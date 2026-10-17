"""Tests for the append-only SQLite event log."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aar.hub.events import Actor, EventLog, IdempotencyConflict, new_event_id


def _log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "hub.db")


def test_actor_rejects_unknown_kind_and_empty_id() -> None:
    with pytest.raises(ValueError):
        Actor("robot", "x")
    with pytest.raises(ValueError):
        Actor("agent", "")


def test_update_and_delete_are_blocked_by_database_triggers(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    log = EventLog(db)
    log.append("note", Actor("agent", "a1"), {"k": 1})

    raw = sqlite3.connect(db)
    raw.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.Error) as update_err:
        raw.execute("UPDATE events SET type = 'tampered' WHERE seq = 1")
        raw.commit()
    assert "append-only" in str(update_err.value)

    with pytest.raises(sqlite3.Error) as delete_err:
        raw.execute("DELETE FROM events WHERE seq = 1")
        raw.commit()
    assert "append-only" in str(delete_err.value)

    row = raw.execute("SELECT type FROM events WHERE seq = 1").fetchone()
    assert row["type"] == "note"
    raw.close()


def test_insert_or_replace_cannot_rewrite_a_row(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    log = EventLog(db)
    original = log.append("note", Actor("agent", "a1"), {"k": 1})

    raw = sqlite3.connect(db)
    raw.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.Error) as err:
        raw.execute(
            "INSERT OR REPLACE INTO events "
            "(seq, id, project_id, type, actor_kind, actor_id, ts, payload) "
            "VALUES (1, 'FORGED', NULL, 'forged', 'agent', 'x', 'now', '{}')"
        )
        raw.commit()
    assert "append-only" in str(err.value)

    row = raw.execute("SELECT id, type FROM events WHERE seq = 1").fetchone()
    assert row["id"] == original.id
    assert row["type"] == "note"
    raw.close()


def test_recursive_triggers_pragma_is_on(tmp_path: Path) -> None:
    log = _log(tmp_path)
    assert log._conn.execute("PRAGMA recursive_triggers").fetchone()[0] == 1


def test_open_with_create_false_requires_append_only_triggers(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    EventLog(db).close()  # creates the schema (with triggers)
    EventLog(db, create=False).close()  # triggers present: fine

    bare = tmp_path / "bare.db"
    sqlite3.connect(bare).close()  # an empty database with no triggers
    with pytest.raises(RuntimeError):
        EventLog(bare, create=False)


def test_append_assigns_monotonic_seq_and_unique_ids(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    for i in range(5):
        log.append("note", actor, {"i": i})

    events = log.read()
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert [e.payload["i"] for e in events] == [0, 1, 2, 3, 4]
    ids = [e.id for e in events]
    assert len(set(ids)) == 5
    assert all(len(i) == 26 for i in ids)
    assert log.latest_seq() == 5


def test_same_key_dedupes_but_different_payload_conflicts(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    first = log.append("note", actor, {"n": 1}, idempotency_key="k1")
    same = log.append("note", actor, {"n": 1}, idempotency_key="k1")

    assert same.seq == first.seq
    assert same.id == first.id
    assert log.latest_seq() == 1

    with pytest.raises(IdempotencyConflict):
        log.append("note", actor, {"n": 2}, idempotency_key="k1")
    assert log.latest_seq() == 1


def test_append_recovers_from_idempotency_key_race(tmp_path: Path, monkeypatch) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    first = log.append("note", actor, {"n": 1}, idempotency_key="k")

    # Simulate losing the pre-check race: pretend the key is absent so the INSERT runs
    # and trips the UNIQUE constraint, exercising the recovery branch.
    monkeypatch.setattr(log, "_lookup_idempotent", lambda *a, **k: None)
    recovered = log.append("note", actor, {"n": 1}, idempotency_key="k")

    assert recovered.seq == first.seq
    assert recovered.id == first.id
    assert log.latest_seq() == 1


def test_read_filters_by_since_project_and_type(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    log.append("a", actor, project_id="prj_1")
    log.append("b", actor, project_id="prj_2")
    log.append("a", actor)  # global event, no project

    assert len(log.read(project_id="prj_1")) == 1
    assert len(log.read(project_id="prj_2")) == 1
    assert {e.type for e in log.read(types=["a"])} == {"a"}
    assert len(log.read(types=["a"])) == 2
    assert log.read(types=[]) == []

    tail = log.read(since_seq=2)
    assert [e.seq for e in tail] == [3]


def test_read_respects_limit_and_iter_pages_everything(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("node", "n1")
    for i in range(10):
        log.append("tick", actor, {"i": i})

    assert len(log.read(limit=3)) == 3
    collected = list(log.iter_events(batch_size=3))
    assert [e.payload["i"] for e in collected] == list(range(10))


def test_new_event_id_is_time_ordered_across_milliseconds() -> None:
    early = new_event_id(ts_ms=1)
    later = new_event_id(ts_ms=2_000_000_000_000)
    assert early < later
    assert len(early) == len(later) == 26


def test_new_event_id_is_monotonic_within_a_millisecond() -> None:
    ids = [new_event_id(ts_ms=1_700_000_000_000) for _ in range(2000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 2000


def test_read_only_log_refuses_writes(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    EventLog(db).append("note", Actor("agent", "a1"), {"k": 1})
    ro = EventLog(db, read_only=True)
    assert len(ro.read()) == 1
    with pytest.raises(RuntimeError):
        ro.append("note", Actor("agent", "a1"), {"k": 2})


def test_insert_or_replace_cannot_rewrite_via_idempotency_key(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    log = EventLog(db)
    original = log.append("note", Actor("agent", "a1"), {"k": 1}, idempotency_key="key-1")

    # A raw connection (recursive_triggers off) tries to REPLACE via the key, with a
    # fresh id/seq. The BEFORE INSERT guard must still block it.
    raw = sqlite3.connect(db)
    raw.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.Error) as err:
        raw.execute(
            "INSERT OR REPLACE INTO events "
            "(id, project_id, type, actor_kind, actor_id, ts, payload, idempotency_key) "
            "VALUES ('FORGED', NULL, 'forged', 'agent', 'x', 'now', '{}', 'key-1')"
        )
        raw.commit()
    assert "append-only" in str(err.value)

    row = raw.execute("SELECT id, payload FROM events WHERE idempotency_key = 'key-1'").fetchone()
    assert row["id"] == original.id
    assert row["payload"] == '{"k":1}'
    raw.close()


def test_read_only_open_of_missing_db_raises_and_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(FileNotFoundError):
        EventLog(missing, read_only=True)
    assert not missing.exists()


def test_transaction_cannot_be_used_after_its_block(tmp_path: Path) -> None:
    log = _log(tmp_path)
    captured: list = []
    with log.transaction() as txn:
        captured.append(txn)
    stale = captured[0]
    with pytest.raises(RuntimeError):
        stale.append("note", Actor("agent", "a1"), {})
    with pytest.raises(RuntimeError):
        stale.read()
