"""Tests for the append-only SQLite event log."""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
from pathlib import Path, PureWindowsPath

import pytest

from aar.hub.events import (
    Actor,
    EventLog,
    IdempotencyConflict,
    match_or_conflict,
    new_event_id,
)


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


def _miss_precheck_once(log: EventLog, monkeypatch) -> None:
    """Make the pre-check miss once (a simulated lost race) but keep the recovery real."""
    real = log._lookup_idempotent
    state = {"missed": False}

    def wrapper(key, event_type, project_id, payload_json, actor):
        if not state["missed"]:
            state["missed"] = True
            return None  # pre-check misses: the INSERT runs and trips the UNIQUE key
        return real(key, event_type, project_id, payload_json, actor)

    monkeypatch.setattr(log, "_lookup_idempotent", wrapper)


def test_append_recovery_branch_returns_the_winner_on_a_match(tmp_path: Path, monkeypatch) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    first = log.append("note", actor, {"n": 1}, idempotency_key="k")

    _miss_precheck_once(log, monkeypatch)
    recovered = log.append("note", actor, {"n": 1}, idempotency_key="k")  # same request

    assert recovered.seq == first.seq
    assert recovered.id == first.id
    assert log.latest_seq() == 1


def test_append_recovery_branch_conflicts_on_a_mismatch(tmp_path: Path, monkeypatch) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    log.append("note", actor, {"n": 1}, idempotency_key="k")

    _miss_precheck_once(log, monkeypatch)
    # Different payload under the same key: the recovery branch must re-check and raise,
    # not hand back the winner's event.
    with pytest.raises(IdempotencyConflict):
        log.append("note", actor, {"n": 2}, idempotency_key="k")
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


def _stored(log: EventLog, payload: dict) -> object:
    """Append a payload then reload it, so it has been through JSON like a real event."""
    log.append("thing", Actor("agent", "a1"), payload, idempotency_key="K")
    return log.get_by_idempotency_key("K")


@pytest.mark.parametrize(
    ("stored_payload", "expected_payload"),
    [
        ({"v": [1, 2]}, {"v": (1, 2)}),  # a tuple request reloads as a list
        ({"v": {"1": "a"}}, {"v": {1: "a"}}),  # an int dict key reloads as a string
    ],
)
def test_match_or_conflict_accepts_json_equivalent_retries(
    tmp_path: Path, stored_payload: dict, expected_payload: dict
) -> None:
    existing = _stored(_log(tmp_path), stored_payload)
    # Same request once JSON-normalised: must be treated as a retry, not a conflict.
    match_or_conflict(existing, "thing", expected_payload, ["v"])


@pytest.mark.parametrize(
    ("stored_value", "expected_value"),
    [
        (True, 1),  # bool vs int
        (1, True),
        (False, 0),
        (1, 1.0),  # int vs float
    ],
)
def test_match_or_conflict_rejects_json_distinct_requests(
    tmp_path: Path, stored_value: object, expected_value: object
) -> None:
    existing = _stored(_log(tmp_path), {"v": stored_value})
    with pytest.raises(IdempotencyConflict):
        match_or_conflict(existing, "thing", {"v": expected_value}, ["v"])


def test_append_rejects_non_finite_floats(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            log.append("thing", actor, {"v": bad})
    assert log.latest_seq() == 0  # nothing was written


def test_read_only_memory_database_is_rejected() -> None:
    with pytest.raises(ValueError):
        EventLog(":memory:", read_only=True)


def test_read_only_open_handles_special_characters_in_path(tmp_path: Path) -> None:
    db = tmp_path / "weird ?#name.db"
    EventLog(db).append("note", Actor("agent", "a1"), {"k": 1})
    ro = EventLog(db, read_only=True)
    assert len(ro.read()) == 1


def _race_worker(db_path: str, key: str, index: int, barrier, queue) -> None:
    """One forked contender: append the same key with a unique payload, report the outcome."""
    from aar.hub.events import Actor, EventLog, IdempotencyConflict

    log = EventLog(db_path)
    try:
        barrier.wait(timeout=20)
        event = log.append("race", Actor("agent", f"w{index}"), {"n": index}, idempotency_key=key)
        queue.put(("ok", index, event.payload["n"]))
    except IdempotencyConflict:
        queue.put(("conflict", index, None))
    except Exception as exc:  # report any other failure to the parent
        queue.put(("error", index, repr(exc)))
    finally:
        log.close()


def test_append_fork_race_resolves_to_one_winner(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    EventLog(db).close()  # create the schema before forking
    n = 6
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(n)
    queue: multiprocessing.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_race_worker, args=(str(db), "RACE", i, barrier, queue))
        for i in range(n)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=30) for _ in range(n)]
    for proc in procs:
        proc.join(timeout=30)

    errors = [r for r in results if r[0] == "error"]
    oks = [r for r in results if r[0] == "ok"]
    conflicts = [r for r in results if r[0] == "conflict"]
    assert errors == []
    assert len(oks) == 1  # exactly one winner, never a non-winner handed someone else's event
    assert len(conflicts) == n - 1

    stored = EventLog(db, read_only=True).read(types=["race"])
    assert len(stored) == 1
    winner_index = oks[0][1]
    assert oks[0][2] == winner_index == stored[0].payload["n"]


def test_append_rejects_cross_project_key_reuse(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    actor = Actor("agent", "a1")
    log.append("t", actor, {"n": 1}, project_id="prj_A", idempotency_key="K")
    # Same key, same type, same payload, different project: a conflict, not a silent reuse.
    with pytest.raises(IdempotencyConflict):
        log.append("t", actor, {"n": 1}, project_id="prj_B", idempotency_key="K")


def test_lookup_tolerates_legacy_non_finite_payload(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    log = EventLog(db)
    # A legacy row written before allow_nan=False: insert NaN text directly.
    raw = sqlite3.connect(db)
    raw.execute(
        "INSERT INTO events (id, project_id, type, actor_kind, actor_id, ts, payload, "
        "idempotency_key) VALUES "
        "('legacy0000000000000000000', NULL, 'legacy', 'agent', 'a', 'now', '{\"v\":NaN}', 'LK')"
    )
    raw.commit()
    raw.close()
    # Retrying that key with a finite payload must conflict, not raise a bare ValueError.
    with pytest.raises(IdempotencyConflict):
        log.append("legacy", Actor("agent", "a"), {"v": 1}, idempotency_key="LK")


def test_read_only_open_of_directory_raises_oserror(tmp_path: Path) -> None:
    directory = tmp_path / "adir"
    directory.mkdir()
    with pytest.raises(OSError) as err:
        EventLog(directory, read_only=True)
    # A directory exists, so it is an OSError, not the missing-file FileNotFoundError.
    assert not isinstance(err.value, FileNotFoundError)


def test_read_only_open_of_non_sqlite_file_raises_oserror(tmp_path: Path) -> None:
    junk = tmp_path / "notadb.txt"
    junk.write_text("definitely not sqlite")
    with pytest.raises(OSError) as err:
        EventLog(junk, read_only=True)
    assert not isinstance(err.value, FileNotFoundError)


def test_read_only_open_rejects_nul_byte_in_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EventLog(f"{tmp_path / 'x.db'}\x00evil", read_only=True)


def test_read_only_open_tolerates_non_utf8_path(tmp_path: Path) -> None:
    weird = os.fsdecode(os.fsencode(tmp_path) + b"/\xff\xfe.db")  # a non-UTF-8 filename
    # Must reach the missing-file branch, not crash with UnicodeEncodeError in the URI.
    with pytest.raises(FileNotFoundError):
        EventLog(weird, read_only=True)


def test_read_only_open_handles_leading_double_slash(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    EventLog(db).append("note", Actor("agent", "a1"), {"k": 1})
    doubled = "/" + str(db)  # e.g. //tmp/.../x.db -- must not parse as a URI authority
    assert doubled.startswith("//")
    assert len(EventLog(doubled, read_only=True).read()) == 1


def test_read_only_open_handles_percent_in_path(tmp_path: Path) -> None:
    db = tmp_path / "a%2Fb.db"  # a literal percent must be encoded, not left raw
    EventLog(db).append("note", Actor("agent", "a1"), {"k": 1})
    assert len(EventLog(db, read_only=True).read()) == 1


# -- read-only open: path handling (round 3) --------------------------------------


def test_read_only_open_does_not_collapse_dotdot_through_symlink(tmp_path: Path) -> None:
    a = tmp_path / "A"
    (a / "sub" / "subsub").mkdir(parents=True)
    EventLog(a / "sub" / "x.db").append("note", Actor("agent", "a1"), {"which": "real"})
    EventLog(a / "x.db").append("note", Actor("agent", "a1"), {"which": "decoy"})
    (a / "link").symlink_to(a / "sub" / "subsub")

    # The OS must resolve link -> A/sub/subsub, then '..' -> A/sub, reaching A/sub/x.db.
    # Collapsing '..' as text (os.path.abspath) would wrongly reach the decoy A/x.db.
    through = f"{a}/link/../x.db"
    assert EventLog(through, read_only=True).read()[0].payload["which"] == "real"


def test_windows_paths_make_valid_file_uris() -> None:
    # The read-only open builds its URI with Path.as_uri(); confirm that is correct for
    # Windows drive letters, backslashes and UNC paths (verified via PureWindowsPath).
    assert PureWindowsPath("C:/data/hub.db").as_uri() == "file:///C:/data/hub.db"
    assert PureWindowsPath(r"C:\data\hub.db").as_uri() == "file:///C:/data/hub.db"
    assert PureWindowsPath(r"\\server\share\hub.db").as_uri() == "file://server/share/hub.db"


# -- create=False must not touch a foreign database (round 3) ---------------------


def test_create_false_leaves_a_foreign_database_untouched(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.db"
    con = sqlite3.connect(foreign)
    con.execute("CREATE TABLE t (x)")
    con.commit()
    con.close()
    before = foreign.read_bytes()
    mtime = foreign.stat().st_mtime_ns

    with pytest.raises(RuntimeError):
        EventLog(foreign, create=False)

    assert foreign.read_bytes() == before  # schema checked before any WAL switch
    assert foreign.stat().st_mtime_ns == mtime
    assert not (tmp_path / "foreign.db-wal").exists()
    assert not (tmp_path / "foreign.db-shm").exists()


def test_failed_open_closes_the_connection(tmp_path: Path, monkeypatch) -> None:
    import aar.hub.events as events_mod

    real_connect = events_mod.sqlite3.connect
    opened: list = []

    def spy(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    foreign = tmp_path / "foreign.db"
    con = real_connect(foreign)
    con.execute("CREATE TABLE t (x)")
    con.commit()
    con.close()

    monkeypatch.setattr(events_mod.sqlite3, "connect", spy)
    with pytest.raises(RuntimeError):
        EventLog(foreign, create=False)

    assert opened, "no connection was opened"
    with pytest.raises(sqlite3.ProgrammingError):
        opened[-1].execute("SELECT 1")  # the failed open's connection must be closed


# -- idempotency matching survivors (round 3) -------------------------------------


def test_lookup_conflicts_on_a_different_type(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    log.append("a", actor, {"n": 1}, idempotency_key="K")
    # Same payload/project/actor, different type: a conflict (pins the type clause).
    with pytest.raises(IdempotencyConflict):
        log.append("b", actor, {"n": 1}, idempotency_key="K")


def test_lookup_distinguishes_true_from_one(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    log.append("t", actor, {"v": True}, idempotency_key="K")
    # True and 1 are Python-equal but text-distinct; reusing the key must conflict.
    with pytest.raises(IdempotencyConflict):
        log.append("t", actor, {"v": 1}, idempotency_key="K")


def test_same_key_different_actor_conflicts(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("t", Actor("agent", "a1"), {"n": 1}, idempotency_key="K")
    with pytest.raises(IdempotencyConflict):
        log.append("t", Actor("agent", "a2"), {"n": 1}, idempotency_key="K")


def test_same_key_same_actor_and_payload_dedupes(tmp_path: Path) -> None:
    log = _log(tmp_path)
    actor = Actor("agent", "a1")
    first = log.append("t", actor, {"n": 1}, idempotency_key="K")
    again = log.append("t", actor, {"n": 1}, idempotency_key="K")  # later ts, still a retry
    assert again.seq == first.seq
    assert log.latest_seq() == 1


def test_read_only_connection_rejects_direct_writes(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    EventLog(db).append("t", Actor("agent", "a1"), {"k": 1})
    ro = EventLog(db, read_only=True)
    # query_only must reject a write issued straight on the connection.
    with pytest.raises(sqlite3.OperationalError):
        ro._conn.execute(
            "INSERT INTO events (id, type, actor_kind, actor_id, ts, payload) "
            "VALUES ('x', 't', 'agent', 'a', 'now', '{}')"
        )


def test_transaction_find_after_block_raises(tmp_path: Path) -> None:
    log = _log(tmp_path)
    captured: list = []
    with log.transaction() as txn:
        captured.append(txn)
    with pytest.raises(RuntimeError):
        captured[0].find_by_idempotency_key("k")
