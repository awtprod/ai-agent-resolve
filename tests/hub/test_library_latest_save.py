"""S25e: the latest save always wins. The pickup records a return whenever the settled file
differs from the newest state the hub recorded for the hand-off (the placed project, or the
newest return), never more than once for one save, whichever workers see it."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest
from ccn_recipe_support import PC_ROOT, WORKER, CcnEnv, agent_cut, make_run, outputs, start, work
from library_handoff_support import make_library_setup, save, trim_tail

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import EventLog
from aar.recipes import desktop_returns, library_returns
from aar.recipes.desktop_returns import returned_edits
from aar.recipes.library_returns import Limits, scan_library_returns
from ccn_support import needs_ffmpeg

pytestmark = needs_ffmpeg
pytest.importorskip("zstandard")

ONE = {"ranges": [{"start_s": 0.9, "end_s": 11.1}]}
FAST = Limits(settle_seconds=0.0)


def _placed(env: CcnEnv) -> tuple[str, Path]:
    show = env.show("cam-a.mp4")
    library = make_library_setup(env.media / "share", [f"{PC_ROOT}\\cam-a.mp4"])
    run = make_run(env, env.params(show, library=library), key="lib")
    start(env, run, "lib")
    assert work(env) == 5
    agent_cut(env, run, keep=ONE, key="r1")
    assert work(env) == 2
    handoff = outputs(env, run, "export")["library_handoff"]
    assert handoff["status"] == "placed", handoff
    return run, Path(handoff["library"]).joinpath(*handoff["parts"]) / "Project.db"


def sweep(env: CcnEnv, log: EventLog | None = None, store: Any = None) -> Any:
    return scan_library_returns(log or env.hub.log, store or env.hub.store, WORKER, limits=FAST)


def put(db: Path, data: bytes) -> None:
    db.write_bytes(data)
    os.utime(db, (1_000_000_000, 1_000_000_000))


def segment_of(env: CcnEnv, ret: dict[str, Any]) -> tuple[str, int]:
    [seg] = json.loads(env.hub.store.read_bytes(ret["diff"]))["segments"]
    return seg["status"], seg.get("tail_frames", 0) if seg["status"] == "trimmed" else 0


def test_edit_save_undo_save_comes_back_as_the_plan(ccn_env: CcnEnv) -> None:
    env = ccn_env
    run, db = _placed(env)
    placed = db.read_bytes()
    save(db, trim_tail(12))  # the edit, saved
    assert sweep(env).returned == 1
    put(db, placed)  # undone, saved again: back to the placed content
    result = sweep(env)
    assert (result.returned, result.unchanged, result.repeated) == (1, 0, 0), result
    edits = returned_edits(env.hub.log, env.hub.project_id, run)
    assert [segment_of(env, e) for e in edits] == [("trimmed", 12), ("unchanged", 0)]
    # The newest record is the file as it is now: the timeline equals the plan.
    last = json.loads(env.hub.store.read_bytes(edits[-1]["diff"]))
    assert last["unchanged"] is True and not last["not_carried"]
    assert edits[-1]["snapshot_sha256"] == edits[-1]["project_db"]
    assert edits[0]["follows"] == "placed" and edits[1]["follows"].startswith("return:")
    # The same file again: the newest recorded state, nothing new.
    seq = env.hub.log.latest_seq()
    result = sweep(env)
    assert (result.returned, result.repeated) == (0, 1)
    assert env.hub.log.latest_seq() == seq


def test_a_b_a_b_records_four_returns(ccn_env: CcnEnv) -> None:
    env = ccn_env
    run, db = _placed(env)
    save(db, trim_tail(4))
    a = db.read_bytes()
    save(db, trim_tail(4))
    b = db.read_bytes()
    for data in (a, b, a, b):
        put(db, data)
        assert sweep(env).returned == 1
        assert sweep(env).returned == 0  # each save once
    edits = returned_edits(env.hub.log, env.hub.project_id, run)
    assert [segment_of(env, e) for e in edits] == [("trimmed", t) for t in (4, 8, 4, 8)]
    assert len({e["follows"] for e in edits}) == 4


def test_a_save_identical_to_the_last_recorded_state_records_nothing(ccn_env: CcnEnv) -> None:
    env = ccn_env
    run, db = _placed(env)
    placed = db.read_bytes()
    put(db, placed)  # saved with nothing changed: the placed content, nothing recorded yet
    assert (sweep(env).returned, sweep(env).unchanged) == (0, 1)
    save(db, trim_tail(3))
    assert sweep(env).returned == 1
    again = db.read_bytes()
    put(db, again)
    seq = env.hub.log.latest_seq()
    assert sweep(env).returned == 0
    assert env.hub.log.latest_seq() == seq
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == 1


@pytest.mark.parametrize("states", [["edit", "undo"], ["edit", "edit", "undo", "edit"]])
def test_two_concurrent_workers_record_each_save_once(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch, states: list[str]
) -> None:
    """Two workers (each its own connection to the hub's log, in its own thread) sweep at the
    same time, both past their state check before either records: every save is recorded
    once, none is missing, and each return follows the one before it."""
    env = ccn_env
    run, db = _placed(env)
    placed = db.read_bytes()
    real_read = library_returns._read
    barrier = threading.Barrier(2, timeout=60)

    def both_read(*args: Any, **kwargs: Any) -> Any:
        found = real_read(*args, **kwargs)
        barrier.wait()  # both workers hold the same state and snapshot here
        return found

    monkeypatch.setattr(library_returns, "_read", both_read)
    path = Path(env.hub.log._path)
    stores = env.hub.store._root  # the same artifact folder for both workers
    results: list[Any] = []

    def worker() -> None:
        log = EventLog(path)
        try:
            results.append(sweep(env, log, ArtifactStore(stores, log)))
        finally:
            log.close()

    for step in states:
        if step == "edit":
            save(db, trim_tail(2))
        else:
            put(db, placed)
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(90)
        assert not any(t.is_alive() for t in threads)
    edits = returned_edits(env.hub.log, env.hub.project_id, run)
    assert len(edits) == len(states), [r for r in results]
    assert sum(r.returned for r in results) == len(states)
    assert sum(r.repeated for r in results) == len(states)
    follows = [e["follows"] for e in edits]
    assert follows[0] == "placed" and len(set(follows)) == len(follows)


def test_a_worker_that_dies_before_recording_leaves_the_save_to_the_next(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker stops (an interrupt) after storing the save's artifacts and before the
    return is journalled; the restarted worker records it once, and the next sweeps nothing."""
    env = ccn_env
    run, db = _placed(env)
    save(db, trim_tail(6))
    real = desktop_returns.record_import

    def dies(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(desktop_returns, "record_import", dies)
    with pytest.raises(KeyboardInterrupt):
        sweep(env)
    monkeypatch.setattr(desktop_returns, "record_import", real)
    library_returns._WATCHED.clear()  # a restarted worker remembers nothing
    assert returned_edits(env.hub.log, env.hub.project_id, run) == []
    assert sweep(env).returned == 1
    assert sweep(env).returned == 0
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == 1


def test_a_return_recorded_meanwhile_defers_a_save_based_on_the_older_state(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker took its state before another worker recorded a different save: inside its
    transaction the state is no longer the newest, so it records nothing (counted deferred);
    the next sweep compares the file with the newest record."""
    env = ccn_env
    run, db = _placed(env)
    save(db, trim_tail(2))
    first = db.read_bytes()
    save(db, trim_tail(2))
    real_state = library_returns._recorded_state
    calls: list[int] = []

    def stale(*args: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            # Another worker records the earlier save now, after this one took its state.
            now = db.read_bytes()
            put(db, first)
            assert sweep(env).returned == 1
            put(db, now)
            return real_state(args[0], args[1], args[2], args[3]).__class__(
                "placed", args[2].placed_sha256
            )
        return real_state(*args)

    monkeypatch.setattr(library_returns, "_recorded_state", stale)
    result = sweep(env)
    assert (result.returned, result.deferred) == (0, 1), result
    monkeypatch.setattr(library_returns, "_recorded_state", real_state)
    assert sweep(env).returned == 1  # the file as it is now, after the newest record
    edits = returned_edits(env.hub.log, env.hub.project_id, run)
    assert [segment_of(env, e) for e in edits] == [("trimmed", 2), ("trimmed", 4)]
