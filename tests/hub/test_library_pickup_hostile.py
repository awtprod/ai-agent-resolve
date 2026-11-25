"""S25d: the library pickup against a hostile or changing library. Every case is a typed,
visible condition (a ``recipe.library_pickup`` event, reported once) or nothing; never a crash,
never a return that says every segment was removed. Each test hands one cut over through the
real recipe (one continuous range, so the library can carry it) and then changes the library."""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from ccn_recipe_support import PC_ROOT, WORKER, CcnEnv, agent_cut, make_run, outputs, start, work
from library_handoff_support import make_library_setup, save, trim_tail, write_seed
from write_support import HUMAN

from aar.hub.reviews import decide
from aar.media import settle
from aar.recipes import library_returns
from aar.recipes.desktop_returns import returned_edits
from aar.recipes.library_returns import EVENT_PICKUP, Limits, scan_library_returns
from aar.recipes.run import recipe_status
from aar.recipes.state import load_state
from ccn_support import needs_ffmpeg

pytestmark = needs_ffmpeg
pytest.importorskip("zstandard")

ONE = {"ranges": [{"start_s": 0.9, "end_s": 11.1}]}
FAST = Limits(settle_seconds=0.0)


def _placed(env: CcnEnv) -> tuple[str, dict[str, Any], Path]:
    show = env.show("cam-a.mp4")
    library = make_library_setup(env.media / "share", [f"{PC_ROOT}\\cam-a.mp4"])
    run = make_run(env, env.params(show, library=library), key="lib")
    start(env, run, "lib")
    assert work(env) == 5
    agent_cut(env, run, keep=ONE, key="r1")
    assert work(env) == 2
    handoff = outputs(env, run, "export")["library_handoff"]
    assert handoff["status"] == "placed", handoff
    return run, handoff, Path(handoff["library"]).joinpath(*handoff["parts"])


def sweep(env: CcnEnv, limits: Limits = FAST) -> library_returns.Sweep:
    return scan_library_returns(env.hub.log, env.hub.store, WORKER, limits=limits)


def conditions(env: CcnEnv) -> Counter[str]:
    return Counter(e.payload["condition"] for e in env.hub.log.iter_events(types=[EVENT_PICKUP]))


def snapshots(env: CcnEnv) -> list[str]:
    area = env.work / library_returns.snapshot_folder().name
    return sorted(p.name for p in area.rglob("*") if p.is_file())


def test_each_hostile_state_of_the_project_is_one_visible_condition(ccn_env: CcnEnv) -> None:
    env = ccn_env
    run, _handoff, project = _placed(env)
    db = project / "Project.db"
    original = db.read_bytes()

    def expect(code: str, limits: Limits = FAST) -> None:
        before = conditions(env)[code]
        result = sweep(env, limits)
        assert result.returned == 0 and result.conditions == [code], result
        # The next sweep reports nothing new: a place is looked at again (the same condition,
        # not recorded twice); a file's content that gave a condition is not even read again.
        result = sweep(env, limits)
        assert result.returned == 0
        assert result.conditions == [code] or (result.conditions, result.repeated) == ([], 1)
        assert conditions(env)[code] == before + 1
        assert returned_edits(env.hub.log, env.hub.project_id, run) == []
        assert snapshots(env) == []

    # The owner deleted or renamed the project.
    moved = project.with_name("renamed by the owner")
    project.rename(moved)
    expect("project_missing")
    moved.rename(project)
    assert sweep(env).unchanged == 1

    # Resolve is saving: its rollback journal is beside the file.
    journal = Path(str(db) + "-journal")
    journal.write_bytes(b"")
    expect("busy")
    journal.unlink()

    # Another project saved under the same folder name: its id is not the one handed over.
    db.unlink()
    write_seed(db, project_name="Someone else's")
    expect("project_replaced")
    db.write_bytes(original)
    os.utime(db, (1_000_000_000, 1_000_000_000))

    # The owner renamed the timeline.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE Sm2Timeline SET Name = 'Renamed in Resolve'")
    conn.commit()
    conn.close()
    os.utime(db, (1_000_000_000, 1_000_000_000))
    expect("timeline_missing")

    # A damaged file.
    data = bytearray(original)
    data[4096:8192] = b"\xff" * 4096
    db.write_bytes(bytes(data))
    os.utime(db, (1_000_000_000, 1_000_000_000))
    expect("damaged")

    # A file over the pickup's size bound (and so, a huge one) is refused unread.
    db.write_bytes(original + b"\x00" * 100)
    os.utime(db, (1_000_000_000, 1_000_000_000))
    expect("too_large", Limits(settle_seconds=0.0, db_bytes=len(original)))

    # Every condition shows in the run's status, once each.
    view = recipe_status(env.hub.log, run)
    assert sorted(c["condition"] for c in view["library_pickup"]) == sorted(
        ["project_missing", "busy", "project_replaced", "timeline_missing", "damaged", "too_large"]
    )
    for entry in view["library_pickup"]:
        assert entry["fix_hint"] and entry["handoff"] == latest_export(env, run)

    # Put right and saved: the save comes back as a return.
    db.write_bytes(original)
    save(db, trim_tail(5))
    assert sweep(env).returned == 1
    [ret] = returned_edits(env.hub.log, env.hub.project_id, run)
    assert ret["status"] == "needs_review" and ret["source"] == "library_save"


def latest_export(env: CcnEnv, run: str) -> str:
    op = load_state(env.hub.log, run).done_op("export")
    assert op is not None
    return op.op_id


def test_a_file_that_changes_while_it_is_copied_is_not_taken(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    _run, _handoff, project = _placed(env)
    db = project / "Project.db"
    save(db, trim_tail(5))
    real_open = settle.open_regular
    wrote: list[bool] = []

    @contextlib.contextmanager
    def open_and_write(p, *, dir_fd=None):  # type: ignore[no-untyped-def]
        with real_open(p, dir_fd=dir_fd) as stream:
            read = stream.read

            def read_then_write(n: int = -1) -> bytes:
                chunk = read(n)
                if chunk and not wrote:
                    wrote.append(True)
                    with db.open("ab") as writer:  # Resolve writing as we copy
                        writer.write(b"\x00")
                return chunk

            stream.read = read_then_write
            yield stream

    with monkeypatch.context() as patch:  # (the fixture's own settings stay)
        patch.setattr(settle, "open_regular", open_and_write)
        result = sweep(env)
    assert result.conditions == ["busy"] and result.returned == 0
    assert snapshots(env) == []
    # The writes stopped; the next sweep takes the file as it now is (a valid SQLite file with
    # trailing zero bytes is damaged or a return, never a crash).
    os.utime(db, (1_000_000_000, 1_000_000_000))
    result = sweep(env)
    assert result.returned + len(result.conditions) == 1
    assert conditions(env)["busy"] == 1


def test_the_library_gone_unlinked_or_read_only(ccn_env: CcnEnv) -> None:
    env = ccn_env
    _run, handoff, project = _placed(env)
    library = Path(handoff["library"])
    db = project / "Project.db"
    save(db, trim_tail(5))

    # A link below the library is never followed, even to the real project folder.
    real = project.with_name("real folder")
    project.rename(real)
    project.symlink_to(real, target_is_directory=True)
    assert sweep(env).conditions == ["path_refused"]
    project.unlink()
    real.rename(project)

    # Not a library any more (its Users folder gone).
    users = library / "Resolve Projects" / "Users"
    users.rename(users.with_name("Users.old"))
    assert sweep(env).conditions == ["not_a_library"]
    users.with_name("Users.old").rename(users)

    # The library unplugged.
    away = library.with_name("unplugged")
    library.rename(away)
    assert sweep(env).conditions == ["library_missing"]

    # The library's path now a link leading out of the media root.
    outside = env.work.parent / "outside-media"
    shutil.move(str(away), outside)
    library.symlink_to(outside, target_is_directory=True)
    assert sweep(env).conditions == ["path_outside_media_root"]
    library.unlink()
    shutil.move(str(outside), library)

    # Read-only: the pickup only reads, so a save still comes back.
    modes = []
    for folder in [library, *[p for p in library.rglob("*") if p.is_dir()]]:
        modes.append((folder, folder.stat().st_mode))
        folder.chmod(0o555)
    try:
        if os.access(library, os.W_OK):
            pytest.skip("running with privileges that ignore folder modes")
        result = sweep(env)
        assert result.returned == 1, result
    finally:
        for folder, mode in reversed(modes):
            folder.chmod(mode)
    counts = conditions(env)
    for code in ("path_refused", "not_a_library", "library_missing", "path_outside_media_root"):
        assert counts[code] == 1


def test_pickup_stops_when_the_review_is_decided_and_honours_its_bounds(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    run, _handoff, project = _placed(env)
    db = project / "Project.db"
    save(db, trim_tail(5))

    # Bytes and time per sweep: the project waits for a later sweep, nothing is reported.
    result = sweep(env, Limits(settle_seconds=0.0, bytes=10))
    assert (result.deferred, result.returned, result.conditions) == (1, 0, [])
    result = sweep(env, Limits(settle_seconds=0.0, seconds=0.0))
    assert result.cut_short and result.checked == 0
    result = sweep(env, Limits(settle_seconds=0.0, projects=0))
    assert result.cut_short and result.checked == 0

    # A save not settled yet waits silently; the next sweep after it settles takes it.
    os.utime(db)  # modified just now
    result = sweep(env, Limits())
    assert (result.deferred, result.conditions, result.returned) == (1, [], 0)

    # Snapshots a killed worker left are swept; anything else there is named, not removed.
    folder = library_returns.snapshot_folder()
    pickup = folder / library_returns.PICKUP_AREA
    pickup.mkdir(exist_ok=True)
    stale = [pickup / ".aar-pickup-1-dead.db", folder / ".aar-library-dead.db"]
    for path in stale:
        path.write_bytes(b"x")
        os.utime(path, (1_000_000, 1_000_000))
    fresh = pickup / ".aar-pickup-2-live.db"
    fresh.write_bytes(b"x")
    stranger = folder / "someone's file"
    stranger.write_bytes(b"x")
    result = sweep(env)
    assert result.snapshots_removed == 2 and not any(p.exists() for p in stale)
    assert fresh.exists() and stranger.exists() and "someone's file" in result.snapshot_leftovers
    assert result.returned == 1
    fresh.unlink()
    stranger.unlink()

    # The review is decided: later saves are not picked up.
    op = load_state(env.hub.log, run).latest_in_round("review", 1)
    assert op is not None and op.review_id is not None
    decide(env.hub.log, env.hub.project_id, op.review_id, "approve", "fine", HUMAN)
    work(env)
    save(db, trim_tail(5))
    result = sweep(env)
    assert (result.closed, result.checked, result.returned) == (1, 0, 0)
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == 1


def test_off_posix_the_pickup_is_a_typed_condition(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    _run_id, handoff, _project = _placed(env)
    model = library_returns.Handoff.model_validate(handoff)
    monkeypatch.delattr(library_returns.os, "O_NOFOLLOW")
    with pytest.raises(library_returns._Condition) as caught, library_returns._project_dir(model):
        pass
    assert caught.value.code == "unsupported_platform"


def test_fuzz_damaged_saves_are_returns_or_typed_conditions_never_a_crash(
    ccn_env: CcnEnv,
) -> None:
    """Random damage to the saved file (bytes overwritten, the file cut short or grown, pages
    swapped): each sweep stores a return or reports a typed condition about the content, and
    leaves no snapshot behind. Any other exception fails the test."""
    import random

    env = ccn_env
    run, _handoff, project = _placed(env)
    db = project / "Project.db"
    save(db, trim_tail(3))
    good = db.read_bytes()
    page = 4096
    rng = random.Random(25_04)
    allowed = {"damaged", "project_replaced", "timeline_missing", "too_large"}
    seen: Counter[str] = Counter()
    for case in range(60):
        data = bytearray(good)
        how = case % 4
        if how == 0:  # bytes overwritten somewhere
            at = rng.randrange(len(data))
            data[at : at + rng.randint(1, 64)] = bytes(rng.randrange(256) for _ in range(64))
        elif how == 1:  # cut short
            del data[rng.randrange(16, len(data)) :]
        elif how == 2:  # grown with junk
            data += bytes(rng.randrange(256) for _ in range(rng.randint(1, 5000)))
        else:  # two pages swapped
            pages = len(data) // page
            a, b = rng.sample(range(1, pages), 2)
            pa, pb = data[a * page : (a + 1) * page], data[b * page : (b + 1) * page]
            data[a * page : (a + 1) * page], data[b * page : (b + 1) * page] = pb, pa
        db.write_bytes(bytes(data))
        os.utime(db, (1_000_000_000, 1_000_000_000))
        result = sweep(env)
        assert set(result.conditions) <= allowed, (case, result)
        assert result.returned + len(result.conditions) + result.repeated <= 1
        seen.update(result.conditions or (["returned"] if result.returned else ["repeated"]))
        assert snapshots(env) == []
    assert seen["damaged"] > 0, seen
    edits = returned_edits(env.hub.log, env.hub.project_id, run)
    assert all(e["source"] == "library_save" for e in edits)
