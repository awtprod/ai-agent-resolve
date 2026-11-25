"""S25d fix round: the pickup's rules that the first tests left unpinned. Conditions of one code
with different causes stay apart; a file rewritten all the time is reported ``busy`` once; a
copy refused after it was made still costs its bytes; the library is opened from the media
root's descriptor, so a link swapped in after the path check is refused, not followed; only
the runner's hand-off of the open round is looked at; a save another worker recorded first
is not recorded twice; and what the owner sees when a save brings back content that was
already returned, or changes nothing in the timeline."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ccn_recipe_support import PC_ROOT, WORKER, CcnEnv, agent_cut, make_run, outputs, start, work
from library_handoff_support import make_library_setup, save, trim_tail
from write_support import AGENT, HUMAN

from aar.hub.reviews import decide
from aar.recipes import library_returns
from aar.recipes.desktop_returns import returned_edits
from aar.recipes.events import EVENT_RESULT
from aar.recipes.library_returns import EVENT_PICKUP, Limits, scan_library_returns
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


def events(env: CcnEnv) -> list[dict[str, Any]]:
    return [e.payload for e in env.hub.log.iter_events(types=[EVENT_PICKUP])]


def test_conditions_keep_their_causes_and_a_file_that_never_settles_is_busy(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    run, _handoff, project = _placed(env)
    db = project / "Project.db"
    original = db.read_bytes()

    # Two different link problems: two path_refused events, each with its cause.
    real = project.with_name("real folder")
    project.rename(real)
    project.symlink_to(real, target_is_directory=True)
    assert sweep(env).conditions == ["path_refused"]
    project.unlink()
    real.rename(project)
    db.rename(project / "elsewhere.db")
    db.symlink_to(project / "elsewhere.db")
    assert sweep(env).conditions == ["path_refused"]
    db.unlink()
    (project / "elsewhere.db").rename(db)
    refused = [e for e in events(env) if e["condition"] == "path_refused"]
    assert [e["cause"] for e in refused] == ["link_below_library:4", "db_not_regular"]
    assert refused[0]["message"] != refused[1]["message"]

    # A journal appears only after the copy: refused as busy, and the copy's bytes count.
    save(db, trim_tail(5))
    size = db.stat().st_size
    calls: list[int] = []
    real_side = library_returns._side_file

    def journal_after_copy(folder_fd: int) -> str | None:
        calls.append(1)
        return "Project.db-journal" if len(calls) == 2 else real_side(folder_fd)

    with monkeypatch.context() as patch:
        patch.setattr(library_returns, "_side_file", journal_after_copy)
        result = sweep(env)
    assert result.conditions == ["busy"] and result.returned == 0
    assert result.bytes == size  # the refused copy cost its bytes
    # ... so a second project that would no longer fit waits for the next sweep.
    with monkeypatch.context() as patch:
        calls.clear()
        patch.setattr(library_returns, "_side_file", journal_after_copy)
        limits = Limits(settle_seconds=0.0, bytes=size + size // 2)
        result = sweep(env, limits)
    assert result.bytes == size

    # A file rewritten all the time: silent for busy_after - 1 sweeps, then busy (once).
    slow = Limits(settle_seconds=3600.0, busy_after=3)
    seen = []
    for _ in range(5):
        os.utime(db)  # modified just now, every time
        result = sweep(env, slow)
        seen.append((result.deferred, tuple(result.conditions)))
    assert seen == [(1, ()), (1, ()), (0, ("busy",)), (0, ("busy",)), (0, ("busy",))]
    busy = [e for e in events(env) if e["condition"] == "busy"]
    assert [e["cause"] for e in busy] == ["side_file_after_copy", "never_settles"]
    assert "has not stopped changing in 3 sweeps" in busy[1]["message"]
    # Once it settles it is taken, and the count starts again.
    os.utime(db, (1_000_000_000, 1_000_000_000))
    assert sweep(env).returned == 1
    os.utime(db)
    assert sweep(env, slow).conditions == []
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == 1
    assert db.read_bytes() != original


def test_the_library_is_opened_from_the_media_root_without_following_a_relink(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    run, handoff, project = _placed(env)
    save(project / "Project.db", trim_tail(5))
    library = Path(handoff["library"])

    # The library moved to a folder beside the media root whose name starts with the root's,
    # and a link left in its place: outside the media root.
    sibling = env.media.with_name(env.media.name + "-old")
    sibling.mkdir()
    shutil.move(str(library), sibling / library.name)
    library.symlink_to(sibling / library.name, target_is_directory=True)
    assert sweep(env).conditions == ["path_outside_media_root"]

    # The path checked while it still led inside (realpath as it was before the link was
    # swapped in), then opened: the link is refused, the project outside never read.
    real_realpath = os.path.realpath

    def before_the_swap(path: Any, *args: Any, **kwargs: Any) -> Any:
        if os.fspath(path) == str(library):
            return str(library)
        return real_realpath(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os.path, "realpath", before_the_swap)
        result = sweep(env)
    assert result.conditions == ["path_refused"] and result.returned == 0
    [relinked] = [e for e in events(env) if e["cause"] == "library_relinked"]
    assert "is now a link" in relinked["message"]
    assert returned_edits(env.hub.log, env.hub.project_id, run) == []

    # Put back: the save comes back.
    library.unlink()
    shutil.move(str(sibling / library.name), library)
    assert sweep(env).returned == 1


def test_only_the_runner_s_hand_off_of_the_open_round_is_picked_up_once(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    run, handoff, project = _placed(env)
    db = project / "Project.db"
    export_op = load_state(env.hub.log, run).done_op("export")
    assert export_op is not None
    [result_event] = [
        e
        for e in env.hub.log.iter_events(types=[EVENT_RESULT])
        if e.payload.get("op_node") == export_op.op_id
    ]

    # A result written by someone other than the runner, naming the same op and project with
    # another plan, is ignored: only the runner's record is looked at.
    forged = dict(result_event.payload)
    forged["outputs"] = dict(forged["outputs"])
    forged["outputs"]["library_handoff"] = {**handoff, "plan_sha256": "b" * 64}
    for actor in (AGENT, HUMAN):
        env.hub.log.append(EVENT_RESULT, actor, forged, project_id=env.hub.project_id)
    save(db, trim_tail(5))
    result = sweep(env)
    assert (result.checked, result.returned, result.invalid) == (1, 1, 0), result
    [ret] = returned_edits(env.hub.log, env.hub.project_id, run)
    assert ret["base_plan"] == handoff["plan_sha256"]

    # Another worker recorded this save first (between this sweep's check and its record):
    # not recorded twice, and counted as repeated, not returned.
    with monkeypatch.context() as patch:
        patch.setattr(library_returns, "_seen", lambda *args: False)
        result = sweep(env)
    assert (result.returned, result.repeated) == (0, 1), result
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == 1

    # The review rejected: the round rolls back to the cut (round 2). A save of the round-1
    # project after that is not picked up.
    op = load_state(env.hub.log, run).latest_in_round("review", 1)
    assert op is not None and op.review_id is not None
    decide(env.hub.log, env.hub.project_id, op.review_id, "reject", "again", HUMAN)
    work(env)
    assert load_state(env.hub.log, run).current_round() == 2
    save(db, trim_tail(7))
    result = sweep(env)
    assert (result.closed, result.checked, result.returned) == (1, 0, 0), result
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == 1


def _found(**change: Any) -> library_returns._Found:
    handoff = library_returns.Handoff.model_validate(
        {
            "status": "placed",
            "library": "/media/Library",
            "user": "guest",
            "folder": "Show - Cut r1",
            "parts": ["Resolve Projects", "Users", "guest", "Projects", "Show - Cut r1"],
            "project_id": "id",
            "timeline": "Cut",
            "plan_sha256": "a" * 64,
            "placed_sha256": "b" * 64,
            "round": 1,
            "recipe_node": "nd_run",
            "media_map": {"/media": "E:\\Shows"},
            **change,
        }
    )
    event = SimpleNamespace(project_id="p")
    return library_returns._Found(event, handoff, "nd_export")  # type: ignore[arg-type]


class _State:
    """A recipe state as ``load_state`` folds it: only what ``_review_open`` asks."""

    def __init__(self, *, round_: int = 1, done: bool = True, decided: bool = False) -> None:
        self.recipe_status = "active"
        self._round, self._done, self._decided = round_, done, decided
        self.recipe_def = SimpleNamespace(steps=[SimpleNamespace(id="review", kind="gate")])

    def current_round(self) -> int:
        return self._round

    def op_by_id(self, op_id: str) -> Any:
        return SimpleNamespace(is_done=self._done) if op_id == "nd_export" else None

    def all_in_round(self, step: str, round_: int) -> list[Any]:
        return [SimpleNamespace(is_cancelled=False)]

    def gate_decision(self, op: Any) -> Any:
        return "approve" if self._decided else None


@pytest.mark.parametrize(
    ("state", "is_open"),
    [
        (_State(), True),
        (_State(round_=2), False),  # an earlier round's hand-off (no decision in its round)
        (_State(done=False), False),  # its export superseded by a reopen or a rollback
        (_State(decided=True), False),
    ],
)
def test_a_hand_off_is_open_only_in_its_own_round_while_its_export_stands(
    monkeypatch: pytest.MonkeyPatch, state: _State, is_open: bool
) -> None:
    import aar.recipes.state as state_module

    monkeypatch.setattr(state_module, "load_state", lambda *args, **kwargs: state)
    assert library_returns._review_open(None, _found()) is is_open  # type: ignore[arg-type]


@pytest.mark.parametrize("part", ["..", ".", "", "a/b", "a\x00b"])
def test_a_recorded_path_part_that_climbs_or_hides_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part: str
) -> None:
    monkeypatch.setenv("AAR_MEDIA_ROOT", str(tmp_path))
    (tmp_path / "Library" / "Resolve Projects" / "Users" / "guest").mkdir(parents=True)
    parts = ["Resolve Projects", "Users", part, "guest", "Show - Cut r1"]
    found = _found(library=str(tmp_path / "Library"), parts=parts)
    with (
        pytest.raises(library_returns._Condition) as caught,
        library_returns._project_dir(found.handoff),
    ):
        pass
    assert caught.value.code == "path_refused" and caught.value.cause == "bad_part"


def test_what_the_owner_sees_for_a_save_of_old_content_or_no_timeline_change(
    ccn_env: CcnEnv,
) -> None:
    """Pinned as DESKTOP-HANDOFF describes it: a save that brings the file back to content
    already returned, or to the placed content, records nothing new; a save that changes
    nothing in the timeline (the project renamed) is a return with no segment changed."""
    env = ccn_env
    run, _handoff, project = _placed(env)
    db = project / "Project.db"
    placed = db.read_bytes()
    save(db, trim_tail(5))
    assert sweep(env).returned == 1
    first = db.read_bytes()
    save(db, trim_tail(5))
    assert sweep(env).returned == 1

    def put_back(data: bytes) -> library_returns.Sweep:
        db.write_bytes(data)
        os.utime(db, (1_000_000_000, 1_000_000_000))
        return sweep(env)

    seq = env.hub.log.latest_seq()
    result = put_back(first)
    assert (result.returned, result.repeated) == (0, 1)
    result = put_back(placed)
    assert (result.returned, result.unchanged) == (0, 1)
    assert env.hub.log.latest_seq() == seq  # nothing recorded for either
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == 2

    conn = sqlite3.connect(db)
    conn.execute("UPDATE SM_Project SET ProjectName = 'Renamed in Resolve'")
    conn.commit()
    conn.close()
    os.utime(db, (1_000_000_000, 1_000_000_000))
    assert sweep(env).returned == 1
    renamed = returned_edits(env.hub.log, env.hub.project_id, run)[-1]
    diff = json.loads(env.hub.store.read_bytes(renamed["diff"]))
    assert Counter(s["status"] for s in diff["segments"]) == Counter(unchanged=1), diff
    assert not diff["added"] and not diff["foreign"]
