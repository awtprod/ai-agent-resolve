"""S13h1 build ops: the branch guard, the no-destruction AST pin and a run through the job
runner.

- Every new op but ``media.import`` is refused on a timeline that is not the calling agent's
  branch, and runs on one it owns; ``media.import`` is the only new op with a guard exemption,
  and its text is pinned.
- An AST scan of ``build.py`` finds no call that deletes a track, clip, timeline or media, moves
  or relinks media, or changes the project or database, except the one ``DeleteClips`` that
  undoes a transition the same call made.
- The six ops run as one ``ops.sequence`` job through the real ``JobRunner`` over the fake, and
  a rerun of the job finds everything ``present``.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from aar.node.jobs import JobState
from aar.node.ops import OpError, build, get_op, make_hot_op
from aar.node.ops.framework import TimelineRef, _branch_refusal

from .fake_resolve import FakeTimeline, FakeTimelineItem, FakeTrack
from .jobs_support import AGENT_ID, MAIN_REF, JobWorld
from .ops_support import AGENT, Show, content_writes

START = 108000
NEW_OPS = (
    "media.import",
    "track.ensure",
    "transition.add",
    "item.set_color",
    "item.set_transform",
    "title.place",
)
TIMELINE_OPS = NEW_OPS[1:]


def ref(item: FakeTimelineItem) -> dict[str, Any]:
    return {
        "track_type": item.track_type,
        "track": item.timeline.track_index_of(item),
        "item_id": item.GetUniqueId(),
    }


def _stage(timeline: FakeTimeline, pool: Any) -> dict[str, Any]:
    """Two clips meeting at a cut on V1 and a title template: what every op below needs."""
    clip = pool.add_clip(("Footage", "Day 1"), "cam S", fps=29.97, frames=20000)
    template = pool.add_clip(
        ("Footage",), "lower third", kind="video", frames=600, title_text="Title"
    )
    a = timeline.place("video", 1, clip, START, 100, src_start=1000)
    b = timeline.place("video", 1, clip, START + 100, 100, src_start=5000)
    return {"a": a, "b": b, "template": template}


def _params(tl: dict[str, str], staged: dict[str, Any]) -> dict[str, dict[str, Any]]:
    a, b, template = staged["a"], staged["b"], staged["template"]
    return {
        "media.import": {"paths": ["X:/acme/media/a.mov"], "bin_path": ["Footage", "Day 2"]},
        "track.ensure": {
            "timeline": tl,
            "tracks": [
                {"type": "video", "index": 2, "name": "Titles"},
                {"type": "audio", "index": 2, "audio_format": "mono"},
            ],
        },
        "transition.add": {
            "timeline": tl,
            "targets": [{"item": ref(a), "position": "end", "alignment": "center", "frames": 12}],
        },
        "item.set_color": {
            "timeline": tl,
            "targets": [{"item": ref(a), "cdl": {"saturation": 0.9}, "lut": "Film/look.cube"}],
        },
        "item.set_transform": {"timeline": tl, "targets": [{"item": ref(b), "zoom": 1.2}]},
        "title.place": {
            "timeline": tl,
            "titles": [
                {
                    "template": {"unique_id": template.GetUniqueId()},
                    "track": 2,
                    "record_in": 30,
                    "record_out": 120,
                    "lines": ["Acme Name"],
                }
            ],
        },
    }


# -- guard exemption ---------------------------------------------------------------------------


def test_media_import_is_the_only_new_op_with_a_guard_exemption() -> None:
    exempt = {name: get_op(name).guard_exemption for name in NEW_OPS}
    assert {name for name, reason in exempt.items() if reason is not None} == {"media.import"}
    assert exempt["media.import"] == (
        "additive: only adds media-pool items to the named bin; never alters, moves, relinks "
        "or deletes existing items or any timeline"
    )
    for name in NEW_OPS:
        spec = get_op(name)
        assert (spec.hot, spec.mutates, spec.creates_timeline) == (True, True, False), name
        assert spec.submit_check is None, name
    for name in TIMELINE_OPS:
        field = get_op(name).params.model_fields["timeline"]
        assert field.annotation is TimelineRef and field.is_required(), name


# -- the branch guard --------------------------------------------------------------------------


def _show_with_stage() -> tuple[Show, dict[str, Any]]:
    show = Show()
    show.fake.edit_files = {"X:/acme/media/a.mov": {}}
    return show, _stage(show.main, show.project.media_pool)


@pytest.mark.parametrize("name", TIMELINE_OPS)
def test_a_new_op_on_a_human_timeline_is_refused_at_submit(name: str) -> None:
    show, staged = _show_with_stage()
    params = _params(show.ref("Main"), staged)[name]
    with pytest.raises(OpError) as err:
        make_hot_op(
            name,
            params,
            target=show.locator(),
            holder=AGENT,
            require_branch=True,
            branch_registry=show.branch_registry,
        )
    assert err.value.code == "not_an_agent_branch"
    assert show.broker.pending() == [] and content_writes(show.fake) == []


@pytest.mark.parametrize("name", TIMELINE_OPS)
def test_a_new_op_without_its_timeline_id_is_refused_at_submit(name: str) -> None:
    show, staged = _show_with_stage()
    params = _params({"name": "Main"}, staged)[name]
    with pytest.raises(OpError) as err:
        make_hot_op(name, params, target=show.locator(), holder=AGENT, require_branch=True)
    assert err.value.code == "timeline_id_required"


def test_media_import_passes_the_guard_only_by_its_exemption() -> None:
    show, staged = _show_with_stage()
    params = _params(show.ref("Main"), staged)["media.import"]
    hot = make_hot_op(
        "media.import",
        params,
        target=show.locator(),
        holder=AGENT,
        require_branch=True,
        branch_registry=show.branch_registry,
    )
    assert hot.label == "media.import" and hot.timeline_id is None
    spec = get_op("media.import")
    refusal = _branch_refusal(
        dataclasses.replace(spec, guard_exemption=None),
        spec.params.model_validate(params),
        holder=AGENT,
        branch_registry=show.branch_registry,
        project_id=show.pid,
    )
    assert refusal is not None and refusal.code == "requires_approval"


def test_the_new_ops_run_with_the_guard_on_the_agents_own_branch() -> None:
    show = Show()
    show.fake.edit_files = {"X:/acme/media/a.mov": {}}
    branch = show.make_branch()
    timeline = show.project.timeline(branch.timeline.name)
    staged = _stage(timeline, show.project.media_pool)
    tl = {"name": branch.timeline.name, "unique_id": branch.timeline.unique_id}
    params = _params(tl, staged)
    for name in NEW_OPS:
        show.ok(name, params[name], require_branch=True)
    (title,) = timeline.items("video", 2)
    assert title.comps[0].tools[0].inputs["StyledText"] == "Acme Name"
    assert show.main.items("video", 1) == [] and show.main.GetTrackCount("video") == 1


# -- no destruction ----------------------------------------------------------------------------

FORBIDDEN_CALLS = frozenset(
    {
        # the brief's list
        "DeleteTrack",
        "DeleteClips",
        "DeleteTimelines",
        "SetCurrentDatabase",
        "LoadProject",
        "SetCurrentProject",
        # moving, relinking or changing existing media, and deleting bins
        "MoveClips",
        "MoveFolders",
        "RelinkClips",
        "UnlinkClips",
        "ReplaceClip",
        "DeleteFolders",
        "SetClipProperty",
    }
)
ROLLBACK = "_undo_transition"


def _calls_by_function(tree: ast.AST) -> list[tuple[str, str]]:
    """``(enclosing function, name)`` for every call of an attribute or name, and for every
    string constant naming a forbidden call (a ``getattr`` route)."""
    found: list[tuple[str, str]] = []

    def visit(node: ast.AST, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, ast.FunctionDef) else where
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute):
                    found.append((inner, func.attr))
                elif isinstance(func, ast.Name):
                    found.append((inner, func.id))
            if isinstance(child, ast.Constant) and child.value in FORBIDDEN_CALLS:
                found.append((inner, f"str:{child.value}"))
            visit(child, inner)

    visit(tree, "<module>")
    return found


def test_build_ops_call_nothing_destructive_but_the_transition_rollback() -> None:
    source = Path(build.__file__).read_text(encoding="utf-8")
    calls = _calls_by_function(ast.parse(source))
    bad = [
        (where, name) for where, name in calls if name in FORBIDDEN_CALLS or name.startswith("str:")
    ]
    assert bad == [(ROLLBACK, "DeleteClips")]
    # The rollback deletes only the one transition it is given, never with a ripple.
    rollback = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == ROLLBACK
    )
    (delete,) = [
        n
        for n in ast.walk(rollback)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "DeleteClips"
    ]
    assert ast.unparse(delete.args[0]) == "[made]"
    assert isinstance(delete.args[1], ast.Constant) and delete.args[1].value is False
    # And only transition.add reaches it.
    callers = {where for where, name in calls if name == ROLLBACK}
    assert callers == {"_add_one"}


def test_the_scan_sees_a_forbidden_call() -> None:
    planted = "def f(t):\n    t.DeleteTrack('audio', 1)\n    getattr(t, 'LoadProject')('x')\n"
    calls = _calls_by_function(ast.parse(planted))
    assert ("f", "DeleteTrack") in calls and ("f", "str:LoadProject") in calls


# -- ops.sequence through the job runner -------------------------------------------------------


def _job(world: JobWorld, params: dict[str, dict[str, Any]]) -> Any:
    steps = [{"op": name, "params": params[name]} for name in NEW_OPS]
    job_id = world.runner.submit(
        "ops.sequence", {"steps": steps}, world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None
    return view


def test_the_six_ops_run_as_one_ops_sequence_job_and_rerun_present(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.fake.edit_files = {"X:/acme/media/a.mov": {}}
    staged = _stage(world.main, world.project.media_pool)
    params = _params(MAIN_REF, staged)
    view = _job(world, params)
    assert view.state == JobState.SUCCEEDED.value, view.error
    assert view.steps_total == 6 and view.current_step == 6
    imported, tracks, transitions, colour, transform, titles = view.results
    assert imported["media"][0]["status"] == "imported"
    assert [t["created"] for t in tracks["tracks"]] == [True, True]
    assert transitions["transitions"][0]["status"] == "added"
    assert colour["items"][0]["cdl"]["verified"] is False
    assert colour["items"][0]["lut"]["status"] == "set"
    assert transform["items"][0]["status"] == "applied"
    assert titles["titles"][0]["status"] == "placed"
    assert world.main.GetTrackCount("video") == 2
    assert [t.name for t in world.main.tracks["video"]] == ["Video 1", "Titles"]

    world.fake.calls.clear()
    again = _job(world, params)
    assert again.state == JobState.SUCCEEDED.value, again.error
    assert again.results[0]["media"][0]["status"] == "present"
    assert [t["changed"] for t in again.results[1]["tracks"]] == [[], []]
    assert again.results[2]["transitions"][0]["status"] == "present"
    assert again.results[3]["items"][0]["lut"]["status"] == "present"
    assert again.results[4]["items"][0]["status"] == "present"
    assert again.results[5]["titles"][0]["status"] == "present"
    # The CDL is sent again (no getter, absolute values); nothing else is written.
    assert [c[0] for c in content_writes(world.fake)] == ["SetCDL"]


def test_a_build_op_on_a_human_timeline_waits_for_approval_in_a_job(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, own_timelines=False)
    staged = _stage(world.main, world.project.media_pool)
    params = _params(MAIN_REF, staged)["track.ensure"]
    job_id = world.runner.submit(
        "ops.sequence",
        {"steps": [{"op": "track.ensure", "params": params}]},
        world.locator(),
        agent_id=AGENT_ID,
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.WAITING.value
    assert view.waiting["detail"]["guard"]["code"] == "not_an_agent_branch"
    assert world.main.GetTrackCount("video") == 1 and isinstance(
        world.main.tracks["video"][0], FakeTrack
    )
