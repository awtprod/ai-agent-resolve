"""Round-3 collaboration tests: the validator's d01, d03, d05, d06 and d07 probes, ported."""

from __future__ import annotations

from typing import Any

import pytest

from aar.node.branch_registry import BranchRegistry, OwnershipConflict
from aar.node.broker import OpStatus
from aar.node.leases import Holder
from aar.node.ops import OpError, submit_op

from .fake_resolve import FakeNode, FakeResolve
from .ops_support import AGENT, Show, content_writes, human_settles, media

START = 108000
B = Holder.agent("thread-2")


def _ref(timeline: Any) -> dict[str, str]:
    return {"name": timeline.GetName(), "unique_id": timeline.GetUniqueId()}


def _submit(show: Show, name: str, params: dict, *, holder: Holder = AGENT, **kw: Any) -> str:
    return submit_op(
        show.broker,
        name,
        params,
        target=kw.pop("target", None) or show.locator("Show"),
        holder=holder,
        branch_registry=kw.pop("registry", None) or show.branch_registry,
        **kw,
    )


def _run(show: Show, name: str, params: dict, **kw: Any):
    op_id = _submit(show, name, params, **kw)
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None
    return result


def _refusal(show: Show, name: str, params: dict, **kw: Any) -> OpError:
    with pytest.raises(OpError) as err:
        _submit(show, name, params, **kw)
    return err.value


# -- d01: an agent must never come to own a human timeline ------------------------------------


def test_a_human_timeline_carrying_the_agents_branch_name_is_never_reused_nor_owned() -> None:
    fake = FakeResolve(
        {"Show": ["Main", "Main [agent]", "Main [agent:thread-ops v1]"], "Other": ["Other main"]},
        current="Show",
        folder=("Shows",),
    )
    show = Show(fake)
    human = show.project.timeline("Main [agent:thread-ops v1]")  # made or renamed by a human
    edit = {"timeline": _ref(human), "markers": [{"frame": 5, "name": "x"}]}
    assert _refusal(show, "markers.add", edit).code == "not_an_agent_branch"

    result = _run(show, "timeline.branch", {"source": show.ref("Main")})
    assert result.status is OpStatus.OK and result.value.created
    # The look-alike only takes a version number: the agent gets a fresh v2.
    assert result.value.timeline.name == "Main [agent:thread-ops v2]"
    assert result.value.timeline.unique_id != human.GetUniqueId()
    assert show.branch_registry.owner(show.pid, human.GetUniqueId()) is None
    assert _refusal(show, "markers.add", edit).code == "not_an_agent_branch"
    assert human.GetMarkers() == {}


def test_after_a_restart_reuse_does_not_reclaim_an_old_branch_by_name() -> None:
    show = Show()
    old = show.make_branch()
    fresh = BranchRegistry()  # the node restarted: its in-memory registry is empty
    edit = {"timeline": old.timeline.model_dump(), "markers": [{"frame": 1}]}
    assert _refusal(show, "markers.add", edit, registry=fresh).code == "not_an_agent_branch"

    again = _run(show, "timeline.branch", {"source": show.ref("Main")}, registry=fresh)
    assert again.value.created and again.value.version == 2  # a new branch, never the old one
    assert fresh.owner(show.pid, old.timeline.unique_id) is None
    assert fresh.owned_by(show.pid, again.value.timeline.unique_id, AGENT.id)


def test_reuse_only_returns_a_branch_of_this_very_source() -> None:
    show = Show()
    first = show.make_branch()
    # A human renames the source away and makes a new "Main": same name, a new unique id.
    show.main._name = "Main (old)"
    new_main = show.main.DuplicateTimeline("Main")
    human_settles(show)
    result = _run(show, "timeline.branch", {"source": _ref(new_main)})
    assert result.value.created and result.value.timeline.unique_id != first.timeline.unique_id
    owner = show.branch_registry.owner(show.pid, result.value.timeline.unique_id)
    assert owner is not None and owner.source_timeline_id == new_main.GetUniqueId()


def test_ownership_is_keyed_by_project_too() -> None:
    show = Show()
    branch = show.make_branch()
    other = show.fake.project("Other")
    human_tl = other.timelines[0]
    human_tl._uid = branch.timeline.unique_id  # the same id in another project (e.g. a copy)
    target = show.locator("Other")
    edit = {"timeline": _ref(human_tl), "markers": [{"frame": 3}]}
    assert _refusal(show, "markers.add", edit, target=target).code == "not_an_agent_branch"
    assert not show.branch_registry.owned_by(target.project_id, branch.timeline.unique_id, AGENT.id)
    assert human_tl.GetMarkers() == {}


def test_a_branch_can_never_be_re_registered_to_another_agent() -> None:
    registry = BranchRegistry()
    registry.register("p", "tl-x", source_timeline_id="s", agent_id="A", version=1)
    registry.register("p", "tl-x", source_timeline_id="s", agent_id="A", version=1)  # idempotent
    with pytest.raises(OwnershipConflict):
        registry.register("p", "tl-x", source_timeline_id="s", agent_id="B", version=1)
    assert registry.owner("p", "tl-x").agent_id == "A"
    registry.register("q", "tl-x", source_timeline_id="s", agent_id="B", version=1)  # other project
    assert registry.owned_by("q", "tl-x", "B") and not registry.owned_by("q", "tl-x", "A")


@pytest.mark.parametrize(
    ("project", "branch", "source", "agent", "version"),
    [
        ("", "b", "s", "a", 1),
        ("p", "", "s", "a", 1),
        ("p", "b", "", "a", 1),
        ("p", "b", "s", "", 1),
        ("p", "b", "s", "a", 0),
    ],
    ids=["no-project", "no-branch", "no-source", "no-agent", "version-0"],
)
def test_the_registry_refuses_an_empty_id_or_a_version_below_one(
    project: str, branch: str, source: str, agent: str, version: int
) -> None:
    registry = BranchRegistry()
    with pytest.raises(ValueError):
        registry.register(
            project, branch, source_timeline_id=source, agent_id=agent, version=version
        )
    assert registry.owner(project, branch) is None


def test_concurrent_branch_requests_resolve_to_distinct_versions() -> None:
    show = Show()
    requests = [(AGENT, "reuse"), (AGENT, "reuse"), (AGENT, "new"), (B, "reuse")]
    ops = [
        _submit(show, "timeline.branch", {"source": show.ref("Main"), "if_exists": mode}, holder=h)
        for h, mode in requests
    ]
    show.broker.run_until_idle()
    got = [
        (show.broker.result(o).value.timeline.name, show.broker.result(o).value.created)
        for o in ops
    ]
    assert got == [
        ("Main [agent:thread-ops v1]", True),
        ("Main [agent:thread-ops v1]", False),
        ("Main [agent:thread-ops v2]", True),
        ("Main [agent:thread-2 v1]", True),
    ]


def test_branch_refuses_to_run_without_an_owner_registry() -> None:
    show = Show()
    with pytest.raises(OpError) as err:  # refused at submit (S11f LOW 8): never queued
        submit_op(
            show.broker,
            "timeline.branch",
            {"source": show.ref("Main")},
            target=show.locator("Show"),
            holder=AGENT,
            branch_registry=None,
        )
    assert err.value.code == "no_branch_registry"
    assert [t.GetName() for t in show.project.timelines] == ["Main", "Main [agent]"]


# -- d07: branch reuse honours the requested resolution ---------------------------------------


def test_branch_reuse_honours_the_requested_resolution() -> None:
    show = Show()
    landscape = show.make_branch()
    vertical = {"source": show.ref("Main"), "resolution": {"width": 1080, "height": 1920}}
    first = show.ok("timeline.branch", vertical)
    settings = show.project.timeline(first.timeline.name).GetSettings()
    assert first.created and first.version == 2
    assert (settings["timelineResolutionWidth"], settings["timelineResolutionHeight"]) == (
        "1080",
        "1920",
    )
    again = show.ok("timeline.branch", vertical)  # idempotent for the same request
    assert not again.created and again.timeline.unique_id == first.timeline.unique_id
    # No resolution means the source's (S11f LOW 6): the landscape v1, never the vertical v2.
    plain = show.ok("timeline.branch", {"source": show.ref("Main")})
    assert not plain.created and plain.version == 1
    assert plain.timeline.unique_id == landscape.timeline.unique_id


# -- d03: mirror atomicity --------------------------------------------------------------------


def _branch_show() -> tuple[Show, Any]:
    show = Show()
    branch = show.make_branch()
    timeline = show.project.timeline(branch.timeline.name)
    timeline.place("video", 1, show.cam_a, START, 1000)
    return show, timeline


def _mirror(show: Show, timeline: Any, notes: list[dict], **kw: Any):
    params = {"timeline": _ref(timeline), "review_id": "rev1", "notes": notes, **kw}
    return _run(show, "markers.mirror_review", params)


def _marks(timeline: Any) -> dict[int, tuple[str, str]]:
    return {f: (m["customData"], m["note"]) for f, m in sorted(timeline.markers.items())}


def _fail_nth(timeline: Any, method: str, n: int) -> None:
    """Make the ``n``-th call of ``timeline.method`` return False without acting."""
    real = getattr(timeline, method)
    count = [0]

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        count[0] += 1
        return False if count[0] == n else real(*args, **kwargs)

    setattr(timeline, method, wrapped)


def _note(note_id: str, frame: int, text: str = "t") -> dict[str, Any]:
    return {"note_id": note_id, "frame": frame, "text": text}


def _cd(note_id: str) -> str:
    return f"aar:note:rev1:{note_id}"


def test_moving_a_note_to_an_earlier_frame_works_and_a_rerun_is_present() -> None:
    show, tl = _branch_show()
    assert _mirror(show, tl, [_note("n1", 50)]).status is OpStatus.OK
    moved = _mirror(show, tl, [_note("n1", 10)])
    assert moved.status is OpStatus.OK, moved.error
    assert _marks(tl) == {10: (_cd("n1"), "t")}
    assert [(m.frame, m.status) for m in moved.value.markers] == [(10, "updated")]
    rerun = _mirror(show, tl, [_note("n1", 10)])
    assert [m.status for m in rerun.value.markers] == ["present"]


def test_moving_a_note_to_a_later_frame_works() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10)])
    assert _mirror(show, tl, [_note("n1", 50)]).status is OpStatus.OK
    assert _marks(tl) == {50: (_cd("n1"), "t")}


def test_a_move_adds_the_new_marker_before_deleting_the_old_one() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 50)])
    start = len(show.fake.calls)
    _mirror(show, tl, [_note("n1", 10)])
    writes = [
        c[:2] for c in show.fake.calls[start:] if c[0] in ("AddMarker", "DeleteMarkerAtFrame")
    ]
    assert writes == [("AddMarker", 10), ("DeleteMarkerAtFrame", 50)]
    assert not [c for c in show.fake.calls[start:] if c[0] == "DeleteMarkerByCustomData"]


def test_a_failed_add_on_a_move_keeps_the_old_marker() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 50, "keep me")])
    _fail_nth(tl, "AddMarker", 1)
    result = _mirror(show, tl, [_note("n1", 10, "keep me")])
    assert result.status is OpStatus.FAILED and result.error.code == "add_marker_failed"
    assert _marks(tl) == {50: (_cd("n1"), "keep me")}  # nothing was lost
    assert result.error.detail["applied"] == {"added": [], "deleted": []}


def test_swapping_two_notes_frames_works() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10, "a"), _note("n2", 20, "b")])
    result = _mirror(show, tl, [_note("n1", 20, "a"), _note("n2", 10, "b")])
    assert result.status is OpStatus.OK, result.error
    assert _marks(tl) == {10: (_cd("n2"), "b"), 20: (_cd("n1"), "a")}
    assert sorted((m.note_id, m.status) for m in result.value.markers) == [
        ("n1", "updated"),
        ("n2", "updated"),
    ]


def test_a_three_note_rotation_works() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("a", 10), _note("b", 20), _note("c", 30)])
    result = _mirror(show, tl, [_note("a", 20), _note("b", 30), _note("c", 10)])
    assert result.status is OpStatus.OK, result.error
    assert {f: cd for f, (cd, _) in _marks(tl).items()} == {
        10: _cd("c"),
        20: _cd("a"),
        30: _cd("b"),
    }


def test_a_new_note_on_a_pruned_notes_frame_works() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("old", 30, "stale")])
    result = _mirror(show, tl, [_note("new", 30, "fresh")])
    assert result.status is OpStatus.OK, result.error
    assert _marks(tl) == {30: (_cd("new"), "fresh")}
    assert [(p.note_id, p.frame) for p in result.value.pruned] == [("old", 30)]


def test_an_in_place_change_whose_add_fails_reports_the_delete() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10, "a"), _note("n2", 20, "b")])
    _fail_nth(tl, "AddMarker", 1)
    result = _mirror(show, tl, [_note("n1", 10, "a CHANGED"), _note("n2", 20, "b")])
    assert result.status is OpStatus.FAILED and result.error.code == "add_marker_failed"
    assert result.error.detail["applied"] == {
        "added": [],
        "deleted": [{"note_id": "n1", "frame": 10}],
    }
    assert _marks(tl) == {20: (_cd("n2"), "b")}  # exactly what the report says
    rerun = _mirror(show, tl, [_note("n1", 10, "a CHANGED"), _note("n2", 20, "b")])
    assert rerun.status is OpStatus.OK and _marks(tl)[10] == (_cd("n1"), "a CHANGED")


def test_a_failed_second_add_reports_the_first() -> None:
    show, tl = _branch_show()
    _fail_nth(tl, "AddMarker", 2)
    result = _mirror(show, tl, [_note("n1", 10, "a"), _note("n2", 20, "b")])
    assert result.status is OpStatus.FAILED
    assert result.error.detail["applied"] == {
        "added": [{"note_id": "n1", "frame": 10}],
        "deleted": [],
    }
    assert _marks(tl) == {10: (_cd("n1"), "a")}


def test_a_move_whose_old_delete_fails_reports_the_add() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10, "a")])
    _fail_nth(tl, "DeleteMarkerAtFrame", 1)
    result = _mirror(show, tl, [_note("n1", 40, "a")])
    assert result.status is OpStatus.FAILED and result.error.code == "marker_delete_failed"
    assert result.error.detail["applied"] == {
        "added": [{"note_id": "n1", "frame": 40}],
        "deleted": [],
    }
    # Both markers are there, as reported; a re-run refuses rather than guess which is right.
    assert set(tl.markers) == {10, 40}
    again = _mirror(show, tl, [_note("n1", 40, "a")])
    assert again.error.code == "duplicate_marker_custom_data"


def test_a_human_marker_landing_mid_run_is_reported_and_kept() -> None:
    show, tl = _branch_show()
    real_add = tl.AddMarker
    armed = [True]

    def racing_add(frame: int, *args: Any, **kwargs: Any) -> bool:
        if armed[0] and frame == 20:
            armed[0] = False
            real_add(20, "Blue", "", "human, mid-run", 1, "")
        return real_add(frame, *args, **kwargs)

    tl.AddMarker = racing_add
    result = _mirror(show, tl, [_note("n1", 10, "a"), _note("n2", 20, "b")])
    assert result.status is OpStatus.FAILED and result.error.code == "add_marker_failed"
    assert result.error.detail["applied"]["added"] == [{"note_id": "n1", "frame": 10}]
    assert _marks(tl)[20] == ("", "human, mid-run")  # the human's marker is untouched


def test_the_old_marker_is_deleted_only_if_it_is_still_ours() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10, "a")])
    tl.markers[10]["customData"] = ""  # a human re-purposed the marker between read and write
    real_read = tl.GetMarkers
    tl.GetMarkers = lambda: {**real_read(), 10: {**real_read()[10], "customData": _cd("n1")}}
    result = _mirror(show, tl, [_note("n1", 40, "a")])
    tl.GetMarkers = real_read
    assert result.status is OpStatus.FAILED and result.error.code == "marker_delete_failed"
    assert 10 in tl.markers  # never deleted another writer's marker


def test_with_prune_off_a_kept_marker_is_in_the_pre_flight() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("keep", 30, "k"), _note("n1", 10, "a")])
    before = dict(tl.markers)
    calls = len(content_writes(show.fake))
    result = _mirror(show, tl, [_note("n1", 30, "a")], prune=False)
    assert result.status is OpStatus.FAILED and result.error.code == "marker_conflict"
    assert tl.markers == before and len(content_writes(show.fake)) == calls  # nothing touched
    kept = _mirror(show, tl, [_note("n1", 20, "a")], prune=False)
    assert kept.status is OpStatus.OK and kept.value.pruned == []
    assert sorted(tl.markers) == [20, 30]


def test_a_note_on_the_frame_after_the_last_is_out_of_range() -> None:
    show, tl = _branch_show()
    duration = tl.GetEndFrame() - tl.GetStartFrame()
    result = _mirror(show, tl, [_note("n1", duration)])
    assert result.status is OpStatus.FAILED and result.error.code == "note_out_of_range"
    assert _mirror(show, tl, [_note("n1", duration - 1)]).status is OpStatus.OK


def test_marker_deletes_count_as_content_writes() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10)])
    start = len(content_writes(show.fake))
    _mirror(show, tl, [])  # prunes n1
    assert [c[0] for c in content_writes(show.fake)[start:]] == ["DeleteMarkerAtFrame"]


# -- d05: with the guard on, every mutating op works on a branch the agent owns ---------------


def _item(item: Any) -> dict[str, Any]:
    return {
        "track_type": item.track_type,
        "track": item.timeline.track_index_of(item),
        "item_id": item.GetUniqueId(),
    }


def test_every_mutating_op_works_on_an_owned_branch_under_the_guard() -> None:
    show = Show()
    reference = show.main.place("video", 1, show.cam_a, START, 100)
    reference.graph.nodes = [FakeNode("balance", tools=("Primaries",))]
    branch = show.make_branch()
    tl = show.project.timeline(branch.timeline.name)
    video = [tl.place("video", 1, show.cam_a, START + 200 + i * 50, 50) for i in range(2)]
    audio = tl.place("audio", 1, show.music, START + 200, 50)
    ref = _ref(tl)
    ops = [
        ("markers.add", {"timeline": ref, "markers": [{"frame": 210}]}),
        (
            "item.set_audio_gain",
            {"timeline": ref, "targets": [{"item": _item(audio), "gain_db": -6.0}]},
        ),
        (
            "item.set_fades",
            {"timeline": ref, "targets": [{"item": _item(video[0]), "fade_in": 10}]},
        ),
        (
            "grades.copy",
            {
                "timeline": ref,
                "groups": [
                    {
                        "camera": "A",
                        "reference": _item(reference),
                        "reference_timeline": _ref(show.main),
                        "targets": [_item(v) for v in video],
                    }
                ],
            },
        ),
        ("markers.mirror_review", {"timeline": ref, "review_id": "r1", "notes": [_note("n1", 5)]}),
        ("timeline.duplicate", {"source": ref, "name": "copy of branch"}),
    ]
    for name, params in ops:
        result = show.run(name, params, require_branch=True)
        assert result.status is OpStatus.OK, (name, result.error)
    fresh = Show()
    fresh_branch = fresh.make_branch()
    appended = fresh.run(
        "timeline.append_ranges",
        {
            "timeline": fresh_branch.timeline.model_dump(),
            "ranges": [{"media": media(fresh.cam_a), "src_in": 0, "src_out": 30}],
        },
        require_branch=True,
    )
    assert appended.status is OpStatus.OK, appended.error
    # A copy of the branch has a fresh unique id, so it is not the agent's: refused at submit.
    copy = show.project.timeline("copy of branch")
    edit = {"timeline": _ref(copy), "markers": [{"frame": 1}]}
    assert _refusal(show, "markers.add", edit).code == "not_an_agent_branch"


def test_item_refs_cannot_reach_a_human_timelines_items_through_an_owned_branch() -> None:
    show = Show()
    branch = show.make_branch()
    tl = show.project.timeline(branch.timeline.name)
    human_item = show.main.place("video", 2, show.cam_a, START + 500, 50)
    before = len(content_writes(show.fake))
    result = show.run(
        "item.set_fades",
        {"timeline": _ref(tl), "targets": [{"item": _item(human_item), "fade_in": 5}]},
        require_branch=True,
    )
    assert result.status is OpStatus.FAILED and result.error.code == "item_not_found"
    assert len(content_writes(show.fake)) == before


# -- d06: a refused op never touches the human's Resolve context ------------------------------


def test_a_refused_timeline_op_makes_no_resolve_call() -> None:
    show = Show()
    show.project.current_timeline = show.project.timeline("Main [agent]")
    human_settles(show)
    start = len(show.fake.calls)
    edit = {"timeline": show.ref("Main"), "markers": [{"frame": 1}]}
    assert _refusal(show, "markers.add", edit).code == "not_an_agent_branch"
    show.broker.run_until_idle()
    assert show.fake.calls[start:] == []  # no SetCurrentTimeline, no render-preset juggling
    assert show.project.current_timeline is show.project.timeline("Main [agent]")
    assert show.broker.pending() == []


def test_a_refused_shared_state_op_on_another_project_never_loads_it() -> None:
    show = Show()
    start = len(show.fake.calls)
    params = {"items": [media(show.cam_a)], "color_space": "Rec.709 (Scene)"}
    error = _refusal(show, "clip.set_input_color_space", params, target=show.locator("Other"))
    assert error.code == "requires_approval"
    show.broker.run_until_idle()
    names = {call[0] for call in show.fake.calls[start:]}
    assert not names & {"LoadProject", "SaveProject", "SetCurrentTimeline"}
    assert show.fake.calls[start:] == []


def test_the_next_version_follows_the_highest_even_with_a_hole() -> None:
    show = Show()
    show.make_branch()
    v2 = show.make_branch(if_exists="new")
    show.make_branch(if_exists="new")
    show.project.timelines = [  # a human deleted v2
        t for t in show.project.timelines if t.GetUniqueId() != v2.timeline.unique_id
    ]
    human_settles(show)
    nxt = show.make_branch(if_exists="new")
    assert nxt.created and nxt.version == 4  # O04: never reuse a number below the highest
