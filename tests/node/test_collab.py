"""Tests for collaboration: contention, agent branches, the owner guard, and markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from aar.hub import reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.node.broker import OpStatus
from aar.node.collab import contention, explain_op_contention
from aar.node.leases import Holder, LeaseManager, Scope
from aar.node.ops import OpError, submit_op
from aar.node.presence import PresenceGuard

from .fake_resolve import FakeClock, FakeResolve
from .ops_support import AGENT, Show, content_writes, human_settles, media

WALL = 1_800_000_000.0
B = Holder.agent("thread-2")
HUMAN = Holder.human("operator")
START = 108000


def _result(
    show: Show, name: str, params: dict, *, holder: Holder = AGENT, require_branch: bool = True
):
    op_id = submit_op(
        show.broker,
        name,
        params,
        target=show.locator("Show"),
        holder=holder,
        require_branch=require_branch,
        branch_registry=show.branch_registry,
    )
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None
    return result


def _refused(show: Show, name: str, params: dict, *, holder: Holder = AGENT) -> OpError:
    """Submit under the guard and return the refusal. The guard refuses at submit, so the
    op is never queued: the broker never switches project or timeline, and nothing runs."""
    calls = len(show.fake.calls)
    with pytest.raises(OpError) as err:
        submit_op(
            show.broker,
            name,
            params,
            target=show.locator("Show"),
            holder=holder,
            branch_registry=show.branch_registry,
        )
    assert show.broker.pending() == []
    assert show.fake.calls[calls:] == []
    return err.value


def _branch_with_content(show: Show, *, holder: Holder = AGENT):
    """A registered agent branch owned by ``holder``, with a 1000-frame item so markers fit."""
    branch = show.make_branch(holder=holder)
    timeline = show.project.timeline(branch.timeline.name)
    timeline.place("video", 1, show.cam_a, START, 1000)
    return branch, timeline


# -- contention: matches the broker --------------------------------------------------------


def test_contention_reports_a_human_instance_lease_the_broker_would_defer_on() -> None:
    show = Show()
    show.leases.acquire(HUMAN, Scope.INSTANCE, ("default",), ttl=None)
    show.presence.observe(show.gw.get_context(), show.clock())
    report = explain_op_contention(
        show.leases,
        show.presence,
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 1}]},
        target=show.locator("Show"),
        holder=AGENT,
        now=show.clock(),
    )
    op = submit_op(
        show.broker,
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 1}]},
        target=show.locator("Show"),
        holder=AGENT,
        require_branch=False,
    )
    outcome = show.broker.tick()
    assert report.blocked and any(lease.scope == "instance" for lease in report.leases)
    assert outcome.kind.value == "deferred"  # the broker agrees
    assert show.broker.result(op) is None


def test_contention_during_warmup_does_not_claim_a_human_is_active() -> None:
    guard = PresenceGuard(idle_window=60)
    report = contention(LeaseManager(FakeClock()), guard, holder=AGENT, project_id="p", now=1000.0)
    assert report.code == "no_observation" and report.human_active is False
    assert "a human is active" not in report.text


def test_a_human_lease_is_reported_with_who_what_and_a_zoned_since() -> None:
    show = Show()
    show.leases.acquire(HUMAN, Scope.PROJECT, (show.pid,), ttl=None)
    show.presence.observe(show.gw.get_context(), show.clock())
    report = contention(
        show.leases,
        show.presence,
        holder=AGENT,
        project_id=show.pid,
        now=show.clock(),
        wall_offset=WALL - 1000,
    )
    assert report.code == "held_by_human" and report.leases[0].holder_id == "operator"
    assert "held by human:operator" in report.text and "UTC" in report.text
    assert report.leases[0].since is not None


def test_clear_contention_reads_as_not_blocked() -> None:
    show = Show()
    show.presence.observe(show.gw.get_context(), show.clock())
    report = contention(
        show.leases, show.presence, holder=AGENT, project_id=show.pid, now=show.clock()
    )
    assert not report.blocked and report.code == "clear"


# -- agent branch timelines ----------------------------------------------------------------


def test_branch_names_the_copy_after_the_holder_and_records_the_handoff() -> None:
    show = Show()
    result = show.make_branch(holder=Holder.agent("thread-1"))
    assert result.created and result.version == 1 and result.agent_id == "thread-1"
    assert result.timeline.name == "Main [agent:thread-1 v1]"
    assert any(note.code == "merge_is_human" for note in result.notes)
    assert show.branch_registry.owned_by(show.pid, result.timeline.unique_id, "thread-1")


def test_branch_cuts_the_next_version_after_an_existing_v1_and_v2() -> None:
    show = Show()
    show.make_branch(holder=AGENT)
    show.make_branch(holder=AGENT, if_exists="new")
    third = show.make_branch(holder=AGENT, if_exists="new")
    assert third.version == 3 and third.timeline.name == "Main [agent:thread-ops v1]".replace(
        "v1", "v3"
    )


def test_branch_is_idempotent_a_rerun_finds_the_branch_it_made() -> None:
    show = Show()
    first = show.make_branch(holder=AGENT)
    again = show.make_branch(holder=AGENT)
    assert again.created is False and again.timeline.unique_id == first.timeline.unique_id
    assert len([t for t in show.project.timelines if "[agent:thread-ops v" in t.GetName()]) == 1


def test_branch_ownership_is_the_submitting_holder_not_a_caller_value() -> None:
    show = Show()
    # thread-2 cannot create or claim thread-1's branch: the owner is always the holder.
    result = show.make_branch(holder=B)
    assert result.agent_id == "thread-2" and result.timeline.name == "Main [agent:thread-2 v1]"
    assert show.branch_registry.owned_by(show.pid, result.timeline.unique_id, "thread-2")
    assert not show.branch_registry.owned_by(show.pid, result.timeline.unique_id, "thread-1")


def test_branch_refuses_a_holder_id_that_is_not_a_usable_owner() -> None:
    show = Show()
    with pytest.raises(OpError) as err:  # refused at submit (S11f LOW 8): never queued
        _result(
            show,
            "timeline.branch",
            {"source": show.ref("Main")},
            holder=Holder.agent("a\nb\x00" + "x" * 300),
            require_branch=False,
        )
    assert err.value.code == "invalid_agent_id"


def test_branch_source_must_carry_its_unique_id() -> None:
    show = Show()
    with pytest.raises(OpError) as err:
        submit_op(
            show.broker,
            "timeline.branch",
            {"source": {"name": "Main"}},
            target=show.locator("Show"),
            holder=AGENT,
            branch_registry=show.branch_registry,
        )
    assert "timeline_id_required" in str(err.value)


# -- the owner guard (fail-closed, by unique id) -------------------------------------------


def test_an_editing_op_is_refused_on_a_human_timeline_by_default() -> None:
    show = Show()
    error = _refused(show, "markers.add", {"timeline": show.ref("Main"), "markers": [{"frame": 5}]})
    assert error.code == "not_an_agent_branch"
    assert "require_branch" not in str(error.detail["fix_hint"])
    assert "timeline.branch" in str(error.detail["fix_hint"])
    assert show.main.GetMarkers() == {}


def test_a_timeline_that_merely_looks_like_a_branch_is_still_refused() -> None:
    fake = FakeResolve(
        {"Show": ["Main", "Main [agent]", "Interview [agent:review v2]"], "Other": ["Other main"]},
        current="Show",
        folder=("Shows",),
    )
    show = Show(fake)
    look_alike = show.project.timeline("Interview [agent:review v2]")
    error = _refused(
        show,
        "markers.add",
        {
            "timeline": {"name": look_alike.GetName(), "unique_id": look_alike.GetUniqueId()},
            "markers": [{"frame": 5}],
        },
        holder=Holder.agent("review"),
    )
    assert error.code == "not_an_agent_branch"
    assert look_alike.GetMarkers() == {}


def test_an_agent_cannot_edit_another_agents_branch() -> None:
    show = Show()
    branch = show.make_branch(holder=AGENT)  # owned by thread-ops
    error = _refused(
        show,
        "markers.add",
        {
            "timeline": {"name": branch.timeline.name, "unique_id": branch.timeline.unique_id},
            "markers": [{"frame": 5}],
        },
        holder=B,  # a different agent
    )
    assert error.code == "not_an_agent_branch"


def test_a_human_renaming_a_timeline_to_a_branch_name_does_not_grant_access() -> None:
    show = Show()
    show.main._name = "Main [agent:thread-ops v1]"  # a human renames in the Resolve UI
    human_settles(show)
    error = _refused(
        show,
        "markers.add",
        {"timeline": {"unique_id": show.main.GetUniqueId()}, "markers": [{"frame": 9}]},
    )
    assert error.code == "not_an_agent_branch"


def test_an_op_on_the_current_human_timeline_is_refused() -> None:
    show = Show()
    show.project.current_timeline = show.main
    error = _refused(show, "markers.add", {"markers": [{"frame": 11}]})
    assert error.code == "not_an_agent_branch"
    assert show.main.GetMarkers() == {}


def test_a_shared_state_op_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    show = Show()
    before = show.cam_a.GetClipProperty("Input Color Space")
    error = _refused(
        show,
        "clip.set_input_color_space",
        {"items": [media(show.cam_a)], "color_space": "Rec.709 (Scene)"},
    )
    assert error.code == "requires_approval"
    assert content_writes(show.fake) == []  # nothing was written to the shared media pool
    assert show.cam_a.GetClipProperty("Input Color Space") == before


def test_the_owner_may_edit_its_branch_under_the_guard() -> None:
    show = Show()
    branch, _ = _branch_with_content(show, holder=AGENT)
    result = _result(
        show,
        "markers.add",
        {
            "timeline": {"name": branch.timeline.name, "unique_id": branch.timeline.unique_id},
            "markers": [{"frame": 5}],
        },
    )
    assert result.status is OpStatus.OK


def test_the_override_lets_an_op_touch_a_human_timeline() -> None:
    show = Show()
    result = _result(
        show,
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 5}]},
        require_branch=False,
    )
    assert result.status is OpStatus.OK


def test_duplicate_and_branch_are_exempt_from_the_guard() -> None:
    show = Show()  # both create from a human source and must stay allowed under the guard
    dup = _result(show, "timeline.duplicate", {"source": show.ref("Main"), "name": "Main v2"})
    assert dup.status is OpStatus.OK
    branch = _result(show, "timeline.branch", {"source": show.ref("Main")})
    assert branch.status is OpStatus.OK


# -- markers.mirror_review ------------------------------------------------------------------


_NOTES = [
    {"note_id": "n1", "frame": 10, "timecode": "01:00:00:10", "text": "tighten the top"},
    {"note_id": "n2", "frame": 50, "timecode": "01:00:02:00", "text": "colour the wide"},
]


def _mirror(show: Show, timeline, notes, **kw):
    return show.run(
        "markers.mirror_review",
        {
            "timeline": {"name": timeline.GetName(), "unique_id": timeline.GetUniqueId()},
            "review_id": "rev1",
            "notes": notes,
            **kw,
        },
    )


def _adds(show: Show) -> int:
    return len([call for call in show.fake.calls if call[0] == "AddMarker"])


def test_mirror_creates_updates_and_prunes_without_duplicating() -> None:
    show = Show()
    _, branch = _branch_with_content(show)

    created = show.ok(
        "markers.mirror_review",
        {
            "timeline": {"name": branch.GetName(), "unique_id": branch.GetUniqueId()},
            "review_id": "rev1",
            "notes": _NOTES,
        },
    )
    assert [(m.note_id, m.frame, m.status) for m in created.markers] == [
        ("n1", 10, "added"),
        ("n2", 50, "added"),
    ]
    assert branch.GetMarkerByCustomData("aar:note:rev1:n1")["note"] == "tighten the top"

    before = _adds(show)
    same = _mirror(show, branch, _NOTES)
    assert [m.status for m in same.value.markers] == ["present", "present"]
    assert _adds(show) == before  # a re-run adds nothing

    changed = [{**_NOTES[0], "text": "tighten a lot more"}, _NOTES[1]]
    updated = _mirror(show, branch, changed)
    assert next(m.status for m in updated.value.markers if m.note_id == "n1") == "updated"
    assert len(branch.markers) == 2
    assert branch.GetMarkerByCustomData("aar:note:rev1:n1")["note"] == "tighten a lot more"

    pruned = _mirror(show, branch, [changed[0]])
    assert [(p.note_id, p.frame) for p in pruned.value.pruned] == [("n2", 50)]
    assert set(branch.markers) == {10}


def test_mirror_moving_onto_a_human_marker_is_refused_before_any_mutation() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    _mirror(show, branch, [{"note_id": "n1", "frame": 10, "text": "a"}])
    branch.AddMarker(20, "Blue", "", "human note", 1, "")
    result = _mirror(show, branch, [{"note_id": "n1", "frame": 20, "text": "a"}])
    assert result.status is OpStatus.FAILED and result.error.code == "marker_conflict"
    # The note's old marker at 10 is untouched: nothing was lost.
    assert branch.GetMarkerByCustomData("aar:note:rev1:n1")["note"] == "a"
    assert 10 in branch.markers and 20 in branch.markers


def test_mirror_partial_conflict_applies_nothing_and_reports_the_plan() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    _mirror(
        show,
        branch,
        [{"note_id": "n1", "frame": 10, "text": "a"}, {"note_id": "n2", "frame": 30, "text": "b"}],
    )
    branch.AddMarker(40, "Blue", "", "human", 1, "")
    before = dict(branch.markers)
    result = _mirror(
        show,
        branch,
        [
            {"note_id": "n1", "frame": 10, "text": "a CHANGED"},
            {"note_id": "n2", "frame": 40, "text": "b"},
        ],
    )
    assert result.status is OpStatus.FAILED and result.error.code == "marker_conflict"
    assert branch.markers == before  # pre-flight refused before touching anything


def test_mirror_refuses_duplicate_custom_data_rather_than_guess() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    _mirror(show, branch, [{"note_id": "n1", "frame": 10, "text": "a"}])
    branch.AddMarker(3, "Cyan", "", "a", 1, "aar:note:rev1:n1")  # a stray duplicate
    result = _mirror(show, branch, [{"note_id": "n1", "frame": 10, "text": "a2"}])
    assert result.status is OpStatus.FAILED and result.error.code == "duplicate_marker_custom_data"


def test_mirror_leaves_human_and_other_review_markers_alone_when_pruning() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    branch.AddMarker(5, "Blue", "", "human", 1, "")
    branch.AddMarker(6, "Red", "", "other review", 1, "aar:note:rev2:n1")
    added = _mirror(show, branch, [{"note_id": "n1", "frame": 10, "text": "a"}])
    assert added.status is OpStatus.OK, added.error
    assert [(m.note_id, m.frame, m.status) for m in added.value.markers] == [("n1", 10, "added")]
    pruned = _mirror(show, branch, [])  # prune everything of rev1
    assert pruned.status is OpStatus.OK, pruned.error
    assert [(p.note_id, p.frame) for p in pruned.value.pruned] == [("n1", 10)]
    assert sorted(branch.markers) == [5, 6]
    assert branch.markers[6]["customData"] == "aar:note:rev2:n1"  # the other review's, intact


def test_mirror_converts_render_frames_to_timeline_frames() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    result = _mirror(
        show, branch, [{"note_id": "n1", "frame": 48, "text": "two seconds"}], render_fps=24.0
    )
    # 48 frames at 24 fps is 2.0 s, which is 60 frames at the branch's 29.97.
    assert result.value.markers[0].frame == 60


def test_mirror_refuses_a_frame_outside_the_timeline() -> None:
    show = Show()
    _, branch = _branch_with_content(show)  # duration 1000
    result = _mirror(show, branch, [{"note_id": "n1", "frame": 5000, "text": "far"}])
    assert result.status is OpStatus.FAILED and result.error.code == "note_out_of_range"


@pytest.mark.parametrize(
    "params",
    [
        {"notes": [{"note_id": "n1\n", "frame": 10, "text": "x"}]},
        {"notes": [{"note_id": "n1", "frame": 10**30, "text": "x"}]},
        {"notes": [{"note_id": "n1", "frame": 10, "text": "A" * 5_000_000}]},
        {"notes": [{"note_id": "n1", "frame": 10, "text": "x\x00y"}]},
        {"notes": [{"note_id": "n1", "frame": 10, "text": "x"}], "render_fps": float("inf")},
        {"notes": [{"note_id": "n1", "frame": 10, "text": "x"}], "render_fps": 1e300},
    ],
    ids=["newline-id", "huge-frame", "huge-text", "control-text", "inf-fps", "over-max-fps"],
)
def test_mirror_params_refuse_bad_input_at_submit(params: dict[str, Any]) -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    ref = {"name": branch.GetName(), "unique_id": branch.GetUniqueId()}
    with pytest.raises(ValidationError):  # at submit, before any mutation
        submit_op(
            show.broker,
            "markers.mirror_review",
            {"timeline": ref, "review_id": "rev1", **params},
            target=show.locator("Show"),
            holder=AGENT,
            require_branch=False,
            branch_registry=show.branch_registry,
        )
    assert branch.markers == {}


def test_mirror_does_not_run_while_a_human_holds_the_branch_lease() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    show.leases.acquire(HUMAN, Scope.TIMELINE, (show.pid, branch.GetUniqueId()), ttl=None)
    op = submit_op(
        show.broker,
        "markers.mirror_review",
        {
            "timeline": {"name": branch.GetName(), "unique_id": branch.GetUniqueId()},
            "review_id": "rev1",
            "notes": [{"note_id": "n1", "frame": 10}],
        },
        target=show.locator("Show"),
        holder=AGENT,
        require_branch=False,
        branch_registry=show.branch_registry,
    )
    show.broker.run_until_idle()
    assert show.broker.result(op) is None and branch.markers == {}


# -- markers.read_review --------------------------------------------------------------------


def test_read_maps_mirrored_and_surfaces_human_markers() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    _mirror(show, branch, [_NOTES[0]])
    branch.AddMarker(20, "Blue", "", "please brighten the open", 1, "")
    result = show.ok(
        "markers.read_review", {"timeline": {"name": branch.GetName()}, "review_id": "rev1"}
    )
    assert [(m.note_id, m.frame) for m in result.mirrored] == [("n1", 10)]
    assert [(h.frame, h.text) for h in result.human] == [(20, "please brighten the open")]


def test_read_ignores_malformed_custom_data_without_crashing() -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    branch.AddMarker(30, "Green", "", "junk", 1, "aar:note:")  # prefix but no ids
    branch.AddMarker(40, "Red", "", "another review", 1, "aar:note:other:n9")
    result = show.ok(
        "markers.read_review", {"timeline": {"name": branch.GetName()}, "review_id": "rev1"}
    )
    assert result.mirrored == [] and result.human == [] and result.other == 2


def test_human_markers_flow_back_into_the_hub_review_as_notes(tmp_path: Path) -> None:
    show = Show()
    _, branch = _branch_with_content(show)
    branch.AddMarker(20, "Blue", "", "please brighten the open", 1, "")
    read = show.ok(
        "markers.read_review", {"timeline": {"name": branch.GetName()}, "review_id": "rev1"}
    )

    log = EventLog(tmp_path / "hub.db")
    store = ArtifactStore(tmp_path / "artifacts", log)
    agent, node = Actor("agent", "a"), Actor("node", "node-1")
    project = Registry(log).create_project("demo-show", Actor("human", "op")).id
    subject = WorkGraph(log, project).add_node("op", agent, title="cut").id
    render = store.put(b"render", actor=agent, mime="video/mp4", project_id=project)
    review = reviews.request_review(
        log, store, project, subject, [render], agent, title="v1", fps=29.97
    )

    for marker in read.human:
        reviews.add_note(log, project, review.id, marker.frame, marker.text, node)
    got = reviews.get_review(log, project, review.id)
    assert [(note.frame, note.text) for note in got.notes] == [(20, "please brighten the open")]
