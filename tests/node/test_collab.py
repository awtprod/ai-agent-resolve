"""Tests for collaboration: contention, agent branches, and review-note markers."""

from __future__ import annotations

from pathlib import Path

from aar.hub import reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.registry import Registry
from aar.node.broker import OpStatus
from aar.node.collab import contention, explain_op_contention
from aar.node.leases import Holder, Scope
from aar.node.ops import submit_op

from .ops_support import AGENT, Show

WALL_OFFSET = 1_800_000_000.0
HUMAN = Holder.human("operator")


def _result(show: Show, name: str, params: dict, *, require_branch: bool):
    op_id = submit_op(
        show.broker,
        name,
        params,
        target=show.locator("Show"),
        holder=AGENT,
        require_branch=require_branch,
    )
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None
    return result


# -- contention: who holds what, since when ----------------------------------


def test_a_lease_held_by_a_human_is_reported_in_structured_fields_and_text() -> None:
    show = Show()
    branch = show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1"})
    branch_id = branch.timeline.unique_id
    show.leases.acquire(HUMAN, Scope.TIMELINE, (show.pid, branch_id), ttl=None)

    report = explain_op_contention(
        show.leases,
        show.presence,
        "markers.add",
        {
            "timeline": {"name": branch.timeline.name, "unique_id": branch_id},
            "markers": [{"frame": 10}],
        },
        target=show.locator("Show"),
        holder=AGENT,
        now=show.clock(),
        wall_offset=WALL_OFFSET,
    )

    assert report.blocked and report.code == "held_by_human"
    assert report.leases[0].holder_kind == "human" and report.leases[0].holder_id == "operator"
    assert report.leases[0].since is not None
    assert "held by human:operator" in report.text and "since" in report.text


def test_a_present_human_is_reported_even_with_no_lease() -> None:
    show = Show()
    show.fake.human_set_page("color")  # a human moved the page since the broker last looked
    show.presence.observe(show.gw.get_context(), show.clock())

    report = contention(
        show.leases, show.presence, holder=AGENT, project_id=show.pid, now=show.clock()
    )

    assert report.blocked and report.human_active and report.code == "human_active"
    assert "a human is active on this node" in report.text


def test_clear_contention_reads_as_not_blocked() -> None:
    show = Show()
    show.presence.observe(show.gw.get_context(), show.clock())  # a fresh, quiet observation
    report = contention(
        show.leases, show.presence, holder=AGENT, project_id=show.pid, now=show.clock()
    )
    assert not report.blocked and report.code == "clear"
    assert report.text == "no lease or presence blocks this work"


# -- agent branch timelines --------------------------------------------------


def test_branch_names_the_copy_and_records_the_handoff() -> None:
    show = Show()
    result = show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1"})

    assert result.created and result.version == 1
    assert result.timeline.name == "Main [agent:thread-1 v1]"
    assert result.source.name == "Main"
    assert any(note.code == "merge_is_human" for note in result.notes)
    assert (
        show.project.timeline("Main [agent:thread-1 v1]").GetUniqueId() == result.timeline.unique_id
    )


def test_branch_cuts_the_next_version_after_an_existing_v1_and_v2() -> None:
    show = Show()
    show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1"})  # v1
    show.ok(
        "timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1", "if_exists": "new"}
    )  # v2
    third = show.ok(
        "timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1", "if_exists": "new"}
    )

    assert third.version == 3 and third.timeline.name == "Main [agent:thread-1 v3]"


def test_branch_is_idempotent_a_rerun_finds_the_branch_it_already_made() -> None:
    show = Show()
    first = show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1"})
    again = show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1"})

    assert again.created is False and again.version == 1
    assert again.timeline.unique_id == first.timeline.unique_id
    branches = [t for t in show.project.timelines if "[agent:thread-1 v" in t.GetName()]
    assert len(branches) == 1


def test_branches_are_per_agent() -> None:
    show = Show()
    one = show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1"})
    two = show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-2"})

    assert one.timeline.name == "Main [agent:thread-1 v1]"
    assert two.timeline.name == "Main [agent:thread-2 v1]"


# -- editing ops refuse a non-branch timeline --------------------------------


def test_an_editing_op_refuses_a_non_branch_timeline_when_required() -> None:
    show = Show()
    result = _result(
        show,
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 10, "name": "x"}]},
        require_branch=True,
    )
    assert result.status is OpStatus.FAILED
    assert result.error.code == "not_an_agent_branch"
    assert result.error.detail["timeline"] == "Main"


def test_an_editing_op_is_allowed_on_an_agent_branch() -> None:
    show = Show()
    branch = show.ok("timeline.branch", {"source": show.ref("Main"), "agent_id": "thread-1"})
    result = _result(
        show,
        "markers.add",
        {
            "timeline": {"name": branch.timeline.name, "unique_id": branch.timeline.unique_id},
            "markers": [{"frame": 10, "name": "x"}],
        },
        require_branch=True,
    )
    assert result.status is OpStatus.OK


def test_the_override_lets_an_editing_op_touch_a_production_timeline() -> None:
    show = Show()
    result = _result(
        show,
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 10, "name": "x"}]},
        require_branch=False,
    )
    assert result.status is OpStatus.OK


# -- review-note markers: mirror --------------------------------------------


_NOTES = [
    {"note_id": "n1", "frame": 10, "timecode": "01:00:00:10", "text": "tighten the top"},
    {"note_id": "n2", "frame": 50, "timecode": "01:00:02:00", "text": "colour the wide"},
]


def _adds(show: Show) -> int:
    return len([call for call in show.fake.calls if call[0] == "AddMarker"])


def test_mirror_creates_updates_and_prunes_markers_without_duplicating() -> None:
    show = Show()
    tl = show.ref("Main [agent]")
    branch = show.project.timeline("Main [agent]")

    created = show.ok(
        "markers.mirror_review", {"timeline": tl, "review_id": "rev1", "notes": _NOTES}
    )
    assert [(m.note_id, m.frame, m.status) for m in created.markers] == [
        ("n1", 10, "added"),
        ("n2", 50, "added"),
    ]
    assert branch.GetMarkerByCustomData("aar:note:rev1:n1")["note"] == "tighten the top"

    before = _adds(show)
    same = show.ok("markers.mirror_review", {"timeline": tl, "review_id": "rev1", "notes": _NOTES})
    assert [m.status for m in same.markers] == ["present", "present"]
    assert _adds(show) == before  # a re-run adds nothing

    changed = [{**_NOTES[0], "text": "tighten a lot more"}, _NOTES[1]]
    updated = show.ok(
        "markers.mirror_review", {"timeline": tl, "review_id": "rev1", "notes": changed}
    )
    assert next(m.status for m in updated.markers if m.note_id == "n1") == "updated"
    assert len(branch.markers) == 2
    assert branch.GetMarkerByCustomData("aar:note:rev1:n1")["note"] == "tighten a lot more"

    pruned = show.ok(
        "markers.mirror_review", {"timeline": tl, "review_id": "rev1", "notes": [changed[0]]}
    )
    assert [(p.note_id, p.frame) for p in pruned.pruned] == [("n2", 50)]
    assert set(branch.markers) == {10}


def test_mirror_converts_render_frames_to_timeline_frames() -> None:
    show = Show()  # the branch runs at 29.97
    result = show.ok(
        "markers.mirror_review",
        {
            "timeline": show.ref("Main [agent]"),
            "review_id": "rev1",
            "notes": [{"note_id": "n1", "frame": 48, "text": "at two seconds"}],
            "render_fps": 24.0,
        },
    )
    # 48 frames at 24 fps is 2.0s, which is 60 frames at 29.97.
    assert result.markers[0].frame == 60


# -- review-note markers: read back -----------------------------------------


def test_read_maps_mirrored_markers_back_and_surfaces_human_markers() -> None:
    show = Show()
    tl = show.ref("Main [agent]")
    branch = show.project.timeline("Main [agent]")
    show.ok("markers.mirror_review", {"timeline": tl, "review_id": "rev1", "notes": [_NOTES[0]]})
    branch.AddMarker(20, "Blue", "", "please brighten the open", 1, "")  # a human's own marker

    result = show.ok("markers.read_review", {"timeline": tl, "review_id": "rev1"})

    assert [(m.note_id, m.frame) for m in result.mirrored] == [("n1", 10)]
    assert [(h.frame, h.text) for h in result.human] == [(20, "please brighten the open")]


def test_read_ignores_a_malformed_custom_data_without_crashing() -> None:
    show = Show()
    branch = show.project.timeline("Main [agent]")
    branch.AddMarker(30, "Green", "", "junk", 1, "aar:note:")  # our prefix, but no ids
    branch.AddMarker(40, "Red", "", "another review", 1, "aar:note:other:n9")

    result = show.ok(
        "markers.read_review", {"timeline": show.ref("Main [agent]"), "review_id": "rev1"}
    )

    assert result.mirrored == [] and result.human == []
    assert result.other == 2  # both ignored, the read did not raise


def test_human_markers_flow_back_into_the_hub_review_as_notes(tmp_path: Path) -> None:
    show = Show()
    branch = show.project.timeline("Main [agent]")
    branch.AddMarker(20, "Blue", "", "please brighten the open", 1, "")
    read = show.ok(
        "markers.read_review", {"timeline": show.ref("Main [agent]"), "review_id": "rev1"}
    )

    log = EventLog(tmp_path / "hub.db")
    store = ArtifactStore(tmp_path / "artifacts", log)
    human = Actor("human", "operator")
    node = Actor("node", "node-1")
    project = Registry(log).create_project("demo-show", human).id
    from aar.hub.graph import WorkGraph

    subject = WorkGraph(log, project).add_node("op", Actor("agent", "a"), title="cut").id
    render = store.put(b"render", actor=Actor("agent", "a"), mime="video/mp4", project_id=project)
    review = reviews.request_review(
        log, store, project, subject, [render], Actor("agent", "a"), title="v1", fps=29.97
    )

    for marker in read.human:
        reviews.add_note(log, project, review.id, marker.frame, marker.text, node)

    got = reviews.get_review(log, project, review.id)
    assert [(note.frame, note.text) for note in got.notes] == [(20, "please brighten the open")]
