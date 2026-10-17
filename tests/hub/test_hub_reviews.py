"""Tests for review requests, timecoded notes and human decisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub import reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.common import HumanRequired, InvalidInput
from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.graph import ProtectedNode, WorkGraph
from aar.hub.registry import Registry

AGENT = Actor("agent", "agent-1")
NODE = Actor("node", "node-1")
HUMAN = Actor("human", "operator-1")
OTHER_HUMAN = Actor("human", "operator-2")


class Hub:
    def __init__(self, tmp_path: Path) -> None:
        self.log = EventLog(tmp_path / "hub.db")
        self.store = ArtifactStore(tmp_path / "artifacts", self.log)
        self.project = Registry(self.log).create_project("demo-show", HUMAN).id
        self.graph = WorkGraph(self.log, self.project)
        self.op = self.graph.add_node("op", AGENT, title="build the acme cut").id
        self.render = self.store.put(
            b"render-bytes", actor=AGENT, mime="video/mp4", project_id=self.project
        )

    def request(self, **kwargs: object) -> reviews.Review:
        params: dict[str, object] = {"title": "acme cut v1", "fps": 24.0}
        params.update(kwargs)
        return reviews.request_review(
            self.log, self.store, self.project, self.op, [self.render], AGENT, **params
        )

    def count(self, event_type: str) -> int:
        return len(self.log.read(types=[event_type]))


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return Hub(tmp_path)


def test_request_is_an_active_review_node_linked_to_its_subject(hub: Hub) -> None:
    review = hub.request(timeline="Main [agent:agent-1 v1]")
    graph = WorkGraph(hub.log, hub.project)
    node = graph.node(review.id)
    assert node is not None and node.kind == "review" and node.status == "active"
    assert node.payload["subject"] == hub.op
    assert node.payload["renders"] == [hub.render]
    edges = [e for e in graph.edges() if e.type == "reviews"]
    assert [(e.src, e.dst) for e in edges] == [(review.id, hub.op)]
    assert review.pending and review.timeline == "Main [agent:agent-1 v1]" and review.fps == 24.0
    assert review.requested_by == "agent:agent-1"


def test_request_rejects_bad_input(hub: Hub, tmp_path: Path) -> None:
    missing = "0" * 64
    with pytest.raises(reviews.UnknownArtifact):
        reviews.request_review(hub.log, hub.store, hub.project, hub.op, [missing], AGENT, title="x")
    with pytest.raises(InvalidInput):
        reviews.request_review(
            hub.log, hub.store, hub.project, hub.op, ["../../etc/passwd"], AGENT, title="x"
        )
    with pytest.raises(InvalidInput):
        reviews.request_review(hub.log, hub.store, hub.project, hub.op, [], AGENT, title="x")
    with pytest.raises(reviews.SubjectNotFound):
        reviews.request_review(
            hub.log, hub.store, hub.project, "nd_missing", [hub.render], AGENT, title="x"
        )
    with pytest.raises(reviews.ProjectNotFound):
        reviews.request_review(
            hub.log, hub.store, "prj_missing", hub.op, [hub.render], AGENT, title="x"
        )
    with pytest.raises(InvalidInput):
        hub.request(title="   ")
    with pytest.raises(InvalidInput):
        hub.request(fps=0)
    # Nothing was written by any refused request.
    assert reviews.list_reviews(hub.log) == []


def test_request_retry_with_same_key_returns_the_same_review(hub: Hub) -> None:
    first = hub.request(idempotency_key="req-1")
    again = hub.request(idempotency_key="req-1")
    assert again.id == first.id
    assert len(reviews.list_reviews(hub.log)) == 1
    assert len([e for e in WorkGraph(hub.log, hub.project).edges() if e.type == "reviews"]) == 1


def test_notes_are_anchored_at_frame_and_timecode(hub: Hub) -> None:
    review = hub.request(fps=24.0)
    derived = reviews.add_note(hub.log, hub.project, review.id, 24 * 61 + 5, "hold here", AGENT)
    assert derived.timecode == "00:01:01:05"
    explicit = reviews.add_note(
        hub.log, hub.project, review.id, 10, "label", HUMAN, timecode="01:00:00:10"
    )
    assert explicit.timecode == "01:00:00:10" and explicit.author == HUMAN
    stored = reviews.get_review(hub.log, hub.project, review.id)
    assert stored is not None
    assert [(n.frame, n.timecode, n.text) for n in stored.notes] == [
        (10, "01:00:00:10", "label"),
        (1469, "00:01:01:05", "hold here"),
    ]
    with pytest.raises(InvalidInput):
        reviews.add_note(hub.log, hub.project, review.id, 1, "x", AGENT, timecode="1:2:3")
    with pytest.raises(InvalidInput):
        reviews.add_note(hub.log, hub.project, review.id, -1, "x", AGENT)
    with pytest.raises(reviews.ReviewNotFound):
        reviews.add_note(hub.log, hub.project, "nd_missing", 1, "x", AGENT)


def test_note_needs_a_timecode_when_the_review_has_no_frame_rate(hub: Hub) -> None:
    review = hub.request(fps=None)
    with pytest.raises(InvalidInput, match="timecode"):
        reviews.add_note(hub.log, hub.project, review.id, 5, "x", AGENT)


def test_frames_to_timecode_uses_the_rounded_timebase() -> None:
    assert reviews.frames_to_timecode(0, 23.976) == "00:00:00:00"
    assert reviews.frames_to_timecode(24, 23.976) == "00:00:01:00"
    assert reviews.frames_to_timecode(30 * 3600 + 29, 29.97) == "01:00:00:29"


@pytest.mark.parametrize(("decision", "status"), [("approve", "done"), ("reject", "failed")])
def test_a_human_decision_moves_the_review_status(hub: Hub, decision: str, status: str) -> None:
    review = hub.request()
    recorded = reviews.decide(hub.log, hub.project, review.id, decision, "because", HUMAN)
    assert recorded.decision == decision and recorded.by == HUMAN
    stored = reviews.get_review(hub.log, hub.project, review.id)
    assert stored is not None
    assert stored.status == status and not stored.pending
    assert stored.decision == recorded
    # The graph is the one place status lives.
    node = WorkGraph(hub.log, hub.project).node(review.id)
    assert node is not None and node.status == status


@pytest.mark.parametrize("actor", [AGENT, NODE])
def test_agents_and_nodes_cannot_decide(hub: Hub, actor: Actor) -> None:
    review = hub.request()
    before = hub.log.latest_seq()
    with pytest.raises(HumanRequired):
        reviews.decide(hub.log, hub.project, review.id, "approve", "self-approve", actor)
    assert hub.log.latest_seq() == before
    stored = reviews.get_review(hub.log, hub.project, review.id)
    assert stored is not None and stored.status == "active" and stored.decision is None


def test_a_review_is_decided_only_once(hub: Hub) -> None:
    review = hub.request()
    first = reviews.decide(
        hub.log, hub.project, review.id, "approve", "ok", HUMAN, idempotency_key="k1"
    )
    # A different decision is a conflict that names the decision that stands.
    with pytest.raises(reviews.ReviewAlreadyDecided) as err:
        reviews.decide(hub.log, hub.project, review.id, "reject", "no", OTHER_HUMAN)
    assert err.value.decision == first
    # So is the same decision from someone else (a second, separate decision).
    with pytest.raises(reviews.ReviewAlreadyDecided):
        reviews.decide(hub.log, hub.project, review.id, "approve", "ok", OTHER_HUMAN)
    # The same decision under the same key is a no-op retry.
    retry = reviews.decide(
        hub.log, hub.project, review.id, "approve", "ok", HUMAN, idempotency_key="k1"
    )
    assert retry == first
    # Reusing the key for a different request is a conflict, not a silent no-op.
    with pytest.raises(IdempotencyConflict):
        reviews.decide(
            hub.log, hub.project, review.id, "approve", "changed", HUMAN, idempotency_key="k1"
        )
    assert hub.count(reviews.EVENT_DECIDED) == 1
    statuses = [
        e.payload["to"]
        for e in hub.log.read(types=["graph.node_status_changed"])
        if e.payload["node_id"] == review.id
    ]
    assert statuses == ["done"]


def test_a_stale_reader_cannot_decide_twice(hub: Hub, tmp_path: Path) -> None:
    review = hub.request()
    second = EventLog(tmp_path / "hub.db")  # another connection, opened before the decision
    reviews.decide(hub.log, hub.project, review.id, "approve", "ok", HUMAN)
    with pytest.raises(reviews.ReviewAlreadyDecided):
        reviews.decide(second, hub.project, review.id, "reject", "late", OTHER_HUMAN)
    assert hub.count(reviews.EVENT_DECIDED) == 1


def test_a_closed_review_cannot_be_decided_and_nothing_is_written(hub: Hub) -> None:
    review = hub.request()
    reviews.withdraw(hub.log, hub.project, review.id, "superseded by v2", AGENT)
    before = hub.log.latest_seq()
    with pytest.raises(reviews.ReviewNotOpen):
        reviews.decide(hub.log, hub.project, review.id, "approve", "ok", HUMAN)
    # The decision and its status change roll back together.
    assert hub.log.latest_seq() == before
    assert hub.count(reviews.EVENT_DECIDED) == 0


def test_a_pending_review_from_the_generic_graph_api_is_activated_then_decided(hub: Hub) -> None:
    node = hub.graph.add_node("review", AGENT, title="made by hand")
    hub.graph.add_edge("reviews", node.id, hub.op, AGENT)
    reviews.decide(hub.log, hub.project, node.id, "reject", "redo", HUMAN)
    statuses = [
        e.payload["to"]
        for e in hub.log.read(types=["graph.node_status_changed"])
        if e.payload["node_id"] == node.id
    ]
    assert statuses == ["active", "failed"]


def test_a_decision_forged_onto_the_log_by_an_agent_is_ignored(hub: Hub) -> None:
    review = hub.request()
    hub.log.append(
        reviews.EVENT_DECIDED,
        AGENT,
        {"review_id": review.id, "decision": "approve", "reason": "trust me"},
        project_id=hub.project,
    )
    stored = reviews.get_review(hub.log, hub.project, review.id)
    assert stored is not None and stored.decision is None and stored.pending
    # And it does not block the real human decision.
    reviews.decide(hub.log, hub.project, review.id, "reject", "no", HUMAN)
    stored = reviews.get_review(hub.log, hub.project, review.id)
    assert stored is not None and stored.decision is not None
    assert stored.decision.by == HUMAN


def test_list_reviews_pending_only_and_per_project(hub: Hub) -> None:
    open_review = hub.request(title="open")
    closed = hub.request(title="closed")
    reviews.decide(hub.log, hub.project, closed.id, "approve", "ok", HUMAN)
    other = Registry(hub.log).create_project("demo-show-2", HUMAN).id
    assert [r.id for r in reviews.list_reviews(hub.log, pending_only=True)] == [open_review.id]
    assert {r.id for r in reviews.list_reviews(hub.log)} == {open_review.id, closed.id}
    assert reviews.list_reviews(hub.log, other) == []


def test_a_decision_key_reused_in_another_project_conflicts(hub: Hub) -> None:
    # Two projects with a review node of the same explicit id, so the decision
    # payloads are identical and only the project tells the requests apart.
    other = Registry(hub.log).create_project("demo-show-2", HUMAN).id
    for project in (hub.project, other):
        WorkGraph(hub.log, project).add_node("review", AGENT, node_id="nd_shared", status="active")
    reviews.decide(hub.log, hub.project, "nd_shared", "approve", "ok", HUMAN, idempotency_key="k")
    with pytest.raises(IdempotencyConflict):
        reviews.decide(hub.log, other, "nd_shared", "approve", "ok", HUMAN, idempotency_key="k")
    # The other project's review was not silently treated as decided.
    stored = reviews.get_review(hub.log, other, "nd_shared")
    assert stored is not None and stored.decision is None and stored.status == "active"


def test_a_decided_review_cannot_be_reopened_even_by_a_human(hub: Hub) -> None:
    review = hub.request()
    reviews.decide(hub.log, hub.project, review.id, "reject", "redo the cold open", HUMAN)
    before = hub.log.latest_seq()
    with hub.log.transaction() as txn, pytest.raises(ProtectedNode):
        WorkGraph(hub.log, hub.project).set_status_in(txn, review.id, "active", HUMAN)
    assert hub.log.latest_seq() == before
    # A reopen forged straight onto the log is skipped too.
    forged = hub.log.append(
        "graph.node_status_changed",
        HUMAN,
        {"node_id": review.id, "to": "active"},
        project_id=hub.project,
    )
    graph = WorkGraph(hub.log, hub.project)
    node = graph.node(review.id)
    assert node is not None and node.status == "failed"
    assert forged.id in {item.event_id for item in graph.skipped()}
