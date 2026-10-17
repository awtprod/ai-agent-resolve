"""Tests for the per-project work graph projection and state machine."""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.graph import DuplicateNodeId, IllegalTransition, WorkGraph

ACTOR = Actor("agent", "editor-1")

_STATUS_EVENTS = ["graph.node_status_changed"]


def _graph(tmp_path: Path, project_id: str = "prj_1") -> tuple[EventLog, WorkGraph]:
    log = EventLog(tmp_path / "hub.db")
    return log, WorkGraph(log, project_id)


def test_add_node_rejects_unknown_kind_and_status(tmp_path: Path) -> None:
    _, graph = _graph(tmp_path)
    with pytest.raises(ValueError):
        graph.add_node("widget", ACTOR)
    with pytest.raises(ValueError):
        graph.add_node("op", ACTOR, status="exploded")


def test_status_machine_allows_legal_and_rejects_illegal_transitions(tmp_path: Path) -> None:
    log, graph = _graph(tmp_path)
    node = graph.add_node("op", ACTOR)
    assert node.status == "pending"

    graph.set_status(node.id, "active", ACTOR)
    assert graph.node(node.id).status == "active"

    with pytest.raises(IllegalTransition):
        graph.set_status(node.id, "pending", ACTOR)  # active -> pending is not allowed

    graph.set_status(node.id, "done", ACTOR)
    with pytest.raises(IllegalTransition):
        graph.set_status(node.id, "active", ACTOR)  # done is terminal

    assert graph.node(node.id).status == "done"
    # The rejected transitions wrote nothing: only the two legal changes are logged.
    assert len(log.read(types=_STATUS_EVENTS)) == 2


def test_transition_table_covers_blocked_failed_and_cancelled(tmp_path: Path) -> None:
    _, graph = _graph(tmp_path)
    node = graph.add_node("op", ACTOR)

    graph.set_status(node.id, "blocked", ACTOR)  # pending -> blocked
    graph.set_status(node.id, "active", ACTOR)  # blocked -> active
    graph.set_status(node.id, "failed", ACTOR)  # active -> failed
    graph.set_status(node.id, "active", ACTOR)  # failed -> active
    graph.set_status(node.id, "cancelled", ACTOR)  # active -> cancelled
    assert graph.node(node.id).status == "cancelled"

    # cancelled is terminal: nothing is reachable from it.
    for target in ("active", "pending", "done", "blocked", "failed"):
        with pytest.raises(IllegalTransition):
            graph.set_status(node.id, target, ACTOR)

    # blocked and failed cannot jump straight to done.
    blocked = graph.add_node("op", ACTOR)
    graph.set_status(blocked.id, "blocked", ACTOR)
    with pytest.raises(IllegalTransition):
        graph.set_status(blocked.id, "done", ACTOR)

    failed = graph.add_node("op", ACTOR)
    graph.set_status(failed.id, "active", ACTOR)
    graph.set_status(failed.id, "failed", ACTOR)
    with pytest.raises(IllegalTransition):
        graph.set_status(failed.id, "done", ACTOR)


def test_set_status_on_unknown_node_raises(tmp_path: Path) -> None:
    _, graph = _graph(tmp_path)
    with pytest.raises(KeyError):
        graph.set_status("nd_missing", "active", ACTOR)


def test_set_status_retry_with_same_key_returns_existing(tmp_path: Path) -> None:
    log, graph = _graph(tmp_path)
    node = graph.add_node("op", ACTOR)
    graph.set_status(node.id, "active", ACTOR)
    graph.set_status(node.id, "done", ACTOR, idempotency_key="finish")

    # The same-key retry must return the existing node, not re-check (done -> done).
    again = graph.set_status(node.id, "done", ACTOR, idempotency_key="finish")
    assert again.status == "done"
    assert len(log.read(types=_STATUS_EVENTS)) == 2  # active, done -- the retry wrote nothing


def test_reused_explicit_node_id_is_rejected_and_does_not_reset(tmp_path: Path) -> None:
    _, graph = _graph(tmp_path)
    graph.add_node("op", ACTOR, node_id="nd_fixed")
    graph.set_status("nd_fixed", "active", ACTOR)
    graph.set_status("nd_fixed", "done", ACTOR)

    with pytest.raises(DuplicateNodeId):
        graph.add_node("op", ACTOR, node_id="nd_fixed")
    assert graph.node("nd_fixed").status == "done"  # not reset to pending


def test_second_writer_cannot_append_illegal_history(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    g1 = WorkGraph(log, "prj_1")
    g1.add_node("op", ACTOR, node_id="nd_x")
    g1.set_status("nd_x", "active", ACTOR)

    g2 = WorkGraph(log, "prj_1")  # g2 is built now, seeing nd_x as active
    g1.set_status("nd_x", "done", ACTOR)  # g1 advances; g2's view is now stale

    # g2 must catch up before validating, see "done", and reject done -> blocked.
    with pytest.raises(IllegalTransition):
        g2.set_status("nd_x", "blocked", ACTOR)

    rebuilt = WorkGraph(log, "prj_1")
    assert rebuilt.node("nd_x").status == "done"
    assert len(log.read(types=_STATUS_EVENTS)) == 2  # active, done -- nothing illegal written


def test_reused_key_for_a_different_node_conflicts(tmp_path: Path) -> None:
    log, graph = _graph(tmp_path)
    first = graph.add_node("op", ACTOR, title="one", idempotency_key="K1")
    # A genuine retry (same request) returns the original node.
    assert graph.add_node("op", ACTOR, title="one", idempotency_key="K1").id == first.id
    # A different request reusing the key must not be treated as a retry.
    with pytest.raises(IdempotencyConflict):
        graph.add_node("plan", ACTOR, title="two", idempotency_key="K1")
    assert len(log.read(types=["graph.node_added"])) == 1


def test_reused_key_for_a_different_status_change_conflicts(tmp_path: Path) -> None:
    log, graph = _graph(tmp_path)
    n1 = graph.add_node("op", ACTOR)
    n2 = graph.add_node("op", ACTOR)
    graph.set_status(n1.id, "active", ACTOR, idempotency_key="K2")

    with pytest.raises(IdempotencyConflict):
        graph.set_status(n2.id, "done", ACTOR, idempotency_key="K2")
    assert graph.node(n2.id).status == "pending"  # unchanged
    assert len(log.read(types=_STATUS_EVENTS)) == 1  # nothing written for the conflict


def test_reused_node_key_on_add_edge_conflicts(tmp_path: Path) -> None:
    _, graph = _graph(tmp_path)
    op = graph.add_node("op", ACTOR, idempotency_key="NODE-KEY")
    artifact = graph.add_node("artifact", ACTOR)
    # Reusing a node event's key for an edge must raise IdempotencyConflict, not KeyError.
    with pytest.raises(IdempotencyConflict):
        graph.add_edge("produces", op.id, artifact.id, ACTOR, idempotency_key="NODE-KEY")


def test_add_edge_retry_with_same_key_returns_existing(tmp_path: Path) -> None:
    log, graph = _graph(tmp_path)
    op = graph.add_node("op", ACTOR)
    artifact = graph.add_node("artifact", ACTOR)
    first = graph.add_edge("produces", op.id, artifact.id, ACTOR, idempotency_key="E1")
    again = graph.add_edge("produces", op.id, artifact.id, ACTOR, idempotency_key="E1")
    assert again.id == first.id
    assert len(log.read(types=["graph.edge_added"])) == 1


def test_typed_edges_validate_kinds_and_endpoints(tmp_path: Path) -> None:
    _, graph = _graph(tmp_path)
    op = graph.add_node("op", ACTOR)
    artifact = graph.add_node("artifact", ACTOR)

    edge = graph.add_edge("produces", op.id, artifact.id, ACTOR)
    assert edge.type == "produces"
    assert (edge.src, edge.dst) == (op.id, artifact.id)

    with pytest.raises(ValueError):
        graph.add_edge("produces", artifact.id, op.id, ACTOR)  # wrong endpoint kinds
    with pytest.raises(ValueError):
        graph.add_edge("teleports", op.id, artifact.id, ACTOR)  # unknown edge type
    with pytest.raises(KeyError):
        graph.add_edge("blocks", op.id, "nd_missing", ACTOR)  # dst does not exist


def test_projection_rebuild_equals_live_state_and_is_project_scoped(tmp_path: Path) -> None:
    log, graph = _graph(tmp_path, "prj_1")
    brief = graph.add_node("brief", ACTOR, title="CCN long-form")
    plan = graph.add_node("plan", ACTOR)
    graph.add_edge("derives_from", plan.id, brief.id, ACTOR)
    graph.set_status(plan.id, "active", ACTOR)

    # A rejected transition must not mutate state or the log.
    with pytest.raises(IllegalTransition):
        graph.set_status(brief.id, "done", ACTOR)
    assert len(log.read(types=_STATUS_EVENTS)) == 1  # only plan -> active was written

    assert graph.rebuild() == graph.snapshot()

    rebuilt = WorkGraph(log, "prj_1")
    assert rebuilt.snapshot() == graph.snapshot()
    assert rebuilt.node(plan.id).status == "active"

    # Another project's graph sees none of this.
    other = WorkGraph(log, "prj_2")
    assert other.nodes() == []
    assert other.edges() == []
