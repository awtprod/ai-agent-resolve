"""Idempotent approval consume across a lost ack, for the node API (S15a item 4).

``consume`` keeps its single-use semantics; ``consume_for_node`` adds a node-scoped replay.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub.approvals import (
    EVENT_CONSUMED,
    ApprovalAlreadyConsumed,
    ApprovalScope,
    consume,
    consume_for_node,
    grant,
)
from aar.hub.events import Actor, EventLog

HUMAN = Actor("human", "andrew")
AGENT = Actor("agent", "agent-1#job-7")
SCOPE = ApprovalScope.PROJECT_WRITE
TARGET = "proj-1#op:write#step:3"


def _granted(tmp_path: Path) -> tuple[EventLog, str]:
    log = EventLog(str(tmp_path / "hub.db"))
    token = grant(log, SCOPE, TARGET, 3600, "ok", HUMAN)
    return log, token.token_id


def _consumed_count(log: EventLog) -> int:
    return sum(1 for e in log.iter_events(types=(EVENT_CONSUMED,)))


def test_first_consume_records_via_node_and_not_replayed(tmp_path: Path) -> None:
    log, token_id = _granted(tmp_path)
    approval, replayed = consume_for_node(
        log, token_id, SCOPE, TARGET, AGENT, via_node="node-1", request_id="r1"
    )
    assert replayed is False
    assert approval.consumed_at is not None
    assert approval.consumed_by == AGENT
    (event,) = list(log.iter_events(types=(EVENT_CONSUMED,)))
    assert event.payload["via_node"] == "node-1"
    assert event.payload["request_id"] == "r1"


def test_replay_through_same_node_returns_first_result_and_writes_nothing(
    tmp_path: Path,
) -> None:
    log, token_id = _granted(tmp_path)
    first, _ = consume_for_node(log, token_id, SCOPE, TARGET, AGENT, via_node="node-1")
    assert _consumed_count(log) == 1
    # The lost-ack retry carries the same fingerprint and node (even a different request_id).
    again, replayed = consume_for_node(
        log, token_id, SCOPE, TARGET, AGENT, via_node="node-1", request_id="retry"
    )
    assert replayed is True
    assert again == first  # the exact first result
    assert _consumed_count(log) == 1  # nothing new written


def test_replay_across_a_different_node_is_refused(tmp_path: Path) -> None:
    log, token_id = _granted(tmp_path)
    consume_for_node(log, token_id, SCOPE, TARGET, AGENT, via_node="node-1")
    with pytest.raises(ApprovalAlreadyConsumed):
        consume_for_node(log, token_id, SCOPE, TARGET, AGENT, via_node="node-2")


def test_replay_with_a_different_actor_is_refused(tmp_path: Path) -> None:
    log, token_id = _granted(tmp_path)
    consume_for_node(log, token_id, SCOPE, TARGET, AGENT, via_node="node-1")
    with pytest.raises(ApprovalAlreadyConsumed):
        consume_for_node(
            log, token_id, SCOPE, TARGET, Actor("agent", "other#job-9"), via_node="node-1"
        )


def test_replay_with_a_different_target_is_refused(tmp_path: Path) -> None:
    log, token_id = _granted(tmp_path)
    consume_for_node(log, token_id, SCOPE, TARGET, AGENT, via_node="node-1")
    with pytest.raises(ApprovalAlreadyConsumed):
        consume_for_node(log, token_id, SCOPE, "proj-1#op:write#step:99", AGENT, via_node="node-1")


def test_a_locally_consumed_token_cannot_be_replayed_remotely(tmp_path: Path) -> None:
    log, token_id = _granted(tmp_path)
    # Consumed the old way (LocalHubLink): the event has no via_node.
    consume(log, token_id, SCOPE, TARGET, AGENT)
    with pytest.raises(ApprovalAlreadyConsumed):
        consume_for_node(log, token_id, SCOPE, TARGET, AGENT, via_node="node-1")


def test_consume_semantics_unchanged_for_local_callers(tmp_path: Path) -> None:
    log, token_id = _granted(tmp_path)
    consume(log, token_id, SCOPE, TARGET, AGENT)
    with pytest.raises(ApprovalAlreadyConsumed):
        consume(log, token_id, SCOPE, TARGET, AGENT)  # still single-use, never a silent replay


def test_normal_refusals_still_apply(tmp_path: Path) -> None:
    from aar.hub.approvals import ApprovalScopeMismatch

    log, token_id = _granted(tmp_path)
    # A scope mismatch on an unconsumed token refuses as usual (not a replay path).
    with pytest.raises(ApprovalScopeMismatch):
        consume_for_node(log, token_id, ApprovalScope.DB_SWITCH, TARGET, AGENT, via_node="node-1")
