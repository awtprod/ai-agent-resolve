"""The in-process hub link: control status and approval consume over a shared event log."""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub import control
from aar.hub.approvals import ApprovalAlreadyConsumed, ApprovalScope, grant
from aar.hub.events import Actor, EventLog
from aar.node.hublink import LocalHubLink

HUMAN = Actor("human", "andrew")
NODE = Actor("node", "node-1")


def _link(tmp_path: Path) -> tuple[LocalHubLink, EventLog]:
    log = EventLog(str(tmp_path / "hub.db"))
    return LocalHubLink(log), log


def test_assert_running_raises_when_stopped(tmp_path: Path) -> None:
    link, log = _link(tmp_path)
    link.assert_running(project_id="proj-1", node_id="node-1")  # no stop: returns
    control.stop(log, control.StopScope.all(), "safety", HUMAN)
    with pytest.raises(control.Stopped):
        link.assert_running(project_id="proj-1", node_id="node-1")
    control.resume(log, control.StopScope.all(), "cleared", HUMAN)
    link.assert_running(project_id="proj-1", node_id="node-1")  # resumed: returns again


def test_consume_approval_is_single_use(tmp_path: Path) -> None:
    link, log = _link(tmp_path)
    token = grant(log, ApprovalScope.PROJECT_WRITE, "proj-1", 3600, "approved", HUMAN)
    approval = link.consume_approval(token.token_id, ApprovalScope.PROJECT_WRITE, "proj-1", NODE)
    assert approval.consumed_at is not None
    with pytest.raises(ApprovalAlreadyConsumed):
        link.consume_approval(token.token_id, ApprovalScope.PROJECT_WRITE, "proj-1", NODE)
