"""RemoteHubLink returns exactly what LocalHubLink returns for the same hub state (S15a item 5).

The behaviour that must match is parametrised over both links; the returned objects are also
compared for equality directly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aar.hub import approvals, control, nodes
from aar.hub.approvals import ApprovalScope
from aar.hub.control import Stopped
from aar.hub.events import Actor, EventLog
from aar.hub.registry import ProjectSource, Registry
from aar.node.hublink import LocalHubLink
from aar.node.locator import ProjectLocator

from .remote_support import ServerThread, hub_app, remote_link, write_token

HUMAN = Actor("human", "andrew")
NODE = Actor("node", "node-1")
AGENT = Actor("agent", "thread-1#job-1")


@pytest.fixture
def world(tmp_path: Path) -> SimpleNamespace:
    log = EventLog(str(tmp_path / "hub.db"))
    show = Registry(log).create_project(
        "Show", HUMAN, sources=[ProjectSource("local_db", {"name": "Show", "uid": "uid-1"})]
    )
    loc = ProjectLocator(("Shows",), "Show", "uid-1")
    _, token = nodes.enroll(log, "node-1", HUMAN)
    token_file = write_token(tmp_path, token)
    # A branch recorded as the node, so both links see it in branches().
    LocalHubLink(log).record_branch(
        loc,
        NODE,
        source_timeline_id="src-1",
        source_timeline_name="Main",
        agent_id="thread-1",
        branch_timeline_id="br-1",
        branch_timeline_name="Main [agent:thread-1 v1]",
        version=1,
    )
    return SimpleNamespace(log=log, loc=loc, token_file=token_file, show=show.id)


@pytest.fixture(params=["local", "remote"])
def link(request: pytest.FixtureRequest, world: SimpleNamespace):
    if request.param == "local":
        yield LocalHubLink(world.log)
    else:
        with ServerThread(hub_app(world.log)) as server:
            yield remote_link(server.base_url, world.token_file)


# -- parametrised behaviour (both links) --------------------------------------------------


def test_assert_running_returns_then_raises_on_stop(link, world: SimpleNamespace) -> None:
    link.assert_running(project=world.loc, node_id="node-1")  # not stopped: returns
    control.stop(world.log, control.StopScope.node("node-1"), "safety", HUMAN)
    with pytest.raises(Stopped):
        link.assert_running(project=world.loc, node_id="node-1")


def test_branches_lists_the_recorded_branch(link, world: SimpleNamespace) -> None:
    found = link.branches(world.loc)
    assert [b.branch_timeline_id for b in found] == ["br-1"]


def test_approval_read_of_an_unknown_token_is_none(link, world: SimpleNamespace) -> None:
    assert link.approval("appr_missing") is None


# -- direct equality: remote == local for the same state ----------------------------------


def test_control_status_equal(world: SimpleNamespace) -> None:
    control.stop(world.log, control.StopScope.node("node-1"), "safety", HUMAN)
    local = LocalHubLink(world.log)
    with ServerThread(hub_app(world.log)) as server:
        remote = remote_link(server.base_url, world.token_file)
        assert remote.control_status() == local.control_status()


def test_approval_read_equal(world: SimpleNamespace) -> None:
    granted = approvals.grant(world.log, ApprovalScope.PROJECT_WRITE, "proj-x", 3600, "ok", HUMAN)
    local = LocalHubLink(world.log)
    with ServerThread(hub_app(world.log)) as server:
        remote = remote_link(server.base_url, world.token_file)
        assert remote.approval(granted.token_id) == local.approval(granted.token_id)


def test_branches_equal(world: SimpleNamespace) -> None:
    local = LocalHubLink(world.log)
    with ServerThread(hub_app(world.log)) as server:
        remote = remote_link(server.base_url, world.token_file)
        assert remote.branches(world.loc) == local.branches(world.loc)


def test_consume_result_equals_local_readback(world: SimpleNamespace) -> None:
    granted = approvals.grant(world.log, ApprovalScope.PROJECT_WRITE, "proj-x", 3600, "ok", HUMAN)
    local = LocalHubLink(world.log)
    with ServerThread(hub_app(world.log)) as server:
        remote = remote_link(server.base_url, world.token_file)
        consumed = remote.consume_approval(
            granted.token_id, ApprovalScope.PROJECT_WRITE, "proj-x", AGENT
        )
    # What remote returned equals what local reads back for the now-consumed token.
    assert consumed == local.approval(granted.token_id)
    assert consumed.consumed_by == AGENT


def test_record_branch_returns_the_same_hub_id(world: SimpleNamespace) -> None:
    local = LocalHubLink(world.log)
    local_id = local.hub_project_id(world.loc)
    with ServerThread(hub_app(world.log)) as server:
        remote = remote_link(server.base_url, world.token_file)
        # Re-record the identical branch (idempotent): same hub id, no new record.
        hub_id = remote.record_branch(
            world.loc,
            NODE,
            source_timeline_id="src-1",
            source_timeline_name="Main",
            agent_id="thread-1",
            branch_timeline_id="br-1",
            branch_timeline_name="Main [agent:thread-1 v1]",
            version=1,
        )
    assert hub_id == local_id
    assert len(local.branches(world.loc)) == 1
