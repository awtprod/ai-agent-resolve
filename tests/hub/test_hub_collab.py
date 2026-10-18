"""Tests for the hub collaboration reads: per-project board, feed gaps, branch records."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from aar.hub.artifacts import ArtifactStore
from aar.hub.collab import (
    EVENT_BRANCH_RECORDED,
    EVENT_FEED_GAP,
    branches,
    feed_gaps,
    project_board,
    record_branch,
    scan_branches,
)
from aar.hub.common import InvalidInput, iso
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import record_lease_event, record_presence
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.node.leases import Holder, LeaseManager, Scope

W = 1_800_000_000.0
HUMAN = Holder.human("operator")
AGENT = Holder.agent("thread-1")
NODE = Actor("node", "node-1")


class _Mono:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _board_clock(mono: float):
    return lambda: datetime.fromtimestamp(mono + W, UTC)


def _seed_leases(log: EventLog) -> None:
    events: list[Any] = []
    manager = LeaseManager(clock=_Mono(), sink=events.append)
    manager.acquire(HUMAN, Scope.PROJECT, ("proj-A",), ttl=None)
    manager.acquire(AGENT, Scope.TIMELINE, ("proj-B", "tl-9"), ttl=600.0)
    manager.acquire(Holder.agent("broker"), Scope.INSTANCE, ("default",), ttl=120.0)
    for event in events:
        record_lease_event(log, event, "node-1", wall_offset=W)
    at = datetime.fromtimestamp(1000.0 + W, UTC)
    record_presence(
        log, "node-1", active=True, at=at, human_id="operator", ttl=timedelta(minutes=5)
    )


# -- project_board -----------------------------------------------------------


def test_project_board_keeps_only_this_projects_leases_and_all_presence(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    _seed_leases(log)
    board = project_board(log, "proj-A", clock=_board_clock(1010.0))
    assert [(e.scope, e.key, e.holder_kind) for e in board.leases] == [
        ("project", ("proj-A",), "human")
    ]
    assert [e.human_id for e in board.presence] == ["operator"]
    assert board.to_dict()["stale"] is False


def test_project_board_of_another_project_sees_its_timeline_lease(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    _seed_leases(log)
    board = project_board(log, "proj-B", clock=_board_clock(1010.0))
    assert [(e.scope, e.key) for e in board.leases] == [("timeline", ("proj-B", "tl-9"))]


def test_project_board_leaves_out_the_instance_lease(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    _seed_leases(log)
    assert project_board(log, "default", clock=_board_clock(1010.0)).leases == ()


def test_project_board_flags_a_feed_gap_as_stale(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    _seed_leases(log)
    log.append(
        EVENT_FEED_GAP, NODE, {"node_id": "node-1", "dropped": 3, "at": iso(datetime.now(UTC))}
    )
    board = project_board(log, "proj-A", clock=_board_clock(1010.0))
    assert board.to_dict()["stale"] is True
    assert [(g.node_id, g.dropped) for g in board.gaps] == [("node-1", 3)]


def test_feed_gap_from_a_non_node_actor_is_ignored(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    log.append(
        EVENT_FEED_GAP,
        Actor("agent", "evil"),
        {"node_id": "node-1", "dropped": 3, "at": iso(datetime.now(UTC))},
    )
    assert feed_gaps(log) == []


# -- record_branch -----------------------------------------------------------


def _base(**overrides: Any) -> dict[str, Any]:
    base = {
        "source_timeline_id": "tl-src",
        "source_timeline_name": "Main",
        "agent_id": "thread-1",
        "branch_timeline_id": "tl-br",
        "branch_timeline_name": "Main [agent:thread-1 v1]",
        "version": 1,
        "handoff": "merge by hand with Compare Timelines",
    }
    base.update(overrides)
    return base


def _hub(tmp_path: Path) -> tuple[EventLog, str]:
    log = EventLog(tmp_path / "hub.db")
    project = Registry(log).create_project("demo", HUMAN).id
    return log, project


def test_a_branch_is_recorded_linked_to_its_source_and_agent(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base())
    found = branches(log, project)
    assert len(found) == 1
    assert (found[0].source_timeline_id, found[0].agent_id, found[0].branch_timeline_id) == (
        "tl-src",
        "thread-1",
        "tl-br",
    )
    assert found[0].recorded_by == NODE


@pytest.mark.parametrize(
    "project_id",
    ["prj_does_not_exist", None, "x" * 100_000, "bad\x00\nid"],
    ids=["unknown", "none", "huge", "control"],
)
def test_record_branch_refuses_a_bad_or_unknown_project(tmp_path: Path, project_id: Any) -> None:
    log, _ = _hub(tmp_path)
    with pytest.raises(InvalidInput):
        record_branch(log, project_id, NODE, **_base())
    assert log.read(types=[EVENT_BRANCH_RECORDED]) == []


def test_only_a_node_may_record_a_branch(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    for actor in (Actor("agent", "evil"), Actor("human", "op")):
        with pytest.raises(InvalidInput):
            record_branch(log, project, actor, **_base())
    assert log.read(types=[EVENT_BRANCH_RECORDED]) == []


def test_a_forged_branch_event_from_a_non_node_is_skipped_by_the_projection(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    # Even if an agent appends the raw event, the projection drops it (node authorship).
    log.append(
        EVENT_BRANCH_RECORDED,
        Actor("agent", "evil"),
        {**_base(), "project_id": project},
        project_id=project,
    )
    scan = scan_branches(log, project)
    assert scan.branches == () and len(scan.skipped) == 1


def test_control_characters_in_names_and_handoff_are_refused(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    with pytest.raises(InvalidInput):
        record_branch(log, project, NODE, **_base(branch_timeline_name="\x1b[31mred\x00"))
    with pytest.raises(InvalidInput):
        record_branch(log, project, NODE, **_base(handoff="\x07" * 10))


def test_a_later_record_for_the_same_branch_wins(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base(handoff="first"))
    record_branch(log, project, NODE, **_base(handoff="second"))
    found = branches(log, project)
    assert len(found) == 1 and found[0].handoff == "second"


def test_cross_project_branches_with_the_same_timeline_id_do_not_collide(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    reg = Registry(log)
    pa = reg.create_project("a", HUMAN).id
    pb = reg.create_project("b", HUMAN).id
    record_branch(log, pa, NODE, **_base())
    record_branch(log, pb, NODE, **_base(agent_id="intruder"))
    listed = branches(log)
    assert sorted((b.project_id, b.agent_id) for b in listed) == sorted(
        [(pa, "thread-1"), (pb, "intruder")]
    )


def test_record_branch_rejects_a_bad_version(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    with pytest.raises(InvalidInput):
        record_branch(log, project, NODE, **_base(version=0))
    assert log.read(types=[EVENT_BRANCH_RECORDED]) == []


# -- MCP tools (caps, cursor, bounded input) ---------------------------------


def _payload(result: Any) -> dict[str, Any]:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def test_mcp_project_board_and_agent_branches_are_capped_and_bounded(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    registry = Registry(log)
    artifacts = ArtifactStore(tmp_path / "artifacts", log)
    _seed_leases(log)
    project = registry.create_project("big", HUMAN).id
    for i in range(250):
        record_branch(log, project, NODE, **_base(branch_timeline_id=f"tl-{i:04d}"))
    server = build_server(log, registry, artifacts, clock=_board_clock(1010.0))

    from mcp import Client

    async def body() -> None:
        async with Client(server) as client:
            board = _payload(await client.call_tool("project_board", {"project_id": "proj-A"}))
            assert board["project_id"] == "proj-A" and [
                lease["scope"] for lease in board["leases"]
            ] == ["project"]
            first = _payload(await client.call_tool("agent_branches", {"project_id": project}))
            assert len(first["branches"]) == 100 and first["next_cursor"] == 100
            rest = _payload(
                await client.call_tool(
                    "agent_branches", {"project_id": project, "cursor": 100, "limit": 200}
                )
            )
            assert len(rest["branches"]) == 150 and rest["next_cursor"] is None
            huge = await client.call_tool("project_board", {"project_id": "P" * 1_000_000})
            assert huge.is_error is True

    asyncio.run(body())
