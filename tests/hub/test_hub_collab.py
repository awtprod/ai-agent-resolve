"""Tests for the hub collaboration reads: the per-project board and branch records."""

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
    branches,
    project_board,
    record_branch,
    scan_branches,
)
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
    events = []
    manager = LeaseManager(clock=_Mono(), sink=events.append)
    manager.acquire(HUMAN, Scope.PROJECT, ("proj-A",), ttl=None)
    manager.acquire(AGENT, Scope.TIMELINE, ("proj-B", "tl-9"), ttl=600.0)
    manager.acquire(Holder.agent("broker"), Scope.INSTANCE, ("default",), ttl=120.0)
    for event in events:
        record_lease_event(log, event, "node-1", wall_offset=W)
    # Report presence at the same wall epoch the leases map to (mono 1000 + offset).
    at = datetime.fromtimestamp(1000.0 + W, UTC)
    record_presence(
        log, "node-1", active=True, at=at, human_id="operator", ttl=timedelta(minutes=5)
    )


# -- project_board -----------------------------------------------------------


def test_project_board_keeps_only_this_projects_leases_and_all_presence(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    _seed_leases(log)

    board = project_board(log, "proj-A", clock=_board_clock(1010.0))

    assert [(entry.scope, entry.key, entry.holder_kind) for entry in board.leases] == [
        ("project", ("proj-A",), "human")
    ]
    assert [entry.human_id for entry in board.presence] == ["operator"]
    assert board.project_id == "proj-A"


def test_project_board_of_another_project_sees_its_timeline_lease(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    _seed_leases(log)

    board = project_board(log, "proj-B", clock=_board_clock(1010.0))

    assert [(entry.scope, entry.key) for entry in board.leases] == [
        ("timeline", ("proj-B", "tl-9"))
    ]


def test_project_board_leaves_out_the_instance_lease(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    _seed_leases(log)

    board = project_board(log, "default", clock=_board_clock(1010.0))

    assert board.leases == ()  # the instance lease is not a project


# -- branch records ----------------------------------------------------------


def _record(log: EventLog, project: str, **overrides: Any):
    kwargs: dict[str, Any] = {
        "source_timeline_id": "tl-src",
        "source_timeline_name": "Main",
        "agent_id": "thread-1",
        "branch_timeline_id": "tl-br",
        "branch_timeline_name": "Main [agent:thread-1 v1]",
        "version": 1,
        "handoff": "merge by hand with Compare Timelines",
    }
    kwargs.update(overrides)
    return record_branch(log, project, NODE, **kwargs)


def test_a_branch_is_recorded_linked_to_its_source_and_agent(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    project = Registry(log).create_project("demo", HUMAN).id
    _record(log, project)

    found = branches(log, project)
    assert len(found) == 1
    branch = found[0]
    assert branch.source_timeline_id == "tl-src" and branch.source_timeline_name == "Main"
    assert branch.agent_id == "thread-1" and branch.version == 1
    assert branch.branch_timeline_id == "tl-br"
    assert branch.handoff and branch.recorded_by == NODE


def test_rerecording_the_same_branch_is_idempotent(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    project = Registry(log).create_project("demo", HUMAN).id
    first = _record(log, project)
    again = _record(log, project)

    assert again.id == first.id
    assert len(branches(log, project)) == 1
    assert len(log.read(types=[EVENT_BRANCH_RECORDED])) == 1


def test_a_malformed_branch_event_is_skipped_not_fatal(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    project = Registry(log).create_project("demo", HUMAN).id
    _record(log, project)
    log.append(EVENT_BRANCH_RECORDED, NODE, {"source_timeline_id": "x"}, project_id=project)

    scan = scan_branches(log, project)
    assert len(scan.branches) == 1 and len(scan.skipped) == 1


def test_record_branch_rejects_a_bad_version(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    project = Registry(log).create_project("demo", HUMAN).id
    with pytest.raises(ValueError, match="version"):
        _record(log, project, version=0)
    assert log.read(types=[EVENT_BRANCH_RECORDED]) == []


# -- MCP tools ---------------------------------------------------------------


def _payload(result: Any) -> dict[str, Any]:
    assert result.is_error is False
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def test_mcp_exposes_project_board_and_agent_branches(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    registry = Registry(log)
    artifacts = ArtifactStore(tmp_path / "artifacts", log)
    _seed_leases(log)
    project = registry.create_project("demo", HUMAN).id
    _record(log, project)
    server = build_server(log, registry, artifacts, clock=_board_clock(1010.0))

    from mcp import Client

    async def body() -> None:
        async with Client(server) as client:
            board = _payload(await client.call_tool("project_board", {"project_id": "proj-A"}))
            assert board["project_id"] == "proj-A"
            assert [lease["scope"] for lease in board["leases"]] == ["project"]
            listed = _payload(await client.call_tool("agent_branches", {"project_id": project}))
            assert [b["agent_id"] for b in listed["branches"]] == ["thread-1"]

    asyncio.run(body())
