"""Round-3 hub collaboration tests: the d04 probe, restored round-1 tests, fold and MCP pins."""

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
    EVENT_LEASE_SNAPSHOT,
    MAX_HANDOFF,
    MAX_ID,
    MAX_NAME,
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
NODE = Actor("node", "node-1")
NODE2 = Actor("node", "node-2")
AT = iso(datetime(2027, 1, 15, 14, 0, tzinfo=UTC))


def _base(**overrides: Any) -> dict[str, Any]:
    base = {
        "source_timeline_id": "tl-src",
        "source_timeline_name": "Main",
        "agent_id": "thread-1",
        "branch_timeline_id": "tl-br",
        "branch_timeline_name": "Main [agent:thread-1 v3]",
        "version": 3,
        "handoff": "v3",
    }
    base.update(overrides)
    return base


def _hub(tmp_path: Path) -> tuple[EventLog, str]:
    log = EventLog(tmp_path / "hub.db")
    return log, Registry(log).create_project("demo", Actor("human", "op")).id


def _forge(log: EventLog, project: str, actor: Actor = NODE, **payload: Any) -> None:
    """Append a branch event by hand, bypassing record_branch (a buggy or racing writer)."""
    log.append(
        EVENT_BRANCH_RECORDED,
        actor,
        {**_base(**payload), "project_id": project},
        project_id=project,
    )


# -- d04 #1/#2: a record can be rolled back, or overwritten by another node ------------------


def test_a_delayed_lower_version_record_is_refused_and_never_rolls_back(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base())
    with pytest.raises(InvalidInput, match="behind"):
        record_branch(
            log, project, NODE, **_base(version=1, branch_timeline_name="v1", handoff="stale")
        )
    _forge(log, project, version=1, handoff="stale, written by hand")  # the fold re-checks
    found = branches(log, project)
    assert [(b.version, b.handoff) for b in found] == [(3, "v3")]
    assert any("behind" in item.reason for item in scan_branches(log, project).skipped)


def test_another_node_cannot_overwrite_a_branch_record(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base())
    with pytest.raises(InvalidInput, match="node-1"):
        record_branch(log, project, NODE2, **_base(agent_id="someone-else", version=9))
    _forge(log, project, NODE2, agent_id="someone-else", version=9)
    found = branches(log, project)
    assert [(b.agent_id, b.version, b.recorded_by) for b in found] == [("thread-1", 3, NODE)]
    # The owning node may still update its record (same or higher version).
    record_branch(log, project, NODE, **_base(handoff="merged by the editor"))
    assert branches(log, project)[0].handoff == "merged by the editor"


def test_rerecording_the_same_branch_keeps_one_record(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base())
    record_branch(log, project, NODE, **_base())  # a retry
    assert len(branches(log, project)) == 1


# -- restored round-1: a malformed branch event is skipped, never fatal (H06 ...) ------------


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 0},
        {"version": "3"},
        {"source_timeline_name": "x" * (MAX_NAME + 1)},
        {"branch_timeline_name": 7},
        {"handoff": "h" * (MAX_HANDOFF + 1)},
        {"handoff": None},
        {"agent_id": "a" * (MAX_ID + 1)},
        {"agent_id": ""},
        {"source_timeline_id": ""},
        {"branch_timeline_id": "b" * (MAX_ID + 1)},
    ],
    ids=[
        "version-0",
        "version-text",
        "long-source-name",
        "non-text-branch-name",
        "long-handoff",
        "null-handoff",
        "long-agent-id",
        "empty-agent-id",
        "empty-source-id",
        "long-branch-id",
    ],
)
def test_a_malformed_branch_event_is_skipped_not_fatal(tmp_path: Path, bad: dict) -> None:
    log, project = _hub(tmp_path)
    _forge(log, project, **bad)
    record_branch(log, project, NODE, **_base(branch_timeline_id="tl-good"))
    scan = scan_branches(log, project)
    assert [b.branch_timeline_id for b in scan.branches] == ["tl-good"]
    assert len(scan.skipped) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"handoff": "h" * (MAX_HANDOFF + 1)},
        {"branch_timeline_name": "n" * (MAX_NAME + 1)},
        {"agent_id": "a" * (MAX_ID + 1)},
        {"agent_id": "bad\nid"},
    ],
    ids=["long-handoff", "long-branch-name", "long-agent-id", "control-agent-id"],
)
def test_record_branch_refuses_unbounded_or_unclean_fields(tmp_path: Path, override: dict) -> None:
    log, project = _hub(tmp_path)
    with pytest.raises(InvalidInput):
        record_branch(log, project, NODE, **_base(**override))
    assert log.read(types=[EVENT_BRANCH_RECORDED]) == []


def test_branches_of_one_project_never_list_another_projects(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    registry = Registry(log)
    pa = registry.create_project("a", Actor("human", "op")).id
    pb = registry.create_project("b", Actor("human", "op")).id
    record_branch(log, pa, NODE, **_base(branch_timeline_id="tl-a"))
    record_branch(log, pb, NODE, **_base(branch_timeline_id="tl-b"))
    assert [b.branch_timeline_id for b in branches(log, pa)] == ["tl-a"]
    assert [b.branch_timeline_id for b in scan_branches(log, pb).branches] == ["tl-b"]
    assert {b.branch_timeline_id for b in branches(log)} == {"tl-a", "tl-b"}


# -- gaps and snapshots on the hub side -------------------------------------------------------


def test_a_malformed_gap_is_skipped_and_listed(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    for dropped in (0, -1, "3", None):
        log.append(EVENT_FEED_GAP, NODE, {"node_id": "node-1", "dropped": dropped, "at": AT})
    board = project_board(log, "p")
    assert board.gaps == () and board.to_dict()["stale"] is False
    assert len([item for item in board.skipped if item.type == EVENT_FEED_GAP]) == 4


def test_only_a_later_snapshot_from_the_same_node_clears_its_gap(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    snap = {"node_id": "node-1", "at": AT, "lease_ids": []}
    log.append(EVENT_LEASE_SNAPSHOT, NODE, snap)  # before the gap: clears nothing
    log.append(EVENT_FEED_GAP, NODE, {"node_id": "node-1", "dropped": 2, "at": AT})
    log.append(EVENT_LEASE_SNAPSHOT, NODE2, {**snap, "node_id": "node-2"})  # another node's
    assert [g.node_id for g in feed_gaps(log)] == ["node-1"]
    log.append(EVENT_LEASE_SNAPSHOT, NODE, snap)
    assert feed_gaps(log) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"node_id": "node-1", "at": AT, "lease_ids": "l-1"},
        {"node_id": "node-1", "at": AT, "lease_ids": [7]},
        {"node_id": "node-1", "at": "yesterday", "lease_ids": []},
        {"node_id": "node-1", "at": AT, "lease_ids": ["l"] * 10_001},
    ],
    ids=["ids-not-a-list", "id-not-text", "bad-at", "too-many-ids"],
)
def test_a_malformed_snapshot_removes_nothing(tmp_path: Path, payload: dict) -> None:
    log = EventLog(tmp_path / "hub.db")
    events: list[Any] = []
    LeaseManager(clock=lambda: 1000.0, sink=events.append).acquire(
        HUMAN, Scope.PROJECT, ("proj-A",), ttl=None
    )
    record_lease_event(log, events[0], "node-1", wall_offset=W)
    log.append(EVENT_FEED_GAP, NODE, {"node_id": "node-1", "dropped": 1, "at": AT})
    log.append(EVENT_LEASE_SNAPSHOT, NODE, payload)
    board = project_board(log, "proj-A")
    assert len(board.leases) == 1 and board.to_dict()["stale"] is True


# -- MCP: d04 #4/#5 board paging, and the restored exposure test -----------------------------


def _payload(result: Any) -> dict[str, Any]:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def _server(tmp_path: Path, log: EventLog, registry: Registry, mono: float = 1001.0):
    store = ArtifactStore(tmp_path / "artifacts", log)
    return build_server(log, registry, store, clock=lambda: datetime.fromtimestamp(mono + W, UTC))


def test_project_board_pages_its_leases_and_says_when_it_truncated(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    registry = Registry(log)
    manager = LeaseManager(
        lambda: 1000.0, sink=lambda e: record_lease_event(log, e, "node-1", wall_offset=W)
    )
    for i in range(520):
        manager.acquire(Holder.agent(f"a{i}"), Scope.TIMELINE, ("projR", f"tl-{i:03d}"), ttl=600)
    server = _server(tmp_path, log, registry)

    from mcp import Client

    async def body() -> None:
        async with Client(server) as client:
            first = _payload(await client.call_tool("project_board", {"project_id": "projR"}))
            assert len(first["leases"]) == 500
            assert first["truncated"] is True and first["next_cursor"] == 500
            rest = _payload(
                await client.call_tool("project_board", {"project_id": "projR", "cursor": 500})
            )
            assert len(rest["leases"]) == 20
            assert rest["truncated"] is False and rest["next_cursor"] is None
            keys = [lease["key"][1] for page in (first, rest) for lease in page["leases"]]
            assert sorted(keys) == [f"tl-{i:03d}" for i in range(520)]
            small = _payload(
                await client.call_tool(
                    "project_board", {"project_id": "projR", "cursor": 10, "limit": 5}
                )
            )
            assert len(small["leases"]) == 5 and small["next_cursor"] == 15
            negative = await client.call_tool(
                "project_board", {"project_id": "projR", "cursor": -1}
            )
            assert negative.is_error is True

    asyncio.run(body())


def test_mcp_exposes_project_board_and_agent_branches(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    registry = Registry(log)
    pa = registry.create_project("a", Actor("human", "op")).id
    pb = registry.create_project("b", Actor("human", "op")).id
    record_branch(log, pa, NODE, **_base(branch_timeline_id="tl-a"))
    record_branch(log, pb, NODE, **_base(branch_timeline_id="tl-b"))
    log.append("lease.granted", Actor("agent", "x"), {"node_id": "node-1"})  # malformed
    server = _server(tmp_path, log, registry)

    from mcp import Client

    async def body() -> None:
        async with Client(server) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert {"project_board", "agent_branches", "lease_board"} <= tools
            only_a = _payload(await client.call_tool("agent_branches", {"project_id": pa}))
            assert [b["branch"]["id"] for b in only_a["branches"]] == ["tl-a"]
            every = _payload(await client.call_tool("agent_branches", {}))
            assert sorted(b["branch"]["id"] for b in every["branches"]) == ["tl-a", "tl-b"]
            board = _payload(await client.call_tool("project_board", {"project_id": "p"}))
            assert board["skipped"]["count"] == 1  # the malformed event is reported
            for bad in ({"project_id": ""}, {"project_id": "P" * 300}, {"cursor": -1}):
                result = await client.call_tool("agent_branches", bad)
                assert result.is_error is True, bad
            limited = _payload(await client.call_tool("agent_branches", {"limit": 0}))
            assert len(limited["branches"]) == 1 and limited["next_cursor"] == 1  # clamped to 1

    asyncio.run(body())


def test_agent_branches_pages_stay_small(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    for i in range(120):
        record_branch(
            log,
            project,
            NODE,
            **_base(branch_timeline_id=f"tl-{i:03d}", handoff="é" * MAX_HANDOFF),
        )
    server = _server(tmp_path, log, Registry(log))

    from mcp import Client

    async def body() -> None:
        async with Client(server) as client:
            seen: list[str] = []
            cursor: int | None = 0
            pages = 0
            while cursor is not None:
                result = await client.call_tool(
                    "agent_branches", {"project_id": project, "cursor": cursor, "limit": 1000}
                )
                page = _payload(result)
                rows = [json.dumps(row, ensure_ascii=False).encode() for row in page["branches"]]
                # Each row is ~4 KB (2000 two-byte characters), so the 256 KiB page budget
                # ends a page well before the 100-row limit (R3M01).
                assert 0 < len(rows) < 100 and sum(map(len, rows)) <= 256 * 1024 + len(rows[-1])
                seen += [b["branch"]["id"] for b in page["branches"]]
                cursor = page["next_cursor"]
                pages += 1
            assert seen == [f"tl-{i:03d}" for i in range(120)] and pages >= 2

    asyncio.run(body())


def test_branches_stay_in_first_recorded_order_after_an_update(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base(branch_timeline_id="tl-a"))
    record_branch(log, project, NODE, **_base(branch_timeline_id="tl-b"))
    record_branch(log, project, NODE, **_base(branch_timeline_id="tl-a", handoff="updated"))
    found = branches(log, project)
    assert [(b.branch_timeline_id, b.handoff) for b in found] == [
        ("tl-a", "updated"),
        ("tl-b", "v3"),
    ]


def test_project_board_caps_presence_and_says_so(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    at = datetime.fromtimestamp(1000.0 + W, UTC)
    for i in range(501):
        record_presence(log, f"node-{i}", active=True, at=at, human_id="op", ttl=timedelta(hours=1))
    server = _server(tmp_path, log, Registry(log))

    from mcp import Client

    async def body() -> None:
        async with Client(server) as client:
            board = _payload(await client.call_tool("project_board", {"project_id": "p"}))
            assert len(board["presence"]) == 500 and board["truncated"] is True
            assert board["next_cursor"] is None  # no lease page left; presence is capped

    asyncio.run(body())
