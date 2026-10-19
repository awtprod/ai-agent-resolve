"""S11f hub follow-ups: branch record identity, the rollback boundary, board paging."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from aar.hub.artifacts import ArtifactStore
from aar.hub.collab import (
    EVENT_BRANCH_RECORDED,
    branches,
    project_board,
    record_branch,
    scan_branches,
)
from aar.hub.common import InvalidInput
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import HubEventSpec, deliver_feed_event, lease_board, record_lease_event
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.node.leases import Holder, Lease, LeaseManager, Scope

NODE = Actor("node", "node-1")
W = 1_800_000_000.0


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


def _forge(log: EventLog, project: str, **payload: Any) -> None:
    """Append a branch event by hand, bypassing record_branch (a buggy or racing writer)."""
    log.append(
        EVENT_BRANCH_RECORDED, NODE, {**_base(**payload), "project_id": project}, project_id=project
    )


# -- LOW 13: the owning agent is part of a branch record's identity --------------------------


@pytest.mark.parametrize("version", [3, 4], ids=["same-version", "higher-version"])
def test_the_owning_node_cannot_hand_a_branch_record_to_another_agent(
    tmp_path: Path, version: int
) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base())
    with pytest.raises(InvalidInput, match="belongs to agent 'thread-1'"):
        record_branch(log, project, NODE, **_base(agent_id="thread-9", version=version))
    _forge(log, project, agent_id="thread-9", version=version, handoff="swapped by hand")
    scan = scan_branches(log, project)
    assert [(b.agent_id, b.version, b.handoff) for b in scan.branches] == [("thread-1", 3, "v3")]
    assert any("belongs to agent 'thread-1'" in item.reason for item in scan.skipped)


def test_the_owning_node_may_still_update_its_own_agents_record(tmp_path: Path) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base())
    record_branch(log, project, NODE, **_base(handoff="same version, new note"))
    record_branch(log, project, NODE, **_base(version=4, handoff="v4"))
    assert [(b.agent_id, b.version, b.handoff) for b in branches(log, project)] == [
        ("thread-1", 4, "v4")
    ]


# -- V46: a rollback by exactly one version is refused (the `<` boundary) ---------------------


def test_a_rollback_by_exactly_one_version_is_refused_at_write_and_in_the_fold(
    tmp_path: Path,
) -> None:
    log, project = _hub(tmp_path)
    record_branch(log, project, NODE, **_base())
    with pytest.raises(InvalidInput, match="version 2 is behind the recorded version 3"):
        record_branch(log, project, NODE, **_base(version=2, handoff="one behind"))
    _forge(log, project, version=2, handoff="one behind, by hand")
    scan = scan_branches(log, project)
    assert [(b.version, b.handoff) for b in scan.branches] == [(3, "v3")]
    assert [item.reason for item in scan.skipped] == ["version 2 is behind the recorded version 3"]


# -- LOW 9, V65, V67: project_board pages on a keyset cursor, capped at 500 -------------------


class _Board:
    """A hub with ``n`` timeline leases on ``projR`` (one per second) and an MCP client."""

    def __init__(self, tmp_path: Path, n: int) -> None:
        self.log = EventLog(tmp_path / "hub.db")
        self.now = [1000.0]
        self.leases = LeaseManager(
            lambda: self.now[0],
            sink=lambda e: record_lease_event(self.log, e, "node-1", wall_offset=W),
        )
        self.held: list[Lease] = []
        for i in range(n):
            self.held.append(
                self.leases.acquire(
                    Holder.agent(f"a{i:03d}"), Scope.TIMELINE, ("projR", f"tl-{i:03d}"), ttl=600
                )
            )
            self.now[0] += 1
        store = ArtifactStore(tmp_path / "artifacts", self.log)
        self.server = build_server(
            self.log,
            Registry(self.log),
            store,
            clock=lambda: datetime.fromtimestamp(self.now[0] + W, UTC),
        )

    def pages(self, limit: int, between=lambda: None) -> list[list[str]]:
        """Every page, following next_cursor; ``between`` runs after the first page."""

        async def body() -> list[list[str]]:
            pages: list[list[str]] = []
            cursor = None
            async with Client(self.server) as client:
                while True:
                    args: dict[str, Any] = {"project_id": "projR", "limit": limit}
                    if cursor is not None:
                        args["cursor"] = cursor
                    result = await client.call_tool("project_board", args)
                    data = json.loads(result.content[0].text)
                    pages.append([lease["holder"]["id"] for lease in data["leases"]])
                    assert data["truncated"] is (data["next_cursor"] is not None)
                    cursor = data["next_cursor"]
                    if len(pages) == 1:
                        between()
                    if cursor is None:
                        return pages

        return asyncio.run(body())


def test_a_lease_released_between_pages_never_makes_the_board_skip_a_live_one(
    tmp_path: Path,
) -> None:
    board = _Board(tmp_path, 8)
    first = board.held[0]
    pages = board.pages(3, between=lambda: board.leases.release(first.id, first.holder))
    seen = [holder for page in pages for holder in page]
    # a000 was on page 1 when it was released; every lease live after it is still listed once.
    assert seen == [f"a{i:03d}" for i in range(8)]


def test_a_lease_expiring_between_pages_never_makes_the_board_skip_or_repeat(
    tmp_path: Path,
) -> None:
    board = _Board(tmp_path, 8)
    board.leases.renew(board.held[1].id, board.held[1].holder, ttl=1)  # a001 ends first

    def tick() -> None:
        board.now[0] += 5

    seen = [holder for page in board.pages(3, between=tick) for holder in page]
    assert seen == ["a000", "a001", "a002", *(f"a{i:03d}" for i in range(3, 8))]


def test_a_board_that_ends_exactly_on_a_page_boundary_has_no_next_cursor(tmp_path: Path) -> None:
    board = _Board(tmp_path, 6)
    pages = board.pages(3)
    assert pages == [["a000", "a001", "a002"], ["a003", "a004", "a005"]]  # no empty third page


def test_the_board_page_size_is_capped_at_500(tmp_path: Path) -> None:
    board = _Board(tmp_path, 501)
    pages = board.pages(10_000)
    assert [len(page) for page in pages] == [500, 1]


# -- V44 and the board's own order --------------------------------------------------------------


def _stamped(spec_type: str, payload: dict[str, Any], seq: int, key: str) -> HubEventSpec:
    return HubEventSpec(spec_type, {**payload, "feed": {"nonce": "n" * 16, "seq": seq}}, key)


def test_an_unauthored_renewal_after_the_snapshot_never_keeps_a_lease_on_the_board(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path / "hub.db")
    manager = LeaseManager(
        lambda: 1000.0, sink=lambda e: record_lease_event(log, e, "node-1", wall_offset=W)
    )
    lease = manager.acquire(Holder.human("op"), Scope.PROJECT, ("pV",), ttl=None)
    snapshot = {"node_id": "node-1", "at": "2027-01-15T14:00:00+00:00", "lease_ids": []}
    log.append("lease.snapshot", Actor("node", "node-1"), snapshot)  # the lease is gone
    renewed = log.read(types=["lease.granted"])[0].payload
    log.append("lease.renewed", Actor("agent", "someone"), renewed)  # forged, after it
    board = project_board(log, "pV", clock=lambda: datetime.fromtimestamp(W + 2000, UTC))
    assert board.leases == ()
    assert any("written by agent:someone" in s.reason for s in board.skipped)
    assert lease.id not in {b.lease_id for b in lease_board(log).leases}


def test_a_late_older_state_of_a_lease_never_overwrites_a_newer_one(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    base = {
        "node_id": "node-1",
        "lease_id": "L1",
        "scope": "project",
        "key": ["pO"],
        "holder": {"kind": "agent", "id": "a"},
        "acquired_at": "2027-01-15T14:00:00+00:00",
        "ttl": 60.0,
        "at": "2027-01-15T14:00:00+00:00",
        "by": None,
    }
    grant = {**base, "expires_at": "2027-01-15T14:01:00+00:00"}
    renewal = {**base, "expires_at": "2027-01-15T15:00:00+00:00", "ttl": 3600.0}
    deliver_feed_event(log, _stamped("lease.renewed", renewal, 2, "renew"), "node-1")
    deliver_feed_event(log, _stamped("lease.granted", grant, 1, "grant"), "node-1")  # late
    at = datetime(2027, 1, 15, 14, 30, tzinfo=UTC)
    board = lease_board(log, clock=lambda: at)
    assert [(b.lease_id, b.expires_at) for b in board.leases] == [
        ("L1", "2027-01-15T15:00:00+00:00")
    ]
