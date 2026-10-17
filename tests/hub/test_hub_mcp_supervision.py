"""Tests for the hub's read-only supervision tools over MCP (reviews, stop, leases, tokens)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp import Client

from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import record_lease_event
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.node.leases import Holder, LeaseEvent, LeaseManager, Scope

AGENT = Actor("agent", "agent-1")
HUMAN = Actor("human", "operator-1")
WALL = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


def _call(server: Any, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    async def body() -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(tool, args or {})
            assert result.is_error is False, result.content
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(body())


def _seed(tmp_path: Path) -> tuple[str, str, str, str]:
    log = EventLog(tmp_path / "hub.db")
    store = ArtifactStore(tmp_path / "artifacts", log)
    project = Registry(log).create_project("demo-show", HUMAN).id
    op = WorkGraph(log, project).add_node("op", AGENT, title="build").id
    render = store.put(b"render", actor=AGENT, mime="video/mp4", project_id=project)
    open_review = reviews.request_review(
        log, store, project, op, [render], AGENT, title="open", fps=25
    ).id
    reviews.add_note(log, project, open_review, 50, "hold", AGENT)
    closed = reviews.request_review(log, store, project, op, [render], AGENT, title="closed").id
    reviews.decide(log, project, closed, "reject", "audio pops at the join", HUMAN)

    events: list[LeaseEvent] = []
    manager = LeaseManager(clock=lambda: 500.0, sink=events.append)
    manager.acquire(Holder.agent("agent-1"), Scope.PROJECT, ("demo-show",), ttl=120)
    for event in events:
        record_lease_event(log, event, "node-1", wall_offset=WALL.timestamp() - 500.0)

    approvals.grant(log, "db.switch", "Local Database", 60, "sandbox", HUMAN, clock=lambda: WALL)
    used = approvals.grant(log, "project.load", project, 60, "load it", HUMAN, clock=lambda: WALL)
    approvals.consume(log, used.token_id, "project.load", project, AGENT, clock=lambda: WALL)
    control.stop(log, f"project:{project}", "wrong timeline open", HUMAN)
    log.close()
    return project, open_review, closed, render


def _reader(tmp_path: Path, now: datetime) -> Any:
    # The way `aar hub serve` opens it: read-only (query_only), so no tool can write.
    log = EventLog(tmp_path / "hub.db", read_only=True)
    store = ArtifactStore(tmp_path / "artifacts", log)
    return build_server(log, Registry(log), store, clock=lambda: now)


def test_review_tools(tmp_path: Path) -> None:
    project, open_review, closed, render = _seed(tmp_path)
    server = _reader(tmp_path, WALL)
    pending = _call(server, "reviews_pending")["reviews"]
    assert [r["id"] for r in pending] == [open_review]
    assert pending[0]["renders"] == [render]
    assert pending[0]["notes"][0]["timecode"] == "00:00:02:00"
    assert _call(server, "reviews_pending", {"project_id": "prj_other"})["reviews"] == []

    got = _call(server, "review_get", {"project_id": project, "review_id": closed})
    assert got["found"] is True
    assert got["review"]["status"] == "failed"
    assert got["review"]["decision"]["decision"] == "reject"
    assert got["review"]["decision"]["reason"] == "audio pops at the join"
    assert got["review"]["decision"]["by"] == {"kind": "human", "id": "operator-1"}
    missing = _call(server, "review_get", {"project_id": project, "review_id": "nd_missing"})
    assert missing["found"] is False


def test_control_status_shows_an_active_stop(tmp_path: Path) -> None:
    project, *_ = _seed(tmp_path)
    status = _call(_reader(tmp_path, WALL), "control_status")
    assert status["running"] is False
    assert status["stops"] == [
        {
            "scope": f"project:{project}",
            "reason": "wrong timeline open",
            "by": {"kind": "human", "id": "operator-1"},
            "at": status["stops"][0]["at"],
            "event_id": status["stops"][0]["event_id"],
        }
    ]


def test_lease_board_drops_expired_leases(tmp_path: Path) -> None:
    _seed(tmp_path)
    held = _call(_reader(tmp_path, WALL + timedelta(seconds=119)), "lease_board")
    assert [(lease["node_id"], lease["key"]) for lease in held["leases"]] == [
        ("node-1", ["demo-show"])
    ]
    assert held["leases"][0]["holder"] == {"kind": "agent", "id": "agent-1"}
    gone = _call(_reader(tmp_path, WALL + timedelta(seconds=120)), "lease_board")
    assert gone["leases"] == []


def test_approvals_list_is_metadata_and_hides_dead_tokens_by_default(tmp_path: Path) -> None:
    _seed(tmp_path)
    server = _reader(tmp_path, WALL)
    active = _call(server, "approvals_list")["approvals"]
    assert [(t["scope"], t["state"]) for t in active] == [("db.switch", "active")]
    everything = _call(server, "approvals_list", {"include_inactive": True})["approvals"]
    assert sorted(t["state"] for t in everything) == ["active", "consumed"]
    later = _call(_reader(tmp_path, WALL + timedelta(seconds=61)), "approvals_list")
    assert later["approvals"] == []
