"""Tests for the read-only hub MCP server, driven through the SDK client."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client
from mcp_types import CallToolResult

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.mcp_server import build_server
from aar.hub.registry import ProjectSource, Registry

TOOL_NAMES = {
    "projects_list",
    "project_get",
    "events_tail",
    "work_graph",
    "artifact_meta",
    "reviews_pending",
    "review_get",
    "control_status",
    "lease_board",
    "project_board",
    "agent_branches",
    "approvals_list",
}


def _payload(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is False
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def _seed(tmp_path: Path) -> tuple[EventLog, Registry, ArtifactStore, str, str]:
    log = EventLog(tmp_path / "hub.db")
    registry = Registry(log)
    artifacts = ArtifactStore(tmp_path / "artifacts", log)

    locator = {"library": "Acme", "project": "Acme Show 2024"}
    source = ProjectSource("cloud_library", locator, host="edit-pc-1")
    project = registry.create_project(
        "Acme Show 2024", Actor("human", "owner"), sources=[source], aliases=["acme"]
    )

    graph = WorkGraph(log, project.id)
    brief = graph.add_node("brief", Actor("agent", "ed"), title="long-form cut")
    graph.set_status(brief.id, "active", Actor("agent", "ed"))

    artifact_id = artifacts.put(
        b"review-render-bytes",
        actor=Actor("agent", "ed"),
        mime="video/mp4",
        project_id=project.id,
        producing_op="op_render",
    )
    return log, registry, artifacts, project.id, artifact_id


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"server did not start on {host}:{port}")


def test_in_process_client_round_trip(tmp_path: Path) -> None:
    log, registry, artifacts, project_id, artifact_id = _seed(tmp_path)
    server = build_server(log, registry, artifacts)

    async def body() -> None:
        async with Client(server) as client:
            listing = await client.list_tools()
            assert {tool.name for tool in listing.tools} == TOOL_NAMES
            assert all(tool.description for tool in listing.tools)
            # Every tool advertises that it only reads.
            assert all(
                tool.annotations is not None and tool.annotations.read_only_hint is True
                for tool in listing.tools
            )

            projects = _payload(await client.call_tool("projects_list", {}))["projects"]
            assert any(p["id"] == project_id and p["name"] == "Acme Show 2024" for p in projects)

            got = _payload(await client.call_tool("project_get", {"project_id": project_id}))
            assert got["found"] is True
            assert got["project"]["aliases"] == ["acme"]

            missing = _payload(await client.call_tool("project_get", {"project_id": "prj_nope"}))
            assert missing["found"] is False

            tail = _payload(
                await client.call_tool("events_tail", {"project_id": project_id, "limit": 50})
            )
            assert tail["events"]
            assert tail["next_seq"] == tail["events"][-1]["seq"]

            # A bad limit is rejected with the real reason (not a generic tool error),
            # and limit=0 is rejected rather than clamped up to 1.
            for bad_limit in (-1, 0):
                bad = await client.call_tool("events_tail", {"limit": bad_limit})
                assert bad.is_error is True
                assert "limit" in bad.content[0].text

            graph = _payload(await client.call_tool("work_graph", {"project_id": project_id}))
            assert graph["found"] is True
            assert graph["nodes"][0]["kind"] == "brief"
            assert graph["nodes"][0]["status"] == "active"

            no_graph = _payload(await client.call_tool("work_graph", {"project_id": "prj_nope"}))
            assert no_graph["found"] is False

            meta = _payload(await client.call_tool("artifact_meta", {"artifact_id": artifact_id}))
            assert meta["found"] is True
            assert meta["meta"]["mime"] == "video/mp4"

            unknown = _payload(await client.call_tool("artifact_meta", {"artifact_id": "0" * 64}))
            assert unknown["found"] is False

    asyncio.run(body())


def test_streamable_http_round_trip(tmp_path: Path) -> None:
    log, registry, artifacts, project_id, _ = _seed(tmp_path)
    server = build_server(log, registry, artifacts)
    port = _free_port()

    thread = threading.Thread(
        target=lambda: server.run(transport="streamable-http", host="127.0.0.1", port=port),
        daemon=True,
    )
    thread.start()
    _wait_for_port("127.0.0.1", port)

    url = f"http://127.0.0.1:{port}/mcp"

    async def body() -> None:
        last_error: Exception | None = None
        for _ in range(20):
            try:
                async with Client(url, read_timeout_seconds=10) as client:
                    projects = _payload(await client.call_tool("projects_list", {}))["projects"]
                    assert any(p["id"] == project_id for p in projects)
                    return
            except (httpx2.HTTPError, ConnectionError, OSError) as exc:
                # The ASGI app may not be ready for a request the instant the socket
                # accepts; retry connection-level failures only.
                last_error = exc
                await asyncio.sleep(0.25)
        raise AssertionError(f"HTTP round trip never succeeded: {last_error!r}")

    asyncio.run(body())
