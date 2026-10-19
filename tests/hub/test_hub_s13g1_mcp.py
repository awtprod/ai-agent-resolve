"""S13g1: the opt-in MCP write tools, driven through the SDK client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp_types import CallToolResult
from write_support import AGENT, Hub, brief_data, make_hub, plan_fixture

from aar.cli import main
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry

READ_TOOLS = {
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
WRITE_TOOLS = {
    "brief_create",
    "recipe_create",
    "plan_submit",
    "node_set_status",
    "facts_record",
    "facts_retract",
}
NEW_READS = {"facts_get", "brief_get", "plan_get"}
HUMAN_ONLY_WORDS = ("decide", "approve", "grant", "revoke", "resume", "stop", "withdraw")


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _server(hub: Hub, **kw: Any) -> Any:
    return build_server(hub.log, Registry(hub.log), hub.store, **kw)


def _run(server: Any, body: Any) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await body(client)

    return asyncio.run(go())


def _ok(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is False, result.content
    return result.structured_content


def _refusal(result: CallToolResult, code: str) -> dict[str, Any]:
    assert result.is_error is True
    body = result.structured_content
    assert body["ok"] is False and body["error"]["code"] == code, body
    assert set(body["error"]) == {"code", "message", "fix_hint", "details"}
    assert json.loads(result.content[0].text) == body
    return body["error"]


def _names(server: Any) -> set[str]:
    async def body(client: Client) -> set[str]:
        return {tool.name for tool in (await client.list_tools()).tools}

    return _run(server, body)


def test_default_server_lists_exactly_the_read_only_tools(hub: Hub) -> None:
    """No-change pin for the default; the explicit agent_writes=False is new."""
    assert _names(_server(hub)) == READ_TOOLS
    assert _names(_server(hub, agent_writes=False)) == READ_TOOLS


def test_agent_writes_adds_the_write_tools_and_their_reads(hub: Hub) -> None:
    server = _server(hub, agent_writes=True)

    async def body(client: Client) -> Any:
        return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in _run(server, body)}
    assert set(tools) == READ_TOOLS | WRITE_TOOLS | NEW_READS
    for name in WRITE_TOOLS:
        tool = tools[name]
        assert tool.annotations.read_only_hint is False
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.idempotent_hint is True
        assert "self-declared" in tool.description and "Example" in tool.description
        assert "idempotency_key" in tool.description and "fix_hint" in tool.description
        assert {"agent_id", "idempotency_key"} <= set(tool.input_schema["required"])
    for name in NEW_READS | READ_TOOLS:
        assert tools[name].annotations.read_only_hint is True
    # Human-only actions are never tools, with or without agent writes.
    for name in tools:
        assert not any(word in name for word in HUMAN_ONLY_WORDS), name


def test_agent_writes_needs_a_writable_log(hub: Hub) -> None:
    read_only = EventLog(hub.root / "hub.db", read_only=True)
    try:
        assert read_only.read_only is True
        with pytest.raises(ValueError, match="writable"):
            build_server(read_only, Registry(read_only), hub.store, agent_writes=True)
        assert _names(build_server(read_only, Registry(read_only), hub.store)) == READ_TOOLS
    finally:
        read_only.close()
    assert hub.log.read_only is False


def test_write_tools_end_to_end(hub: Hub) -> None:
    server = _server(hub, agent_writes=True)
    pid = hub.project_id

    async def body(client: Client) -> None:
        brief = _ok(
            await client.call_tool(
                "brief_create",
                {"agent_id": "ed-1", "idempotency_key": "b1", "brief": brief_data(pid)},
            )
        )
        assert brief["op"] == "brief.create" and brief["replayed"] is False
        brief_id = brief["node"]["id"]
        again = _ok(
            await client.call_tool(
                "brief_create",
                {"agent_id": "ed-1", "idempotency_key": "b1", "brief": brief_data(pid)},
            )
        )
        assert again["replayed"] is True and again["node"] == brief["node"]

        recipe = _ok(
            await client.call_tool(
                "recipe_create",
                {
                    "agent_id": "ed-1",
                    "idempotency_key": "r1",
                    "project_id": pid,
                    "recipe": {"recipe": "ccn-short", "brief_id": brief_id},
                },
            )
        )
        assert recipe["edge"]["dst"] == brief_id

        clean = _ok(
            await client.call_tool(
                "plan_submit",
                {
                    "agent_id": "ed-1",
                    "idempotency_key": "p1",
                    "project_id": pid,
                    "derives_from": recipe["node"]["id"],
                    "plan": plan_fixture("short_clean"),
                },
            )
        )
        assert clean["node"]["status"] == "pending"
        broken = _ok(
            await client.call_tool(
                "plan_submit",
                {
                    "agent_id": "ed-1",
                    "idempotency_key": "p2",
                    "project_id": pid,
                    "derives_from": brief_id,
                    "plan": plan_fixture("short_broken"),
                    "title": "v0",
                },
            )
        )
        assert broken["node"]["status"] == "blocked" and broken["node"]["title"] == "v0"

        moved = _ok(
            await client.call_tool(
                "node_set_status",
                {
                    "agent_id": "ed-1",
                    "idempotency_key": "s1",
                    "project_id": pid,
                    "node_id": brief_id,
                    "status": "active",
                },
            )
        )
        assert (moved["from"], moved["to"]) == ("pending", "active")

        fact = _ok(
            await client.call_tool(
                "facts_record",
                {
                    "agent_id": "ed-1",
                    "idempotency_key": "f1",
                    "project_id": pid,
                    "key": "qa.lufs.short-1",
                    "value": {"value": -14.6, "unit": "LUFS"},
                    "source": {"recipe_node": recipe["node"]["id"]},
                },
            )
        )
        assert fact["fact"]["source"] == {"recipe_node": recipe["node"]["id"], "artifact": None}
        _ok(
            await client.call_tool(
                "facts_record",
                {
                    "agent_id": "ed-1",
                    "idempotency_key": "f2",
                    "project_id": pid,
                    "key": "path.map.edit-pc",
                    "value": {"from": "E:\\", "to": "/srv"},
                },
            )
        )
        gone = _ok(
            await client.call_tool(
                "facts_retract",
                {
                    "agent_id": "ed-1",
                    "idempotency_key": "f3",
                    "project_id": pid,
                    "key": "path.map.edit-pc",
                    "reason": "drive moved",
                },
            )
        )
        assert gone["retracted"]["reason"] == "drive moved"

        got = _ok(await client.call_tool("facts_get", {"project_id": pid}))
        assert [f["key"] for f in got["facts"]] == ["qa.lufs.short-1"]
        assert got["next_cursor"] is None and got["skipped"]["count"] == 0
        both = _ok(
            await client.call_tool("facts_get", {"project_id": pid, "include_retracted": True})
        )
        assert [(f["key"], f.get("retracted", False)) for f in both["facts"]] == [
            ("path.map.edit-pc", True),
            ("qa.lufs.short-1", False),
        ]
        assert both["facts"][0]["value"] is None

        brief_view = _ok(
            await client.call_tool("brief_get", {"project_id": pid, "node_id": brief_id})
        )
        assert brief_view["found"] is True and brief_view["brief"]["recipe"] == "ccn-short"
        assert {d["kind"] for d in brief_view["derived"]} == {"recipe", "plan"}
        plan_view = _ok(
            await client.call_tool("plan_get", {"project_id": pid, "node_id": clean["node"]["id"]})
        )
        assert plan_view["plan"]["timeline"] == plan_fixture("short_clean")["timeline"]
        missing = _ok(await client.call_tool("plan_get", {"project_id": pid, "node_id": "nd_x"}))
        assert missing["found"] is False
        assert (
            _ok(await client.call_tool("brief_get", {"project_id": pid, "node_id": "nd_x"}))[
                "found"
            ]
            is False
        )

    _run(server, body)
    # Every write is attributed to Actor("agent", agent_id).
    graph = WorkGraph(hub.log, pid)
    assert {graph.creator(node.id) for node in graph.nodes()} == {Actor("agent", "ed-1")}
    writers = {
        (event.actor.kind, event.actor.id)
        for event in hub.log.iter_events()
        if event.type in ("hub.write", "facts.recorded", "facts.retracted")
    }
    assert writers == {("agent", "ed-1")}


def test_refusals_are_structured_error_results(hub: Hub) -> None:
    server = _server(hub, agent_writes=True)
    pid = hub.project_id

    async def body(client: Client) -> None:
        for bad in ("", "has space", "human:andrew", "x" * 129):
            error = _refusal(
                await client.call_tool(
                    "brief_create",
                    {"agent_id": bad, "idempotency_key": "b", "brief": brief_data(pid)},
                ),
                "invalid_input",
            )
            assert error["details"]["field"] == "agent_id"
        bad_brief = brief_data(pid, kind="trailer", extra=1)
        error = _refusal(
            await client.call_tool(
                "brief_create", {"agent_id": "ed", "idempotency_key": "b", "brief": bad_brief}
            ),
            "invalid_brief",
        )
        assert {e["loc"] for e in error["details"]["errors"]} >= {"kind", "extra"}
        _refusal(
            await client.call_tool(
                "brief_create", {"agent_id": "ed", "idempotency_key": "", "brief": brief_data(pid)}
            ),
            "invalid_input",
        )
        error = _refusal(
            await client.call_tool(
                "plan_submit",
                {
                    "agent_id": "ed",
                    "idempotency_key": "p",
                    "project_id": pid,
                    "derives_from": "nd_x",
                    "plan": {"version": 1},
                },
            ),
            "invalid_plan",
        )
        assert any(e["loc"] == "project" for e in error["details"]["errors"])
        _refusal(
            await client.call_tool(
                "node_set_status",
                {
                    "agent_id": "ed",
                    "idempotency_key": "s",
                    "project_id": pid,
                    "node_id": "nd_x",
                    "status": "active",
                },
            ),
            "unknown_node",
        )
        _refusal(
            await client.call_tool(
                "facts_record",
                {
                    "agent_id": "ed",
                    "idempotency_key": "f",
                    "project_id": pid,
                    "key": "mood.today",
                    "value": {},
                },
            ),
            "unknown_namespace",
        )
        _refusal(
            await client.call_tool(
                "facts_retract",
                {
                    "agent_id": "ed",
                    "idempotency_key": "r",
                    "project_id": pid,
                    "key": "audio.gain.A1",
                    "reason": "x",
                },
            ),
            "unknown_fact",
        )
        _refusal(
            await client.call_tool("facts_get", {"project_id": pid, "prefix": "p" * 201}),
            "invalid_input",
        )
        _refusal(await client.call_tool("facts_get", {"project_id": ""}), "invalid_input")
        unknown = _ok(await client.call_tool("facts_get", {"project_id": "prj_missing"}))
        assert unknown == {"found": False, "project_id": "prj_missing"}

    _run(server, body)
    assert hub.counts()["hub.write"] == 0


def test_an_mcp_caller_cannot_move_another_agents_node_or_a_review(hub: Hub) -> None:
    from aar.hub import writes
    from aar.hub.reviews import request_review

    brief_id = writes.create_brief(
        hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b"
    ).body["node"]["id"]
    render = hub.store.put(b"r", actor=AGENT, project_id=hub.project_id)
    review = request_review(
        hub.log, hub.store, hub.project_id, brief_id, [render], AGENT, title="look"
    )
    server = _server(hub, agent_writes=True)

    async def body(client: Client) -> None:
        _refusal(
            await client.call_tool(
                "node_set_status",
                {
                    "agent_id": "someone-else",
                    "idempotency_key": "s",
                    "project_id": hub.project_id,
                    "node_id": brief_id,
                    "status": "cancelled",
                },
            ),
            "not_creator",
        )
        # Even the requesting agent cannot close its review through this tool.
        _refusal(
            await client.call_tool(
                "node_set_status",
                {
                    "agent_id": AGENT.id,
                    "idempotency_key": "s2",
                    "project_id": hub.project_id,
                    "node_id": review.id,
                    "status": "cancelled",
                },
            ),
            "protected_review",
        )

    _run(server, body)


def test_params_validator_reaches_recipe_create(hub: Hub) -> None:
    from aar.hub import writes

    brief_id = writes.create_brief(
        hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b"
    ).body["node"]["id"]
    seen: list[str] = []

    def validator(recipe: str, params: Any, brief: Any) -> list[tuple[str, str]]:
        seen.append(recipe)
        return [("params.x", "not a ccn-short parameter")] if "x" in params else []

    server = _server(hub, agent_writes=True, params_validator=validator)

    async def body(client: Client) -> None:
        error = _refusal(
            await client.call_tool(
                "recipe_create",
                {
                    "agent_id": "ed",
                    "idempotency_key": "r",
                    "project_id": hub.project_id,
                    "recipe": {"recipe": "ccn-short", "brief_id": brief_id, "params": {"x": 1}},
                },
            ),
            "invalid_params",
        )
        assert error["details"]["problems"] == [
            {"loc": "params.x", "msg": "not a ccn-short parameter"}
        ]

    _run(server, body)
    assert seen == ["ccn-short"]


def test_facts_get_pages_by_key_and_by_size(hub: Hub) -> None:
    from aar.hub import facts

    for index in range(7):
        facts.record_fact(
            hub.log,
            hub.project_id,
            f"template.trap.t{index}",
            {"text": "t" * 2000},
            AGENT,
            idempotency_key=f"t{index}",
        )
    facts.record_fact(
        hub.log, hub.project_id, "audio.gain.A1", {"db": 0}, AGENT, idempotency_key="g"
    )
    server = _server(hub, agent_writes=True)

    async def body(client: Client) -> None:
        page = _ok(await client.call_tool("facts_get", {"project_id": hub.project_id, "limit": 3}))
        assert [f["key"] for f in page["facts"]] == [
            "audio.gain.A1",
            "template.trap.t0",
            "template.trap.t1",
        ]
        rest = _ok(
            await client.call_tool(
                "facts_get",
                {"project_id": hub.project_id, "cursor": page["next_cursor"], "limit": 200},
            )
        )
        assert rest["facts"][0]["key"] == "template.trap.t2"
        assert rest["next_cursor"] is None
        traps = _ok(
            await client.call_tool(
                "facts_get", {"project_id": hub.project_id, "prefix": "template.trap.", "limit": 0}
            )
        )
        assert len(traps["facts"]) == 1  # limit is clamped to at least 1

    _run(server, body)


def test_facts_get_ends_a_page_early_to_stay_small(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aar.hub import facts, mcp_writes

    for index in range(4):
        facts.record_fact(
            hub.log,
            hub.project_id,
            f"template.trap.t{index}",
            {"text": "t" * 2000},
            AGENT,
            idempotency_key=f"t{index}",
        )
    row = facts.facts_view(hub.log, hub.project_id).get("template.trap.t0").to_dict()
    size = len(json.dumps(row, ensure_ascii=False).encode())
    monkeypatch.setattr(mcp_writes, "MAX_FACTS_BYTES", int(size * 2.5))
    server = _server(hub, agent_writes=True)

    async def body(client: Client) -> None:
        page = _ok(await client.call_tool("facts_get", {"project_id": hub.project_id}))
        assert len(page["facts"]) == 2 and page["next_cursor"] == "template.trap.t1"

    _run(server, body)


def test_serve_agent_writes_flag_opens_the_log_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    EventLog(tmp_path / "hub.db").close()
    (tmp_path / "artifacts").mkdir()
    served: list[Any] = []
    monkeypatch.setattr(
        "aar.hub.mcp_server.MCPServer.run", lambda self, **kwargs: served.append(self)
    )
    args = [
        "hub",
        "serve",
        "--db",
        str(tmp_path / "hub.db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
    ]
    assert main(args) == 0
    assert _names(served[-1]) == READ_TOOLS
    assert main([*args, "--agent-writes"]) == 0
    assert _names(served[-1]) == READ_TOOLS | WRITE_TOOLS | NEW_READS
    assert "self-declared" in capsys.readouterr().err
    # --agent-writes never creates a database.
    missing = [
        "hub",
        "serve",
        "--db",
        str(tmp_path / "nope.db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--agent-writes",
    ]
    assert main(missing) == 2
    assert not (tmp_path / "nope.db").exists()
