"""The recipe surfaces are thin wrappers: the MCP tools and the aar recipe CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from mcp import Client
from recipe_support import make_run
from write_support import AGENT, Hub, brief_data, make_hub

from aar.cli import main
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.recipes import registry
from aar.recipes.state import load_state


@pytest.fixture
def hub(tmp_path: Any) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _server(hub: Hub, **kw: Any) -> Any:
    return build_server(
        hub.log,
        Registry(hub.log),
        hub.store,
        params_validator=registry.registry_params_validator,
        **kw,
    )


def _run(server: Any, body: Any) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await body(client)

    return asyncio.run(go())


def _names(server: Any) -> set[str]:
    async def body(client: Client) -> Any:
        return {tool.name for tool in (await client.list_tools()).tools}

    return _run(server, body)


def test_recipe_status_is_listed_without_agent_writes(hub: Hub, registered_recipe: Any) -> None:
    names = _names(_server(hub))
    assert "recipe_status" in names
    assert not any(n in names for n in ("recipe_start", "step_complete", "recipe_retry"))


def test_recipe_write_tools_only_with_agent_writes(hub: Hub, registered_recipe: Any) -> None:
    names = _names(_server(hub, agent_writes=True))
    assert {"recipe_start", "step_complete", "recipe_retry", "recipe_cancel"} <= names
    assert "recipe_status" in names


def test_mcp_recipe_create_refuses_unregistered_then_starts(
    hub: Hub, registered_recipe: Any
) -> None:
    server = _server(hub, agent_writes=True)
    pid = hub.project_id

    async def body(client: Client) -> None:
        # brief for the test recipe
        brief = await client.call_tool(
            "brief_create",
            {"agent_id": "ed", "idempotency_key": "b", "brief": brief_data(pid, recipe="rectest")},
        )
        brief_id = brief.structured_content["node"]["id"]
        # recipe_create for an unregistered name is refused with unknown_recipe
        bad = await client.call_tool(
            "recipe_create",
            {
                "agent_id": "ed",
                "idempotency_key": "r-bad",
                "project_id": pid,
                "recipe": {"recipe": "nope", "brief_id": brief_id},
            },
        )
        assert bad.is_error and bad.structured_content["error"]["code"] in (
            "unknown_recipe",
            "recipe_mismatch",
        )
        # a registered recipe works; a stored QA report is required by params
        qa = __import__("recipe_support").store_qa_report(
            hub.store, __import__("recipe_support").passing_qa_report(), AGENT
        )
        ok = await client.call_tool(
            "recipe_create",
            {
                "agent_id": "ed",
                "idempotency_key": "r-ok",
                "project_id": pid,
                "recipe": {"recipe": "rectest", "brief_id": brief_id, "params": {"qa_report": qa}},
            },
        )
        recipe_node = ok.structured_content["node"]["id"]
        started = await client.call_tool(
            "recipe_start",
            {"agent_id": "ed", "idempotency_key": "s", "recipe_node": recipe_node},
        )
        assert started.structured_content["ok"] is True
        assert started.structured_content["status"] == "active"
        # recipe_status read tool reflects it; the hot step waits with the PC off.
        status = await client.call_tool("recipe_status", {"recipe_node": recipe_node})
        states = {s["step_id"]: s["state"] for s in status.structured_content["steps"]}
        assert states["h"] == "waiting:no_node_dispatcher"

    _run(server, body)


def test_mcp_recipe_status_unknown_node(hub: Hub, registered_recipe: Any) -> None:
    server = _server(hub)

    async def body(client: Client) -> None:
        result = await client.call_tool("recipe_status", {"recipe_node": "nd_missing"})
        assert result.structured_content["found"] is False

    _run(server, body)


# -- CLI ----------------------------------------------------------------------


def test_cli_list_show_status_and_start(hub: Hub, registered_recipe: Any, capsys: Any) -> None:
    db = str(hub.root / "hub.db")
    recipe_node = make_run(hub)
    hub.log.close()

    assert main(["recipe", "list"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert any(r["name"] == "rectest" for r in listing["recipes"])

    assert main(["recipe", "show", "rectest"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["found"] and {s["id"] for s in shown["steps"]} >= {"c", "h", "p", "g", "q"}

    assert main(["recipe", "status", "--db", db, recipe_node]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "pending"

    code = main(
        [
            "recipe",
            "start",
            "--db",
            db,
            "--actor",
            "agent:agent-1",
            "--idempotency-key",
            "k",
            recipe_node,
        ]
    )
    assert code == 0
    started = json.loads(capsys.readouterr().out)
    assert started["status"] == "active"
    # Reopen read-only to confirm the recipe really moved.
    from aar.hub.events import EventLog

    hub.log = EventLog(hub.root / "hub.db", read_only=True)
    assert load_state(hub.log, recipe_node).recipe_status == "active"


def test_cli_start_unknown_node_refuses_exit_2(
    hub: Hub, registered_recipe: Any, capsys: Any
) -> None:
    db = str(hub.root / "hub.db")
    code = main(
        [
            "recipe",
            "start",
            "--db",
            db,
            "--actor",
            "agent:agent-1",
            "--idempotency-key",
            "k",
            "nd_missing",
        ]
    )
    assert code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["error"]["code"] == "unknown_node"
