"""Round-2 validator probes: who can produce the runner actor, status moves, review withdrawal.

Each test asserts the SAFE property; a failure is a finding. Run with -s for OBS lines.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from recipe_support import make_run
from val_support import approve, gate_op, mcp_run, start_run, to_gate, worker_idle
from write_support import AGENT, HUMAN

from aar.hub.graph import WorkGraph
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.hub.reviews import get_review, withdraw
from aar.hub.writes import WriteError, accept_any_params, set_own_status
from aar.recipes.events import RUNNER
from aar.recipes.run import recipe_start
from aar.recipes.state import load_state

AAR = str(Path(sys.executable).parent / "aar")


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)[:3000]}")


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "AAR_LIVE"}
    return subprocess.run([AAR, *args], capture_output=True, text=True, env=env, timeout=120)


# -- 1. string surfaces that might produce the runner actor ----------------------------------


def test_parse_actor_functions_refuse_runner() -> None:
    from aar.hub import cli as hub_cli
    from aar.hub import write_cli
    from aar.recipes import cli as recipe_cli

    got = {}
    for name, fn in (
        ("hub.cli", hub_cli._parse_actor),
        ("hub.write_cli", write_cli._parse_actor),
        ("recipes.cli", recipe_cli._parse_actor),
    ):
        try:
            got[name] = str(fn("runner:recipes"))
        except Exception as exc:
            got[name] = f"refused: {type(exc).__name__}"
    _obs("_parse_actor('runner:recipes')", got)
    assert all(v.startswith("refused") for v in got.values()), got


def test_media_ingest_cli_cannot_write_as_runner(hub: Any, tmp_path: Path) -> None:
    src = tmp_path / "clip.mov"
    src.write_bytes(b"\x00acme clip bytes")
    before = max(e.seq for e in hub.log.iter_events())
    cp = _cli(
        "media",
        "ingest",
        str(src),
        "--db",
        str(hub.root / "hub.db"),
        "--artifacts",
        str(hub.root / "artifacts"),
        "--project",
        hub.project_id,
        "--actor",
        "runner:recipes",
    )
    written = [(e.type, e.actor.kind, e.actor.id) for e in hub.log.iter_events() if e.seq > before]
    _obs(
        "aar media ingest --actor runner:recipes",
        {"rc": cp.returncode, "stderr": cp.stderr[-300:], "written": written},
    )
    assert not any(k == "runner" for _, k, _ in written), (
        "a CLI string surface wrote events as the runner identity"
    )


def test_node_api_body_actor_refuses_runner() -> None:
    from aar.hub import nodeapi

    ctx = SimpleNamespace(body={"actor": {"kind": "runner", "id": "recipes"}}, node_id="node-1")
    try:
        actor = nodeapi._body_actor(ctx)  # type: ignore[arg-type]
        got = f"accepted: {actor.kind}:{actor.id} (== RUNNER: {actor == RUNNER})"
    except Exception as exc:
        got = f"refused: {type(exc).__name__}: {exc}"
    _obs("node API _body_actor({kind: runner, id: recipes})", got)
    assert got.startswith("refused"), got


def test_node_api_body_actor_refuses_a_human() -> None:
    """R04: the shared parser's allow-set is enforced -- the node API forbids a human body actor."""
    from aar.hub import nodeapi

    ctx = SimpleNamespace(body={"actor": {"kind": "human", "id": "operator-1"}}, node_id="node-1")
    try:
        nodeapi._body_actor(ctx)  # type: ignore[arg-type]
        got = "accepted"
    except Exception as exc:
        got = f"refused: {type(exc).__name__}"
    _obs("node API _body_actor({kind: human})", got)
    assert got.startswith("refused"), got


def test_mcp_agent_id_cannot_be_runner(hub: Any, registered_recipe: Any) -> None:
    server = build_server(hub.log, Registry(hub.log), hub.store, agent_writes=True)

    async def body(client: Any) -> Any:
        out = {}
        for aid in ("runner:recipes", "runner", "recipes"):
            r = await client.call_tool(
                "brief_create",
                {
                    "agent_id": aid,
                    "idempotency_key": f"k-{len(out)}",
                    "brief": __import__("write_support").brief_data(hub.project_id),
                },
            )
            out[aid] = [r.is_error, (r.structured_content or {}).get("error", {}).get("code")]
        return out

    got = mcp_run(server, body)
    kinds = sorted({e.actor.kind for e in hub.log.iter_events()})
    _obs("MCP brief_create agent_id variants", {"results": got, "actor kinds in log": kinds})
    assert "runner" not in kinds


def test_mcp_step_complete_schema_has_round(hub: Any) -> None:
    server = build_server(hub.log, Registry(hub.log), hub.store, agent_writes=True)

    async def body(client: Any) -> Any:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        return sorted(tools["step_complete"].input_schema.get("properties", {}))

    props = mcp_run(server, body)
    _obs("MCP step_complete input properties", props)
    assert "round" in props, "brief F4: step_complete gains a round; the MCP tool has none"


# -- 2. status moves (human and agent) -----------------------------------------------------


def test_human_status_moves_on_runner_nodes_are_refused(hub: Any, registered_recipe: Any) -> None:
    rn, _fake = to_gate(hub)
    g = gate_op(hub, rn)
    got = {}
    for label, node, to in (
        ("gate op -> done", g.op_id, "done"),
        ("gate op -> failed", g.op_id, "failed"),
        ("recipe -> cancelled", rn, "cancelled"),
        ("recipe -> blocked", rn, "blocked"),
    ):
        for who in (HUMAN, AGENT):
            try:
                set_own_status(
                    hub.log, hub.project_id, node, to, who, idempotency_key=f"m-{len(got)}"
                )
                got[f"{who.kind}: {label}"] = "accepted"
            except WriteError as exc:
                got[f"{who.kind}: {label}"] = exc.code
    st = load_state(hub.log, rn)
    _obs("status moves on runner-managed nodes", got)
    assert set(got.values()) == {"runner_managed"}, got
    assert st.recipe_status == "active" and st.latest_in_round("g", 1).status == "active"


def test_pre_start_status_move_is_refused_at_start(hub: Any, registered_recipe: Any) -> None:
    """N5: if the creator moves the recipe node before recipe.started, recipe_start refuses it
    (step_not_open), so the graph mirror can never end up lying about a started run."""
    rn = make_run(hub)
    set_own_status(hub.log, hub.project_id, rn, "active", AGENT, idempotency_key="pre1")
    set_own_status(hub.log, hub.project_id, rn, "done", AGENT, idempotency_key="pre2")
    fake = __import__("recipe_support").hot_fake()
    with pytest.raises(WriteError) as exc:
        recipe_start(hub.log, rn, AGENT, idempotency_key="s", dispatcher=fake)
    _obs("pre-start finalized recipe, then start refused", {"code": exc.value.code})
    assert exc.value.code == "step_not_open"
    assert not load_state(hub.log, rn).started


# -- 3. gate review withdrawal ---------------------------------------------------------------


def test_agent_cannot_withdraw_the_gate_review_library_or_cli(
    hub: Any, registered_recipe: Any
) -> None:
    rn, _fake = to_gate(hub)
    g = gate_op(hub, rn)
    try:
        withdraw(hub.log, hub.project_id, g.review_id, "mine", AGENT)
        lib = "accepted"
    except Exception as exc:
        lib = type(exc).__name__
    cp = _cli(
        "hub",
        "review",
        "withdraw",
        "--db",
        str(hub.root / "hub.db"),
        "--project",
        hub.project_id,
        "--review",
        g.review_id,
        "--reason",
        "mine",
        "--actor",
        "agent:agent-1",
    )
    cp_runner = _cli(
        "hub",
        "review",
        "withdraw",
        "--db",
        str(hub.root / "hub.db"),
        "--project",
        hub.project_id,
        "--review",
        g.review_id,
        "--reason",
        "mine",
        "--actor",
        "runner:recipes",
    )
    review = get_review(hub.log, hub.project_id, g.review_id)
    _obs(
        "agent withdraw of the gate review",
        {
            "library": lib,
            "cli_rc": cp.returncode,
            "cli_err": cp.stderr[-200:],
            "cli_as_runner_rc": cp_runner.returncode,
            "cli_as_runner_err": cp_runner.stderr[-200:],
            "review_pending": review.pending,
            "requested_by": review.requested_by,
        },
    )
    assert lib != "accepted" and cp.returncode != 0 and cp_runner.returncode != 0
    assert review.pending


def test_human_withdraw_then_creator_retry_reopens_same_round_and_mirror_follows(
    hub: Any, registered_recipe: Any
) -> None:
    from aar.recipes.run import recipe_retry

    rn, fake = to_gate(hub)
    g1 = gate_op(hub, rn)
    withdraw(hub.log, hub.project_id, g1.review_id, "wrong render", HUMAN)
    from aar.recipes.engine import advance

    advance(hub.log, rn, actor=AGENT, dispatcher=fake)
    st = load_state(hub.log, rn)
    mirror1 = WorkGraph(hub.log, hub.project_id).node(rn).status
    blocked = (
        st.recipe_status,
        st.latest_in_round("g", 1).status,
        (st.latest_in_round("g", 1).result_error or {}).get("code"),
        mirror1,
    )
    recipe_retry(hub.log, rn, "g", AGENT, idempotency_key="rg", dispatcher=fake)
    st = load_state(hub.log, rn)
    g2 = st.latest_in_round("g", 1)
    mirror2 = WorkGraph(hub.log, hub.project_id).node(rn).status
    _obs(
        "withdraw -> blocked -> creator retry",
        {
            "blocked": blocked,
            "after_retry": [
                st.recipe_status,
                g2.attempt,
                g2.round,
                g2.review_id != g1.review_id,
                mirror2,
            ],
        },
    )
    approve(hub, rn, fake)
    worker_idle(hub, fake)
    st = load_state(hub.log, rn)
    _obs("after approve", [st.recipe_status, WorkGraph(hub.log, hub.project_id).node(rn).status])
    assert blocked[0] == "blocked" and blocked[2] == "review_withdrawn" and blocked[3] == "blocked"
    assert g2.attempt == 2 and g2.round == 1 and mirror2 == "active"
    assert st.recipe_status == "done"


@pytest.mark.parametrize("who", ["creator", "other"])
def test_retry_of_withdrawn_gate_permissions(hub: Any, registered_recipe: Any, who: str) -> None:
    from write_support import OTHER

    from aar.recipes.engine import advance
    from aar.recipes.run import recipe_retry

    rn, fake = to_gate(hub)
    withdraw(hub.log, hub.project_id, gate_op(hub, rn).review_id, "x", HUMAN)
    advance(hub.log, rn, actor=AGENT, dispatcher=fake)
    actor = AGENT if who == "creator" else OTHER
    try:
        recipe_retry(hub.log, rn, "g", actor, idempotency_key="rg", dispatcher=fake)
        got = "accepted"
    except WriteError as exc:
        got = exc.code
    _obs(f"retry withdrawn gate by {who}", got)
    assert got == ("accepted" if who == "creator" else "not_creator")


def test_runner_actor_never_decides(hub: Any, registered_recipe: Any) -> None:
    from aar.hub.reviews import decide

    rn, _fake = to_gate(hub)
    g = gate_op(hub, rn)
    try:
        decide(hub.log, hub.project_id, g.review_id, "approve", "self", RUNNER)
        got = "accepted"
    except Exception as exc:
        got = type(exc).__name__
    _obs("decide as RUNNER", got)
    assert got != "accepted"
    _ = (start_run, accept_any_params)
