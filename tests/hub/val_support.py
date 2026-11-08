"""Shared helpers for the validator probes (not the implementer's)."""

from __future__ import annotations

import asyncio
from typing import Any

from recipe_support import hot_fake, make_run
from write_support import AGENT, HUMAN

from aar.hub.reviews import decide
from aar.recipes.engine import advance
from aar.recipes.run import recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.worker import run_worker

WORKER = __import__("aar.hub.events", fromlist=["Actor"]).Actor("agent", "worker-1")


def worker_idle(hub: Any, fake: Any, actor: Any = WORKER, clock: Any = None) -> int:
    ran = 0
    for _ in range(30):
        kw = {"clock": clock} if clock is not None else {}
        n = run_worker(hub.log, hub.store, actor, dispatcher=fake, once=True, **kw)
        ran += n
        if n == 0:
            break
    return ran


def start_run(hub: Any, *, creator: Any = AGENT, fake: Any = None, **kw: Any) -> tuple[str, Any]:
    fake = fake if fake is not None else hot_fake()
    recipe_node = make_run(hub, actor=creator, **kw)
    recipe_start(hub.log, recipe_node, creator, idempotency_key="start-1", dispatcher=fake)
    return recipe_node, fake


def _prev_rejected_review(state: Any, current: int) -> str:
    """The review of this recipe's round-(current-1) gate that a human rejected."""
    for op in state.ops:
        if op.kind == "gate" and op.round == current - 1 and op.review_id is not None:
            review = state.reviews.get(op.review_id)
            if (
                review is not None
                and review.decision is not None
                and (review.decision.decision == "reject")
            ):
                return op.review_id
    raise AssertionError(f"no rejected round-{current - 1} review to derive a plan from")


def complete_plan(hub: Any, recipe_node: str, fake: Any, key: str, actor: Any = AGENT) -> str:
    """Submit and complete the plan step for the current round, honouring the round-3 F4 contract:
    round 1 derives from the recipe node; a later round derives from the previous round's rejected
    review, and the plan's bytes differ each round (so plan_unchanged does not trip)."""
    from write_support import plan_fixture

    from aar.hub.writes import submit_plan

    state = load_state(hub.log, recipe_node)
    current = state.current_round()
    parent = recipe_node if current == 1 else _prev_rejected_review(state, current)
    plan = dict(plan_fixture("short_clean"))
    plan["timeline"] = f"{plan.get('timeline', 'short')} [{key}]"  # distinct bytes per round
    plan_node = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan,
        actor,
        derives_from=parent,
        idempotency_key=f"plan-{key}",
    ).body["node"]["id"]
    step_complete(
        hub.log,
        hub.store,
        recipe_node,
        "p",
        None,
        {"plan_node": plan_node},
        actor,
        idempotency_key=f"complete-{key}",
        dispatcher=fake,
    )
    return plan_node


def to_gate(hub: Any, *, creator: Any = AGENT, **kw: Any) -> tuple[str, Any]:
    """Start a run, run cold/hot, complete the plan: the gate review is open."""
    recipe_node, fake = start_run(hub, creator=creator, **kw)
    worker_idle(hub, fake)
    complete_plan(hub, recipe_node, fake, "r1")
    return recipe_node, fake


def gate_op(hub: Any, recipe_node: str) -> Any:
    state = load_state(hub.log, recipe_node)
    return state.latest_in_round("g", state.current_round())


def reject(hub: Any, recipe_node: str, fake: Any, reason: str = "tighter") -> None:
    op = gate_op(hub, recipe_node)
    decide(hub.log, hub.project_id, op.review_id, "reject", reason, HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)


def approve(hub: Any, recipe_node: str, fake: Any) -> None:
    op = gate_op(hub, recipe_node)
    decide(hub.log, hub.project_id, op.review_id, "approve", "ship it", HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)


def mcp_run(server: Any, body: Any) -> Any:
    from mcp import Client

    async def go() -> Any:
        async with Client(server) as client:
            return await body(client)

    return asyncio.run(go())


def step_states(view: dict[str, Any]) -> dict[str, str]:
    return {s["step_id"]: s["state"] for s in view["steps"]}
