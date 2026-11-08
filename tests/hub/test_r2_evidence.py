"""Round-2 validator probes: evidence scoping and forged runner events (generic event API).

Each test asserts the SAFE property; a failure is a finding. Run with -s for OBS lines.
"""

from __future__ import annotations

import json
from typing import Any

from recipe_support import RECIPE_NAME, hot_fake, passing_qa_report, store_qa_report
from val_support import approve, complete_plan, gate_op, start_run, to_gate, worker_idle
from write_support import AGENT, OTHER, brief_data

from aar.hub.events import Actor
from aar.hub.graph import WorkGraph
from aar.hub.reviews import get_review
from aar.hub.writes import create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.engine import advance
from aar.recipes.run import recipe_cancel, recipe_start
from aar.recipes.state import load_state
from aar.recipes.status import status_view


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)[:3000]}")


def _second_recipe(hub: Any, actor: Actor, key: str) -> str:
    """Another rectest recipe in the SAME project (fresh keys, so not a replay of make_run)."""
    qa = store_qa_report(hub.store, passing_qa_report(), actor)
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe=RECIPE_NAME), actor, idempotency_key=f"b-{key}"
    )
    body = {
        "recipe": RECIPE_NAME,
        "brief_id": brief.body["node"]["id"],
        "params": {"qa_report": qa},
    }
    return create_recipe(
        hub.log,
        hub.project_id,
        body,
        actor,
        idempotency_key=f"r-{key}",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]


def _ops(hub: Any, rn: str) -> list[Any]:
    return [
        n
        for n in WorkGraph(hub.log, hub.project_id).nodes()
        if n.kind == "op" and isinstance(n.payload, dict) and n.payload.get("recipe_node") == rn
    ]


# -- cross-recipe scoping of recipe.started / recipe.run_cancelled ---------------------------


def test_starting_one_recipe_does_not_start_another_in_the_project(
    hub: Any, registered_recipe: Any
) -> None:
    """Agent-2's recipe B is never started by agent-2; agent-1 starts its own recipe A."""
    b = _second_recipe(hub, OTHER, "b")  # created (pending) by agent-2, never started
    _a, fake = start_run(hub)  # agent-1 creates and starts A
    sb = load_state(hub.log, b)
    _obs("B (never started) after A started", {"started": sb.started, "status": sb.recipe_status})
    worker_idle(hub, fake)  # a normal worker pass over the hub
    sb = load_state(hub.log, b)
    view = status_view(hub.log, b)
    _obs(
        "B after a worker pass",
        {
            "status": sb.recipe_status,
            "ops_created_for_B": len(_ops(hub, b)),
            "B_steps": {s["step_id"]: s["state"] for s in view["steps"]},
            "B_graph_status": WorkGraph(hub.log, hub.project_id).node(b).status,
        },
    )
    assert not sb.started, "recipe B counts as started because recipe A was started"
    assert not _ops(hub, b), "the worker ran recipe B, which nobody started"


def test_cancelling_one_recipe_does_not_cancel_another(hub: Any, registered_recipe: Any) -> None:
    a, fake = start_run(hub)
    b = _second_recipe(hub, AGENT, "b2")
    recipe_start(hub.log, b, AGENT, idempotency_key="start-b", dispatcher=fake)
    assert load_state(hub.log, b).recipe_status == "active"
    recipe_cancel(hub.log, a, AGENT, idempotency_key="cancel-a")
    sb = load_state(hub.log, b)
    worker_idle(hub, fake)
    _obs(
        "B after cancelling A",
        {
            "A": load_state(hub.log, a).recipe_status,
            "B": sb.recipe_status,
            "B_graph_status": WorkGraph(hub.log, hub.project_id).node(b).status,
            "B_ops_status": [o.status for o in load_state(hub.log, b).ops],
        },
    )
    assert sb.recipe_status != "cancelled", "cancelling recipe A cancelled recipe B too"


# -- forged runner events through the generic event API (local trust; kills M3) --------------


def _forge(hub: Any, type_: str, payload: dict[str, Any], actor: Actor = AGENT) -> None:
    hub.log.append(type_, actor, payload, project_id=hub.project_id)


def test_forged_recipe_result_on_the_gate_op_is_ignored(hub: Any, registered_recipe: Any) -> None:
    rn, fake = to_gate(hub)
    g = gate_op(hub, rn)
    _forge(
        hub,
        "recipe.result",
        {
            "op_node": g.op_id,
            "outcome": "done",
            "outputs": {},
            "artifact_ids": [],
            "error": None,
            "by": "agent:agent-1",
        },
    )
    for _ in range(3):
        advance(hub.log, rn, actor=AGENT, dispatcher=fake)
    worker_idle(hub, fake)
    st = load_state(hub.log, rn)
    review = get_review(hub.log, hub.project_id, g.review_id)
    _obs(
        "forged recipe.result on gate",
        {
            "gate": st.latest_in_round("g", 1).status,
            "recipe": st.recipe_status,
            "review_decided": review.decision is not None,
            "ignored": st.ignored[:3],
        },
    )
    assert st.latest_in_round("g", 1).status != "done"
    assert st.recipe_status != "done"
    assert any(i.get("type") == "recipe.result" for i in st.ignored)


def test_forged_link_to_another_recipes_approved_runner_review_is_ignored(
    hub: Any, registered_recipe: Any
) -> None:
    """Recipe A's gate review (RUNNER-created) is approved by a human; an agent forges a link
    from recipe B's gate op to A's review."""
    a, fake = to_gate(hub)
    approve(hub, a, fake)
    a_review = gate_op(hub, a).review_id
    b = _second_recipe(hub, AGENT, "b3")
    recipe_start(hub.log, b, AGENT, idempotency_key="start-b3", dispatcher=fake)
    worker_idle(hub, fake)
    complete_plan(hub, b, fake, "b3")
    gb = load_state(hub.log, b).latest_in_round("g", 1)
    _forge(
        hub,
        "recipe.review_linked",
        {"op_node": gb.op_id, "review_id": a_review, "round": 1, "subject": "x"},
    )
    for _ in range(3):
        advance(hub.log, b, actor=AGENT, dispatcher=fake)
    st = load_state(hub.log, b)
    _obs(
        "B gate after forged link to A's approved review",
        {
            "gate": st.latest_in_round("g", 1).status,
            "review_id": st.latest_in_round("g", 1).review_id,
            "a_review": a_review,
            "recipe": st.recipe_status,
        },
    )
    assert st.latest_in_round("g", 1).status != "done"


def test_forged_started_and_cancel_events_are_ignored(hub: Any, registered_recipe: Any) -> None:
    from recipe_support import make_run

    rn = make_run(hub)  # not started
    _forge(hub, "recipe.started", {"recipe_node": rn})
    s1 = load_state(hub.log, rn)
    fake = hot_fake()
    rn2 = _second_recipe(hub, AGENT, "fs")
    recipe_start(hub.log, rn2, AGENT, idempotency_key="s-fs", dispatcher=fake)
    _forge(hub, "recipe.run_cancelled", {"recipe_node": rn2, "reason": "x"})
    op = load_state(hub.log, rn2).latest_in_round("c", 1)
    _forge(hub, "recipe.cancelled", {"op_node": op.op_id, "reason": "x"})
    s2 = load_state(hub.log, rn2)
    _obs(
        "forged started/run_cancelled/cancelled",
        {
            "rn_started": s1.started,
            "rn2_status": s2.recipe_status,
            "c_op": s2.op_by_id(op.op_id).status,
        },
    )
    assert not s1.started
    assert s2.recipe_status != "cancelled"
    assert s2.op_by_id(op.op_id).status != "cancelled"
