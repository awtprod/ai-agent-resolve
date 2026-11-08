"""Validator probes: can an agent get past a human gate?

Each test asserts the SAFE property. A failure is a finding; the failure message carries the
evidence. Run with -s to see the printed observations.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from recipe_support import hot_fake, make_run, passing_qa_report, store_qa_report
from val_support import (
    approve,
    complete_plan,
    gate_op,
    mcp_run,
    reject,
    start_run,
    step_states,
    to_gate,
    worker_idle,
)
from write_support import AGENT, HUMAN, OTHER, brief_data, plan_fixture

from aar.hub.events import Actor
from aar.hub.graph import WorkGraph
from aar.hub.reviews import get_review, withdraw
from aar.hub.writes import WriteError, set_own_status
from aar.recipes import registry
from aar.recipes.engine import advance
from aar.recipes.report import build_report_text
from aar.recipes.run import recipe_cancel, recipe_retry, recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)}")


# -- 1. the creator agent moves its own gate op to done ------------------------------------


def test_creator_agent_cannot_mark_gate_op_done(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub)
    op = gate_op(hub, recipe_node)
    graph = WorkGraph(hub.log, hub.project_id)
    _obs("gate op creator", [graph.creator(op.op_id).kind, graph.creator(op.op_id).id])
    _obs("gate op status / review", [op.status, op.review_id])
    try:
        res = set_own_status(
            hub.log, hub.project_id, op.op_id, "done", AGENT, idempotency_key="sneak-gate"
        )
        _obs("set_own_status(gate op -> done)", res.body)
    except WriteError as exc:
        _obs("set_own_status refused", exc.code)
        return
    # Accepted: does the runner treat the gate as passed?
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    worker_idle(hub, fake)
    state = load_state(hub.log, recipe_node)
    view = status_view(hub.log, recipe_node, dispatcher=fake)
    review = get_review(hub.log, hub.project_id, op.review_id)
    _obs("after: recipe status", state.recipe_status)
    _obs("after: step states", step_states(view))
    _obs("after: review decision", None if review.decision is None else review.decision.decision)
    report = build_report_text(hub.log, recipe_node)
    gates = report.split("## Gates")[1].split("##")[0].strip()
    _obs("after: REPORT gates section", gates)
    assert not state.is_step_done("g"), (
        "GATE BYPASS: an agent moved its own gate op to done; the scheduler treats the gate as "
        f"passed with no human decision (recipe={state.recipe_status}, steps={step_states(view)})"
    )


def test_gate_bypass_through_mcp_tools_only(hub: Any, registered_recipe: Any) -> None:
    """The same route through the MCP write tools an agent has with --agent-writes."""
    from aar.hub.mcp_server import build_server
    from aar.hub.registry import Registry

    server = build_server(
        hub.log,
        Registry(hub.log),
        hub.store,
        agent_writes=True,
        params_validator=registry.registry_params_validator,
    )
    pid = hub.project_id
    qa = store_qa_report(hub.store, passing_qa_report(), Actor("agent", "ed"))
    out: dict[str, Any] = {}

    async def setup(client: Any) -> None:
        b = await client.call_tool(
            "brief_create",
            {"agent_id": "ed", "idempotency_key": "b", "brief": brief_data(pid, recipe="rectest")},
        )
        brief_id = b.structured_content["node"]["id"]
        r = await client.call_tool(
            "recipe_create",
            {
                "agent_id": "ed",
                "idempotency_key": "r",
                "project_id": pid,
                "recipe": {"recipe": "rectest", "brief_id": brief_id, "params": {"qa_report": qa}},
            },
        )
        out["recipe"] = r.structured_content["node"]["id"]
        s = await client.call_tool(
            "recipe_start",
            {"agent_id": "ed", "idempotency_key": "s", "recipe_node": out["recipe"]},
        )
        assert s.structured_content["ok"] is True
        p = await client.call_tool(
            "plan_submit",
            {
                "agent_id": "ed",
                "idempotency_key": "p1",
                "project_id": pid,
                "derives_from": out["recipe"],
                "plan": plan_fixture("short_clean"),
            },
        )
        plan_node = p.structured_content["node"]["id"]
        c = await client.call_tool(
            "step_complete",
            {
                "agent_id": "ed",
                "idempotency_key": "c1",
                "recipe_node": out["recipe"],
                "step_id": "p",
                "outputs": {"plan_node": plan_node},
            },
        )
        assert not c.is_error, c.structured_content
        st = await client.call_tool("recipe_status", {"recipe_node": out["recipe"]})
        g = next(s for s in st.structured_content["steps"] if s["step_id"] == "g")
        out["gate_state_before"] = g["state"]
        out["gate_op"] = g["op_node"]
        res = await client.call_tool(
            "node_set_status",
            {
                "agent_id": "ed",
                "idempotency_key": "sneak",
                "project_id": pid,
                "node_id": g["op_node"],
                "status": "done",
            },
        )
        out["node_set_status"] = res.structured_content
        out["node_set_status_is_error"] = res.is_error

    mcp_run(server, setup)
    _obs("mcp", out)
    # A worker (any process running `aar recipe worker`) then advances the run.
    fake = hot_fake()
    worker_idle(hub, fake)
    state = load_state(hub.log, out["recipe"])
    view = status_view(hub.log, out["recipe"], dispatcher=fake)
    _obs("after worker: recipe status", state.recipe_status)
    _obs("after worker: steps", step_states(view))
    assert out["node_set_status_is_error"], (
        "GATE BYPASS via MCP: node_set_status moved the gate op to done "
        f"(recipe={state.recipe_status}, steps={step_states(view)})"
    )


def test_creator_agent_cannot_mark_recipe_done(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub)
    try:
        set_own_status(hub.log, hub.project_id, recipe_node, "done", AGENT, idempotency_key="rd")
    except WriteError as exc:
        _obs("recipe -> done refused", exc.code)
        return
    view = status_view(hub.log, recipe_node, dispatcher=fake)
    _obs("status view after agent moved recipe to done", [view["status"], step_states(view)])
    from aar.recipes.worker import active_recipes

    _obs("worker still sees it", (hub.project_id, recipe_node) in active_recipes(hub.log))
    assert view["status"] != "done", (
        "the creator agent moved the recipe node to done with the gate undecided; the status "
        f"view reports status={view['status']!r} (steps {step_states(view)})"
    )


def test_creator_agent_cannot_skip_the_cold_qa_step(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub)
    approve(hub, recipe_node, fake)
    q = load_state(hub.log, recipe_node).latest_in_round("q", 1)
    _obs("q op before", [q.status, q.kind])
    try:
        set_own_status(hub.log, hub.project_id, q.op_id, "active", AGENT, idempotency_key="q1")
        set_own_status(hub.log, hub.project_id, q.op_id, "done", AGENT, idempotency_key="q2")
    except WriteError as exc:
        _obs("q skip refused", exc.code)
        return
    worker_idle(hub, fake)
    state = load_state(hub.log, recipe_node)
    report = build_report_text(hub.log, recipe_node)
    qa = report.split("## QA")[1].split("##")[0].strip()
    _obs("after: recipe", state.recipe_status)
    _obs("after: REPORT QA section", qa)
    assert not state.is_step_done("q"), (
        f"QA BYPASS: the agent marked the cold QA op done without running it; recipe="
        f"{state.recipe_status}; REPORT QA section: {qa!r}"
    )


def test_exhausted_gate_cannot_be_revived_by_creator(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub, max_review_rounds=1)
    reject(hub, recipe_node, fake)
    op = gate_op(hub, recipe_node)
    st = load_state(hub.log, recipe_node)
    _obs("exhausted gate", [op.status, (op.result_error or {}).get("code"), st.recipe_status])
    try:
        set_own_status(hub.log, hub.project_id, op.op_id, "active", AGENT, idempotency_key="g1")
        set_own_status(hub.log, hub.project_id, op.op_id, "done", AGENT, idempotency_key="g2")
    except WriteError as exc:
        _obs("revive refused", exc.code)
        return
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    worker_idle(hub, fake)
    state = load_state(hub.log, recipe_node)
    _obs("after revive: steps", step_states(status_view(hub.log, recipe_node)))
    _obs("after revive: recipe", state.recipe_status)
    assert not state.is_step_done("g"), (
        "an agent revived a review_rounds_exhausted gate (failed->active->done); the scheduler "
        f"counts it done (q done={state.is_step_done('q')}, recipe={state.recipe_status})"
    )


# -- 2. step_complete on non-agent steps, wrong attempt, wrong round -----------------------


@pytest.mark.parametrize("step_id", ["g", "c", "h", "q", "report"])
def test_step_complete_refuses_non_agent_steps(
    hub: Any, registered_recipe: Any, step_id: str
) -> None:
    recipe_node, fake = to_gate(hub)
    with pytest.raises(WriteError) as excinfo:
        step_complete(
            hub.log,
            hub.store,
            recipe_node,
            step_id,
            None,
            {"decision": "approve"},
            OTHER,
            idempotency_key=f"x-{step_id}",
            dispatcher=fake,
        )
    _obs(f"step_complete({step_id})", excinfo.value.code)
    assert excinfo.value.code == "step_not_open"
    assert not load_state(hub.log, recipe_node).is_step_done("g")


def test_step_complete_wrong_attempt_refused(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = start_run(hub)
    plan = __import__("recipe_support").submit_test_plan(hub, recipe_node, AGENT, key="pl")
    for bad in (0, 2, -1):
        with pytest.raises(WriteError) as excinfo:
            step_complete(
                hub.log,
                hub.store,
                recipe_node,
                "p",
                bad,
                {"plan_node": plan},
                AGENT,
                idempotency_key=f"wa{bad}".replace("-", "m"),
                dispatcher=fake,
            )
        assert excinfo.value.code == "step_not_open", bad
    assert load_state(hub.log, recipe_node).latest_in_round("p", 1).status == "active"


def test_replayed_round1_step_complete_does_not_complete_round2(
    hub: Any, registered_recipe: Any
) -> None:
    """A retried round-1 call (same key, same rejected plan, attempt=1) arrives in round 2."""
    recipe_node, fake = start_run(hub)
    worker_idle(hub, fake)
    plan1 = complete_plan(hub, recipe_node, fake, "r1")
    review1 = gate_op(hub, recipe_node).review_id
    reject(hub, recipe_node, fake)
    assert load_state(hub.log, recipe_node).current_round() == 2
    outcome: Any
    try:
        res = step_complete(
            hub.log,
            hub.store,
            recipe_node,
            "p",
            1,
            {"plan_node": plan1},
            AGENT,
            idempotency_key="complete-r1",  # the SAME key as the round-1 call
            dispatcher=fake,
        )
        # A4: F9 replay contract: a replayed call returns replayed=True and must not complete r2.
        outcome = "replayed" if res.get("replayed") else "accepted"
        p2 = load_state(hub.log, recipe_node).latest_in_round("p", 2)
        assert p2 is None or p2.status != "done", "replay completed round 2"
    except WriteError as exc:
        outcome = exc.code
    state = load_state(hub.log, recipe_node)
    g2 = state.latest_in_round("g", 2)
    review2 = None if g2 is None else get_review(hub.log, hub.project_id, g2.review_id)
    _obs("replayed round-1 step_complete in round 2", outcome)
    _obs(
        "round-2 gate review subject vs rejected plan",
        [None if review2 is None else review2.subject, plan1, review1],
    )
    assert outcome != "accepted", (
        "a replay of the round-1 step_complete (same idempotency key, the plan the human just "
        "rejected) completed the round-2 plan attempt and re-opened a review of the rejected plan"
    )


# -- 3. recipe_retry on a gate that failed review_rounds_exhausted ---------------------------


def test_retry_of_exhausted_gate(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub, max_review_rounds=1)
    reject(hub, recipe_node, fake)
    op1 = gate_op(hub, recipe_node)
    assert (op1.result_error or {}).get("code") == "review_rounds_exhausted"
    try:
        res = recipe_retry(hub.log, recipe_node, "g", AGENT, idempotency_key="rg", dispatcher=fake)
        outcome = "accepted"
    except WriteError as exc:
        outcome = exc.code
        res = None
    _obs("retry exhausted gate", outcome)
    state = load_state(hub.log, recipe_node)
    gates = [(o.round, o.attempt, o.status, o.review_id) for o in state.attempts_for("g")]
    _obs("gate attempts after retry", gates)
    _obs("round after retry / max", [state.current_round(), state.max_review_rounds])
    if res is not None:
        new = state.latest_in_round("g", state.current_round())
        if new.review_id is not None:
            approve(hub, recipe_node, fake)
            worker_idle(hub, fake)
            st2 = load_state(hub.log, recipe_node)
            _obs(
                "after human approves the retried gate",
                [st2.recipe_status, step_states(status_view(hub.log, recipe_node))],
            )
    assert outcome != "accepted", (
        "recipe_retry on a gate that failed review_rounds_exhausted is accepted: it opens a "
        f"fresh review past max_review_rounds (gate attempts {gates})"
    )


# -- 4. the requester withdraws the gate review ----------------------------------------------


def test_requester_withdrawal_of_gate_review(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub)
    op = gate_op(hub, recipe_node)
    review = get_review(hub.log, hub.project_id, op.review_id)
    _obs("review requested_by", review.requested_by)
    try:
        withdraw(hub.log, hub.project_id, op.review_id, "agent changed its mind", AGENT)
        outcome = "withdrawn"
    except Exception as exc:  # NotRequester etc.
        outcome = type(exc).__name__
    _obs("agent withdraw of the gate review", outcome)
    for _ in range(3):
        advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    state = load_state(hub.log, recipe_node)
    op2 = gate_op(hub, recipe_node)
    view = status_view(hub.log, recipe_node, dispatcher=fake)
    g = next(s for s in view["steps"] if s["step_id"] == "g")
    act = [a for a in view["next_actions"] if a.get("step_id") == "g"]
    _obs("after withdraw: gate op", [op2.status, op2.review_id, state.is_step_done("g")])
    _obs("after withdraw: status view gate", [g["state"], act])
    rv = get_review(hub.log, hub.project_id, op2.review_id)
    _obs("after withdraw: review status", rv.status)
    assert not state.is_step_done("g"), "a withdrawn review let the gate pass"
    # Liveness: the runner must not leave the gate waiting on a closed review forever.
    assert rv.pending or op2.review_id != op.review_id, (
        "after the requester withdrew the gate review, the runner keeps the gate 'needs_human' "
        "pointing at a CLOSED review; no human can decide it, and advance never re-opens one"
    )


# -- 5. a decision by a non-human, written through the generic event API ---------------------


@pytest.mark.parametrize("kind", ["agent", "node"])
def test_non_human_decision_is_ignored(hub: Any, registered_recipe: Any, kind: str) -> None:
    recipe_node, fake = to_gate(hub)
    op = gate_op(hub, recipe_node)
    forger = Actor(kind, "agent-1" if kind == "agent" else "node-1")
    hub.log.append(
        "graph.node_status_changed",
        forger,
        {"node_id": op.review_id, "to": "done"},
        project_id=hub.project_id,
    )
    hub.log.append(
        "review.decided",
        forger,
        {"review_id": op.review_id, "decision": "approve", "reason": "lgtm"},
        project_id=hub.project_id,
    )
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    worker_idle(hub, fake)
    state = load_state(hub.log, recipe_node)
    rv = get_review(hub.log, hub.project_id, op.review_id)
    _obs(f"forged {kind} decision: review decision", None if rv.decision is None else "set")
    _obs(f"forged {kind} decision: gate done", state.is_step_done("g"))
    assert not state.is_step_done("g")
    assert rv.decision is None


def test_forged_review_link_to_an_approved_review(hub: Any, registered_recipe: Any) -> None:
    """recipe.* events are folded from any actor: link the gate op to a review a human approved
    for something else (generic event API, local trust)."""
    from aar.hub.reviews import decide

    recipe_node, fake = to_gate(hub)
    op = gate_op(hub, recipe_node)
    # Some other review in the project, requested by the agent and approved by a human
    # (opened the generic way the hub's review request does: a review node + reviews edge).
    plan_other = __import__("recipe_support").submit_test_plan(hub, recipe_node, AGENT, key="o")
    graph = WorkGraph(hub.log, hub.project_id)
    node = graph.add_node(
        "review",
        AGENT,
        title="unrelated",
        status="active",
        data={"subject": plan_other, "renders": [], "requested_by": "agent:agent-1"},
    )
    graph.add_edge("reviews", node.id, plan_other, AGENT)
    other_id = node.id
    decide(hub.log, hub.project_id, other_id, "approve", "fine", HUMAN)
    hub.log.append(
        "recipe.review_linked",
        AGENT,
        {"op_node": op.op_id, "review_id": other_id, "round": 1, "subject": plan_other},
        project_id=hub.project_id,
    )
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    state = load_state(hub.log, recipe_node)
    _obs("forged link: gate done", state.is_step_done("g"))
    assert not state.is_step_done("g"), (
        "a recipe.review_linked event appended by the agent (generic event API) re-pointed the "
        "gate at an unrelated human-approved review; the runner counted the gate approved"
    )


# -- 6. who may start / cancel / retry --------------------------------------------------------


def test_start_cancel_retry_permissions(hub: Any, registered_recipe: Any) -> None:
    fake = hot_fake()
    recipe_node = make_run(hub)
    for fn, kw in (
        (recipe_start, {"dispatcher": fake}),
        (recipe_cancel, {}),
    ):
        with pytest.raises(WriteError) as excinfo:
            fn(hub.log, recipe_node, OTHER, idempotency_key="o1", **kw)
        _obs(f"{fn.__name__} by other agent", excinfo.value.code)
        assert excinfo.value.code == "not_creator"
    # a human may start someone else's recipe
    recipe_start(hub.log, recipe_node, HUMAN, idempotency_key="h1", dispatcher=fake)
    assert load_state(hub.log, recipe_node).recipe_status == "active"
    with pytest.raises(WriteError) as excinfo:
        recipe_retry(hub.log, recipe_node, "c", OTHER, idempotency_key="o2", dispatcher=fake)
    _obs("retry by other agent (no failed step)", excinfo.value.code)
    # retry of a step that has not failed
    with pytest.raises(WriteError) as excinfo:
        recipe_retry(hub.log, recipe_node, "p", AGENT, idempotency_key="a2", dispatcher=fake)
    _obs("retry of a non-failed step", excinfo.value.code)
    assert excinfo.value.code == "step_not_open"


def test_human_cancel_of_agent_recipe_withdraws_review(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub)
    op = gate_op(hub, recipe_node)
    res = recipe_cancel(hub.log, recipe_node, HUMAN, idempotency_key="hc")
    state = load_state(hub.log, recipe_node)
    rv = get_review(hub.log, hub.project_id, op.review_id)
    live = [o.step_id for o in state.ops if o.is_live]
    _obs("human cancel", [res["status"], rv.status, live])
    assert res["status"] == "cancelled" and rv.status == "cancelled" and not live
    # the worker no longer touches it
    assert worker_idle(hub, fake) == 0


def test_agent_cancel_then_restart_refused(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = to_gate(hub)
    recipe_cancel(hub.log, recipe_node, AGENT, idempotency_key="ac")
    with pytest.raises(WriteError) as excinfo:
        recipe_start(hub.log, recipe_node, AGENT, idempotency_key="again", dispatcher=fake)
    _obs("start after cancel", excinfo.value.code)


def test_retry_replay_with_same_key_is_idempotent(hub: Any, tmp_path: Any) -> None:
    """recipe_retry twice with ONE idempotency key must not create two new attempts."""
    from pydantic import BaseModel

    from aar.recipes.context import StepFailure
    from aar.recipes.models import RecipeDef, StepDef

    class P(BaseModel):
        pass

    def boom(ctx: Any) -> Any:
        raise StepFailure("boom", "always fails")

    registry.clear()
    registry.register_callable("val.boom", boom)
    registry.register(
        RecipeDef(
            name="valboom",
            version=1,
            params_model=P,
            steps=(StepDef(id="b", kind="cold", callable_id="val.boom", max_attempts=5),),
        )
    )
    try:
        from aar.hub.writes import create_brief, create_recipe

        brief = create_brief(
            hub.log, brief_data(hub.project_id, recipe="valboom"), AGENT, idempotency_key="bb"
        )
        rn = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "valboom", "brief_id": brief.body["node"]["id"], "params": {}},
            AGENT,
            idempotency_key="rr",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        recipe_start(hub.log, rn, AGENT, idempotency_key="s")
        worker_idle(hub, None)
        recipe_retry(hub.log, rn, "b", AGENT, idempotency_key="retry-1")
        worker_idle(hub, None)
        outcome: Any
        try:
            recipe_retry(hub.log, rn, "b", AGENT, idempotency_key="retry-1")  # a network retry
            outcome = "accepted"
        except WriteError as exc:
            outcome = exc.code
        attempts = [o.attempt for o in load_state(hub.log, rn).attempts_for("b")]
        _obs("second recipe_retry with the same key", [outcome, attempts])
        assert outcome in ("replayed", "idempotency_conflict") or len(attempts) == 2, (
            f"a replay of one recipe_retry call (same key) created another attempt: {attempts}"
        )
    finally:
        registry.clear()
