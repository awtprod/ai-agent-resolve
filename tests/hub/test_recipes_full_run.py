"""A full run of the test recipe on the fake, asserting on the log, not just return values."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest
from recipe_support import (
    hot_fake,
    make_run,
    submit_test_plan,
    test_recipe,  # noqa: F401 (fixture)
)
from write_support import AGENT, Hub, make_hub

from aar.hub.reviews import decide, scan_reviews
from aar.recipes.engine import advance
from aar.recipes.report import build_report_text
from aar.recipes.run import recipe_start, step_complete
from aar.recipes.state import load_state


@pytest.fixture
def hub(tmp_path: Any) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


HUMAN = __import__("write_support").HUMAN


def _open_gate_review(hub: Hub, recipe_node: str) -> str:
    state = load_state(hub.log, recipe_node)
    current = state.current_round()
    op = state.latest_in_round("g", current)
    assert op is not None and op.review_id is not None, "the gate review should be open"
    return op.review_id


def _run_worker_idle(hub: Hub, fake: Any) -> None:
    from aar.recipes.worker import run_worker

    for _ in range(20):
        if run_worker(hub.log, hub.store, AGENT, dispatcher=fake, once=True) == 0:
            break


def test_full_run_start_worker_hot_agent_reject_then_approve(hub: Hub, test_recipe: Any) -> None:
    fake = hot_fake()
    recipe_node = make_run(hub)

    # Start: cold c pending, hot h submitted and completed on the fake, agent p active.
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start-1", dispatcher=fake)
    state = load_state(hub.log, recipe_node)
    assert state.recipe_status == "active"
    assert state.is_step_done("h"), "the hot step completes on the fake during advance"
    assert fake.calls == [state.latest_in_round("h", 1).op_id]

    # The worker runs the cold step.
    _run_worker_idle(hub, fake)
    assert load_state(hub.log, recipe_node).is_step_done("c")

    # Round 1: the agent submits a plan and completes its step; the gate opens.
    plan_node = submit_test_plan(hub, recipe_node, AGENT, key="plan-r1")
    step_complete(
        hub.log, hub.store, recipe_node, "p", None, {"plan_node": plan_node}, AGENT,
        idempotency_key="complete-r1", dispatcher=fake,
    )
    review1 = _open_gate_review(hub, recipe_node)

    # The human rejects once: round 2 re-runs from the plan step.
    decide(hub.log, hub.project_id, review1, "reject", "needs a tighter cut", HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    state = load_state(hub.log, recipe_node)
    assert state.current_round() == 2
    p_round1 = [op for op in state.attempts_for("p") if op.round == 1][0]
    assert p_round1.is_cancelled, "the round-1 plan attempt is superseded"
    assert not state.is_step_done("p"), "the plan must run again in round 2"

    # Round 2: resubmit a plan, complete, and the human approves.
    plan_node2 = submit_test_plan(hub, recipe_node, AGENT, key="plan-r2")
    step_complete(
        hub.log, hub.store, recipe_node, "p", None, {"plan_node": plan_node2}, AGENT,
        idempotency_key="complete-r2", dispatcher=fake,
    )
    review2 = _open_gate_review(hub, recipe_node)
    assert review2 != review1
    decide(hub.log, hub.project_id, review2, "approve", "ship it", HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)

    # The QA step runs and the report is written; the recipe finishes.
    _run_worker_idle(hub, fake)
    state = load_state(hub.log, recipe_node)
    assert state.recipe_status == "done", state.recipe_status
    assert state.is_step_done("q") and state.is_step_done("report")

    # Assert on the log: the runner's event families are all present.
    counts = hub.counts()
    assert counts["recipe.hot_intent"] == 1
    assert counts["recipe.hot_submitted"] == 1
    assert counts["recipe.review_linked"] == 2  # one review per round
    assert counts["recipe.cancelled"] >= 2  # round-1 plan and gate superseded
    assert counts["recipe.claimed"] >= 3  # c, q, report claimed by the worker
    assert counts["recipe.result"] >= 6

    # Two reviews exist, the first rejected and the second approved.
    reviews = {r.id: r for r in scan_reviews(hub.log, hub.project_id).reviews}
    assert reviews[review1].decision.decision == "reject"
    assert reviews[review2].decision.decision == "approve"

    # The REPORT.md artifact is present and names the gate decisions and the QA pass.
    report_step = state.done_op("report")
    assert report_step is not None and report_step.result_artifacts
    report_text = build_report_text(hub.log, recipe_node)
    assert "## Gates" in report_text
    assert "approve by human:operator-1" in report_text
    assert "qa.lufs.master" in report_text


def test_pc_off_no_dispatcher_waits_but_cold_runs(hub: Hub, test_recipe: Any) -> None:
    # No dispatcher: the hot step stays pending (waiting: no_node_dispatcher), cold still runs.
    recipe_node = make_run(hub)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start-1")  # no dispatcher
    _run_worker_idle(hub, None)
    state = load_state(hub.log, recipe_node)
    assert state.is_step_done("c"), "the cold step runs with the PC off"
    from aar.recipes.status import status_view

    view = status_view(hub.log, recipe_node)
    hot = next(step for step in view["steps"] if step["step_id"] == "h")
    assert hot["state"] == "waiting:no_node_dispatcher"
    h_counts = Counter(step["state"] for step in view["steps"])
    assert h_counts["waiting:no_node_dispatcher"] == 1
