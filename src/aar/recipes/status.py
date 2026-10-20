"""The agent-native status view: what each step is waiting on, and who acts next.

:func:`status_view` folds the run and returns a bounded JSON object: a per-step state, attempt,
round, op node id, an outputs summary and the error, plus a ``next_actions`` list that says who
acts (agent, human, worker or node), what to do, and -- for agents -- the exact tool and argument
template. It is a pure read over the log; with a :class:`~aar.recipes.ports.HotDispatcher` it also
surfaces the node's ``waiting_for_node`` / ``node_offline`` states unchanged (poll is a read).
"""

from __future__ import annotations

from typing import Any

from aar.hub.common import Clock, utcnow
from aar.hub.events import EventLog
from aar.recipes.models import StepDef
from aar.recipes.state import OpAttempt, RunState, load_state

MAX_OUTPUTS_KEYS = 32


def status_view(
    log: EventLog, recipe_node: str, *, dispatcher: Any = None, clock: Clock = utcnow
) -> dict[str, Any]:
    """Return the bounded status view for the run of ``recipe_node``."""
    state = load_state(log, recipe_node)
    if state.recipe_data is None:
        return {"found": False, "recipe_node": recipe_node}
    current = state.current_round()
    steps_out: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    registered = state.recipe_def is not None
    step_defs: list[StepDef] = list(state.recipe_def.steps) if registered else []
    for step in step_defs:
        op = state.latest_in_round(step.id, current)
        step_state, action = _step_state(state, step, op, current, dispatcher)
        entry: dict[str, Any] = {
            "step_id": step.id,
            "kind": step.kind,
            "state": step_state,
            "round": current,
            "attempt": op.attempt if op is not None else None,
            "op_node": op.op_id if op is not None else None,
            "outputs": _summary(op.result_outputs) if op is not None else None,
            "error": op.result_error if op is not None else None,
        }
        steps_out.append(entry)
        if action is not None:
            actions.append({"step_id": step.id, **action})
    recipe_state = state.recipe_status
    if state.rounds_exhausted():
        actions.append(
            {
                "step_id": None,
                "who": "human",
                "what": "review rounds exhausted; intervene in the console",
            }
        )
    return {
        "found": True,
        "recipe_node": recipe_node,
        "project_id": state.project_id,
        "recipe": state.recipe_data.recipe,
        "registered": registered,
        "status": recipe_state,
        "round": current,
        "max_review_rounds": state.max_review_rounds,
        "steps": steps_out,
        "next_actions": actions,
    }


def _summary(outputs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not outputs:
        return None
    keys = sorted(outputs)[:MAX_OUTPUTS_KEYS]
    return {key: outputs[key] for key in keys}


def _retry_action(recipe_node: str, step: StepDef) -> dict[str, Any]:
    return {
        "who": "agent",
        "what": f"retry the failed {step.kind} step {step.id!r}",
        "tool": "recipe_retry",
        "args": {"recipe_node": recipe_node, "step_id": step.id, "idempotency_key": "<unique>"},
    }


def _step_state(
    state: RunState, step: StepDef, op: OpAttempt | None, current: int, dispatcher: Any
) -> tuple[str, dict[str, Any] | None]:
    if op is None:
        if all(state.is_step_done(need) for need in step.needs):
            return "ready", None
        return "waiting_on_needs", None
    if op.status == "done":
        return "done", None
    if op.status == "failed":
        attempts = len([o for o in state.in_round(step.id, current) if o.status == "failed"])
        if attempts < step.max_attempts:
            return "failed", _retry_action(state.recipe_node, step)
        return "failed", {"who": "human", "what": f"step {step.id!r} is out of attempts"}
    # pending / active
    if step.kind == "cold":
        if op.status == "pending":
            return "ready", {"who": "worker", "what": f"run cold step {step.id!r}"}
        return "running", {"who": "worker", "what": f"cold step {step.id!r} is running"}
    if step.kind == "hot":
        return _hot_state(op, step, dispatcher)
    if step.kind == "agent":
        return "needs_agent", {
            "who": "agent",
            "what": f"complete agent step {step.id!r}: {step.instruction}",
            "tool": "step_complete",
            "args": {
                "recipe_node": state.recipe_node,
                "step_id": step.id,
                "attempt": op.attempt,
                "outputs": "<per the expects contract>",
                "idempotency_key": "<unique>",
            },
        }
    if step.kind == "gate":
        review_id = op.review_id
        return "needs_human", {
            "who": "human",
            "what": f"decide the review for gate {step.id!r} in the console",
            "review_id": review_id,
        }
    return op.status, None


def _hot_state(op: OpAttempt, step: StepDef, dispatcher: Any) -> tuple[str, dict[str, Any] | None]:
    if not op.hot_submitted:
        if not op.hot_intent:
            return "waiting:no_node_dispatcher", {
                "who": "node",
                "what": f"bring the Resolve node online for hot step {step.id!r}",
            }
        return "waiting:submitting", {"who": "node", "what": f"re-submit hot step {step.id!r}"}
    if dispatcher is not None and op.hot_job_ref is not None:
        view = dispatcher.poll(op.hot_job_ref)
        if view.state in ("waiting_for_node", "node_offline"):
            return f"waiting:{view.state}", {
                "who": "node",
                "what": f"the node reports {view.state}",
            }
        if view.state == "running":
            return "running", None
    return "waiting:node", {"who": "node", "what": f"hot step {step.id!r} is on the node"}
