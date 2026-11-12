"""When a recipe's agent step may be reopened (S19f), as a pure check over the folded run.

:func:`reopen_refusal` is the one precondition check: ``recipe_reopen`` raises what it returns,
and the status view offers ``recipe_reopen`` exactly when it returns ``None``
(:func:`reopen_offer`). The writes are :func:`aar.recipes.engine.begin_reopen` and
:func:`aar.recipes.engine.finish_reopen`; :func:`reopen_unfinished` and :func:`reopening` tell
the engine, the write commands and the status view that a reopen's writes are not all made yet.
"""

from __future__ import annotations

from typing import Any

from aar.hub.writes import WriteError
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.state import OpAttempt, RunState

_FINISHED = frozenset({"done", "cancelled"})


def reopen_refusal(state: RunState, definition: RecipeDef, step: StepDef) -> WriteError | None:
    """The first precondition of a reopen of ``step`` that fails, as a typed refusal, or
    ``None``. The same check decides whether the status view offers ``recipe_reopen``."""
    current = state.current_round()
    status = state.recipe_status
    if status in _FINISHED or not state.started:
        return WriteError(
            "recipe_not_open",
            f"recipe {state.recipe_node!r} is {status!r}; only a running recipe can be reopened",
            "A cancelled or finished run cannot be reopened; start a new run (recipe_create).",
            status=status,
        )
    blocker = reopen_blocker(state, definition, step)
    if blocker is not None:
        return WriteError(
            "recipe_blocked",
            f"recipe {state.recipe_node!r} is blocked by {blocker!r}, which a reopen of "
            f"{step.id!r} does not clear",
            "Read recipe_status for what the block needs (a human, or recipe_cancel).",
            reason=blocker,
        )
    if step.kind != "agent":
        return WriteError(
            "not_agent_step",
            f"step {step.id!r} is a {step.kind} step; only an agent step can be reopened",
            "Reopen the agent step a failed step's on_fail names (recipe_status shows it).",
            step_id=step.id,
            kind=step.kind,
        )
    op = state.latest_in_round(step.id, current)
    if op is None or op.status != "done":
        return WriteError(
            "step_not_done",
            f"step {step.id!r} has no done attempt in round {current} (it is "
            f"{op.status if op is not None else 'not started'})",
            "Only a completed agent step can be reopened; complete the open attempt instead.",
            step_id=step.id,
            round=current,
            open_attempt=op.attempt if op is not None else None,
        )
    if state.upstream_failure(step.id, current) is None:
        return WriteError(
            "no_upstream_failure",
            f"no step declaring on_fail={step.id!r} has, as its latest attempt in round "
            f"{current}, a failure caused by {step.id!r}'s output",
            "Reopen only after such a failure; otherwise retry the failed step (recipe_retry).",
            step_id=step.id,
            round=current,
        )
    used = len(state.all_in_round(step.id, current))
    if used >= step.max_attempts:
        return WriteError(
            "no_attempts_left",
            f"step {step.id!r} has used all {step.max_attempts} attempts in round {current}",
            "Cancel the run (recipe_cancel) and start a new one.",
            step_id=step.id,
            max_attempts=step.max_attempts,
            used=used,
        )
    return None


def reopen_blocker(state: RunState, definition: RecipeDef, step: StepDef) -> str | None:
    """Why the run is blocked when a reopen of ``step`` would not clear it, else ``None``. The
    one block a reopen clears: a cold step declaring ``on_fail == step`` whose latest attempt
    failed upstream and that has run out of attempts."""
    if state.blocked_reason() is None:
        return None
    if state.rounds_exhausted():
        return "review_rounds_exhausted"
    current = state.current_round()
    for candidate in definition.steps:
        op = state.latest_in_round(candidate.id, current)
        if op is None or op.status != "failed":
            continue
        code = op.result_error.get("code") if isinstance(op.result_error, dict) else None
        if code == "review_withdrawn":
            return code
        failed = len([o for o in state.in_round(candidate.id, current) if o.status == "failed"])
        if failed < candidate.max_attempts:
            continue
        if candidate.kind == "cold" and candidate.on_fail == step.id and op.upstream:
            continue
        return code or "step_failed"
    return None


def _plan_shape_ok(plan: dict[str, Any]) -> bool:
    supersede = plan.get("supersede")
    return (
        isinstance(plan.get("step"), str)
        and _plain_int(plan.get("round"))
        and _plain_int(plan.get("attempt"))
        and isinstance(supersede, list)
        and all(isinstance(op_id, str) for op_id in supersede)
        and isinstance(plan.get("reopened_for"), dict)
    )


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def reopen_unfinished(state: RunState, marker_op: OpAttempt) -> bool:
    """Whether the reopen recorded on ``marker_op`` still has a write to make in this round: an op
    it lists not yet superseded, or its new attempt not yet created."""
    plan = marker_op.reopen
    if plan is None or not _plan_shape_ok(plan) or plan["round"] != state.current_round():
        return False
    for op_id in plan["supersede"]:
        op = state.op_by_id(op_id)
        if op is not None and not op.is_cancelled:
            return True
    made = state.all_in_round(plan["step"], plan["round"])
    return not any(op.attempt == plan["attempt"] for op in made)


def unfinished_reopens(state: RunState) -> list[OpAttempt]:
    """The marker ops of every reopen of the current round whose writes are not all made yet
    (the window between a reopen's first write and the next engine pass or recipe command)."""
    return [op for op in state.ops if op.reopen is not None and reopen_unfinished(state, op)]


def reopening(state: RunState) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """For the status view and the ready pass, while reopens are unfinished: ``(steps, ops)``,
    the step ids whose new attempt they will still create and the op ids they will still
    supersede, each mapped to its reopen's plan."""
    steps: dict[str, dict[str, Any]] = {}
    ops: dict[str, dict[str, Any]] = {}
    for marker in unfinished_reopens(state):
        plan = marker.reopen or {}
        steps[plan["step"]] = plan
        for op_id in plan["supersede"]:
            ops[op_id] = plan
    return steps, ops


def in_progress_action(plan: dict[str, Any]) -> dict[str, Any]:
    """The status view's action while a reopen is unfinished: nothing for anyone to do; the next
    engine tick, or any recipe command, finishes it. Never a retry (it would race the reopen)."""
    why = plan.get("reopened_for") or {}
    return {
        "who": "worker",
        "what": (
            f"a reopen of {plan.get('step')!r} after {why.get('step')!r} failed is being finished: "
            "the next engine tick (or any recipe command) completes it; do not retry"
        ),
    }


def reopen_offer(state: RunState, failed_step: StepDef) -> dict[str, Any] | None:
    """For the status view: what to do about ``failed_step``'s upstream failure, or ``None``
    when it is not one (no ``on_fail``, or the failure is not marked upstream)."""
    if failed_step.on_fail is None or state.recipe_def is None:
        return None
    current = state.current_round()
    op = state.latest_in_round(failed_step.id, current)
    if op is None or op.status != "failed" or not op.upstream:
        return None
    target = state.recipe_def.step(failed_step.on_fail)
    code = op.result_error.get("code") if isinstance(op.result_error, dict) else None
    refusal = reopen_refusal(state, state.recipe_def, target)
    if refusal is None:
        return {
            "who": "agent",
            "what": (
                f"recipe_reopen {target.id}: step {failed_step.id!r} failed ({code}) because of "
                f"the {target.id!r} step's output; reopen {target.id!r} and complete it with a "
                "changed result (an unchanged plan is refused)"
            ),
            "tool": "recipe_reopen",
            "args": {"recipe_node": state.recipe_node, "step_id": target.id},
            "fill": ["agent_id", "idempotency_key"],
        }
    if refusal.code == "no_attempts_left":
        used = len(state.all_in_round(target.id, current))
        return {
            "who": "human",
            "what": (
                f"step {failed_step.id!r} failed ({code}) because of the {target.id!r} step's "
                f"output, and {target.id!r} has no attempts left ({used} of "
                f"{target.max_attempts} in round {current}); what remains is recipe_cancel and "
                "a new run"
            ),
        }
    return None


__all__ = [
    "in_progress_action",
    "reopen_blocker",
    "reopen_offer",
    "reopen_refusal",
    "reopen_unfinished",
    "reopening",
    "unfinished_reopens",
]
