"""The agent-native status view: what each step is waiting on, and who acts next.

:func:`status_view` folds the run from evidence and returns a bounded JSON object: a per-step
state, attempt, round, op node id, an outputs summary and the error, plus a ``next_actions`` list
that says who acts (agent, human, worker or node), what to do, and -- for agents -- the exact tool,
its concrete arguments and a ``fill`` list naming the fields the caller supplies (``agent_id``,
``idempotency_key``, ``outputs``). It is a pure read over the log; with a
:class:`~aar.recipes.ports.HotDispatcher` it also surfaces the node's ``waiting_for_node`` /
``node_offline`` states (poll is a read), and ``waiting:dispatcher_error`` if the port raises.

The view is bounded (F6): each step's outputs summary and error message are capped, and the whole
object is capped, with ``truncated: true`` when anything was cut.
"""

from __future__ import annotations

import json
from typing import Any

from aar.hub.common import Clock, utcnow
from aar.hub.events import EventLog
from aar.recipes.models import StepDef
from aar.recipes.ports import HOT_STATES
from aar.recipes.reopen import in_progress_action, reopen_offer, reopening
from aar.recipes.state import OpAttempt, RunState, load_state

MAX_OUTPUTS_KEYS = 32
MAX_STEP_OUTPUTS_BYTES = 8 * 1024
MAX_STEP_ERROR_BYTES = 4 * 1024
MAX_VIEW_BYTES = 64 * 1024
MAX_ACTION_WHAT_BYTES = 256
#: The state of a step while an unfinished reopen will still supersede its op or create its
#: attempt (S19f): the worker acts next (``in_progress_action``), not a human or an agent.
REOPENING_STATE = "waiting:reopen"


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
    truncated = False
    # S19f: while a reopen is unfinished, the ops it will supersede and the step it will create
    # get its in-progress action instead of a retry or a run (MED-1 of the validation).
    held_steps, held_ops = reopening(state)
    for step in step_defs:
        op = state.latest_in_round(step.id, current)
        # F17: a step done in an earlier round carries forward as done here.
        if op is None and state.is_step_done(step.id):
            op = state.done_op(step.id)
        step_state, action = _step_state(state, step, op, current, dispatcher)
        held = held_ops.get(op.op_id) if op is not None else held_steps.get(step.id)
        if held is not None and step_state != "done":
            # the state agrees with the action: the worker finishes the reopen, nobody else acts
            step_state, action = REOPENING_STATE, in_progress_action(held)
        outputs, out_trunc = _summary(op.result_outputs) if op is not None else (None, False)
        error, err_trunc = _error(op.result_error) if op is not None else (None, False)
        truncated = truncated or out_trunc or err_trunc
        entry: dict[str, Any] = {
            "step_id": step.id,
            "kind": step.kind,
            "state": step_state,
            "round": current,
            "attempt": op.attempt if op is not None else None,
            "op_node": op.op_id if op is not None else None,
            "outputs": outputs,
            "error": error,
        }
        if op is not None and op.reopened_for is not None:
            # S19f: why this agent attempt exists (the failure its last output caused)
            reopened, re_trunc = _reopened(op.reopened_for)
            entry["reopened_for"] = reopened
            truncated = truncated or re_trunc
        steps_out.append(entry)
        if action is not None:
            actions.append({"step_id": step.id, **action})
    if state.rounds_exhausted():
        actions.append(
            {
                "step_id": None,
                "who": "human",
                "what": "review rounds exhausted; cancel the recipe in the console",
            }
        )
    view = {
        "found": True,
        "recipe_node": recipe_node,
        "project_id": state.project_id,
        "recipe": state.recipe_data.recipe,
        "registered": registered,
        "status": state.recipe_status,
        "round": current,
        "max_review_rounds": state.max_review_rounds,
        "steps": steps_out,
        "next_actions": actions,
        "ignored": state.ignored,
        "diagnostics": _diagnostics(log, state.project_id, recipe_node),
        "truncated": truncated,
    }
    return _enforce_total_bound(view)


MAX_WORKER_ERRORS_LISTED = 5
MAX_WORKER_ERROR_ENTRY_BYTES = 1024
_MAX_EXC_TYPE_BYTES = 128


def _diagnostics(log: EventLog, project_id: str, recipe_node: str) -> dict[str, Any]:
    entries, total = _worker_errors(log, project_id, recipe_node)
    return {"worker_errors": entries, "worker_error_count": total}


def _worker_errors(
    log: EventLog, project_id: str, recipe_node: str
) -> tuple[list[dict[str, Any]], int]:
    """The runner's ``recipe.worker_error`` records for this recipe (R3-3/R4-1): the first
    :data:`MAX_WORKER_ERRORS_LISTED` as entries, each field bounded on STORED bytes (the same
    escaped-JSON measure as the whole view), plus the total count of records."""
    from aar.recipes.engine import _bound_text
    from aar.recipes.events import OP_WORKER_ERROR, RUNNER

    entries: list[dict[str, Any]] = []
    total = 0
    for ev in log.read(0, project_id=project_id, types=["hub.write"], limit=-1):
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        if payload.get("op") != OP_WORKER_ERROR or payload.get("recipe_node") != recipe_node:
            continue
        if ev.actor != RUNNER:  # only the runner writes these; anything else is not evidence
            continue
        total += 1
        if len(entries) < MAX_WORKER_ERRORS_LISTED:
            exc_type, error = payload.get("exc_type"), payload.get("error")
            first_seen = payload.get("first_seen")
            entries.append(
                {
                    # R5-3: when it first failed; later occurrences are only in the WARNING log.
                    "first_seen": _bound_text(first_seen, _MAX_EXC_TYPE_BYTES)
                    if isinstance(first_seen, str)
                    else None,
                    "exc_type": _bound_text(exc_type, _MAX_EXC_TYPE_BYTES)
                    if isinstance(exc_type, str)
                    else None,
                    "error": _bound_text(error, MAX_WORKER_ERROR_ENTRY_BYTES)
                    if isinstance(error, str)
                    else None,
                }
            )
    return entries, total


def _enforce_total_bound(view: dict[str, Any]) -> dict[str, Any]:
    """Keep the whole view under :data:`MAX_VIEW_BYTES`: drop the step bodies first, then the
    worker-error entries (R4-1: ``diagnostics`` counts toward the bound; the count is kept)."""
    if len(_canon(view)) <= MAX_VIEW_BYTES:
        return view
    for entry in view["steps"]:
        entry["outputs"] = None
        entry["error"] = None
    view["truncated"] = True
    if len(_canon(view)) > MAX_VIEW_BYTES:
        view["diagnostics"]["worker_errors"] = []
    if len(_canon(view)) > MAX_VIEW_BYTES:
        from aar.recipes.engine import _bound_text

        # Last resort (R5-2): a registered instruction is capped per step, but 64 of them can
        # still exceed the whole bound; cut each action's text, keeping the tool and its args.
        for action in view["next_actions"]:
            what = action.get("what")
            if isinstance(what, str) and len(what.encode("utf-8")) > MAX_ACTION_WHAT_BYTES:
                action["what"] = _bound_text(what, MAX_ACTION_WHAT_BYTES)
    return view


def _canon(value: Any) -> bytes:
    # The STORED (escaped) form the log writes, so bounds are measured on stored bytes (R3-1).
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError):
        return b"\x00" * (MAX_VIEW_BYTES + 1)


def _summary(outputs: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    if not outputs:
        return None, False
    keys = sorted(outputs)[:MAX_OUTPUTS_KEYS]
    summary = {key: outputs[key] for key in keys}
    size = len(_canon(summary))
    if size <= MAX_STEP_OUTPUTS_BYTES:
        return summary, False
    # F6: the marker itself must stay small -- never echo the (possibly huge) keys, just counts.
    return {"truncated": True, "key_count": len(outputs), "bytes": size}, True


def _reopened(reopened_for: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    keys = ("step", "attempt", "code", "message", "fix_hint", "op_node")
    out = {k: reopened_for.get(k) for k in keys}
    cut = False
    for k in ("message", "fix_hint"):
        text = out.get(k)
        if isinstance(text, str) and len(text.encode("utf-8")) > MAX_STEP_ERROR_BYTES:
            out[k] = text.encode("utf-8")[:MAX_STEP_ERROR_BYTES].decode("utf-8", "ignore")
            cut = True
    return out, cut


def _error(error: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    if not error:
        return None, False
    out = {k: error.get(k) for k in ("code", "message", "fix_hint") if k in error}
    if error.get("upstream") is True:
        out["upstream"] = True  # S19f: the on_fail step's output caused it
    msg = out.get("message")
    if isinstance(msg, str) and len(msg.encode("utf-8")) > MAX_STEP_ERROR_BYTES:
        out["message"] = msg.encode("utf-8")[:MAX_STEP_ERROR_BYTES].decode("utf-8", "ignore")
        return out, True
    return out, False


def _reopened_note(op: OpAttempt) -> str:
    """`` (attempt N, reopened because ...)`` for a reopened agent attempt (S19f), else empty."""
    why = op.reopened_for
    if why is None:
        return ""
    return (
        f" (attempt {op.attempt}, reopened because step {why.get('step')!r} attempt "
        f"{why.get('attempt')} failed {why.get('code')}: see this step's reopened_for; submit a "
        "changed result)"
    )


def _retry_action(recipe_node: str, step: StepDef) -> dict[str, Any]:
    return {
        "who": "agent",
        "what": f"retry the failed {step.kind} step {step.id!r}",
        "tool": "recipe_retry",
        "args": {"recipe_node": recipe_node, "step_id": step.id},
        "fill": ["agent_id", "idempotency_key"],
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
        code = op.result_error.get("code") if isinstance(op.result_error, dict) else None
        if step.kind == "gate" or code in ("review_rounds_exhausted", "review_withdrawn"):
            if code == "review_withdrawn":
                return "failed", _retry_action(state.recipe_node, step)
            return "failed", {"who": "human", "what": f"gate {step.id!r}: {code}; cancel to exit"}
        offer = reopen_offer(state, step)
        if offer is not None:  # S19f: retrying it unchanged would fail the same way
            return "failed", offer
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
            "what": f"complete agent step {step.id!r}{_reopened_note(op)}: {step.instruction}",
            "tool": "step_complete",
            "args": {
                "recipe_node": state.recipe_node,
                "step_id": step.id,
                "attempt": op.attempt,
            },
            "fill": ["agent_id", "idempotency_key", "outputs"],
        }
    if step.kind == "gate":
        return "needs_human", {
            "who": "human",
            "what": f"decide the review for gate {step.id!r} in the console",
            "review_id": op.review_id,
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
        try:
            view = dispatcher.poll(op.hot_job_ref)
        except Exception:
            # F10: a port that raises must not break the read; surface a bounded wait reason.
            return "waiting:dispatcher_error", {
                "who": "node",
                "what": f"the node dispatcher failed to report on hot step {step.id!r}",
            }
        if view is None:
            # N7: a port that returns nothing is unknown, not a crash.
            return "dispatcher_unknown", {
                "who": "node",
                "what": f"the node dispatcher returned no status for hot step {step.id!r}",
            }
        state_value = getattr(view, "state", None)
        if not isinstance(state_value, str) or state_value not in HOT_STATES:
            # N8: a garbage response gets a typed, bounded reason (not a raise, not silent).
            return "waiting:dispatcher_bad_response", {
                "who": "node",
                "what": f"the node dispatcher returned an unusable status for hot step {step.id!r}",
                "got": repr(state_value)[:200],
            }
        if state_value in ("waiting_for_node", "node_offline"):
            return f"waiting:{state_value}", {
                "who": "node",
                "what": f"the node reports {state_value}",
            }
        if state_value == "running":
            return "running", None
    return "waiting:node", {"who": "node", "what": f"hot step {step.id!r} is on the node"}
