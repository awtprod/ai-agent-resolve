"""The scheduler: ``advance`` drives a recipe run one pass, as a pure function over the log.

``advance`` folds the run, decides what should change, and makes each change with S13g1's
atomicity pattern (one ``BEGIN IMMEDIATE`` transaction per write, the ``hub.write`` record last).
It never runs a cold step itself -- it only creates the op node a worker then claims. It submits
and polls hot steps through the injected :class:`~aar.recipes.ports.HotDispatcher`, opens and reads
gate reviews, advances review rounds, and marks the recipe done or blocked.

``advance`` is safe to call concurrently. Two callers produce exactly one op node per attempt,
because the op-creating event uses a global (not per-actor) idempotency key, so the database's
``UNIQUE(idempotency_key)`` refuses the second one; the loser treats the op as already created.

Every recipe write tool and the worker loop call ``advance``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from aar.hub.common import Clock, iso, parse_ts, utcnow
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph, allowed_transitions
from aar.hub.writes import (
    _append_record,
    _in_transaction,
    request_sha256,
)
from aar.recipes import registry
from aar.recipes.events import (
    EVENT_CANCELLED,
    EVENT_CLAIMED,
    EVENT_HEARTBEAT,
    EVENT_HOT_INTENT,
    EVENT_HOT_SUBMITTED,
    EVENT_RESULT,
    EVENT_REVIEW_LINKED,
    OP_ADVANCE,
    OP_CLAIM,
    OP_HEARTBEAT,
    OP_HOT_INTENT,
    OP_HOT_SUBMITTED,
    OP_RESULT,
    OP_REVIEW_OPEN,
    OUTCOME_DONE,
    OUTCOME_FAILED,
    STEP_SCHEMA,
)
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.state import OpAttempt, RunState, load_state, project_of_node

DEFAULT_LEASE_S = 60
DEFAULT_HEARTBEAT_S = 15
_MAX_REASON = 500


def _runner_actor(graph: WorkGraph, recipe_node: str, fallback: Actor) -> Actor:
    """The stable actor the engine attributes its own writes to: the recipe's creator."""
    creator = graph.creator(recipe_node)
    return creator if creator is not None else fallback


# -- op creation --------------------------------------------------------------


def _op_base_key(recipe_node: str, step_id: str, round_: int, attempt: int) -> str:
    """A GLOBAL (not per-actor) idempotency key, so one attempt is one op across all callers."""
    return f"aar.recipe.op/{recipe_node}/{step_id}/r{round_}/a{attempt}"


def create_attempt(
    log: EventLog,
    *,
    project_id: str,
    recipe_node: str,
    step: StepDef,
    round_: int,
    attempt: int,
    actor: Actor,
    status: str,
) -> str | None:
    """Create the op node for one (step, round, attempt); return its id, or ``None`` if it exists.

    The op links to the recipe node with ``derives_from``. The global ``#node`` key makes this the
    single op for the attempt even under two concurrent callers.
    """
    base = _op_base_key(recipe_node, step.id, round_, attempt)
    if log.get_by_idempotency_key(base + "#node") is not None:
        return None
    data = {
        "schema": STEP_SCHEMA,
        "recipe_node": recipe_node,
        "step_id": step.id,
        "attempt": attempt,
        "round": round_,
        "step_kind": step.kind,
    }
    title = f"{step.kind} {step.id} r{round_} a{attempt}"
    request = {
        "recipe_node": recipe_node,
        "step_id": step.id,
        "round": round_,
        "attempt": attempt,
        "kind": step.kind,
    }
    try:
        with _in_transaction(log) as txn:
            graph = WorkGraph(log, project_id)
            node_event = graph.add_node_in(
                txn,
                "op",
                actor,
                title=title,
                status=status,
                data=data,
                idempotency_key=base + "#node",
            )
            op_id = node_event.payload["node_id"]
            edge_event = graph.add_edge_in(
                txn, "derives_from", op_id, recipe_node, actor, idempotency_key=base + "#edge"
            )
            _append_record(
                txn,
                base,
                OP_ADVANCE,
                project_id,
                actor,
                request_sha256(OP_ADVANCE, {"project_id": project_id, **request}),
                node_id=op_id,
                edge_id=edge_event.payload["edge_id"],
            )
    except Exception as exc:  # lost the create race, or rolled back: another caller has it
        if _is_conflict(exc):
            return None
        raise
    return op_id


def _is_conflict(exc: Exception) -> bool:
    from aar.hub.writes import WriteError

    return isinstance(exc, WriteError) and exc.code == "idempotency_conflict"


# -- recording a step result --------------------------------------------------


def _artifact_node(
    txn: Any, graph: WorkGraph, op_id: str, artifact_id: str, actor: Actor, base: str
) -> None:
    """Add an ``artifact`` node for ``artifact_id`` and a ``produces`` edge from the op."""
    node_event = graph.add_node_in(
        txn,
        "artifact",
        actor,
        title=f"artifact {artifact_id[:12]}",
        status="done",
        data={"artifact_id": artifact_id},
        idempotency_key=f"{base}#art:{artifact_id}#node",
    )
    graph.add_edge_in(
        txn,
        "produces",
        op_id,
        node_event.payload["node_id"],
        actor,
        idempotency_key=f"{base}#art:{artifact_id}#edge",
    )


def record_result(
    log: EventLog,
    *,
    project_id: str,
    op_id: str,
    actor: Actor,
    outcome: str,
    outputs: dict[str, Any] | None = None,
    artifact_ids: tuple[str, ...] = (),
    error: dict[str, Any] | None = None,
) -> bool:
    """Record a step attempt's result: move the op to done/failed and link produced artifacts.

    Idempotent per op (the ``#result`` key); returns ``True`` if it wrote, ``False`` on replay.
    """
    base = f"aar.recipe.result/{op_id}"
    if log.get_by_idempotency_key(base + "#result") is not None:
        return False
    new_status = "done" if outcome == OUTCOME_DONE else "failed"
    payload = {
        "op_node": op_id,
        "outcome": outcome,
        "outputs": dict(outputs or {}),
        "artifact_ids": list(artifact_ids),
        "error": error,
    }
    try:
        with _in_transaction(log) as txn:
            graph = WorkGraph(log, project_id)
            node = graph.node(op_id)
            if node is None or node.status in ("done", "failed", "cancelled"):
                return False
            if node.status == "pending":
                graph.set_status_in(txn, op_id, "active", actor)
            graph.set_status_in(txn, op_id, new_status, actor)
            txn.append(
                EVENT_RESULT,
                actor,
                payload,
                project_id=project_id,
                idempotency_key=base + "#result",
            )
            if outcome == OUTCOME_DONE:
                for aid in artifact_ids:
                    _artifact_node(txn, graph, op_id, aid, actor, base)
            _append_record(
                txn,
                base,
                OP_RESULT,
                project_id,
                actor,
                request_sha256(
                    OP_RESULT, {"project_id": project_id, "op_node": op_id, "outcome": outcome}
                ),
                op_node=op_id,
                outcome=outcome,
            )
    except Exception as exc:
        if _is_conflict(exc):
            return False
        raise
    return True


# -- claim / heartbeat (used by the worker) -----------------------------------


def claim(
    log: EventLog,
    *,
    project_id: str,
    op: OpAttempt,
    worker: Actor,
    idempotent: bool,
    clock: Clock,
    lease_s: int = DEFAULT_LEASE_S,
) -> bool:
    """Claim a cold op with a lease, if it is claimable as of the transaction.

    A ``pending`` op is claimed (moved to ``active``). An ``active`` op is stolen only when its
    lease has lapsed and the step is idempotent; a live lease is never stolen, and a lapsed
    non-idempotent op is not re-claimed here (``advance`` fails it ``lease_lost``). Returns whether
    the claim succeeded.
    """
    now = clock()
    lease_until = iso(now + timedelta(seconds=lease_s))
    base = f"aar.recipe.claim/{op.op_id}"
    try:
        with _in_transaction(log) as txn:
            graph = WorkGraph(log, project_id)
            node = graph.node(op.op_id)
            if node is None:
                return False
            # Count prior claims to key this one (and detect a steal).
            prior = _claim_events(txn, project_id, op.op_id)
            if node.status == "pending":
                graph.set_status_in(txn, op.op_id, "active", worker)
            elif node.status == "active":
                lease_events = _lease_events(txn, project_id, op.op_id)
                if _lease_live(lease_events, now) or not idempotent:
                    return False  # never steal a live lease; non-idempotent is not re-claimed
            else:
                return False  # done/failed/cancelled
            n = len(prior)
            key = f"{base}/{n}"
            txn.append(
                EVENT_CLAIMED,
                worker,
                {
                    "op_node": op.op_id,
                    "step_id": op.step_id,
                    "attempt": op.attempt,
                    "round": op.round,
                    "lease_until": lease_until,
                },
                project_id=project_id,
                idempotency_key=key + "#claimed",
            )
            _append_record(
                txn,
                key,
                OP_CLAIM,
                project_id,
                worker,
                request_sha256(OP_CLAIM, {"project_id": project_id, "op_node": op.op_id, "n": n}),
                op_node=op.op_id,
            )
    except Exception as exc:
        if _is_conflict(exc):
            return False
        raise
    return True


def heartbeat(
    log: EventLog,
    *,
    project_id: str,
    op_id: str,
    worker: Actor,
    clock: Clock,
    lease_s: int = DEFAULT_LEASE_S,
    n: int,
) -> None:
    """Extend the lease on a claimed op. ``n`` disambiguates successive heartbeats."""
    lease_until = iso(clock() + timedelta(seconds=lease_s))
    base = f"aar.recipe.heartbeat/{op_id}/{n}"
    if log.get_by_idempotency_key(base + "#hb") is not None:
        return
    with _in_transaction(log) as txn:
        txn.append(
            EVENT_HEARTBEAT,
            worker,
            {"op_node": op_id, "lease_until": lease_until},
            project_id=project_id,
            idempotency_key=base + "#hb",
        )
        _append_record(
            txn,
            base,
            OP_HEARTBEAT,
            project_id,
            worker,
            request_sha256(OP_HEARTBEAT, {"project_id": project_id, "op_node": op_id, "n": n}),
            op_node=op_id,
        )


def _claim_events(txn: Any, project_id: str, op_id: str) -> list[Any]:
    events = txn.read(0, project_id=project_id, types=[EVENT_CLAIMED])
    return [e for e in events if isinstance(e.payload, dict) and e.payload.get("op_node") == op_id]


def _lease_events(txn: Any, project_id: str, op_id: str) -> list[Any]:
    events = txn.read(0, project_id=project_id, types=[EVENT_CLAIMED, EVENT_HEARTBEAT])
    return [e for e in events if isinstance(e.payload, dict) and e.payload.get("op_node") == op_id]


def _lease_live(claim_events: list[Any], now: Any) -> bool:
    """Whether any claim/heartbeat gives a lease that has not yet lapsed."""
    latest = None
    for event in claim_events:
        value = event.payload.get("lease_until")
        if isinstance(value, str):
            try:
                ts = parse_ts(value)
            except (ValueError, OverflowError):
                continue
            if latest is None or ts > latest:
                latest = ts
    return latest is not None and now <= latest


def op_lease_live(state: RunState, op: OpAttempt, now: Any) -> bool:
    return op.lease_until is not None and now <= op.lease_until


# -- advance ------------------------------------------------------------------


def advance(
    log: EventLog,
    recipe_node: str,
    *,
    actor: Actor,
    dispatcher: Any = None,
    clock: Clock = utcnow,
) -> RunState:
    """Drive the recipe run of ``recipe_node`` one pass; return the resulting :class:`RunState`."""
    registry.ensure_builtins()
    project_id = project_of_node(log, recipe_node)
    state = load_state(log, recipe_node, project_id=project_id)
    if project_id is None or state.recipe_data is None or state.recipe_def is None:
        return state
    if state.recipe_status not in ("active", "blocked"):
        return state
    graph = WorkGraph(log, project_id)
    runner = _runner_actor(graph, recipe_node, actor)
    recipe_def = state.recipe_def
    now = clock()

    if _reap_leases(log, state, recipe_def, runner, now):
        state = load_state(log, recipe_node, project_id=project_id)

    if _process_gates(log, state, recipe_def, runner):
        state = load_state(log, recipe_node, project_id=project_id)

    if _create_ready(log, state, recipe_def, runner):
        state = load_state(log, recipe_node, project_id=project_id)

    if _drive_hot(log, state, recipe_def, runner, dispatcher):
        state = load_state(log, recipe_node, project_id=project_id)

    if _open_gate_reviews(log, state, recipe_def, runner):
        state = load_state(log, recipe_node, project_id=project_id)

    if _finish(log, state, recipe_def, runner):
        state = load_state(log, recipe_node, project_id=project_id)
    return state


def _reap_leases(
    log: EventLog, state: RunState, recipe_def: RecipeDef, runner: Actor, now: Any
) -> bool:
    changed = False
    for op in state.ops:
        if op.kind != "cold" or op.status != "active" or op.is_cancelled:
            continue
        try:
            step = recipe_def.step(op.step_id)
        except KeyError:
            continue
        if op_lease_live(state, op, now):
            continue
        if step.idempotent:
            continue  # lapsed but re-claimable; leave active for another worker
        if record_result(
            log,
            project_id=state.project_id,
            op_id=op.op_id,
            actor=runner,
            outcome=OUTCOME_FAILED,
            error={
                "code": "lease_lost",
                "message": "the worker's lease lapsed mid-step",
                "fix_hint": "Retry the step (it is not idempotent, so it was not re-claimed).",
            },
        ):
            changed = True
    return changed


def _process_gates(log: EventLog, state: RunState, recipe_def: RecipeDef, runner: Actor) -> bool:
    changed = False
    current = state.current_round()
    exhausted = state.rounds_exhausted()
    for op in state.ops:
        if op.kind != "gate" or op.review_id is None or op.is_cancelled:
            continue
        review = state.reviews.get(op.review_id)
        if review is None or review.decision is None:
            continue
        if review.decision.decision == "approve":
            if op.status == "active" and record_result(
                log,
                project_id=state.project_id,
                op_id=op.op_id,
                actor=runner,
                outcome=OUTCOME_DONE,
                outputs={"decision": "approve"},
            ):
                changed = True
        elif exhausted and op.round >= state.max_review_rounds:
            if op.status in ("pending", "active") and record_result(
                log,
                project_id=state.project_id,
                op_id=op.op_id,
                actor=runner,
                outcome=OUTCOME_FAILED,
                error={
                    "code": "review_rounds_exhausted",
                    "message": f"the review was rejected in the final round {op.round}",
                    "fix_hint": "A human must intervene; the recipe is blocked.",
                },
            ):
                changed = True
        elif op.round < current:
            # Rejected below the cap: supersede from on_reject onward in the rejected round.
            step = recipe_def.step(op.step_id)
            if _rollback_round(log, state, recipe_def, step.on_reject, op.round, runner):
                changed = True
    if exhausted:
        changed |= _set_recipe_status(log, state, "blocked", runner)
    return changed


def _rollback_round(
    log: EventLog,
    state: RunState,
    recipe_def: RecipeDef,
    on_reject: str | None,
    round_: int,
    runner: Actor,
) -> bool:
    if on_reject is None:
        return False
    try:
        start = recipe_def.index(on_reject)
    except KeyError:
        return False
    changed = False
    for step in recipe_def.steps[start:]:
        for op in state.in_round(step.id, round_):
            if _supersede_op(log, state.project_id, op, runner, "superseded by review rejection"):
                changed = True
    return changed


def _supersede_op(
    log: EventLog, project_id: str, op: OpAttempt, runner: Actor, reason: str
) -> bool:
    base = f"aar.recipe.cancel/{op.op_id}"
    if log.get_by_idempotency_key(base + "#cancelled") is not None:
        return False
    try:
        with _in_transaction(log) as txn:
            graph = WorkGraph(log, project_id)
            node = graph.node(op.op_id)
            if node is None:
                return False
            # A done attempt stays 'done' in the graph (terminal) but is marked superseded;
            # a live attempt is also moved to cancelled.
            if node.status in ("pending", "active"):
                graph.set_status_in(txn, op.op_id, "cancelled", runner)
            txn.append(
                EVENT_CANCELLED,
                runner,
                {"op_node": op.op_id, "reason": reason[:_MAX_REASON]},
                project_id=project_id,
                idempotency_key=base + "#cancelled",
            )
            _append_record(
                txn,
                base,
                OP_ADVANCE,
                project_id,
                runner,
                request_sha256(OP_ADVANCE, {"project_id": project_id, "cancel": op.op_id}),
                op_node=op.op_id,
            )
    except Exception as exc:
        if _is_conflict(exc):
            return False
        raise
    return True


def _create_ready(log: EventLog, state: RunState, recipe_def: RecipeDef, runner: Actor) -> bool:
    current = state.current_round()
    changed = False
    for step in recipe_def.steps:
        if state.is_step_done(step.id):
            continue  # already done (carried forward from an earlier round); never re-run it
        if state.in_round(step.id, current):
            continue  # already has a (non-cancelled) op this round
        if not all(state.is_step_done(need) for need in step.needs):
            continue
        status = "pending" if step.kind in ("cold", "hot") else "active"
        op_id = create_attempt(
            log,
            project_id=state.project_id,
            recipe_node=state.recipe_node,
            step=step,
            round_=current,
            attempt=1,
            actor=runner,
            status=status,
        )
        if op_id is not None:
            changed = True
    return changed


def _drive_hot(
    log: EventLog, state: RunState, recipe_def: RecipeDef, runner: Actor, dispatcher: Any
) -> bool:
    current = state.current_round()
    changed = False
    for step in recipe_def.steps:
        if step.kind != "hot":
            continue
        op = state.latest_in_round(step.id, current)
        if op is None or op.status in ("done", "failed") or op.is_cancelled:
            continue
        if not op.hot_submitted:
            if dispatcher is None:
                continue  # PC off: the op stays pending; status shows waiting:no_node_dispatcher
            _record_hot_intent(log, state.project_id, op, runner)
            job_ref = dispatcher.submit(
                request_id=op.op_id,
                project_id=state.project_id,
                kind=step.hot_kind or step.id,
                params=dict(step.hot_params or {}),
                agent_id=runner.id,
            )
            _record_hot_submitted(log, state.project_id, op, runner, job_ref)
            changed = True
            view = dispatcher.poll(job_ref)
        elif dispatcher is not None:
            view = dispatcher.poll(op.hot_job_ref)
        else:
            continue
        if view.is_terminal:
            if view.state == "done":
                result = view.result
                if record_result(
                    log,
                    project_id=state.project_id,
                    op_id=op.op_id,
                    actor=runner,
                    outcome=OUTCOME_DONE,
                    outputs=dict(result.outputs) if result else {},
                    artifact_ids=result.artifact_ids if result else (),
                ):
                    changed = True
            elif record_result(
                log,
                project_id=state.project_id,
                op_id=op.op_id,
                actor=runner,
                outcome=OUTCOME_FAILED,
                error=view.error or {"code": "hot_failed", "message": ""},
            ):
                changed = True
    return changed


def _record_hot_intent(log: EventLog, project_id: str, op: OpAttempt, runner: Actor) -> None:
    base = f"aar.recipe.hotintent/{op.op_id}"
    if log.get_by_idempotency_key(base + "#intent") is not None:
        return
    with _in_transaction(log) as txn:
        txn.append(
            EVENT_HOT_INTENT,
            runner,
            {"op_node": op.op_id, "request_id": op.op_id},
            project_id=project_id,
            idempotency_key=base + "#intent",
        )
        _append_record(
            txn,
            base,
            OP_HOT_INTENT,
            project_id,
            runner,
            request_sha256(OP_HOT_INTENT, {"project_id": project_id, "op_node": op.op_id}),
            op_node=op.op_id,
        )


def _record_hot_submitted(
    log: EventLog, project_id: str, op: OpAttempt, runner: Actor, job_ref: str
) -> None:
    base = f"aar.recipe.hotsubmitted/{op.op_id}"
    if log.get_by_idempotency_key(base + "#submitted") is not None:
        return
    with _in_transaction(log) as txn:
        graph = WorkGraph(log, project_id)
        node = graph.node(op.op_id)
        if node is not None and node.status == "pending":
            graph.set_status_in(txn, op.op_id, "active", runner)
        txn.append(
            EVENT_HOT_SUBMITTED,
            runner,
            {"op_node": op.op_id, "request_id": op.op_id, "job_ref": job_ref},
            project_id=project_id,
            idempotency_key=base + "#submitted",
        )
        _append_record(
            txn,
            base,
            OP_HOT_SUBMITTED,
            project_id,
            runner,
            request_sha256(OP_HOT_SUBMITTED, {"project_id": project_id, "op_node": op.op_id}),
            op_node=op.op_id,
        )


def _open_gate_reviews(
    log: EventLog, state: RunState, recipe_def: RecipeDef, runner: Actor
) -> bool:
    current = state.current_round()
    changed = False
    for step in recipe_def.steps:
        if step.kind != "gate":
            continue
        op = state.latest_in_round(step.id, current)
        if op is None or op.status != "active" or op.review_id is not None or op.is_cancelled:
            continue
        subject_outputs = state.done_outputs(step.subject) if step.subject else {}
        subject_node, renders = _review_subject(subject_outputs)
        if subject_node is None or not renders:
            continue  # subject not ready yet
        base = f"aar.recipe.review/{op.op_id}"
        if log.get_by_idempotency_key(base + "#link") is not None:
            continue
        review_id = _open_review_node(
            log,
            state.project_id,
            subject_node,
            renders,
            runner,
            title=f"gate {step.id} (round {op.round})",
            base=base,
        )
        with _in_transaction(log) as txn:
            txn.append(
                EVENT_REVIEW_LINKED,
                runner,
                {
                    "op_node": op.op_id,
                    "review_id": review_id,
                    "round": op.round,
                    "subject": subject_node,
                },
                project_id=state.project_id,
                idempotency_key=base + "#link",
            )
            _append_record(
                txn,
                base,
                OP_REVIEW_OPEN,
                state.project_id,
                runner,
                request_sha256(
                    OP_REVIEW_OPEN, {"project_id": state.project_id, "op_node": op.op_id}
                ),
                op_node=op.op_id,
                review_id=review_id,
            )
        changed = True
    return changed


def _open_review_node(
    log: EventLog,
    project_id: str,
    subject: str,
    renders: tuple[str, ...],
    runner: Actor,
    *,
    title: str,
    base: str,
) -> str:
    """Create a ``review`` node (status active) plus its ``reviews`` edge, like reviews.request.

    Done without the artifact store (``advance`` has no store): the renders are the subject step's
    already-stored outputs, so they exist. A human still decides it in the console; nothing here
    can decide a review.
    """
    data = {
        "subject": subject,
        "renders": list(renders),
        "timeline": None,
        "fps": None,
        "requested_by": f"{runner.kind}:{runner.id}",
    }
    graph = WorkGraph(log, project_id)
    node = graph.add_node(
        "review",
        runner,
        title=title,
        status="active",
        data=data,
        idempotency_key=f"{base}:node",
    )
    graph.add_edge("reviews", node.id, subject, runner, idempotency_key=f"{base}:edge")
    return node.id


def _review_subject(outputs: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    """The subject node and render ids for a gate, read from the reviewed step's outputs."""
    plan_node = outputs.get("plan_node")
    plan_sha = outputs.get("plan_sha256")
    if isinstance(plan_node, str) and isinstance(plan_sha, str):
        return plan_node, (plan_sha,)
    subject = outputs.get("subject_node")
    renders = outputs.get("artifact_ids")
    if isinstance(subject, str) and isinstance(renders, list):
        return subject, tuple(r for r in renders if isinstance(r, str))
    return None, ()


def _finish(log: EventLog, state: RunState, recipe_def: RecipeDef, runner: Actor) -> bool:
    if all(state.is_step_done(step.id) for step in recipe_def.steps):
        return _set_recipe_status(log, state, "done", runner)
    current = state.current_round()
    for step in recipe_def.steps:
        failed = [op for op in state.in_round(step.id, current) if op.status == "failed"]
        if failed and len(failed) >= step.max_attempts:
            return _set_recipe_status(log, state, "blocked", runner)
    return False


def _set_recipe_status(log: EventLog, state: RunState, status: str, runner: Actor) -> bool:
    if state.recipe_status == status:
        return False
    if status not in allowed_transitions(state.recipe_status):
        return False
    base = f"aar.recipe.status/{state.recipe_node}/{status}"
    if log.get_by_idempotency_key(base + "#status") is not None:
        return False
    try:
        with _in_transaction(log) as txn:
            graph = WorkGraph(log, state.project_id)
            node = graph.node(state.recipe_node)
            if node is None or status not in allowed_transitions(node.status):
                return False
            graph.set_status_in(txn, state.recipe_node, status, runner)
            _append_record(
                txn,
                base,
                OP_ADVANCE,
                state.project_id,
                runner,
                request_sha256(
                    OP_ADVANCE,
                    {"project_id": state.project_id, "recipe": state.recipe_node, "to": status},
                ),
                recipe_node=state.recipe_node,
                to=status,
            )
    except Exception as exc:
        if _is_conflict(exc):
            return False
        raise
    return True
