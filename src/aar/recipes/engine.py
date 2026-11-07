"""The scheduler: ``advance`` drives a recipe run one pass, as a pure function over the log.

``advance`` folds the run from runner-written evidence, decides what should change, and makes each
change with S13g1's atomicity pattern (one ``BEGIN IMMEDIATE`` transaction per write, the
``hub.write`` record last). It never runs a cold step itself -- it only creates the op node a
worker then claims. It submits and polls hot steps through the injected
:class:`~aar.recipes.ports.HotDispatcher`, opens and reads gate reviews, advances review rounds,
and reconciles the recipe node's graph status to the evidence status (a mirror only).

**Every structural write is attributed to the constant runner identity**
(:data:`aar.recipes.events.RUNNER`), never to the caller, and the real caller is recorded in the
``hub.write`` record (``caller``) and, where it matters, in the payload (``worker`` on a claim,
``by`` on a result). No tool surface can produce the runner identity, so the fold can trust a
``recipe.*`` event or an ``aar.step/1`` op node only when RUNNER authored it; a graph status change
an agent writes on an op it happens to own is simply not evidence.

``advance`` is safe to call concurrently. Two callers produce exactly one op node per attempt,
because the op-creating event uses a global (not per-actor) idempotency key under the single runner
identity, so the database's ``UNIQUE(idempotency_key)`` refuses the second one.

Every recipe write tool and the worker loop call ``advance``.
"""

from __future__ import annotations

import json
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
    EVENT_HOT_BAD_POLL,
    EVENT_HOT_INTENT,
    EVENT_HOT_SUBMITTED,
    EVENT_RESULT,
    EVENT_REVIEW_LINKED,
    EVENT_RUN_CANCELLED,
    EVENT_STARTED,
    OP_ADVANCE,
    OP_CANCEL,
    OP_CLAIM,
    OP_HEARTBEAT,
    OP_HOT_BAD_POLL,
    OP_HOT_INTENT,
    OP_HOT_SUBMITTED,
    OP_RESULT,
    OP_REVIEW_OPEN,
    OP_START,
    OUTCOME_DONE,
    OUTCOME_FAILED,
    RUNNER,
    STEP_SCHEMA,
)
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.ports import HOT_STATES
from aar.recipes.state import OpAttempt, RunState, load_state, project_of_node

MAX_BAD_POLLS = 5

DEFAULT_LEASE_S = 60
DEFAULT_HEARTBEAT_S = 15
_MAX_REASON = 500
# The brief's and DESIGN's bound: a result's canonical-JSON outputs are at most 64 KiB.
MAX_OUTPUT_BYTES = 64 * 1024


def _caller_ref(actor: Actor) -> str:
    """``kind:id`` for the real caller/worker, recorded alongside a runner write."""
    return f"{actor.kind}:{actor.id}"


MAX_ERROR_BYTES = 4 * 1024
_MAX_OUTPUT_DEPTH = 64


def _canonical(value: Any) -> bytes:
    """Canonical UTF-8 JSON. ``allow_nan=False`` so NaN/±inf raise rather than emit invalid JSON;
    ``sort_keys`` raises ``TypeError`` on non-string keys."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _within_depth(value: Any, limit: int) -> bool:
    """Whether ``value`` nests no deeper than ``limit`` (iterative, so no RecursionError)."""
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return False
        if isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend((v, depth + 1) for v in node)
    return True


def _outputs_problem(outputs: dict[str, Any]) -> str | None:
    """``None`` if ``outputs`` is JSON-valid and within bounds, else a refusal code.

    ``outputs_invalid``: NaN/±inf, non-string keys or nesting over 64. ``outputs_too_large``: over
    64 KiB of canonical UTF-8 JSON. This runs on the boundary so no cold/hot result can crash the
    worker or produce log-invalid JSON (N3)."""
    if not _within_depth(outputs, _MAX_OUTPUT_DEPTH):
        return "outputs_invalid"
    try:
        blob = _canonical(outputs)
    except (TypeError, ValueError):
        return "outputs_invalid"
    if len(blob) > MAX_OUTPUT_BYTES:
        return "outputs_too_large"
    return None


_OUTPUT_ERRORS = {
    "outputs_too_large": {
        "code": "outputs_too_large",
        "message": f"the step's outputs exceed {MAX_OUTPUT_BYTES} bytes of canonical JSON",
        "fix_hint": "Store large data as an artifact and return its id, not inline outputs.",
    },
    "outputs_invalid": {
        "code": "outputs_invalid",
        "message": "the step's outputs are not valid JSON (NaN/inf, non-string keys, or too deep)",
        "fix_hint": "Return finite numbers, string keys and nesting under 64 levels.",
    },
}


def _bound_text(value: Any, limit: int) -> Any:
    """Truncate a string to ``limit`` UTF-8 bytes; pass non-strings through."""
    if not isinstance(value, str):
        return value
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    return raw[:limit].decode("utf-8", "ignore")


def _bound_error(error: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only ``{code, message, fix_hint}`` and cap each text field to 4 KiB of UTF-8 (F6)."""
    if not isinstance(error, dict):
        return error
    bounded = {
        "code": _bound_text(error.get("code"), MAX_ERROR_BYTES),
        "message": _bound_text(error.get("message"), MAX_ERROR_BYTES),
        "fix_hint": _bound_text(error.get("fix_hint"), MAX_ERROR_BYTES),
    }
    return {k: v for k, v in bounded.items() if v is not None}


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
    caller: Actor,
    status: str,
) -> str | None:
    """Create the op node for one (step, round, attempt); return its id, or ``None`` if it exists.

    The op is created by RUNNER (so an agent cannot later move its graph status as evidence) and
    links to the recipe node with ``derives_from``. The global ``#node`` key makes this the single
    op for the attempt even under two concurrent callers. ``caller`` is the real caller, recorded.
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
                RUNNER,
                title=title,
                status=status,
                data=data,
                idempotency_key=base + "#node",
            )
            op_id = node_event.payload["node_id"]
            edge_event = graph.add_edge_in(
                txn, "derives_from", op_id, recipe_node, RUNNER, idempotency_key=base + "#edge"
            )
            _append_record(
                txn,
                base,
                OP_ADVANCE,
                project_id,
                RUNNER,
                request_sha256(OP_ADVANCE, {"project_id": project_id, **request}),
                node_id=op_id,
                edge_id=edge_event.payload["edge_id"],
                caller=_caller_ref(caller),
            )
    except Exception as exc:  # lost the create race, or rolled back: another caller has it
        if _is_conflict(exc):
            return None
        raise
    return op_id


def _is_conflict(exc: Exception) -> bool:
    from aar.hub.writes import WriteError

    return isinstance(exc, WriteError) and exc.code == "idempotency_conflict"


# -- run lifecycle evidence ---------------------------------------------------


def record_started(log: EventLog, *, project_id: str, recipe_node: str, caller: Actor) -> bool:
    """Record the ``recipe.started`` evidence (idempotent). The run is 'started' from this alone."""
    base = f"aar.recipe.start/{recipe_node}"
    if log.get_by_idempotency_key(base + "#started") is not None:
        return False
    with _in_transaction(log) as txn:
        txn.append(
            EVENT_STARTED,
            RUNNER,
            {"recipe_node": recipe_node},
            project_id=project_id,
            idempotency_key=base + "#started",
        )
        _append_record(
            txn,
            base,
            OP_START,
            project_id,
            RUNNER,
            request_sha256(OP_START, {"project_id": project_id, "recipe": recipe_node}),
            recipe_node=recipe_node,
            caller=_caller_ref(caller),
        )
    return True


def record_run_cancelled(
    log: EventLog, *, project_id: str, recipe_node: str, caller: Actor, reason: str
) -> bool:
    """Record the ``recipe.run_cancelled`` evidence (idempotent): the run is cancelled from this."""
    base = f"aar.recipe.runcancel/{recipe_node}"
    if log.get_by_idempotency_key(base + "#cancelled") is not None:
        return False
    with _in_transaction(log) as txn:
        txn.append(
            EVENT_RUN_CANCELLED,
            RUNNER,
            {"recipe_node": recipe_node, "reason": reason[:_MAX_REASON]},
            project_id=project_id,
            idempotency_key=base + "#cancelled",
        )
        _append_record(
            txn,
            base,
            OP_CANCEL,
            project_id,
            RUNNER,
            request_sha256(OP_CANCEL, {"project_id": project_id, "recipe": recipe_node}),
            recipe_node=recipe_node,
            caller=_caller_ref(caller),
        )
    return True


# -- recording a step result --------------------------------------------------


def _artifact_node(txn: Any, graph: WorkGraph, op_id: str, artifact_id: str, base: str) -> None:
    """Add an ``artifact`` node for ``artifact_id`` and a ``produces`` edge from the op (RUNNER)."""
    node_event = graph.add_node_in(
        txn,
        "artifact",
        RUNNER,
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
        RUNNER,
        idempotency_key=f"{base}#art:{artifact_id}#edge",
    )


def _latest_claimant(txn: Any, project_id: str, op_id: str) -> str | None:
    """The ``worker`` of the latest ``recipe.claimed`` for ``op_id`` (by seq), or ``None``."""
    claims = _claim_events(txn, project_id, op_id)
    if not claims:
        return None
    latest = max(claims, key=lambda e: e.seq)
    worker = latest.payload.get("worker")
    return worker if isinstance(worker, str) else None


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
    enforce_holder: bool = True,
) -> bool:
    """Record a step attempt's result: move the op to done/failed and link produced artifacts.

    The write is attributed to RUNNER; ``actor`` is the real worker/caller, recorded as ``by`` in
    the payload and ``caller`` in the record. If the op is still active and has been claimed,
    only its latest claimant may record a result (``enforce_holder``); a stale worker's result is
    refused quietly (``False``). Done outputs that are not JSON-valid, nest over 64, or exceed
    64 KiB are converted to a ``failed`` result with ``outputs_invalid`` / ``outputs_too_large``
    (N3/F6), so no result can crash the worker or write log-invalid JSON. The error dict is bounded
    to ``{code, message, fix_hint}`` with each field capped at 4 KiB of UTF-8. Idempotent per op.
    """
    base = f"aar.recipe.result/{op_id}"
    if log.get_by_idempotency_key(base + "#result") is not None:
        return False
    out = dict(outputs or {})
    if outcome == OUTCOME_DONE:
        problem = _outputs_problem(out)
        if problem is not None:
            outcome, out, artifact_ids, error = (
                OUTCOME_FAILED,
                {},
                (),
                dict(_OUTPUT_ERRORS[problem]),
            )
    error = _bound_error(error)
    new_status = "done" if outcome == OUTCOME_DONE else "failed"
    payload = {
        "op_node": op_id,
        "outcome": outcome,
        "by": _caller_ref(actor),
        "outputs": out,
        "artifact_ids": list(artifact_ids),
        "error": error,
    }
    try:
        with _in_transaction(log) as txn:
            graph = WorkGraph(log, project_id)
            node = graph.node(op_id)
            if node is None or node.status in ("done", "failed", "cancelled"):
                return False
            if enforce_holder and node.status == "active":
                claimant = _latest_claimant(txn, project_id, op_id)
                if claimant is not None and claimant != _caller_ref(actor):
                    return False  # the lease moved on; this worker may no longer finish the op
            if node.status == "pending":
                graph.set_status_in(txn, op_id, "active", RUNNER)
            graph.set_status_in(txn, op_id, new_status, RUNNER)
            txn.append(
                EVENT_RESULT,
                RUNNER,
                payload,
                project_id=project_id,
                idempotency_key=base + "#result",
            )
            if outcome == OUTCOME_DONE:
                for aid in artifact_ids:
                    _artifact_node(txn, graph, op_id, aid, base)
            _append_record(
                txn,
                base,
                OP_RESULT,
                project_id,
                RUNNER,
                request_sha256(
                    OP_RESULT, {"project_id": project_id, "op_node": op_id, "outcome": outcome}
                ),
                op_node=op_id,
                outcome=outcome,
                caller=_caller_ref(actor),
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

    The claim event is written by RUNNER with the real ``worker`` and a ``claim_index`` in the
    payload, so heartbeats key per claim and the fold reads the holder from the latest claim. A
    ``pending`` op is claimed (moved to ``active``). An ``active`` op is stolen only when the
    latest claim's lease has lapsed and the step is idempotent; a live lease is never stolen, and
    a lapsed non-idempotent op is not re-claimed here (``advance`` fails it ``lease_lost``).
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
            prior = _claim_events(txn, project_id, op.op_id)
            if node.status == "pending":
                graph.set_status_in(txn, op.op_id, "active", RUNNER)
            elif node.status == "active":
                if _latest_lease_live(prior, _hb_events(txn, project_id, op.op_id), now):
                    return False  # never steal a live lease
                if not idempotent:
                    return False  # a lapsed non-idempotent op is not re-claimed
            else:
                return False  # done/failed/cancelled
            n = len(prior)
            key = f"{base}/{n}"
            txn.append(
                EVENT_CLAIMED,
                RUNNER,
                {
                    "op_node": op.op_id,
                    "step_id": op.step_id,
                    "attempt": op.attempt,
                    "round": op.round,
                    "lease_until": lease_until,
                    "worker": _caller_ref(worker),
                    "claim_index": n,
                },
                project_id=project_id,
                idempotency_key=key + "#claimed",
            )
            _append_record(
                txn,
                key,
                OP_CLAIM,
                project_id,
                RUNNER,
                request_sha256(OP_CLAIM, {"project_id": project_id, "op_node": op.op_id, "n": n}),
                op_node=op.op_id,
                caller=_caller_ref(worker),
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
    """Extend the lease on a claimed op, if ``worker`` is still its latest claimant.

    Heartbeats key per claim (``.../claim<index>/<n>``), so a new holder's beats never collide
    with the old holder's; a beat from a worker that lost the lease is refused silently. ``n``
    disambiguates successive beats from one holder.
    """
    with _in_transaction(log) as txn:
        claims = _claim_events(txn, project_id, op_id)
        if not claims:
            return
        latest = max(claims, key=lambda e: e.seq)
        if latest.payload.get("worker") != _caller_ref(worker):
            return  # not the current holder: its lease has moved on
        idx = latest.payload.get("claim_index")
        idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else 0
        base = f"aar.recipe.heartbeat/{op_id}/claim{idx}/{n}"
        if txn.find_by_idempotency_key(base + "#hb") is not None:
            return
        lease_until = iso(clock() + timedelta(seconds=lease_s))
        txn.append(
            EVENT_HEARTBEAT,
            RUNNER,
            {
                "op_node": op_id,
                "lease_until": lease_until,
                "claim_index": idx,
                "worker": _caller_ref(worker),
            },
            project_id=project_id,
            idempotency_key=base + "#hb",
        )
        _append_record(
            txn,
            base,
            OP_HEARTBEAT,
            project_id,
            RUNNER,
            request_sha256(OP_HEARTBEAT, {"project_id": project_id, "op_node": op_id, "n": n}),
            op_node=op_id,
            caller=_caller_ref(worker),
        )


def _claim_events(txn: Any, project_id: str, op_id: str) -> list[Any]:
    events = txn.read(0, project_id=project_id, types=[EVENT_CLAIMED])
    return [e for e in events if isinstance(e.payload, dict) and e.payload.get("op_node") == op_id]


def _hb_events(txn: Any, project_id: str, op_id: str) -> list[Any]:
    events = txn.read(0, project_id=project_id, types=[EVENT_HEARTBEAT])
    return [e for e in events if isinstance(e.payload, dict) and e.payload.get("op_node") == op_id]


def _latest_lease_live(claims: list[Any], beats: list[Any], now: Any) -> bool:
    """Whether the latest claim's lease (with its own heartbeats) has not yet lapsed."""
    if not claims:
        return False
    latest = max(claims, key=lambda e: e.seq)
    idx = latest.payload.get("claim_index")
    lease = _parse(latest.payload.get("lease_until"))
    for beat in beats:
        if beat.payload.get("claim_index") != idx:
            continue
        until = _parse(beat.payload.get("lease_until"))
        if until is not None and (lease is None or until > lease):
            lease = until
    return lease is not None and now <= lease


def _parse(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return parse_ts(value)
    except (ValueError, OverflowError):
        return None


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
    """Drive the recipe run of ``recipe_node`` one pass; return the resulting :class:`RunState`.

    ``actor`` is the real caller (a worker, a tool caller, a human); it is recorded on every write,
    but the writes themselves are made as RUNNER.
    """
    registry.ensure_builtins()
    project_id = project_of_node(log, recipe_node)
    state = load_state(log, recipe_node, project_id=project_id)
    if project_id is None or state.recipe_data is None or state.recipe_def is None:
        return state
    recipe_def = state.recipe_def
    now = clock()

    if state.recipe_status in ("active", "blocked"):
        if _reap_leases(log, state, recipe_def, actor, now):
            state = load_state(log, recipe_node, project_id=project_id)
        if _process_gates(log, state, recipe_def, actor):
            state = load_state(log, recipe_node, project_id=project_id)
        if _create_ready(log, state, recipe_def, actor):
            state = load_state(log, recipe_node, project_id=project_id)
        if _drive_hot(log, state, recipe_def, actor, dispatcher):
            state = load_state(log, recipe_node, project_id=project_id)
        if _open_gate_reviews(log, state, recipe_def, actor):
            state = load_state(log, recipe_node, project_id=project_id)

    if _reconcile_mirror(log, state, actor):
        state = load_state(log, recipe_node, project_id=project_id)
    return state


def _reap_leases(
    log: EventLog, state: RunState, recipe_def: RecipeDef, caller: Actor, now: Any
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
            actor=caller,
            outcome=OUTCOME_FAILED,
            enforce_holder=False,
            error={
                "code": "lease_lost",
                "message": "the worker's lease lapsed mid-step",
                "fix_hint": "Retry the step (it is not idempotent, so it was not re-claimed).",
            },
        ):
            changed = True
    return changed


def _process_gates(log: EventLog, state: RunState, recipe_def: RecipeDef, caller: Actor) -> bool:
    changed = False
    current = state.current_round()
    for op in state.ops:
        if op.kind != "gate" or op.review_id is None or op.is_cancelled:
            continue
        if op.status in ("done", "failed"):
            continue
        review = state.reviews.get(op.review_id)
        if review is None:
            continue
        if review.decision is not None and review.decision.decision == "approve":
            if record_result(
                log,
                project_id=state.project_id,
                op_id=op.op_id,
                actor=caller,
                outcome=OUTCOME_DONE,
                outputs={"decision": "approve"},
            ):
                changed = True
        elif review.decision is not None:  # rejected
            if op.round >= state.max_review_rounds:
                # Rejected in the final allowed round: the gate is terminal (F7).
                if record_result(
                    log,
                    project_id=state.project_id,
                    op_id=op.op_id,
                    actor=caller,
                    outcome=OUTCOME_FAILED,
                    error={
                        "code": "review_rounds_exhausted",
                        "message": f"the review was rejected in the final round {op.round}",
                        "fix_hint": "A human must intervene; the recipe is blocked (cancel it).",
                    },
                ):
                    changed = True
            elif op.round < current:
                step = recipe_def.step(op.step_id)
                if _rollback_round(log, state, recipe_def, step.on_reject, op.round, caller):
                    changed = True
        elif not review.pending and record_result(
            # The review closed without a decision: a human withdrew it (F8). Fail the gate
            # attempt; recipe_retry on this gate reopens a fresh review in the SAME round.
            log,
            project_id=state.project_id,
            op_id=op.op_id,
            actor=caller,
            outcome=OUTCOME_FAILED,
            error={
                "code": "review_withdrawn",
                "message": "the gate review was withdrawn before a human decided it",
                "fix_hint": "Retry the gate (recipe_retry) to open a fresh review, or cancel.",
            },
        ):
            changed = True
    return changed


def _rollback_round(
    log: EventLog,
    state: RunState,
    recipe_def: RecipeDef,
    on_reject: str | None,
    round_: int,
    caller: Actor,
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
            if _supersede_op(log, state.project_id, op, caller, "superseded by review rejection"):
                changed = True
    return changed


def _supersede_op(
    log: EventLog, project_id: str, op: OpAttempt, caller: Actor, reason: str
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
            if node.status in ("pending", "active"):
                graph.set_status_in(txn, op.op_id, "cancelled", RUNNER)
            txn.append(
                EVENT_CANCELLED,
                RUNNER,
                {"op_node": op.op_id, "reason": reason[:_MAX_REASON]},
                project_id=project_id,
                idempotency_key=base + "#cancelled",
            )
            _append_record(
                txn,
                base,
                OP_ADVANCE,
                project_id,
                RUNNER,
                request_sha256(OP_ADVANCE, {"project_id": project_id, "cancel": op.op_id}),
                op_node=op.op_id,
                caller=_caller_ref(caller),
            )
    except Exception as exc:
        if _is_conflict(exc):
            return False
        raise
    return True


def _create_ready(log: EventLog, state: RunState, recipe_def: RecipeDef, caller: Actor) -> bool:
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
            caller=caller,
            status=status,
        )
        if op_id is not None:
            changed = True
    return changed


def _drive_hot(
    log: EventLog, state: RunState, recipe_def: RecipeDef, caller: Actor, dispatcher: Any
) -> bool:
    current = state.current_round()
    changed = False
    for step in recipe_def.steps:
        if step.kind != "hot":
            continue
        op = state.latest_in_round(step.id, current)
        if op is None or op.status in ("done", "failed") or op.is_cancelled:
            continue
        try:
            changed |= _drive_one_hot(log, state, step, op, caller, dispatcher)
        except Exception:
            # F10: a dispatcher that raises must not escape advance or crash the worker loop.
            # The intent (if written) persists, so the next pass re-submits the same request_id;
            # the status view surfaces waiting:dispatcher_error. This is not a step failure.
            continue
    return changed


def _drive_one_hot(
    log: EventLog,
    state: RunState,
    step: StepDef,
    op: OpAttempt,
    caller: Actor,
    dispatcher: Any,
) -> bool:
    changed = False
    if not op.hot_submitted:
        if dispatcher is None:
            return False  # PC off: the op stays pending; status shows waiting:no_node_dispatcher
        _record_hot_intent(log, state.project_id, op, caller)
        job_ref = dispatcher.submit(
            request_id=op.op_id,
            project_id=state.project_id,
            kind=step.hot_kind or step.id,
            params=dict(step.hot_params or {}),
            agent_id=RUNNER.id,
        )
        _record_hot_submitted(log, state.project_id, op, caller, job_ref)
        changed = True
        view = dispatcher.poll(job_ref)
    elif dispatcher is not None:
        view = dispatcher.poll(op.hot_job_ref)
    else:
        return changed
    state_value = getattr(view, "state", None)
    if view is None or not isinstance(state_value, str) or state_value not in HOT_STATES:
        # N8: a None or garbage poll response is not a crash and not a step failure (yet). Record
        # it; after MAX_BAD_POLLS consecutive bad responses the step fails dispatcher_bad_response.
        if _record_bad_poll(log, state.project_id, op, caller):
            changed = True
        if op.bad_polls + 1 >= MAX_BAD_POLLS and record_result(
            log,
            project_id=state.project_id,
            op_id=op.op_id,
            actor=caller,
            outcome=OUTCOME_FAILED,
            error={
                "code": "dispatcher_bad_response",
                "message": f"the node dispatcher returned {MAX_BAD_POLLS} unusable responses",
                "fix_hint": "Check the node; retry the step once it reports properly.",
            },
        ):
            changed = True
        return changed
    if view.is_terminal:
        if view.state == "done":
            result = view.result
            if record_result(
                log,
                project_id=state.project_id,
                op_id=op.op_id,
                actor=caller,
                outcome=OUTCOME_DONE,
                outputs=dict(result.outputs) if result else {},
                artifact_ids=result.artifact_ids if result else (),
            ):
                changed = True
        elif record_result(
            log,
            project_id=state.project_id,
            op_id=op.op_id,
            actor=caller,
            outcome=OUTCOME_FAILED,
            error=view.error or {"code": "hot_failed", "message": ""},
        ):
            changed = True
    return changed


def _record_bad_poll(log: EventLog, project_id: str, op: OpAttempt, caller: Actor) -> bool:
    """Record one bad/None poll response for ``op`` (idempotent per count), for the N8 counter."""
    base = f"aar.recipe.badpoll/{op.op_id}/{op.bad_polls}"
    if log.get_by_idempotency_key(base + "#bad") is not None:
        return False
    with _in_transaction(log) as txn:
        txn.append(
            EVENT_HOT_BAD_POLL,
            RUNNER,
            {"op_node": op.op_id, "n": op.bad_polls},
            project_id=project_id,
            idempotency_key=base + "#bad",
        )
        _append_record(
            txn,
            base,
            OP_HOT_BAD_POLL,
            project_id,
            RUNNER,
            request_sha256(OP_HOT_BAD_POLL, {"project_id": project_id, "op_node": op.op_id}),
            op_node=op.op_id,
            caller=_caller_ref(caller),
        )
    return True


def _record_hot_intent(log: EventLog, project_id: str, op: OpAttempt, caller: Actor) -> None:
    base = f"aar.recipe.hotintent/{op.op_id}"
    if log.get_by_idempotency_key(base + "#intent") is not None:
        return
    with _in_transaction(log) as txn:
        txn.append(
            EVENT_HOT_INTENT,
            RUNNER,
            {"op_node": op.op_id, "request_id": op.op_id},
            project_id=project_id,
            idempotency_key=base + "#intent",
        )
        _append_record(
            txn,
            base,
            OP_HOT_INTENT,
            project_id,
            RUNNER,
            request_sha256(OP_HOT_INTENT, {"project_id": project_id, "op_node": op.op_id}),
            op_node=op.op_id,
            caller=_caller_ref(caller),
        )


def _record_hot_submitted(
    log: EventLog, project_id: str, op: OpAttempt, caller: Actor, job_ref: str
) -> None:
    base = f"aar.recipe.hotsubmitted/{op.op_id}"
    if log.get_by_idempotency_key(base + "#submitted") is not None:
        return
    with _in_transaction(log) as txn:
        graph = WorkGraph(log, project_id)
        node = graph.node(op.op_id)
        if node is not None and node.status == "pending":
            graph.set_status_in(txn, op.op_id, "active", RUNNER)
        txn.append(
            EVENT_HOT_SUBMITTED,
            RUNNER,
            {"op_node": op.op_id, "request_id": op.op_id, "job_ref": job_ref},
            project_id=project_id,
            idempotency_key=base + "#submitted",
        )
        _append_record(
            txn,
            base,
            OP_HOT_SUBMITTED,
            project_id,
            RUNNER,
            request_sha256(OP_HOT_SUBMITTED, {"project_id": project_id, "op_node": op.op_id}),
            op_node=op.op_id,
            caller=_caller_ref(caller),
        )


def _open_gate_reviews(
    log: EventLog, state: RunState, recipe_def: RecipeDef, caller: Actor
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
            title=f"gate {step.id} (round {op.round})",
            base=base,
        )
        with _in_transaction(log) as txn:
            txn.append(
                EVENT_REVIEW_LINKED,
                RUNNER,
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
                RUNNER,
                request_sha256(
                    OP_REVIEW_OPEN, {"project_id": state.project_id, "op_node": op.op_id}
                ),
                op_node=op.op_id,
                review_id=review_id,
                caller=_caller_ref(caller),
            )
        changed = True
    return changed


def _open_review_node(
    log: EventLog,
    project_id: str,
    subject: str,
    renders: tuple[str, ...],
    *,
    title: str,
    base: str,
) -> str:
    """Create a ``review`` node (status active) plus its ``reviews`` edge, requested by RUNNER.

    The requester is RUNNER, so no agent can withdraw the gate review (F2/F8); a human still can.
    Done without the artifact store (``advance`` has no store): the renders are the subject step's
    already-stored outputs. A human decides it in the console; nothing here can decide a review.
    """
    data = {
        "subject": subject,
        "renders": list(renders),
        "timeline": None,
        "fps": None,
        "requested_by": _caller_ref(RUNNER),
    }
    graph = WorkGraph(log, project_id)
    node = graph.add_node(
        "review",
        RUNNER,
        title=title,
        status="active",
        data=data,
        idempotency_key=f"{base}:node",
    )
    graph.add_edge("reviews", node.id, subject, RUNNER, idempotency_key=f"{base}:edge")
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


# -- recipe-node status mirror ------------------------------------------------


def _reconcile_mirror(log: EventLog, state: RunState, caller: Actor) -> bool:
    """Move the recipe node's graph status toward the evidence status (a mirror only).

    Graph status is never read as truth (the fold ignores it); this keeps the human-facing node
    status honest. F7: a recipe whose blocking condition cleared (a legitimate retry) is moved
    ``blocked -> active`` here, and a finished run ``blocked -> active -> done``.
    """
    target = state.recipe_status
    changed = False
    for _ in range(4):
        graph = WorkGraph(log, state.project_id)
        node = graph.node(state.recipe_node)
        if node is None or node.status == target:
            break
        path = _mirror_path(node.status, target)
        if not path:
            break
        if _move_recipe_node(log, state.project_id, state.recipe_node, path[0], caller):
            changed = True
        else:
            break
    return changed


def _mirror_path(current: str, target: str) -> list[str]:
    """The shortest sequence of one-step transitions from ``current`` to ``target`` (BFS)."""
    from collections import deque

    seen = {current}
    queue: deque[tuple[str, list[str]]] = deque([(current, [])])
    while queue:
        node, path = queue.popleft()
        for nxt in sorted(allowed_transitions(node)):
            if nxt in seen:
                continue
            step = [*path, nxt]
            if nxt == target:
                return step
            seen.add(nxt)
            queue.append((nxt, step))
    return []


def _move_recipe_node(
    log: EventLog, project_id: str, recipe_node: str, to: str, caller: Actor
) -> bool:
    """Move the recipe node one step (RUNNER). No hub.write record: this is a cosmetic mirror,
    guarded by the transition table, and blocked<->active may legitimately repeat."""
    from aar.hub.graph import IllegalTransition

    graph = WorkGraph(log, project_id)
    try:
        graph.set_status(recipe_node, to, RUNNER)
    except (IllegalTransition, KeyError):
        return False
    return True
