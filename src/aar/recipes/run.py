"""The recipe library ops the CLI and MCP tools wrap: start, complete, retry, reopen, cancel,
status.

Each returns plain JSON-ready data and refuses with :class:`~aar.hub.writes.WriteError`
(``{code, message, fix_hint, details}``). Every write calls :func:`~aar.recipes.engine.advance`
after it changes the log, so the run always moves as far as it can in one call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aar.hub.artifacts import ArtifactStore
from aar.hub.common import Clock, utcnow
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.reviews import get_review, withdraw
from aar.hub.writes import (
    WriteError,
    _append_record,
    _find_record,
    _in_transaction,
    check_actor,
    check_id,
    request_sha256,
    typed_plan,
    typed_recipe,
    write_key,
)
from aar.recipes import registry
from aar.recipes.engine import (
    _outputs_problem,
    _supersede_op,
    advance,
    begin_reopen,
    create_attempt,
    finish_reopen,
    finish_reopens,
    record_result,
    record_run_cancelled,
    record_started,
    reopen_plan,
)
from aar.recipes.events import (
    OP_REOPEN,
    OP_RETRY,
    OP_STEP_COMPLETE,
    OUTCOME_DONE,
    RUNNER,
)
from aar.recipes.gates import plan_gate
from aar.recipes.models import ArtifactExpect, JsonExpect, PlanExpect, RecipeDef, StepDef
from aar.recipes.reopen import reopen_refusal
from aar.recipes.state import OpAttempt, RunState, load_state, project_of_node
from aar.recipes.status import status_view

_FINISHED = frozenset({"done", "cancelled"})


def _unknown_node(recipe_node: str) -> WriteError:
    return WriteError(
        "unknown_node",
        f"no recipe node {recipe_node!r}",
        "Pass a recipe node id (recipe_create returns it).",
        node_id=recipe_node,
    )


def _resolve(log: EventLog, recipe_node: str) -> tuple[str, WorkGraph, RunState, RecipeDef]:
    """Resolve a recipe node to its project, graph, run state and registered def, or refuse."""
    recipe_node = check_id(recipe_node, "recipe_node")
    project_id = project_of_node(log, recipe_node)
    if project_id is None:
        raise _unknown_node(recipe_node)
    graph = WorkGraph(log, project_id)
    node = graph.node(recipe_node)
    if node is None:
        raise _unknown_node(recipe_node)
    if node.kind != "recipe":
        raise WriteError(
            "wrong_parent_kind",
            f"node {recipe_node!r} is a {node.kind}, not a recipe",
            "Pass a recipe node id.",
            node_id=recipe_node,
            kind=node.kind,
        )
    data = typed_recipe(node)
    if data is None:
        raise WriteError(
            "untyped_parent",
            f"recipe node {recipe_node!r} has no typed recipe data",
            "Create the recipe with recipe_create.",
            node_id=recipe_node,
        )
    definition = registry.get(data.recipe)
    if definition is None:
        raise WriteError(
            "unknown_recipe",
            f"recipe {data.recipe!r} is not registered",
            "Register the recipe in code (recipe_list shows registered recipes).",
            recipe=data.recipe,
            known=list(registry.names()),
        )
    state = load_state(log, recipe_node, project_id=project_id)
    return project_id, graph, state, definition


def _finish_reopens_first(
    log: EventLog, state: RunState, definition: RecipeDef, actor: Actor
) -> RunState:
    """S19f: every recipe write command first finishes a reopen whose writes were cut short (a
    crash after its first write), then decides on the state that leaves. Without this a command
    run inside that window (a ``recipe_retry`` of the failed step, say) writes next to the
    half-made reopen and the reopen never supersedes what it wrote."""
    if finish_reopens(log, state, definition, actor):
        return load_state(log, state.recipe_node, project_id=state.project_id)
    return state


def _revalidate_params(state: RunState, definition: RecipeDef) -> None:
    """Re-validate the stored recipe params against the registered model (F12).

    A recipe created through the generic graph API (or any surface that skipped the registry
    validator) may hold params the model rejects; refuse to start it with ``invalid_params``.
    """
    from pydantic import ValidationError

    raw = state.recipe_data.params if state.recipe_data is not None else {}
    try:
        definition.params_model.model_validate(dict(raw))
    except ValidationError as exc:
        raise WriteError(
            "invalid_params",
            f"recipe params do not validate ({exc.error_count()} problem(s))",
            "Recreate the recipe with params the registered model accepts.",
        ) from None


def _require_owner(graph: WorkGraph, recipe_node: str, actor: Actor, action: str) -> None:
    if actor.kind == "human":
        return
    if graph.creator(recipe_node) != actor:
        raise WriteError(
            "not_creator",
            f"{actor.kind}:{actor.id} did not create recipe {recipe_node!r}",
            f"Only the recipe's creator or a human may {action} it.",
            node_id=recipe_node,
        )


def recipe_start(
    log: EventLog,
    recipe_node: str,
    actor: Actor,
    *,
    idempotency_key: str,
    dispatcher: Any = None,
    clock: Clock = utcnow,
) -> dict[str, Any]:
    """Start a recipe run: record the ``recipe.started`` evidence and advance it.

    'Started' is the runner-written ``recipe.started`` event, not a graph status an agent could
    forge; the recipe node's graph status is only a mirror the engine reconciles in ``advance``.
    """
    actor = check_actor(actor)
    write_key(actor, idempotency_key)  # validate the key shape early
    project_id, _graph, state, definition = _resolve(log, recipe_node)
    _require_owner(_graph, recipe_node, actor, "start")
    state = _finish_reopens_first(log, state, definition, actor)
    _revalidate_params(state, definition)
    if state.recipe_status in _FINISHED:
        raise WriteError(
            "step_not_open",
            f"recipe {recipe_node!r} is {state.recipe_status!r} and cannot be started",
            "Start a fresh recipe run (recipe_create) instead.",
            node_id=recipe_node,
            status=state.recipe_status,
        )
    # N5: the recipe node must still be open. If it has not started yet, its graph status must be
    # pending; a creator that pre-moved it (active/done) can no longer start it here.
    node = _graph.node(recipe_node)
    if not state.started and node is not None and node.status != "pending":
        raise WriteError(
            "step_not_open",
            f"recipe {recipe_node!r} node is {node.status!r}, not an open (pending) recipe",
            "A recipe must be started from pending; recreate it if its node was moved.",
            node_id=recipe_node,
            status=node.status,
        )
    record_started(log, project_id=project_id, recipe_node=recipe_node, caller=actor)
    advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
    return _result("recipe.start", log, recipe_node, dispatcher, clock)


def step_complete(
    log: EventLog,
    store: ArtifactStore,
    recipe_node: str,
    step_id: str,
    attempt: int | None,
    outputs: Mapping[str, Any],
    actor: Actor,
    *,
    round: int | None = None,
    idempotency_key: str,
    dispatcher: Any = None,
    clock: Clock = utcnow,
) -> dict[str, Any]:
    """Complete an agent step: validate ``outputs`` against its contract, then mark it done.

    Any agent (or a human) may complete an agent step; the write is made by the runner with the
    caller recorded as ``by``. The caller's ``idempotency_key`` is honoured (F9): the same key and
    request replays the first result (``replayed``); the same key with a different request is
    ``idempotency_conflict``. ``round`` and ``attempt`` are optional guards that must name the live
    attempt, which stops a replayed earlier-round call from completing a later round (F4). If the
    outputs fail the contract the call is refused (``expects_unmet``) and the step stays open.
    """
    actor = check_actor(actor)
    caller_key = write_key(actor, idempotency_key)
    step_id = check_id(step_id, "step_id")
    if not isinstance(outputs, Mapping):
        raise WriteError(
            "expects_unmet",
            "outputs must be a JSON object",
            "Send the step's outputs as an object.",
        )
    # N3: reject bad raw outputs (NaN/inf, non-string keys, nesting over 64) FIRST -- before the
    # fingerprint canonicalises them -- so a deep structure can never crash with a RecursionError.
    raw_problem = _outputs_problem(dict(outputs))
    if raw_problem is not None:
        raise WriteError(
            raw_problem,
            f"the agent step's outputs are {raw_problem.replace('_', ' ')}",
            "Return finite, string-keyed JSON under 64 levels and 64 KiB.",
            step_id=step_id,
        )
    project_id, _graph, state, definition = _resolve(log, recipe_node)
    state = _finish_reopens_first(log, state, definition, actor)
    fingerprint = request_sha256(
        OP_STEP_COMPLETE,
        {
            "project_id": project_id,
            "recipe_node": recipe_node,
            "step_id": step_id,
            "outputs": dict(outputs),
        },
    )
    if (
        _find_record(
            log, caller_key, idempotency_key, OP_STEP_COMPLETE, project_id, actor, fingerprint
        )
        is not None
    ):
        advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
        return {
            **_result("recipe.step_complete", log, recipe_node, dispatcher, clock),
            "replayed": True,
        }
    step = _step(definition, step_id)
    if step.kind != "agent":
        raise WriteError(
            "step_not_open",
            f"step {step_id!r} is a {step.kind} step; only agent steps are completed explicitly",
            "Agent steps are the only ones completed with step_complete.",
            step_id=step_id,
            kind=step.kind,
        )
    current = state.current_round()
    if round is not None and round != current:
        raise WriteError(
            "step_not_open",
            f"step {step_id!r} is open in round {current}, not {round}",
            "Pass the round from recipe_status, or omit it.",
            step_id=step_id,
            round=current,
        )
    op = state.latest_in_round(step_id, current)
    if op is None or op.status != "active":
        raise WriteError(
            "step_not_open",
            f"step {step_id!r} has no open attempt in round {current}",
            "Start the recipe (recipe_start) so the step is activated first.",
            step_id=step_id,
            round=current,
        )
    if attempt is not None and op.attempt != attempt:
        raise WriteError(
            "step_not_open",
            f"step {step_id!r} open attempt is {op.attempt}, not {attempt}",
            "Pass the attempt from recipe_status, or omit it.",
            step_id=step_id,
            open_attempt=op.attempt,
        )
    validated = _validate_expects(log, store, state, recipe_node, step, dict(outputs))
    problem = _outputs_problem(validated)
    if problem is not None:
        raise WriteError(
            problem,
            f"the agent step's outputs are {problem.replace('_', ' ')}",
            "Return finite, string-keyed JSON under 64 levels and 64 KiB.",
            step_id=step_id,
        )
    artifact_ids = tuple(a for a in validated.get("artifact_ids", []) if isinstance(a, str))
    record_result(
        log,
        project_id=project_id,
        op_id=op.op_id,
        actor=actor,
        outcome=OUTCOME_DONE,
        outputs=validated,
        artifact_ids=artifact_ids,
    )
    with _in_transaction(log) as txn:
        _append_record(
            txn,
            caller_key,
            OP_STEP_COMPLETE,
            project_id,
            actor,
            fingerprint,
            recipe_node=recipe_node,
            step_id=step_id,
            op_node=op.op_id,
        )
    advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
    return _result("recipe.step_complete", log, recipe_node, dispatcher, clock)


def _validate_expects(
    log: EventLog,
    store: ArtifactStore,
    state: RunState,
    recipe_node: str,
    step: StepDef,
    outputs: dict[str, Any],
) -> dict[str, Any]:
    expects = step.expects
    if isinstance(expects, PlanExpect):
        return _validate_plan(log, state, recipe_node, step, outputs, expects)
    if isinstance(expects, ArtifactExpect):
        return _validate_artifact(store, state, step, outputs, expects)
    if isinstance(expects, JsonExpect):
        return _validate_json(step, outputs, expects)
    raise WriteError("expects_unmet", f"step {step.id!r} has no contract", "Fix the recipe.")


def _validate_plan(
    log: EventLog,
    state: RunState,
    recipe_node: str,
    step: StepDef,
    outputs: dict[str, Any],
    expects: PlanExpect,
) -> dict[str, Any]:
    plan_node_id = outputs.get("plan_node")
    if not isinstance(plan_node_id, str):
        raise WriteError(
            "expects_unmet",
            f"step {step.id!r} expects a plan: outputs must name plan_node",
            "Submit a plan (plan_submit) and pass its node id as plan_node.",
            step_id=step.id,
        )
    graph = WorkGraph(log, state.project_id)
    plan_node = graph.node(plan_node_id)
    if plan_node is None or plan_node.kind != "plan":
        raise WriteError(
            "expects_unmet",
            f"plan_node {plan_node_id!r} is not a plan node in this project",
            "Pass the plan node id that plan_submit returned.",
            step_id=step.id,
            plan_node=plan_node_id,
        )
    parent = _plan_parent(graph, plan_node_id)
    # F4: the round is the run's current round from the event log, never a caller number.
    #   round 1    -> the plan must derive from the recipe node.
    #   round n>=2 -> the plan must derive from round n-1's REJECTED gate review (not the recipe
    #                 node, and not any other round's or recipe's review).
    current = state.current_round()
    if current == 1:
        if parent != recipe_node:
            raise WriteError(
                "expects_unmet",
                f"plan {plan_node_id!r} must derive from the recipe node in round 1",
                "Submit the plan with derives_from set to the recipe node.",
                step_id=step.id,
                derives_from=parent,
            )
    else:
        prev_rejected = {
            op.review_id
            for op in state.ops
            if op.kind == "gate"
            and op.round == current - 1
            and op.review_id is not None
            and (rev := state.reviews.get(op.review_id)) is not None
            and rev.decision is not None
            and rev.decision.decision == "reject"
        }
        if not (_is_review(graph, parent) and parent in prev_rejected):
            raise WriteError(
                "expects_unmet",
                f"plan {plan_node_id!r} must derive from round {current - 1}'s rejected review",
                "In a later round, derive the new plan from the review that rejected the last one.",
                step_id=step.id,
                derives_from=parent,
            )
    data = typed_plan(plan_node)
    if data is None:
        raise WriteError(
            "expects_unmet",
            f"plan {plan_node_id!r} has no typed lint summary",
            "Submit the plan with plan_submit so it carries a lint summary.",
            step_id=step.id,
        )
    # F4 / S19f: refuse a plan whose canonical bytes equal any plan this recipe accepted before
    # (any round, any attempt, superseded or not): one a human rejected, or one whose render or
    # export failed because of it (a reopened attempt must change the plan).
    accepted_shas = {
        op.result_outputs.get("plan_sha256")
        for op in state.ops
        if op.kind == "agent"
        and op.result_outcome == OUTCOME_DONE
        and isinstance(op.result_outputs, dict)
    }
    accepted_shas.discard(None)
    if data.plan_sha256 in accepted_shas:
        raise WriteError(
            "plan_unchanged",
            f"plan {plan_node_id!r} is byte-for-byte a plan this recipe already took (a human "
            "rejected it, or its render or export failed)",
            "Change the plan in response to the rejection or the failure, then resubmit.",
            step_id=step.id,
            plan_sha256=data.plan_sha256,
        )
    gate = plan_gate(data, required_rules=expects.required_rules)
    if not gate.passed:
        raise WriteError(
            "expects_unmet",
            f"plan {plan_node_id!r} does not pass the plan gate",
            "Fix the plan's lint errors (and run every required rule), then resubmit.",
            step_id=step.id,
            gate=gate.to_dict(),
        )
    return {"plan_node": plan_node_id, "plan_sha256": data.plan_sha256, "gate": gate.to_dict()}


def _plan_parent(graph: WorkGraph, plan_node_id: str) -> str | None:
    for edge in graph.edges():
        if edge.type == "derives_from" and edge.src == plan_node_id:
            return edge.dst
    return None


def _is_review(graph: WorkGraph, node_id: str | None) -> bool:
    if node_id is None:
        return False
    node = graph.node(node_id)
    return node is not None and node.kind == "review"


def _validate_artifact(
    store: ArtifactStore,
    state: RunState,
    step: StepDef,
    outputs: dict[str, Any],
    expects: ArtifactExpect,
) -> dict[str, Any]:
    artifact_id = outputs.get("artifact_id")
    if not isinstance(artifact_id, str) or not store.has(artifact_id):
        raise WriteError(
            "expects_unmet",
            f"step {step.id!r} expects an artifact: outputs.artifact_id must be a stored artifact",
            "Store the artifact (ctx.put / the hub) and pass its id as artifact_id.",
            step_id=step.id,
        )
    meta = store.get_meta(artifact_id)
    # F5: the artifact must belong to this recipe's project.
    if meta is None or meta.project_id != state.project_id:
        raise WriteError(
            "expects_unmet",
            f"artifact {artifact_id} is not stored in this recipe's project",
            "Produce the artifact in this project (ctx.put records the project).",
            step_id=step.id,
            project_id=state.project_id,
        )
    if expects.media_kind != "any" and meta.meta.get("kind") != expects.media_kind:
        raise WriteError(
            "expects_unmet",
            f"artifact {artifact_id} is not of media kind {expects.media_kind!r}",
            "Produce an artifact of the stated media kind.",
            step_id=step.id,
            media_kind=expects.media_kind,
        )
    current = state.current_round()
    # N6: in a later round, refuse the artifact (id or sha) a rejected review already covered.
    rejected_artifacts = {
        aid
        for prior in state.ops
        if prior.step_id == step.id
        and prior.round < current
        and isinstance(prior.result_outputs, dict)
        for aid in (
            prior.result_outputs.get("artifact_id"),
            *(prior.result_outputs.get("artifact_ids") or []),
        )
        if isinstance(aid, str)
    }
    if artifact_id in rejected_artifacts:
        raise WriteError(
            "subject_unchanged",
            f"artifact {artifact_id} is the one a human already rejected",
            "Produce a new artifact in response to the rejection.",
            step_id=step.id,
        )
    op = state.latest_in_round(step.id, current)
    subject = op.op_id if op is not None else None
    return {"artifact_id": artifact_id, "subject_node": subject, "artifact_ids": [artifact_id]}


def _validate_json(step: StepDef, outputs: dict[str, Any], expects: JsonExpect) -> dict[str, Any]:
    from pydantic import ValidationError

    payload = outputs.get("json")
    if payload is None:
        raise WriteError(
            "expects_unmet",
            f"step {step.id!r} expects json: outputs must carry a json object",
            "Send the result under the 'json' key.",
            step_id=step.id,
        )
    try:
        model = expects.model.model_validate(payload)
    except ValidationError as exc:
        raise WriteError(
            "expects_unmet",
            f"step {step.id!r} json does not validate ({exc.error_count()} problem(s))",
            "Fix the fields the model rejects.",
            step_id=step.id,
        ) from None
    return {"json": model.model_dump(mode="json")}


def recipe_retry(
    log: EventLog,
    recipe_node: str,
    step_id: str,
    actor: Actor,
    *,
    idempotency_key: str,
    dispatcher: Any = None,
    clock: Clock = utcnow,
) -> dict[str, Any]:
    """Retry a failed step that still has attempts left: create its next attempt, then advance.

    A gate that failed ``review_rounds_exhausted`` is terminal: retry is refused with
    ``rounds_exhausted`` (only ``recipe_cancel`` exits, F7). A gate whose review was withdrawn
    (F8) may be retried by the
    creator or a human to open a fresh review **in the same round** -- it does not use a round. The
    caller's ``idempotency_key`` is honoured (F9): the same key and failed op replays; a different
    request is ``idempotency_conflict``.
    """
    actor = check_actor(actor)
    caller_key = write_key(actor, idempotency_key)
    step_id = check_id(step_id, "step_id")
    project_id, graph, state, definition = _resolve(log, recipe_node)
    _require_owner(graph, recipe_node, actor, "retry")
    state = _finish_reopens_first(log, state, definition, actor)
    step = _step(definition, step_id)
    # F9: a repeated retry with the same key replays the first result whatever has happened since.
    # The key is checked BEFORE the "has a failed attempt" check (so a replay after the step moved
    # on still replays, not step_not_open), and is fingerprinted on (recipe_node, step_id) so a
    # different step under the same key is a conflict.
    fingerprint = request_sha256(
        OP_RETRY,
        {"project_id": project_id, "recipe_node": recipe_node, "step_id": step_id},
    )
    if (
        _find_record(log, caller_key, idempotency_key, OP_RETRY, project_id, actor, fingerprint)
        is not None
    ):
        advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
        return {**_result("recipe.retry", log, recipe_node, dispatcher, clock), "replayed": True}
    current = state.current_round()
    op = state.latest_in_round(step_id, current)
    if op is None or op.status != "failed":
        raise WriteError(
            "step_not_open",
            f"step {step_id!r} has no failed attempt in round {current} to retry",
            "Only a failed step can be retried.",
            step_id=step_id,
        )
    code = op.result_error.get("code") if isinstance(op.result_error, dict) else None
    if step.kind == "gate" and code == "review_rounds_exhausted":
        raise WriteError(
            "rounds_exhausted",
            f"gate {step_id!r} used all {state.max_review_rounds} review round(s)",
            "The review rounds are used up; only recipe_cancel can exit this recipe.",
            step_id=step_id,
            max_review_rounds=state.max_review_rounds,
        )
    failed = len([o for o in state.in_round(step_id, current) if o.status == "failed"])
    if failed >= step.max_attempts:
        raise WriteError(
            "no_attempts_left",
            f"step {step_id!r} has used all {step.max_attempts} attempts",
            "A human must intervene; the recipe is blocked.",
            step_id=step_id,
            max_attempts=step.max_attempts,
        )
    status = "pending" if step.kind in ("cold", "hot") else "active"
    create_attempt(
        log,
        project_id=project_id,
        recipe_node=recipe_node,
        step=step,
        round_=current,
        attempt=state.next_attempt(step_id, current),
        caller=actor,
        status=status,
    )
    with _in_transaction(log) as txn:
        _append_record(
            txn,
            caller_key,
            OP_RETRY,
            project_id,
            actor,
            fingerprint,
            recipe_node=recipe_node,
            step_id=step_id,
            failed_op=op.op_id,
        )
    advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
    return _result("recipe.retry", log, recipe_node, dispatcher, clock)


def recipe_reopen(
    log: EventLog,
    recipe_node: str,
    step_id: str,
    actor: Actor,
    *,
    idempotency_key: str,
    dispatcher: Any = None,
    clock: Clock = utcnow,
) -> dict[str, Any]:
    """Reopen agent step ``step_id`` in the current round after a later cold step that declares
    ``on_fail == step_id`` failed because of its output (S19f).

    A reopen is a new attempt of the agent step in the same round: it supersedes every live or
    done op of the round from ``step_id`` onward (the set a round rollback supersedes), then
    creates the step's next attempt carrying ``reopened_for``. Rounds are untouched. Only the
    recipe's creator or a human may reopen; each precondition is a typed refusal (see
    :func:`~aar.recipes.reopen.reopen_refusal`). The caller's ``idempotency_key`` is honoured as for
    ``recipe_retry``.

    **Crash safety.** The first write supersedes the agent's done op with the whole reopen plan on
    the event; the other writes (supersedes, the new attempt) are idempotent and are made by
    :func:`~aar.recipes.engine.finish_reopen`. A crash after the first write is finished by the
    next ``advance`` (any engine tick: the worker loop), by any recipe write command (each one
    finishes it before it decides anything: start, step_complete, retry, reopen, cancel) or by
    re-running this command with the same key, whichever comes first; the caller's record is
    written last, so a re-run after a finished reopen whose record was lost still writes it and
    returns ok.
    """
    actor = check_actor(actor)
    caller_key = write_key(actor, idempotency_key)
    step_id = check_id(step_id, "step_id")
    project_id, graph, state, definition = _resolve(log, recipe_node)
    _require_owner(graph, recipe_node, actor, "reopen")
    state = _finish_reopens_first(log, state, definition, actor)
    step = _step(definition, step_id)
    fingerprint = request_sha256(
        OP_REOPEN,
        {"project_id": project_id, "recipe_node": recipe_node, "step_id": step_id},
    )
    if (
        _find_record(log, caller_key, idempotency_key, OP_REOPEN, project_id, actor, fingerprint)
        is not None
    ):
        advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
        return {**_result("recipe.reopen", log, recipe_node, dispatcher, clock), "replayed": True}
    mine = _own_reopen(state, step_id, caller_key)
    if mine is not None and state.recipe_status in _FINISHED:
        mine = None  # a run cancelled since the crash is not reopened: refused below
    if mine is None:
        refusal = reopen_refusal(state, definition, step)
        if refusal is not None:
            raise refusal
        current = state.current_round()
        agent_op = state.latest_in_round(step_id, current)
        failed = state.upstream_failure(step_id, current)
        if agent_op is None or failed is None:  # reopen_refusal has checked both; never None
            raise WriteError("step_not_done", f"step {step_id!r} cannot be reopened", "")
        plan = reopen_plan(state, definition, step, failed, caller_key)
        begin_reopen(log, state, agent_op, plan, actor)
        state = load_state(log, recipe_node, project_id=project_id)
        mine = _own_reopen(state, step_id, caller_key)
        if mine is None:
            # Lost a race: another call superseded the done op first. Help finish its reopen (so
            # the run never waits on it), then refuse: one reopen happened, not two.
            winner = state.op_by_id(agent_op.op_id)
            if winner is not None and winner.reopen is not None:
                finish_reopen(log, state, definition, winner, actor)
            advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
            raise WriteError(
                "already_reopened",
                f"step {step_id!r} was reopened (or superseded) by another call just now",
                "Read recipe_status: the step's new attempt is open; do not reopen it again.",
                step_id=step_id,
            )
    finish_reopen(log, state, definition, mine, actor)
    with _in_transaction(log) as txn:
        _append_record(
            txn,
            caller_key,
            OP_REOPEN,
            project_id,
            actor,
            fingerprint,
            recipe_node=recipe_node,
            step_id=step_id,
            reopened_op=mine.op_id,
        )
    advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
    return _result("recipe.reopen", log, recipe_node, dispatcher, clock)


def _own_reopen(state: RunState, step_id: str, caller_key: str) -> OpAttempt | None:
    """The op carrying this caller's reopen plan of ``step_id`` in the current round, if one was
    written (a re-run after a crash finds it here and finishes it)."""
    current = state.current_round()
    for op in state.all_in_round(step_id, current):
        plan = op.reopen
        if plan is not None and plan.get("caller") == caller_key and plan.get("round") == current:
            return op
    return None


def recipe_cancel(
    log: EventLog,
    recipe_node: str,
    actor: Actor,
    *,
    idempotency_key: str,
    reason: str = "recipe cancelled",
    clock: Clock = utcnow,
) -> dict[str, Any]:
    """Cancel a run: cancel live ops, withdraw open reviews, mark the run cancelled.

    All writes are the runner's (with the caller recorded). The gate reviews are withdrawn as
    RUNNER (their requester), so this works even though no agent may withdraw them; a cancel's own
    withdrawals never count as human rejections (F8). The run is cancelled from the
    ``recipe.run_cancelled`` evidence regardless, and ``advance`` reconciles the node mirror.
    """
    actor = check_actor(actor)
    write_key(actor, idempotency_key)
    project_id, graph, state, definition = _resolve(log, recipe_node)
    _require_owner(graph, recipe_node, actor, "cancel")
    state = _finish_reopens_first(log, state, definition, actor)
    # N5: cancelling a finished run is a no-op that reports already_finished (never flips a done
    # run to cancelled).
    if state.recipe_status in _FINISHED:
        return {
            "ok": True,
            "op": "recipe.cancel",
            "already_finished": True,
            **status_view(log, recipe_node, clock=clock),
        }
    for op in state.ops:
        if op.is_live:
            _supersede_op(log, project_id, op, actor, reason)
    for op in state.ops:
        if op.review_id is None:
            continue
        review = get_review(log, project_id, op.review_id)
        if review is not None and review.pending:
            withdraw(log, project_id, op.review_id, reason, RUNNER, idempotency_key=None)
    record_run_cancelled(
        log, project_id=project_id, recipe_node=recipe_node, caller=actor, reason=reason
    )
    advance(log, recipe_node, actor=actor, dispatcher=None, clock=clock)
    return _result("recipe.cancel", log, recipe_node, None, clock)


def recipe_status(
    log: EventLog, recipe_node: str, *, dispatcher: Any = None, clock: Clock = utcnow
) -> dict[str, Any]:
    """The agent-native status view (read-only)."""
    return status_view(log, recipe_node, dispatcher=dispatcher, clock=clock)


def recipe_list() -> dict[str, Any]:
    """The registered recipes, with versions and step counts."""
    recipes = []
    for name in registry.names():
        definition = registry.get(name)
        if definition is not None:
            recipes.append(
                {"name": name, "version": definition.version, "steps": len(definition.steps)}
            )
    return {"recipes": recipes}


def recipe_show(name: str) -> dict[str, Any]:
    """One registered recipe's steps, or ``{found: false}``."""
    definition = registry.get(name)
    if definition is None:
        return {"found": False, "recipe": name, "known": list(registry.names())}
    steps = [
        {
            "id": step.id,
            "kind": step.kind,
            "needs": list(step.needs),
            "max_attempts": step.max_attempts,
            "timeout_s": step.timeout_s,
            "idempotent": step.idempotent,
        }
        for step in definition.steps
    ]
    return {"found": True, "recipe": name, "version": definition.version, "steps": steps}


def _step(definition: RecipeDef, step_id: str) -> StepDef:
    try:
        return definition.step(step_id)
    except KeyError:
        raise WriteError(
            "unknown_step",
            f"recipe {definition.name!r} has no step {step_id!r}",
            "Use a step id from recipe_show.",
            step_id=step_id,
            steps=[s.id for s in definition.steps],
        ) from None


def _result(
    op: str, log: EventLog, recipe_node: str, dispatcher: Any, clock: Clock
) -> dict[str, Any]:
    return {
        "ok": True,
        "op": op,
        **status_view(log, recipe_node, dispatcher=dispatcher, clock=clock),
    }
