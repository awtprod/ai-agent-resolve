"""The recipe library ops the CLI and MCP tools wrap: start, complete, retry, cancel, status.

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
    MAX_OUTPUT_BYTES,
    _canonical,
    _supersede_op,
    advance,
    create_attempt,
    record_result,
    record_run_cancelled,
    record_started,
)
from aar.recipes.events import (
    OP_RETRY,
    OP_STEP_COMPLETE,
    OUTCOME_DONE,
    RUNNER,
)
from aar.recipes.gates import plan_gate
from aar.recipes.models import ArtifactExpect, JsonExpect, PlanExpect, RecipeDef, StepDef
from aar.recipes.state import RunState, load_state, project_of_node
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
    _revalidate_params(state, definition)
    if state.recipe_status in _FINISHED:
        raise WriteError(
            "step_not_open",
            f"recipe {recipe_node!r} is {state.recipe_status!r} and cannot be started",
            "Start a fresh recipe run (recipe_create) instead.",
            node_id=recipe_node,
            status=state.recipe_status,
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
    project_id, _graph, state, definition = _resolve(log, recipe_node)
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
    if len(_canonical(validated)) > MAX_OUTPUT_BYTES:
        raise WriteError(
            "outputs_too_large",
            f"the step's outputs exceed {MAX_OUTPUT_BYTES} bytes of canonical JSON",
            "Store large data as an artifact and return its id, not inline outputs.",
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
    # F4: round 1 derives from the recipe node. A review parent is accepted only when it is a gate
    # review of THIS recipe for the previous round (current-1): a plan deriving from another
    # recipe's review, or from an earlier round's review, is refused.
    current = state.current_round()
    allowed_reviews = {
        op.review_id
        for op in state.ops
        if op.kind == "gate" and op.review_id is not None and op.round == current - 1
    }
    if parent == recipe_node or (_is_review(graph, parent) and parent in allowed_reviews):
        pass
    else:
        raise WriteError(
            "expects_unmet",
            f"plan {plan_node_id!r} must derive from the recipe node or this recipe's "
            f"round-{current - 1} gate review",
            "Submit the plan with derives_from set to the recipe node (or the current round's "
            "review).",
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
    op = state.latest_in_round(step.id, state.current_round())
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
    step = _step(definition, step_id)
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
    fingerprint = request_sha256(
        OP_RETRY,
        {
            "project_id": project_id,
            "recipe_node": recipe_node,
            "step_id": step_id,
            "failed_op": op.op_id,
        },
    )
    if (
        _find_record(log, caller_key, idempotency_key, OP_RETRY, project_id, actor, fingerprint)
        is not None
    ):
        advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
        return {**_result("recipe.retry", log, recipe_node, dispatcher, clock), "replayed": True}
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
        attempt=op.attempt + 1,
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
    project_id, graph, state, _ = _resolve(log, recipe_node)
    _require_owner(graph, recipe_node, actor, "cancel")
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
