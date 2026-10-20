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
from aar.hub.graph import WorkGraph, allowed_transitions
from aar.hub.reviews import get_review, withdraw
from aar.hub.writes import (
    WriteError,
    check_actor,
    check_id,
    set_own_status,
    typed_plan,
    typed_recipe,
    write_key,
)
from aar.recipes import registry
from aar.recipes.engine import advance, create_attempt, record_result
from aar.recipes.events import OUTCOME_DONE
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
    """Start a recipe run: move the recipe node to ``active`` and advance it."""
    actor = check_actor(actor)
    write_key(actor, idempotency_key)  # validate the key shape early
    project_id, graph, _state, _ = _resolve(log, recipe_node)
    _require_owner(graph, recipe_node, actor, "start")
    node = graph.node(recipe_node)
    assert node is not None
    if node.status == "pending":
        set_own_status(
            log, project_id, recipe_node, "active", actor, idempotency_key=idempotency_key
        )
    elif node.status in _FINISHED:
        raise WriteError(
            "step_not_open",
            f"recipe {recipe_node!r} is {node.status!r} and cannot be started",
            "Start a fresh recipe run (recipe_create) instead.",
            node_id=recipe_node,
            status=node.status,
        )
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
    idempotency_key: str,
    dispatcher: Any = None,
    clock: Clock = utcnow,
) -> dict[str, Any]:
    """Complete an agent step: validate ``outputs`` against its contract, then mark it done.

    Any agent (or a human) may complete an agent step; attribution is the caller. If the outputs
    fail the contract the call is refused and the step stays open.
    """
    actor = check_actor(actor)
    write_key(actor, idempotency_key)
    step_id = check_id(step_id, "step_id")
    if not isinstance(outputs, Mapping):
        raise WriteError(
            "expects_unmet",
            "outputs must be a JSON object",
            "Send the step's outputs as an object.",
        )
    project_id, _graph, state, definition = _resolve(log, recipe_node)
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
    record_result(
        log,
        project_id=project_id,
        op_id=op.op_id,
        actor=actor,
        outcome=OUTCOME_DONE,
        outputs=validated,
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
    if parent != recipe_node and not _is_review(graph, parent):
        raise WriteError(
            "expects_unmet",
            f"plan {plan_node_id!r} must derive from the recipe node or the round's review",
            "Submit the plan with derives_from set to the recipe node (or the review).",
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
    if expects.media_kind != "any" and (
        meta is None or meta.meta.get("kind") != expects.media_kind
    ):
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
    """Retry a failed step that still has attempts left: create its next attempt, then advance."""
    actor = check_actor(actor)
    write_key(actor, idempotency_key)
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
    failed = len([o for o in state.in_round(step_id, current) if o.status == "failed"])
    if failed >= step.max_attempts:
        raise WriteError(
            "no_attempts_left",
            f"step {step_id!r} has used all {step.max_attempts} attempts",
            "A human must intervene; the recipe is blocked.",
            step_id=step_id,
            max_attempts=step.max_attempts,
        )
    runner = graph.creator(recipe_node) or actor
    status = "pending" if step.kind in ("cold", "hot") else "active"
    create_attempt(
        log,
        project_id=project_id,
        recipe_node=recipe_node,
        step=step,
        round_=current,
        attempt=op.attempt + 1,
        actor=runner,
        status=status,
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
    """Cancel a run: cancel live ops, withdraw open reviews, move the recipe to ``cancelled``."""
    actor = check_actor(actor)
    write_key(actor, idempotency_key)
    project_id, graph, state, _ = _resolve(log, recipe_node)
    _require_owner(graph, recipe_node, actor, "cancel")
    from aar.recipes.engine import _supersede_op

    for op in state.ops:
        if op.is_live:
            _supersede_op(log, project_id, op, actor, reason)
    for op in state.ops:
        if op.review_id is None:
            continue
        review = get_review(log, project_id, op.review_id)
        if review is not None and review.pending:
            withdraw(log, project_id, op.review_id, reason, actor, idempotency_key=None)
    node = graph.node(recipe_node)
    if node is not None and "cancelled" in allowed_transitions(node.status):
        set_own_status(
            log,
            project_id,
            recipe_node,
            "cancelled",
            actor,
            idempotency_key=f"{idempotency_key}:recipe-cancel",
        )
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
