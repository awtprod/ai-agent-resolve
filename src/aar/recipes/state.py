"""The folded state of one recipe run: a pure projection over the event log.

Everything the runner needs is derived here from the op nodes (graph), the ``recipe.*`` events
and the reviews -- nothing is kept in memory between calls. A killed worker, a crashed tool or a
second process all rebuild the same :class:`RunState` from the log. The scheduler
(:mod:`aar.recipes.engine`) reads a :class:`RunState`, decides what to do, and writes; it never
keeps mutable state of its own.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aar.hub.common import parse_ts
from aar.hub.events import Actor, EventLog
from aar.hub.graph import GRAPH_EVENT_TYPES, WorkGraph
from aar.hub.reviews import Review, review_view
from aar.hub.writes import RecipeData, typed_recipe
from aar.recipes import registry
from aar.recipes.events import (
    EVENT_CANCELLED,
    EVENT_CLAIMED,
    EVENT_HEARTBEAT,
    EVENT_HOT_INTENT,
    EVENT_HOT_SUBMITTED,
    EVENT_RESULT,
    EVENT_REVIEW_LINKED,
    RECIPE_EVENT_TYPES,
    STEP_SCHEMA,
)
from aar.recipes.models import RecipeDef

_LIVE = frozenset({"pending", "active"})


def _as_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_ts(value)
    except (ValueError, OverflowError):
        return None


@dataclass
class OpAttempt:
    """One op node (one step attempt) plus the ``recipe.*`` facts folded onto it."""

    op_id: str
    step_id: str
    attempt: int
    round: int
    kind: str
    status: str
    created: str | None = None
    lease_until: datetime | None = None
    lease_worker: Actor | None = None
    hot_request_id: str | None = None
    hot_job_ref: str | None = None
    hot_submitted: bool = False
    hot_intent: bool = False
    review_id: str | None = None
    review_round: int | None = None
    result_outcome: str | None = None
    result_at: str | None = None
    result_outputs: dict[str, Any] | None = None
    result_artifacts: tuple[str, ...] = ()
    result_error: dict[str, Any] | None = None
    cancelled_reason: str | None = None
    superseded: bool = False

    @property
    def is_live(self) -> bool:
        return self.status in _LIVE and not self.is_cancelled

    @property
    def is_done(self) -> bool:
        return self.status == "done" and not self.superseded

    @property
    def is_cancelled(self) -> bool:
        # A done attempt cannot leave the terminal 'done' graph status, so a round rollback
        # supersedes it with a recipe.cancelled event rather than a status change.
        return self.superseded or self.status == "cancelled"


@dataclass
class RunState:
    """A recipe run as folded from the log."""

    project_id: str
    recipe_node: str
    recipe_status: str
    recipe_data: RecipeData | None
    recipe_def: RecipeDef | None
    max_review_rounds: int
    ops: list[OpAttempt] = field(default_factory=list)
    reviews: dict[str, Review] = field(default_factory=dict)

    # -- grouping -----------------------------------------------------------

    def attempts_for(self, step_id: str) -> list[OpAttempt]:
        return [op for op in self.ops if op.step_id == step_id]

    def non_cancelled(self, step_id: str) -> list[OpAttempt]:
        return [op for op in self.attempts_for(step_id) if not op.is_cancelled]

    def in_round(self, step_id: str, round_: int) -> list[OpAttempt]:
        return [
            op for op in self.attempts_for(step_id) if op.round == round_ and not op.is_cancelled
        ]

    def latest_in_round(self, step_id: str, round_: int) -> OpAttempt | None:
        ops = sorted(self.in_round(step_id, round_), key=lambda op: op.attempt)
        return ops[-1] if ops else None

    def is_step_done(self, step_id: str) -> bool:
        """A step is done if it has a non-cancelled, done op in any round."""
        return any(op.is_done for op in self.non_cancelled(step_id))

    def done_outputs(self, step_id: str) -> dict[str, Any]:
        """The recorded outputs of a step's done attempt (empty if none)."""
        for op in self.non_cancelled(step_id):
            if op.is_done and op.result_outputs is not None:
                return op.result_outputs
        return {}

    def done_op(self, step_id: str) -> OpAttempt | None:
        for op in self.non_cancelled(step_id):
            if op.is_done:
                return op
        return None

    # -- rounds -------------------------------------------------------------

    def _rejected_rounds(self) -> set[int]:
        rejected: set[int] = set()
        for op in self.ops:
            if op.kind != "gate" or op.review_id is None:
                continue
            review = self.reviews.get(op.review_id)
            if (
                review is not None
                and review.decision is not None
                and review.decision.decision == "reject"
            ):
                rejected.add(op.round)
        return rejected

    def current_round(self) -> int:
        """The round steps run in now: 1, plus one per consecutive rejected round below the cap."""
        rejected = self._rejected_rounds()
        highest = max(rejected) if rejected else 0
        if highest >= self.max_review_rounds:
            return self.max_review_rounds
        return highest + 1

    def rounds_exhausted(self) -> bool:
        """Whether a gate was rejected in the final allowed round."""
        rejected = self._rejected_rounds()
        return bool(rejected) and max(rejected) >= self.max_review_rounds


# -- loading ------------------------------------------------------------------


def project_of_node(log: EventLog, node_id: str) -> str | None:
    """The project id of the graph node ``node_id`` (its creation event's project), or ``None``."""
    for event in log.iter_events(types=[GRAPH_EVENT_TYPES[0]]):
        payload = event.payload
        if isinstance(payload, dict) and payload.get("node_id") == node_id:
            return event.project_id
    return None


def _fold_recipe_events(ops: dict[str, OpAttempt], events: Iterable[Any]) -> None:
    for event in events:
        payload = event.payload
        if not isinstance(payload, dict):
            continue
        op_id = payload.get("op_node")
        op = ops.get(op_id) if isinstance(op_id, str) else None
        if op is None:
            continue
        if event.type in (EVENT_CLAIMED, EVENT_HEARTBEAT):
            until = _as_ts(payload.get("lease_until"))
            if until is not None and (op.lease_until is None or until > op.lease_until):
                op.lease_until = until
            if event.type == EVENT_CLAIMED:
                op.lease_worker = event.actor
        elif event.type == EVENT_HOT_INTENT:
            op.hot_intent = True
            rid = payload.get("request_id")
            if isinstance(rid, str):
                op.hot_request_id = rid
        elif event.type == EVENT_HOT_SUBMITTED:
            op.hot_submitted = True
            ref = payload.get("job_ref")
            if isinstance(ref, str):
                op.hot_job_ref = ref
            rid = payload.get("request_id")
            if isinstance(rid, str):
                op.hot_request_id = rid
        elif event.type == EVENT_RESULT:
            op.result_outcome = payload.get("outcome")
            op.result_at = event.ts
            outputs = payload.get("outputs")
            op.result_outputs = outputs if isinstance(outputs, dict) else {}
            artifacts = payload.get("artifact_ids")
            op.result_artifacts = (
                tuple(a for a in artifacts if isinstance(a, str))
                if isinstance(artifacts, list)
                else ()
            )
            err = payload.get("error")
            op.result_error = err if isinstance(err, dict) else None
        elif event.type == EVENT_REVIEW_LINKED:
            rid = payload.get("review_id")
            if isinstance(rid, str):
                op.review_id = rid
            rnd = payload.get("round")
            if isinstance(rnd, int) and not isinstance(rnd, bool):
                op.review_round = rnd
        elif event.type == EVENT_CANCELLED:
            op.superseded = True
            reason = payload.get("reason")
            op.cancelled_reason = reason if isinstance(reason, str) else None


def _op_from_node(node: Any) -> OpAttempt | None:
    data = node.payload if isinstance(node.payload, dict) else {}
    if data.get("schema") != STEP_SCHEMA:
        return None
    step_id = data.get("step_id")
    attempt = data.get("attempt")
    round_ = data.get("round")
    if not (isinstance(step_id, str) and isinstance(attempt, int) and isinstance(round_, int)):
        return None
    if isinstance(attempt, bool) or isinstance(round_, bool):
        return None
    return OpAttempt(
        op_id=node.id,
        step_id=step_id,
        attempt=attempt,
        round=round_,
        kind=data.get("step_kind") if isinstance(data.get("step_kind"), str) else "cold",
        status=node.status,
        created=node.created,
    )


def load_state(log: EventLog, recipe_node: str, *, project_id: str | None = None) -> RunState:
    """Fold the run of ``recipe_node`` from the log.

    ``project_id`` is derived from the node when not given. A recipe node that does not exist (or
    is not a recipe) yields a :class:`RunState` whose ``recipe_data`` is ``None``.
    """
    if project_id is None:
        project_id = project_of_node(log, recipe_node)
    if project_id is None:
        return RunState(
            project_id="",
            recipe_node=recipe_node,
            recipe_status="unknown",
            recipe_data=None,
            recipe_def=None,
            max_review_rounds=1,
        )
    graph = WorkGraph(log, project_id)
    recipe_node_obj = graph.node(recipe_node)
    recipe_data = typed_recipe(recipe_node_obj) if recipe_node_obj is not None else None
    recipe_def = registry.get(recipe_data.recipe) if recipe_data is not None else None
    max_rounds = recipe_data.max_review_rounds if recipe_data is not None else 1
    ops: dict[str, OpAttempt] = {}
    for node in graph.nodes():
        if node.kind != "op":
            continue
        data = node.payload if isinstance(node.payload, dict) else {}
        if data.get("recipe_node") != recipe_node:
            continue
        op = _op_from_node(node)
        if op is not None:
            ops[op.op_id] = op
    _fold_recipe_events(ops, log.read(0, project_id=project_id, types=RECIPE_EVENT_TYPES, limit=-1))
    reviews: dict[str, Review] = {}
    for op in ops.values():
        if op.review_id is not None and op.review_id not in reviews:
            review, _ = review_view(log, project_id, op.review_id)
            if review is not None:
                reviews[op.review_id] = review
    return RunState(
        project_id=project_id,
        recipe_node=recipe_node,
        recipe_status=recipe_node_obj.status if recipe_node_obj is not None else "unknown",
        recipe_data=recipe_data,
        recipe_def=recipe_def,
        max_review_rounds=max_rounds,
        ops=sorted(ops.values(), key=lambda op: (op.created or "", op.op_id)),
        reviews=reviews,
    )
