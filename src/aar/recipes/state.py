"""The folded state of one recipe run: a pure projection over runner-written evidence.

Everything the scheduler needs is derived here from evidence the runner identity (RUNNER) wrote:
the ``recipe.*`` events and the op nodes it created, plus human review decisions. **Graph node
status is never read as truth** -- it is only a mirror the runner writes. An op counts only when
its creation event and its ``recipe.*`` events were authored by RUNNER, and a linked review counts
only when the review node was created by RUNNER; anything else is ignored and listed in
``ignored`` for the status view's diagnostics. This is what stops an agent that owns a node (op
nodes are an agent-ownable kind) from walking the run forward with ``node_set_status``: a graph
status change it writes is simply not evidence.

A killed worker, a crashed tool or a second process all rebuild the same :class:`RunState` from the
log. The scheduler reads a :class:`RunState`, decides, and writes; it keeps no mutable state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aar.hub.common import parse_ts
from aar.hub.events import EventLog
from aar.hub.graph import GRAPH_EVENT_TYPES, WorkGraph
from aar.hub.reviews import Decision, Review, review_view
from aar.hub.writes import RecipeData, typed_recipe
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
    RECIPE_EVENT_TYPES,
    RUNNER,
    STEP_SCHEMA,
)
from aar.recipes.models import RecipeDef

_LIVE = frozenset({"pending", "active"})
_MAX_IGNORED = 50


def _as_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_ts(value)
    except (ValueError, OverflowError):
        return None


@dataclass
class _Claim:
    seq: int
    index: int
    worker: str | None
    lease_until: datetime | None


@dataclass
class OpAttempt:
    """One runner-created op node (one step attempt), with its runner evidence folded on.

    ``status`` is derived from evidence, never from the op's graph node status: done/failed from a
    runner ``recipe.result`` (for a gate, from the human review decision on the runner-linked
    review), cancelled from a runner ``recipe.cancelled``, active once claimed/submitted/under
    review, else pending.
    """

    op_id: str
    step_id: str
    attempt: int
    round: int
    kind: str
    status: str = "pending"
    created: str | None = None
    # lease (F3): the latest runner-written claim and its heartbeats decide the holder and lease.
    claim_count: int = 0
    latest_claim_index: int | None = None
    holder: str | None = None
    lease_until: datetime | None = None
    # hot
    hot_request_id: str | None = None
    hot_job_ref: str | None = None
    hot_submitted: bool = False
    hot_intent: bool = False
    bad_polls: int = 0
    # gate
    review_id: str | None = None
    review_round: int | None = None
    # result
    result_outcome: str | None = None
    result_at: str | None = None
    result_by: str | None = None
    result_outputs: dict[str, Any] | None = None
    result_artifacts: tuple[str, ...] = ()
    result_error: dict[str, Any] | None = None
    # supersede
    superseded: bool = False
    cancelled_reason: str | None = None
    # reopen (S19f): ``reopened_for`` is the op node's own data (why this agent attempt exists);
    # ``reopen`` is the plan of a reopen, recorded on the runner's supersede of the agent's done op
    reopened_for: dict[str, Any] | None = None
    reopen: dict[str, Any] | None = None

    @property
    def upstream(self) -> bool:
        """Whether its failed result is marked upstream (``error.upstream`` exactly True)."""
        return isinstance(self.result_error, dict) and self.result_error.get("upstream") is True

    @property
    def is_live(self) -> bool:
        return self.status in _LIVE

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    @property
    def is_cancelled(self) -> bool:
        return self.status == "cancelled"


@dataclass
class RunState:
    """A recipe run as folded from runner evidence."""

    project_id: str
    recipe_node: str
    recipe_data: RecipeData | None
    recipe_def: RecipeDef | None
    max_review_rounds: int
    started: bool = False
    run_cancelled: bool = False
    ops: list[OpAttempt] = field(default_factory=list)
    reviews: dict[str, Review] = field(default_factory=dict)
    ignored: list[dict[str, Any]] = field(default_factory=list)

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

    def op_by_id(self, op_id: str) -> OpAttempt | None:
        for op in self.ops:
            if op.op_id == op_id:
                return op
        return None

    def all_in_round(self, step_id: str, round_: int) -> list[OpAttempt]:
        """Every op of ``step_id`` in ``round_``, superseded ones included."""
        return [op for op in self.attempts_for(step_id) if op.round == round_]

    def next_attempt(self, step_id: str, round_: int) -> int:
        """The next attempt number of ``step_id`` in ``round_``: one past every op of the round,
        superseded ones included, so a new op never collides with a superseded one's key."""
        return 1 + max((op.attempt for op in self.all_in_round(step_id, round_)), default=0)

    def upstream_failure(self, step_id: str, round_: int) -> OpAttempt | None:
        """The latest attempt, in ``round_``, of the first cold step (recipe order) declaring
        ``on_fail == step_id`` whose latest attempt failed with a result marked upstream."""
        if self.recipe_def is None:
            return None
        for step in self.recipe_def.steps:
            if step.kind != "cold" or step.on_fail != step_id:
                continue
            op = self.latest_in_round(step.id, round_)
            if op is not None and op.status == "failed" and op.upstream:
                return op
        return None

    def is_step_done(self, step_id: str) -> bool:
        """A step is done if it has a non-cancelled, done op in any round (evidence only)."""
        return any(op.is_done for op in self.non_cancelled(step_id))

    def done_outputs(self, step_id: str) -> dict[str, Any]:
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

    def reopen_superseded(self) -> frozenset[str]:
        """The op ids a reopen supersedes (S19f): every op listed in the plan of a reopen whose
        first write landed, whether or not the rest of its writes are made yet."""
        listed: set[str] = set()
        for op in self.ops:
            if op.reopen is not None and plan_shape_ok(op.reopen):
                listed.update(op.reopen["supersede"])
        return frozenset(listed)

    def gate_decision(self, op: OpAttempt) -> Decision | None:
        """The human decision on gate op ``op``'s review, or ``None``. A decision on the review of a
        gate op a reopen superseded is ignored (S19f): the reopen withdraws that review, but a
        decision made before the withdrawal landed (in the window, or racing it) must neither open
        a round nor count as the gate's decision."""
        if op.kind != "gate" or op.review_id is None:
            return None
        review = self.reviews.get(op.review_id)
        if review is None or review.decision is None:
            return None
        if op.op_id in self.reopen_superseded():
            return None
        return review.decision

    def _rejected_rounds(self) -> set[int]:
        rejected: set[int] = set()
        for op in self.ops:
            decision = self.gate_decision(op)
            if decision is not None and decision.decision == "reject":
                rejected.add(op.round)
        return rejected

    def current_round(self) -> int:
        """The round steps run in now: 1, plus one per rejected round below the cap (clamped)."""
        rejected = self._rejected_rounds()
        highest = max(rejected) if rejected else 0
        if highest >= self.max_review_rounds:
            return self.max_review_rounds
        return highest + 1

    def rounds_exhausted(self) -> bool:
        """Whether a gate was rejected in the final allowed round."""
        rejected = self._rejected_rounds()
        return bool(rejected) and max(rejected) >= self.max_review_rounds

    # -- recipe-level status (evidence, not the graph mirror) ----------------

    def blocked_reason(self) -> str | None:
        """Why the run is blocked (needs a human), or ``None``."""
        if self.recipe_def is None:
            return None
        current = self.current_round()
        if self.rounds_exhausted():
            return "review_rounds_exhausted"
        for step in self.recipe_def.steps:
            op = self.latest_in_round(step.id, current)
            if op is None or op.status != "failed":
                continue
            if op.result_error and op.result_error.get("code") == "review_withdrawn":
                return "review_withdrawn"
            failed = len([o for o in self.in_round(step.id, current) if o.status == "failed"])
            if failed >= step.max_attempts:
                return op.result_error.get("code") if op.result_error else "step_failed"
        return None

    @property
    def recipe_status(self) -> str:
        """The run status from evidence: pending, active, blocked, done or cancelled."""
        if self.run_cancelled:
            return "cancelled"
        if not self.started:
            return "pending"
        if self.recipe_def is not None and all(
            self.is_step_done(step.id) for step in self.recipe_def.steps
        ):
            return "done"
        if self.blocked_reason() is not None:
            return "blocked"
        return "active"


# -- loading ------------------------------------------------------------------


def project_of_node(log: EventLog, node_id: str) -> str | None:
    """The project id of the graph node ``node_id`` (its creation event's project), or ``None``."""
    for event in log.iter_events(types=[GRAPH_EVENT_TYPES[0]]):
        payload = event.payload
        if isinstance(payload, dict) and payload.get("node_id") == node_id:
            return event.project_id
    return None


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
    reopened_for = data.get("reopened_for")
    return OpAttempt(
        op_id=node.id,
        step_id=step_id,
        attempt=attempt,
        round=round_,
        kind=data.get("step_kind") if isinstance(data.get("step_kind"), str) else "cold",
        created=node.created,
        reopened_for=reopened_for if isinstance(reopened_for, dict) else None,
    )


def _fold_recipe_events(
    ops: dict[str, OpAttempt],
    events: Iterable[Any],
    ignored: list[dict[str, Any]],
    recipe_node: str,
) -> tuple[bool, bool]:
    """Fold runner ``recipe.*`` events onto their ops; return (started, run_cancelled).

    Events whose actor is not RUNNER are ignored (and listed): no tool surface can produce the
    runner identity, so this is what makes the evidence trustworthy. ``recipe.started`` and
    ``recipe.run_cancelled`` count only for THIS recipe node (N1): the log read spans the whole
    project, so a start/cancel of another recipe in the project must not touch this run.
    """
    started = run_cancelled = False
    claims: dict[str, list[_Claim]] = {}
    heartbeats: dict[str, list[tuple[int, datetime | None]]] = {}
    for event in events:
        if event.actor != RUNNER:
            if len(ignored) < _MAX_IGNORED:
                ignored.append(
                    {
                        "kind": "event",
                        "type": event.type,
                        "id": event.id,
                        "reason": f"not written by the runner ({event.actor.kind})",
                    }
                )
            continue
        payload = event.payload
        if not isinstance(payload, dict):
            continue
        if event.type == EVENT_STARTED:
            if payload.get("recipe_node") == recipe_node:
                started = True
            continue
        if event.type == EVENT_RUN_CANCELLED:
            if payload.get("recipe_node") == recipe_node:
                run_cancelled = True
            continue
        op_id = payload.get("op_node")
        op = ops.get(op_id) if isinstance(op_id, str) else None
        if op is None:
            continue
        if event.type == EVENT_CLAIMED:
            idx = payload.get("claim_index")
            idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else op.claim_count
            claims.setdefault(op_id, []).append(
                _Claim(
                    event.seq, idx, _str(payload.get("worker")), _as_ts(payload.get("lease_until"))
                )
            )
        elif event.type == EVENT_HEARTBEAT:
            idx = payload.get("claim_index")
            idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else -1
            heartbeats.setdefault(op_id, []).append((idx, _as_ts(payload.get("lease_until"))))
        elif event.type == EVENT_HOT_INTENT:
            op.hot_intent = True
            op.hot_request_id = _str(payload.get("request_id")) or op.hot_request_id
        elif event.type == EVENT_HOT_SUBMITTED:
            op.hot_submitted = True
            op.hot_job_ref = _str(payload.get("job_ref")) or op.hot_job_ref
            op.hot_request_id = _str(payload.get("request_id")) or op.hot_request_id
        elif event.type == EVENT_HOT_BAD_POLL:
            op.bad_polls += 1
        elif event.type == EVENT_RESULT:
            op.result_outcome = payload.get("outcome")
            op.result_at = event.ts
            op.result_by = _str(payload.get("by"))
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
            op.review_id = _str(payload.get("review_id")) or op.review_id
            rnd = payload.get("round")
            if isinstance(rnd, int) and not isinstance(rnd, bool):
                op.review_round = rnd
        elif event.type == EVENT_CANCELLED:
            op.superseded = True
            op.cancelled_reason = _str(payload.get("reason"))
            reopen = payload.get("reopen")
            op.reopen = reopen if isinstance(reopen, dict) else None
    for op_id, op in ops.items():
        _apply_lease(op, claims.get(op_id, []), heartbeats.get(op_id, []))
    return started, run_cancelled


def plan_shape_ok(plan: dict[str, Any]) -> bool:
    """Whether a reopen plan (S19f) recorded on a runner ``recipe.cancelled`` has the shape the
    engine acts on; a plan without it is ignored everywhere."""
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


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _apply_lease(
    op: OpAttempt, claims: list[_Claim], beats: list[tuple[int, datetime | None]]
) -> None:
    op.claim_count = len(claims)
    if not claims:
        return
    latest = max(claims, key=lambda c: c.seq)
    op.latest_claim_index = latest.index
    op.holder = latest.worker
    lease = latest.lease_until
    for idx, until in beats:
        if idx == latest.index and until is not None and (lease is None or until > lease):
            lease = until
    op.lease_until = lease


def _status_of(op: OpAttempt, review: Review | None) -> str:
    if op.superseded:
        return "cancelled"
    if op.result_outcome == "done":
        return "done"
    if op.result_outcome == "failed":
        return "failed"
    if op.kind == "gate":
        # A gate op is open from creation: advance opens its review (needs status active), then it
        # awaits a human decision. An approve is recorded as a done result; a reject leaves it
        # active until the round rollback supersedes it.
        return "active"
    if op.kind == "agent":
        # An agent op is open for step_complete from creation; there is no claim/submit step.
        return "active"
    # cold / hot: pending until a worker claims it or the hot step is submitted.
    if op.claim_count > 0 or op.hot_submitted:
        return "active"
    return "pending"


def load_state(log: EventLog, recipe_node: str, *, project_id: str | None = None) -> RunState:
    """Fold the run of ``recipe_node`` from runner evidence (never from graph status)."""
    if project_id is None:
        project_id = project_of_node(log, recipe_node)
    if project_id is None:
        return RunState(
            project_id="",
            recipe_node=recipe_node,
            recipe_data=None,
            recipe_def=None,
            max_review_rounds=1,
        )
    graph = WorkGraph(log, project_id)
    recipe_node_obj = graph.node(recipe_node)
    recipe_data = typed_recipe(recipe_node_obj) if recipe_node_obj is not None else None
    recipe_def = registry.get(recipe_data.recipe) if recipe_data is not None else None
    max_rounds = recipe_data.max_review_rounds if recipe_data is not None else 1
    ignored: list[dict[str, Any]] = []
    ops: dict[str, OpAttempt] = {}
    for node in graph.nodes():
        if node.kind != "op":
            continue
        data = node.payload if isinstance(node.payload, dict) else {}
        if data.get("recipe_node") != recipe_node or data.get("schema") != STEP_SCHEMA:
            continue
        if graph.creator(node.id) != RUNNER:
            if len(ignored) < _MAX_IGNORED:
                ignored.append(
                    {
                        "kind": "op_node",
                        "id": node.id,
                        "reason": "op node not created by the runner",
                    }
                )
            continue
        op = _op_from_node(node)
        if op is not None:
            ops[op.op_id] = op
    started, run_cancelled = _fold_recipe_events(
        ops,
        log.read(0, project_id=project_id, types=RECIPE_EVENT_TYPES, limit=-1),
        ignored,
        recipe_node,
    )
    reviews: dict[str, Review] = {}
    for op in ops.values():
        rid = op.review_id
        # A linked review counts only if the review node itself was created by the runner.
        if rid is not None and rid not in reviews and graph.creator(rid) == RUNNER:
            review, _ = review_view(log, project_id, rid)
            if review is not None:
                reviews[rid] = review
    for op in ops.values():
        op.status = _status_of(op, reviews.get(op.review_id) if op.review_id else None)
    return RunState(
        project_id=project_id,
        recipe_node=recipe_node,
        recipe_data=recipe_data,
        recipe_def=recipe_def,
        max_review_rounds=max_rounds,
        started=started,
        run_cancelled=run_cancelled,
        ops=sorted(ops.values(), key=lambda op: (op.created or "", op.op_id)),
        reviews=reviews,
        ignored=ignored,
    )
