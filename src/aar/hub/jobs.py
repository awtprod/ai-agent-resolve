"""Durable remote Resolve jobs folded from the hub's append-only event log.

There is deliberately no jobs table.  Every mutation validates the complete current
projection and appends one event in the same ``BEGIN IMMEDIATE`` transaction.  A claim is
sticky to its authenticated node; an expired lease changes only the displayed state to
``node_offline``.  Reassignment requires an explicit human release and increments a fencing
generation, so messages from the old owner can never become current again.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pydantic import ValidationError

from aar.hub.common import (
    Clock,
    HumanRequired,
    InvalidInput,
    clean_text,
    iso,
    parse_ts,
    require_human,
    utcnow,
)
from aar.hub.events import Actor, Event, EventLog, match_or_conflict
from aar.node.branches import valid_agent_id
from aar.node.jobs import get_kind
from aar.node.jobs.model import TERMINAL_STATES
from aar.node.locator import ProjectLocator
from aar.node.ops import get_op

EVENT_SUBMITTED = "job.submitted"
EVENT_CLAIMED = "job.claimed"
EVENT_HEARTBEAT = "job.heartbeat"
EVENT_STATE = "job.state"
EVENT_FINISHED = "job.finished"
EVENT_CANCEL = "job.cancel_requested"
EVENT_RELEASED = "job.released"
EVENT_LOST = "job.lost_on_node"
JOB_EVENT_TYPES = frozenset(
    {
        EVENT_SUBMITTED,
        EVENT_CLAIMED,
        EVENT_HEARTBEAT,
        EVENT_STATE,
        EVENT_FINISHED,
        EVENT_CANCEL,
        EVENT_RELEASED,
        EVENT_LOST,
    }
)

DEFAULT_LEASE_SECONDS = 60.0
DEFAULT_HEARTBEAT_SECONDS = 15.0
MAX_JSON_BYTES = 256 * 1024
MAX_JSON_DEPTH = 32
MAX_ID = 128
MAX_JOBS = 10_000
MAX_JOB_EVENTS = 100_000
RECONCILE_PAGE_SIZE = 100
_LOCAL_TERMINAL = frozenset(state.value for state in TERMINAL_STATES)
_LOCAL_STATES = frozenset({"queued", "running", "waiting", "stalled", *_LOCAL_TERMINAL})


class JobError(ValueError):
    """A stable, typed refusal from the hub job queue."""

    code = "job_error"


class JobNotFound(JobError):
    code = "job_not_found"


class JobConflict(JobError):
    code = "job_conflict"


class StaleGeneration(JobError):
    code = "stale_generation"


class JobCorrupt(JobError):
    code = "job_corrupt"


@dataclass(slots=True)
class HubJob:
    job_id: str
    kind: str
    params: dict[str, Any]
    target: ProjectLocator
    agent_id: str
    priority: int
    approval_token: str | None
    creator: Actor
    assigned_node: str
    generation: int
    submitted_at: str
    submitted_seq: int
    claimed_node: str | None = None
    lease_expires_at: str | None = None
    reported_state: str | None = None
    report: dict[str, Any] | None = None
    terminal_state: str | None = None
    result: dict[str, Any] | None = None
    cancel_requested: bool = False
    cancel_reason: str | None = None
    lost_on_node: bool = False
    updated_at: str = ""
    event_seq: int = 0

    def state(self, now: datetime) -> str:
        if self.terminal_state is not None:
            return self.terminal_state
        if self.cancel_requested:
            return "cancel_requested"
        if self.lost_on_node:
            return "lost_on_node"
        if self.claimed_node is None:
            return "waiting_for_node"
        if self.lease_expires_at is not None and parse_ts(self.lease_expires_at) <= now:
            return "node_offline"
        if self.reported_state == "waiting":
            reason = (self.report or {}).get("waiting")
            kind = reason.get("kind") if isinstance(reason, dict) else None
            return f"waiting:{kind}" if isinstance(kind, str) and kind else "waiting"
        return self.reported_state or "claimed"

    def to_dict(self, now: datetime, *, include_payload: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "project": self.target.to_dict(),
            "agent_id": self.agent_id,
            "priority": self.priority,
            "creator": {"kind": self.creator.kind, "id": self.creator.id},
            "assigned_node": self.assigned_node,
            "claimed_node": self.claimed_node,
            "generation": self.generation,
            "lease_expires_at": self.lease_expires_at,
            "state": self.state(now),
            "cancel_requested": self.cancel_requested,
            "cancel_reason": self.cancel_reason,
            "submitted_at": self.submitted_at,
            "updated_at": self.updated_at,
            "event_seq": self.event_seq,
        }
        if include_payload:
            value.update(
                {
                    "params": self.params,
                    "approval_token": self.approval_token,
                    "report": self.report,
                    "result": self.result,
                }
            )
        return value


def _bounded_json(value: Any, field_name: str, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise InvalidInput(field_name, f"must not nest deeper than {MAX_JSON_DEPTH} levels")
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise InvalidInput(field_name, "object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list | tuple):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise InvalidInput(field_name, "numbers must be finite")
        elif item is not None and not isinstance(item, str | int | float | bool):
            raise InvalidInput(field_name, f"contains unsupported {type(item).__name__}")
    try:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidInput(field_name, f"must be JSON: {exc}") from exc
    if len(raw) > max_bytes:
        raise InvalidInput(field_name, f"must be at most {max_bytes} encoded bytes")
    return json.loads(raw)


def _key(operation: str, actor: Actor, supplied: str) -> str:
    supplied = clean_text(supplied, "idempotency_key", max_len=128)
    return f"job:{operation}:{actor.kind}:{actor.id}:{supplied}"


def _node(value: Any, field_name: str = "node_id") -> str:
    return clean_text(value, field_name, max_len=MAX_ID)


def _job_id(value: Any) -> str:
    value = clean_text(value, "job_id", max_len=MAX_ID)
    if not value.startswith("job_"):
        raise InvalidInput("job_id", "must start with 'job_'")
    return value


def _generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInput("generation", "must be a positive integer")
    return value


def _lease(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidInput("lease_seconds", "must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not 1 <= value <= 3600:
        raise InvalidInput("lease_seconds", "must be between 1 and 3600")
    return value


def _require_current(job: HubJob, node_id: str, generation: int) -> None:
    if job.claimed_node != node_id or job.generation != generation:
        raise StaleGeneration(
            f"job {job.job_id} belongs to node {job.claimed_node!r} generation "
            f"{job.generation}, not node {node_id!r} generation {generation}"
        )
    if job.terminal_state is not None:
        raise JobConflict(f"job {job.job_id} is already terminal ({job.terminal_state})")


def _must(jobs: dict[str, HubJob], job_id: str) -> HubJob:
    try:
        return jobs[job_id]
    except KeyError:
        raise JobNotFound(f"no job {job_id!r}") from None


def _payload(event: Event, names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(event.payload, dict) or any(name not in event.payload for name in names):
        raise JobCorrupt(f"malformed {event.type} event {event.id}")
    return event.payload


def _fold_events(events: Iterable[Event]) -> dict[str, HubJob]:
    jobs: dict[str, HubJob] = {}
    for index, event in enumerate(events, 1):
        if index > MAX_JOB_EVENTS:
            raise JobCorrupt(f"hub job event limit ({MAX_JOB_EVENTS}) exceeded")
        try:
            parse_ts(event.ts)
        except (TypeError, ValueError) as exc:
            raise JobCorrupt(f"job event {event.id} has an invalid timestamp") from exc
        p = event.payload
        if event.type == EVENT_SUBMITTED:
            p = _payload(
                event,
                (
                    "job_id",
                    "kind",
                    "params",
                    "project",
                    "agent_id",
                    "priority",
                    "assigned_node",
                    "generation",
                ),
            )
            job_id = _job_id(p["job_id"])
            if job_id in jobs:
                raise JobCorrupt(f"job {job_id} was submitted twice")
            if not isinstance(p["kind"], str) or not 0 < len(p["kind"]) <= 128:
                raise JobCorrupt(f"job {job_id} has an invalid kind")
            if not isinstance(p["params"], dict):
                raise JobCorrupt(f"job {job_id} params are not an object")
            try:
                _bounded_json(p["params"], "params")
            except InvalidInput as exc:
                raise JobCorrupt(f"job {job_id} has invalid params: {exc}") from exc
            if not isinstance(p["agent_id"], str) or not valid_agent_id(p["agent_id"]):
                raise JobCorrupt(f"job {job_id} has an invalid agent_id")
            if (
                isinstance(p["priority"], bool)
                or not isinstance(p["priority"], int)
                or abs(p["priority"]) > 1_000_000
            ):
                raise JobCorrupt(f"job {job_id} has an invalid priority")
            if isinstance(p["generation"], bool) or p["generation"] != 1:
                raise JobCorrupt(f"job {job_id} must start at generation 1")
            assigned_node = _node(p["assigned_node"], "assigned_node")
            approval_token = p.get("approval_token")
            if approval_token is not None and (
                not isinstance(approval_token, str) or not 0 < len(approval_token) <= MAX_ID
            ):
                raise JobCorrupt(f"job {job_id} has an invalid approval token reference")
            try:
                target = ProjectLocator.from_dict(p["project"])
            except (TypeError, ValueError, KeyError) as exc:
                raise JobCorrupt(f"job {job_id} has a bad project: {exc}") from exc
            if target.project_id is None:
                raise JobCorrupt(f"job {job_id} project has no project_id")
            jobs[job_id] = HubJob(
                job_id=job_id,
                kind=p["kind"],
                params=p["params"],
                target=target,
                agent_id=p["agent_id"],
                priority=p["priority"],
                approval_token=approval_token,
                creator=event.actor,
                assigned_node=assigned_node,
                generation=p["generation"],
                submitted_at=event.ts,
                submitted_seq=event.seq,
                updated_at=event.ts,
                event_seq=event.seq,
            )
            continue
        p = _payload(event, ("job_id", "generation"))
        job = _must(jobs, _job_id(p["job_id"]))
        generation = _generation(p["generation"])
        if generation != job.generation:
            raise JobCorrupt(
                f"event {event.id} has generation {generation}, expected {job.generation}"
            )
        if job.terminal_state is not None:
            raise JobCorrupt(f"event {event.id} follows terminal job {job.job_id}")
        if event.type == EVENT_CLAIMED:
            if job.claimed_node is not None or job.lost_on_node or job.cancel_requested:
                raise JobCorrupt(f"job {job.job_id} was claimed from a non-waiting state")
            node_id = _node(p.get("node_id"))
            if node_id != job.assigned_node:
                raise JobCorrupt(f"job {job.job_id} was claimed by its non-assigned node")
            job.claimed_node = node_id
            job.lease_expires_at = p.get("lease_expires_at")
            parse_ts(job.lease_expires_at)
            job.reported_state = "claimed"
        elif event.type == EVENT_HEARTBEAT:
            if _node(p.get("node_id")) != job.claimed_node:
                raise JobCorrupt(f"job {job.job_id} heartbeat came from the wrong node")
            job.lease_expires_at = p.get("lease_expires_at")
            parse_ts(job.lease_expires_at)
        elif event.type == EVENT_STATE:
            if (
                _node(p.get("node_id")) != job.claimed_node
                or p.get("state") not in _LOCAL_STATES
                or not isinstance(p.get("report"), dict)
            ):
                raise JobCorrupt(f"job {job.job_id} has an invalid state report")
            job.reported_state = p["state"]
            job.report = p.get("report")
        elif event.type == EVENT_FINISHED:
            if (
                _node(p.get("node_id")) != job.claimed_node
                or p.get("state") not in _LOCAL_TERMINAL
                or not isinstance(p.get("result"), dict)
            ):
                raise JobCorrupt(f"job {job.job_id} has an invalid terminal report")
            job.terminal_state = p["state"]
            job.result = p.get("result")
            job.lease_expires_at = None
        elif event.type == EVENT_CANCEL:
            if event.actor.kind != "human" and event.actor != job.creator:
                raise JobCorrupt(f"job {job.job_id} cancellation has no authority")
            job.cancel_requested = True
            job.cancel_reason = p.get("reason")
        elif event.type == EVENT_LOST:
            if _node(p.get("node_id")) != job.claimed_node:
                raise JobCorrupt(f"job {job.job_id} lost report came from the wrong node")
            job.lost_on_node = True
            job.reported_state = "lost_on_node"
            job.lease_expires_at = None
        elif event.type == EVENT_RELEASED:
            if event.actor.kind != "human":
                raise JobCorrupt(f"job {job.job_id} release was not written by a human")
            if job.claimed_node is None:
                raise JobCorrupt(f"job {job.job_id} was released before it was claimed")
            new_generation = p.get("new_generation")
            if new_generation != job.generation + 1:
                raise JobCorrupt(f"job {job.job_id} release has a non-monotonic generation")
            job.generation = new_generation
            job.assigned_node = _node(p.get("assigned_node"), "assigned_node")
            job.claimed_node = None
            job.lease_expires_at = None
            job.reported_state = None
            job.report = None
            job.cancel_requested = False
            job.cancel_reason = None
            job.lost_on_node = False
        else:  # caller filtered, but keep the fold closed
            raise JobCorrupt(f"unknown job event {event.type!r}")
        job.updated_at = event.ts
        job.event_seq = event.seq
    return jobs


def _events(source: EventLog | Any) -> Iterable[Event]:
    if isinstance(source, EventLog):
        return source.iter_events(types=JOB_EVENT_TYPES)
    return source.read(types=JOB_EVENT_TYPES)


def fold(source: EventLog | Any) -> dict[str, HubJob]:
    """Strictly fold every job event; malformed or impossible history is refused."""
    return _fold_events(_events(source))


def submit(
    log: EventLog,
    actor: Actor,
    *,
    kind: str,
    params: dict[str, Any],
    project: ProjectLocator,
    agent_id: str,
    assigned_node: str,
    idempotency_key: str,
    priority: int = 0,
    approval_token: str | None = None,
    clock: Clock = utcnow,
) -> tuple[HubJob, bool]:
    if actor.kind not in {"agent", "human"}:
        raise InvalidInput("actor", "only an agent or human may submit a job")
    kind = clean_text(kind, "kind", max_len=128)
    assigned_node = _node(assigned_node, "assigned_node")
    if not isinstance(agent_id, str) or not valid_agent_id(agent_id):
        raise InvalidInput("agent_id", "must be a valid agent id")
    if project.project_id is None:
        raise InvalidInput("project", "project_id is required")
    project_id = clean_text(project.project_id, "project.project_id", max_len=MAX_ID)
    if project_id != project.project_id:
        raise InvalidInput("project.project_id", "must have no surrounding whitespace")
    project_data = _bounded_json(project.to_dict(), "project", max_bytes=16 * 1024)
    if isinstance(priority, bool) or not isinstance(priority, int) or abs(priority) > 1_000_000:
        raise InvalidInput("priority", "must be an integer between -1000000 and 1000000")
    approval_token = (
        clean_text(approval_token, "approval_token", max_len=MAX_ID)
        if approval_token is not None
        else None
    )
    # Inspect the caller's raw tree before Pydantic's JSON-mode dump can coerce a NaN to
    # null. This is the durable boundary: malformed/non-finite input never becomes an event.
    params = _bounded_json(params, "params")
    try:
        spec = get_kind(kind)
        model = spec.params_model.model_validate(params)
        validated = model.model_dump(mode="json")
        steps = spec.build(model, project)
        for step in steps:
            op_name, op_params = getattr(step, "op", None), getattr(step, "params", None)
            if isinstance(op_name, str):
                get_op(op_name).params.model_validate(op_params)
    except (KeyError, ValueError, TypeError, ValidationError) as exc:
        raise InvalidInput("params", str(exc)) from exc
    validated = _bounded_json(validated, "params")
    key = _key("submit", actor, idempotency_key)
    base = {
        "kind": kind,
        "params": validated,
        "project": project_data,
        "agent_id": agent_id,
        "priority": priority,
        "approval_token": approval_token,
        "assigned_node": assigned_node,
        "generation": 1,
    }
    with log.transaction() as txn:
        existing = txn.find_by_idempotency_key(key)
        if existing is not None:
            match_or_conflict(existing, EVENT_SUBMITTED, base, base.keys(), actor=actor)
            jobs = fold(txn)
            return _must(jobs, existing.payload["job_id"]), True
        if len(fold(txn)) >= MAX_JOBS:
            raise JobConflict(f"hub job limit ({MAX_JOBS}) reached")
        payload = {"job_id": "job_" + uuid.uuid4().hex, **base}
        event = txn.append(
            EVENT_SUBMITTED,
            actor,
            payload,
            project_id=project_id,
            idempotency_key=key,
            ts=iso(clock()),
        )
        return _must(fold(txn), event.payload["job_id"]), False


def list_jobs(log: EventLog) -> list[HubJob]:
    return sorted(
        fold(log).values(),
        key=lambda job: (-job.priority, job.submitted_seq, job.job_id),
    )


def get_job(log: EventLog, job_id: str) -> HubJob | None:
    return fold(log).get(_job_id(job_id))


def claim(
    log: EventLog,
    node_id: str,
    projects: list[str],
    *,
    request_id: str,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
    clock: Clock = utcnow,
) -> tuple[HubJob | None, bool]:
    node_id = _node(node_id)
    lease_seconds = _lease(lease_seconds)
    if not isinstance(projects, list) or not projects:
        raise InvalidInput("projects", "must explicitly list at least one configured project id")
    allowed = {_node(item, "projects") for item in projects}
    actor = Actor("node", node_id)
    key = _key("claim", actor, request_id)
    now = clock()
    with log.transaction() as txn:
        existing = txn.find_by_idempotency_key(key)
        if existing is not None:
            job = _must(fold(txn), existing.payload["job_id"])
            if job.target.project_id not in allowed:
                raise JobConflict("claim replay is outside the requested projects")
            _require_current(job, node_id, existing.payload["generation"])
            return job, True
        jobs = fold(txn)
        candidates = [
            job
            for job in jobs.values()
            if job.assigned_node == node_id
            and job.target.project_id in allowed
            and job.claimed_node is None
            and job.terminal_state is None
            and not job.cancel_requested
        ]
        if not candidates:
            return None, False
        job = min(candidates, key=lambda item: (-item.priority, item.submitted_seq, item.job_id))
        txn.append(
            EVENT_CLAIMED,
            actor,
            {
                "job_id": job.job_id,
                "node_id": node_id,
                "generation": job.generation,
                "lease_expires_at": iso(now + timedelta(seconds=lease_seconds)),
            },
            project_id=job.target.project_id,
            idempotency_key=key,
            ts=iso(now),
        )
        return _must(fold(txn), job.job_id), False


def heartbeat(
    log: EventLog,
    node_id: str,
    job_id: str,
    generation: int,
    *,
    request_id: str,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
    clock: Clock = utcnow,
) -> tuple[HubJob, bool]:
    return _node_event(
        log,
        EVENT_HEARTBEAT,
        node_id,
        job_id,
        generation,
        request_id=request_id,
        extra={"lease_expires_at": iso(clock() + timedelta(seconds=_lease(lease_seconds)))},
        clock=clock,
    )


def report(
    log: EventLog,
    node_id: str,
    job_id: str,
    generation: int,
    state: str,
    report_data: dict[str, Any],
    *,
    request_id: str,
    clock: Clock = utcnow,
) -> tuple[HubJob, bool]:
    if state not in _LOCAL_STATES:
        raise InvalidInput("state", f"must be one of {sorted(_LOCAL_STATES)}")
    if not isinstance(report_data, dict):
        raise InvalidInput("report", "must be an object")
    report_data = _bounded_json(report_data, "report")
    terminal = state in _LOCAL_TERMINAL
    return _node_event(
        log,
        EVENT_FINISHED if terminal else EVENT_STATE,
        node_id,
        job_id,
        generation,
        request_id=request_id,
        extra={"state": state, "result" if terminal else "report": report_data},
        clock=clock,
    )


def lost_on_node(
    log: EventLog,
    node_id: str,
    job_id: str,
    generation: int,
    *,
    request_id: str,
    clock: Clock = utcnow,
) -> tuple[HubJob, bool]:
    return _node_event(
        log,
        EVENT_LOST,
        node_id,
        job_id,
        generation,
        request_id=request_id,
        extra={},
        clock=clock,
    )


def _node_event(
    log: EventLog,
    event_type: str,
    node_id: str,
    job_id: str,
    generation: int,
    *,
    request_id: str,
    extra: dict[str, Any],
    clock: Clock,
) -> tuple[HubJob, bool]:
    node_id, job_id, generation = _node(node_id), _job_id(job_id), _generation(generation)
    actor = Actor("node", node_id)
    key = _key(event_type, actor, request_id)
    payload = {"job_id": job_id, "node_id": node_id, "generation": generation, **extra}
    with log.transaction() as txn:
        existing = txn.find_by_idempotency_key(key)
        job = _must(fold(txn), job_id)
        # Fence before replay too: an old request key cannot confer ownership after release.
        if job.claimed_node != node_id or job.generation != generation:
            raise StaleGeneration(f"job {job_id} no longer belongs to this node/generation")
        if existing is not None:
            compare = payload.keys() - {"lease_expires_at"}
            match_or_conflict(existing, event_type, payload, compare, actor=actor)
            return job, True
        if (
            event_type == EVENT_FINISHED
            and job.claimed_node == node_id
            and job.generation == generation
            and job.terminal_state == extra.get("state")
            and job.result == extra.get("result")
        ):
            # A terminal local job is deliberately re-sent after restart. Its first ACK may
            # have been lost, so accept an identical result under a fresh request id without
            # adding another terminal event.
            return job, True
        _require_current(job, node_id, generation)
        txn.append(
            event_type,
            actor,
            payload,
            project_id=job.target.project_id,
            idempotency_key=key,
            ts=iso(clock()),
        )
        return _must(fold(txn), job_id), False


def reconcile(
    log: EventLog,
    node_id: str,
    local_jobs: list[dict[str, Any]],
    projects: list[str],
    *,
    request_id: str,
    clock: Clock = utcnow,
    range_start: str | None = None,
    range_end: str | None = None,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Reconcile a complete ID range; paginate replies independently of the input range.

    Bounds are exclusive start, inclusive end. The client covers the whole keyspace with
    consecutive ranges, so missing jobs in other input pages are never marked lost here.
    """
    node_id = _node(node_id)
    if not isinstance(local_jobs, list) or len(local_jobs) > RECONCILE_PAGE_SIZE:
        raise InvalidInput("jobs", f"must contain at most {RECONCILE_PAGE_SIZE} summaries")
    if not isinstance(projects, list) or not projects or len(projects) > 256:
        raise InvalidInput("projects", "must explicitly list 1 to 256 configured project ids")
    allowed = {_node(item, "projects") for item in projects}
    for bound in (range_start, range_end, cursor):
        if bound is not None:
            _job_id(bound)
    if range_start is not None and range_end is not None and range_start >= range_end:
        raise InvalidInput("range_end", "must be greater than range_start")

    def in_range(job_id: str) -> bool:
        return (range_start is None or job_id > range_start) and (
            range_end is None or job_id <= range_end
        )

    if cursor is not None and not in_range(cursor):
        raise InvalidInput("cursor", "must be within the reconciliation range")
    local: dict[str, int] = {}
    for item in local_jobs:
        if not isinstance(item, dict):
            raise InvalidInput("jobs", "each summary must be an object")
        job_id = _job_id(item.get("job_id"))
        if job_id in local or not in_range(job_id):
            raise InvalidInput("jobs", "summary ids must be unique and within the range")
        local[job_id] = _generation(item.get("generation"))
        if item.get("state") not in _LOCAL_STATES:
            raise InvalidInput("jobs", "summary state is invalid")
    # Fold, check and append under the same lock as release/claim. A reconnect cannot mark
    # a newly reassigned generation lost based on a stale projection.
    with log.transaction() as txn:
        current = fold(txn)
        for job in current.values():
            if (
                in_range(job.job_id)
                and job.claimed_node == node_id
                and job.target.project_id in allowed
                and job.terminal_state is None
                and not job.lost_on_node
                and local.get(job.job_id) != job.generation
            ):
                txn.append(
                    EVENT_LOST,
                    Actor("node", node_id),
                    {"job_id": job.job_id, "node_id": node_id, "generation": job.generation},
                    project_id=job.target.project_id,
                    ts=iso(clock()),
                )
        current = fold(txn)
        relevant = sorted(
            (
                job
                for job in current.values()
                if in_range(job.job_id)
                and (cursor is None or job.job_id > cursor)
                and job.target.project_id in allowed
                and (job.assigned_node == node_id or job.claimed_node == node_id)
            ),
            key=lambda job: job.job_id,
        )
        page = relevant[:RECONCILE_PAGE_SIZE]
        return {
            "jobs": [
                job.to_dict(clock(), include_payload=False)
                for job in page
                if job.claimed_node == node_id
            ],
            "known_job_ids": [job.job_id for job in page],
            "next_cursor": page[-1].job_id if len(relevant) > len(page) else None,
        }


def cancel(
    log: EventLog,
    actor: Actor,
    job_id: str,
    reason: str,
    *,
    idempotency_key: str,
    clock: Clock = utcnow,
) -> tuple[HubJob, bool]:
    job_id = _job_id(job_id)
    reason = clean_text(reason, "reason", max_len=1000, multiline=True)
    key = _key("cancel", actor, idempotency_key)
    with log.transaction() as txn:
        existing = txn.find_by_idempotency_key(key)
        jobs = fold(txn)
        job = _must(jobs, job_id)
        if actor.kind != "human" and actor != job.creator:
            raise HumanRequired("cancel a job created by another actor", actor)
        payload = {"job_id": job_id, "generation": job.generation, "reason": reason}
        if existing is not None:
            match_or_conflict(existing, EVENT_CANCEL, payload, payload.keys(), actor=actor)
            return job, True
        if job.terminal_state is not None:
            raise JobConflict(f"job {job_id} is already terminal")
        txn.append(
            EVENT_CANCEL,
            actor,
            payload,
            project_id=job.target.project_id,
            idempotency_key=key,
            ts=iso(clock()),
        )
        return _must(fold(txn), job_id), False


def release(
    log: EventLog,
    actor: Actor,
    job_id: str,
    assigned_node: str,
    reason: str,
    *,
    idempotency_key: str,
    clock: Clock = utcnow,
) -> tuple[HubJob, bool]:
    require_human(actor, "release and reassign a claimed job")
    job_id = _job_id(job_id)
    assigned_node = _node(assigned_node, "assigned_node")
    reason = clean_text(reason, "reason", max_len=1000, multiline=True)
    key = _key("release", actor, idempotency_key)
    with log.transaction() as txn:
        existing = txn.find_by_idempotency_key(key)
        jobs = fold(txn)
        job = _must(jobs, job_id)
        payload = {
            "job_id": job_id,
            "generation": job.generation,
            "new_generation": job.generation + 1,
            "assigned_node": assigned_node,
            "reason": reason,
        }
        if existing is not None:
            # On replay the current generation is already one higher; compare against stored.
            match_or_conflict(
                existing,
                EVENT_RELEASED,
                existing.payload
                | {"job_id": job_id, "assigned_node": assigned_node, "reason": reason},
                ("job_id", "assigned_node", "reason"),
                actor=actor,
            )
            return job, True
        if job.claimed_node is None:
            raise JobConflict(f"job {job_id} has not been claimed")
        if job.terminal_state is not None:
            raise JobConflict(f"job {job_id} is already terminal")
        txn.append(
            EVENT_RELEASED,
            actor,
            payload,
            project_id=job.target.project_id,
            idempotency_key=key,
            ts=iso(clock()),
        )
        return _must(fold(txn), job_id), False
