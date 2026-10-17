"""Emergency stop: halt hot work everywhere, on one project, or on one node.

A stop is an event; so is a resume. The current state is a projection over those
events, so every reader (the console, agents over MCP, nodes before a hot op) sees
the same answer. Scopes are independent: resuming ``all`` does not lift a stop on one
project, and a project stop holds while ``all`` is running.

Anyone may pull the brake: a human from the console, or an agent or node that sees
something wrong. Only a human may resume. The projection enforces that too, so a
resume an agent appends straight to the log is ignored.

Nodes call :func:`assert_running` before each hot op (wired in S7/S10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aar.hub.common import InvalidInput, actor_dict, clean_text, require_human
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    IdempotencyConflict,
    Transaction,
    match_or_conflict,
)

EVENT_STOPPED = "control.stopped"
EVENT_RESUMED = "control.resumed"
_CONTROL_TYPES = (EVENT_STOPPED, EVENT_RESUMED)

SCOPE_KINDS = ("all", "project", "node")
MAX_REASON = 1000
MAX_SCOPE_ID = 128


@dataclass(frozen=True, slots=True)
class StopScope:
    """``all``, ``project:<id>`` or ``node:<id>``."""

    kind: str
    id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in SCOPE_KINDS:
            raise InvalidInput("scope", f"kind must be one of {list(SCOPE_KINDS)}")
        if self.kind == "all":
            if self.id is not None:
                raise InvalidInput("scope", "'all' takes no id")
        else:
            clean_text(self.id, "scope", max_len=MAX_SCOPE_ID)
            if self.id != self.id.strip():
                raise InvalidInput("scope", "id must not have surrounding whitespace")

    @classmethod
    def parse(cls, value: StopScope | str) -> StopScope:
        """Parse ``all``, ``project:<id>`` or ``node:<id>``."""
        if isinstance(value, StopScope):
            return value
        if not isinstance(value, str):
            raise InvalidInput("scope", f"must be a string, got {type(value).__name__}")
        if value == "all":
            return cls("all")
        kind, sep, ident = value.partition(":")
        if not sep or kind not in ("project", "node"):
            raise InvalidInput("scope", f"must be 'all', 'project:<id>' or 'node:<id>': {value!r}")
        return cls(kind, ident)

    @classmethod
    def all(cls) -> StopScope:
        return cls("all")

    @classmethod
    def project(cls, project_id: str) -> StopScope:
        return cls("project", project_id)

    @classmethod
    def node(cls, node_id: str) -> StopScope:
        return cls("node", node_id)

    def __str__(self) -> str:
        return "all" if self.kind == "all" else f"{self.kind}:{self.id}"


@dataclass(frozen=True, slots=True)
class StopRecord:
    """An active stop: what, why, who and when."""

    scope: StopScope
    reason: str
    by: Actor
    at: str
    event_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": str(self.scope),
            "reason": self.reason,
            "by": actor_dict(self.by),
            "at": self.at,
            "event_id": self.event_id,
        }


@dataclass(frozen=True, slots=True)
class ControlStatus:
    """Every active stop. ``running`` is true only when there are none."""

    stops: tuple[StopRecord, ...]

    @property
    def running(self) -> bool:
        return not self.stops

    def to_dict(self) -> dict[str, Any]:
        return {"running": self.running, "stops": [stop.to_dict() for stop in self.stops]}


class Stopped(Exception):
    """Raised by :func:`assert_running` when work in scope is stopped."""

    def __init__(self, record: StopRecord) -> None:
        super().__init__(
            f"emergency stop on {record.scope}: {record.reason} "
            f"(by {record.by.kind}:{record.by.id} at {record.at})"
        )
        self.record = record
        self.scope = record.scope
        self.reason = record.reason
        self.by = record.by
        self.at = record.at


class NotStopped(Exception):
    """Raised by :func:`resume` when the scope is not stopped."""


def _fold(events: list[Event]) -> dict[str, StopRecord]:
    stops: dict[str, StopRecord] = {}
    for event in events:
        try:
            scope = StopScope.parse(event.payload.get("scope"))
        except InvalidInput:
            continue
        key = str(scope)
        if event.type == EVENT_STOPPED:
            reason = event.payload.get("reason")
            stops[key] = StopRecord(
                scope=scope,
                reason=reason if isinstance(reason, str) else "",
                by=event.actor,
                at=event.ts,
                event_id=event.id,
            )
        elif event.type == EVENT_RESUMED and event.actor.kind == "human":
            stops.pop(key, None)
    return stops


def _order(stops: dict[str, StopRecord]) -> tuple[StopRecord, ...]:
    rank = {kind: i for i, kind in enumerate(SCOPE_KINDS)}
    return tuple(sorted(stops.values(), key=lambda s: (rank[s.scope.kind], s.scope.id or "")))


def status(log: EventLog) -> ControlStatus:
    """Return every active stop (``all`` first, then projects, then nodes)."""
    return ControlStatus(_order(_fold(list(log.iter_events(types=_CONTROL_TYPES)))))


def assert_running(
    log: EventLog, project_id: str | None = None, node_id: str | None = None
) -> None:
    """Raise :class:`Stopped` if ``all``, this project or this node is stopped."""
    stops = _fold(list(log.iter_events(types=_CONTROL_TYPES)))
    wanted = ["all"]
    if project_id is not None:
        wanted.append(str(StopScope.project(project_id)))
    if node_id is not None:
        wanted.append(str(StopScope.node(node_id)))
    for key in wanted:
        if key in stops:
            raise Stopped(stops[key])


def _project_of(scope: StopScope) -> str | None:
    return scope.id if scope.kind == "project" else None


def _retry(
    txn: Transaction,
    idempotency_key: str | None,
    event_type: str,
    payload: dict[str, Any],
    actor: Actor,
) -> Event | None:
    """Return the event an earlier identical request stored under this key, if any."""
    if idempotency_key is None:
        return None
    existing = txn.find_by_idempotency_key(idempotency_key)
    if existing is not None:
        match_or_conflict(existing, event_type, payload, payload.keys())
        if existing.actor != actor:
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key!r} was already used by another actor"
            )
    return existing


def _record(event: Event) -> StopRecord:
    return StopRecord(
        scope=StopScope.parse(event.payload["scope"]),
        reason=event.payload["reason"],
        by=event.actor,
        at=event.ts,
        event_id=event.id,
    )


def stop(
    log: EventLog,
    scope: StopScope | str,
    reason: str,
    by: Actor,
    *,
    idempotency_key: str | None = None,
) -> StopRecord:
    """Stop hot work in ``scope`` and return the stop as recorded.

    Stopping a scope that is already stopped records the new stop (and its reason)
    rather than failing: pressing STOP twice must never error.
    """
    scope = StopScope.parse(scope)
    reason = clean_text(reason, "reason", max_len=MAX_REASON, multiline=True)
    payload = {"scope": str(scope), "reason": reason}
    with log.transaction() as txn:
        existing = _retry(txn, idempotency_key, EVENT_STOPPED, payload, by)
        if existing is not None:
            return _record(existing)
        event = txn.append(
            EVENT_STOPPED,
            by,
            payload,
            project_id=_project_of(scope),
            idempotency_key=idempotency_key,
        )
    return _record(event)


def resume(
    log: EventLog,
    scope: StopScope | str,
    reason: str,
    by: Actor,
    *,
    idempotency_key: str | None = None,
) -> StopRecord:
    """Lift the stop on ``scope``. Only a human may resume.

    Returns the stop that was lifted. Raises :class:`NotStopped` if the scope is not
    stopped (unless this is a retry under the same ``idempotency_key``).
    """
    require_human(by, "resume after an emergency stop")
    scope = StopScope.parse(scope)
    reason = clean_text(reason, "reason", max_len=MAX_REASON, multiline=True)
    payload = {"scope": str(scope), "reason": reason}
    with log.transaction() as txn:
        events = txn.read(0, types=_CONTROL_TYPES)
        existing = _retry(txn, idempotency_key, EVENT_RESUMED, payload, by)
        if existing is not None:
            # The stop this resume lifted is the last stop of the scope before it.
            lifted = [
                e
                for e in events
                if e.type == EVENT_STOPPED
                and e.seq < existing.seq
                and e.payload.get("scope") == str(scope)
            ]
            if not lifted:  # pragma: no cover - a resume is only written over a stop
                raise NotStopped(f"{scope} was not stopped")
            return _record(lifted[-1])
        active = _fold(events).get(str(scope))
        if active is None:
            raise NotStopped(f"{scope} is not stopped")
        txn.append(
            EVENT_RESUMED,
            by,
            payload,
            project_id=_project_of(scope),
            idempotency_key=idempotency_key,
        )
    return active
