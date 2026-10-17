"""Lease board: who holds what across nodes, and where a human is working.

Nodes keep their leases in memory (:mod:`aar.node.leases`) and report every change
as a :class:`~aar.node.leases.LeaseEvent`. :func:`lease_event_to_hub` turns one of
those, plus the reporting node's id, into a hub event; it is pure, so a node can map
once and retry the same event. :func:`lease_board` projects the board from the log.

Node lease times are monotonic seconds, which mean nothing off the node, so the
mapping takes ``wall_offset`` (Unix time minus the node's monotonic clock, see
:func:`current_wall_offset`) and records wall-clock times.

Expiry is applied when the board is read, against an injectable clock: a lease whose
expiry has passed drops off even if its node never reported the expiry (because it
crashed, say). Presence ("a human is active on node X since T") works the same way.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from aar.hub.common import Clock, InvalidInput, clean_text, iso, parse_ts, utcnow
from aar.hub.events import Actor, Event, EventLog
from aar.node.leases import Holder, LeaseEvent

LEASE_EVENT_TYPES: dict[str, str] = {
    "granted": "lease.granted",
    "renewed": "lease.renewed",
    "released": "lease.released",
    "expired": "lease.expired",
    "preempted": "lease.preempted",
}
_HELD = frozenset({"lease.granted", "lease.renewed"})
_ENDED = frozenset({"lease.released", "lease.expired", "lease.preempted"})
EVENT_PRESENCE = "presence.changed"
_BOARD_TYPES = (*LEASE_EVENT_TYPES.values(), EVENT_PRESENCE)

MAX_NODE_ID = 128
MAX_PRESENCE_TTL = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class HubEventSpec:
    """An event ready to append to the hub log."""

    type: str
    payload: dict[str, Any]
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class BoardLease:
    """One held lease on the board."""

    node_id: str
    lease_id: str
    scope: str
    key: tuple[str, ...]
    holder_kind: str
    holder_id: str
    since: str
    expires_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "lease_id": self.lease_id,
            "scope": self.scope,
            "key": list(self.key),
            "holder": {"kind": self.holder_kind, "id": self.holder_id},
            "since": self.since,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class Presence:
    """A human active on a node."""

    node_id: str
    human_id: str | None
    since: str
    expires_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "human_id": self.human_id,
            "since": self.since,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class LeaseBoard:
    """The board as of ``as_of``."""

    leases: tuple[BoardLease, ...]
    presence: tuple[Presence, ...]
    as_of: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "leases": [lease.to_dict() for lease in self.leases],
            "presence": [entry.to_dict() for entry in self.presence],
        }


def current_wall_offset() -> float:
    """Return Unix time minus monotonic time on this host, for :func:`lease_event_to_hub`."""
    return time.time() - time.monotonic()


def _node_id(node_id: Any) -> str:
    return clean_text(node_id, "node_id", max_len=MAX_NODE_ID)


def _wall(monotonic: float, wall_offset: float, field: str) -> str:
    value = monotonic + wall_offset
    if not math.isfinite(value):
        raise InvalidInput(field, f"must be a finite time, got {value!r}")
    try:
        return iso(datetime.fromtimestamp(value, UTC))
    except (OverflowError, OSError, ValueError) as exc:
        raise InvalidInput(field, f"is not a representable time: {value!r}") from exc


def _holder(holder: Holder) -> dict[str, str]:
    return {"kind": holder.kind.value, "id": holder.id}


def lease_event_to_hub(event: LeaseEvent, node_id: str, *, wall_offset: float) -> HubEventSpec:
    """Map a node's lease event to the hub event that records it. Pure."""
    event_type = LEASE_EVENT_TYPES.get(event.kind)
    if event_type is None:
        raise InvalidInput("kind", f"unknown lease event kind {event.kind!r}")
    node_id = _node_id(node_id)
    lease = event.lease
    if lease.ttl is not None and not math.isfinite(lease.ttl):
        raise InvalidInput("ttl", f"must be finite, got {lease.ttl!r}")
    payload = {
        "node_id": node_id,
        "lease_id": lease.id,
        "scope": lease.scope.value,
        "key": list(lease.key),
        "holder": _holder(lease.holder),
        "acquired_at": _wall(lease.acquired_at, wall_offset, "acquired_at"),
        "expires_at": (
            None if lease.expires_at is None else _wall(lease.expires_at, wall_offset, "expires_at")
        ),
        "ttl": lease.ttl,
        "at": _wall(event.at, wall_offset, "at"),
        "by": None if event.by is None else _holder(event.by),
    }
    key = f"lease:{node_id}:{lease.id}:{event.kind}:{event.at!r}"
    return HubEventSpec(event_type, payload, key)


def record_lease_event(
    log: EventLog, event: LeaseEvent, node_id: str, *, wall_offset: float
) -> Event:
    """Append a node's lease event to the log, as that node."""
    spec = lease_event_to_hub(event, node_id, wall_offset=wall_offset)
    return log.append(
        spec.type,
        Actor("node", spec.payload["node_id"]),
        spec.payload,
        idempotency_key=spec.idempotency_key,
    )


def presence_to_hub(
    node_id: str,
    *,
    active: bool,
    at: datetime,
    human_id: str | None = None,
    ttl: timedelta | None = None,
) -> HubEventSpec:
    """Map "a human is (or is no longer) active on ``node_id`` since ``at``". Pure.

    ``ttl`` bounds how long an ``active`` report stays on the board without a fresh
    one, so a node that dies does not leave a human "present" forever.
    """
    node_id = _node_id(node_id)
    if human_id is not None:
        human_id = clean_text(human_id, "human_id", max_len=MAX_NODE_ID)
    if ttl is not None and (ttl <= timedelta(0) or ttl > MAX_PRESENCE_TTL):
        raise InvalidInput("ttl", f"must be positive and at most {MAX_PRESENCE_TTL}")
    payload = {
        "node_id": node_id,
        "active": bool(active),
        "human_id": human_id,
        "at": iso(at),
        "expires_at": iso(at + ttl) if (active and ttl is not None) else None,
    }
    return HubEventSpec(EVENT_PRESENCE, payload)


def record_presence(
    log: EventLog,
    node_id: str,
    *,
    active: bool,
    at: datetime,
    human_id: str | None = None,
    ttl: timedelta | None = None,
) -> Event:
    """Append a presence change for ``node_id``, as that node."""
    spec = presence_to_hub(node_id, active=active, at=at, human_id=human_id, ttl=ttl)
    return log.append(spec.type, Actor("node", spec.payload["node_id"]), spec.payload)


def _expired(expires_at: str | None, now: datetime) -> bool:
    return expires_at is not None and now >= parse_ts(expires_at)


def _lease_from(payload: dict[str, Any]) -> BoardLease | None:
    """Build a board entry from a lease event payload, or ``None`` if it is malformed."""
    try:
        holder = payload["holder"]
        key = tuple(payload["key"])
        entry = BoardLease(
            node_id=payload["node_id"],
            lease_id=payload["lease_id"],
            scope=payload["scope"],
            key=key,
            holder_kind=holder["kind"],
            holder_id=holder["id"],
            since=payload["acquired_at"],
            expires_at=payload.get("expires_at"),
        )
        if entry.expires_at is not None:
            parse_ts(entry.expires_at)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(isinstance(part, str) for part in (*key, entry.node_id, entry.lease_id)):
        return None
    return entry


def _presence_from(payload: dict[str, Any], previous: Presence | None) -> Presence | None:
    """Fold one presence event: the human now present, or ``None`` if nobody (or junk)."""
    at = payload.get("at")
    expires_at = payload.get("expires_at")
    human = payload.get("human_id")
    if payload.get("active") is not True or not isinstance(at, str):
        return None
    try:
        at_time = parse_ts(at)
        if expires_at is not None:
            parse_ts(expires_at)
    except (TypeError, ValueError):
        return None
    # A refresh of an ongoing presence keeps the original "since".
    ongoing = previous is not None and not _expired(previous.expires_at, at_time)
    return Presence(
        node_id=payload["node_id"],
        human_id=human if isinstance(human, str) else None,
        since=previous.since if ongoing and previous else at,
        expires_at=expires_at,
    )


def lease_board(log: EventLog, *, clock: Clock = utcnow) -> LeaseBoard:
    """Project who holds what, and which humans are active, as of ``clock()``.

    Only events whose actor is the node they describe count. Actors are self-declared
    (there is no auth in v1), so this stops mislabelled events, not a hostile writer.
    """
    now = clock()
    leases: dict[tuple[str, str], BoardLease] = {}
    presence: dict[str, Presence] = {}
    for event in log.iter_events(types=_BOARD_TYPES):
        payload = event.payload
        node_id = payload.get("node_id")
        if event.actor.kind != "node" or event.actor.id != node_id:
            continue
        if event.type == EVENT_PRESENCE:
            entry = _presence_from(payload, presence.get(node_id))
            if entry is None:
                presence.pop(node_id, None)
            else:
                presence[node_id] = entry
            continue
        lease_id = payload.get("lease_id")
        if not isinstance(lease_id, str):
            continue
        if event.type in _HELD:
            entry = _lease_from(payload)
            if entry is not None:
                leases[(node_id, lease_id)] = entry
        elif event.type in _ENDED:
            leases.pop((node_id, lease_id), None)
    held = sorted(
        (lease for lease in leases.values() if not _expired(lease.expires_at, now)),
        key=lambda lease: (lease.since, lease.node_id, lease.lease_id),
    )
    present = sorted(
        (entry for entry in presence.values() if not _expired(entry.expires_at, now)),
        key=lambda entry: (entry.since, entry.node_id),
    )
    return LeaseBoard(tuple(held), tuple(present), iso(now))
