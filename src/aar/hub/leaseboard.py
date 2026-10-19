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

A node's feed (:mod:`aar.node.feed`) can lose events (a bounded outbox overflows) and
deliver them late or twice (a timed-out send that still commits, a resend after a lost
ack). The board copes with all three, in :func:`lease_board`, the one fold every view
reads:

- **Stamps.** Every event a feed sends carries ``feed: {nonce, seq}``: the feed process's
  random nonce and its own running order. The board orders a node's events by that
  (``_older``), never by when the hub committed them, so a late delivery never overwrites
  a newer state.
- **Claims.** The hub admits a feed process the first time it hears its nonce
  (``feed.claimed``, :func:`deliver_feed_event`); a newer process of the same node id
  replaces the older one, whose later events are refused (:class:`FeedConflict`) and, if
  written anyway, skipped. That is a restart; two live processes sharing a node id are
  refused loudly rather than left to erase each other's leases.
- **Gaps and snapshots.** A feed that dropped events sends ``feed.gap`` (the node is
  ``stale``), then reconciles: it re-announces its live leases and sends a
  ``lease.snapshot`` of their ids. A lease last held before the snapshot and not in it is
  no longer held (its release was lost, or a previous process held it); a gap before the
  snapshot is cleared. A ``feed.superseded`` record (a refused process) also marks the
  node stale until its current process next reconciles.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from aar.hub.common import (
    Clock,
    InvalidInput,
    SkippedEvent,
    clean_text,
    is_int,
    is_member,
    is_text,
    is_ts,
    iso,
    parse_ts,
    utcnow,
)
from aar.hub.events import Actor, Event, EventLog, Transaction
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
EVENT_FEED_GAP = "feed.gap"
EVENT_LEASE_SNAPSHOT = "lease.snapshot"
EVENT_FEED_CLAIMED = "feed.claimed"
EVENT_FEED_SUPERSEDED = "feed.superseded"
_BOARD_TYPES = (
    *LEASE_EVENT_TYPES.values(),
    EVENT_PRESENCE,
    EVENT_FEED_GAP,
    EVENT_LEASE_SNAPSHOT,
    EVENT_FEED_CLAIMED,
    EVENT_FEED_SUPERSEDED,
)

MAX_SNAPSHOT_LEASES = 10_000
"""Lease ids one ``lease.snapshot`` may carry; a node with more does not reconcile."""
MAX_FEED_NONCE = 64
MAX_FEED_SEQ = 10**18
MAX_NODE_ID = 128
MAX_LEASE_ID = 128
MAX_KEY_PART = 256
MAX_TTL_SECONDS = 10.0**9
MAX_PRESENCE_TTL = timedelta(hours=24)
_KEY_ARITY = {"instance": 1, "project": 1, "timeline": 2}
_HOLDER_KINDS = frozenset({"human", "agent"})


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

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """The board's order: ``(since, node id, lease id)``. Unique (a lease id is unique on
        its node) and fixed for the lease's lifetime, so a keyset cursor can page on it."""
        return (self.since, self.node_id, self.lease_id)

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


class FeedConflict(ValueError):
    """The hub refused a feed event: a newer process with the same node id replaced the
    process that sent it (see :func:`deliver_feed_event`). The sender must stop."""


@dataclass(frozen=True, slots=True)
class FeedGap:
    """A node's report that its feed dropped events, so its board history may be stale."""

    node_id: str
    dropped: int
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "dropped": self.dropped, "at": self.at}


@dataclass(frozen=True, slots=True)
class SupersededFeed:
    """A feed process the hub refused because a newer one (``owner``) holds its node id."""

    node_id: str
    nonce: str
    owner: str
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "nonce": self.nonce, "owner": self.owner, "at": self.at}


@dataclass(frozen=True, slots=True)
class LeaseBoard:
    """The board as of ``as_of``.

    ``gaps`` and ``superseded`` list the nodes whose view may be stale: a feed gap, or a
    refused second feed process, that no later snapshot from the node has reconciled.
    """

    leases: tuple[BoardLease, ...]
    presence: tuple[Presence, ...]
    as_of: str
    skipped: tuple[SkippedEvent, ...] = ()
    gaps: tuple[FeedGap, ...] = ()
    superseded: tuple[SupersededFeed, ...] = ()

    @property
    def stale(self) -> bool:
        return bool(self.gaps or self.superseded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "stale": self.stale,
            "leases": [lease.to_dict() for lease in self.leases],
            "presence": [entry.to_dict() for entry in self.presence],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "superseded": [entry.to_dict() for entry in self.superseded],
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
    problem = _lease_problem(payload)
    if problem is not None:
        raise InvalidInput("lease", problem)
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
    problem = _presence_problem(payload)
    if problem is not None:
        raise InvalidInput("presence", problem)
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


def _holder_ok(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and is_member(value.get("kind"), _HOLDER_KINDS)
        and is_text(value.get("id"), MAX_NODE_ID)
    )


def _lease_problem(payload: Any) -> str | None:
    """Why a lease event payload is unusable, or ``None``. Shared by writer and fold."""
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_NODE_ID):
        return "node_id is not an id"
    if not is_text(payload.get("lease_id"), MAX_LEASE_ID):
        return "lease_id is not an id"
    scope = payload.get("scope")
    if not is_member(scope, _KEY_ARITY):
        return "scope is not instance, project or timeline"
    key = payload.get("key")
    if (
        not isinstance(key, list)
        or len(key) != _KEY_ARITY[scope]
        or not all(is_text(part, MAX_KEY_PART) for part in key)
    ):
        return f"key is not {_KEY_ARITY[scope]} id(s) for a {scope} lease"
    if not _holder_ok(payload.get("holder")):
        return "holder is not a human or agent with an id"
    if not is_ts(payload.get("acquired_at")) or not is_ts(payload.get("at")):
        return "acquired_at or at is not a timestamp"
    expires_at = payload.get("expires_at")
    if expires_at is not None and not is_ts(expires_at):
        return "expires_at is not a timestamp"
    ttl = payload.get("ttl")
    if ttl is not None and not (
        isinstance(ttl, (int, float))
        and not isinstance(ttl, bool)
        and math.isfinite(ttl)
        and 0 < ttl <= MAX_TTL_SECONDS
    ):
        return "ttl is not a positive finite number of seconds"
    by = payload.get("by")
    if by is not None and not _holder_ok(by):
        return "by is not a human or agent with an id"
    return None


def _presence_problem(payload: Any) -> str | None:
    """Why a presence payload is unusable, or ``None``. Shared by writer and fold."""
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_NODE_ID):
        return "node_id is not an id"
    if not isinstance(payload.get("active"), bool):
        return "active is not true or false"
    human = payload.get("human_id")
    if human is not None and not is_text(human, MAX_NODE_ID):
        return "human_id is not an id"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    expires_at = payload.get("expires_at")
    if expires_at is not None and not is_ts(expires_at):
        return "expires_at is not a timestamp"
    return None


def _author_problem(event: Event) -> str | None:
    node_id = event.payload.get("node_id")
    if event.actor.kind != "node" or event.actor.id != node_id:
        return f"written by {event.actor.kind}:{event.actor.id}, not by node {node_id}"
    return None


# -- the feed protocol: gaps, snapshots, claims, and the node-side order ----------------------


def _stamp_problem(payload: dict[str, Any]) -> str | None:
    """Why a payload's ``feed`` stamp is unusable, or ``None`` (also when it has none)."""
    stamp = payload.get("feed")
    if stamp is None:
        return None
    if not (
        isinstance(stamp, dict)
        and is_text(stamp.get("nonce"), MAX_FEED_NONCE)
        and is_int(stamp.get("seq"), 1, MAX_FEED_SEQ)
    ):
        return "feed is not a stamp of a feed nonce and a positive sequence number"
    return None


def _stamp_of(payload: dict[str, Any]) -> tuple[str, int] | None:
    stamp = payload.get("feed")
    return None if stamp is None else (stamp["nonce"], stamp["seq"])


def _gap_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_NODE_ID):
        return "node_id is not an id"
    if not is_int(payload.get("dropped"), 1, 10**18):
        return "dropped is not a positive count"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    return None


def _snapshot_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_NODE_ID):
        return "node_id is not an id"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    ids = payload.get("lease_ids")
    if not isinstance(ids, list) or len(ids) > MAX_SNAPSHOT_LEASES:
        return f"lease_ids is not a list of at most {MAX_SNAPSHOT_LEASES} ids"
    if not all(is_text(lease_id, MAX_LEASE_ID) for lease_id in ids):
        return "lease_ids holds something that is not a lease id"
    return None


def _claim_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_NODE_ID):
        return "node_id is not an id"
    if not is_text(payload.get("nonce"), MAX_FEED_NONCE):
        return "nonce is not a feed nonce"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    return None


def _superseded_problem(payload: Any) -> str | None:
    problem = _claim_problem(payload)
    if problem is None and not is_text(payload.get("owner"), MAX_FEED_NONCE):
        problem = "owner is not a feed nonce"
    return problem


_PROBLEMS = {
    **dict.fromkeys(LEASE_EVENT_TYPES.values(), _lease_problem),
    EVENT_PRESENCE: _presence_problem,
    EVENT_FEED_GAP: _gap_problem,
    EVENT_LEASE_SNAPSHOT: _snapshot_problem,
    EVENT_FEED_CLAIMED: _claim_problem,
    EVENT_FEED_SUPERSEDED: _superseded_problem,
}
_STAMPED = frozenset(
    {*LEASE_EVENT_TYPES.values(), EVENT_PRESENCE, EVENT_FEED_GAP, EVENT_LEASE_SNAPSHOT}
)

_Order = tuple[int, int]
"""A feed event's node-side order: ``(rank of its feed process, its feed seq)``."""
_Pos = tuple[_Order | None, int]
"""Where an event stands: its node-side order (``None`` if unstamped) and its log seq."""


def _older(a: _Pos, b: _Pos) -> bool:
    """Whether ``a`` happened before ``b`` on its node.

    Two stamped events of a node compare by the node's own order (the rank of the feed
    process that sent them, then its sequence number), never by when the hub happened to
    commit them: a delivery that timed out and committed late keeps its place. An
    unstamped event (written by hand, or by :func:`record_lease_event`) has no node-side
    order, so it falls back to log order.
    """
    if a[0] is not None and b[0] is not None:
        return a[0] < b[0]
    return a[1] < b[1]


def _board_lease(payload: dict[str, Any]) -> BoardLease:
    return BoardLease(
        node_id=payload["node_id"],
        lease_id=payload["lease_id"],
        scope=payload["scope"],
        key=tuple(payload["key"]),
        holder_kind=payload["holder"]["kind"],
        holder_id=payload["holder"]["id"],
        since=payload["acquired_at"],
        expires_at=payload.get("expires_at"),
    )


def _presence_entry(payload: dict[str, Any], previous: Presence | None) -> Presence:
    at = payload["at"]
    # A refresh of an ongoing presence keeps the original "since".
    ongoing = previous is not None and not _expired(previous.expires_at, parse_ts(at))
    return Presence(
        node_id=payload["node_id"],
        human_id=payload.get("human_id"),
        since=previous.since if ongoing and previous else at,
        expires_at=payload.get("expires_at"),
    )


class _Fold:
    """One pass over the log, in seq order, keeping every node's feed state."""

    def __init__(self) -> None:
        self.leases: dict[tuple[str, str], tuple[BoardLease, _Pos]] = {}
        self.ended: set[tuple[str, str]] = set()
        self.presence: dict[str, Presence] = {}
        self.presence_at: dict[str, _Pos] = {}
        self.claims: dict[str, dict[str, int]] = {}  # node -> feed nonce -> rank
        self.owner: dict[str, str] = {}  # node -> the nonce of its current feed process
        self.snapshots: dict[str, tuple[frozenset[str], _Pos]] = {}
        self.gaps: dict[str, tuple[FeedGap, _Pos]] = {}
        self.superseded: dict[str, tuple[SupersededFeed, _Pos]] = {}
        self.skipped: list[SkippedEvent] = []

    def order(self, event: Event) -> tuple[_Order | None, str | None]:
        """The event's node-side order, or why its stamp makes it unusable."""
        stamp = _stamp_of(event.payload)
        if stamp is None:
            return None, None
        node_id = event.payload["node_id"]
        rank = self.claims.get(node_id, {}).get(stamp[0])
        if rank is None:
            return None, f"sent by a feed process the hub never admitted for node {node_id}"
        if self.owner[node_id] != stamp[0]:
            return None, f"sent by a feed process of node {node_id} that a newer one replaced"
        return (rank, stamp[1]), None

    def add(self, event: Event) -> None:
        problem = _PROBLEMS[event.type](event.payload)
        if problem is None and event.type in _STAMPED:
            problem = _stamp_problem(event.payload)
        problem = problem or _author_problem(event)
        order: _Order | None = None
        if problem is None and event.type in _STAMPED:
            order, problem = self.order(event)
        if problem is None:
            problem = self._apply(event, (order, event.seq))
        if problem is not None:
            self.skipped.append(SkippedEvent.of(event, problem))

    def _apply(self, event: Event, pos: _Pos) -> str | None:
        payload = event.payload
        node_id = payload["node_id"]
        if event.type == EVENT_FEED_CLAIMED:
            ranks = self.claims.setdefault(node_id, {})
            if payload["nonce"] in ranks:
                return "claims a feed process the hub already admitted"
            ranks[payload["nonce"]] = len(ranks)
            self.owner[node_id] = payload["nonce"]
        elif event.type == EVENT_FEED_SUPERSEDED:
            record = SupersededFeed(node_id, payload["nonce"], payload["owner"], payload["at"])
            self.superseded[node_id] = (record, pos)
        elif event.type == EVENT_FEED_GAP:
            gap = FeedGap(node_id, payload["dropped"], payload["at"])
            if node_id not in self.gaps or not _older(pos, self.gaps[node_id][1]):
                self.gaps[node_id] = (gap, pos)
        elif event.type == EVENT_LEASE_SNAPSHOT:
            held = frozenset(payload["lease_ids"])
            if node_id not in self.snapshots or not _older(pos, self.snapshots[node_id][1]):
                self.snapshots[node_id] = (held, pos)
        elif event.type == EVENT_PRESENCE:
            if node_id in self.presence_at and _older(pos, self.presence_at[node_id]):
                return None  # a late delivery of an older report: the newer one stands
            self.presence_at[node_id] = pos
            if payload["active"]:
                self.presence[node_id] = _presence_entry(payload, self.presence.get(node_id))
            else:
                self.presence.pop(node_id, None)
        else:
            key = (node_id, payload["lease_id"])
            if event.type in _HELD:
                if key in self.ended:
                    return "the lease had already ended; a late grant or renewal is ignored"
                if key in self.leases and _older(pos, self.leases[key][1]):
                    return None  # a late delivery of an older state: the newer one stands
                self.leases[key] = (_board_lease(payload), pos)
            else:
                self.leases.pop(key, None)
                self.ended.add(key)
        return None

    def released(self, lease: BoardLease, pos: _Pos) -> bool:
        """Whether the node's latest snapshot says it no longer holds ``lease``.

        A lease last granted or renewed before the snapshot (in the node's own order) and
        not listed in it is no longer held: its release was dropped, or a previous process
        of the node held it. One granted after the snapshot stays, even if the snapshot
        reached the hub later (a timed-out delivery that committed late).
        """
        snapshot = self.snapshots.get(lease.node_id)
        return (
            snapshot is not None and lease.lease_id not in snapshot[0] and _older(pos, snapshot[1])
        )

    def cleared(self, node_id: str, pos: _Pos) -> bool:
        """Whether a snapshot from ``node_id`` came after ``pos`` (so it reconciled it)."""
        snapshot = self.snapshots.get(node_id)
        return snapshot is not None and _older(pos, snapshot[1])


def lease_board(log: EventLog, *, clock: Clock = utcnow) -> LeaseBoard:
    """Project who holds what, and which humans are active, as of ``clock()``.

    This is the one reconciled view every reader uses (the console's leases page, the MCP
    ``lease_board`` and ``project_board`` tools): a node's leases are checked against its
    latest ``lease.snapshot``, so a lease whose release was lost is not shown, and
    ``stale`` says when a node's feed has dropped events (or a second feed process for one
    node id was refused) and has not reconciled since.

    Total: a malformed event, one not written by the node it describes, one sent by a feed
    process the hub did not admit or has since replaced, or a grant for a lease that
    already ended (a late retry) is skipped and listed, never raised. A late delivery of an
    older state (by the node's own order) never overwrites a newer one. Actors are
    self-declared (no auth in v1), so the author check stops mislabelled events, not a
    hostile writer.
    """
    now = clock()
    fold = _Fold()
    for event in log.iter_events(types=_BOARD_TYPES):
        fold.add(event)
    held = sorted(
        (
            lease
            for lease, pos in fold.leases.values()
            if not _expired(lease.expires_at, now) and not fold.released(lease, pos)
        ),
        key=lambda lease: lease.sort_key,
    )
    present = sorted(
        (entry for entry in fold.presence.values() if not _expired(entry.expires_at, now)),
        key=lambda entry: (entry.since, entry.node_id),
    )
    gaps = sorted(
        (gap for node, (gap, pos) in fold.gaps.items() if not fold.cleared(node, pos)),
        key=lambda gap: gap.node_id,
    )
    refused = sorted(
        (rec for node, (rec, pos) in fold.superseded.items() if not fold.cleared(node, pos)),
        key=lambda rec: rec.node_id,
    )
    return LeaseBoard(
        tuple(held), tuple(present), iso(now), tuple(fold.skipped), tuple(gaps), tuple(refused)
    )


# -- the hub side of a node's feed ------------------------------------------------------------


def _claimed(txn: Transaction, node_id: str) -> list[str]:
    """The feed nonces the hub has admitted for ``node_id``, oldest first."""
    nonces: list[str] = []
    for event in txn.read(types=[EVENT_FEED_CLAIMED]):
        if (
            _claim_problem(event.payload) is None
            and _author_problem(event) is None
            and event.payload["node_id"] == node_id
            and event.payload["nonce"] not in nonces
        ):
            nonces.append(event.payload["nonce"])
    return nonces


def deliver_feed_event(
    log: EventLog, spec: HubEventSpec, node_id: str, *, clock: Clock = utcnow
) -> Event:
    """Append one event a node's feed sent, as node ``node_id``: the hub side of a feed.

    A spec stamped with a feed nonce (every spec a :class:`~aar.node.feed.NodeFeed`
    sends) is checked against the feed processes the hub has admitted for that node id:

    - a resend of a spec already in the log (same idempotency key) is answered with the
      stored event, so a lost ack never adds a second one;
    - the node's current process is accepted;
    - a nonce the hub has never seen is a new process of the node (a restart): it is
      admitted with a ``feed.claimed`` event and replaces the previous one;
    - a nonce the hub admitted before but has since replaced is refused with
      :class:`FeedConflict`, and the refusal is recorded once (``feed.superseded``), which
      marks the node stale on the board. Two live processes with one node id therefore
      never take turns erasing each other's leases with their snapshots: the older one is
      refused, loudly, from the moment the newer one first speaks.

    The check and the append share one transaction. A remote transport (F-S11-1) calls this
    on the hub and must hand :class:`FeedConflict` back to the node as itself.
    """
    actor = Actor("node", node_id)
    problem = _stamp_problem(spec.payload)
    if problem is not None:
        raise ValueError(problem)
    stamp = _stamp_of(spec.payload)
    if stamp is None:
        return log.append(spec.type, actor, spec.payload, idempotency_key=spec.idempotency_key)
    nonce = stamp[0]
    refusal: FeedConflict | None = None
    with log.transaction() as txn:
        key = spec.idempotency_key
        if key is not None and txn.find_by_idempotency_key(key) is not None:
            return txn.append(spec.type, actor, spec.payload, idempotency_key=key)
        admitted = _claimed(txn, node_id)
        owner = admitted[-1] if admitted else None
        if nonce in admitted and nonce != owner:
            record_key = f"{EVENT_FEED_SUPERSEDED}:{node_id}:{nonce}"
            if txn.find_by_idempotency_key(record_key) is None:
                txn.append(
                    EVENT_FEED_SUPERSEDED,
                    actor,
                    {"node_id": node_id, "nonce": nonce, "owner": owner, "at": iso(clock())},
                    idempotency_key=record_key,
                )
            refusal = FeedConflict(
                f"node {node_id!r} has a newer feed process ({owner}); this one ({nonce}) was "
                "replaced and may send nothing more. Two live processes must not share a "
                "node id: stop this one."
            )
        else:
            if nonce not in admitted:
                txn.append(
                    EVENT_FEED_CLAIMED,
                    actor,
                    {"node_id": node_id, "nonce": nonce, "replaces": owner, "at": iso(clock())},
                    idempotency_key=f"{EVENT_FEED_CLAIMED}:{node_id}:{nonce}",
                )
            event = txn.append(spec.type, actor, spec.payload, idempotency_key=key)
    if refusal is not None:
        raise refusal
    return event
