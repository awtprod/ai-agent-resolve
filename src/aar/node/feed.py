"""Node-to-hub feed: a node's lease and presence changes, written to the hub log.

The broker's lease table (:mod:`aar.node.leases`) emits a :class:`LeaseEvent` on every
grant, renewal, release, expiry and preemption; the presence guard
(:mod:`aar.node.presence`) tracks whether a human is working. This feed turns both into
hub events in exactly the shapes the console's lease board projects
(:mod:`aar.hub.leaseboard`).

It must never raise into, nor block, the LeaseManager or the broker. So the sink only
*maps and enqueues*; sending happens in :meth:`NodeFeed.flush`, which the node's pump
calls between ops. A mapped spec is stored once (the wall offset is read at map time,
not resend time), so a retried send replays the same spec and the hub dedupes it on its
idempotency key. A send that fails leaves its spec queued for the next flush. The outbox
is bounded: on overflow the oldest spec is dropped, counted, and a ``feed.gap`` event is
sent on the next successful flush so the board can show the node's history may be stale.
Pending renewals for one lease are coalesced to the latest; expiries emitted while the
lease table is read go through the same path. A spec the hub keeps refusing is
quarantined (counted as dropped and gap-flagged) rather than left to block the queue.

After a gap, and on a new process's first flush, the feed reconciles: it re-announces its
live leases and sends a ``lease.snapshot`` of their ids, so the hub's project board drops a
lease whose release was lost (or that an earlier process held) and clears the stale flag.
Every key the feed mints itself carries a per-process nonce, so a restarted node never
collides with the keys of an earlier one.

Bad input that the hub could not map is refused *before* a lease is granted
(:meth:`aar.node.leases.LeaseManager.acquire`), so the feed never has to drop a lease
event it cannot map; if one still cannot be mapped it is dropped and gap-flagged, never
raised.

Delivery is behind :class:`FeedTransport`. In process, :class:`EventLogTransport` appends
to a hub :class:`~aar.hub.events.EventLog`; a remote node would implement the same
one-method protocol over HTTP (follow-up F-S11-1). Lease and presence only; pulling jobs
and approval tokens are S7's ``aar.node.hublink``.
"""

from __future__ import annotations

import secrets
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

from aar.hub.collab import EVENT_FEED_GAP, EVENT_LEASE_SNAPSHOT, MAX_SNAPSHOT_LEASES
from aar.hub.common import InvalidInput, clean_text, iso
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import (
    MAX_NODE_ID,
    MAX_PRESENCE_TTL,
    HubEventSpec,
    current_wall_offset,
    lease_event_to_hub,
    presence_to_hub,
)
from aar.node.leases import Lease, LeaseEvent
from aar.node.presence import PresenceGuard

DEFAULT_PRESENCE_TTL = timedelta(minutes=10)
"""How long an ``active`` presence report stays on the board without a fresh one."""

DEFAULT_REFRESH_INTERVAL = 120.0
"""Re-report an ongoing presence this often (seconds); must be shorter than the TTL."""

DEFAULT_OUTBOX_MAX = 4096
"""Specs held before the oldest is dropped (and the gap recorded)."""

# PresenceGuard.may_switch codes that mean a human is positively here.
_HUMAN_PRESENT_CODES = frozenset({"human_context_change", "human_input"})
_RENEWED = "lease.renewed"


class FeedTransport(Protocol):
    """Where a mapped hub event goes.

    In process this appends to an :class:`~aar.hub.events.EventLog`; a remote transport
    would POST to the hub. It must be idempotent on ``spec.idempotency_key``: delivering
    the same spec twice must not add a second event. On failure it raises: a
    ``ConnectionError``, ``TimeoutError`` or other ``OSError`` when the hub cannot be
    reached (the feed keeps the spec and retries it), a ``ValueError`` when the hub refuses
    the spec (the feed quarantines it), anything else for a failure that may pass (retried
    up to ``MAX_SPEC_ATTEMPTS`` times, then quarantined).
    """

    def deliver(self, spec: HubEventSpec, *, node_id: str) -> None: ...


class EventLogTransport:
    """A :class:`FeedTransport` that appends to a local hub event log, as the node."""

    def __init__(self, log: EventLog) -> None:
        self._log = log

    def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
        self._log.append(
            spec.type,
            Actor("node", node_id),
            spec.payload,
            idempotency_key=spec.idempotency_key,
        )


@dataclass(frozen=True, slots=True)
class FlushReport:
    """The outcome of one :meth:`NodeFeed.flush`."""

    sent: int
    pending: int
    dropped: int
    delivered_gap: bool
    error: str | None = None


def _clean_node_id(node_id: str) -> str:
    """Validate a node id with the hub's own rules; reject anything not already clean."""
    try:
        cleaned = clean_text(node_id, "node_id", max_len=MAX_NODE_ID)
    except InvalidInput as exc:
        raise ValueError(str(exc)) from exc
    if cleaned != node_id:
        raise ValueError(
            f"node_id must be clean (no surrounding whitespace), so the transport actor and "
            f"the payload id are identical: {node_id!r}"
        )
    return cleaned


# Delivery failures, by what the feed does about them (see :meth:`NodeFeed.flush`).
# Unreachable: the hub is down or the link is; retried every flush until it comes back.
_UNREACHABLE = (ConnectionError, TimeoutError, OSError)
# Rejected: the hub refused this spec (a payload it will not take, or a reused idempotency
# key). It can never succeed, so it is quarantined at once rather than block the queue.
_REJECTED = (ValueError, TypeError)

MAX_SPEC_ATTEMPTS = 3
"""Tries before a spec that fails in any other way (neither unreachable nor rejected, e.g.
a locked database) is quarantined, so no spec blocks the queue forever."""

QUARANTINE_MEMORY = 64
"""Quarantined specs remembered for inspection (oldest forgotten first); all are counted."""


@dataclass(frozen=True, slots=True)
class Quarantined:
    """A spec the feed gave up on, and why. It counts as dropped and is gap-flagged."""

    type: str
    idempotency_key: str | None
    error: str


class NodeFeed:
    """Maps a node's lease and presence changes and sends them to the hub on ``flush``.

    ``lease_source`` and ``clock`` (the lease table's ``active`` and its monotonic clock)
    let the feed reconcile the board after it lost events: on its first flush, and after
    any gap, it re-announces every live lease and then sends a ``lease.snapshot`` of their
    ids, so the board picks up a grant it missed, drops a lease whose release it missed
    (or that a previous process of this node held), and clears the node's stale flag.
    """

    def __init__(
        self,
        node_id: str,
        transport: FeedTransport,
        *,
        wall_offset: Callable[[], float] = current_wall_offset,
        outbox_max: int = DEFAULT_OUTBOX_MAX,
        lease_source: Callable[[], Iterable[Lease]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if outbox_max < 1:
            raise ValueError("outbox_max must be at least 1")
        if (lease_source is None) != (clock is None):
            raise ValueError("lease_source and clock go together (the lease table and its clock)")
        self._node_id = _clean_node_id(node_id)
        self._transport = transport
        self._wall_offset = wall_offset
        self._outbox_max = outbox_max
        self._lease_source = lease_source
        self._clock = clock
        # A per-process nonce in every key the feed mints itself (gaps, snapshots, lease
        # re-announcements), so a restarted node never reuses a key an earlier process of
        # the same node already used for a different event (which the hub would refuse on
        # every flush from then on).
        self._epoch = secrets.token_hex(8)
        self._minted = 0
        self._outbox: list[HubEventSpec] = []
        self._failing: HubEventSpec | None = None  # the head spec, while it keeps failing
        self._failures = 0
        self._dropped = 0
        self._gap_pending = False
        self._gap_at: datetime | None = None
        # A new process knows nothing of what an earlier one left on the board: reconcile.
        self._reconcile_pending = lease_source is not None
        self._quarantined: deque[Quarantined] = deque(maxlen=QUARANTINE_MEMORY)

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def pending(self) -> int:
        return len(self._outbox)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def quarantined(self) -> tuple[Quarantined, ...]:
        """The most recent specs the feed gave up on (up to ``QUARANTINE_MEMORY``)."""
        return tuple(self._quarantined)

    @property
    def reconciled(self) -> bool:
        """Whether the board has this node's current leases (no reconcile is pending)."""
        return not self._reconcile_pending

    # -- enqueue (never raises into the LeaseManager or the broker) ------------------------

    def lease_sink(self, event: LeaseEvent) -> None:
        """Map and enqueue one lease event. Pass as ``LeaseManager(sink=feed.lease_sink)``.

        The wall offset is read once, now, and baked into the mapped spec, so a later
        resend never re-maps against a drifted clock. An event the hub could not map is
        dropped and gap-flagged, never raised.
        """
        try:
            spec = lease_event_to_hub(event, self._node_id, wall_offset=self._wall_offset())
        except InvalidInput:
            self._note_gap()  # unmappable despite acquire-time validation: lose it, flag staleness
            return
        self._enqueue(spec)

    def report_presence(
        self,
        *,
        active: bool,
        at: datetime,
        human_id: str | None = None,
        ttl: timedelta | None = None,
    ) -> None:
        """Map and enqueue a presence change. ``at`` must be timezone-aware.

        A change the hub could not map is dropped and gap-flagged, never raised, like a
        lease event.
        """
        if at.tzinfo is None:
            raise ValueError("at must be timezone-aware")  # a caller bug, not a feed failure
        try:
            spec = presence_to_hub(self._node_id, active=active, at=at, human_id=human_id, ttl=ttl)
        except InvalidInput:
            self._note_gap()
            return
        state = "active" if active else "inactive"
        key = f"presence:{self._node_id}:{state}:{at.astimezone(UTC).isoformat()}"
        self._enqueue(replace(spec, idempotency_key=key))

    def _enqueue(self, spec: HubEventSpec) -> None:
        if spec.type == _RENEWED:
            lease_id = spec.payload.get("lease_id")
            self._outbox = [
                queued
                for queued in self._outbox
                if not (queued.type == _RENEWED and queued.payload.get("lease_id") == lease_id)
            ]
        self._outbox.append(spec)
        while len(self._outbox) > self._outbox_max:
            self._outbox.pop(0)
            self._note_gap()

    def _note_gap(self) -> None:
        self._dropped += 1
        self._gap_pending = True
        self._reconcile_pending = self._lease_source is not None
        if self._gap_at is None:
            self._gap_at = datetime.now(UTC)

    # -- flush (called by the pump; never raises into it) ----------------------------------

    def flush(self) -> FlushReport:
        """Send the pending gap, then reconcile, then the queued specs, in order.

        Never raises. Failing to reach the hub stops the flush and keeps everything for the
        next one. Any other failure never blocks what follows: a gap or reconcile that fails
        is retried on the next flush, and a queued spec the hub rejects is quarantined at
        once (one that fails any other way, after ``MAX_SPEC_ATTEMPTS`` tries), counted as
        dropped and gap-flagged, so a poison spec never blocks the specs behind it.
        """
        delivered_gap = False
        problems: list[str] = []
        if self._gap_pending:
            try:
                self._transport.deliver(self._gap_spec(), node_id=self._node_id)
            except _UNREACHABLE as exc:  # re-sent (same key) on the next flush
                return self._report(0, delivered_gap, repr(exc))
            except Exception as exc:
                problems.append(f"gap: {exc!r}")
            else:
                self._gap_pending = False
                self._gap_at = None
                delivered_gap = True
                # The stale flag this gap raises is cleared only by a later snapshot.
                self._reconcile_pending = self._lease_source is not None
        if self._reconcile_pending:
            try:
                self._reconcile()
            except _UNREACHABLE as exc:  # reconciled again, from scratch, on the next flush
                return self._report(0, delivered_gap, f"reconcile: {exc!r}")
            except Exception as exc:
                problems.append(f"reconcile: {exc!r}")
            else:
                self._reconcile_pending = False
        sent = 0
        while self._outbox:
            spec = self._outbox[0]
            if spec is not self._failing:
                self._failing, self._failures = spec, 0
            try:
                self._transport.deliver(spec, node_id=self._node_id)
            except _UNREACHABLE as exc:
                problems.append(repr(exc))
                break
            except Exception as exc:
                self._failures += 1
                if not isinstance(exc, _REJECTED) and self._failures < MAX_SPEC_ATTEMPTS:
                    problems.append(repr(exc))
                    break
                self._outbox.pop(0)
                self._quarantined.append(Quarantined(spec.type, spec.idempotency_key, repr(exc)))
                self._note_gap()
                problems.append(f"quarantined {spec.type}: {exc!r}")
                continue
            self._outbox.pop(0)
            sent += 1
        return self._report(sent, delivered_gap, "; ".join(problems) or None)

    def _report(self, sent: int, delivered_gap: bool, error: str | None) -> FlushReport:
        return FlushReport(sent, len(self._outbox), self._dropped, delivered_gap, error)

    def _mint_key(self, kind: str) -> str:
        self._minted += 1
        return f"{kind}:{self._node_id}:{self._epoch}:{self._minted}"

    def _gap_spec(self) -> HubEventSpec:
        # Keyed by this process's epoch and the running drop count: a resend of the same
        # pending gap replays the same key and payload, and a later gap (more drops) or a
        # later process never reuses it.
        at = self._gap_at or datetime.now(UTC)
        return HubEventSpec(
            EVENT_FEED_GAP,
            {"node_id": self._node_id, "dropped": self._dropped, "at": iso(at)},
            f"{EVENT_FEED_GAP}:{self._node_id}:{self._epoch}:{self._dropped}",
        )

    def _reconcile(self) -> None:
        """Re-announce every live lease, then send a snapshot of their ids. Raises on failure.

        The re-announcements are ``lease.renewed`` events the lease board already folds, so
        a lease whose grant was dropped reappears; the snapshot lets the hub drop a lease
        this node no longer holds. Both are built fresh (fresh keys) on every attempt.
        """
        assert self._lease_source is not None and self._clock is not None
        live = list(self._lease_source())
        if len(live) > MAX_SNAPSHOT_LEASES:
            raise ValueError(
                f"{len(live)} live leases is over the snapshot cap of {MAX_SNAPSHOT_LEASES}"
            )
        now = self._clock()
        offset = self._wall_offset()
        for lease in live:
            spec = lease_event_to_hub(
                LeaseEvent("renewed", now, lease), self._node_id, wall_offset=offset
            )
            spec = replace(spec, idempotency_key=self._mint_key("lease.reannounced"))
            self._transport.deliver(spec, node_id=self._node_id)
        snapshot = HubEventSpec(
            EVENT_LEASE_SNAPSHOT,
            {
                "node_id": self._node_id,
                "at": iso(datetime.fromtimestamp(now + offset, UTC)),
                "lease_ids": [lease.id for lease in live],
            },
            self._mint_key(EVENT_LEASE_SNAPSHOT),
        )
        self._transport.deliver(snapshot, node_id=self._node_id)


class PresenceReporter:
    """Turns a :class:`PresenceGuard`'s verdict into hub presence events.

    Call :meth:`sync` from the node's pump after the guard has observed the latest
    context. It reports ``active`` when the guard sees a human, ``inactive`` once the node
    has been quiet for a whole idle window, and re-reports an ongoing ``active`` every
    ``refresh_interval`` so the board's TTL never lapses. While the verdict is unknown
    (warm-up or a stale observation) it emits nothing and holds the last report.
    """

    def __init__(
        self,
        guard: PresenceGuard,
        feed: NodeFeed,
        *,
        human_id: str | None = None,
        ttl: timedelta = DEFAULT_PRESENCE_TTL,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("ttl must be positive so a dead node's presence drops off")
        if ttl > MAX_PRESENCE_TTL:
            raise ValueError(f"ttl must be at most {MAX_PRESENCE_TTL} (the hub's cap)")
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if refresh_interval >= ttl.total_seconds():
            raise ValueError(
                "refresh_interval must be shorter than the ttl, else a present human lapses "
                "off the board between refreshes"
            )
        if human_id is not None:
            try:
                clean_text(human_id, "human_id", max_len=MAX_NODE_ID)
            except InvalidInput as exc:
                raise ValueError(str(exc)) from exc
        self._guard = guard
        self._feed = feed
        self._human_id = human_id
        self._ttl = ttl
        self._refresh_interval = refresh_interval
        self._active: bool | None = None
        self._reported_at: float | None = None

    def sync(self, now: float, wall_now: datetime) -> bool:
        """Reconcile the guard's verdict with the hub. Returns whether it emitted an event."""
        verdict = self._verdict(now)
        if verdict is None:
            return False
        if verdict:
            due = self._reported_at is None or now - self._reported_at >= self._refresh_interval
            if self._active and not due:
                return False
            self._feed.report_presence(
                active=True, at=wall_now, human_id=self._human_id, ttl=self._ttl
            )
            self._active, self._reported_at = True, now
            return True
        if not self._active:  # never reported active, or already absent: nothing to clear
            self._active = False
            return False
        self._feed.report_presence(active=False, at=wall_now, human_id=self._human_id)
        self._active, self._reported_at = False, now
        return True

    def _verdict(self, now: float) -> bool | None:
        """``True`` human present, ``False`` quiet, ``None`` not yet known."""
        decision = self._guard.may_switch(now)
        if decision.allowed:  # "idle": quiet for a full window
            return False
        if decision.code in _HUMAN_PRESENT_CODES:
            return True
        return None  # warming up, or no/stale observation
