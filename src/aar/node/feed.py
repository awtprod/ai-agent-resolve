"""Node-to-hub feed: a node's lease and presence changes, written to the hub log.

The broker's lease table (:mod:`aar.node.leases`) emits a :class:`LeaseEvent` on every
grant, renewal, release, expiry and preemption; the presence guard
(:mod:`aar.node.presence`) tracks whether a human is working. This feed turns both into
hub events in exactly the shapes the console's lease board projects
(:mod:`aar.hub.leaseboard`).

It must never raise into, nor block, the LeaseManager or the broker. So the sink only
*maps, stamps and enqueues*; sending happens in :meth:`NodeFeed.flush`, which the node's
pump calls between ops. A mapped spec is stored once (the wall offset is read at map time,
not resend time), so a retried send replays the same spec and the hub dedupes it on its
idempotency key. Every spec is stamped ``feed: {nonce, seq}`` when it is made: the feed
process's random nonce and the feed's own running order, so the hub orders a node's events
as the node made them, whatever order they arrive in. The outbox is bounded: on overflow
the oldest spec is dropped, counted, and a ``feed.gap`` event is sent on the next flush so
the board can show the node's history may be stale. Pending renewals for one lease are
coalesced to the latest.

After a gap, and on a new process's first flush, the feed reconciles: it re-announces the
live leases the hub may have missed (those with no grant or renewal still queued) and the
latest presence report, then sends a ``lease.snapshot`` of every live lease id. That batch
is built once, with its keys and stamps, and resent unchanged until it is through, so a
lost ack never adds events and a snapshot that commits late never erases a lease granted
after it (the hub compares the stamps). The snapshot lets the hub drop a lease whose
release was lost (or that an earlier process held) and clear the stale flag.

Delivery and retries, for every way a send can fail (see :meth:`NodeFeed.flush`):

- **Unreachable** (the transport raises an ``OSError``: ``ConnectionError``,
  ``TimeoutError``, ...): nothing is ever dropped for it. The flush stops and keeps every
  spec, and the feed backs off before it tries again, ``BACKOFF_FIRST`` seconds of its clock
  doubling to at most ``BACKOFF_MAX`` (back to ``BACKOFF_FIRST`` whenever a flush got
  something through, so a lossy link still drains). A send that timed out may have
  committed; its resend carries the same key and the hub answers with the stored event.
- **Refused** (``ValueError`` or ``TypeError``): the spec can never succeed. A queued spec is
  quarantined at once (counted as dropped and gap-flagged), so it never blocks the specs
  behind it. A refused gap does not hold up the reconcile it asked for; it is tried again on
  later flushes, ``MAX_SPEC_ATTEMPTS`` times in all, then given up. A refused
  reconcile spec abandons that reconcile; a new one (new keys: it is a new reconcile, not a
  resend) is tried on the next flush, at most ``MAX_SPEC_ATTEMPTS`` times in all, then it is
  given up and the node stays honestly stale on the board until its next gap.
- **Anything else** (a locked database, say): the same spec is retried, with the backoff,
  ``MAX_SPEC_ATTEMPTS`` times in all, then treated as refused.
- **Replaced** (:class:`~aar.hub.leaseboard.FeedConflict`: the hub admitted a newer process
  with this node id): the feed stops for good, and every later flush says so at once. Its
  process must be stopped; two live processes must never share a node id.

Bad input that the hub could not map is refused *before* a lease is granted
(:meth:`aar.node.leases.LeaseManager.acquire`), so the feed never has to drop a lease
event it cannot map; if one still cannot be mapped it is dropped and gap-flagged, never
raised.

Delivery is behind :class:`FeedTransport`. In process, :class:`EventLogTransport` hands
each spec to the hub's :func:`~aar.hub.leaseboard.deliver_feed_event`; a remote node uses
:class:`aar.node.remote.RemoteFeedTransport`, the same one-method protocol over HTTP
(F-S11-1, implemented in S15a), which keeps the spec's idempotency key on every resend.
Lease and presence only; pulling jobs and approval tokens are S7's ``aar.node.hublink``.
"""

from __future__ import annotations

import secrets
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

from aar.hub.common import InvalidInput, clean_text, iso, utcnow
from aar.hub.events import EventLog
from aar.hub.leaseboard import (
    EVENT_FEED_GAP,
    EVENT_LEASE_SNAPSHOT,
    LEASE_EVENT_TYPES,
    MAX_NODE_ID,
    MAX_PRESENCE_TTL,
    MAX_SNAPSHOT_LEASES,
    FeedConflict,
    HubEventSpec,
    current_wall_offset,
    deliver_feed_event,
    lease_event_to_hub,
    presence_to_hub,
)
from aar.node.leases import Lease, LeaseEvent, LeaseManager
from aar.node.presence import PresenceGuard

DEFAULT_PRESENCE_TTL = timedelta(minutes=10)
"""How long an ``active`` presence report stays on the board without a fresh one."""

DEFAULT_REFRESH_INTERVAL = 120.0
"""Re-report an ongoing presence this often (seconds); must be shorter than the TTL."""

DEFAULT_OUTBOX_MAX = 4096
"""Specs held before the oldest is dropped (and the gap recorded)."""

BACKOFF_FIRST = 1.0
"""Seconds (of the feed's clock) to wait after a failed flush that got nothing through."""

BACKOFF_MAX = 60.0
"""The longest the feed ever waits between tries while the hub is unreachable."""

# PresenceGuard.may_switch codes that mean a human is positively here.
_HUMAN_PRESENT_CODES = frozenset({"human_context_change", "human_input"})
_RENEWED = LEASE_EVENT_TYPES["renewed"]
_HELD = frozenset({LEASE_EVENT_TYPES["granted"], _RENEWED})


class FeedTransport(Protocol):
    """Where a mapped hub event goes.

    In process this hands the spec to the hub's ``deliver_feed_event``; a remote transport
    would POST it to the hub, which runs the same. It must be idempotent on
    ``spec.idempotency_key``: delivering the same spec twice must not add a second event.
    On failure it raises: an ``OSError`` (``ConnectionError``, ``TimeoutError``, ...) when
    the hub cannot be reached or did not answer (the feed keeps the spec and backs off), a
    ``ValueError`` when the hub refuses the spec (the feed quarantines it), the hub's
    ``FeedConflict`` as itself (the feed stops), anything else for a failure that may pass
    (retried up to ``MAX_SPEC_ATTEMPTS`` times, then quarantined).
    """

    def deliver(self, spec: HubEventSpec, *, node_id: str) -> None: ...


class EventLogTransport:
    """A :class:`FeedTransport` into a local hub event log, through the hub's acceptor.

    ``clock`` is the hub's wall clock (it dates the hub's own records of the feed)."""

    def __init__(self, log: EventLog, *, clock: Callable[[], datetime] = utcnow) -> None:
        self._log = log
        self._clock = clock

    def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
        deliver_feed_event(self._log, spec, node_id, clock=self._clock)


@dataclass(frozen=True, slots=True)
class FlushReport:
    """The outcome of one :meth:`NodeFeed.flush`.

    ``retry_at`` (the feed's clock) is set while the feed backs off: a flush before it
    sends nothing. ``stopped`` is set once the hub has replaced this feed process.
    """

    sent: int
    pending: int
    dropped: int
    delivered_gap: bool
    error: str | None = None
    retry_at: float | None = None
    stopped: bool = False


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


# Delivery failures, by what the feed does about them (see the module docstring).
# Unreachable: the hub is down, or the link is, or the reply was lost; never dropped.
_UNREACHABLE = OSError  # ConnectionError and TimeoutError are OSErrors
# Refused: the hub will not take this spec (a payload it rejects, or a reused key).
_REFUSED = (ValueError, TypeError)

MAX_SPEC_ATTEMPTS = 3
"""Tries before a spec that fails in any other way (neither unreachable nor refused, e.g.
a locked database) is treated as refused, and the most reconciles tried after a gap."""

QUARANTINE_MEMORY = 64
"""Quarantined specs remembered for inspection (oldest forgotten first); all are counted."""


@dataclass(frozen=True, slots=True)
class Quarantined:
    """A spec the feed gave up on, and why. A queued one counts as dropped (gap-flagged)."""

    type: str
    idempotency_key: str | None
    error: str


class _Halt(Exception):
    """Ends a flush: the hub is unreachable, a spec should be retried later, or the hub
    replaced this feed process."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(repr(cause))
        self.cause = cause


class NodeFeed:
    """Maps a node's lease and presence changes and sends them to the hub on ``flush``.

    ``lease_source`` and ``clock`` are required: the lease table's ``active`` and its
    monotonic clock (bind them late, see :func:`wire_feed`). With them the feed reconciles
    the board after it lost events: on its first flush, and after any gap, it re-announces
    what the hub may have missed and then sends a ``lease.snapshot`` of every live lease
    id, so the board picks up a grant it missed, drops a lease whose release it missed (or
    that a previous process of this node held), and clears the node's stale flag. The clock
    also times the backoff.
    """

    def __init__(
        self,
        node_id: str,
        transport: FeedTransport,
        *,
        lease_source: Callable[[], Iterable[Lease]],
        clock: Callable[[], float],
        wall_offset: Callable[[], float] = current_wall_offset,
        outbox_max: int = DEFAULT_OUTBOX_MAX,
    ) -> None:
        if outbox_max < 1:
            raise ValueError("outbox_max must be at least 1")
        if not callable(lease_source) or not callable(clock):
            raise TypeError(
                "lease_source and clock are required: the lease table's active() and its "
                "clock, so the feed can reconcile after a gap"
            )
        self._node_id = _clean_node_id(node_id)
        self._transport = transport
        self._wall_offset = wall_offset
        self._outbox_max = outbox_max
        self._lease_source = lease_source
        self._clock = clock
        # This process's nonce, in every stamp and in every key the feed mints itself, so a
        # restarted node never reuses a key of an earlier process, and the hub can tell the
        # processes of one node id apart (it admits the newest, refuses the one it replaced).
        self._nonce = secrets.token_hex(8)
        self._seq = 0
        self._minted = 0
        self._outbox: list[HubEventSpec] = []
        self._batch: list[HubEventSpec] = []  # the reconcile in progress, resent unchanged
        self._head: HubEventSpec | None = None  # the spec that failed last ("other" errors)
        self._head_failures = 0
        self._dropped = 0
        self._gap_pending = False
        self._gap_at: datetime | None = None
        self._gap: HubEventSpec | None = None  # the pending gap's spec, resent unchanged
        self._gap_failures = 0
        # A new process knows nothing of what an earlier one left on the board: reconcile.
        self._reconcile_pending = True
        self._reconcile_failures = 0
        self._last_presence: HubEventSpec | None = None
        self._retry_at: float | None = None
        self._delay = 0.0
        self._stopped: str | None = None
        self._quarantined: deque[Quarantined] = deque(maxlen=QUARANTINE_MEMORY)

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def nonce(self) -> str:
        """This feed process's nonce, as stamped on everything it sends."""
        return self._nonce

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
        """Whether the board has this node's current leases (no gap or reconcile is due)."""
        return not (self._gap_pending or self._reconcile_pending or self._batch)

    @property
    def stopped(self) -> str | None:
        """Why the feed stopped for good (the hub replaced this process), or ``None``."""
        return self._stopped

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
        self._enqueue(self._stamp(spec))

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
        lease event. The latest change is also re-announced by every reconcile, so one
        the hub never got (quarantined, or dropped) cannot leave a human shown as present.
        """
        if at.tzinfo is None:
            raise ValueError("at must be timezone-aware")  # a caller bug, not a feed failure
        try:
            spec = presence_to_hub(self._node_id, active=active, at=at, human_id=human_id, ttl=ttl)
        except InvalidInput:
            self._note_gap()
            return
        state = "active" if active else "inactive"
        key = f"presence:{self._node_id}:{self._nonce}:{state}:{at.astimezone(UTC).isoformat()}"
        self._last_presence = replace(spec, idempotency_key=key)
        self._enqueue(self._stamp(self._last_presence))

    def _stamp(self, spec: HubEventSpec) -> HubEventSpec:
        """``spec`` stamped with this process's nonce and the next number in its order."""
        self._seq += 1
        stamp = {"nonce": self._nonce, "seq": self._seq}
        return replace(spec, payload={**spec.payload, "feed": stamp})

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
        self._gap = None  # a new count: the next gap spec says so (with a new key)
        self._reconcile_pending = True
        if self._gap_at is None:
            self._gap_at = datetime.now(UTC)

    # -- flush (called by the pump; never raises into it) ----------------------------------

    def flush(self) -> FlushReport:
        """Send the pending gap, then the reconcile, then the queued specs, in that order.

        Never raises. While backing off (see the module docstring) it sends nothing and
        says until when. Failing to reach the hub stops the flush, keeps everything and
        backs off. A refused spec never blocks what follows (see the module docstring).
        """
        if self._stopped is not None:
            return self._report(0, False, self._stopped)
        if self._retry_at is not None and self._clock() < self._retry_at:
            return self._report(0, False, "backing off: the hub was not reachable")
        counts = {"sent": 0}
        delivered_gap = False
        problems: list[str] = []
        try:
            delivered_gap = self._flush_gap(counts, problems)
            self._flush_reconcile(counts, problems)
            self._flush_outbox(counts, problems)
        except _Halt as halt:
            return self._halt(counts["sent"], delivered_gap, halt.cause)
        self._retry_at, self._delay = None, 0.0
        return self._report(counts["sent"], delivered_gap, "; ".join(problems) or None)

    def _flush_gap(self, counts: dict[str, int], problems: list[str]) -> bool:
        if not self._gap_pending:
            return False
        spec = self._gap_spec()
        refused = self._send(spec)
        if refused is not None:
            # The reconcile _note_gap asked for still runs, now. The same gap is tried again
            # on later flushes, MAX_SPEC_ATTEMPTS times in all, then given up.
            self._gap_failures += 1
            if self._gap_failures < MAX_SPEC_ATTEMPTS:
                problems.append(f"gap refused, will retry: {refused!r}")
                return False
            self._quarantined.append(Quarantined(spec.type, spec.idempotency_key, repr(refused)))
            problems.append(f"gap given up: {refused!r}")
        else:
            counts["sent"] += 1
            # The stale flag this gap raises is cleared only by a later snapshot.
            self._reconcile_pending = True
        self._gap_pending, self._gap, self._gap_at, self._gap_failures = False, None, None, 0
        return refused is None

    def _flush_reconcile(self, counts: dict[str, int], problems: list[str]) -> None:
        if self._reconcile_pending:
            # A new reconcile: any unfinished earlier one is abandoned (its snapshot, should
            # it still land, is older than this one's and clears nothing this one should).
            self._reconcile_pending = False
            try:
                self._batch = self._reconcile_batch()
            except ValueError as exc:  # over the snapshot cap: it can never succeed
                self._batch = []
                self._quarantined.append(Quarantined(EVENT_LEASE_SNAPSHOT, None, repr(exc)))
                problems.append(f"reconcile given up: {exc!r}")
        while self._batch:
            spec = self._batch[0]
            refused = self._send(spec)
            if refused is not None:
                self._batch = []
                self._reconcile_failures += 1
                if self._reconcile_failures < MAX_SPEC_ATTEMPTS:
                    self._reconcile_pending = True  # try a new reconcile on the next flush
                    problems.append(f"reconcile refused, will retry: {refused!r}")
                else:  # give up: the node's gap stays on the board (honestly stale)
                    self._reconcile_failures = 0
                    self._quarantined.append(
                        Quarantined(spec.type, spec.idempotency_key, repr(refused))
                    )
                    problems.append(f"reconcile given up: {refused!r}")
                return
            self._batch.pop(0)
            counts["sent"] += 1
            if spec.type == EVENT_LEASE_SNAPSHOT:
                self._reconcile_failures = 0

    def _flush_outbox(self, counts: dict[str, int], problems: list[str]) -> None:
        while self._outbox:
            spec = self._outbox[0]
            refused = self._send(spec)
            self._outbox.pop(0)
            if refused is not None:
                self._quarantined.append(
                    Quarantined(spec.type, spec.idempotency_key, repr(refused))
                )
                self._note_gap()
                problems.append(f"quarantined {spec.type}: {refused!r}")
                continue
            counts["sent"] += 1

    def _send(self, spec: HubEventSpec) -> Exception | None:
        """Deliver ``spec`` once. ``None`` if it went through, the refusal if it never will;
        raises :class:`_Halt` when the flush must stop (and keep ``spec``) for now."""
        try:
            self._transport.deliver(spec, node_id=self._node_id)
        except FeedConflict as exc:
            self._stopped = f"stopped: the hub replaced this feed process: {exc}"
            raise _Halt(exc) from exc
        except _UNREACHABLE as exc:
            raise _Halt(exc) from exc
        except _REFUSED as exc:
            self._head, self._head_failures = None, 0
            return exc
        except Exception as exc:
            if spec is not self._head:
                self._head, self._head_failures = spec, 0
            self._head_failures += 1
            if self._head_failures < MAX_SPEC_ATTEMPTS:
                raise _Halt(exc) from exc
            self._head, self._head_failures = None, 0
            return exc
        if spec is self._head:
            self._head, self._head_failures = None, 0
        return None

    def _halt(self, sent: int, delivered_gap: bool, cause: Exception) -> FlushReport:
        if self._stopped is not None:
            return self._report(sent, delivered_gap, self._stopped)
        # Back off: from the start again if this flush got anything through (the link
        # works, if lossily), else twice as long as last time, up to the cap.
        self._delay = (
            BACKOFF_FIRST if sent else min(BACKOFF_MAX, max(BACKOFF_FIRST, 2 * self._delay))
        )
        self._retry_at = self._clock() + self._delay
        return self._report(sent, delivered_gap, repr(cause))

    def _report(self, sent: int, delivered_gap: bool, error: str | None) -> FlushReport:
        return FlushReport(
            sent,
            len(self._outbox),
            self._dropped,
            delivered_gap,
            error,
            self._retry_at,
            self._stopped is not None,
        )

    def _mint_key(self, kind: str) -> str:
        self._minted += 1
        return f"{kind}:{self._node_id}:{self._nonce}:{self._minted}"

    def _gap_spec(self) -> HubEventSpec:
        # Built once per pending gap and resent unchanged (same key, payload and stamp); a
        # later drop makes a new one (the count, and so the key, changes).
        if self._gap is None:
            at = self._gap_at or datetime.now(UTC)
            self._gap = self._stamp(
                HubEventSpec(
                    EVENT_FEED_GAP,
                    {"node_id": self._node_id, "dropped": self._dropped, "at": iso(at)},
                    f"{EVENT_FEED_GAP}:{self._node_id}:{self._nonce}:{self._dropped}",
                )
            )
        return self._gap

    def _reconcile_batch(self) -> list[HubEventSpec]:
        """The reconcile, built once: re-announcements, then a snapshot of every live lease.

        A live lease is re-announced (as a ``lease.renewed`` event the board already folds)
        unless a grant or renewal of it is still queued, which will announce it anyway. The
        latest presence report is re-announced too, unless it is still queued. All are
        stamped before the snapshot, so the hub orders them before it; a lease granted
        after this is stamped after it.
        """
        live = list(self._lease_source())
        if len(live) > MAX_SNAPSHOT_LEASES:
            raise ValueError(
                f"{len(live)} live leases is over the snapshot cap of {MAX_SNAPSHOT_LEASES}"
            )
        now = self._clock()
        offset = self._wall_offset()
        queued = {spec.payload.get("lease_id") for spec in self._outbox if spec.type in _HELD}
        batch: list[HubEventSpec] = []
        for lease in live:
            if lease.id in queued:
                continue
            spec = lease_event_to_hub(
                LeaseEvent("renewed", now, lease), self._node_id, wall_offset=offset
            )
            batch.append(
                self._stamp(replace(spec, idempotency_key=self._mint_key("lease.reannounced")))
            )
        presence = self._last_presence
        if presence is not None and not any(
            spec.idempotency_key == presence.idempotency_key for spec in self._outbox
        ):
            key = self._mint_key("presence.reannounced")
            batch.append(self._stamp(replace(presence, idempotency_key=key)))
        snapshot = HubEventSpec(
            EVENT_LEASE_SNAPSHOT,
            {
                "node_id": self._node_id,
                "at": iso(datetime.fromtimestamp(now + offset, UTC)),
                "lease_ids": [lease.id for lease in live],
            },
            self._mint_key(EVENT_LEASE_SNAPSHOT),
        )
        batch.append(self._stamp(snapshot))
        return batch


def wire_feed(
    node_id: str,
    transport: FeedTransport,
    *,
    clock: Callable[[], float],
    wall_offset: Callable[[], float] = current_wall_offset,
    outbox_max: int = DEFAULT_OUTBOX_MAX,
) -> tuple[NodeFeed, LeaseManager]:
    """A node's lease table and its feed, bound to each other on one clock.

    Each needs the other at construction (the table's sink is the feed; the feed's
    ``lease_source`` is the table's ``active``), so the feed is bound late, the way a node
    runtime must wire them. Build one per node id.
    """
    table: list[LeaseManager] = []
    feed = NodeFeed(
        node_id,
        transport,
        lease_source=lambda: table[0].active(),
        clock=clock,
        wall_offset=wall_offset,
        outbox_max=outbox_max,
    )
    table.append(LeaseManager(clock, sink=feed.lease_sink))
    return feed, table[0]


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
