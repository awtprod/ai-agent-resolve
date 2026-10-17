"""Node-to-hub feed: a node's lease and presence changes, written to the hub log.

The broker's lease table (:mod:`aar.node.leases`) emits a :class:`LeaseEvent` on every
grant, renewal, release, expiry and preemption; the presence guard
(:mod:`aar.node.presence`) tracks whether a human is working. This module turns both
into hub events in exactly the shapes the console's lease board projects
(:mod:`aar.hub.leaseboard`), so a supervisor sees who holds what and where a human is
active.

The event shapes and their idempotency keys come from
:func:`aar.hub.leaseboard.lease_event_to_hub` and ``presence_to_hub``, which are pure.
A lease event already carries a per-(node, lease, kind, time) key; this feed adds the
matching key for a presence report. So a retried event is deduped by the hub rather
than double-counted.

Delivery is behind :class:`FeedTransport`. In process, :class:`EventLogTransport`
appends straight to a hub :class:`~aar.hub.events.EventLog`. A remote node would
implement the same one-method protocol over HTTP and nothing else would change
(follow-up F-S11-1).

Scope: lease and presence only. Pulling jobs and human approval tokens are S7's
``aar.node.hublink``; this feed stays separate from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import (
    HubEventSpec,
    current_wall_offset,
    lease_event_to_hub,
    presence_to_hub,
)
from aar.node.leases import LeaseEvent
from aar.node.presence import PresenceGuard

DEFAULT_PRESENCE_TTL = timedelta(minutes=10)
"""How long an ``active`` presence report stays on the board without a fresh one, so a
node that dies does not leave a human "present" forever."""

DEFAULT_REFRESH_INTERVAL = 120.0
"""Re-report an ongoing presence this often (seconds) so its board TTL never lapses."""

# PresenceGuard.may_switch codes that mean a human is *positively* here (not just that
# the broker cannot act yet, as in warm-up or a stale observation).
_HUMAN_PRESENT_CODES = frozenset({"human_context_change", "human_input"})


class FeedTransport(Protocol):
    """Where a mapped hub event goes.

    In process this appends to an :class:`~aar.hub.events.EventLog`; a remote transport
    would POST to the hub. An implementation must be idempotent on
    ``spec.idempotency_key``: delivering the same spec twice must not add a second event.
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


class NodeFeed:
    """Maps a node's lease and presence changes and hands them to a transport."""

    def __init__(
        self,
        node_id: str,
        transport: FeedTransport,
        *,
        wall_offset: Callable[[], float] = current_wall_offset,
    ) -> None:
        if not node_id:
            raise ValueError("node_id must be non-empty")
        self._node_id = node_id
        self._transport = transport
        self._wall_offset = wall_offset

    @property
    def node_id(self) -> str:
        return self._node_id

    def lease_sink(self, event: LeaseEvent) -> None:
        """Deliver one lease event. Wire it in as ``LeaseManager(sink=feed.lease_sink)``.

        The node clock is monotonic, so the wall offset is read now and the mapping
        records wall-clock times (see :func:`aar.hub.leaseboard.lease_event_to_hub`).
        """
        spec = lease_event_to_hub(event, self._node_id, wall_offset=self._wall_offset())
        self._transport.deliver(spec, node_id=self._node_id)

    def report_presence(
        self,
        *,
        active: bool,
        at: datetime,
        human_id: str | None = None,
        ttl: timedelta | None = None,
    ) -> None:
        """Report that a human is (``active``) or is no longer working on this node.

        ``at`` is wall-clock: for ``active`` it is the board's "since", and ``ttl`` bounds
        how long it stays without a refresh. The report is keyed on
        ``(node, active, at)`` so a retry of the *same* report is deduped while a later
        refresh (a new ``at``) is a genuinely new event.
        """
        spec = presence_to_hub(self._node_id, active=active, at=at, human_id=human_id, ttl=ttl)
        state = "active" if active else "inactive"
        key = f"presence:{self._node_id}:{state}:{at.astimezone(UTC).isoformat()}"
        self._transport.deliver(replace(spec, idempotency_key=key), node_id=self._node_id)


class PresenceReporter:
    """Turns a :class:`PresenceGuard`'s verdict into hub presence events.

    Call :meth:`sync` from the node's poll loop after the guard has observed the latest
    context. It reports ``active`` when the guard sees a human (a context change it did
    not cause, or recent input), ``inactive`` once the node has been quiet for a whole
    idle window, and re-reports an ongoing ``active`` every ``refresh_interval`` so the
    board's TTL never lapses while a human is still here. While the verdict is unknown
    (warm-up, or no/stale observation) it emits nothing and holds the last report, so an
    unobserved node is never reported as either present or absent.
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
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
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
        if not self._active:  # never reported active, or already reported absent: nothing to clear
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
