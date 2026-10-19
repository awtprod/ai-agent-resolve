"""One node's runtime: the broker, lease table, feed, presence, branch registry and job runner,
wired together exactly once per node id (the S7 x S11 integration, policy P5).

The wiring, in one place so nothing is left to a default by accident:

- one :class:`~aar.node.branch_registry.BranchRegistry` per node, handed to the job runner,
  which passes it to every ``make_hot_op`` it submits (every op of a job runs as the job's
  own agent, ``Holder.agent(<agent id>)``, P2);
- the :class:`~aar.node.leases.LeaseManager` and :class:`~aar.node.feed.NodeFeed` built by
  :func:`aar.node.feed.wire_feed`, bound late so each sees the other: the feed's
  ``lease_source`` is the table's ``active`` and both run on the runtime's monotonic clock;
- one :class:`~aar.node.feed.PresenceReporter` over the broker's presence guard;
- the pump (:meth:`aar.node.jobs.JobRunner.pump`) calls ``PresenceReporter.sync`` and then
  ``NodeFeed.flush`` on every tick, and records each ``timeline.branch`` result on the hub
  (:func:`aar.hub.collab.record_branch`) under the hub's ``prj_`` id, as
  ``Actor("node", <the feed's node id>)``;
- exactly one live feed per node id in a process: a second :class:`NodeRuntime` for a node id
  that is still open is refused (:class:`NodeIdInUse`); across processes the hub refuses the
  feed it replaced (:class:`~aar.hub.leaseboard.FeedConflict`) and the job store's
  single-writer lock refuses a second runner on the same store.

``NodeRuntime.local`` builds the in-process form (hub on the same host, over its event log).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from aar.hub.events import EventLog
from aar.node.branch_registry import BranchRegistry
from aar.node.broker import Broker, BrokerConfig
from aar.node.feed import EventLogTransport, FeedTransport, PresenceReporter, wire_feed
from aar.node.gateway import ResolveGateway
from aar.node.hublink import HubLink, LocalHubLink
from aar.node.jobs import ApprovalPolicy, JobRunner, JobStore, PumpOutcome, Watchdog
from aar.node.jobs.runner import _DRAIN_BUDGET
from aar.node.presence import PresenceGuard

DEFAULT_IDLE_WINDOW = 60.0
"""Seconds of quiet before the presence guard calls a human gone (the broker's default)."""


class NodeIdInUse(RuntimeError):
    """A runtime for this node id is already open in this process (one live feed per node)."""


_LIVE: dict[str, NodeRuntime] = {}
_LIVE_LOCK = threading.Lock()


class NodeRuntime:
    """Builds and owns one node's broker, lease table, feed, presence and job runner."""

    def __init__(
        self,
        gateway: ResolveGateway,
        store: JobStore,
        hublink: HubLink,
        transport: FeedTransport,
        *,
        node_id: str,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        broker_config: BrokerConfig | None = None,
        idle_window: float = DEFAULT_IDLE_WINDOW,
        human_id: str | None = None,
        approval: ApprovalPolicy | None = None,
        watchdog: Watchdog | None = None,
        drain_budget: float | None = _DRAIN_BUDGET,
    ) -> None:
        with _LIVE_LOCK:
            if node_id in _LIVE:
                raise NodeIdInUse(
                    f"node {node_id!r} already has a live runtime in this process; close it "
                    "first (two live feeds for one node id would fight on the hub)"
                )
            _LIVE[node_id] = self  # reserved now; released if construction fails
        try:
            self.node_id = node_id
            self.clock = clock
            self.feed, self.leases = wire_feed(node_id, transport, clock=clock)
            self.presence = PresenceGuard(idle_window=idle_window)
            self.presence_reporter = PresenceReporter(self.presence, self.feed, human_id=human_id)
            self.branch_registry = BranchRegistry()
            self.broker = Broker(
                gateway,
                leases=self.leases,
                presence=self.presence,
                clock=clock,
                config=broker_config,
            )
            self.store = store
            self.hublink = hublink
            self.runner = JobRunner(
                self.broker,
                store,
                hublink,
                node_id=node_id,
                branch_registry=self.branch_registry,
                feed=self.feed,
                presence_reporter=self.presence_reporter,
                clock=clock,
                wall_clock=wall_clock,
                approval=approval,
                watchdog=watchdog,
                drain_budget=drain_budget,
            )
        except BaseException:
            self._release()
            raise
        self._closed = False

    @classmethod
    def local(
        cls,
        gateway: ResolveGateway,
        store: JobStore,
        log: EventLog,
        *,
        node_id: str,
        **kwargs: object,
    ) -> NodeRuntime:
        """A runtime whose hub is in process: the link and the feed both over ``log``."""
        return cls(
            gateway,
            store,
            LocalHubLink(log),
            EventLogTransport(log),
            node_id=node_id,
            **kwargs,  # type: ignore[arg-type]
        )

    def pump(self) -> PumpOutcome:
        return self.runner.pump()

    def close(self, timeout: float = 5.0) -> bool:
        """Stop the runner (no op it submitted runs any more) and free the node id. The job
        store stays open; its owner closes it. Returns whether the drain ended in time."""
        if self._closed:
            return True
        self._closed = True
        try:
            return self.runner.close(timeout)
        finally:
            self._release()

    def _release(self) -> None:
        with _LIVE_LOCK:
            if _LIVE.get(self.node_id) is self:
                del _LIVE[self.node_id]
