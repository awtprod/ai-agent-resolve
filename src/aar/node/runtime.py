"""One node's runtime: the broker, lease table, feed, presence, branch registry and job runner,
wired together exactly once per node id (the S7 x S11 integration, policy P5).

The wiring, in one place so nothing is left to a default by accident:

- one :class:`~aar.node.branch_registry.BranchRegistry` per node, handed to the job runner,
  which passes it to every ``make_hot_op`` it submits (every op of a job runs as the job's
  own agent, ``Holder.agent(<agent id>)``, P2);
- the :class:`~aar.node.leases.LeaseManager` and :class:`~aar.node.feed.NodeFeed` built by
  :func:`aar.node.feed.wire_feed`, bound late so each sees the other: the feed's
  ``lease_source`` is the table's ``active`` and both run on the runtime's monotonic clock;
- the broker's :class:`~aar.node.presence.PresenceGuard` consults the OS keyboard/mouse idle
  time (``GetLastInputInfo`` on Windows, :func:`~aar.node.presence.default_input_idle_provider`)
  unless the caller injects another provider; a provider that fails refuses (fail closed);
- one :class:`~aar.node.feed.PresenceReporter` over that guard;
- the pump (:meth:`aar.node.jobs.JobRunner.pump`) calls ``PresenceReporter.sync`` and then
  ``NodeFeed.flush`` on every tick -- at the end of the drain it runs, on the broker's thread,
  because the lease table, presence guard and feed are not thread-safe -- and records each
  ``timeline.branch`` result on the hub
  (:func:`aar.hub.collab.record_branch`) under the hub's ``prj_`` id, as
  ``Actor("node", <the feed's node id>)``;
- exactly one live feed per node id in a process: a second :class:`NodeRuntime` for a node id
  that is still open is refused (:class:`NodeIdInUse`); across processes the hub refuses the
  feed it replaced (:class:`~aar.hub.leaseboard.FeedConflict`) and the job store's
  single-writer lock refuses a second runner on the same store.

``NodeRuntime.local`` builds the in-process form (hub on the same host, over its event log).
"""

from __future__ import annotations

import os
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
from aar.node.locator import ProjectLocator
from aar.node.presence import InputIdleProvider, PresenceGuard, default_input_idle_provider
from aar.node.remote import (
    DEFAULT_RESPONSE_CAP,
    DEFAULT_TIMEOUT,
    RemoteConfig,
    RemoteFeedTransport,
    RemoteHubLink,
    _NodeClient,
)

DEFAULT_IDLE_WINDOW = 60.0
"""Seconds of quiet before the presence guard calls a human gone (the broker's default)."""


class _OsDefault:
    """Sentinel: use the OS input-idle provider (``None`` means: deliberately none)."""

    def __repr__(self) -> str:
        return "OS_DEFAULT"


OS_DEFAULT = _OsDefault()


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
        input_idle: InputIdleProvider | _OsDefault | None = OS_DEFAULT,
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
            # Resolved here, not at import, so the provider matches the platform it runs on.
            self.input_idle = (
                default_input_idle_provider() if isinstance(input_idle, _OsDefault) else input_idle
            )
            """Seconds since the last keyboard/mouse input (``None``: the OS cannot tell)."""
            self.presence = PresenceGuard(idle_window=idle_window, input_idle=self.input_idle)
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

    @classmethod
    def remote(
        cls,
        gateway: ResolveGateway,
        store: JobStore,
        *,
        node_id: str,
        hub_url: str,
        token_file: str | os.PathLike[str],
        connect_timeout: float = DEFAULT_TIMEOUT,
        read_timeout: float = DEFAULT_TIMEOUT,
        response_cap: int = DEFAULT_RESPONSE_CAP,
        **kwargs: object,
    ) -> NodeRuntime:
        """A runtime whose hub is on another host, reached over HTTP (``aar hub nodeapi``).

        Same wiring as :meth:`local`, but the link and the feed go through
        :class:`~aar.node.remote.RemoteHubLink` and
        :class:`~aar.node.remote.RemoteFeedTransport` over one stdlib HTTP client. The gateway is
        injected, as it is for ``local``. Does not add ``aar node run`` (that is S15b).
        """
        config = RemoteConfig(
            hub_url=hub_url,
            token_file=token_file,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            response_cap=response_cap,
        )
        client = _NodeClient(config)
        return cls(
            gateway,
            store,
            RemoteHubLink(client),
            RemoteFeedTransport(client),
            node_id=node_id,
            **kwargs,  # type: ignore[arg-type]
        )

    def pump(self) -> PumpOutcome:
        return self.runner.pump()

    def rebuild_branches(self, projects: tuple[ProjectLocator, ...]) -> int:
        """Replace stale in-memory ownership with authenticated hub records.

        This is called before a remote worker pumps any local job, both on process startup
        and after reconnect. A failure leaves the new registry empty (fail closed); callers
        must not pump until the rebuild succeeds.
        """
        current = self.branch_registry
        # Hot ops already queued by the runner hold this exact registry object. Clear it in
        # place so reconnect cannot leave such a closure pointing at stale ownership.
        current._owners.clear()
        fresh = BranchRegistry()
        count = 0
        for project in projects:
            if project.project_id is None:
                raise ValueError("branch rebuild projects need a project_id")
            for branch in self.hublink.branches(project):
                fresh.register(
                    project.project_id,
                    branch.branch_timeline_id,
                    source_timeline_id=branch.source_timeline_id,
                    agent_id=branch.agent_id,
                    version=branch.version,
                )
                count += 1
        current._owners.update(fresh._owners)
        return count

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
