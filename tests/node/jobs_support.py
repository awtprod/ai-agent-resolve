"""A fake Resolve with a broker, a hub event log and a job runner wired together.

Mirrors ``tests/node/ops_support`` for the job runner: a real broker over a real
``ScriptApiGateway`` over the in-memory :class:`FakeResolve`, a real hub ``EventLog`` behind a
``LocalHubLink``, and a real SQLite :class:`JobStore` in ``tmp_path``. Nothing here is mocked;
only Resolve is faked. ``restart`` models a node restart: a fresh broker and runner reopen the
same store and hub log against the same (persistent) Resolve.

The broker uses a very large restore delay so it holds the instance for the whole test instead
of restoring the borrowed context between a job's ops (in production the default 15 s delay
does the same while work is pending).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aar.hub.events import EventLog
from aar.node.broker import Broker, BrokerConfig
from aar.node.gateway import ScriptApiGateway
from aar.node.hublink import LocalHubLink
from aar.node.jobs import ApprovalPolicy, JobRunner, JobStore, Watchdog, WatchdogConfig
from aar.node.leases import Holder, LeaseManager
from aar.node.locator import ProjectLocator
from aar.node.presence import PresenceGuard

from .fake_resolve import FakeClock, FakeResolve

IDLE = 60.0
NODE_ID = "node-shows"

JOB_CONFIG = BrokerConfig(
    batch_max_ops=50,
    batch_max_seconds=1e6,
    max_wait_seconds=1e6,
    restore_delay_seconds=1e6,
)


class JobWorld:
    """A project "Show" (timelines "Main" and "Main [agent]") with a job runner in front."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        approval: ApprovalPolicy | None = None,
        watchdog: Watchdog | None = None,
        node_id: str = NODE_ID,
    ) -> None:
        self.fake = FakeResolve(
            {"Show": ["Main", "Main [agent]"], "Other": ["Other main"]},
            current="Show",
            folder=("Shows",),
        )
        self.project = self.fake.project("Show")
        self.main = self.project.timeline("Main")
        self.gw = ScriptApiGateway(self.fake)
        self.clock = FakeClock()
        self.node_id = node_id
        self.db_path = str(tmp_path / "jobs.db")
        self.log = EventLog(str(tmp_path / "hub.db"))
        self.hublink = LocalHubLink(self.log)
        self._approval = approval or ApprovalPolicy(live=False)
        self._watchdog = watchdog or Watchdog(WatchdogConfig())
        self._fresh_broker()
        self.store = JobStore(self.db_path)
        self.runner = self._new_runner()

    def _fresh_broker(self) -> None:
        self.leases = LeaseManager(self.clock)
        self.presence = PresenceGuard(idle_window=IDLE)
        self.broker = Broker(
            self.gw, leases=self.leases, presence=self.presence, clock=self.clock, config=JOB_CONFIG
        )
        self.presence.observe(self.gw.get_context(), self.clock())
        self.clock.advance(IDLE)
        self.fake.calls.clear()

    def _new_runner(self) -> JobRunner:
        return JobRunner(
            self.broker,
            self.store,
            self.hublink,
            node_id=self.node_id,
            clock=self.clock,
            approval=self._approval,
            watchdog=self._watchdog,
        )

    def restart(
        self, *, approval: ApprovalPolicy | None = None, watchdog: Watchdog | None = None
    ) -> JobRunner:
        """Model a node restart: a fresh broker and runner reopen the same store and log."""
        self.store.close()
        if approval is not None:
            self._approval = approval
        if watchdog is not None:
            self._watchdog = watchdog
        self._fresh_broker()
        self.store = JobStore(self.db_path)
        self.runner = self._new_runner()
        return self.runner

    # -- helpers --------------------------------------------------------------------------

    def locator(self, name: str = "Show") -> ProjectLocator:
        project = self.fake.project(name)
        return ProjectLocator(project.folder, name, project.GetUniqueId())

    def pid(self, name: str = "Show") -> str:
        return self.fake.project(name).GetUniqueId()

    def ref(self, timeline: str, project: str = "Show") -> dict[str, str]:
        return {
            "name": timeline,
            "unique_id": self.fake.project(project).timeline(timeline).GetUniqueId(),
        }

    def human_leases_project(self, name: str = "Show") -> None:
        from aar.node.leases import Scope, project_key

        self.leases.acquire(
            Holder.human("andrew"), Scope.PROJECT, project_key(self.pid(name)), None
        )

    def marker_fields(self, **over: Any) -> dict[str, Any]:
        """A marker dict shaped like the fake stores it, matching markers.add defaults."""
        base = {"color": "Blue", "duration": 1, "note": "", "name": "", "customData": ""}
        base.update(over)
        return base
