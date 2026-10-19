"""A fake Resolve with a broker, a hub event log and a job runner wired together.

Mirrors ``tests/node/ops_support`` for the job runner, through the real node wiring
(:class:`~aar.node.runtime.NodeRuntime`, P5): a real broker over a real ``ScriptApiGateway`` over
the in-memory :class:`FakeResolve`, the lease table and the S11 feed bound to each other, a
presence reporter, one branch registry, and a real hub ``EventLog`` behind a ``LocalHubLink`` and
the feed's ``EventLogTransport``, with a real SQLite :class:`JobStore` in ``tmp_path``. Nothing
here is mocked; only Resolve is faked. ``restart`` models a node restart: a fresh runtime reopens
the same store and hub log against the same (persistent) Resolve.

Every job is submitted for the agent :data:`AGENT_ID` (P2). By default the world's branch
registry says that agent owns the "Show" timelines and "Other main" (``own_timelines``), so the
runner-mechanics tests edit them with the branch guard ON (an agent editing its own branch);
the S7 x S11 policy tests build a world with ``own_timelines=False`` and real human timelines.

The broker uses the production restore delay (15 s). With a frozen clock it therefore holds the
borrowed context for the test (nothing to restore), and a test that advances the clock past the
delay sees the real restore -- the behaviour production has while work is pending.

Determinism: the runner's drain budget is ``None`` by default here, so a pump waits for its
drain to FINISH and every test counts pumps on the injected clock alone, never on how fast the
machine is. A test about a hung gateway call passes an explicit ``drain_budget`` and drives the
hang with events (``threading.Event``), waiting on completion signals (``runner.wait_drain``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from aar.hub.events import EventLog
from aar.node.broker import BrokerConfig
from aar.node.gateway import ScriptApiGateway
from aar.node.hublink import LocalHubLink
from aar.node.jobs import ApprovalPolicy, JobRunner, JobStore, Watchdog, WatchdogConfig
from aar.node.leases import Holder
from aar.node.locator import ProjectLocator
from aar.node.runtime import NodeRuntime

from .fake_resolve import FakeClock, FakeResolve

IDLE = 60.0
NODE_ID = "node-shows"
AGENT_ID = "test-agent"
"""The agent every test job works for (``Holder.agent(AGENT_ID)``)."""
WALL_EPOCH = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
LAYOUT = {"Show": ["Main", "Main [agent]"], "Other": ["Other main"]}


def _main_ref() -> dict[str, str]:
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    return {"name": "Main", "unique_id": fake.project("Show").timeline("Main").GetUniqueId()}


LIVE_WORLDS: list[JobWorld] = []
"""Every world built in the current test; ``tests/node/conftest.py`` closes them afterwards
(one live runtime per node id, so a test's world must not outlive it)."""

MAIN_REF = _main_ref()
"""The "Main" timeline by unique id (the fake's ids are deterministic per layout): what an
editing op must name for the branch guard (an op on "whatever is current" is refused)."""
OWNED = (("Show", "Main"), ("Show", "Main [agent]"), ("Other", "Other main"))

JOB_CONFIG = BrokerConfig(
    batch_max_ops=50,
    batch_max_seconds=1e6,
    max_wait_seconds=1e6,
    restore_delay_seconds=15.0,  # the production default, not a test-only value
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
        drain_budget: float | None = None,
        own_timelines: bool = True,
        broker_config: BrokerConfig = JOB_CONFIG,
    ) -> None:
        self.fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
        self.project = self.fake.project("Show")
        self.main = self.project.timeline("Main")
        assert self.main.GetUniqueId() == MAIN_REF["unique_id"]
        self.gw = ScriptApiGateway(self.fake)
        self.clock = FakeClock()
        self.node_id = node_id
        self.db_path = str(tmp_path / "jobs.db")
        self.log = EventLog(str(tmp_path / "hub.db"))
        self.hublink = LocalHubLink(self.log)
        self._approval = approval or ApprovalPolicy(live=False)
        self._watchdog = watchdog or Watchdog(WatchdogConfig())
        self.drain_budget = drain_budget
        self.own_timelines = own_timelines
        self._broker_config = broker_config
        self.runtime: NodeRuntime | None = None
        self.store = JobStore(self.db_path)
        LIVE_WORLDS.append(self)
        self._start()

    def close(self) -> None:
        """Close the runtime (frees the node id, stops the drain) and the store."""
        if self.runtime is not None:
            self.runtime.close()
        self.store.close()

    def wall(self) -> datetime:
        """The node's wall clock, tied to the fake monotonic clock (deterministic)."""
        return WALL_EPOCH + timedelta(seconds=self.clock())

    def _start(self) -> None:
        """Build the node runtime (P5 wiring) over the current store, as a fresh process."""
        self.runtime = NodeRuntime.local(
            self.gw,
            self.store,
            self.log,
            node_id=self.node_id,
            clock=self.clock,
            wall_clock=self.wall,
            broker_config=self._broker_config,
            idle_window=IDLE,
            approval=self._approval,
            watchdog=self._watchdog,
            drain_budget=self.drain_budget,
        )
        rt = self.runtime
        self.broker, self.leases, self.presence = rt.broker, rt.leases, rt.presence
        self.feed, self.branch_registry, self.runner = rt.feed, rt.branch_registry, rt.runner
        self.presence.observe(self.gw.get_context(), self.clock())
        self.clock.advance(IDLE)
        self.fake.calls.clear()
        if self.own_timelines:
            for project, timeline in OWNED:
                self.own(timeline, project)

    def own(self, timeline: str, project: str = "Show", agent_id: str = AGENT_ID) -> None:
        """Register ``timeline`` as an agent branch ``agent_id`` owns (as timeline.branch would)."""
        proj = self.fake.project(project)
        uid = proj.timeline(timeline).GetUniqueId()
        self.branch_registry.register(
            proj.GetUniqueId(), uid, source_timeline_id=uid, agent_id=agent_id, version=1
        )

    def restart(
        self,
        *,
        approval: ApprovalPolicy | None = None,
        watchdog: Watchdog | None = None,
        broker_config: BrokerConfig | None = None,
    ) -> JobRunner:
        """Model a node restart: a fresh runtime (broker, leases, feed, registry, runner)
        reopens the same store and log.

        The old runtime is closed first, so a drain thread of the "dead" process can never
        run another op against the shared fake Resolve after the restart.
        """
        if self.runtime is not None:
            self.runtime.close()
        self.store.close()
        if approval is not None:
            self._approval = approval
        if watchdog is not None:
            self._watchdog = watchdog
        if broker_config is not None:
            self._broker_config = broker_config
        self.store = JobStore(self.db_path)
        self._start()
        return self.runner

    def submit(self, kind: str, params: Any, target: ProjectLocator, **kwargs: Any) -> str:
        """Submit a job for :data:`AGENT_ID` (unless ``agent_id`` is given)."""
        kwargs.setdefault("agent_id", AGENT_ID)
        return self.runner.submit(kind, params, target, **kwargs)

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
