"""The session broker, run against ``ScriptApiGateway`` on the fake Resolve."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aar.node.broker import (
    Broker,
    BrokerConfig,
    BrokerEvent,
    HotOp,
    OpStatus,
    TickKind,
)
from aar.node.gateway import ResolveGateway, ScriptApiGateway
from aar.node.leases import Holder, LeaseManager, Scope, instance_key, project_key, timeline_key
from aar.node.locator import ProjectLocator
from aar.node.presence import PresenceGuard

from .fake_resolve import DeliverState, FakeClock, FakeResolve

IDLE = 60.0
AGENT = Holder.agent("thread-1")
OTHER_AGENT = Holder.agent("thread-2")
ANDREW = Holder.human("andrew")

TEST_CONFIG = BrokerConfig(
    batch_max_ops=10,
    batch_max_seconds=1e6,
    max_wait_seconds=1e6,
    restore_delay_seconds=0.0,
)


class World:
    """A fake Resolve with a human's working context in project A, plus a broker."""

    def __init__(
        self,
        config: BrokerConfig = TEST_CONFIG,
        on_event: Callable[[World, BrokerEvent], None] | None = None,
        fake: FakeResolve | None = None,
    ) -> None:
        self.fake = fake or FakeResolve(
            {"A": ["A main", "A alt"], "B": ["B main", "B alt"], "C": ["C main"]}, current="A"
        )
        a = self.fake.project("A")
        a.SetCurrentTimeline(a.timeline("A alt"))
        a.timeline("A alt").timecode = "01:00:10:00"
        a.media_pool.current = a.media_pool.folder("Footage", "Day 2")
        a.deliver = DeliverState(
            "mov", "ProRes422HQ", 0, {"TargetDir": "E:/Human", "CustomName": "cut v3"}
        )
        self.fake.page = "color"
        self.fake.calls.clear()

        self.gw = ScriptApiGateway(self.fake)
        self.clock = FakeClock()
        self.leases = LeaseManager(self.clock)
        self.input_idle: float | None = None
        self.presence = PresenceGuard(idle_window=IDLE, input_idle=lambda: self.input_idle)
        self.events: list[BrokerEvent] = []
        self.ran: list[tuple[str, str | None, str | None]] = []
        self._on_event = on_event
        self.broker = Broker(
            self.gw,
            leases=self.leases,
            presence=self.presence,
            clock=self.clock,
            sink=self._sink,
            config=config,
        )

    def _sink(self, event: BrokerEvent) -> None:
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(self, event)

    def settle(self) -> None:
        """Let the presence guard watch a quiet idle window."""
        self.presence.observe(self.gw.get_context(), self.clock())
        self.clock.advance(IDLE)

    def loc(self, project: str | ProjectLocator) -> ProjectLocator:
        """A locator with the project's real folder and id (none for unknown projects)."""
        if isinstance(project, ProjectLocator):
            return project
        known = self.fake.projects.get(project)
        if known is None:
            return ProjectLocator(("CCN",), project)
        return ProjectLocator(known.folder, project, known.GetUniqueId())

    def op(
        self,
        project: str | ProjectLocator,
        name: str,
        *,
        cost: float = 0.0,
        action: Callable[[ResolveGateway], Any] | None = None,
        holder: Holder = AGENT,
        **kwargs: Any,
    ) -> HotOp:
        def fn(gw: ResolveGateway) -> str:
            ctx = gw.get_context()
            self.ran.append((name, ctx.project, ctx.timeline))
            self.clock.advance(cost)
            if action is not None:
                action(gw)
            return name

        return HotOp(project=self.loc(project), fn=fn, holder=holder, label=name, **kwargs)

    def submit(self, project: str | ProjectLocator, name: str, **kwargs: Any) -> str:
        return self.broker.submit(self.op(project, name, **kwargs))

    def names(self) -> list[str]:
        return [name for name, _, _ in self.ran]

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def of(self, kind: str) -> list[BrokerEvent]:
        return [e for e in self.events if e.kind == kind]

    def loads(self) -> list[str]:
        return [c[1] for c in self.fake.calls if c[0] == "LoadProject"]


@pytest.fixture
def world() -> World:
    return World()


# -- snapshot and restore -----------------------------------------------------------------


def test_human_context_round_trips_through_snapshot_and_restore(world: World) -> None:
    before = world.gw.get_context()
    deliver_before = world.fake.project("A").deliver.copy()
    world.settle()

    def clobber(gw: ResolveGateway) -> None:
        gw.load_render_preset("YouTube 1080p")
        gw.set_current_timeline("A main")
        gw.set_playhead("01:05:00:00")
        gw.open_page("edit")

    world.submit("A", "a1", action=clobber)
    world.submit("B", "b1", timeline="B alt", page="deliver")
    outcomes = world.broker.run_until_idle()

    assert [o.kind for o in outcomes] == [
        TickKind.RAN,
        TickKind.RAN,
        TickKind.RESTORED,
        TickKind.IDLE,
    ]
    assert outcomes[2].code == "ok"
    assert world.ran == [("a1", "A", "A alt"), ("b1", "B", "B alt")]
    assert world.gw.get_context() == before
    assert world.fake.project("A").deliver == deliver_before
    assert not [p for p in world.fake.presets if p.startswith("aar-restore-")]
    report = world.broker.last_restore
    assert report is not None and report.ok
    assert report.mismatches == {}
    assert not world.broker.in_control
    assert world.leases.active() == []
    snapshot_events = world.of("snapshot")
    assert len(snapshot_events) == 1
    assert snapshot_events[0].data["context"]["playhead"] == "01:00:10:00"
    assert snapshot_events[0].data["context"]["project_manager_path"] == ("CCN",)
    assert snapshot_events[0].data["project_locator"] == world.loc("A").to_dict()
    assert [(e.data["source"], e.data["target"]) for e in world.of("switch")] == [("A", "CCN/B")]
    # the human's project was saved before the broker moved away from it
    names = world.fake.call_names()
    assert names.index("SaveProject") < names.index("LoadProject")
    assert world.loads() == ["B", "A"]


def test_restore_waits_for_restore_delay() -> None:
    world = World(BrokerConfig(restore_delay_seconds=15.0))
    world.settle()
    world.submit("B", "b1")
    assert world.broker.tick().kind is TickKind.RAN
    assert world.broker.tick().kind is TickKind.RESTORE_PENDING
    world.clock.advance(14.9)
    assert world.broker.tick().kind is TickKind.RESTORE_PENDING
    world.clock.advance(0.1)
    assert world.broker.tick().kind is TickKind.RESTORED
    assert world.gw.get_context().project == "A"
    assert world.kinds().count("restore_pending") == 1


def test_restore_never_happens_while_human_input_is_recent(world: World) -> None:
    world.settle()
    world.submit("B", "b1")
    assert world.broker.tick().kind is TickKind.RAN

    world.input_idle = 2.0  # someone touched the mouse
    for _ in range(3):
        outcome = world.broker.tick()
        assert outcome.kind is TickKind.RESTORE_DEFERRED
        assert outcome.code == "human_input"
    assert world.gw.get_context().project == "B"
    assert world.loads() == ["B"]
    assert world.kinds().count("restore_deferred") == 1

    world.input_idle = 600.0
    assert world.broker.tick().kind is TickKind.RESTORED
    assert world.gw.get_context().project == "A"


def test_restore_is_abandoned_when_the_human_takes_over(world: World) -> None:
    world.settle()
    world.submit("B", "b1")
    world.broker.tick()
    preset = world.broker.snapshot.deliver_preset if world.broker.snapshot else None
    assert preset is not None

    world.fake.human_open_project("C")
    outcome = world.broker.tick()

    assert outcome.kind is TickKind.IDLE
    assert world.gw.get_context().project == "C"
    assert world.loads() == ["B"]
    abandoned = world.of("restore_abandoned")
    assert len(abandoned) == 1
    assert abandoned[0].data["deliver_preset_kept"] == preset
    assert "project" in abandoned[0].data["changed"]
    assert preset in world.fake.presets  # left for the human to recover Deliver settings
    assert not world.broker.in_control
    assert world.leases.active() == []


def test_failed_deliver_snapshot_is_reported_not_hidden(world: World) -> None:
    world.fake.project("A").SaveAsNewRenderPreset = lambda presetName: False  # type: ignore[method-assign]
    world.settle()
    world.submit("B", "b1")
    world.broker.tick()
    snapshot = world.broker.snapshot
    assert snapshot is not None and snapshot.deliver_preset is None
    assert "preset_save_failed" in snapshot.warnings[0]
    assert world.of("snapshot")[0].data["warnings"] == list(snapshot.warnings)


# -- refusing to act while a human is present ----------------------------------------------------


def test_refuses_to_switch_while_human_is_changing_the_context(world: World) -> None:
    world.settle()
    world.fake.human_move_playhead("01:00:20:00")
    op_id = world.submit("B", "b1")

    for _ in range(3):
        outcome = world.broker.tick()
        assert outcome.kind is TickKind.DEFERRED
        assert outcome.code == "human_context_change"
    assert world.fake.calls == []  # not even a render-preset snapshot
    assert [op.id for op in world.broker.pending()] == [op_id]
    assert world.kinds().count("deferred") == 1

    world.clock.advance(IDLE)
    assert world.broker.tick().kind is TickKind.RAN
    assert world.names() == ["b1"]


def test_refuses_while_os_input_is_recent(world: World) -> None:
    world.settle()
    world.input_idle = 1.0
    world.submit("B", "b1")
    outcome = world.broker.tick()
    assert (outcome.kind, outcome.code) == (TickKind.DEFERRED, "human_input")
    assert world.fake.calls == []


def test_fresh_broker_waits_a_whole_idle_window(world: World) -> None:
    world.submit("A", "a1")
    outcome = world.broker.tick()
    assert (outcome.kind, outcome.code) == (TickKind.DEFERRED, "warming_up")
    world.clock.advance(IDLE)
    assert world.broker.tick().kind is TickKind.RAN


def test_human_instance_lease_blocks_the_broker(world: World) -> None:
    lease = world.leases.acquire(ANDREW, Scope.INSTANCE, instance_key(), ttl=None)
    world.settle()
    world.submit("B", "b1")
    outcome = world.broker.tick()
    assert (outcome.kind, outcome.code) == (TickKind.DEFERRED, "instance_leased")
    assert "human:andrew" in outcome.reason
    assert world.fake.calls == []
    world.leases.release(lease.id, ANDREW)
    assert world.broker.tick().kind is TickKind.RAN


def test_human_lease_on_current_project_blocks_switching_away(world: World) -> None:
    world.leases.acquire(ANDREW, Scope.TIMELINE, timeline_key("A", "A alt"), ttl=None)
    world.settle()
    world.submit("B", "b1")
    outcome = world.broker.tick()
    assert (outcome.kind, outcome.code) == (TickKind.DEFERRED, "human_lease_on_current")
    assert world.loads() == []


def test_no_project_switch_while_rendering(world: World) -> None:
    world.fake.project("A").rendering = True
    world.settle()
    world.submit("B", "b1")
    outcome = world.broker.tick()
    assert (outcome.kind, outcome.code) == (TickKind.DEFERRED, "render_in_progress")
    assert world.loads() == []


def test_human_takeover_mid_batch_stops_the_batch() -> None:
    def human_grabs_resolve(world: World, event: BrokerEvent) -> None:
        if event.kind == "op_succeeded" and not world.of("restore_abandoned"):
            world.fake.human_open_project("C")

    world = World(on_event=human_grabs_resolve)
    world.settle()
    for name in ("b1", "b2", "b3"):
        world.submit("B", name)
    outcome = world.broker.tick()

    assert outcome.kind is TickKind.RAN
    assert world.ran == [("b1", "B", "B main")]
    assert [op.label for op in world.broker.pending()] == ["b2", "b3"]
    assert len(world.of("restore_abandoned")) == 1
    assert world.gw.get_context().project == "C"
    follow_up = world.broker.tick()
    assert (follow_up.kind, follow_up.code) == (TickKind.DEFERRED, "human_context_change")
    assert world.gw.get_context().project == "C"


# -- project assertion -----------------------------------------------------------------------


def test_op_never_runs_against_the_wrong_project(world: World) -> None:
    def wander_off(gw: ResolveGateway) -> None:
        gw.load_project(world.loc("C"))
        world.fake.load_silently_fails.add("B")  # and B can no longer be re-selected

    world.settle()
    world.submit("B", "b1", action=wander_off)
    stuck = world.submit("B", "b2")
    world.broker.run_until_idle()

    assert world.names() == ["b1"]
    result = world.broker.result(stuck)
    assert result is not None and result.status is OpStatus.FAILED
    assert result.error is not None
    assert result.error.code == "wrong_project"
    assert result.error.detail["expected"] == "B"
    assert result.error.detail["actual"] == "C"
    assert result.error.detail["gateway_code"] == "project_load_failed"
    assert result.error.detail["expected_id"] == world.loc("B").project_id
    assert [e.data["expected"] for e in world.of("reselect")] == ["CCN/B"]


def test_reselects_the_project_after_an_op_wanders_off(world: World) -> None:
    world.settle()
    world.submit("B", "b1", action=lambda gw: gw.load_project(world.loc("C")))
    world.submit("B", "b2")
    world.broker.run_until_idle()
    assert world.ran == [("b1", "B", "B main"), ("b2", "B", "B main")]
    assert len(world.of("reselect")) == 1


def test_unloadable_project_fails_its_ops_with_a_structured_error(world: World) -> None:
    world.settle()
    ghost = world.submit("Ghost", "g1")
    world.submit("B", "b1")
    world.broker.submit(world.op("Ghost", "g2", priority=1))
    world.broker.run_until_idle()

    result = world.broker.result(ghost)
    assert result is not None and result.status is OpStatus.FAILED
    assert result.error is not None and result.error.code == "project_unavailable"
    assert result.error.detail["gateway_code"] == "project_not_found"
    assert world.names() == ["b1"]
    assert world.broker.pending() == []
    assert world.of("switch_failed")[0].data["target"] == "CCN/Ghost"


def test_timeline_is_asserted_before_the_op(world: World) -> None:
    world.settle()
    missing = world.submit("B", "b-missing", timeline="No such timeline")
    world.submit("B", "b-alt", timeline="B alt")
    world.broker.run_until_idle()
    result = world.broker.result(missing)
    assert result is not None and result.error is not None
    assert result.error.code == "timeline_unavailable"
    assert world.ran == [("b-alt", "B", "B alt")]


# -- per-op results ---------------------------------------------------------------------------


def test_a_failing_op_does_not_abort_the_batch(world: World) -> None:
    def boom(gw: ResolveGateway) -> None:
        raise RuntimeError("AppendToTimeline returned None")

    world.settle()
    ok1 = world.submit("A", "ok1")
    bad = world.submit("A", "bad", action=boom)
    ok2 = world.submit("A", "ok2")
    world.broker.run_until_idle()

    statuses = [world.broker.result(i).status for i in (ok1, bad, ok2)]  # type: ignore[union-attr]
    assert statuses == [OpStatus.OK, OpStatus.FAILED, OpStatus.OK]
    error = world.broker.result(bad).error  # type: ignore[union-attr]
    assert error is not None and error.code == "op_raised"
    assert "AppendToTimeline returned None" in error.message
    assert "RuntimeError" in str(error.detail["traceback"])
    assert world.broker.result(ok1).value == "ok1"  # type: ignore[union-attr]


def test_abort_flag_cancels_the_rest_of_that_holders_batch(world: World) -> None:
    def boom(gw: ResolveGateway) -> None:
        raise RuntimeError("step 1 failed")

    world.settle()
    x1 = world.submit("B", "x1", action=boom, abort_batch_on_failure=True)
    x2 = world.submit("B", "x2")
    y1 = world.submit("B", "y1", holder=OTHER_AGENT)
    x3 = world.submit("B", "x3")
    first = world.broker.tick()

    assert first.ran == (x1,)
    for op_id in (x2, x3):
        result = world.broker.result(op_id)
        assert result is not None and result.status is OpStatus.CANCELLED
        assert result.error is not None and result.error.code == "batch_aborted"
    assert world.broker.result(y1) is None
    world.broker.run_until_idle()
    assert world.names() == ["x1", "y1"]


def test_cancelled_op_never_runs(world: World) -> None:
    world.settle()
    op_id = world.submit("B", "b1")
    assert world.broker.cancel(op_id)
    assert not world.broker.cancel(op_id)
    world.broker.run_until_idle()
    assert world.ran == []
    result = world.broker.result(op_id)
    assert result is not None and result.status is OpStatus.CANCELLED


# -- batching, fairness and starvation ---------------------------------------------------------


def test_prefers_current_project_and_batches_to_minimise_switches(world: World) -> None:
    world.settle()
    for i in range(3):
        world.submit("B", f"b{i}")
        world.clock.advance(1)
        world.submit("A", f"a{i}")
        world.clock.advance(1)
    world.broker.run_until_idle()
    assert world.names() == ["a0", "a1", "a2", "b0", "b1", "b2"]
    assert world.loads() == ["B", "A"]  # one switch out, one restore


def test_batch_quota_rotates_between_projects() -> None:
    world = World(BrokerConfig(batch_max_ops=2, max_wait_seconds=1e6, restore_delay_seconds=0))
    world.settle()
    for i in range(6):
        world.submit("A", f"a{i}")
    for i in range(3):
        world.submit("B", f"b{i}")
    world.broker.run_until_idle()
    assert world.names() == ["a0", "a1", "b0", "b1", "a2", "a3", "b2", "a4", "a5"]


def test_batch_time_quota_ends_the_batch() -> None:
    config = BrokerConfig(
        batch_max_ops=10, batch_max_seconds=50, max_wait_seconds=1e6, restore_delay_seconds=0
    )
    world = World(config)
    world.settle()
    for i in range(5):
        world.submit("A", f"a{i}", cost=20)
    world.submit("B", "b0")
    world.broker.run_until_idle()
    assert world.names() == ["a0", "a1", "a2", "b0", "a3", "a4"]
    assert world.of("batch_ended")[0].data["stop"] == "time"


def test_starvation_is_bounded_by_max_wait(world: World) -> None:
    config = BrokerConfig(
        batch_max_ops=100, batch_max_seconds=1e6, max_wait_seconds=100, restore_delay_seconds=0
    )
    world = World(config)
    world.settle()
    for i in range(10):
        world.submit("A", f"a{i}", cost=30)
    waiting = world.submit("B", "b0")
    world.broker.run_until_idle()

    assert world.names()[:5] == ["a0", "a1", "a2", "a3", "b0"]
    result = world.broker.result(waiting)
    assert result is not None and result.started_at is not None
    submitted = next(e.at for e in world.of("submitted") if e.data["op_id"] == waiting)
    assert result.started_at - submitted <= config.max_wait_seconds + 30
    assert world.of("batch_yielded")[0].data == {"project": "CCN/A", "to": "CCN/B"}
    assert world.of("batch_started")[1].data["why"] == "starved"


def test_priority_orders_ops_within_a_project(world: World) -> None:
    world.settle()
    world.submit("A", "low")
    world.submit("A", "high", priority=5)
    world.submit("A", "mid", priority=1)
    world.broker.run_until_idle()
    assert world.names() == ["high", "mid", "low"]


def test_lease_blocked_op_waits_while_others_run(world: World) -> None:
    held = world.leases.acquire(OTHER_AGENT, Scope.TIMELINE, timeline_key("B", "B main"), ttl=600)
    world.leases.acquire(OTHER_AGENT, Scope.PROJECT, project_key("C"), ttl=600)
    world.settle()
    blocked = world.submit("B", "b-main", timeline="B main")
    c1 = world.submit("C", "c1")
    world.submit("B", "b-alt", timeline="B alt")
    outcomes = world.broker.run_until_idle()

    assert world.names() == ["b-alt"]
    assert [(op.id, op.label) for op in world.broker.pending()] == [
        (blocked, "b-main"),
        (c1, "c1"),
    ]
    assert outcomes[-2].kind is TickKind.RESTORED  # blocked ops do not hold the human hostage
    skipped = {e.data["project"] for e in world.of("skipped")}
    assert skipped == {"CCN/B", "CCN/C"}

    world.leases.release(held.id, OTHER_AGENT)
    world.clock.advance(IDLE)
    world.broker.run_until_idle()
    assert world.names() == ["b-alt", "b-main"]


def test_op_without_timeline_needs_the_whole_project(world: World) -> None:
    world.leases.acquire(OTHER_AGENT, Scope.TIMELINE, timeline_key("B", "B alt"), ttl=600)
    world.settle()
    blocked = world.submit("B", "b-any")
    outcomes = world.broker.run_until_idle()

    assert [o.kind for o in outcomes] == [TickKind.IDLE]
    assert world.fake.calls == []  # never switched (or snapshotted) for an op it could not run
    assert [op.id for op in world.broker.pending()] == [blocked]
    assert world.of("skipped")[0].data["code"].startswith("lease:timeline lease on B/B alt")


def test_lease_taken_mid_batch_is_respected() -> None:
    def other_agent_grabs_b(world: World, event: BrokerEvent) -> None:
        if event.kind == "op_succeeded" and not world.leases.active()[1:]:
            world.leases.acquire(OTHER_AGENT, Scope.PROJECT, project_key("B"), ttl=600)

    world = World(on_event=other_agent_grabs_b)
    world.settle()
    world.submit("B", "b1")
    world.submit("B", "b2")
    world.broker.run_until_idle()
    assert world.names() == ["b1"]
    assert [op.label for op in world.broker.pending()] == ["b2"]
    assert world.gw.get_context().project == "A"  # restored; the blocked op does not hold A


def test_human_preempting_the_instance_lease_pauses_the_restore(world: World) -> None:
    world.settle()
    world.submit("B", "b1")
    assert world.broker.tick().kind is TickKind.RAN
    human = world.leases.acquire(ANDREW, Scope.INSTANCE, instance_key(), ttl=None)

    outcome = world.broker.tick()
    assert (outcome.kind, outcome.code) == (TickKind.RESTORE_DEFERRED, "instance_leased")
    assert world.gw.get_context().project == "B"

    world.leases.release(human.id, ANDREW)
    assert world.broker.tick().kind is TickKind.RESTORED
    assert world.gw.get_context().project == "A"
    lost = world.of("lease_lost")
    assert len(lost) == 1 and lost[0].data["reason"] == "preempted by human:andrew"


class TrustingGateway(ScriptApiGateway):
    """A gateway that takes ``LoadProject``'s word for it, as a naive one would."""

    def load_project(self, locator: ProjectLocator) -> None:
        self.resolve.GetProjectManager().LoadProject(locator.name)


def test_broker_asserts_the_project_itself_and_does_not_trust_the_gateway() -> None:
    world = World()
    world.gw = TrustingGateway(world.fake)
    world.broker = Broker(
        world.gw,
        leases=world.leases,
        presence=world.presence,
        clock=world.clock,
        sink=world._sink,
        config=TEST_CONFIG,
    )
    world.fake.load_silently_fails.add("B")  # LoadProject("B") returns B but A stays open
    world.settle()
    op_id = world.submit("B", "b1")
    world.broker.run_until_idle()

    assert world.ran == []
    result = world.broker.result(op_id)
    assert result is not None and result.error is not None
    assert result.error.code == "wrong_project"
    assert (result.error.detail["expected"], result.error.detail["actual"]) == ("B", "A")


# -- Project Manager folders and same-named projects (trap pm-folder-scoped-load) ------------

PROJECTS = {"A": ["A main", "A alt"], "B": ["B main", "B alt"], "C": ["C main"]}


def test_restore_puts_the_project_manager_back_in_the_humans_folder() -> None:
    fake = FakeResolve(PROJECTS, current="A", folders={"B": ("Clients", "Acme")})
    fake.add_pm_folder(("Archive", "2024"))
    fake.human_browse_project_manager(("Archive", "2024"))  # browsed away after opening A
    world = World(fake=fake)
    before = world.gw.get_context(with_pm_path=True)
    world.settle()
    world.submit("B", "b1")
    outcomes = world.broker.run_until_idle()

    assert [o.kind for o in outcomes] == [TickKind.RAN, TickKind.RESTORED, TickKind.IDLE]
    assert world.ran == [("b1", "B", "B main")]
    # A is not in the human's PM folder, so the snapshot found it by walking the tree.
    assert world.of("snapshot")[0].data["project_locator"] == world.loc("A").to_dict()
    report = world.broker.last_restore
    assert report is not None and report.ok, report
    assert report.final.project_manager_path == ("Archive", "2024")
    assert world.gw.get_context(with_pm_path=True) == before


def test_restore_reports_a_project_manager_path_it_could_not_restore() -> None:
    fake = FakeResolve(PROJECTS, current="A", folders={"B": ("Clients", "Acme")})
    fake.add_pm_folder(("Archive", "2024"))
    fake.human_browse_project_manager(("Archive", "2024"))
    world = World(fake=fake)
    world.settle()
    world.submit("B", "b1", action=lambda gw: fake.pm_extra_folders.clear())  # folder gone
    world.broker.run_until_idle()

    report = world.broker.last_restore
    assert report is not None and not report.ok
    assert report.errors["project_manager_path"].startswith("pm_open_failed")
    assert report.mismatches["project_manager_path"] == (("Archive", "2024"), ())
    assert world.gw.get_context().project == "A"  # the project itself came back


def test_same_named_project_in_another_folder_is_switched_to_by_id() -> None:
    fake = FakeResolve(PROJECTS, current="A")
    twin = fake.add_project("A", ["A main", "A alt"], ("Clients", "Acme"))
    world = World(fake=fake)
    human_id = fake.project("A").GetUniqueId()
    seen: list[str | None] = []
    world.settle()
    world.submit(
        ProjectLocator(("Clients", "Acme"), "A", twin.GetUniqueId()),
        "twin",
        action=lambda gw: seen.append(gw.get_context().project_id),
    )
    world.broker.run_until_idle()

    assert seen == [twin.GetUniqueId()]  # a name-only match would have run it in the human's A
    assert world.loads() == ["A", "A"]
    assert world.gw.get_context().project_id == human_id
    report = world.broker.last_restore
    assert report is not None and report.ok


def test_unlocatable_human_project_is_reported_not_guessed() -> None:
    fake = FakeResolve(PROJECTS, current="A")
    fake.add_project("A", ["A main"], ("Clients", "Acme"))
    fake.add_pm_folder(("Archive",))
    fake.human_browse_project_manager(("Archive",))  # and there are two projects named A
    world = World(fake=fake)
    world.settle()
    world.submit("B", "b1")
    world.broker.run_until_idle()

    snapshot_warnings = world.of("snapshot")[0].data["warnings"]
    assert any("cannot be reloaded on restore" in w for w in snapshot_warnings)
    report = world.broker.last_restore
    assert report is not None and not report.ok
    assert report.errors["project"].startswith("human_project_unlocated")
    assert world.loads() == ["B"]  # never loaded a guessed "A"
    assert world.gw.get_context().project == "B"


def test_pm_path_snapshot_can_be_turned_off_for_read_only_use() -> None:
    config = BrokerConfig(
        restore_delay_seconds=0.0,
        snapshot_deliver_preset=False,
        snapshot_project_manager_path=False,
        save_before_switch=False,
    )
    world = World(config)
    world.settle()
    world.submit("A", "a1")
    world.broker.tick()
    names = world.fake.call_names()
    assert not {"GotoParentFolder", "GotoRootFolder", "OpenFolder"} & set(names)
    assert "SaveAsNewRenderPreset" not in names
    snapshot = world.broker.snapshot
    assert snapshot is not None and snapshot.context.project_manager_path is None
