"""S7g: the node's context poll must not crash Resolve 21.1.

On 2026-11-07 ``aar node run`` read ``ScriptApiGateway.get_context()`` on every worker tick
(0.25 s), and Resolve 21.1 crashed after about 270 reads. An isolation run then showed that
``Project.GetCurrentRenderMode()`` raises one "Main view page is changed to 11" signal per call.
``GetCurrentRenderFormatAndCodec`` was not tested, so it is treated as suspect too.

These pins hold the fix:

- the presence poll reads only the working position (``get_position``), never Deliver state;
- it runs at most once per ``BrokerConfig.context_poll_seconds`` (2 s), not once per tick;
- ``Broker.tick`` reaches a full read (``get_context``) only through the snapshot it takes when
  it borrows the human's context and the restore that gives it back;
- the presence guard ignores Deliver fields a cheap read did not take, without losing a change.
"""

from __future__ import annotations

import ast
import logging
import math
from collections.abc import Iterable
from pathlib import Path

import pytest

from aar.node import broker as broker_module
from aar.node import gateway as gateway_module
from aar.node.broker import Broker, BrokerConfig, HotOp, TickKind
from aar.node.gateway import ResolveContext, ScriptApiGateway
from aar.node.jobs import JobState
from aar.node.leases import Holder, LeaseManager
from aar.node.presence import PresenceGuard

from .fake_resolve import DeliverState, FakeClock, FakeResolve
from .jobs_support import JobWorld
from .test_broker import World

DELIVER_GETTERS = ("GetCurrentRenderMode", "GetCurrentRenderFormatAndCodec")
TICK = 0.25
"""``aar node run``'s default ``--tick-seconds``: the rate the incident polled at."""
INCIDENT_TICKS = 420
"""1 min 45 s of 0.25 s ticks: how long the node ran before Resolve crashed."""
POLL = 2.0
"""The default ``BrokerConfig.context_poll_seconds`` (pinned below): the budget tests use the
production default, so they measure what ``aar node run`` does without the flag."""


def _deliver_reads(fake: FakeResolve) -> int:
    return sum(fake.reads[name] for name in DELIVER_GETTERS)


def _position_reads(fake: FakeResolve) -> int:
    """Context reads, of either kind: each one asks for the current page exactly once."""
    return fake.reads["GetCurrentPage"]


def _budget(elapsed: float, interval: float) -> int:
    return math.ceil(elapsed / interval) + 1


# == the rate budget (decision 3) ================================================================


def test_idle_broker_polls_position_only_and_at_most_once_per_interval() -> None:
    world = World()
    world.settle()
    full_reads = []
    original = world.gw.get_context
    world.gw.get_context = lambda **kw: full_reads.append(kw) or original(**kw)  # type: ignore[method-assign]
    world.fake.reads.clear()
    start = world.clock()
    for _ in range(INCIDENT_TICKS):
        assert world.broker.tick().kind is TickKind.IDLE
        world.clock.advance(TICK)
    elapsed = world.clock() - TICK - start  # from the first tick to the last
    assert _deliver_reads(world.fake) == 0
    assert full_reads == []
    polls = _position_reads(world.fake)
    assert polls <= _budget(elapsed, POLL)
    assert polls >= elapsed // POLL  # still polled: presence keeps being fed


def test_deferred_job_polls_position_only_at_the_interval(tmp_path: Path) -> None:
    """The incident itself: a claimed job (ops.sequence [timeline.list]) deferred for 1 min 45 s
    because the owner was at the console, pumped at the worker's 0.25 s tick."""
    world = JobWorld(tmp_path, input_idle=lambda: 0.0)  # keyboard/mouse input just now
    job_id = world.submit(
        "ops.sequence", {"steps": [{"op": "timeline.list", "params": {}}]}, world.locator()
    )
    world.fake.reads.clear()
    start = world.clock()
    for _ in range(INCIDENT_TICKS):
        world.runner.pump()
        world.clock.advance(TICK)
    elapsed = world.clock() - TICK - start
    assert _deliver_reads(world.fake) == 0
    assert _position_reads(world.fake) <= _budget(elapsed, POLL)
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.RUNNING.value
    assert len(world.broker.pending()) == 1  # deferred, never run: the human is present
    assert world.presence.may_switch(world.clock()).code == "human_input"


def test_one_borrowed_context_reads_deliver_state_three_times() -> None:
    """Borrow the human's context, switch to B, run an op, restore: the snapshot, the check
    before the Deliver restore and the restore's final check are the only full reads, so each
    Deliver getter is called three times (S7g round 2, F1). The final check cannot be folded
    into the one before: it verifies setters that may report success without applying."""
    world = World()
    world.settle()
    world.fake.reads.clear()
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "done", Holder.agent("thread-1")))
    kinds = [outcome.kind for outcome in world.broker.run_until_idle()]
    assert kinds == [TickKind.RAN, TickKind.RESTORED, TickKind.IDLE]
    assert world.fake.reads["GetCurrentRenderMode"] == 3
    assert world.fake.reads["GetCurrentRenderFormatAndCodec"] == 3
    report = world.broker.last_restore
    assert report is not None and report.ok, report
    assert report.final.render_measured and report.final.render_format == "mov"
    assert world.broker.snapshot is None


def test_render_job_polls_never_read_deliver_state(tmp_path: Path) -> None:
    """render.poll runs once per two pumps while a render runs; neither it nor the broker's
    checks around it may call a Deliver getter. Only the build (its own Deliver snapshot), the
    broker's snapshot and the broker's restore do."""
    from .test_jobs_render import _render_params

    world = JobWorld(tmp_path)
    world.project.render_complete_after = 12
    job_id = world.submit("render", _render_params(world), world.locator())
    world.fake.reads.clear()
    for _ in range(10):
        world.runner.pump()
        if world.project.render_jobs:  # render.build ran: the render is queued and started
            break
    assert [job.status for job in world.project.render_jobs] == ["Rendering"]
    render = world.project.render_jobs[0]
    building = _deliver_reads(world.fake)
    assert building == 2 * len(DELIVER_GETTERS)  # render.build's own snapshot + the broker's
    world.fake.reads.clear()
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.SUCCEEDED.value, view.error
    assert render.polls >= 12  # many polls ...
    assert _deliver_reads(world.fake) == 0  # ... and none read Deliver state
    world.clock.advance(world.broker.config.restore_delay_seconds + 1)
    world.runner.pump()  # the queue stayed empty: the broker restores the human's context
    assert world.broker.snapshot is None
    assert _deliver_reads(world.fake) == 2 * len(DELIVER_GETTERS)  # before and after the restore


# == the static pins (decision 3) ================================================================


def _class(path: str, name: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {f.name: f for f in node.body if isinstance(f, ast.FunctionDef)}
    raise AssertionError(f"class {name} not found in {path}")


def _self_calls(fn: ast.FunctionDef) -> set[str]:
    """Methods ``fn`` calls on ``self`` (``self.x(...)``, or ``self.x`` passed as a callable)."""
    return {
        node.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _attributes(fn: ast.FunctionDef) -> set[str]:
    return {node.attr for node in ast.walk(fn) if isinstance(node, ast.Attribute)}


def _reach(methods: dict[str, ast.FunctionDef], start: str, skip: Iterable[str] = ()) -> set[str]:
    seen, todo = set(), [start]
    skipped = set(skip)
    while todo:
        name = todo.pop()
        if name in seen or name in skipped or name not in methods:
            continue
        seen.add(name)
        todo.extend(_self_calls(methods[name]))
    return seen


def test_tick_reaches_a_full_read_only_through_the_snapshot_and_the_restore() -> None:
    methods = _class(broker_module.__file__, "Broker")
    full_readers = {name for name, fn in methods.items() if "get_context" in _attributes(fn)}
    # Each needs Deliver state: the snapshot keeps the human's format, codec and mode; the
    # check before the restore tells a human's Deliver edit from the broker's own; the read
    # after an op that reported changing Deliver records what it left; the restore checks it
    # put them back. Every other read is a position read.
    assert full_readers == {
        "_take_control",
        "_human_deliver_change",
        "_note_deliver_left",
        "_restore",
    }
    poll_path = _reach(methods, "tick", skip=full_readers)
    assert not poll_path & full_readers
    assert {"_observe", "_look_again", "_run_batch", "_run_op", "_switch"} <= poll_path
    assert "get_position" in _attributes(methods["_observe"])
    assert "_observe" in _self_calls(methods["tick"])
    module = ast.parse(Path(broker_module.__file__).read_text(encoding="utf-8"))
    called = {node.attr for node in ast.walk(module) if isinstance(node, ast.Attribute)}
    assert not called & {*DELIVER_GETTERS, "IsRenderingInProgress"}  # only via the gateway


def test_get_position_cannot_reach_the_deliver_getters() -> None:
    methods = _class(gateway_module.__file__, "ScriptApiGateway")
    assert "get_position" in methods
    reached = _reach(methods, "get_position")
    for name in reached:
        assert not _attributes(methods[name]) & set(DELIVER_GETTERS), name
    full = set().union(*(_attributes(methods[n]) for n in _reach(methods, "get_context")))
    assert set(DELIVER_GETTERS) <= full  # the pin would see them


# == the cheap read (decision 1) =================================================================


def test_get_position_reads_the_working_position_without_deliver_state() -> None:
    world = World()
    world.fake.reads.clear()
    position = world.gw.get_position()
    assert _deliver_reads(world.fake) == 0
    full = world.gw.get_context()
    assert world.fake.reads["GetCurrentRenderMode"] == 1
    assert not position.render_measured and full.render_measured
    assert (position.render_format, position.render_codec, position.render_mode) == (None,) * 3
    assert (full.render_format, full.render_codec, full.render_mode) == ("mov", "ProRes422HQ", 0)
    assert position.diff(full) == {}  # same position; the Deliver fields are not compared
    deliver = {"render_format": "mov", "render_codec": "ProRes422HQ", "render_mode": 0}
    assert position.position() | deliver == full.position()
    assert world.fake.calls == []  # read-only


def test_get_position_with_no_project_open() -> None:
    fake = FakeResolve({"A": ["A main"]}, current="A")
    fake.current = None
    position = ScriptApiGateway(fake).get_position()
    assert position.project is None and position.page == "edit"
    assert not position.render_measured and not position.rendering
    assert _deliver_reads(fake) == 0


def test_get_position_still_reads_rendering() -> None:
    """IsRenderingInProgress raised no signal in the isolation run, and the gate needs it."""
    world = World()
    world.fake.project("A").rendering = True
    assert world.gw.get_position().rendering is True


# == presence with cheap reads (decision 1) ======================================================

_POSITION = {
    "project": "A",
    "project_id": "pa",
    "timeline": "A main",
    "page": "deliver",
    "playhead": "01:00:00:00",
}
_DELIVER = {"render_format": "mov", "render_codec": "ProRes422HQ", "render_mode": 0}


def _full(**over: object) -> ResolveContext:
    """What ``get_context`` returns: the position and the Deliver fields."""
    return ResolveContext(**{**_POSITION, **_DELIVER, **over})  # type: ignore[arg-type]


def _cheap(**over: object) -> ResolveContext:
    """What ``get_position`` returns: the position only, Deliver unread."""
    return ResolveContext(**{**_POSITION, **over}, render_measured=False)  # type: ignore[arg-type]


def _guard() -> PresenceGuard:
    return PresenceGuard(idle_window=60.0, max_observation_age=30.0)


def test_diff_ignores_deliver_fields_a_side_did_not_read() -> None:
    assert _full().diff(_cheap()) == {} and _cheap().diff(_full()) == {}
    assert _full().diff(_full(render_codec="H264")) == {"render_codec": ("ProRes422HQ", "H264")}
    moved = _cheap(playhead="01:00:05:00")
    assert _full().diff(moved) == {"playhead": ("01:00:00:00", "01:00:05:00")}


def test_unknown_to_a_value_is_not_human_activity() -> None:
    g = _guard()
    g.observe(_cheap(), 0.0)
    assert g.observe(_full(), 2.0) is False
    assert g.observe(_cheap(), 4.0) is False  # nor a value to unknown
    g.note_broker_context(_cheap(), 5.0)
    assert g.observe(_cheap(), 6.0) is False  # unread stays unread: nothing to carry
    assert g.observe(_full(render_format="mp4"), 8.0) is False
    assert g.last_human_change_at is None


def test_a_deliver_change_between_two_full_reads_is_not_lost_to_cheap_polls() -> None:
    g = _guard()
    g.observe(_full(), 0.0)
    for t in (2.0, 4.0, 6.0):
        assert g.observe(_cheap(), t) is False
    assert g.observe(_full(render_format="mp4"), 8.0) is True
    assert g.last_human_change == {"render_format": ("mov", "mp4")}


def test_a_position_change_is_seen_by_a_cheap_poll_after_a_full_read() -> None:
    g = _guard()
    g.observe(_full(), 0.0)
    assert g.observe(_cheap(page="edit"), 2.0) is True
    assert g.last_human_change == {"page": ("deliver", "edit")}


def test_another_projects_deliver_fields_are_never_carried_over() -> None:
    g = _guard()
    g.observe(_full(), 0.0)
    assert g.observe(_cheap(project="B", project_id="pb"), 2.0) is True  # a human switched
    b_full = _full(project="B", project_id="pb", render_format="mp4", render_codec="H264")
    assert g.observe(b_full, 70.0) is False  # B's Deliver was never read before: not a change
    assert g.last_human_change_at == 2.0


def test_the_brokers_cheap_note_forgets_the_deliver_fields() -> None:
    """The broker's own op may have changed Deliver state (render.build does), so a cheap note
    of its result must not keep the old values for a later full read to compare against."""
    g = _guard()
    g.observe(_full(), 0.0)
    g.note_broker_context(_cheap(), 1.0)
    assert g.observe(_full(render_codec="H264"), 2.0) is False


# == the poll interval (decision 2) ==============================================================


def test_the_poll_interval_defaults_to_two_seconds() -> None:
    assert BrokerConfig().context_poll_seconds == POLL == 2.0
    assert BrokerConfig(context_poll_seconds=0.5).context_poll_seconds == 0.5


@pytest.mark.parametrize("bad", [0.49, 0.0, -1.0, math.nan, math.inf, -math.inf, True, "2", None])
def test_the_poll_interval_is_finite_and_at_least_half_a_second(bad: object) -> None:
    with pytest.raises(ValueError, match="context_poll_seconds"):
        BrokerConfig(context_poll_seconds=bad)  # type: ignore[arg-type]


def test_the_poll_interval_must_be_shorter_than_the_max_observation_age() -> None:
    fake = FakeResolve({"A": ["A main"]}, current="A")
    clock = FakeClock()

    def broker(poll: float, max_age: float) -> Broker:
        return Broker(
            ScriptApiGateway(fake),
            leases=LeaseManager(clock),
            presence=PresenceGuard(idle_window=60.0, max_observation_age=max_age),
            clock=clock,
            config=BrokerConfig(context_poll_seconds=poll),
        )

    broker(29.5, 30.0)
    for poll, max_age in ((30.0, 30.0), (31.0, 30.0), (2.0, 1.0)):
        with pytest.raises(ValueError, match="max_observation_age"):
            broker(poll, max_age)


def test_presence_windows_hold_at_the_poll_interval() -> None:
    """At 0.25 s ticks and a 2 s poll: an observation is never stale, a human change is seen
    within one interval and dated when it is seen (later, so the idle window only grows), and
    the broker acts once the idle window has passed after it."""
    world = World()
    world.settle()
    interval = world.broker.config.context_poll_seconds
    for _ in range(int(4 * interval / TICK)):
        world.broker.tick()
        assert world.presence.may_switch(world.clock()).allowed
        world.clock.advance(TICK)
    world.fake.human_set_page("edit")
    changed_at = world.clock()
    seen_at = None
    for _ in range(int(interval / TICK) + 1):
        world.broker.tick()
        if world.presence.last_human_change_at is not None:
            seen_at = world.presence.last_human_change_at
            break
        world.clock.advance(TICK)
    assert seen_at is not None and changed_at <= seen_at <= changed_at + interval
    assert world.presence.last_human_change == {"page": ("color", "edit")}
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "done", Holder.agent("thread-1")))
    while world.clock() < seen_at + world.presence.idle_window:
        assert world.broker.tick().kind is TickKind.DEFERRED
        decision = world.presence.may_switch(world.clock())
        assert decision.code == "human_context_change"
        world.clock.advance(TICK)
    assert world.broker.tick().kind is TickKind.RAN


def test_a_human_who_moved_since_the_last_poll_is_seen_before_a_restore() -> None:
    """Between polls a tick reuses the last read, but it looks again before acting on it: a
    human who opened another project 0.5 s after the poll takes over, and is not undone."""
    world = World()
    world.settle()
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "done", Holder.agent("thread-1")))
    assert world.broker.tick().kind is TickKind.RAN  # polled, switched to B, ran
    world.clock.advance(0.5)  # not due: the next poll is 2 s after the last
    world.fake.human_open_project("C")
    assert world.broker.tick().kind is TickKind.IDLE
    assert world.fake.current is world.fake.project("C")
    assert world.broker.snapshot is None
    assert any(e.kind == "restore_abandoned" for e in world.events)


def test_a_human_who_moved_since_the_last_poll_is_seen_before_a_switch() -> None:
    world = World()
    world.settle()
    assert world.broker.tick().kind is TickKind.IDLE  # the poll
    world.clock.advance(0.5)
    world.fake.human_set_page("edit")
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "done", Holder.agent("thread-1")))
    outcome = world.broker.tick()
    assert outcome.kind is TickKind.DEFERRED and outcome.code == "human_context_change"
    assert world.fake.current is world.fake.project("A") and world.broker.snapshot is None


def test_a_tick_between_polls_starts_from_the_brokers_own_latest_read() -> None:
    """After the broker switched to B, a tick before the next poll knows B is open (its own
    read replaced the poll's), so it serves B's queue first instead of switching back to A."""
    world = World()
    world.settle()
    holder = Holder.agent("thread-1")
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "b1", holder))
    assert world.broker.tick().kind is TickKind.RAN
    world.broker.submit(HotOp(world.loc("A"), lambda gw: "a1", holder))
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "b2", holder))
    outcome = world.broker.tick()  # same instant: no poll
    assert outcome.kind is TickKind.RAN and outcome.project == world.loc("B").key
    assert world.fake.current is world.fake.project("B")
    assert [e.kind for e in world.events].count("switch") == 1  # only the first, to B


def test_a_quiet_tick_between_polls_acts_on_the_brokers_latest_read() -> None:
    """The broker's own reads replace the poll's: right after its switch to B, the next tick
    restores from B (one look again, no Deliver read until the restore's checks)."""
    world = World()
    world.settle()
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "done", Holder.agent("thread-1")))
    assert world.broker.tick().kind is TickKind.RAN
    assert world.fake.current is world.fake.project("B")
    world.fake.reads.clear()
    assert world.broker.tick().kind is TickKind.RESTORED
    assert world.fake.current is world.fake.project("A")
    assert world.broker.last_restore is not None and world.broker.last_restore.ok
    assert world.fake.reads["GetCurrentRenderMode"] == 2  # the restore's two checks only


# == a human's Deliver edit is kept (round 2, F1) ===================================

HUMAN_DELIVER = ("mxf", "DNxHR", 1)


def _deliver(world: World, project: str = "A") -> tuple[str, str, int]:
    d = world.fake.project(project).deliver
    return (d.format, d.codec, d.mode)


def _edit_deliver(world: World, project: str = "A") -> None:
    """A human changes the Deliver settings only (no page or position change, no input
    provider in ``World``): the poll cannot see it."""
    a = world.fake.project(project)
    a.deliver = DeliverState(*HUMAN_DELIVER, {**a.deliver.settings, "TargetDir": "F:/Mine"})


def _restore_after(world: World, ticks: int) -> TickKind:
    for _ in range(ticks):
        world.clock.advance(TICK)
        outcome = world.broker.tick()
        if outcome.kind is TickKind.RESTORED:
            return outcome.kind
    return outcome.kind


def test_val_probe_deliver_edit_while_render_queued(caplog: pytest.LogCaptureFixture) -> None:
    """The validator's probe (s7g-val, deviation c), now asserting the edit is KEPT: the owner
    edits A's Deliver settings while the broker holds the borrowed context in B. The poll does
    not see it; the check before the restore does, and the restore leaves Deliver alone."""
    world = World()
    world.settle()
    snapshot_deliver = _deliver(world)
    world.broker.submit(HotOp(world.loc("B"), lambda gw: "done", Holder.agent("thread-1")))
    assert world.broker.tick().kind is TickKind.RAN
    _edit_deliver(world)
    world.fake.reads.clear()
    with caplog.at_level(logging.WARNING, logger="aar.node.broker"):
        assert _restore_after(world, int(3 / TICK)) is TickKind.RESTORED
    assert world.presence.last_human_change_at is None  # the poll never saw it
    assert world.fake.current is world.fake.project("A")  # the position came back
    assert world.fake.page == "color"
    assert _deliver(world) == HUMAN_DELIVER  # the human's edit is kept ...
    assert world.fake.project("A").deliver.settings["TargetDir"] == "F:/Mine"  # ... all of it
    report = world.broker.last_restore
    assert report is not None and report.ok, report
    fields = {
        "render_format": [snapshot_deliver[0], HUMAN_DELIVER[0]],
        "render_codec": [snapshot_deliver[1], HUMAN_DELIVER[1]],
        "render_mode": [snapshot_deliver[2], HUMAN_DELIVER[2]],
    }
    assert report.deliver_restore_skipped == {"reason": "human_change", "fields": fields}
    assert report.kept_preset is None  # no op ran in A: the snapshot preset is redundant
    assert not [p for p in world.fake.presets if p.startswith("aar-restore-")]
    assert not world.fake.call_names().count("SetCurrentRenderFormatAndCodec")
    restored = [e for e in world.events if e.kind == "restored"][-1]
    assert restored.data["deliver_restore_skipped"] == report.deliver_restore_skipped
    assert any("deliver_restore_skipped: human_change" in r.getMessage() for r in caplog.records)
    # Only the check before the restore read Deliver state; the final check is a position read.
    assert world.fake.reads["GetCurrentRenderMode"] == 1
    assert world.fake.reads["GetCurrentRenderFormatAndCodec"] == 1


def test_a_deliver_edit_in_the_borrowed_project_itself_is_kept() -> None:
    """The broker stays in the human's project A (an op on A), the human edits Deliver there:
    the snapshot preset is kept and named, because an op ran in A."""
    world = World()
    world.settle()
    world.broker.submit(HotOp(world.loc("A"), lambda gw: "done", Holder.agent("thread-1")))
    assert world.broker.tick().kind is TickKind.RAN
    preset = world.broker.snapshot.deliver_preset if world.broker.snapshot else None
    assert preset is not None
    _edit_deliver(world)
    assert _restore_after(world, int(3 / TICK)) is TickKind.RESTORED
    assert _deliver(world) == HUMAN_DELIVER
    report = world.broker.last_restore
    assert report is not None and report.deliver_restore_skipped is not None
    assert set(report.deliver_restore_skipped["fields"]) == {  # type: ignore[arg-type]
        "render_format",
        "render_codec",
        "render_mode",
    }
    assert report.kept_preset == preset and preset in world.fake.presets


def test_without_a_human_edit_the_snapshot_deliver_settings_come_back() -> None:
    """The check must not stop the normal restore: an op that changed A's Deliver settings
    through the gateway reported it (the gateway's setters do), so it is the broker's change
    and the snapshot is put back over it."""
    world = World()
    world.settle()
    snapshot_deliver = _deliver(world)

    def clobbers_deliver(gw) -> str:
        gw.set_render_format_and_codec("mp4", "H265")
        return "done"

    world.broker.submit(HotOp(world.loc("A"), clobbers_deliver, Holder.agent("thread-1")))
    world.fake.reads.clear()
    assert world.broker.tick().kind is TickKind.RAN
    # The op said so: one full read records what it left (snapshot + that read so far).
    assert world.fake.reads["GetCurrentRenderMode"] == 2
    assert _restore_after(world, int(3 / TICK)) is TickKind.RESTORED
    assert _deliver(world) == snapshot_deliver
    report = world.broker.last_restore
    assert report is not None and report.ok and report.deliver_restore_skipped is None


def test_a_reported_deliver_change_in_another_project_is_not_the_humans() -> None:
    """Deliver settings are per project: an op in B that changes B's is no reason to read
    them, and the restore of A compares A's with the snapshot (no skip, no extra read)."""
    world = World()
    world.settle()
    snapshot_deliver = _deliver(world)

    def changes_b(gw) -> str:
        gw.set_render_format_and_codec("mp4", "H265")
        return "done"

    world.broker.submit(HotOp(world.loc("B"), changes_b, Holder.agent("thread-1")))
    world.fake.reads.clear()
    kinds = [outcome.kind for outcome in world.broker.run_until_idle()]
    assert kinds == [TickKind.RAN, TickKind.RESTORED, TickKind.IDLE]
    assert _deliver(world, "B")[:2] == ("mp4", "H265")
    assert _deliver(world) == snapshot_deliver
    report = world.broker.last_restore
    assert report is not None and report.ok and report.deliver_restore_skipped is None, report
    assert world.fake.reads["GetCurrentRenderMode"] == 3  # snapshot, check, final: no more


def test_an_unreported_deliver_change_by_an_op_is_taken_for_a_humans() -> None:
    """Fail-safe direction: op code that changes Deliver through the raw API without reporting
    it cannot be told from a human, so the broker keeps what it finds rather than overwrite a
    possible human edit."""
    world = World()
    world.settle()

    def silently_clobbers(gw) -> str:
        project = gw.resolve.GetProjectManager().GetCurrentProject()
        assert project.SetCurrentRenderFormatAndCodec("mp4", "H265")
        return "done"

    world.broker.submit(HotOp(world.loc("A"), silently_clobbers, Holder.agent("thread-1")))
    assert world.broker.tick().kind is TickKind.RAN
    assert _restore_after(world, int(3 / TICK)) is TickKind.RESTORED
    report = world.broker.last_restore
    assert report is not None and report.deliver_restore_skipped is not None
    assert _deliver(world)[:2] == ("mp4", "H265")


def test_a_failed_render_build_restore_is_the_brokers_change_not_a_humans(
    tmp_path: Path,
) -> None:
    """render.build drives the API directly and reports a failed Deliver restore
    (note_deliver_changed), so the broker reads what it left once and its own restore puts the
    human's settings back."""
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 3
    original = world.project.deliver.copy()
    proj = world.project
    load = proj.LoadRenderPreset
    proj.LoadRenderPreset = lambda name: False if name.startswith("aar-job-") else load(name)  # type: ignore[method-assign]
    from .test_jobs_render import _render_params

    job_id = world.submit("render", _render_params(world), world.locator())
    world.fake.reads.clear()
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.SUCCEEDED.value
    assert view.measured["deliver_restore_error"]["code"] == "preset_load_failed"
    assert (proj.deliver.format, proj.deliver.codec) == ("mov", "ProRes")  # the job's
    # snapshot + render.build's own + the one read of what the failed build left
    assert _deliver_reads(world.fake) == 3 * len(DELIVER_GETTERS)
    world.clock.advance(world.broker.config.restore_delay_seconds + 1)
    world.runner.pump()
    report = world.broker.last_restore
    assert report is not None and report.deliver_restore_skipped is None, report
    assert (proj.deliver.format, proj.deliver.codec, proj.deliver.mode) == (
        original.format,
        original.codec,
        original.mode,
    )
    assert proj.deliver.settings == original.settings


def test_a_reporting_op_in_the_human_project_costs_one_more_full_read() -> None:
    """An op in A that reports its own Deliver change (``set_render_mode`` does, through the
    gateway) makes the broker read what it left, once: four full reads, not three. The restore
    then puts the snapshot's mode back over the op's, which is not a human's edit."""
    world = World()
    world.settle()
    snapshot_deliver = _deliver(world)

    def sets_mode(gw) -> str:
        gw.set_render_mode(1 - snapshot_deliver[2])
        return "done"

    world.fake.reads.clear()
    world.broker.submit(HotOp(world.loc("A"), sets_mode, Holder.agent("thread-1")))
    kinds = [outcome.kind for outcome in world.broker.run_until_idle()]
    assert kinds == [TickKind.RAN, TickKind.RESTORED, TickKind.IDLE]
    report = world.broker.last_restore
    assert report is not None and report.ok and report.deliver_restore_skipped is None, report
    assert _deliver(world) == snapshot_deliver
    assert world.fake.reads["GetCurrentRenderMode"] == 4  # snapshot, the op's, check, final
    assert world.fake.reads["GetCurrentRenderFormatAndCodec"] == 4


def test_after_an_op_reports_its_deliver_change_a_later_human_edit_is_kept() -> None:
    """The claim the docs make: once an op in A has reported its own Deliver change, the broker
    compares against what the op left, so a human edit made after that is kept (and the
    snapshot preset is named, because an op ran in A)."""
    world = World()
    world.settle()

    def sets_mode(gw) -> str:
        gw.set_render_mode(1)
        return "done"

    world.broker.submit(HotOp(world.loc("A"), sets_mode, Holder.agent("thread-1")))
    assert world.broker.tick().kind is TickKind.RAN
    _edit_deliver(world)
    assert _restore_after(world, int(3 / TICK)) is TickKind.RESTORED
    assert _deliver(world) == HUMAN_DELIVER
    report = world.broker.last_restore
    assert report is not None and report.deliver_restore_skipped is not None
    assert report.kept_preset is not None and report.kept_preset in world.fake.presets


def test_a_human_edit_before_a_reporting_op_in_the_same_project_is_overwritten() -> None:
    """The documented limit of "kept": a human edit made before an op in the same project
    reports its own change is absorbed into what the broker left, so the restore overwrites it."""
    world = World()
    world.settle()
    snapshot_deliver = _deliver(world)
    world.broker.submit(HotOp(world.loc("A"), lambda gw: "done", Holder.agent("thread-1")))
    assert world.broker.tick().kind is TickKind.RAN
    _edit_deliver(world)

    def sets_mode(gw) -> str:
        gw.set_render_mode(snapshot_deliver[2])
        return "done"

    world.broker.submit(HotOp(world.loc("A"), sets_mode, Holder.agent("thread-1")))
    world.broker.run_until_idle()
    report = world.broker.last_restore
    assert report is not None and report.deliver_restore_skipped is None, report
    assert _deliver(world) == snapshot_deliver


def test_a_crashed_render_builds_recovered_snapshot_is_not_a_reported_change(
    tmp_path: Path,
) -> None:
    """Crash after render.build's job settings went in and before its own restore: the next run
    recovers the snapshot preset (``snapshot_recovered``). That recovery must not report a
    Deliver change, or the broker would take the job's leftovers as its own and the recovery
    would not hold; the pre-job settings come back."""
    from .jobs_support import AGENT_ID
    from .test_jobs_render import _render_params

    class Crash(BaseException):
        pass

    world = JobWorld(tmp_path)
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    load = world.project.LoadRenderPreset

    def die(name: str) -> bool:
        if name.startswith("aar-job-"):
            raise Crash("power cut before the build's Deliver restore")
        return load(name)

    world.project.LoadRenderPreset = die  # type: ignore[method-assign]
    with pytest.raises(Crash):
        world.broker.run_until_idle()
    world.project.LoadRenderPreset = load  # type: ignore[method-assign]
    assert (world.project.deliver.format, world.project.deliver.codec) != (
        original.format,
        original.codec,
    )  # the crashed attempt left the job's settings in place
    runner = world.restart()
    runner.run_until_settled()
    world.clock.advance(16)
    runner.pump()
    assert runner.get(job_id) is not None
    assert (world.project.deliver.format, world.project.deliver.codec) == (
        original.format,
        original.codec,
    )
    assert world.project.deliver.settings == original.settings


# == the CLI flag (decision 2) ===================================================================


def _node_run_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "node",
        "run",
        "--node-id",
        "edit-pc",
        "--hub-url",
        "http://127.0.0.1:1",
        "--token-file",
        str(tmp_path / "token"),
        "--store",
        str(tmp_path / "store.db"),
        "--project",
        '{"folder_path":[],"name":"Show","project_id":"uid"}',
        *extra,
    ]


def test_node_run_takes_the_poll_interval_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aar.cli import main
    from aar.node import cli as node_cli

    seen: list[float] = []

    def stop_at_the_gateway(args, gateway_factory=None):
        seen.append(args.context_poll_seconds)
        return 0

    monkeypatch.setattr(node_cli, "run_node", stop_at_the_gateway)
    assert main(_node_run_args(tmp_path)) == 0
    assert main(_node_run_args(tmp_path, "--context-poll-seconds", "5")) == 0
    assert seen == [2.0, 5.0]


@pytest.mark.parametrize("bad", ["0.1", "0", "-2", "nan", "inf"])
def test_node_run_refuses_a_bad_poll_interval_before_touching_resolve(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    bad: str,
) -> None:
    from aar.cli import main
    from aar.node import cli as node_cli

    def unexpected():
        pytest.fail("a bad --context-poll-seconds reached Resolve")

    original_run = node_cli.run_node
    monkeypatch.setattr(
        node_cli, "run_node", lambda args: original_run(args, gateway_factory=unexpected)
    )
    assert main(_node_run_args(tmp_path, "--context-poll-seconds", bad)) == 2
    assert "context_poll_seconds" in capsys.readouterr().err
    assert not (tmp_path / "store.db").exists()


def test_node_run_hands_the_poll_interval_to_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse
    import json

    from aar.node import cli as node_cli
    from aar.node.runtime import NodeRuntime

    seen: list[BrokerConfig] = []

    class Stop(Exception):
        pass

    def remote(*args, **kwargs):
        seen.append(kwargs["broker_config"])
        raise Stop

    monkeypatch.setattr(NodeRuntime, "remote", remote)
    args = argparse.Namespace(
        project=[json.dumps({"folder_path": [], "name": "Show", "project_id": "uid"})],
        node_id="edit-pc",
        hub_url="http://127.0.0.1:1",
        token_file=str(tmp_path / "token"),
        store=str(tmp_path / "jobs.db"),
        connect_timeout=1,
        read_timeout=1,
        response_cap=1024,
        lease_seconds=60,
        heartbeat_seconds=15,
        tick_seconds=0.25,
        context_poll_seconds=3.5,
    )
    with pytest.raises(Stop):
        node_cli.run_node(args, gateway_factory=lambda: ScriptApiGateway(FakeResolve({})))
    assert [config.context_poll_seconds for config in seen] == [3.5]
