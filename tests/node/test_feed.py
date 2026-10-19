"""Tests for the node-to-hub feed: never raises into the broker, outbox, flush, gaps."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aar.hub.collab import EVENT_FEED_GAP, feed_gaps
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import lease_board
from aar.node.broker import Broker
from aar.node.feed import EventLogTransport, NodeFeed, PresenceReporter
from aar.node.gateway import ScriptApiGateway
from aar.node.leases import Holder, Lease, LeaseEvent, LeaseManager, Scope
from aar.node.locator import ProjectLocator
from aar.node.ops import submit_op
from aar.node.presence import Decision, PresenceGuard

from .fake_resolve import FakeClock, FakeResolve
from .ops_support import CONFIG, IDLE

WALL = 1_800_000_000.0
NODE = "node-1"
AGENT = Holder.agent("thread-1")


def _feed(log: EventLog, **kw) -> NodeFeed:
    return NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL, **kw)


def _board_clock(mono: float):
    return lambda: datetime.fromtimestamp(mono + WALL, UTC)


class Flaky:
    """A transport that raises for the first ``fail`` deliveries, then writes to the log."""

    def __init__(self, log: EventLog, fail: int) -> None:
        self._real = EventLogTransport(log)
        self.remaining = fail
        self.attempts = 0

    def deliver(self, spec, *, node_id: str) -> None:
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ConnectionError("hub unreachable")
        self._real.deliver(spec, node_id=node_id)


# -- the sink never raises into the LeaseManager or the broker (P01) -------------------------


def test_a_failing_transport_does_not_raise_through_acquire_or_the_broker(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, Flaky(log, fail=1000), wall_offset=lambda: WALL)
    fake = FakeResolve({"Show": ["Main"]}, current="Show", folder=("Shows",))
    gw = ScriptApiGateway(fake)
    clock = FakeClock()
    leases = LeaseManager(clock, sink=feed.lease_sink)
    presence = PresenceGuard(idle_window=IDLE)
    broker = Broker(gw, leases=leases, presence=presence, clock=clock, config=CONFIG)
    presence.observe(gw.get_context(), clock())
    clock.advance(IDLE)

    # An agent acquire, a human preempt, and a whole op run: none of these raise, even though
    # every delivery attempt fails. The lease table and the broker stay consistent.
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)
    leases.acquire(Holder.human("op"), Scope.PROJECT, ("p1",), ttl=None)  # preempts the agent
    assert [lease.holder.id for lease in leases.active()] == ["op"]

    pid = fake.project("Show").GetUniqueId()
    op = submit_op(
        broker, "timeline.list", {}, target=ProjectLocator(("Shows",), "Show", pid), holder=AGENT
    )
    broker.run_until_idle()
    assert broker.result(op).status.value == "ok"
    assert broker._instance_lease is None  # released cleanly; nothing leaked
    assert feed.pending > 0  # the events are still queued for a later flush, not lost or raised


def test_an_unmappable_lease_event_is_dropped_and_gap_flagged_not_raised(tmp_path: Path) -> None:
    # The LeaseManager refuses a non-finite ttl before granting, but the sink must still drop
    # (and gap-flag) any event it cannot map rather than raise into whoever emitted it.
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    lease = Lease("l-1", Scope.PROJECT, ("p1",), AGENT, 0.0, math.inf, math.inf)
    feed.lease_sink(LeaseEvent("granted", 0.0, lease))  # does not raise
    assert feed.pending == 0 and feed.dropped == 1
    feed.flush()
    assert [gap.dropped for gap in feed_gaps(log)] == [1]


@pytest.mark.parametrize("ttl", [math.inf, -math.inf, math.nan, 0.0, -1.0])
def test_acquire_refuses_a_non_finite_or_non_positive_ttl_before_granting(ttl: float) -> None:
    events: list[LeaseEvent] = []
    leases = LeaseManager(FakeClock(), sink=events.append)
    with pytest.raises(ValueError, match="ttl"):
        leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=ttl)
    assert leases.active() == [] and events == []  # never granted, so never announced


# -- acquire refuses bad input before granting (P02b) ----------------------------------------


def test_acquire_refuses_a_too_long_holder_id_and_too_large_ttl_before_granting() -> None:
    leases = LeaseManager(FakeClock())
    with pytest.raises(ValueError, match="holder id"):
        leases.acquire(Holder.agent("x" * 129), Scope.PROJECT, ("p1",), ttl=60)
    with pytest.raises(ValueError, match="ttl"):
        leases.acquire(Holder.agent("y"), Scope.PROJECT, ("p2",), ttl=2e9)
    assert leases.active() == []  # nothing was granted


# -- node_id must be clean (P02c) ------------------------------------------------------------


def test_node_id_with_surrounding_whitespace_is_refused_at_construction(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    with pytest.raises(ValueError, match="node_id"):
        NodeFeed("node-1 ", EventLogTransport(log))


# -- idempotency: one stored spec, replayed on retry (P02a / #11) ----------------------------


def test_a_failed_send_is_retried_with_the_same_spec_despite_offset_drift(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    offset = [WALL]
    feed = NodeFeed(NODE, Flaky(log, fail=1), wall_offset=lambda: offset[0])
    leases = LeaseManager(lambda: 1000.0, sink=feed.lease_sink)
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)  # mapped once, at this offset

    first = feed.flush()
    assert first.sent == 0 and first.error is not None and feed.pending == 1
    offset[0] += 0.001  # the wall offset drifts before the retry

    second = feed.flush()  # replays the SAME stored spec, not a re-mapping
    assert second.sent == 1 and feed.pending == 0
    rows = log.read(types=["lease.granted"])
    assert len(rows) == 1
    # The stored wall times are the ones from the first mapping, not the drifted offset.
    assert rows[0].payload["acquired_at"] == datetime.fromtimestamp(1000.0 + WALL, UTC).isoformat()


# -- renewal coalescing (#14, P09) -----------------------------------------------------------


def test_pending_renewals_are_coalesced_to_the_latest(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    clock = FakeClock()
    leases = LeaseManager(clock, sink=feed.lease_sink)
    lease = leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)
    for _ in range(5):
        clock.advance(1)
        leases.renew(lease.id, AGENT)

    feed.flush()
    kinds = [event.type for event in log.read()]
    assert kinds.count("lease.granted") == 1
    assert kinds.count("lease.renewed") == 1  # four intermediate renewals coalesced away
    board = lease_board(log, clock=_board_clock(1064.0))
    # The surviving renewal is the last one (expiry 1005 + 60), not the first (1001 + 60).
    assert board.leases[0].expires_at == datetime.fromtimestamp(1065.0 + WALL, UTC).isoformat()


# -- the outbox cap drops the oldest, counts it, and reports a gap ----------------------------


def test_the_outbox_cap_drops_the_oldest_counts_it_and_reports_a_gap(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log, outbox_max=2)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    for i in range(3):
        leases.acquire(AGENT, Scope.PROJECT, (f"p{i}",), ttl=60)

    assert feed.dropped == 1 and feed.pending == 2
    report = feed.flush()
    assert report.delivered_gap and report.sent == 2
    gaps = feed_gaps(log)
    assert [(gap.node_id, gap.dropped) for gap in gaps] == [(NODE, 1)]
    # The two newest leases survived; the oldest (p0) was dropped.
    kept = {lease.key[0] for lease in lease_board(log, clock=_board_clock(1030.0)).leases}
    assert kept == {"p1", "p2"}


# -- lease board reflects broker activity end to end -----------------------------------------


def test_the_board_reflects_broker_lease_activity_end_to_end(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    fake = FakeResolve({"Show": ["Main"]}, current="Show", folder=("Shows",))
    gw = ScriptApiGateway(fake)
    clock = FakeClock()
    leases = LeaseManager(clock, sink=feed.lease_sink)
    presence = PresenceGuard(idle_window=IDLE)
    broker = Broker(gw, leases=leases, presence=presence, clock=clock, config=CONFIG)
    presence.observe(gw.get_context(), clock())
    clock.advance(IDLE)

    pid = fake.project("Show").GetUniqueId()
    leases.acquire(AGENT, Scope.PROJECT, (pid,), ttl=600.0)
    op = submit_op(
        broker, "timeline.list", {}, target=ProjectLocator(("Shows",), "Show", pid), holder=AGENT
    )
    broker.run_until_idle()
    assert broker.result(op).status.value == "ok"

    feed.flush()  # the pump pushes the node's queued lease events to the hub
    board = lease_board(log, clock=_board_clock(clock()))
    assert [(e.scope, e.key, e.holder_id) for e in board.leases] == [
        ("project", (pid,), "thread-1")
    ]
    granted = log.read(types=["lease.granted"])
    assert any(event.payload["scope"] == "instance" for event in granted)
    assert log.read(types=["lease.released"])  # the broker's instance hold was released


# -- presence reporter -----------------------------------------------------------------------


class _Guard:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def may_switch(self, now: float) -> Decision:
        return self.decision


def test_reporter_emits_active_then_inactive_and_holds_through_warmup(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    guard = _Guard(Decision(False, "warming_up", "warming up"))
    reporter = PresenceReporter(guard, feed, human_id="op", ttl=timedelta(minutes=5))
    wall = datetime(2027, 1, 1, tzinfo=UTC)

    assert reporter.sync(1000.0, wall) is False  # warm-up: unknown, hold
    guard.decision = Decision(False, "human_context_change", "human changed timeline")
    assert reporter.sync(1001.0, wall) is True
    guard.decision = Decision(True, "idle", "quiet")
    assert reporter.sync(1002.0, wall + timedelta(seconds=2)) is True

    feed.flush()
    assert [e.payload["active"] for e in log.read(types=["presence.changed"])] == [True, False]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl": timedelta(0)},
        {"ttl": timedelta(days=2)},
        {"refresh_interval": 600.0, "ttl": timedelta(seconds=60)},
        {"human_id": "h" * 129},
    ],
    ids=["zero-ttl", "ttl-over-24h", "refresh-over-ttl", "long-human-id"],
)
def test_reporter_validates_its_settings_at_construction(tmp_path: Path, kwargs: dict) -> None:
    log = EventLog(tmp_path / "hub.db")
    with pytest.raises(ValueError):
        PresenceReporter(_Guard(Decision(True, "idle", "")), _feed(log), **kwargs)


def test_reporter_rejects_a_naive_wall_time(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    reporter = PresenceReporter(
        _Guard(Decision(False, "human_input", "typing")), _feed(log), human_id="op"
    )
    with pytest.raises(ValueError, match="timezone"):
        reporter.sync(1000.0, datetime(2027, 1, 1))  # naive


def test_feed_gap_events_name_their_own_node(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log, outbox_max=1)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    leases.acquire(AGENT, Scope.PROJECT, ("p0",), ttl=60)
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)  # drops p0
    feed.flush()
    gap = log.read(types=[EVENT_FEED_GAP])
    assert gap and gap[0].actor == Actor("node", NODE)
