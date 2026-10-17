"""Tests for the node-to-hub feed: lease and presence changes written to the hub log."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import lease_board
from aar.node.broker import Broker
from aar.node.feed import EventLogTransport, NodeFeed, PresenceReporter
from aar.node.gateway import ScriptApiGateway
from aar.node.leases import Holder, LeaseManager, Scope
from aar.node.locator import ProjectLocator
from aar.node.ops import submit_op
from aar.node.presence import Decision, PresenceGuard

from .fake_resolve import FakeClock, FakeResolve
from .ops_support import CONFIG, IDLE

WALL_OFFSET = 1_800_000_000.0
NODE = "node-1"
AGENT = Holder.agent("thread-1")
HUMAN = Holder.human("operator")


def _feed(log: EventLog, node_id: str = NODE) -> NodeFeed:
    return NodeFeed(node_id, EventLogTransport(log), wall_offset=lambda: WALL_OFFSET)


def _board_clock(mono: float):
    return lambda: datetime.fromtimestamp(mono + WALL_OFFSET, UTC)


# -- feed idempotency --------------------------------------------------------


def test_a_retried_lease_event_is_not_double_counted(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    seen = []
    manager = LeaseManager(
        FakeClock(), sink=lambda event: (seen.append(event), feed.lease_sink(event))
    )
    manager.acquire(AGENT, Scope.PROJECT, ("proj-9",), ttl=60.0)

    feed.lease_sink(seen[0])  # the node retries delivering the same grant

    rows = log.read(types=["lease.granted"])
    assert len(rows) == 1
    assert rows[0].actor == Actor("node", NODE)


def test_a_retried_presence_report_is_not_double_counted(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    at = datetime(2027, 1, 15, 14, 2, tzinfo=UTC)

    feed.report_presence(active=True, at=at, human_id="operator", ttl=timedelta(minutes=5))
    feed.report_presence(active=True, at=at, human_id="operator", ttl=timedelta(minutes=5))

    assert len(log.read(types=["presence.changed"])) == 1
    # A later refresh (a new time) is a genuinely new event, not a retry.
    feed.report_presence(
        active=True, at=at + timedelta(minutes=2), human_id="operator", ttl=timedelta(minutes=5)
    )
    assert len(log.read(types=["presence.changed"])) == 2


# -- lease board reflects broker activity end to end -------------------------


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
    leases.acquire(AGENT, Scope.PROJECT, (pid,), ttl=600.0)  # an agent takes the project
    locator = ProjectLocator(("Shows",), "Show", pid)
    op_id = submit_op(broker, "timeline.list", {}, target=locator, holder=AGENT)
    broker.run_until_idle()  # the broker holds and releases its own instance lease to run it

    assert broker.result(op_id).status.value == "ok"
    board = lease_board(log, clock=_board_clock(clock()))
    agent_leases = [entry for entry in board.leases if entry.holder_kind == "agent"]
    assert [(entry.scope, entry.key, entry.holder_id) for entry in agent_leases] == [
        ("project", (pid,), "thread-1")
    ]
    # The broker's instance lease was granted and released over the op: both reached the hub.
    instance_grants = [
        event for event in log.read(types=["lease.granted"]) if event.payload["scope"] == "instance"
    ]
    assert instance_grants and log.read(types=["lease.released"])


# -- presence reporter -------------------------------------------------------


class _Guard:
    """A stand-in that returns the decision it is told to, so the reporter's rules are isolated."""

    def __init__(self) -> None:
        self.decision = Decision(False, "warming_up", "warming up")

    def may_switch(self, now: float) -> Decision:
        return self.decision


def _reports(log: EventLog):
    return [
        (event.payload["active"], event.payload["human_id"])
        for event in log.read(types=["presence.changed"])
    ]


def test_reporter_emits_active_when_a_human_appears_and_inactive_when_quiet(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    guard = _Guard()
    reporter = PresenceReporter(guard, feed, human_id="operator", ttl=timedelta(minutes=5))
    wall = datetime(2027, 1, 15, 14, 0, tzinfo=UTC)

    # Warming up: unknown, so nothing is reported.
    assert reporter.sync(1000.0, wall) is False
    assert log.read(types=["presence.changed"]) == []

    guard.decision = Decision(False, "human_context_change", "human changed timeline 2s ago")
    assert reporter.sync(1001.0, wall) is True
    guard.decision = Decision(True, "idle", "no human activity")
    assert reporter.sync(1002.0, wall + timedelta(seconds=2)) is True

    assert _reports(log) == [(True, "operator"), (False, "operator")]


def test_reporter_holds_steady_and_refreshes_only_after_the_interval(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    guard = _Guard()
    guard.decision = Decision(False, "human_input", "input 1s ago")
    reporter = PresenceReporter(
        guard, feed, human_id="operator", ttl=timedelta(minutes=5), refresh_interval=60.0
    )
    base = datetime(2027, 1, 15, 14, 0, tzinfo=UTC)

    assert reporter.sync(1000.0, base) is True  # first active
    assert reporter.sync(1030.0, base + timedelta(seconds=30)) is False  # within interval: steady
    assert reporter.sync(1065.0, base + timedelta(seconds=65)) is True  # past interval: refresh

    reports = log.read(types=["presence.changed"])
    assert [event.payload["active"] for event in reports] == [True, True]
    board = lease_board(log, clock=lambda: base + timedelta(seconds=90))
    # The refresh keeps the original "since" and pushes the expiry out.
    assert [entry.since for entry in board.presence] == [base.isoformat()]


def test_reporter_does_not_report_absent_before_it_ever_reported_present(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = _feed(log)
    guard = _Guard()
    guard.decision = Decision(True, "idle", "no human activity")
    reporter = PresenceReporter(guard, feed, human_id="operator", ttl=timedelta(minutes=5))

    assert reporter.sync(1000.0, datetime(2027, 1, 15, 14, 0, tzinfo=UTC)) is False
    assert log.read(types=["presence.changed"]) == []


def test_reporter_needs_a_positive_ttl() -> None:
    with pytest.raises(ValueError, match="ttl"):
        PresenceReporter(_Guard(), None, ttl=timedelta(0))  # type: ignore[arg-type]
