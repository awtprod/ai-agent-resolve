"""Round-3 feed tests: the validator's d02 probes, restored round-1 tests, reporter pins."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aar.hub.collab import EVENT_LEASE_SNAPSHOT, feed_gaps, project_board
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import HubEventSpec, lease_board
from aar.node.feed import (
    MAX_SPEC_ATTEMPTS,
    EventLogTransport,
    NodeFeed,
    PresenceReporter,
)
from aar.node.leases import Holder, LeaseManager, Scope
from aar.node.presence import Decision

from .fake_resolve import FakeClock

WALL = 1_800_000_000.0
NODE = "node-1"
AGENT = Holder.agent("thread-1")
HUMAN = Holder.human("op")
T0 = datetime(2027, 1, 15, 14, 0, tzinfo=UTC)


class Switch:
    """A transport that can be taken down: raises ConnectionError while ``up`` is false."""

    def __init__(self, log: EventLog) -> None:
        self.up = True
        self.inner = EventLogTransport(log)
        self.delivered: list[HubEventSpec] = []

    def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
        if not self.up:
            raise ConnectionError("hub down")
        self.inner.deliver(spec, node_id=node_id)
        self.delivered.append(spec)


def _wired(log: EventLog, transport, *, outbox_max: int = 4096) -> tuple[NodeFeed, LeaseManager]:
    """A feed wired to a lease table the way a node runtime would (so it can reconcile)."""
    clock = FakeClock()
    holder: list[LeaseManager] = []
    feed = NodeFeed(
        NODE,
        transport,
        wall_offset=lambda: WALL,
        outbox_max=outbox_max,
        lease_source=lambda: holder[0].active(),
        clock=clock,
    )
    holder.append(LeaseManager(clock, sink=feed.lease_sink))
    return feed, holder[0]


def _far_future() -> datetime:
    return datetime.fromtimestamp(WALL + 10**7, UTC)


# -- d02 #2: a dropped release of a ttl=None lease left a phantom, stale forever ----------------


def test_a_dropped_release_leaves_no_phantom_and_the_stale_flag_clears(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    switch = Switch(log)
    feed, leases = _wired(log, switch, outbox_max=3)
    human = leases.acquire(HUMAN, Scope.PROJECT, ("projX",), ttl=None)
    feed.flush()  # the grant (and the start-up snapshot) reach the hub
    assert [lease.holder_id for lease in project_board(log, "projX").leases] == ["op"]

    switch.up = False
    leases.release(human.id, HUMAN)  # queued ...
    for i in range(3):  # ... and pushed out of the 3-slot outbox: the release is lost
        leases.acquire(Holder.agent(f"b{i}"), Scope.PROJECT, (f"q{i}",), ttl=600)
    assert feed.dropped == 1
    switch.up = True

    report = feed.flush()
    assert report.delivered_gap and report.error is None and feed.reconciled
    board = project_board(log, "projX", clock=_far_future)
    assert board.leases == ()  # no phantom human lease, even 115 days later
    assert board.to_dict()["stale"] is False  # the snapshot after the gap reconciled it
    assert feed_gaps(log) == []


def test_a_grant_lost_in_a_gap_is_re_announced(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    switch = Switch(log)
    feed, leases = _wired(log, switch, outbox_max=1)
    switch.up = False
    leases.acquire(AGENT, Scope.PROJECT, ("pA",), ttl=600)  # its grant is pushed out ...
    leases.acquire(Holder.agent("other"), Scope.PROJECT, ("pB",), ttl=600)
    switch.up = True
    feed.flush()
    # ... yet the board shows both live leases, and is not stale.
    board = lease_board(log, clock=lambda: datetime.fromtimestamp(WALL + 1, UTC))
    assert sorted(lease.key[0] for lease in board.leases) == ["pA", "pB"]
    assert project_board(log, "pA").to_dict()["stale"] is False


def test_a_restarted_node_clears_what_its_previous_process_left_on_the_board(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path / "hub.db")
    first, leases = _wired(log, EventLogTransport(log))
    leases.acquire(HUMAN, Scope.PROJECT, ("projR",), ttl=None)
    first.flush()
    assert len(project_board(log, "projR").leases) == 1
    # The node process dies holding the lease; a new process starts with no leases.
    second, _ = _wired(log, EventLogTransport(log))
    assert not second.reconciled
    second.flush()
    assert second.reconciled and project_board(log, "projR").leases == ()


def test_a_lease_granted_after_the_snapshot_stays_on_the_board(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed, leases = _wired(log, EventLogTransport(log))
    feed.flush()  # start-up snapshot: no leases
    leases.acquire(AGENT, Scope.PROJECT, ("pN",), ttl=600)
    feed.flush()
    assert [lease.key for lease in project_board(log, "pN").leases] == [("pN",)]
    assert len(log.read(types=[EVENT_LEASE_SNAPSHOT])) == 1  # no reconcile without a gap


def test_a_snapshot_from_another_writer_is_ignored(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed, leases = _wired(log, EventLogTransport(log))
    leases.acquire(HUMAN, Scope.PROJECT, ("pF",), ttl=None)
    feed.flush()
    forged = {"node_id": NODE, "at": T0.isoformat(), "lease_ids": []}
    log.append(EVENT_LEASE_SNAPSHOT, Actor("agent", NODE), forged)  # not written by the node
    board = project_board(log, "pF")
    assert len(board.leases) == 1
    assert any("not by the node" in item.reason for item in board.skipped)


def test_reconcile_needs_the_lease_source_and_its_clock_together(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    with pytest.raises(ValueError, match="together"):
        NodeFeed(NODE, EventLogTransport(log), lease_source=list)


# -- d02 #4: after a restart the gap key collided and every flush failed ---------------------


def test_gap_keys_never_collide_across_node_restarts(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feeds = []
    for process in range(3):
        feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL, outbox_max=1)
        leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
        leases.acquire(AGENT, Scope.PROJECT, (f"a{process}",), ttl=60)
        leases.acquire(AGENT, Scope.PROJECT, (f"b{process}",), ttl=60)  # drops a{process}
        report = feed.flush()
        assert report.error is None and report.sent == 1 and report.delivered_gap, process
        feeds.append(feed)
    assert len(log.read(types=["feed.gap"])) == 3  # one per process, none refused


def test_a_resent_gap_replays_the_same_key(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    switch = Switch(log)
    feed = NodeFeed(NODE, switch, wall_offset=lambda: WALL, outbox_max=1)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    leases.acquire(AGENT, Scope.PROJECT, ("p0",), ttl=60)
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)
    keys = []

    class Recording:
        def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
            keys.append(spec.idempotency_key)
            raise ConnectionError("down")

    feed._transport = Recording()
    feed.flush()
    feed.flush()
    assert len(keys) == 2 and keys[0] == keys[1]  # the same pending gap, the same key


# -- d02 #5: a spec the hub rejects blocked everything queued behind it ----------------------


def test_a_poison_spec_is_quarantined_and_the_rest_are_delivered(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    real = EventLogTransport(log)

    class Picky:
        def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
            if spec.payload.get("holder", {}).get("id") == "poison":
                raise ValueError("hub rejects this spec")
            real.deliver(spec, node_id=node_id)

    feed = NodeFeed(NODE, Picky(), wall_offset=lambda: WALL)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    leases.acquire(Holder.agent("poison"), Scope.PROJECT, ("p0",), ttl=60)
    for i in range(5):
        leases.acquire(Holder.agent(f"ok{i}"), Scope.PROJECT, (f"p{i + 1}",), ttl=60)

    report = feed.flush()
    assert report.sent == 5 and feed.pending == 0 and "quarantined" in (report.error or "")
    assert [q.type for q in feed.quarantined] == ["lease.granted"]
    assert feed.dropped == 1  # counted, and flagged as a gap on the next flush
    assert feed.flush().delivered_gap
    assert len(log.read(types=["lease.granted"])) == 5


def test_an_unknown_failure_is_retried_then_quarantined(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    real = EventLogTransport(log)
    attempts = []

    class Locked:
        def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
            if spec.payload.get("holder", {}).get("id") == "stuck":
                attempts.append(spec.idempotency_key)
                raise RuntimeError("database is locked")
            real.deliver(spec, node_id=node_id)

    feed = NodeFeed(NODE, Locked(), wall_offset=lambda: WALL)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    leases.acquire(Holder.agent("stuck"), Scope.PROJECT, ("p0",), ttl=60)
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)
    for _ in range(MAX_SPEC_ATTEMPTS - 1):
        report = feed.flush()
        assert report.sent == 0 and feed.pending == 2  # transient: kept, in order
    report = feed.flush()
    assert report.sent == 1 and feed.pending == 0 and len(attempts) == MAX_SPEC_ATTEMPTS


def test_an_unreachable_hub_never_quarantines(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    switch = Switch(log)
    switch.up = False
    feed = NodeFeed(NODE, switch, wall_offset=lambda: WALL)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)
    for _ in range(MAX_SPEC_ATTEMPTS * 3):
        assert feed.flush().sent == 0
    assert feed.pending == 1 and feed.quarantined == () and feed.dropped == 0
    switch.up = True
    assert feed.flush().sent == 1


# -- d02 #1/#3: overflow order and coalescing completeness (no defect; pinned) ---------------


def test_overflow_recovery_sends_the_gap_first_then_the_survivors_in_order(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    switch = Switch(log)
    switch.up = False
    feed = NodeFeed(NODE, switch, wall_offset=lambda: WALL, outbox_max=5)
    clock = FakeClock()
    leases = LeaseManager(clock, sink=feed.lease_sink)
    held = []
    for i in range(4):
        held.append(leases.acquire(Holder.agent(f"a{i}"), Scope.PROJECT, (f"p{i}",), ttl=600))
        clock.advance(1)
    for lease in held:
        leases.release(lease.id, lease.holder)
        clock.advance(1)
    assert (feed.pending, feed.dropped) == (5, 3)
    switch.up = True
    report = feed.flush()
    assert (report.sent, report.delivered_gap, report.error) == (5, True, None)
    assert [spec.type for spec in switch.delivered] == ["feed.gap", "lease.granted"] + [
        "lease.released"
    ] * 4
    assert lease_board(log).leases == ()


# -- restored round-1 tests (deleted in round 2; they killed F03, F05, F07, F13) -------------


def test_a_retried_lease_event_is_not_double_counted(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL)
    seen = []
    manager = LeaseManager(
        FakeClock(), sink=lambda event: (seen.append(event), feed.lease_sink(event))
    )
    manager.acquire(AGENT, Scope.PROJECT, ("proj-9",), ttl=60.0)
    feed.lease_sink(seen[0])  # the node retries the same grant
    assert feed.pending == 2
    feed.flush()
    rows = log.read(types=["lease.granted"])
    assert len(rows) == 1 and rows[0].actor == Actor("node", NODE)


def test_a_retried_presence_report_is_not_double_counted(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL)
    ttl = timedelta(minutes=5)
    feed.report_presence(active=True, at=T0, human_id="operator", ttl=ttl)
    feed.report_presence(active=True, at=T0, human_id="operator", ttl=ttl)
    feed.flush()
    assert len(log.read(types=["presence.changed"])) == 1
    # A later refresh (a new time) is a genuinely new event, not a retry.
    feed.report_presence(active=True, at=T0 + timedelta(minutes=2), human_id="operator", ttl=ttl)
    assert feed.flush().error is None
    assert len(log.read(types=["presence.changed"])) == 2


def test_active_and_inactive_at_the_same_instant_are_two_events(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL)
    feed.report_presence(active=True, at=T0, human_id="op", ttl=timedelta(minutes=5))
    feed.report_presence(active=False, at=T0, human_id="op")
    assert feed.flush().error is None
    assert [e.payload["active"] for e in log.read(types=["presence.changed"])] == [True, False]


class _Guard:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def may_switch(self, now: float) -> Decision:
        return self.decision


_HUMAN_HERE = Decision(False, "human_input", "input 1s ago")
_QUIET = Decision(True, "idle", "no human activity")


def _reporter(log: EventLog, guard: _Guard, **kw) -> tuple[PresenceReporter, NodeFeed]:
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL)
    kw.setdefault("ttl", timedelta(minutes=5))
    return PresenceReporter(guard, feed, human_id="operator", **kw), feed


def _actives(log: EventLog) -> list[bool]:
    return [event.payload["active"] for event in log.read(types=["presence.changed"])]


def test_reporter_holds_steady_and_refreshes_only_after_the_interval(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    reporter, feed = _reporter(log, _Guard(_HUMAN_HERE), refresh_interval=60.0)
    assert reporter.sync(1000.0, T0) is True  # first active
    assert reporter.sync(1030.0, T0 + timedelta(seconds=30)) is False  # within interval
    assert reporter.sync(1059.0, T0 + timedelta(seconds=59)) is False
    assert reporter.sync(1060.0, T0 + timedelta(seconds=60)) is True  # at the interval: due
    assert reporter.sync(1061.0, T0 + timedelta(seconds=61)) is False  # the refresh reset it
    feed.flush()
    assert _actives(log) == [True, True]
    board = lease_board(log, clock=lambda: T0 + timedelta(seconds=90))
    # The refresh keeps the original "since" and its TTL pushes the expiry out.
    assert [(e.since, e.expires_at) for e in board.presence] == [
        (T0.isoformat(), (T0 + timedelta(seconds=60, minutes=5)).isoformat())
    ]


def test_reporter_reports_a_human_returning_right_after_going_quiet(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    guard = _Guard(_HUMAN_HERE)
    reporter, feed = _reporter(log, guard, refresh_interval=60.0)
    assert reporter.sync(1000.0, T0) is True
    guard.decision = _QUIET
    assert reporter.sync(1001.0, T0 + timedelta(seconds=1)) is True  # inactive
    assert reporter.sync(1002.0, T0 + timedelta(seconds=2)) is False  # still quiet: no repeat
    guard.decision = _HUMAN_HERE
    assert reporter.sync(1003.0, T0 + timedelta(seconds=3)) is True  # back within the interval
    feed.flush()
    assert _actives(log) == [True, False, True]


def test_reporter_does_not_report_absent_before_it_ever_reported_present(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    reporter, feed = _reporter(log, _Guard(_QUIET))
    assert reporter.sync(1000.0, T0) is False
    assert reporter.sync(1500.0, T0 + timedelta(seconds=500)) is False
    feed.flush()
    assert log.read(types=["presence.changed"]) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"refresh_interval": 0.0}, {"refresh_interval": 300.0}, {"ttl": timedelta(seconds=-1)}],
    ids=["zero-refresh", "refresh-equal-to-ttl", "negative-ttl"],
)
def test_reporter_refuses_a_refresh_that_cannot_keep_presence_alive(
    tmp_path: Path, kwargs: dict
) -> None:
    log = EventLog(tmp_path / "hub.db")
    with pytest.raises(ValueError):
        _reporter(log, _Guard(_QUIET), **kwargs)


def test_an_unmappable_presence_change_is_gap_flagged_not_raised(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL)
    feed.report_presence(active=True, at=T0, human_id="h" * 500, ttl=timedelta(minutes=5))
    assert feed.pending == 0 and feed.dropped == 1


def test_a_reconcile_the_hub_keeps_refusing_is_given_up_and_never_blocks_the_outbox(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path / "hub.db")
    real = EventLogTransport(log)
    snapshots = []

    class NoSnapshots:
        def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
            if spec.type == EVENT_LEASE_SNAPSHOT:
                snapshots.append(spec.idempotency_key)
                raise ValueError("hub refuses snapshots")
            real.deliver(spec, node_id=node_id)

    feed, leases = _wired(log, NoSnapshots())
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=600)
    for attempt in range(MAX_SPEC_ATTEMPTS + 2):
        report = feed.flush()
        assert feed.pending == 0, attempt  # the outbox is delivered every time regardless
    assert len(snapshots) == MAX_SPEC_ATTEMPTS  # then given up, not retried forever
    assert feed.reconciled and [q.type for q in feed.quarantined] == [EVENT_LEASE_SNAPSHOT]
    assert len(log.read(types=["lease.renewed"])) == MAX_SPEC_ATTEMPTS  # bounded re-sends
    assert report.error is None


# -- more survivor pins: N07 N08 N10 R3F05 R3F08 ----------------------------------------------


def test_a_delivered_gap_is_not_sent_again(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL, outbox_max=1)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    leases.acquire(AGENT, Scope.PROJECT, ("p0",), ttl=60)
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)
    assert feed.flush().delivered_gap is True
    assert feed.flush().delivered_gap is False  # N07: cleared once delivered


def test_two_gaps_in_one_process_both_reach_the_hub(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL, outbox_max=1)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    for batch in range(2):
        leases.acquire(AGENT, Scope.PROJECT, (f"a{batch}",), ttl=60)
        leases.acquire(AGENT, Scope.PROJECT, (f"b{batch}",), ttl=60)  # one more drop
        report = feed.flush()
        assert report.delivered_gap and report.error is None, batch
    gaps = [event.payload["dropped"] for event in log.read(types=["feed.gap"])]
    assert gaps == [1, 2]  # N08: each gap has its own key


def test_coalescing_keeps_the_latest_renewal_of_each_lease(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed(NODE, EventLogTransport(log), wall_offset=lambda: WALL)
    clock = FakeClock()
    leases = LeaseManager(clock, sink=feed.lease_sink)
    a = leases.acquire(AGENT, Scope.PROJECT, ("pa",), ttl=60)
    b = leases.acquire(Holder.agent("other"), Scope.PROJECT, ("pb",), ttl=60)
    clock.advance(1)
    leases.renew(a.id, AGENT)
    leases.renew(b.id, b.holder)
    feed.flush()
    renewed = {event.payload["lease_id"] for event in log.read(types=["lease.renewed"])}
    assert renewed == {a.id, b.id}  # N10: one lease's renewal never drops another's


def test_a_gap_delivered_after_a_reconcile_is_reconciled_again(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    real = EventLogTransport(log)
    refuse_gap = [True]

    class GapOnce:
        def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
            if spec.type == "feed.gap" and refuse_gap[0]:
                refuse_gap[0] = False
                raise ValueError("hub refuses this gap once")
            real.deliver(spec, node_id=node_id)

    feed, leases = _wired(log, GapOnce(), outbox_max=1)
    leases.acquire(AGENT, Scope.PROJECT, ("p0",), ttl=60)
    leases.acquire(AGENT, Scope.PROJECT, ("p1",), ttl=60)  # drops p0's grant
    first = feed.flush()
    assert "gap" in (first.error or "") and first.sent == 1  # R3F08: the outbox still went
    assert feed.flush().delivered_gap is True
    # The gap landed after the first reconcile, so it was reconciled again after it.
    assert feed_gaps(log) == [] and feed.reconciled  # R3F05
