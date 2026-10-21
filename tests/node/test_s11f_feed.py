"""S11f: the feed and the board under faults (F1, F2, F3, LOW 10, LOW 11, LOW 12).

``Faulty`` is a fault-injecting transport into a real hub log: it can be down, lose the
ack of a delivery that committed, hold a delivery and commit it later (a timed-out POST
the hub still commits), deliver twice, and refuse. It pins the feed's retry behaviour
before the remote transport (F-S11-1) exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from starlette.testclient import TestClient

from aar.hub.artifacts import ArtifactStore
from aar.hub.collab import project_board
from aar.hub.console import ConsoleConfig, build_app
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import (
    EVENT_FEED_CLAIMED,
    EVENT_FEED_SUPERSEDED,
    EVENT_LEASE_SNAPSHOT,
    SUPERSEDED_STALE,
    FeedConflict,
    HubEventSpec,
    deliver_feed_event,
    lease_board,
)
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.node.feed import (
    BACKOFF_FIRST,
    BACKOFF_MAX,
    MAX_SPEC_ATTEMPTS,
    EventLogTransport,
    NodeFeed,
    wire_feed,
)
from aar.node.leases import Holder, LeaseConflict, LeaseManager, LeaseNotHeld, Scope

from .fake_resolve import FakeClock

WALL = 1_800_000_000.0
NODE = "node-1"
HUMAN = Holder.human("op")


class Faulty:
    """A transport into ``log`` that misbehaves on demand (see the module docstring)."""

    def __init__(
        self,
        log: EventLog,
        rng: random.Random | None = None,
        hub_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.inner = EventLogTransport(log, clock=hub_clock or (lambda: datetime.now(UTC)))
        self.rng = rng or random.Random(0)
        self.down = False
        self.ack_loss = 0.0  # chance a committed delivery's ack is lost
        self.lose_first_ack = False  # lose the ack of every key's first delivery
        self.late = 0.0  # chance a delivery is held and committed later
        self.hold: Callable[[HubEventSpec], bool] = lambda spec: False
        self.duplicate = False
        self.refuse: Callable[[HubEventSpec], bool] = lambda spec: False
        self.held: list[tuple[HubEventSpec, str]] = []
        self.attempts: list[HubEventSpec] = []
        self._seen: set[str | None] = set()

    def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
        self.attempts.append(spec)
        if self.down:
            raise ConnectionError("hub unreachable")
        if self.refuse(spec):
            raise ValueError("the hub refuses this spec")
        if self.hold(spec) or self.rng.random() < self.late:
            self.held.append((spec, node_id))
            raise TimeoutError("timed out; the hub will still commit it, later")
        self.inner.deliver(spec, node_id=node_id)
        if self.duplicate:
            self.inner.deliver(spec, node_id=node_id)
        first = spec.idempotency_key not in self._seen
        self._seen.add(spec.idempotency_key)
        if (self.lose_first_ack and first) or self.rng.random() < self.ack_loss:
            raise TimeoutError("committed, but the ack was lost")

    def land(self, count: int | None = None) -> None:
        """Commit held deliveries now (all, or the oldest ``count``), as a slow hub would."""
        landing, self.held = self.held[:count], self.held[len(self.held[:count]) :]
        for spec, node_id in landing:
            # A replaced process's late delivery is refused at the hub, as it should be.
            with contextlib.suppress(FeedConflict):
                self.inner.deliver(spec, node_id=node_id)


def _wired(log: EventLog, transport: Any, clock: FakeClock, **kw: Any):
    return wire_feed(NODE, transport, clock=clock, wall_offset=lambda: WALL, **kw)


def _at(clock: FakeClock, later: float = 0.0) -> Callable[[], datetime]:
    return lambda: datetime.fromtimestamp(WALL + clock() + later, UTC)


def _drain(feed: NodeFeed, clock: FakeClock, flushes: int = 50) -> int:
    """Flush until the feed is reconciled and its outbox is empty; returns flushes used."""
    for used in range(1, flushes + 1):
        feed.flush()
        if feed.reconciled and feed.pending == 0:
            return used
        clock.advance(BACKOFF_MAX)
    raise AssertionError(f"the feed did not drain in {flushes} flushes")


def _holders(log: EventLog, project: str, clock: FakeClock) -> list[str]:
    return sorted(lease.holder_id for lease in project_board(log, project, clock=_at(clock)).leases)


# -- F1: a live lease stays on the board after a reconcile -----------------------------------


def test_a_live_lease_stays_on_every_view_after_a_reconcile(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    feed, leases = _wired(log, transport, clock, outbox_max=2)
    leases.acquire(HUMAN, Scope.PROJECT, ("pLive",), ttl=None)
    _drain(feed, clock)
    transport.down = True
    for i in range(3):  # overflow the 2-slot outbox with other grants: a gap
        leases.acquire(Holder.agent(f"b{i}"), Scope.PROJECT, (f"q{i}",), ttl=600)
    assert feed.dropped == 1
    transport.down = False
    clock.advance(BACKOFF_MAX)
    _drain(feed, clock)
    assert len(log.read(types=[EVENT_LEASE_SNAPSHOT])) == 2  # the start-up one and this one
    # The snapshot lists the human's lease, so the board keeps it; the gap is reconciled.
    board = project_board(log, "pLive", clock=_at(clock, 10**7))
    assert [lease.holder_id for lease in board.leases] == ["op"]
    assert board.to_dict()["stale"] is False
    every = lease_board(log, clock=_at(clock))
    assert sorted(lease.key[0] for lease in every.leases) == ["pLive", "q0", "q1", "q2"]


# -- F2: every view of the leases is reconciled, and says when it is stale -------------------


def _console_leases(log: EventLog, tmp_path: Path, clock: FakeClock) -> str:
    app = build_app(
        log,
        ArtifactStore(tmp_path / "artifacts", log),
        ConsoleConfig(operator="operator-1", csrf_secret=b"0123456789abcdef0123456789abcdef"),
        clock=_at(clock, 5),
    )
    client = TestClient(app, base_url="http://127.0.0.1:8788", client=("127.0.0.1", 50000))
    return client.get("/leases").text


def _mcp_lease_board(log: EventLog, tmp_path: Path, clock: FakeClock) -> dict[str, Any]:
    server = build_server(
        log, Registry(log), ArtifactStore(tmp_path / "mcp-artifacts", log), clock=_at(clock, 5)
    )

    async def body() -> dict[str, Any]:
        async with Client(server) as client:
            return json.loads((await client.call_tool("lease_board", {})).content[0].text)

    return asyncio.run(body())


def test_the_console_and_the_mcp_lease_board_drop_the_phantom_too(tmp_path: Path) -> None:
    """The validator's a_console_phantom: a human lease whose release was dropped."""
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    feed, leases = _wired(log, transport, clock, outbox_max=3)
    human = leases.acquire(HUMAN, Scope.PROJECT, ("projX",), ttl=None)
    feed.flush()
    transport.down = True
    clock.advance(1)
    leases.release(human.id, HUMAN)  # queued, then pushed out of the outbox: lost
    for i in range(3):
        leases.acquire(Holder.agent(f"b{i}"), Scope.PROJECT, (f"q{i}",), ttl=600)
        clock.advance(1)
    # While the hub cannot hear from the node, it still shows the human's lease.
    assert "projX" in _console_leases(log, tmp_path, clock)
    transport.down = False
    clock.advance(BACKOFF_MAX)
    _drain(feed, clock)  # the gap, then the reconcile
    page = _console_leases(log, tmp_path, clock)
    assert "projX" not in page and "q0" in page and "This view may be stale" not in page
    data = _mcp_lease_board(log, tmp_path, clock)
    assert sorted(lease["key"][0] for lease in data["leases"]) == ["q0", "q1", "q2"]
    assert data["stale"] is False and data["gaps"] == []


def test_an_unreconciled_gap_shows_as_stale_on_the_console_and_the_mcp_tool(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    transport.refuse = lambda spec: spec.type == EVENT_LEASE_SNAPSHOT  # it never reconciles
    feed, leases = _wired(log, transport, clock, outbox_max=1)
    leases.acquire(HUMAN, Scope.PROJECT, ("projX",), ttl=None)
    leases.acquire(Holder.agent("b"), Scope.PROJECT, ("q",), ttl=600)  # drops the first grant
    for _ in range(MAX_SPEC_ATTEMPTS + 1):
        feed.flush()
    page = _console_leases(log, tmp_path, clock)
    assert "This view may be stale" in page and "dropped 1 feed event" in page
    data = _mcp_lease_board(log, tmp_path, clock)
    assert data["stale"] is True and [gap["node_id"] for gap in data["gaps"]] == [NODE]
    assert project_board(log, "projX").to_dict()["stale"] is True


# -- F3: a reconcile is safe to retry --------------------------------------------------------


def test_a_resent_reconcile_reuses_its_keys(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    feed, leases = _wired(log, transport, clock, outbox_max=1)
    feed.flush()
    a = leases.acquire(Holder.agent("a"), Scope.PROJECT, ("pa",), ttl=600)
    feed.flush()
    transport.down = True
    leases.acquire(Holder.agent("b"), Scope.PROJECT, ("pb",), ttl=600)
    leases.renew(a.id, a.holder, ttl=600)  # pushes b's grant out: a gap
    transport.down = False
    transport.hold = lambda spec: spec.type == EVENT_LEASE_SNAPSHOT  # its POST times out
    for _ in range(3):
        feed.flush()
        clock.advance(BACKOFF_MAX)
    snapshots = [s for s in transport.attempts if s.type == EVENT_LEASE_SNAPSHOT]
    assert len(snapshots) == 4  # the start-up one, then three tries of the same reconcile
    assert len({s.idempotency_key for s in snapshots[1:]}) == 1
    assert len({json.dumps(s.payload, sort_keys=True) for s in snapshots[1:]}) == 1
    reannounced = [
        s.idempotency_key for s in transport.attempts if "reannounced" in str(s.idempotency_key)
    ]
    assert len(reannounced) == 1  # sent once (it went through), never re-minted
    transport.hold = lambda spec: False
    transport.land()
    _drain(feed, clock)
    assert len(log.read(types=[EVENT_LEASE_SNAPSHOT])) == 2  # the held one landed once


def test_a_snapshot_that_commits_late_never_erases_a_lease_granted_after_it(
    tmp_path: Path,
) -> None:
    """The validator's a_feed_attacks #2."""
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    transport.hold = lambda spec: spec.type == EVENT_LEASE_SNAPSHOT and not transport.held
    feed, leases = _wired(log, transport, clock)
    assert "TimeoutError" in (feed.flush().error or "")  # the start-up snapshot times out
    leases.acquire(Holder.agent("L2"), Scope.PROJECT, ("P",), ttl=600)
    clock.advance(BACKOFF_MAX)
    transport.hold = lambda spec: False
    _drain(feed, clock)
    assert _holders(log, "P", clock) == ["L2"]
    transport.land()  # the first snapshot (no leases) lands after L2's grant
    assert _holders(log, "P", clock) == ["L2"]
    assert project_board(log, "P", clock=_at(clock)).to_dict()["stale"] is False


def test_a_late_snapshot_never_clears_a_gap_that_came_after_it(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    transport.hold = lambda spec: spec.type == EVENT_LEASE_SNAPSHOT and not transport.held
    feed, leases = _wired(log, transport, clock, outbox_max=1)
    feed.flush()  # the start-up snapshot is in flight
    transport.hold = lambda spec: spec.type == EVENT_LEASE_SNAPSHOT  # so is every later one
    leases.acquire(Holder.agent("a"), Scope.PROJECT, ("pa",), ttl=600)
    leases.acquire(Holder.agent("b"), Scope.PROJECT, ("pb",), ttl=600)  # a drop: a new gap
    clock.advance(BACKOFF_MAX)
    feed.flush()  # the gap lands; the new reconcile's snapshot is held too
    transport.land(1)  # only the old start-up snapshot lands, after the gap
    assert project_board(log, "pa", clock=_at(clock)).to_dict()["stale"] is True


def test_lost_acks_never_starve_the_outbox_nor_double_an_event(tmp_path: Path) -> None:
    """The validator's a_feed_attacks #4: every first delivery of a key loses its ack."""
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    transport.lose_first_ack = True
    feed, leases = _wired(log, transport, clock)
    x = leases.acquire(Holder.agent("x"), Scope.PROJECT, ("P",), ttl=600)
    leases.renew(x.id, x.holder, ttl=600)
    leases.release(x.id, x.holder)
    used = _drain(feed, clock)
    types = [event.type for event in log.read()]
    assert types.count("lease.snapshot") == 1
    assert [t for t in types if t.startswith("lease.") and t != "lease.snapshot"] == [
        "lease.granted",
        "lease.renewed",
        "lease.released",
    ]
    assert used <= 2 * 4  # each spec costs at most two tries: the lost ack, then the resend


@pytest.mark.parametrize("loss", [0.1, 0.2])
def test_a_lossy_link_drains_and_the_hub_log_does_not_grow_per_flush(
    tmp_path: Path, loss: float
) -> None:
    """The validator's a_feed_lossy: 50 live leases, a share of acks lost after commit."""
    for seed in range(5):
        log = EventLog(tmp_path / f"lossy-{loss}-{seed}.db")
        clock = FakeClock()
        transport = Faulty(log, random.Random(seed))
        transport.ack_loss = loss
        feed, leases = _wired(log, transport, clock)
        for i in range(50):
            leases.acquire(Holder.agent(f"a{i}"), Scope.TIMELINE, ("P", f"tl{i}"), ttl=3600)
        leases.acquire(Holder.agent("late"), Scope.PROJECT, ("Q",), ttl=3600)
        used = _drain(feed, clock, flushes=200)
        assert used < 100, (seed, used)
        # Exactly the ideal: 51 grants and one snapshot, however many resends it took.
        assert len(log.read(limit=10_000)) == 51 + 1 + 1, seed  # + the feed's claim


def test_a_duplicate_delivery_adds_nothing(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    transport.duplicate = True
    feed, leases = _wired(log, transport, clock)
    lease = leases.acquire(Holder.agent("a"), Scope.PROJECT, ("P",), ttl=600)
    leases.release(lease.id, lease.holder)
    _drain(feed, clock)
    assert [e.type for e in log.read()] == [
        EVENT_FEED_CLAIMED,
        EVENT_LEASE_SNAPSHOT,
        "lease.granted",
        "lease.released",
    ]


def test_an_unreachable_hub_backs_off_boundedly_and_drops_nothing(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    transport.down = True
    feed, leases = _wired(log, transport, clock)
    leases.acquire(Holder.agent("a"), Scope.PROJECT, ("P",), ttl=600)
    delays = []
    for _ in range(10):
        before = len(transport.attempts)
        report = feed.flush()
        assert len(transport.attempts) == before + 1 and report.retry_at is not None
        delays.append(report.retry_at - clock())
        assert feed.flush().sent == 0 and len(transport.attempts) == before + 1  # backing off
        clock.advance(delays[-1])
    assert delays == [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]
    assert delays[0] == BACKOFF_FIRST and max(delays) == BACKOFF_MAX
    assert feed.pending == 1 and feed.dropped == 0 and feed.quarantined == ()
    transport.down = False
    report = feed.flush()
    assert report.sent == 2 and report.retry_at is None and report.error is None
    transport.down = True
    leases.acquire(Holder.agent("b"), Scope.PROJECT, ("P2",), ttl=600)
    assert feed.flush().retry_at == clock() + BACKOFF_FIRST  # a fresh outage starts small


def test_a_flush_that_got_something_through_backs_off_from_the_start(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    feed, leases = _wired(log, transport, clock)
    transport.down = True
    leases.acquire(Holder.agent("a"), Scope.PROJECT, ("P",), ttl=600)
    for _ in range(4):  # backing off 1, 2, 4, 8 seconds
        feed.flush()
        clock.advance(BACKOFF_MAX)
    transport.down = False
    transport.lose_first_ack = True
    report = feed.flush()  # the snapshot's first try loses its ack: nothing through
    assert report.sent == 0 and report.retry_at - clock() == 16  # so it doubles again
    clock.advance(BACKOFF_MAX)
    report = feed.flush()  # the snapshot's resend goes through, the grant's ack is lost
    assert report.sent == 1 and report.retry_at - clock() == BACKOFF_FIRST


def test_a_refused_gap_still_reconciles_away_the_phantom(tmp_path: Path) -> None:
    """V27: a drop asks for a reconcile even when the hub refuses the gap itself."""
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    transport.refuse = lambda spec: spec.type == "feed.gap"
    feed, leases = _wired(log, transport, clock, outbox_max=3)
    human = leases.acquire(HUMAN, Scope.PROJECT, ("projX",), ttl=None)
    feed.flush()
    transport.down = True
    leases.release(human.id, HUMAN)
    for i in range(3):
        leases.acquire(Holder.agent(f"b{i}"), Scope.PROJECT, (f"q{i}",), ttl=600)
    transport.down = False
    clock.advance(BACKOFF_MAX)
    report = feed.flush()
    assert "gap refused" in (report.error or "")
    assert _holders(log, "projX", clock) == []  # the snapshot still went: no phantom
    for _ in range(MAX_SPEC_ATTEMPTS):
        feed.flush()
    assert [q.type for q in feed.quarantined] == ["feed.gap"]  # retried, then given up


# -- LOW 10: two live feeds with one node id --------------------------------------------------


def test_a_second_live_feed_with_the_same_node_id_is_refused_loudly(tmp_path: Path) -> None:
    """The validator's a_feed_attacks #1."""
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = EventLogTransport(log, clock=_at(clock))
    fa, la = _wired(log, transport, clock)
    la.acquire(Holder.agent("A"), Scope.PROJECT, ("P",), ttl=600)
    fa.flush()
    fb, lb = _wired(log, transport, clock)  # a second process with the same node id
    lb.acquire(Holder.agent("B"), Scope.PROJECT, ("Q",), ttl=600)
    fb.flush()  # the hub admits it as the node's newer process
    la.acquire(Holder.agent("A2"), Scope.PROJECT, ("P2",), ttl=600)
    report = fa.flush()
    assert report.stopped and "replaced" in (report.error or "") and fa.stopped
    assert fa.flush().stopped and fa.flush().sent == 0  # it stays stopped
    # A's snapshot never got to erase B's lease; the board is stale and says why.
    assert _holders(log, "Q", clock) == ["B"]
    assert _holders(log, "P2", clock) == []  # A's events are refused
    board = project_board(log, "Q", clock=_at(clock)).to_dict()
    assert board["stale"] is True
    assert [(s["nonce"], s["owner"]) for s in board["superseded"]] == [(fa.nonce, fb.nonce)]
    refusals = log.read(types=[EVENT_FEED_SUPERSEDED])
    assert len(refusals) == 1  # recorded once, however often A tries
    # The flag clears when B next reconciles, or after the window at the latest.
    later = _at(clock, SUPERSEDED_STALE.total_seconds())
    assert project_board(log, "Q", clock=later).to_dict()["stale"] is False
    fb._reconcile_pending = True  # (as after a gap of B's own)
    fb.flush()
    assert project_board(log, "Q", clock=_at(clock)).to_dict()["stale"] is False


def test_events_of_a_replaced_feed_written_by_hand_are_skipped(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    fa, _ = _wired(log, EventLogTransport(log), clock)
    fa.flush()
    fb, lb = _wired(log, EventLogTransport(log), clock)
    lb.acquire(Holder.agent("B"), Scope.PROJECT, ("Q",), ttl=600)
    fb.flush()
    # A's snapshot, appended around the hub's acceptor: the fold refuses it as well.
    forged = {"node_id": NODE, "at": "2027-01-01T00:00:00+00:00", "lease_ids": []}
    forged["feed"] = {"nonce": fa.nonce, "seq": 10**6}
    log.append(EVENT_LEASE_SNAPSHOT, Actor("node", NODE), forged)
    board = lease_board(log, clock=_at(clock))
    assert [lease.holder_id for lease in board.leases] == ["B"]
    assert any("a newer one replaced" in item.reason for item in board.skipped)
    never = {**forged, "feed": {"nonce": "f" * 16, "seq": 1}}
    log.append(EVENT_LEASE_SNAPSHOT, Actor("node", NODE), never)
    board = lease_board(log, clock=_at(clock))
    assert [lease.holder_id for lease in board.leases] == ["B"]
    assert any("never admitted" in item.reason for item in board.skipped)


def test_a_restarted_node_takes_over_without_a_conflict(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    first, leases = _wired(log, EventLogTransport(log), clock)
    leases.acquire(HUMAN, Scope.PROJECT, ("projR",), ttl=None)
    first.flush()
    second, _ = _wired(log, EventLogTransport(log), clock)  # the first process died
    second.flush()
    board = project_board(log, "projR", clock=_at(clock)).to_dict()
    assert board["leases"] == [] and board["stale"] is False and board["superseded"] == []


def test_the_acceptor_dedupes_a_resend_and_refuses_a_bad_stamp(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    stamp = {"nonce": "a" * 16, "seq": 1}
    spec = HubEventSpec(
        EVENT_LEASE_SNAPSHOT,
        {"node_id": NODE, "at": "2027-01-01T00:00:00+00:00", "lease_ids": [], "feed": stamp},
        "snap-1",
    )
    first = deliver_feed_event(log, spec, NODE)
    assert deliver_feed_event(log, spec, NODE).id == first.id
    assert [e.type for e in log.read()] == [EVENT_FEED_CLAIMED, EVENT_LEASE_SNAPSHOT]
    for bad in ({"nonce": "", "seq": 1}, {"nonce": "a", "seq": 0}, {"nonce": "a", "seq": True}, []):
        with pytest.raises(ValueError, match="stamp"):
            deliver_feed_event(log, HubEventSpec(spec.type, {**spec.payload, "feed": bad}), NODE)


# -- LOW 11: a quarantined presence change never leaves a human shown as active -------------


def test_a_quarantined_inactive_report_is_re_announced_by_the_reconcile(tmp_path: Path) -> None:
    """The validator's a_feed_attacks #3: a busy hub database while a human goes quiet."""
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    real = EventLogTransport(log)
    busy = [0]

    class Busy:
        def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
            if busy[0] and spec.type == "presence.changed" and not spec.payload["active"]:
                busy[0] -= 1
                raise sqlite3.OperationalError("database is locked")
            real.deliver(spec, node_id=node_id)

    feed, _ = _wired(log, Busy(), clock)
    at = _at(clock)
    feed.report_presence(active=True, at=at(), human_id="op", ttl=timedelta(minutes=10))
    feed.flush()
    busy[0] = MAX_SPEC_ATTEMPTS
    feed.report_presence(active=False, at=at() + timedelta(seconds=1), human_id="op")
    for _ in range(MAX_SPEC_ATTEMPTS + 1):
        feed.flush()
        clock.advance(BACKOFF_MAX)
    assert [q.type for q in feed.quarantined] == ["presence.changed"]
    board = project_board(log, "P", clock=_at(clock, 5))
    assert board.presence == () and board.to_dict()["stale"] is False


def test_an_older_presence_report_landing_late_never_overrides_a_newer_one(
    tmp_path: Path,
) -> None:
    """A reconcile's re-announced presence goes before the queued reports; by the node's
    own order the newest report stands, whatever order the hub commits them in."""
    log = EventLog(tmp_path / "hub.db")
    at = datetime(2027, 1, 15, 14, 0, tzinfo=UTC)
    stamp = {"nonce": "a" * 16}
    active = {"node_id": NODE, "active": True, "human_id": "op", "at": at.isoformat()}
    active["expires_at"] = (at + timedelta(minutes=10)).isoformat()
    quiet = {"node_id": NODE, "active": False, "human_id": "op", "expires_at": None}
    quiet["at"] = (at + timedelta(seconds=1)).isoformat()
    newer = HubEventSpec("presence.changed", {**quiet, "feed": {**stamp, "seq": 2}}, "quiet")
    older = HubEventSpec("presence.changed", {**active, "feed": {**stamp, "seq": 1}}, "active")
    deliver_feed_event(log, newer, NODE)
    deliver_feed_event(log, older, NODE)  # lands last
    assert lease_board(log, clock=lambda: at + timedelta(seconds=5)).presence == ()


# -- LOW 12: a feed cannot be built in a mode that never reconciles ---------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clock": FakeClock()},
        {"lease_source": list},
        {"lease_source": None, "clock": FakeClock()},
        {"lease_source": list, "clock": None},
    ],
    ids=["no-lease-source", "no-clock", "lease-source-none", "clock-none"],
)
def test_a_feed_needs_its_lease_source_and_clock(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(TypeError, match=r"lease_source and clock are required|missing"):
        NodeFeed(NODE, EventLogTransport(EventLog(tmp_path / "hub.db")), **kwargs)


def test_wire_feed_binds_the_lease_table_and_the_feed_on_one_clock(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    feed, leases = wire_feed(NODE, EventLogTransport(log), clock=clock, wall_offset=lambda: WALL)
    assert isinstance(leases, LeaseManager)
    lease = leases.acquire(HUMAN, Scope.PROJECT, ("pW",), ttl=None)
    feed.flush()
    assert log.read(types=[EVENT_LEASE_SNAPSHOT])[0].payload["lease_ids"] == [lease.id]


# -- the fault fuzz: after the faults stop and the feed drains, every view is exact ----------

PROJECTS = [f"p{i}" for i in range(4)]


def _fuzz(seed: int, tmp_path: Path) -> list[str]:
    rng = random.Random(seed)
    log = EventLog(tmp_path / f"fuzz-{seed}.db")
    clock = FakeClock()
    transport = Faulty(log, rng, hub_clock=_at(clock))

    def process() -> tuple[NodeFeed, LeaseManager]:
        return _wired(log, transport, clock, outbox_max=rng.choice([2, 3, 5, 50]))

    feed, leases = process()
    holders = [Holder.agent(f"a{i}") for i in range(3)] + [HUMAN]
    for _ in range(rng.randint(5, 60)):
        transport.down = rng.random() < 0.25
        transport.ack_loss = rng.choice([0.0, 0.0, 0.2])
        transport.late = rng.choice([0.0, 0.0, 0.1])
        transport.duplicate = rng.random() < 0.1
        refusing = rng.random() < 0.1
        transport.refuse = lambda spec, refusing=refusing: refusing and rng.random() < 0.3
        step = rng.random()
        try:
            if step < 0.3:
                holder = rng.choice(holders)
                ttl = None if holder.is_human and rng.random() < 0.5 else rng.choice([5, 30, 600])
                leases.acquire(holder, Scope.PROJECT, (rng.choice(PROJECTS),), ttl=ttl)
            elif step < 0.42:
                live = leases.active()
                if live:
                    lease = rng.choice(live)
                    ttl = None if lease.ttl is None else rng.choice([5, 30, 600])
                    leases.renew(lease.id, lease.holder, ttl=ttl)
            elif step < 0.57:
                live = leases.active()
                if live:
                    lease = rng.choice(live)
                    leases.release(lease.id, lease.holder)
            elif step < 0.85:
                feed.flush()
            elif step < 0.9:
                transport.land(rng.randint(0, 3))  # some held deliveries commit, late
            elif step < 0.96:
                clock.advance(rng.choice([1, 10, 40, 70]))
            else:
                feed, leases = process()  # a restart: the old process dies with its state
        except (LeaseConflict, LeaseNotHeld):
            pass
        clock.advance(0.5)
    transport.down, transport.ack_loss, transport.late, transport.duplicate = False, 0, 0, False
    transport.refuse = lambda spec: False
    transport.land()  # whatever was still in flight commits now, late
    for _ in range(4):
        feed.flush()
        clock.advance(BACKOFF_MAX)
    problems = []
    gave_up = any(q.type == EVENT_LEASE_SNAPSHOT for q in feed.quarantined)
    # A dead process's delivery that committed after a restart is refused and marks the
    # node stale for a while; once that window is over, a clean drain is never stale.
    for later in (0.0, SUPERSEDED_STALE.total_seconds()):
        now = _at(clock, later)
        every = lease_board(log, clock=now)
        for project in PROJECTS:
            board = project_board(log, project, clock=now)
            on_board = {lease.lease_id for lease in board.leases}
            mine = {
                lease.id
                for lease in leases.active()
                if lease.key == (project,)
                and (lease.expires_at is None or lease.expires_at > clock() + later)
            }
            stale = board.to_dict()["stale"]
            if not stale and on_board != mine:
                problems.append(f"{project}: phantom {on_board - mine} lost {mine - on_board}")
            if stale and not (gave_up or (later == 0.0 and board.superseded)):
                problems.append(f"{project}: stale after a clean drain (+{later}s)")
            in_every = {lease.lease_id for lease in every.leases if lease.key == (project,)}
            if in_every != on_board:
                problems.append(f"{project}: lease_board and project_board disagree")
    return problems


@pytest.mark.parametrize("block", range(3))
def test_feed_fault_fuzz_every_view_matches_the_node_after_a_clean_drain(
    tmp_path: Path, block: int
) -> None:
    failures = {}
    for seed in range(block * 50, block * 50 + 50):
        problems = _fuzz(seed, tmp_path)
        if problems:
            failures[seed] = problems
    assert failures == {}


def test_a_gap_that_grows_while_pending_reports_its_latest_count(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    feed, leases = _wired(log, transport, clock, outbox_max=1)
    feed.flush()
    transport.down = True
    for i in range(2):
        leases.acquire(Holder.agent(f"a{i}"), Scope.PROJECT, (f"p{i}",), ttl=600)  # 1 drop
    feed.flush()  # the gap (1 dropped) cannot be sent
    clock.advance(BACKOFF_MAX)
    leases.acquire(Holder.agent("a2"), Scope.PROJECT, ("p2",), ttl=600)  # a 2nd drop
    transport.down = False
    _drain(feed, clock)
    gaps = [e.payload["dropped"] for e in log.read(types=["feed.gap"])]
    assert gaps == [2]  # the pending gap was re-made with the new count (and a new key)


def test_a_dead_processs_late_resend_after_a_restart_is_answered_not_refused(
    tmp_path: Path,
) -> None:
    """A resend of a spec already in the log is answered with the stored event, even from a
    process the hub has since replaced: it adds nothing and marks nothing stale."""
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log, hub_clock=_at(clock))
    first, leases = _wired(log, transport, clock)
    first.flush()
    transport.lose_first_ack = True
    leases.acquire(Holder.agent("a"), Scope.PROJECT, ("P",), ttl=600)
    first.flush()  # the grant commits; its ack is lost
    transport.lose_first_ack = False
    transport.hold = lambda spec: spec.type == "lease.granted"
    clock.advance(BACKOFF_MAX)
    first.flush()  # the resend is in flight when the process dies
    transport.hold = lambda spec: False
    second, _ = _wired(log, EventLogTransport(log, clock=_at(clock)), clock)
    second.flush()  # the restarted process takes over
    transport.land()  # the dead process's resend lands now
    assert log.read(types=[EVENT_FEED_SUPERSEDED]) == []
    assert len(log.read(types=["lease.granted"])) == 1
    assert project_board(log, "P", clock=_at(clock)).to_dict()["stale"] is False


def test_a_hand_written_re_claim_never_brings_back_a_replaced_process(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    fa, _ = _wired(log, EventLogTransport(log), clock)
    fa.flush()
    fb, lb = _wired(log, EventLogTransport(log), clock)
    lb.acquire(Holder.agent("B"), Scope.PROJECT, ("Q",), ttl=600)
    fb.flush()
    claim = {"node_id": NODE, "nonce": fa.nonce, "at": "2027-01-01T00:00:00+00:00"}
    log.append(EVENT_FEED_CLAIMED, Actor("node", NODE), claim)  # by hand, around the acceptor
    snapshot = {"node_id": NODE, "at": "2027-01-01T00:00:00+00:00", "lease_ids": []}
    snapshot["feed"] = {"nonce": fa.nonce, "seq": 10**6}
    log.append(EVENT_LEASE_SNAPSHOT, Actor("node", NODE), snapshot)
    board = lease_board(log, clock=_at(clock))
    assert [lease.holder_id for lease in board.leases] == ["B"]
    assert any("already admitted" in item.reason for item in board.skipped)


def test_a_refused_second_process_shows_as_stale_on_every_view(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = EventLogTransport(log, clock=_at(clock, 5))
    fa, _ = _wired(log, transport, clock)
    fa.flush()
    fb, _ = _wired(log, transport, clock)
    fb.flush()
    fa._reconcile_pending = True  # A has something to say again
    assert fa.flush().stopped
    assert lease_board(log, clock=_at(clock, 5)).stale is True
    assert project_board(log, "P", clock=_at(clock, 5)).to_dict()["stale"] is True
    page = _console_leases(log, tmp_path, clock)
    assert "This view may be stale" in page and "a second feed process" in page
    data = _mcp_lease_board(log, tmp_path, clock)
    assert data["stale"] is True and [s["nonce"] for s in data["superseded"]] == [fa.nonce]


def test_a_late_older_gap_never_replaces_the_newer_gaps_count(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    stamp = {"nonce": "a" * 16}
    gap = {"node_id": NODE, "at": "2027-01-15T14:00:00+00:00"}
    newer = HubEventSpec("feed.gap", {**gap, "dropped": 3, "feed": {**stamp, "seq": 9}}, "g3")
    older = HubEventSpec("feed.gap", {**gap, "dropped": 1, "feed": {**stamp, "seq": 4}}, "g1")
    deliver_feed_event(log, newer, NODE)
    deliver_feed_event(log, older, NODE)  # lands late
    assert [g.dropped for g in lease_board(log).gaps] == [3]


def test_a_feed_is_not_reconciled_while_its_reconcile_is_still_in_flight(tmp_path: Path) -> None:
    # S43: the gap is through, but the reconcile's snapshot is not; the feed must say so.
    log = EventLog(tmp_path / "hub.db")
    clock = FakeClock()
    transport = Faulty(log)
    feed, leases = _wired(log, transport, clock, outbox_max=1)
    _drain(feed, clock)
    leases.acquire(Holder.agent("a"), Scope.PROJECT, ("pa",), ttl=600)
    leases.acquire(Holder.agent("b"), Scope.PROJECT, ("pb",), ttl=600)  # a drop
    transport.hold = lambda spec: spec.type == EVENT_LEASE_SNAPSHOT
    report = feed.flush()
    assert report.delivered_gap and report.retry_at is not None
    assert feed.pending == 1 and not feed.reconciled
    transport.hold = lambda spec: False
    clock.advance(BACKOFF_MAX)
    feed.flush()
    assert feed.reconciled and feed.pending == 0
