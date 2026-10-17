"""Tests for the hub lease board: node lease events mapped to hub events, and the projection."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from aar.hub.common import InvalidInput
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import (
    current_wall_offset,
    lease_board,
    lease_event_to_hub,
    presence_to_hub,
    record_lease_event,
    record_presence,
)
from aar.node.leases import Holder, LeaseEvent, LeaseManager, Scope

WALL_OFFSET = 1_800_000_000.0  # Unix time minus the node's monotonic clock
NODE = "node-1"
PROJECT = "demo-show"
AGENT = Holder.agent("agent-1")
HUMAN = Holder.human("operator-1")
START = 1000.0  # the node's monotonic seconds when a scenario begins
T0 = datetime(2027, 1, 15, 8, 0, tzinfo=UTC)


class _Mono:
    """A hand-wound monotonic clock for a node's LeaseManager."""

    def __init__(self, now: float = START) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _node() -> tuple[LeaseManager, _Mono, list[LeaseEvent]]:
    """A real LeaseManager on a fake clock, with every event it emits collected."""
    clock = _Mono()
    events: list[LeaseEvent] = []
    return LeaseManager(clock=clock, sink=events.append), clock, events


def _wall(mono: float) -> str:
    return datetime.fromtimestamp(mono + WALL_OFFSET, UTC).isoformat()


def _at(mono: float) -> Callable[[], datetime]:
    """A board clock reading the wall time of node-monotonic second ``mono``."""
    return lambda: datetime.fromtimestamp(mono + WALL_OFFSET, UTC)


def _map(event: LeaseEvent, node_id: str = NODE) -> Any:
    return lease_event_to_hub(event, node_id, wall_offset=WALL_OFFSET)


def _record(log: EventLog, events: list[LeaseEvent], node_id: str = NODE) -> None:
    for event in events:
        record_lease_event(log, event, node_id, wall_offset=WALL_OFFSET)


def _log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "hub.db")


# -- lease_event_to_hub ------------------------------------------------------


def test_granted_event_maps_to_lease_granted_with_wall_clock_times() -> None:
    manager, _clock, events = _node()
    lease = manager.acquire(AGENT, Scope.TIMELINE, (PROJECT, "main"), ttl=60.0)

    spec = _map(events[0])

    assert spec.type == "lease.granted"
    assert spec.payload == {
        "node_id": "node-1",
        "lease_id": lease.id,
        "scope": "timeline",
        "key": [PROJECT, "main"],
        "holder": {"kind": "agent", "id": "agent-1"},
        "acquired_at": _wall(1000.0),
        "expires_at": _wall(1060.0),
        "ttl": 60.0,
        "at": _wall(1000.0),
        "by": None,
    }


def test_each_lease_event_kind_maps_to_its_own_hub_event_type() -> None:
    manager, clock, events = _node()
    agent_lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)  # granted
    clock.now = 1010.0
    manager.renew(agent_lease.id, AGENT)  # renewed
    clock.now = 1020.0
    human_lease = manager.acquire(HUMAN, Scope.PROJECT, (PROJECT,), ttl=None)  # preempted, granted
    clock.now = 1030.0
    manager.release(human_lease.id, HUMAN)  # released
    manager.acquire(AGENT, Scope.INSTANCE, ("default",), ttl=5.0)  # granted
    clock.now = 1040.0
    manager.active()  # expired

    assert [event.kind for event in events] == [
        "granted",
        "renewed",
        "preempted",
        "granted",
        "released",
        "granted",
        "expired",
    ]
    assert [_map(event).type for event in events] == [
        "lease.granted",
        "lease.renewed",
        "lease.preempted",
        "lease.granted",
        "lease.released",
        "lease.granted",
        "lease.expired",
    ]


def test_renewed_event_keeps_the_original_acquired_at_and_moves_expiry() -> None:
    manager, clock, events = _node()
    lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1010.0
    manager.renew(lease.id, AGENT)

    payload = _map(events[1]).payload

    assert payload["acquired_at"] == _wall(1000.0)
    assert payload["expires_at"] == _wall(1070.0)
    assert payload["at"] == _wall(1010.0)


def test_expired_event_is_stamped_when_the_node_noticed_not_when_it_lapsed() -> None:
    manager, clock, events = _node()
    manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1100.0
    assert manager.active() == []

    spec = _map(events[1])

    assert spec.type == "lease.expired"
    assert spec.payload["expires_at"] == _wall(1060.0)
    assert spec.payload["at"] == _wall(1100.0)


def test_preemption_event_names_the_victim_as_holder_and_the_human_as_by() -> None:
    manager, clock, events = _node()
    agent_lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1020.0
    manager.acquire(HUMAN, Scope.PROJECT, (PROJECT,), ttl=None)

    preempted = _map(events[1])
    granted = _map(events[2])

    assert preempted.type == "lease.preempted"
    assert preempted.payload["lease_id"] == agent_lease.id
    assert preempted.payload["holder"] == {"kind": "agent", "id": "agent-1"}
    assert preempted.payload["by"] == {"kind": "human", "id": "operator-1"}
    # The human's own grant has no preemptor, and a human lease without a ttl never expires.
    assert granted.type == "lease.granted"
    assert granted.payload["by"] is None
    assert granted.payload["expires_at"] is None
    assert granted.payload["ttl"] is None


def test_mapping_is_pure_and_its_idempotency_key_is_per_node_and_event() -> None:
    manager, clock, events = _node()
    lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1010.0
    manager.renew(lease.id, AGENT)

    assert _map(events[0]) == _map(events[0])
    assert _map(events[1]) == _map(events[1])
    # A retry of one event reuses its key; a different event or a different node does not.
    keys = {
        _map(events[0]).idempotency_key,
        _map(events[1]).idempotency_key,
        _map(events[0], "node-2").idempotency_key,
    }
    assert None not in keys
    assert len(keys) == 3


def test_unknown_lease_event_kind_raises_invalid_input() -> None:
    manager, _clock, events = _node()
    manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    bogus = LeaseEvent("vanished", events[0].at, events[0].lease)

    with pytest.raises(InvalidInput) as raised:
        _map(bogus)

    assert raised.value.field == "kind"


@pytest.mark.parametrize("node_id", ["", "   ", "n" * 129])
def test_lease_event_needs_a_usable_node_id(node_id: str) -> None:
    manager, _clock, events = _node()
    manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)

    with pytest.raises(InvalidInput) as raised:
        _map(events[0], node_id)

    assert raised.value.field == "node_id"


def test_current_wall_offset_turns_monotonic_time_into_unix_time() -> None:
    offset = current_wall_offset()

    assert offset + time.monotonic() == pytest.approx(time.time(), abs=5.0)


# -- record_lease_event ------------------------------------------------------


def test_record_lease_event_appends_as_the_node_and_dedupes_a_retry(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, _clock, events = _node()
    manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)

    first = record_lease_event(log, events[0], NODE, wall_offset=WALL_OFFSET)
    again = record_lease_event(log, events[0], NODE, wall_offset=WALL_OFFSET)

    assert first.actor == Actor("node", "node-1")
    assert first.type == "lease.granted"
    assert again.id == first.id
    assert len(log.read(types=["lease.granted"])) == 1


def test_two_renewals_at_different_times_are_both_recorded(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, clock, events = _node()
    lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1010.0
    manager.renew(lease.id, AGENT)
    clock.now = 1020.0
    manager.renew(lease.id, AGENT)

    _record(log, events)

    assert len(log.read(types=["lease.renewed"])) == 2


# -- lease_board -------------------------------------------------------------


def test_board_lists_a_granted_lease_with_scope_key_holder_node_and_since(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, _clock, events = _node()
    lease = manager.acquire(AGENT, Scope.TIMELINE, (PROJECT, "main"), ttl=60.0)
    _record(log, events)

    board = lease_board(log, clock=_at(1030.0))

    assert board.as_of == _wall(1030.0)
    assert [entry.to_dict() for entry in board.leases] == [
        {
            "node_id": "node-1",
            "lease_id": lease.id,
            "scope": "timeline",
            "key": [PROJECT, "main"],
            "holder": {"kind": "agent", "id": "agent-1"},
            "since": _wall(1000.0),
            "expires_at": _wall(1060.0),
        }
    ]
    assert board.presence == ()


def test_expired_lease_drops_off_the_board(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, _clock, events = _node()
    lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    _record(log, events)  # only the grant: this node crashed and never reports an expiry
    assert log.read(types=["lease.expired"]) == []

    before = lease_board(log, clock=_at(1059.0))
    assert [entry.lease_id for entry in before.leases] == [lease.id]

    # The expiry instant itself counts as expired, and so does anything later.
    assert lease_board(log, clock=_at(1060.0)).leases == ()
    assert lease_board(log, clock=_at(5000.0)).leases == ()


def test_human_lease_without_a_ttl_never_drops_off_the_board(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, _clock, events = _node()
    lease = manager.acquire(HUMAN, Scope.PROJECT, (PROJECT,), ttl=None)
    _record(log, events)

    board = lease_board(log, clock=_at(1000.0 + 10 * 365 * 86400.0))

    assert [entry.lease_id for entry in board.leases] == [lease.id]
    assert board.leases[0].expires_at is None


def test_released_lease_leaves_the_board(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, clock, events = _node()
    lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    _record(log, events)
    assert len(lease_board(log, clock=_at(1015.0)).leases) == 1

    clock.now = 1020.0
    manager.release(lease.id, AGENT)
    _record(log, events)  # the grant is replayed (deduped) and the release is new

    assert lease_board(log, clock=_at(1030.0)).leases == ()


def test_preempted_agent_lease_is_replaced_on_the_board_by_the_humans(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, clock, events = _node()
    agent_lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1020.0
    human_lease = manager.acquire(HUMAN, Scope.PROJECT, (PROJECT,), ttl=None)
    _record(log, events)

    board = lease_board(log, clock=_at(1030.0))

    assert [entry.lease_id for entry in board.leases] == [human_lease.id]
    assert agent_lease.id not in {entry.lease_id for entry in board.leases}
    assert board.leases[0].holder_kind == "human"
    assert board.leases[0].holder_id == "operator-1"
    assert board.leases[0].since == _wall(1020.0)


def test_expired_event_removes_the_lease_even_if_the_board_clock_is_behind(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, clock, events = _node()
    manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1100.0
    manager.active()  # the node reports the expiry
    _record(log, events)

    # The reader's clock says 1030, before the lease's own expiry: the report still wins.
    assert lease_board(log, clock=_at(1030.0)).leases == ()


def test_renewed_event_extends_the_lease_past_its_old_expiry(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, clock, events = _node()
    lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)  # expires at 1060
    clock.now = 1050.0
    manager.renew(lease.id, AGENT)  # now expires at 1110
    _record(log, events)

    board = lease_board(log, clock=_at(1080.0))

    assert [entry.lease_id for entry in board.leases] == [lease.id]
    assert board.leases[0].expires_at == _wall(1110.0)
    assert board.leases[0].since == _wall(1000.0)  # renewing is not a new acquisition
    assert lease_board(log, clock=_at(1110.0)).leases == ()


def test_board_orders_leases_by_when_they_were_acquired(tmp_path: Path) -> None:
    log = _log(tmp_path)
    first_node, first_clock, first_events = _node()
    second_node, second_clock, second_events = _node()
    first_clock.now = 1000.0
    late = first_node.acquire(AGENT, Scope.PROJECT, ("demo-show",), ttl=600.0)
    second_clock.now = 900.0
    early = second_node.acquire(AGENT, Scope.PROJECT, ("demo-reel",), ttl=600.0)
    _record(log, first_events, "node-1")  # recorded first, but acquired later
    _record(log, second_events, "node-2")

    board = lease_board(log, clock=_at(1100.0))

    assert [(e.node_id, e.lease_id) for e in board.leases] == [
        ("node-2", early.id),
        ("node-1", late.id),
    ]


# -- actor must be the node named in the payload -----------------------------


def _grant_spec() -> Any:
    manager, _clock, events = _node()
    manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    return _map(events[0])


@pytest.mark.parametrize(
    "actor",
    [Actor("agent", "agent-1"), Actor("human", "operator-1"), Actor("node", "node-2")],
    ids=["agent", "human", "other-node"],
)
def test_lease_event_from_an_actor_that_is_not_its_node_is_ignored(
    tmp_path: Path, actor: Actor
) -> None:
    log = _log(tmp_path)
    spec = _grant_spec()
    assert spec.payload["node_id"] == "node-1"

    log.append(spec.type, actor, spec.payload)

    assert lease_board(log, clock=_at(1030.0)).leases == ()

    # The identical payload from the node it names is accepted, so the filter is what
    # kept the first one off the board.
    log.append(spec.type, Actor("node", "node-1"), spec.payload)
    assert len(lease_board(log, clock=_at(1030.0)).leases) == 1


def test_release_forged_by_a_non_node_actor_does_not_free_the_lease(tmp_path: Path) -> None:
    log = _log(tmp_path)
    manager, clock, events = _node()
    lease = manager.acquire(AGENT, Scope.PROJECT, (PROJECT,), ttl=60.0)
    clock.now = 1020.0
    manager.release(lease.id, AGENT)
    _record(log, events[:1])  # the real node only reported the grant
    release = _map(events[1])

    log.append(release.type, Actor("agent", "agent-1"), release.payload)

    assert [entry.lease_id for entry in lease_board(log, clock=_at(1030.0)).leases] == [lease.id]


@pytest.mark.parametrize(
    "damage",
    [
        lambda p: p.pop("holder"),
        lambda p: p.update(expires_at="not-a-date"),
        lambda p: p.update(key=[1, 2]),
        lambda p: p.update(lease_id=None),
        lambda p: p.pop("acquired_at"),
    ],
    ids=["no-holder", "bad-expiry", "non-string-key", "no-lease-id", "no-acquired-at"],
)
def test_malformed_lease_event_from_the_node_is_skipped_not_fatal(
    tmp_path: Path, damage: Callable[[dict[str, Any]], object]
) -> None:
    log = _log(tmp_path)
    spec = _grant_spec()
    broken = dict(spec.payload)
    damage(broken)
    node = Actor("node", "node-1")

    log.append(spec.type, node, broken)
    assert lease_board(log, clock=_at(1030.0)).leases == ()

    log.append(spec.type, node, spec.payload)  # a good event after it still shows
    assert len(lease_board(log, clock=_at(1030.0)).leases) == 1


# -- presence ----------------------------------------------------------------


def test_presence_shows_the_human_since_the_report_with_its_expiry(tmp_path: Path) -> None:
    log = _log(tmp_path)
    record_presence(log, NODE, active=True, at=T0, human_id="operator-1", ttl=timedelta(minutes=5))

    board = lease_board(log, clock=lambda: T0 + timedelta(minutes=1))

    assert [entry.to_dict() for entry in board.presence] == [
        {
            "node_id": "node-1",
            "human_id": "operator-1",
            "since": T0.isoformat(),
            "expires_at": (T0 + timedelta(minutes=5)).isoformat(),
        }
    ]
    assert board.leases == ()


def test_presence_refresh_keeps_since_and_pushes_the_expiry(tmp_path: Path) -> None:
    log = _log(tmp_path)
    ttl = timedelta(minutes=5)
    record_presence(log, NODE, active=True, at=T0, human_id="operator-1", ttl=ttl)
    record_presence(
        log, NODE, active=True, at=T0 + timedelta(minutes=1), human_id="operator-1", ttl=ttl
    )

    # 5.5 minutes in: past the first report's expiry, inside the refreshed one.
    board = lease_board(log, clock=lambda: T0 + timedelta(minutes=5, seconds=30))

    assert len(board.presence) == 1
    assert board.presence[0].since == T0.isoformat()
    assert board.presence[0].expires_at == (T0 + timedelta(minutes=6)).isoformat()


def test_presence_set_inactive_removes_the_human(tmp_path: Path) -> None:
    log = _log(tmp_path)
    record_presence(log, NODE, active=True, at=T0, human_id="operator-1", ttl=timedelta(minutes=5))
    assert len(lease_board(log, clock=lambda: T0 + timedelta(minutes=1)).presence) == 1

    record_presence(log, NODE, active=False, at=T0 + timedelta(minutes=2))

    assert lease_board(log, clock=lambda: T0 + timedelta(minutes=2)).presence == ()


def test_presence_drops_off_once_its_expiry_passes(tmp_path: Path) -> None:
    log = _log(tmp_path)
    record_presence(log, NODE, active=True, at=T0, human_id="operator-1", ttl=timedelta(minutes=5))

    just_before = T0 + timedelta(minutes=5) - timedelta(seconds=1)
    assert len(lease_board(log, clock=lambda: just_before).presence) == 1
    assert lease_board(log, clock=lambda: T0 + timedelta(minutes=5)).presence == ()
    assert lease_board(log, clock=lambda: T0 + timedelta(hours=3)).presence == ()


def test_presence_reported_again_after_it_lapsed_starts_a_new_since(tmp_path: Path) -> None:
    log = _log(tmp_path)
    ttl = timedelta(minutes=5)
    record_presence(log, NODE, active=True, at=T0, human_id="operator-1", ttl=ttl)
    later = T0 + timedelta(minutes=10)
    record_presence(log, NODE, active=True, at=later, human_id="operator-1", ttl=ttl)

    board = lease_board(log, clock=lambda: later + timedelta(minutes=1))

    assert [entry.since for entry in board.presence] == [later.isoformat()]


def test_presence_without_a_ttl_stays_until_reported_inactive(tmp_path: Path) -> None:
    log = _log(tmp_path)
    record_presence(log, NODE, active=True, at=T0, human_id="operator-1")

    board = lease_board(log, clock=lambda: T0 + timedelta(days=30))

    assert [entry.expires_at for entry in board.presence] == [None]
    assert [entry.human_id for entry in board.presence] == ["operator-1"]


def test_presence_inactive_report_carries_no_expiry_even_with_a_ttl() -> None:
    inactive = presence_to_hub(NODE, active=False, at=T0, ttl=timedelta(minutes=5))
    active = presence_to_hub(NODE, active=True, at=T0, ttl=timedelta(minutes=5))

    assert inactive.type == "presence.changed"
    assert inactive.payload["active"] is False
    assert inactive.payload["expires_at"] is None
    assert active.payload["expires_at"] == (T0 + timedelta(minutes=5)).isoformat()
    # Pure: the same arguments give an equal spec.
    assert presence_to_hub(NODE, active=False, at=T0, ttl=timedelta(minutes=5)) == inactive


def test_presence_report_from_a_non_node_actor_is_ignored(tmp_path: Path) -> None:
    log = _log(tmp_path)
    spec = presence_to_hub(NODE, active=True, at=T0, human_id="operator-1")

    log.append(spec.type, Actor("agent", "agent-1"), spec.payload)

    assert lease_board(log, clock=lambda: T0 + timedelta(minutes=1)).presence == ()


@pytest.mark.parametrize(
    "ttl",
    [timedelta(0), timedelta(seconds=-1), timedelta(hours=24, microseconds=1)],
    ids=["zero", "negative", "over-24h"],
)
def test_presence_ttl_outside_zero_to_24_hours_is_rejected(tmp_path: Path, ttl: timedelta) -> None:
    log = _log(tmp_path)

    with pytest.raises(InvalidInput) as raised:
        record_presence(log, NODE, active=True, at=T0, human_id="operator-1", ttl=ttl)

    assert raised.value.field == "ttl"
    assert log.latest_seq() == 0  # nothing was written


def test_presence_ttl_of_exactly_24_hours_is_accepted(tmp_path: Path) -> None:
    log = _log(tmp_path)
    record_presence(log, NODE, active=True, at=T0, human_id="operator-1", ttl=timedelta(hours=24))

    board = lease_board(log, clock=lambda: T0 + timedelta(hours=23))

    assert [entry.expires_at for entry in board.presence] == [
        (T0 + timedelta(hours=24)).isoformat()
    ]
