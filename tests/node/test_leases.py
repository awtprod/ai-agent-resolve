"""Lease conflicts, expiry and human precedence."""

from __future__ import annotations

import pytest

from aar.node.leases import (
    Holder,
    LeaseConflict,
    LeaseEvent,
    LeaseManager,
    LeaseNotHeld,
    Scope,
    instance_key,
    project_key,
    timeline_key,
)

from .fake_resolve import FakeClock

ANDREW = Holder.human("andrew")
EDITOR = Holder.human("editor-2")
AGENT_1 = Holder.agent("thread-1")
AGENT_2 = Holder.agent("thread-2")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def events() -> list[LeaseEvent]:
    return []


@pytest.fixture
def leases(clock: FakeClock, events: list[LeaseEvent]) -> LeaseManager:
    return LeaseManager(clock, events.append)


def test_agents_conflict_on_the_same_key_but_not_on_different_keys(leases: LeaseManager) -> None:
    held = leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=60)
    with pytest.raises(LeaseConflict) as err:
        leases.acquire(AGENT_2, Scope.PROJECT, project_key("A"), ttl=60)
    assert err.value.blocking == (held,)
    leases.acquire(AGENT_2, Scope.PROJECT, project_key("B"), ttl=60)
    assert len(leases.active()) == 2


def test_project_lease_conflicts_with_timelines_inside_it_only(leases: LeaseManager) -> None:
    leases.acquire(AGENT_1, Scope.TIMELINE, timeline_key("A", "main"), ttl=60)
    leases.acquire(AGENT_2, Scope.TIMELINE, timeline_key("A", "alt"), ttl=60)
    with pytest.raises(LeaseConflict):
        leases.acquire(AGENT_2, Scope.PROJECT, project_key("A"), ttl=60)
    leases.acquire(AGENT_2, Scope.PROJECT, project_key("B"), ttl=60)
    with pytest.raises(LeaseConflict):
        leases.acquire(AGENT_1, Scope.TIMELINE, timeline_key("B", "main"), ttl=60)


def test_instance_lease_is_independent_of_project_leases(leases: LeaseManager) -> None:
    leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=60)
    leases.acquire(AGENT_2, Scope.INSTANCE, instance_key(), ttl=60)
    with pytest.raises(LeaseConflict):
        leases.acquire(AGENT_1, Scope.INSTANCE, instance_key(), ttl=60)


def test_reacquire_by_same_holder_renews_the_same_lease(
    leases: LeaseManager, clock: FakeClock
) -> None:
    first = leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=60)
    clock.advance(50)
    again = leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=60)
    assert again.id == first.id
    assert again.expires_at == clock.now + 60


def test_lease_expires_and_frees_the_key(
    leases: LeaseManager, clock: FakeClock, events: list[LeaseEvent]
) -> None:
    lease = leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=30)
    clock.advance(29.9)
    with pytest.raises(LeaseConflict):
        leases.acquire(AGENT_2, Scope.PROJECT, project_key("A"), ttl=30)
    clock.advance(0.1)
    leases.acquire(AGENT_2, Scope.PROJECT, project_key("A"), ttl=30)
    with pytest.raises(LeaseNotHeld) as err:
        leases.renew(lease.id, AGENT_1)
    assert err.value.reason == "expired"
    assert [e.kind for e in events if e.lease.id == lease.id] == ["granted", "expired"]


def test_renew_extends_by_the_original_ttl(leases: LeaseManager, clock: FakeClock) -> None:
    lease = leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=30)
    clock.advance(20)
    renewed = leases.renew(lease.id, AGENT_1)
    assert renewed.expires_at == clock.now + 30
    clock.advance(25)  # past the original expiry, inside the renewed one
    assert leases.get(lease.id) is not None


def test_human_preempts_agent_and_agent_learns_why(
    leases: LeaseManager, events: list[LeaseEvent]
) -> None:
    agent_lease = leases.acquire(AGENT_1, Scope.TIMELINE, timeline_key("A", "main"), ttl=60)
    human_lease = leases.acquire(ANDREW, Scope.PROJECT, project_key("A"), ttl=None)
    assert leases.active() == [human_lease]
    assert human_lease.expires_at is None
    with pytest.raises(LeaseNotHeld) as err:
        leases.renew(agent_lease.id, AGENT_1)
    assert err.value.reason == "preempted by human:andrew"
    preempted = [e for e in events if e.kind == "preempted"]
    assert len(preempted) == 1
    assert preempted[0].lease == agent_lease
    assert preempted[0].by == ANDREW


def test_agent_can_never_preempt_a_human(leases: LeaseManager) -> None:
    human_lease = leases.acquire(ANDREW, Scope.INSTANCE, instance_key(), ttl=5)
    with pytest.raises(LeaseConflict) as err:
        leases.acquire(AGENT_1, Scope.INSTANCE, instance_key(), ttl=60)
    assert err.value.blocking == (human_lease,)
    with pytest.raises(ValueError, match="only a human can force"):
        leases.acquire(AGENT_1, Scope.INSTANCE, instance_key(), ttl=60, force=True)
    assert leases.active() == [human_lease]


def test_human_displaces_human_only_with_force(leases: LeaseManager) -> None:
    first = leases.acquire(ANDREW, Scope.PROJECT, project_key("A"), ttl=60)
    with pytest.raises(LeaseConflict):
        leases.acquire(EDITOR, Scope.PROJECT, project_key("A"), ttl=60)
    assert leases.active() == [first]
    forced = leases.acquire(EDITOR, Scope.PROJECT, project_key("A"), ttl=60, force=True)
    assert leases.active() == [forced]


def test_failed_human_acquire_does_not_revoke_any_agent_lease(leases: LeaseManager) -> None:
    agent_lease = leases.acquire(AGENT_1, Scope.TIMELINE, timeline_key("A", "main"), ttl=60)
    human_lease = leases.acquire(EDITOR, Scope.TIMELINE, timeline_key("A", "alt"), ttl=60)
    with pytest.raises(LeaseConflict):
        leases.acquire(ANDREW, Scope.PROJECT, project_key("A"), ttl=60)
    assert {lease.id for lease in leases.active()} == {agent_lease.id, human_lease.id}


def test_release_rules(leases: LeaseManager) -> None:
    lease = leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=60)
    with pytest.raises(LeaseNotHeld):
        leases.release(lease.id, AGENT_2)
    leases.release(lease.id, AGENT_1)
    with pytest.raises(LeaseNotHeld) as err:
        leases.release(lease.id, AGENT_1)
    assert err.value.reason == "released"
    leases.acquire(AGENT_2, Scope.PROJECT, project_key("A"), ttl=60)


def test_validation(leases: LeaseManager) -> None:
    with pytest.raises(ValueError, match="positive ttl"):
        leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=None)
    with pytest.raises(ValueError, match="positive"):
        leases.acquire(AGENT_1, Scope.PROJECT, project_key("A"), ttl=0)
    with pytest.raises(ValueError, match="key"):
        leases.acquire(AGENT_1, Scope.TIMELINE, project_key("A"), ttl=10)


def test_conflicts_for_reports_without_acquiring(leases: LeaseManager) -> None:
    human = leases.acquire(ANDREW, Scope.TIMELINE, timeline_key("A", "main"), ttl=60)
    assert leases.conflicts_for(AGENT_1, Scope.PROJECT, project_key("A")) == [human]
    assert leases.conflicts_for(ANDREW, Scope.PROJECT, project_key("A")) == []
    assert leases.conflicts_for(AGENT_1, Scope.TIMELINE, timeline_key("A", "alt")) == []
