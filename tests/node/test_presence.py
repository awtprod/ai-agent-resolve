"""Human presence detection."""

from __future__ import annotations

import sys

import pytest

from aar.node.gateway import ResolveContext
from aar.node.presence import (
    PresenceGuard,
    default_input_idle_provider,
    idle_seconds_from_ticks,
    windows_input_idle_seconds,
)

HUMAN = ResolveContext(project="A", timeline="A main", page="edit", playhead="01:00:00:00")


def guard(**kwargs: object) -> PresenceGuard:
    return PresenceGuard(idle_window=60.0, max_observation_age=10.0, **kwargs)  # type: ignore[arg-type]


def test_nothing_observed_means_no() -> None:
    decision = guard().may_switch(0.0)
    assert not decision.allowed
    assert decision.code == "no_observation"


def test_must_watch_a_whole_idle_window_before_allowing() -> None:
    g = guard()
    g.observe(HUMAN, 0.0)
    g.observe(HUMAN, 55.0)
    assert g.may_switch(55.0).code == "warming_up"
    g.observe(HUMAN, 60.0)
    decision = g.may_switch(60.0)
    assert decision.allowed, decision
    assert decision.code == "idle"


def test_stale_observation_is_refused() -> None:
    g = guard()
    g.observe(HUMAN, 0.0)
    g.observe(HUMAN, 100.0)
    assert g.may_switch(100.0).allowed
    assert g.may_switch(111.0).code == "stale_observation"


def test_context_change_not_caused_by_broker_means_human_active() -> None:
    g = guard()
    g.observe(HUMAN, 0.0)
    moved = ResolveContext(project="A", timeline="A main", page="edit", playhead="01:00:09:00")
    assert g.observe(moved, 100.0) is True
    decision = g.may_switch(100.0)
    assert not decision.allowed
    assert decision.code == "human_context_change"
    assert "playhead" in decision.reason
    assert g.last_human_change == {"playhead": ("01:00:00:00", "01:00:09:00")}
    g.observe(moved, 159.0)
    assert not g.may_switch(159.0).allowed
    g.observe(moved, 160.0)
    assert g.may_switch(160.0).allowed


def test_broker_caused_changes_are_not_human_activity() -> None:
    g = guard()
    g.observe(HUMAN, 0.0)
    g.observe(HUMAN, 60.0)
    agent_ctx = ResolveContext(project="B", timeline="B main", page="color", playhead="02:00:00:00")
    g.note_broker_context(agent_ctx, 61.0)
    assert g.observe(agent_ctx, 62.0) is False
    assert g.may_switch(62.0).allowed
    assert not g.human_changed_since(0.0)


def test_human_change_after_broker_action_is_detected() -> None:
    g = guard()
    g.observe(HUMAN, 0.0)
    agent_ctx = ResolveContext(project="B", timeline="B main", page="edit")
    g.note_broker_context(agent_ctx, 70.0)
    human_back = ResolveContext(project="A", timeline="A main", page="edit")
    assert g.observe(human_back, 75.0) is True
    assert g.human_changed_since(70.0)
    assert g.may_switch(75.0).code == "human_context_change"


def test_render_progress_alone_is_not_human_activity() -> None:
    g = guard()
    g.observe(HUMAN, 0.0)
    rendering = ResolveContext(**{**HUMAN.position(), "rendering": True})  # type: ignore[arg-type]
    assert g.observe(rendering, 60.0) is False
    assert g.may_switch(60.0).allowed


def test_recent_os_input_means_human_active() -> None:
    idle = {"seconds": 5.0}
    g = guard(input_idle=lambda: idle["seconds"])
    g.observe(HUMAN, 0.0)
    g.observe(HUMAN, 60.0)
    decision = g.may_switch(60.0)
    assert decision.code == "human_input"
    idle["seconds"] = 61.0
    assert g.may_switch(60.0).allowed


def test_input_provider_that_cannot_tell_is_ignored_and_failure_refuses() -> None:
    g = guard(input_idle=lambda: None)
    g.observe(HUMAN, 0.0)
    g.observe(HUMAN, 60.0)
    assert g.may_switch(60.0).allowed

    def broken() -> float:
        raise OSError("no session")

    g2 = guard(input_idle=broken)
    g2.observe(HUMAN, 0.0)
    g2.observe(HUMAN, 60.0)
    decision = g2.may_switch(60.0)
    assert not decision.allowed
    assert decision.code == "input_idle_error"
    assert "no session" in decision.reason


def test_tick_difference_survives_32_bit_wrap() -> None:
    assert idle_seconds_from_ticks(1_000, 4_000) == 3.0
    assert idle_seconds_from_ticks(0xFFFFFFFF - 999, 2_000) == 3.0


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows fallback")
def test_input_idle_is_a_noop_off_windows() -> None:
    assert windows_input_idle_seconds() is None
    assert default_input_idle_provider() is None


def test_rejects_nonpositive_windows() -> None:
    with pytest.raises(ValueError):
        PresenceGuard(idle_window=0)
