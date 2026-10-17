"""Tests for the emergency stop: scoped stops, human-only resume, idempotent retries."""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub.common import HumanRequired, InvalidInput
from aar.hub.control import (
    NotStopped,
    Stopped,
    StopScope,
    assert_running,
    resume,
    status,
    stop,
)
from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.registry import Registry

HUMAN = Actor("human", "operator-1")
OTHER_HUMAN = Actor("human", "operator-2")
AGENT = Actor("agent", "agent-1")
NODE = Actor("node", "node-1")

STOPPED = "control.stopped"
RESUMED = "control.resumed"


def _log(tmp_path: Path) -> EventLog:
    """A hub with the two projects these tests stop (stops on unknown projects are refused)."""
    log = EventLog(tmp_path / "hub.db")
    registry = Registry(log)
    for project_id in ("prj_demo", "prj_other"):
        registry.create_project(f"demo-show {project_id}", HUMAN, project_id=project_id)
    return log


def _scopes(log: EventLog) -> list[str]:
    return [str(record.scope) for record in status(log).stops]


# -- stop and resume ---------------------------------------------------------


def test_stop_all_halts_work_and_a_human_resume_restarts_it(tmp_path: Path) -> None:
    log = _log(tmp_path)
    assert status(log).running is True
    assert_running(log)  # nothing stopped: does not raise

    record = stop(log, "all", "bad render on the bus", HUMAN)

    assert status(log).running is False
    stop_event = log.read(types=[STOPPED])[0]
    assert record.scope == StopScope.all()
    assert record.reason == "bad render on the bus"
    assert record.by == HUMAN
    assert record.at == stop_event.ts
    assert record.event_id == stop_event.id
    assert status(log).stops == (record,)
    assert status(log).to_dict() == {"running": False, "stops": [record.to_dict()]}

    with pytest.raises(Stopped) as raised:
        assert_running(log)
    assert raised.value.scope == StopScope.all()
    assert raised.value.reason == "bad render on the bus"
    assert raised.value.by == HUMAN
    assert raised.value.at == record.at
    assert "bad render on the bus" in str(raised.value)

    lifted = resume(log, "all", "fixed", HUMAN)

    assert lifted == record  # resume returns the stop it lifted
    assert status(log).running is True
    assert_running(log)
    resumed_event = log.read(types=[RESUMED])[0]
    assert resumed_event.actor == HUMAN
    assert resumed_event.payload == {"scope": "all", "reason": "fixed"}


def test_stopping_an_already_stopped_scope_records_the_new_reason_and_does_not_error(
    tmp_path: Path,
) -> None:
    log = _log(tmp_path)
    stop(log, "all", "first", HUMAN)

    second = stop(log, "all", "second", AGENT)

    assert [(r.reason, r.by) for r in status(log).stops] == [("second", AGENT)]
    assert second.reason == "second"
    resume(log, "all", "fixed", HUMAN)  # one resume lifts the scope however often it was stopped
    assert status(log).running is True


def test_status_lists_all_then_projects_then_nodes(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "node:node-1", "n", HUMAN)
    stop(log, "project:prj_demo", "p", HUMAN)
    stop(log, "all", "a", HUMAN)

    assert _scopes(log) == ["all", "project:prj_demo", "node:node-1"]


# -- scopes are independent --------------------------------------------------


def test_project_stop_only_blocks_that_project(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "project:prj_demo", "bad edit", HUMAN)

    with pytest.raises(Stopped) as raised:
        assert_running(log, project_id="prj_demo")
    assert raised.value.scope == StopScope.project("prj_demo")
    assert_running(log, project_id="prj_other")
    assert_running(log)
    assert_running(log, node_id="node-1")
    assert status(log).running is False
    # A project stop is filed under its project so project-scoped readers can see it.
    assert log.read(types=[STOPPED])[0].project_id == "prj_demo"


def test_node_stop_only_blocks_that_node(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "node:node-1", "disk failing", HUMAN)

    with pytest.raises(Stopped) as raised:
        assert_running(log, node_id="node-1")
    assert raised.value.scope == StopScope.node("node-1")
    assert_running(log, node_id="node-2")
    assert_running(log)
    assert_running(log, project_id="prj_demo")
    assert log.read(types=[STOPPED])[0].project_id is None


def test_stop_all_blocks_every_project_and_node(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "all", "everything stop", HUMAN)

    for kwargs in ({}, {"project_id": "prj_other"}, {"node_id": "node-9"}):
        with pytest.raises(Stopped) as raised:
            assert_running(log, **kwargs)
        assert raised.value.scope == StopScope.all()


def test_resuming_all_does_not_lift_a_project_or_node_stop(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "all", "a", HUMAN)
    stop(log, "project:prj_demo", "p", HUMAN)
    stop(log, "node:node-1", "n", HUMAN)

    resume(log, "all", "all clear", HUMAN)

    assert _scopes(log) == ["project:prj_demo", "node:node-1"]
    assert_running(log)
    with pytest.raises(Stopped):
        assert_running(log, project_id="prj_demo")
    with pytest.raises(Stopped):
        assert_running(log, node_id="node-1")

    resume(log, "project:prj_demo", "p fixed", HUMAN)

    assert _scopes(log) == ["node:node-1"]
    assert_running(log, project_id="prj_demo")


def test_resuming_a_project_does_not_lift_stop_all(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "all", "a", HUMAN)
    stop(log, "project:prj_demo", "p", HUMAN)

    resume(log, "project:prj_demo", "p fixed", HUMAN)

    assert _scopes(log) == ["all"]
    with pytest.raises(Stopped):
        assert_running(log, project_id="prj_demo")


# -- who may stop and who may resume -----------------------------------------


@pytest.mark.parametrize("actor", [AGENT, NODE], ids=["agent", "node"])
def test_agents_and_nodes_may_pull_the_brake(tmp_path: Path, actor: Actor) -> None:
    log = _log(tmp_path)

    record = stop(log, "all", "something looks wrong", actor)

    assert record.by == actor
    with pytest.raises(Stopped) as raised:
        assert_running(log)
    assert raised.value.by == actor


@pytest.mark.parametrize("actor", [AGENT, NODE], ids=["agent", "node"])
def test_agents_and_nodes_cannot_resume(tmp_path: Path, actor: Actor) -> None:
    log = _log(tmp_path)
    stop(log, "all", "halt", HUMAN)

    with pytest.raises(HumanRequired) as raised:
        resume(log, "all", "looks fine to me", actor)

    assert raised.value.actor == actor
    assert status(log).running is False
    assert log.read(types=[RESUMED]) == []
    with pytest.raises(Stopped):
        assert_running(log)


@pytest.mark.parametrize("actor", [AGENT, NODE], ids=["agent", "node"])
def test_a_resume_forged_straight_into_the_log_does_not_lift_the_stop(
    tmp_path: Path, actor: Actor
) -> None:
    log = _log(tmp_path)
    stop(log, "all", "halt", HUMAN)

    log.append(RESUMED, actor, {"scope": "all", "reason": "x"})

    assert status(log).running is False
    with pytest.raises(Stopped):
        assert_running(log)

    # A real human resume still works afterwards: the forged event did not use it up.
    resume(log, "all", "fixed", HUMAN)
    assert status(log).running is True


def test_unreadable_control_events_in_the_log_are_skipped(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append(STOPPED, AGENT, {"scope": "bogus:x", "reason": "junk"})
    log.append(STOPPED, AGENT, {"reason": "no scope"})
    assert status(log).running is True
    assert_running(log)

    stop(log, "all", "real", HUMAN)

    assert _scopes(log) == ["all"]


# -- NotStopped and idempotent retries ---------------------------------------


def test_resume_on_a_running_scope_raises_not_stopped_and_writes_nothing(tmp_path: Path) -> None:
    log = _log(tmp_path)

    with pytest.raises(NotStopped):
        resume(log, "all", "nothing to resume", HUMAN)

    stop(log, "project:prj_demo", "p", HUMAN)
    with pytest.raises(NotStopped):
        resume(log, "all", "wrong scope", HUMAN)  # a different scope is stopped, not this one
    resume(log, "project:prj_demo", "fixed", HUMAN)
    with pytest.raises(NotStopped):
        resume(log, "project:prj_demo", "twice", HUMAN)
    assert len(log.read(types=[RESUMED])) == 1


def test_resume_retry_under_the_same_key_returns_without_a_second_event(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "all", "first look", HUMAN)
    latest = stop(log, "all", "second look", HUMAN)

    first = resume(log, "all", "fixed", HUMAN, idempotency_key="resume-1")
    again = resume(log, "all", "fixed", HUMAN, idempotency_key="resume-1")

    assert first == latest  # the stop in force is the most recent one
    assert again == first
    assert len(log.read(types=[RESUMED])) == 1
    assert status(log).running is True


def test_late_resume_retry_does_not_lift_a_newer_stop(tmp_path: Path) -> None:
    log = _log(tmp_path)
    old = stop(log, "all", "first problem", HUMAN)
    resume(log, "all", "fixed", HUMAN, idempotency_key="resume-1")
    newer = stop(log, "all", "second problem", HUMAN)

    retried = resume(log, "all", "fixed", HUMAN, idempotency_key="resume-1")

    assert retried == old  # the retry reports the stop its first call lifted
    assert len(log.read(types=[RESUMED])) == 1
    assert status(log).stops == (newer,)  # the new stop is untouched


def test_resume_key_reused_for_a_different_request_conflicts(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "all", "halt", HUMAN)
    resume(log, "all", "fixed", HUMAN, idempotency_key="resume-1")

    with pytest.raises(IdempotencyConflict):
        resume(log, "all", "a different reason", HUMAN, idempotency_key="resume-1")
    with pytest.raises(IdempotencyConflict):
        resume(log, "all", "fixed", OTHER_HUMAN, idempotency_key="resume-1")
    with pytest.raises(IdempotencyConflict):
        resume(log, "node:node-1", "fixed", HUMAN, idempotency_key="resume-1")
    assert len(log.read(types=[RESUMED])) == 1


def test_stop_retry_under_the_same_key_writes_one_event(tmp_path: Path) -> None:
    log = _log(tmp_path)

    first = stop(log, "all", "halt", AGENT, idempotency_key="stop-1")
    again = stop(log, "all", "halt", AGENT, idempotency_key="stop-1")

    assert again == first
    assert len(log.read(types=[STOPPED])) == 1


def test_stop_key_reused_for_a_different_request_conflicts(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stop(log, "all", "halt", AGENT, idempotency_key="stop-1")

    with pytest.raises(IdempotencyConflict):
        stop(log, "all", "another reason", AGENT, idempotency_key="stop-1")
    with pytest.raises(IdempotencyConflict):
        stop(log, "project:prj_demo", "halt", AGENT, idempotency_key="stop-1")
    with pytest.raises(IdempotencyConflict):
        stop(log, "all", "halt", NODE, idempotency_key="stop-1")
    # The same key cannot be spent on a resume either.
    with pytest.raises(IdempotencyConflict):
        resume(log, "all", "halt", HUMAN, idempotency_key="stop-1")
    assert len(log.read(types=[STOPPED])) == 1
    assert log.read(types=[RESUMED]) == []


# -- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["", "project:", "node:", "bogus:x", "all:x", ":x", "all ", "project: x", "project", None, 7],
)
def test_stop_scope_parse_rejects_malformed_scopes(value: object) -> None:
    with pytest.raises(InvalidInput) as raised:
        StopScope.parse(value)  # type: ignore[arg-type]

    assert raised.value.field == "scope"


@pytest.mark.parametrize(
    ("text", "kind", "ident"),
    [
        ("all", "all", None),
        ("project:prj_demo", "project", "prj_demo"),
        ("node:node-1", "node", "node-1"),
    ],
)
def test_stop_scope_parse_round_trips_valid_scopes(text: str, kind: str, ident: str | None) -> None:
    scope = StopScope.parse(text)

    assert (scope.kind, scope.id) == (kind, ident)
    assert str(scope) == text
    assert StopScope.parse(scope) is scope


@pytest.mark.parametrize("scope", ["bogus:x", "all:x", "project:", ""])
def test_a_bad_scope_is_rejected_by_stop_and_resume_without_writing(
    tmp_path: Path, scope: str
) -> None:
    log = _log(tmp_path)
    setup_seq = log.latest_seq()

    with pytest.raises(InvalidInput):
        stop(log, scope, "halt", HUMAN)
    with pytest.raises(InvalidInput):
        resume(log, scope, "fixed", HUMAN)

    assert log.latest_seq() == setup_seq


@pytest.mark.parametrize(
    "reason", ["", "   ", "x" * 1001, "bell\x07"], ids=["empty", "blank", "long", "ctrl"]
)
def test_stop_and_resume_need_a_sane_reason(tmp_path: Path, reason: str) -> None:
    log = _log(tmp_path)
    stop(log, "all", "halt", HUMAN)
    seq = log.latest_seq()

    with pytest.raises(InvalidInput):
        stop(log, "all", reason, HUMAN)
    with pytest.raises(InvalidInput):
        resume(log, "all", reason, HUMAN)

    assert log.latest_seq() == seq
    assert status(log).running is False


def test_a_reason_of_exactly_1000_characters_is_accepted(tmp_path: Path) -> None:
    log = _log(tmp_path)

    record = stop(log, "all", "x" * 1000, HUMAN)

    assert len(record.reason) == 1000


def test_stopping_a_project_that_is_not_in_the_registry_is_refused(tmp_path: Path) -> None:
    log = _log(tmp_path)
    before = log.latest_seq()
    with pytest.raises(InvalidInput, match="no project 'prj_typo'"):
        stop(log, "project:prj_typo", "halt", HUMAN)
    assert log.latest_seq() == before
    assert status(log).running is True
    stop(log, "project:prj_demo", "halt", HUMAN)  # a registered project still stops
    assert status(log).running is False
