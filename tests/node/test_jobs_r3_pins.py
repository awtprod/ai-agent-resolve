"""S7 round 3: deterministic pins for mutants that survived the round-3 mutation run.

Each test names the mutant(s) it kills. All run on the injected clock with the drain budget
off (``JobWorld``'s default), or synchronise on events -- none depends on wall-clock timing.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from aar.hub import control
from aar.hub.approvals import ApprovalScope, grant
from aar.hub.events import Actor
from aar.node.jobs import (
    ApprovalPolicy,
    JobKind,
    JobState,
    Watchdog,
    WatchdogConfig,
    register_kind,
)
from aar.node.jobs.model import Done, Step, StepAction, StepContext, Submit

from .jobs_support import JobWorld

HUMAN = Actor("human", "andrew")
WAIT = 30.0


def _markers(*frames: int) -> dict:
    return {"steps": [{"op": "markers.add", "params": {"markers": [{"frame": f}]}} for f in frames]}


def _render_params(world: JobWorld, **over: object) -> dict:
    params = {
        "timeline": world.ref("Main"),
        "target_dir": "D:/Out",
        "custom_name": "show_v1",
        "mark_in": 0,
        "mark_out": 99,
    }
    params.update(over)
    return params


# -- a test-only kind whose steps use the cursor --------------------------------------------------


class _CursorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int = 2


class _CursorStep(Step):
    """Counts its own advances in the cursor; reports what the cursor held when it started."""

    def __init__(self, index: int) -> None:
        self.name = f"cursor-{index}"
        self.timeout = 300.0
        self.index = index

    def advance(self, ctx: StepContext) -> StepAction:
        if ctx.outcome is None:
            ctx.cursor.setdefault("seen_at_start", dict(ctx.cursor))
            return Submit("timeline.list", {}, key=self.name)
        return Done({"step": self.index, "seen_at_start": ctx.cursor["seen_at_start"]})


def _build_cursor(params: BaseModel, target: Any) -> list[Step]:
    assert isinstance(params, _CursorParams)
    return [_CursorStep(i) for i in range(params.steps)]


register_kind(JobKind("test.cursor_steps", _CursorParams, _build_cursor))


def test_each_step_starts_with_an_empty_cursor(tmp_path: Path) -> None:
    """Kills R27 (cursor not reset per step)."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("test.cursor_steps", {"steps": 2}, world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "succeeded"
    assert [r["seen_at_start"] for r in view.results] == [{}, {}]


def test_measured_is_the_last_steps_result(tmp_path: Path) -> None:
    """Kills R28 (measured = first result)."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("test.cursor_steps", {"steps": 3}, world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.measured["step"] == 2 == view.results[-1]["step"]


# -- approval -------------------------------------------------------------------------------------


def test_approve_reactivates_a_waiting_job_at_once(tmp_path: Path) -> None:
    """Kills R08: the view approve() returns shows the job queued, no longer waiting."""
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "waiting"
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "ok", HUMAN)
    view = world.runner.approve(job_id, token.token_id, HUMAN)
    assert view.state == "queued" and view.waiting is None
    assert world.store.get(job_id).state is JobState.QUEUED


def test_read_only_op_on_production_needs_no_token(tmp_path: Path) -> None:
    """Kills R05 (the approval gate also gates reads)."""
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit(
        "ops.sequence", {"steps": [{"op": "timeline.list", "params": {}}]}, world.locator()
    )
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert world.log.read(types=["approval.consumed", "approval.consume_refused"]) == []


# -- pump outcome ---------------------------------------------------------------------------------


def test_a_queued_job_counts_as_active(tmp_path: Path) -> None:
    """Kills R30: a render queued behind another render's slot is still active."""
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 50
    world.runner.submit("render", _render_params(world), world.locator())
    world.clock.advance(1.0)  # the first job is older: it takes the render slot
    second = world.runner.submit(
        "render",
        _render_params(world, timeline=world.ref("Main [agent]"), custom_name="b"),
        world.locator(),
    )
    outcome = world.runner.pump()
    assert world.runner.get(second).state == "queued"
    assert outcome.active == 2 and outcome.running == 1


# -- hung ops after resume / after a stop lifts ---------------------------------------------------


def _hung_marker(world: JobWorld):
    entered, release = threading.Event(), threading.Event()
    orig = world.main.AddMarker

    def hung(*a, **k):
        entered.set()
        assert release.wait(WAIT)
        return orig(*a, **k)

    world.main.AddMarker = hung  # type: ignore[method-assign]
    return entered, release


def test_resumed_job_whose_op_still_hangs_is_watched_again(tmp_path: Path) -> None:
    """Kills F35: resume keeps a still-running op tracked, RUNNING, with a fresh op window,
    so the watchdog stalls it again if it keeps hanging."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)), drain_budget=0.5)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.pump()
    entered, release = _hung_marker(world)
    try:
        world.runner.pump()
        assert entered.wait(WAIT)
        world.clock.advance(5.0)
        world.runner.pump()
        assert world.runner.get(job_id).state == "stalled"
        view = world.runner.resume(job_id)
        assert view.state == "running"
        world.runner.pump()
        assert world.runner.get(job_id).state == "running"  # fresh window, not re-stalled
        world.clock.advance(5.0)
        world.runner.pump()
        assert world.runner.get(job_id).state == "stalled"  # still hung: stalled again
    finally:
        release.set()
    assert world.runner.wait_drain(WAIT)
    world.runner.resume(job_id)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert sorted(world.main.markers) == [100]  # the hung op's own result, not a re-run


def test_stop_lifted_while_an_op_still_hangs_puts_it_back_under_the_watchdog(
    tmp_path: Path,
) -> None:
    """Kills F36: the job leaves 'waiting' once the stop lifts, so a hang is still caught."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)), drain_budget=0.5)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.pump()
    entered, release = _hung_marker(world)
    try:
        world.runner.pump()
        assert entered.wait(WAIT)
        control.stop(world.log, control.StopScope.all(), "STOP", HUMAN)
        world.runner.pump()
        assert world.runner.get(job_id).state == "waiting"
        control.resume(world.log, control.StopScope.all(), "go", HUMAN)
        world.runner.pump()
        assert world.runner.get(job_id).state == "running"
        world.clock.advance(5.0)
        world.runner.pump()
        assert world.runner.get(job_id).state == "stalled"
    finally:
        release.set()
    assert world.runner.wait_drain(WAIT)


# -- events ---------------------------------------------------------------------------------------


def test_a_long_stop_records_one_waiting_event_and_resume_is_labelled(tmp_path: Path) -> None:
    """Kills R13 (enter_wait churns: one 'waiting' event per pump) and R31 (the 'resumed'
    label is lost)."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.pump()
    control.stop(world.log, control.StopScope.all(), "lunch", HUMAN)
    for _ in range(20):
        world.runner.pump()
    kinds = [e["kind"] for e in world.runner.get(job_id).events]
    assert kinds.count("waiting") == 1
    control.resume(world.log, control.StopScope.all(), "back", HUMAN)
    world.runner.run_until_settled()
    events = world.runner.get(job_id).events
    resumed = [e for e in events if e["kind"] == "resumed"]
    assert len(resumed) == 1 and resumed[0]["data"] == {"from": "waiting"}


# -- store ----------------------------------------------------------------------------------------


def test_open_existing_store_refuses_one_missing_its_append_only_triggers(tmp_path: Path) -> None:
    """Kills S02 (the create=False trigger check disabled)."""
    import sqlite3

    import pytest

    from aar.node.jobs import JobStore

    db = tmp_path / "jobs.db"
    JobStore(db).close()
    JobStore(db, create=False).close()  # triggers present: fine
    conn = sqlite3.connect(db)
    conn.execute("DROP TRIGGER job_events_no_delete")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="missing triggers"):
        JobStore(db, create=False)
    JobStore(db).close()  # the lock was released on the failed open


def test_store_refuses_non_finite_values_and_writes_nothing(tmp_path: Path) -> None:
    """Kills S01 (allow_nan=True): a NaN would make the row invalid JSON for every reader."""
    import pytest

    from aar.node.jobs import JobRecord, JobStore
    from aar.node.locator import ProjectLocator

    store = JobStore(tmp_path / "jobs.db")
    try:
        record = JobRecord(
            job_id="job_nan",
            kind="ops.sequence",
            node_id="n",
            target=ProjectLocator((), "Show", "uid-1"),
            params={},
            results=[{"frame_rate": float("nan")}],
        )
        with pytest.raises(ValueError):
            store.save(record)
        with pytest.raises(ValueError):
            store.save(JobRecord("job_inf", "k", "n", record.target, {"x": float("inf")}))
        assert store.load_all() == []
    finally:
        store.close()


# -- stop and watchdog details --------------------------------------------------------------------


def test_stop_cancels_the_queued_op_before_the_broker_runs(tmp_path: Path) -> None:
    """Kills P-R12: the runner itself cancels a stopped job's queued op (H3); the broker's
    precheck is only the second line of defence."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.pump()
    op_id = next(
        e["data"]["op_id"] for e in world.runner.get(job_id).events if e["kind"] == "op_submitted"
    )
    control.stop(world.log, control.StopScope.all(), "STOP", HUMAN)
    world.runner.pump()
    result = world.broker.result(op_id)
    assert result.error.code == "cancelled"  # cancelled by the runner, not refused by the broker
    assert world.main.markers == {}


def test_a_long_render_poll_loop_is_not_an_op_timeout(tmp_path: Path) -> None:
    """Kills P-R22: once an op's result is in, its op timer is cleared, so a render that polls
    for longer than op_timeout (between short polls) is not stalled as a hung op."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=60.0)))
    world.project.render_complete_after = 10_000
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    for _ in range(40):
        world.clock.advance(10.0)  # 400 s of polling, each poll returns at once
        world.runner.pump()
    assert world.runner.get(job_id).state == "running"


def test_watchdog_boundaries_are_strict(tmp_path: Path) -> None:
    """Kills F55 (step boundary >=) and pins W03 (op boundary): exactly at a limit is not
    overdue; just past it is."""
    from aar.node.jobs import Watchdog as W

    dog = W(WatchdogConfig(op_timeout=10.0))
    assert (
        dog.overdue(now=10.0, step_started_at=0.0, step_timeout=10.0, op_submitted_at=None) is None
    )
    assert dog.overdue(now=10.5, step_started_at=0.0, step_timeout=10.0, op_submitted_at=None)
    assert (
        dog.overdue(now=10.0, step_started_at=None, step_timeout=None, op_submitted_at=0.0) is None
    )
    assert dog.overdue(now=10.5, step_started_at=None, step_timeout=None, op_submitted_at=0.0)
    # A deadline used up (restart / pause) is overdue as soon as time passes; None = no limit.
    assert dog.overdue(now=1.0, step_started_at=0.0, step_timeout=0.0, op_submitted_at=None)
    assert (
        dog.overdue(now=1e9, step_started_at=0.0, step_timeout=None, op_submitted_at=None) is None
    )


def test_an_op_the_library_refuses_at_submit_fails_the_job_structurally(tmp_path: Path) -> None:
    """Kills R32: a HotOpFailure from building the op (a write naming a timeline without its
    unique id) fails the job with that structured error; it does not escape the pump."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence",
        {
            "steps": [
                {
                    "op": "markers.add",
                    "params": {"timeline": {"name": "Main"}, "markers": [{"frame": 1}]},
                }
            ]
        },
        world.locator(),
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "timeline_id_required"
    assert view.error["fix_hint"]  # lifted out of the op library's detail
    assert world.main.markers == {}


def test_render_params_are_validated_at_submit(tmp_path: Path) -> None:
    """Kills D08 (no unique_id requirement) and D09 (mark validator off by one): a bad render
    is refused at submit, before any job exists."""
    import pytest

    world = JobWorld(tmp_path)
    with pytest.raises(ValueError, match="unique_id"):
        world.runner.submit(
            "render", _render_params(world, timeline={"name": "Main"}), world.locator()
        )
    with pytest.raises(ValueError, match="mark_out"):
        world.runner.submit(
            "render", _render_params(world, mark_in=10, mark_out=10), world.locator()
        )
    assert world.runner.list() == [] and world.store.load_all() == []
