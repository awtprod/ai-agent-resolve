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


# -- the broker precheck (S3 hook used by the runner) ----------------------------------------------


def _marker_op(world: JobWorld, frame: int, precheck):
    from dataclasses import replace

    from aar.node.leases import Holder
    from aar.node.ops import make_hot_op

    hot = make_hot_op(
        "markers.add",
        {"timeline": world.ref("Main"), "markers": [{"frame": frame}]},
        target=world.locator(),
        holder=Holder.agent("pin"),
    )
    return replace(hot, precheck=precheck)


def test_broker_precheck_refuses_runs_and_fails_closed(tmp_path: Path) -> None:
    """Kills F01/F02/F03/F54: a refusing precheck finishes the op cancelled with its error and
    never runs it (at the tick sweep AND right before the op); a precheck that raises refuses;
    no precheck or ``None`` runs the op."""
    from aar.node.broker import OpError, OpStatus

    world = JobWorld(tmp_path)
    calls: list[str] = []

    def refuse():
        calls.append("refuse")
        return OpError("nope", "refused by the test")

    def boom():
        raise RuntimeError("cannot check")

    ok = world.broker.submit(_marker_op(world, 1, lambda: None))
    refused = world.broker.submit(_marker_op(world, 2, refuse))
    raised = world.broker.submit(_marker_op(world, 3, boom))
    world.broker.run_until_idle()
    assert world.broker.result(ok).status is OpStatus.OK
    r = world.broker.result(refused)
    assert r.status is OpStatus.CANCELLED and r.error.code == "nope" and r.started_at is None
    r = world.broker.result(raised)
    assert r.status is OpStatus.CANCELLED and r.error.code == "precheck_failed"
    assert sorted(world.main.markers) == [1]
    assert calls == ["refuse"]  # refused at the tick sweep: never reached the batch


def test_broker_precheck_is_asked_again_right_before_the_op(tmp_path: Path) -> None:
    """Kills F01: the gate is re-asked between ops of one batch, after the sweep passed."""
    from aar.node.broker import OpError, OpStatus

    world = JobWorld(tmp_path)
    gate = {"open": True}

    def first_op_closes_the_gate(*a, **k):
        gate["open"] = False
        return orig(*a, **k)

    orig = world.main.AddMarker
    world.main.AddMarker = first_op_closes_the_gate  # type: ignore[method-assign]
    a = world.broker.submit(_marker_op(world, 1, lambda: None))
    b = world.broker.submit(
        _marker_op(world, 2, lambda: None if gate["open"] else OpError("closed", "gate closed"))
    )
    world.broker.run_until_idle()
    assert world.broker.result(a).status is OpStatus.OK
    assert world.broker.result(b).error.code == "closed"
    assert sorted(world.main.markers) == [1]


# -- render op details -----------------------------------------------------------------------------


def test_cleanup_failure_is_noted_in_measured(tmp_path: Path) -> None:
    """Kills D12 (cleanup failure note dropped) and O18 (delete_job unverified): a completed
    render job that is still listed after DeleteRenderJob is reported, not claimed removed."""
    world = JobWorld(tmp_path)
    world.project.DeleteRenderJob = lambda job_id: True  # type: ignore[method-assign]
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "succeeded"
    assert "still listed" in view.measured["cleanup_note"]
    assert len(world.project.render_jobs) == 1


def test_start_that_does_not_start_fails_and_removes_our_job(tmp_path: Path) -> None:
    """Kills O10 (start not verified): StartRendering says True but nothing renders."""
    world = JobWorld(tmp_path)
    world.project.StartRendering = lambda ids, interactive=False: True  # type: ignore[method-assign]
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "render_not_started"
    assert view.error["detail"]["cleanup"]["deleted"] is True
    assert world.project.render_jobs == []


def test_failed_render_reports_resolves_error_text(tmp_path: Path) -> None:
    """Kills O20 (poll drops the error text)."""
    world = JobWorld(tmp_path)
    world.project.render_outcome = "fail"
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.error["code"] == "render_failed"
    assert "frames offline" in view.error["message"]
    assert view.alerts == []  # the render is over: not an orphan


def test_render_removed_by_a_human_mid_poll_fails_without_an_orphan_alert(tmp_path: Path) -> None:
    """Kills O21 (an unknown job id is not refused by render.poll)."""
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 50
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    for _ in range(6):
        world.runner.pump()
    assert world.store.get(job_id).cursor.get("phase") == "poll"
    world.project.render_jobs.clear()  # a human deleted it from the queue
    world.project.rendering = False
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "render_job_unknown"
    assert view.alerts == []


def test_render_poll_reports_is_rendering(tmp_path: Path) -> None:
    """Kills O13 (is_rendering always False)."""
    from aar.node.leases import Holder
    from aar.node.ops import submit_op

    world = JobWorld(tmp_path)
    world.project.render_complete_after = 50
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    for _ in range(6):
        world.runner.pump()
    rid = world.store.get(job_id).cursor["render_job_id"]
    op_id = submit_op(
        world.broker,
        "render.poll",
        {"render_job_id": rid},
        target=world.locator(),
        holder=Holder.agent("reader"),
    )
    world.broker.run_until_idle()
    value = world.broker.result(op_id).value
    assert value.status == "Rendering" and value.is_rendering is True


# -- third batch ----------------------------------------------------------------------------------


def test_unknown_op_is_refused_at_submit(tmp_path: Path) -> None:
    """Kills K02 (unknown op not checked at submit): no job is created for it."""
    import pytest

    world = JobWorld(tmp_path)
    with pytest.raises(ValueError, match=r"no\.such\.op"):
        world.runner.submit(
            "ops.sequence", {"steps": [{"op": "no.such.op", "params": {}}]}, world.locator()
        )
    assert world.runner.list() == [] and world.store.load_all() == []


def test_structured_error_lifts_fix_hint_out_of_the_detail(tmp_path: Path) -> None:
    """Kills M01 (from_broker leaves fix_hint/trap_id in the detail)."""
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
    error = world.runner.get(job_id).error
    assert error["fix_hint"] and "fix_hint" not in error["detail"]
    assert "trap_id" not in error["detail"]


def test_store_loads_a_nodes_jobs_oldest_first(tmp_path: Path) -> None:
    """Kills S06/S06' (node load order reversed): the documented order of load_all."""
    from aar.node.jobs import JobRecord, JobStore
    from aar.node.locator import ProjectLocator

    store = JobStore(tmp_path / "jobs.db")
    try:
        target = ProjectLocator((), "Show", "uid-1")
        for job_id, created in (("job_b", 2.0), ("job_c", 3.0), ("job_a", 1.0)):
            store.save(JobRecord(job_id, "ops.sequence", "n1", target, {}, created_at=created))
        store.save(JobRecord("job_other", "ops.sequence", "n2", target, {}, created_at=0.5))
        assert [r.job_id for r in store.load_all("n1")] == ["job_a", "job_b", "job_c"]
        assert [r.job_id for r in store.load_all()] == ["job_other", "job_a", "job_b", "job_c"]
    finally:
        store.close()


def test_render_mode_and_format_are_put_back_after_a_good_build(tmp_path: Path) -> None:
    """Kills B06/B07 (restore skips the format/codec or the render mode reset)."""
    world = JobWorld(tmp_path)
    world.project.deliver.mode = 0
    world.presence.observe(world.gw.get_context(), world.clock())
    world.clock.advance(60.0)
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render",
        _render_params(world, render_format="mov", render_codec="ProRes", render_mode=1),
        world.locator(),
    )
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    rendered = world.fake.calls
    assert ("SetCurrentRenderMode", 1) in rendered or world.project.render_jobs == []
    deliver = world.project.deliver
    assert (deliver.format, deliver.codec, deliver.mode) == (
        original.format,
        original.codec,
        original.mode,
    )


def test_stopped_job_with_a_hung_op_is_not_stalled_by_the_stop_time(tmp_path: Path) -> None:
    """Kills R20' (watchdog also checks waiting jobs): a stop pauses the clocks, even when an
    op it could not cancel is still out."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)), drain_budget=0.5)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.pump()
    entered, release = _hung_marker(world)
    try:
        world.runner.pump()
        assert entered.wait(WAIT)
        control.stop(world.log, control.StopScope.all(), "STOP", HUMAN)
        world.runner.pump()
        world.clock.advance(500.0)
        world.runner.pump()
        assert world.runner.get(job_id).state == "waiting"
    finally:
        release.set()
    assert world.runner.wait_drain(WAIT)
    control.resume(world.log, control.StopScope.all(), "go", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"


def test_render_that_completes_instantly_after_start_succeeds(tmp_path: Path) -> None:
    """Kills O19 (Complete not counted as already running): a render that is Complete by the
    time the build checks it started is not 'not started'."""
    world = JobWorld(tmp_path)
    proj = world.project
    real_start = proj.StartRendering

    def instant(ids, interactive=False):
        ok = real_start(ids, interactive)
        for job in proj.render_jobs:
            if job.job_id in ids:
                job.status, job.completion = "Complete", 100
        proj.rendering = False
        return ok

    proj.StartRendering = instant  # type: ignore[method-assign]
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"


def test_adopted_job_a_human_deleted_meanwhile_fails_visibly(tmp_path: Path) -> None:
    """Kills O07 (the queue listing is not verified): the journaled JobId is gone after the
    restart, so the resumed build fails with render_job_missing instead of guessing."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.pump()
    world.broker.run_until_idle()  # the build ran; the node dies before the checkpoint
    world.project.render_jobs.clear()  # a human removed it while the node was down
    world.project.rendering = False
    runner = world.restart()
    runner.run_until_settled()
    view = runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "render_job_missing"
    assert [c for c in world.fake.calls if c[0] in ("AddRenderJob", "StartRendering")] == []


def test_added_job_missing_from_the_queue_is_not_trusted(tmp_path: Path) -> None:
    """Kills O07 on the fresh path: AddRenderJob's id must be listed exactly once."""
    world = JobWorld(tmp_path)
    proj = world.project
    real_add = proj.AddRenderJob

    def add_but_unlisted():
        rid = real_add()
        proj.render_jobs.clear()
        return rid

    proj.AddRenderJob = add_but_unlisted  # type: ignore[method-assign]
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "add_render_job_unverified"
    assert [c for c in world.fake.calls if c[0] == "StartRendering"] == []
