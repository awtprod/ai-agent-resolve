"""S7 round 3: the round-2 validator probes as permanent regression tests, plus round-3 pins.

Ported from the validator's ``test_zz_val_r2.py`` (PASS = fixed). Every probe that depended on
wall-clock timing (a ``time.sleep`` racing the 0.5 s drain budget) is rewritten to synchronise
on events instead: a hung or slow gateway call blocks on a ``threading.Event`` the test
releases, and the test waits on completion signals (``runner.wait_drain``), never on a sleep.
A crash is emulated with a ``BaseException`` raised from a Resolve call on the test thread,
which -- like a killed process -- runs no cleanup code in the op.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

import pytest

from aar.hub import control
from aar.hub.approvals import ApprovalScope, grant
from aar.hub.events import Actor
from aar.hub.registry import ProjectSource, Registry
from aar.node.broker import BrokerConfig, OpStatus
from aar.node.jobs import (
    ApprovalPolicy,
    JobRunner,
    JobState,
    JobStore,
    JobStoreLocked,
    Watchdog,
    WatchdogConfig,
)
from aar.node.jobs.render import snapshot_preset_name
from aar.node.leases import Holder, Scope, timeline_key
from aar.node.ops import submit_op

from .fake_resolve import DeliverState, FakeRenderJob
from .jobs_support import AGENT_ID, IDLE, MAIN_REF, JobWorld

HUMAN = Actor("human", "andrew")
WAIT = 30.0  # generous upper bound for an event that MUST happen; never a timing assumption


class _Crash(BaseException):
    """A process death inside a Resolve call: no ``except Exception`` cleanup runs."""


def _render_params(world: JobWorld, **over: object) -> dict:
    params = {
        "timeline": world.ref("Main"),
        "target_dir": "D:/Out",
        "custom_name": "show_v1",
        "preset": "YouTube 1080p",
        "render_format": "mov",
        "render_codec": "ProRes",
        "render_mode": 1,
        "mark_in": 0,
        "mark_out": 99,
    }
    params.update(over)
    return params


def _markers(*frames: int) -> dict:
    return {
        "steps": [
            {"op": "markers.add", "params": {"timeline": MAIN_REF, "markers": [{"frame": f}]}}
            for f in frames
        ]
    }


def _calls(world: JobWorld, name: str) -> list:
    return [c for c in world.fake.calls if c[0] == name]


def _human_job(world: JobWorld, timeline: str, name: str, status: str = "Ready") -> str:
    tl = world.project.timeline(timeline)
    job_id = "rj-human-1"
    info = {
        "JobId": job_id,
        "RenderJobName": "Human Job",
        "TimelineName": tl.GetName(),
        "TargetDir": "E:/Human",
        "OutputFilename": name,
        "MarkIn": 0,
        "MarkOut": 9,
        "FrameRate": "29.97",
    }
    world.project.render_jobs.append(FakeRenderJob(job_id, info, status=status))
    return job_id


def _prod_broker(world: JobWorld, **cfg: object) -> None:
    """Restart the node with a production BrokerConfig (the 15 s restore delay)."""
    world.restart(broker_config=BrokerConfig(**cfg))


def _pump_until(world: JobWorld, job_id: str, pred, n: int = 40) -> None:
    for _ in range(n):
        world.runner.pump()
        if pred(world.store.get(job_id)):
            return
    raise AssertionError("condition never reached")


def _aar_presets(world: JobWorld) -> list[str]:
    """The JOB's snapshot presets (aar-job-*). aar-restore-* belong to the S6 broker."""
    return [p for p in world.fake.presets if p.startswith("aar-job-")]


def _kinds(view) -> list[str]:
    return [e["kind"] for e in view.events]


def _op_ids(view, op: str) -> list[str]:
    return [
        e["data"]["op_id"]
        for e in view.events
        if e["kind"] == "op_submitted" and e["data"]["op"] == op
    ]


# == P1: a human render job is never adopted =====================================================


def test_p1a_human_ready_job_same_name_other_timeline_untouched(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    human = _human_job(world, "Main [agent]", "show_v1")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.state == "succeeded" and len(_calls(world, "AddRenderJob")) == 1
    assert all(human not in c[1] for c in _calls(world, "StartRendering"))
    assert ("DeleteRenderJob", human) not in world.fake.calls
    assert [j.job_id for j in world.project.render_jobs] == [human]


def test_p1b_human_complete_job_same_name_untouched(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    human = _human_job(world, "Main [agent]", "show_v1", status="Complete")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.state == "succeeded" and len(_calls(world, "StartRendering")) == 1
    assert ("DeleteRenderJob", human) not in world.fake.calls


# == P2: crash after the build op ran, before its result was checkpointed ========================


def test_p2_crash_between_build_and_checkpoint_does_not_duplicate(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_output_extension = ".mov"
    world.project.render_complete_after = 50
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()  # build op queued (cursor phase=build checkpointed)
    assert world.store.get(job_id).cursor.get("phase") == "build"
    world.broker.run_until_idle()  # the build runs in the process that is about to die
    first = [j.job_id for j in world.project.render_jobs]
    assert len(first) == 1 and world.project.rendering
    # The journal recorded the JobId during the op, before any checkpoint of its result.
    assert world.store.get(job_id).cursor["journal"]["render_job_id"] == first[0]
    runner = world.restart()
    for _ in range(80):
        runner.pump()
    v = runner.get(job_id)
    assert _calls(world, "AddRenderJob") == []  # no second render job on resume
    assert [j.job_id for j in world.project.render_jobs] == first
    assert v.state in ("running", "succeeded")


def test_p2b_crash_mid_build_after_add_adopts_and_finishes(tmp_path: Path) -> None:
    """The node dies inside StartRendering (after AddRenderJob, after the Deliver restore of a
    real build would... no: before the build's restore ran). The resumed build adopts the
    journaled JobId, recovers the Deliver settings from its own snapshot, and starts it."""
    world = JobWorld(tmp_path)
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()  # build queued
    real_restore_load = world.project.LoadRenderPreset

    def die_on_restore(name: str) -> bool:
        if name.startswith("aar-job-"):
            raise _Crash("power cut before the Deliver restore")
        return real_restore_load(name)

    world.project.LoadRenderPreset = die_on_restore  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        world.broker.run_until_idle()
    world.project.LoadRenderPreset = real_restore_load  # type: ignore[method-assign]
    added = [j.job_id for j in world.project.render_jobs]
    assert len(added) == 1 and not world.project.rendering
    assert world.project.deliver.settings != original.settings  # clobbered by the dead attempt
    runner = world.restart()
    runner.run_until_settled()
    v = runner.get(job_id)
    assert v.state == "succeeded", v.error
    assert _calls(world, "AddRenderJob") == []  # adopted, not re-added
    assert [c[1] for c in _calls(world, "StartRendering")] == [added]
    assert world.project.deliver.settings == original.settings
    assert _aar_presets(world) == []


def test_p2c_crash_inside_add_render_job_that_added_fails_visibly(tmp_path: Path) -> None:
    """The node dies INSIDE AddRenderJob after Resolve added the job: the id was never
    journaled. The resumed build cannot tell the new queue entry is its own, so it neither
    adopts nor re-adds: the job fails with the candidate ids and an alert, the entry is left
    untouched, and the Deliver settings are recovered."""
    world = JobWorld(tmp_path)
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    real_add = world.project.AddRenderJob

    def add_then_die() -> str:
        real_add()
        raise _Crash("died inside AddRenderJob")

    world.project.AddRenderJob = add_then_die  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        world.broker.run_until_idle()
    world.project.AddRenderJob = real_add  # type: ignore[method-assign]
    left = [j.job_id for j in world.project.render_jobs]
    runner = world.restart()
    runner.run_until_settled()
    v = runner.get(job_id)
    assert v.state == "failed" and v.error["code"] == "render_build_ambiguous"
    assert v.error["detail"]["candidates"] == left
    assert [a["code"] for a in v.alerts] == ["render_build_ambiguous"]
    assert [j.job_id for j in world.project.render_jobs] == left  # never touched
    assert _calls(world, "StartRendering") == [] and _calls(world, "DeleteRenderJob") == []
    assert world.project.deliver.settings == original.settings and _aar_presets(world) == []


def test_p2d_crash_inside_add_render_job_before_it_added_reruns(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    real_add = world.project.AddRenderJob

    def die_before_add() -> str:
        raise _Crash("died inside AddRenderJob before Resolve added anything")

    world.project.AddRenderJob = die_before_add  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        world.broker.run_until_idle()
    world.project.AddRenderJob = real_add  # type: ignore[method-assign]
    runner = world.restart()
    runner.run_until_settled()
    assert runner.get(job_id).state == "succeeded"
    assert len(_calls(world, "AddRenderJob")) == 1  # nothing was added before: one fresh add


def test_p2e_adopted_job_that_completed_meanwhile_is_not_restarted(tmp_path: Path) -> None:
    """Pins O19: an adopted render that already finished is polled, never started again."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    world.broker.run_until_idle()  # the build ran; the node dies before the checkpoint
    rj = world.project.render_jobs[0]
    rj.status, rj.completion = "Complete", 100  # it finished while the node was down
    world.project.rendering = False
    runner = world.restart()  # (the restart clears the fake's call log)
    runner.run_until_settled()
    assert runner.get(job_id).state == "succeeded"
    assert _calls(world, "StartRendering") == [] and _calls(world, "AddRenderJob") == []


# == P3: emergency stop ==========================================================================


def test_p3a_queued_write_does_not_run_after_stop(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.pump()
    control.stop(world.log, control.StopScope.all(), "STOP", HUMAN)
    world.runner.pump()
    v = world.runner.get(job_id)
    assert v.state == "waiting" and world.main.markers == {}


def test_p3b_queued_build_does_not_start_render_after_stop(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.runner.submit("render", _render_params(world), world.locator(), agent_id=AGENT_ID)
    world.runner.pump()  # build queued
    control.stop(world.log, control.StopScope.all(), "STOP", HUMAN)
    world.runner.pump()
    assert _calls(world, "StartRendering") == [] and not world.project.rendering


def test_p3c_stop_pressed_during_build_does_not_lose_render_id(tmp_path: Path) -> None:
    """The STOP lands while render.build is running: the build completes atomically (fine), and
    the runner keeps its result instead of re-running the build on resume."""
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 20
    orig = world.project.AddRenderJob
    pressed: list[int] = []

    def add_and_human_hits_stop():
        rid = orig()
        if not pressed:
            pressed.append(1)
            control.stop(world.log, control.StopScope.all(), "STOP mid-build", HUMAN)
        return rid

    world.project.AddRenderJob = add_and_human_hits_stop  # type: ignore[method-assign]
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    world.runner.pump()
    assert world.runner.get(job_id).state == "waiting"
    control.resume(world.log, control.StopScope.all(), "ok", HUMAN)
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert len(_calls(world, "AddRenderJob")) == 1
    assert v.state == "succeeded"


def test_p3c_restart_while_stopped_after_build_adopts(tmp_path: Path) -> None:
    """Same, but the node restarts while stopped (the in-memory result is lost): the journal
    still makes the resumed build adopt the render job instead of adding a second one."""
    world = JobWorld(tmp_path)
    orig = world.project.AddRenderJob

    def add_and_stop():
        rid = orig()
        control.stop(world.log, control.StopScope.all(), "STOP mid-build", HUMAN)
        return rid

    world.project.AddRenderJob = add_and_stop  # type: ignore[method-assign]
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    world.runner.pump()
    world.project.AddRenderJob = orig  # type: ignore[method-assign]
    runner = world.restart()
    control.resume(world.log, control.StopScope.all(), "ok", HUMAN)
    runner.run_until_settled()
    assert runner.get(job_id).state == "succeeded"
    assert _calls(world, "AddRenderJob") == []  # adopted after the restart
    assert len(_calls(world, "StartRendering")) == 0  # it was already rendering


def test_p3d_stop_takes_effect_between_ops_inside_one_batch(tmp_path: Path) -> None:
    """Two jobs' writes sit in ONE broker batch. The first op is slow (blocked on an event), so
    the pump returns while the drain still runs it. STOP is committed then. The second write
    must not run -- and its timeline must not even be selected."""
    world = JobWorld(tmp_path, drain_budget=0.5)
    other = world.project.timeline("Main [agent]")
    om, oo = world.main.AddMarker, other.AddMarker
    entered, release = threading.Event(), threading.Event()
    wrote2: list[int] = []

    def slow(*a, **k):
        entered.set()
        assert release.wait(WAIT)
        return om(*a, **k)

    def second(*a, **k):
        wrote2.append(1)
        return oo(*a, **k)

    world.main.AddMarker = slow  # type: ignore[method-assign]
    other.AddMarker = second  # type: ignore[method-assign]
    one = {
        "op": "markers.add",
        "params": {"timeline": world.ref("Main"), "markers": [{"frame": 1}]},
    }
    two = {
        "op": "markers.add",
        "params": {"timeline": world.ref("Main [agent]"), "markers": [{"frame": 2}]},
    }
    j1 = world.runner.submit("ops.sequence", {"steps": [one]}, world.locator(), agent_id=AGENT_ID)
    world.clock.advance(1.0)  # j1 is older, so its op is first in the batch
    j2 = world.runner.submit("ops.sequence", {"steps": [two]}, world.locator(), agent_id=AGENT_ID)
    world.runner.pump()  # both ops queued (one batch for project Show)
    world.runner.pump()  # the drain starts op 1 and blocks in it; the pump returns
    assert entered.wait(WAIT)
    assert not world.runner.wait_drain(0)  # the drain is still busy inside op 1
    control.stop(world.log, control.StopScope.all(), "STOP", HUMAN)
    release.set()
    assert world.runner.wait_drain(WAIT)
    world.runner.pump()
    assert wrote2 == [] and other.markers == {}
    op2 = _op_ids(world.runner.get(j2), "markers.add")[0]
    result = world.broker.result(op2)
    assert result.status is OpStatus.CANCELLED and result.error.code == "emergency_stop"
    assert world.project.current_timeline.GetName() == "Main"  # op 2's timeline never selected
    assert world.runner.get(j2).state == "waiting"
    assert world.runner.get(j1).state == "waiting"  # op 1 ran (already started); paused after
    control.resume(world.log, control.StopScope.all(), "go", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(j1).state == world.runner.get(j2).state == "succeeded"
    assert sorted(world.main.markers) == [1] and sorted(other.markers) == [2]


# == P4/P5: deadlines =============================================================================


def test_p4a_approval_wait_does_not_burn_deadline(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    world.clock.advance(400.0)
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "ok", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"


def test_p4b_stop_wait_does_not_burn_deadline(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    control.stop(world.log, control.StopScope.all(), "lunch", HUMAN)
    world.runner.run_until_settled()
    world.clock.advance(400.0)
    control.resume(world.log, control.StopScope.all(), "back", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"


def test_p4c_stop_mid_render_does_not_burn_deadline(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1e6)))
    world.project.render_complete_after = 10_000
    job_id = world.runner.submit(
        "render", _render_params(world, timeout_seconds=600.0), world.locator(), agent_id=AGENT_ID
    )
    _pump_until(world, job_id, lambda r: r.cursor.get("phase") == "poll")
    world.clock.advance(100.0)
    world.runner.pump()
    control.stop(world.log, control.StopScope.all(), "long stop", HUMAN)
    world.runner.pump()
    world.clock.advance(1000.0)
    control.resume(world.log, control.StopScope.all(), "go", HUMAN)
    for _ in range(4):
        world.runner.pump()
    assert world.runner.get(job_id).state == "running"


def test_p5_restart_while_waiting_keeps_step_watchdog(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1e6)))
    world.project.render_outcome = "hang"
    job_id = world.runner.submit(
        "render", _render_params(world, timeout_seconds=30.0), world.locator(), agent_id=AGENT_ID
    )
    control.stop(world.log, control.StopScope.all(), "pause", HUMAN)
    world.runner.run_until_settled()
    runner = world.restart()
    control.resume(world.log, control.StopScope.all(), "go", HUMAN)
    for _ in range(40):
        runner.pump()
    for _ in range(20):
        world.clock.advance(1000.0)
        runner.pump()
    assert runner.get(job_id).state == "stalled"


def test_p5b_restart_loop_shorter_than_deadline_still_stalls(tmp_path: Path) -> None:
    """A node that restarts every 20 s during a hung 30 s render step still stalls it: the
    remaining deadline is persisted at every checkpoint, not reset on load."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1e6)))
    world.project.render_outcome = "hang"
    job_id = world.runner.submit(
        "render", _render_params(world, timeout_seconds=30.0), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(10):
        world.runner.pump()
    states = []
    for _ in range(10):  # 10 restarts x 20 s = 200 s of a 30 s deadline
        world.clock.advance(20.0)
        world.runner.pump()
        states.append(world.runner.get(job_id).state)
        if states[-1] == "stalled":
            break
        world.restart()
        for _ in range(3):
            world.runner.pump()
    assert "stalled" in states
    assert states.index("stalled") <= 2  # within the second 20 s window, not the tenth
    assert world.runner.get(job_id).stall["code"] == "watchdog_step_timeout"


def test_p5c_crash_loop_before_any_checkpoint_stalls_after_the_restart_limit(
    tmp_path: Path,
) -> None:
    """The process dies in the same op every time, before anything is checkpointed (so the
    persisted deadline never moves). The persistent restart counter stalls it."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.pump()  # running, op queued
    real = world.main.AddMarker

    def die(*a, **k):
        raise _Crash("the op kills the node")

    restarts = []
    for _ in range(10):
        world.main.AddMarker = die  # type: ignore[method-assign]
        with pytest.raises(_Crash):
            world.broker.run_until_idle()
        world.main.AddMarker = real  # type: ignore[method-assign]
        runner = world.restart()
        rec = world.store.get(job_id)
        restarts.append(rec.restarts)
        if rec.state is JobState.STALLED:
            break
        runner.pump()  # the op is re-queued (and will kill the node again)
    view = world.runner.get(job_id)
    assert view.state == "stalled" and view.stall["code"] == "crash_loop"
    assert restarts[-1] == JobRunner.__init__.__kwdefaults__["max_restarts"] + 1
    assert world.main.markers == {}
    world.runner.resume(job_id)  # the explicit recovery resets the counter
    assert world.store.get(job_id).restarts == 0
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"


def test_restarts_counter_resets_when_a_step_completes(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence", _markers(100, 200), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    world.restart()
    assert world.store.get(job_id).restarts == 1
    world.runner.pump()
    world.runner.pump()  # step 0 completes
    assert world.store.get(job_id).current_step == 1
    assert world.store.get(job_id).restarts == 0


# == P6: refused token =============================================================================


def test_p6_refused_token_not_retried(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_LOAD, world.pid(), 3600, "wrong", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    for _ in range(50):
        world.runner.pump()
    assert len(world.log.read(types=["approval.consume_refused"])) == 1


# == P7: production broker restore around a blocked build =======================================


def test_p7_lease_blocked_build_then_restore_renders_job_settings(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    _prod_broker(world)
    original = world.project.deliver.copy()
    lease = world.leases.acquire(
        Holder.human("andrew"),
        Scope.TIMELINE,
        timeline_key(world.pid(), world.main.GetUniqueId()),
        None,
    )
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    world.runner.pump()
    world.clock.advance(20.0)
    world.runner.pump()
    world.leases.release(lease.id, Holder.human("andrew"))
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.state == "succeeded" and v.measured["output_path"] == "D:/Out/show_v1"
    assert world.project.deliver.settings == original.settings


def test_p7b_stall_resume_in_build_phase(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    _prod_broker(world)
    lease = world.leases.acquire(
        Holder.human("andrew"),
        Scope.TIMELINE,
        timeline_key(world.pid(), world.main.GetUniqueId()),
        None,
    )
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(14):
        world.runner.pump()
        world.clock.advance(10.0)
    stalled = world.runner.get(job_id).state
    world.leases.release(lease.id, Holder.human("andrew"))
    world.runner.resume(job_id)
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert stalled == "stalled" and v.state == "succeeded"
    assert v.measured["output_path"] == "D:/Out/show_v1"


# == P8/P9/P10: Deliver restored, no preset leak ================================================


def test_p8_no_preset_left_and_deliver_restored_after_build(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 5
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    _pump_until(world, job_id, lambda r: r.cursor.get("phase") == "poll")
    assert _aar_presets(world) == [] and world.project.deliver.settings == original.settings


def test_p9_cancel_mid_render_alerts_the_orphaned_render(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 50
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    _pump_until(world, job_id, lambda r: r.cursor.get("phase") == "poll")
    rid = world.store.get(job_id).cursor["render_job_id"]
    world.runner.cancel(job_id)
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert _aar_presets(world) == [] and world.project.deliver.settings == original.settings
    # The render keeps running in Resolve; the job says so instead of leaving it silently.
    assert [(a["code"], a["render_job_id"]) for a in v.alerts] == [("render_orphaned", rid)]
    assert world.runner.alerts()[0]["job_id"] == job_id


def test_p10_stall_mid_render(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1e6)))
    world.project.render_outcome = "hang"
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world, timeout_seconds=30.0), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(15):
        world.runner.pump()
    world.clock.advance(35.0)
    world.runner.pump()
    assert world.runner.get(job_id).state == "stalled"
    assert world.project.deliver.settings == original.settings
    assert [j.status for j in world.project.render_jobs] == ["Rendering"]  # never killed


# == P11: single writer ===========================================================================


def test_p11_second_store_open_is_refused(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    with pytest.raises(JobStoreLocked):
        JobStore(world.db_path)


# == P12: poison rows ==============================================================================


def _forge(world: JobWorld, **cols: object) -> None:
    good = world.runner.submit("ops.sequence", _markers(300), world.locator(), agent_id=AGENT_ID)
    world.runner.close()
    world.store.close()
    conn = sqlite3.connect(world.db_path)
    names = [d[0] for d in conn.execute("SELECT * FROM jobs").description]
    row = dict(
        zip(
            names,
            conn.execute("SELECT * FROM jobs WHERE job_id=?", (good,)).fetchone(),
            strict=True,
        )
    )
    row.update(job_id="job_forged", **cols)
    conn.execute(
        f"INSERT INTO jobs ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
        list(row.values()),
    )
    conn.commit()
    conn.close()
    world.store = JobStore(world.db_path)


def _raw_row(world: JobWorld, job_id: str) -> tuple:
    conn = sqlite3.connect(world.db_path)
    try:
        return conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    finally:
        conn.close()


@pytest.mark.parametrize(
    "cols",
    [
        {"kind": "render", "params": '{"nope": 1}'},
        {"kind": "no.such.kind"},
        {"state": "bogus"},
        {"cursor": "{not json"},
        {"results": "[1,"},
        {"target": '{"name": 5}'},
        {"error": '{"code": 1}'},
    ],
    ids=[
        "bad-params",
        "bad-kind",
        "bad-state",
        "bad-json",
        "bad-results",
        "bad-target",
        "bad-error",
    ],
)
def test_p12_poison_row_does_not_brick_runner(tmp_path: Path, cols: dict) -> None:
    world = JobWorld(tmp_path)
    _forge(world, **cols)
    before = _raw_row(world, "job_forged")
    runner = world.restart()  # must NOT raise
    bad = runner.get("job_forged")
    assert bad.state == "failed" and bad.error["code"] == "quarantined_bad_record"
    runner.run_until_settled()  # the good job still runs
    assert [v.state for v in runner.list() if v.job_id != "job_forged"] == ["succeeded"]
    assert runner.cancel("job_forged") is False and runner.resume("job_forged").state == "failed"
    world.restart()  # quarantined again, the row untouched, the event recorded once
    assert _raw_row(world, "job_forged") == before
    assert [e.kind for e in world.store.events("job_forged")].count("quarantined") == 1


def test_p12_quarantined_record_is_never_written(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    _forge(world, state="bogus")
    world.restart()
    rec = world.runner._jobs["job_forged"]
    assert rec.readonly
    with pytest.raises(ValueError):
        world.store.save(rec)


# == P13: forged consumed flag ====================================================================


def test_p13_forged_consumed_flag(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    world.runner.close()
    world.store.close()
    conn = sqlite3.connect(world.db_path)
    conn.execute(
        "UPDATE jobs SET approval_consumed=1, state='queued', waiting=NULL WHERE job_id=?",
        (job_id,),
    )
    conn.commit()
    conn.close()
    world.store = JobStore(world.db_path)
    runner = world.restart(approval=ApprovalPolicy(live=True))
    runner.run_until_settled()
    assert world.main.markers == {}


def _forge_consumed(world: JobWorld, job_id: str, token_id: str) -> None:
    world.runner.close()
    world.store.close()
    conn = sqlite3.connect(world.db_path)
    conn.execute(
        "UPDATE jobs SET approval_consumed=1, approval_token=? WHERE job_id=?", (token_id, job_id)
    )
    conn.commit()
    conn.close()
    world.store = JobStore(world.db_path)


def test_p13b_forged_flag_reusing_a_token_this_node_consumed_for_another_job(
    tmp_path: Path,
) -> None:
    """Job A legitimately consumes token T. A forged row for job B (in ANOTHER project) claims
    T as its consumed approval. The consume is bound to job A, so B's claim is not verified."""
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    a = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "A only", HUMAN)
    world.runner.approve(a, token.token_id, HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(a).state == "succeeded"
    b = world.runner.submit(
        "ops.sequence",
        {
            "steps": [
                {
                    "op": "timeline.duplicate",
                    "params": {
                        "source": world.ref("Other main", "Other"),
                        "name": "Forged",
                        "if_exists": "error",
                    },
                }
            ]
        },
        world.locator("Other"),
        agent_id=AGENT_ID,
    )
    _forge_consumed(world, b, token.token_id)
    runner = world.restart(approval=ApprovalPolicy(live=True))
    runner.run_until_settled()
    other = world.fake.project("Other")
    assert not any(t.GetName() == "Forged" for t in other.timelines)
    assert "approval_unverified" in _kinds(runner.get(b))


def test_p13c_forged_flag_reusing_another_jobs_token_in_the_same_project(tmp_path: Path) -> None:
    """Same project, same scope and target: only the per-job binding catches it."""
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    a = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "A only", HUMAN)
    world.runner.approve(a, token.token_id, HUMAN)
    world.runner.run_until_settled()
    b = world.runner.submit("ops.sequence", _markers(200), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    _forge_consumed(world, b, token.token_id)
    runner = world.restart(approval=ApprovalPolicy(live=True))
    runner.run_until_settled()
    assert sorted(world.main.markers) == [100]  # B never wrote
    assert runner.get(b).state == "waiting"
    consumed = world.log.read(types=["approval.consumed"])
    # The consume is audited as the job's agent, bound to job A (P2, P13b).
    assert [e.actor for e in consumed] == [Actor("agent", f"{AGENT_ID}#{a}")]


def test_p13d_genuine_consume_survives_restart(tmp_path: Path) -> None:
    """The binding does not break the legitimate case: a job's own consume verifies on load."""
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit(
        "ops.sequence", _markers(100, 200), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "ok", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    for _ in range(3):
        world.runner.pump()
    assert world.store.get(job_id).approval_consumed
    runner = world.restart(approval=ApprovalPolicy(live=True))
    assert world.store.get(job_id).approval_consumed  # verified, kept
    runner.run_until_settled()
    assert runner.get(job_id).state == "succeeded"
    assert sorted(world.main.markers) == [100, 200]


# == P14: non-finite values ======================================================================


def test_p14_inf_timeout_rejected_no_ghost(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    with pytest.raises(ValueError):
        world.runner.submit(
            "ops.sequence",
            {
                "steps": [
                    {
                        "op": "markers.add",
                        "params": {"timeline": MAIN_REF, "markers": [{"frame": 100}]},
                        "timeout": float("inf"),
                    }
                ]
            },
            world.locator(),
            agent_id=AGENT_ID,
        )
    assert len(world.runner.list()) == len(world.store.load_all())


def test_p14b_nan_in_step_params_no_ghost_no_wedge(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    other = world.runner.submit("ops.sequence", _markers(200), world.locator(), agent_id=AGENT_ID)
    # Rejected at submit, or stored as JSON null and refused when the op is built.
    with contextlib.suppress(ValueError):
        world.runner.submit(
            "ops.sequence",
            {
                "steps": [
                    {
                        "op": "markers.add",
                        "params": {
                            "timeline": MAIN_REF,
                            "markers": [{"frame": 100}],
                            "nan_here": float("nan"),
                        },
                    }
                ]
            },
            world.locator(),
            agent_id=AGENT_ID,
        )
    assert len(world.runner.list()) == len(world.store.load_all())  # no ghost job
    world.runner.run_until_settled()
    assert world.runner.get(other).state == "succeeded"


# == P15: hung gateway call =======================================================================


def _hang(world: JobWorld, name: str):
    """Make ``project.<name>`` block until released. Returns (entered, release)."""
    entered, release = threading.Event(), threading.Event()
    orig = getattr(world.project, name)

    def hung(*a, **k):
        entered.set()
        assert release.wait(WAIT)
        return orig(*a, **k)

    setattr(world.project, name, hung)
    return entered, release


def test_p15_hung_build_landing_after_stall_and_cancel_undoes_itself(tmp_path: Path) -> None:
    """A hung AddRenderJob: the pump returns and the watchdog stalls the job; the job is then
    cancelled. When the hang ends, the build sees the job no longer wants it and deletes the
    render job it just added (by its JobId) -- no orphan render, Deliver restored."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)), drain_budget=0.5)
    world.project.render_complete_after = 50
    original = world.project.deliver.copy()
    entered, release = _hang(world, "AddRenderJob")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()  # build queued
    world.runner.pump()  # the drain starts the build; it hangs in AddRenderJob
    assert entered.wait(WAIT)
    assert not world.runner.wait_drain(0)  # the pump returned while the op still hangs
    world.clock.advance(5.0)
    world.runner.pump()
    assert world.runner.get(job_id).state == "stalled"
    assert world.runner.cancel(job_id)
    release.set()
    assert world.runner.wait_drain(WAIT)
    world.runner.pump()
    v = world.runner.get(job_id)
    assert world.project.render_jobs == [], "a render started after its job was cancelled"
    assert _calls(world, "StartRendering") == []
    assert _aar_presets(world) == [] and world.project.deliver.settings == original.settings
    assert v.state == "cancelled" and v.alerts == []
    late = [e for e in v.events if e["kind"] == "late_result"]
    assert len(late) == 1 and late[0]["data"]["code"] == "render_build_aborted"


def test_p15b_render_started_after_cancel_is_alerted_not_silent(tmp_path: Path) -> None:
    """The hang is inside StartRendering itself, past the last point the build can back out:
    the render starts after the job was cancelled. It is reported, with its JobId."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)), drain_budget=0.5)
    world.project.render_complete_after = 50
    entered, release = _hang(world, "StartRendering")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    world.runner.pump()
    assert entered.wait(WAIT)
    world.runner.cancel(job_id)
    release.set()
    assert world.runner.wait_drain(WAIT)
    world.runner.pump()
    v = world.runner.get(job_id)
    rid = world.project.render_jobs[0].job_id
    assert [(a["code"], a["render_job_id"]) for a in v.alerts] == [("render_orphaned", rid)]


def test_p15c_hung_op_does_not_block_reads(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)), drain_budget=0.5)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.pump()
    entered, release = threading.Event(), threading.Event()
    orig = world.main.AddMarker

    def hung(*a, **k):
        entered.set()
        assert release.wait(WAIT)
        return orig(*a, **k)

    world.main.AddMarker = hung  # type: ignore[method-assign]
    try:
        world.runner.pump()
        assert entered.wait(WAIT)
        got: list = []
        reader = threading.Thread(target=lambda: got.append(world.runner.get(job_id)))
        reader.start()
        reader.join(WAIT)
        assert got and got[0].state == "running"
    finally:
        release.set()
    assert world.runner.wait_drain(WAIT)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"  # the late result was collected


def test_p15d_stalled_job_keeps_its_hung_op_and_resumes_with_its_result(tmp_path: Path) -> None:
    """A stall does not drop an op that is still running: when it comes back, the resumed job
    uses its result instead of running it again."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)), drain_budget=0.5)
    entered, release = _hang(world, "AddRenderJob")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()
    world.runner.pump()
    assert entered.wait(WAIT)
    world.clock.advance(5.0)
    world.runner.pump()
    assert world.runner.get(job_id).state == "stalled"
    release.set()
    assert world.runner.wait_drain(WAIT)
    world.runner.resume(job_id)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    # The hung build's own result was used: one render job ever added, one start.
    assert len(_calls(world, "AddRenderJob")) == 1
    assert len(_calls(world, "StartRendering")) == 1


# == P16: restart mid-render + the broker's own restore ==========================================


def test_p16_restart_mid_render_broker_restore(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    _prod_broker(world)
    original = world.project.deliver.copy()
    world.project.render_complete_after = 6
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    _pump_until(world, job_id, lambda r: r.cursor.get("phase") == "poll")
    _prod_broker(world)  # the node restarts mid-render
    world.runner.run_until_settled()
    for _ in range(3):
        world.clock.advance(30.0)
        world.runner.pump()
    assert world.project.deliver.settings == original.settings


def test_n_broker_restore_of_human_context_still_happens(tmp_path: Path) -> None:
    """Finding 1: the broker restores the human's project after the queue drains, because the
    runner ticks it on every pump (it never did once nothing was pending)."""
    world = JobWorld(tmp_path)
    _prod_broker(world)
    assert world.fake.current.GetName() == "Show"
    job_id = world.runner.submit(
        "ops.sequence",
        {"steps": [{"op": "timeline.list", "params": {}}]},
        world.locator("Other"),
        agent_id=AGENT_ID,
    )
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert world.fake.current.GetName() == "Other" and world.broker.in_control
    for _ in range(6):
        world.clock.advance(30.0)
        world.runner.pump()
    assert world.fake.current.GetName() == "Show" and not world.broker.in_control
    assert world.broker.last_restore is not None and world.broker.last_restore.ok


# == P17/P18 ======================================================================================


def _show_registered(world: JobWorld, name: str = "Show", library: str = "CCN"):
    return Registry(world.log).create_project(
        name,
        HUMAN,
        sources=[ProjectSource("cloud_library", {"library": library, "name": "Show"})],
    )


def test_p18_registry_stop_reaches_job(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    show = _show_registered(world)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    control.stop(world.log, control.StopScope.project(show.id), "stop show", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "waiting" and world.main.markers == {}


def test_p18b_registry_rename_does_not_break_stop(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    reg = Registry(world.log)
    show = _show_registered(world)
    reg.rename_project(show.id, "Show (Season 2)", HUMAN)  # display name only
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    control.stop(world.log, control.StopScope.project(show.id), "stop show", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "waiting" and world.main.markers == {}


def test_p18c_duplicate_registry_name_does_not_break_stop(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    _show_registered(world, library="Archive 2024")
    current = _show_registered(world, library="CCN")
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    control.stop(world.log, control.StopScope.project(current.id), "stop show", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "waiting" and world.main.markers == {}


def test_p18d_stop_reaches_job_by_alias_or_uid_in_a_source(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    reg = Registry(world.log)
    by_alias = reg.create_project("Season Two", HUMAN)
    reg.add_alias(by_alias.id, "Show", HUMAN)
    by_uid = reg.create_project(
        "Renamed elsewhere", HUMAN, sources=[ProjectSource("local_db", {"uid": world.pid()})]
    )
    assert world.hublink.registry_ids(world.locator()) == sorted([by_alias.id, by_uid.id])
    for pid in (by_alias.id, by_uid.id):
        job_id = world.runner.submit(
            "ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID
        )
        control.stop(world.log, control.StopScope.project(pid), "stop", HUMAN)
        world.runner.run_until_settled()
        assert world.runner.get(job_id).state == "waiting"
        control.resume(world.log, control.StopScope.project(pid), "go", HUMAN)
        world.runner.cancel(job_id)
    assert world.main.markers == {}


def test_p18e_unrelated_registry_project_stop_does_not_pause(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    reg = Registry(world.log)
    unrelated = reg.create_project(
        "Other", HUMAN, sources=[ProjectSource("cloud_library", {"library": "CCN", "name": "X"})]
    )
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    control.stop(world.log, control.StopScope.project(unrelated.id), "stop other", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"


# == P19/P20/P21/P22 ==============================================================================


def test_p19_build_on_job_timeline(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    proj = world.project
    proj.current_timeline = proj.timeline("Main [agent]")
    world.presence.observe(world.gw.get_context(), world.clock())
    world.clock.advance(IDLE)
    seen = []
    o_set, o_add = proj.SetRenderSettings, proj.AddRenderJob

    def s(x):
        seen.append(("Set", proj.current_timeline.GetName()))
        return o_set(x)

    def a():
        seen.append(("Add", proj.current_timeline.GetName()))
        return o_add()

    proj.SetRenderSettings, proj.AddRenderJob = s, a  # type: ignore[method-assign]
    world.runner.submit("render", _render_params(world), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    assert seen == [("Set", "Main"), ("Add", "Main")]


def test_p20_two_renders_same_project(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    a = world.runner.submit(
        "render",
        _render_params(world, target_dir="D:/A", custom_name="a_v1", cleanup=False),
        world.locator(),
        agent_id=AGENT_ID,
    )
    b = world.runner.submit(
        "render",
        _render_params(
            world,
            timeline=world.ref("Main [agent]"),
            target_dir="D:/B",
            custom_name="b_v1",
            cleanup=False,
        ),
        world.locator(),
        agent_id=AGENT_ID,
    )
    world.runner.run_until_settled()
    va, vb = world.runner.get(a), world.runner.get(b)
    q = [(j.job_id, j.info["TimelineName"], j.info["TargetDir"]) for j in world.project.render_jobs]
    assert va.measured["output_path"] == "D:/A/a_v1" and vb.measured["output_path"] == "D:/B/b_v1"
    assert {t for _, t, _ in q} == {"Main", "Main [agent]"}


def test_p21_render_does_not_stall_other_project(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 10_000
    world.runner.submit("render", _render_params(world), world.locator(), agent_id=AGENT_ID)
    for _ in range(30):
        world.runner.pump()
        if world.project.rendering:
            break
    o = world.runner.submit(
        "ops.sequence",
        {"steps": [{"op": "timeline.list", "params": {}}]},
        world.locator("Other"),
        agent_id=AGENT_ID,
    )
    for _ in range(14):
        world.clock.advance(10.0)
        world.runner.pump()
    assert world.runner.get(o).state != "stalled"


def test_p22_events_bounded(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 200
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    n = len(world.store.events(job_id))
    assert n < 30  # 200 polls; a per-poll event (mutant N22) would be 200+


# == N: render.build failure paths ===============================================================


def _fault(world: JobWorld, name: str, mode: str) -> None:
    proj = world.project
    orig = getattr(proj, name)

    def bad(*a, **k):
        if mode == "raise":
            raise RuntimeError(f"{name} blew up")
        if name == "AddRenderJob":
            return ""
        return False

    if name == "LoadRenderPreset":  # only the delivery preset fails, not the snapshot restore

        def bad_load(preset, *a, **k):
            if preset == "YouTube 1080p":
                return bad()
            return orig(preset, *a, **k)

        setattr(proj, name, bad_load)
    else:
        setattr(proj, name, bad)


@pytest.mark.parametrize(
    "fault",
    [
        ("SaveAsNewRenderPreset", "false"),
        ("LoadRenderPreset", "false"),
        ("SetRenderSettings", "false"),
        ("SetCurrentRenderFormatAndCodec", "false"),
        ("SetCurrentRenderMode", "false"),
        ("AddRenderJob", "false"),
        ("StartRendering", "false"),
        ("SetRenderSettings", "raise"),
        ("AddRenderJob", "raise"),
        ("StartRendering", "raise"),
    ],
    ids=lambda f: f"{f[0]}-{f[1]}",
)
def test_n1_build_failure_at_each_step(tmp_path: Path, fault: tuple[str, str]) -> None:
    world = JobWorld(tmp_path)
    original = world.project.deliver.copy()
    if fault[0] == "SetCurrentRenderFormatAndCodec":
        orig = world.project.SetCurrentRenderFormatAndCodec
        world.project.SetCurrentRenderFormatAndCodec = (  # type: ignore[method-assign]
            lambda f, c: False if (f, c) == ("mov", "ProRes") else orig(f, c)
        )
    elif fault[0] == "SetCurrentRenderMode":
        world.project.deliver.mode = 0
        world.presence.observe(world.gw.get_context(), world.clock())
        world.clock.advance(IDLE)
        original = world.project.deliver.copy()
        orig_m = world.project.SetCurrentRenderMode
        world.project.SetCurrentRenderMode = (  # type: ignore[method-assign]
            lambda m: False if m == 1 else orig_m(m)
        )
    else:
        _fault(world, *fault)
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.state == "failed"
    assert world.project.deliver.settings == original.settings
    assert (world.project.deliver.format, world.project.deliver.codec) == (
        original.format,
        original.codec,
    )
    assert _aar_presets(world) == []
    assert world.project.render_jobs == [], "the failed build left its job in the human's queue"
    assert v.alerts == []


def test_n1_start_failure_deletes_only_our_job(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    human = _human_job(world, "Main [agent]", "other")
    _fault(world, "StartRendering", "false")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.error["code"] == "start_rendering_failed"
    assert v.error["detail"]["cleanup"]["deleted"] is True
    assert [j.job_id for j in world.project.render_jobs] == [human]
    assert [c[1] for c in _calls(world, "DeleteRenderJob")] == [
        v.error["detail"]["cleanup"]["render_job_id"]
    ]


def test_n1_job_we_cannot_delete_is_alerted_with_its_id(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    _fault(world, "StartRendering", "false")
    world.project.DeleteRenderJob = lambda job_id: False  # type: ignore[method-assign]
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    rid = world.project.render_jobs[0].job_id
    assert v.error["code"] == "start_rendering_failed"
    assert v.error["detail"]["cleanup"]["orphaned"] is True
    assert [(a["code"], a["render_job_id"]) for a in v.alerts] == [("render_orphaned", rid)]


def test_n1b_restore_failure_does_not_mask_the_original_error(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.SetRenderSettings = lambda s: False  # type: ignore[method-assign]
    world.project.DeleteRenderPreset = lambda p: False  # type: ignore[method-assign]
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.error["code"] == "render_settings_failed"  # the PRIMARY error
    assert v.error["detail"]["restore_error"]["code"] == "preset_delete_failed"
    assert [a["code"] for a in v.alerts] == ["deliver_not_restored"]


def test_n1c_restore_while_rendering_refused(tmp_path: Path) -> None:
    """If Resolve refuses LoadRenderPreset while a render is in progress, nothing breaks: the
    build restores Deliver BEFORE it starts the render."""
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 5
    proj = world.project
    original = proj.deliver.copy()
    orig = proj.LoadRenderPreset

    def load(name):
        if proj.rendering:
            return False
        return orig(name)

    proj.LoadRenderPreset = load  # type: ignore[method-assign]
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.state == "succeeded" and proj.deliver.settings == original.settings
    assert _aar_presets(world) == []


def test_n1c_restore_failure_after_add_keeps_tracking_the_render(tmp_path: Path) -> None:
    """The restore fails but the render job is fine: the render runs and is TRACKED to the
    end; the restore failure is reported separately (measured + alert), and the snapshot
    preset that holds the originals is kept for the human."""
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 3
    proj = world.project
    orig = proj.LoadRenderPreset
    proj.LoadRenderPreset = lambda name: False if name.startswith("aar-job-") else orig(name)  # type: ignore[method-assign]
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.state == "succeeded"
    assert v.measured["deliver_restore_error"]["code"] == "preset_load_failed"
    assert [a["code"] for a in v.alerts] == ["deliver_not_restored"]
    assert _aar_presets(world) == [snapshot_preset_name(job_id)]  # kept: it holds the originals


def test_n2_human_preset_with_snapshot_name_is_never_loaded_or_deleted(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    name = snapshot_preset_name(job_id)
    human = DeliverState("mxf", "DNxHR", 0, {"TargetDir": "H:/Humans"})
    world.fake.presets[name] = human
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert world.project.deliver.settings == original.settings
    assert world.fake.presets[name] is human  # still there, untouched
    assert ("LoadRenderPreset", name) not in world.fake.calls
    assert ("DeleteRenderPreset", name) not in world.fake.calls
    assert ("SaveAsNewRenderPreset", f"{name}-2") in world.fake.calls


def test_n2b_all_snapshot_names_taken_fails_without_touching_them(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    base = snapshot_preset_name(job_id)
    names = [base] + [f"{base}-{i}" for i in range(2, 11)]
    for n in names:
        world.fake.presets[n] = DeliverState("mxf", "DNxHR", 0, {})
    world.runner.run_until_settled()
    v = world.runner.get(job_id)
    assert v.error["code"] == "snapshot_preset_names_taken"
    assert all(n in world.fake.presets for n in names)
    assert world.project.render_jobs == []


def test_n3_crash_mid_build_recovers_originals(tmp_path: Path) -> None:
    """A real crash inside the build after the snapshot and the clobber (no cleanup ran):
    Resolve is left with the snapshot preset and agent settings, no render job. The journal
    says the preset is the job's own, so the resumed build recovers the originals from it."""
    world = JobWorld(tmp_path)
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()  # build queued and checkpointed
    real = world.project.SetCurrentRenderFormatAndCodec

    def die(f, c):
        real(f, c)  # clobbered
        raise _Crash("power cut mid-build")

    world.project.SetCurrentRenderFormatAndCodec = die  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        world.broker.run_until_idle()
    world.project.SetCurrentRenderFormatAndCodec = real  # type: ignore[method-assign]
    assert _aar_presets(world) == [snapshot_preset_name(job_id)]
    assert world.project.deliver.format != original.format
    runner = world.restart()
    runner.run_until_settled()
    assert runner.get(job_id).state == "succeeded"
    assert world.project.deliver.settings == original.settings
    assert world.project.deliver.format == original.format and _aar_presets(world) == []
    assert len(world.project.render_jobs) == 0  # cleanup deleted the completed job
    assert len(_calls(world, "AddRenderJob")) == 1


def test_render_build_outside_a_job_is_refused(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    op_id = submit_op(
        world.broker,
        "render.build",
        {
            "timeline": world.ref("Main"),
            "target_dir": "D:/Out",
            "custom_name": "x",
            "snapshot_preset": "aar-x",
        },
        target=world.locator(),
        holder=Holder.agent("someone"),
    )
    world.broker.run_until_idle()
    result = world.broker.result(op_id)
    assert result.status is OpStatus.FAILED and result.error.code == "render_needs_job"
    assert world.project.render_jobs == [] and _aar_presets(world) == []
    assert not any(c[1].startswith("aar-job") for c in _calls(world, "SaveAsNewRenderPreset"))


@pytest.mark.parametrize("op", ["render.build", "render.delete_job"])
def test_render_ops_are_refused_in_ops_sequence(tmp_path: Path, op: str) -> None:
    world = JobWorld(tmp_path)
    human = _human_job(world, "Main", "theirs")
    with pytest.raises(ValueError, match="render"):
        world.runner.submit(
            "ops.sequence",
            {"steps": [{"op": op, "params": {"render_job_id": human}}]},
            world.locator(),
            agent_id=AGENT_ID,
        )
    assert [j.job_id for j in world.project.render_jobs] == [human]


# == N5/N6/N10 + drain =============================================================================


def test_n5_concurrent_pumps_share_one_drain(tmp_path: Path) -> None:
    """Finding 9: two concurrent pumps start ONE drain; Resolve never sees two calls at once."""
    world = JobWorld(tmp_path, drain_budget=0.5)
    tl2 = world.project.timeline("Main [agent]")
    active, peak = [0], [0]
    lock = threading.Lock()
    entered, release = threading.Event(), threading.Event()

    def wrap(orig):
        def f(*a, **k):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            entered.set()
            assert release.wait(WAIT)
            with lock:
                active[0] -= 1
            return orig(*a, **k)

        return f

    world.main.AddMarker = wrap(world.main.AddMarker)  # type: ignore[method-assign]
    tl2.AddMarker = wrap(tl2.AddMarker)  # type: ignore[method-assign]
    drains = [0]
    real_drain = world.runner._drain

    def counting(done):
        drains[0] += 1
        real_drain(done)

    world.runner._drain = counting  # type: ignore[method-assign]
    for timeline, frame in (("Main", 100), ("Main [agent]", 200)):
        step = {
            "op": "markers.add",
            "params": {"timeline": world.ref(timeline), "markers": [{"frame": frame}]},
        }
        world.runner.submit("ops.sequence", {"steps": [step]}, world.locator(), agent_id=AGENT_ID)
    world.runner.pump()  # both queued
    before = drains[0]
    ts = [threading.Thread(target=world.runner.pump) for _ in range(2)]
    for t in ts:
        t.start()
    assert entered.wait(WAIT)
    for t in ts:
        t.join(WAIT)
        assert not t.is_alive()
    # At most ONE drain served both concurrent pumps (none if the first pump's drain was
    # still running -- e.g. under the test-only drain delay -- and they shared that one).
    assert drains[0] - before <= 1
    release.set()
    assert world.runner.wait_drain(WAIT)
    world.runner.run_until_settled()
    assert peak[0] == 1
    assert sorted(world.main.markers) == [100] and sorted(tl2.markers) == [200]


def test_n6_flock_released_when_owner_process_dies(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    db = tmp_path / "jobs.db"
    code = (
        "import sys,time; from aar.node.jobs import JobStore; "
        f"s=JobStore({str(db)!r}); print('locked', flush=True); time.sleep(60)"
    )
    p = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=dict(os.environ)
    )
    try:
        assert p.stdout.readline().strip() == "locked"
        with pytest.raises(JobStoreLocked):
            JobStore(db)
        p.kill()  # my own child
        p.wait(WAIT)
        JobStore(db).close()  # the kernel released the dead owner's lock
    finally:
        if p.poll() is None:
            p.kill()
        p.wait(WAIT)


def test_n10_unknown_project_and_sandbox_exempt_types() -> None:
    p = ApprovalPolicy()
    assert p.live and p.is_production(None) and p.is_production("x")
    with pytest.raises(TypeError):
        ApprovalPolicy(sandbox_exempt="proj-1")  # type: ignore[arg-type]
    assert ApprovalPolicy(sandbox_exempt=frozenset({"proj-1"})).is_production("proj")


def test_broker_crash_is_recorded_logged_and_explains_the_stall(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Finding 10: a gateway error inside tick is no longer swallowed: it is logged, recorded
    once as a job event, and named in the stall when the watchdog gives up on the op."""
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=10.0)))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.pump()  # op queued

    def gone():
        raise RuntimeError("Resolve went away")

    world.gw.get_context = gone  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="aar.node.jobs.runner"):
        for _ in range(5):
            world.runner.pump()
    assert any("broker crashed" in r.getMessage() for r in caplog.records)
    v = world.runner.get(job_id)
    errors = [e for e in v.events if e["kind"] == "broker_error"]
    assert len(errors) == 1  # bounded: the same crash is recorded once
    assert errors[0]["data"]["message"] == "Resolve went away"
    world.clock.advance(20.0)
    world.runner.pump()
    v = world.runner.get(job_id)
    assert v.state == "stalled" and v.stall["code"] == "watchdog_op_timeout"
    assert v.stall["detail"]["broker_error"]["message"] == "Resolve went away"
    assert "RuntimeError" in v.stall["detail"]["broker_error"]["traceback"]


@pytest.mark.parametrize("budget", [0, -1.0, float("inf"), float("nan")])
def test_drain_budget_must_be_positive_and_finite(tmp_path: Path, budget: float) -> None:
    world = JobWorld(tmp_path)
    with pytest.raises(ValueError):
        JobRunner(
            world.broker,
            world.store,
            world.hublink,
            node_id=world.node_id,
            branch_registry=world.branch_registry,
            feed=world.feed,
            presence_reporter=world.runtime.presence_reporter,
            drain_budget=budget,
        )


def test_closed_runner_runs_nothing_more(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.pump()  # op queued
    assert world.runner.close()
    world.broker.run_until_idle()  # whoever ticks the old broker, the op is refused
    assert world.main.markers == {}
    op_id = _op_ids(world.runner.get(job_id), "markers.add")[0]
    assert world.broker.result(op_id).error.code == "job_inactive"


def test_events_json_is_readable(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    for e in world.runner.get(job_id).events:
        json.dumps(e)


@pytest.mark.parametrize("old_version", [1, 2])
def test_old_store_is_migrated_to_v3_and_agentless_jobs_never_run(
    tmp_path: Path, old_version: int
) -> None:
    """Schema v3 adds the agent id and the consumed approvals; v1 (no restart counter) and v2
    stores are migrated in place. A migrated job has no agent id, so it fails at load with
    ``no_agent_id`` instead of running as nobody (P2)."""
    world = JobWorld(tmp_path)
    job_id = world.submit("ops.sequence", _markers(100), world.locator())
    world.runtime.close()
    world.store.close()
    conn = sqlite3.connect(world.db_path)
    for column in ("agent_id", "approvals", "approval_pending_key") + (
        ("restarts",) if old_version == 1 else ()
    ):
        conn.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
    conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(old_version),))
    conn.commit()
    conn.close()
    world.store = JobStore(world.db_path)
    runner = world.restart()
    conn = sqlite3.connect(world.db_path)
    try:
        version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    finally:
        conn.close()
    assert version == ("3",)
    assert {"restarts", "agent_id", "approvals", "approval_pending_key"} <= columns
    runner.run_until_settled()
    view = runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "no_agent_id"
    assert world.main.markers == {}
