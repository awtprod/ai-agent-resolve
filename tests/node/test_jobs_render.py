"""Render jobs: submit, poll to completion and to failure, Deliver restore, watchdog.

The render job runs the render ops over the broker against the FakeResolve render queue. The
fake's ``render_outcome`` drives whether the render completes, fails or hangs.
"""

from __future__ import annotations

from pathlib import Path

from aar.hub.approvals import ApprovalScope, grant
from aar.hub.events import Actor
from aar.node.jobs import ApprovalPolicy, JobState, Watchdog, WatchdogConfig

from .jobs_support import JobWorld

HUMAN = Actor("human", "andrew")


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


def test_render_polls_to_completion_and_reports_measured_values(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_outcome = "complete"
    world.project.render_complete_after = 3  # needs more than one poll
    world.project.render_time_ms = 7200
    original = world.project.deliver.copy()

    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None, "job vanished"
    assert view.state == JobState.SUCCEEDED.value, view.error
    measured = view.measured
    assert measured["status"] == "Complete"
    assert measured["output_path"] == "D:/Out/show_v1"
    assert measured["frame_count"] == 100  # mark 0..99 inclusive
    assert abs(measured["frame_rate"] - 29.97) < 1e-6
    assert abs(measured["duration_seconds"] - 100 / 29.97) < 1e-6
    assert measured["render_time_ms"] == 7200

    # The runner really polled more than once (across pumps) before the render completed.
    poll_submits = [
        e for e in view.events if e["kind"] == "op_submitted" and e["data"]["op"] == "render.poll"
    ]
    assert len(poll_submits) >= 2
    # cleanup=True removed the finished job from the queue.
    assert world.project.render_jobs == []
    # Deliver settings were snapshotted before the preset loaded and restored afterwards.
    assert world.project.deliver.format == original.format
    assert world.project.deliver.codec == original.codec
    assert world.project.deliver.mode == original.mode
    assert world.project.deliver.settings == original.settings


def test_render_poll_to_failure_restores_deliver_settings(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_outcome = "fail"
    world.project.render_complete_after = 2
    world.project.render_error = "frames offline: cam A"
    original = world.project.deliver.copy()

    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.FAILED.value
    assert view.error["code"] == "render_failed"
    assert "frames offline" in view.error["message"]
    assert view.error["trap_id"] == "render-uses-original-media"

    # Deliver settings are restored even though the render failed.
    assert world.project.deliver.format == original.format
    assert world.project.deliver.codec == original.codec
    assert world.project.deliver.mode == original.mode
    assert world.project.deliver.settings == original.settings
    # The failed job is left in the queue for inspection (cleanup only on success).
    assert len(world.project.render_jobs) == 1
    assert world.project.render_jobs[0].status == "Failed"


def test_render_restores_deliver_settings_on_success(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    original = world.project.deliver.copy()
    job_id = world.runner.submit("render", _render_params(world, cleanup=False), world.locator())
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value
    # The delivery preset changed format/codec mid-render; the snapshot was restored.
    assert world.project.deliver.format == original.format
    assert world.project.deliver.codec == original.codec
    assert world.project.deliver.settings == original.settings
    # cleanup=False leaves the completed job in the queue.
    assert len(world.project.render_jobs) == 1


def test_watchdog_stalls_a_hung_render_without_touching_resolve(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1e6)))
    world.project.render_outcome = "hang"
    job_id = world.runner.submit(
        "render", _render_params(world, timeout_seconds=30.0), world.locator()
    )
    for _ in range(15):  # reach the poll phase; the render never finishes
        world.runner.pump()
    assert world.runner.get(job_id).state == JobState.RUNNING.value

    world.clock.advance(35.0)  # past the render step deadline
    world.runner.pump()
    stalled = world.runner.get(job_id)
    assert stalled is not None and stalled.state == JobState.STALLED.value
    assert stalled.stall["code"] == "watchdog_step_timeout"
    # The watchdog reports and stops; it never deletes or restarts the render.
    assert len(world.project.render_jobs) == 1
    assert world.project.render_jobs[0].status == "Rendering"


def test_production_render_waits_for_approval(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    pid = world.pid()
    job_id = world.runner.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.WAITING.value
    assert view.waiting["kind"] == "needs_approval"
    # Nothing was changed while waiting: no render job, no preset snapshot side effects visible.
    assert world.project.render_jobs == []

    token = grant(world.log, ApprovalScope.PROJECT_WRITE, pid, 3600, "render approved", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    done = world.runner.get(job_id)
    assert done is not None and done.state == JobState.SUCCEEDED.value
    assert done.measured["output_path"] == "D:/Out/show_v1"
