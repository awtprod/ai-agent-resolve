"""Render jobs: atomic build, poll, Deliver safety, the hijack/cross-wire/crash fixes.

Every test runs the render ops over the broker against the FakeResolve render queue. Several
are ports of the S7 round-2 validator probes, asserting the FIXED behaviour (they failed on the
round-1 code).
"""

from __future__ import annotations

from pathlib import Path

from aar.hub import control
from aar.hub.approvals import ApprovalScope, grant
from aar.hub.events import Actor
from aar.node.jobs import ApprovalPolicy, JobState, Watchdog, WatchdogConfig

from .fake_resolve import FakeRenderJob
from .jobs_support import AGENT_ID, IDLE, JobWorld

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


# -- happy path --------------------------------------------------------------------------------


def test_render_completes_and_reports_measured_values(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 3  # more than one poll
    world.project.render_time_ms = 7200
    original = world.project.deliver.copy()

    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.SUCCEEDED.value, view and view.error
    m = view.measured
    assert m["status"] == "Complete"
    assert m["output_path"] == "D:/Out/show_v1"
    assert m["frame_count"] == 100
    assert abs(m["frame_rate"] - 29.97) < 1e-6
    assert abs(m["duration_seconds"] - 100 / 29.97) < 1e-6
    assert m["render_time_ms"] == 7200
    assert world.project.render_jobs == []  # cleanup removed the finished job
    # Deliver settings were snapshotted before the preset loaded and restored afterwards.
    assert world.project.deliver.format == original.format
    assert world.project.deliver.codec == original.codec
    assert world.project.deliver.settings == original.settings
    assert not [p for p in world.fake.presets if p.startswith("aar-job-")]  # no preset leak


def test_render_failure_restores_deliver(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_outcome = "fail"
    world.project.render_error = "frames offline: cam A"
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.FAILED.value
    assert view.error["code"] == "render_failed"
    assert view.error["trap_id"] == "render-uses-original-media"
    assert world.project.deliver.settings == original.settings
    assert not [p for p in world.fake.presets if p.startswith("aar-job-")]


def test_render_waits_for_approval_on_production(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    pid = world.pid()
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.WAITING.value
    assert view.waiting["kind"] == "needs_approval"
    assert world.project.render_jobs == []  # nothing built while waiting
    assert world.project.deliver.settings == original.settings

    token = grant(world.log, ApprovalScope.PROJECT_WRITE, pid, 3600, "render ok", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    done = world.runner.get(job_id)
    assert done is not None and done.state == JobState.SUCCEEDED.value
    assert done.measured["output_path"] == "D:/Out/show_v1"


# -- P1: a human's render job is never adopted -------------------------------------------------


def test_p1a_agent_never_adopts_a_human_render_job(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    human = _human_job(world, "Main [agent]", "show_v1")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.SUCCEEDED.value
    assert _calls(world, "AddRenderJob")  # the agent queued its OWN render
    assert ("StartRendering", [human], False) not in world.fake.calls  # never started the human's
    assert ("DeleteRenderJob", human) not in world.fake.calls  # never deleted the human's
    assert any(j.job_id == human for j in world.project.render_jobs)  # the human's job survives
    assert view.measured["output_path"] == "D:/Out/show_v1"  # the agent's own output


def test_p1b_completed_human_job_is_left_alone(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    human = _human_job(world, "Main [agent]", "show_v1", status="Complete")
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value
    assert ("DeleteRenderJob", human) not in world.fake.calls
    assert any(j.job_id == human for j in world.project.render_jobs)


# -- P2 / H1: reuse by recorded id survives a restart and the live filename extension ----------


def test_p2_resume_reuses_recorded_id_even_with_filename_extension(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_output_extension = ".mov"  # live 21.1 appends the container extension
    world.project.render_complete_after = 5  # still rendering when we restart
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(30):
        world.runner.pump()
        if world.store.get(job_id).cursor.get("render_job_id"):
            break
    recorded = world.store.get(job_id).cursor["render_job_id"]
    assert len(world.project.render_jobs) == 1

    runner = world.restart()
    for _ in range(50):
        runner.pump()
        if runner.get(job_id).state == JobState.SUCCEEDED.value:
            break
    # No second render job was added on resume: the recorded JobId was reused, not the name.
    assert _calls(world, "AddRenderJob") == []  # calls were cleared at restart
    assert runner.get(job_id).state == JobState.SUCCEEDED.value
    assert runner.get(job_id).measured["render_job_id"] == recorded


# -- P7 / P16: the Deliver settings are never clobbered between a render's ops ------------------


def test_p7_production_broker_restore_never_renders_human_settings(tmp_path: Path) -> None:
    from aar.node.broker import BrokerConfig

    world = JobWorld(tmp_path)
    world.restart(broker_config=BrokerConfig())  # production restore_delay
    original = world.project.deliver.copy()
    world.project.render_complete_after = 5
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    # Advance the clock past the restore delay partway through the render, forcing a broker
    # restore while the render is still polling.
    for i in range(40):
        world.runner.pump()
        if i == 6:
            world.clock.advance(30.0)
        if world.runner.get(job_id).state in (JobState.SUCCEEDED.value, JobState.FAILED.value):
            break
    view = world.runner.get(job_id)
    assert view.state == JobState.SUCCEEDED.value, view.error
    assert view.measured["output_path"] == "D:/Out/show_v1"  # rendered with the agent's settings
    assert world.project.deliver.settings == original.settings  # human's settings intact


def test_p16_restart_mid_render_keeps_deliver_original(tmp_path: Path) -> None:
    from aar.node.broker import BrokerConfig

    world = JobWorld(tmp_path)
    world.restart(broker_config=BrokerConfig())  # production restore_delay
    original = world.project.deliver.copy()
    world.project.render_complete_after = 6
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(40):
        world.runner.pump()
        if world.store.get(job_id).cursor.get("phase") == "poll":
            break
    # Node restarts mid-render.
    world.restart(broker_config=BrokerConfig())
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value
    # Let the broker's own restore run after the queue drains.
    for _ in range(3):
        world.clock.advance(30.0)
        world.runner.pump()
    assert world.project.deliver.settings == original.settings  # never re-clobbered


# -- P8 / P9 / P10: Deliver is restored (inside build) on every path ---------------------------


def test_p8_snapshot_preset_gone_after_build_not_mid_render(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 5
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(40):
        world.runner.pump()
        if world.store.get(job_id).cursor.get("phase") == "poll":
            break
    # By the time the render is polling, build already restored Deliver and deleted its preset.
    assert not [p for p in world.fake.presets if p.startswith("aar-job-")]
    assert world.project.deliver.settings == original.settings
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value


def test_p9_cancel_mid_render_leaves_deliver_clean(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 50
    original = world.project.deliver.copy()
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(40):
        world.runner.pump()
        if world.store.get(job_id).cursor.get("phase") == "poll":
            break
    world.runner.cancel(job_id)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.CANCELLED.value
    assert world.project.deliver.settings == original.settings
    assert not [p for p in world.fake.presets if p.startswith("aar-job-")]  # no preset leak


def test_p10_stall_mid_render_leaves_deliver_clean(tmp_path: Path) -> None:
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
    stalled = world.runner.get(job_id)
    assert stalled.state == JobState.STALLED.value
    assert stalled.stall["code"] == "watchdog_step_timeout"
    assert world.project.deliver.settings == original.settings
    # The watchdog reports and stops; the render is left running, never killed.
    assert len(world.project.render_jobs) == 1
    assert world.project.render_jobs[0].status == "Rendering"


# -- P19: configure runs on the job's own timeline, not the editor's ---------------------------


def test_p19_build_runs_on_the_jobs_timeline(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    proj = world.project
    proj.current_timeline = proj.timeline("Main [agent]")  # the editor is on another timeline
    world.presence.observe(world.gw.get_context(), world.clock())
    world.clock.advance(IDLE)
    seen: list[tuple[str, str]] = []
    orig_set, orig_add = proj.SetRenderSettings, proj.AddRenderJob

    def set_settings(settings):
        seen.append(("SetRenderSettings", proj.current_timeline.GetName()))
        return orig_set(settings)

    def add():
        seen.append(("AddRenderJob", proj.current_timeline.GetName()))
        return orig_add()

    proj.SetRenderSettings, proj.AddRenderJob = set_settings, add  # type: ignore[method-assign]
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value
    assert seen[0] == ("SetRenderSettings", "Main")
    assert seen[-1] == ("AddRenderJob", "Main")


# -- P20: two renders in one project never cross-wire ------------------------------------------


def test_p20_two_renders_same_project_do_not_cross_wire(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    a = world.runner.submit(
        "render",
        _render_params(
            world, timeline=world.ref("Main"), target_dir="D:/A", custom_name="a_v1", cleanup=False
        ),
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
    assert va.state == JobState.SUCCEEDED.value and vb.state == JobState.SUCCEEDED.value
    assert va.measured["output_path"] == "D:/A/a_v1"
    assert vb.measured["output_path"] == "D:/B/b_v1"
    by_timeline = {j.info["TimelineName"]: j.info for j in world.project.render_jobs}
    assert by_timeline["Main"]["TargetDir"] == "D:/A"
    assert by_timeline["Main [agent]"]["TargetDir"] == "D:/B"


# -- P21: a long render in one project does not stall a job in another -------------------------


def test_p21_long_render_does_not_stall_another_project(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 10_000  # effectively never finishes in the test
    world.runner.submit("render", _render_params(world), world.locator(), agent_id=AGENT_ID)
    for _ in range(30):
        world.runner.pump()
        if world.project.rendering:
            break
    assert world.project.rendering
    other = world.runner.submit(
        "ops.sequence",
        {"steps": [{"op": "timeline.list", "params": {}}]},
        world.locator("Other"),
        agent_id=AGENT_ID,
    )
    for _ in range(14):
        world.clock.advance(10.0)
        world.runner.pump()
    assert world.runner.get(other).state != JobState.STALLED.value


# -- P3b: an emergency stop stops a render before it starts ------------------------------------


def test_p3b_stop_before_build_never_starts_the_render(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.pump()  # the build op is queued, not yet run
    assert _calls(world, "AddRenderJob") == []
    control.stop(world.log, control.StopScope.all(), "STOP NOW", HUMAN)
    world.runner.pump()
    assert world.runner.get(job_id).state == JobState.WAITING.value
    assert _calls(world, "StartRendering") == [] and world.project.render_jobs == []


# -- P5: a render's step deadline survives a restart while waiting ------------------------------


def test_p5_step_deadline_survives_restart_while_waiting(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1e6)))
    world.project.render_outcome = "hang"
    job_id = world.runner.submit(
        "render", _render_params(world, timeout_seconds=30.0), world.locator(), agent_id=AGENT_ID
    )
    control.stop(world.log, control.StopScope.all(), "pause", HUMAN)
    world.runner.run_until_settled()
    runner = world.restart(watchdog=Watchdog(WatchdogConfig(op_timeout=1e6)))
    control.resume(world.log, control.StopScope.all(), "go", HUMAN)
    for _ in range(40):
        runner.pump()
    for _ in range(20):
        world.clock.advance(1000.0)  # far past the 30 s render deadline
        runner.pump()
    assert runner.get(job_id).state == JobState.STALLED.value  # the deadline was enforced


# -- P22: the event log does not grow per poll during a long render ----------------------------


def test_p22_long_render_event_log_is_bounded(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 120  # a long render, polled many times
    job_id = world.runner.submit(
        "render", _render_params(world), world.locator(), agent_id=AGENT_ID
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == JobState.SUCCEEDED.value
    # Transitions, not polls: the log is bounded by the steps, far below the poll count.
    assert len(view.events) < 20
