"""The node job runner: checkpointed resume, the watchdog, emergency stop and the F15 gate.

No test calls Resolve; every job runs over the broker against the in-memory FakeResolve, with a
real SQLite job store and a real hub event log. ``restart`` reopens the store in a fresh runner
and broker to model a node restart.
"""

from __future__ import annotations

from pathlib import Path

from aar.hub import control
from aar.hub.approvals import ApprovalScope, get_approval, grant
from aar.hub.common import utcnow
from aar.hub.events import Actor
from aar.node.jobs import ApprovalPolicy, JobState, Watchdog, WatchdogConfig

from .jobs_support import NODE_ID, JobWorld

HUMAN = Actor("human", "andrew")


def _markers_job(world: JobWorld, *frames: int) -> list[dict]:
    return [{"op": "markers.add", "params": {"markers": [{"frame": f}]}} for f in frames]


def _add_marker_frames(world: JobWorld) -> list[int]:
    return [call[1] for call in world.fake.calls if call[0] == "AddMarker"]


# -- happy path and events ---------------------------------------------------------------------


def test_ops_sequence_succeeds_with_measured_values(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence",
        {
            "steps": [
                {"op": "markers.add", "params": {"markers": [{"frame": 100, "name": "cut"}]}},
                {"op": "timeline.list", "params": {}},
            ]
        },
        world.locator(),
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None
    assert view.state == JobState.SUCCEEDED.value
    assert view.steps_total == 2
    assert view.current_step == 2
    # Each step keeps its measured value; the final measured view is the last step's result.
    assert view.results[0]["markers"][0]["frame"] == 100
    assert view.results[0]["markers"][0]["status"] == "added"
    assert "timelines" in view.measured and view.measured["project"] == "Show"
    # Every transition is an event.
    kinds = [event["kind"] for event in view.events]
    for expected in ("submitted", "running", "op_submitted", "step_completed", "succeeded"):
        assert expected in kinds, kinds


def test_op_failure_becomes_a_structured_job_error(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    # markers.add with two markers on the same frame is refused by the op's validator.
    job_id = world.runner.submit(
        "ops.sequence",
        {"steps": [{"op": "markers.add", "params": {"markers": [{"frame": 5}, {"frame": 5}]}}]},
        world.locator(),
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None
    assert view.state == JobState.FAILED.value
    assert view.error is not None
    assert view.error["code"] == "op_submit_error"
    assert view.error["fix_hint"]


# -- checkpointed resume -----------------------------------------------------------------------


def test_completed_step_is_not_rerun_after_restart(tmp_path: Path) -> None:
    """Step 0 duplicates a timeline with if_exists='error'; if resume re-ran it the duplicate
    would fail 'timeline_exists'. The job succeeding proves the completed step is skipped."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence",
        {
            "steps": [
                {
                    "op": "timeline.duplicate",
                    "params": {
                        "source": world.ref("Main"),
                        "name": "Render Copy",
                        "if_exists": "error",
                    },
                },
                {"op": "markers.add", "params": {"markers": [{"frame": 50}]}},
            ]
        },
        world.locator(),
    )
    # Pump until step 0 is checkpointed (the duplicate exists), but step 1 is not done yet.
    for _ in range(6):
        world.runner.pump()
        if (view := world.runner.get(job_id)) and view.current_step >= 1:
            break
    view = world.runner.get(job_id)
    assert view is not None and view.current_step == 1
    assert sum(t.GetName() == "Render Copy" for t in world.project.timelines) == 1

    runner = world.restart()
    reloaded = runner.get(job_id)
    assert reloaded is not None and reloaded.current_step == 1  # resumed from the store
    runner.run_until_settled()
    done = runner.get(job_id)
    assert done is not None and done.state == JobState.SUCCEEDED.value  # step 0 was NOT re-run
    assert sum(t.GetName() == "Render Copy" for t in world.project.timelines) == 1


def test_inflight_step_reruns_idempotently_after_restart(tmp_path: Path) -> None:
    """A step whose op was in flight at the kill re-runs on resume; idempotency finds its
    earlier work (the marker already present) instead of adding it twice."""
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100, 200)}, world.locator()
    )
    world.runner.pump()  # step 0's op is submitted to the broker but not yet run
    view = world.runner.get(job_id)
    assert view is not None and view.current_step == 0
    assert _add_marker_frames(world) == []  # nothing applied yet
    # Simulate the op having applied its write in the process that died before checkpointing.
    world.main.markers[100] = world.marker_fields()

    runner = world.restart()
    runner.run_until_settled()
    done = runner.get(job_id)
    assert done is not None and done.state == JobState.SUCCEEDED.value
    # Step 0 re-ran but found frame 100 already present; only frame 200 was newly added.
    assert done.results[0]["markers"][0]["status"] == "present"
    assert done.results[1]["markers"][0]["status"] == "added"
    assert _add_marker_frames(world) == [200]
    assert set(world.main.markers) == {100, 200}


# -- watchdog ----------------------------------------------------------------------------------


def test_watchdog_stalls_a_blocked_op_and_stops_the_project(tmp_path: Path) -> None:
    from aar.node.leases import Holder, Scope, project_key

    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=10.0)))
    # A human holds the project, so the runner's op can never run.
    human_lease = world.leases.acquire(
        Holder.human("andrew"), Scope.PROJECT, project_key(world.pid()), None
    )
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100)}, world.locator()
    )
    world.runner.pump()  # op submitted, but lease-blocked in the broker
    assert world.runner.get(job_id).state == JobState.RUNNING.value

    world.clock.advance(15.0)
    world.runner.pump()
    stalled = world.runner.get(job_id)
    assert stalled is not None and stalled.state == JobState.STALLED.value
    assert stalled.stall["code"] == "watchdog_op_timeout"
    assert "stalled" in [e["kind"] for e in stalled.events]

    # A stalled project stops scheduling: a new job for it stays queued.
    other_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 300)}, world.locator()
    )
    world.runner.pump()
    assert world.runner.get(other_id).state == JobState.QUEUED.value

    # Recovery is an explicit action: release the human lease and resume.
    world.leases.release(human_lease.id, Holder.human("andrew"))
    world.runner.resume(job_id)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value
    assert world.runner.get(other_id).state == JobState.SUCCEEDED.value


# -- emergency stop ----------------------------------------------------------------------------


def test_emergency_stop_all_pauses_then_resumes(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100, 200)}, world.locator()
    )
    control.stop(world.log, control.StopScope.all(), "safety check", HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.WAITING.value
    assert view.waiting["kind"] == "stop" and view.waiting["detail"]["scope"] == "all"
    assert world.main.markers == {}  # paused before any step ran

    control.resume(world.log, control.StopScope.all(), "cleared", HUMAN)
    world.runner.run_until_settled()
    done = world.runner.get(job_id)
    assert done is not None and done.state == JobState.SUCCEEDED.value
    assert set(world.main.markers) == {100, 200}


def test_emergency_stop_node_scope(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100)}, world.locator()
    )
    control.stop(world.log, control.StopScope.node(NODE_ID), "node paused", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.WAITING.value
    control.resume(world.log, control.StopScope.node(NODE_ID), "node ok", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value


def test_emergency_stop_midjob_pauses_at_checkpoint(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100, 200)}, world.locator()
    )
    for _ in range(4):
        world.runner.pump()
        if (view := world.runner.get(job_id)) and view.current_step >= 1:
            break
    assert world.runner.get(job_id).current_step == 1
    assert set(world.main.markers) == {100}  # step 0 done, step 1 not

    control.stop(world.log, control.StopScope.all(), "mid-job", HUMAN)
    world.runner.run_until_settled()
    paused = world.runner.get(job_id)
    assert paused.state == JobState.WAITING.value and paused.current_step == 1
    assert set(world.main.markers) == {100}  # step 1 did not run while stopped

    control.resume(world.log, control.StopScope.all(), "go", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value
    assert set(world.main.markers) == {100, 200}


# -- F15 approval gate -------------------------------------------------------------------------


def test_production_write_waits_for_approval_then_consumes_once(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    pid = world.pid()
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100)}, world.locator()
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.WAITING.value
    assert view.waiting["kind"] == "needs_approval"
    assert view.waiting["detail"] == {"scope": "project.write", "target": pid}
    assert view.approval_consumed is False
    assert world.main.markers == {}  # nothing written while waiting
    assert _add_marker_frames(world) == []

    token = grant(world.log, ApprovalScope.PROJECT_WRITE, pid, 3600, "approved", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    done = world.runner.get(job_id)
    assert done is not None and done.state == JobState.SUCCEEDED.value
    assert done.approval_consumed is True
    assert set(world.main.markers) == {100}
    # Consumed exactly once.
    consumed = world.log.read(types=["approval.consumed"])
    assert len(consumed) == 1
    assert get_approval(world.log, token.token_id).state(utcnow()) == "consumed"


def test_approval_wrong_scope_is_refused_and_job_keeps_waiting(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    pid = world.pid()
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100)}, world.locator()
    )
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_LOAD, pid, 3600, "wrong scope", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.WAITING.value
    assert view.waiting["detail"]["refused_code"] == "wrong_scope"
    assert view.approval_consumed is False
    assert world.main.markers == {}
    refused = world.log.read(types=["approval.consume_refused"])
    assert any(e.payload["code"] == "wrong_scope" for e in refused)
    assert get_approval(world.log, token.token_id).state(utcnow()) == "active"  # not consumed


def test_approval_wrong_target_is_refused(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100)}, world.locator()
    )
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, "proj-not-this-one", 3600, "wrong", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.WAITING.value
    assert view.waiting["detail"]["refused_code"] == "wrong_target"
    assert view.approval_consumed is False


def test_sandbox_exempt_project_skips_approval(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    pid = world.pid()
    world.runner.policy = ApprovalPolicy(live=True, sandbox_exempt=frozenset({pid}))
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100)}, world.locator()
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view is not None and view.state == JobState.SUCCEEDED.value
    assert set(world.main.markers) == {100}


# -- cancel ------------------------------------------------------------------------------------


def test_cancel_stops_a_running_job(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit(
        "ops.sequence", {"steps": _markers_job(world, 100, 200)}, world.locator()
    )
    world.runner.pump()
    assert world.runner.cancel(job_id, "no longer needed") is True
    assert world.runner.get(job_id).state == JobState.CANCELLED.value
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.CANCELLED.value
    assert world.runner.cancel(job_id) is False  # already terminal


def test_list_filters_by_state_and_project(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    a = world.runner.submit("ops.sequence", {"steps": _markers_job(world, 100)}, world.locator())
    world.runner.run_until_settled()
    assert [v.job_id for v in world.runner.list(state=JobState.SUCCEEDED)] == [a]
    assert world.runner.list(project_id="nope") == []
