"""Regression ports of the S7 round-2 validator probes (non-render).

Each test asserts the FIXED behaviour; every one failed on the round-1 code. The probe id is
in the test name. The render probes (P1/P2/P5/P7/P8/P9/P10/P16/P19/P20/P21) live in
``test_jobs_render.py``.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from aar.hub import control
from aar.hub.approvals import ApprovalScope, get_approval, grant
from aar.hub.common import utcnow
from aar.hub.events import Actor
from aar.hub.registry import ProjectSource, Registry
from aar.node.jobs import (
    ApprovalPolicy,
    JobState,
    JobStore,
    JobStoreLocked,
    JobStoreVersionError,
    Watchdog,
    WatchdogConfig,
)

from .jobs_support import JobWorld

HUMAN = Actor("human", "andrew")


# -- L4: an unknown store schema version is refused --------------------------------------------


def test_l4_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    path = str(tmp_path / "jobs.db")
    JobStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(JobStoreVersionError):
        JobStore(path, create=False)


# -- L5: production policy fails closed and rejects a string sandbox list -----------------------


def test_l5_approval_policy_fails_closed_and_rejects_a_string() -> None:
    assert ApprovalPolicy(live=True).is_production(None) is True  # unknown project = production
    assert ApprovalPolicy(live=True, sandbox_exempt=frozenset({"p1"})).is_production("p1") is False
    with pytest.raises(TypeError):
        ApprovalPolicy(live=True, sandbox_exempt="p1")  # type: ignore[arg-type]


def _markers(frame: int) -> dict:
    return {"steps": [{"op": "markers.add", "params": {"markers": [{"frame": frame}]}}]}


# -- P3a: an emergency stop stops an already-queued op -----------------------------------------


def test_p3a_stop_prevents_a_queued_op_from_running(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.pump()  # the op is queued in the broker, not yet run
    assert world.main.markers == {}
    control.stop(world.log, control.StopScope.all(), "STOP NOW", HUMAN)
    world.runner.pump()
    view = world.runner.get(job_id)
    assert view.state == JobState.WAITING.value
    assert world.main.markers == {}  # the write did NOT land after the stop


# -- P4: time spent waiting does not count against the step deadline ---------------------------


def test_p4a_approval_wait_does_not_burn_the_step_deadline(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.WAITING.value
    world.clock.advance(400.0)  # a human takes ~7 min to approve; the step timeout is 300 s
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "ok", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == JobState.SUCCEEDED.value  # not stalled by the wait
    assert view.approval_consumed and 100 in world.main.markers


def test_p4b_stop_wait_does_not_burn_the_step_deadline(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    control.stop(world.log, control.StopScope.all(), "lunch", HUMAN)
    world.runner.run_until_settled()
    world.clock.advance(400.0)
    control.resume(world.log, control.StopScope.all(), "back", HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.SUCCEEDED.value  # not stalled


# -- P6: a refused token is not retried on every pump ------------------------------------------


def test_p6_refused_token_is_not_retried(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_LOAD, world.pid(), 3600, "wrong", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    for _ in range(200):
        world.runner.pump()
    refused = world.log.read(types=["approval.consume_refused"])
    assert len(refused) == 1  # the bad token was tried once, not 200 times
    assert world.runner.get(job_id).state == JobState.WAITING.value


# -- P11: two runners on one store cannot both drive the jobs ----------------------------------


def test_p11_second_runner_on_same_store_is_locked_out(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.runner.submit("ops.sequence", _markers(100), world.locator())
    with pytest.raises(JobStoreLocked):
        JobStore(world.db_path)  # the first store still holds the single-writer lock


# -- P12: a corrupt row is quarantined, not fatal ----------------------------------------------


def test_p12_corrupt_row_is_quarantined_not_fatal(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    good = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.store.close()
    conn = sqlite3.connect(world.db_path)
    row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (good,)).fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM jobs").description]
    forged = dict(zip(cols, row, strict=True))
    forged.update(job_id="job_corrupt", kind="render", params='{"nope": 1}')
    conn.execute(
        f"INSERT INTO jobs ({','.join(forged)}) VALUES ({','.join('?' * len(forged))})",
        list(forged.values()),
    )
    conn.commit()
    conn.close()
    runner = world.restart()  # must NOT raise
    bad = runner.get("job_corrupt")
    assert bad is not None and bad.state == JobState.FAILED.value
    assert bad.error["code"] == "quarantined_bad_record"
    runner.run_until_settled()  # the good job still runs
    assert runner.get(good).state == JobState.SUCCEEDED.value


# -- P13: a forged approval_consumed does not bypass F15 ---------------------------------------


def test_p13_forged_consumed_flag_is_not_trusted(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == JobState.WAITING.value
    world.store.close()
    conn = sqlite3.connect(world.db_path)
    conn.execute(
        "UPDATE jobs SET approval_consumed=1, state='queued', waiting=NULL WHERE job_id=?",
        (job_id,),
    )
    conn.commit()
    conn.close()
    runner = world.restart(approval=ApprovalPolicy(live=True))
    runner.run_until_settled()
    assert world.main.markers == {}  # no write without a verified token
    assert world.log.read(types=["approval.consumed"]) == []
    assert runner.get(job_id).state == JobState.WAITING.value


# -- P14: a non-finite timeout is rejected at submit, not a ghost job ---------------------------


def test_p14_non_finite_timeout_is_rejected_at_submit(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    other = world.runner.submit("ops.sequence", _markers(200), world.locator())
    with pytest.raises(ValueError):
        world.runner.submit(
            "ops.sequence",
            {
                "steps": [
                    {
                        "op": "markers.add",
                        "params": {"markers": [{"frame": 100}]},
                        "timeout": float("inf"),
                    }
                ]
            },
            world.locator(),
        )
    assert len(world.runner.list()) == 1  # no ghost job was created
    world.runner.run_until_settled()  # the unrelated job runs, no wedge
    assert world.runner.get(other).state == JobState.SUCCEEDED.value


# -- P15: the watchdog can see a hung gateway call; the pump is not blocked ---------------------


def test_p15_hung_gateway_call_does_not_block_the_pump(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, watchdog=Watchdog(WatchdogConfig(op_timeout=1.0)))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.pump()  # the op is queued, not yet run
    release = threading.Event()
    orig = world.main.AddMarker

    def hung(*args, **kwargs):
        release.wait(5.0)
        return orig(*args, **kwargs)

    world.main.AddMarker = hung  # type: ignore[method-assign]
    world.clock.advance(5.0)  # far past op_timeout, while the call will be hung
    try:
        pump_thread = threading.Thread(target=world.runner.pump)
        pump_thread.start()
        pump_thread.join(2.0)
        assert not pump_thread.is_alive()  # the pump returned despite the hung op
        # A read from another thread is not blocked while the op hangs on the worker.
        got: list = []
        reader = threading.Thread(target=lambda: got.append(world.runner.get(job_id)))
        reader.start()
        reader.join(1.0)
        assert got and got[0] is not None
    finally:
        release.set()
        pump_thread.join()
    assert world.runner.get(job_id).state == JobState.STALLED.value
    assert world.runner.get(job_id).stall["code"] == "watchdog_op_timeout"


# -- P18: a project-scope stop reaches the job via the registry id -----------------------------


def test_p18_project_scope_stop_reaches_the_job(tmp_path: Path) -> None:
    from aar.hub.common import InvalidInput

    world = JobWorld(tmp_path)
    reg = Registry(world.log)
    show = reg.create_project(
        "Show", HUMAN, sources=[ProjectSource("cloud_library", {"library": "CCN", "name": "Show"})]
    )
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    # The console's "Stop Show" button stops project:<registry id>.
    control.stop(world.log, control.StopScope.project(show.id), "stop this show", HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == JobState.WAITING.value  # the registry-id stop reached the job
    assert world.main.markers == {}
    assert view.waiting["detail"]["scope"] == f"project:{show.id}"
    # A stop by the raw Resolve uid is refused by the hub (it is not a registry id).
    with pytest.raises(InvalidInput):
        control.stop(world.log, control.StopScope.project(world.pid()), "by uid", HUMAN)


# -- P23: a crash between the hub consume and the local checkpoint strands safely ---------------


def test_p23_crash_after_consume_leaves_job_waiting_not_written(tmp_path: Path) -> None:
    world = JobWorld(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.runner.submit("ops.sequence", _markers(100), world.locator())
    world.runner.run_until_settled()
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "ok", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    real_save = world.store.save

    class Crash(Exception):
        pass

    def save(record, events=()):
        events = list(events)
        if any(kind == "approval_consumed" for kind, _, _ in events):
            raise Crash("power cut after the hub consume committed")
        return real_save(record, events)

    world.store.save = save  # type: ignore[method-assign]
    with pytest.raises(Crash):
        world.runner.pump()
    runner = world.restart(approval=ApprovalPolicy(live=True))
    runner.run_until_settled()
    view = runner.get(job_id)
    assert view.state == JobState.WAITING.value
    assert view.waiting["detail"]["refused_code"] == "already_consumed"
    assert world.main.markers == {}  # the write never happened; the token is burned
    assert get_approval(world.log, token.token_id).state(utcnow()) == "consumed"
