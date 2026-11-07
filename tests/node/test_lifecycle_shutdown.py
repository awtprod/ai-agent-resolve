"""Shutdown and crash recovery through the production runner and real fake side effects."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from aar.hub import jobs
from aar.hub.events import Actor
from aar.node.jobs import JobStore
from aar.node.jobs.store import JobStoreLocked
from aar.node.runtime import NodeIdInUse, NodeRuntime

from .jobs_support import JobWorld
from .test_remote_worker import AGENT, NODE, _direct_world, _DirectJobs


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_close_checkpoints_mutation_without_dispatching_next_step(
    tmp_path: Path,
    timeout: bool,
    cancel: bool,
):
    with _direct_world(tmp_path) as (log, fake, locator, runtime, store, worker, clock):
        entered, release = threading.Event(), threading.Event()
        timeline = fake.project("Show").timeline("Main")
        original = timeline.DuplicateTimeline
        calls = []

        def duplicate(name):
            value = original(name)
            calls.append(name)
            entered.set()
            assert release.wait(5)
            return value

        timeline.DuplicateTimeline = duplicate
        runtime.runner._drain_budget = 0.01
        job, _ = jobs.submit(
            log,
            AGENT,
            kind="ops.sequence",
            params={
                "steps": [
                    {
                        "op": "timeline.branch",
                        "params": {"source": {"name": "Main", "unique_id": timeline.GetUniqueId()}},
                    },
                    {"op": "timeline.list", "params": {}},
                ]
            },
            project=locator,
            agent_id="editor",
            assigned_node=NODE,
            idempotency_key="branch",
        )
        for _ in range(4):
            worker.tick()
            if entered.is_set():
                break
        assert entered.wait(2)
        if cancel:
            assert runtime.runner.cancel(job.job_id, "cancel during mutation")
        results = []
        closer = threading.Thread(
            target=lambda: results.append(worker.close(0.01 if timeout else 2))
        )
        closer.start()
        try:
            if timeout:
                closer.join(1)
                assert results == [False]
                assert worker.close(0.01) is False
                assert runtime.close(0.01) is False
                with pytest.raises(JobStoreLocked):
                    JobStore(tmp_path / "jobs.db")
                with pytest.raises(NodeIdInUse):
                    NodeRuntime.local(runtime.broker._gw, store, log, node_id=NODE)
            else:
                # The drain lock establishes that close has disabled dispatch.
                for _ in range(1000):
                    if runtime.runner._closed:
                        break
                    threading.Event().wait(0.001)
                assert runtime.runner._closed
        finally:
            release.set()
            closer.join(3)
            assert runtime.runner.wait_drain(2)
        assert worker.close(2)
        assert results == [not timeout]
        store2 = JobStore(tmp_path / "jobs.db")
        try:
            record = store2.get(job.job_id)
            assert record.current_step == (0 if cancel else 1)
            assert len(record.results) == (0 if cancel else 1)
            event = "late_result" if cancel else "step_completed"
            assert sum(e.kind == event for e in store2.events(job.job_id)) == 1
            assert "_uncertain_op" not in record.cursor
            assert sum(e.kind == "branch_recorded" for e in store2.events(job.job_id)) == 1
            assert sum(e.kind == "op_submitted" for e in store2.events(job.job_id)) == 1
            assert len(calls) == 1
            replacement = NodeRuntime.local(
                runtime.broker._gw,
                store2,
                log,
                node_id=NODE,
                clock=clock,
                input_idle=None,
                approval=runtime.runner.policy,
                drain_budget=None,
            )
            from aar.node.worker import RemoteWorker

            replacement.presence.observe(replacement.broker._gw.get_context(), clock())
            clock.advance(60)
            worker2 = RemoteWorker(
                replacement,
                store2,
                _DirectJobs(log, locator.project_id),
                (locator,),
                clock=clock,
            )
            try:
                for _ in range(8):
                    worker2.tick()
                assert replacement.runner.get(job.job_id).state == (
                    "cancelled" if cancel else "succeeded"
                )
                assert len(calls) == 1
            finally:
                worker2.close()
        finally:
            store2.close()


def test_abrupt_exit_in_mutation_requires_human_recovery(tmp_path: Path):
    script = """
import os, sys
from pathlib import Path
from tests.node.jobs_support import JobWorld
world = JobWorld(Path(sys.argv[1]), own_timelines=False)
original = world.main.DuplicateTimeline
def crash(name):
    original(name)
    os._exit(23)
world.main.DuplicateTimeline = crash
world.submit("ops.sequence", {"steps": [{"op": "timeline.branch", "params": {
    "source": world.ref("Main")}}]}, world.locator())
world.runner.pump()
world.runner.pump()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "AAR_LIVE": "0"},
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23, result.stderr
    world = JobWorld(tmp_path, own_timelines=False)
    view = world.runner.list()[0]
    assert view.state == "stalled"
    assert view.stall["code"] == "op_uncertain"
    world.runner.pump()
    assert not any(name == "DuplicateTimeline" for name, *_ in world.fake.calls)
    with pytest.raises(ValueError, match="human"):
        world.runner.resume(view.job_id)
    with pytest.raises(ValueError, match="human"):
        world.runner.resume(view.job_id, Actor("agent", "editor"))
    world.runner.resume(view.job_id, Actor("human", "andrew"))
    world.runner.run_until_settled()
    assert world.runner.get(view.job_id).state == "succeeded"


def test_shutdown_recovery_quarantines_bad_cursor(tmp_path: Path):
    world = JobWorld(tmp_path)
    job_id = world.submit(
        "ops.sequence",
        {"steps": [{"op": "timeline.list"}]},
        world.locator(),
    )
    record = world.store.get(job_id)
    record.cursor = []  # type: ignore[assignment]  # malformed persisted boundary data
    world.store.save(record)
    world.restart()
    view = world.runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "quarantined_bad_record"
    world.runner.pump()
    assert world.store.get(job_id).cursor == []  # original evidence is never rewritten


def test_cli_reports_incomplete_shutdown_and_retains_exclusion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    from aar.node import cli, remote
    from aar.node import jobs as node_jobs
    from aar.node.worker import RemoteWorker

    with _direct_world(tmp_path) as (log, fake, locator, runtime, store, worker, _):
        entered, release = threading.Event(), threading.Event()
        timeline = fake.project("Show").timeline("Main")
        original = timeline.DuplicateTimeline

        def duplicate(name):
            value = original(name)
            entered.set()
            assert release.wait(5)
            return value

        timeline.DuplicateTimeline = duplicate
        runtime.runner._drain_budget = 0.01
        job, _ = jobs.submit(
            log,
            AGENT,
            kind="ops.sequence",
            params={
                "steps": [
                    {
                        "op": "timeline.branch",
                        "params": {"source": {"name": "Main", "unique_id": timeline.GetUniqueId()}},
                    },
                ]
            },
            project=locator,
            agent_id="editor",
            assigned_node=NODE,
            idempotency_key="branch",
        )

        def run(stop):
            for _ in range(4):
                worker.tick()
                if entered.is_set():
                    break
            assert entered.wait(2)
            return 0

        original_close = worker.close
        monkeypatch.setattr(worker, "run", run)
        monkeypatch.setattr(worker, "close", lambda: original_close(0.01))
        monkeypatch.setattr(NodeRuntime, "remote", lambda *args, **kw: runtime)
        monkeypatch.setattr(node_jobs, "JobStore", lambda *args: store)
        monkeypatch.setattr(remote, "RemoteJobClient", lambda *args, **kw: worker.jobs)
        monkeypatch.setattr(remote, "_NodeClient", lambda *args: None)
        monkeypatch.setattr("aar.node.worker.RemoteWorker", lambda *args, **kw: worker)
        args = argparse.Namespace(
            project=[json.dumps(locator.to_dict())],
            node_id=NODE,
            hub_url="http://127.0.0.1:1",
            token_file=str(tmp_path / "unused-token"),
            store=str(tmp_path / "jobs.db"),
            connect_timeout=1,
            read_timeout=1,
            response_cap=1024,
            lease_seconds=60,
            heartbeat_seconds=15,
            tick_seconds=0.25,
        )
        try:
            assert cli.run_node(args, gateway_factory=lambda: runtime.broker._gw) == 2
            diagnostic = capsys.readouterr().err
            assert "shutdown incomplete" in diagnostic and "uncertain" in diagnostic
            assert store.get(job.job_id).cursor["_uncertain_op"]["op"] == "timeline.branch"
            with pytest.raises(JobStoreLocked):
                JobStore(tmp_path / "jobs.db")
        finally:
            release.set()
            assert original_close(2)
        assert isinstance(worker, RemoteWorker)


def test_uncertainty_checkpoint_failure_prevents_the_mutation(tmp_path: Path, monkeypatch):
    world = JobWorld(tmp_path, own_timelines=False)
    job_id = world.submit(
        "ops.sequence",
        {"steps": [{"op": "timeline.branch", "params": {"source": world.ref("Main")}}]},
        world.locator(),
    )
    world.runner.pump()
    original_save = world.store.save

    def failing_save(record, events=()):
        if "_uncertain_op" in record.cursor:
            raise OSError("uncertainty checkpoint unavailable")
        return original_save(record, events)

    monkeypatch.setattr(world.store, "save", failing_save)
    world.runner.pump()
    assert world.runner.get(job_id).state == "failed"
    assert not any(call[0] == "DuplicateTimeline" for call in world.fake.calls)


def test_abrupt_exit_in_read_only_op_can_retry(tmp_path: Path):
    script = """
import os, sys
from pathlib import Path
from tests.node.jobs_support import JobWorld
world = JobWorld(Path(sys.argv[1]))
def crash():
    os._exit(23)
world.main.GetEndFrame = crash
world.submit("ops.sequence", {"steps": [{"op": "timeline.list"}]}, world.locator())
world.runner.pump()
world.runner.pump()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "AAR_LIVE": "0"},
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23, result.stderr
    world = JobWorld(tmp_path)
    job = world.runner.list()[0]
    assert job.state == "running"
    world.runner.run_until_settled()
    assert world.runner.get(job.job_id).state == "succeeded"


def test_close_during_render_build_restores_settings_and_checkpoints_poll(tmp_path: Path):
    world = JobWorld(tmp_path, drain_budget=0.01)
    entered, release = threading.Event(), threading.Event()
    original_add = world.project.AddRenderJob
    original_deliver = world.project.deliver.copy()

    def add():
        value = original_add()
        entered.set()
        assert release.wait(5)
        return value

    world.project.AddRenderJob = add
    job_id = world.submit(
        "render",
        {
            "timeline": world.ref("Main"),
            "target_dir": "X:/Exports",
            "custom_name": "close-render",
            "render_format": "mov",
            "render_codec": "ProRes",
        },
        world.locator(),
    )
    world.runner.pump()
    world.runner.pump()
    assert entered.wait(2)
    results = []
    closer = threading.Thread(target=lambda: results.append(world.runtime.close(2)))
    closer.start()
    try:
        for _ in range(1000):
            if world.runner._closed:
                break
            threading.Event().wait(0.001)
        assert world.runner._closed
    finally:
        release.set()
        closer.join(3)
    assert results == [True]
    record = world.store.get(job_id)
    assert record.cursor["phase"] == "poll"
    assert record.cursor["render_job_id"] == world.project.render_jobs[0].job_id
    assert "_uncertain_op" not in record.cursor
    assert sum(e.kind == "op_submitted" for e in world.store.events(job_id)) == 1
    assert world.project.deliver == original_deliver
    world.restart()
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert sum(call[0] == "AddRenderJob" for call in world.fake.calls) == 0
