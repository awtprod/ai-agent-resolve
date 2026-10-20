from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from aar.hub import jobs, nodes
from aar.hub.events import EventLog
from aar.node.jobs import JobStore

from .fake_resolve import FakeResolve
from .jobs_support import LAYOUT
from .remote_support import write_token
from .test_remote_worker import AGENT, HUMAN, NODE, PARAMS, _registered, _submit, _subprocess_hub


def _wait(
    predicate: Callable[[], bool], *, description: str, node: subprocess.Popen | None = None
) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return
        if node is not None and node.poll() is not None:
            raise AssertionError(
                f"node {node.pid} exited {node.returncode} while waiting for {description}"
            )
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {description}")


@contextlib.contextmanager
def _node(
    tmp_path: Path, url: str, token: Path, *, block: bool = False
) -> Iterator[subprocess.Popen]:
    argv = [
        sys.executable,
        "-m",
        "tests.node.remote_worker_process_helper",
        "--url",
        url,
        "--token",
        str(token),
        "--store",
        str(tmp_path / "jobs.db"),
        "--observation",
        str(tmp_path / "observations.jsonl"),
    ]
    if block:
        argv.append("--block-third")
    with (tmp_path / "node.log").open("ab") as output:
        process = subprocess.Popen(argv, stdout=output, stderr=output)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    returncode = process.wait(timeout=10)
                    if sys.platform != "win32":  # Windows terminate() forcibly ends the process.
                        assert returncode == 0
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    raise AssertionError(f"node {process.pid} failed to stop") from None


def _observations(tmp_path: Path) -> list[dict]:
    path = tmp_path / "observations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_subprocess_cli_crash_restart_keeps_claim_and_completed_step(tmp_path: Path) -> None:
    with contextlib.closing(EventLog(tmp_path / "hub.db")) as log:
        locator = _registered(log, FakeResolve(LAYOUT, current="Show", folder=("Shows",)))
        _, token = nodes.enroll(log, NODE, HUMAN)
        token_file = write_token(tmp_path, token)
        first = _submit(log, locator)
        assert first.state(jobs.utcnow()) == "waiting_for_node"
        with _subprocess_hub(tmp_path / "hub.db", token_file):
            assert jobs.get_job(log, first.job_id).claimed_node is None
        # Hub process restart with node absent preserves the pending submission.
        with _subprocess_hub(tmp_path / "hub.db", token_file) as url:
            with _node(tmp_path, url, token_file, block=True) as process:
                _wait(
                    lambda: jobs.get_job(log, first.job_id).terminal_state == "succeeded",
                    description="first node-off submission to finish",
                    node=process,
                )
                second, _ = jobs.submit(
                    log,
                    AGENT,
                    kind="ops.sequence",
                    params={"steps": PARAMS["steps"] * 2},
                    project=locator,
                    agent_id="editor",
                    assigned_node=NODE,
                    idempotency_key="two-step",
                )
                _wait(
                    lambda: any(e["event"] == "op_blocked" for e in _observations(tmp_path)),
                    description="second job's second op to block",
                    node=process,
                )
                # Kill only the exact child we spawned, without signal handlers or store close.
                process.kill()
                process.wait(timeout=5)
                assert process.returncode != 0
            with contextlib.closing(JobStore(tmp_path / "jobs.db")) as store:
                local = store.get(second.job_id)
                assert local.current_step == 1 and len(local.results) == 1
                assert sum(e.kind == "step_completed" for e in store.events(second.job_id)) == 1
            _wait(
                lambda: jobs.get_job(log, second.job_id).state(jobs.utcnow()) == "node_offline",
                description="abruptly killed node's lease to expire",
            )
            assert (
                jobs.claim(log, "another-node", [locator.project_id], request_id="no-steal")[0]
                is None
            )
            assert len([e for e in _observations(tmp_path) if e["event"] == "op_completed"]) == 2
            with _node(tmp_path, url, token_file) as restarted:
                _wait(
                    lambda: jobs.get_job(log, second.job_id).terminal_state == "succeeded",
                    description="same local journal to resume",
                    node=restarted,
                )
            with contextlib.closing(JobStore(tmp_path / "jobs.db")) as store:
                local = store.get(second.job_id)
                assert local.state.value == "succeeded" and local.current_step == 2
                assert sum(e.kind == "step_completed" for e in store.events(second.job_id)) == 2
                assert sum(e.kind == "restarted" for e in store.events(second.job_id)) >= 1
            assert len([e for e in _observations(tmp_path) if e["event"] == "op_completed"]) == 3
            assert len(log.read(types=[jobs.EVENT_CLAIMED])) == 2
            assert len(log.read(types=[jobs.EVENT_FINISHED])) == 2
            assert not jobs.get_job(log, second.job_id).lost_on_node
