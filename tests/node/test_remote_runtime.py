"""NodeRuntime.remote end to end (S15a items 4 and 6): a lost ack, and a two-process hot op."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from aar.hub import nodes
from aar.hub.approvals import ApprovalScope, grant
from aar.hub.events import Actor, EventLog
from aar.node.gateway import ScriptApiGateway
from aar.node.jobs import ApprovalPolicy, JobState, JobStore
from aar.node.locator import ProjectLocator
from aar.node.remote import (
    HubUnreachable,
    RemoteConfig,
    RemoteFeedTransport,
    RemoteHubLink,
    _NodeClient,
)
from aar.node.runtime import NodeRuntime

from .fake_resolve import FakeClock, FakeResolve
from .jobs_support import (
    AGENT_ID,
    IDLE,
    JOB_CONFIG,
    LAYOUT,
    MAIN_REF,
    NODE_ID,
    OWNED,
    WALL_EPOCH,
)
from .remote_support import ServerThread, hub_app, write_token

HUMAN = Actor("human", "andrew")


def _markers_job(frame: int) -> dict[str, Any]:
    return {
        "steps": [
            {"op": "markers.add", "params": {"timeline": MAIN_REF, "markers": [{"frame": frame}]}}
        ]
    }


def _own(rt: NodeRuntime, fake: FakeResolve) -> None:
    for project, timeline in OWNED:
        proj = fake.project(project)
        uid = proj.timeline(timeline).GetUniqueId()
        rt.branch_registry.register(
            proj.GetUniqueId(), uid, source_timeline_id=uid, agent_id=AGENT_ID, version=1
        )


def _wall(clock: FakeClock):
    return lambda: WALL_EPOCH + timedelta(seconds=clock())


class _DropFirstConsumeAck:
    """Wrap a client so the first consume_approval commits on the hub but loses its reply."""

    def __init__(self, inner: _NodeClient) -> None:
        self._inner = inner
        self.dropped = False

    def call(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        if op == "consume_approval" and not self.dropped:
            self.dropped = True
            with contextlib.suppress(Exception):  # the forward should succeed (the hub commits)
                self._inner.call(op, body)
            raise HubUnreachable("dropped the consume ack after the hub committed (test)")
        return self._inner.call(op, body)


def _consumed(log: EventLog) -> int:
    return len(log.read(types=["approval.consumed"]))


def test_lost_ack_heals_and_the_op_runs_exactly_once(tmp_path: Path) -> None:
    log = EventLog(str(tmp_path / "hub.db"))
    _, token = nodes.enroll(log, NODE_ID, HUMAN)
    token_file = write_token(tmp_path, token)
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    project = fake.project("Show")
    gw = ScriptApiGateway(fake)
    clock = FakeClock()
    store = JobStore(str(tmp_path / "jobs.db"))

    with ServerThread(hub_app(log)) as server:
        config = RemoteConfig(hub_url=server.base_url, token_file=token_file)
        dropping = _DropFirstConsumeAck(_NodeClient(config))
        rt = NodeRuntime(
            gw,
            store,
            RemoteHubLink(dropping),
            RemoteFeedTransport(_NodeClient(config)),
            node_id=NODE_ID,
            clock=clock,
            wall_clock=_wall(clock),
            broker_config=JOB_CONFIG,
            idle_window=IDLE,
            input_idle=None,
            approval=ApprovalPolicy(live=True),
            drain_budget=None,
        )
        try:
            rt.presence.observe(gw.get_context(), clock())
            clock.advance(IDLE)
            _own(rt, fake)
            pid = project.GetUniqueId()
            loc = ProjectLocator(project.folder, "Show", pid)
            tok = grant(log, ApprovalScope.PROJECT_WRITE, pid, 3600, "approved", HUMAN)
            job_id = rt.runner.submit(
                "ops.sequence",
                _markers_job(100),
                loc,
                agent_id=AGENT_ID,
                approval_token=tok.token_id,
            )

            # Pump until the dropped ack parks the job: the hub committed, the node did not hear.
            for _ in range(10):
                rt.pump()
                view = rt.runner.get(job_id)
                assert view is not None
                if view.state not in (JobState.QUEUED.value, JobState.RUNNING.value):
                    break
            assert dropping.dropped
            assert view.state == JobState.WAITING.value
            assert view.waiting["kind"] == "hub_unreachable"
            assert project.timeline("Main").markers == {}  # nothing ran while waiting
            assert _consumed(log) == 1  # the hub committed the consume exactly once

            # The link is back: the runner heals from its write-ahead intent (re-reading the
            # token, not re-consuming) and the op runs.
            clock.advance(120)
            rt.runner.run_until_settled()
            done = rt.runner.get(job_id)
            assert done is not None and done.state == JobState.SUCCEEDED.value
            assert set(project.timeline("Main").markers) == {100}  # ran exactly once
            assert _consumed(log) == 1  # still exactly one approval.consumed
        finally:
            rt.close()
    store.close()


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_up(link: RemoteHubLink, proc: subprocess.Popen[bytes]) -> None:
    for _ in range(100):
        if proc.poll() is not None:  # pragma: no cover - startup failure
            raise RuntimeError(f"nodeapi subprocess exited early: {proc.returncode}")
        try:
            link.control_status()
            return
        except HubUnreachable:
            time.sleep(0.1)
    raise RuntimeError("nodeapi subprocess did not come up")  # pragma: no cover


def test_two_process_hot_op_through_a_subprocess_hub(tmp_path: Path) -> None:
    db = str(tmp_path / "hub.db")
    EventLog(db).close()  # create the hub DB, then let the subprocess be its sole writer
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    project = fake.project("Show")
    pid = project.GetUniqueId()

    setup = EventLog(db, create=False)
    _, token = nodes.enroll(setup, NODE_ID, HUMAN)
    tok = grant(setup, ApprovalScope.PROJECT_WRITE, pid, 3600, "approved", HUMAN)
    setup.close()
    token_file = write_token(tmp_path, token)

    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aar",
            "hub",
            "nodeapi",
            "--db",
            db,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        probe = RemoteHubLink(
            _NodeClient(RemoteConfig(url, token_file, connect_timeout=1, read_timeout=1))
        )
        _wait_up(probe, proc)

        gw = ScriptApiGateway(fake)
        clock = FakeClock()
        store = JobStore(str(tmp_path / "jobs.db"))
        rt = NodeRuntime.remote(
            gw,
            store,
            node_id=NODE_ID,
            hub_url=url,
            token_file=token_file,
            clock=clock,
            wall_clock=_wall(clock),
            broker_config=JOB_CONFIG,
            idle_window=IDLE,
            input_idle=None,
            approval=ApprovalPolicy(live=True),
            drain_budget=None,
        )
        try:
            rt.presence.observe(gw.get_context(), clock())
            clock.advance(IDLE)
            _own(rt, fake)
            loc = ProjectLocator(project.folder, "Show", pid)
            job_id = rt.runner.submit(
                "ops.sequence",
                _markers_job(100),
                loc,
                agent_id=AGENT_ID,
                approval_token=tok.token_id,
            )
            rt.runner.run_until_settled()
            done = rt.runner.get(job_id)
            assert done is not None and done.state == JobState.SUCCEEDED.value
            assert set(project.timeline("Main").markers) == {100}
        finally:
            rt.close()
            store.close()

        # The subprocess hub committed exactly one consume and at least one feed lease.
        read = EventLog(db, create=False, read_only=True)
        try:
            assert _consumed(read) == 1
            assert read.read(types=["lease.granted"]) or read.read(types=["lease.snapshot"])
        finally:
            read.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=10)
