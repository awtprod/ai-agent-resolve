from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from aar.hub import jobs, nodes
from aar.hub.events import Actor, EventLog
from aar.hub.registry import ProjectSource, Registry
from aar.node.gateway import ScriptApiGateway
from aar.node.hublink import HubUnreachable
from aar.node.jobs import ApprovalPolicy, JobStore
from aar.node.locator import ProjectLocator
from aar.node.remote import RemoteConfig, RemoteHubLink, RemoteJobClient, _NodeClient
from aar.node.runtime import NodeRuntime
from aar.node.worker import RemoteWorker

from .fake_resolve import FakeClock, FakeResolve
from .jobs_support import IDLE, JOB_CONFIG, LAYOUT
from .remote_support import ServerThread, hub_app, write_token

HUMAN = Actor("human", "andrew")
AGENT = Actor("agent", "editor")
NODE = "edit-pc"
PARAMS = {"steps": [{"op": "timeline.list", "params": {}}]}


class _DirectJobs:
    """No-socket companion for the bounded tick test; TCP coverage remains below."""

    lease_seconds = 60.0

    def __init__(self, log: EventLog, project_id: str) -> None:
        self.log = log
        self.projects = (project_id,)
        self.sequence = 0

    def _request(self, prefix: str) -> str:
        self.sequence += 1
        return f"{prefix}-{self.sequence}"

    def claim(self) -> dict | None:
        job, _ = jobs.claim(self.log, NODE, list(self.projects), request_id=self._request("claim"))
        return job.to_dict(jobs.utcnow()) if job is not None else None

    def heartbeat(self, job_id: str, generation: int) -> dict:
        job, _ = jobs.heartbeat(
            self.log,
            NODE,
            job_id,
            generation,
            request_id=self._request("heartbeat"),
        )
        return job.to_dict(jobs.utcnow())

    def report(self, job_id: str, generation: int, state: str, report: dict) -> dict:
        job, _ = jobs.report(
            self.log,
            NODE,
            job_id,
            generation,
            state,
            report,
            request_id=self._request("report"),
        )
        return job.to_dict(jobs.utcnow())

    def reconcile(self, summaries: list[dict]) -> dict:
        return jobs.reconcile(
            self.log,
            NODE,
            summaries,
            list(self.projects),
            request_id=self._request("reconcile"),
        )


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


@contextlib.contextmanager
def _subprocess_hub(db: Path, token_file: Path) -> Iterator[str]:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aar",
            "hub",
            "nodeapi",
            "--db",
            str(db),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    probe = RemoteHubLink(
        _NodeClient(RemoteConfig(url, token_file, connect_timeout=0.2, read_timeout=0.2))
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = (process.stderr.read() if process.stderr is not None else b"")[-2000:]
                raise RuntimeError(f"node API exited early: {detail.decode(errors='replace')}")
            try:
                probe.control_status()
                break
            except HubUnreachable:
                time.sleep(0.02)
        else:
            raise RuntimeError("node API did not start within 10 seconds")
        yield url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()


def _registered(log: EventLog, fake: FakeResolve) -> ProjectLocator:
    project = fake.project("Show")
    project_id = project.GetUniqueId()
    Registry(log).create_project(
        "Show",
        HUMAN,
        project_id="prj_show",
        sources=[ProjectSource("local_db", {"name": "Show", "uid": project_id})],
    )
    return ProjectLocator(project.folder, "Show", project_id)


def _runtime(
    tmp_path: Path,
    base_url: str,
    token_file: Path,
    fake: FakeResolve,
    locator: ProjectLocator,
    *,
    clock: FakeClock | None = None,
    lease_seconds: float = 60.0,
    heartbeat_seconds: float = 15.0,
) -> tuple[NodeRuntime, JobStore, RemoteWorker, FakeClock]:
    clock = clock or FakeClock()
    store = JobStore(tmp_path / "jobs.db")
    gateway = ScriptApiGateway(fake)
    runtime = NodeRuntime.remote(
        gateway,
        store,
        node_id=NODE,
        hub_url=base_url,
        token_file=token_file,
        clock=clock,
        broker_config=JOB_CONFIG,
        idle_window=IDLE,
        input_idle=None,
        approval=ApprovalPolicy(live=False),
        drain_budget=None,
    )
    runtime.presence.observe(gateway.get_context(), clock())
    clock.advance(IDLE)
    config = RemoteConfig(base_url, token_file)
    remote_jobs = RemoteJobClient(
        _NodeClient(config),
        (locator.project_id,),  # type: ignore[arg-type]
        lease_seconds=lease_seconds,
    )
    worker = RemoteWorker(
        runtime,
        store,
        remote_jobs,
        (locator,),
        clock=clock,
        heartbeat_seconds=heartbeat_seconds,
    )
    return runtime, store, worker, clock


def _submit(log: EventLog, locator: ProjectLocator, key: str = "submit") -> jobs.HubJob:
    job, _ = jobs.submit(
        log,
        AGENT,
        kind="ops.sequence",
        params=PARAMS,
        project=locator,
        agent_id="editor",
        assigned_node=NODE,
        idempotency_key=key,
    )
    return job


def test_bounded_tick_runs_the_production_worker_without_a_socket(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    locator = _registered(log, fake)
    submitted = _submit(log, locator)
    clock = FakeClock()
    store = JobStore(tmp_path / "jobs.db")
    gateway = ScriptApiGateway(fake)
    runtime = NodeRuntime.local(
        gateway,
        store,
        log,
        node_id=NODE,
        clock=clock,
        broker_config=JOB_CONFIG,
        idle_window=IDLE,
        input_idle=None,
        approval=ApprovalPolicy(live=False),
        drain_budget=None,
    )
    runtime.presence.observe(gateway.get_context(), clock())
    clock.advance(IDLE)
    worker = RemoteWorker(
        runtime,
        store,
        _DirectJobs(log, locator.project_id),  # type: ignore[arg-type]
        (locator,),
        clock=clock,
    )
    try:
        for _ in range(10):
            worker.tick()
            clock.advance(0.5)
            current = jobs.get_job(log, submitted.job_id)
            if current is not None and current.terminal_state is not None:
                break
        current = jobs.get_job(log, submitted.job_id)
        assert current is not None and current.terminal_state == "succeeded"
        local = runtime.runner.get(submitted.job_id)
        assert local is not None and local.state == "succeeded"
    finally:
        worker.close()
        log.close()


def test_tcp_worker_persists_mapping_claims_once_and_completes(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    locator = _registered(log, fake)
    _, token = nodes.enroll(log, NODE, HUMAN)
    token_file = write_token(tmp_path, token)
    job = _submit(log, locator)
    assert job.state(jobs.utcnow()) == "waiting_for_node"

    # Starting a second server over the same durable log models a hub process restart.
    with _subprocess_hub(tmp_path / "hub.db", token_file):
        pass
    assert jobs.get_job(log, job.job_id) is not None

    with _subprocess_hub(tmp_path / "hub.db", token_file) as hub_url:
        runtime, _store, worker, clock = _runtime(
            tmp_path,
            hub_url,
            token_file,
            fake,
            locator,
            lease_seconds=1,
            heartbeat_seconds=0.25,
        )
        try:
            for _ in range(20):
                worker.tick()
                clock.advance(0.5)
                current = jobs.get_job(log, job.job_id)
                if current is not None and current.terminal_state is not None:
                    break
            current = jobs.get_job(log, job.job_id)
            assert current is not None and current.terminal_state == "succeeded"
            assert runtime.runner.external_job_ids() == {job.job_id}
            local = runtime.runner.get(job.job_id)
            assert local is not None and local.state == "succeeded"
            assert len([event for event in local.events if event["kind"] == "op_submitted"]) == 1
            assert len(log.read(types=[jobs.EVENT_CLAIMED])) == 1

            cancelled_job = _submit(log, locator, "cancel-submit")
            assert worker.tick().claimed == cancelled_job.job_id
            jobs.cancel(
                log,
                AGENT,
                cancelled_job.job_id,
                "changed plan",
                idempotency_key="cancel",
            )
            clock.advance(15)
            worker.tick()
            cancelled_local = runtime.runner.get(cancelled_job.job_id)
            assert cancelled_local is not None and cancelled_local.state == "cancelled"
            assert not [
                event for event in cancelled_local.events if event["kind"] == "op_submitted"
            ]
        finally:
            worker.close()

        # Reopening the same JobStore reconciles terminal local results. Re-sending after a
        # potentially lost ACK does not append a second hub terminal event or execute again.
        runtime2, _store2, worker2, _clock2 = _runtime(
            tmp_path,
            hub_url,
            token_file,
            fake,
            locator,
            lease_seconds=1,
            heartbeat_seconds=0.25,
        )
        offline_job = _submit(log, locator, "offline-submit")
        try:
            restarted = worker2.tick()
            assert restarted.connected and restarted.claimed == offline_job.job_id
            assert len(log.read(types=[jobs.EVENT_FINISHED])) == 2
            assert runtime2.runner.get(job.job_id).state == "succeeded"  # type: ignore[union-attr]
        finally:
            worker2.close()

    # The node disappeared after persisting the claim but before its first execution pump.
    # Lease lapse is only visible state; it never hands the work to another node.
    time.sleep(1.1)
    offline = jobs.get_job(log, offline_job.job_id)
    assert offline is not None and offline.state(jobs.utcnow()) == "node_offline"
    assert jobs.claim(log, "other-pc", [locator.project_id], request_id="steal")[0] is None


def test_reconcile_missing_local_marks_lost_and_never_reexecutes(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    locator = _registered(log, fake)
    _, token = nodes.enroll(log, NODE, HUMAN)
    token_file = write_token(tmp_path, token)
    job = _submit(log, locator)
    claimed, _ = jobs.claim(log, NODE, [locator.project_id], request_id="claim")  # type: ignore[list-item]
    assert claimed is not None

    with ServerThread(hub_app(log)) as server:
        runtime, _store, worker, _ = _runtime(tmp_path, server.base_url, token_file, fake, locator)
        try:
            first = worker.tick()
            assert first.connected
            current = jobs.get_job(log, job.job_id)
            assert current is not None and current.state(jobs.utcnow()) == "lost_on_node"
            assert runtime.runner.get(job.job_id) is None
            for _ in range(3):
                worker.tick()
            assert runtime.runner.get(job.job_id) is None
        finally:
            worker.close()


def test_revoked_enrollment_stops_claiming_without_retry(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    locator = _registered(log, fake)
    _, token = nodes.enroll(log, NODE, HUMAN)
    token_file = write_token(tmp_path, token)

    with ServerThread(hub_app(log)) as server:
        _runtime_obj, _store, worker, _ = _runtime(
            tmp_path, server.base_url, token_file, fake, locator
        )
        try:
            assert worker.tick().connected
            nodes.revoke(log, NODE, HUMAN)
            result = worker.tick()
            assert result.stopped and worker.permanently_stopped
            assert worker.tick().stopped
        finally:
            worker.close()


def test_failed_branch_rebuild_leaves_registry_empty_and_no_claim(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    project = fake.project("Show")
    locator = ProjectLocator(project.folder, "Show", project.GetUniqueId())
    _, token = nodes.enroll(log, NODE, HUMAN)
    token_file = write_token(tmp_path, token)
    job = _submit(log, locator)

    with ServerThread(hub_app(log)) as server:
        runtime, _store, worker, _ = _runtime(tmp_path, server.base_url, token_file, fake, locator)
        try:
            result = worker.tick()
            assert result.stopped and worker.permanently_stopped
            assert runtime.branch_registry.owner(locator.project_id, "anything") is None
            assert jobs.get_job(log, job.job_id).claimed_node is None  # type: ignore[union-attr]
        finally:
            worker.close()


@contextlib.contextmanager
def _direct_world(tmp_path: Path) -> Iterator[tuple]:
    log = EventLog(tmp_path / "hub.db")
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    locator = _registered(log, fake)
    clock = FakeClock()
    store = JobStore(tmp_path / "jobs.db")
    gateway = ScriptApiGateway(fake)
    runtime = NodeRuntime.local(
        gateway,
        store,
        log,
        node_id=NODE,
        clock=clock,
        broker_config=JOB_CONFIG,
        idle_window=IDLE,
        input_idle=None,
        approval=ApprovalPolicy(live=False),
        drain_budget=None,
    )
    runtime.presence.observe(gateway.get_context(), clock())
    clock.advance(IDLE)
    worker = RemoteWorker(
        runtime, store, _DirectJobs(log, locator.project_id), (locator,), clock=clock
    )
    try:
        yield log, fake, locator, runtime, store, worker, clock
    finally:
        worker.close()
        log.close()


def test_claim_mapping_contains_generation_before_first_pump(tmp_path: Path) -> None:
    with _direct_world(tmp_path) as (log, _, locator, runtime, store, worker, _):
        job = _submit(log, locator)
        assert worker.tick().claimed == job.job_id
        events = store.events(job.job_id)
        assert events[0].kind == "external_assigned"
        assert events[0].data == {"hub_job_id": job.job_id, "generation": 1}
        assert runtime.runner.external_generation(job.job_id) == 1
        assert not any(event.kind == "op_submitted" for event in events)


def test_legacy_and_assigned_but_unclaimed_jobs_are_quarantined(tmp_path: Path) -> None:
    with _direct_world(tmp_path) as (log, _, locator, runtime, _, worker, _):
        legacy = runtime.runner.submit("ops.sequence", PARAMS, locator, agent_id="editor")
        hub_job = _submit(log, locator)
        runtime.runner.submit(
            "ops.sequence", PARAMS, locator, agent_id="editor", job_id=hub_job.job_id, generation=1
        )
        worker.stop_claiming()
        assert worker.tick().connected
        for job_id in (legacy, hub_job.job_id):
            view = runtime.runner.get(job_id)
            assert view.state == "cancelled"
            assert any(e["kind"] == "external_quarantined" for e in view.events)
            assert not any(e["kind"] == "op_submitted" for e in view.events)
        assert jobs.get_job(log, hub_job.job_id).claimed_node is None


def test_remote_attachment_blocks_already_queued_legacy_op(tmp_path: Path) -> None:
    with _direct_world(tmp_path) as (_, _fake, locator, runtime, _, worker, _):
        job_id = runtime.runner.submit("ops.sequence", PARAMS, locator, agent_id="editor")
        runtime.runner.remote_claim_check = None  # before attachment: local-only behaviour
        runtime.pump()  # queue an op, but do not execute it yet
        assert any(e["kind"] == "op_submitted" for e in runtime.runner.get(job_id).events)
        runtime.runner.remote_claim_check = worker._check_claim
        submitted = next(
            e for e in runtime.runner.get(job_id).events if e["kind"] == "op_submitted"
        )
        runtime.pump()  # no reconciliation, even a direct pump must not dispatch
        result = runtime.broker.result(submitted["data"]["op_id"])
        assert result.status.value == "cancelled" and result.error.code == "hub_unreachable"
        view = runtime.runner.get(job_id)
        assert not any(e["kind"] == "step_completed" for e in view.events)


@pytest.mark.parametrize("assigned_node", [NODE, "other-pc"])
@pytest.mark.parametrize("terminal", [False, True])
def test_restart_never_adopts_released_generation(
    tmp_path: Path,
    assigned_node: str,
    terminal: bool,
) -> None:
    with _direct_world(tmp_path) as (log, _, locator, runtime, _, worker, clock):
        job = _submit(log, locator)
        assert worker.tick().claimed == job.job_id
        if terminal:
            # Local completion with a lost terminal ACK; the hub remains nonterminal.
            for _ in range(5):
                runtime.pump()
                clock.advance(0.5)
            assert runtime.runner.get(job.job_id).state == "succeeded"
        jobs.release(
            log, HUMAN, job.job_id, assigned_node, "old node checked", idempotency_key="release"
        )
        if assigned_node == NODE:
            jobs.claim(log, NODE, [locator.project_id], request_id="new-generation")
        worker.close()
        store = JobStore(tmp_path / "jobs.db")
        # Re-open through production runtime construction; no memory generation survives.
        gateway = ScriptApiGateway(FakeResolve(LAYOUT, current="Show", folder=("Shows",)))
        restarted = NodeRuntime.local(
            gateway,
            store,
            log,
            node_id=NODE,
            clock=clock,
            broker_config=JOB_CONFIG,
            idle_window=IDLE,
            input_idle=None,
            approval=ApprovalPolicy(live=False),
            drain_budget=None,
        )
        client = _DirectJobs(log, locator.project_id)
        client.sequence = 100
        replacement = RemoteWorker(restarted, store, client, (locator,), clock=clock)
        replacement.stop_claiming()
        try:
            assert replacement.tick().connected
            view = restarted.runner.get(job.job_id)
            assert any(e["kind"] == "external_quarantined" for e in view.events)
            assert view.state == ("succeeded" if terminal else "cancelled")
            current = jobs.get_job(log, job.job_id)
            assert current.generation == 2 and current.terminal_state is None
            assert not log.read(types=[jobs.EVENT_FINISHED])
            assert job.job_id not in replacement._generation
            if assigned_node == NODE:
                assert current.lost_on_node
        finally:
            replacement.close()


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        {},
        {"hub_job_id": "job_wrong", "generation": 1},
        {"generation": True},
        {"generation": 0},
    ],
)
def test_bad_mapping_is_quarantined_and_hub_claim_is_lost(
    tmp_path: Path, mapping: dict | None
) -> None:
    with _direct_world(tmp_path) as (log, _, locator, _runtime_obj, store, worker, clock):
        job = _submit(log, locator)
        jobs.claim(log, NODE, [locator.project_id], request_id="claim-local")
        from aar.node.jobs.model import JobState
        from aar.node.jobs.store import JobRecord

        record = JobRecord(
            job.job_id,
            "ops.sequence",
            NODE,
            locator,
            PARAMS,
            agent_id="editor",
            state=JobState.QUEUED,
        )
        events = (
            []
            if mapping is None
            else [("external_assigned", {"hub_job_id": job.job_id, **mapping}, clock())]
        )
        store.save(record, events)
        worker.close()
        reopened_store = JobStore(tmp_path / "jobs.db")
        gateway = ScriptApiGateway(FakeResolve(LAYOUT, current="Show", folder=("Shows",)))
        reopened = NodeRuntime.local(
            gateway,
            reopened_store,
            log,
            node_id=NODE,
            clock=clock,
            broker_config=JOB_CONFIG,
            idle_window=IDLE,
            input_idle=None,
            approval=ApprovalPolicy(live=False),
            drain_budget=None,
        )
        client = _DirectJobs(log, locator.project_id)
        client.sequence = 100
        replacement = RemoteWorker(reopened, reopened_store, client, (locator,), clock=clock)
        replacement.stop_claiming()
        try:
            for _ in range(3):
                assert replacement.tick().connected
            assert jobs.get_job(log, job.job_id).lost_on_node
            assert reopened.runner.get(job.job_id).state == "cancelled"
            assert not any(e.kind == "op_submitted" for e in reopened_store.events(job.job_id))
        finally:
            replacement.close()


def test_release_between_queue_and_dispatch_prevents_fake_mutation(tmp_path: Path) -> None:
    with _direct_world(tmp_path) as (log, fake, locator, runtime, _, worker, clock):
        from aar.hub.collab import record_branch

        main = fake.project("Show").timeline("Main")
        record_branch(
            log,
            "prj_show",
            Actor("node", NODE),
            source_timeline_id=main.GetUniqueId(),
            source_timeline_name="Main",
            agent_id="editor",
            branch_timeline_id=main.GetUniqueId(),
            branch_timeline_name="Main",
            version=1,
            handoff="test",
        )
        job, _ = jobs.submit(
            log,
            AGENT,
            kind="ops.sequence",
            project=locator,
            agent_id="editor",
            assigned_node=NODE,
            idempotency_key="mutate",
            params={
                "steps": [
                    {
                        "op": "timeline.duplicate",
                        "params": {
                            "source": {"name": "Main", "unique_id": main.GetUniqueId()},
                            "name": "Should never exist",
                        },
                    }
                ]
            },
        )
        assert worker.tick().claimed == job.job_id
        worker.tick()  # queue op; heartbeat deadline is still far away
        assert any(e["kind"] == "op_submitted" for e in runtime.runner.get(job.job_id).events)
        assert clock() < worker._next_heartbeat[job.job_id]
        jobs.release(log, HUMAN, job.job_id, NODE, "checked old node", idempotency_key="release")
        jobs.claim(log, NODE, [locator.project_id], request_id="new-claim")
        result = worker.tick()
        assert result.stopped
        assert "Should never exist" not in [t.GetName() for t in fake.project("Show").timelines]
        assert runtime.runner.get(job.job_id).state == "cancelled"
        assert not any(e["kind"] == "step_completed" for e in runtime.runner.get(job.job_id).events)


def test_lost_terminal_ack_resends_without_duplicate_op_or_finish(tmp_path: Path) -> None:
    with _direct_world(tmp_path) as (log, _, locator, runtime, store, worker, clock):
        job = _submit(log, locator)
        original = worker.jobs.report
        dropped = False

        def drop_ack(job_id: str, generation: int, state: str, report: dict) -> dict:
            nonlocal dropped
            result = original(job_id, generation, state, report)
            if state == "succeeded" and not dropped:
                dropped = True
                raise HubUnreachable("terminal ACK lost after hub commit")
            return result

        worker.jobs.report = drop_ack
        for _ in range(15):
            worker.tick()
            clock.advance(1)
            if any(e.kind == "external_result_acked" for e in store.events(job.job_id)):
                break
        assert dropped
        assert runtime.runner.get(job.job_id).state == "succeeded"
        assert len(log.read(types=[jobs.EVENT_FINISHED])) == 1
        assert sum(e.kind == "op_submitted" for e in store.events(job.job_id)) == 1
        assert sum(e.kind == "external_result_acked" for e in store.events(job.job_id)) == 1


def test_tcp_reconcile_over_thousand_historical_jobs_then_pending(tmp_path: Path) -> None:
    import json

    from aar.node.jobs.model import JobState
    from aar.node.jobs.store import JobRecord

    log = EventLog(tmp_path / "hub.db")
    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    locator = _registered(log, fake)
    _, token = nodes.enroll(log, NODE, HUMAN)
    token_file = write_token(tmp_path, token)
    template = _submit(log, locator)
    payload = log.read(types=[jobs.EVENT_SUBMITTED])[0].payload
    # Model genuine acknowledged history without quadratic public-API setup. Both journals
    # still contain real production event/record shapes, then the real HTTP client pages them.
    historical = [template.job_id, *(f"job_history{i:04d}" for i in range(1004))]
    store = JobStore(tmp_path / "jobs.db")
    try:
        with log.transaction() as txn:
            for index, job_id in enumerate(historical):
                if job_id != template.job_id:
                    txn.append(jobs.EVENT_SUBMITTED, AGENT, {**payload, "job_id": job_id})
                txn.append(
                    jobs.EVENT_CLAIMED,
                    Actor("node", NODE),
                    {
                        "job_id": job_id,
                        "generation": 1,
                        "node_id": NODE,
                        "lease_expires_at": "2026-10-20T23:59:59+00:00",
                    },
                )
                body = {
                    "current_step": 1,
                    "steps_total": 1,
                    "step_name": None,
                    "results": [{"history": index}],
                    "measured": {"history": index},
                    "error": None,
                    "waiting": None,
                    "stall": None,
                    "alerts": [],
                    "updated_at": 0.0,
                }
                txn.append(
                    jobs.EVENT_FINISHED,
                    Actor("node", NODE),
                    {
                        "job_id": job_id,
                        "generation": 1,
                        "node_id": NODE,
                        "state": "succeeded",
                        "result": body,
                    },
                )
                record = JobRecord(
                    job_id,
                    "ops.sequence",
                    NODE,
                    locator,
                    PARAMS,
                    agent_id="editor",
                    state=JobState.SUCCEEDED,
                    current_step=1,
                    results=[{"history": index}],
                )
                fingerprint = json.dumps({"state": "succeeded", "report": body}, sort_keys=True)
                store.save(
                    record,
                    [
                        ("external_assigned", {"hub_job_id": job_id, "generation": 1}, 0),
                        ("external_result_acked", {"generation": 1, "fingerprint": fingerprint}, 0),
                    ],
                )
            # A claimed local pending job sorts AFTER the first thousand summaries. An
            # incomplete reconciliation must not falsely mark it lost on an earlier page.
            pending_id = "job_zzpending"
            txn.append(jobs.EVENT_SUBMITTED, AGENT, {**payload, "job_id": pending_id})
            txn.append(
                jobs.EVENT_CLAIMED,
                Actor("node", NODE),
                {
                    "job_id": pending_id,
                    "generation": 1,
                    "node_id": NODE,
                    "lease_expires_at": "2026-10-20T23:59:59+00:00",
                },
            )
            store.save(
                JobRecord(pending_id, "ops.sequence", NODE, locator, PARAMS, agent_id="editor"),
                [
                    ("external_assigned", {"hub_job_id": pending_id, "generation": 1}, 0),
                ],
            )
    finally:
        store.close()
    try:
        with ServerThread(hub_app(log)) as server:
            runtime, _, worker, clock = _runtime(
                tmp_path, server.base_url, token_file, fake, locator
            )
            try:
                for _ in range(10):
                    assert worker.tick().connected
                    clock.advance(0.5)
                    if jobs.get_job(log, pending_id).terminal_state:
                        break
                current = jobs.get_job(log, pending_id)
                assert current.terminal_state == "succeeded" and not current.lost_on_node
                assert runtime.runner.get(pending_id).state == "succeeded"
                assert not worker.permanently_stopped
                assert len(worker._generation) == 1006
                assert len(log.read(types=[jobs.EVENT_LOST])) == 0
                assert len(list(log.iter_events(types=[jobs.EVENT_FINISHED]))) == 1006
                assert not any(
                    e["kind"] == "external_quarantined"
                    for e in runtime.runner.get(historical[-1]).events
                )
            finally:
                worker.close()
    finally:
        log.close()


@pytest.mark.parametrize("failure", ["branches", "reconcile"])
def test_failed_reconcile_blocks_previously_queued_op(tmp_path: Path, failure: str) -> None:
    with _direct_world(tmp_path) as (log, fake, locator, runtime, _, worker, _):
        job = _submit(log, locator)
        worker.tick()
        worker.tick()  # real current claim, with its first op queued
        assert any(e["kind"] == "op_submitted" for e in runtime.runner.get(job.job_id).events)
        main = fake.project("Show").timeline("Main")
        calls = []
        original_start = main.GetStartFrame

        def observe() -> int:
            calls.append("executed")
            return original_start()

        main.GetStartFrame = observe

        def unavailable(*args: object) -> None:
            raise HubUnreachable("injected reconciliation outage")

        worker._connected = False
        if failure == "branches":
            runtime.rebuild_branches = unavailable
        else:
            worker.jobs.reconcile = unavailable
        assert not worker.tick().connected
        runtime.pump()  # even a queued broker drain cannot use old permission
        assert not calls
        assert not any(e["kind"] == "step_completed" for e in runtime.runner.get(job.job_id).events)


def test_remote_result_boundaries_refuse_nonfinite_cyclic_and_oversize() -> None:
    class NoRequests:
        def call(self, *args: object, **kw: object) -> dict:
            raise AssertionError("malformed result must not reach transport")

    from aar.node.remote import HubRefused

    remote = RemoteJobClient(NoRequests(), ("project",))
    cyclic: dict = {}
    cyclic["self"] = cyclic
    for payload in ({"bad": float("nan")}, cyclic, {"large": "x" * (256 * 1024)}):
        with pytest.raises(HubRefused):
            remote.report("job_test", 1, "succeeded", payload)


def test_reconcile_rejects_nonprogressing_output_cursor() -> None:
    class BrokenPages:
        def call(self, *args: object, **kw: object) -> dict:
            return {"result": {"jobs": [], "known_job_ids": ["job_one"], "next_cursor": "job_zero"}}

    remote = RemoteJobClient(BrokenPages(), ("project",))
    with pytest.raises(HubUnreachable, match="cursor"):
        remote.reconcile([])


def test_reconcile_pages_large_hub_replies_with_empty_local_journal(tmp_path: Path) -> None:
    import json

    log = EventLog(tmp_path / "hub.db")
    locator = ProjectLocator(("Shows",), "x" * 12000, "project-large")
    _, token = nodes.enroll(log, NODE, HUMAN)
    token_file = write_token(tmp_path, token)
    template = _submit(log, locator)
    payload = log.read(types=[jobs.EVENT_SUBMITTED])[0].payload
    try:
        with log.transaction() as txn:
            for index in range(360):
                job_id = template.job_id if index == 0 else f"job_large{index:04d}"
                if index:
                    txn.append(jobs.EVENT_SUBMITTED, AGENT, {**payload, "job_id": job_id})
                txn.append(
                    jobs.EVENT_CLAIMED,
                    Actor("node", NODE),
                    {
                        "job_id": job_id,
                        "generation": 1,
                        "node_id": NODE,
                        "lease_expires_at": "2026-10-20T23:59:59+00:00",
                    },
                )
        unpaged = [job.to_dict(jobs.utcnow(), include_payload=False) for job in jobs.list_jobs(log)]
        assert len(json.dumps(unpaged).encode()) > 4 << 20
        with ServerThread(hub_app(log)) as server:
            remote = RemoteJobClient(
                _NodeClient(RemoteConfig(server.base_url, token_file)), (locator.project_id,)
            )
            result = remote.reconcile([])
        assert len(result["jobs"]) == 360 and len(result["known_job_ids"]) == 360
        assert all(job["state"] == "lost_on_node" for job in result["jobs"])
        assert len(log.read(types=[jobs.EVENT_LOST])) == 360
        assert jobs.claim(log, NODE, [locator.project_id], request_id="no-rerun")[0] is None
    finally:
        log.close()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0, -1])
def test_worker_rejects_nonfinite_or_invalid_timing(tmp_path: Path, value: float) -> None:
    with (
        _direct_world(tmp_path) as (_, _, locator, runtime, store, worker, _),
        pytest.raises(ValueError, match="positive finite"),
    ):
        RemoteWorker(runtime, store, worker.jobs, (locator,), tick_seconds=value)


def test_malformed_claim_reply_leaves_no_dispatch(tmp_path: Path) -> None:
    with _direct_world(tmp_path) as (log, _, locator, runtime, _, worker, _):
        job = _submit(log, locator)
        original_claim = worker.jobs.claim

        def malformed() -> dict:
            return {**original_claim(), "cancel_requested": "false"}

        worker.jobs.claim = malformed
        result = worker.tick()
        assert not result.connected and result.retry_in > 0
        assert runtime.runner.get(job.job_id) is None


def test_scheduled_heartbeat_refusal_persists_stale_quarantine(tmp_path: Path) -> None:
    with _direct_world(tmp_path) as (log, _, locator, runtime, _, worker, clock):
        job = _submit(log, locator)
        assert worker.tick().claimed == job.job_id
        jobs.release(
            log, HUMAN, job.job_id, "other-pc", "old node stopped", idempotency_key="release"
        )
        clock.advance(15)
        assert worker.tick().stopped
        assert runtime.runner.get(job.job_id).state == "cancelled"
        assert runtime.runner.external_generation(job.job_id) is None
