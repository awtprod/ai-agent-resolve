from __future__ import annotations

import json
import math
import threading
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aar.cli import main
from aar.hub import jobs
from aar.hub.common import HumanRequired, InvalidInput
from aar.hub.events import Actor, EventLog
from aar.node.locator import ProjectLocator

AGENT = Actor("agent", "editor")
OTHER = Actor("agent", "other")
HUMAN = Actor("human", "andrew")
PROJECT = ProjectLocator(("Shows",), "Show", "resolve-show")
PARAMS = {"steps": [{"op": "timeline.list", "params": {}}]}


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 20, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _submit(log: EventLog, key: str = "submit-1", **kw: object) -> jobs.HubJob:
    job, _ = jobs.submit(
        log,
        AGENT,
        kind="ops.sequence",
        params=PARAMS,
        project=PROJECT,
        agent_id="editor",
        assigned_node="edit-pc",
        idempotency_key=key,
        **kw,  # type: ignore[arg-type]
    )
    return job


def test_submit_is_validated_before_the_durable_event_and_replays(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    first = _submit(log)
    replay, replayed = jobs.submit(
        log,
        AGENT,
        kind="ops.sequence",
        params=PARAMS,
        project=PROJECT,
        agent_id="editor",
        assigned_node="edit-pc",
        idempotency_key="submit-1",
    )
    assert replayed is True and replay.job_id == first.job_id
    assert len(log.read(types=[jobs.EVENT_SUBMITTED])) == 1

    malformed = {"steps": [{"op": "timeline.list", "params": {"bad": math.nan}}]}
    with pytest.raises(InvalidInput, match="finite"):
        jobs.submit(
            log,
            AGENT,
            kind="ops.sequence",
            params=malformed,
            project=PROJECT,
            agent_id="editor",
            assigned_node="edit-pc",
            idempotency_key="bad",
        )
    assert len(log.read(types=[jobs.EVENT_SUBMITTED])) == 1
    log.close()


@pytest.mark.parametrize("actor", [AGENT, HUMAN])
def test_cancel_unclaimed_is_terminal_on_reopen_and_replay(tmp_path: Path, actor: Actor) -> None:
    db = tmp_path / "cancel.db"
    with closing(EventLog(db)) as log:
        job = _submit(log)
        with pytest.raises(HumanRequired):
            jobs.cancel(log, OTHER, job.job_id, "stop", idempotency_key="unauthorized")
        cancelled, replay = jobs.cancel(log, actor, job.job_id, "stop", idempotency_key="stop")
        assert not replay and cancelled.terminal_state == "cancelled"
    with closing(EventLog(db, create=False)) as log:
        again, replay = jobs.cancel(log, actor, job.job_id, "stop", idempotency_key="stop")
        assert replay and again.state(jobs.utcnow()) == "cancelled"
        assert jobs.claim(log, "edit-pc", ["resolve-show"], request_id="claim")[0] is None
        assert len(log.read(types=[jobs.EVENT_CANCEL])) == 1


def test_legacy_unclaimed_cancel_event_is_terminal(tmp_path: Path) -> None:
    with closing(EventLog(tmp_path / "hub.db")) as log:
        job = _submit(log)
        log.append(
            jobs.EVENT_CANCEL,
            AGENT,
            {
                "job_id": job.job_id,
                "generation": 1,
                "reason": "legacy cancellation",
            },
        )
        # Older hubs accepted a second authorized cancellation with a fresh request key.
        log.append(
            jobs.EVENT_CANCEL,
            HUMAN,
            {
                "job_id": job.job_id,
                "generation": 1,
                "reason": "legacy repeated cancellation",
            },
        )
        assert jobs.get_job(log, job.job_id).terminal_state == "cancelled"
        assert jobs.claim(log, "edit-pc", ["resolve-show"], request_id="claim")[0] is None


@pytest.mark.parametrize("claim_first", [False, True])
def test_cancel_and_claim_order_is_atomic(tmp_path: Path, claim_first: bool) -> None:
    with closing(EventLog(tmp_path / "hub.db")) as log:
        job = _submit(log)
        if claim_first:
            assert jobs.claim(log, "edit-pc", ["resolve-show"], request_id="first")[0]
        current, _ = jobs.cancel(log, AGENT, job.job_id, "stop", idempotency_key="stop")
        assert current.terminal_state == (None if claim_first else "cancelled")
        assert current.state(jobs.utcnow()) == ("cancel_requested" if claim_first else "cancelled")
        assert jobs.claim(log, "edit-pc", ["resolve-show"], request_id="second")[0] is None


def test_sticky_claim_expires_visible_but_never_requeues_and_release_fences(tmp_path: Path) -> None:
    clock = Clock()
    log = EventLog(tmp_path / "hub.db")
    job, _ = jobs.submit(
        log,
        AGENT,
        kind="ops.sequence",
        params=PARAMS,
        project=PROJECT,
        agent_id="editor",
        assigned_node="edit-pc",
        idempotency_key="s",
        clock=clock,
    )
    claimed, _ = jobs.claim(
        log, "edit-pc", ["resolve-show"], request_id="c", lease_seconds=60, clock=clock
    )
    assert claimed is not None and claimed.job_id == job.job_id
    clock.advance(61)
    assert jobs.get_job(log, job.job_id).state(clock()) == "node_offline"  # type: ignore[union-attr]
    assert jobs.claim(log, "other-pc", ["resolve-show"], request_id="other", clock=clock)[0] is None

    released, _ = jobs.release(
        log,
        HUMAN,
        job.job_id,
        "other-pc",
        "checked Resolve; safe to retry",
        idempotency_key="release",
        clock=clock,
    )
    assert released.generation == 2 and released.state(clock()) == "waiting_for_node"
    with pytest.raises(jobs.StaleGeneration):
        jobs.heartbeat(log, "edit-pc", job.job_id, 1, request_id="stale", clock=clock)
    newer, _ = jobs.claim(log, "other-pc", ["resolve-show"], request_id="new", clock=clock)
    assert newer is not None and newer.generation == 2
    log.close()


def test_two_claimers_race_to_one_append(tmp_path: Path) -> None:
    path = tmp_path / "hub.db"
    setup = EventLog(path)
    job = _submit(setup)
    setup.close()
    barrier = threading.Barrier(2)
    results: list[str | None] = []
    errors: list[BaseException] = []

    def claim(key: str) -> None:
        log = EventLog(path, create=False)
        try:
            barrier.wait(timeout=5)
            found, _ = jobs.claim(log, "edit-pc", ["resolve-show"], request_id=key)
            results.append(found.job_id if found is not None else None)
        except BaseException as exc:  # surfaced below, not swallowed
            errors.append(exc)
        finally:
            log.close()

    threads = [threading.Thread(target=claim, args=(key,)) for key in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    assert sorted(value is not None for value in results) == [False, True]
    assert job.job_id in results
    check = EventLog(path, create=False, read_only=True)
    assert len(check.read(types=[jobs.EVENT_CLAIMED])) == 1
    check.close()


def test_cancel_authority_and_terminal_resend_are_idempotent(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    job = _submit(log)
    with pytest.raises(HumanRequired):
        jobs.cancel(log, OTHER, job.job_id, "no", idempotency_key="c0")
    cancelled, _ = jobs.cancel(log, AGENT, job.job_id, "stop", idempotency_key="c1")
    assert cancelled.cancel_requested

    other = _submit(log, "submit-2")
    jobs.claim(log, "edit-pc", ["resolve-show"], request_id="claim-2")
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(jobs.MAX_JSON_DEPTH + 1):
        child: dict[str, object] = {}
        cursor["x"] = child
        cursor = child
    with pytest.raises(InvalidInput, match="nest"):
        jobs.report(
            log,
            "edit-pc",
            other.job_id,
            1,
            "running",
            nested,
            request_id="too-deep",
        )
    finished, _ = jobs.report(
        log,
        "edit-pc",
        other.job_id,
        1,
        "succeeded",
        {"results": [{"count": 2}]},
        request_id="done-1",
    )
    again, replayed = jobs.report(
        log,
        "edit-pc",
        other.job_id,
        1,
        "succeeded",
        {"results": [{"count": 2}]},
        request_id="done-after-lost-ack",
    )
    assert replayed and again.result == finished.result
    assert len(log.read(types=[jobs.EVENT_FINISHED])) == 1
    log.close()


def test_job_cli_submit_list_show_cancel_and_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "hub.db"
    EventLog(db).close()
    base = ["hub", "job"]
    assert (
        main(
            [
                *base,
                "submit",
                "--db",
                str(db),
                "--actor",
                "agent:editor",
                "--kind",
                "ops.sequence",
                "--params",
                json.dumps(PARAMS),
                "--project",
                json.dumps(PROJECT.to_dict()),
                "--node",
                "edit-pc",
                "--agent-id",
                "editor",
                "--idempotency-key",
                "cli-submit",
            ]
        )
        == 0
    )
    job_id = json.loads(capsys.readouterr().out)["job"]["job_id"]
    assert main([*base, "list", "--db", str(db), "--state", "waiting_for_node"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"][0]["job_id"] == job_id
    assert main([*base, "show", "--db", str(db), job_id]) == 0
    assert json.loads(capsys.readouterr().out)["job"]["job_id"] == job_id

    writable = EventLog(db, create=False)
    jobs.claim(writable, "edit-pc", ["resolve-show"], request_id="cli-claim")
    writable.close()
    assert (
        main(
            [
                *base,
                "cancel",
                "--db",
                str(db),
                "--actor",
                "agent:editor",
                job_id,
                "--reason",
                "stop",
                "--idempotency-key",
                "cli-cancel",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                *base,
                "release",
                "--db",
                str(db),
                "--actor",
                "human:andrew",
                job_id,
                "--node",
                "other-pc",
                "--reason",
                "checked Resolve",
                "--idempotency-key",
                "cli-release",
            ]
        )
        == 0
    )
    released = json.loads(capsys.readouterr().out)["job"]
    assert released["generation"] == 2 and released["assigned_node"] == "other-pc"


def test_release_key_cannot_replay_against_another_job(tmp_path: Path) -> None:
    from aar.hub.events import IdempotencyConflict

    with closing(EventLog(tmp_path / "hub.db")) as log:
        first, second = _submit(log), _submit(log, "second")
        jobs.claim(log, "edit-pc", [PROJECT.project_id], request_id="claim-first")
        jobs.release(log, HUMAN, first.job_id, "other", "safe", idempotency_key="release")
        with pytest.raises(IdempotencyConflict):
            jobs.release(log, HUMAN, second.job_id, "other", "safe", idempotency_key="release")


def test_claim_replay_respects_project_scope_and_current_generation(tmp_path: Path) -> None:
    with closing(EventLog(tmp_path / "hub.db")) as log:
        job = _submit(log)
        jobs.claim(log, "edit-pc", [PROJECT.project_id], request_id="claim")
        with pytest.raises(jobs.JobConflict):
            jobs.claim(log, "edit-pc", ["another-project"], request_id="claim")
        jobs.release(log, HUMAN, job.job_id, "edit-pc", "safe", idempotency_key="release")
        with pytest.raises(jobs.StaleGeneration):
            jobs.claim(log, "edit-pc", [PROJECT.project_id], request_id="claim")


@pytest.mark.parametrize("operation", ["heartbeat", "report"])
def test_old_request_replay_is_fenced_after_release(tmp_path: Path, operation: str) -> None:
    with closing(EventLog(tmp_path / "hub.db")) as log:
        job = _submit(log)
        jobs.claim(log, "edit-pc", [PROJECT.project_id], request_id="claim")

        def send() -> None:
            if operation == "heartbeat":
                jobs.heartbeat(log, "edit-pc", job.job_id, 1, request_id="same")
            else:
                jobs.report(log, "edit-pc", job.job_id, 1, "running", {}, request_id="same")

        send()
        jobs.release(log, HUMAN, job.job_id, "edit-pc", "safe", idempotency_key="release")
        jobs.claim(log, "edit-pc", [PROJECT.project_id], request_id="new-claim")
        with pytest.raises(jobs.StaleGeneration):
            send()


def test_job_fold_reads_heartbeat_and_completion_beyond_first_thousand(tmp_path: Path) -> None:
    clock = Clock()
    with closing(EventLog(tmp_path / "hub.db")) as log:
        job = _submit(log)
        jobs.claim(log, "edit-pc", [PROJECT.project_id], request_id="claim", clock=clock)
        # Append valid history directly to avoid quadratic setup through the fold on each write.
        with log.transaction() as txn:
            for index in range(1001):
                txn.append(
                    jobs.EVENT_HEARTBEAT,
                    Actor("node", "edit-pc"),
                    {
                        "job_id": job.job_id,
                        "generation": 1,
                        "node_id": "edit-pc",
                        "lease_expires_at": (clock() + timedelta(seconds=index + 60)).isoformat(),
                    },
                    ts=clock().isoformat(),
                )
        folded = jobs.get_job(log, job.job_id)
        assert folded is not None
        assert folded.lease_expires_at == (clock() + timedelta(seconds=1060)).isoformat()
        jobs.report(log, "edit-pc", job.job_id, 1, "succeeded", {"count": 1}, request_id="done")
        folded = jobs.get_job(log, job.job_id)
        assert folded is not None and folded.terminal_state == "succeeded"
        assert folded.result == {"count": 1} and folded.event_seq > 1000


@pytest.mark.parametrize(
    "summary",
    [
        {"job_id": "job_test", "state": "running"},
        {"job_id": "job_test", "state": "running", "generation": True},
        {"job_id": "job_test", "state": "running", "generation": 0},
        {"job_id": "job_test", "state": "invented", "generation": 1},
    ],
)
def test_reconcile_validates_durable_summary_generation(tmp_path: Path, summary: dict) -> None:
    with closing(EventLog(tmp_path / "hub.db")) as log:
        job = _submit(log)
        jobs.claim(log, "edit-pc", [PROJECT.project_id], request_id="claim")
        with pytest.raises(InvalidInput):
            jobs.reconcile(log, "edit-pc", [summary], [PROJECT.project_id], request_id="bad")
        assert not jobs.get_job(log, job.job_id).lost_on_node
