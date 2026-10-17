"""Tests for human approval tokens: grant, revoke, and atomic single-use consume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aar.hub import approvals
from aar.hub.approvals import ApprovalScope
from aar.hub.common import HumanRequired, InvalidInput
from aar.hub.events import Actor, EventLog, IdempotencyConflict

HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")
NODE = Actor("node", "node-1")
T0 = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
TARGET = "prj_demo-show"


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "hub.db")


def _grant(log: EventLog, clock: Clock, **kwargs: object) -> approvals.Approval:
    params: dict[str, object] = {
        "scope": ApprovalScope.PROJECT_WRITE,
        "target": TARGET,
        "ttl": timedelta(minutes=15),
        "reason": "write the acme cut back",
        "by": HUMAN,
        "clock": clock,
    }
    params.update(kwargs)
    return approvals.grant(log, **params)  # type: ignore[arg-type]


def _refusal_codes(log: EventLog) -> list[str]:
    return [e.payload["code"] for e in log.read(types=[approvals.EVENT_CONSUME_REFUSED])]


def test_grant_records_a_bound_expiring_token(log: EventLog) -> None:
    clock = Clock()
    token = _grant(log, clock)
    assert token.token_id.startswith("apv_")
    assert token.scope is ApprovalScope.PROJECT_WRITE and token.target == TARGET
    assert token.granted_by == HUMAN
    assert token.expires_at == (T0 + timedelta(minutes=15)).isoformat()
    assert token.state(T0) == "active"
    assert approvals.list_approvals(log) == [token]
    assert [e.type for e in log.read()] == [approvals.EVENT_GRANTED]


@pytest.mark.parametrize("actor", [AGENT, NODE])
def test_only_humans_grant_or_revoke(log: EventLog, actor: Actor) -> None:
    clock = Clock()
    with pytest.raises(HumanRequired):
        _grant(log, clock, by=actor)
    token = _grant(log, clock)
    with pytest.raises(HumanRequired):
        approvals.revoke(log, token.token_id, "no", actor, clock=clock)
    assert approvals.get_approval(log, token.token_id) == token  # unchanged


def test_grant_validates_scope_ttl_and_text(log: EventLog) -> None:
    clock = Clock()
    with pytest.raises(InvalidInput):
        _grant(log, clock, scope="db.drop")
    with pytest.raises(InvalidInput):
        _grant(log, clock, ttl=timedelta(hours=25))
    with pytest.raises(InvalidInput):
        _grant(log, clock, ttl=0)
    with pytest.raises(InvalidInput):
        _grant(log, clock, reason="")
    with pytest.raises(InvalidInput):
        _grant(log, clock, target="x" * 201)
    assert log.latest_seq() == 0


def test_grant_retry_returns_the_same_token(log: EventLog) -> None:
    clock = Clock()
    first = _grant(log, clock, idempotency_key="form-1")
    clock.now = T0 + timedelta(seconds=3)
    again = _grant(log, clock, idempotency_key="form-1")
    assert again == first
    assert len(approvals.list_approvals(log)) == 1
    with pytest.raises(IdempotencyConflict):
        _grant(log, clock, idempotency_key="form-1", target="prj_other")


def test_consume_is_single_use(log: EventLog) -> None:
    clock = Clock()
    token = _grant(log, clock)
    used = approvals.consume(log, token.token_id, "project.write", TARGET, NODE, clock=clock)
    assert used.state(T0) == "consumed" and used.consumed_by == NODE
    with pytest.raises(approvals.ApprovalAlreadyConsumed):
        approvals.consume(log, token.token_id, "project.write", TARGET, AGENT, clock=clock)
    assert len(log.read(types=[approvals.EVENT_CONSUMED])) == 1
    assert _refusal_codes(log) == ["already_consumed"]


def test_expired_token_is_refused(log: EventLog) -> None:
    clock = Clock()
    token = _grant(log, clock, ttl=60)
    clock.now = T0 + timedelta(seconds=60)
    with pytest.raises(approvals.ApprovalExpired):
        approvals.consume(log, token.token_id, "project.write", TARGET, NODE, clock=clock)
    assert approvals.list_approvals(log)[0].state(clock.now) == "expired"
    assert _refusal_codes(log) == ["expired"]
    # A second before expiry the same token would have worked.
    fresh = _grant(log, Clock(T0), ttl=60)
    clock.now = T0 + timedelta(seconds=59)
    approvals.consume(log, fresh.token_id, "project.write", TARGET, NODE, clock=clock)


def test_revoked_token_is_refused(log: EventLog) -> None:
    clock = Clock()
    token = _grant(log, clock)
    revoked = approvals.revoke(log, token.token_id, "wrong project", HUMAN, clock=clock)
    assert revoked.state(T0) == "revoked" and revoked.revoked_by == HUMAN
    with pytest.raises(approvals.ApprovalRevoked):
        approvals.consume(log, token.token_id, "project.write", TARGET, NODE, clock=clock)
    with pytest.raises(approvals.ApprovalRevoked):
        approvals.revoke(log, token.token_id, "again", HUMAN, clock=clock)
    assert _refusal_codes(log) == ["revoked"]


def test_wrong_scope_and_wrong_target_are_refused_without_burning_the_token(
    log: EventLog,
) -> None:
    clock = Clock()
    token = _grant(log, clock, scope="db.switch", target="Local Database")
    with pytest.raises(approvals.ApprovalScopeMismatch):
        approvals.consume(log, token.token_id, "cloud.restore", "Local Database", NODE, clock=clock)
    with pytest.raises(approvals.ApprovalTargetMismatch):
        approvals.consume(log, token.token_id, "db.switch", "Other Database", NODE, clock=clock)
    with pytest.raises(approvals.ApprovalNotFound):
        approvals.consume(log, "apv_missing", "db.switch", "Local Database", NODE, clock=clock)
    assert _refusal_codes(log) == ["wrong_scope", "wrong_target", "not_found"]
    # The refused attempts did not use it up.
    approvals.consume(log, token.token_id, "db.switch", "Local Database", NODE, clock=clock)


def test_consume_of_a_used_token_cannot_be_revoked(log: EventLog) -> None:
    clock = Clock()
    token = _grant(log, clock)
    approvals.consume(log, token.token_id, "project.write", TARGET, NODE, clock=clock)
    with pytest.raises(approvals.ApprovalAlreadyConsumed):
        approvals.revoke(log, token.token_id, "too late", HUMAN, clock=clock)


def test_a_grant_forged_onto_the_log_by_an_agent_is_not_a_token(log: EventLog) -> None:
    clock = Clock()
    log.append(
        approvals.EVENT_GRANTED,
        AGENT,
        {
            "token_id": "apv_forged",
            "scope": "db.switch",
            "target": "Local Database",
            "reason": "trust me",
            "ttl_seconds": 900.0,
            "granted_at": T0.isoformat(),
            "expires_at": (T0 + timedelta(hours=1)).isoformat(),
        },
    )
    assert approvals.list_approvals(log) == []
    with pytest.raises(approvals.ApprovalNotFound):
        approvals.consume(log, "apv_forged", "db.switch", "Local Database", AGENT, clock=clock)


_WORKER = r"""
import json, sys, time
from pathlib import Path
from aar.hub import approvals
from aar.hub.events import Actor, EventLog

db, token, go, ready, name = sys.argv[1:6]
log = EventLog(db, create=False)
Path(ready).touch()
deadline = time.monotonic() + 30
while not Path(go).exists():
    if time.monotonic() > deadline:
        print(json.dumps({"result": "barrier-timeout"}))
        sys.exit(1)
    time.sleep(0.0005)
try:
    approvals.consume(log, token, "db.switch", "Local Database", Actor("node", name))
    print(json.dumps({"result": "ok"}))
except approvals.ApprovalError as exc:
    print(json.dumps({"result": exc.code}))
"""


def test_two_process_consume_race_has_exactly_one_winner(tmp_path: Path) -> None:
    db = tmp_path / "hub.db"
    log = EventLog(db)
    token = approvals.grant(
        log, "db.switch", "Local Database", timedelta(minutes=5), "sandbox switch", HUMAN
    )
    workers = 6
    go = tmp_path / "go"
    procs = []
    for i in range(workers):
        ready = tmp_path / f"ready-{i}"
        argv = [sys.executable, "-c", _WORKER, str(db), token.token_id, str(go), str(ready)]
        procs.append(
            (
                ready,
                subprocess.Popen(
                    [*argv, f"node-{i}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ.copy(),
                ),
            )
        )
    try:
        deadline = time.monotonic() + 60
        while not all(ready.exists() for ready, _ in procs):
            failed = [p for _, p in procs if p.poll() is not None]
            assert not failed, failed[0].communicate()[1]
            assert time.monotonic() < deadline, "workers never became ready"
            time.sleep(0.01)
        go.touch()  # release every worker at once
        results = []
        for _, proc in procs:
            out, err = proc.communicate(timeout=60)
            assert proc.returncode == 0, err
            results.append(json.loads(out.strip().splitlines()[-1])["result"])
    finally:
        for _, proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    assert sorted(results) == ["already_consumed"] * (workers - 1) + ["ok"]
    assert len(log.read(types=[approvals.EVENT_CONSUMED])) == 1
    assert _refusal_codes(log) == ["already_consumed"] * (workers - 1)
