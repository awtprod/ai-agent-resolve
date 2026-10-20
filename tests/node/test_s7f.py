"""S7f: follow-ups to the S7 x S11 integration validation (MED-1, LOW-2..8).

No Windows is available: the Windows input-idle wiring is driven through a fake
``ctypes.windll`` (with ``sys.platform`` patched), never a real Win32 call.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

import pytest

from aar.hub.approvals import ApprovalScope, grant
from aar.hub.events import Actor
from aar.node import presence
from aar.node.jobs import ApprovalPolicy
from aar.node.presence import (
    NO_CONSOLE_SESSION,
    InputIdleUnavailable,
    PresenceGuard,
    session_refusal,
)

from .test_s7_s11_integration import _human_world, _marker_job, _op_grant

# == MED-1: input idle fails closed outside the active console session ==========================


@pytest.mark.parametrize(
    ("process", "console", "allowed"),
    [
        (1, 1, True),  # the node runs in the active console session
        (2, 2, True),
        (0, 0, False),  # session 0 (a service) even if the console reports 0
        (0, 1, False),  # a service; the human is on console session 1
        (2, 1, False),  # another session (Remote Desktop / another user)
        (1, NO_CONSOLE_SESSION, False),  # nobody attached to the console
        (NO_CONSOLE_SESSION, NO_CONSOLE_SESSION, False),  # 0xFFFFFFFF is never a match
        (None, 1, False),  # ProcessIdToSessionId failed
        (1, None, False),  # WTSGetActiveConsoleSessionId failed
        (None, None, False),
    ],
)
def test_med1_session_decision_table(process, console, allowed: bool) -> None:
    refusal = session_refusal(process, console)
    assert (refusal is None) is allowed
    if not allowed:
        assert isinstance(refusal, str) and refusal


class _Fn:
    """A fake Win32 function: callable, with a settable ``restype``."""

    def __init__(self, impl: Any) -> None:
        self._impl = impl
        self.restype = None
        self.calls = 0

    def __call__(self, *args: Any) -> Any:
        self.calls += 1
        return self._impl(*args)


class _FakeWindll:
    """``ctypes.windll`` with kernel32/user32 reporting the given sessions and input time."""

    def __init__(
        self,
        *,
        process_session: int | None,
        console_session: int | None,
        last_input_tick: int = 1_000,
        now_tick: int = 4_000,
    ) -> None:
        def process_id_to_session_id(pid: int, ref: Any) -> int:
            if process_session is None:
                return 0  # the call failed
            ref._obj.value = process_session
            return 1

        def console() -> int:
            if console_session is None:
                raise OSError("WTSGetActiveConsoleSessionId failed")
            return console_session

        def last_input(ref: Any) -> int:
            ref._obj.dwTime = last_input_tick
            return 1

        self.kernel32 = type("k32", (), {})()
        self.kernel32.GetCurrentProcessId = _Fn(lambda: 4242)
        self.kernel32.ProcessIdToSessionId = _Fn(process_id_to_session_id)
        self.kernel32.WTSGetActiveConsoleSessionId = _Fn(console)
        self.kernel32.GetTickCount = _Fn(lambda: now_tick)
        self.user32 = type("u32", (), {})()
        self.user32.GetLastInputInfo = _Fn(last_input)


def _on_windows(monkeypatch: pytest.MonkeyPatch, windll: _FakeWindll) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda *a: OSError("WinError"), raising=False)


def test_med1_the_console_session_reads_the_real_idle_time(monkeypatch) -> None:
    windll = _FakeWindll(process_session=1, console_session=1)
    _on_windows(monkeypatch, windll)
    assert presence.windows_input_idle_seconds() == 3.0
    assert windll.user32.GetLastInputInfo.calls == 1


@pytest.mark.parametrize(
    ("process", "console", "said"),
    [
        (0, 1, "session 0"),
        (2, 1, "not the active console session"),
        (1, NO_CONSOLE_SESSION, "no session is attached"),
        (None, 1, "ProcessIdToSessionId failed"),
        (1, None, "WTSGetActiveConsoleSessionId failed"),
    ],
)
def test_med1_outside_the_console_session_it_refuses_before_reading_input(
    monkeypatch, process, console, said: str
) -> None:
    """The service case reports a huge, false idle (here 3 s; the point is it is never read)."""
    windll = _FakeWindll(process_session=process, console_session=console)
    _on_windows(monkeypatch, windll)
    with pytest.raises(InputIdleUnavailable) as err:
        presence.windows_input_idle_seconds()
    assert said in str(err.value)
    assert "interactive user's session" in str(err.value) and "not a service" in str(err.value)
    assert windll.user32.GetLastInputInfo.calls == 0


def test_med1_the_guard_refuses_to_switch_for_a_node_in_session_0(monkeypatch) -> None:
    """Through the OS default provider: a quiet context and an elapsed idle window, but the
    node is a service in session 0 -- may_switch waits with a clear reason."""
    windll = _FakeWindll(process_session=0, console_session=1, last_input_tick=0, now_tick=10**9)
    _on_windows(monkeypatch, windll)
    provider = presence.default_input_idle_provider()
    assert provider is presence.windows_input_idle_seconds
    from aar.node.gateway import ScriptApiGateway

    from .fake_resolve import FakeResolve

    ctx = ScriptApiGateway(FakeResolve({"Show": ["Main"]}, current="Show")).get_context()
    guard = PresenceGuard(idle_window=60.0, input_idle=provider)
    guard.observe(ctx, 0.0)
    guard.observe(ctx, 60.0)
    decision = guard.may_switch(60.0)
    assert not decision.allowed and decision.code == "input_idle_error"
    assert decision.reason.startswith("input idle unavailable: the node runs in session 0")
    assert "logon task, not a service" in decision.reason


# == LOW-2: an override approval licenses its own op only, never the F15 gate for others ==========

HUMAN = Actor("human", "andrew")
PROD = ApprovalPolicy(live=True)


def _project_grant(world) -> str:
    return grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "ok", HUMAN).token_id


def _consumes(world) -> int:
    return len(world.log.read(types=["approval.consumed"]))


def _guarded_then_own_branch(world) -> dict:
    """Step 0 edits the human's "Main" (guarded: needs its override); step 1 edits the
    agent's own branch (guard-allowed, but a production write)."""
    world.own("Main [agent]")
    own = world.ref("Main [agent]")
    return {
        "steps": [
            *_marker_job(100)["steps"],
            *_marker_job(200, timeline=own)["steps"],
        ]
    }


def test_low2a_the_overridden_op_runs_on_its_override_alone(tmp_path: Path) -> None:
    world = _human_world(tmp_path, approval=PROD)
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    assert world.runner.get(job_id).waiting["detail"]["key"].startswith("op:")
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert sorted(world.main.markers) == [100] and _consumes(world) == 1


@pytest.mark.parametrize("restart", [False, True])
def test_low2b_a_later_guard_allowed_write_waits_for_the_project_approval(
    tmp_path: Path, restart: bool
) -> None:
    world = _human_world(tmp_path, approval=PROD)
    job_id = world.submit("ops.sequence", _guarded_then_own_branch(world), world.locator())
    world.runner.run_until_settled()
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    world.runner.run_until_settled()
    if restart:
        world.restart(approval=PROD)
        world.own("Main [agent]")  # (the in-memory branch registry is rebuilt by timeline.branch)
        world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert sorted(world.main.markers) == [100]  # the overridden op ran on its override
    assert view.state == "waiting" and view.current_step == 1
    assert view.waiting["detail"]["key"] == "project"
    assert view.waiting["detail"]["target"] == world.pid()
    assert world.project.timeline("Main [agent]").markers == {}
    world.runner.approve(job_id, _project_grant(world), HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert sorted(world.project.timeline("Main [agent]").markers) == [200]
    assert _consumes(world) == 2


@pytest.mark.parametrize("restart", [False, True])
def test_low2c_the_project_approval_licenses_the_jobs_later_writes_once(
    tmp_path: Path, restart: bool
) -> None:
    world = _human_world(tmp_path, approval=PROD)
    world.own("Main [agent]")
    own = world.ref("Main [agent]")
    job_id = world.submit("ops.sequence", _marker_job(1, 2, 3, timeline=own), world.locator())
    world.runner.run_until_settled()
    world.runner.approve(job_id, _project_grant(world), HUMAN)
    world.runner.pump()
    world.runner.pump()  # step 0 written; steps 1-2 to go
    if restart:
        world.restart(approval=PROD)
        world.own("Main [agent]")
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert sorted(world.project.timeline("Main [agent]").markers) == [1, 2, 3]
    assert _consumes(world) == 1  # once per job, as before


def test_low2d_restart_mid_override_keeps_it_for_its_op_only(tmp_path: Path) -> None:
    """The override is consumed and its op queued when the node dies: after the restart the
    op runs on the held override (no new grant), and the next write still needs the project
    approval."""
    world = _human_world(tmp_path, approval=PROD)
    job_id = world.submit("ops.sequence", _guarded_then_own_branch(world), world.locator())
    world.runner.run_until_settled()
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    world.runner.pump()  # consume + submit; the node dies before the op ran
    assert world.main.markers == {} and _consumes(world) == 1
    world.restart(approval=PROD)
    world.own("Main [agent]")
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert sorted(world.main.markers) == [100] and _consumes(world) == 1
    assert view.state == "waiting" and view.waiting["detail"]["key"] == "project"


# == LOW-8: crash recovery through an op-override consume ========================================


class _Crash(Exception):
    pass


@pytest.mark.parametrize("production", [False, True])
def test_low8_a_crash_between_the_override_consume_and_its_checkpoint_is_recovered(
    tmp_path: Path, production: bool
) -> None:
    """The hub consumed the override token, then the node died before ``_hold_approval``
    checkpointed it. On restart the persisted pending intent (token AND its approval key) is
    recovered under the override's own key; the op runs exactly once with no new grant. On
    a production project the recovered override still licenses only its own op."""
    policy = ApprovalPolicy(live=production)
    world = _human_world(tmp_path, approval=policy)
    job = _guarded_then_own_branch(world) if production else _marker_job(100)
    job_id = world.submit("ops.sequence", job, world.locator())
    world.runner.run_until_settled()
    token = _op_grant(world, "markers.add")
    world.runner.approve(job_id, token, HUMAN)
    real_save = world.store.save

    def save(record, events=()):
        events = list(events)
        if any(kind == "approval_consumed" for kind, _, _ in events):
            raise _Crash("power cut after the hub consume committed")
        return real_save(record, events)

    world.store.save = save  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        world.runner.pump()
    stored = world.store.get(job_id)
    key = "op:0/0:markers.add/markers.add"
    assert (stored.approval_pending, stored.approval_pending_key) == (token, key)
    assert _consumes(world) == 1 and world.main.markers == {}
    world.restart(approval=policy)
    if production:
        world.own("Main [agent]")
    recovered = world.store.get(job_id)
    assert recovered.approvals == {key: token} and recovered.approval_pending is None
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert sorted(world.main.markers) == [100]  # exactly once
    assert _consumes(world) == 1  # no new grant was needed or consumed
    events = [e for e in view.events if e["kind"] == "approval_recovered"]
    assert len(events) == 1 and events[0]["data"]["key"] == key
    if production:
        assert view.state == "waiting" and view.waiting["detail"]["key"] == "project"
    else:
        assert view.state == "succeeded"
