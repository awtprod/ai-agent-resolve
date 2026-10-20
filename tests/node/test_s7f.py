"""S7f: follow-ups to the S7 x S11 integration validation (MED-1, LOW-2..8).

No Windows is available: the Windows input-idle wiring is driven through a fake
``ctypes.windll`` (with ``sys.platform`` patched), never a real Win32 call.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

import pytest

from aar.node import presence
from aar.node.presence import (
    NO_CONSOLE_SESSION,
    InputIdleUnavailable,
    PresenceGuard,
    session_refusal,
)

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
