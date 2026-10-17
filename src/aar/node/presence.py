"""Is a human working in Resolve right now?

``PresenceGuard`` is fed timestamped context observations (polls of
``ResolveGateway.get_context``) and the broker's own actions. A change of working position
that the broker did not cause counts as human activity. So does recent keyboard or mouse
input, when an OS input-idle provider is available (``GetLastInputInfo`` on Windows).

It is deliberately conservative: until the context has been watched and found quiet for a
whole idle window, a human is assumed to be present.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from aar.node.gateway import ResolveContext

InputIdleProvider = Callable[[], float | None]
"""Seconds since the last keyboard/mouse input, or ``None`` when the OS cannot tell."""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: str
    reason: str


class PresenceGuard:
    """Decides whether the broker may take or keep the instance (switch context, run ops)."""

    def __init__(
        self,
        *,
        idle_window: float = 300.0,
        max_observation_age: float = 30.0,
        input_idle: InputIdleProvider | None = None,
    ) -> None:
        if idle_window <= 0 or max_observation_age <= 0:
            raise ValueError("idle_window and max_observation_age must be positive")
        self.idle_window = idle_window
        self.max_observation_age = max_observation_age
        self._input_idle = input_idle
        self._first_observed_at: float | None = None
        self._last_observed_at: float | None = None
        self._last_seen: ResolveContext | None = None
        self._last_human_change_at: float | None = None
        self._last_human_change: dict[str, tuple[object, object]] = {}

    @property
    def last_human_change_at(self) -> float | None:
        return self._last_human_change_at

    @property
    def last_human_change(self) -> dict[str, tuple[object, object]]:
        """The fields that changed in the last human change, as ``{field: (before, after)}``."""
        return dict(self._last_human_change)

    def human_changed_since(self, t: float) -> bool:
        """Whether a context change not caused by the broker was seen at or after ``t``."""
        return self._last_human_change_at is not None and self._last_human_change_at >= t

    def observe(self, context: ResolveContext, at: float) -> bool:
        """Record a poll. Returns True if it shows a change the broker did not cause.

        The baseline is the last poll or the last ``note_broker_context``, whichever is later,
        so call this from the broker's own worker: a poll taken while an op is half-way
        through would count the op's own changes as human (which only makes the broker back
        off, never act).
        """
        human = False
        previous = self._last_seen
        if previous is not None and not context.same_position(previous):
            human = True
            self._last_human_change_at = at
            self._last_human_change = previous.diff(context)
        if self._first_observed_at is None:
            self._first_observed_at = at
        self._last_seen = context
        self._last_observed_at = at
        return human

    def note_broker_context(self, context: ResolveContext, at: float) -> None:
        """Record a position the broker itself just produced, so it is not mistaken for a human."""
        self._last_seen = context
        if self._first_observed_at is None:
            self._first_observed_at = at
        self._last_observed_at = at

    def may_switch(self, now: float) -> Decision:
        """May the broker change Resolve's context (or act in the human's context) now?"""
        if self._last_observed_at is None or self._first_observed_at is None:
            return Decision(False, "no_observation", "Resolve's context has not been observed")
        age = now - self._last_observed_at
        if age > self.max_observation_age:
            return Decision(
                False,
                "stale_observation",
                f"last context observation is {age:.1f}s old (max {self.max_observation_age}s)",
            )
        if self._last_human_change_at is not None:
            since = now - self._last_human_change_at
            if since < self.idle_window:
                changed = ", ".join(sorted(self._last_human_change)) or "context"
                return Decision(
                    False,
                    "human_context_change",
                    f"human changed {changed} {since:.1f}s ago (idle window {self.idle_window}s)",
                )
        watched = now - self._first_observed_at
        if watched < self.idle_window:
            return Decision(
                False,
                "warming_up",
                f"context watched for {watched:.1f}s, less than the idle window "
                f"({self.idle_window}s)",
            )
        if self._input_idle is not None:
            try:
                idle = self._input_idle()
            except Exception as exc:
                return Decision(False, "input_idle_error", f"input-idle provider failed: {exc!r}")
            if idle is not None and idle < self.idle_window:
                return Decision(
                    False,
                    "human_input",
                    f"keyboard/mouse input {idle:.1f}s ago (idle window {self.idle_window}s)",
                )
        return Decision(True, "idle", f"no human activity for at least {self.idle_window}s")


def idle_seconds_from_ticks(last_input_tick: int, now_tick: int) -> float:
    """Milliseconds-tick difference as seconds, tolerating the 32-bit wrap of ``GetTickCount``."""
    return ((now_tick - last_input_tick) & 0xFFFFFFFF) / 1000.0


def windows_input_idle_seconds() -> float | None:
    """Seconds since the last input in this session via ``GetLastInputInfo``; ``None`` off Windows.

    Only meaningful when the node runs in the interactive user's session: a service in
    session 0 sees no input at all.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)
    windll = ctypes.windll  # type: ignore[attr-defined]
    if not windll.user32.GetLastInputInfo(ctypes.byref(info)):
        raise ctypes.WinError()  # type: ignore[attr-defined]
    windll.kernel32.GetTickCount.restype = wintypes.DWORD
    return idle_seconds_from_ticks(int(info.dwTime), int(windll.kernel32.GetTickCount()))


def default_input_idle_provider() -> InputIdleProvider | None:
    """The OS input-idle provider for this platform, or ``None`` where there is none."""
    return windows_input_idle_seconds if sys.platform == "win32" else None
