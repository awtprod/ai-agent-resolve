"""The job watchdog: notice a job that has stopped making progress and stall it.

A job is stalled when its current op has not returned within ``op_timeout`` (a hung gateway
call, or a project the broker cannot get into), or its current step has run past its own
deadline. The watchdog only *reports*: it marks the job stalled with a clear reason and the
runner stops scheduling that project. It never kills or restarts Resolve or any process it did
not start (DESIGN §6); recovery is a human or an explicit agent action.

Time is injected (the runner's monotonic clock), so tests are deterministic and nothing
sleeps.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import StructuredError


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    op_timeout: float = 120.0
    """Seconds a single op (gateway call) may be outstanding before the job is stalled."""

    def __post_init__(self) -> None:
        if self.op_timeout <= 0:
            raise ValueError("op_timeout must be positive")


class Watchdog:
    """Decides, from timestamps and ``now``, whether a job's current work is overdue."""

    def __init__(self, config: WatchdogConfig | None = None) -> None:
        self.config = config or WatchdogConfig()

    def overdue(
        self,
        *,
        now: float,
        step_started_at: float | None,
        step_timeout: float,
        op_submitted_at: float | None,
    ) -> StructuredError | None:
        """Return why the job is stalled, or ``None`` if it is still within its deadlines."""
        if op_submitted_at is not None:
            age = now - op_submitted_at
            if age > self.config.op_timeout:
                return StructuredError(
                    "watchdog_op_timeout",
                    f"the current op has not returned in {age:.1f}s (limit "
                    f"{self.config.op_timeout:.0f}s); the project is not making progress",
                    "check the Resolve node by hand: it may be hung or a human may be holding "
                    "the project. The watchdog never touches Resolve; resume the job with "
                    "resume(job_id) once the node is healthy.",
                    detail={"op_age_seconds": age, "op_timeout": self.config.op_timeout},
                )
        if step_started_at is not None and step_timeout > 0:
            age = now - step_started_at
            if age > step_timeout:
                return StructuredError(
                    "watchdog_step_timeout",
                    f"the current step has not finished in {age:.1f}s (limit {step_timeout:.0f}s)",
                    "check the render or node by hand; the watchdog never touches Resolve. "
                    "Resume the job with resume(job_id) once it is healthy.",
                    detail={"step_age_seconds": age, "step_timeout": step_timeout},
                )
        return None
