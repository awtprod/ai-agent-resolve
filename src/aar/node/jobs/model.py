"""Job data model: states, structured results, steps and the persisted job record.

A job has an id, a kind, typed params and an ordered sequence of :class:`Step`. Each step
runs through the broker (the op library), never against Resolve directly. A step may take
several broker ops and span several runner pumps; its progress lives in a per-step ``cursor``
that is checkpointed, so a step resumes exactly where it was after a process restart.

Every result an agent sees carries measured values, and every error is structured with a fix
hint and (when known) the trap that caused it (DESIGN R5). ``StructuredError`` and
``WaitReason`` are those shapes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from aar.node.broker import OpError as BrokerOpError
from aar.node.broker import OpResult, OpStatus
from aar.node.locator import ProjectLocator


class JobState(StrEnum):
    """The lifecycle states a job moves through. Every transition is an event."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    STALLED = "stalled"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.FAILED, JobState.SUCCEEDED, JobState.CANCELLED})
"""States a job never leaves."""


@dataclass(frozen=True, slots=True)
class StructuredError:
    """A machine-readable failure: what failed, how to fix it, and the known trap behind it.

    This is DESIGN R5's error shape, aligned with the op library's ``OpError``.
    """

    code: str
    message: str
    fix_hint: str = ""
    trap_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "trap_id": self.trap_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredError:
        return cls(
            code=data["code"],
            message=data["message"],
            fix_hint=data.get("fix_hint", ""),
            trap_id=data.get("trap_id"),
            detail=dict(data.get("detail", {})),
        )

    @classmethod
    def from_broker(cls, error: BrokerOpError) -> StructuredError:
        """Build from a broker ``OpResult.error``. The op library stuffs ``fix_hint`` and
        ``trap_id`` into the error detail; lift them back out into first-class fields."""
        detail = dict(error.detail)
        fix_hint = detail.pop("fix_hint", "")
        trap_id = detail.pop("trap_id", None)
        return cls(
            code=error.code,
            message=error.message,
            fix_hint=fix_hint if isinstance(fix_hint, str) else "",
            trap_id=trap_id if isinstance(trap_id, str) else None,
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class WaitReason:
    """Why a job is paused and what is needed to continue it.

    ``kind`` is ``stop`` (an emergency stop) or ``needs_approval`` (a human token is needed).
    For ``needs_approval`` the detail names the exact scope and target to grant, so an agent
    can tell a human precisely what to approve.
    """

    kind: str
    code: str
    message: str
    fix_hint: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "code": self.code,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaitReason:
        return cls(
            kind=data["kind"],
            code=data["code"],
            message=data["message"],
            fix_hint=data.get("fix_hint", ""),
            detail=dict(data.get("detail", {})),
        )


# -- op outcomes and step actions --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpOutcome:
    """The result of the op a step most recently submitted, as the step sees it."""

    ok: bool
    value: BaseModel | None
    error: StructuredError | None

    def value_dict(self) -> dict[str, Any]:
        return self.value.model_dump(mode="json") if self.value is not None else {}

    @classmethod
    def from_result(cls, result: OpResult) -> OpOutcome:
        if result.status is OpStatus.OK:
            value = result.value if isinstance(result.value, BaseModel) else None
            return cls(True, value, None)
        error = (
            StructuredError.from_broker(result.error)
            if result.error is not None
            else StructuredError("op_failed", "the op failed without an error", "")
        )
        return cls(False, None, error)


@dataclass(frozen=True, slots=True)
class Submit:
    """Submit op ``op`` with ``params`` through the broker and wait for its result.

    ``key`` labels the op within the step, for the event log only.
    """

    op: str
    params: dict[str, Any] | BaseModel
    key: str = "op"


@dataclass(frozen=True, slots=True)
class Done:
    """The step is finished; ``value`` is its measured result."""

    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Fail:
    """The step failed; the job fails with this structured error."""

    error: StructuredError


@dataclass(frozen=True, slots=True)
class Pause:
    """The step made no progress this pump and should be re-tried next pump.

    Used for an op that is still working (for example a render still rendering). The job stays
    ``running``: unlike a :class:`WaitReason`, nothing external is needed to continue.
    """

    code: str
    message: str
    progress: Any = None
    """An observable progress mark (JSON-serialisable), e.g. a render's status and percent.
    When it changes, the step demonstrably moved on, so the runner forgets the node restarts
    it counted against this step (a healthy long render is never a crash loop)."""


StepAction = Submit | Done | Fail | Pause


@dataclass
class StepContext:
    """What a step sees when it is advanced.

    ``cursor`` is the step's own persisted state; a step mutates it in place and the runner
    checkpoints it. ``outcome`` is the result of the op the step last submitted, or ``None``
    when the step has no op outstanding (its first advance, or after a :class:`Pause`).
    """

    job_id: str
    params: BaseModel
    target: ProjectLocator
    prior: list[dict[str, Any]]
    cursor: dict[str, Any]
    outcome: OpOutcome | None


class Step(ABC):
    """One unit of a job. Runs only through the broker; drives itself from its ``cursor``."""

    name: str
    timeout: float
    """Wall-clock seconds the step may take before the watchdog calls it stalled."""
    renders_project: bool = False
    """True for a step that drives a render; the runner serialises these per project."""

    @abstractmethod
    def advance(self, ctx: StepContext) -> StepAction:
        """Decide the next action from the cursor and the last op outcome."""
        ...


__all__ = [
    "TERMINAL_STATES",
    "Done",
    "Fail",
    "JobState",
    "OpOutcome",
    "Pause",
    "Step",
    "StepAction",
    "StepContext",
    "StructuredError",
    "Submit",
    "WaitReason",
]
