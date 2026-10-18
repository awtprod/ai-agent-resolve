"""The node job runner: checkpointed, resumable jobs over the session broker.

A job has an id, a kind, typed params and a sequence of steps; each step runs through the op
library, never against Resolve directly. State is checkpointed to a node-side SQLite store
that survives a process restart, so a fresh runner resumes every job from its last checkpoint.
The runner wires in the hub's emergency stop and the F15 approval gate, and a watchdog stalls
a job that stops making progress.

Importing this package registers the built-in job kinds (``ops.sequence`` and ``render``).
"""

from __future__ import annotations

from .kinds import (
    DEFAULT_STEP_TIMEOUT,
    JobKind,
    OpSequenceParams,
    OpStep,
    OpStepSpec,
    get_kind,
    list_kinds,
    register_kind,
)
from .model import (
    JobState,
    StructuredError,
    WaitReason,
)
from .render import DEFAULT_RENDER_TIMEOUT, RenderParams, RenderStep, _build_render
from .runner import ApprovalPolicy, JobRunner, JobView, PumpOutcome
from .store import (
    SCHEMA_VERSION,
    JobEvent,
    JobRecord,
    JobStore,
    JobStoreLocked,
    JobStoreVersionError,
)
from .watchdog import Watchdog, WatchdogConfig

# Register the render kind here, so render.py need not import the kind registry.
RENDER_KIND = register_kind(JobKind("render", RenderParams, _build_render))

__all__ = [
    "DEFAULT_RENDER_TIMEOUT",
    "DEFAULT_STEP_TIMEOUT",
    "RENDER_KIND",
    "SCHEMA_VERSION",
    "ApprovalPolicy",
    "JobEvent",
    "JobKind",
    "JobRecord",
    "JobRunner",
    "JobState",
    "JobStore",
    "JobStoreLocked",
    "JobStoreVersionError",
    "JobView",
    "OpSequenceParams",
    "OpStep",
    "OpStepSpec",
    "PumpOutcome",
    "RenderParams",
    "RenderStep",
    "StructuredError",
    "WaitReason",
    "Watchdog",
    "WatchdogConfig",
    "get_kind",
    "list_kinds",
    "register_kind",
]
