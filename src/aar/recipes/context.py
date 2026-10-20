"""What a cold step is given to do its work: :class:`StepContext`.

The engine hands a cold callable a :class:`StepContext`: the recipe's typed params and brief, the
outputs of the steps it needs, an artifact store (``put``/``get``), a facts writer that attributes
to the running op node, a deadline, and a ``cancelled()`` check. A callable returns a
:class:`StepOutput` (bounded JSON outputs plus any artifact ids it produced) or raises
:class:`StepFailure` to fail the step with a typed error. Any other exception is a failure too,
recorded with code ``error``.

Timeouts are cooperative. The context carries a ``deadline``; a well-behaved callable checks
``cancelled()`` and ``time_left()`` and stops. An in-process step cannot be killed: if the callable
runs past its deadline, the engine records a ``timeout`` failure when it finally returns, but it
could not have interrupted it. Only a separate process (the hot node, S15b) can be hard-killed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from os import PathLike
from typing import Any

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.facts import record_fact
from aar.hub.writes import Brief
from aar.plan.profile import Profile


class StepFailure(Exception):
    """A controlled step failure: a stable ``code``, a ``message`` and a ``fix_hint``."""

    def __init__(self, code: str, message: str, fix_hint: str = "") -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.fix_hint = fix_hint

    def to_error(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "fix_hint": self.fix_hint}


@dataclass(frozen=True, slots=True)
class StepOutput:
    """What a cold callable returns: bounded JSON outputs and any produced artifact ids."""

    outputs: dict[str, Any] = field(default_factory=dict)
    artifact_ids: tuple[str, ...] = ()


@dataclass
class StepContext:
    """Everything a cold step may read and the few things it may write.

    ``params`` and ``brief`` are the recipe's typed params object and its brief. ``needs`` maps each
    needed step id to that step's recorded outputs. ``put``/``get`` are the artifact store.
    ``record_fact`` writes a project fact sourced to the recipe node. ``deadline`` is when the step
    must be done; ``cancelled()`` is True once a stop was requested.
    """

    log: EventLog
    store: ArtifactStore
    project_id: str
    recipe_node: str
    op_node: str
    actor: Actor
    params: Any
    brief: Brief | None
    profile: Profile | None
    needs: Mapping[str, Mapping[str, Any]]
    deadline: datetime
    _cancelled: Callable[[], bool]
    _clock: Callable[[], datetime]

    def cancelled(self) -> bool:
        """Whether a stop has been requested (shutdown, or the op cancelled)."""
        return bool(self._cancelled())

    def time_left(self) -> float:
        """Seconds until the deadline (negative once it has passed)."""
        return (self.deadline - self._clock()).total_seconds()

    def put(
        self,
        source: bytes | bytearray | str | PathLike[str],
        *,
        mime: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Store ``source`` and return its artifact id; the op node is recorded as producer."""
        return self.store.put(
            source,
            actor=self.actor,
            mime=mime,
            project_id=self.project_id,
            producing_op=self.op_node,
            meta=meta,
        )

    def get(self, artifact_id: str) -> bytes:
        """Read a stored artifact's bytes (verified against its hash)."""
        return self.store.read_bytes(artifact_id)

    def record_fact(self, key: str, value: Mapping[str, Any]) -> str:
        """Record a project fact sourced to the recipe node; returns the fact key.

        Idempotent per ``(op_node, key)`` so a re-run of an idempotent step does not conflict.
        """
        idem = f"recipe-fact:{self.op_node}:{key}"
        record_fact(
            self.log,
            self.project_id,
            key,
            dict(value),
            self.actor,
            idempotency_key=idem,
            source={"recipe_node": self.recipe_node},
        )
        return key
