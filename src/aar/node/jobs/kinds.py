"""Job kinds and the simplest step: one verified op per step.

A :class:`JobKind` binds a name to a params model and a ``build`` that turns validated params
and the job's target into an ordered list of :class:`~aar.node.jobs.model.Step`. The runner
looks a kind up by name, validates params against its model, and builds the steps once.

``ops.sequence`` is the general kind: a list of ops, each run as one :class:`OpStep`. Because
the op library's ops are idempotent (markers already there count as present, a duplicate is
reused, batch ops take a resume cursor), re-running a step after a restart finds its earlier
work instead of doing it twice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aar.node.locator import ProjectLocator
from aar.node.ops import get_op

from .model import Done, Fail, Step, StepAction, StepContext, Submit

DEFAULT_STEP_TIMEOUT = 300.0
"""A step should finish well under the live MCP 60 s cap per op; a generous whole-step limit."""

RENDER_KIND_ONLY_OPS = frozenset({"render.build", "render.delete_job"})
"""Ops only the ``render`` job kind may run: as a bare op step, a re-run build would add a
second render job (the step would not adopt the journaled one) and ``render.delete_job`` would
delete any JobId it is given, a human's included."""


@dataclass(frozen=True, slots=True)
class JobKind:
    """A named job type: how to validate its params and build its steps."""

    name: str
    params_model: type[BaseModel]
    build: Callable[[BaseModel, ProjectLocator], list[Step]]


_KINDS: dict[str, JobKind] = {}


def register_kind(kind: JobKind) -> JobKind:
    if kind.name in _KINDS:
        raise ValueError(f"job kind {kind.name!r} is already registered")
    _KINDS[kind.name] = kind
    return kind


def get_kind(name: str) -> JobKind:
    try:
        return _KINDS[name]
    except KeyError:
        raise KeyError(f"no job kind {name!r}; known: {sorted(_KINDS)}") from None


def list_kinds() -> list[JobKind]:
    return [_KINDS[name] for name in sorted(_KINDS)]


class OpStep(Step):
    """One op. It submits the op, then reports its measured value or its structured error."""

    def __init__(
        self,
        op: str,
        params: dict[str, Any],
        *,
        name: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.op = op
        self.params = params
        self.name = name or op
        self.timeout = timeout if timeout is not None else DEFAULT_STEP_TIMEOUT

    def advance(self, ctx: StepContext) -> StepAction:
        if ctx.outcome is None:
            return Submit(self.op, self.params, key=self.name)
        if ctx.outcome.ok:
            return Done(ctx.outcome.value_dict())
        assert ctx.outcome.error is not None
        return Fail(ctx.outcome.error)


# -- ops.sequence ------------------------------------------------------------------------------


class OpStepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class OpSequenceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[OpStepSpec] = Field(min_length=1, max_length=500)


def _build_op_sequence(params: BaseModel, target: ProjectLocator) -> list[Step]:
    assert isinstance(params, OpSequenceParams)
    steps: list[Step] = []
    for index, spec in enumerate(params.steps):
        try:
            get_op(spec.op)  # fail at submit, not mid-job, on an unknown op
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if spec.op in RENDER_KIND_ONLY_OPS:
            raise ValueError(f"op {spec.op!r} runs only inside a 'render' job, not ops.sequence")
        steps.append(
            OpStep(
                spec.op, spec.params, name=spec.name or f"{index}:{spec.op}", timeout=spec.timeout
            )
        )
    return steps


OPS_SEQUENCE = register_kind(JobKind("ops.sequence", OpSequenceParams, _build_op_sequence))
