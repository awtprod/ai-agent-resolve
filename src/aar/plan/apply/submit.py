"""Turn compiled steps into an ``ops.sequence`` job submission (the ``--submit`` path).

This builds and validates the job params exactly as ``aar hub job submit`` would -- the same
``OpSequenceParams`` envelope and the same per-step op-params validation -- and derives the
idempotency key from the plan hash plus the phase (decision 6). Actually appending it to a hub
event log is the CLI's job, behind the hub flags; this module stays pure so it is easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aar.node.jobs.kinds import OpSequenceParams
from aar.node.ops import get_op
from aar.plan.apply.compile import BranchRef, plan_hash
from aar.plan.apply.errors import CompileError
from aar.plan.apply.model import Step
from aar.plan.model import EditPlan

KIND = "ops.sequence"


@dataclass
class Submission:
    """A validated ``ops.sequence`` submission: feed ``kind``/``params``/``idempotency_key``
    straight to ``aar.hub.jobs.submit``."""

    kind: str
    params: dict[str, Any]
    idempotency_key: str
    phase: str
    warnings: list[str] = field(default_factory=list)


def build_submission(
    plan: EditPlan,
    phase: str,
    steps: list[Step],
    *,
    template_described: bool = False,
    branch: BranchRef | None = None,
) -> Submission:
    """Build (and validate) the ops.sequence job for one phase.

    Phase A refuses without a ``--template-describe`` (decision 3): ``timeline.branch`` must know
    the template is small and needs its ``unique_id``. Phase B refuses with no steps, i.e. no
    phase-A result to resolve item ids from (decision 6)."""
    if phase not in ("A", "B"):
        raise ValueError(f"phase must be 'A' or 'B', not {phase!r}")
    if phase == "A" and not template_described:
        raise CompileError(
            "template_size_unchecked",
            "cannot submit phase A without --template-describe",
            "run timeline.describe on the template and pass it as --template-describe; "
            "timeline.branch needs the template's unique_id and a size check",
        )
    if not steps:
        raise ValueError("build_submission needs at least one step")

    spec_steps = [{"op": step.op, "name": step.name, "params": step.params} for step in steps]
    params = {"steps": spec_steps}
    _validate(params)

    warnings: list[str] = []
    if phase == "A" and branch is None:
        warnings.append(
            "phase A names the working timeline by placeholder: a correct run needs the branch "
            "identity threaded into the track/append steps (ops.sequence does not thread results)"
        )

    return Submission(
        kind=KIND,
        params=params,
        idempotency_key=f"plan.apply:{plan_hash(plan)}:{phase}",
        phase=phase,
        warnings=warnings,
    )


def _validate(params: dict[str, Any]) -> None:
    """The envelope and per-step op-params checks ``aar hub job submit`` runs before queueing."""
    envelope = OpSequenceParams.model_validate(params)
    for index, spec in enumerate(envelope.steps):
        try:
            op = get_op(spec.op)
        except KeyError as exc:
            raise CompileError(
                "phase_a_result_mismatch",
                f"step {index} names unknown op {spec.op!r}",
                "every compiled step must name a registered op",
            ) from exc
        op.params.model_validate(spec.params)
