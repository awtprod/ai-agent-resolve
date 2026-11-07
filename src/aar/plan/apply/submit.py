"""Turn one phase's compiled steps into a validated ``ops.sequence`` submission.

A submission is refused before the hub is ever touched if any step still carries an unresolved
reference (:func:`~aar.plan.apply.refs.find_unresolved`) -- no placeholder is ever queued (F1).
The idempotency key is the plan hash, the phase, and the digests of the phase's input results,
so re-running a phase after an upstream redo is a new key, never a conflict (F2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from aar.node.jobs.kinds import OpSequenceParams
from aar.node.ops import get_op
from aar.plan.apply.compile import plan_hash
from aar.plan.apply.errors import CompileError
from aar.plan.apply.model import Step
from aar.plan.apply.refs import find_unresolved
from aar.plan.model import EditPlan

KIND = "ops.sequence"
_PHASES = ("prepare", "assemble", "finish")


@dataclass
class Submission:
    kind: str
    params: dict[str, Any]
    idempotency_key: str
    phase: str
    warnings: list[str] = field(default_factory=list)


def build_submission(
    plan: EditPlan, phase: str, steps: list[Step], *, input_hashes: list[str] | None = None
) -> Submission:
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES}, not {phase!r}")
    if not steps:
        raise ValueError("build_submission needs at least one step")

    _refuse_unresolved(steps)

    spec_steps = [{"op": s.op, "name": s.name, "params": s.params} for s in steps]
    params = {"steps": spec_steps}
    _validate(params)

    digest = hashlib.sha256(
        "\n".join([plan_hash(plan), phase, *(input_hashes or [])]).encode("utf-8")
    ).hexdigest()
    return Submission(KIND, params, f"plan.apply:{digest}", phase)


def _refuse_unresolved(steps: list[Step]) -> None:
    found: list[dict[str, Any]] = []
    for step in steps:
        for path, inner in find_unresolved(step.params):
            found.append({"step": step.name, "op": step.op, "at": path, **inner})
    if found:
        where = "; ".join(f"{f['step']}.{f['at']} ({f['kind']}: {f['hint']})" for f in found)
        raise CompileError(
            "unresolved_reference",
            f"the payload has {len(found)} unresolved reference(s): {where}",
            "compile this phase from the real result files of the earlier phases",
            unresolved=found,
        )


def _validate(params: dict[str, Any]) -> None:
    """The envelope and per-step op-params checks ``aar hub job submit`` runs before queueing."""
    envelope = OpSequenceParams.model_validate(params)
    for index, spec in enumerate(envelope.steps):
        try:
            op = get_op(spec.op)
        except KeyError as exc:
            raise CompileError(
                "phase_result_mismatch",
                f"step {index} names unknown op {spec.op!r}",
                "every compiled step must name a registered op",
            ) from exc
        op.params.model_validate(spec.params)
