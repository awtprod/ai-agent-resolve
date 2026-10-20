"""Recipe gates: pass/fail decisions read from evidence, never from node status.

A recipe advances a step only when its evidence says so. Two gates matter here:

- :func:`plan_gate` reads a plan node's stored lint summary (``aar.plan-node/1`` data). It
  passes only when the immutable lint result has no errors and every rule the caller requires
  actually ran. It never looks at the plan node's graph status. That is the S13g1 F2 fix: an
  agent can legally move its own lint-``blocked`` plan to ``active`` (a legal graph transition),
  but it cannot thereby clear the lint errors, so the gate still refuses it.
- :func:`qa_gate` reads a :class:`~aar.qa.report.QAReport`. ``aar qa check`` exiting 0 is not a
  pass: a check that was skipped, that ran but could not check everything (``incomplete``), or
  that only warned is not an all-clear. The gate makes that explicit, per required check.

Both gates return a :class:`GateResult` with a per-item reason, so a runner (and a person
reading ``REPORT.md``) can see exactly why a gate passed or failed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from aar.hub.writes import PlanNodeData, typed_plan
from aar.qa.report import QAReport


@dataclass(frozen=True, slots=True)
class GateReason:
    """One check's contribution to a gate decision."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class GateResult:
    """Whether a gate passed, with a reason per check."""

    passed: bool
    reasons: tuple[GateReason, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": [reason.to_dict() for reason in self.reasons]}


def _plan_node_data(plan: PlanNodeData | Mapping[str, Any]) -> PlanNodeData | None:
    """Coerce a plan node's data to :class:`PlanNodeData`; ``None`` when it is not typed.

    Accepts either the typed model or a node's raw ``payload`` mapping. Crucially it never
    receives, or reads, a graph status: the gate decides on the stored lint summary alone.
    """
    if isinstance(plan, PlanNodeData):
        return plan
    if isinstance(plan, Mapping):
        return typed_plan(_NodeShim(dict(plan)))
    return None


class _NodeShim:
    """Minimal stand-in so :func:`typed_plan` can validate a bare payload mapping."""

    kind = "plan"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


def plan_gate(
    plan_node_data: PlanNodeData | Mapping[str, Any],
    *,
    required_rules: Iterable[str] = (),
) -> GateResult:
    """Pass only when the plan's lint summary has no errors and every required rule ran.

    ``required_rules`` names rules that must have executed; a required rule that the hub could
    not run (it is in ``lint.skipped_rules`` -- e.g. it needs a transcript the hub did not have)
    fails the gate, because a clean lint over rules that never ran is not a pass. The plan
    node's graph status is deliberately ignored.
    """
    required = list(dict.fromkeys(required_rules))
    data = _plan_node_data(plan_node_data)
    if data is None:
        return GateResult(
            False,
            (GateReason("typed", False, "plan node has no aar.plan-node/1 lint summary"),),
        )
    lint = data.lint
    reasons = [
        GateReason(
            "lint.errors",
            lint.errors == 0,
            "no lint errors" if lint.errors == 0 else f"{lint.errors} lint error(s)",
        )
    ]
    skipped = set(lint.skipped_rules)
    for rule in required:
        ran = rule not in skipped
        reasons.append(
            GateReason(
                f"rule:{rule}",
                ran,
                "ran" if ran else "required rule could not run (listed in skipped_rules)",
            )
        )
    return GateResult(all(reason.passed for reason in reasons), tuple(reasons))


_SKIP_OK = "not-applicable"


def qa_gate(
    report: QAReport,
    *,
    required: Iterable[str],
    allow_warn: frozenset[str] = frozenset(),
) -> GateResult:
    """Pass only when every required QA check is present and clear.

    Per required check:

    - missing from the report -> fail;
    - ``incomplete`` (ran but could not check everything) -> fail;
    - ``skipped`` -> fail, unless its ``skip_reason`` is ``not-applicable``;
    - status ``pass`` -> pass;
    - status ``warn`` -> pass only when the check id is in ``allow_warn``;
    - status ``fail`` -> fail.

    A required check that the report lists in ``unchecked`` fails too, unless it was skipped as
    not applicable -- which is the same thing said from the report's point of view. Exit 0 from
    ``aar qa check`` is not a pass.
    """
    required_ids = list(dict.fromkeys(required))
    allow = frozenset(allow_warn)
    by_id = {check.id: check for check in report.checks}
    unchecked = set(report.unchecked)
    reasons: list[GateReason] = []
    for check_id in required_ids:
        check = by_id.get(check_id)
        if check is None:
            reasons.append(GateReason(check_id, False, "required check is not in the report"))
            continue
        skipped_na = check.status == "skipped" and check.skip_reason == _SKIP_OK
        if check.incomplete:
            reasons.append(
                GateReason(check_id, False, "ran but could not check everything (incomplete)")
            )
        elif check.status == "skipped":
            if skipped_na:
                reasons.append(GateReason(check_id, True, "skipped: not applicable"))
            else:
                reasons.append(
                    GateReason(check_id, False, f"skipped: {check.skip_reason or 'unknown'}")
                )
        elif check.status == "pass":
            reasons.append(GateReason(check_id, True, "pass"))
        elif check.status == "warn":
            allowed = check_id in allow
            reasons.append(
                GateReason(check_id, allowed, "warn (allowed)" if allowed else "warn (not allowed)")
            )
        else:  # fail
            reasons.append(GateReason(check_id, False, "fail"))
        # Cross-check the report's own unchecked list: anything it names that is not a
        # not-applicable skip must not pass, even if a check's own fields disagreed.
        if check_id in unchecked and not skipped_na and reasons[-1].passed:
            reasons[-1] = GateReason(check_id, False, "listed in report.unchecked")
    return GateResult(all(reason.passed for reason in reasons), tuple(reasons))
