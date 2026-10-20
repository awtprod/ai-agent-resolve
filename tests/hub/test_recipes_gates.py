"""The gate library: qa_gate's table of statuses/skip reasons, and plan_gate's F2 fix."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from write_support import AGENT, Hub, make_hub, plan_fixture

from aar.hub.writes import PlanNodeData, set_own_status, submit_plan, typed_plan
from aar.qa.report import QACheck, QAReport, build_report
from aar.recipes.gates import plan_gate, qa_gate


@pytest.fixture
def hub(tmp_path: Any) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _report(check: QACheck, *, also: list[QACheck] | None = None) -> QAReport:
    checks = [check, *(also or [])]
    return build_report(file="f.mp4", kind="short", profile="ccn", media=None, checks=checks)


def _check(
    status: str, *, skip_reason: str | None = None, incomplete: bool = False
) -> QACheck:
    return QACheck(
        id="decode", status=status, message="m", skip_reason=skip_reason, incomplete=incomplete
    )


# (status, skip_reason, incomplete, allow_warn) -> passed
QA_CASES = [
    ("pass", None, False, frozenset(), True),
    ("warn", None, False, frozenset(), False),
    ("warn", None, False, frozenset({"decode"}), True),
    ("fail", None, False, frozenset(), False),
    ("pass", None, True, frozenset(), False),  # incomplete fails even when status is pass
    ("warn", None, True, frozenset({"decode"}), False),  # incomplete fails despite allow_warn
    ("skipped", "not-applicable", False, frozenset(), True),
    ("skipped", "no-plan", False, frozenset(), False),
    ("skipped", "no-transcript", False, frozenset(), False),
    ("skipped", "render-unreadable", False, frozenset(), False),
    ("skipped", None, False, frozenset(), False),
]


@pytest.mark.parametrize("status,skip_reason,incomplete,allow_warn,expected", QA_CASES)
def test_qa_gate_table(
    status: str, skip_reason: str | None, incomplete: bool, allow_warn: Any, expected: bool
) -> None:
    report = _report(_check(status, skip_reason=skip_reason, incomplete=incomplete))
    result = qa_gate(report, required=("decode",), allow_warn=allow_warn)
    assert result.passed is expected, result.to_dict()


def test_qa_gate_missing_required_check_fails() -> None:
    report = _report(QACheck(id="loudness", status="pass", message="ok"))
    result = qa_gate(report, required=("decode",))
    assert result.passed is False
    assert result.reasons[0].name == "decode" and not result.reasons[0].passed


def test_qa_gate_all_required_present_and_clear_passes() -> None:
    report = build_report(
        file="f.mp4", kind="short", profile="ccn", media=None,
        checks=[
            QACheck(id="decode", status="pass", message="ok"),
            QACheck(id="loudness", status="warn", message="close"),
        ],
    )
    assert qa_gate(report, required=("decode", "loudness"), allow_warn=frozenset({"loudness"})).passed
    assert not qa_gate(report, required=("decode", "loudness")).passed  # warn not allowed


def test_qa_gate_unchecked_cross_check() -> None:
    # A skipped (not-applicable) check lands in report.unchecked but still passes.
    report = _report(_check("skipped", skip_reason="not-applicable"))
    assert "decode" in report.unchecked
    assert qa_gate(report, required=("decode",)).passed is True


# -- plan_gate ----------------------------------------------------------------


def _plan_data(errors: int, skipped_rules: tuple[str, ...] = ()) -> PlanNodeData:
    from aar.hub.writes import LintSummary

    return PlanNodeData(
        plan_sha256="0" * 64,
        plan_version=1,
        rule_profile="ccn",
        derives_from="nd_x",
        lint=LintSummary(
            ok=errors == 0, errors=errors, warnings=0, infos=0,
            skipped_rules=list(skipped_rules), findings=[], omitted=0,
        ),
    )


def test_plan_gate_passes_clean_and_fails_on_errors() -> None:
    assert plan_gate(_plan_data(0)).passed is True
    assert plan_gate(_plan_data(2)).passed is False


def test_plan_gate_required_rule_that_did_not_run_fails() -> None:
    data = _plan_data(0, skipped_rules=("end-on-laugh",))
    assert plan_gate(data, required_rules=("end-on-laugh",)).passed is False
    assert plan_gate(data, required_rules=("cut-density",)).passed is True


def test_plan_gate_ignores_node_status_f2(hub: Hub) -> None:
    """S13g1 F2: an agent moving its own lint-blocked plan active must not pass the gate."""
    brief = __import__("write_support").brief_data(hub.project_id)
    from aar.hub.writes import create_brief

    brief_id = create_brief(hub.log, brief, AGENT, idempotency_key="b1").body["node"]["id"]
    result = submit_plan(
        hub.log, hub.store, hub.project_id, plan_fixture("short_broken"), AGENT,
        derives_from=brief_id, idempotency_key="p1",
    )
    plan_node_id = result.body["node"]["id"]
    assert result.body["node"]["status"] == "blocked"  # stored blocked by lint errors
    # The agent flips its own blocked plan to active (a legal graph transition).
    set_own_status(hub.log, hub.project_id, plan_node_id, "active", AGENT, idempotency_key="s1")
    from aar.hub.graph import WorkGraph

    node = WorkGraph(hub.log, hub.project_id).node(plan_node_id)
    assert node.status == "active"  # status now says active...
    assert plan_gate(typed_plan(node)).passed is False  # ...but the gate still refuses it
