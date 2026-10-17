"""Small helpers shared by the ``aar.plan`` tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aar.plan.lint import Finding, LintReport

FIXTURES = Path(__file__).parent / "fixtures"

PlanData = dict[str, Any]
RefTuple = tuple[int | None, str | None, str | None, str | None]


def only(report: LintReport, rule: str) -> list[Finding]:
    """The findings of one rule."""
    return [f for f in report.findings if f.rule == rule]


def refs(finding: Finding) -> list[RefTuple]:
    """A finding's refs as (segment, point, record timecode, source timecode) tuples."""
    return [(r.segment, r.point, r.record_tc, r.source_tc) for r in finding.refs]
