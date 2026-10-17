"""``aar plan`` commands: print the EditPlan JSON Schema and lint a plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.plan.inputs import LaughMap, Transcript, load_laughs, load_transcript
from aar.plan.lint import Finding, LintReport, Ref, lint
from aar.plan.model import EditPlan
from aar.plan.profile import ProfileError, load_profile

EXIT_OK = 0
EXIT_ERRORS = 1  # the plan was linted and has error findings
EXIT_INPUT = 2  # a file could not be read or parsed, or the profile is invalid


def register(subparsers: Any) -> None:
    """Add the ``plan`` command group to the top-level parser."""
    plan = subparsers.add_parser("plan", help="EditPlan schema and linting.")
    sub = plan.add_subparsers(dest="plan_command", metavar="COMMAND", required=True)

    schema = sub.add_parser("schema", help="Print the EditPlan JSON Schema.")
    schema.set_defaults(func=_cmd_schema)

    check = sub.add_parser(
        "lint",
        help="Check a plan against a rule profile.",
        description="Exit status: 0 no errors, 1 error findings, 2 unreadable input.",
    )
    check.add_argument("plan", type=Path, help="EditPlan JSON file.")
    check.add_argument("--transcript", type=Path, help="Transcript JSON (words with times).")
    check.add_argument("--laughs", type=Path, help="Laugh map JSON.")
    check.add_argument("--profile", help="Profile name or .toml path (default: the plan's).")
    check.add_argument("--json", action="store_true", help="Print the report as JSON.")
    check.set_defaults(func=_cmd_lint)


def _cmd_schema(args: argparse.Namespace) -> int:
    print(json.dumps(EditPlan.model_json_schema(), indent=2))
    return EXIT_OK


def _input_error(kind: str, message: str, details: list[Any], as_json: bool) -> int:
    if as_json:
        body = {"ok": False, "error": {"type": kind, "message": message, "details": details}}
        print(json.dumps(body, indent=2))
    else:
        print(f"error: {message}", file=sys.stderr)
        for detail in details:
            where = ".".join(str(part) for part in detail["loc"])
            print(f"  {where}: {detail['msg']}", file=sys.stderr)
    return EXIT_INPUT


def _validation_details(exc: ValidationError) -> list[Any]:
    return json.loads(exc.json(include_url=False, include_context=False, include_input=False))


def _cmd_lint(args: argparse.Namespace) -> int:
    as_json: bool = args.json
    try:
        plan = EditPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
        transcript: Transcript | None = (
            load_transcript(args.transcript) if args.transcript else None
        )
        laughs: LaughMap | None = load_laughs(args.laughs) if args.laughs else None
        profile = load_profile(args.profile or plan.profile)
        report = lint(plan, profile, transcript=transcript, laughs=laughs)
    except ValidationError as exc:
        return _input_error(
            "invalid-input", "input failed validation", _validation_details(exc), as_json
        )
    except (OSError, json.JSONDecodeError, ProfileError) as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)
    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        print(_format_report(report, len(plan.segments)))
    return EXIT_OK if report.ok else EXIT_ERRORS


def _format_ref(ref: Ref) -> str:
    parts: list[str] = []
    if ref.segment is not None:
        parts.append(f"segment {ref.segment}" + (f" {ref.point}" if ref.point else ""))
    if ref.item:
        parts.append(ref.item)
    if ref.record_tc:
        parts.append(f"record {ref.record_tc}")
    if ref.source_tc:
        parts.append(f"source {ref.source_tc}" + (f" ({ref.source_id})" if ref.source_id else ""))
    return ", ".join(parts)


def _format_finding(finding: Finding) -> str:
    lines = [f"{finding.severity.upper():5} {finding.rule}: {finding.message}"]
    lines.extend(f"      at {_format_ref(ref)}" for ref in finding.refs)
    lines.append(f"      fix: {finding.fix_hint}")
    return "\n".join(lines)


def _format_report(report: LintReport, segments: int) -> str:
    counts = report.counts
    head = (
        f"profile {report.profile}: {segments} segment(s), {report.duration_s:g} s "
        f"({report.duration_frames} frames): {counts['error']} error, "
        f"{counts['warn']} warn, {counts['info']} info"
    )
    return "\n".join([head, *(_format_finding(f) for f in report.findings)])
