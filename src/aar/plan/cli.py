"""``aar plan`` commands: print the EditPlan JSON Schema, lint a plan and produce one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar import cli_io
from aar.node.ops.read import DescribeResult
from aar.plan.apply import (
    AssembleResult,
    CompileError,
    PrepareResult,
    build_submission,
    compile_assemble,
    compile_finish,
    compile_prepare,
)
from aar.plan.apply.render import dumps_json, render_json, render_text
from aar.plan.inputs import LaughMap, Transcript, load_laughs, load_transcript
from aar.plan.lint import Finding, LintReport, Ref, lint
from aar.plan.model import EditPlan
from aar.plan.produce import cli as produce_cli
from aar.plan.profile import ProfileError, load_profile

_PROG = "aar plan"
EXIT_OK = 0
EXIT_ERRORS = 1  # the plan was linted and has error findings
EXIT_INPUT = 2  # a file could not be read or parsed, or the profile is invalid
EXIT_REFUSED = 2  # apply refused: lint errors, or a typed compile refusal


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
    produce_cli.register(sub)

    apply = sub.add_parser(
        "apply",
        help="Compile a linted EditPlan v2 into Resolve build-op steps (dry-run).",
        description="Lints first (refuses a plan with lint errors), then compiles three phases: "
        "prepare (branch, import, find title templates), assemble (tracks, append), finish "
        "(transitions, colour, transforms, titles, gain, fades, markers). Each later phase needs "
        "the earlier phases' real result files. Exit: 0 ok, 2 refused.",
    )
    apply.add_argument("plan", type=Path, help="EditPlan JSON file.")
    apply.add_argument("--profile", help="Profile name or .toml path (default: the plan's).")
    apply.add_argument("--transcript", type=Path, help="Transcript JSON, for the lint.")
    apply.add_argument("--laughs", type=Path, help="Laugh map JSON, for the lint.")
    apply.add_argument(
        "--template-describe",
        type=Path,
        help="A timeline.describe of the template (its unique_id and size); phase 'prepare' input.",
    )
    apply.add_argument(
        "--prepare-result",
        type=Path,
        help="The phase-'prepare' result (branch identity, imported sources, found titles).",
    )
    apply.add_argument(
        "--assemble-result",
        type=Path,
        help="The phase-'assemble' result (the append results), to resolve finish item ids.",
    )
    apply.add_argument(
        "--branch-describe",
        type=Path,
        help="A timeline.describe of an existing agent branch, for an idempotency diff.",
    )
    apply.add_argument(
        "--batch-size", type=int, default=10, help="Max ranges per append step (1..10)."
    )
    apply.add_argument("--dry-run", action="store_true", help="Print the dry-run diff (default).")
    apply.add_argument("--json", action="store_true", help="Print the dry-run as JSON.")
    apply.add_argument(
        "--submit",
        choices=["prepare", "assemble", "finish"],
        help="Build and validate the ops.sequence submission for one phase (refused if any "
        "reference is unresolved; thin, for manual use with 'aar hub job submit').",
    )
    apply.set_defaults(func=_cmd_apply)


def _cmd_schema(args: argparse.Namespace) -> int:
    return cli_io.emit(_PROG, json.dumps(EditPlan.model_json_schema(), indent=2) + "\n")


def _input_error(kind: str, message: str, details: list[Any], as_json: bool) -> int:
    if as_json:
        body = {"ok": False, "error": {"type": kind, "message": message, "details": details}}
        return cli_io.emit(_PROG, json.dumps(body, indent=2) + "\n") or EXIT_INPUT
    cli_io.warn(f"error: {message}")
    for detail in details:
        where = ".".join(str(part) for part in detail["loc"])
        cli_io.warn(f"  {where}: {detail['msg']}")
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
    out = (
        report.model_dump_json(indent=2) if as_json else _format_report(report, len(plan.segments))
    )
    written = cli_io.emit(_PROG, out + "\n")
    if written != EXIT_OK:
        return written
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


def _load_describe(path: Path) -> DescribeResult:
    return DescribeResult.model_validate_json(path.read_text(encoding="utf-8"))


def _sha(path: Path | None) -> list[str]:
    if path is None:
        return []
    import hashlib

    return [hashlib.sha256(path.read_bytes()).hexdigest()]


def _compile_refusal(exc: CompileError, as_json: bool) -> int:
    if as_json:
        body = {"ok": False, "error": exc.as_dict()}
        return cli_io.emit(_PROG, json.dumps(body, indent=2) + "\n") or EXIT_REFUSED
    cli_io.warn(f"refused ({exc.code}): {exc.message}")
    cli_io.warn(f"  fix: {exc.fix_hint}")
    return EXIT_REFUSED


def _cmd_apply(args: argparse.Namespace) -> int:
    as_json: bool = args.json
    try:
        plan = EditPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
        transcript: Transcript | None = (
            load_transcript(args.transcript) if args.transcript else None
        )
        laughs: LaughMap | None = load_laughs(args.laughs) if args.laughs else None
        profile = load_profile(args.profile or plan.profile)
        template = _load_describe(args.template_describe) if args.template_describe else None
        prepare_res = (
            PrepareResult.model_validate_json(args.prepare_result.read_text(encoding="utf-8"))
            if args.prepare_result
            else None
        )
        assemble_res = (
            AssembleResult.model_validate_json(args.assemble_result.read_text(encoding="utf-8"))
            if args.assemble_result
            else None
        )
        branch_describe = _load_describe(args.branch_describe) if args.branch_describe else None
    except ValidationError as exc:
        return _input_error(
            "invalid-input", "input failed validation", _validation_details(exc), as_json
        )
    except (OSError, json.JSONDecodeError, ProfileError) as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)

    report = lint(plan, profile, transcript=transcript, laughs=laughs)
    if not report.ok:
        if as_json:
            body = {"ok": False, "error": {"type": "lint_errors", "report": report.model_dump()}}
            return cli_io.emit(_PROG, json.dumps(body, indent=2) + "\n") or EXIT_REFUSED
        cli_io.warn(f"refused (lint_errors): the plan has {report.counts['error']} lint error(s)")
        for finding in report.findings:
            if finding.severity == "error":
                cli_io.warn(f"  {finding.rule}: {finding.message}")
        return EXIT_REFUSED
    for finding in report.findings:  # warnings are printed, not fatal
        if finding.severity == "warn":
            cli_io.warn(f"warning {finding.rule}: {finding.message}")

    try:
        prepare = compile_prepare(plan, template=template)
        assemble = compile_assemble(plan, prepare=prepare_res, batch_size=args.batch_size)
        finish = compile_finish(
            plan, prepare=prepare_res, assemble=assemble_res, batch_size=args.batch_size
        )
        if args.submit is not None:
            return _do_submit(args, plan, prepare, assemble, finish, as_json)
    except CompileError as exc:
        return _compile_refusal(exc, as_json)
    except ValidationError as exc:
        return _input_error(
            "invalid-compile",
            "a compiled op param failed validation",
            _validation_details(exc),
            as_json,
        )

    if as_json:
        out = dumps_json(
            render_json(plan, prepare, assemble, finish, branch_describe=branch_describe)
        )
    else:
        out = render_text(plan, prepare, assemble, finish, branch_describe=branch_describe)
    return cli_io.emit(_PROG, out)


_SUBMIT_INPUTS = {
    "prepare": lambda a: _sha(a.template_describe),
    "assemble": lambda a: _sha(a.prepare_result),
    "finish": lambda a: _sha(a.prepare_result) + _sha(a.assemble_result),
}


def _do_submit(
    args: argparse.Namespace,
    plan: EditPlan,
    prepare: Any,
    assemble: Any,
    finish: Any,
    as_json: bool,
) -> int:
    phase = args.submit
    comp = {"prepare": prepare, "assemble": assemble, "finish": finish}[phase]
    # Surface every note and warning before submitting, so nothing is hidden on the submit path
    # (R4): mute warnings, dropped-title notes, the head-fade [I] note, etc.
    for warning in comp.warnings:
        cli_io.warn(f"warning: {warning}")
    for note in comp.notes:
        cli_io.warn(f"note: {note}")
    steps = comp.steps
    if not steps:
        cli_io.warn(f"phase {phase!r} compiled to no steps; nothing to submit")
        return EXIT_OK
    submission = build_submission(plan, phase, steps, input_hashes=_SUBMIT_INPUTS[phase](args))
    if as_json:
        body = {
            "ok": True,
            "kind": submission.kind,
            "idempotency_key": submission.idempotency_key,
            "phase": submission.phase,
            "params": submission.params,
        }
        return cli_io.emit(_PROG, json.dumps(body, indent=2) + "\n")
    lines = [
        f"submission ready: kind {submission.kind}, phase {submission.phase}, "
        f"{len(submission.params['steps'])} step(s)",
        f"idempotency-key: {submission.idempotency_key}",
        "enqueue with: aar hub job submit --kind ops.sequence --params <file> "
        f"--idempotency-key {submission.idempotency_key} ...",
    ]
    return cli_io.emit(_PROG, "\n".join(lines) + "\n")


def _format_report(report: LintReport, segments: int) -> str:
    counts = report.counts
    head = (
        f"profile {report.profile}: {segments} segment(s), {report.duration_s:g} s "
        f"({report.duration_frames} frames): {counts['error']} error, "
        f"{counts['warn']} warn, {counts['info']} info"
    )
    return "\n".join([head, *(_format_finding(f) for f in report.findings)])
