"""``aar plan`` commands: print the EditPlan JSON Schema, lint, compile (apply), produce, export
and render plans."""

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
from aar.plan.export import (
    ExportError,
    export_fcpxml,
    load_media_map,
    verify_plan_fcpxml,
)
from aar.plan.export.files import write_pair
from aar.plan.export.reader import MAX_BYTES as FCPXML_MAX_BYTES
from aar.plan.inputs import LaughMap, Transcript, load_laughs, load_transcript
from aar.plan.lint import Finding, LintReport, Ref, lint
from aar.plan.model import EditPlan
from aar.plan.produce import cli as produce_cli
from aar.plan.profile import ProfileError, load_profile
from aar.plan.render import RUNTIME_CODES as RENDER_RUNTIME_CODES
from aar.plan.render import RenderError, RenderTerminated, render_plan

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

    export = sub.add_parser(
        "export",
        help="Export a linted EditPlan v2 as an FCPXML 1.10 timeline (no Resolve needed).",
        description="Lints first (refuses a plan with lint errors), writes FCPXML 1.10 for "
        "File > Import > Timeline in Resolve, reads it back and checks it reproduces the plan, "
        "and writes the export report beside it (<name>.report.json). --verify FILE reads an "
        "FCPXML file back and compares it with the plan. Exit: 0 ok, 1 --verify found a "
        "mismatch, 2 refused or unreadable input.",
    )
    export.add_argument("plan", type=Path, help="EditPlan JSON file.")
    export.add_argument("--profile", help="Profile name or .toml path (default: the plan's).")
    export.add_argument("--format", choices=["fcpxml"], default="fcpxml", help="Output format.")
    target = export.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="FCPXML file to write (never overwritten).")
    target.add_argument(
        "--verify", type=Path, metavar="FILE", help="Read FILE back and compare it with the plan."
    )
    export.add_argument(
        "--media-map",
        type=Path,
        help="JSON object of linux_root -> windows_root, for sources with hub (Linux) paths.",
    )
    export.add_argument("--transcript", type=Path, help="Transcript JSON, for the lint.")
    export.add_argument("--laughs", type=Path, help="Laugh map JSON, for the lint.")
    export.add_argument(
        "--overwrite", action="store_true", help="Replace an existing FCPXML and report."
    )
    export.add_argument(
        "--allow-unknown-timecode",
        action="store_true",
        help="Export sources with no start_timecode anyway, their assets at 0s (Resolve links "
        "their clips only if the file's timecode starts at 00:00:00:00).",
    )
    export.add_argument("--json", action="store_true", help="Print the report as JSON.")
    export.set_defaults(func=_cmd_export)

    render = sub.add_parser(
        "render",
        help="Render a linted EditPlan v2 to MP4 with ffmpeg on the hub (no Resolve needed).",
        description="Lints first (refuses a plan with lint errors), renders the plan from the "
        "hub's own media copies with ffmpeg (the FCPXML export's layout: cuts, dissolves, "
        "handles, reframe crops, titles, crossfades, gains, loudness to the profile), runs "
        "aar qa check on the result, and writes the file and its report (<stem>.render.json) "
        "as one pair. Exit: 0 rendered and QA did not fail, 1 rendered and QA failed, 2 refused "
        "or unreadable input, 3 ffmpeg failed, timed out or left an incomplete file, 143 "
        "stopped by SIGTERM (ffmpeg stopped, work dir removed).",
    )
    render.add_argument("plan", type=Path, help="EditPlan JSON file.")
    render.add_argument("--profile", help="Profile name or .toml path (default: the plan's).")
    render.add_argument("--out", type=Path, required=True, help="MP4 (or .mov) file to write.")
    render.add_argument(
        "--media-map",
        type=Path,
        help="JSON object of linux_root -> windows_root (the export's map, read the other way), "
        "for sources with edit-PC paths.",
    )
    render.add_argument(
        "--preset",
        choices=["review", "master"],
        default="review",
        help="review: H.264 High + AAC with faststart; master: higher quality H.264, or ProRes "
        "(.mov) when the profile's [render] master_codec says so.",
    )
    render.add_argument("--transcript", type=Path, help="Transcript JSON, for the lint and QA.")
    render.add_argument("--laughs", type=Path, help="Laugh map JSON, for the lint.")
    render.add_argument(
        "--overwrite", action="store_true", help="Replace an existing file and report."
    )
    render.add_argument(
        "--memory-budget",
        type=_memory_mb,
        metavar="MB",
        help="The memory one ffmpeg of the render may use, in MB (the picture is encoded in "
        "chunks that fit it); default min(6 GiB, half of MemAvailable, the cgroup's free "
        "headroom), or the profile's [render] memory_budget_mb.",
    )
    render.add_argument("--json", action="store_true", help="Print the report as JSON.")
    render.set_defaults(func=_cmd_render)


def _memory_mb(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of MB") from None
    if value < 64:
        raise argparse.ArgumentTypeError("the memory budget must be at least 64 MB")
    return value


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


# -- export (S18) ------------------------------------------------------------------------------


def report_path(out: Path) -> Path:
    """Where the export report goes: beside the FCPXML, ``<name>.report.json``."""
    return out.with_name(f"{out.stem}.report.json")


def _write_export(out: Path, report: Path, fcpxml: str, body: str, overwrite: bool) -> None:
    """The FCPXML and its report land together or not at all (round 3, item 9)."""
    write_pair([(out, fcpxml.encode("utf-8")), (report, body.encode("utf-8"))], overwrite=overwrite)


def _render_export(report: dict[str, Any], out: Path, report_file: Path) -> str:
    timeline = report["timeline"]
    counts = report["exported"]
    lines = [
        f"exported FCPXML {report['fcpxml_version']}: {out} "
        f"(sha256 {report['fcpxml_sha256'][:12]})",
        f"timeline {report['plan']['timeline']!r}: {timeline['fps']} fps, starts "
        f"{timeline['start']}, {timeline['total_frames']} frames, {timeline['resolution']}",
        f"exported: {counts['segments']} segment(s), outro {'yes' if counts['outro'] else 'no'}, "
        f"{counts['transitions']} transition(s), {counts['titles']} title(s), "
        f"{counts['assets']} asset(s)",
        "import into Resolve:",
        *(f"  - {step}" for step in report["import"]),
        "media:",
        *(f"  {m['source']}: {m['src']} ({m['rule']})" for m in report["media"]),
        "lanes (after import):",
        *(
            f"  lane {row['lane']}: {row['carries']} -> {row['imported_video']}"
            + (f", sound {row['imported_audio']}" if row["imported_audio"] else "")
            for row in report["lanes"]
        ),
        "titles (after import):",
        *(
            f"  {t['location']} {t['name']!r} at {t['timecode']} -> {t['imported_video']}"
            for t in report["titles"]
        ),
        "audio tracks (plan -> after import):",
        *(
            f"  {row['track']} {row['name']!r} -> {', '.join(row['imported'])}"
            for row in report["audio_tracks"]
        ),
        *(
            f"  split crossfade {row['location']} at {row['timecode']}: {row['overlap']}"
            for row in report["split_crossfade_audio"]
        ),
        f"{report['not_imported']['heading']} ({len(report['not_imported']['markers'])}):",
        *(f"  marker {m['name']!r} at {m['timecode']}" for m in report["not_imported"]["markers"]),
        f"not expressible in FCPXML ({len(report['not_expressible'])}):",
        *(
            f"  {item['location']}: {item['detail']} -> {item['action']}"
            for item in report["not_expressible"]
        ),
        f"notes ({len(report['notes'])}):",
        *(f"  {item['location']}: {item['detail']}" for item in report["notes"]),
        f"self-check: {'ok' if report['self_check']['ok'] else 'FAILED'} "
        + ", ".join(f"{v} {k}" for k, v in report["self_check"]["checked"].items()),
        f"report: {report_file}",
    ]
    return "\n".join(lines) + "\n"


def _export_refusal(exc: ExportError, as_json: bool) -> int:
    code = _compile_refusal(exc, as_json)
    if not as_json:
        for error in exc.details.get("errors", []):
            cli_io.warn(f"  {error['rule']}: {error['message']}")
        for problem in exc.details.get("problems", []):
            cli_io.warn(f"  {problem}")
    return code


def _cmd_export(args: argparse.Namespace) -> int:
    as_json: bool = args.json
    try:
        plan = EditPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
        transcript: Transcript | None = (
            load_transcript(args.transcript) if args.transcript else None
        )
        laughs: LaughMap | None = load_laughs(args.laughs) if args.laughs else None
        profile = load_profile(args.profile or plan.profile)
        media_map = load_media_map(args.media_map) if args.media_map else None
    except ValidationError as exc:
        return _input_error(
            "invalid-input", "input failed validation", _validation_details(exc), as_json
        )
    except ExportError as exc:
        return _export_refusal(exc, as_json)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ProfileError) as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)

    if args.verify is not None:
        return _do_verify(args.verify, plan, profile, media_map, as_json)
    try:
        result = export_fcpxml(
            plan,
            profile,
            media_map=media_map,
            transcript=transcript,
            laughs=laughs,
            allow_unknown_timecode=args.allow_unknown_timecode,
        )
    except ExportError as exc:
        return _export_refusal(exc, as_json)
    for finding in result.warnings:  # warnings are printed, not fatal
        cli_io.warn(f"warning {finding.rule}: {finding.message}")
    out: Path = args.out
    report_file = report_path(out)
    report = {**result.report, "file": out.name}
    body = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    try:
        _write_export(out, report_file, result.fcpxml, body, args.overwrite)
    except ExportError as exc:
        return _export_refusal(exc, as_json)
    except OSError as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)
    text = body if as_json else _render_export(report, out, report_file)
    return cli_io.emit(_PROG, text)


def _do_verify(path: Path, plan: EditPlan, profile: Any, media_map: Any, as_json: bool) -> int:
    try:
        if path.stat().st_size > FCPXML_MAX_BYTES:
            return _input_error(
                "too-large", f"{path} is over {FCPXML_MAX_BYTES} bytes", [], as_json
            )
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)
    try:
        result = verify_plan_fcpxml(text, plan, profile, media_map=media_map)
    except ExportError as exc:
        return _export_refusal(exc, as_json)
    if as_json:
        out = json.dumps({**result.as_dict(), "file": str(path)}, indent=2) + "\n"
    else:
        counts = ", ".join(f"{v} {k}" for k, v in result.checked.items())
        head = f"verify {path}: {'ok' if result.ok else 'MISMATCH'} ({counts})"
        out = "\n".join([head, *(f"  {p}" for p in result.problems)]) + "\n"
    written = cli_io.emit(_PROG, out)
    if written != EXIT_OK:
        return written
    return EXIT_OK if result.ok else EXIT_ERRORS


# -- render (S18b) ------------------------------------------------------------------------------

EXIT_RENDER_FAILED = 3  # ffmpeg failed or timed out while rendering
EXIT_TERMINATED = 143  # SIGTERM (128 + 15), after ffmpeg was stopped and the work dir removed


def _render_refusal(exc: RenderError, as_json: bool) -> int:
    code = _export_refusal(exc, as_json)
    if exc.code in RENDER_RUNTIME_CODES:
        tail = exc.details.get("stderr_tail")
        if tail and not as_json:
            cli_io.warn(f"  ffmpeg: {tail.strip().splitlines()[-1] if tail.strip() else ''}")
        return EXIT_RENDER_FAILED
    return code


def _render_text(report: dict[str, Any], out: Path, report_file: Path) -> str:
    timeline = report["timeline"]
    loud = report["loudness"]
    qa = report["qa"]
    measured = loud["output"].get("integrated_lufs")
    lines = [
        f"rendered {report['preset']} {report['format']}: {out} (sha256 {report['sha256'][:12]}, "
        f"{report['size_bytes']} bytes)",
        f"timeline {report['plan']['timeline']!r}: {timeline['fps']} fps, "
        f"{timeline['total_frames']} frames, {timeline['resolution']}",
        f"memory: {len(report['memory']['chunks'])} picture chunk(s), "
        f"{report['memory']['mix_passes']} mix pass(es), within a "
        f"{report['memory']['budget']['mb']} MB budget",
        f"loudness: {loud['method']} to {loud['target_lufs']:g} LUFS"
        + (f" -> {measured:g} LUFS" if measured is not None else "")
        + (f" ({loud['note']})" if loud.get("note") else ""),
        f"qa: {qa['status']}"
        + (f" (unchecked: {', '.join(qa['unchecked'])})" if qa.get("unchecked") else ""),
        *(
            f"  {c['id']}: {c['status']}: {c['message']}"
            for c in qa["checks"]
            if c["status"] in ("fail", "warn")
        ),
        f"not rendered ({len(report['not_rendered'])}):",
        *(
            f"  {item['location']}: {item['detail']} -> {item['action']}"
            for item in report["not_rendered"]
        ),
        f"notes ({len(report['notes'])}):",
        *(f"  {item['location']}: {item['detail']}" for item in report["notes"]),
        f"report: {report_file}",
    ]
    return "\n".join(lines) + "\n"


def _cmd_render(args: argparse.Namespace) -> int:
    as_json: bool = args.json
    try:
        plan = EditPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
        transcript: Transcript | None = (
            load_transcript(args.transcript) if args.transcript else None
        )
        laughs: LaughMap | None = load_laughs(args.laughs) if args.laughs else None
        profile = load_profile(args.profile or plan.profile)
        media_map = load_media_map(args.media_map) if args.media_map else None
    except ValidationError as exc:
        return _input_error(
            "invalid-input", "input failed validation", _validation_details(exc), as_json
        )
    except ExportError as exc:
        return _export_refusal(exc, as_json)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ProfileError) as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)
    try:
        result = render_plan(
            plan,
            profile,
            out=args.out,
            preset=args.preset,
            media_map=media_map,
            media_root=args.plan.resolve().parent,
            transcript=transcript,
            laughs=laughs,
            overwrite=args.overwrite,
            memory_budget_mb=args.memory_budget,
        )
    except RenderError as exc:
        return _render_refusal(exc, as_json)
    except RenderTerminated as exc:
        cli_io.warn(f"render stopped: {exc}; ffmpeg was stopped and the work dir removed")
        return EXIT_TERMINATED
    except OSError as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)
    for finding in result.warnings:
        cli_io.warn(f"warning {finding.rule}: {finding.message}")
    if result.report["loudness"].get("warning"):
        cli_io.warn(f"warning loudness: {result.report['loudness']['warning']}")
    if as_json:
        text = json.dumps(result.report, indent=2, ensure_ascii=False) + "\n"
    else:
        text = _render_text(result.report, result.out, result.report_path)
    written = cli_io.emit(_PROG, text)
    if written != EXIT_OK:
        return written
    return EXIT_OK if result.qa_ok else EXIT_ERRORS
