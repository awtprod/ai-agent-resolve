"""``aar qa check``: run media QA on a rendered file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar import cli_io
from aar.analyze.errors import AnalyzeError
from aar.analyze.whisper import WhisperTranscriber
from aar.media import MediaError
from aar.plan.inputs import Transcript, load_transcript
from aar.plan.model import EditPlan
from aar.plan.profile import ProfileError, load_profile
from aar.qa.check import check
from aar.qa.report import QACheck, QAReport

_PROG = "aar qa"
EXIT_OK = 0  # every check passed, warned or was skipped
EXIT_FAILED = 1  # at least one check failed
EXIT_INPUT = 2  # an input could not be read, the profile is invalid, or ffmpeg is missing

_SHOWN_DETAILS = 5


def register(subparsers: Any) -> None:
    """Add the ``qa`` command group to the top-level parser."""
    qa = subparsers.add_parser("qa", help="Media QA on rendered files.")
    sub = qa.add_subparsers(dest="qa_command", metavar="COMMAND", required=True)
    run = sub.add_parser(
        "check",
        help="Check a render: loudness, clipping, decode, silence, size, plan joins, captions.",
        description=(
            "Exit status: 0 no check failed (warnings and skips allowed), 1 the checks ran and "
            "one failed (including media that exists but does not decode), 2 an input, "
            "ffmpeg/ffprobe or the whisper model cannot be located or opened, or the profile "
            "is invalid."
        ),
    )
    run.add_argument(
        "file",
        type=Path,
        help=(
            "The rendered media file. A relative name that starts with '-' needs '--' before "
            "it or './' in front of it."
        ),
    )
    run.add_argument(
        "--kind",
        required=True,
        choices=("short", "longform", "review"),
        help="Render kind: picks the loudness target; review renders also get the size cap.",
    )
    run.add_argument(
        "--profile", help="Profile name or .toml path (default: the plan's, else ccn)."
    )
    run.add_argument("--plan", type=Path, help="EditPlan JSON the render was made from.")
    run.add_argument(
        "--transcript", type=Path, help="Source transcript JSON (words, recording time)."
    )
    run.add_argument(
        "--silence-db",
        type=float,
        metavar="N",
        help="Silence threshold in dB for the longest-silence check (default: profile, -42).",
    )
    run.add_argument(
        "--whisper-model",
        metavar="MODEL",
        help=(
            "Run join-retranscribe with this faster-whisper model (a name looked up under "
            "--model-dir or $AAR_MODEL_DIR, or a model directory; never downloaded). Needs the "
            "'whisper' extra. Without it join-retranscribe is skipped (no-transcriber)."
        ),
    )
    run.add_argument("--model-dir", type=Path, help="Model directory for --whisper-model.")
    run.add_argument("--json", action="store_true", help="Print the report as JSON.")
    run.set_defaults(func=_cmd_check)


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


def _cmd_check(args: argparse.Namespace) -> int:
    as_json: bool = args.json
    file: Path = args.file
    if args.silence_db is not None and not (math.isfinite(args.silence_db) and args.silence_db < 0):
        return _input_error(
            "invalid-argument", "--silence-db must be a negative number of dB", [], as_json
        )
    try:
        if not file.exists():
            return _input_error("FileNotFoundError", f"file not found: {file}", [], as_json)
        if not file.is_file():
            return _input_error("NotAFile", f"not a regular file: {file}", [], as_json)
    except OSError as exc:
        return _input_error(type(exc).__name__, f"cannot read {file}: {exc}", [], as_json)
    try:
        plan = (
            EditPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
            if args.plan
            else None
        )
        transcript: Transcript | None = (
            load_transcript(args.transcript) if args.transcript else None
        )
        profile = load_profile(args.profile or (plan.profile if plan else "ccn"))
        transcriber = None
        if args.whisper_model is not None:
            transcriber = WhisperTranscriber(args.whisper_model, model_dir=args.model_dir)
        elif args.model_dir is not None:
            return _input_error(
                "invalid-argument", "--model-dir needs --whisper-model", [], as_json
            )
        report = check(
            file,
            kind=args.kind,
            profile=profile,
            plan=plan,
            transcript=transcript,
            transcriber=transcriber,
            silence_db=args.silence_db,
            media_root=args.plan.parent if args.plan else None,
        )
    except ValidationError as exc:
        return _input_error(
            "invalid-input", "input failed validation", _validation_details(exc), as_json
        )
    except AnalyzeError as exc:  # --whisper-model: the extra or the model is missing
        message = f"{exc} ({exc.hint})" if exc.hint else str(exc)
        return _input_error(exc.type, message, [], as_json)
    except (OSError, json.JSONDecodeError, ProfileError, MediaError, ValueError) as exc:
        return _input_error(type(exc).__name__, str(exc), [], as_json)
    # json.dumps, not model_dump_json: a file name that is not valid UTF-8 carries lone
    # surrogates, which pydantic refuses to encode and cli_io.emit escapes instead.
    out = (
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
        if as_json
        else format_report(report)
    )
    written = cli_io.emit(_PROG, out + "\n")
    if written != EXIT_OK:
        return written
    return EXIT_OK if report.ok else EXIT_FAILED


_LABEL = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skipped": "SKIP"}


def _detail_line(detail: dict[str, Any]) -> str:
    if "join" in detail:
        head = f"join {detail['join']} at {detail.get('record_tc', '')}"
        reason = detail.get("hint") or detail.get("reason")
        return f"{head}: {detail['status']}" + (f" ({reason})" if reason else "")
    if "word" in detail:
        return f"{detail['word']!r} at {detail['record_start_s']} s"
    return json.dumps(detail, sort_keys=True)


def _format_check(entry: QACheck) -> list[str]:
    lines = [f"{_LABEL[entry.status]:4}  {entry.id}: {entry.message}"]
    if entry.status in ("fail", "warn"):
        shown = [d for d in entry.details if d.get("status", "fail") != "pass"]
        lines.extend(f"      {_detail_line(d)}" for d in shown[:_SHOWN_DETAILS])
        if len(shown) > _SHOWN_DETAILS:
            lines.append(f"      ... {len(shown) - _SHOWN_DETAILS} more (use --json)")
    return lines


def format_report(report: QAReport) -> str:
    counts = report.counts
    head = (
        f"qa {report.file} ({report.kind}, profile {report.profile}): "
        f"{report.status.upper()}: {counts['pass']} pass, {counts['warn']} warn, "
        f"{counts['fail']} fail, {counts['skipped']} skipped"
    )
    lines = [head]
    if report.unchecked:
        lines.append(f"not checked or not fully checked: {', '.join(report.unchecked)}")
    for entry in report.checks:
        lines.extend(_format_check(entry))
    return "\n".join(lines)
