"""``aar plan produce``: write an EditPlan from a stub, a transcript and a keep list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aar import cli_io
from aar.analyze.common import write_atomic
from aar.plan.produce.build import produce
from aar.plan.produce.errors import EXIT_INPUT, EXIT_OK, ProduceError
from aar.plan.produce.inputs import (
    load_faces,
    load_keep,
    load_schedule,
    load_stub,
    load_sync,
    read_laughs,
    read_transcript,
)
from aar.plan.produce.report import ProduceReport
from aar.plan.profile import ProfileError, load_profile

_PROG = "aar plan produce"


def register(sub: Any) -> None:
    run = sub.add_parser(
        "produce",
        help="Make a lint-clean EditPlan from a stub, a transcript and a keep list.",
        description=(
            "Pause trim, joins in source-audio quiet, sliver fold, end on the laugh and (shorts) "
            "a steady camera path, then lint under the profile. --no-pause-trim / --pause-trim "
            "override the profile's [produce] pause_trim. Exit status: 0 the plan was "
            "written (warnings allowed), 1 no lint-clean plan could be made, 2 an input or the "
            "profile could not be read or is malformed."
        ),
    )
    run.add_argument("stub", type=Path, help="EditPlan JSON with empty segments: the sources.")
    run.add_argument("--transcript", type=Path, required=True, help="Transcript JSON.")
    run.add_argument("--keep", type=Path, required=True, help="Keep list JSON (recording time).")
    run.add_argument("--out", type=Path, required=True, help="Where to write the plan JSON.")
    run.add_argument("--schedule", type=Path, help="Camera schedule JSON (default: one camera).")
    run.add_argument("--sync", type=Path, help="aar analyze sync report: source origins.")
    run.add_argument(
        "--faces", type=Path, action="append", default=[], help="aar.faces/1 track (repeatable)."
    )
    run.add_argument("--laughs", type=Path, help="Laugh map JSON: end on the final laugh.")
    run.add_argument("--profile", help="Profile name or .toml path (default: the stub's).")
    run.add_argument("--venue", help="A [produce.venues] entry, e.g. loud.")
    trim = run.add_mutually_exclusive_group()
    trim.add_argument(
        "--pause-trim",
        dest="pause_trim",
        action="store_const",
        const=True,
        help="Trim long pauses (default: the profile's [produce] pause_trim).",
    )
    trim.add_argument(
        "--no-pause-trim",
        dest="pause_trim",
        action="store_const",
        const=False,
        help="Do not trim pauses: keep ranges play whole (one camera on room microphones).",
    )
    run.add_argument("--json", action="store_true", help="Print the report as JSON.")
    run.set_defaults(func=_cmd_produce)


def _fail(error: dict[str, Any], exit_code: int, as_json: bool) -> int:
    if as_json:
        body = json.dumps({"ok": False, "error": error}, indent=2) + "\n"
        return cli_io.emit(_PROG, body) or exit_code
    cli_io.warn(f"error: {error['message']}")
    for detail in error.get("details", []):
        if isinstance(detail, dict) and "msg" in detail:
            where = ".".join(str(part) for part in detail.get("loc", ()))
            cli_io.warn(f"  {where}: {detail['msg']}")
        elif isinstance(detail, dict) and "message" in detail:
            cli_io.warn(f"  {detail.get('rule', '')}: {detail['message']}")
    if error.get("hint"):
        cli_io.warn(f"hint: {error['hint']}")
    return exit_code


def _cmd_produce(args: argparse.Namespace) -> int:
    as_json: bool = args.json
    try:
        stub = load_stub(args.stub)
        profile = load_profile(args.profile or stub.profile)
        result = produce(
            stub,
            transcript=read_transcript(args.transcript),
            keep=load_keep(args.keep),
            profile=profile,
            schedule=load_schedule(args.schedule) if args.schedule else None,
            sync=load_sync(args.sync) if args.sync else None,
            faces=[load_faces(path) for path in args.faces],
            laughs=read_laughs(args.laughs) if args.laughs else None,
            venue=args.venue,
            media_root=args.stub.parent,
            pause_trim=args.pause_trim,
        )
    except ProduceError as exc:
        return _fail(exc.to_dict(), exc.exit_code, as_json)
    except ProfileError as exc:
        return _fail({"type": "ProfileError", "message": str(exc)}, EXIT_INPUT, as_json)
    text = result.plan.model_dump_json(indent=2, exclude_none=True) + "\n"
    try:
        write_atomic(args.out, text)
    except OSError as exc:
        error = {"type": "output-error", "message": f"cannot write {args.out}: {exc}"}
        return _fail(error, cli_io.EXIT_OUTPUT_ERROR, as_json)
    report = result.report(str(args.out))
    out = report.model_dump_json(indent=2) if as_json else _format(report)
    return cli_io.emit(_PROG, out + "\n") or EXIT_OK


def _format(report: ProduceReport) -> str:
    counts = report.lint.counts
    lines = [
        f"wrote {report.out}: {report.kind}, {report.segments} segment(s), {report.cuts} cut(s), "
        f"{report.duration_s:g} s; lint {counts['error']} error, {counts['warn']} warn, "
        f"{counts['info']} info; {len(report.warnings)} planner warning(s)"
    ]
    for warning in report.warnings:
        lines.append(f"WARN  planner {warning.code}: {warning.message}")
    for finding in report.lint.findings:
        if finding.severity != "info":
            lines.append(f"{finding.severity.upper():5} {finding.rule}: {finding.message}")
    log = report.log
    trims = f"{len(log['trims'])} pause trim(s)" if log["pause_trim"]["ran"] else "pause trim off"
    lines.append(
        f"      {trims}, {len(log['nudges'])} nudged join(s), "
        f"{len(log['folds'])} fold(s), {len(log['snaps'])} keep edge(s) moved inward past a word"
    )
    if log["ending"] is not None:
        lines.append(
            f"      ends at {log['ending']['out_s']:.3f} s (laugh onset "
            f"{log['ending']['onset_s']:.3f} s)"
        )
    return "\n".join(lines)
