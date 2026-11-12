"""``aar plan stub``: write the plan stub ``aar plan produce`` takes, from hub media files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from aar import cli_io
from aar.analyze.common import write_atomic
from aar.plan.stub import StubError, StubSource, build_stub

_PROG = "aar plan stub"
EXIT_OK = 0
EXIT_REFUSED = 2
MAX_BASE_BYTES = 1024 * 1024
_RESOLUTION = re.compile(r"^([1-9][0-9]{0,4})x([1-9][0-9]{0,4})$")


def register(sub: Any) -> None:
    run = sub.add_parser(
        "stub",
        help="Write a plan stub (sources, no segments) from hub media files.",
        description=(
            "Probe each media file (frame count, frame size, constant frame rate, sound, start "
            "timecode) and write the EditPlan stub aar plan produce takes. --camera, "
            "--shot-size and --source-id, when given, are given once per media file, in order. "
            "Exit status: 0 the stub was written, 2 a file or an argument was refused."
        ),
    )
    run.add_argument("media", nargs="+", type=Path, help="Media files on the hub (the proxies).")
    run.add_argument("--out", type=Path, required=True, help="Where to write the stub JSON.")
    run.add_argument("--recording", required=True, help="The recording id of every source.")
    run.add_argument("--project", required=True, help="The Resolve project name.")
    run.add_argument("--timeline", required=True, help="The timeline name.")
    run.add_argument("--kind", choices=("longform", "short"), default="longform")
    run.add_argument("--profile", default="ccn", help="The rule profile name (default ccn).")
    run.add_argument("--camera", action="append", help="Camera label per media file (A, B...).")
    run.add_argument(
        "--shot-size",
        action="append",
        choices=("close", "medium", "wide", "unknown"),
        help="Shot size per media file.",
    )
    run.add_argument("--source-id", action="append", help="Source id per media file.")
    run.add_argument(
        "--resolution", help="WIDTHxHEIGHT (default: the first file's size, long-form only)."
    )
    run.add_argument(
        "--base", type=Path, help="JSON whose template, head, outro and audio_tracks pass through."
    )
    run.add_argument(
        "--allow-unknown-timecode",
        action="store_true",
        help="Write a source whose start timecode cannot be read (as the export's opt-out).",
    )
    run.add_argument("--force", action="store_true", help="Replace an existing --out.")
    run.add_argument("--json", action="store_true", help="Print the report as JSON.")
    run.set_defaults(func=_cmd_stub)


def _refuse(error: dict[str, Any], as_json: bool) -> int:
    if as_json:
        cli_io.emit(_PROG, json.dumps({"ok": False, "error": error}, indent=2) + "\n")
        return EXIT_REFUSED
    cli_io.warn(f"error: {error['code']}: {error['message']}")
    if error.get("fix_hint"):
        cli_io.warn(f"hint: {error['fix_hint']}")
    return EXIT_REFUSED


def _bad(message: str, hint: str = "") -> dict[str, Any]:
    return {"code": "invalid_argument", "message": message, "fix_hint": hint, "details": {}}


def _per_file(values: list[str] | None, count: int, flag: str) -> list[Any]:
    if values is None:
        return [None] * count
    if len(values) != count:
        raise ValueError(f"{flag} is given {len(values)} time(s) for {count} media file(s)")
    return values


def _read_base(path: Path) -> Any:
    with path.open("rb") as handle:
        raw = handle.read(MAX_BASE_BYTES + 1)
    if len(raw) > MAX_BASE_BYTES:
        raise ValueError(f"the base {path} is over {MAX_BASE_BYTES} bytes")
    return json.loads(raw.decode("utf-8"))


def _cmd_stub(args: argparse.Namespace) -> int:
    as_json: bool = args.json
    count = len(args.media)
    try:
        cameras = _per_file(args.camera, count, "--camera")
        sizes = _per_file(args.shot_size, count, "--shot-size")
        ids = _per_file(args.source_id, count, "--source-id")
    except ValueError as exc:
        return _refuse(_bad(str(exc), "give it once per media file, or not at all"), as_json)
    resolution = None
    if args.resolution is not None:
        match = _RESOLUTION.fullmatch(args.resolution)
        if match is None:
            return _refuse(_bad("--resolution must be WIDTHxHEIGHT, e.g. 1920x1080"), as_json)
        resolution = (int(match.group(1)), int(match.group(2)))
    if args.out.exists() and not args.force:
        return _refuse(_bad(f"{args.out} exists", "pass --force to replace it"), as_json)
    base = None
    if args.base is not None:
        try:
            base = _read_base(args.base)
        except (OSError, ValueError) as exc:  # JSONDecodeError and UnicodeDecodeError too
            return _refuse(_bad(f"cannot read the base {args.base}: {exc}"), as_json)
    sources = [
        StubSource(media, camera=camera, shot_size=size or "unknown", id=sid)
        for media, camera, size, sid in zip(args.media, cameras, sizes, ids, strict=True)
    ]
    try:
        result = build_stub(
            sources,
            recording_id=args.recording,
            project=args.project,
            timeline=args.timeline,
            kind=args.kind,
            profile=args.profile,
            resolution=resolution,
            base=base,
            allow_unknown_timecode=args.allow_unknown_timecode,
        )
    except StubError as exc:
        return _refuse(exc.to_dict(), as_json)
    try:
        write_atomic(args.out, result.plan.model_dump_json(indent=2, exclude_none=True) + "\n")
    except OSError as exc:
        cli_io.warn(f"error: cannot write {args.out}: {exc}")
        return cli_io.EXIT_OUTPUT_ERROR
    report = {"ok": True, "out": str(args.out), **result.report}
    if as_json:
        return cli_io.emit(_PROG, json.dumps(report, indent=2) + "\n") or EXIT_OK
    lines = [f"wrote {args.out}: {report['fps']} fps, {report['resolution']}"]
    for source in report["sources"]:
        tc = source["start_timecode"]
        label = tc["timecode"] if tc else "unknown"
        lines.append(
            f"  {source['id']}: {source['duration']} frames, start {label}, {source['media']}"
        )
    for warning in report["warnings"]:
        lines.append(f"WARN  {warning['code']}: {warning['message']}")
    return cli_io.emit(_PROG, "\n".join(lines) + "\n") or EXIT_OK
