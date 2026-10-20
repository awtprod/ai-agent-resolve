"""``aar analyze``: the cold analysis workers on the command line. Output is JSON on stdout.

Exit status, as ``aar qa check``:

- 0: the analysis ran and every result is usable (``sync``: every pair is ``ok``);
- 1: it ran and something failed (``sync``: a pair is ``drift``, ``low-confidence``,
  ``no-overlap`` or ``failed``; ``transcript``, ``faces``, ``laughs``: the decode, the
  model, the detector or the Gemini call failed), including media that does not decode;
- 2: an input, a tool or a model cannot be located or opened, an extra is missing, a checkpoint
  does not match, an argument is bad, a paid call was not allowed (``--allow-paid``) or its
  key is missing;
- 74: the result could not be written (``output-error``; a checkpoint is kept).

An error prints ``{"ok": false, "error": {"type", "message", "hint"}}``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar import cli_io
from aar.analyze.errors import EXIT_FAILED, AnalyzeError
from aar.analyze.faces import MODEL_ENV as FACE_MODEL_ENV
from aar.analyze.faces import FaceSettings, analyse_faces
from aar.analyze.gemini import KEY_ENV
from aar.analyze.laughs import LaughSettings, analyse_laughs
from aar.analyze.sync import SyncSettings, sync_files
from aar.analyze.whisper import (
    CCN_PRESET,
    MODEL_DIR_ENV,
    RecordingOffset,
    WhisperSettings,
    explicit_offset,
    offset_from_plan,
    transcribe_file,
    write_atomic,
)
from aar.plan.model import EditPlan, Fps

_PROG = "aar analyze"
EXIT_OK = 0
_EXITS = (
    "Exit status: 0 every result is usable, 1 the analysis ran and something failed "
    "(including media that does not decode), 2 an input, tool, model or extra is missing, "
    "or an argument is bad."
)


def register(subparsers: Any) -> None:
    """Add the ``analyze`` command group to the top-level parser."""
    analyze = subparsers.add_parser(
        "analyze",
        help="Cold analysis: audio sync offsets, word-level transcripts, faces and laughs.",
    )
    sub = analyze.add_subparsers(dest="analyze_command", metavar="COMMAND", required=True)
    _register_sync(sub)
    _register_transcript(sub)
    _register_faces(sub)
    _register_laughs(sub)


def _register_sync(sub: Any) -> None:
    d = SyncSettings()
    run = sub.add_parser(
        "sync",
        help="Audio offsets of other recordings against a reference, with a confidence.",
        description=(
            "offset_s is where the other file's first sample falls on the reference's clock; "
            "suggested_recording_origin = --ref-origin + offset_s * fps. " + _EXITS
        ),
    )
    run.add_argument("--ref", required=True, type=Path, help="The reference media file.")
    run.add_argument(
        "--other", required=True, action="append", type=Path, help="Another file (repeatable)."
    )
    run.add_argument("--fps", required=True, help="Frame rate for frames, e.g. 24000/1001.")
    run.add_argument(
        "--ref-origin", type=int, default=0, help="The reference's recording_origin (frames)."
    )
    run.add_argument(
        "--max-offset-s",
        type=float,
        default=d.max_offset_s,
        help=f"Search offsets within ± this many seconds (default {d.max_offset_s:g}).",
    )
    run.add_argument("--min-ratio", type=float, default=d.min_ratio, help="Peak ratio needed.")
    run.add_argument(
        "--min-agreement",
        type=float,
        default=d.min_agreement,
        help="Fraction of eligible anchor windows that must agree.",
    )
    run.add_argument(
        "--anchors",
        type=int,
        default=d.anchors,
        help=f"Anchor windows spread over the reference (default {d.anchors}); a file much "
        "shorter than the reference needs more, so that two fall inside it.",
    )
    run.add_argument(
        "--max-duration-s",
        type=float,
        default=d.max_duration_s,
        help="Read at most this much of each file (memory: 115 MB per hour per file).",
    )
    run.add_argument("--timeout-s", type=float, help="Timeout of each decode (default: scaled).")
    run.add_argument("--out", type=Path, help="Also write the JSON report to this file.")
    run.set_defaults(func=_cmd_sync)


def _register_transcript(sub: Any) -> None:
    d = CCN_PRESET
    run = sub.add_parser(
        "transcript",
        help="Word-level transcript (faster-whisper) in recording time; resumable.",
        description=(
            "Writes an aar Transcript JSON to --out and prints a report. The model is a name "
            f"looked up under --model-dir or ${MODEL_DIR_ENV}, or a model directory; it is "
            "never downloaded. " + _EXITS
        ),
    )
    run.add_argument("media", type=Path, help="The media file to transcribe.")
    run.add_argument("--out", required=True, type=Path, help="Transcript JSON to write.")
    run.add_argument("--model", default=d.model, help=f"Model name or path (default {d.model}).")
    run.add_argument("--model-dir", type=Path, help=f"Model directory (default ${MODEL_DIR_ENV}).")
    run.add_argument("--threads", type=int, default=d.cpu_threads, help="CPU threads (0: default).")
    _add_offset_args(run, "the transcript")
    run.add_argument("--resume", action="store_true", help="Continue from the checkpoint.")
    run.add_argument("--beam-size", type=int, default=d.beam_size)
    run.add_argument("--language", default=d.language, help="Language code ('' to detect).")
    run.add_argument("--no-vad", action="store_true", help="Turn the VAD filter off.")
    run.add_argument("--vad-min-silence-ms", type=int, default=d.vad_min_silence_ms)
    run.add_argument("--condition-on-previous-text", action="store_true")
    run.add_argument("--device", default=d.device)
    run.add_argument("--compute-type", default=d.compute_type)
    run.add_argument("--no-speech-prob-max", type=float, default=d.no_speech_prob_max)
    run.add_argument("--avg-logprob-min", type=float, default=d.avg_logprob_min)
    run.add_argument("--compression-ratio-max", type=float, default=d.compression_ratio_max)
    run.add_argument("--max-duration-s", type=float, default=d.max_duration_s)
    run.add_argument("--timeout-s", type=float, help="Timeout of the decode (default: scaled).")
    run.set_defaults(func=_cmd_transcript)


def _add_offset_args(run: Any, what: str) -> None:
    """Recording time: ``--plan`` with ``--source``, or ``--offset-s``, and ``--recording``."""
    run.add_argument("--plan", type=Path, help="EditPlan JSON: take the offset from --source.")
    run.add_argument("--source", help="The plan source this file is (with --plan).")
    run.add_argument(
        "--offset-s", type=float, help="Seconds to add to file time (not with --plan)."
    )
    run.add_argument("--recording", help=f"Recording name for {what} (default: the plan's).")


def _register_faces(sub: Any) -> None:
    d = FaceSettings()
    run = sub.add_parser(
        "faces",
        help="Faces in sampled frames (OpenCV YuNet), in recording time; resumable.",
        description=(
            "Writes aar.faces/1 JSON to --out and prints a short report. The YuNet model is "
            f"--model or ${FACE_MODEL_ENV}, an existing .onnx file; it is never downloaded. "
            "Needs the 'faces' extra. " + _EXITS
        ),
    )
    run.add_argument("media", type=Path, help="The video file.")
    run.add_argument("--out", required=True, type=Path, help="aar.faces/1 JSON to write.")
    run.add_argument("--model", type=Path, help=f"YuNet .onnx file (default ${FACE_MODEL_ENV}).")
    run.add_argument(
        "--sample-fps",
        default="2",
        help="Frames sampled per second, e.g. 2 or 1/2 (default 2).",
    )
    run.add_argument("--max-side", type=int, default=d.max_side, help="Longest frame side.")
    run.add_argument("--score", type=float, default=d.score, help="Detector score threshold.")
    run.add_argument("--nms", type=float, default=d.nms, help="Detector NMS threshold.")
    run.add_argument("--top-k", type=int, default=d.top_k, help="Detector top-k before NMS.")
    run.add_argument("--resume", action="store_true", help="Continue from the checkpoint.")
    _add_offset_args(run, "the faces")
    run.add_argument("--timeout-s", type=float, help="Timeout of the decode (default: scaled).")
    run.set_defaults(func=_cmd_faces)


def _register_laughs(sub: Any) -> None:
    d = LaughSettings()
    run = sub.add_parser(
        "laughs",
        help="Audience laughs from Gemini (paid), checked against the audio envelope.",
        description=(
            "Writes aar.laughs/1 JSON to --out (and a LaughMap with --map-out). Calling Gemini "
            f"is paid and needs --allow-paid and ${KEY_ENV} (environment only, no flag); "
            "--from-raw re-parses a saved reply offline. " + _EXITS
        ),
    )
    run.add_argument("media", type=Path, help="The media file (its first audio stream).")
    run.add_argument("--out", required=True, type=Path, help="aar.laughs/1 JSON to write.")
    run.add_argument(
        "--work-dir", required=True, type=Path, help="Where audio.flac and raw.json go."
    )
    paid = run.add_mutually_exclusive_group()
    paid.add_argument("--allow-paid", action="store_true", help="Allow the paid Gemini call.")
    paid.add_argument("--from-raw", type=Path, help="Re-parse a saved raw.json; no network.")
    run.add_argument("--model", default=d.model, help=f"Gemini model id (default {d.model}).")
    run.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Put unconfirmed and unchecked laughs in the map too.",
    )
    run.add_argument("--map-out", type=Path, help="Also write a LaughMap JSON here.")
    run.add_argument("--confirm-bump-db", type=float, default=d.confirm_bump_db)
    run.add_argument("--confirm-frac", type=float, default=d.confirm_frac)
    run.add_argument(
        "--max-audio-s",
        type=float,
        default=d.max_audio_s,
        help="Refuse longer audio before uploading it (default 4 h).",
    )
    _add_offset_args(run, "the laugh map")
    run.add_argument("--timeout-s", type=float, help="Timeout of each ffmpeg call.")
    run.set_defaults(func=_cmd_laughs)


# ---------------------------------------------------------------- output


def _emit(body: dict[str, Any]) -> int:
    # json.dumps, not pydantic: a file name that is not valid UTF-8 carries lone surrogates,
    # which cli_io.emit escapes.
    return cli_io.emit(_PROG, json.dumps(body, indent=2, ensure_ascii=False) + "\n")


def _fail(exc: AnalyzeError) -> int:
    return _emit({"ok": False, "error": exc.to_dict()}) or exc.exit_code


def _bad(message: str) -> AnalyzeError:
    return AnalyzeError("invalid-argument", message)


def _check_out(path: Path | None) -> None:
    """Refuse an output whose folder does not exist before any long analysis starts."""
    if path is None:
        return
    folder = path.parent if str(path.parent) else Path(".")
    if not folder.is_dir():
        raise _bad(f"the folder of {path} does not exist")
    if path.exists() and not path.is_file():
        raise _bad(f"{path} exists and is not a regular file")


def _check_input(path: Path, what: str) -> None:
    try:
        if not path.exists():
            raise AnalyzeError("input-missing", f"{what} not found: {path}")
        if not path.is_file():
            raise AnalyzeError("input-missing", f"{what} is not a regular file: {path}")
    except OSError as exc:
        raise AnalyzeError("input-missing", f"cannot read {what} {path}: {exc}") from None


def _parse_fps(text: str) -> Fraction:
    try:
        fps = Fps.model_validate(text)
    except ValidationError:
        raise _bad(f"--fps {text!r} is not 'N' or 'N/D' with N, D > 0") from None
    return Fraction(fps.num, fps.den)


# ---------------------------------------------------------------- sync


def _cmd_sync(args: argparse.Namespace) -> int:
    try:
        fps = _parse_fps(args.fps)
        try:
            settings = SyncSettings(
                max_offset_s=args.max_offset_s,
                min_ratio=args.min_ratio,
                min_agreement=args.min_agreement,
                anchors=args.anchors,
                max_duration_s=args.max_duration_s,
                timeout_s=args.timeout_s,
            )
        except ValueError as exc:
            raise _bad(str(exc)) from None
        _check_out(args.out)
        _check_input(args.ref, "reference")
        for other in args.other:
            _check_input(other, "file")
        report = sync_files(
            args.ref, list(args.other), fps=fps, ref_origin=args.ref_origin, settings=settings
        )
    except AnalyzeError as exc:
        return _fail(exc)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out is not None:
        try:
            write_atomic(args.out, text)
        except OSError as exc:
            return _fail(AnalyzeError("output-error", f"cannot write {args.out}: {exc}"))
    written = cli_io.emit(_PROG, text)
    if written != EXIT_OK:
        return written
    return EXIT_OK if report["ok"] else EXIT_FAILED


# ---------------------------------------------------------------- transcript


def _settings(args: argparse.Namespace) -> WhisperSettings:
    values = {
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "cpu_threads": args.threads,
        "beam_size": args.beam_size,
        "language": args.language or None,
        "vad_filter": not args.no_vad,
        "vad_min_silence_ms": args.vad_min_silence_ms,
        "condition_on_previous_text": args.condition_on_previous_text,
        "no_speech_prob_max": args.no_speech_prob_max,
        "avg_logprob_min": args.avg_logprob_min,
        "compression_ratio_max": args.compression_ratio_max,
        "max_duration_s": args.max_duration_s,
        "timeout_s": args.timeout_s,
    }
    assert set(values) <= {f.name for f in fields(WhisperSettings)}
    try:
        return replace(CCN_PRESET, **values)
    except ValueError as exc:
        raise _bad(str(exc)) from None


def _offset(args: argparse.Namespace) -> RecordingOffset:
    if args.offset_s is not None and (args.plan is not None or args.source is not None):
        raise _bad("--offset-s and --plan/--source are mutually exclusive")
    if (args.plan is None) != (args.source is None):
        raise _bad("--plan and --source go together")
    if args.offset_s is not None:
        return explicit_offset(args.offset_s)
    if args.plan is None:
        return RecordingOffset()
    _check_input(args.plan, "plan")
    try:
        plan = EditPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # ValidationError is a ValueError
        raise AnalyzeError("invalid-plan", f"cannot use the plan {args.plan}: {exc}") from None
    return offset_from_plan(plan, args.source)


def _cmd_transcript(args: argparse.Namespace) -> int:
    try:
        settings = _settings(args)
        offset = _offset(args)
        _check_out(args.out)
        _check_input(args.media, "input")
        report = transcribe_file(
            args.media,
            args.out,
            settings=settings,
            model_dir=args.model_dir,
            offset=offset,
            recording=args.recording,
            resume=args.resume,
        )
    except AnalyzeError as exc:
        return _fail(exc)
    except OSError as exc:  # the checkpoint beside --out could not be written
        return _fail(AnalyzeError("output-error", f"cannot write beside {args.out}: {exc}"))
    return _emit(report) or EXIT_OK


# ---------------------------------------------------------------- faces


def _face_settings(args: argparse.Namespace) -> FaceSettings:
    try:
        sample_fps = Fraction(args.sample_fps)
    except (ValueError, ZeroDivisionError):
        raise _bad(f"--sample-fps {args.sample_fps!r} is not a number or N/D") from None
    try:
        return FaceSettings(
            sample_fps=sample_fps,
            max_side=args.max_side,
            score=args.score,
            nms=args.nms,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        raise _bad(str(exc)) from None


def _cmd_faces(args: argparse.Namespace) -> int:
    try:
        settings = _face_settings(args)
        offset = _offset(args)
        _check_out(args.out)
        _check_input(args.media, "input")
        report = analyse_faces(
            args.media,
            args.out,
            model=args.model,
            settings=settings,
            offset=offset,
            recording=args.recording,
            resume=args.resume,
        )
    except AnalyzeError as exc:
        return _fail(exc)
    except OSError as exc:  # the checkpoint beside --out could not be written
        return _fail(AnalyzeError("output-error", f"cannot write beside {args.out}: {exc}"))
    return _emit(report) or EXIT_OK


# ---------------------------------------------------------------- laughs


def _cmd_laughs(args: argparse.Namespace) -> int:
    try:
        try:
            settings = LaughSettings(
                model=args.model,
                confirm_bump_db=args.confirm_bump_db,
                confirm_frac=args.confirm_frac,
                include_unconfirmed=args.include_unconfirmed,
                max_audio_s=args.max_audio_s,
                timeout_s=args.timeout_s,
            )
        except ValueError as exc:
            raise _bad(str(exc)) from None
        offset = _offset(args)
        _check_out(args.out)
        _check_out(args.map_out)
        _check_input(args.media, "input")
        report = analyse_laughs(
            args.media,
            args.out,
            work_dir=args.work_dir,
            allow_paid=args.allow_paid,
            from_raw=args.from_raw,
            settings=settings,
            offset=offset,
            recording=args.recording,
            map_out=args.map_out,
        )
    except AnalyzeError as exc:
        return _fail(exc)
    except OSError as exc:  # the work dir could not be created or written
        return _fail(AnalyzeError("output-error", f"cannot write the work dir or output: {exc}"))
    return _emit(report) or EXIT_OK
