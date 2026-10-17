"""``aar drp diff A B``: semantic diff of two files, with ``diff(1)`` exit codes."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import TextIO

from aar.drp.diff import SECTIONS, diff_projects
from aar.drp.errors import DrpError
from aar.drp.reader import read_drp

__all__ = [
    "EXIT_DIFFERENT",
    "EXIT_ERROR",
    "EXIT_GATE_FAILED",
    "EXIT_PIPE_CLOSED",
    "EXIT_SAME",
    "register",
]

EXIT_SAME = 0
EXIT_DIFFERENT = 1
EXIT_GATE_FAILED = 1  # --gate: the diff cannot vouch for every change (reasons are printed)
EXIT_ERROR = 2
EXIT_PIPE_CLOSED = 141  # 128 + SIGPIPE, what a shell reports for a reader that went away

EXIT_STATUS_HELP = """\
exit status:
  0    no differences (with --gate: every change is decoded and matched by DbId)
  1    differences found (with --gate: the gate failed; the reasons are printed)
  2    error: a file cannot be read whole, usage error, or an internal error
  141  the output pipe was closed early (for example by `head`)
--only filters what is shown; the exit status always reflects the whole diff.
"""

# Characters the text rendering uses that a narrow terminal encoding may lack.
_ASCII = str.maketrans({"→": "->", "…": "..."})


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``diff`` to the ``aar drp`` subcommands."""
    p = sub.add_parser(
        "diff",
        help="semantic diff of two files (exit 0 same, 1 different, 2 error)",
        description=(
            "Compare two .drp/.drt/.drb files: header, media pool, timelines, markers, and "
            "the raw bytes of everything not decoded."
        ),
        epilog=EXIT_STATUS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("a", help="the earlier file")
    p.add_argument("b", help="the later file")
    p.add_argument("--json", action="store_true", help="print the diff as JSON")
    p.add_argument(
        "--only",
        action="append",
        choices=SECTIONS,
        metavar="{" + ",".join(SECTIONS) + "}",
        help="show only this section (repeatable); hidden changes are counted, not dropped",
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="exit 0 only if every change is decoded and DbId-matched (S8 post-write check)",
    )
    p.set_defaults(func=_diff)


def _diff(args: argparse.Namespace) -> int:
    try:
        a = read_drp(args.a)
        b = read_drp(args.b)
        result = diff_projects(a, b, gate=args.gate)
        if args.only:
            result = result.only(args.only)
        text = (
            json.dumps(result.to_dict(), indent=1, ensure_ascii=False) + "\n"
            if args.json
            else result.to_text()
        )
    except DrpError as exc:
        _err(f"aar drp diff: {exc}")
        return EXIT_ERROR
    except Exception as exc:  # an internal error must never look like "different" (1)
        _err(f"aar drp diff: internal error: {type(exc).__name__}: {exc}")
        _err(traceback.format_exc())
        return EXIT_ERROR
    if _emit(text) == EXIT_PIPE_CLOSED:
        return EXIT_PIPE_CLOSED
    if args.gate:
        return EXIT_SAME if result.gate.passed else EXIT_GATE_FAILED
    return EXIT_SAME if result.identical else EXIT_DIFFERENT


def _safe(text: str, stream: TextIO) -> str:
    """Make ``text`` encodable on ``stream``: arrows become ASCII, anything else is escaped."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        pass
    text = text.translate(_ASCII)
    try:
        return text.encode(encoding, "backslashreplace").decode(encoding, "replace")
    except LookupError:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def _err(text: str) -> None:
    with contextlib.suppress(BrokenPipeError):
        print(_safe(text, sys.stderr), file=sys.stderr)


def _emit(text: str) -> int:
    """Write ``text`` to stdout. A closed pipe ends quietly with ``EXIT_PIPE_CLOSED``."""
    try:
        sys.stdout.write(_safe(text, sys.stdout))
        sys.stdout.flush()
    except BrokenPipeError:
        _silence_stdout()
        return EXIT_PIPE_CLOSED
    return EXIT_SAME


def _silence_stdout() -> None:
    """Point stdout at /dev/null so the interpreter's final flush cannot fail again."""
    try:
        fd = sys.stdout.fileno()
    except (OSError, ValueError):
        return  # not backed by a file descriptor, so nothing is left to flush into the pipe
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
    finally:
        os.close(devnull)
