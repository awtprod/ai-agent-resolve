"""``aar drp diff A B``: semantic diff of two files, with ``diff(1)`` exit codes."""

from __future__ import annotations

import argparse
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
    "EXIT_OUTPUT_ERROR",
    "EXIT_PIPE_CLOSED",
    "EXIT_SAME",
    "register",
]

EXIT_SAME = 0
EXIT_DIFFERENT = 1
EXIT_GATE_FAILED = 1  # --gate: the diff cannot vouch for every change (reasons are printed)
EXIT_ERROR = 2
EXIT_OUTPUT_ERROR = 74  # EX_IOERR from sysexits.h: stdout is closed, full or failing (as S1h)
EXIT_PIPE_CLOSED = 141  # 128 + SIGPIPE, what a shell reports for a reader that went away

EXIT_STATUS_HELP = """\
exit status:
  0    no differences (with --gate: every change is decoded and matched by DbId)
  1    differences found (with --gate: the gate failed; the reasons are printed)
  2    error: a file cannot be read whole, usage error, or an internal error
  74   the output could not be written (stdout closed, disk full, I/O error)
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
        # JSON is pure ASCII (\\u escapes), so it stays valid JSON in any output encoding.
        text = (
            json.dumps(result.to_dict(), indent=1, ensure_ascii=True) + "\n"
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
    written = _emit(text)
    if written != EXIT_SAME:
        return written  # 141 (pipe closed) or 74 (output error); never 1, which means "different"
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
    """Best-effort line on stderr; silent if stderr is closed or failing too."""
    stream = sys.stderr
    if stream is None:
        return
    try:
        stream.write(_safe(text, stream) + "\n")
        stream.flush()
    except (OSError, ValueError):
        # stderr is closed or failing (2>&- or 2>/dev/full): point it at /dev/null so the
        # interpreter's own flush at exit cannot fail again and turn the exit code into 120.
        _silence(stream)


def _emit(text: str) -> int:
    """Write ``text`` to stdout.

    A closed pipe ends quietly with ``EXIT_PIPE_CLOSED``. A missing stdout (``>&-``) or a
    failing one (``>/dev/full``, I/O error) ends with ``EXIT_OUTPUT_ERROR`` and one line on
    stderr. Never a traceback, and never exit 1.
    """
    stream = sys.stdout
    if stream is None:
        _err("aar drp diff: stdout is closed; nothing written")
        return EXIT_OUTPUT_ERROR
    try:
        stream.write(_safe(text, stream))
        stream.flush()
    except BrokenPipeError:
        _silence(sys.stdout)
        return EXIT_PIPE_CLOSED
    except (OSError, ValueError) as exc:  # ValueError: write to a closed file object
        _silence(sys.stdout)
        _err(f"aar drp diff: cannot write output: {exc}")
        return EXIT_OUTPUT_ERROR
    return EXIT_SAME


def _silence(stream: TextIO | None) -> None:
    """Point a std stream at /dev/null so the interpreter's final flush cannot fail again."""
    try:
        fd = stream.fileno()  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError):
        return  # not backed by a file descriptor, so nothing is left to flush into it
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
    finally:
        os.close(devnull)
