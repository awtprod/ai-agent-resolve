"""Shared stdout/stderr handling for the ``aar`` CLIs: consistent exit codes, no traceback.

The ``drp`` group grew this handling first (closed/full/failing stdout, broken pipes); this
module offers the same behaviour to the other groups. A closed reader pipe ends quietly with
:data:`EXIT_PIPE_CLOSED`; a missing (``>&-``), full (``>/dev/full``) or failing stdout ends with
:data:`EXIT_OUTPUT_ERROR` and one line on stderr. Never a traceback, and never the interpreter's
own exit-120 flush failure: after a failed write the stream's descriptor is pointed at
``/dev/null`` so the final flush cannot fail again.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

__all__ = ["EXIT_OUTPUT_ERROR", "EXIT_PIPE_CLOSED", "emit", "warn"]

EXIT_OUTPUT_ERROR = 74  # EX_IOERR from sysexits.h: stdout is closed, full or failing
EXIT_PIPE_CLOSED = 141  # 128 + SIGPIPE, what a shell reports for a reader that went away


def _safe(text: str, stream: TextIO) -> str:
    """Make ``text`` encodable on ``stream``; lone surrogates etc. become ``\\uXXXX`` escapes."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return text.encode(encoding, "backslashreplace").decode(encoding, "replace")


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


def warn(line: str) -> None:
    """Best-effort ``line`` on stderr (written as-is); silent if stderr is closed or failing."""
    stream = sys.stderr
    if stream is None:
        return
    try:
        stream.write(_safe(line, stream) + "\n")
        stream.flush()
    except (OSError, ValueError):
        _silence(stream)


def emit(prog: str, text: str) -> int:
    """Write ``text`` to stdout; return ``0``, :data:`EXIT_PIPE_CLOSED` or
    :data:`EXIT_OUTPUT_ERROR`. The caller maps ``0`` to its own success/semantic exit code."""
    stream = sys.stdout
    if stream is None:
        warn(f"{prog}: stdout is closed; nothing written")
        return EXIT_OUTPUT_ERROR
    try:
        stream.write(_safe(text, stream))
        stream.flush()
    except BrokenPipeError:
        _silence(sys.stdout)
        return EXIT_PIPE_CLOSED
    except (OSError, ValueError) as exc:  # ValueError: write to a closed file object
        _silence(sys.stdout)
        warn(f"{prog}: cannot write output: {exc}")
        return EXIT_OUTPUT_ERROR
    return 0
