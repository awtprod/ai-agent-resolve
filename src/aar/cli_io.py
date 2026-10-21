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

__all__ = ["EXIT_OUTPUT_ERROR", "EXIT_PIPE_CLOSED", "emit", "finalize", "warn"]

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


def _flush(stream: TextIO | None, prog: str | None = None) -> int:
    """Flush a std stream; return 0, 141 (broken pipe) or 74 (closed/full/failing). Silences
    the stream on failure so the interpreter's own exit-time flush cannot fail (exit 120).
    With ``prog``, a 74 also puts the standard one-line reason on stderr (141 stays quiet)."""
    if stream is None:
        if prog is not None:
            warn(f"{prog}: stdout is closed; nothing written")
        return EXIT_OUTPUT_ERROR
    try:
        stream.flush()
    except BrokenPipeError:
        _silence(stream)
        return EXIT_PIPE_CLOSED
    except (OSError, ValueError) as exc:
        _silence(stream)
        if prog is not None:
            warn(f"{prog}: cannot write output: {exc}")
        return EXIT_OUTPUT_ERROR
    return 0


def finalize(code: int, prog: str = "aar") -> int:
    """Flush stdout and stderr after a path that wrote with bare ``print`` (argparse help,
    usage and version); return the exit code. Never an exit-120 flush failure.

    A non-zero ``code`` always wins: a usage error (2) stays 2 whatever state stdout and stderr
    are in. That includes stdout closed or full, where a drp input error also stays 2, and
    stderr closed, where argparse prints its usage text to stdout instead (``print_usage(None)``
    falls back to ``sys.stdout``) and that flush fails. Only a success (``code`` 0) is overridden,
    by 74 (stdout closed, full or failing, with the same ``prog: cannot write output: ...`` line
    on stderr as every other command prints) or 141 (the reader went away, quiet). A failed
    stderr flush is silenced and never changes the code."""
    out = _flush(sys.stdout, prog if code == 0 else None)
    _flush(sys.stderr)
    return code or out
