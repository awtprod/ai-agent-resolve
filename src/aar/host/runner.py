"""The runner seam: how an argv actually runs.

The transport never calls :mod:`subprocess` itself. It takes a :class:`Runner` and asks it to
run an argv list, so tests can substitute a fake that answers from a script instead of touching a
host. The real runner, :class:`SubprocessRunner`, runs the argv with no shell (``shell=False``,
argument list) and a timeout. ssh/scp options such as ``-F``, ``BatchMode`` and ``ConnectTimeout``
are supplied by the transport in the argv, not here.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["RunResult", "Runner", "SubprocessRunner"]


@dataclass(frozen=True)
class RunResult:
    """The outcome of one run: exit code and captured output, or ``timed_out`` set."""

    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


@runtime_checkable
class Runner(Protocol):
    """Runs an argv list with optional stdin and timeout, returning a :class:`RunResult`."""

    def run(
        self, argv: list[str], *, stdin: bytes | None = None, timeout: float | None = None
    ) -> RunResult: ...


class SubprocessRunner:
    """The production runner. Never uses a shell; the argv is passed as a list to the OS."""

    def run(
        self, argv: list[str], *, stdin: bytes | None = None, timeout: float | None = None
    ) -> RunResult:
        # When the caller supplies no stdin, give the child /dev/null rather than the parent's
        # stdin: ssh/scp must not consume our stdin, and leaving the remote PowerShell's stdin
        # open-but-idle can hang Windows PowerShell.
        try:
            completed = subprocess.run(  # argv list, shell=False, never a user shell
                argv,
                input=stdin if stdin is not None else None,
                stdin=subprocess.DEVNULL if stdin is None else None,
                capture_output=True,
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, bytes) else b""
            err = exc.stderr if isinstance(exc.stderr, bytes) else b""
            return RunResult(returncode=-1, stdout=out, stderr=err, timed_out=True)
        return RunResult(completed.returncode, completed.stdout, completed.stderr)
