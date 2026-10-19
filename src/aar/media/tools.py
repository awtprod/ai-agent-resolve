"""Run ffmpeg and ffprobe safely: argument lists, a timeout on every call, bounded output.

The binaries come from ``AAR_FFMPEG`` / ``AAR_FFPROBE`` when set, otherwise from ``PATH``.
Nothing here goes through a shell. Every call has a timeout; on expiry the whole process group
is killed, so a wrapper script cannot leave a child holding the pipes open. stderr is kept as a
bounded tail; stdout is either collected up to a limit or streamed to a sink.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

Tool = Literal["ffmpeg", "ffprobe"]

ENV_VARS: dict[Tool, str] = {"ffmpeg": "AAR_FFMPEG", "ffprobe": "AAR_FFPROBE"}

DEFAULT_TIMEOUT_S = 60.0
STDOUT_LIMIT = 16 * 1024 * 1024
STDERR_LIMIT = 64 * 1024
ERROR_TAIL_CHARS = 2000
_CHUNK = 64 * 1024
_REAP_S = 5.0


class MediaError(RuntimeError):
    """An ffmpeg/ffprobe call failed. Carries the exit code and the end of stderr."""

    def __init__(
        self,
        message: str,
        *,
        tool: str = "",
        exit_code: int | None = None,
        stderr_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.tool = tool
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail[-ERROR_TAIL_CHARS:]

    def summary(self) -> str:
        """One line: the message plus the last non-empty stderr line, if any."""
        last = next((ln for ln in reversed(self.stderr_tail.splitlines()) if ln.strip()), "")
        return f"{self}: {last.strip()}" if last else str(self)


class MediaToolMissing(MediaError):
    """The binary is not on PATH, or the ``AAR_FF*`` override does not point at a program."""


class MediaTimeout(MediaError):
    """The call ran past its timeout and was killed."""

    def __init__(
        self, message: str, *, timeout_s: float, tool: str = "", stderr_tail: str = ""
    ) -> None:
        super().__init__(message, tool=tool, exit_code=None, stderr_tail=stderr_tail)
        self.timeout_s = timeout_s


class OutputTooLarge(MediaError):
    """stdout passed its limit; the process was killed."""


class MediaInputError(MediaError):
    """The input is not something ffmpeg should be pointed at (missing, not a regular file)."""


@dataclass(frozen=True)
class RunResult:
    """What a finished call left behind."""

    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: str
    stderr_truncated: bool
    elapsed_s: float

    @property
    def stderr_tail(self) -> str:
        return self.stderr[-ERROR_TAIL_CHARS:]


def resolve_tool(tool: Tool) -> str:
    """Absolute path of ``tool``: the ``AAR_FF*`` override, otherwise the first on PATH."""
    var = ENV_VARS[tool]
    override = os.environ.get(var, "").strip()
    if override:
        found = shutil.which(override)
        if found is None:
            raise MediaToolMissing(f"{var}={override!r} is not an executable program", tool=tool)
        return found
    found = shutil.which(tool)
    if found is None:
        raise MediaToolMissing(f"{tool} not found on PATH (or set {var})", tool=tool)
    return found


def input_url(path: str | os.PathLike[str]) -> str:
    """The argument ffmpeg gets for a local file.

    Absolute and prefixed with ``file:`` so a name starting with ``-`` is never read as an
    option and a name containing ``:`` is never read as a protocol.
    """
    return "file:" + os.path.abspath(os.fspath(path))


def require_regular_file(path: str | os.PathLike[str], tool: str = "") -> Path:
    """Refuse missing paths and anything that is not a regular file (FIFOs, devices, dirs)."""
    p = Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        raise MediaInputError(f"file not found: {p}", tool=tool) from None
    except OSError as exc:
        raise MediaInputError(f"cannot read {p}: {exc.strerror or exc}", tool=tool) from None
    if not stat.S_ISREG(st.st_mode):
        raise MediaInputError(f"not a regular file: {p}", tool=tool)
    return p


class _Tail:
    """Keeps the last ``limit`` bytes written to it."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self.buf = bytearray()
        self.truncated = False

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        extra = len(self.buf) - self.limit
        if extra > 0:
            del self.buf[:extra]
            self.truncated = True

    def text(self) -> str:
        data = bytes(self.buf)
        if self.truncated:
            # Drop the partial first line so parsers only see whole lines.
            cut = data.find(b"\n")
            data = data[cut + 1 :] if cut >= 0 else data
        return data.decode("utf-8", errors="replace")


class _Pump(threading.Thread):
    def __init__(
        self, stream: IO[bytes], feed: Callable[[bytes], None], on_error: Callable[[], None]
    ) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.feed = feed
        self.on_error = on_error
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            while True:
                chunk = self.stream.read1(_CHUNK) if hasattr(self.stream, "read1") else b""
                if not chunk:
                    break
                if self.error is None:
                    try:
                        self.feed(chunk)
                    except BaseException as exc:  # recorded and re-raised by run_tool
                        self.error = exc
                        self.on_error()
        except (OSError, ValueError):
            pass  # the pipe closed under us after a kill


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - the hub is POSIX
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def run_tool(
    tool: Tool,
    args: Sequence[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    stdout_sink: Callable[[bytes], None] | None = None,
    stdout_limit: int = STDOUT_LIMIT,
    stderr_limit: int = STDERR_LIMIT,
    check: bool = True,
) -> RunResult:
    """Run ``tool`` with ``args`` (a list, never a shell string) and wait at most ``timeout_s``.

    stdout goes to ``stdout_sink`` chunk by chunk when given, otherwise it is collected and the
    process is killed if it passes ``stdout_limit`` bytes. stderr keeps only its last
    ``stderr_limit`` bytes (``stderr_truncated`` says whether anything was dropped).

    Raises ``MediaToolMissing``, ``MediaTimeout``, ``OutputTooLarge``, any exception raised by
    the sink, and, when ``check`` is true, ``MediaError`` for a non-zero exit.
    """
    if isinstance(args, (str, bytes)) or not all(isinstance(a, str) for a in args):
        raise TypeError("args must be a sequence of str (no shell strings)")
    if not timeout_s or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    exe = resolve_tool(tool)
    argv = (exe, *args)
    kwargs: dict[str, object] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    started = time.monotonic()
    try:
        proc = subprocess.Popen(  # an argument list, never a shell
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            **kwargs,  # type: ignore[call-overload]
        )
    except OSError as exc:
        raise MediaToolMissing(f"cannot start {tool} ({exe}): {exc}", tool=tool) from None

    collected = bytearray()

    def collect(chunk: bytes) -> None:
        if len(collected) + len(chunk) > stdout_limit:
            raise OutputTooLarge(f"{tool} wrote more than {stdout_limit} bytes", tool=tool)
        collected.extend(chunk)

    err = _Tail(stderr_limit)
    assert proc.stdout is not None and proc.stderr is not None
    out_pump = _Pump(proc.stdout, stdout_sink or collect, lambda: _kill(proc))
    err_pump = _Pump(proc.stderr, err.feed, lambda: _kill(proc))
    out_pump.start()
    err_pump.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(proc)
    finally:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_REAP_S)
        out_pump.join(_REAP_S)
        err_pump.join(_REAP_S)
        for stream in (proc.stdout, proc.stderr):
            with contextlib.suppress(OSError):
                stream.close()
    elapsed = time.monotonic() - started
    stderr_text = err.text()
    rc = proc.returncode if proc.returncode is not None else -1
    if timed_out:
        raise MediaTimeout(
            f"{tool} timed out after {timeout_s:g} s",
            timeout_s=timeout_s,
            tool=tool,
            stderr_tail=stderr_text,
        )
    if out_pump.error is not None:
        exc = out_pump.error
        if isinstance(exc, MediaError) and not exc.stderr_tail:
            exc.stderr_tail = stderr_text[-ERROR_TAIL_CHARS:]
        raise exc
    if check and rc != 0:
        raise MediaError(
            f"{tool} exited with status {rc}", tool=tool, exit_code=rc, stderr_tail=stderr_text
        )
    return RunResult(
        argv=argv,
        returncode=rc,
        stdout=bytes(collected),
        stderr=stderr_text,
        stderr_truncated=err.truncated,
        elapsed_s=elapsed,
    )


def timeout_for(
    duration_s: float | None, *, base_s: float = 60.0, per_media_s: float = 2.0
) -> float:
    """A timeout that grows with the media length: ``base_s + per_media_s * duration``."""
    if duration_s is None or duration_s != duration_s or duration_s < 0:
        return base_s
    return base_s + per_media_s * min(duration_s, 24 * 3600.0)
