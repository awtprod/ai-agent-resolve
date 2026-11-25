"""Measure media files on the edit PC for the library writer: one JSON line of facts per file.

``aar media facts PATH... --out FACTS.jsonl`` (also ``python -m aar.media.facts`` from a copy
of ``src`` on the PC, standard library only, like ``aar media shrink`` and ``aar media
receipt``). For each file it records what :func:`aar.drp.mediapool.facts.facts_from_json`
reads on the hub: the path as the edit PC names it, the size, the modification time on the
PC's wall clock and in UTC, the creation time in UTC where the system keeps one, and
``ffprobe -show_format -show_streams`` as JSON. A library project's media-pool clips are
authored from these facts (DESIGN "Media-pool rows (S25c1)"), so they must be measured on the
PC's own file, not on the hub's working copy.

**Read-only.** The files are only stat'ed and handed to ffprobe (an argument list, never a
shell, with a ``file:`` URL so no name reads as an option or a protocol). Nothing is written
but the output file, which is made under a temporary name beside it and given its name without
replacing anything (an existing ``--out`` is refused before any file is measured).

**Bounded.** At most :data:`MAX_FILES` files a run (``--max-files``, refused before any probe
when more are named), each probe killed after ``--timeout`` seconds, each file's line at most
:data:`MAX_LINE_BYTES` (the hub's own limit per line), and the whole output at most
``--max-bytes``: a file whose line would pass it is listed with the error ``output_cap`` instead,
as is every file after it.

**Nothing is skipped silently.** A file that is missing, not a regular file, changed while it
was measured, that ffprobe refuses or that times out is written as a line of its own,
``{"path", "exists", "error": {"code", "message"}}``, printed, and makes the run exit 2. The
hub reads such a line as "measured with an error" and names it when a cut needs that file.

**Deterministic.** Files are measured and written in the order of their PC paths (case
folded, then as given), every JSON object with sorted keys; the same files give the same bytes.

A folder argument takes the media files directly in it (:data:`MEDIA_SUFFIXES`, not hidden,
not in subfolders). ``--pc-folder LOCAL=PC`` names a folder as the edit PC knows it when the
command runs elsewhere (on the PC itself the absolute path is already the PC's).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import stat
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from aar.media.tools import (
    MediaError,
    MediaTimeout,
    MediaToolMissing,
    OutputTooLarge,
    input_url,
    run_tool,
)

__all__ = [
    "MAX_FILES",
    "MAX_LINE_BYTES",
    "MEDIA_SUFFIXES",
    "FactsRunError",
    "main",
    "measure",
    "measure_file",
]

#: The most files one run measures (a show's cameras and sound files fit many times over).
MAX_FILES = 256
#: One file's line at most: the hub's own limit (``aar.drp.mediapool.facts.MAX_FACTS_BYTES``).
MAX_LINE_BYTES = 1024 * 1024
#: The whole output at most, by default.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_S = 60.0
MEDIA_SUFFIXES = frozenset(
    {".mp4", ".m4v", ".mov", ".mxf", ".wav", ".mp3", ".m4a", ".aac", ".png", ".braw"}
)
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class FactsRunError(RuntimeError):
    """The run as a whole is refused (nothing measured): a typed code and a fix hint."""

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "fix_hint": self.hint}


@dataclass(frozen=True)
class Target:
    """One file to measure: where it is here and the path the edit PC names it by."""

    local: Path
    pc_path: str


# --------------------------------------------------------------------------- times


def local_text(seconds: int) -> str:
    """The PC's wall clock as ``Fri Mar 13 04:05:06 2026`` (C ``ctime`` form, English names
    whatever the locale)."""
    t = time.localtime(seconds)
    return (
        f"{_DAYS[t.tm_wday]} {_MONTHS[t.tm_mon - 1]} {t.tm_mday:2d} "
        f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d} {t.tm_year:04d}"
    )


def utc_text(ns: int) -> str:
    """Nanoseconds since 1970 as ``2026-03-13T09:05:06.1234567Z`` (.NET's round-trip form:
    seven fractional digits, the 100 ns a Windows file time holds)."""
    seconds, rest = divmod(ns, 1_000_000_000)
    t = time.gmtime(seconds)
    return (
        f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T{t.tm_hour:02d}:{t.tm_min:02d}:"
        f"{t.tm_sec:02d}.{rest // 100:07d}Z"
    )


# --------------------------------------------------------------------------- paths


def _pc_path(local: Path, folders: Sequence[tuple[str, str]]) -> str | None:
    """The path the edit PC names ``local`` by: under a ``--pc-folder`` (the longest that
    matches), else the absolute path itself when it is a Windows path; None otherwise."""
    absolute = os.path.abspath(local)
    best: tuple[int, str] | None = None
    for here, there in folders:
        here_abs = os.path.abspath(here)
        rel = os.path.relpath(absolute, here_abs) if _inside(absolute, here_abs) else None
        if rel is None:
            continue
        if best is None or len(here_abs) > best[0]:
            parts = [] if rel == "." else Path(rel).parts
            best = (len(here_abs), str(PureWindowsPath(there, *parts)))
    if best is not None:
        return best[1]
    return absolute if _windows_absolute(absolute) else None


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # another drive on Windows
        return False


def _windows_absolute(text: str) -> bool:
    """A drive path (``E:\\...``) or a UNC path (``\\\\server\\share\\...``), backslashes."""
    if "/" in text:
        return False
    win = PureWindowsPath(text)
    return bool(win.drive) and bool(win.root) and len(win.parts) >= 2


def _folder_files(folder: Path) -> list[Path]:
    """The media files directly in ``folder`` (not hidden), by name."""
    found = []
    with os.scandir(folder) as entries:
        for n, entry in enumerate(entries):
            if n >= 20 * MAX_FILES:
                raise FactsRunError(
                    "too_many_entries",
                    f"{str(folder)!r} holds more than {20 * MAX_FILES} entries",
                    "measure a folder of media, or name the files",
                )
            name = entry.name
            if name.startswith(".") or Path(name).suffix.lower() not in MEDIA_SUFFIXES:
                continue
            if entry.is_file(follow_symlinks=True):
                found.append(Path(entry.path))
    return sorted(found, key=lambda p: p.name)


def targets(
    paths: Sequence[Path], folders: Sequence[tuple[str, str]], *, max_files: int = MAX_FILES
) -> tuple[list[Target], int]:
    """Every file to measure, in output order, and how many repeated paths were dropped.
    ``too_many_files`` past ``max_files``; ``not_a_pc_path`` for a file the PC has no name for.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(_folder_files(path))
        else:
            files.append(path)  # missing or odd files are measured, and listed with the error
        if len(files) > max_files:
            raise FactsRunError(
                "too_many_files",
                f"more than {max_files} files are named; one run measures at most {max_files}",
                "measure the files in several runs (one facts file each), or raise --max-files",
            )
    out: dict[str, Target] = {}
    repeated = 0
    for local in files:
        pc = _pc_path(local, folders)
        if pc is None:
            raise FactsRunError(
                "not_a_pc_path",
                f"{str(local)!r} is not a Windows path, so the edit PC's name for it is unknown",
                "run this on the edit PC, or name its folder there with --pc-folder LOCAL=PC",
            )
        key = pc.casefold()
        if key in out:
            repeated += 1
            continue
        out[key] = Target(local, pc)
    ordered = sorted(out.values(), key=lambda t: (t.pc_path.casefold(), t.pc_path))
    return ordered, repeated


# --------------------------------------------------------------------------- one file


def _identity(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _failed(target: Target, code: str, message: str, *, exists: bool = True) -> dict[str, Any]:
    return {
        "path": target.pc_path,
        "exists": exists,
        "error": {"code": code, "message": message[:2000]},
    }


def measure_file(target: Target, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """The facts of one file, or its error line (module docstring); never raises for the
    file's own faults (a missing ffprobe is the run's: :class:`FactsRunError`)."""
    try:
        before = os.stat(target.local)
    except FileNotFoundError:
        return _failed(target, "missing", "the file does not exist", exists=False)
    except OSError as exc:
        return _failed(target, "unreadable", f"cannot read it: {exc.strerror or exc}")
    if not stat.S_ISREG(before.st_mode):
        return _failed(target, "not_a_file", "it is not a regular file (a folder or a device)")
    if before.st_size == 0:
        return _failed(target, "empty", "the file is empty")
    try:
        result = run_tool(
            "ffprobe",
            [
                "-hide_banner",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                input_url(target.local),
            ],
            timeout_s=timeout_s,
            stdout_limit=MAX_LINE_BYTES,
        )
    except MediaToolMissing as exc:
        raise FactsRunError(
            "tool_missing", str(exc), "install ffprobe (with ffmpeg) or set AAR_FFPROBE"
        ) from None
    except MediaTimeout:
        return _failed(target, "timeout", f"ffprobe did not finish within {timeout_s:g} s")
    except OutputTooLarge:
        return _failed(target, "too_large", f"ffprobe printed more than {MAX_LINE_BYTES} bytes")
    except MediaError as exc:
        return _failed(target, "probe_failed", exc.summary())
    try:
        data = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return _failed(target, "bad_probe_output", f"ffprobe printed unreadable JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("streams"), list):
        return _failed(target, "bad_probe_output", "ffprobe printed no stream list")
    if not data["streams"]:
        return _failed(target, "no_streams", "ffprobe found no streams in it")
    try:
        after = os.stat(target.local)
    except OSError as exc:
        return _failed(target, "changed_while_measured", f"it went away while measured: {exc}")
    if _identity(after) != _identity(before):
        return _failed(
            target, "changed_while_measured", "it changed while it was measured; measure again"
        )
    facts: dict[str, Any] = {
        "path": target.pc_path,
        "exists": True,
        "size": before.st_size,
        "mtime_local": local_text(before.st_mtime_ns // 1_000_000_000),
        "mtime_utc": utc_text(before.st_mtime_ns),
        "ffprobe": {"format": data.get("format") or {}, "streams": data["streams"]},
    }
    born = getattr(before, "st_birthtime_ns", None)  # Windows (3.12+) and the BSDs keep it
    if isinstance(born, int) and born >= 0:
        facts["ctime_utc"] = utc_text(born)
    return facts


def _line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- the run


def measure(
    files: Sequence[Target],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[bytes, list[dict[str, Any]]]:
    """The facts file's bytes and one summary per file (``path``, ``ok``, ``error``)."""
    out = bytearray()
    summary: list[dict[str, Any]] = []
    capped = False
    for target in files:
        record = (
            _failed(target, "output_cap", "the output reached its size cap before this file")
            if capped
            else measure_file(target, timeout_s=timeout_s)
        )
        line = _line(record)
        if len(line) > MAX_LINE_BYTES:
            record = _failed(target, "too_large", "its facts are longer than one line may be")
            line = _line(record)
        if not capped and len(out) + len(line) > max_bytes:
            capped = True
            record = _failed(target, "output_cap", "the output reached its size cap at this file")
            line = _line(record)
        out += line
        entry: dict[str, Any] = {"path": target.pc_path, "ok": "error" not in record}
        if "error" in record:
            entry["error"] = record["error"]
        summary.append(entry)
    return bytes(out), summary


def write_new(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path``, which must not exist: under a temporary name beside it,
    flushed, then given its name without replacing anything."""
    temp = path.with_name(f".{path.name}.aar-facts-{secrets.token_hex(4)}.part")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "posix":
            os.rename(temp, path)  # Windows: refuses an existing name
        else:
            try:
                os.link(temp, path)  # refuses an existing name
            except FileExistsError:
                raise
            except OSError:  # a file system without hard links
                if os.path.lexists(path):
                    raise FileExistsError(str(path)) from None
                os.rename(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)


def _folder_pair(text: str) -> tuple[str, str]:
    here, sep, there = text.partition("=")
    if not sep or not here or not _windows_absolute(there):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not LOCAL=PC (a local folder, '=', and the folder's absolute "
            "Windows path on the edit PC)"
        )
    return here, there.rstrip("\\") or there


def _positive(kind: type) -> Any:
    def parse(text: str) -> Any:
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
        if not value > 0 or value != value or value == float("inf"):
            raise argparse.ArgumentTypeError(f"{text!r} must be above 0")
        return value

    return parse


DESCRIPTION = (
    "Measure media files on the edit PC (ffprobe, size and times) into a facts file, one JSON "
    "line per file, for the hub's library export. Read-only on the media. Exit 0 when every "
    "file was measured, 2 when any was not (listed with its error) or the run was refused."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", type=Path, nargs="+", help="Media files, or folders of them.")
    parser.add_argument(
        "--out", type=Path, required=True, help="The facts file to write (never overwritten)."
    )
    parser.add_argument(
        "--pc-folder",
        type=_folder_pair,
        action="append",
        default=[],
        metavar="LOCAL=PC",
        help="A local folder and its path on the edit PC (when not run on the PC itself).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive(float),
        default=DEFAULT_TIMEOUT_S,
        help=f"Seconds one ffprobe may take (default {DEFAULT_TIMEOUT_S:g}).",
    )
    parser.add_argument(
        "--max-files",
        type=_positive(int),
        default=MAX_FILES,
        help=f"The most files one run measures (default {MAX_FILES}).",
    )
    parser.add_argument(
        "--max-bytes",
        type=_positive(int),
        default=DEFAULT_MAX_BYTES,
        help=f"The most bytes the facts file may hold (default {DEFAULT_MAX_BYTES}).",
    )
    parser.set_defaults(func=run_command)


def run_command(args: argparse.Namespace) -> int:
    def refuse(error: FactsRunError) -> int:
        print(json.dumps({"ok": False, "error": error.as_dict()}, sort_keys=True))
        return 2

    out: Path = args.out
    if os.path.lexists(out):
        return refuse(
            FactsRunError(
                "output_exists", f"{str(out)!r} exists; it is never overwritten", "give a new name"
            )
        )
    if not out.parent.is_dir():
        return refuse(
            FactsRunError(
                "output_folder_missing",
                f"the folder of {str(out)!r} does not exist",
                "create it first",
            )
        )
    try:
        files, repeated = targets(args.paths, args.pc_folder, max_files=args.max_files)
        data, summary = measure(files, timeout_s=args.timeout, max_bytes=args.max_bytes)
    except FactsRunError as exc:
        return refuse(exc)
    try:
        write_new(out, data)
    except FileExistsError:
        return refuse(
            FactsRunError(
                "output_exists", f"{str(out)!r} appeared while measuring", "give a new name"
            )
        )
    except OSError as exc:
        return refuse(
            FactsRunError(
                "output_unwritable", f"cannot write {str(out)!r}: {exc}", "check the folder"
            )
        )
    for entry in summary:
        print(json.dumps(entry, sort_keys=True, ensure_ascii=False))
    failed = sum(not entry["ok"] for entry in summary)
    print(
        json.dumps(
            {
                "ok": failed == 0,
                "written": str(out),
                "files": len(summary),
                "failed": failed,
                "repeated_paths_dropped": repeated,
            },
            sort_keys=True,
        )
    )
    return 0 if failed == 0 else 2


def build_parser(prog: str = "python -m aar.media.facts") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=DESCRIPTION)
    add_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": {"code": "interrupted"}}), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
