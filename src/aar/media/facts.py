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
shell, with a ``file:`` URL so no name reads as an option or a protocol; on Windows ffprobe is
looked for only in the absolute folders of ``PATH``, never the current folder). Nothing is
written but the output file, which is made under a temporary name beside it and given its name
without replacing anything (an existing ``--out`` is refused before any file is measured).

**Bounded.** At most :data:`MAX_FILES` files a run (``--max-files``, refused before any probe
when more are named), each probe killed after ``--timeout`` seconds, the whole run at most
``--deadline`` seconds (a file not measured by then is listed ``deadline``), each file's line at
most :data:`MAX_LINE_BYTES` (the hub's own limit per line), and the whole output at most
``--max-bytes``, error lines included: a file whose line would pass it is listed with the error
``output_cap`` instead, as is every file after it, and an error line that does not fit either
is only printed (``"in_file": false``). A run cut short (the deadline, or Ctrl+C: exit 130)
still writes and prints what it measured and lists what it did not.

**Nothing is skipped silently.** A file that is missing, not a regular file, changed while it
was measured, that ffprobe refuses, that times out, whose name or times cannot be written (a
name that is not valid Unicode; a time Windows cannot convert) is written as a line of its own,
``{"path", "exists", "error": {"code", "message"}}``, printed, and makes the run exit 2. The
hub reads such a line as "measured with an error" and names it when a cut needs that file. In a
folder, what is not measured (hidden, another suffix, a subfolder, a dangling link, not a
regular file) is counted by reason and the first names are printed in the summary; when
nothing at all is left to measure the run is refused (``no_media``, exit 2).

**Any console.** Everything printed is ASCII JSON (other characters escaped), so no console
encoding can fail on a file name; the facts file itself is always UTF-8.

**Deterministic.** Files are measured and written in the order of their PC paths (case
folded, then as given), every JSON object with sorted keys; the same files give the same bytes.

A folder argument takes the media files directly in it (:data:`MEDIA_SUFFIXES`, not hidden,
not in subfolders). ``--pc-folder LOCAL=PC`` names a folder as the edit PC knows it when the
command runs elsewhere (on the PC itself the absolute path is already the PC's): a drive or
share root, or a folder below one; on Windows matched as Windows compares paths (case and
separators ignored). Once it is given, a file under none of its folders is refused
(``not_under_pc_folder``), never recorded under its own name. A file directly at a drive's or
share's root is the error line ``at_root``: the library writer needs a folder below the root.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import ntpath
import os
import posixpath
import secrets
import stat
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
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
    "DEFAULT_DEADLINE_S",
    "MAX_FILES",
    "MAX_LINE_BYTES",
    "MEDIA_SUFFIXES",
    "FactsRunError",
    "Measured",
    "Selection",
    "TimeTextError",
    "birth_ns",
    "main",
    "measure",
    "measure_file",
    "say",
    "targets",
    "write_new",
]

#: The most files one run measures (a show's cameras and sound files fit many times over).
MAX_FILES = 256
#: One file's line at most: the hub's own limit (``aar.drp.mediapool.facts.MAX_FACTS_BYTES``).
MAX_LINE_BYTES = 1024 * 1024
#: The whole output at most, by default.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_S = 60.0
#: The whole run at most, by default (a show's files take seconds each).
DEFAULT_DEADLINE_S = 1800.0
#: Skipped folder entries named in the summary (all are counted).
MAX_SKIPPED_LISTED = 20
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


@dataclass
class Selection:
    """What the paths given name: the files to measure in output order, how many repeated
    paths were dropped, and what a folder held that is not measured (``(path, reason)``)."""

    files: list[Target]
    repeated: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def skipped_summary(self) -> dict[str, Any]:
        by_reason: dict[str, int] = {}
        for _, reason in self.skipped:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return {
            "count": len(self.skipped),
            "by_reason": dict(sorted(by_reason.items())),
            "first": [
                {"path": path, "reason": reason}
                for path, reason in self.skipped[:MAX_SKIPPED_LISTED]
            ],
        }


# --------------------------------------------------------------------------- times


class TimeTextError(ValueError):
    """A file time this system cannot write as a date (Windows refuses times before 1970 and
    after the year 3000; nothing past the year 9999 fits the facts' form)."""


def _year_checked(t: time.struct_time, what: str) -> time.struct_time:
    if not 1 <= t.tm_year <= 9999:
        raise TimeTextError(f"{what} falls in the year {t.tm_year}")
    return t


def local_text(seconds: int) -> str:
    """The PC's wall clock as ``Fri Mar 13 04:05:06 2026`` (C ``ctime`` form, English names
    whatever the locale). :class:`TimeTextError` when the system cannot convert it."""
    try:
        t = _year_checked(time.localtime(seconds), "the time")
    except (OSError, OverflowError, ValueError) as exc:
        raise TimeTextError(f"{seconds} s since 1970 has no local date here ({exc})") from None
    return (
        f"{_DAYS[t.tm_wday]} {_MONTHS[t.tm_mon - 1]} {t.tm_mday:2d} "
        f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d} {t.tm_year:04d}"
    )


def utc_text(ns: int) -> str:
    """Nanoseconds since 1970 as ``2026-03-13T09:05:06.1234567Z`` (.NET's round-trip form:
    seven fractional digits, the 100 ns a Windows file time holds). :class:`TimeTextError` when
    the system cannot convert it."""
    seconds, rest = divmod(ns, 1_000_000_000)
    try:
        t = _year_checked(time.gmtime(seconds), "the time")
    except (OSError, OverflowError, ValueError) as exc:
        raise TimeTextError(f"{seconds} s since 1970 has no UTC date here ({exc})") from None
    return (
        f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T{t.tm_hour:02d}:{t.tm_min:02d}:"
        f"{t.tm_sec:02d}.{rest // 100:07d}Z"
    )


# --------------------------------------------------------------------------- paths


_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\")


def _pc_path(local: Path, folders: Sequence[tuple[str, str]]) -> str | None:
    """The path the edit PC names ``local`` by: under a ``--pc-folder`` (the longest that
    matches), else, when no folder is given, the absolute path itself when it is a Windows
    path; None otherwise. A local path is compared by this system's rules (Windows': case and
    ``/`` or ``\\`` ignored); the PC path is built by Windows' rules wherever this runs."""
    local_rules = ntpath if _WINDOWS else posixpath
    flavour = PureWindowsPath if _WINDOWS else PurePosixPath
    absolute = flavour(local_rules.abspath(os.fspath(local)))
    best: tuple[int, str] | None = None
    for here, there in folders:
        here_abs = flavour(local_rules.abspath(here))
        if not absolute.is_relative_to(here_abs):  # by parts, case-insensitive on Windows
            continue
        if best is None or len(here_abs.parts) > best[0]:
            parts = absolute.parts[len(here_abs.parts) :]
            best = (len(here_abs.parts), str(PureWindowsPath(there, *parts)))
    if best is not None:
        return best[1]
    if folders:
        return None
    text = str(absolute)
    return text if _windows_absolute(text) else None


def _windows_absolute(text: str) -> bool:
    """A drive path (``E:\\...``) or a UNC path (``\\\\server\\share\\...``), backslashes; not a
    device path (``\\\\?\\``, ``\\\\.\\``), which the library writer refuses."""
    if "/" in text or text.startswith(_DEVICE_PREFIXES):
        return False
    win = PureWindowsPath(text)
    return bool(win.drive) and bool(win.root) and len(win.parts) >= 2


def _at_root(pc_path: str) -> bool:
    """A file directly at a drive's or share's root (``D:\\a.mov``), which the library writer
    refuses (:func:`aar.drp.mediapool.facts.check_pc_path`: a folder below the root)."""
    return len(PureWindowsPath(pc_path).parts) <= 2


def _windows_folder(text: str) -> str | None:
    """``text`` as an absolute folder on the edit PC, in the form Windows joins under: a drive
    root (``D:\\``), a share root (``\\\\server\\share``, given with or without its trailing
    backslash) or a folder below either, backslashes only; None for anything else (a
    drive-relative ``D:`` or ``D:Show``, ``/``, an empty, ``.`` or ``..`` name, a device path)."""
    if "/" in text or text.startswith(_DEVICE_PREFIXES):
        return None
    win = PureWindowsPath(text)
    if not win.drive or not win.root or not text.startswith(win.drive):
        return None  # drive-relative, rooted without a drive, or a server without a share
    if "\\\\" in text[2:]:
        return None  # an empty name: a doubled backslash past a UNC path's leading pair
    names = text.lstrip("\\").split("\\")
    if any(name in (".", "..") for name in names):
        return None
    return str(win)


def _folder_files(folder: Path, skipped: list[tuple[str, str]]) -> list[Path]:
    """The media files directly in ``folder`` (not hidden), by name; every other entry is added
    to ``skipped`` with its reason. ``folder_unreadable`` when it cannot be listed."""
    found = []
    left: list[tuple[str, str]] = []
    try:
        with os.scandir(folder) as entries:
            for n, entry in enumerate(entries):
                if n >= 20 * MAX_FILES:
                    raise FactsRunError(
                        "too_many_entries",
                        f"{str(folder)!r} holds more than {20 * MAX_FILES} entries",
                        "measure a folder of media, or name the files",
                    )
                reason = _skip_reason(entry)
                if reason is None:
                    found.append(Path(entry.path))
                else:
                    left.append((entry.path, reason))
    except OSError as exc:
        raise FactsRunError(
            "folder_unreadable",
            f"the folder {str(folder)!r} cannot be listed ({exc.strerror or exc})",
            "check the folder and its permissions, or name the files",
        ) from None
    skipped.extend(sorted(left))
    return sorted(found, key=lambda p: p.name)


def _skip_reason(entry: os.DirEntry[str]) -> str | None:
    """Why a folder entry is not measured, or None when it is a media file to measure."""
    name = entry.name
    if name.startswith("."):
        return "hidden"
    try:
        if entry.is_dir(follow_symlinks=False):
            return "subfolder"
        if entry.is_symlink() and not os.path.exists(entry.path):
            return "dangling_link"
        suffix = Path(name).suffix.lower()
        if suffix not in MEDIA_SUFFIXES:
            return f"not_a_media_suffix:{suffix or '(none)'}"
        if not entry.is_file(follow_symlinks=True):
            return "not_a_regular_file"
    except OSError:
        return "unreadable"
    return None


def targets(
    paths: Sequence[Path], folders: Sequence[tuple[str, str]], *, max_files: int = MAX_FILES
) -> Selection:
    """Every file to measure, in output order, how many repeated paths were dropped and what
    the folders held that is not measured. ``too_many_files`` past ``max_files`` (exactly
    ``max_files`` is fine); ``not_a_pc_path`` for a file the PC has no name for;
    ``folder_unreadable`` for a folder that cannot be listed."""
    files: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for path in paths:
        if path.is_dir():
            files.extend(_folder_files(path, skipped))
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
        if pc is None and folders:
            raise FactsRunError(
                "not_under_pc_folder",
                f"{str(local)!r} is in none of the --pc-folder folders given, so the edit PC's "
                "name for it is unknown",
                "add a --pc-folder for its folder (on the edit PC itself, FOLDER=FOLDER keeps "
                "its own name), or measure it in a run of its own",
            )
        if pc is None:
            raise FactsRunError(
                "not_a_pc_path",
                f"{str(local)!r} is not a Windows path, so the edit PC's name for it is unknown",
                "run this on the edit PC, or name its folder there with --pc-folder LOCAL=PC",
            )
        key = pc.lower()  # as Windows and the hub (load_facts) tell PC paths apart
        if key in out:
            repeated += 1
            continue
        out[key] = Target(local, pc)
    ordered = sorted(out.values(), key=lambda t: (t.pc_path.casefold(), t.pc_path))
    return Selection(ordered, repeated, skipped)


# --------------------------------------------------------------------------- one file


def _identity(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _utf8(text: str) -> str:
    """``text`` as it can be written in UTF-8: a lone surrogate (a name that is not valid
    Unicode) shown as ``\\udcff``."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _failed(target: Target, code: str, message: str, *, exists: bool = True) -> dict[str, Any]:
    return {
        "path": _utf8(target.pc_path),
        "exists": exists,
        "error": {"code": code, "message": _utf8(message[:2000])},
    }


def birth_ns(st: Any) -> int | None:
    """The file's creation time in ns since 1970 where the system keeps one, else None: the
    ``st_birthtime_ns`` of Windows (Python 3.12+) and the BSDs, the ``st_birthtime`` seconds of
    macOS, the ``st_ctime_ns`` of Windows before 3.12 (its creation time there; on every other
    system ``st_ctime`` is the last status change, never used). 0 or less is "unknown"."""
    ns = getattr(st, "st_birthtime_ns", None)
    if isinstance(ns, int) and not isinstance(ns, bool):
        return ns if ns > 0 else None
    seconds = getattr(st, "st_birthtime", None)
    if isinstance(seconds, float | int) and not isinstance(seconds, bool):
        if math.isfinite(seconds) and seconds > 0:
            return round(seconds * 1_000_000) * 1000
        return None
    if _WINDOWS and sys.version_info < (3, 12):
        ctime = getattr(st, "st_ctime_ns", None)
        return ctime if isinstance(ctime, int) and ctime > 0 else None
    return None


#: Read once, so tests can simulate Windows' rules on another system.
_WINDOWS = os.name == "nt"


def measure_file(
    target: Target, *, timeout_s: float = DEFAULT_TIMEOUT_S, notes: list[str] | None = None
) -> dict[str, Any]:
    """The facts of one file, or its error line (module docstring); never raises for the
    file's own faults (a missing ffprobe is the run's: :class:`FactsRunError`). What was left
    out of the facts (a creation time that cannot be written) is added to ``notes``."""
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
    try:
        mtime_local = local_text(before.st_mtime_ns // 1_000_000_000)
        mtime_utc = utc_text(before.st_mtime_ns)
    except TimeTextError as exc:
        return _failed(target, "bad_time", f"its modification time cannot be written: {exc}")
    facts: dict[str, Any] = {
        "path": target.pc_path,
        "exists": True,
        "size": before.st_size,
        "mtime_local": mtime_local,
        "mtime_utc": mtime_utc,
        "ffprobe": {"format": data.get("format") or {}, "streams": data["streams"]},
    }
    born = birth_ns(before)
    if born is not None:
        # A creation time this system cannot write is left out (the hub treats it as not
        # kept); the modification time, which the writer needs, is never left out.
        try:
            facts["ctime_utc"] = utc_text(born)
        except TimeTextError as exc:
            if notes is not None:
                notes.append(f"its creation time is left out: {exc}")
    return facts


def _line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- the run


@dataclass
class Measured:
    """A run's result: the facts file's bytes, one summary per file (``path``, ``ok``,
    ``error``, and ``in_file: false`` for an error line the byte cap kept out of the file),
    and whether the run was interrupted (Ctrl+C) before every file was measured."""

    data: bytes
    summary: list[dict[str, Any]]
    interrupted: bool = False


def measure(
    files: Sequence[Target],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    deadline_s: float = DEFAULT_DEADLINE_S,
    clock: Any = time.monotonic,
) -> Measured:
    """Measure ``files`` in order (module docstring): each probe at most ``timeout_s`` and the
    whole run at most ``deadline_s`` seconds, the output at most ``max_bytes`` bytes, error
    lines included. A Ctrl+C stops the run: the file being measured and every later one are
    listed ``interrupted``, and what was measured is kept."""
    out = bytearray()
    summary: list[dict[str, Any]] = []
    capped = False
    interrupted = False
    stop_at = clock() + deadline_s
    for target in files:
        notes: list[str] = []
        left = stop_at - clock()
        if interrupted:
            record = _failed(target, "interrupted", "not measured: the run was interrupted")
        elif capped:
            record = _failed(
                target, "output_cap", "the output reached its size cap before this file"
            )
        elif left <= 0:
            record = _failed(
                target,
                "deadline",
                f"not measured: the run's deadline of {deadline_s:g} s passed before this file",
            )
        elif _at_root(target.pc_path):
            record = _failed(
                target,
                "at_root",
                "not measured: it is directly at the root of a drive or share, and a library "
                "project needs its media in a folder below the root; move it into a folder",
                exists=os.path.lexists(target.local),
            )
        elif _utf8(target.pc_path) != target.pc_path:
            record = _failed(
                target,
                "unrepresentable_name",
                "its name is not valid Unicode, so the facts file cannot hold it; rename it",
            )
        else:
            try:
                record = measure_file(target, timeout_s=min(timeout_s, left), notes=notes)
            except KeyboardInterrupt:
                interrupted = True
                record = _failed(target, "interrupted", "not measured: the run was interrupted")
        try:
            line = _line(record)
        except UnicodeEncodeError:
            record = _failed(
                target, "unrepresentable_text", "its facts hold text that is not Unicode"
            )
            line = _line(record)
        if len(line) > MAX_LINE_BYTES:
            record = _failed(target, "too_large", "its facts are longer than one line may be")
            line = _line(record)
        if not capped and "error" not in record and len(out) + len(line) > max_bytes:
            capped = True
            record = _failed(target, "output_cap", "the output reached its size cap at this file")
            line = _line(record)
        entry: dict[str, Any] = {"path": record["path"], "ok": "error" not in record}
        if "error" in record:
            entry["error"] = record["error"]
        elif notes:
            entry["notes"] = [_utf8(note) for note in notes]
        if len(out) + len(line) <= max_bytes:
            out += line
        else:
            capped = True
            entry["in_file"] = False
        summary.append(entry)
    return Measured(bytes(out), summary, interrupted)


def write_new(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path``, which must not exist: under a temporary name beside it,
    flushed, then given its name without replacing anything (``FileExistsError`` if it exists
    by then). On a POSIX file system without hard links the name itself is created exclusively
    and written (a run killed meanwhile can leave it short: the hub refuses a cut line). The
    temporary file is removed on every path."""
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
                _write_exclusive(path, data)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)


def _write_exclusive(path: Path, data: bytes) -> None:
    """Create ``path`` (``FileExistsError`` if anything has that name) and write ``data``; the
    file is removed again if the write fails."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise


def _folder_pair(text: str) -> tuple[str, str]:
    here, sep, there = text.partition("=")
    folder = _windows_folder(there) if sep and here else None
    if folder is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not LOCAL=PC (a local folder, '=', and the folder's absolute "
            "Windows path on the edit PC: a drive or share root, or a folder below one, with "
            "backslashes)"
        )
    return here, folder


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
    "file was measured, 2 when any was not (listed with its error) or the run was refused "
    "(no media found among the paths included), 130 when interrupted (what was measured is "
    "written)."
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
        "--deadline",
        type=_positive(float),
        default=DEFAULT_DEADLINE_S,
        help=f"Seconds the whole run may take (default {DEFAULT_DEADLINE_S:g}); files not "
        "measured by then are listed.",
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


def say(value: Any, stream: Any = None) -> None:
    """Print ``value`` as one line of ASCII JSON (every other character escaped), which any
    console encoding can show; a closed pipe is not an error of the run."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n"
    with contextlib.suppress(BrokenPipeError):
        (stream or sys.stdout).write(text)
        (stream or sys.stdout).flush()


def run_command(args: argparse.Namespace) -> int:
    def refuse(error: FactsRunError, **extra: Any) -> int:
        say({"ok": False, "error": error.as_dict(), **extra})
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
        selection = targets(args.paths, args.pc_folder, max_files=args.max_files)
    except FactsRunError as exc:
        return refuse(exc)
    skipped = selection.skipped_summary()
    if not selection.files:
        return refuse(
            FactsRunError(
                "no_media",
                "no media file was found in the paths given (see skipped)",
                "name the files to measure, or a folder holding them ("
                + " ".join(sorted(MEDIA_SUFFIXES))
                + ")",
            ),
            skipped=skipped,
        )
    try:
        measured = measure(
            selection.files,
            timeout_s=args.timeout,
            max_bytes=args.max_bytes,
            deadline_s=args.deadline,
        )
    except FactsRunError as exc:
        return refuse(exc)
    try:
        write_new(out, measured.data)
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
    for entry in measured.summary:
        say(entry)
    failed = sum(not entry["ok"] for entry in measured.summary)
    final: dict[str, Any] = {
        "ok": failed == 0 and not measured.interrupted,
        "written": str(out),
        "files": len(measured.summary),
        "failed": failed,
        "not_in_file": sum(entry.get("in_file") is False for entry in measured.summary),
        "repeated_paths_dropped": selection.repeated,
        "skipped": skipped,
    }
    if measured.interrupted:
        final["interrupted"] = True
    say(final)
    if measured.interrupted:
        return 130
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
        say({"ok": False, "error": {"code": "interrupted"}}, sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
