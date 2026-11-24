"""``aar drp`` subcommands: ``inspect``, ``describe`` and ``library`` (``diff`` and ``modify``
live in their own modules)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import TextIO

from aar.drp import diff_cli
from aar.drp.describe import DEFAULT_MAX_ITEMS, describe, printable
from aar.drp.errors import DrpError
from aar.drp.library import LibraryError, LibraryOptions, list_library
from aar.drp.model import DrpFile
from aar.drp.reader import read_drp

__all__ = ["EXIT_NOTHING_READ", "EXIT_OK", "EXIT_OUTPUT_ERROR", "EXIT_PIPE_CLOSED", "register"]

EXIT_OK = 0
EXIT_NOTHING_READ = 1
EXIT_OUTPUT_ERROR = 74  # EX_IOERR from sysexits.h: stdout is closed, full or failing
EXIT_PIPE_CLOSED = 141  # 128 + SIGPIPE, what a shell reports for a reader that went away

EXIT_STATUS_HELP = """\
exit status:
  0    the file was read (problems found inside it are listed as warnings)
  1    the file could not be opened, or it holds no project, bin or timeline data
  2    usage error
  74   the output could not be written (stdout closed, disk full, I/O error)
  141  the output pipe was closed early (for example by `head`)
"""


LIBRARY_WAIT_SECONDS = 5.0
LIBRARY_EXIT_STATUS_HELP = """\
exit status:
  0    every project in the library was read
  1    the folder is not a library or holds no project, or a project was refused (it is
       listed with the refusal's code, message and fix hint)
  2    usage error
  74   the output could not be written (stdout closed, disk full, I/O error)
  141  the output pipe was closed early (for example by `head`)
"""


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``drp`` command group to the top-level parser."""
    fmt = argparse.RawDescriptionHelpFormatter
    drp = subparsers.add_parser(
        "drp",
        help="Read .drp, .drt and .drb files, and Resolve disk library projects, offline.",
        epilog=EXIT_STATUS_HELP,
        formatter_class=fmt,
    )
    sub = drp.add_subparsers(dest="drp_command", required=True)

    inspect = sub.add_parser(
        "inspect",
        help="summary of a file, or the full model with --json",
        epilog=EXIT_STATUS_HELP,
        formatter_class=fmt,
    )
    inspect.add_argument("file", help="path to a .drp, .drt or .drb file, or a library Project.db")
    inspect.add_argument("--json", action="store_true", help="print the full model as JSON")
    inspect.add_argument(
        "--blobs", action="store_true", help="fully decode every hex blob and report statistics"
    )
    inspect.set_defaults(func=_inspect)

    desc = sub.add_parser(
        "describe",
        help="plain-text description for an agent to read",
        epilog=EXIT_STATUS_HELP,
        formatter_class=fmt,
    )
    desc.add_argument("file", help="path to a .drp, .drt or .drb file, or a library Project.db")
    desc.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help=f"media clips and items per track to list (default {DEFAULT_MAX_ITEMS}, 0 = all)",
    )
    desc.set_defaults(func=_describe)

    lib = sub.add_parser(
        "library",
        help="list the projects of a Resolve disk library (each Project.db read)",
        description=(
            "List every project saved in a Resolve disk project library: its folder, "
            "ProjectName, project id, project version, last-modified time and timeline names. "
            "Each Project.db is read from a settled private snapshot; a project that is being "
            "saved, damaged or not readable is listed with its refusal, never skipped."
        ),
        epilog=LIBRARY_EXIT_STATUS_HELP,
        formatter_class=fmt,
    )
    lib.add_argument("folder", help="the library's folder (the one holding 'Resolve Projects')")
    lib.add_argument("--json", action="store_true", help="print the listing as JSON")
    lib.add_argument(
        "--wait",
        type=float,
        default=LIBRARY_WAIT_SECONDS,
        metavar="SECONDS",
        help="how long to wait for a project being saved before listing it as busy "
        f"(default {LIBRARY_WAIT_SECONDS:g})",
    )
    lib.set_defaults(func=_library)

    diff_cli.register(sub)
    # Imported here: aar.drp.modify.cli reuses this module's output helpers.
    from aar.drp.modify import cli as modify_cli

    modify_cli.register(sub)


def _safe(text: str, stream: TextIO) -> str:
    """Make ``text`` encodable on ``stream``: lone surrogates etc. become ``\\uXXXX`` escapes."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return text.encode(encoding, "backslashreplace").decode(encoding, "replace")


def _emit(text: str) -> int:
    """Write ``text`` to stdout.

    A closed pipe ends quietly with ``EXIT_PIPE_CLOSED``. A missing stdout (``>&-``) or a
    failing one (``>/dev/full``, I/O error) ends with ``EXIT_OUTPUT_ERROR`` and a one-line
    message on stderr. Never a traceback.
    """
    stream = sys.stdout
    if stream is None:
        _warn("aar drp: stdout is closed; nothing written")
        return EXIT_OUTPUT_ERROR
    try:
        stream.write(_safe(text, stream))
        stream.flush()
    except BrokenPipeError:
        _silence(sys.stdout)
        return EXIT_PIPE_CLOSED
    except (OSError, ValueError) as exc:  # ValueError: write to a closed file object
        _silence(sys.stdout)
        _warn(f"aar drp: cannot write output: {exc}")
        return EXIT_OUTPUT_ERROR
    return EXIT_OK


def _warn(message: str) -> None:
    """Best-effort message on stderr; stays silent if stderr is closed or failing too."""
    stream = sys.stderr
    if stream is None:
        return
    try:
        stream.write(_safe(printable(message), stream) + "\n")
        stream.flush()
    except (OSError, ValueError):
        # stderr is closed or failing (2>&- or 2>/dev/full): point it at /dev/null so the
        # interpreter's own flush at exit cannot fail again and turn the exit code into 120.
        _silence(stream)


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


def _read(path: str, *, check_blobs: bool = False) -> DrpFile | None:
    try:
        return read_drp(path, check_blobs=check_blobs)
    except DrpError as exc:
        _warn(f"aar drp: {exc}")
        return None


def _status(drp: DrpFile, written: int) -> int:
    if written != EXIT_OK:
        return written
    empty = drp.project is None and not drp.bins and not drp.media and not drp.timelines
    return EXIT_NOTHING_READ if empty else EXIT_OK


def _inspect(args: argparse.Namespace) -> int:
    drp = _read(args.file, check_blobs=args.blobs)
    if drp is None:
        return EXIT_NOTHING_READ
    if args.json:
        # ASCII-only JSON is valid in every output encoding (cp1252 pipes on Windows), and
        # allow_nan=False fails closed rather than emit the invalid tokens Infinity/NaN.
        try:
            text = json.dumps(drp.to_dict(), indent=1, ensure_ascii=True, allow_nan=False) + "\n"
        except ValueError as exc:  # a non-finite value slipped the decoders; never emit bad JSON
            _warn(f"aar drp: cannot encode JSON: {exc}")
            return EXIT_OUTPUT_ERROR
        return _status(drp, _emit(text))
    v = drp.versions
    lines = [
        f"file: {drp.path}",
        f"kind: {drp.kind} (from {drp.kind_source}; contents look like {drp.sniffed_kind})",
        f"size: {drp.size} bytes  sha256: {drp.sha256}",
        f"DbAppVer: {v.app_version if v else None}  DbPrjVer: {v.project_version if v else None}",
        f"zstd backend: {drp.zstd_backend or 'none (raw blocks only)'}",
        f"members: {len(drp.members)}",
    ]
    for m in drp.members:
        flag = "" if m.interpreted or m.role == "directory" else "  (raw)"
        lines.append(f"  {m.size:>10}  {m.role:<14} {m.name}{flag}")
    if drp.project:
        lines.append(f"project: {drp.project.name!r}")
    lines.append(f"bins: {len(drp.bins)}  media: {len(drp.media)}  timelines: {len(drp.timelines)}")
    for tl in drp.timelines:
        items = sum(len(t.items) for t in tl.tracks)
        lines.append(f"  {tl.kind}: {tl.name!r} tracks={len(tl.tracks)} items={items}")
    if drp.blob_stats.checked:
        lines.append(f"blobs: {drp.blob_stats.total}")
        for tag, kinds in drp.blob_stats.by_tag.items():
            lines.append(f"  {tag}: " + ", ".join(f"{k}={n}" for k, n in kinds.items()))
        for failure, n in drp.blob_stats.failures.items():
            lines.append(f"  FAILED x{n}: {failure}")
        for tag, n in drp.blob_stats.undecoded.items():
            lines.append(f"  not decoded (kept raw) x{n}: {tag}")
    for issue in drp.issues:
        lines.append(f"[{issue.severity}] {issue.code}: {issue.message}")
    return _status(drp, _emit("\n".join(printable(line) for line in lines) + "\n"))


def _describe(args: argparse.Namespace) -> int:
    drp = _read(args.file)
    if drp is None:
        return EXIT_NOTHING_READ
    return _status(drp, _emit(describe(drp, max_items=args.max_items)))


def _library(args: argparse.Namespace) -> int:
    if not math.isfinite(args.wait) or args.wait < 0:
        _warn("aar drp library: --wait must be a number of seconds, 0 or more")
        return 2
    options = LibraryOptions(deadline_seconds=args.wait)
    try:
        listing = list_library(args.folder, options=options)
    except LibraryError as exc:
        _warn(f"aar drp library: {exc}")
        return EXIT_NOTHING_READ
    refused = sum(1 for p in listing.projects if p.refusal is not None)
    if args.json:
        text = json.dumps(listing.to_dict(), indent=1, ensure_ascii=True, allow_nan=False) + "\n"
    else:
        lines = [
            f"library: {listing.root}",
            f"projects: {len(listing.projects)}"
            + (f" ({refused} refused)" if refused else "")
            + (" (list cut short: too many entries)" if listing.truncated else ""),
        ]
        for p in listing.projects:
            where = f"  {p.user}/{p.folder}"
            if p.refusal is not None:
                r = p.refusal
                lines.append(f"{where}: REFUSED {r['code']}: {r['message']}; {r['hint']}")
                continue
            names = ", ".join(f'"{n}"' for n in p.timelines) or "none"
            lines.append(
                f'{where}: "{p.name or "?"}" version {p.project_version}, id {p.id}, '
                f"modified {p.last_modified or '?'}, timelines: {names}"
            )
            if p.warnings:
                lines.append(f"    read with warnings: {', '.join(p.warnings)}")
        text = "\n".join(printable(line) for line in lines) + "\n"
    written = _emit(text)
    if written != EXIT_OK:
        return written
    return EXIT_NOTHING_READ if refused or not listing.projects else EXIT_OK
