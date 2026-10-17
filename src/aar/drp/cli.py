"""``aar drp`` subcommands: ``inspect`` and ``describe``."""

from __future__ import annotations

import argparse
import json
import sys

from aar.drp.describe import DEFAULT_MAX_ITEMS, describe
from aar.drp.errors import DrpError
from aar.drp.model import DrpFile
from aar.drp.reader import read_drp

__all__ = ["register"]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``drp`` command group to the top-level parser."""
    drp = subparsers.add_parser("drp", help="Read .drp, .drt and .drb files offline.")
    sub = drp.add_subparsers(dest="drp_command", required=True)

    inspect = sub.add_parser("inspect", help="summary of a file, or the full model with --json")
    inspect.add_argument("file", help="path to a .drp, .drt or .drb file")
    inspect.add_argument("--json", action="store_true", help="print the full model as JSON")
    inspect.add_argument(
        "--blobs", action="store_true", help="fully decode every hex blob and report statistics"
    )
    inspect.set_defaults(func=_inspect)

    desc = sub.add_parser("describe", help="plain-text description for an agent to read")
    desc.add_argument("file", help="path to a .drp, .drt or .drb file")
    desc.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help=f"media clips and items per track to list (default {DEFAULT_MAX_ITEMS}, 0 = all)",
    )
    desc.set_defaults(func=_describe)


def _read(path: str, *, check_blobs: bool = False) -> DrpFile | None:
    try:
        return read_drp(path, check_blobs=check_blobs)
    except DrpError as exc:
        print(f"aar drp: {exc}", file=sys.stderr)
        return None


def _inspect(args: argparse.Namespace) -> int:
    drp = _read(args.file, check_blobs=args.blobs)
    if drp is None:
        return 1
    if args.json:
        json.dump(drp.to_dict(), sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    v = drp.versions
    print(f"file: {drp.path}")
    print(f"kind: {drp.kind} (from {drp.kind_source}; contents look like {drp.sniffed_kind})")
    print(f"size: {drp.size} bytes  sha256: {drp.sha256}")
    print(f"DbAppVer: {v.app_version if v else None}  DbPrjVer: {v.project_version if v else None}")
    print(f"zstd backend: {drp.zstd_backend or 'none (raw blocks only)'}")
    print(f"members: {len(drp.members)}")
    for m in drp.members:
        flag = "" if m.interpreted or m.role == "directory" else "  (raw)"
        print(f"  {m.size:>10}  {m.role:<14} {m.name}{flag}")
    if drp.project:
        print(f"project: {drp.project.name!r}")
    print(f"bins: {len(drp.bins)}  media: {len(drp.media)}  timelines: {len(drp.timelines)}")
    for tl in drp.timelines:
        items = sum(len(t.items) for t in tl.tracks)
        print(f"  {tl.kind}: {tl.name!r} tracks={len(tl.tracks)} items={items}")
    if drp.blob_stats.checked:
        print(f"blobs: {drp.blob_stats.total}")
        for tag, kinds in drp.blob_stats.by_tag.items():
            print(f"  {tag}: " + ", ".join(f"{k}={n}" for k, n in kinds.items()))
        for failure, n in drp.blob_stats.failures.items():
            print(f"  FAILED x{n}: {failure}")
        for tag, n in drp.blob_stats.undecoded.items():
            print(f"  not decoded (kept raw) x{n}: {tag}")
    for issue in drp.issues:
        print(f"[{issue.severity}] {issue.code}: {issue.message}")
    return 0


def _describe(args: argparse.Namespace) -> int:
    drp = _read(args.file)
    if drp is None:
        return 1
    sys.stdout.write(describe(drp, max_items=args.max_items))
    return 0
