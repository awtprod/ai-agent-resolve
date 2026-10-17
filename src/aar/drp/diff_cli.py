"""``aar drp diff A B``: semantic diff of two files, with ``diff(1)`` exit codes."""

from __future__ import annotations

import argparse
import json
import sys

from aar.drp.diff import SECTIONS, diff_projects
from aar.drp.errors import DrpError
from aar.drp.reader import read_drp

__all__ = ["EXIT_DIFFERENT", "EXIT_ERROR", "EXIT_SAME", "register"]

EXIT_SAME = 0
EXIT_DIFFERENT = 1
EXIT_ERROR = 2


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``diff`` to the ``aar drp`` subcommands."""
    p = sub.add_parser(
        "diff",
        help="semantic diff of two files (exit 0 same, 1 different, 2 error)",
        description=(
            "Compare two .drp/.drt/.drb files: header, media pool, timelines, markers, and "
            "the raw bytes of everything not decoded. Exit status follows diff(1): 0 when the "
            "selected sections have no differences, 1 when they do, 2 on error."
        ),
    )
    p.add_argument("a", help="the earlier file")
    p.add_argument("b", help="the later file")
    p.add_argument("--json", action="store_true", help="print the diff as JSON")
    p.add_argument(
        "--only",
        action="append",
        choices=SECTIONS,
        metavar="{" + ",".join(SECTIONS) + "}",
        help="show only this section (repeatable); the exit status follows the shown sections",
    )
    p.set_defaults(func=_diff)


def _diff(args: argparse.Namespace) -> int:
    try:
        a = read_drp(args.a)
        b = read_drp(args.b)
        result = diff_projects(a, b)
    except DrpError as exc:
        print(f"aar drp diff: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.only:
        result = result.only(args.only)
    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(result.to_text())
    return EXIT_SAME if result.identical else EXIT_DIFFERENT
