"""Print the S25c2 timeline-item scoreboard: per group, items checked, fields equal, differences.

    python scripts/items_scoreboard.py [--other-project] PROJECT.db [PROJECT.db ...]

Every picture and sound item of a supported shape is authored leave-one-out from another real
item of its kind and compared with Resolve's own rows (:mod:`aar.drp.items.gate`); items of an
unsupported shape are counted per reason. Nothing is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aar.drp.items import gate
from aar.drp.items.errors import ItemError
from aar.drp.library import LibraryError, LibraryOptions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--other-project", action="store_true", help="templates from another project"
    )
    parser.add_argument(
        "--linked-templates",
        action="store_true",
        help="unlinked items authored from linked templates",
    )
    parser.add_argument("projects", type=Path, nargs="+", help="Resolve-saved Project.db files")
    args = parser.parse_args(argv)
    try:
        board = gate.run(
            args.projects,
            options=LibraryOptions(settle_seconds=0.0),
            other_project=args.other_project,
            linked_templates=args.linked_templates,
        )
    except (ItemError, LibraryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(gate.format_scoreboard(board))
    print(f"gate {'passes' if board.passed() else 'FAILS'}")
    return 0 if board.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
