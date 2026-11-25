"""Print the S25c1 media-pool scoreboard: per kind, files checked, fields equal, fields differing.

    python scripts/media_scoreboard.py --facts FACTS.jsonl PROJECT.db [PROJECT.db ...]

For every file with facts, its rows are authored leave-one-out from another file of its kind and
compared with Resolve's own rows (:mod:`aar.drp.mediapool.gate`). File names appear only in the
examples printed here; nothing is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aar.drp.library import LibraryError, LibraryOptions
from aar.drp.mediapool import gate
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.kinds import KINDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--facts", type=Path, required=True, help="the facts file (JSON lines)")
    parser.add_argument("projects", type=Path, nargs="+", help="Resolve-saved Project.db files")
    args = parser.parse_args(argv)
    try:
        facts = gate.load_facts(args.facts)
        board = gate.run(args.projects, facts, options=LibraryOptions(settle_seconds=0.0))
    except (MediaPoolError, LibraryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(gate.format_scoreboard(board))
    declared = [k for k, s in KINDS.items() if s.status == "verified"]
    passed = board.passed(declared)
    print(f"declared verified kinds: {len(declared)}; gate {'passes' if passed else 'FAILS'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
