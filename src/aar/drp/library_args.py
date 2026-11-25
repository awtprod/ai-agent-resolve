"""The command-line options that raise the budgets of a library read, defined once.

Every command that can read a project from a Resolve disk library (``aar drp inspect``,
``describe``, ``diff`` and ``library``, and ``aar plan import-timeline``) adds them with
:func:`add_library_budget_options` and turns them into :class:`~aar.drp.library.LibraryOptions`
with :func:`library_options`. Each refusal's hint names the option that raises the limit it hit.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from typing import Any

from aar.drp.container import DEFAULT_MAX_MEMBER_BYTES
from aar.drp.library import (
    DEFAULT_LISTING_BYTES,
    DEFAULT_MAX_DB_BYTES,
    DEFAULT_MAX_LOAD_BYTES,
    DEFAULT_MAX_READ_SECONDS,
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_VALUE_BYTES,
    LAYOUT_PER_LOAD_BYTE,
    LibraryOptions,
)
from aar.drp.reader import DEFAULT_MAX_TOTAL_XML_BYTES

__all__ = ["add_library_budget_options", "add_listing_bytes_option", "library_options"]

#: Option name -> LibraryOptions field.
_FIELDS = {
    "max_rows": "max_rows",
    "max_load_bytes": "max_load_bytes",
    "max_layout_bytes": "max_layout_bytes",
    "max_value_bytes": "max_value_bytes",
    "max_db_bytes": "max_db_bytes",
    "max_read_seconds": "max_read_seconds",
}


def _count(most: int | None = None, what: str = "") -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
        if value < 1:
            raise argparse.ArgumentTypeError("must be 1 or more")
        if most is not None and value > most:
            raise argparse.ArgumentTypeError(f"must be at most {most} ({what})")
        return value

    return parse


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a number of seconds over 0")
    return value


def add_library_budget_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that raise a library read's budgets (``Project.db`` only)."""
    group = parser.add_argument_group(
        "library read budgets (a Project.db only)",
        "Raise one of these when a library project is refused as too_large and really is "
        "one project; the refusal names the option.",
    )
    group.add_argument(
        "--max-rows",
        type=_count(),
        metavar="N",
        help=f"rows read from the project's tables, in total (default {DEFAULT_MAX_ROWS})",
    )
    group.add_argument(
        "--max-load-bytes",
        type=_count(),
        metavar="BYTES",
        help="bytes of values read, plus 16 per value (default "
        f"{DEFAULT_MAX_LOAD_BYTES}); also raises the layout budget unless --max-layout-bytes "
        "is given",
    )
    group.add_argument(
        "--max-layout-bytes",
        type=_count(DEFAULT_MAX_TOTAL_XML_BYTES, "the most the reader holds"),
        metavar="BYTES",
        help=f"bytes the rows lay out to (default {LAYOUT_PER_LOAD_BYTE} x --max-load-bytes, "
        f"at most {DEFAULT_MAX_TOTAL_XML_BYTES})",
    )
    group.add_argument(
        "--max-value-bytes",
        type=_count(DEFAULT_MAX_MEMBER_BYTES, "the reader's member limit"),
        metavar="BYTES",
        help=f"one value at most (default {DEFAULT_MAX_VALUE_BYTES})",
    )
    group.add_argument(
        "--max-db-bytes",
        type=_count(),
        metavar="BYTES",
        help=f"size of the Project.db file (default {DEFAULT_MAX_DB_BYTES})",
    )
    group.add_argument(
        "--max-read-seconds",
        type=_seconds,
        metavar="SECONDS",
        help="time to read the project once it is copied: integrity check, queries and layout "
        f"(default {DEFAULT_MAX_READ_SECONDS:g})",
    )


def add_listing_bytes_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--max-listing-bytes``, the byte budget of a whole listing."""
    parser.add_argument(
        "--max-listing-bytes",
        type=_count(),
        default=DEFAULT_LISTING_BYTES,
        metavar="BYTES",
        help="sizes of the project files read, in total; projects past it are listed as not "
        f"read (default {DEFAULT_LISTING_BYTES})",
    )


def library_options(args: argparse.Namespace, **fields: Any) -> LibraryOptions:
    """The :class:`LibraryOptions` the budget options in ``args`` ask for, with ``fields``."""
    given = {
        field: getattr(args, option)
        for option, field in _FIELDS.items()
        if getattr(args, option, None) is not None
    }
    return LibraryOptions(**{**given, **fields})
