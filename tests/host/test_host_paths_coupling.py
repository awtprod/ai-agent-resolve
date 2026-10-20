"""Item 6 guard: aar.host.winpath reuses internals of the S8c-owned aar.drp.modify.paths.

paths.py is owned by S8c (round 3 is editing it now). winpath deliberately reuses its private
parse/validate primitives rather than re-implementing NTFS path rules. If S8c changes the shape
of those names, this test fails loudly with a clear message, instead of a surprising failure deep
in a transfer. A later slice should make these names public (see PROGRESS).
"""

from __future__ import annotations

import inspect

from aar.drp.modify import paths

_REUSED = (
    "_parse_win",
    "_irregular",
    "_WIN_BAD",
    "_WIN_RESERVED",
    "_Win",
    "ntfs_upper",
    "PathMapper",
)


def test_reused_names_still_exist() -> None:
    for name in _REUSED:
        assert hasattr(paths, name), (
            f"aar.drp.modify.paths.{name} is gone; aar.host.winpath (S13f) reuses it. "
            "Make it public or update winpath."
        )


def test_parse_win_signature_and_shape() -> None:
    assert list(inspect.signature(paths._parse_win).parameters) == ["path"], (
        "aar.host.winpath calls paths._parse_win(path)"
    )
    win = paths._parse_win("E:\\A\\B")
    assert win is not None
    for attr in ("root", "unc", "parts", "trailing_sep"):
        assert hasattr(win, attr), f"paths._Win lost .{attr}; aar.host.winpath depends on it"
    assert callable(win.key) and isinstance(win.key(), tuple), (
        "paths._Win.key() must return a tuple"
    )


def test_irregular_and_regex_shapes() -> None:
    assert paths._irregular(("..",)) is True
    assert paths._irregular(("ok",)) is False
    assert paths._WIN_BAD.search(":") is not None, "paths._WIN_BAD must flag illegal characters"
    assert paths._WIN_RESERVED.match("CON") is not None, (
        "paths._WIN_RESERVED must flag device names"
    )


def test_ntfs_upper_is_one_to_one() -> None:
    # aar.host.winpath relies on NTFS-style per-character upcasing (not full casefold).
    assert paths.ntfs_upper("show") == "SHOW"
    assert paths.ntfs_upper("stra\u00dfe") == "STRA\u00dfE"  # eszett is not expanded to SS
