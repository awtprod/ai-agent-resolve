"""S1h6: pins the upper bound of the unread-tail DbId search (S1h5 validator LOW-1, mutant M6).

``aar.drp.diff_opaque`` looks for a ``DbId`` among the attributes the scanner did not decode (past
the 4096-attribute cap) so that the change detail can say one is there. The search is bounded by
the end of the truncated element's start tag. A ``DbId`` on a *later sibling* in the same member
lies past that bound, so it is not an attribute of the truncated element and must not be reported
as one. With the bound widened to the end of the member (M6) the sibling's ``DbId`` is attributed
to the truncated element: this file fails then, and passes on the tip. Every input is built here,
byte by byte, in the layout of ``test_drp_s1h5``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from diff_testing import diff
from test_drp_s1h5 import _CAP, _NAME, _drp, _says_dbid_unread, _truncated

# An element after the truncated ``<J>``, with an identity attribute of its own. It is equal on
# both sides, so it is not a change; it only sits in the member after ``<J>``'s start tag.
_SIBLING = '<K DbId="k9"/>'


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_a_dbid_on_a_later_sibling_is_not_attributed_to_the_truncated_tail(
    tmp_path: Path, swap: bool
) -> None:
    # ``<J>`` has 4097 attributes and differs only in its last one (``t`` x against y), which is in
    # the unread tail; ``<K DbId="k9"/>`` follows it. The tail ends at ``<J>``'s ``/>``, so the
    # sibling's DbId is not in it, and the detail says nothing about a DbId.
    a = _drp(tmp_path / "a.drp", _NAME + _truncated("a", "x") + _SIBLING)
    b = _drp(tmp_path / "b.drp", _NAME + _truncated("a", "y") + _SIBLING)
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "J")
    assert f"attributes past the first {_CAP} differ" in c.detail
    assert "DbId" not in c.detail and "is not decoded" not in c.detail, c.detail


def test_the_same_layout_with_a_dbid_in_the_tail_is_named(tmp_path: Path) -> None:
    # The control for the test above: with the DbId inside ``<J>``'s own unread tail (and the same
    # sibling after it) the wording is there, so its absence above is not a search that finds
    # nothing anywhere.
    a = _drp(tmp_path / "a.drp", _NAME + _truncated("a").replace(' t="x"', ' DbId="z"') + _SIBLING)
    b = _drp(tmp_path / "b.drp", _NAME + _truncated("a").replace(' t="x"', ' DbId="w"') + _SIBLING)
    (c,) = diff(a, b).changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "J")
    assert _says_dbid_unread(c.detail, "A and B"), c.detail
