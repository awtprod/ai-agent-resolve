"""S1h5: pins for the four S1h4 round-3 mutants that survived, and the follow-ups to them.

Every input below is built here, byte by byte; none is a copy of a corpus file. The first block
pins the byte-level comparison that ``aar.drp.diff_opaque`` does on an element the scanner could
only half read (more than 4096 attributes, so the attributes past the cap sit in an unread
"tail"). Each test names the validator mutant (``V05``, ``V08``, ``V10``/``T03``, ``V15``) it
kills: with that mutant applied to ``diff_opaque.py`` the test fails, and it passes on the tip.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from diff_testing import diff
from drp_helpers import xml

from aar.cli import main

_CAP = 4096
# 4095 filler attributes: with the leading ``k0`` and a trailing ``t`` the element has 4097, so
# its last attribute is past the cap and the element is truncated.
_FILL = "".join(f' k{i}="v"' for i in range(1, _CAP))
# 4096 attributes exactly: an element carrying these is not truncated.
_FULL = "".join(f' k{i}="v"' for i in range(_CAP))
_NAME = "<ProjectName>P</ProjectName>"
_POOL = "MediaPool/Master/MpFolder.xml"


def _drp(path: Path, project_body: str, *, pool: bytes | None = None) -> Path:
    members: dict[str, bytes | str] = {
        "project.xml": xml(f'<SM_Project DbId="p1">{project_body}</SM_Project>\n'),
        "Gallery.xml": xml('<Gallery::GyGallery DbId="g"/>\n'),
        _POOL: pool
        if pool is not None
        else xml('<Sm2MpFolder DbId="b1"><Name>Master</Name><MediaVec/></Sm2MpFolder>\n'),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _truncated(first: str, last: str = "x", content: str | None = None) -> str:
    """A ``<J>`` element with 4097 attributes: ``k0=first``, 4095 fillers, then ``t=last``."""
    head = f'<J k0="{first}"{_FILL} t="{last}"'
    return f"{head}/>" if content is None else f"{head}>{content}</J>"


# ---- V15: a kept attribute changed on a truncated element --------------------------------------


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_v15_a_kept_attribute_change_on_a_truncated_element_is_named(
    tmp_path: Path, swap: bool
) -> None:
    # Both sides carry the same unread tail (``t="x"``), so the raw-tail comparison cannot see the
    # change: only the by-name comparison of the kept attributes reports it. Dropping that
    # comparison for truncated elements (V15) leaves the pair "identical".
    a = _drp(tmp_path / "a.drp", _NAME + _truncated("a"))
    b = _drp(tmp_path / "b.drp", _NAME + _truncated("b"))
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "J")
    assert "attribute(s) k0 changed" in c.detail
    assert "attributes past the first" not in c.detail  # the tails are byte-equal
    assert not d.to_dict()["identical"]


# ---- V05: the tail stops at the end of the start tag, not the end of the element ---------------


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_v05_inner_text_change_on_a_truncated_element_is_reported_once(
    tmp_path: Path, swap: bool
) -> None:
    # The element's attributes are identical on both sides; only its text differs ("1" vs "2").
    # The unread tail is the attribute bytes up to the start tag's ``>``; if it ran on to the end
    # of the element (V05) the same text change would be counted a second time, as a bogus
    # "attributes past the first 4096 differ".
    a = _drp(tmp_path / "a.drp", _NAME + _truncated("a", content="1"))
    b = _drp(tmp_path / "b.drp", _NAME + _truncated("a", content="2"))
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "J")
    assert c.byte_count == 1
    assert "not decoded" in c.detail
    assert "attributes past" not in c.detail and "attribute(s)" not in c.detail


# ---- V08: the two tails are each cut at their own offset ---------------------------------------


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_v08_a_shift_earlier_in_the_member_does_not_change_a_truncated_element(
    tmp_path: Path, swap: bool
) -> None:
    # B has one byte more before the truncated ``<J>`` (``<O>22</O>`` against ``<O>1</O>``), so
    # every offset of ``<J>`` in B is one past its offset in A. Cutting B's tail at A's offset
    # (V08) compares shifted bytes and reports ``<J>`` as changed. Only ``<O>`` changed.
    a = _drp(tmp_path / "a.drp", f"{_NAME}<O>1</O>{_truncated('a')}")
    b = _drp(tmp_path / "b.drp", f"{_NAME}<O>22</O>{_truncated('a')}")
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "O")
    assert not [x for x in d.changes if x.field_name == "J"]


# ---- V10 / T03: an element that is not truncated on one side only ------------------------------

# A's bin member has no XML declaration and starts with text, so its root is ``<T">`` (the ``<``
# and ``T"`` come from the bytes ``z="<T"``), carrying exactly 4096 attributes: not truncated.
# B's member repeats ``z="<T"`` plus the same 4096 attributes, so B's root has more than 4096
# attributes and is truncated; B's unread tail is byte-for-byte A's whole member. With the
# "one side was not truncated" clause gone (V10), ``data_a[None:end]`` is that whole member and
# the pair compares equal.
_T_A = f' z="<T"{_FULL}/>'
_T_B = f' z="<T"{_FULL} z="<T"{_FULL}/>'


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_v10_an_element_truncated_on_one_side_only_is_a_change(tmp_path: Path, swap: bool) -> None:
    a = _drp(tmp_path / "a.drp", _NAME, pool=_T_A.encode())
    b = _drp(tmp_path / "b.drp", _NAME, pool=_T_B.encode())
    first, second = (b, a) if swap else (a, b)
    d = diff(first, second)
    (c,) = d.changes
    assert (c.section, c.kind) == ("opaque", "opaque-changed")
    assert "attributes past the first 4096 differ" in c.detail
    assert c.where_a is not None and c.where_a.member == _POOL


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_v10_the_same_pair_exits_1_from_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], swap: bool
) -> None:
    # The validator's t03 pair: exit 1 ("differences found"), never 0 ("identical").
    a = _drp(tmp_path / "a.drp", _NAME, pool=_T_A.encode())
    b = _drp(tmp_path / "b.drp", _NAME, pool=_T_B.encode())
    first, second = (b, a) if swap else (a, b)
    code = main(["drp", "diff", str(first), str(second)])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "summary: 1 change" in out and "opaque 1" in out
