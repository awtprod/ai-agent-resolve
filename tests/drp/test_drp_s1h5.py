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


def _drp(
    path: Path,
    project_body: str,
    *,
    pool: bytes | None = None,
    root_attrs: str = ' DbId="p1"',
) -> Path:
    members: dict[str, bytes | str] = {
        "project.xml": xml(f"<SM_Project{root_attrs}>{project_body}</SM_Project>\n"),
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


# ---- A8 / D9: an identity attribute (DbId) past the 4096-attribute cap -------------------------
#
# Below the cap a repeated attribute is last-wins; past it only the first 4096 attributes are
# decoded, so a DbId in the unread tail is never read and the semantic "DbId p1 -> p2" line is not
# available. The pair is still reported (and the gate still refuses it); the change detail says
# that a DbId sits in the part that was not decoded.


def _says_dbid_unread(detail: str, sides: str) -> bool:
    return f"DbId appears past the first {_CAP} attributes in {sides} and is not decoded" in detail


def _bin_root(dbid_tail: str) -> bytes:
    root = f'<Sm2MpFolder DbId="b1"{_FILL} k0="v"{dbid_tail}>'
    return xml(f"{root}<Name>Master</Name><MediaVec/></Sm2MpFolder>\n").encode()


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_a8_a_second_dbid_past_the_cap_on_the_project_root_is_named(
    tmp_path: Path, swap: bool
) -> None:
    # The validator's A8 pair: B's root is ``DbId="p1"``, 4095 fillers, then a second ``DbId="p2"``.
    a = _drp(tmp_path / "a.drp", _NAME)
    b = _drp(
        tmp_path / "b.drp",
        _NAME,
        root_attrs=' DbId="p1"' + "".join(f' k{i}="v"' for i in range(_CAP - 1)) + ' DbId="p2"',
    )
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "SM_Project")
    assert _says_dbid_unread(c.detail, "A" if swap else "B"), c.detail


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_a8_the_tail_that_differs_only_by_a_dbid_says_so(tmp_path: Path, swap: bool) -> None:
    # The kept attributes are equal on both sides (the same 4096), B has one more: a DbId.
    root = ' DbId="p1"' + "".join(f' k{i}="v"' for i in range(_CAP - 1))
    a = _drp(tmp_path / "a.drp", _NAME, root_attrs=root)
    b = _drp(tmp_path / "b.drp", _NAME, root_attrs=root + ' DbId="p2"')
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert f"attributes past the first {_CAP} differ" in c.detail
    assert _says_dbid_unread(c.detail, "A" if swap else "B"), c.detail
    assert c.detail.count("DbId appears") == 1


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
def test_d9_a_changed_dbid_past_the_cap_on_a_bin_root_names_both_sides(
    tmp_path: Path, swap: bool
) -> None:
    # The validator's D9 pair: a second DbId past the cap on a bin root, q1 against q2.
    a = _drp(tmp_path / "a.drp", _NAME, pool=_bin_root(' DbId="q1"'))
    b = _drp(tmp_path / "b.drp", _NAME, pool=_bin_root(' DbId="q2"'))
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "Sm2MpFolder")
    assert f"attributes past the first {_CAP} differ" in c.detail
    assert _says_dbid_unread(c.detail, "A and B"), c.detail


def test_a_dbid_in_the_tail_is_named_next_to_a_kept_attribute_change(tmp_path: Path) -> None:
    # The by-name wording also carries it: k0 changed, and both sides have a DbId past the cap.
    tail = ' DbId="z"'
    a = _drp(tmp_path / "a.drp", _NAME + _truncated("a").replace(' t="x"', tail))
    b = _drp(tmp_path / "b.drp", _NAME + _truncated("b").replace(' t="x"', tail))
    (c,) = diff(a, b).changes
    assert "attribute(s) k0 changed" in c.detail
    assert _says_dbid_unread(c.detail, "A and B"), c.detail


@pytest.mark.parametrize("name", ["xDbId", "DbIdx", "Db", "dbid"])
def test_other_names_in_the_tail_do_not_trigger_the_dbid_wording(tmp_path: Path, name: str) -> None:
    a = _drp(tmp_path / "a.drp", _NAME + _truncated("a").replace(' t="x"', f' {name}="1"'))
    b = _drp(tmp_path / "b.drp", _NAME + _truncated("a").replace(' t="x"', f' {name}="2"'))
    (c,) = diff(a, b).changes
    assert f"attributes past the first {_CAP} differ" in c.detail
    assert "DbId" not in c.detail


def test_a_tail_without_a_dbid_says_nothing_about_one(tmp_path: Path) -> None:
    a = _drp(tmp_path / "a.drp", _NAME + _truncated("a", "x"))
    b = _drp(tmp_path / "b.drp", _NAME + _truncated("a", "y"))
    (c,) = diff(a, b).changes
    assert f"attributes past the first {_CAP} differ" in c.detail
    assert "DbId" not in c.detail


# ---- the tail is compared more strictly than the kept attributes (DESIGN.md, Diff (S2)) --------

_STRICT = {
    "quote-style": ('<J k0="a"{f} t="x"/>', "<J k0=\"a\"{f} t='x'/>"),
    "whitespace": ('<J k0="a"{f} t="x"/>', '<J k0="a"{f}  t="x" />'),
    "entity-spelling": ('<J k0="a"{f} t="A"/>', '<J k0="a"{f} t="&#65;"/>'),
    "attribute-order": ('<J k0="a"{f} t="x" u="y"/>', '<J k0="a"{f} u="y" t="x"/>'),
    "empty-element-form": ('<J k0="a"{f} t="x"/>', '<J k0="a"{f} t="x"></J>'),
}


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
@pytest.mark.parametrize("case", list(_STRICT))
def test_a_respelling_inside_the_truncated_tail_is_a_change(
    tmp_path: Path, case: str, swap: bool
) -> None:
    # In the kept attributes these spellings are equal (see the quote-style test in
    # test_drp_s1h4_cap_gate); past the cap they are bytes, so each counts. It fails closed.
    one, two = (x.replace("{f}", _FILL) for x in _STRICT[case])
    a = _drp(tmp_path / "a.drp", _NAME + one)
    b = _drp(tmp_path / "b.drp", _NAME + two)
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "J")
    assert f"attributes past the first {_CAP} differ" in c.detail


def test_the_same_respellings_in_the_kept_attributes_are_not_changes(tmp_path: Path) -> None:
    # The contrast the DESIGN.md sentence draws: below the cap quote style and attribute order
    # are normalised away.
    kept = "".join(f' k{i}="v"' for i in range(1, 6))
    kept_sq = kept.replace('="v"', "='v'")
    a = _drp(tmp_path / "a.drp", f'{_NAME}<J k0="a"{kept} u="y" t="x"/>')
    b = _drp(tmp_path / "b.drp", f"{_NAME}<J t='x' u='y'{kept_sq} k0='a'/>")
    assert not diff(a, b).changes
