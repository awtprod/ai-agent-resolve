"""S1h4 round 2, C1: an element past the 4096-attribute cap must not blind the diff gate.

The scanner keeps 4096 attributes per element and drops the rest; ``diff_opaque`` compares only
what was kept. At 410a6b3 the drop was silent, so two files that differed only past attribute
4096 (a changed value, an added attribute, a repeated name, a second ``DbId``) passed
``aar drp diff --gate``. Each case below reads two real zips and runs the gate; it fails on
410a6b3 and passes now because the reader warns (``scan-attributes-truncated``) and the gate
refuses a file that was read with warnings. The last test goes through S8's post-write check.

Round 3 (MED): the plain ``aar drp diff`` (no ``--gate``) said "0 changes" and ``"identical": true``
for those same pairs, where fbfdcc5 reported the opaque change. The tests at the bottom pin that
the plain diff reports a pair as different, says why in a note, and still reports a pair whose
over-cap elements are identical as identical.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from diff_testing import diff
from drp_helpers import xml

from aar.cli import main
from aar.drp.modify import verify_write
from aar.drp.reader import read_drp

_CAP = 4096


def _attrs(count: int) -> str:
    return "".join(f' k{i}="v"' for i in range(count))


def _drp(path: Path, project_body: str, *, root_attrs: str = ' DbId="p1"') -> Path:
    members = {
        "project.xml": xml(f"<SM_Project{root_attrs}>{project_body}</SM_Project>\n"),
        "Gallery.xml": xml('<Gallery::GyGallery DbId="g"/>\n'),
        "MediaPool/Master/MpFolder.xml": xml(
            '<Sm2MpFolder DbId="b1"><Name>Master</Name><MediaVec/></Sm2MpFolder>\n'
        ),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


_NAME = "<ProjectName>P</ProjectName>"
# (body of A, body of B, root attributes of A, root attributes of B) -- the validator's A1/A2/A3/A8
_CASES = {
    "A1-value-changed-past-cap": (
        f'{_NAME}<Junk{_attrs(_CAP)} last="X"/>',
        f'{_NAME}<Junk{_attrs(_CAP)} last="Y"/>',
        ' DbId="p1"',
        ' DbId="p1"',
    ),
    "A2-attribute-added-past-cap": (
        f"{_NAME}<Junk{_attrs(_CAP)}/>",
        f'{_NAME}<Junk{_attrs(_CAP)} extra="1"/>',
        ' DbId="p1"',
        ' DbId="p1"',
    ),
    "A3-name-repeated-past-cap": (
        f"{_NAME}<Junk{_attrs(_CAP)}/>",
        f'{_NAME}<Junk{_attrs(_CAP)} k0="CHANGED"/>',
        ' DbId="p1"',
        ' DbId="p1"',
    ),
    "A8-second-dbid-past-cap": (
        _NAME,
        _NAME,
        ' DbId="p1"',
        ' DbId="p1"' + _attrs(_CAP - 1) + ' DbId="p2"',
    ),
}


def _pair(tmp_path: Path, case: str) -> tuple[Path, Path]:
    body_a, body_b, root_a, root_b = _CASES[case]
    return (
        _drp(tmp_path / "a.drp", body_a, root_attrs=root_a),
        _drp(tmp_path / "b.drp", body_b, root_attrs=root_b),
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_a_change_past_the_attribute_cap_fails_the_gate(tmp_path: Path, case: str) -> None:
    a, b = _pair(tmp_path, case)
    gate = diff(a, b, gate=True).gate
    assert not gate.passed, f"{case}: the gate passed over an element it could only half read"
    assert any("read with warnings" in r and "scan-attributes-truncated" in r for r in gate.reasons)


@pytest.mark.parametrize("case", list(_CASES))
def test_the_reader_warns_about_the_dropped_attributes(tmp_path: Path, case: str) -> None:
    a, b = _pair(tmp_path, case)
    warned = {i.code for path in (a, b) for i in read_drp(path).issues if i.severity == "warning"}
    assert "scan-attributes-truncated" in warned


def test_an_element_at_exactly_the_cap_is_compared_whole(tmp_path: Path) -> None:
    # The control: 4096 attributes lose nothing, so a change at the last one is a plain change.
    a = _drp(tmp_path / "a.drp", f'{_NAME}<Junk{_attrs(_CAP - 1)} last="X"/>')
    b = _drp(tmp_path / "b.drp", f'{_NAME}<Junk{_attrs(_CAP - 1)} last="Y"/>')
    d = diff(a, b, gate=True)
    assert not [i for i in read_drp(b).issues if i.code == "scan-attributes-truncated"]
    assert any("last" in c.detail for c in d.changes), d.to_text()


@pytest.mark.parametrize("case", ["A1-value-changed-past-cap", "A3-name-repeated-past-cap"])
def test_the_post_write_check_refuses_a_change_past_the_cap(tmp_path: Path, case: str) -> None:
    # S8's verify_write runs diff_projects(gate=True): a write that altered attribute 4097 must
    # not verify just because the comparison could not see it.
    a, b = _pair(tmp_path, case)
    check = verify_write(a, b, [], {"project.xml"})
    assert not check.ok
    refusal = next(r for r in check.refusals if r.code == "gate-failed")
    assert "scan-attributes-truncated" in refusal.message


# ---- round 3: the plain diff (no --gate) must not call these pairs identical -------------------

_NOTE = (
    "an element has more than 4096 attributes; the attributes past that point are compared as "
    "raw bytes, not by name"
)
_PAST_CAP = "attributes past the first 4096 differ"
# A8 differs in the 4095 attributes the root carries on one side only, so the existing by-name
# comparison already reports it; A1-A3 differ only past the cap.
_PAST_CAP_ONLY = [
    "A1-value-changed-past-cap",
    "A2-attribute-added-past-cap",
    "A3-name-repeated-past-cap",
]


@pytest.mark.parametrize("swap", [False, True], ids=["a-to-b", "b-to-a"])
@pytest.mark.parametrize("case", _PAST_CAP_ONLY)
def test_plain_diff_reports_a_change_past_the_cap(tmp_path: Path, case: str, swap: bool) -> None:
    a, b = _pair(tmp_path, case)
    d = diff(b, a) if swap else diff(a, b)
    (c,) = d.changes
    assert (c.section, c.kind, c.field_name) == ("opaque", "opaque-changed", "Junk")
    assert _PAST_CAP in c.detail and c.byte_count is not None and c.byte_count > 0
    assert c.where_a is not None and c.where_a.member == "project.xml"
    assert c.where_b is not None and c.where_b.member == "project.xml"
    assert not d.to_dict()["identical"]


@pytest.mark.parametrize("case", list(_CASES))
def test_plain_diff_names_the_truncation_in_a_note(tmp_path: Path, case: str) -> None:
    a, b = _pair(tmp_path, case)
    d = diff(a, b)
    assert d.changes, f"{case}: the plain diff reported no change"
    notes = [n for n in d.notes if "attributes" in n]
    # Only A1 has an element over the cap on side A; in the others only B's is, so only B warns.
    sides = {n.split(":", 1)[0] for n in notes}
    expected = {"A1-value-changed-past-cap": {"A", "B"}}.get(case, {"B"})
    assert sides >= expected and all(n.split(": ", 1)[1] == _NOTE for n in notes)


def test_plain_diff_cli_text_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _pair(tmp_path, "A1-value-changed-past-cap")
    code = main(["drp", "diff", str(a), str(b)])
    out = capsys.readouterr().out.splitlines()
    assert code == 1
    assert any("opaque <Junk> changed" in line and _PAST_CAP in line for line in out)
    assert f"note: A: {_NOTE}" in out and f"note: B: {_NOTE}" in out
    assert out[-1].startswith(
        "summary: 1 change (header 0, media 0, timelines 0, markers 0, opaque 1)"
    )
    code = main(["drp", "diff", "--json", str(a), str(b)])
    doc = json.loads(capsys.readouterr().out)
    assert code == 1 and doc["identical"] is False and doc["summary"]["changes"] == 1
    assert doc["notes"] == [f"A: {_NOTE}", f"B: {_NOTE}"]


def test_plain_diff_still_calls_identical_over_cap_elements_identical(tmp_path: Path) -> None:
    # Same element over the cap on both sides, byte for byte: nothing to report for it. The files
    # differ elsewhere (the project name) so the member is really walked, not skipped as equal.
    junk = f'<Junk{_attrs(_CAP + 5)} last="X"/>'
    a = _drp(tmp_path / "a.drp", f"<ProjectName>P</ProjectName>{junk}")
    b = _drp(tmp_path / "b.drp", f"<ProjectName>Q</ProjectName>{junk}")
    d = diff(a, b)
    assert d.changes and not [c for c in d.changes if c.section == "opaque"], d.to_text()
    assert not [c for c in d.changes if c.field_name == "Junk"]
    assert any("attributes" in n for n in d.notes)  # the warning still shows
    # the same two files with identical bytes are identical, with no change and no hidden pass
    same = diff(a, a)
    assert not same.changes and same.to_dict()["identical"]


def test_plain_diff_leaves_the_quote_style_of_kept_attributes_alone(tmp_path: Path) -> None:
    # Quote style in the attributes the scanner kept is not a change anywhere; over the cap it
    # must not become one either (only the unread tail is compared).
    head_dq = _attrs(_CAP)
    head_sq = head_dq.replace('="v"', "='v'")
    a = _drp(tmp_path / "a.drp", f'{_NAME}<Junk{head_dq} last="X"/>')
    b = _drp(tmp_path / "b.drp", f'{_NAME}<Junk{head_sq} last="X"/>')
    d = diff(a, b)
    assert not d.changes, d.to_text()
