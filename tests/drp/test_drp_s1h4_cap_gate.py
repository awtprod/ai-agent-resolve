"""S1h4 round 2, C1: an element past the 4096-attribute cap must not blind the diff gate.

The scanner keeps 4096 attributes per element and drops the rest; ``diff_opaque`` compares only
what was kept. At 410a6b3 the drop was silent, so two files that differed only past attribute
4096 (a changed value, an added attribute, a repeated name, a second ``DbId``) passed
``aar drp diff --gate``. Each case below reads two real zips and runs the gate; it fails on
410a6b3 and passes now because the reader warns (``scan-attributes-truncated``) and the gate
refuses a file that was read with warnings. The last test goes through S8's post-write check.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from diff_testing import diff
from drp_helpers import xml

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
