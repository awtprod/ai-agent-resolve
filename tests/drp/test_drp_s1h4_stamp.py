"""S1h4 item 5: what DESIGN.md (S2 Diff) says about a version-stamp bump.

Bumping the stamp in only ``project.xml``, or only in the members the reader interprets, fails
the gate as a ``version-mismatch`` warning. Bumping it in every member fails only through the
raw bytes of the member the reader does not interpret.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from diff_testing import H264, POOL, SEQ_MEMBER, diff, edited
from drp_edit import once

_BUMP = once(b'DbPrjVer="17"', b'DbPrjVer="18"')


@pytest.mark.parametrize(
    "members",
    [
        pytest.param(["project.xml"], id="project.xml only"),
        pytest.param(["project.xml", POOL, SEQ_MEMBER], id="the interpreted members only"),
    ],
)
def test_partial_stamp_bump_fails_the_gate_as_a_version_mismatch(
    tmp_path: Path, members: list[str]
) -> None:
    # DESIGN.md (S2 Diff): the failure is a version-mismatch warning, not an opaque change.
    b = edited(tmp_path, "b.drp", {m: _BUMP for m in members})
    d = diff(H264, b, gate=True)
    assert not d.gate.passed
    assert any("version-mismatch" in reason for reason in d.gate.reasons), d.gate.reasons
    assert not any(c.section == "opaque" for c in d.changes)
    assert [(c.section, c.field_name) for c in d.changes] == [("header", "DbPrjVer")]


def test_stamp_bump_in_every_member_fails_only_through_the_raw_member_bytes(
    tmp_path: Path,
) -> None:
    # The decoded header alone would pass; the failure is the member the reader does not
    # interpret (Gallery.xml), compared as opaque bytes.
    b = edited(
        tmp_path,
        "b.drp",
        {m: _BUMP for m in ("project.xml", POOL, SEQ_MEMBER, "Gallery.xml")},
    )
    d = diff(H264, b, gate=True)
    assert not d.gate.passed
    assert not any("version-mismatch" in reason for reason in d.gate.reasons)
    opaque = [c for c in d.changes if c.section == "opaque"]
    assert [c.object_id for c in opaque] == ["Gallery.xml"]
    assert [(c.section, c.field_name) for c in d.changes if c.section == "header"] == [
        ("header", "DbPrjVer")
    ]
