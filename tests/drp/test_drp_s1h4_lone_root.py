"""S1h4 round 3, HIGH: a model root that exists on one side only must be diffed, not crash.

``diff_opaque._xml_member`` hands the lone root to ``_lone`` with a path. Round 2 turned paths
into parent-linked ``_Path`` objects but left two call sites on the old ``(tag,)`` tuple, so a
member whose root is a model object on one side only (the bin root renamed or replaced in B)
raised ``AttributeError: 'tuple' object has no attribute 'text'``: ``aar drp diff`` and
``--gate`` both exited 2 with a traceback where fbfdcc5 exits 1 and reports two changes.

Each case below builds a real zip pair, runs the CLI in text, ``--json`` and ``--gate`` form in
both directions, and pins the exact change lines. The expected strings are what fbfdcc5 prints,
so they also fail if the path text of a one-sided carrier goes empty (the validator's mutant P07).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from diff_testing import H264, MARKER_BLOB_ID, SEQUENCE, diff, edited
from drp_edit import add_carrier, carrier, kd_bytes
from drp_helpers import envelope, kd, pb_var, xml

from aar.cli import main
from aar.drp.model import Provenance

_BIN_A = '<Sm2MpFolder DbId="b1"><Name>Master</Name><MediaVec/></Sm2MpFolder>\n'
_MEMBER = "MediaPool/Master/MpFolder.xml"
# case -> (root tag in B, bin xml in B, bytes of that element, `end` offset of the element)
_CASES = {
    "root-renamed": (
        "Sm2MpFolderX",
        '<Sm2MpFolderX DbId="b1"><Name>Master</Name><MediaVec/></Sm2MpFolderX>\n',
        69,
    ),
    "root-not-a-model-unit": (
        "NotAUnit",
        '<NotAUnit DbId="b1"><Name>Master</Name></NotAUnit>\n',
        50,
    ),
}
_NOTE = (
    "note: no DbId is shared between A and B: they look like different projects, so objects show "
    "as added and removed rather than changed"
)
_SUMMARY = (
    "summary: 2 changes (header 0, media 1, timelines 0, markers 0, opaque 1); 0 heuristic matches"
)


def _drp(path: Path, bin_xml: str) -> Path:
    members = {
        "project.xml": xml('<SM_Project DbId="p1"><ProjectName>P</ProjectName></SM_Project>\n'),
        "Gallery.xml": xml('<Gallery::GyGallery DbId="g"/>\n'),
        _MEMBER: xml(bin_xml),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def _run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["drp", "diff", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def _pair(tmp_path: Path, case: str, *, reverse: bool) -> tuple[Path, Path, str, int]:
    tag, bin_b, size = _CASES[case]
    a = _drp(tmp_path / "a.drp", _BIN_A)
    b = _drp(tmp_path / "b.drp", bin_b)
    return (b, a, tag, size) if reverse else (a, b, tag, size)


def _change_lines(case_tag: str, size: int, *, reverse: bool) -> list[str]:
    side = "removed" if reverse else "added"
    bin_side = "added" if reverse else "removed"
    return [
        f'Media pool bin "Master" (DbId b1): {bin_side} (0 item(s))',
        f"Member {_MEMBER}: opaque <{case_tag}> {side} ({size} bytes, {_MEMBER} @83)",
        _NOTE,
    ]


@pytest.mark.parametrize("reverse", [False, True], ids=["a-to-b", "b-to-a"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_lone_root_text_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, case: str, reverse: bool
) -> None:
    a, b, tag, size = _pair(tmp_path, case, reverse=reverse)
    code, out, err = _run(capsys, str(a), str(b))
    assert (code, err) == (1, "")
    assert out.splitlines()[2:] == [*_change_lines(tag, size, reverse=reverse), _SUMMARY]


@pytest.mark.parametrize("reverse", [False, True], ids=["a-to-b", "b-to-a"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_lone_root_gate_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, case: str, reverse: bool
) -> None:
    a, b, tag, size = _pair(tmp_path, case, reverse=reverse)
    code, out, err = _run(capsys, "--gate", str(a), str(b))
    assert (code, err) == (1, "")
    warned = "A" if reverse else "B"
    gate = (
        "gate: FAIL: 1 opaque change(s): bytes differ that no decoded value explains; "
        f"{warned}: read with warnings (unexpected-root)"
    )
    assert out.splitlines()[2:] == [*_change_lines(tag, size, reverse=reverse), gate, _SUMMARY]


@pytest.mark.parametrize("reverse", [False, True], ids=["a-to-b", "b-to-a"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_lone_root_json_fields(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, case: str, reverse: bool
) -> None:
    a, b, tag, size = _pair(tmp_path, case, reverse=reverse)
    code, out, err = _run(capsys, "--json", str(a), str(b))
    assert (code, err) == (1, "")
    doc = json.loads(out)
    assert doc["summary"]["changes"] == 2 and doc["summary"]["by_section"]["opaque"] == 1
    (opaque,) = (c for c in doc["changes"] if c["section"] == "opaque")
    kind = "opaque-removed" if reverse else "opaque-added"
    assert (opaque["kind"], opaque["field"], opaque["object_id"]) == (kind, tag, _MEMBER)
    assert (
        opaque["byte_count"] == size
        and opaque["text"] == _change_lines(tag, size, reverse=reverse)[1]
    )
    span = {"member": _MEMBER, "kind": "element", "blob_path": None, "start": 83, "end": 83 + size}
    assert (opaque["where_a"], opaque["where_b"]) == ((span, None) if reverse else (None, span))


@pytest.mark.parametrize("reverse", [False, True], ids=["a-to-b", "b-to-a"])
@pytest.mark.parametrize("gate", [False, True], ids=["plain", "gate"])
def test_one_sided_marker_blob_the_reader_cannot_decode(
    tmp_path: Path, reverse: bool, gate: bool
) -> None:
    """The other path that builds a ``_Path`` for a lone element: ``_lone_carrier``.

    A marker blob on one side only whose list the reader cannot decode is reported as an opaque
    add or remove; the reported path (``field_name``) is the blob's tag, not empty (mutant P07).
    """
    garbage = envelope(10001, 0x80, pb_var(3, 7))  # a protobuf that is not a marker list
    blob = kd([kd_bytes("BlobData", garbage)]).hex()
    b = edited(
        tmp_path, "b.drp", {"project.xml": add_carrier(carrier(MARKER_BLOB_ID, SEQUENCE, blob))}
    )
    d = diff(b, H264, gate=gate) if reverse else diff(H264, b, gate=gate)
    (c,) = d.changes
    verb = "removed" if reverse else "added"
    span = Provenance("project.xml", 34401, 34696, "element")
    assert (c.section, c.kind, c.object_kind, c.field_name) == (
        "opaque",
        f"opaque-{verb}",
        "markers",
        "Sm2SequenceLockableBlob",
    )
    assert c.byte_count == 295
    assert (c.where_a, c.where_b) == ((span, None) if reverse else (None, span))
    assert c.detail == (
        f"opaque <Sm2SequenceLockableBlob> {verb} (295 bytes, project.xml @34401); "
        "marker blob with content the reader does not decode"
    )
    assert not d.gate.passed
    assert d.gate.reasons[0] == "1 opaque change(s): bytes differ that no decoded value explains"
