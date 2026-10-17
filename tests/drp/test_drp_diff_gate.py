"""The S8 post-write gate (``diff_projects(..., gate=True)``, ``aar drp diff --gate``) and
unreadable inputs, which are errors (exit 2) in every mode."""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from diff_testing import (
    A1_ITEM,
    H264,
    POOL,
    SEQ_MEMBER,
    V1,
    V1_ITEM,
    diff,
    edited,
    marker_file,
    only_change,
)
from drp_edit import (
    after,
    chain,
    edit_zip,
    element_span,
    every,
    flip_hex_byte,
    marker_entry,
    once,
    rewrite_clip_blobs,
    set_fields,
)

from aar.cli import main
from aar.drp.diff import DiffError, diff_projects
from aar.drp.errors import DrpError
from aar.drp.reader import read_drp

EDITS: dict[str, dict[str, Callable[[bytes], bytes]]] = {
    "trim": {SEQ_MEMBER: after(V1, b"<Duration>3664<", b"<Duration>3652<")},
    "move": {SEQ_MEMBER: after(V1, b"<Start>86400<", b"<Start>86424<")},
    "rename": {POOL: every(b"<Name>MediaTemplate<", b"<Name>Cut 2<", 2)},
    "relink": {POOL: rewrite_clip_blobs(set_fields({1: b"/Volumes/Media", 2: b"a.mp4"}))},
}


@pytest.mark.parametrize("name", sorted(EDITS))
def test_gate_passes_decoded_dbid_matched_edits(name: str, tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", EDITS[name])
    d = diff(H264, b, gate=True)
    assert len(d.changes) == 1 and d.gate.passed, d.gate.reasons
    assert d.to_text().splitlines()[-2].startswith("gate: PASS")


def test_gate_passes_identical_files_and_marker_adds(tmp_path: Path) -> None:
    assert diff(H264, H264, gate=True).gate.passed
    d = diff(H264, marker_file(tmp_path, "m.drp", marker_entry(48, 32, "Laugh")), gate=True)
    assert [c.match for c in d.changes] == ["owner+frame"] and d.gate.passed


def test_gate_fails_on_opaque_changes(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: flip_hex_byte(V1, b"FieldsBlob", 40)})
    d = diff(H264, b, gate=True)
    assert not d.gate.passed
    assert d.gate.reasons == ("1 opaque change(s): bytes differ that no decoded value explains",)
    assert d.to_text().splitlines()[-2].startswith("gate: FAIL: 1 opaque change(s)")


def test_gate_fails_on_heuristic_matches(tmp_path: Path) -> None:
    strip = chain(
        once(f' DbId="{V1_ITEM}"'.encode(), b""), once(f' DbId="{A1_ITEM}"'.encode(), b"")
    )
    a = edited(tmp_path, "a.drp", {SEQ_MEMBER: strip})
    b = edited(
        tmp_path,
        "b.drp",
        {SEQ_MEMBER: chain(after(V1, b"<Duration>3664<", b"<Duration>3600<"), strip)},
    )
    d = diff(a, b, gate=True)
    assert [c.match for c in d.changes] == ["heuristic"]
    assert d.changes[0].text.endswith("[heuristic match]")
    assert d.gate.reasons == ("1 change(s) on objects not matched by DbId",)


def duplicated_item(data: bytes) -> bytes:
    """The V1 item twice with the same DbId, the copy placed after it."""
    start, end = element_span(data, V1_ITEM)
    piece = data[start:end]
    return data[:end] + b"\n     " + piece.replace(b"<Start>86400<", b"<Start>90064<") + data[end:]


def test_gate_fails_on_duplicate_dbids(tmp_path: Path) -> None:
    a = edited(tmp_path, "a.drp", {SEQ_MEMBER: duplicated_item})
    b = edited(
        tmp_path,
        "b.drp",
        {SEQ_MEMBER: chain(duplicated_item, after(V1, b"<Duration>3664<", b"<Duration>3652<"))},
    )
    d = diff(a, b, gate=True)
    reason = f"A: DbId {V1_ITEM} occurs 2 times among items"
    assert (
        reason in d.gate.reasons
        and f"B: DbId {V1_ITEM} occurs 2 times among items" in d.gate.reasons
    )
    assert any(n.startswith(f"duplicate DbId: {reason}") for n in d.notes)
    assert {c.match for c in d.changes} == {"heuristic"}


def test_gate_fails_on_a_degraded_read(tmp_path: Path) -> None:
    def odd_hex(data: bytes) -> bytes:
        at = data.index(V1)
        i = data.index(b"<MediaTimemapBA>", at) + len(b"<MediaTimemapBA>")
        return data[:i] + b"0" + data[i:]  # odd hex length: the reader warns bad-hex

    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: odd_hex})
    assert any(i.code == "bad-hex" for i in read_drp(b).issues)
    d = diff(H264, b, gate=True)
    assert "B: read with warnings (bad-hex)" in d.gate.reasons


def test_gate_mode_pairs_no_markers_by_name(tmp_path: Path) -> None:
    a = marker_file(tmp_path, "a.drp", marker_entry(48, 32, "Laugh"))
    b = marker_file(tmp_path, "b.drp", marker_entry(60, 32, "Laugh"))
    loose = only_change(a, b)
    assert (loose.kind, loose.match) == ("moved", "heuristic")
    assert not diff(a, b).gate.passed
    strict = diff(a, b, gate=True)
    assert sorted((c.kind, c.match) for c in strict.changes) == [
        ("added", "owner+frame"),
        ("removed", "owner+frame"),
    ]
    assert strict.gate.passed


def test_cli_gate_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ok = edited(tmp_path, "ok.drp", EDITS["trim"])
    bad = edited(tmp_path, "bad.drp", {SEQ_MEMBER: flip_hex_byte(V1, b"FieldsBlob", 40)})
    assert main(["drp", "diff", "--gate", str(H264), str(ok)]) == 0
    assert "gate: PASS" in capsys.readouterr().out
    assert main(["drp", "diff", "--gate", str(H264), str(bad)]) == 1
    assert "gate: FAIL: 1 opaque change(s)" in capsys.readouterr().out
    assert main(["drp", "diff", "--gate", "--only", "timelines", str(H264), str(bad)]) == 1


# --------------------------------------------------------------------------- unreadable inputs


def truncated(tmp_path: Path) -> Path:
    data = H264.read_bytes()
    path = tmp_path / "truncated.drp"
    path.write_bytes(data[: len(data) // 2])
    return path


def bad_crc(tmp_path: Path) -> Path:
    stored = edit_zip(H264, tmp_path / "stored.drp", stored=True)
    data = bytearray(stored.read_bytes())
    at = data.index(b"<ProjectName>_media_tmpl</ProjectName>") + len(b"<ProjectName>")
    data[at] ^= 0x01
    path = tmp_path / "badcrc.drp"
    path.write_bytes(bytes(data))
    return path


def no_project(tmp_path: Path) -> Path:
    path = tmp_path / "noproject.drp"
    with zipfile.ZipFile(H264) as zin, zipfile.ZipFile(path, "w") as zout:
        for info in zin.infolist():
            if info.filename != "project.xml":
                zout.writestr(info, zin.read(info))
    return path


@pytest.mark.parametrize("make", [truncated, bad_crc, no_project])
@pytest.mark.parametrize("gate", [False, True])
def test_unreadable_inputs_exit_2(
    make: Callable[[Path], Path], gate: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = make(tmp_path)
    args = ["drp", "diff", *(["--gate"] if gate else []), str(H264), str(broken)]
    assert main(args) == 2
    out = capsys.readouterr()
    assert out.out == "" and out.err.startswith("aar drp diff: ")


@pytest.mark.parametrize("make", [truncated, bad_crc, no_project])
def test_unreadable_inputs_raise_in_the_api(make: Callable[[Path], Path], tmp_path: Path) -> None:
    broken = make(tmp_path)
    try:
        drp = read_drp(broken)
    except DrpError:  # the reader may refuse the file itself (S1h hardening); also an error
        return
    with pytest.raises(DiffError, match="B "):
        diff_projects(read_drp(H264), drp)
