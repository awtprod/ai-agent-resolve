"""Tests for ``aar drp diff``: output, ``--json``, ``--only`` and diff(1) exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from drp_edit import after, edit_zip, member_named

from aar.cli import main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"
H264 = FIXTURES / "media-clip-h264.drp"
R19 = FIXTURES / "media-clip-r19.drp"


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["drp", "diff", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_identical_files_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, str(H264), str(H264))
    assert code == 0 and err == ""
    assert out.splitlines()[-1].startswith("summary: 0 changes (header 0, media 0, timelines 0")


def test_different_files_exit_1_with_one_line_per_change(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    seq = member_named(H264, "SeqContainer/")
    v1 = b'DbId="ad1fd562-da22-4b1f-b79f-f3106f9c2ac5"'
    b = edit_zip(H264, tmp_path / "b.drp", {seq: after(v1, b"<Duration>3664<", b"<Duration>3652<")})
    code, out, _ = run(capsys, str(H264), str(b))
    assert code == 1
    lines = out.splitlines()
    assert lines[2] == (
        'Timeline "MediaTemplate" V1 item "sample.mp4" (DbId ad1fd562…): '
        "tail trimmed -12f (01:02:32:16 → 01:02:32:04); "
        "source out -12f (00:02:32:16 → 00:02:32:04)"
    )
    assert lines[-1].startswith("summary: 1 change (header 0, media 0, timelines 1")


def test_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--json", str(R19), str(H264))
    assert code == 1
    data = json.loads(out)
    assert data["identical"] is False
    assert data["a"]["project_version"] == 14 and data["b"]["project_version"] == 17
    assert data["summary"]["changes"] == len(data["changes"])
    kinds = {(c["section"], c["field"]) for c in data["changes"] if c["section"] == "header"}
    assert {("header", "DbAppVer"), ("header", "DbPrjVer")} <= kinds


def test_only_filters_sections_and_sets_the_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--only", "markers", str(R19), str(H264))
    assert code == 0
    assert out.splitlines()[-1] == "summary: 0 changes (markers 0); 0 heuristic matches"
    code, out, _ = run(capsys, "--only", "header", "--only", "media", str(R19), str(H264))
    assert code == 1
    body = [x for x in out.splitlines()[2:-1] if not x.startswith("note: ")]
    assert body and all(line.startswith(("Header", "Project", "Media")) for line in body)
    assert out.splitlines()[-1].startswith("summary: 11 changes (header 6, media 5)")


def test_missing_file_exits_2(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out, err = run(capsys, str(H264), str(tmp_path / "missing.drp"))
    assert code == 2 and out == ""
    assert err.startswith("aar drp diff: cannot read")


def test_bad_section_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, "--only", "nope", str(H264), str(H264))
    assert code == 2 and "invalid choice" in err
