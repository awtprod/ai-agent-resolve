"""Tests for ``aar drp inspect`` and ``aar drp describe``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from drp_helpers import needs_zstandard

from aar.cli import main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"


@needs_zstandard
def test_describe_prints_timeline_and_items(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["drp", "describe", str(FIXTURES / "media-clip-h264.drp")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("DRP file (suffix): Resolve 21.0.0.0048 (DbPrjVer 17)")
    assert (
        'Timeline "MediaTemplate": 24 fps, 1920x1080, start 01:00:00:00 (frame 86400), '
        "length 00:02:32:16 (3664 frames)"
    ) in out
    assert (
        '    01:00:00:00-01:02:32:16 video "sample.mp4" src 00:00:00:00-00:02:32:16 '
        "media 'sample.mp4'"
    ) in out
    assert "[video] sample.mp4; 4576 frames @ 29.97 fps; 352x262" in out
    assert "differs from the Media Pool path" in out


def test_describe_max_items_truncates(capsys: pytest.CaptureFixture[str]) -> None:
    path = str(FIXTURES / "media-clip-r19.drp")
    assert main(["drp", "describe", "--max-items", "0", path]) == 0
    full = capsys.readouterr().out
    assert "A16: 0 item(s)" in full and "more" not in full
    assert "[video] r19_template_media.mp4" in full
    assert main(["drp", "describe", "--max-items", "1", path]) == 0
    capped = capsys.readouterr().out
    assert "[video] r19_template_media.mp4" in capped and "more" not in capped


def test_inspect_json_is_the_full_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["drp", "inspect", "--json", str(FIXTURES / "media-clip-r19.drp")]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "drp"
    assert data["versions"]["app_version"] == "19.1.3.0007"
    assert data["versions"]["project_version"] == 14
    [tl] = data["timelines"]
    assert tl["resolution"] == [1920, 1080]
    item = tl["tracks"][0]["items"][0]
    assert item["record_start"] == 86400
    assert item["spans"]["name"]["member"].startswith("SeqContainer/")
    assert {b["path"] for b in data["bins"]} == {"Master", "Master/Archive"}


@needs_zstandard
def test_inspect_with_blob_stats(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["drp", "inspect", "--blobs", str(FIXTURES / "media-clip-h264.drp")]) == 0
    out = capsys.readouterr().out
    assert "DbAppVer: 21.0.0.0048  DbPrjVer: 17" in out
    assert "Clip: envelope-zstd=2" in out
    assert "FAILED" not in out
    assert "not decoded (kept raw) x1: PreConformMediaExtents" in out


def test_unreadable_file_returns_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["drp", "describe", str(tmp_path / "missing.drp")]) == 1
    assert "aar drp: cannot read" in capsys.readouterr().err


def test_drp_without_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["drp"]) == 2
    assert "usage: aar drp" in capsys.readouterr().err
