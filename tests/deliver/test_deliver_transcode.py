"""``FfmpegTranscoder``: the argv it builds, the decode check, and log cleanup (real ffmpeg)."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
from deliver_support import make_source

import aar.deliver.transcode as transcode
from aar.deliver.errors import DeliverError
from aar.deliver.transcode import FfmpegTranscoder
from aar.media import input_url


def _run(tmp_path: Path) -> tuple[FfmpegTranscoder, Path, str, Path]:
    src = make_source(tmp_path / "src.mp4", width=320, height=180, duration=1.0, fps=10)
    out = tmp_path / "out.mp4.part"
    pl = str(tmp_path / "ffpass")
    tc = FfmpegTranscoder()
    tc.transcode(
        str(src),
        str(out),
        width=1280,
        height=720,
        video_kbps=500,
        audio_kbps=128,
        passlogfile=pl,
        timeout_s=120.0,
    )
    return tc, src, pl, out


def test_argv_pins_and_log_cleanup(tmp_path: Path) -> None:
    tc, src, pl, out = _run(tmp_path)
    one, two = tc.argv_log
    assert "+faststart" in two  # V39
    assert "0:a:0?" in two  # V40
    assert "scale=1280:720" in two and "scale=1280:720" in one  # V41
    assert f"file:{src}" in two or input_url(str(src)) in two  # V42
    assert input_url(str(src)) in two
    assert two[two.index("-b:a") + 1] == "128k"  # V44
    assert one[one.index("-passlogfile") + 1] == pl  # V45
    assert two[two.index("-passlogfile") + 1] == pl
    assert glob.glob(glob.escape(pl) + "-*.log*") == []  # F12
    assert out.stat().st_size > 0


def test_v43_undecodable_output_raises_transcode_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_tool(tool, argv, **kw):
        if "-pass" in argv and argv[argv.index("-pass") + 1] == "2":
            Path(argv[-1]).write_bytes(b"garbage" * 100)

    monkeypatch.setattr(transcode, "run_tool", fake_run_tool)
    out = tmp_path / "out.mp4.part"
    with pytest.raises(DeliverError) as info:
        FfmpegTranscoder().transcode(
            str(tmp_path / "src.mp4"),
            str(out),
            width=640,
            height=360,
            video_kbps=500,
            audio_kbps=128,
            passlogfile=str(tmp_path / "ffpass"),
            timeout_s=60.0,
        )
    assert info.value.code == "transcode_failed"
