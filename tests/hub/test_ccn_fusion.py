"""Tests for Fusion graphics & extra_sources in ``ccn-longform`` params, stub, and handoff notes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aar.fusion.render import render_fusion_clip
from aar.media import resolve_tool
from aar.plan.stub import StubSource, build_stub
from aar.recipes.ccn.handoff import handoff_notes
from aar.recipes.ccn.params import CcnLongformParams

PIL = pytest.importorskip("PIL.Image")


SIMPLE_SETTING = """{
\tTools = ordered() {
\t\tSimpleGraphic = MacroOperator {
\t\t\tInputs = ordered() { },
\t\t\tOutputs = {
\t\t\t\tMainOutput1 = InstanceOutput { SourceOp = "Bg", Source = "Output", },
\t\t\t},
\t\t\tTools = ordered() {
\t\t\t\tBg = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 320, },
\t\t\t\t\t\tHeight = Input { Value = 180, },
\t\t\t\t\t\tTopLeftRed = Input { Value = 1, },
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 1, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
"""


def _make_camera(path: Path) -> Path:
    ffmpeg = str(resolve_tool("ffmpeg"))
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:r=24:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-timecode",
            "00:00:00:00",
            str(path),
        ],
        check=True,
    )
    return path


def test_ccn_params_extra_sources_and_fusion_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAR_MEDIA_ROOT", str(tmp_path))
    cam = tmp_path / "cam_a.mp4"
    gfx = tmp_path / "lower_third.mov"
    tpl = tmp_path / "LowerThird.setting"
    cam.write_bytes(b"cam")
    gfx.write_bytes(b"gfx")
    tpl.write_text(SIMPLE_SETTING, encoding="utf-8")

    params = CcnLongformParams.model_validate(
        {
            "sources": [{"path": str(cam), "camera": "A"}],
            "extra_sources": [{"path": str(gfx), "id": "gfx-lt"}],
            "fusion_templates": [str(tpl)],
            "recording_id": "rec-1",
            "timeline": "Show Ep 1",
            "media_map": {str(tmp_path): "D:/Media"},
        }
    )
    assert len(params.extra_sources) == 1
    assert params.fusion_templates == [str(tpl)]
    # pause_trim auto counts only camera sources, not extra_sources
    assert params.pause_trim_on() is False

    # Non-.setting/.drfx file in fusion_templates must be rejected
    bad_tpl = tmp_path / "not_fusion.txt"
    bad_tpl.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.setting or \.drfx"):
        CcnLongformParams.model_validate(
            {
                "sources": [{"path": str(cam)}],
                "fusion_templates": [str(bad_tpl)],
                "recording_id": "rec-1",
                "timeline": "Show Ep 1",
                "media_map": {str(tmp_path): "D:/Media"},
            }
        )

    # Duplicate across sources and extra_sources is rejected
    with pytest.raises(ValueError, match="already given"):
        CcnLongformParams.model_validate(
            {
                "sources": [{"path": str(cam)}],
                "extra_sources": [{"path": str(cam)}],
                "recording_id": "rec-1",
                "timeline": "Show Ep 1",
                "media_map": {str(tmp_path): "D:/Media"},
            }
        )


def test_build_stub_with_video_only_alpha_extra_source(tmp_path: Path) -> None:
    cam = _make_camera(tmp_path / "cam.mp4")
    tpl = tmp_path / "SimpleGraphic.setting"
    tpl.write_text(SIMPLE_SETTING, encoding="utf-8")
    gfx = tmp_path / "gfx.mov"
    render_fusion_clip(
        tpl,
        gfx,
        duration_s=1.0,
        fps=24.0,
        resolution=(320, 180),
        codec="qtrle",
    )

    stub = build_stub(
        [StubSource(cam, camera="A")],
        extra_sources=[StubSource(gfx, id="gfx-1")],
        recording_id="rec-1",
        project="CCN",
        timeline="Ep 1",
        kind="longform",
        profile="default",
    )
    by_id = stub.plan.source_map()
    assert by_id["cam-a"].recording_id == "rec-1"
    assert by_id["gfx-1"].recording_id is None
    assert by_id["gfx-1"].duration == 24
    assert by_id["gfx-1"].start_timecode == 0


def test_handoff_notes_includes_fusion_graphics_section() -> None:
    report = {
        "not_expressible": [],
        "media": [{"windows_path": "D:/Media/cam.mp4"}],
        "import": ["Import the FCPXML."],
        "notes": [],
        "fusion": {
            "schema_version": 1,
            "timeline_name": "Ep 1",
            "fps": 24.0,
            "resolution": [1920, 1080],
            "placements": [
                {
                    "index": 0,
                    "kind": "lower_third",
                    "setting": "CCN_LowerThird.setting",
                    "track": 2,
                    "record_in_s": 5.0,
                    "record_out_s": 10.0,
                    "duration_s": 5.0,
                    "lines": [],
                    "inputs": {"Name": "Jerry Morrison", "Title": "Host"},
                }
            ],
        },
    }
    text = handoff_notes(
        project="CCN",
        timeline="Ep 1",
        round_=1,
        report=report,
        report_id="art-1",
        loudness=None,
    )
    assert "FUSION GRAPHICS (1)" in text
    assert "CCN_LowerThird.setting" in text
    assert "Name=Jerry Morrison" in text
