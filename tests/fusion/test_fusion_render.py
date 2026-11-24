"""Tests for ``aar.fusion.render`` and ``aar fusion render`` / ``manifest`` CLI."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aar.cli import main
from aar.fusion.render import render_fusion_clip, render_fusion_frame
from aar.media import probe

PIL = pytest.importorskip("PIL.Image")


DYNAMIC_BAR_MACRO = """{
\tTools = ordered() {
\t\tDynamicBar = MacroOperator {
\t\t\tInputs = ordered() {
\t\t\t\tInput1 = InstanceInput {
\t\t\t\t\tSourceOp = "LabelText",
\t\t\t\t\tSource = "StyledText",
\t\t\t\t\tName = "Label",
\t\t\t\t},
\t\t\t},
\t\t\tOutputs = {
\t\t\t\tMainOutput1 = InstanceOutput {
\t\t\t\t\tSourceOp = "FinalMerge",
\t\t\t\t\tSource = "Output",
\t\t\t\t},
\t\t\t},
\t\t\tTools = ordered() {
\t\t\t\tCanvas = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 640, },
\t\t\t\t\t\tHeight = Input { Value = 360, },
\t\t\t\t\t\tTopLeftRed = Input { Value = 0, },
\t\t\t\t\t\tTopLeftGreen = Input { Value = 0, },
\t\t\t\t\t\tTopLeftBlue = Input { Value = 0, },
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 0, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tLabelText = TextPlus {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 640, },
\t\t\t\t\t\tHeight = Input { Value = 360, },
\t\t\t\t\t\tStyledText = Input { Value = "HI", },
\t\t\t\t\t\tFont = Input { Value = "DejaVu Sans", },
\t\t\t\t\t\tStyle = Input { Value = "Bold", },
\t\t\t\t\t\tSize = Input { Value = 0.08, },
\t\t\t\t\t\tCenter = Input { Value = { 0.1, 0.25 }, },
\t\t\t\t\t\tHorizontalLeftCenterRight = Input { Value = -1, },
\t\t\t\t\t\tRed1 = Input { Value = 1, },
\t\t\t\t\t\tGreen1 = Input { Value = 1, },
\t\t\t\t\t\tBlue1 = Input { Value = 1, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tBarMask = RectangleMask {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input {
\t\t\t\t\t\t\tValue = 0.2,
    Expression = "(LabelText.Output.DataWindow[3] - LabelText.Output.DataWindow[1]) / 640 + 0.06",
\t\t\t\t\t\t},
\t\t\t\t\t\tHeight = Input { Value = 0.14, },
\t\t\t\t\t\tCenter = Input {
\t\t\t\t\t\t\tValue = { 0.2, 0.25 },
\t\t\t\t\t\t\tExpression = "Point(0.07 + Width / 2, 0.25)",
\t\t\t\t\t\t},
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tBarBg = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 640, },
\t\t\t\t\t\tHeight = Input { Value = 360, },
\t\t\t\t\t\tTopLeftRed = Input { Value = 0.9, },
\t\t\t\t\t\tTopLeftGreen = Input { Value = 0.1, },
\t\t\t\t\t\tTopLeftBlue = Input { Value = 0.1, },
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 1, },
\t\t\t\t\t\tEffectMask = Input { SourceOp = "BarMask", Source = "Mask", },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tBarMerge = Merge {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tBackground = Input { SourceOp = "Canvas", Source = "Output", },
\t\t\t\t\t\tForeground = Input { SourceOp = "BarBg", Source = "Output", },
\t\t\t\t\t\tBlend = Input {
\t\t\t\t\t\t\tValue = 1,
\t\t\t\t\t\t\tExpression = "min(1, time / 10)",
\t\t\t\t\t\t},
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tFinalMerge = Merge {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tBackground = Input { SourceOp = "BarMerge", Source = "Output", },
\t\t\t\t\t\tForeground = Input { SourceOp = "LabelText", Source = "Output", },
\t\t\t\t\t\tBlend = Input {
\t\t\t\t\t\t\tValue = 1,
\t\t\t\t\t\t\tExpression = "min(1, time / 10)",
\t\t\t\t\t\t},
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
"""

LOADER_DRFX_MACRO = """{
\tTools = ordered() {
\t\tBadgeMacro = MacroOperator {
\t\t\tInputs = ordered() { },
\t\t\tOutputs = {
\t\t\t\tMainOutput1 = InstanceOutput {
\t\t\t\t\tSourceOp = "OutMerge",
\t\t\t\t\tSource = "Output",
\t\t\t\t},
\t\t\t},
\t\t\tTools = ordered() {
\t\t\t\tCanvas = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 320, },
\t\t\t\t\t\tHeight = Input { Value = 180, },
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 0, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tBadgeLoader = Loader {
\t\t\t\t\tClips = {
\t\t\t\t\t\tClip {
\t\t\t\t\t\t\tID = "Clip1",
\t\t\t\t\t\t\tFilename = "Templates:\\\\Edit\\\\Titles\\\\badge.png",
\t\t\t\t\t\t},
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tOutMerge = Merge {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tBackground = Input { SourceOp = "Canvas", Source = "Output", },
\t\t\t\t\t\tForeground = Input { SourceOp = "BadgeLoader", Source = "Output", },
\t\t\t\t\t\tCenter = Input { Value = { 0.5, 0.5 }, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
"""


def test_render_frame_dynamic_datawindow_and_animation(tmp_path: Path) -> None:
    setting = tmp_path / "DynamicBar.setting"
    setting.write_text(DYNAMIC_BAR_MACRO, encoding="utf-8")

    # Frame 0: Blend = 0 -> fully transparent canvas
    f0 = render_fusion_frame(
        setting,
        inputs={"Label": "SHORT"},
        resolution=(640, 360),
        time=0.0,
    )
    assert f0.size == (640, 360)
    assert f0.getbbox() is None

    # Frame 10: Blend = 1 -> red bar behind text; longer text produces wider bar!
    short_img = render_fusion_frame(
        setting,
        inputs={"Label": "AI"},
        resolution=(640, 360),
        time=10.0,
    )
    long_img = render_fusion_frame(
        setting,
        inputs={"Label": "LONG SPEAKER NAME HERE"},
        resolution=(640, 360),
        time=10.0,
    )
    short_box = short_img.getbbox()
    long_box = long_img.getbbox()
    assert short_box is not None and long_box is not None
    assert (long_box[2] - long_box[0]) > (short_box[2] - short_box[0]) + 80
    # Top-right corner must remain transparent (alpha == 0)
    assert short_img.getpixel((630, 10))[3] == 0


def test_render_drfx_with_embedded_loader_png(tmp_path: Path) -> None:
    badge = PIL.new("RGBA", (60, 60), (20, 210, 40, 255))
    badge_path = tmp_path / "badge.png"
    badge.save(badge_path, format="PNG")

    drfx = tmp_path / "Badge.drfx"
    with zipfile.ZipFile(drfx, "w") as zf:
        zf.writestr("Edit/Titles/BadgeMacro.setting", LOADER_DRFX_MACRO)
        zf.write(badge_path, "Edit/Titles/badge.png")

    frame = render_fusion_frame(drfx, resolution=(320, 180), time=0.0)
    # Center pixel is green from the embedded badge PNG; corner is transparent
    r, g, _b, a = frame.getpixel((160, 90))
    assert g > 180 and r < 50 and a == 255
    assert frame.getpixel((5, 5))[3] == 0


def test_render_fusion_clip_alpha_mov_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setting = tmp_path / "DynamicBar.setting"
    setting.write_text(DYNAMIC_BAR_MACRO, encoding="utf-8")
    out_mov = tmp_path / "bar.mov"

    res = render_fusion_clip(
        setting,
        out_mov,
        duration_s=0.5,
        fps=24.0,
        resolution=(320, 180),
        inputs={"Label": "HOST"},
        codec="qtrle",
    )
    assert res.frames == 12
    assert res.animated is True
    assert out_mov.is_file()

    info = probe(out_mov)
    assert info.video is not None
    assert info.video.pix_fmt in ("argb", "rgba", "bgra", "yuva444p10le")
    assert info.timecode == "00:00:00:00"

    # CLI render to PNG and MOV
    png_out = tmp_path / "cli_frame.png"
    assert (
        main(
            [
                "fusion",
                "render",
                str(setting),
                "--out",
                str(png_out),
                "--frame",
                "10",
                "--width",
                "320",
                "--height",
                "180",
                "--input",
                "Label=CLI",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert png_out.is_file()


def test_render_bounds_corrupt_loader_and_transform_limit(tmp_path: Path) -> None:
    from aar.fusion.errors import FusionError

    setting = tmp_path / "DynamicBar.setting"
    setting.write_text(DYNAMIC_BAR_MACRO, encoding="utf-8")

    # Oversized target resolution must raise FusionError("invalid_setting")
    with pytest.raises(FusionError) as exc_res:
        render_fusion_frame(setting, resolution=(10000, 10000), time=0.0)
    assert exc_res.value.code == "invalid_setting"

    # Corrupt PNG inside .drfx Loader must raise FusionError("render_failed"), not raw PIL error
    drfx = tmp_path / "CorruptBadge.drfx"
    with zipfile.ZipFile(drfx, "w") as zf:
        zf.writestr("Edit/Titles/BadgeMacro.setting", LOADER_DRFX_MACRO)
        zf.writestr("Edit/Titles/badge.png", b"not a valid png file")
    with pytest.raises(FusionError) as exc_bad:
        render_fusion_frame(drfx, resolution=(320, 180), time=0.0)
    assert exc_bad.value.code == "render_failed"

    # Extreme Transform scale must raise FusionError("invalid_setting")
    huge_xf = """{
\tTools = ordered() {
\t\tHugeXf = MacroOperator {
\t\t\tInputs = ordered() { },
\t\t\tOutputs = { MainOutput1 = InstanceOutput { SourceOp = "Xf", Source = "Output", }, },
\t\t\tTools = ordered() {
\t\t\t\tBg = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 320, },
\t\t\t\t\t\tHeight = Input { Value = 180, },
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 1, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tXf = Transform {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tInput = Input { SourceOp = "Bg", Source = "Output", },
\t\t\t\t\t\tSize = Input { Value = 500, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
"""
    xf_path = tmp_path / "HugeXf.setting"
    xf_path.write_text(huge_xf, encoding="utf-8")
    with pytest.raises(FusionError) as exc_xf:
        render_fusion_frame(xf_path, resolution=(320, 180), time=0.0)
    assert exc_xf.value.code == "invalid_setting"


def test_render_fusion_clip_timeout_and_atomic_cleanup(tmp_path: Path) -> None:
    import stat
    import sys

    from aar.fusion.errors import FusionError

    setting = tmp_path / "DynamicBar.setting"
    setting.write_text(DYNAMIC_BAR_MACRO, encoding="utf-8")

    # Fake ffmpeg that sleeps until killed -> raises render_timeout and cleans up temp/output
    hang_bin = tmp_path / "fake_ffmpeg_hang.py"
    hang_bin.write_text(
        f"#!{sys.executable}\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    hang_bin.chmod(hang_bin.stat().st_mode | stat.S_IXUSR)

    out_dir = tmp_path / "out_timeout"
    out_dir.mkdir()
    out_mov = out_dir / "bar.mov"
    with pytest.raises(FusionError) as exc_to:
        render_fusion_clip(
            setting,
            out_mov,
            duration_s=0.1,
            fps=10.0,
            resolution=(160, 90),
            ffmpeg_bin=str(hang_bin),
            timeout_s=0.3,
        )
    assert exc_to.value.code == "render_timeout"
    assert not out_mov.exists()
    assert list(out_dir.iterdir()) == []

    # Fake ffmpeg that writes partial bytes then exits non-zero -> atomic cleanup removes temp file
    fail_bin = tmp_path / "fake_ffmpeg_fail.py"
    fail_bin.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "open(sys.argv[-1], 'wb').write(b'partial')\n"
        "sys.stderr.write('boom\\n')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    fail_bin.chmod(fail_bin.stat().st_mode | stat.S_IXUSR)

    out_dir2 = tmp_path / "out_fail"
    out_dir2.mkdir()
    out_mov2 = out_dir2 / "bar.mov"
    with pytest.raises(FusionError) as exc_fail:
        render_fusion_clip(
            setting,
            out_mov2,
            duration_s=0.1,
            fps=10.0,
            resolution=(160, 90),
            ffmpeg_bin=str(fail_bin),
            timeout_s=5.0,
        )
    assert exc_fail.value.code == "render_failed"
    assert not out_mov2.exists()
    assert list(out_dir2.iterdir()) == []
