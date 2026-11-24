"""Tests for Fusion graphics integration in ``aar.plan``: model v2, export, alpha graph, render."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aar.cli import main
from aar.fusion.render import render_fusion_clip
from aar.media import resolve_tool
from aar.plan.apply.inputs import plan_hash
from aar.plan.export import build_fusion_manifest, export_fcpxml, layout_for
from aar.plan.model import EditPlan, Resolution
from aar.plan.profile import load_profile
from aar.plan.render import RenderError, render_plan
from aar.plan.render.framing import Framing, picture_filters
from aar.plan.render.graph import video_graph
from aar.plan.render.media import HubSource, pix_fmt_has_alpha
from aar.plan.render.timeline import build_timeline

PIL = pytest.importorskip("PIL.Image")

GREEN_BOX_MACRO = """{
\tTools = ordered() {
\t\tGreenBadge = MacroOperator {
\t\t\tInputs = ordered() {
\t\t\t\tInput1 = InstanceInput {
\t\t\t\t\tSourceOp = "LabelText",
\t\t\t\t\tSource = "StyledText",
\t\t\t\t\tName = "Label",
\t\t\t\t},
\t\t\t},
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
\t\t\t\tBoxMask = RectangleMask {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 0.5, },
\t\t\t\t\t\tHeight = Input { Value = 0.25, },
\t\t\t\t\t\tCenter = Input { Value = { 0.5, 0.25 }, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tBoxBg = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 320, },
\t\t\t\t\t\tHeight = Input { Value = 180, },
\t\t\t\t\t\tTopLeftRed = Input { Value = 0.05, },
\t\t\t\t\t\tTopLeftGreen = Input { Value = 0.9, },
\t\t\t\t\t\tTopLeftBlue = Input { Value = 0.1, },
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 1, },
\t\t\t\t\t\tEffectMask = Input { SourceOp = "BoxMask", Source = "Mask", },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tLabelText = TextPlus {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 320, },
\t\t\t\t\t\tHeight = Input { Value = 180, },
\t\t\t\t\t\tStyledText = Input { Value = "HELLO", },
\t\t\t\t\t\tFont = Input { Value = "DejaVu Sans", },
\t\t\t\t\t\tSize = Input { Value = 0.08, },
\t\t\t\t\t\tCenter = Input { Value = { 0.5, 0.25 }, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tBarMerge = Merge {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tBackground = Input { SourceOp = "Canvas", Source = "Output", },
\t\t\t\t\t\tForeground = Input { SourceOp = "BoxBg", Source = "Output", },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tOutMerge = Merge {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tBackground = Input { SourceOp = "BarMerge", Source = "Output", },
\t\t\t\t\t\tForeground = Input { SourceOp = "LabelText", Source = "Output", },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
"""


def _base_plan_dict(
    media_path: str,
    *,
    titles: list[dict[str, Any]] | None = None,
    extra_sources: list[dict[str, Any]] | None = None,
    extra_segments: list[dict[str, Any]] | None = None,
    version: int = 2,
) -> dict[str, Any]:
    sources = [
        {
            "id": "cam-a",
            "media": media_path,
            "camera": "A",
            "duration": 72,
            "frame_size": [320, 180],
            "start_timecode": 0,
            "start_timecode_rate": {"num": 24, "den": 1},
        },
        *(extra_sources or []),
    ]
    segments = [
        {
            "source": "cam-a",
            "src_in": 0,
            "src_out": 48,
            "track": 1,
        },
        *(extra_segments or []),
    ]
    return {
        "version": version,
        "project": "TestShow",
        "timeline": "Fusion Test",
        "kind": "longform",
        "fps": {"num": 24, "den": 1},
        "resolution": {"width": 320, "height": 180},
        "profile": "ccn",
        "sources": sources,
        "segments": segments,
        "titles": titles or [],
    }


def _make_solid_clip(path: Path, color: str = "blue", seconds: float = 3.0) -> Path:
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
            f"color=c={color}:s=320x180:r=24:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={seconds}",
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


def _extract_frame(video: Path, frame_index: int, out_png: Path) -> Any:
    ffmpeg = str(resolve_tool("ffmpeg"))
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vframes",
            "1",
            str(out_png),
        ],
        check=True,
    )
    return PIL.open(out_png).convert("RGB")


def test_editplan_v2_fusion_fields_and_v1_rejection() -> None:
    data = _base_plan_dict(
        "/media/cam.mp4",
        titles=[
            {
                "kind": "graphic",
                "record_in": 0,
                "record_out": 24,
                "setting": "GreenBadge.setting",
                "inputs": {"Label": "Host", "Scale": 1.0, "Pos": [0.5, 0.25]},
            }
        ],
    )
    plan = EditPlan.model_validate(data)
    assert plan.titles[0].kind == "graphic"
    assert plan.titles[0].lines == []
    assert plan.titles[0].setting == "GreenBadge.setting"

    # v1 rejects Fusion fields
    data_v1 = {**data, "version": 1}
    with pytest.raises(ValueError, match="version 2"):
        EditPlan.model_validate(data_v1)

    # Without lines, setting, or inputs -> invalid
    bad = _base_plan_dict(
        "/media/cam.mp4",
        titles=[{"kind": "lower-third", "record_in": 0, "record_out": 24, "lines": []}],
    )
    with pytest.raises(ValueError, match="line"):
        EditPlan.model_validate(bad)


def test_plan_hash_invariant_for_plain_titles() -> None:
    plain_a = EditPlan.model_validate(
        _base_plan_dict(
            "/media/cam.mp4",
            titles=[
                {
                    "kind": "lower-third",
                    "record_in": 0,
                    "record_out": 24,
                    "lines": ["Alice"],
                }
            ],
        )
    )
    plain_b = EditPlan.model_validate(
        _base_plan_dict(
            "/media/cam.mp4",
            titles=[
                {
                    "kind": "lower-third",
                    "record_in": 0,
                    "record_out": 24,
                    "lines": ["Alice"],
                    "setting": None,
                    "inputs": {},
                }
            ],
        )
    )
    assert plan_hash(plain_a) == plan_hash(plain_b)


def test_export_fcpxml_and_cli_write_fusion_manifest(tmp_path: Path) -> None:
    setting = tmp_path / "GreenBadge.setting"
    setting.write_text(GREEN_BOX_MACRO, encoding="utf-8")
    plan_data = _base_plan_dict(
        "X:/show/cam.mp4",
        titles=[
            {
                "kind": "graphic",
                "record_in": 4,
                "record_out": 28,
                "setting": str(setting),
                "inputs": {"Label": "Guest"},
            }
        ],
    )
    plan = EditPlan.model_validate(plan_data)
    profile = load_profile("ccn")
    res = export_fcpxml(plan, profile, allow_unknown_timecode=True)
    assert "fusion" in res.report
    assert len(res.report["fusion"]["placements"]) == 1
    p0 = res.report["fusion"]["placements"][0]
    assert p0["kind"] == "graphic"
    assert p0["inputs"] == {"Label": "Guest"}

    # Customized .setting emission via build_fusion_manifest
    custom_dir = tmp_path / "custom"
    manifest = build_fusion_manifest(res.layout, customized_dir=custom_dir)
    assert manifest.placements[0].customized_setting is not None
    assert (custom_dir / manifest.placements[0].customized_setting).is_file()

    # CLI export writes <stem>.fusion.json beside <stem>.fcpxml
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan_data), encoding="utf-8")
    fcpxml_out = tmp_path / "show.fcpxml"
    assert (
        main(
            [
                "plan",
                "export",
                str(plan_file),
                "--out",
                str(fcpxml_out),
                "--allow-unknown-timecode",
            ]
        )
        == 0
    )
    fusion_json = tmp_path / "show.fusion.json"
    assert fusion_json.is_file()
    loaded = json.loads(fusion_json.read_text(encoding="utf-8"))
    assert loaded["fcpxml"] == "show.fcpxml"
    assert len(loaded["placements"]) == 1


def test_alpha_pix_fmt_and_picture_filters_preserve_alpha() -> None:
    assert pix_fmt_has_alpha("argb") is True
    assert pix_fmt_has_alpha("rgba") is True
    assert pix_fmt_has_alpha("yuva444p10le") is True
    assert pix_fmt_has_alpha("yuv420p") is False
    assert pix_fmt_has_alpha(None) is False

    res = Resolution(width=320, height=180)
    fit_framing = Framing("fit", (200, 180), fitted=(200, 180))
    opaque = picture_filters(fit_framing, res)
    alpha = picture_filters(fit_framing, res, preserve_alpha=True)
    assert any("yuv444p" in f for f in opaque)
    assert any("yuva444p" in f for f in alpha)
    assert any("black@0" in f for f in alpha)

    # Connected alpha overlay keeps yuva420p in video_graph, whereas opaque stays untouched
    plan = EditPlan.model_validate(
        _base_plan_dict(
            "X:/show/cam.mp4",
            extra_sources=[
                {
                    "id": "gfx",
                    "media": "X:/show/gfx.mov",
                    "duration": 48,
                    "frame_size": [320, 180],
                    "start_timecode": 0,
                    "start_timecode_rate": {"num": 24, "den": 1},
                }
            ],
            extra_segments=[
                {
                    "source": "gfx",
                    "src_in": 0,
                    "src_out": 24,
                    "track": 2,
                    "record_in": 0,
                }
            ],
        )
    )
    profile = load_profile("ccn")
    layout = layout_for(plan, profile, allow_unknown_timecode=True)
    from fractions import Fraction

    from aar.media import MediaInfo

    def _hub(source_id: str, path: str, has_alpha: bool) -> HubSource:
        p = Path(path)
        return HubSource(
            source_id=source_id,
            plan_media=f"X:/show/{p.name}",
            path=p,
            rule="plan-path",
            info=MediaInfo(
                path=str(p), format="mov", duration_s=3.0, size_bytes=1024, bit_rate=None
            ),
            display=(320, 180),
            sample_aspect=None,
            fps=Fraction(24, 1),
            frames=72,
            channels=2 if not has_alpha else None,
            audio_index=1 if not has_alpha else None,
            video_time_base=Fraction(1, 24000),
            has_alpha=has_alpha,
        )

    hub_alpha = {
        "cam-a": _hub("cam-a", "/hub/cam.mp4", False),
        "gfx": _hub("gfx", "/hub/gfx.mov", True),
    }
    tl = build_timeline(layout, hub_alpha)
    g_alpha = video_graph(tl, hub_alpha, font=None, luts={}, out_pix_fmt="yuv420p")
    assert "setsar=1,format=yuva420p[p1]" in g_alpha.text
    assert "format=yuv420p,format=yuva420p[p1]" not in g_alpha.text

    hub_opaque = {
        "cam-a": _hub("cam-a", "/hub/cam.mp4", False),
        "gfx": _hub("gfx", "/hub/gfx.mov", False),
    }
    g_opaque = video_graph(tl, hub_opaque, font=None, luts={}, out_pix_fmt="yuv420p")
    assert "format=yuv420p,format=yuva420p[p1]" in g_opaque.text


def test_render_plan_alpha_v2_mov_and_native_fusion_setting(tmp_path: Path) -> None:
    blue_mp4 = _make_solid_clip(tmp_path / "blue.mp4", color="blue", seconds=3.0)
    setting = tmp_path / "GreenBadge.setting"
    setting.write_text(GREEN_BOX_MACRO, encoding="utf-8")

    # 1. Pre-rendered alpha .mov on V2 composited over blue V1 must preserve blue background
    alpha_mov = tmp_path / "badge.mov"
    render_fusion_clip(
        setting,
        alpha_mov,
        duration_s=2.0,
        fps=24.0,
        resolution=(320, 180),
        inputs={"Label": "V2"},
        codec="qtrle",
    )
    v2_plan = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            extra_sources=[
                {
                    "id": "badge",
                    "media": str(alpha_mov),
                    "duration": 48,
                    "frame_size": [320, 180],
                    "start_timecode": 0,
                    "start_timecode_rate": {"num": 24, "den": 1},
                }
            ],
            extra_segments=[
                {
                    "source": "badge",
                    "src_in": 0,
                    "src_out": 24,
                    "track": 2,
                    "record_in": 0,
                }
            ],
        )
    )
    profile = load_profile("ccn")
    v2_out = tmp_path / "v2_alpha.mp4"
    render_plan(v2_plan, profile, out=v2_out)
    frame_v2 = _extract_frame(v2_out, 10, tmp_path / "v2_f10.png")
    # Top-left corner (outside green badge) MUST remain blue, not black!
    r, g, b = frame_v2.getpixel((10, 10))
    assert b > 180 and r < 40 and g < 40, f"expected blue background, got {(r, g, b)}"
    # Inside green badge (x=160, y=135 -> Fusion y=0.25 from bottom -> PIL y=0.75*180=135)
    r2, g2, b2 = frame_v2.getpixel((120, 135))
    assert g2 > 160 and b2 < 80, f"expected green badge overlay, got {(r2, g2, b2)}"

    # 2. Native Fusion Title.setting rendered directly by render_plan
    fusion_plan = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            titles=[
                {
                    "kind": "graphic",
                    "record_in": 4,
                    "record_out": 36,
                    "setting": str(setting),
                    "inputs": {"Label": "LIVE"},
                }
            ],
        )
    )
    fusion_out = tmp_path / "native_fusion.mp4"
    res = render_plan(fusion_plan, profile, out=fusion_out)
    assert res.report["rendered"]["titles"][0]["fusion_template"] == "GreenBadge.setting"
    assert not any(item["feature"] == "title_template" for item in res.report["not_rendered"])

    # Before title (frame 1): solid blue everywhere
    f_before = _extract_frame(fusion_out, 1, tmp_path / "nf_f1.png")
    assert f_before.getpixel((120, 135))[2] > 180
    # During title (frame 15): blue corner, green badge inside box
    f_during = _extract_frame(fusion_out, 15, tmp_path / "nf_f15.png")
    r_bg, g_bg, b_bg = f_during.getpixel((10, 10))
    assert b_bg > 180 and r_bg < 40 and g_bg < 40
    _r_fg, g_fg, b_fg = f_during.getpixel((120, 135))
    assert g_fg > 160 and b_fg < 80


def test_render_plan_refuses_missing_fusion_setting(tmp_path: Path) -> None:
    blue_mp4 = _make_solid_clip(tmp_path / "blue.mp4", color="blue", seconds=3.0)
    plan = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            titles=[
                {
                    "kind": "graphic",
                    "record_in": 0,
                    "record_out": 24,
                    "setting": "DoesNotExist.setting",
                }
            ],
        )
    )
    with pytest.raises(RenderError) as exc:
        render_plan(plan, load_profile("ccn"), out=tmp_path / "missing.mp4")
    assert exc.value.code == "fusion_template_missing"


def test_v1_plan_rejects_graphic_title_kind_without_setting_or_inputs() -> None:
    from pydantic import ValidationError

    raw = _base_plan_dict(
        "X:/show/a.mp4",
        titles=[
            {
                "kind": "graphic",
                "lines": ["Just text"],
                "record_in": 0,
                "record_out": 24,
            }
        ],
    )
    raw["version"] = 1
    with pytest.raises(ValidationError, match="graphic"):
        EditPlan.model_validate(raw)


def test_plain_title_template_not_hijacked_by_matching_setting(tmp_path: Path) -> None:
    """A plain title with ``template`` and no ``setting`` must NOT switch to Fusion rendering."""
    blue_mp4 = _make_solid_clip(tmp_path / "blue.mp4", color="blue", seconds=2.0)
    # Place a .setting with an unsupported node whose stem matches title.template!
    # If plain titles still resolved by template stem, this would fail with fusion_render_failed.
    unsupported = """{
\tTools = ordered() {
\t\tPlainName = MacroOperator {
\t\t\tInputs = ordered() { },
\t\t\tOutputs = { MainOutput1 = InstanceOutput { SourceOp = "Blur1", Source = "Output", }, },
\t\t\tTools = ordered() {
\t\t\t\tBlur1 = Blur { Inputs = { }, },
\t\t\t},
\t\t},
\t},
}
"""
    (tmp_path / "PlainName.setting").write_text(unsupported, encoding="utf-8")

    plan = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            titles=[
                {
                    "kind": "lower-third",
                    "template": "PlainName",
                    "lines": ["Hello World"],
                    "record_in": 0,
                    "record_out": 24,
                }
            ],
        )
    )
    out = tmp_path / "plain_title.mp4"
    res = render_plan(
        plan,
        load_profile("ccn"),
        out=out,
        media_root=tmp_path,
        fusion_dirs=[tmp_path],
    )
    assert out.is_file()
    assert "fusion_template" not in res.report["rendered"]["titles"][0]


def test_render_plan_confines_fusion_templates_to_allowlist(tmp_path: Path) -> None:
    blue_mp4 = _make_solid_clip(tmp_path / "blue.mp4", color="blue", seconds=2.0)
    tpl_allowed = tmp_path / "Allowed.setting"
    tpl_sibling = tmp_path / "Sibling.setting"
    tpl_allowed.write_text(GREEN_BOX_MACRO, encoding="utf-8")
    tpl_sibling.write_text(GREEN_BOX_MACRO, encoding="utf-8")

    plan = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            titles=[
                {
                    "kind": "graphic",
                    "record_in": 0,
                    "record_out": 24,
                    "setting": "Sibling.setting",
                }
            ],
        )
    )
    with pytest.raises(RenderError) as exc:
        render_plan(
            plan,
            load_profile("ccn"),
            out=tmp_path / "blocked.mp4",
            media_root=tmp_path,
            fusion_templates=[tpl_allowed],
        )
    assert exc.value.code == "fusion_template_missing"


def test_video_graph_composites_fusion_and_plain_titles_in_lane_order(tmp_path: Path) -> None:
    from aar.plan.render import build_timeline, render_layout
    from aar.plan.render.graph import video_graph
    from aar.plan.render.media import resolve_source

    blue_mp4 = _make_solid_clip(tmp_path / "blue.mp4", color="blue", seconds=2.0)
    profile = load_profile("ccn")
    # Overlapping plain title (title 0 -> lane 1) and Fusion graphic (title 1 -> lane 2)
    plan_plain_below = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            titles=[
                {
                    "kind": "lower-third",
                    "lines": ["Below"],
                    "record_in": 0,
                    "record_out": 24,
                },
                {
                    "kind": "graphic",
                    "setting": "GreenBadge.setting",
                    "record_in": 0,
                    "record_out": 24,
                },
            ],
        )
    )
    hub = {
        s.id: resolve_source(s, plan_plain_below.fps, media_map=None, media_root=None)
        for s in plan_plain_below.sources
    }
    tl = build_timeline(render_layout(plan_plain_below, hub, profile), hub)
    assert tl.titles[0].lane < tl.titles[1].lane
    fake_mov = tmp_path / "f.mov"
    g1 = video_graph(
        tl,
        hub,
        span=(0, 24),
        font="font.ttf",
        luts={},
        out_pix_fmt="yuv420p",
        fusion_clips={tl.titles[1].location: fake_mov},
    )
    assert g1.text.index("drawtext=") < g1.text.index("[flane1]overlay=")

    # Reverse: Fusion graphic on lower lane, plain title on higher lane
    plan_fusion_below = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            titles=[
                {
                    "kind": "graphic",
                    "setting": "GreenBadge.setting",
                    "record_in": 0,
                    "record_out": 24,
                },
                {
                    "kind": "lower-third",
                    "lines": ["Above"],
                    "record_in": 0,
                    "record_out": 24,
                },
            ],
        )
    )
    tl2 = build_timeline(render_layout(plan_fusion_below, hub, profile), hub)
    assert tl2.titles[0].lane < tl2.titles[1].lane
    g2 = video_graph(
        tl2,
        hub,
        span=(0, 24),
        font="font.ttf",
        luts={},
        out_pix_fmt="yuv420p",
        fusion_clips={tl2.titles[0].location: fake_mov},
    )
    assert g2.text.index("[flane0]overlay=") < g2.text.index("drawtext=")


def test_render_plan_wrong_typed_input_raises_render_error(tmp_path: Path) -> None:
    """Wrong-typed numeric plan inputs (e.g. Size='big') must raise RenderError
    ('fusion_render_failed'), never escape as a raw ValueError.
    """
    blue_mp4 = _make_solid_clip(tmp_path / "blue.mp4", color="blue", seconds=2.0)
    tpl_dir = tmp_path / "tpl"
    tpl_dir.mkdir()
    setting = tpl_dir / "TextTitle.setting"
    setting.write_text(
        """{
\tTools = ordered() {
\t\tTextTitle = MacroOperator {
\t\t\tInputs = ordered() {
\t\t\t\tSize = InstanceInput { SourceOp = "Title", Source = "Size", },
\t\t\t},
\t\t\tOutputs = { MainOutput1 = InstanceOutput { SourceOp = "Title", Source = "Output", }, },
\t\t\tTools = ordered() {
\t\t\t\tTitle = TextPlus {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tStyledText = Input { Value = "Hello", },
\t\t\t\t\t\tSize = Input { Value = 0.08, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
""",
        encoding="utf-8",
    )
    plan = EditPlan.model_validate(
        _base_plan_dict(
            str(blue_mp4),
            titles=[
                {
                    "kind": "graphic",
                    "setting": "TextTitle.setting",
                    "inputs": {"Size": "big"},
                    "record_in": 0,
                    "record_out": 24,
                }
            ],
        )
    )
    with pytest.raises(RenderError) as exc:
        render_plan(
            plan,
            load_profile("ccn"),
            out=tmp_path / "out.mp4",
            fusion_templates=[setting],
        )
    assert exc.value.code == "fusion_render_failed"
