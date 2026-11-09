"""S18b's small helper extractions, pinned: S18's pair writer places a finished file, the export
layout takes a media resolver, ``run_tool`` runs in a given directory, and the ``[render]``
profile block. Each default keeps S18's and S13b's behaviour (their own suites still pass)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from export_support import EXPORT_FIXTURES, load_plan, permissive
from pydantic import ValidationError

from aar.media import MediaError, run_tool
from aar.plan.export.errors import ExportError
from aar.plan.export.files import write_pair
from aar.plan.export.layout import build_layout
from aar.plan.export.media import MediaRef
from aar.plan.profile import RenderSettings, load_profile, parse_profile


def test_write_pair_moves_a_finished_file_into_place(tmp_path: Path) -> None:
    work = tmp_path / ".work"
    work.mkdir()
    made = work / "render.mp4"
    made.write_bytes(b"movie")
    inode = made.stat().st_ino
    out, report = tmp_path / "x.mp4", tmp_path / "x.render.json"
    write_pair([(out, made), (report, b"{}")], overwrite=False)
    assert out.read_bytes() == b"movie" and out.stat().st_ino == inode  # moved, not copied
    assert not made.exists() and report.read_bytes() == b"{}"


def test_write_pair_with_a_file_refuses_an_existing_output_and_touches_nothing(
    tmp_path: Path,
) -> None:
    made = tmp_path / ".r.mp4"
    made.write_bytes(b"new")
    out = tmp_path / "x.mp4"
    out.write_bytes(b"old")
    with pytest.raises(ExportError) as info:
        write_pair([(out, made), (tmp_path / "x.render.json", b"{}")], overwrite=False)
    assert info.value.code == "output_exists"
    assert out.read_bytes() == b"old" and not (tmp_path / "x.render.json").exists()


def test_write_pair_with_a_file_keeps_the_previous_output_when_the_report_fails(
    tmp_path: Path,
) -> None:
    made = tmp_path / ".r.mp4"
    made.write_bytes(b"new")
    out = tmp_path / "x.mp4"
    out.write_bytes(b"old")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(blocked, 0o500)  # the report cannot be written there
    try:
        with pytest.raises(OSError):
            write_pair([(out, made), (blocked / "x.render.json", b"{}")], overwrite=True)
    finally:
        os.chmod(blocked, 0o700)
    assert out.read_bytes() == b"old"
    assert not made.exists() or made.read_bytes() == b"new"


def test_build_layout_takes_a_media_resolver() -> None:
    plan = load_plan(EXPORT_FIXTURES / "ccn_longform_v2.plan.json")
    seen: list[str] = []

    def resolve(source):  # type: ignore[no-untyped-def]
        seen.append(source.id)
        return MediaRef(source.id, source.media, f"/hub/{source.id}", "hub-path", "", source.id)

    layout = build_layout(plan, allow_unknown_timecode=True, resolve_media=resolve)
    assert seen == ["camA", "camB", "camA2", "outroclip"]
    assert {c.src for c in layout.clips} == {f"/hub/{s}" for s in seen}
    # Without it, the FCPXML media rules hold, as before.
    default = build_layout(plan, allow_unknown_timecode=True)
    assert {c.src for c in default.clips} == {
        "file:///E:/CCN/Ep1/A001.braw",
        "file:///E:/CCN/Ep1/B001.braw",
        "file:///E:/CCN/Ep1/A002.mov",
        "file:///E:/CCN/Assets/Outro.mov",
    }


def test_run_tool_runs_in_the_given_directory(tmp_path: Path) -> None:
    (tmp_path / "g.txt").write_text("anullsrc=r=8000:cl=mono,atrim=end_sample=80[out]\n")
    args = ["-v", "error", "-filter_complex_script", "g.txt", "-map", "[out]", "-f", "null", "-"]
    assert run_tool("ffmpeg", args, timeout_s=30, cwd=tmp_path).returncode == 0
    with pytest.raises(MediaError):
        run_tool("ffmpeg", args, timeout_s=30)  # not found relative to this process's cwd


def test_render_settings_defaults_and_the_ccn_block() -> None:
    assert RenderSettings().font == "Montserrat" and RenderSettings().master_codec == "h264"
    ccn = load_profile("ccn").render
    assert ccn.limiter_warn_db == 3.0 and ccn.true_peak_margin_db == 0.5
    assert ccn.gain_tolerance_lu == 0.1 and ccn.decode_memory_mb == 16384
    assert ccn.review_x264_preset == "medium" and ccn.master_crf == 14
    assert permissive().render == RenderSettings()  # a profile without [render] still loads
    bad_values = (
        {"master_codec": "dnxhd"},
        {"true_peak_margin_db": float("nan")},
        {"crf": 1},
        {"fps_tolerance": 0.001},  # gone in round 2: a source's rate must be the plan's
        {"longform_lra": 20.0},  # gone in round 2: no loudnorm
    )
    for bad in bad_values:
        with pytest.raises((ValidationError, ValueError)):
            RenderSettings.model_validate(bad)
    with pytest.raises(ValueError):
        parse_profile(
            {
                "name": "x",
                "qa": {
                    "longform_lufs": -22,
                    "short_lufs": -14.5,
                    "lufs_tolerance": 0.5,
                    "review_mp4_max_mb": 20,
                },
                "rules": {},
                "render": {"review_x264_preset": "warp"},
            }
        )


def test_the_title_font_is_the_profiles_family_else_dejavu(tmp_path, monkeypatch) -> None:
    from aar.plan.render import fonts

    fallback = fonts.find_font("No Such Family")
    assert fallback.fallback and fallback.used in (fonts.FALLBACK_FAMILY, None)
    if fallback.path is not None:
        assert fallback.path.name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")
        for name in ("Montserrat[wght].ttf", "Montserrat-Regular.ttf", "Montserrat-SemiBold.ttf"):
            (tmp_path / name).write_bytes(fallback.path.read_bytes())
    else:
        pytest.skip("no DejaVu font on this machine")
    monkeypatch.setenv(fonts.FONTS_DIR_ENV, str(tmp_path))
    chosen = fonts.find_font("Montserrat")
    assert not chosen.fallback and chosen.path == tmp_path / "Montserrat-SemiBold.ttf"
