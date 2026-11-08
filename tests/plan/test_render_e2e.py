"""S18b end to end on small synthetic plans: dissolves from the handles, transforms (rotation,
flip, pan past the picture), connected clips and titles, loudness for each kind (gain plus a
limiter for shorts, never loudnorm), mute, LUTs, the master presets, the CLI, and the safety
rules (atomic pair, timeouts, argument lists, the work dir)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from export_support import permissive
from render_support import (
    Spec,
    blend_of,
    fast,
    iter_frames,
    make_media,
    read_frame,
    read_pcm,
    sample_at,
    tone_dbfs,
)

from aar.cli import main
from aar.media import timeout_for
from aar.plan import render
from aar.plan.export.layout import transform_spec
from aar.plan.model import EditPlan, Segment
from aar.plan.profile import Profile, load_profile
from aar.plan.render import RenderError, render_plan
from aar.plan.render import runner as render_runner

pytestmark = pytest.mark.slow
SIZE = (320, 180)


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Spec]]:
    root = tmp_path_factory.mktemp("small_media")
    specs = {
        "a": Spec("red", 440, 200, centre=(110, 70), quiet=((22, 26),)),
        "b": Spec("green", 660, 200, centre=(200, 100), quiet=((22, 26),)),
        "voice": Spec("blue", 440, 240, tone_dbfs=-30.0, bursts=((60, 90, 8.0),)),
        # A short loud laugh: too brief to move the integrated loudness much, so after the gain
        # its peak goes past the ceiling and the limiter must hold it.
        "laugh": Spec("yellow", 440, 240, tone_dbfs=-30.0, bursts=((60, 62, 18.0),)),
    }
    return {name: (make_media(root / f"{name}.mov", spec), spec) for name, spec in specs.items()}


def plan_of(media: dict[str, tuple[Path, Spec]], segments: list[dict[str, Any]], **extra: Any):
    used = {seg["source"] for seg in segments}
    sources = [
        {"id": name, "media": str(path), "duration": spec.frames, "frame_size": list(SIZE)}
        for name, (path, spec) in media.items()
        if name in used
    ]
    body = {
        "project": "P",
        "timeline": "T",
        "kind": "longform",
        "fps": "24000/1001",
        "resolution": {"width": SIZE[0], "height": SIZE[1]},
        "profile": "test",
        "sources": sources,
        "segments": segments,
        **extra,
    }
    return EditPlan.model_validate(body)


def quick() -> Profile:
    return fast(permissive())


def frames(result: render.RenderResult, picks: list[int]) -> dict[int, np.ndarray]:
    return dict(iter_frames(result.out, SIZE, picks))


def test_a_dissolve_plays_both_handles_with_its_audio_crossfade(media, tmp_path: Path) -> None:
    plan = plan_of(
        media,
        [
            {"source": "a", "src_in": 24, "src_out": 72},
            {"source": "b", "src_in": 24, "src_out": 72},
        ],
        transitions=[
            {
                "after": 0,
                "video": "dissolve",
                "video_frames": 12,
                "audio": {"type": "Cross Fade 0 dB", "frames": 12},
            }
        ],
    )
    result = render_plan(plan, quick(), out=tmp_path / "d.mp4")
    # Centred (both sides have the media): window [42, 54), 6 frames each side of the cut at 48.
    assert result.report["rendered"]["dissolves"] == [
        {"location": "transitions[0]", "cut": 48, "window": [42, 54], "frames": 12}
    ]
    got = {n: read_frame(f) for n, f in frames(result, [41, 48, 54, 95]).items()}
    assert (got[41].colour, got[41].index) == ("red", 65)
    assert (got[54].colour, got[54].index) == ("green", 30)  # b's head handle came first
    assert (got[95].colour, got[95].index) == ("green", 71)
    share = blend_of(got[48].background, "red", "green")
    assert share is not None and 0.35 < share < 0.65
    pcm = read_pcm(result.out)
    level = tone_dbfs(pcm, sample_at(10), sample_at(40), 440)
    # Both sounds play over the window: a's tail handle and b's head handle, crossfaded.
    assert tone_dbfs(pcm, sample_at(44), sample_at(45), 440) == pytest.approx(
        level + 20 * math.log10(9.5 / 12), abs=1.0
    )
    assert tone_dbfs(pcm, sample_at(51), sample_at(52), 660) == pytest.approx(
        level + 20 * math.log10(9.5 / 12), abs=1.0
    )
    assert tone_dbfs(pcm, sample_at(56), sample_at(60), 440) < level - 40


def _forward(seg: Segment, u: float, v: float) -> tuple[float, float]:
    """S18's FCPXML transform, forward: source pixel -> output pixel (image coordinates)."""
    width, height = SIZE
    spec = transform_spec(seg, SIZE, width, height)
    px, py = u - width / 2, height / 2 - v
    a = math.radians(spec.rotation)
    qx = math.cos(a) * spec.scale[0] * px - math.sin(a) * spec.scale[1] * py
    qy = math.sin(a) * spec.scale[0] * px + math.cos(a) * spec.scale[1] * py
    qx += spec.position[0] * height / 100
    qy += spec.position[1] * height / 100
    return width / 2 + qx, height / 2 - qy


def _ring_centre(frame: np.ndarray, near: tuple[float, float]) -> tuple[float, float]:
    """The centroid of the black ring's pixels within 45 px of ``near`` (a window that leaves
    the picture shows black at the frame's edges, which must not count)."""
    x0, y0 = max(0, int(near[0]) - 45), max(0, int(near[1]) - 45)
    region = frame[0, y0 : int(near[1]) + 45, x0 : int(near[0]) + 45].astype(float)
    bg = float(np.median(region))
    ys, xs = np.nonzero(region < 16 + (bg - 16) / 4)
    assert len(xs) > 50, "no ring near the expected place"
    return x0 + float(xs.mean()) + 0.5, y0 + float(ys.mean()) + 0.5


@pytest.mark.parametrize(
    "transform",
    [
        {"zoom": 1.3, "rotation_deg": 9.0},
        {"zoom": 1.2, "rotation_deg": -6.0, "flip_h": True, "pan": 18.0, "tilt": -12.0},
        {"zoom": 1.1, "flip_v": True, "pan": -25.0},
    ],
)
def test_transforms_put_the_picture_where_s18_does(media, tmp_path: Path, transform) -> None:
    seg = {"source": "a", "src_in": 0, "src_out": 24, "transform": transform}
    plan = plan_of(media, [seg])
    result = render_plan(plan, quick(), out=tmp_path / "t.mp4")
    assert result.report["rendered"]["pictures"][0]["framing"] == "window"
    spec = media["a"][1]
    left, top, right, bottom = spec.block
    want = _forward(plan.segments[0], (left + right) / 2, (top + bottom) / 2)
    for n, frame in frames(result, [3, 20]).items():
        got = _ring_centre(frame, want)
        assert math.dist(got, want) < 1.0, (n, got, want)
        flips = {k: transform.get(k, False) for k in ("flip_h", "flip_v")}
        x0, y0 = max(0, int(want[0]) - 45), max(0, int(want[1]) - 45)
        near = frame[:, y0 : int(want[1]) + 45, x0 : int(want[0]) + 45]
        reading = read_frame(near, **flips)
        if not transform.get("rotation_deg"):
            assert reading.index == n and reading.colour == "red"


def test_a_window_past_the_picture_shows_black(media, tmp_path: Path) -> None:
    plan = plan_of(media, [{"source": "a", "src_in": 0, "src_out": 12, "transform": {"pan": 100}}])
    result = render_plan(plan, quick(), out=tmp_path / "pan.mp4")
    window = result.report["rendered"]["pictures"][0]["window"]
    assert window["pad"] > 0 and window["first"][0] == pytest.approx(60.0)
    frame = frames(result, [5])[5]
    assert frame[0, :, :95].max() < 24  # S18's window starts 100 px left of the picture
    assert abs(float(frame[0, 10:20, 110:120].mean()) - float(frame[0, 10:20, 300:310].mean())) < 3


def test_connected_clips_and_titles_on_their_frames(media, tmp_path: Path) -> None:
    plan = plan_of(
        media,
        [
            {"source": "a", "src_in": 0, "src_out": 96},
            {"track": 2, "source": "b", "src_in": 100, "src_out": 124, "record_in": 30},
        ],
        titles=[{"kind": "title", "lines": ["Hello"], "record_in": 60, "record_out": 80}],
        audio_tracks=[{"name": "A1"}, {"name": "A2", "gain_db": -6.0}],
    )
    result = render_plan(plan, quick(), out=tmp_path / "v2.mp4")
    got = frames(result, [29, 30, 53, 54, 59, 60, 79, 80])
    readings = {n: read_frame(f) for n, f in got.items()}
    assert (readings[29].colour, readings[29].index) == ("red", 29)
    assert (readings[30].colour, readings[30].index) == ("green", 100)
    assert (readings[53].colour, readings[53].index) == ("green", 123)
    assert (readings[54].colour, readings[54].index) == ("red", 54)
    band = {n: got[n][0, 70:110, 100:220].max() for n in (59, 60, 79, 80)}
    assert band[59] < 150 and band[80] < 150 and band[60] > 200 and band[79] > 200
    pcm = read_pcm(result.out)
    a = tone_dbfs(pcm, sample_at(32), sample_at(52), 440)
    b = tone_dbfs(pcm, sample_at(32), sample_at(52), 660)
    assert b - a == pytest.approx(-6.0, abs=0.5)  # V2's sound is on A2, at A2's gain
    assert tone_dbfs(pcm, sample_at(60), sample_at(90), 660) < a - 40


def _short(media, source: str) -> EditPlan:
    plan = plan_of(media, [{"source": source, "src_in": 0, "src_out": 240}])
    return plan.model_copy(update={"kind": "short"})


def test_a_short_gets_gain_and_a_limiter_and_keeps_its_range(media, tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    real = render_runner.run_tool

    def spy(tool: str, args: list[str], **kwargs: Any):
        calls.append(list(args))
        return real(tool, args, **kwargs)

    monkeypatch.setattr(render_runner, "run_tool", spy)
    result = render_plan(_short(media, "voice"), quick(), out=tmp_path / "s.mp4")
    loud = result.report["loudness"]
    assert loud["method"] == "gain-limiter"
    chains = " ".join(" ".join(c) for c in calls)
    assert "alimiter=" in chains and "loudnorm" not in chains
    qa = {c["id"]: c for c in result.report["qa"]["checks"]}
    assert qa["loudness"]["status"] == "pass"
    pcm = read_pcm(result.out)
    voice = tone_dbfs(pcm, sample_at(10), sample_at(55), 440)
    laugh = tone_dbfs(pcm, sample_at(62), sample_at(88), 440)
    assert laugh - voice == pytest.approx(8.0, abs=0.5)  # the laugh-to-voice range is kept


def test_a_loud_burst_is_held_under_the_true_peak_maximum(media, tmp_path: Path) -> None:
    result = render_plan(_short(media, "laugh"), quick(), out=tmp_path / "l.mp4")
    loud = result.report["loudness"]
    assert loud["method"] == "gain-limiter"
    # Without the limiter the burst would peak above full scale.
    assert -30.0 + 18.0 + loud["gain_db"] > 0.0
    pcm = read_pcm(result.out)
    burst = 20 * math.log10(float(np.abs(pcm[sample_at(60) : sample_at(62)]).max()))
    assert burst <= loud["process_true_peak_dbtp"] + 0.3
    # Limiting the laugh lowered the loudness; the gain was measured again and corrected.
    assert loud["passes"] >= 2 and abs(loud["output"]["integrated_lufs"] + 14.5) <= 0.15
    qa = {c["id"]: c for c in result.report["qa"]["checks"]}
    assert qa["loudness"]["status"] == "pass", qa["loudness"]["message"]
    assert qa["loudness"]["measured"]["true_peak_dbtp"] <= -1.0
    assert qa["clipping"]["status"] == "pass"
    assert qa["duration"]["status"] == "pass"


def test_a_long_form_gets_two_pass_loudnorm(media, tmp_path: Path) -> None:
    result = render_plan(
        _short(media, "voice").model_copy(update={"kind": "longform"}),
        quick(),
        out=tmp_path / "lf.mp4",
    )
    loud = result.report["loudness"]
    assert loud["method"] == "loudnorm-2pass" and loud["passes"] == 2
    qa = {c["id"]: c for c in result.report["qa"]["checks"]}
    assert qa["loudness"]["measured"]["integrated_lufs"] == pytest.approx(-22.0, abs=0.5)


def test_a_muted_track_with_no_bus_is_silent(media, tmp_path: Path) -> None:
    plan = plan_of(
        media,
        [{"source": "a", "src_in": 0, "src_out": 48}],
        audio_tracks=[{"name": "A1", "mute": True}],
    )
    result = render_plan(plan, quick(), out=tmp_path / "m.mp4")
    assert result.out.exists() and not result.qa_ok  # rendered; QA's loudness fails it
    assert result.report["loudness"]["method"] == "none"
    assert [n["feature"] for n in result.report["notes"]] == ["audio_mute"]
    assert float(np.abs(read_pcm(result.out)).max()) < 1e-4


def test_a_lut_file_is_applied(media, tmp_path: Path) -> None:
    lut = tmp_path / "swap.cube"
    corners = [(r, g, b) for b in (0, 1) for g in (0, 1) for r in (0, 1)]
    lut.write_text("LUT_3D_SIZE 2\n" + "".join(f"{b} {g} {r}\n" for r, g, b in corners))
    body = plan_of(media, [{"source": "a", "src_in": 0, "src_out": 12}]).model_dump()
    body["sources"][0]["color"] = {"lut": str(lut)}
    result = render_plan(EditPlan.model_validate(body), quick(), out=tmp_path / "lut.mp4")
    assert result.report["rendered"]["luts"] == {"a": str(lut)}
    assert "lut" not in {item["feature"] for item in result.report["not_rendered"]}
    assert read_frame(frames(result, [4])[4]).colour == "blue"  # red with R and B swapped


def _probe(path: Path) -> dict[str, Any]:
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def test_review_and_master_presets(media, tmp_path: Path) -> None:
    plan = plan_of(media, [{"source": "a", "src_in": 0, "src_out": 24}])
    review = render_plan(plan, quick(), out=tmp_path / "r.mp4")
    streams = _probe(review.out)
    assert streams["video"]["codec_name"] == "h264" and streams["video"]["profile"] == "High"
    assert streams["video"]["pix_fmt"] == "yuv420p" and streams["audio"]["codec_name"] == "aac"
    assert (streams["video"]["width"], streams["video"]["height"]) == SIZE
    assert streams["video"]["r_frame_rate"] == "24000/1001"
    master = render_plan(plan, quick(), out=tmp_path / "m.mp4", preset="master")
    assert "14" in master.report["encode"]["args"]
    profile = quick()
    prores = profile.model_copy(
        update={"render": profile.render.model_copy(update={"master_codec": "prores"})}
    )
    with pytest.raises(RenderError) as info:
        render_plan(plan, prores, out=tmp_path / "p.mp4", preset="master")
    assert info.value.code == "bad_output_path"
    mov = render_plan(plan, prores, out=tmp_path / "p.mov", preset="master")
    streams = _probe(mov.out)
    assert streams["video"]["codec_name"] == "prores"
    assert streams["video"]["pix_fmt"] == "yuv422p10le"
    assert streams["audio"]["codec_name"] == "pcm_s24le"


def test_the_moov_atom_comes_first(media, tmp_path: Path) -> None:
    plan = plan_of(media, [{"source": "a", "src_in": 0, "src_out": 24}])
    result = render_plan(plan, quick(), out=tmp_path / "fs.mp4")
    data = result.out.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")  # faststart


def test_every_ffmpeg_call_is_bounded_and_stays_in_the_work_dir(media, tmp_path, monkeypatch):
    calls: list[tuple[list[str], float, Path]] = []
    real = render_runner.run_tool

    def spy(tool: str, args: list[str], **kwargs: Any):
        calls.append((list(args), kwargs["timeout_s"], Path(kwargs["cwd"])))
        assert Path(kwargs["cwd"]).is_dir()
        return real(tool, args, **kwargs)

    monkeypatch.setattr(render_runner, "run_tool", spy)
    plan = plan_of(media, [{"source": "a", "src_in": 0, "src_out": 48}])
    profile = quick()
    out = tmp_path / "out" / "b.mp4"
    out.parent.mkdir()
    render_plan(plan, profile, out=out)
    duration = 48 * 1001 / 24000
    want = timeout_for(
        duration,
        base_s=profile.render.timeout_base_s,
        per_media_s=profile.render.timeout_per_media_s,
    )
    assert len(calls) >= 4
    work = calls[0][2]
    assert work.parent == out.parent and work.name.startswith(".b.mp4.aar-render-")
    assert not work.exists()  # removed afterwards
    media_url = f"file:{media['a'][0]}"
    for args, timeout, cwd in calls:
        assert timeout == pytest.approx(want) and cwd == work
        assert "-filter_complex" not in args  # graphs go in -filter_complex_script files
        for arg in args:
            if arg.startswith(("/", "file:")):
                assert arg == media_url or arg.startswith(f"file:{work}/"), arg


def test_a_failed_render_leaves_nothing_and_keeps_the_previous_pair(media, tmp_path, monkeypatch):
    plan = plan_of(media, [{"source": "a", "src_in": 0, "src_out": 24}])
    out = tmp_path / "x.mp4"
    first = render_plan(plan, quick(), out=out)
    before = (out.read_bytes(), first.report_path.read_bytes())
    real = render.run_ffmpeg

    def failing(args: list[str], stage: str, timeout_s: float, cwd: Path) -> str:
        if stage == "encode":
            raise RenderError("render_failed", "boom", "none", stage=stage)
        return real(args, stage, timeout_s, cwd)

    monkeypatch.setattr(render, "run_ffmpeg", failing)
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=out, overwrite=True)
    assert info.value.code == "render_failed"
    assert (out.read_bytes(), first.report_path.read_bytes()) == before
    fresh = tmp_path / "y.mp4"
    with pytest.raises(RenderError):
        render_plan(plan, quick(), out=fresh)
    assert not fresh.exists() and not render.render_report_path(fresh).exists()
    assert not [p for p in tmp_path.iterdir() if ".aar-render-" in p.name or ".tmp" in p.name]
    monkeypatch.setattr(render, "run_ffmpeg", real)
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=out)  # exists, no --overwrite: refused before rendering
    assert info.value.code == "output_exists"


def test_lint_errors_and_missing_media_are_refused_before_rendering(media, tmp_path) -> None:
    plan = plan_of(
        media,
        [{"source": "a", "src_in": 0, "src_out": 24}, {"source": "b", "src_in": 0, "src_out": 24}],
        transitions=[{"after": 0, "video": "dissolve", "video_frames": 12}],
    )
    with pytest.raises(RenderError) as info:
        render_plan(plan, fast(load_profile("ccn")), out=tmp_path / "x.mp4")
    assert info.value.code == "lint_errors"  # ccn: no dissolves between content
    missing = plan_of(media, [{"source": "a", "src_in": 0, "src_out": 24}])
    gone = missing.model_copy(
        update={"sources": [missing.sources[0].model_copy(update={"media": "/no/such.mov"})]}
    )
    with pytest.raises(RenderError) as info:
        render_plan(gone, quick(), out=tmp_path / "y.mp4")
    assert info.value.code == "media_not_on_hub"
    assert not list(tmp_path.iterdir())


# -- CLI ------------------------------------------------------------------------------------------


def _cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(["plan", "render", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _write(tmp_path: Path, plan: EditPlan, name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    return path


def _profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "test.toml"
    path.write_text(
        'name = "test"\n[qa]\nlongform_lufs = -22.0\nshort_lufs = -14.5\nlufs_tolerance = 0.5\n'
        'review_mp4_max_mb = 20\n[render]\nreview_x264_preset = "veryfast"\n'
        "[rules.ranges-valid]\n",
        encoding="utf-8",
    )
    return path


def test_cli_renders_through_the_media_map(media, tmp_path: Path, capsys) -> None:
    plan = plan_of(media, [{"source": "a", "src_in": 30, "src_out": 54}])
    hub = media["a"][0].parent
    edit = plan.model_copy(
        update={"sources": [plan.sources[0].model_copy(update={"media": "E:\\Show\\a.mov"})]}
    )
    plan_file = _write(tmp_path, edit)
    map_file = tmp_path / "map.json"
    map_file.write_text(json.dumps({str(hub): "E:\\Show"}), encoding="utf-8")
    out = tmp_path / "cli.mp4"
    profile = _profile_file(tmp_path)
    code, stdout, _ = _cli(
        capsys,
        str(plan_file),
        "--profile",
        str(profile),
        "--out",
        str(out),
        "--media-map",
        str(map_file),
        "--json",
    )
    assert code == 0, stdout
    report = json.loads(stdout)
    assert report["media"][0]["rule"] == "media-map" and report["qa"]["ok"]
    assert out.exists() and (tmp_path / "cli.render.json").exists()
    code, _, err = _cli(capsys, str(plan_file), "--profile", str(profile), "--out", str(out))
    assert code == 2 and "output_exists" in err
    code, _, err = _cli(
        capsys, str(plan_file), "--profile", str(profile), "--out", str(tmp_path / "n.mp4")
    )
    assert code == 2 and "media_not_on_hub" in err  # no --media-map: the edit-PC path is refused


def test_cli_exit_codes_for_qa_failure_and_ffmpeg_failure(media, tmp_path, capsys, monkeypatch):
    # a then b: different recordings, and b's in point (frame 60) is not in a pause: QA fails.
    plan = plan_of(
        media,
        [{"source": "a", "src_in": 0, "src_out": 30}, {"source": "b", "src_in": 60, "src_out": 90}],
    )
    plan_file = _write(tmp_path, plan)
    profile = _profile_file(tmp_path)
    code, stdout, _ = _cli(
        capsys, str(plan_file), "--profile", str(profile), "--out", str(tmp_path / "q.mp4")
    )
    assert code == 1 and "join-quiet: fail" in stdout
    assert (tmp_path / "q.mp4").exists()  # rendered; the report says why QA failed

    def failing(args: list[str], stage: str, timeout_s: float, cwd: Path) -> str:
        raise RenderError("render_failed", "boom", "none", stage=stage, stderr_tail="x\nlast line")

    monkeypatch.setattr(render, "run_ffmpeg", failing)
    code, _, err = _cli(
        capsys, str(plan_file), "--profile", str(profile), "--out", str(tmp_path / "f.mp4")
    )
    assert code == 3 and "render_failed" in err
