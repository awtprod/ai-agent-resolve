"""S18b round 2: the validation's findings, each pinned on a small synthetic render.

One input per use (no sound lost when a file is reused out of order, every sound placed to the
sample, at any rate), stream start offsets kept, the right channel kept, connected clips stacked
by track, left-aligned dissolves, loudness by gain plus a true-peak limiter for both kinds (laugh
range kept, a limiter that works hard is a warning), memory and size refused up front, an
incomplete encode never placed, and SIGTERM handled like Ctrl-C."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from fractions import Fraction
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

from aar.plan import render
from aar.plan.model import EditPlan
from aar.plan.profile import Profile
from aar.plan.render import RenderError, render_plan

pytestmark = pytest.mark.slow
SIZE = (320, 180)
F25 = Fraction(25)


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Spec]]:
    root = tmp_path_factory.mktemp("r2_media")
    specs = {
        "A": Spec("red", 440, 400, rate=F25, centre=(100, 60)),
        "B": Spec("green", 660, 400, rate=F25, centre=(220, 120)),
        "C": Spec("blue", 880, 400, rate=F25, centre=(160, 90)),
        "stereo": Spec("yellow", 440, 200, rate=F25, channels=2, right_hz=660),
        "late": Spec("red", 440, 200, rate=F25, quiet=((60, 64),), audio_offset_s=0.25),
        "quiet_peaks": Spec("blue", 440, 400, rate=F25, tone_dbfs=-40.0,
                            clicks=tuple(1.0 + 2.5 * i for i in range(6))),
        "wide": Spec("green", 440, 400, rate=F25, tone_dbfs=-40.0, bursts=((0, 200, 30.0),)),
    }  # fmt: skip
    return {name: (make_media(root / f"{name}.mov", spec), spec) for name, spec in specs.items()}


def plan_of(media, segments: list[dict[str, Any]], fps: Fraction = F25, **extra: Any) -> EditPlan:
    used = {seg["source"] for seg in segments}
    if "outro" in extra:
        used.add(extra["outro"]["source"])
    body = {
        "project": "P",
        "timeline": "T",
        "kind": "longform",
        "fps": {"num": fps.numerator, "den": fps.denominator},
        "resolution": {"width": SIZE[0], "height": SIZE[1]},
        "profile": "test",
        "sources": [
            {"id": name, "media": str(path), "duration": spec.frames, "frame_size": list(SIZE)}
            for name, (path, spec) in media.items()
            if name in used
        ],
        "segments": segments,
        **extra,
    }
    return EditPlan.model_validate(body)


def quick() -> Profile:
    return fast(permissive())


def _levels(pcm: np.ndarray, sounds: list[dict], tones: dict[str, float]) -> dict[str, float]:
    """Each sound's own tone level over its middle (away from fades)."""
    out = {}
    for s in sounds:
        a, b = s["record"]
        lo = a + s["fade_in_frames"] + 2
        hi = b - s["fade_out_frames"] - 2
        out[s["location"]] = tone_dbfs(
            pcm, sample_at(lo, F25), sample_at(hi, F25), tones[s["source"]]
        )
    return out


@pytest.mark.parametrize("case", ["broll_then_outro", "nonchrono_split"])
def test_a_file_reused_out_of_order_keeps_every_sound(media, tmp_path: Path, case: str) -> None:
    """The validator's amix probes: V2 B-roll from the outro's file before the outro, and a
    split crossfade into an earlier part of a file. Each sound is its own input now."""
    if case == "broll_then_outro":
        plan = plan_of(
            media,
            [
                {"source": "A", "src_in": 0, "src_out": 100},
                {"source": "B", "src_in": 200, "src_out": 230, "track": 2, "record_in": 20},
            ],
            outro={"source": "B", "src_in": 0, "src_out": 40, "audio_track": "A1"},
            audio_tracks=[{"name": "A1"}, {"name": "A2"}],
        )
    else:
        plan = plan_of(
            media,
            [
                {"source": "A", "src_in": 300, "src_out": 350},
                {"source": "B", "src_in": 50, "src_out": 100},
                {"source": "A", "src_in": 100, "src_out": 150},
            ],
            transitions=[
                {"after": 0, "audio": {"frames": 2}},
                {"after": 1, "audio": {"frames": 2}},
            ],
        )
    result = render_plan(plan, quick(), out=tmp_path / f"{case}.mp4")
    pcm = read_pcm(result.out)
    sounds = result.report["rendered"]["sounds"]
    levels = _levels(pcm, sounds, {"A": 440, "B": 660})
    reference = levels["segments[0]"]
    for location, level in levels.items():
        assert level > -60, (location, level)
        assert abs(level - reference) < 0.5, (location, level, reference)


def test_a_late_audio_start_stays_in_sync(media, tmp_path: Path) -> None:
    """The source's audio starts 0.25 s (6.25 frames at 25) after its video. Its quiet span was
    made 2.4-2.56 s into the audio, so in the file it plays at 2.65-2.81 s: cut from frame 48
    (1.92 s), it must land 0.73-0.89 s into the render (not 0.48-0.64, counting samples from
    the audio's first one)."""
    plan = plan_of(media, [{"source": "late", "src_in": 48, "src_out": 120}])
    result = render_plan(plan, quick(), out=tmp_path / "late.mp4")
    assert [n["feature"] for n in result.report["notes"]] == ["audio_offset"]
    pcm = read_pcm(result.out)
    win = 48
    lo, hi = sample_at(10, F25), sample_at(30, F25)
    quiet = [s for s in range(lo, hi, win) if tone_dbfs(pcm, s, s + win, 440) < -50]
    assert quiet
    assert abs(quiet[0] - round(0.73 * 48000)) <= 2 * win
    assert abs(quiet[-1] + win - round(0.89 * 48000)) <= 2 * win
    # Before frame 48 - 6 = 42 of the file's audio there is nothing: a cut from frame 2 starts
    # with silence until the audio begins.
    early = plan_of(media, [{"source": "late", "src_in": 2, "src_out": 40}])
    first = render_plan(early, quick(), out=tmp_path / "early.mp4")
    pcm = read_pcm(first.out)
    assert tone_dbfs(pcm, 0, sample_at(3, F25), 440) < -60  # record 0-3 = video 2-5: no audio yet
    assert tone_dbfs(pcm, sample_at(6, F25), sample_at(20, F25), 440) > -40


def test_the_right_channel_is_the_sources_right(media, tmp_path: Path) -> None:
    plan = plan_of(media, [{"source": "stereo", "src_in": 10, "src_out": 60}])
    result = render_plan(plan, quick(), out=tmp_path / "st.mp4")
    pcm = read_pcm(result.out)
    a, b = sample_at(5, F25), sample_at(45, F25)
    left_440, left_660 = tone_dbfs(pcm, a, b, 440), tone_dbfs(pcm, a, b, 660)
    right = pcm[:, 1:2]
    right_440, right_660 = tone_dbfs(right, a, b, 440), tone_dbfs(right, a, b, 660)
    assert left_440 > left_660 + 40 and right_660 > right_440 + 40


def test_a_higher_track_is_drawn_on_top(media, tmp_path: Path) -> None:
    plan = plan_of(
        media,
        [
            {"source": "A", "src_in": 0, "src_out": 60},
            {"track": 3, "source": "C", "src_in": 100, "src_out": 130, "record_in": 10},
            {"track": 2, "source": "B", "src_in": 100, "src_out": 140, "record_in": 5},
        ],
    )
    result = render_plan(plan, quick(), out=tmp_path / "v3.mp4")
    got = {n: read_frame(f) for n, f in iter_frames(result.out, SIZE, [4, 7, 20, 41, 50])}
    assert got[4].colour == "red"
    assert (got[7].colour, got[7].index) == ("green", 102)  # V2 alone
    assert (got[20].colour, got[20].index) == ("blue", 110)  # V3 over V2
    assert (got[41].colour, got[41].index) == ("green", 136)  # V3 ended at 40, V2 runs to 45
    assert got[50].colour == "red"


def test_a_left_aligned_dissolve_plays_the_incoming_head(media, tmp_path: Path) -> None:
    """The outgoing clip ends at its source's last frame (no tail): the dissolve ends on the cut
    and plays 12 frames of the incoming clip's head before it, picture and sound."""
    plan = plan_of(
        media,
        [
            {"source": "A", "src_in": 340, "src_out": 400},
            {"source": "B", "src_in": 100, "src_out": 160},
        ],
        transitions=[
            {"after": 0, "video": "dissolve", "video_frames": 12, "audio": {"frames": 12}}
        ],
    )
    result = render_plan(plan, quick(), out=tmp_path / "left.mp4")
    assert result.report["rendered"]["dissolves"] == [
        {"location": "transitions[0]", "cut": 60, "window": [48, 60], "frames": 12}
    ]
    got = {n: read_frame(f) for n, f in iter_frames(result.out, SIZE, [47, 54, 60])}
    assert (got[47].colour, got[47].index) == ("red", 387)
    assert (got[60].colour, got[60].index) == ("green", 100)  # the incoming clip's in point
    share = blend_of(got[54].background, "red", "green")
    assert share is not None and 0.35 < share < 0.65
    sounds = {s["location"]: s for s in result.report["rendered"]["sounds"]}
    assert sounds["segments[0]"]["record"] == [0, 60]  # the outgoing sound has no tail to play
    assert sounds["segments[1]"]["record"] == [48, 120]
    pcm = read_pcm(result.out)
    level = tone_dbfs(pcm, sample_at(5, F25), sample_at(40, F25), 440)
    mid = tone_dbfs(pcm, sample_at(53, F25), sample_at(55, F25), 660)
    assert mid == pytest.approx(level + 20 * math.log10(6 / 12), abs=1.0)


def test_sound_is_placed_to_the_sample_at_2997(tmp_path: Path) -> None:
    """At 29.97 a frame is 1601.6 samples, so frame positions round; every sound must sit at lag
    0 against its source (a PCM master keeps the samples)."""
    rate = Fraction(30000, 1001)
    spec = Spec("red", 440, 300, rate=rate, noise_seed=7, channels=1)
    path = make_media(tmp_path / "noise.mov", spec)
    plan = plan_of(
        {"N": (path, spec)},
        [
            {"source": "N", "src_in": 7, "src_out": 70},
            {"source": "N", "src_in": 151, "src_out": 211},
            {"source": "N", "src_in": 33, "src_out": 97},
        ],
        fps=rate,
    )
    profile = quick()
    prores = profile.model_copy(
        update={"render": profile.render.model_copy(update={"master_codec": "prores"})}
    )
    result = render_plan(plan, prores, out=tmp_path / "n.mov", preset="master")
    pcm = read_pcm(result.out)[:, 0].astype(float)
    source = spec.noise()[:, 0].astype(float)
    for seg, (start, end) in zip(plan.segments, plan.record_ranges(), strict=True):
        out0 = round(Fraction(start) / rate * 48000) + 480
        src0 = round(Fraction(seg.src_in) / rate * 48000) + 480
        n = round(Fraction(end - start) / rate * 48000) - 960
        y = pcm[out0 : out0 + n]
        lags = {lag: float(np.dot(y, source[src0 + lag : src0 + lag + n])) for lag in range(-3, 4)}
        assert max(lags, key=lags.get) == 0, (seg.src_in, lags)


def test_loudness_is_one_gain_and_a_limiter_for_long_forms(media, tmp_path: Path) -> None:
    # A wide range (a section 30 dB louder than the rest): kept, on target.
    wide = plan_of(media, [{"source": "wide", "src_in": 0, "src_out": 400}])
    result = render_plan(wide, quick(), out=tmp_path / "wide.mp4")
    loud = result.report["loudness"]
    assert loud["method"] == "gain-limiter" and loud["output_samples"] == sample_at(400, F25)
    qa = {c["id"]: c for c in result.report["qa"]["checks"]}
    assert qa["loudness"]["measured"]["integrated_lufs"] == pytest.approx(-22, abs=0.5)
    pcm = read_pcm(result.out)
    loud_part = tone_dbfs(pcm, sample_at(20, F25), sample_at(180, F25), 440)
    quiet_part = tone_dbfs(pcm, sample_at(220, F25), sample_at(380, F25), 440)
    assert loud_part - quiet_part == pytest.approx(30.0, abs=1.0)
    assert loud["limiter_reduction_db"] < 1.0 and loud["warning"] is None
    # A quiet recording with sharp peaks: the gain would put them far over the maximum, so the
    # limiter works hard; on target, and a warning says so.
    peaky = plan_of(media, [{"source": "quiet_peaks", "src_in": 0, "src_out": 400}])
    result = render_plan(peaky, quick(), out=tmp_path / "peaky.mp4")
    loud = result.report["loudness"]
    qa = {c["id"]: c for c in result.report["qa"]["checks"]}
    assert qa["loudness"]["status"] == "pass", qa["loudness"]["message"]
    assert loud["limiter_reduction_db"] > 3.0 and "limiter" in loud["warning"]
    assert "limiter" in [n["feature"] for n in result.report["notes"]]


def test_refusals_before_any_work(media, tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(render, "run_ffmpeg", lambda *a, **k: calls.append("ffmpeg"))
    odd = plan_of(media, [{"source": "A", "src_in": 0, "src_out": 20}])
    odd = odd.model_copy(update={"resolution": odd.resolution.model_copy(update={"height": 179})})
    with pytest.raises(RenderError) as info:
        render_plan(odd, quick(), out=tmp_path / "odd.mp4")
    assert info.value.code == "odd_resolution"
    many = plan_of(
        media, [{"source": "A", "src_in": 3 * i, "src_out": 3 * i + 3} for i in range(120)]
    )
    profile = quick()

    def tight(mb: int) -> Profile:
        return profile.model_copy(
            update={"render": profile.render.model_copy(update={"memory_budget_mb": mb})}
        )

    with pytest.raises(RenderError) as info:  # under the encoder's own base (328 MB): it says so
        render_plan(many, tight(256), out=tmp_path / "many.mp4")
    assert info.value.code == "render_too_large" and info.value.details["budget_mb"] == 256
    assert info.value.details["reason"] == "encoder_base" and "encoder alone" in info.value.message
    with pytest.raises(RenderError) as info:  # the base fits, not with a picture (331 MB)
        render_plan(many, tight(330), out=tmp_path / "many.mp4")
    assert info.value.details["reason"] == "pictures" and info.value.details["pictures"] == 1
    assert info.value.details["start"] == 0 and "hold one picture" in info.value.message
    assert calls == [] and not list(tmp_path.iterdir())


def test_an_incomplete_encode_is_never_placed(media, tmp_path: Path, monkeypatch) -> None:
    real = render.run_ffmpeg

    def short(args: list[str], stage: str, timeout_s: float, cwd: Path) -> str:
        if stage == "encode":  # ffmpeg stops a frame early (a broken source, a full disk)
            args = [*args[:-1], "-frames:v", "39", args[-1]]
        return real(args, stage, timeout_s, cwd)

    monkeypatch.setattr(render, "run_ffmpeg", short)
    plan = plan_of(media, [{"source": "A", "src_in": 0, "src_out": 40}])
    out = tmp_path / "short.mp4"
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=out)
    assert info.value.code == "render_incomplete" and "39 frames" in info.value.message
    assert not out.exists() and not [p for p in tmp_path.iterdir() if "aar-render" in p.name]


def test_the_sound_track_is_the_timelines_exact_length(media, tmp_path: Path) -> None:
    plan = plan_of(media, [{"source": "A", "src_in": 0, "src_out": 37}])
    result = render_plan(plan, quick(), out=tmp_path / "len.mp4")
    out = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries"),
            *("stream=duration", "-of", "csv=p=0", str(result.out)),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert round(float(out) * 48000) == sample_at(37, F25)  # the edit list, not AAC frames


def _ffmpeg_in(work: Path) -> list[int]:
    """Processes whose working directory is the render's work dir (its ffmpeg calls)."""
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if os.readlink(entry / "cwd") == str(work):
                    found.append(int(entry.name))
            except OSError:
                continue
    return found


def test_sigterm_stops_ffmpeg_and_removes_the_work_dir(media, tmp_path: Path) -> None:
    plan = plan_of(media, [{"source": "A", "src_in": 0, "src_out": 400}])
    plan = plan.model_copy(update={"resolution": plan.resolution.model_copy(
        update={"width": 1280, "height": 720})})  # fmt: skip
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(plan.model_dump_json(), encoding="utf-8")
    profile = tmp_path / "slow.toml"
    profile.write_text(
        'name = "test"\n[qa]\nlongform_lufs = -22.0\nshort_lufs = -14.5\nlufs_tolerance = 0.5\n'
        'review_mp4_max_mb = 20\n[render]\nreview_x264_preset = "veryslow"\n'
        "[rules.ranges-valid]\n",
        encoding="utf-8",
    )
    out = tmp_path / "out" / "sig.mp4"
    out.parent.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-m", "aar", "plan", "render", str(plan_file), "--profile",
         str(profile), "--out", str(out)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )  # fmt: skip
    work = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and proc.poll() is None:
        dirs = [p for p in out.parent.iterdir() if ".aar-render-" in p.name]
        if dirs and (work := dirs[0]) and (work / "final.wav").exists() and _ffmpeg_in(work):
            break  # the encode is running
        time.sleep(0.05)
    assert work is not None and proc.poll() is None, "the render never reached its encode"
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 143, err
    assert b"SIGTERM" in err or b"signal 15" in err
    assert not work.exists() and not _ffmpeg_in(work)
    assert not out.exists() and not render.render_report_path(out).exists()


def test_a_plan_report_lists_round_2_facts(media, tmp_path: Path) -> None:
    plan = plan_of(media, [{"source": "late", "src_in": 0, "src_out": 50}])
    result = render_plan(plan, quick(), out=tmp_path / "facts.mp4")
    media_row = result.report["media"][0]
    assert media_row["audio_start_s"] - media_row["video_start_s"] == pytest.approx(0.25)
    assert media_row["frames"] == 200
    on_disk = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert on_disk["loudness"]["output_samples"] == sample_at(50, F25)
