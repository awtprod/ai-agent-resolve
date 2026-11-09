"""S18b goldens: the CCN long-form and the reframed short fixtures (S18's export fixtures) rendered
from synthetic media mapped in with --media-map, then read back frame by frame and sample by
sample.

* Frames: every output frame shows the source and source frame the S18 layout puts there (cuts
  exact), except inside dissolves (a blend, checked at the midpoint) and the head's fade from
  black; the short's moving reframe crops sit where S18's crop windows put them, within 1 source
  pixel, located by the pattern block's edges.
* Sound: each segment's tone where it plays and nowhere else, the split crossfade's overlap and
  ramps, the gains (segment and track) within 0.5 dB, the head fade, sample-exact placement, and
  each kind's loudness method and target (QA).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from export_support import EXPORT_FIXTURES, ccn, load_plan
from plan_support import FIXTURES
from render_support import (
    SAMPLE_RATE,
    Spec,
    blend_of,
    fast,
    frame_count,
    iter_frames,
    make_media,
    read_frame,
    read_pcm,
    sample_at,
    tone_dbfs,
)

from aar.plan.export import media_map_from_mapping
from aar.plan.export.verify import window_from_plan
from aar.plan.model import EditPlan, ReframeKey, Transform
from aar.plan.render import RenderResult, render_plan

pytestmark = pytest.mark.slow
RENDER_FIXTURES = FIXTURES / "render"


@dataclass
class Rendered:
    name: str
    plan: EditPlan
    result: RenderResult
    specs: dict[str, Spec]  # by source id
    size: tuple[int, int]

    @property
    def report(self) -> dict:
        return self.result.report


def _render(root: Path, name: str, files: dict[str, tuple[str, Spec]]) -> Rendered:
    plan = load_plan(EXPORT_FIXTURES / f"{name}.plan.json")
    media = root / "media"
    specs: dict[str, Spec] = {}
    for source_id, (relative, spec) in files.items():
        path = media / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        make_media(path, spec)
        specs[source_id] = spec
    media_map = media_map_from_mapping({str(media): "E:\\CCN"})
    result = render_plan(plan, fast(ccn()), out=root / f"{name}.mp4", media_map=media_map)
    res = plan.resolution
    return Rendered(name, plan, result, specs, (res.width, res.height))


@pytest.fixture(scope="module")
def longform(tmp_path_factory: pytest.TempPathFactory) -> Rendered:
    # camA2's cut comes in at source frame 30: its sound is quiet there (QA's join check).
    files = {
        "camA": ("Ep1/A001.braw", Spec("red", 440, 1000)),
        "camB": ("Ep1/B001.braw", Spec("green", 660, 1000)),
        # A second quiet span (source 50-54, record 236-240) where camA2 plays alone pins the
        # sound's placement to the sample.
        "camA2": ("Ep1/A002.mov", Spec("blue", 880, 1000, quiet=((28, 32), (50, 54)))),
        # The sting is centred over the outro: its pattern sits top left.
        "outroclip": ("Assets/Outro.mov", Spec("yellow", 330, 120, centre=(64, 40))),
    }
    return _render(tmp_path_factory.mktemp("lf"), "ccn_longform_v2", files)


def _keys_at(keys: list[ReframeKey], t: int) -> tuple[float, float, float]:
    """The plan's reframe semantics, written out here (linear, held outside the keys)."""
    if not keys:
        return 0.5, 0.5, 1.0
    if t <= keys[0].time:
        return keys[0].x, keys[0].y, keys[0].zoom
    for a, b in itertools.pairwise(keys):
        if t < b.time:
            f = (t - a.time) / (b.time - a.time)
            return a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.zoom + (b.zoom - a.zoom) * f
    return keys[-1].x, keys[-1].y, keys[-1].zoom


def s18_window(plan: EditPlan, index: int, t: int, media: tuple[int, int]):
    """S18's crop window for segment ``index`` at its frame ``t``, in the hub file's pixels."""
    seg = plan.segments[index]
    source = plan.source_map()[seg.source]
    assert source.frame_size is not None
    fs = (source.frame_size[0], source.frame_size[1])
    res = plan.resolution
    x, y, zoom = _keys_at(list(seg.reframe), t)
    cx, cy, w, h = window_from_plan(
        x, y, zoom, seg.transform or Transform(), fs, res.width, res.height
    )
    kx, ky = media[0] / fs[0], media[1] / fs[1]
    return cx * kx, cy * ky, w * kx, h * ky


@pytest.fixture(scope="module")
def short(tmp_path_factory: pytest.TempPathFactory) -> Rendered:
    plan = load_plan(EXPORT_FIXTURES / "short_reframe.plan.json")
    size = (320, 180)
    # Each source's pattern goes where every one of its crop windows shows it (S18's windows).
    centres: dict[str, tuple[int, int]] = {}
    for index, seg in enumerate(plan.segments):
        boxes = [s18_window(plan, index, t, size) for t in range(seg.length)]
        left = max(cx - w / 2 for cx, _, w, _ in boxes)
        right = min(cx + w / 2 for cx, _, w, _ in boxes)
        top = max(cy - h / 2 for _, cy, _, h in boxes)
        bottom = min(cy + h / 2 for _, cy, _, h in boxes)
        assert right - left > 30 and bottom - top > 30
        centres[seg.source] = (round((left + right) / 2), round((top + bottom) / 2))
    files = {
        "camA": ("Ep1/A001.braw", Spec("red", 440, 2000, centre=centres["camA"], cell=4)),
        "camB": (
            "Ep1/B 001 (wide).braw",
            Spec("green", 660, 2000, centre=centres["camB"], cell=4),
        ),
    }
    return _render(tmp_path_factory.mktemp("sh"), "short_reframe", files)


# -- long-form ------------------------------------------------------------------------------------


def _expected(rendered: Rendered, frame: int) -> tuple[str, int] | None:
    """(colour, source frame) the layout puts at ``frame`` on V1; None inside a dissolve."""
    report = rendered.report["rendered"]
    for d in report["dissolves"]:
        if d["window"][0] <= frame < d["window"][1]:
            return None
    pic = [
        p for p in report["pictures"] if p["lane"] == 0 and p["record"][0] <= frame < p["record"][1]
    ]
    assert len(pic) == 1, frame
    spec = rendered.specs[pic[0]["source"]]
    return spec.colour, pic[0]["source_frames"][0] + frame - pic[0]["record"][0]


def test_longform_every_frame_is_the_layouts(longform: Rendered) -> None:
    out = longform.result.out
    total = longform.plan.total_frames()
    assert frame_count(out) == total == 426
    exact = blends = 0
    fade: list[float] = []
    for n, frame in iter_frames(out, longform.size, list(range(total))):
        reading = read_frame(frame)
        want = _expected(longform, n)
        if n < 15:  # head: a 15-frame dissolve from black, rendered as a fade
            fade.append(float(reading.background[0]))
            continue
        if want is None:
            if n == 306 + 7:  # the outro dissolve's midpoint: half camA2, half the outro
                share = blend_of(reading.background, "blue", "yellow")
                assert share is not None and 0.3 < share < 0.7, share
                blends += 1
            continue
        assert (reading.colour, reading.index) == want, n
        exact += 1
    assert exact == total - 15 - 15 and blends == 1
    assert fade[0] < 20 and all(b > a for a, b in itertools.pairwise(fade))


def test_longform_cuts_land_exactly(longform: Rendered) -> None:
    picks = [71, 72, 143, 144, 215, 216, 305, 321]
    got = {n: read_frame(f) for n, f in iter_frames(longform.result.out, longform.size, picks)}
    assert (got[71].colour, got[71].index) == ("red", 71)
    assert (got[72].colour, got[72].index) == ("green", 72)
    assert (got[143].colour, got[143].index) == ("green", 143)
    assert (got[144].colour, got[144].index) == ("red", 144)
    # The split crossfade keeps the picture cut at 216.
    assert (got[215].colour, got[215].index) == ("red", 215)
    assert (got[216].colour, got[216].index) == ("blue", 30)
    # The outro dissolve starts on the cut: frame 305 is camA2's last; 321 is the outro's 15th.
    assert (got[305].colour, got[305].index) == ("blue", 119)
    assert (got[321].colour, got[321].index) == ("yellow", 15)


def _level(pcm: np.ndarray, frame_a: int, frame_b: int, hz: float) -> float:
    return tone_dbfs(pcm, sample_at(frame_a), sample_at(frame_b), hz)


def test_longform_sound(longform: Rendered) -> None:
    pcm = read_pcm(longform.result.out)
    assert pcm.shape[1] == 2 and np.allclose(pcm[:, 0], pcm[:, 1], atol=1e-3)  # mono on both
    tones = {sid: spec.tone_hz for sid, spec in longform.specs.items()}
    sounds = {s["location"]: s for s in longform.report["rendered"]["sounds"]}
    levels: dict[str, float] = {}
    for location, sound in sounds.items():
        start = sound["record"][0] + sound["fade_in_frames"] + 2
        end = sound["record"][1] - sound["fade_out_frames"] - 2
        if location == "segments[3]":
            start = 240 + 2  # after camA2's quiet spans (source 28-32 and 50-54)
        own = _level(pcm, start, end, tones[sound["source"]])
        levels[location] = own
        for other, hz in tones.items():
            if other != sound["source"]:
                assert _level(pcm, start, end, hz) < own - 40, (location, other)
    # Gains: segment 2 is -2 dB, the outro on A5 (-1 dB); every tone was made at one level.
    assert levels["segments[2]"] - levels["segments[0]"] == pytest.approx(-2.0, abs=0.5)
    assert levels["outro"] - levels["segments[0]"] == pytest.approx(-1.0, abs=0.5)
    assert levels["segments[1]"] - levels["segments[0]"] == pytest.approx(0.0, abs=0.5)
    # The split crossfade (12 frames, centred on 216): both sounds play over [210, 222), each
    # ramping linearly; outside it only one.
    outgoing, incoming = levels["segments[2]"], levels["segments[3]"]
    for frame in (211, 212, 220):
        out_ramp = 20 * math.log10((222 - frame - 0.5) / 12)
        assert _level(pcm, frame, frame + 1, 440) == pytest.approx(outgoing + out_ramp, abs=1.0)
    for frame in (211, 212, 219, 220):
        in_ramp = 20 * math.log10((frame + 0.5 - 210) / 12)
        assert _level(pcm, frame, frame + 1, 880) == pytest.approx(incoming + in_ramp, abs=1.0)
    assert _level(pcm, 206, 209, 880) < incoming - 40
    assert _level(pcm, 223, 226, 440) < outgoing - 40
    # The head's 24-frame fade-in.
    head = levels["segments[0]"] + 20 * math.log10(6 / 24)
    assert _level(pcm, 5, 7, 440) == pytest.approx(head, abs=1.0)


def test_longform_sound_is_placed_to_the_sample(longform: Rendered) -> None:
    """camA2's quiet span (its source frames 50-54) lands at record 236-240, to within 2 ms."""
    pcm = read_pcm(longform.result.out)
    win = 48  # 1 ms
    lo, hi = sample_at(233), sample_at(243)
    rms = [float(np.sqrt(np.mean(pcm[s : s + win, 0] ** 2))) for s in range(lo, hi, win)]
    quiet = [lo + i * win for i, value in enumerate(rms) if 20 * math.log10(value + 1e-12) < -50]
    assert quiet
    assert abs(quiet[0] - sample_at(236)) <= 2 * win
    assert abs(quiet[-1] + win - sample_at(240)) <= 2 * win
    total = sample_at(426)
    assert total <= len(pcm) < total + 2048  # the AAC tail pads at most one frame


def test_longform_loudness_and_qa(longform: Rendered) -> None:
    report = longform.report
    loud = report["loudness"]
    assert loud["method"] == "gain-limiter" and loud["target_lufs"] == -22.0
    assert loud["limiter_reduction_db"] < 0.5 and loud["warning"] is None  # tones: no peaks
    assert loud["output_samples"] == 852852  # 426 frames at 23.976, exactly
    qa = {c["id"]: c for c in report["qa"]["checks"]}
    assert report["qa"]["status"] == "pass" and report["qa"]["kind"] == "longform"
    measured = qa["loudness"]["measured"]
    assert measured["integrated_lufs"] == pytest.approx(-22.0, abs=0.5)
    assert measured["true_peak_dbtp"] <= -1.0
    assert qa["duration"]["status"] == "pass" and qa["join-quiet"]["status"] == "pass"
    assert report["qa"]["unchecked"] == ["size-cap", "join-retranscribe", "caption-coverage"]


def test_longform_report(longform: Rendered) -> None:
    report = longform.report
    out = longform.result.out
    assert longform.result.report_path == out.with_name("ccn_longform_v2.render.json")
    on_disk = json.loads(longform.result.report_path.read_text(encoding="utf-8"))
    assert on_disk == report
    assert report["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert report["size_bytes"] == out.stat().st_size
    features = sorted(item["feature"] for item in report["not_rendered"])
    assert features == sorted(
        ["template", "cdl", "lut", "marker", "marker"]
        + ["audio_mute", "audio_bus", "bus_gain"] * 4
        + ["title_template", "title_template"]
    )
    assert [m["rule"] for m in report["media"]] == ["media-map"] * 4
    assert [t["location"] for t in report["rendered"]["titles"]] == ["titles[0]", "outro.sting"]
    assert not list(out.parent.glob(".ccn_longform_v2.mp4.aar-render-*"))  # the work dir is gone


def test_longform_titles_are_on_their_frames(longform: Rendered) -> None:
    # The lower third (48-120) is drawn bottom left in white over a dark box.
    picks = [47, 48, 119, 120]
    got = {
        n: f[0, 760:950, 115:400] for n, f in iter_frames(longform.result.out, longform.size, picks)
    }
    assert got[47].max() < 150 and got[120].max() < 150
    assert got[48].max() > 200 and got[119].max() > 200


# -- the short ------------------------------------------------------------------------------------


def test_short_crops_are_s18s_windows(short: Rendered) -> None:
    """Each frame's crop puts the pattern block where S18's window says, within 1 source px."""
    plan = short.plan
    width, height = short.size
    picks = sorted({*range(0, 480, 3), 119, 120, 121, 239, 240, 479})
    worst = 0.0
    bias = [0.0, 0.0]  # signed error, summed: horizontal edges, vertical edges
    edges = 0
    for n, frame in iter_frames(short.result.out, short.size, picks):
        index = 0 if n < 240 else 1
        seg = plan.segments[index]
        spec = short.specs[seg.source]
        t = n - (0 if index == 0 else 240)
        reading = read_frame(frame)
        assert (reading.colour, reading.index) == (spec.colour, seg.src_in + t), n
        cx, cy, w, h = s18_window(plan, index, t, spec.size)
        left, top, right, bottom = spec.block
        want = (
            (left - (cx - w / 2)) * width / w,
            (top - (cy - h / 2)) * height / h,
            (right - (cx - w / 2)) * width / w,
            (bottom - (cy - h / 2)) * height / h,
        )
        assert reading.ring is not None
        for axis, (got, expect) in enumerate(zip(reading.ring, want, strict=True)):
            per_px = width / w if axis % 2 == 0 else height / h
            worst = max(worst, abs(got - expect) / per_px)
            bias[axis % 2] += (got - expect) / per_px
        edges += 2
    assert worst < 1.0, f"a crop edge is {worst:.3f} source px from S18's window"
    # Measured here: under 0.1 px at worst. The edges' mean signed error is also near zero: a
    # slip of the half-pixel correction (perspective maps pixel indices, the window is in pixel
    # areas) would shift every edge on that axis by about 0.15 px.
    print(
        "crop edge bias (source px)", [round(b / edges, 4) for b in bias], "worst", round(worst, 4)
    )
    assert all(abs(b / edges) < 0.06 for b in bias), bias


def test_short_loudness_is_gain_and_limiter(short: Rendered) -> None:
    report = short.report
    loud = report["loudness"]
    assert loud["method"] == "gain-limiter" and loud["target_lufs"] == -14.5
    assert loud["gain_db"] == pytest.approx(-14.5 - loud["mix"]["integrated_lufs"], abs=0.3)
    qa = {c["id"]: c for c in report["qa"]["checks"]}
    assert qa["loudness"]["status"] == "pass"
    assert qa["loudness"]["measured"]["integrated_lufs"] == pytest.approx(-14.5, abs=0.5)
    assert qa["loudness"]["measured"]["true_peak_dbtp"] <= -1.0
    features = {item["feature"] for item in report["not_rendered"]}
    assert {"captions", "caption_style", "marker", "template"} <= features
    pcm = read_pcm(short.result.out)
    assert pcm.shape[0] >= sample_at(480) and SAMPLE_RATE == 48000


@pytest.mark.parametrize("which", ["longform", "short"])
def test_report_matches_the_golden(which: str, request: pytest.FixtureRequest) -> None:
    """What was rendered and every feature not rendered, pinned (the media paths, font file, sizes,
    hashes and measurements are machine-dependent and left out)."""
    rendered: Rendered = request.getfixturevalue(which)
    report = rendered.report
    body = {
        "timeline": report["timeline"],
        "rendered": {k: v for k, v in report["rendered"].items() if k != "font"},
        "not_rendered": report["not_rendered"],
        "notes": report["notes"],
        "loudness_method": report["loudness"]["method"],
    }
    text = json.dumps(body, indent=1, sort_keys=True) + "\n"
    golden = RENDER_FIXTURES / f"{rendered.name}.render.json"
    if os.environ.get("AAR_UPDATE_RENDER_GOLDEN") == "1":
        golden.write_text(text, encoding="utf-8")
    assert text == golden.read_text(encoding="utf-8")
