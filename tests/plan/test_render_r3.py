"""S18b round 3: memory bounded by construction, a linear mix, timestamp-exact cuts.

The picture is planned in chunks under a memory budget, cut only at hard cuts, encoded one ffmpeg
at a time and joined by stream copy into one continuous stream (exact frame count, the frame grid,
strictly increasing DTS); the mix is layered (no sound padded to the timeline) and may run in
passes; a picture is cut by its timestamps, so a damaged source leaves its chunk short and is
refused naming it; every source's packet timestamps are scanned for a variable rate; a letterbox
with an odd margin is centred; and the validator's surviving round-2 mutants are pinned."""

from __future__ import annotations

import dataclasses
import itertools
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from export_support import permissive
from render_support import Spec, blend_of, fast, iter_frames, make_media, read_frame, read_pcm
from test_render_graph import fake_hub

from aar.plan import render
from aar.plan.model import EditPlan
from aar.plan.profile import Profile
from aar.plan.render import RenderError, render_layout, render_plan
from aar.plan.render import chunks as chunking
from aar.plan.render.chunks import (
    Budget,
    cut_points,
    encode_base_mb,
    memory_budget,
    picture_use_mb,
    plan_chunks,
)
from aar.plan.render.graph import picture_use
from aar.plan.render.media import Timing, _check_rate, resolve_source, used_source_ids
from aar.plan.render.timeline import build_timeline

SIZE = (320, 180)
F25 = Fraction(25)
SCRIPT = render.MAX_FILTER_SCRIPT_BYTES


def quick(**render_settings: Any) -> Profile:
    profile = fast(permissive())
    return profile.model_copy(update={"render": profile.render.model_copy(update=render_settings)})


def _plan(
    sources: dict[str, tuple[Path | str, int, tuple[int, int]]], segments, **extra
) -> EditPlan:
    fps = extra.pop("fps", F25)
    res = extra.pop("resolution", SIZE)
    return EditPlan.model_validate(
        {
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "fps": {"num": fps.numerator, "den": fps.denominator},
            "resolution": {"width": res[0], "height": res[1]},
            "profile": "test",
            "sources": [
                {"id": sid, "media": str(path), "duration": frames}
                | ({} if size is None else {"frame_size": list(size)})
                for sid, (path, frames, size) in sources.items()
            ],
            "segments": segments,
            **extra,
        }
    )


# -- the chunk plan (no ffmpeg) -------------------------------------------------------------------


def long_plan(cuts: int, *, dissolve_every: int = 25) -> EditPlan:
    """The validator's long-form shape: three 1080p cameras, ``cuts`` segments with 2-frame
    sound crossfades, every third punched in; and a 12-frame video dissolve every so often."""
    seg = 36
    segments, transitions = [], []
    for k in range(cuts):
        s: dict[str, Any] = {
            "source": f"c{k % 3}",
            "src_in": 100 + k * seg,
            "src_out": 100 + (k + 1) * seg,
        }
        if k % 3 == 2:
            s["transform"] = {"zoom": 1.25}
        segments.append(s)
        if k:
            t: dict[str, Any] = {"after": k - 1, "audio": {"frames": 2}}
            if k % dissolve_every == 0:
                t.update(video="dissolve", video_frames=12)
            transitions.append(t)
    frames = 100 + (cuts + 2) * seg
    sources = {f"c{i}": (f"/srv/media/c{i}.mov", frames, (1920, 1080)) for i in range(3)}
    return _plan(
        sources,
        segments,
        transitions=transitions,
        fps=Fraction(24000, 1001),
        resolution=(1920, 1080),
    )


def test_a_400_cut_plan_is_planned_in_chunks_under_the_budget() -> None:
    plan = long_plan(400)
    hub = fake_hub(plan, size=(1920, 1080))
    tl = build_timeline(render_layout(plan, hub, permissive()), hub)
    budget = Budget(6144.0, "override", {})
    chunks = plan_chunks(tl, hub, budget, max_script_bytes=SCRIPT)
    assert len(tl.pictures) == 400 and len(tl.dissolves) == 15
    assert 4 <= len(chunks) <= 8  # about 80 uses a chunk at 6 GiB
    assert all(c.planned_mb <= budget.mb for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == tl.total_frames
    assert all(a.end == b.start for a, b in itertools.pairwise(chunks))
    assert sum(c.pictures for c in chunks) == 400
    for chunk in chunks:
        # No picture, so no dissolve window (the outgoing picture plays through it), crosses.
        assert not any(p.start < chunk.start < p.end for p in tl.pictures)
        assert not any(
            d.window_start < chunk.start < d.window_start + d.frames for d in tl.dissolves
        )
    # Uncapped, the same plan would need about 24 GB in one ffmpeg (the round-2 measurement).
    total = encode_base_mb(plan.resolution) + sum(
        picture_use_mb(p, hub[p.source_id]) for p in tl.pictures
    )
    assert total > 20000
    # The model is the validator's measurement, rounded up: 1,177 MB + 44.7 MB a 1080p use.
    assert encode_base_mb(plan.resolution) >= 1177
    plain = next(p for p in tl.pictures if p.framing.mode != "window")
    punched = next(p for p in tl.pictures if p.framing.mode == "window")
    assert picture_use_mb(plain, hub[plain.source_id]) >= 44.7
    assert picture_use_mb(punched, hub[punched.source_id]) >= 75


def test_a_stretch_with_no_hard_cut_over_the_budget_is_refused_naming_it() -> None:
    plan = long_plan(30, dissolve_every=1)  # a dissolve at every cut: no hard cut anywhere
    hub = fake_hub(plan, size=(1920, 1080))
    tl = build_timeline(render_layout(plan, hub, permissive()), hub)
    assert cut_points(tl) == [0, tl.total_frames]
    with pytest.raises(RenderError) as info:
        plan_chunks(tl, hub, Budget(2048.0, "override", {}), max_script_bytes=SCRIPT)
    err = info.value
    assert err.code == "render_too_large" and err.details["pictures"] == 30
    assert (err.details["start"], err.details["end"]) == (0, tl.total_frames)
    assert "record frames 0.." in err.message and "--memory-budget" in err.fix_hint
    # With room for it all, it is one chunk.
    assert len(plan_chunks(tl, hub, Budget(1e9, "override", {}), max_script_bytes=SCRIPT)) == 1


def test_the_default_budget_is_the_smallest_limit(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 64000000 kB\nMemAvailable:    8388608 kB\n")
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/system.slice/cc.service\n")
    root = tmp_path / "fs"
    (root / "system.slice" / "cc.service").mkdir(parents=True)
    (root / "system.slice" / "memory.max").write_text("max\n")
    (root / "system.slice" / "memory.current").write_text("1\n")
    service = root / "system.slice" / "cc.service"
    (service / "memory.max").write_text(f"{3 * 1024**3}\n")
    (service / "memory.current").write_text(f"{1024**3 + 512 * 1024**2}\n")

    def budget(**kw: Any) -> Budget:
        return memory_budget(meminfo=meminfo, proc_cgroup=cgroup, cgroup_root=root, **kw)

    b = budget()
    assert b.source == "default" and b.mb == pytest.approx(1536)  # the cgroup's 1.5 GiB free
    assert b.limits == pytest.approx(
        {"cap": 6144, "mem_available_share": 4096, "cgroup_headroom": 1536}
    )
    (service / "memory.max").write_text("max\n")  # no limit: half of MemAvailable
    assert budget().mb == pytest.approx(4096)
    meminfo.write_text("MemAvailable: 64000000 kB\n")  # plenty: the 6 GiB cap
    assert budget().mb == 6144
    assert budget(override_mb=900).mb == 900 and budget(override_mb=900).source == "override"


# -- the validator's surviving round-2 mutants, as unit pins --------------------------------------


def _timing(**kw: Any) -> Timing:
    base = dict(
        start_s=Fraction(0),
        video_start_s=Fraction(0),
        audio_start_s=Fraction(0),
        base_rate=F25,
        avg_rate=F25,
        frames=250,
        frames_from="nb_frames",
        duration_frames=Fraction(250),
        uneven=None,
        sample_rate=48000,
        time_base=Fraction(1, 12800),
    )
    base.update(kw)
    return Timing(**base)


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"avg_rate": F25 * Fraction(1005, 1000)}, "average rate"),  # R02: 1/1000, not 1/100
        ({"uneven": (Fraction(4), Fraction(3, 2))}, "4.000 s into its video"),  # R03
        ({"frames": 252}, "holds 252 frames"),  # R04: a slack of one frame, not three
        ({"frames": 248, "frames_from": "packet count"}, "holds 248 frames"),
    ],
)
def test_each_rate_test_refuses_on_its_own(change: dict[str, Any], why: str) -> None:
    plan = _plan(
        {"s": ("/srv/media/s.mov", 250, SIZE)}, [{"source": "s", "src_in": 0, "src_out": 10}]
    )
    source = plan.sources[0]
    _check_rate(source, Path("/srv/media/s.mov"), _timing(), F25)  # the clean timing passes
    with pytest.raises(RenderError) as info:
        _check_rate(source, Path("/srv/media/s.mov"), _timing(**change), F25)
    assert info.value.code == "source_vfr" and why in info.value.message


def test_a_stream_without_a_time_base_is_refused() -> None:
    plan = _plan(
        {"s": ("/srv/media/s.mov", 250, SIZE)}, [{"source": "s", "src_in": 0, "src_out": 10}]
    )
    with pytest.raises(RenderError) as info:
        _check_rate(plan.sources[0], Path("/srv/media/s.mov"), _timing(time_base=None), F25)
    assert info.value.code == "media_unreadable"


def test_the_picture_seek_counts_the_video_streams_late_start() -> None:
    """R09: a video stream that starts 250 ms into its file is seeked from its own start. (The
    timestamp cut would still pick the right frames from an earlier seek; this pins the seek, so
    nothing is decoded twice over.)"""
    plan = _plan(
        {"s": ("/srv/media/s.mov", 250, SIZE)}, [{"source": "s", "src_in": 100, "src_out": 110}]
    )
    late = dataclasses.replace(fake_hub(plan)["s"], video_start_s=Fraction(1, 4))
    tl = build_timeline(render_layout(plan, {"s": late}, permissive()), {"s": late})
    use = picture_use(tl.pictures[0], late, plan.fps)
    assert use.seek_s == Fraction(1, 4) + (100 - Fraction(1, 2)) / 25


# -- renders ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Spec]]:
    root = tmp_path_factory.mktemp("r3_media")
    specs = {
        "A": Spec("red", 440, 400, rate=F25, centre=(100, 60)),
        "B": Spec("green", 660, 400, rate=F25, centre=(220, 60)),
        "C": Spec("blue", 880, 400, rate=F25, centre=(160, 70)),
    }
    return {name: (make_media(root / f"{name}.mov", spec), spec) for name, spec in specs.items()}


def plan_of(media, segments, **extra) -> EditPlan:
    used = {s["source"] for s in segments}
    sources = {n: (p, spec.frames, SIZE) for n, (p, spec) in media.items() if n in used}
    return _plan(sources, segments, **extra)


SEGMENTS = [
    {"source": "A", "src_in": 0, "src_out": 30},
    {"source": "B", "src_in": 100, "src_out": 130},
    {"source": "C", "src_in": 200, "src_out": 230},
    {"source": "A", "src_in": 50, "src_out": 80},
    {"source": "B", "src_in": 150, "src_out": 180},
    {"source": "C", "src_in": 250, "src_out": 280},
    {"source": "A", "src_in": 300, "src_out": 330},
    {"source": "B", "src_in": 220, "src_out": 250},
    {"track": 2, "source": "C", "src_in": 10, "src_out": 25, "record_in": 35},
]
COLOUR = {"A": "red", "B": "green", "C": "blue"}


def tiny_budget(plan: EditPlan, uses: float) -> float:
    """A budget that holds the encode's base and ``uses`` of these small pictures."""
    per = chunking.USE_FIXED_MB + chunking.PLAIN_USE_MB * (SIZE[0] * SIZE[1]) / (1920 * 1080)
    return encode_base_mb(plan.resolution) + uses * per


def _packets(path: Path) -> list[tuple[int, int]]:
    out = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pts,dts", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [tuple(int(x) for x in line.split(",")[:2]) for line in out.split()]  # type: ignore[misc]


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["review", "master", "prores"])
def test_tiny_chunks_join_into_one_frame_exact_stream(media, tmp_path: Path, preset: str) -> None:
    plan = plan_of(
        media,
        SEGMENTS,
        transitions=[{"after": 3, "video": "dissolve", "video_frames": 8}],
        titles=[{"kind": "lower-third", "lines": ["Joined"], "record_in": 140, "record_out": 200}],
    )
    profile = quick(master_codec="prores") if preset == "prores" else quick()
    out = tmp_path / ("j.mov" if preset == "prores" else "j.mp4")
    result = render_plan(
        plan,
        profile,
        out=out,
        preset="review" if preset == "review" else "master",
        memory_budget_mb=tiny_budget(plan, 2.5),
    )
    memory = result.report["memory"]
    spans = [(c["start"], c["end"]) for c in memory["chunks"]]
    # Two or three uses a chunk: cut only at hard cuts, never inside the connected clip's
    # segment (30..60) or the dissolve's (90..150).
    assert spans == [(0, 30), (30, 60), (60, 90), (90, 150), (150, 210), (210, 240)]
    assert all(c["planned_mb"] <= memory["budget"]["mb"] for c in memory["chunks"])
    packets = _packets(result.out)
    assert len(packets) == 240
    dts = [d for _, d in packets]
    assert all(b > a for a, b in itertools.pairwise(dts))  # no non-monotonic DTS
    pts = sorted(p for p, _ in packets)
    step = pts[1] - pts[0]
    assert pts == [pts[0] + k * step for k in range(240)]  # one continuous frame grid
    assert all(d <= p for p, d in packets)
    window = result.report["rendered"]["dissolves"][0]["window"]
    frames = dict(iter_frames(result.out, SIZE, list(range(240))))
    bad = []
    for r in range(240):
        got = read_frame(frames[r])
        if window[0] <= r < window[1]:
            share = blend_of(got.background, "red", "green")
            want = (r - window[0]) / (window[1] - window[0])
            if share is None or abs(share - want) > 0.1:
                bad.append((r, "blend", share, want))
            continue
        if 35 <= r < 50:
            want_src, want_index = "C", 10 + r - 35  # the connected clip on V2
        else:
            k = r // 30
            seg = SEGMENTS[k]
            want_src, want_index = seg["source"], seg["src_in"] + r - 30 * k
        if (got.colour, got.index) != (COLOUR[want_src], want_index):
            bad.append((r, got.colour, got.index, want_src, want_index))
    assert bad == []

    # The title spans the join at 150: drawn on both sides of it, on its frames only.
    def titled(r: int) -> bool:
        region = frames[r][0, 136:156, 17:60].astype(float)
        return float(np.abs(region - read_frame(frames[r]).background[0]).mean()) > 15

    assert [titled(r) for r in (139, 140, 149, 150, 199, 200)] == [False, *[True] * 4, False]


@pytest.mark.slow
def test_mix_passes_sum_to_the_one_pass_mix(media, tmp_path: Path, monkeypatch) -> None:
    plan = plan_of(
        media,
        SEGMENTS[:6],
        transitions=[{"after": k, "audio": {"frames": 6}} for k in range(5)],
    )
    hub = {
        sid: resolve_source(plan.source_map()[sid], plan.fps, media_map=None, media_root=None)
        for sid in used_source_ids(plan)
    }
    tl = build_timeline(render_layout(plan, hub, quick()), hub)
    settings = quick().render
    mixes = {}
    for per_pass in (100, 2):
        work = tmp_path / f"w{per_pass}"
        work.mkdir()
        passes = render._mix(tl, hub, per_pass, settings, work)
        mixes[per_pass] = (
            passes,
            read_pcm(work / "mix.wav"),
            sorted(p.name for p in work.iterdir()),
        )
    assert mixes[100][0] == 1 and mixes[2][0] == 3
    one, three = mixes[100][1], mixes[2][1]
    assert one.shape == three.shape == (round(180 / 25 * 48000), 2)
    assert float(np.abs(one - three).max()) < 1e-6
    assert float(np.abs(one).max()) > 0.05  # not silence
    assert not any(n.startswith("group") and n.endswith(".wav") for n in mixes[2][2])  # removed
    # And a budget that holds two sounds a pass renders in passes.
    monkeypatch.setattr(chunking, "SOUND_USE_MB", 1000.0)
    budget = Budget(chunking.MIX_FIXED_MB + 2000.0 + 1, "override", {})
    assert chunking.mix_groups(6, budget) == 2


@pytest.mark.slow
def test_a_damaged_source_is_refused_naming_it(media, tmp_path: Path) -> None:
    """The validator's corrupt_middle: bytes zeroed in the middle of the file, the index
    intact. The frames that no longer decode leave the chunk short; the render names the
    source and its range instead of shifting the rest of the picture."""
    path, spec = media["A"]
    data = bytearray(path.read_bytes())
    mid = len(data) // 2
    data[mid : mid + 200_000] = bytes(200_000)
    damaged = tmp_path / "damaged.mov"
    damaged.write_bytes(bytes(data))
    plan = _plan(
        {"A": (damaged, spec.frames, SIZE)},
        [{"source": "A", "src_in": 50, "src_out": 350}],
    )
    out = tmp_path / "d.mp4"
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=out)
    err = info.value
    assert err.code == "render_incomplete" and err.details["source"] == "A"
    assert (err.details["src_start"], err.details["src_end"]) == (50, 349)
    assert err.details["decoded"] < 300 and "damaged.mov" in err.message
    assert not out.exists()


@pytest.fixture(scope="module")
def cfr(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Spec]:
    root = tmp_path_factory.mktemp("r3_vfr")
    spec = Spec("red", 440, 200, rate=F25)
    return make_media(root / "cfr.mov", spec), spec


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "ext", "vf", "at"),
    [
        ("jump20ms", "mkv", "setpts='if(gte(N,100),PTS+0.020/TB,PTS)'", "3.960 s"),
        ("jump30ms", "mov", "setpts='if(gte(N,120),PTS+0.030/TB,PTS)'", "4.760 s"),
        ("dropped", "mkv", "select='not(eq(n,150))'", "5.960 s"),
    ],
)
def test_an_uneven_packet_step_is_refused_naming_its_time(
    cfr, tmp_path: Path, name: str, ext: str, vf: str, at: str
) -> None:
    path, spec = cfr
    uneven = tmp_path / f"{name}.{ext}"
    subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(path), "-vf", vf),
            *("-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"),
            *("-c:a", "copy", str(uneven)),
        ],
        check=True,
    )
    plan = _plan(
        {"s": (uneven, spec.frames - 2, SIZE)}, [{"source": "s", "src_in": 160, "src_out": 190}]
    )
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=tmp_path / "v.mp4")
    assert info.value.code == "source_vfr" and at in info.value.message


@pytest.mark.slow
def test_a_letterbox_with_an_odd_margin_is_centred(tmp_path: Path) -> None:
    """The validator's c2997 plan 2 at small scale: a square picture in an 854x480 frame is
    480 wide with 187 px each side (4:4:4 at the pad: not rounded down to 186)."""
    spec = Spec("green", 440, 30, size=(240, 240), rate=F25)
    path = make_media(tmp_path / "square.mov", spec)
    plan = _plan(
        {"sq": (path, 30, None)},  # unconformed: fitted inside the frame
        [{"source": "sq", "src_in": 0, "src_out": 10}],
        resolution=(854, 480),
    )
    result = render_plan(plan, quick(), out=tmp_path / "lb.mp4")
    frame = next(iter(iter_frames(result.out, (854, 480), [5])))[1]
    row = frame[0, 20, :].astype(int)  # a background row (above the pattern block)
    picture = np.nonzero(row > 40)[0]
    assert (int(picture[0]), int(picture[-1])) == (187, 666)


@pytest.mark.slow
def test_a_file_without_sound_fails_the_output_check(tmp_path: Path) -> None:
    """R06: the output check refuses a file with no sound stream."""
    path = make_media(tmp_path / "mute.mov", Spec("red", 440, 10, rate=F25, audio=False))
    with pytest.raises(RenderError) as info:
        render._check_output(path, 10, Fraction(1, 25))
    assert info.value.code == "render_incomplete" and "no audio stream" in info.value.message
    with pytest.raises(RenderError) as info:
        render._check_output(path, 11, Fraction(1, 25))
    assert "10 frames, not the timeline's 11" in info.value.message


@pytest.mark.slow
def test_a_short_processed_sound_is_never_placed(media, tmp_path: Path, monkeypatch) -> None:
    """R14: the loudness pass's output length is checked against the timeline's samples."""
    real = render.normalize

    def short(*args: Any, **kwargs: Any):
        result = real(*args, **kwargs)
        result.output_samples -= 1
        return result

    monkeypatch.setattr(render, "normalize", short)
    plan = plan_of(media, [{"source": "A", "src_in": 0, "src_out": 20}])
    out = tmp_path / "s.mp4"
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=out)
    assert info.value.code == "render_incomplete" and info.value.details["stage"] == "loudness"
    assert not out.exists() and not [p for p in tmp_path.iterdir() if "aar-render" in p.name]
