"""S18b round 4: page cache is cgroup headroom, a refusal that names the encoder's base, and the
validator's round-3 surviving mutants pinned (M04-M08, M20, M24)."""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from export_support import permissive
from render_support import Spec, fast, make_media
from test_render_graph import fake_hub
from test_render_r3 import _plan, long_plan

from aar.plan import render
from aar.plan.model import Fps
from aar.plan.profile import Profile
from aar.plan.render import RenderError, render_layout, render_plan
from aar.plan.render.chunks import Budget, Chunk, encode_base_mb, memory_budget, plan_chunks
from aar.plan.render.timeline import build_timeline

GIB = 1024**3
SIZE = (320, 180)
F25 = Fraction(25)
SCRIPT = render.MAX_FILTER_SCRIPT_BYTES


def quick(**render_settings: Any) -> Profile:
    profile = fast(permissive())
    return profile.model_copy(update={"render": profile.render.model_copy(update=render_settings)})


# -- the cgroup's headroom (no ffmpeg) -------------------------------------------------------------


def _cgroup(tmp_path: Path, *, limit: int, current: int, stat: str | None) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 64000000 kB\n")  # plenty: the cgroup or the cap decides
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/system.slice/cc.service\n")
    root = tmp_path / "fs"
    service = root / "system.slice" / "cc.service"
    service.mkdir(parents=True, exist_ok=True)
    (service / "memory.max").write_text(f"{limit}\n")
    (service / "memory.current").write_text(f"{current}\n")
    if stat is not None:
        (service / "memory.stat").write_text(stat)
    return {"meminfo": meminfo, "proc_cgroup": cgroup, "cgroup_root": root}


def _stat(anon: int, inactive_file: int, active_file: int = 0) -> str:
    return (
        f"anon {anon}\nfile {inactive_file + active_file}\nkernel 104857600\n"
        f"active_anon {anon}\ninactive_anon 0\nactive_file {active_file}\n"
        f"inactive_file {inactive_file}\nslab 52428800\n"
    )


def test_page_cache_is_headroom_and_anon_still_limits(tmp_path: Path) -> None:
    limit = 48 * GIB
    near = limit - round(1.1e9)  # 1.1 GB under the limit ...
    # ... and 96 % of it inactive page cache, as Command Center's group runs: the 6 GiB budget.
    cached = _cgroup(
        tmp_path / "a", limit=limit, current=near, stat=_stat(near // 25, near * 24 // 25)
    )
    b = memory_budget(**cached)
    assert b.mb == 6144 and b.limits["cgroup_headroom"] > 40000
    # The same group genuinely full (anon): 1.1 GB is all there is.
    full = _cgroup(tmp_path / "b", limit=limit, current=near, stat=_stat(near, 0))
    assert memory_budget(**full).mb == pytest.approx(1.1e9 / 2**20, rel=1e-6)
    # Active page cache is not counted as room (only what is reclaimed first).
    active = _cgroup(tmp_path / "c", limit=limit, current=near, stat=_stat(0, 0, near))
    assert memory_budget(**active).mb == pytest.approx(1.1e9 / 2**20, rel=1e-6)
    # Over the limit (the kernel lets current pass max briefly): floored at 0.
    over = _cgroup(tmp_path / "d", limit=limit, current=limit + GIB, stat=_stat(limit + GIB, 0))
    assert memory_budget(**over).mb == 0


@pytest.mark.parametrize("stat", ["missing", "unreadable", "no inactive_file"])
def test_without_a_usable_memory_stat_the_headroom_is_max_minus_current(
    tmp_path: Path, stat: str
) -> None:
    limit, current = 8 * GIB, 8 * GIB - 3 * GIB // 2
    text = None if stat != "no inactive_file" else "anon 1\nfile 2\n"
    paths = _cgroup(tmp_path, limit=limit, current=current, stat=text)
    if stat == "unreadable":  # a directory where the file should be: reading it fails
        (paths["cgroup_root"] / "system.slice" / "cc.service" / "memory.stat").mkdir()
    assert memory_budget(**paths).mb == pytest.approx(1536)


def test_a_limiting_parent_with_page_cache_counts_its_own_stat(tmp_path: Path) -> None:
    paths = _cgroup(tmp_path, limit=48 * GIB, current=GIB, stat=_stat(GIB, 0))
    slice_dir = paths["cgroup_root"] / "system.slice"
    (slice_dir / "memory.max").write_text(f"{20 * GIB}\n")
    (slice_dir / "memory.current").write_text(f"{20 * GIB - GIB // 2}\n")  # 512 MB under ...
    (slice_dir / "memory.stat").write_text(_stat(2 * GIB, 17 * GIB))  # ... 17 GiB of it cache
    assert memory_budget(**paths).mb == 6144
    (slice_dir / "memory.stat").write_text(_stat(19 * GIB, 0))
    assert memory_budget(**paths).mb == pytest.approx(512)


# -- 4K: the encoder's base (no ffmpeg) -----------------------------------------------------------


def _uhd(cuts: int):
    plan = long_plan(cuts).model_copy(
        update={
            "resolution": long_plan(1).resolution.model_copy(update={"width": 3840, "height": 2160})
        }
    )
    plan = plan.model_copy(
        update={
            "sources": [s.model_copy(update={"frame_size": (3840, 2160)}) for s in plan.sources]
        }
    )
    hub = fake_hub(plan, size=(3840, 2160))
    return plan, hub, build_timeline(render_layout(plan, hub, permissive()), hub)


def test_a_4k_output_under_its_encoder_base_says_so() -> None:
    plan, hub, tl = _uhd(30)
    assert encode_base_mb(plan.resolution) == pytest.approx(4300)  # 300 + 1000 x 4 (1080p areas)
    with pytest.raises(RenderError) as info:
        plan_chunks(tl, hub, Budget(3072.0, "default", {}), max_script_bytes=SCRIPT)
    err = info.value
    assert err.code == "render_too_large" and err.details["reason"] == "encoder_base"
    assert (err.details["base_mb"], err.details["budget_mb"]) == (4300, 3072)
    assert "the encoder alone needs about 4300 MB at a 3840x2160 output" in err.message
    assert "3072 MB" in err.message and "picture(s)" not in err.message
    assert "--memory-budget" in err.fix_hint and "free memory" in err.fix_hint


def test_a_4k_plan_packs_a_few_uses_a_chunk() -> None:
    """M20: the encoder's base scales with the output's pixels (4,300 MB at 4K), so a 6 GiB
    chunk holds about 7 4K uses (the validator measured 4,941 MB for 7)."""
    _, hub, tl = _uhd(30)
    chunks = plan_chunks(tl, hub, Budget(6144.0, "default", {}), max_script_bytes=SCRIPT)
    assert all(c.planned_mb <= 6144 for c in chunks)
    assert max(c.pictures for c in chunks) <= 8 and len(chunks) >= 4
    assert sum(c.pictures for c in chunks) == 30


# -- the join's checks, by injection ---------------------------------------------------------------


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Spec]]:
    root = tmp_path_factory.mktemp("r4_media")
    specs = {
        "A": Spec("red", 440, 400, rate=F25, centre=(100, 60)),
        "B": Spec("green", 660, 400, rate=F25, centre=(220, 60)),
    }
    return {name: (make_media(root / f"{name}.mov", spec), spec) for name, spec in specs.items()}


def plan_of(media, segments, **extra):
    used = {s["source"] for s in segments}
    return _plan(
        {n: (p, spec.frames, SIZE) for n, (p, spec) in media.items() if n in used},
        segments,
        **extra,
    )


TWO = [{"source": "A", "src_in": 0, "src_out": 30}, {"source": "B", "src_in": 100, "src_out": 130}]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("times", "why"),
    [
        ([(0, -1001), (1001, 0), (3003, 1001)], "not one continuous frame grid"),  # M04
        ([(0, 0), (1001, 0), (2002, 1001)], "decode times are not strictly increasing"),  # M05
        ([(0, -1001), (2002, 0), (1001, 2002)], "a frame is decoded after it is shown"),
    ],
)
def test_the_joined_file_check_refuses_broken_timestamps(
    media, monkeypatch, times: list[tuple[int, int]], why: str
) -> None:
    """M04, M05: the joined file's packets, replaced (the encoder never makes these)."""
    path, _ = media["A"]  # a real file for the sound check; its packets are replaced
    listed = [[(0, -1001), (1001, 0), (2002, 1001)]]

    def packets(_path: Path, _stage: str) -> render.Packets:
        return render.Packets(listed[0], True, Fraction(1, 24000))

    monkeypatch.setattr(render, "_packets", packets)
    render._check_output(path, 3, Fraction(1001, 24000))  # the clean grid passes
    listed[0] = times
    with pytest.raises(RenderError) as info:
        render._check_output(path, 3, Fraction(1001, 24000))
    assert info.value.code == "render_incomplete" and why in info.value.message


@pytest.mark.slow
def test_a_chunk_that_does_not_start_on_a_key_frame_is_never_joined(
    media, tmp_path: Path, monkeypatch
) -> None:
    """M06."""
    real = render._packets

    def packets(path: Path, stage: str) -> render.Packets:
        got = real(path, stage)
        return render.Packets(got.times, False, got.time_base) if stage == "encode" else got

    monkeypatch.setattr(render, "_packets", packets)
    out = tmp_path / "k.mp4"
    with pytest.raises(RenderError) as info:
        render_plan(plan_of(media, TWO), quick(), out=out)
    assert info.value.code == "render_incomplete" and "start on a key frame" in info.value.message
    assert not out.exists()


@pytest.mark.slow
def test_chunks_with_different_parameter_sets_are_never_joined(
    media, tmp_path: Path, monkeypatch
) -> None:
    """M07: the chunks' extradata (H.264 SPS/PPS) must be equal, not just the codec."""
    real = render._stream_setup
    seen: list[Path] = []

    def setup(path: Path) -> dict[str, Any]:
        seen.append(path)
        got = dict(real(path))
        if len(seen) > 1:
            got["extradata_hash"] = "SHA256:another"
        return got

    monkeypatch.setattr(render, "_stream_setup", setup)
    plan = plan_of(media, TWO)
    out = tmp_path / "x.mp4"
    budget = encode_base_mb(plan.resolution) + 4  # one 320x180 use (3.4 MB) a chunk: 2 chunks
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=out, memory_budget_mb=budget)
    err = info.value
    assert err.code == "render_incomplete" and err.details["differ"] == ["extradata_hash"]
    assert "chunk 2" in err.message and not out.exists()


@pytest.mark.slow
def test_a_short_chunk_names_the_picture_that_is_short_not_the_first(media, tmp_path: Path) -> None:
    """M24: the chunk holds a clean picture of A, then a damaged range of B: B is named."""
    path, spec = media["B"]
    data = bytearray(path.read_bytes())
    mid = len(data) // 2
    data[mid : mid + 200_000] = bytes(200_000)
    damaged = tmp_path / "damagedB.mov"
    damaged.write_bytes(bytes(data))
    a_path, a_spec = media["A"]
    plan = _plan(
        {"A": (a_path, a_spec.frames, SIZE), "B": (damaged, spec.frames, SIZE)},
        [
            {"source": "A", "src_in": 0, "src_out": 30},
            {"source": "B", "src_in": 50, "src_out": 350},
        ],
    )
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=tmp_path / "s.mp4")
    err = info.value
    assert err.code == "render_incomplete" and err.details["source"] == "B"
    assert err.details["location"] == "segments[1]" and err.details["decoded"] < 300


# -- the frame grid after many joins (M08) --------------------------------------------------------


@pytest.mark.slow
def test_the_join_puts_90_one_frame_chunks_back_on_the_frame_grid(tmp_path: Path) -> None:
    """The concat demuxer places each chunk at the sum of the durations before it, in whole
    microseconds; a 23.976 frame is 41,708.33 us, so 0.33 us drifts away at every join and past
    about 63 joins a chunk lands a tick early. ``setts`` restores the grid (the validator saw the
    unrounded join fail at 75 chunks). One one-frame chunk, joined 90 times."""
    fps = Fps(num=24000, den=1001)
    profile = quick()
    codec = render._codec_args("review", profile, "mp4")
    work = tmp_path / "w"
    work.mkdir()
    part = work / "part.mp4"
    subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi"),
            *("-i", "testsrc2=s=64x36:r=24000/1001", "-frames:v", "1", "-an"),
            *("-r", "24000/1001", "-fps_mode", "cfr", *codec.video),
            *("-video_track_timescale", str(render._track_timescale(fps)), str(part)),
        ],
        check=True,
    )
    n = 90
    samples = n * 2002
    subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi"),
            *("-i", "anullsrc=r=48000:cl=stereo", "-af", f"atrim=end_sample={samples}"),
            *("-c:a", "pcm_f32le", str(work / "final.wav")),
        ],
        check=True,
    )
    chunks = [Chunk(k, k + 1, 1, 0.0) for k in range(n)]
    out = work / "joined.mp4"
    timeline_s = Fraction(n * 1001, 24000)
    render._join([part] * n, chunks, codec, "mp4", fps, profile.render, timeline_s, out, work)
    render._check_output(out, n, Fraction(1001, 24000))  # one continuous grid, else raises
    probe = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pts", "-of", "csv=p=0", str(out)),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert sorted(int(p) for p in probe) == [k * 1001 for k in range(n)]


# -- a Matroska file that starts late (the PR #16 finding) ----------------------------------------


def test_the_matroska_duration_tag_is_an_end_time() -> None:
    from aar.plan.render.media import _elapsed

    tag = {"tags": {"DURATION": "00:00:15.000000000"}}
    assert _elapsed({**tag, "start_time": "5.000000"}) == 10  # ffmpeg 6.1: the stream's end
    assert _elapsed(tag) == 15  # no start: from 0
    assert _elapsed({**tag, "start_time": "0.000000"}) == 15
    assert _elapsed({**tag, "start_time": "16.0"}) is None  # not a length: no frame-count test
    assert _elapsed({"start_time": "5.0"}) is None


@pytest.fixture(scope="module")
def mkv(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Spec]]:
    root = tmp_path_factory.mktemp("r4_mkv")
    spec = Spec("blue", 880, 250, rate=F25, container="matroska")
    zero = make_media(root / "zero.mkv", spec)
    late = root / "late.mkv"  # the same file, its timestamps starting at 5 s
    subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(zero), "-map", "0"),
            *("-c", "copy", "-output_ts_offset", "5", str(late)),
        ],
        check=True,
    )
    short = Spec("red", 440, 100, rate=F25, container="matroska", audio=False)
    video = make_media(root / "video100.mkv", short)
    longaudio = root / "longaudio.mkv"  # its sound runs 3 s past its 100 frames of video
    subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(video), "-f", "lavfi"),
            *("-i", "sine=f=440:r=48000:d=7", "-map", "0:v", "-map", "1:a", "-c:v", "copy"),
            *("-c:a", "pcm_s16le", str(longaudio)),
        ],
        check=True,
    )
    late_long = root / "late_longaudio.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(longaudio), "-map", "0"),
            *("-c", "copy", "-output_ts_offset", "5", str(late_long)),
        ],
        check=True,
    )
    return {"zero": (zero, spec), "late": (late, spec), "longaudio": (longaudio, short),
            "late_longaudio": (late_long, short)}  # fmt: skip


@pytest.mark.slow
@pytest.mark.parametrize("name", ["zero", "late"])
def test_a_matroska_file_renders_whatever_its_start(mkv, tmp_path: Path, name: str) -> None:
    from render_support import iter_frames, read_frame

    from aar.plan.render.media import resolve_source

    path, spec = mkv[name]
    plan = _plan({"s": (path, spec.frames, SIZE)}, [{"source": "s", "src_in": 100, "src_out": 130}])
    media = resolve_source(plan.sources[0], plan.fps, media_map=None, media_root=None)
    assert media.frames == 250 and media.video_start_s == (5 if name == "late" else 0)
    result = render_plan(plan, quick(), out=tmp_path / f"{name}.mp4")
    got = {n: read_frame(f) for n, f in iter_frames(result.out, SIZE, [0, 15, 29])}
    assert [(got[n].colour, got[n].index) for n in (0, 15, 29)] == [
        ("blue", 100),
        ("blue", 115),
        ("blue", 129),
    ]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["longaudio", "late_longaudio"])
def test_a_matroska_segment_past_its_video_is_still_refused(mkv, tmp_path: Path, name: str) -> None:
    """The validator's round-1 case: the sound runs past the video; the frames are the video's."""
    path, _ = mkv[name]
    plan = _plan({"s": (path, 180, SIZE)}, [{"source": "s", "src_in": 80, "src_out": 140}])
    with pytest.raises(RenderError) as info:
        render_plan(plan, quick(), out=tmp_path / "past.mp4")
    assert info.value.code == "segment_past_media_end" and info.value.details["media_frames"] == 100
