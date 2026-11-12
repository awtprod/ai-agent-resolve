"""S19b: ``aar media shrink``, a small working copy of a camera proxy.

Sources are made here with lavfi (H.264 video, 24-bit PCM, with and without a tmcd timecode).
Every property of the copy is read back with ffprobe/ffmpeg run directly from the test, not
through :mod:`aar.media.shrink`, so a bug there cannot hide in the checks.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from media_support import FFMPEG, FFPROBE, needs_ffmpeg, posix_only

from aar.media import shrink as shrink_mod
from aar.media.probe import stream_rotation
from aar.media.quicktime import TimecodeUnreadable, read_timecode
from aar.media.shrink import (
    KEYFRAME_MAX_S,
    MARK_KEY,
    ShrinkError,
    shrink_file,
)

pytestmark = needs_ffmpeg

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
NOISE = "0.2*sin(2*PI*440*t)+0.02*(random(0)-0.5)"


def run(*args: str) -> bytes:
    assert FFMPEG is not None
    return subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", *args],
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout


def ffprobe_json(path: Path, *entries: str) -> dict:
    assert FFPROBE is not None
    out = subprocess.run(
        [FFPROBE, "-v", "error", *entries, "-of", "json", str(path)],
        check=True,
        capture_output=True,
        timeout=60,
    ).stdout
    return json.loads(out)


def make_source(
    path: Path,
    *,
    seconds: float = 5.0,
    timecode: str | None = "01:00:00:00",
    size: str = "320x180",
    rate: str = "30000/1001",
    second_audio: bool = False,
    vfr: bool = False,
) -> Path:
    """A camera-proxy-like file: H.264 (one keyframe only), 24-bit PCM stereo, a tmcd track."""
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}"]
    args += ["-f", "lavfi", "-i", f"aevalsrc={NOISE}|{NOISE}*0.5:s=48000:d={seconds}"]
    maps = ["-map", "0:v", "-map", "1:a"]
    codecs = ["-c:a:0", "pcm_s24le"]
    if second_audio:
        args += ["-f", "lavfi", "-i", f"aevalsrc={NOISE}:s=44100:d={seconds}"]
        maps += ["-map", "2:a"]
        codecs += ["-c:a:1", "pcm_s16le"]
    video = ["-c:v", "libx264", "-preset", "ultrafast", "-g", "1000", "-pix_fmt", "yuv420p"]
    if vfr:  # a 0.5 s hole after frame 30
        video += ["-vf", "setpts='PTS+gt(N,30)*0.5/TB'", "-fps_mode", "passthrough"]
    tc = ["-timecode", timecode] if timecode else []
    # Written as QuickTime first: ffmpeg 6.1 labels 24-bit PCM it encodes straight into MP4 as
    # 32-bit, but keeps it 24-bit when re-wrapping, as the camera's own MP4 proxies have it.
    mov = path.with_name(path.stem + ".tmp.mov")
    run(*args, *maps, *video, *codecs, *tc, "-y", str(mov))
    if path.suffix == ".mov":
        mov.rename(path)
    else:
        run("-i", str(mov), "-map", "0:v", "-map", "0:a", "-c", "copy", "-y", str(path))
        mov.unlink()
    return path


def video_packets(path: Path) -> list[tuple[int, bool]]:
    data = ffprobe_json(path, "-select_streams", "v:0", "-show_entries", "packet=pts,flags")
    return sorted((int(p["pts"]), p["flags"].startswith("K")) for p in data["packets"])


def streams(path: Path) -> list[dict]:
    return ffprobe_json(path, "-show_streams")["streams"]


def audio_pcm(path: Path, index: int) -> bytes:
    """One audio stream decoded to 32-bit integers."""
    return run("-i", str(path), "-map", f"0:a:{index}", "-c:a", "pcm_s32le", "-f", "s32le", "-")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if ".aar-shrink-" in p.name)


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("shrink-sources")
    return {
        "tc": make_source(root / "tc.mp4", second_audio=True),
        "notc": make_source(root / "notc.mp4", timecode=None),
        "mov": make_source(root / "cam.mov", seconds=3.0, timecode="10:00:00:12"),
        "short": make_source(root / "short.mp4", seconds=2.0),
    }


@pytest.mark.parametrize("name", ["tc", "notc", "mov"])
def test_copy_has_the_same_frames_timecode_and_lossless_audio(
    sources: dict[str, Path], tmp_path: Path, name: str
) -> None:
    src = sources[name]
    dst = tmp_path / src.name
    result = shrink_file(src, dst)
    assert result.status == "shrunk"
    assert dst.is_file() and leftovers(tmp_path) == []

    a, b = streams(src), streams(dst)
    va = next(s for s in a if s["codec_type"] == "video")
    vb = next(s for s in b if s["codec_type"] == "video")
    assert (vb["width"], vb["height"]) == (va["width"], va["height"])
    assert vb["r_frame_rate"] == va["r_frame_rate"] == "30000/1001"
    assert vb["avg_frame_rate"] == "30000/1001"
    assert (vb["codec_name"], vb["pix_fmt"]) == ("h264", "yuv420p")

    # One output frame per input frame: the same count, every step one frame (1001 ticks).
    pa, pb = video_packets(src), video_packets(dst)
    assert len(pb) == len(pa) > 60
    ticks = [y[0] - x[0] for x, y in itertools.pairwise(pb)]
    tb = vb["time_base"].split("/")
    frame = int(tb[1]) * 1001 // (30000 * int(tb[0]))
    assert set(ticks) == {frame}
    # A keyframe at least every two seconds (the source has only one).
    keys = [pts for pts, key in pb if key] + [pb[-1][0] + frame]
    assert sum(1 for _, key in pa if key) == 1
    gaps = [(y - x) * int(tb[0]) / int(tb[1]) for x, y in itertools.pairwise(keys)]
    assert len(keys) > 2 and max(gaps) <= float(KEYFRAME_MAX_S)

    # Every audio stream, same channels, rate and samples, decoded bit for bit.
    aa = [s for s in a if s["codec_type"] == "audio"]
    ab = [s for s in b if s["codec_type"] == "audio"]
    assert [(s["channels"], s["sample_rate"]) for s in ab] == [
        (s["channels"], s["sample_rate"]) for s in aa
    ]
    assert {s["codec_name"] for s in ab} == {"flac" if src.suffix == ".mp4" else "alac"}
    for n in range(len(aa)):
        original = audio_pcm(src, n)
        assert len(original) > 48000 * 4 and audio_pcm(dst, n) == original

    # The start timecode, by the repo's reader and ffprobe's tag; none invented.
    tags = ffprobe_json(dst, "-show_format")["format"].get("tags", {})
    assert tags[MARK_KEY].startswith("aar-shrink v2 ")
    if name == "notc":
        with pytest.raises(TimecodeUnreadable):
            read_timecode(dst)
        assert not any(s["codec_tag_string"] == "tmcd" for s in b)
        assert all("timecode" not in s.get("tags", {}) for s in b)
    else:
        got, want = read_timecode(dst), read_timecode(src)
        assert (got.frames, got.rate, got.drop_frame) == (want.frames, want.rate, want.drop_frame)
        assert vb["tags"]["timecode"] == va["tags"]["timecode"]


def test_rerun_skips_a_verified_copy_and_the_cli_registers(
    sources: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    src, dst = sources["short"], tmp_path / "short.mp4"
    assert main(["media", "shrink", str(src), "--out", str(dst)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "shrunk" and first["output_bytes"] == dst.stat().st_size
    stamp = (dst.stat().st_mtime_ns, sha(dst))
    report = tmp_path / "report.json"
    assert main(["media", "shrink", str(src), "--out", str(dst), "--report", str(report)]) == 0
    line = json.loads(capsys.readouterr().out)
    assert line["status"] == "skipped"
    assert (dst.stat().st_mtime_ns, sha(dst)) == stamp
    assert json.loads(report.read_text())["files"][0]["status"] == "skipped"


def test_an_existing_destination_that_does_not_verify_is_never_overwritten(
    sources: dict[str, Path], tmp_path: Path
) -> None:
    other = tmp_path / "other.mp4"
    shrink_file(sources["short"], other)  # a real shrink, of another source
    before = sha(other)
    with pytest.raises(ShrinkError) as exc:
        shrink_file(sources["notc"], other)
    assert exc.value.code == "destination_exists"
    assert any(d.startswith("frame count") for d in exc.value.details["differences"])
    assert sha(other) == before

    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a movie")
    with pytest.raises(ShrinkError) as exc:
        shrink_file(sources["short"], junk)
    assert exc.value.code == "destination_exists"
    assert junk.read_bytes() == b"not a movie"
    assert leftovers(tmp_path) == []


def test_replace_puts_the_verified_copy_in_place_and_refuses_twice(
    sources: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    src = tmp_path / "cam.mp4"
    src.write_bytes(sources["short"].read_bytes())
    original_audio = audio_pcm(src, 0)
    original_frames = len(video_packets(src))

    # Without --replace or --out nothing runs and nothing changes.
    before = sha(src)
    assert main(["media", "shrink", str(src)]) == 2
    assert sha(src) == before
    capsys.readouterr()

    assert main(["media", "shrink", str(src), "--replace"]) == 0
    line = json.loads(capsys.readouterr().out)
    assert line["status"] == "shrunk" and line["mode"] == "replace"
    assert sha(src) != before and src.stat().st_size == line["output_bytes"]
    assert len(video_packets(src)) == original_frames and audio_pcm(src, 0) == original_audio
    assert MARK_KEY in ffprobe_json(src, "-show_format")["format"]["tags"]
    assert leftovers(tmp_path) == []

    shrunk = sha(src)
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, replace=True)
    assert exc.value.code == "already_shrunk"
    assert "--replace never runs twice" in exc.value.message
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, tmp_path / "again.mp4")
    assert exc.value.code == "already_shrunk"
    assert sha(src) == shrunk and not (tmp_path / "again.mp4").exists()
    # In a folder, an already shrunk file is reported as done, not shrunk again.
    [done] = shrink_mod.shrink(tmp_path, replace=True)
    assert (done.status, done.ok) == ("skipped", True) and sha(src) == shrunk


@posix_only
def test_a_timed_out_encode_is_killed_and_its_partial_file_removed(
    sources: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real ffmpeg, made to read its input at real-time speed so the encode outlasts 1.5 s.
    wrapper = tmp_path / "bin" / "ffmpeg-realtime"
    wrapper.parent.mkdir()
    wrapper.write_text(f'#!/bin/sh\nexec "{FFMPEG}" -re "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("AAR_FFMPEG", str(wrapper))
    out = tmp_path / "out"
    out.mkdir()
    src = sources["tc"]
    before = sha(src)
    seen: list[int] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            for p in out.iterdir():
                if ".aar-shrink-" in p.name:
                    with contextlib.suppress(FileNotFoundError):
                        seen.append(p.stat().st_size)
            time.sleep(0.02)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with pytest.raises(ShrinkError) as exc:
            shrink_file(src, out / "tc.mp4", encode_timeout_s=1.5)
    finally:
        stop.set()
        watcher.join()
    assert exc.value.code == "timeout"
    assert seen and max(seen) > 0  # the partial file existed while ffmpeg ran
    assert list(out.iterdir()) == []
    assert sha(src) == before


def test_a_folder_batch_carries_on_past_a_bad_file(
    sources: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    folder = tmp_path / "Proxy"
    folder.mkdir()
    (folder / "A001.mp4").write_bytes(sources["short"].read_bytes())
    (folder / "B001.mp4").write_bytes(b"\x00" * 4096)  # not a movie
    (folder / "C001.mov").write_bytes(sources["mov"].read_bytes())
    (folder / "notes.txt").write_text("not video")
    out = tmp_path / "hub" / "Show" / "Proxy"  # created by the run
    report = tmp_path / "report.json"
    assert main(["media", "shrink", str(folder), "--out", str(out), "--report", str(report)]) == 2
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(Path(x["source"]).name, x["status"]) for x in lines] == [
        ("A001.mp4", "shrunk"),
        ("B001.mp4", "failed"),
        ("C001.mov", "shrunk"),
    ]
    assert lines[1]["error"]["code"] == "unreadable" and lines[1]["error"]["fix_hint"]
    assert sorted(p.name for p in out.iterdir()) == ["A001.mp4", "C001.mov"]
    data = json.loads(report.read_text())
    assert data["ok"] is False and [f["status"] for f in data["files"]] == [
        "shrunk",
        "failed",
        "shrunk",
    ]
    # A rerun resumes: the two done files are skipped, the bad one fails again.
    assert main(["media", "shrink", str(folder), "--out", str(out)]) == 2
    again = [json.loads(line)["status"] for line in capsys.readouterr().out.splitlines()]
    assert again == ["skipped", "failed", "skipped"]


def test_low_space_is_refused_before_encoding(sources: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(ShrinkError) as exc:
        shrink_file(sources["short"], tmp_path / "x.mp4", min_free_fraction=1e15)
    assert exc.value.code == "low_space" and exc.value.hint
    assert list(tmp_path.iterdir()) == []


def test_a_variable_frame_rate_source_is_refused_with_the_stub_wording(tmp_path: Path) -> None:
    from aar.plan.stub import _RENDER_CODES

    src = make_source(tmp_path / "vfr.mp4", seconds=2.0, vfr=True)
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, tmp_path / "out.mp4")
    assert exc.value.code == "vfr"
    assert "has a variable frame rate: " in exc.value.message
    assert (exc.value.code, exc.value.hint) == _RENDER_CODES["source_vfr"]
    assert not (tmp_path / "out.mp4").exists()


def test_the_plan_stub_of_the_copy_equals_the_original_s(
    sources: dict[str, Path], tmp_path: Path
) -> None:
    from aar.plan.stub import StubSource, build_stub

    src = sources["tc"]
    dst = tmp_path / "tc.mp4"
    shrink_file(src, dst)

    def stub(path: Path) -> tuple[dict, dict]:
        built = build_stub(
            [StubSource(path, camera="A")],
            recording_id="rec",
            project="P",
            timeline="T",
            kind="longform",
            profile="ccn",
        )
        plan = built.plan.model_dump(mode="json")
        report = json.loads(json.dumps(built.report))
        for item in (*plan["sources"], *report["sources"]):
            assert item.pop("media") == str(path)
        return plan, report

    original, copy = stub(src), stub(dst)
    assert copy == original
    assert original[0]["sources"][0]["start_timecode"] == 108000  # 01:00:00:00


def test_runs_from_the_source_tree_with_the_standard_library_only(
    sources: dict[str, Path], tmp_path: Path
) -> None:
    """As on the edit PC: ``python -S -m aar.media.shrink`` with only ``PYTHONPATH=src``. ``-S``
    leaves out site-packages, so no third-party package can be imported (``-I`` cannot be used:
    it ignores ``PYTHONPATH``). ``-X importtime`` lists every module the run imported."""
    dst = tmp_path / "copy.mp4"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(SRC_ROOT)
    check = [
        sys.executable,
        "-S",
        "-c",
        "import importlib.util as u; print(u.find_spec('pydantic'))",
    ]
    assert subprocess.run(check, capture_output=True, text=True, env=env, timeout=60).stdout == (
        "None\n"
    )
    done = subprocess.run(
        [
            *(sys.executable, "-S", "-X", "importtime", "-m", "aar.media.shrink"),
            *(str(sources["short"]), "--out", str(dst)),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1])["status"] == "shrunk"
    imported = set()
    for line in done.stderr.splitlines():
        fields = line.removeprefix("import time:").split("|")
        if line.startswith("import time:") and fields[0].strip().isdigit():
            imported.add(fields[-1].strip().split(".")[0])
    assert "aar" in imported and "json" in imported
    assert sorted(imported - set(sys.stdlib_module_names) - {"aar", "__main__"}) == []
    assert dst.is_file()


def test_settings_live_in_one_place_and_mark_says_the_version() -> None:
    assert shrink_mod.mark_version(shrink_mod.mark_value()) == shrink_mod.SETTINGS_VERSION
    assert shrink_mod.mark_version("something else") is None
    assert shrink_mod.mark_version(None) is None


# --- Fix round: stopping, the verification's every check, the picture, timing, safety --------

TC = ["-timecode", "01:00:00:00"]
ENC = [
    *("-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p"),
    *("-g", "59", "-fps_mode", "passthrough", "-enc_time_base:v", "demux"),
]


def mark_args() -> list[str]:
    return ["-metadata", f"{MARK_KEY}={shrink_mod.mark_value()}", "-movflags", "+use_metadata_tags"]


def make_clip(
    path: Path,
    *,
    seconds: float = 1.0,
    size: str = "320x180",
    pix_fmt: str = "yuv420p",
    layout: str = "stereo",
    acodec: str = "pcm_s24le",
    vf: str | None = None,
    mux: tuple[str, ...] = (),
    gop: int = 1000,
) -> Path:
    """A short source with one audio stream of ``layout`` in ``acodec`` and a timecode."""
    channels = {"stereo": 2, "5.1": 6}[layout]
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30000/1001:duration={seconds}"]
    expr = "|".join([NOISE] * channels)
    args += ["-f", "lavfi", "-i", f"aevalsrc={expr}:s=48000:d={seconds}:c={layout}"]
    video = ["-c:v", "libx264", "-preset", "ultrafast", "-g", str(gop), "-pix_fmt", pix_fmt]
    video += ["-vf", vf] if vf else []
    mov = path.with_name(path.stem + ".tmp.mov")
    run(*args, "-map", "0:v", "-map", "1:a", *video, "-c:a", acodec, *TC, "-y", str(mov))
    if path.suffix == ".mov" and not mux:
        mov.rename(path)
    else:
        run(*mux, "-i", str(mov), "-map", "0:v", "-map", "0:a", "-c", "copy", "-y", str(path))
        mov.unlink()
    return path


def offset_copy(base: Path, path: Path, *, video: str = "0", audio: str = "0") -> Path:
    """``base`` re-wrapped with its video and/or audio starting later (stream copy)."""
    run(
        *("-itsoffset", video, "-i", str(base), "-itsoffset", audio, "-i", str(base)),
        *("-map", "0:v", "-map", "1:a", "-c", "copy", *TC, "-y", str(path)),
    )
    return path


@pytest.fixture(scope="module")
def fool(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """A source (two audio streams, 24-bit and 16-bit, a timecode), its real shrink, and a
    folder for copies that are wrong in one way each."""
    root = tmp_path_factory.mktemp("shrink-fool")
    src = make_source(root / "src.mp4", seconds=2.0, second_audio=True)
    good = root / "good.mp4"
    shrink_file(src, good)
    return src, good, root


# Each copy differs from a true shrink in one way; ``what`` is the set of differences verify
# must name (the part before the first colon) and ``says`` text one of them must contain.
WRONG: dict[str, tuple[list[str], set[str], str]] = {
    "timecode": (
        ["-map", "0:v", "-map", "0:a", "-c", "copy", "-timecode", "01:00:00:01"],
        {"start timecode", "tmcd start timecode"},
        "start timecode: source 01:00:00:00, copy 01:00:00:01",
    ),
    "audio_low_bits": (  # the 24-bit stream cut to 16 bits: equal if hashed at 16 bits
        ["-map", "0:v", "-map", "0:a", "-c:v", "copy", "-c:a", "flac", "-sample_fmt", "s16", *TC],
        {"audio stream 0 decoded samples differ"},
        "",
    ),
    "audio_sample_short": (
        [
            *("-map", "0:v", "-map", "0:a", "-c:v", "copy", "-c:a", "flac"),
            *("-filter:a:0", "atrim=end_sample=95900", *TC),
        ],
        {"audio stream 0 sample count"},
        "source 96000, copy 95900",
    ),
    "audio_stream_missing": (
        ["-map", "0:v", "-map", "0:a:0", "-c", "copy", *TC],
        {"audio streams"},
        "source 2, copy 1",
    ),
    "pixel_aspect": (
        ["-map", "0:v", "-map", "0:a", "-c", "copy", "-aspect", "4:3", *TC],
        {"pixel aspect"},
        "copy 3/4",
    ),
    "frame_size": (
        ["-map", "0:v", "-map", "0:a", *ENC, "-vf", "scale=318:180,setsar=1", "-c:a", "copy", *TC],
        {"frame size"},
        "source (320, 180), copy (318, 180)",
    ),
    "mark_missing": (
        ["-map", "0:v", "-map", "0:a", "-c", "copy", "-map_metadata", "-1", *TC],
        {"mark"},
        "no aar_shrink mark",
    ),
    "uneven_frame": (  # frame 30 half a frame late: same count, same average rate
        [
            *("-map", "0:v", "-map", "0:a", *ENC, "-c:a", "copy", *TC),
            *("-vf", "setpts=PTS+eq(N\\,30)*0.5*1001/30000/TB"),
        ],
        {"frame rate", "constant frame rate"},
        "a frame lasts",
    ),
    "video_start": (
        ["-map", "0:v", "-map", "1:a", "-c", "copy", *TC],
        {"video start"},
        "source 0.000000 s, copy 0.100000 s",
    ),
    "audio_start": (
        ["-map", "0:v", "-map", "1:a", "-c", "copy", *TC],
        {"audio stream 0 start", "audio stream 1 start"},
        "copy 0.100000 s",
    ),
    "picture_negated": (
        ["-map", "0:v", "-map", "0:a", *ENC, "-vf", "negate", "-c:a", "copy", *TC],
        {"picture"},
        "another picture",
    ),
    "picture_shifted": (  # frame N shows source frame N+1 (the last one repeated)
        [
            *("-map", "0:v", "-map", "0:a", *ENC, "-c:a", "copy", *TC),
            *("-vf", "trim=start_frame=1,setpts=PTS-STARTPTS,tpad=stop=1:stop_mode=clone"),
        ],
        {"picture"},
        "shifted by a frame",
    ),
}


@pytest.mark.parametrize("case", sorted(WRONG))
def test_verify_names_a_copy_wrong_in_one_way(fool: tuple[Path, Path, Path], case: str) -> None:
    src, good, root = fool
    args, what, says = WRONG[case]
    inputs = ["-i", str(good)]
    if case == "video_start":
        inputs = ["-itsoffset", "0.1", "-i", str(good), "-i", str(good)]
    elif case == "audio_start":
        inputs = ["-i", str(good), "-itsoffset", "0.1", "-i", str(good)]
    bad = root / f"{case}.mp4"
    run(*inputs, *args, *(mark_args() if case != "mark_missing" else []), "-y", str(bad))
    try:
        with pytest.raises(ShrinkError) as exc:
            shrink_mod.verify(shrink_mod.read_facts(src), bad)
        found = exc.value.details["differences"]
        assert exc.value.code == "verify_failed"
        assert {d.split(":")[0] for d in found} == what, found
        assert any(says in d for d in found), found
    finally:
        bad.unlink()


def test_the_true_shrink_passes_every_check(fool: tuple[Path, Path, Path]) -> None:
    src, good, _ = fool
    facts = shrink_mod.read_facts(src)
    assert shrink_mod.differences(facts, shrink_mod.read_facts(good)) == []
    assert shrink_mod.picture_differences(facts, shrink_mod.read_facts(good), good) == []
    assert shrink_mod.verify(facts, good).frames == facts.frames == 60


def test_an_existing_copy_with_the_wrong_picture_is_not_taken_as_done(
    fool: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    src, good, _ = fool
    dst = tmp_path / "src.mp4"
    run(
        *("-i", str(good), "-map", "0:v", "-map", "0:a", *ENC, "-vf", "negate", "-c:a", "copy"),
        *TC,
        *mark_args(),
        *("-y", str(dst)),
    )
    before = sha(dst)
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, dst)
    assert exc.value.code == "destination_exists"
    assert any(d.startswith("picture") for d in exc.value.details["differences"])
    assert sha(dst) == before


def test_an_average_rate_off_its_base_rate_is_refused(tmp_path: Path) -> None:
    # Every frame step is one frame, but the last frame lasts 30: the average rate is off.
    base = make_clip(tmp_path / "base.mp4", seconds=2.0)
    last = max(pts for pts, _ in video_packets(base))
    src = tmp_path / "avg.mp4"
    run(
        *("-i", str(base), "-map", "0:v", "-map", "0:a", "-c", "copy", *TC),
        *("-bsf:v", f"setts=duration=if(eq(PTS\\,{last})\\,30030\\,1001)", "-y", str(src)),
    )
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, tmp_path / "out.mp4")
    assert exc.value.code == "vfr" and "average rate" in exc.value.message


@pytest.fixture
def no_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if an encode starts: refusals must come before it."""
    real = shrink_mod.run_tool

    def guarded(tool: str, args: list[str], **kw: object) -> object:
        assert "libx264" not in args, "an encode was started"
        return real(tool, args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(shrink_mod, "run_tool", guarded)


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("s32_mov", "audio_unsupported"),
        ("f32_mov", "audio_unsupported"),
        ("odd_size", "odd_size"),
        ("surround_to_mp4", "audio_unsupported"),
        ("cut_mid_gop", "negative_start"),
    ],
)
def test_a_source_that_cannot_be_kept_is_refused_before_encoding(
    tmp_path: Path, no_encode: None, case: str, code: str
) -> None:
    if case == "s32_mov":
        src = make_clip(tmp_path / "s.mov", acodec="pcm_s32le")
    elif case == "f32_mov":
        src = make_clip(tmp_path / "s.mov", acodec="pcm_f32le")
    elif case == "odd_size":
        src = make_clip(tmp_path / "s.mp4", vf="scale=321:181", pix_fmt="yuv444p")
    elif case == "surround_to_mp4":
        src = make_clip(tmp_path / "s.mp4", layout="5.1")
    else:  # cut with -c copy from inside a GOP: an edit list hides the frames before the cut
        base = make_clip(tmp_path / "base.mov", seconds=3.0, gop=30)
        src = tmp_path / "s.mp4"
        run(
            *("-ss", "1.3", "-i", str(base), "-t", "1", "-map", "0:v", "-map", "0:a"),
            *("-c", "copy", "-y", str(src)),
        )
    dst = tmp_path / "out.mp4"
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, dst)
    assert exc.value.code == code and exc.value.hint
    if case == "surround_to_mp4":
        assert "5.1" in exc.value.message and ".mov" in exc.value.hint
    assert not dst.exists() and leftovers(tmp_path) == []


def first_stream(path: Path, kind: str) -> dict:
    return next(s for s in streams(path) if s["codec_type"] == kind)


@pytest.mark.parametrize("case", ["rotated", "video_late", "audio_late", "surround_mov"])
def test_rotation_stream_starts_and_channel_layout_are_kept(tmp_path: Path, case: str) -> None:
    if case == "rotated":
        src = make_clip(tmp_path / "s.mp4", mux=("-display_rotation", "90"))
    elif case == "surround_mov":
        src = make_clip(tmp_path / "s.mov", layout="5.1")
    else:
        base = make_clip(tmp_path / "base.mov", seconds=2.0)
        late = {"video": "0.25"} if case == "video_late" else {"audio": "0.5"}
        src = offset_copy(base, tmp_path / "s.mp4", **late)
    dst = tmp_path / f"out{src.suffix}"
    assert shrink_file(src, dst).status == "shrunk"
    for kind in ("video", "audio"):
        a, b = first_stream(src, kind), first_stream(dst, kind)
        assert b["start_time"] == a["start_time"]
        if kind == "video":
            assert (b["width"], b["height"]) == (a["width"], a["height"]) == (320, 180)
            assert stream_rotation(b) == stream_rotation(a)
        else:
            assert b["channel_layout"] == a["channel_layout"]
    if case == "rotated":
        assert stream_rotation(first_stream(dst, "video")) == 90
    if case == "video_late":
        assert first_stream(dst, "video")["start_time"] == "0.250000"
    if case == "audio_late":
        assert first_stream(dst, "audio")["start_time"] == "0.500000"
        assert audio_pcm(dst, 0) == audio_pcm(src, 0)
    if case == "surround_mov":
        assert first_stream(dst, "audio")["channel_layout"] == "5.1"


def test_a_copy_with_another_channel_layout_is_refused(tmp_path: Path) -> None:
    # FLAC in MP4 turns 5.1 into 5.1(side): the samples are the same, the layout is not.
    src = make_clip(tmp_path / "s.mov", layout="5.1")
    good = tmp_path / "good.mov"
    shrink_file(src, good)
    bad = tmp_path / "bad.mp4"
    run(
        *("-i", str(good), "-map", "0:v", "-map", "0:a", "-c:v", "copy", "-c:a", "flac"),
        *TC,
        *mark_args(),
        *("-y", str(bad)),
    )
    with pytest.raises(ShrinkError) as exc:
        shrink_mod.verify(shrink_mod.read_facts(src), bad)
    assert exc.value.details["differences"] == [
        "audio stream 0 channel layout: source 5.1, copy 5.1(side)"
    ]


def _realtime_ffmpeg(folder: Path, log: Path) -> Path:
    """The real ffmpeg, reading at real-time speed for the encode and the audio hash (so a
    signal can land in either), and logging its process id and arguments."""
    wrapper = folder / "ffmpeg-realtime"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'echo "$$ $*" >> "{log}"\n'
        'case "$*" in *libx264*|*streamhash*) exec "' + str(FFMPEG) + '" -re "$@";; esac\n'
        f'exec "{FFMPEG}" "$@"\n'
    )
    wrapper.chmod(0o755)
    return wrapper


def _default_signals() -> None:  # in the child: a shell may have left SIGINT ignored
    import signal

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal.signal(getattr(signal, name), signal.SIG_DFL)


@posix_only
@pytest.mark.parametrize(
    ("sig", "phase", "mode"),
    [
        ("SIGTERM", "encode", "out"),
        ("SIGTERM", "encode", "replace"),
        ("SIGTERM", "hash", "out"),
        ("SIGTERM", "hash", "replace"),
        ("SIGHUP", "encode", "out"),
        ("SIGHUP", "encode", "replace"),
        ("SIGHUP", "hash", "out"),
        ("SIGHUP", "hash", "replace"),
        ("SIGINT", "encode", "out"),
        ("SIGINT", "hash", "replace"),
    ],
)
def test_a_stop_signal_stops_ffmpeg_and_removes_the_partial_copy(
    sources: dict[str, Path], tmp_path: Path, sig: str, phase: str, mode: str
) -> None:
    import shutil
    import signal

    log = tmp_path / "ffmpeg.log"
    wrapper = _realtime_ffmpeg(tmp_path, log)
    work = tmp_path / "work"
    work.mkdir()
    src = work / "cam.mp4"
    shutil.copyfile(sources["short"], src)
    before = sha(src)
    report = tmp_path / "report.json"
    where = ["--replace"] if mode == "replace" else ["--out", str(work / "copy.mp4")]
    env = {**os.environ, "PYTHONPATH": str(SRC_ROOT), "AAR_FFMPEG": str(wrapper)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "aar.media.shrink", str(src), *where, "--report", str(report)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=tmp_path,
        preexec_fn=_default_signals,
    )
    try:
        marker = "libx264" if phase == "encode" else "streamhash"
        pid = None
        deadline = time.monotonic() + 60
        while pid is None and time.monotonic() < deadline and proc.poll() is None:
            lines = log.read_text().splitlines() if log.exists() else []
            pid = next((int(ln.split()[0]) for ln in lines if marker in ln), None)
            time.sleep(0.02)
        assert pid is not None, "ffmpeg never reached the " + phase
        if phase == "encode":  # until the partial copy holds data
            while not any(p.stat().st_size for p in work.glob("*.part")):
                assert time.monotonic() < deadline
                time.sleep(0.02)
        else:
            time.sleep(0.3)
        proc.send_signal(getattr(signal, sig))
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 2, err
    line = json.loads(out.splitlines()[-1])
    assert line["error"]["code"] == "interrupted" and line["error"]["details"]["signal"] == sig
    assert "Traceback" not in err
    assert json.loads(report.read_text())["interrupted"]["details"]["signal"] == sig
    assert leftovers(work) == [] and sha(src) == before
    assert sorted(p.name for p in work.iterdir()) == ["cam.mp4"]
    with pytest.raises(ProcessLookupError):  # ffmpeg is gone, not left to finish the copy
        os.kill(pid, 0)


def test_the_no_clobber_rename_never_replaces_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    temp, final = tmp_path / ".x.part", tmp_path / "x.mp4"
    temp.write_bytes(b"new")
    final.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        shrink_mod._no_clobber_rename(temp, final)
    assert (final.read_bytes(), temp.read_bytes()) == (b"old", b"new")
    if os.name == "posix":  # the fallback for file systems without hard links refuses too

        def no_links(*_args: object, **_kw: object) -> None:
            raise OSError(errno.EPERM, "Operation not permitted")

        monkeypatch.setattr(shrink_mod.os, "link", no_links)
        with pytest.raises(FileExistsError):
            shrink_mod._no_clobber_rename(temp, final)
        assert final.read_bytes() == b"old"
    final.unlink()
    shrink_mod._no_clobber_rename(temp, final)
    assert final.read_bytes() == b"new" and not temp.exists()


def test_the_source_itself_is_never_a_destination(sources: dict[str, Path]) -> None:
    src = sources["short"]
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, src)
    assert exc.value.code == "destination_exists" and "is the source itself" in exc.value.message


@pytest.mark.parametrize(
    ("change", "mode"),
    [("append", "replace"), ("append", "out"), ("same_size_and_time", "replace")],
)
def test_a_source_that_changes_during_the_encode_is_left_and_the_copy_removed(
    sources: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    mode: str,
) -> None:
    if change == "same_size_and_time" and os.name != "posix":
        pytest.skip("only POSIX has a change time that a rewrite with the old mtime moves")
    src = tmp_path / "cam.mp4"
    src.write_bytes(sources["short"].read_bytes())
    real = shrink_mod.run_tool

    def encode_then_touch(tool: str, args: list[str], **kw: object) -> object:
        done = real(tool, args, **kw)  # type: ignore[arg-type]
        if "libx264" in args:
            st = src.stat()
            if change == "append":
                with src.open("ab") as f:
                    f.write(b"\0" * 16)
            else:  # one byte flipped in place, size and modification time put back
                with src.open("r+b") as f:
                    f.seek(st.st_size - 1)
                    last = f.read(1)
                    f.seek(st.st_size - 1)
                    f.write(bytes([last[0] ^ 1]))
                os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))
        return done

    monkeypatch.setattr(shrink_mod, "run_tool", encode_then_touch)
    expected = sha(src) if change == "append" else None
    dst = tmp_path / "copy.mp4"
    with pytest.raises(ShrinkError) as exc:
        shrink_file(src, replace=True) if mode == "replace" else shrink_file(src, dst)
    assert exc.value.code == "source_changed"
    assert MARK_KEY not in ffprobe_json(src, "-show_format")["format"].get("tags", {})
    assert not dst.exists() and leftovers(tmp_path) == []
    if expected is not None:
        assert sha(src) != expected  # the change, kept: nothing was put over it


def test_low_space_uses_half_the_source_by_default(
    sources: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    size = sources["short"].stat().st_size
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda p: real(p)._replace(free=int(size * 0.49)))
    with pytest.raises(ShrinkError) as exc:
        shrink_file(sources["short"], tmp_path / "x.mp4")
    assert exc.value.code == "low_space" and exc.value.details["need_bytes"] == size // 2
    assert list(tmp_path.iterdir()) == []


def test_a_folder_batch_skips_hidden_files(tmp_path: Path) -> None:
    for name in ("._A001.mp4", ".B001.mov", "C001.mp4", "D001.MOV", "e.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in shrink_mod.video_files(tmp_path)] == ["C001.mp4", "D001.MOV"]


@posix_only
def test_replace_refuses_a_symlink_before_encoding(
    sources: dict[str, Path], tmp_path: Path, no_encode: None
) -> None:
    target = tmp_path / "real.mp4"
    target.write_bytes(sources["short"].read_bytes())
    link = tmp_path / "link.mp4"
    link.symlink_to(target)
    with pytest.raises(ShrinkError) as exc:
        shrink_file(link, replace=True)
    assert exc.value.code == "symlink" and "points to" in exc.value.hint
    assert link.is_symlink() and sha(target) == sha(sources["short"])


@posix_only
def test_a_dangling_symlink_destination_is_refused_before_encoding(
    sources: dict[str, Path], tmp_path: Path, no_encode: None
) -> None:
    dst = tmp_path / "x.mp4"
    dst.symlink_to(tmp_path / "nowhere.mp4")
    with pytest.raises(ShrinkError) as exc:
        shrink_file(sources["short"], dst)
    assert exc.value.code == "destination_exists" and "already exists" in exc.value.message
    assert dst.is_symlink() and not (tmp_path / "nowhere.mp4").exists()


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere")
@pytest.mark.parametrize("mode", ["out", "replace"])
def test_a_folder_that_cannot_be_written_is_named_before_encoding(
    sources: dict[str, Path], tmp_path: Path, no_encode: None, mode: str
) -> None:
    folder = tmp_path / "ro"
    folder.mkdir()
    src = folder / "cam.mp4"
    src.write_bytes(sources["short"].read_bytes())
    folder.chmod(0o555)
    try:
        with pytest.raises(ShrinkError) as exc:
            if mode == "replace":
                shrink_file(src, replace=True)
            else:
                shrink_file(src, folder / "copy.mp4")
        assert exc.value.code == "not_writable"
        assert exc.value.details["folder"] == str(folder) and "writ" in exc.value.hint
        assert sorted(p.name for p in folder.iterdir()) == ["cam.mp4"]
    finally:
        folder.chmod(0o755)


@pytest.mark.parametrize("mode", ["replace", "out"])
def test_the_copy_is_flushed_before_its_rename_and_the_folder_after(
    sources: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import stat

    src = tmp_path / "cam.mp4"
    src.write_bytes(sources["short"].read_bytes())
    events: list[str] = []
    real_fsync, real_replace, real_link = os.fsync, os.replace, getattr(os, "link", None)

    def fsync(fd: int) -> None:
        events.append("fsync folder" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync file")
        real_fsync(fd)

    def replace(a: object, b: object) -> None:
        events.append("rename")
        real_replace(a, b)  # type: ignore[arg-type]

    def link(a: object, b: object) -> None:
        events.append("rename")
        real_link(a, b)  # type: ignore[misc]

    def rename(a: object, b: object) -> None:  # Windows: the no-clobber rename
        events.append("rename")
        real_rename(a, b)  # type: ignore[arg-type]

    real_rename = os.rename
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    if os.name != "posix":
        monkeypatch.setattr(os, "rename", rename)
    if real_link is not None:
        monkeypatch.setattr(os, "link", link)
    if mode == "replace":
        shrink_file(src, replace=True)
    else:
        shrink_file(src, tmp_path / "copy.mp4")
    folder = ["fsync folder"] if os.name == "posix" else []
    assert events == ["fsync file", "rename", *folder]
