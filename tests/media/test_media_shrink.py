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
    assert tags[MARK_KEY].startswith("aar-shrink v1 ")
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
    """As on the edit PC: ``python -m aar.media.shrink`` with ``src`` on the path, in an
    interpreter that cannot import third-party packages (no site-packages)."""
    dst = tmp_path / "copy.mp4"
    script = (
        "import importlib.util, json, runpy, sys\n"
        "assert importlib.util.find_spec('pydantic') is None\n"
        f"sys.path.insert(0, {str(SRC_ROOT)!r})\n"
        f"sys.argv = ['shrink', {str(sources['short'])!r}, '--out', {str(dst)!r}]\n"
        "code = 0\n"
        "try:\n"
        "    runpy.run_module('aar.media.shrink', run_name='__main__', alter_sys=True)\n"
        "except SystemExit as exc:\n"
        "    code = exc.code\n"
        "top = {m.split('.')[0] for m in sys.modules}\n"
        "extra = sorted(top - set(sys.stdlib_module_names) - {'aar', '__main__'})\n"
        "print(json.dumps({'code': code, 'extra': extra}))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    first, last = done.stdout.splitlines()[0], done.stdout.splitlines()[-1]
    assert json.loads(first)["status"] == "shrunk"
    assert json.loads(last) == {"code": 0, "extra": []}
    assert dst.is_file()


def test_settings_live_in_one_place_and_mark_says_the_version() -> None:
    assert shrink_mod.mark_version(shrink_mod.mark_value()) == shrink_mod.SETTINGS_VERSION
    assert shrink_mod.mark_version("something else") is None
    assert shrink_mod.mark_version(None) is None
