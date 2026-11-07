"""``analyze.faces`` without OpenCV: real ffmpeg decode, a detector that finds a bright box.

The fixture is a 160x96, 10 fps, 3 s clip: a 24x16 white box on a dark background whose left
edge is at ``8 + 3 * n`` px in source frame n (overlay snaps it to an even pixel) and whose top
is at 12 px. Rotated copies carry a display matrix of +90 or -90 degrees, which ffmpeg applies
on decode, so the frames it emits are 96x160. Sample k at 2 fps is source frame 5k exactly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from analyze_support import FFMPEG, needs_ffmpeg, tone, write_wav

from aar.analyze import AnalyzeError
from aar.analyze.common import checkpoint_path, explicit_offset, read_checkpoint
from aar.analyze.faces import (
    FaceSettings,
    VideoInfo,
    analyse_faces,
    frame_size,
    normalise,
)
from aar.cli import main

W, H, BOX_W, BOX_H, BOX_Y = 160, 96, 24, 16, 12
SAMPLES = 6  # 3 s at 2 fps


def ffmpeg(*args: str) -> None:
    assert FFMPEG is not None
    subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", *args],
        check=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if FFMPEG is None:
        pytest.skip("ffmpeg not found")
    root = tmp_path_factory.mktemp("clips")
    graph = (
        f"color=c=0x101010:s={W}x{H}:r=10:d=3[bg];color=c=white:s={BOX_W}x{BOX_H}:r=10:d=3[box];"
        f"[bg][box]overlay=x='8+30*t':y={BOX_Y}:shortest=1"
    )
    plain = root / "plain.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", graph, "-c:v", "libx264", "-preset", "ultrafast"),
        *("-pix_fmt", "yuv420p", "-g", "5", str(plain)),
    )
    out = {"plain": plain}
    for name, degrees in (("r90", "90"), ("r270", "-90")):
        out[name] = root / f"{name}.mp4"
        ffmpeg("-display_rotation:v:0", degrees, "-i", str(plain), "-c", "copy", str(out[name]))
    return out


def box_x(sample: int) -> int:
    """Left edge (px, unrotated) of the box in sample ``sample`` (source frame 5 * sample)."""
    return (8 + 3 * 5 * sample) // 2 * 2


def expected_box(sample: int, rotation: str) -> list[float]:
    x = box_x(sample)
    if rotation == "plain":
        return [x / W, BOX_Y / H, BOX_W / W, BOX_H / H]
    if rotation == "r90":  # +90: displayed counter-clockwise, 96x160
        return [BOX_Y / H, (W - x - BOX_W) / W, BOX_H / H, BOX_W / W]
    return [(H - BOX_Y - BOX_H) / H, x / W, BOX_H / H, BOX_W / W]  # -90: clockwise


class BrightBox:
    """A detector that reports the bounding box of the bright pixels, with fake landmarks."""

    def __init__(self, log: dict[str, Any], size: tuple[int, int], fail_at: int | None) -> None:
        self.log, self.size, self.fail_at = log, size, fail_at

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        calls = self.log.setdefault("frames", [])
        if self.fail_at is not None and len(calls) == self.fail_at:
            raise RuntimeError("detector crashed")
        calls.append(frame.shape)
        ys, xs = np.nonzero(frame.max(axis=2) > 128)
        if len(xs) == 0:
            return None
        x0, y0 = float(xs.min()), float(ys.min())
        w, h = float(xs.max() + 1) - x0, float(ys.max() + 1) - y0
        marks = [x0 + w / 2, y0 + h / 2] * 5
        return np.array([[x0, y0, w, h, *marks, 0.9]], dtype=np.float32)


def bright_factory(log: dict[str, Any], fail_at: int | None = None) -> Any:
    def factory(model: Path, size: tuple[int, int], score: float, nms: float, top_k: int) -> Any:
        log["created"] = {"model": model, "size": size, "args": (score, nms, top_k)}
        return BrightBox(log, size, fail_at)

    return factory


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "yunet-acme.onnx"
    path.write_bytes(b"not really a model, but the bytes are hashed")
    return path


def run(clip: Path, out: Path, model: Path, **kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    log: dict[str, Any] = {}
    factory = bright_factory(log, kw.pop("fail_at", None))
    report = analyse_faces(clip, out, model=model, detector_factory=factory, **kw)
    return report, log


# ---------------------------------------------------------------- end to end


@needs_ffmpeg
@pytest.mark.parametrize("rotation", ["plain", "r90", "r270"])
@pytest.mark.parametrize("max_side", [640, 80])
def test_boxes_and_times_follow_the_known_motion(tmp_path, clips, model, rotation, max_side):
    out = tmp_path / "faces.json"
    report, log = run(
        clips[rotation],
        out,
        model,
        settings=FaceSettings(max_side=max_side),
        offset=explicit_offset(10.0),
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    scale = min(1, max_side / W)
    want_size = (W * scale, H * scale) if rotation == "plain" else (H * scale, W * scale)
    assert doc["header"]["frame_size"] == list(want_size) == report["frame_size"]
    assert log["created"]["size"] == tuple(want_size)
    assert log["created"]["args"] == (0.6, 0.3, 50)
    assert set(log["frames"]) == {(want_size[1], want_size[0], 3)}  # (h, w, BGR)
    source = doc["header"]["source"]
    assert source["rotation"] == {"plain": 0, "r90": 90, "r270": 270}[rotation]
    assert (source["display_width"], source["display_height"]) == (
        (W, H) if rotation == "plain" else (H, W)
    )
    assert doc["header"]["time_tolerance_s"] == 0.1  # one source frame at 10 fps
    frames = doc["frames"]
    assert len(frames) == SAMPLES == doc["summary"]["frames"]
    tolerance = 1.01 / min(want_size)  # one pixel of the emitted frame
    for k, frame in enumerate(frames):
        assert frame["file_s"] == float(Fraction(k, 2))
        assert frame["recording_s"] == pytest.approx(10.0 + k / 2)
        [face] = frame["faces"]
        assert face["box"] == pytest.approx(expected_box(k, rotation), abs=tolerance), k
        assert face["score"] == 0.9
    summary = doc["summary"]
    assert summary["frames_with_faces"] == SAMPLES and summary["max_faces"] == 1
    assert summary["dropped"] == {"non-finite": 0, "zero-area": 0}
    assert summary["truncated_tail"] is None


@needs_ffmpeg
def test_output_header_records_model_settings_input_and_offset(tmp_path, clips, model):
    out = tmp_path / "faces.json"
    run(clips["plain"], out, model, settings=FaceSettings(sample_fps=Fraction(1, 2)))
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "aar.faces/1"
    header = doc["header"]
    assert header["model"] == {
        "path": str(model.resolve()),
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "size": model.stat().st_size,
    }
    assert header["settings"] == {
        "sample_fps": "1/2",
        "max_side": 640,
        "score": 0.6,
        "nms": 0.3,
        "top_k": 50,
    }
    st = clips["plain"].stat()
    assert header["input"] == {
        "path": str(clips["plain"]),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }
    assert header["sample_fps"] == "1/2" and header["source"]["fps"] == "10/1"
    assert header["offset"]["from"] == "none"
    assert [f["file_s"] for f in doc["frames"]] == [0.0, 2.0]  # 3 s at one frame per 2 s
    assert not checkpoint_path(out).exists() and not list(tmp_path.glob(".*.tmp"))


@needs_ffmpeg
def test_interrupted_then_resumed_is_byte_identical(tmp_path, clips, model):
    clean = tmp_path / "clean.json"
    run(clips["r90"], clean, model)
    out = tmp_path / "resumed.json"
    with pytest.raises(AnalyzeError) as error:
        run(clips["r90"], out, model, fail_at=3)
    assert error.value.type == "detection-failed" and error.value.exit_code == 1
    assert "--resume" in error.value.hint and not out.exists()
    _, records, _ = read_checkpoint(
        checkpoint_path(out), kind="aar.analyze.faces", what="frame", check=lambda r, p: None
    )
    assert [r["index"] for r in records] == [0, 1, 2]
    report, log = run(clips["r90"], out, model, resume=True)
    assert out.read_bytes() == clean.read_bytes()
    assert report["resume"] == {"resumed": True, "from_index": 3}
    assert len(log["frames"]) == SAMPLES - 3  # only frames 3.. were decoded and detected
    assert not checkpoint_path(out).exists()


@needs_ffmpeg
def test_checkpoint_refusals(tmp_path, clips, model):
    out = tmp_path / "faces.json"
    with pytest.raises(AnalyzeError):
        run(clips["plain"], out, model, fail_at=2)
    with pytest.raises(AnalyzeError) as exists:
        run(clips["plain"], out, model)
    assert exists.value.type == "checkpoint-exists"
    with pytest.raises(AnalyzeError) as mismatch:
        run(clips["plain"], out, model, resume=True, settings=FaceSettings(score=0.7))
    assert mismatch.value.type == "checkpoint-mismatch"
    model.write_bytes(b"another model")
    with pytest.raises(AnalyzeError) as other_model:
        run(clips["plain"], out, model, resume=True)
    assert other_model.value.type == "checkpoint-mismatch"
    ckpt = checkpoint_path(out)
    lines = ckpt.read_bytes().splitlines(keepends=True)
    ckpt.write_bytes(lines[0] + lines[2])  # frame 1 without frame 0
    with pytest.raises(AnalyzeError) as gap:
        run(clips["plain"], out, model, resume=True)
    assert gap.value.type == "checkpoint-corrupt" and "expected 0" in str(gap.value)


# ---------------------------------------------------------------- ffmpeg trouble


def fake_ffmpeg(tmp_path: Path, body: str) -> str:
    script = tmp_path / "ffmpeg"
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


@needs_ffmpeg
def test_a_short_final_read_is_reported_not_parsed(tmp_path, clips, model, monkeypatch):
    frame_bytes = W * H * 3
    body = f"head -c {2 * frame_bytes + 5} /dev/zero"
    monkeypatch.setenv("AAR_FFMPEG", fake_ffmpeg(tmp_path, body))
    out = tmp_path / "faces.json"
    report, log = run(clips["plain"], out, model)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["summary"]["frames"] == 2 and len(log["frames"]) == 2
    assert doc["summary"]["truncated_tail"] == {"bytes": 5}
    assert report["summary"]["truncated_tail"] == {"bytes": 5}


@needs_ffmpeg
def test_a_failing_ffmpeg_is_a_typed_error_with_its_stderr(tmp_path, clips, model, monkeypatch):
    body = 'echo "acme decoder: corrupt slice at frame 3" >&2; exit 1'
    monkeypatch.setenv("AAR_FFMPEG", fake_ffmpeg(tmp_path, body))
    with pytest.raises(AnalyzeError) as error:
        run(clips["plain"], tmp_path / "faces.json", model)
    assert error.value.type == "decode-failed" and error.value.exit_code == 1
    assert "corrupt slice at frame 3" in str(error.value)


@needs_ffmpeg
def test_a_stalled_ffmpeg_times_out(tmp_path, clips, model, monkeypatch):
    monkeypatch.setenv("AAR_FFMPEG", fake_ffmpeg(tmp_path, "sleep 30"))
    with pytest.raises(AnalyzeError) as error:
        run(clips["plain"], tmp_path / "f.json", model, settings=FaceSettings(timeout_s=1.0))
    assert error.value.type == "timeout"


# ---------------------------------------------------------------- pure parts


def test_normalise_clamps_drops_sorts_and_rounds():
    nan = float("nan")
    rows = np.array(
        [
            [10, 10, 20, 20, *([15, 15] * 5), 0.95],  # small
            [-20, 50, 60, 70, *([-5, 200] * 5), 0.80],  # big, clamped to the frame
            [nan, 10, 20, 20, *([0, 0] * 5), 0.99],  # non-finite
            [10, 10, 20, 20, *([0, 0] * 4), 0, float("inf"), 0.99],
            [150, 10, 30, 30, *([0, 0] * 5), 0.99],  # starts past the right edge
            [30, 30, 0, 20, *([0, 0] * 5), 0.99],  # no width
        ],
        dtype=np.float64,
    )
    faces, dropped = normalise(rows, 100, 100)
    assert dropped == {"non-finite": 2, "zero-area": 2}
    big, small = faces
    assert big["box"] == [0.0, 0.5, 0.4, 0.5]  # x -20..40 -> 0..0.4, y 50..120 -> 0.5..1
    assert big["landmarks"] == [[0.0, 1.0]] * 5
    assert small["box"] == [0.1, 0.1, 0.2, 0.2] and small["score"] == 0.95
    assert normalise(None, 100, 100) == ([], {})
    [rounded], _ = normalise(np.array([[1, 2, 3, 4, *([1] * 10), 0.123456]]), 3, 7)
    assert rounded["box"] == [0.3333, 0.2857, 0.6667, 0.5714] and rounded["score"] == 0.1235


@pytest.mark.parametrize(
    ("display", "max_side", "size"),
    [
        ((1920, 1080), 640, (640, 360)),
        ((1080, 1920), 640, (360, 640)),
        ((320, 240), 640, (320, 240)),  # never enlarged
        ((641, 361), 640, (640, 360)),  # rounded down to even
        ((101, 51), 640, (100, 50)),
        ((1440, 1080), 100, (100, 74)),
    ],
)
def test_frame_size(display, max_side, size):
    assert frame_size(display, max_side) == size


@pytest.mark.parametrize(
    ("rotation", "sar", "display"),
    [
        (0, Fraction(1), (160, 96)),
        (90, Fraction(1), (96, 160)),
        (270, Fraction(1), (96, 160)),
        (180, Fraction(1), (160, 96)),
        (0, Fraction(4, 3), (213, 96)),
        (90, Fraction(4, 3), (96, 213)),
    ],
)
def test_display_size_applies_aspect_then_rotation(rotation, sar, display):
    info = VideoInfo(W, H, rotation, sar, Fraction(10), 3.0)
    assert info.display_size == display


# ---------------------------------------------------------------- CLI, model and extras


class FakeYunet:
    calls: ClassVar[list[tuple[Any, ...]]] = []

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    @classmethod
    def create(cls, *args: Any) -> FakeYunet:
        cls.calls.append(args)
        return cls(args[2])

    def detect(self, frame: np.ndarray) -> tuple[int, np.ndarray | None]:
        assert frame.shape == (self.size[1], self.size[0], 3)
        return 1, np.array([[4, 4, 8, 8, *([8, 8] * 5), 0.77]], dtype=np.float32)


@pytest.fixture
def fake_cv2(monkeypatch: pytest.MonkeyPatch) -> type[FakeYunet]:
    module = types.ModuleType("cv2")
    FakeYunet.calls = []
    module.FaceDetectorYN = FakeYunet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", module)
    return FakeYunet


def cli(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any]]:
    code = main(["analyze", "faces", *(str(a) for a in argv)])
    return code, json.loads(capsys.readouterr().out)


@needs_ffmpeg
def test_cli_uses_cv2_yunet_with_the_documented_arguments(tmp_path, clips, model, fake_cv2, capsys):
    out = tmp_path / "faces.json"
    code, report = cli(
        capsys, clips["plain"], "--out", out, "--model", model, "--score", "0.5",
        "--nms", "0.4", "--top-k", "7", "--max-side", "80",
    )  # fmt: skip
    assert code == 0 and report["ok"]
    assert fake_cv2.calls == [(str(model.resolve()), "", (80, 48), 0.5, 0.4, 7)]
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["frames"][0]["faces"][0]["box"] == [0.05, 0.0833, 0.1, 0.1667]  # (4,4,8,8)/80x48


@needs_ffmpeg
def test_cli_takes_the_model_from_the_environment(
    tmp_path, clips, model, fake_cv2, capsys, monkeypatch
):
    monkeypatch.delenv("AAR_YUNET_MODEL", raising=False)
    code, body = cli(capsys, clips["plain"], "--out", tmp_path / "f.json")
    assert code == 2 and body["error"]["type"] == "model-missing"
    assert "AAR_YUNET_MODEL" in body["error"]["hint"]
    monkeypatch.setenv("AAR_YUNET_MODEL", str(model))
    code, report = cli(capsys, clips["plain"], "--out", tmp_path / "f.json")
    assert code == 0 and report["model"]["path"] == str(model.resolve())


def test_cli_errors_exit_2(tmp_path, model, capsys, monkeypatch):
    clip = write_wav(tmp_path / "a.wav", tone(1.0))
    code, body = cli(capsys, clip, "--out", tmp_path / "f.json", "--model", tmp_path)
    assert code == 2 and body["error"]["type"] == "model-missing"
    monkeypatch.setitem(sys.modules, "cv2", None)
    code, body = cli(capsys, clip, "--out", tmp_path / "f.json", "--model", model)
    assert code == 2 and body["error"]["type"] == "missing-extra"
    assert body["error"]["hint"] == "pip install 'ai-agent-resolve[faces]'"
    for bad in (["--sample-fps", "0"], ["--sample-fps", "x"], ["--max-side", "4"]):
        code, body = cli(capsys, clip, "--out", tmp_path / "f.json", "--model", model, *bad)
        assert code == 2 and body["error"]["type"] == "invalid-argument", bad


@needs_ffmpeg
def test_cli_audio_only_input_has_no_video(tmp_path, model, fake_cv2, capsys):
    clip = write_wav(tmp_path / "a.wav", tone(1.0))
    code, body = cli(capsys, clip, "--out", tmp_path / "f.json", "--model", model)
    assert code == 1 and body["error"]["type"] == "no-video"


# ---------------------------------------------------------------- live


LIVE_MODEL = os.environ.get("AAR_YUNET_MODEL", "")


@needs_ffmpeg
@pytest.mark.skipif(
    not (LIVE_MODEL.endswith(".onnx") and Path(LIVE_MODEL).is_file())
    or importlib.util.find_spec("cv2") is None,
    reason="set AAR_YUNET_MODEL to a YuNet .onnx file and install the faces extra",
)
def test_live_yunet_gives_a_valid_document(tmp_path, clips):
    out = tmp_path / "faces.json"
    report = analyse_faces(clips["plain"], out, model=LIVE_MODEL)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "aar.faces/1" and len(doc["frames"]) == SAMPLES
    assert report["summary"]["frames"] == SAMPLES  # a box is not a face; zero faces is fine
