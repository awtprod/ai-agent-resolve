"""Fixture media for the S13b tests, generated with ffmpeg lavfi into pytest's tmp_path.

Every file is 0.5-3 s and tiny. Nothing here uses aar.media, so a bug there cannot hide in
the fixtures. Levels are exact: a stereo 1 kHz sine with peak P dBFS measures P LUFS and a
P dBTP true peak in ebur128 (checked in test_media_measure).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _tool(name: str, var: str) -> str | None:
    override = os.environ.get(var, "").strip()
    return shutil.which(override) if override else shutil.which(name)


FFMPEG = _tool("ffmpeg", "AAR_FFMPEG")
FFPROBE = _tool("ffprobe", "AAR_FFPROBE")
needs_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE),
    reason="ffmpeg/ffprobe not found on PATH or via AAR_FFMPEG/AAR_FFPROBE; media tests need them",
)
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="fake tools are sh scripts")

RATE = 48000


def amp(dbfs: float) -> str:
    """Linear amplitude for a peak level, as an ffmpeg expression literal."""
    return repr(10 ** (dbfs / 20))


def ffmpeg(*args: str) -> None:
    assert FFMPEG is not None
    subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", *args],
        check=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


def aeval(
    path: Path,
    expr: str,
    *,
    seconds: float,
    channels: int = 2,
    rate: int = RATE,
    codec: str = "pcm_f32le",
) -> Path:
    """Write ``expr`` (an aevalsrc expression of t and n) to every channel."""
    exprs = "|".join([expr] * channels)
    ffmpeg("-f", "lavfi", "-i", f"aevalsrc={exprs}:s={rate}:d={seconds}", "-c:a", codec, str(path))
    return path


def sine(path: Path, peak_dbfs: float, *, seconds: float = 2.0, freq: int = 1000, **kw: Any) -> Path:
    return aeval(path, f"{amp(peak_dbfs)}*sin(2*PI*{freq}*t)", seconds=seconds, **kw)


def gapped(
    path: Path,
    *,
    before_s: float,
    gap_s: float,
    after_s: float,
    peak_dbfs: float = -6.0,
    gap_dbfs: float | None = None,
    freq: int = 1000,
    **kw: Any,
) -> Path:
    """Tone, then a gap (digital silence, or a tone at ``gap_dbfs``), then tone again.

    Sample exact: the gap is samples n with before_s <= n/rate < before_s + gap_s.
    """
    end = before_s + gap_s
    loud = f"{amp(peak_dbfs)}*sin(2*PI*{freq}*t)"
    inside = f"between(t\\,{before_s}\\,{end})*lt(t\\,{end})"
    quiet = "0" if gap_dbfs is None else f"{amp(gap_dbfs)}*sin(2*PI*{freq}*t)"
    expr = f"if({inside}\\,{quiet}\\,{loud})"
    return aeval(path, expr, seconds=before_s + gap_s + after_s, **kw)


def square(path: Path, *, seconds: float = 1.0, **kw: Any) -> Path:
    """A full-scale 100 Hz square wave: every sample is clipped."""
    return aeval(path, "if(lt(mod(t*100\\,1)\\,0.5)\\,1\\,-1)", seconds=seconds, codec="pcm_s16le", **kw)


def render(
    path: Path,
    *,
    frames: int,
    fps: str = "30000/1001",
    audio: str | None = None,
    channels: int = 2,
    size: str = "160x90",
    faststart: bool = True,
) -> Path:
    """An H.264/AAC mp4 with exactly ``frames`` video frames and audio of the same length.

    ``audio`` is an aevalsrc expression (default: a -20 dBFS 1 kHz sine); None-like "none"
    leaves the audio out.
    """
    num, _, den = fps.partition("/")
    seconds = frames * int(den or 1) / int(num)
    args = ["-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}"]
    if audio != "none":
        expr = audio or f"{amp(-20)}*sin(2*PI*1000*t)"
        exprs = "|".join([expr] * channels)
        args += ["-f", "lavfi", "-i", f"aevalsrc={exprs}:s={RATE}:d={seconds}"]
    args += ["-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt"]
    args += ["yuv420p", "-g", "15"]
    if audio != "none":
        args += ["-c:a", "aac", "-b:a", "96k", "-t", f"{seconds:.6f}"]
    if faststart:
        args += ["-movflags", "+faststart"]
    ffmpeg(*args, str(path))
    return path


def truncate(src: Path, dst: Path, fraction: float = 0.6) -> Path:
    data = src.read_bytes()
    dst.write_bytes(data[: int(len(data) * fraction)])
    return dst


def fake_tool(tmp_path: Path, name: str, body: str) -> Path:
    """An executable sh script standing in for ffmpeg/ffprobe."""
    script = tmp_path / name
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def plan_dict(
    sources: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    fps: str = "50",
    kind: str = "short",
    captions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "project": "Test Show",
        "timeline": "QA fixture",
        "kind": kind,
        "fps": fps,
        "resolution": {"width": 160, "height": 90},
        "profile": "ccn",
        "sources": sources,
        "segments": segments,
        "captions": captions or [],
    }


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
