"""Synthetic camera files for the S19a stub and recipe tests, made with ffmpeg lavfi.

A "show" is a QuickTime file with a test-pattern picture (testsrc2), a tmcd start timecode and
a sound track whose words are bursts of a three-partial voice at about -16 dBFS over a 120 Hz
room tone at -60 dBFS, so every gap between words is quiet for the producer's join test. The
transcript the tests use is :data:`WORDS` itself, so the two can never disagree. Nothing here
uses aar.media.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


def _tool(name: str, var: str) -> str | None:
    override = os.environ.get(var, "").strip()
    return shutil.which(override) if override else shutil.which(name)


FFMPEG = _tool("ffmpeg", "AAR_FFMPEG")
needs_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and _tool("ffprobe", "AAR_FFPROBE")),
    reason="ffmpeg/ffprobe not found on PATH or via AAR_FFMPEG/AAR_FFPROBE",
)

VOICE = "(0.5*sin(2*PI*220*t)+0.3*sin(2*PI*337*t)+0.2*sin(2*PI*513*t))"
ROOM = "0.001*sin(2*PI*120*t)"

#: (text, start_s, end_s): three sentences with a one-second pause between them.
WORDS: tuple[tuple[str, float, float], ...] = (
    ("So", 1.0, 1.3),
    ("I", 1.4, 1.6),
    ("went", 1.7, 2.0),
    ("to", 2.1, 2.3),
    ("the", 2.4, 2.6),
    ("store", 2.7, 3.2),
    ("yesterday", 3.3, 4.0),
    ("and", 5.0, 5.3),
    ("nobody", 5.4, 5.9),
    ("was", 6.0, 6.3),
    ("there", 6.4, 7.0),
    ("at", 7.1, 7.3),
    ("all", 7.4, 8.0),
    ("Then", 9.0, 9.3),
    ("my", 9.4, 9.6),
    ("wife", 9.7, 10.0),
    ("laughed", 10.1, 10.6),
    ("loudly", 10.7, 11.0),
)
SECONDS = 12.0
#: Two keep ranges with a human cut in the second pause (both edges in quiet).
KEEP = {"ranges": [{"start_s": 0.9, "end_s": 4.1}, {"start_s": 8.9, "end_s": 11.1}]}


def _audio_expr(words: tuple[tuple[str, float, float], ...], gain: float) -> str:
    bursts = "+".join(f"between(t,{a},{b})" for _, a, b in words)
    return f"{ROOM}+{gain}*({bursts})*{VOICE}"


def make_show(
    path: Path,
    *,
    seconds: float = SECONDS,
    rate: str = "30000/1001",
    size: str = "320x180",
    timecode: str | None = "01:00:00:00",
    audio: bool = True,
    words: tuple[tuple[str, float, float], ...] = WORDS,
    gain: float = 0.3,
    drop_frame_at: int | None = None,
) -> Path:
    """Write a synthetic camera file (module docstring); return its path. ``drop_frame_at``
    removes one video frame (a variable frame rate)."""
    assert FFMPEG is not None
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y"]
    cmd += ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}"]
    if audio:
        expr = _audio_expr(words, gain)
        cmd += ["-f", "lavfi", "-i", f"aevalsrc='{expr}|{expr}':s=48000:d={seconds}"]
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "pcm_s16le"]
    if drop_frame_at is not None:
        cmd += ["-vf", f"select='not(eq(n\\,{drop_frame_at}))'", "-fps_mode", "passthrough"]
    if timecode is not None:
        cmd += ["-timecode", timecode]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "30"]
    cmd += ["-f", "mov", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def transcript(recording: str, words: tuple[tuple[str, float, float], ...] = WORDS) -> dict:
    return {
        "recording": recording,
        "words": [{"text": t, "start_s": a, "end_s": b} for t, a, b in words],
    }


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path
