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

#: A voice: three partials plus seeded noise (``random(0)`` is deterministic per sample), so two
#: cameras' copies of one take have a single clear correlation peak for ``aar analyze sync``.
VOICE = "(0.4*sin(2*PI*220*t)+0.25*sin(2*PI*337*t)+0.15*sin(2*PI*513*t)+0.4*(2*random(0)-1))"
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


def _audio_expr(words: tuple[tuple[str, float, float], ...], gain: float, room: bool) -> str:
    bursts = "+".join(f"between(t,{a},{b})" for _, a, b in words)
    return f"{ROOM if room else 0}+{gain}*({bursts})*{VOICE}"


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
    room: bool = True,
    drop_frame_at: int | None = None,
) -> Path:
    """Write a synthetic camera file (module docstring); return its path. ``drop_frame_at``
    removes one video frame (a variable frame rate); ``gain=0, room=False`` is digital silence."""
    assert FFMPEG is not None
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y"]
    cmd += ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={seconds}"]
    if audio:
        expr = _audio_expr(words, gain, room)
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


def delayed_copy(src: Path, dest: Path, *, delay_s: float, gain: float = 0.5) -> Path:
    """``src`` as a second camera started ``delay_s`` later: the same picture, the same sound
    shifted by ``delay_s`` (quieter), the same length."""
    assert FFMPEG is not None
    ms = round(delay_s * 1000)
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(src)]
    cmd += ["-af", f"adelay={ms}|{ms},volume={gain},atrim=0:{SECONDS}", "-c:v", "copy"]
    cmd += ["-c:a", "pcm_s16le", "-timecode", "01:00:00:15", "-f", "mov", str(dest)]
    subprocess.run(cmd, check=True, timeout=120)
    return dest


def frame_count(path: Path) -> int:
    """The video stream's packet count (one per frame for these files), by ffprobe."""
    probe = _tool("ffprobe", "AAR_FFPROBE")
    assert probe is not None
    run = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    return int(run.stdout.strip().strip(","))


def timecode_label(frames: int, nominal: int = 30) -> str:
    """A non-drop ``HH:MM:SS:FF`` label of ``frames`` from 00:00:00:00 at ``nominal`` fps,
    wrapped at midnight."""
    frames %= 86400 * nominal
    seconds, ff = divmod(frames, nominal)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}:{ff:02d}"


def make_parts(
    folder: Path,
    gaps: list[int],
    *,
    name: str = "cam-a",
    start_frames: int = 108000,
    seconds: float = SECONDS,
    **kw: Any,
) -> list[tuple[Path, int]]:
    """One camera's consecutive files (S24a): ``len(gaps) + 1`` synthetic shows (each with
    :data:`WORDS` in its own clock), ``<name>_C001.mov`` first, starting at timecode frame
    ``start_frames`` (01:00:00:00 at 29.97 by default); each next file's start timecode is where
    the previous one ends plus ``gaps[n]`` frames (0: frame-contiguous; a negative gap
    overlaps). Returns ``(path, start timecode frame)`` per file (wrapped at midnight)."""
    out: list[tuple[Path, int]] = []
    at = start_frames
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(len(gaps) + 1):
        path = make_show(
            folder / f"{name}_C{n + 1:03d}.mov", seconds=seconds, timecode=timecode_label(at), **kw
        )
        out.append((path, at % (86400 * 30)))
        if n < len(gaps):
            at += frame_count(path) + gaps[n]
    return out


def transcript(recording: str, words: tuple[tuple[str, float, float], ...] = WORDS) -> dict:
    return {
        "recording": recording,
        "words": [{"text": t, "start_s": a, "end_s": b} for t, a, b in words],
    }


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path
