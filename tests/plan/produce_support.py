"""Synthetic shows for the plan-producer tests, rendered with ffmpeg into pytest's tmp_path.

The JSON inputs (stub, transcript, keep list, schedule, laughs, sync report, face track) are
committed under ``fixtures/produce``. The audio is generated from the committed transcript and
laugh map, so the two can never disagree: every word is a burst of one of three three-partial
"voices" at -20 dBFS peak, every laugh a two-partial burst at -22 dBFS, over a 120 Hz room tone
at -60 dBFS
(camera B: 6 dB quieter voice, room at -50 dBFS). ``AUDIO`` lists the deliberate differences
between what the transcript says and what is on the audio: a mumble too quiet to break a pause,
a word whose sound outlasts Whisper's end and one that starts before Whisper's start (the R6
join traps).

Nothing here uses aar.media, so a bug there cannot hide in the fixtures.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "produce"
RATE = 16000
FPS = 30000 / 1001


def _tool(name: str, var: str) -> str | None:
    override = os.environ.get(var, "").strip()
    return shutil.which(override) if override else shutil.which(name)


FFMPEG = _tool("ffmpeg", "AAR_FFMPEG")
needs_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and _tool("ffprobe", "AAR_FFPROBE")),
    reason="ffmpeg/ffprobe not found on PATH or via AAR_FFMPEG/AAR_FFPROBE",
)

#: Three voices, word i speaking with VOICES[i % 3], so the bursts do not repeat one period and
#: ``aar analyze sync`` finds a single clear correlation peak between the cameras.
VOICES = (
    "(0.5*sin(2*PI*220*t)+0.3*sin(2*PI*337*t)+0.2*sin(2*PI*513*t))",
    "(0.5*sin(2*PI*181*t)+0.3*sin(2*PI*419*t)+0.2*sin(2*PI*647*t))",
    "(0.5*sin(2*PI*263*t)+0.3*sin(2*PI*389*t)+0.2*sin(2*PI*571*t))",
)
LAUGH = "(0.5*sin(2*PI*600*t)+0.5*sin(2*PI*910*t))"

#: (show, word text, transcript start) -> what the audio does instead.
AUDIO: dict[tuple[str, str, float], dict[str, float]] = {
    # A mumble below the silence threshold: the quiet run goes on through it, but it is a
    # transcribed word, so no trim may swallow it.
    ("longform", "um", 8.0): {"peak_db": -46.0},
    # Whisper ended "later" 150 ms early: its sound runs on to 19.45 s, past the second keep
    # range's in point (19.40 s), so that in point must move later (inward) into quiet.
    ("longform", "later", 18.8): {"end_s": 19.45},
    # Whisper put "and" 80 ms late: its sound starts at 19.92 s.
    ("longform", "and", 20.0): {"start_s": 19.92},
}

SHOWS: dict[str, dict[str, Any]] = {
    "longform": {
        "seconds": 40.0,
        "media": {
            "cam-a": {"file": "longform_a.wav", "origin_s": 0.0, "gain_db": 0.0, "room_db": -60.0},
            "cam-b": {
                "file": "longform_b.wav",
                "origin_s": 60 * 1001 / 30000,
                "gain_db": -6.0,
                "room_db": -50.0,
            },
        },
    },
    "short": {
        "seconds": 30.0,
        "media": {
            "master": {
                "file": "short_master.mov",
                "origin_s": 0.0,
                "gain_db": 0.0,
                "room_db": -60.0,
                "video": True,
            }
        },
    },
}


def amp(dbfs: float) -> str:
    return repr(10 ** (dbfs / 20))


def fixture(show: str, part: str) -> Path:
    return FIXTURES / f"{show}.{part}.json"


def load(show: str, part: str) -> Any:
    return json.loads(fixture(show, part).read_text(encoding="utf-8"))


def events(show: str) -> list[tuple[float, float, float, str]]:
    """(start, end, peak dBFS, timbre) in recording seconds, from the committed inputs."""
    out = []
    for index, word in enumerate(load(show, "transcript")["words"]):
        change = AUDIO.get((show, word["text"], word["start_s"]), {})
        start = change.get("start_s", word["start_s"])
        end = change.get("end_s", word["end_s"])
        out.append((start, end, change.get("peak_db", -20.0), VOICES[index % 3]))
    for laugh in load(show, "laughs")["laughs"]:
        out.append((laugh["start_s"], laugh["end_s"], -22.0, LAUGH))
    return sorted(out)


def _expr(show: str, origin_s: float, gain_db: float, room_db: float) -> str:
    """Room tone plus each timbre times the sum of its bursts' gated amplitudes."""
    gates: dict[str, list[str]] = {voice: [] for voice in (*VOICES, LAUGH)}
    for start, end, peak, timbre in events(show):
        a, b = start - origin_s, end - origin_s
        if b > 0:
            gates[timbre].append(f"between(t\\,{a:.6f}\\,{b:.6f})*{amp(peak + gain_db)}")
    terms = [f"{amp(room_db)}*sin(2*PI*120*t)"]
    terms += [f"({'+'.join(g)})*{timbre}" for timbre, g in gates.items() if g]
    return "+".join(terms)


def ffmpeg(*args: str) -> None:
    assert FFMPEG is not None
    subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", *args],
        check=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


def render(show: str, folder: Path) -> dict[str, Path]:
    """Write the show's media into ``folder``; returns source id -> path."""
    spec = SHOWS[show]
    seconds = spec["seconds"]
    paths: dict[str, Path] = {}
    for source_id, media in spec["media"].items():
        path = folder / media["file"]
        expr = _expr(show, media["origin_s"], media["gain_db"], media["room_db"])
        audio = ["-f", "lavfi", "-i", f"aevalsrc={expr}:s={RATE}:d={seconds}"]
        if media.get("video"):
            frames = round(seconds * FPS)
            ffmpeg(
                "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30000/1001", *audio,
                "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-t", f"{seconds:.6f}", str(path),
            )  # fmt: skip
        else:
            ffmpeg(*audio, "-c:a", "pcm_s16le", str(path))
        paths[source_id] = path
    return paths


def stage(show: str, folder: Path, parts: tuple[str, ...]) -> dict[str, Path]:
    """Render the media and copy the named JSON inputs into ``folder``."""
    render(show, folder)
    out = {}
    for part in parts:
        target = folder / f"{show}.{part}.json"
        shutil.copyfile(fixture(show, part), target)
        out[part] = target
    return out
