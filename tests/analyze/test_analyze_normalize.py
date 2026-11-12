"""S19d: ``--normalize`` levels far-field audio before Whisper without moving it in time.

The synthetic "show" is a stereo 48 kHz WAV, as a camera records it: a loud first part (peaks
-12 dBFS) and a quiet second part (peaks -40 dBFS, the level of room microphones), each a train
of sharp-edged tone bursts at known times. The transcriber is an injected fake that "hears" a
word wherever the audio it is handed passes an absolute level, which is how quiet speech goes
missing in Whisper's VAD: the raw run hears only the loud part, the normalised run hears both,
and every word heard in both lands at the same file time within ``TOLERANCE_S``.
"""

from __future__ import annotations

import json
import sys
import types
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from analyze_support import needs_ffmpeg

from aar.analyze.audio import NORMALIZE_FILTER, read_mono
from aar.cli import main

pytestmark = needs_ffmpeg

SRC_RATE = 48000
RATE = 16000
SECONDS = 60.0
#: Loud bursts (start, end, peak dBFS) and quiet ones: 300 ms "words" with sharp edges.
LOUD = [(1.0 + 2.0 * i, 1.3 + 2.0 * i, -12.0) for i in range(8)]  # 1.0 .. 15.3 s
QUIET = [(24.0 + 2.5 * i, 24.3 + 2.5 * i, -40.0) for i in range(12)]  # 24.0 .. 51.8 s
#: The fake hears audio whose 10 ms peak passes this (-26 dBFS): the loud part only, raw.
HEAR = 10 ** (-26 / 20)
FRAME_S = 0.010
#: Every word heard both ways lands within this of itself (two 10 ms detection frames).
TOLERANCE_S = 0.020


def write_show(path: Path) -> Path:
    t = np.arange(round(SECONDS * SRC_RATE)) / SRC_RATE
    left = np.zeros_like(t)
    for start, end, peak in LOUD + QUIET:
        gate = (t >= start) & (t < end)
        left[gate] = 10 ** (peak / 20) * np.sin(2 * np.pi * 440.0 * t[gate])
    rng = np.random.default_rng(7)
    hiss = 10 ** (-70 / 20) * rng.standard_normal(len(t))  # a room is never digital silence
    stereo = np.stack([left + hiss, left + hiss], axis=1)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SRC_RATE)
        out.writeframes((np.clip(stereo, -1, 1) * 32767).round().astype("<i2").tobytes())
    return path


def heard(audio: np.ndarray, base_s: float = 0.0) -> list[tuple[float, float]]:
    """(start, end) file seconds of every run of 10 ms frames whose peak passes ``HEAR``."""
    step = round(FRAME_S * RATE)
    n = len(audio) // step
    loud = np.abs(audio[: n * step]).reshape(n, step).max(axis=1) > HEAR
    runs, start = [], None
    for i, on in enumerate([*loud, False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((base_s + start * FRAME_S, base_s + i * FRAME_S))
            start = None
    return runs


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 0.9


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]
    avg_logprob: float = -0.3
    compression_ratio: float = 1.4
    no_speech_prob: float = 0.05
    tokens: list[int] = field(default_factory=list)


class Ears:
    """A fake ``faster_whisper`` whose model transcribes what the audio holds."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        ears = self

        class WhisperModel:
            def __init__(self, path: str, **kwargs: Any) -> None:
                pass

            def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
                ears.calls.append(audio)
                segments = [Segment(a, b, " word", [Word(a, b, " word")]) for a, b in heard(audio)]
                return iter(segments), types.SimpleNamespace(language="en")

        self.module = types.ModuleType("faster_whisper")
        self.module.WhisperModel = WhisperModel  # type: ignore[attr-defined]


@pytest.fixture
def ears(monkeypatch: pytest.MonkeyPatch) -> Ears:
    fake = Ears()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake.module)
    return fake


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    (root / "medium.en").mkdir(parents=True)
    (root / "medium.en" / "model.bin").write_bytes(b"")
    monkeypatch.setenv("AAR_MODEL_DIR", str(root))
    return root


@pytest.fixture(scope="module")
def show(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_show(tmp_path_factory.mktemp("show") / "camera.wav")


def transcribe(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any]]:
    code = main(["analyze", "transcript", *(str(a) for a in argv)])
    return code, json.loads(capsys.readouterr().out)


def starts(path: Path) -> list[float]:
    return [w["start_s"] for w in json.loads(path.read_text(encoding="utf-8"))["words"]]


# ---------------------------------------------------------------- the decode


def test_normalize_changes_level_not_time(show: Path) -> None:
    raw = read_mono(show, rate=RATE, max_duration_s=120.0, timeout_s=60.0)
    norm = read_mono(show, rate=RATE, max_duration_s=120.0, timeout_s=60.0, normalize=True)
    # Not one sample more or less: nothing is delayed, padded or stretched.
    assert len(norm.samples) == len(raw.samples) == round(SECONDS * RATE)
    # Each burst's onset is at the same sample, to the 1 ms the sharp edge allows.
    for start, _end, _peak in LOUD + QUIET:
        a, b = round((start - 0.05) * RATE), round((start + 0.05) * RATE)
        threshold = 0.5 * np.abs(raw.samples[a:b]).max()
        raw_on = np.flatnonzero(np.abs(raw.samples[a:b]) > threshold)[0]
        norm_threshold = 0.5 * np.abs(norm.samples[a:b]).max()
        norm_on = np.flatnonzero(np.abs(norm.samples[a:b]) > norm_threshold)[0]
        assert abs(int(raw_on) - int(norm_on)) <= RATE // 1000, start
    # The quiet part comes up by well over 15 dB; the loud part is not pushed past full scale.
    q = slice(round(24.0 * RATE), round(52.0 * RATE))
    gain_db = 20 * np.log10(np.abs(norm.samples[q]).max() / np.abs(raw.samples[q]).max())
    assert gain_db > 15.0
    assert np.abs(norm.samples).max() < 0.95


def test_the_filter_is_the_documented_one() -> None:
    assert NORMALIZE_FILTER == "dynaudnorm=f=250:g=15:p=0.9:m=30"


# ---------------------------------------------------------------- the transcript


def test_normalized_transcript_hears_the_quiet_part_at_the_same_times(
    tmp_path: Path, show: Path, ears: Ears, models: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_out, norm_out = tmp_path / "raw.json", tmp_path / "norm.json"
    code, raw_report = transcribe(capsys, show, "--out", raw_out, "--wordless-min-s", "5")
    assert code == 0
    code, norm_report = transcribe(
        capsys, show, "--out", norm_out, "--normalize", "--wordless-min-s", "5"
    )
    assert code == 0
    raw_starts, norm_starts = starts(raw_out), starts(norm_out)
    # Raw, the fake hears only the loud bursts: the quiet half is a hole.
    assert raw_starts == pytest.approx([s for s, _, _ in LOUD], abs=TOLERANCE_S)
    # Normalised, it hears every burst, each where it is in the file.
    assert norm_starts == pytest.approx([s for s, _, _ in LOUD + QUIET], abs=TOLERANCE_S)
    # The words heard both ways land on themselves (file time unchanged by the filter).
    both = norm_starts[: len(raw_starts)]
    assert np.max(np.abs(np.array(both) - np.array(raw_starts))) <= TOLERANCE_S
    # The setting is recorded in the report.
    assert raw_report["settings"]["normalize"] is False and raw_report["audio_filter"] is None
    assert norm_report["settings"]["normalize"] is True
    assert norm_report["audio_filter"] == NORMALIZE_FILTER
    # The raw run's hole is reported, at the source's own (raw) level, not the normalised one.
    [hole] = [s for s in raw_report["wordless"]["stretches"] if s["start_s"] > 15.0]
    assert hole["start_s"] == pytest.approx(15.3, abs=TOLERANCE_S)
    assert hole["end_s"] == pytest.approx(SECONDS, abs=1e-6)
    # 12 bursts of 300 ms at -43 dBFS RMS per channel over 44.7 s: about -51 dBFS (ffmpeg's
    # stereo-to-mono mix adds up to +3 dB for identical channels). Normalised it would be ~-20.
    assert -54.0 < hole["level_dbfs"] < -47.0
    # The normalised run has no such hole; its stretches' levels are still the raw ones.
    norm_stretches = norm_report["wordless"]["stretches"]
    assert all(s["end_s"] - s["start_s"] < 24.0 for s in norm_stretches)
    for stretch in norm_stretches:
        a, b = stretch["file_start_s"], stretch["file_end_s"]
        raw_same = [s for s in raw_report["wordless"]["stretches"] if s["file_start_s"] == a]
        if raw_same and raw_same[0]["file_end_s"] == b:
            assert stretch["level_dbfs"] == raw_same[0]["level_dbfs"]
    tail = norm_stretches[-1]
    assert tail["file_end_s"] == pytest.approx(SECONDS) and tail["level_dbfs"] < -60.0
