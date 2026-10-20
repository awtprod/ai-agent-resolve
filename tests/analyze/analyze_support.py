"""Synthetic audio for the analyze tests: speech-like noise bursts in numpy, written as 16-bit
mono WAV with the stdlib ``wave`` module. Every clip is at most 60 s at 16 kHz (about 2 MB).

``Source`` is one long "event" sampled at 48 kHz (band 150-3500 Hz, so 14x oversampled);
``take`` reads a recorder's view of it at any start time, gain, clock drift and polarity by
linear interpolation, which is accurate to well under a microsecond at that oversampling.
"""

from __future__ import annotations

import os
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

SRC_RATE = 48000
RATE = 16000


def _tool(name: str, var: str) -> str | None:
    override = os.environ.get(var, "").strip()
    return shutil.which(override) if override else shutil.which(name)


FFMPEG = _tool("ffmpeg", "AAR_FFMPEG")
FFPROBE = _tool("ffprobe", "AAR_FFPROBE")
needs_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not found on PATH or via AAR_FFMPEG"
)


def band_noise(n: int, rate: int, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / rate)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, n)


class Source:
    """A speech-like event: band-limited noise in syllable-sized bursts with gaps."""

    def __init__(self, seconds: float = 75.0, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        n = int(seconds * SRC_RATE)
        noise = band_noise(n, SRC_RATE, rng, 150.0, 3500.0)
        env = np.zeros(n)
        t = 0
        while t < n:
            on = int(rng.uniform(0.08, 0.35) * SRC_RATE)
            off = int(rng.uniform(0.03, 0.25) * SRC_RATE)
            m = max(0, min(on, n - t))
            env[t : t + m] = (np.hanning(on) * rng.uniform(0.3, 1.0))[:m]
            t += on + off
        signal = noise * env
        self.samples = signal / np.abs(signal).max() * 0.5
        self.seconds = seconds

    def take(
        self,
        start_s: float,
        seconds: float,
        *,
        rate: int = RATE,
        gain: float = 1.0,
        drift_ppm: float = 0.0,
    ) -> np.ndarray:
        """What a recorder whose sample 0 is at event time ``start_s`` captured.

        With ``drift_ppm`` its clock runs slow: its sample n is at event time
        ``start_s + n / rate * (1 + drift_ppm * 1e-6)``.
        """
        t = start_s + np.arange(round(seconds * rate)) / rate * (1 + drift_ppm * 1e-6)
        assert t[0] >= 0 and t[-1] * SRC_RATE < len(self.samples) - 1, "take outside the source"
        return gain * np.interp(t * SRC_RATE, np.arange(len(self.samples)), self.samples)


def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((data * 32767).round().astype("<i2").tobytes())
    return path


def tone(seconds: float, freq: float = 1000.0, rate: int = RATE, peak: float = 0.5) -> np.ndarray:
    return peak * np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate)


def lowpass(samples: np.ndarray, corner_hz: float, rate: int = RATE) -> np.ndarray:
    """A gentle 4th-order-like low-pass (|H| = 1 / sqrt(1 + (f/fc)^8)), zero phase."""
    spec = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), 1 / rate)
    spec /= np.sqrt(1 + (freqs / corner_hz) ** 8)
    return np.fft.irfft(spec, len(samples))
