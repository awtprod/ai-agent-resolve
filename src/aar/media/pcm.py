"""Decoded audio in memory and pure analysis on it (no ffmpeg here, stdlib only).

Levels are RMS in dBFS with the ffmpeg astats convention: ``20*log10(rms)`` of samples in
[-1, 1], so a full-scale sine reads -3.01 dBFS and digital silence reads ``-inf``.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from itertools import accumulate


@dataclass(frozen=True)
class Pcm:
    """Interleaved 32-bit float samples starting ``start_s`` seconds into the source."""

    rate: int
    channels: int
    start_s: float
    samples: array[float] = field(repr=False)

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration_s(self) -> float:
        return self.frames / self.rate

    def channel(self, index: int) -> array[float]:
        usable = self.frames * self.channels
        return self.samples[index : usable : self.channels]


def db(power: float) -> float:
    """Mean-square power to dB (``-inf`` for 0)."""
    return 10.0 * math.log10(power) if power > 0 else -math.inf


@dataclass(frozen=True)
class QuietRun:
    """A stretch where every analysis window inside it is below the threshold."""

    start_s: float
    end_s: float
    level_dbfs: float  # the loudest window that makes up the run

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class QuietScan:
    threshold_dbfs: float
    window_s: float
    scanned_s: float
    runs: list[QuietRun]
    quietest_dbfs: float | None  # the quietest window anywhere; None if too little audio

    @property
    def longest(self) -> QuietRun | None:
        return max(self.runs, key=lambda r: (r.duration_s, -r.start_s), default=None)


def quiet_runs(pcm: Pcm, *, threshold_dbfs: float, window_s: float = 0.010) -> QuietScan:
    """Find the quiet runs in ``pcm``.

    A window is ``window_s`` long (rounded to whole samples) and may start at any sample. Its
    level is the RMS of the loudest channel. A window is quiet when that level is strictly
    below ``threshold_dbfs``. A quiet run is a maximal stretch covered by overlapping or
    touching quiet windows, so its length is exact to the sample: a 60 ms gap of digital
    silence between loud sounds measures 60 ms plus at most a few samples where the
    neighbouring sound passes through zero.
    """
    width = max(1, round(window_s * pcm.rate))
    frames = pcm.frames
    if frames < width:
        return QuietScan(threshold_dbfs, width / pcm.rate, frames / pcm.rate, [], None)
    prefixes = [
        list(accumulate((x * x for x in pcm.channel(c)), initial=0.0)) for c in range(pcm.channels)
    ]
    limit = 10.0 ** (threshold_dbfs / 10.0) * width  # energy of a window at the threshold
    runs: list[QuietRun] = []
    quietest = math.inf
    run_start = -1
    run_end = -1
    run_peak = 0.0

    def close() -> None:
        runs.append(
            QuietRun(
                start_s=pcm.start_s + run_start / pcm.rate,
                end_s=pcm.start_s + run_end / pcm.rate,
                level_dbfs=db(run_peak / width),
            )
        )

    for i in range(frames - width + 1):
        energy = max(p[i + width] - p[i] for p in prefixes)
        energy = max(energy, 0.0)  # prefix rounding can leave -1e-17 on silence
        quietest = min(quietest, energy)
        if energy < limit:
            if run_start >= 0 and i <= run_end:
                run_end = i + width
                run_peak = max(run_peak, energy)
            else:
                if run_start >= 0:
                    close()
                run_start, run_end, run_peak = i, i + width, energy
    if run_start >= 0:
        close()
    return QuietScan(
        threshold_dbfs=threshold_dbfs,
        window_s=width / pcm.rate,
        scanned_s=frames / pcm.rate,
        runs=runs,
        quietest_dbfs=db(quietest / width),
    )
