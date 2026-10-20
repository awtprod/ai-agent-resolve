"""Decode audio straight into a numpy array: mono, 32-bit float, at a chosen rate.

``aar.media.read_pcm`` returns ``array('f')`` and is meant for the short QA windows. The
analysis workers read up to hours of audio, so this reader has ffmpeg write ``f32le`` into a
preallocated numpy buffer chunk by chunk (through ``aar.media.tools.run_tool``'s stdout sink,
with its timeout and process-group kill). Nothing is copied through Python floats or a second
bytes buffer, and the buffer is sized from a duration cap, so the output is bounded: ffmpeg is
told to stop at the cap (``-t``) and the sink refuses anything past it.

Memory: 4 bytes per sample, e.g. 115 MB for an hour at 8 kHz. The buffer is allocated for the
whole cap but only the pages ffmpeg fills become resident.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from aar.analyze.errors import need_numpy
from aar.media.tools import OutputTooLarge, input_url, require_regular_file, run_tool

_BASE = ["-hide_banner", "-nostdin", "-nostats", "-v", "error"]


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


@dataclass(frozen=True)
class Mono:
    """Mono float32 samples that start ``start_s`` seconds into the file."""

    samples: Any  # numpy.ndarray, float32, 1-D
    rate: int
    start_s: float
    non_finite: int  # NaN or infinite samples in the decode, replaced by 0

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.rate


class _Sink:
    """Copies ffmpeg's f32le stdout into a preallocated float32 buffer."""

    def __init__(self, np: Any, capacity: int) -> None:
        self.np = np
        self.buf = np.empty(capacity, dtype=np.float32)
        self.n = 0
        self.rest = b""
        self.non_finite = 0

    def feed(self, chunk: bytes) -> None:
        data = self.rest + chunk if self.rest else chunk
        usable = len(data) - len(data) % 4
        self.rest = data[usable:]
        count = usable // 4
        if self.n + count > len(self.buf):
            raise OutputTooLarge(f"ffmpeg wrote more than {len(self.buf)} samples", tool="ffmpeg")
        view = self.buf[self.n : self.n + count]
        view[:] = self.np.frombuffer(data, dtype="<f4", count=count)
        bad = ~self.np.isfinite(view)  # per chunk, so the check never holds a file-sized mask
        if bad.any():
            self.non_finite += int(bad.sum())
            view[bad] = 0.0
        self.n += count


def read_mono(
    path: str | os.PathLike[str],
    *,
    rate: int,
    max_duration_s: float,
    timeout_s: float,
    start_s: float = 0.0,
    duration_s: float | None = None,
    band_hz: tuple[float, float] | None = None,
) -> Mono:
    """Decode the first audio stream from ``start_s`` for ``duration_s`` (default: to the end),
    reading at most ``max_duration_s``. Channels are averaged to mono and resampled to ``rate``;
    ``band_hz`` adds a high-pass and a low-pass at those corner frequencies.

    Raises ``MediaError`` subclasses (missing file, missing ffmpeg, timeout, decode failure) and
    ``AnalyzeError("missing-extra")`` without numpy.
    """
    np = need_numpy()
    p = require_regular_file(path, tool="ffmpeg")
    if rate <= 0:
        raise ValueError("rate must be positive")
    if not (math.isfinite(max_duration_s) and max_duration_s > 0):
        raise ValueError("max_duration_s must be a finite number of seconds above 0")
    limit = max_duration_s if duration_s is None else min(max(duration_s, 0.0), max_duration_s)
    start_s = max(0.0, start_s)
    # -t cuts at the cap; one second of slack absorbs the resampler's rounding.
    sink = _Sink(np, math.ceil(limit * rate) + rate)
    args = [*_BASE, "-ss", _fmt(start_s), "-t", _fmt(limit), "-i", input_url(p)]
    args += ["-map", "0:a:0", "-ac", "1"]
    if band_hz is not None:
        low, high = band_hz
        args += ["-af", f"aresample={rate},highpass=f={_fmt(low)},lowpass=f={_fmt(high)}"]
    args += ["-ar", str(rate), "-c:a", "pcm_f32le", "-f", "f32le", "-"]
    run_tool("ffmpeg", args, timeout_s=timeout_s, stdout_sink=sink.feed)
    return Mono(samples=sink.buf[: sink.n], rate=rate, start_s=start_s, non_finite=sink.non_finite)
