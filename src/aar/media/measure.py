"""Measurements made by running ffmpeg on a file: loudness, sample stats, silences, full
decode, raw PCM windows and a clipped-sample count.

Every function takes a local path, refuses anything that is not a regular file, passes the
path as ``file:<abs path>`` and runs with a timeout (see ``aar.media.tools``).
"""

from __future__ import annotations

import os
import sys
from array import array
from dataclasses import dataclass

from aar.media.parse import (
    AStats,
    Ebur128Summary,
    ParseError,
    Silence,
    parse_astats,
    parse_ebur128,
    parse_silencedetect,
)
from aar.media.pcm import Pcm
from aar.media.tools import (
    MediaError,
    input_url,
    require_regular_file,
    run_tool,
)

_BASE = ["-hide_banner", "-nostdin", "-nostats"]
PCM_RATE = 48000
MEASURE_STDERR_LIMIT = 8 * 1024 * 1024
DECODE_ERROR_LINES = 5


@dataclass(frozen=True)
class AudioMeasurements:
    """One decode of the first audio stream through astats, silencedetect and ebur128."""

    loudness: Ebur128Summary
    stats: AStats
    silences: list[Silence]
    silence_db: float
    silence_min_s: float
    log_truncated: bool


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def measure_audio(
    path: str | os.PathLike[str],
    *,
    silence_db: float,
    silence_min_s: float,
    timeout_s: float,
) -> AudioMeasurements:
    """Measure the first audio stream in a single ffmpeg pass.

    astats runs first so its sample peak is of the original samples (ebur128 resamples to
    48 kHz for its own measurement). A silence is every sample of every channel below
    ``silence_db`` for at least ``silence_min_s`` (ffmpeg silencedetect semantics).
    """
    p = require_regular_file(path, tool="ffmpeg")
    chain = ",".join(
        [
            "astats=measure_overall=Peak_level+Number_of_NaNs+Number_of_Infs+Flat_factor"
            "+Peak_count+Number_of_samples:measure_perchannel=none",
            f"silencedetect=n={_fmt(silence_db)}dB:d={_fmt(silence_min_s)}",
            "ebur128=peak=sample+true:framelog=quiet",
        ]
    )
    result = run_tool(
        "ffmpeg",
        [
            *_BASE,
            *("-v", "info", "-i", input_url(p), "-map", "0:a:0"),
            *("-af", chain, "-f", "null", "-"),
        ],
        timeout_s=timeout_s,
        stderr_limit=MEASURE_STDERR_LIMIT,
        stdout_limit=1024 * 1024,
    )
    try:
        loudness = parse_ebur128(result.stderr)
        stats = parse_astats(result.stderr)
    except ParseError as exc:
        raise MediaError(
            f"could not read the audio measurements: {exc}",
            tool="ffmpeg",
            exit_code=result.returncode,
            stderr_tail=result.stderr_tail,
        ) from None
    return AudioMeasurements(
        loudness=loudness,
        stats=stats,
        silences=parse_silencedetect(result.stderr),
        silence_db=silence_db,
        silence_min_s=silence_min_s,
        log_truncated=result.stderr_truncated,
    )


@dataclass(frozen=True)
class DecodeResult:
    """A full decode of every video and audio stream with ``-xerror``."""

    ok: bool
    exit_code: int
    errors: list[str]
    error_lines: int


def decode_check(path: str | os.PathLike[str], *, timeout_s: float) -> DecodeResult:
    """Decode everything; any error line or a non-zero exit is a failure."""
    p = require_regular_file(path, tool="ffmpeg")
    result = run_tool(
        "ffmpeg",
        [
            *_BASE,
            "-v",
            "error",
            "-xerror",
            "-i",
            input_url(p),
            "-map",
            "0:v?",
            "-map",
            "0:a?",
            "-f",
            "null",
            "-",
        ],
        timeout_s=timeout_s,
        check=False,
        stdout_limit=1024 * 1024,
    )
    lines = [ln.strip() for ln in result.stderr.splitlines() if ln.strip()]
    return DecodeResult(
        ok=result.returncode == 0 and not lines,
        exit_code=result.returncode,
        errors=lines[:DECODE_ERROR_LINES],
        error_lines=len(lines),
    )


def _floats(data: bytes) -> array[float]:
    usable = len(data) - len(data) % 4
    samples = array("f")
    samples.frombytes(data[:usable])
    if sys.byteorder == "big":  # pragma: no cover - f32le on a big-endian host
        samples.byteswap()
    return samples


def read_pcm(
    path: str | os.PathLike[str],
    *,
    start_s: float,
    duration_s: float,
    channels: int,
    timeout_s: float = 30.0,
    rate: int = PCM_RATE,
) -> Pcm:
    """Decode ``duration_s`` of the first audio stream from ``start_s`` as 32-bit float.

    The audio is resampled to ``rate`` so sample counts mean the same thing for every source;
    ``channels`` is the stream's channel count (from ``probe``) and is kept as is.
    """
    p = require_regular_file(path, tool="ffmpeg")
    if channels <= 0:
        raise ValueError("channels must be positive")
    start_s = max(0.0, start_s)
    max_bytes = int((duration_s + 1.0) * rate * channels * 4) + 4096
    result = run_tool(
        "ffmpeg",
        [
            *_BASE,
            "-v",
            "error",
            "-ss",
            _fmt(start_s),
            "-t",
            _fmt(max(duration_s, 0.0)),
            "-i",
            input_url(p),
            "-map",
            "0:a:0",
            "-ac",
            str(channels),
            "-ar",
            str(rate),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        ],
        timeout_s=timeout_s,
        stdout_limit=max_bytes,
    )
    return Pcm(rate=rate, channels=channels, start_s=start_s, samples=_floats(result.stdout))


@dataclass(frozen=True)
class ClipCount:
    """Samples at or above the clip level, and when the first one occurs."""

    clipped: int
    first_s: float | None
    level: float
    scanned: int


class _ClipCounter:
    def __init__(self, level: float) -> None:
        self.level = level
        self.rest = b""
        self.scanned = 0
        self.clipped = 0
        self.first_index: int | None = None

    def feed(self, chunk: bytes) -> None:
        data = self.rest + chunk
        usable = len(data) - len(data) % 4
        self.rest = data[usable:]
        samples = _floats(data[:usable])
        level = self.level
        # max/min run in C; only chunks that reach the level are walked in Python.
        if samples and (max(samples) >= level or min(samples) <= -level):
            for i, x in enumerate(samples):
                if x >= level or x <= -level:
                    if self.first_index is None:
                        self.first_index = self.scanned + i
                    self.clipped += 1
        self.scanned += len(samples)


def count_clipped(
    path: str | os.PathLike[str],
    *,
    level: float,
    channels: int,
    sample_rate: int,
    timeout_s: float,
) -> ClipCount:
    """Count decoded samples (any channel) whose magnitude is at or above ``level``.

    The first audio stream is decoded to 32-bit float at its own rate (no resampling, which
    would invent or hide overs) and streamed, so memory stays flat for long files.
    """
    p = require_regular_file(path, tool="ffmpeg")
    if channels <= 0:
        raise ValueError("channels must be positive")
    counter = _ClipCounter(level)
    run_tool(
        "ffmpeg",
        [
            *_BASE,
            "-v",
            "error",
            "-i",
            input_url(p),
            "-map",
            "0:a:0",
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        ],
        timeout_s=timeout_s,
        stdout_sink=counter.feed,
    )
    first = counter.first_index
    return ClipCount(
        clipped=counter.clipped,
        first_s=None if first is None or sample_rate <= 0 else (first // channels) / sample_rate,
        level=level,
        scanned=counter.scanned,
    )
