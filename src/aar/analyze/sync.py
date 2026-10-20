"""``analyze.sync``: the audio offset between recordings of one event, with an honest confidence.

One method replaces the four script variants this was ported from (whole-file PHAT, windowed
PHAT, envelope cross-correlation, restricted-range PHAT). The choices, and why:

**Sign.** ``offset_s`` is where the other file's first sample falls on the reference's clock:
``reference time = other time + offset_s``. If B's frame 0 is at A's 2.0 s, B's offset is
+2.0 and ``B.recording_origin = A.recording_origin + 2.0 * fps`` (frames, rounded to the
nearest frame; halves round up). A negative offset means B started first.

**Decode.** The first audio stream of each file, averaged to mono, resampled to 8 kHz and
band-limited to 150-3500 Hz (ffmpeg high-pass and low-pass), streamed as f32 into numpy
(``aar.analyze.audio``). The band keeps speech and room sound and drops hum, rumble and hiss
that differ between microphones; 8 kHz is the lowest rate that holds the band, so an hour of
audio is 115 MB per file. Each file is read up to ``max_duration_s`` (default 2 h); a longer
file is analysed on its first part and flagged ``truncated``.

**Coarse pass (anchors).** ``anchors`` windows of ``anchor_s`` are spread evenly over the
reference (not one whole-file FFT, which needs memory for the whole lag range and lets one long
stretch of music or a repeated sound decide). Each window is searched in the other file over
every lag within ``max_offset_s`` with GCC-PHAT: the cross-spectrum is divided by its own
magnitude plus a small epsilon (``phat_eps``, 1e-3, times the median in-band magnitude: the
largest would let one loud tone swamp every other bin), so every frequency counts equally and
a loud hum or a peaky microphone cannot dominate the peak. Bins outside the
band are zeroed rather than whitened: there is nothing in them but noise. The search runs in
blocks of a fixed FFT size (overlap-save), so memory does not grow with ``max_offset_s``; only
the running peak and a 5 ms grid of block maxima are kept. The peak is taken on ``|r|`` (a
microphone wired with inverted polarity correlates negatively; the sign is reported) and
refined to a fraction of a sample with a parabola through it and its neighbours.

``max_offset_s`` defaults to 600 s. Cameras on a set usually start within a minute or two of
each other; 10 minutes covers a late start with margin. A wider range costs CPU linearly (about
a second per anchor per 10 minutes of range at 8 kHz) and gives a false peak more places to be,
which the consensus below has to outvote, so it is not unbounded.

**Window confidence.** ``ratio`` is the main peak over the highest peak outside a guard band of
``guard_s`` (20 ms, rounded out to the 5 ms grid) around it. Early reflections and the main
lobe sit inside the guard; a periodic sound (a tone, a click track) or a second, equally good
match sits outside it and drives the ratio towards 1. A peak within the guard band of a limit
of the searched range (the true one may lie just beyond it) is never trusted; the start and end
of the other file are not such limits.

**Consensus.** Window offsets that lie within ``agree_s`` (0.25 s) of each other form a
candidate; the candidate with the most windows (then the highest median ratio) wins and its
offset is their median. A chance peak lands anywhere in ±``max_offset_s``, so two windows
agreeing to 0.25 s is strong evidence; the tolerance also absorbs up to ±0.25 s of clock drift
between the anchors, which the consistency pass then measures. ``agreement`` is the agreeing
windows over the *eligible* ones: windows that are not silent and that lie inside the other
file at the candidate's offset, so a partial overlap is not penalised for anchors that could
never match. The pair is confident when at least ``min_windows`` (2) windows agree,
``agreement`` is at least ``min_agreement`` (0.6) and the agreeing windows' median ratio is at
least ``min_ratio`` (2.0). Noise with no match gives ratios near 1.05 over millions of lags, so
2.0 leaves a wide margin; a real match between two microphones in one room gives tens.

**Consistency and drift.** Over the overlap, ``window_s`` (6 s) windows on a ``step_s`` (2 s)
step, at most ``max_windows`` (120) of them spaced evenly, are each searched within
±``search_s`` (0.25 s) of the coarse offset. Windows that are silent, below ``min_ratio`` or
on the edge of their range are excluded and counted. A Theil-Sen line (the median of pairwise
slopes: robust to a minority of wrong peaks) fits offset against reference time; the drift is
its slope in ppm (positive: the offset grows along the reference, so the other file's clock runs
slow) and ``residual_rms_ms`` is over the windows within max(3 robust sigma, 1 ms) of the line.
The reported offset is the line's value at the middle of the overlap, the best single offset
for the whole overlap. If fewer than ``min_confirm`` (half) of the non-silent windows confirm
the line, the pair is low-confidence after all.

**Status**, one per pair: ``ok`` (confident); ``drift`` (confident, but the line moves more
than one frame across the overlap: reported with a hint, never corrected); ``low-confidence``
(candidates listed, no offset or origin suggested: never a silent number); ``no-overlap``
(less than ``min_overlap_s``, 5 s, of shared audio is possible: a file shorter than that, or a
confident match that overlaps less); ``failed`` (a typed decode error). Audio that simply does
not match (disjoint takes, a dead microphone) cannot be told apart from audio that matches
badly, so it is ``low-confidence``, not ``no-overlap``.

Writing offsets into a plan is out of scope here (S13d).
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any, Literal

from aar.analyze.audio import Mono, read_mono
from aar.analyze.errors import AnalyzeError, from_media_error, need_numpy
from aar.media import MediaError, probe, timeout_for

PairStatus = Literal["ok", "drift", "low-confidence", "no-overlap", "failed"]

CELL_S = 0.005  # grid of the block maxima kept for the second-peak search
RATIO_CAP = 1e6  # a ratio with no competing peak at all (reported, JSON has no inf)
_TINY = 1e-30
_MAD_SIGMA = 1.4826  # median absolute deviation to standard deviation, for normal noise
_INLIER_FLOOR_S = 0.001
_CANDIDATES_SHOWN = 5

SIGN_CONVENTION = (
    "offset_s is where the other file's first sample falls on the reference's clock "
    "(reference time = other time + offset_s); "
    "other.recording_origin = reference.recording_origin + offset_s * fps"
)


@dataclass(frozen=True)
class SyncSettings:
    """Every threshold of the method (see the module docstring for the reasons)."""

    rate: int = 8000
    band_hz: tuple[float, float] = (150.0, 3500.0)
    max_offset_s: float = 600.0
    max_duration_s: float = 7200.0
    anchors: int = 12
    anchor_s: float = 10.0
    guard_s: float = 0.020
    phat_eps: float = 1e-3
    agree_s: float = 0.25
    min_ratio: float = 2.0
    min_agreement: float = 0.6
    min_windows: int = 2
    window_s: float = 6.0
    step_s: float = 2.0
    search_s: float = 0.25
    max_windows: int = 120
    min_confirm: float = 0.5
    min_overlap_s: float = 5.0
    silence_dbfs: float = -60.0
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        positive = {
            "max_offset_s": self.max_offset_s,
            "max_duration_s": self.max_duration_s,
            "anchor_s": self.anchor_s,
            "guard_s": self.guard_s,
            "phat_eps": self.phat_eps,
            "agree_s": self.agree_s,
            "min_ratio": self.min_ratio,
            "window_s": self.window_s,
            "step_s": self.step_s,
            "search_s": self.search_s,
            "min_overlap_s": self.min_overlap_s,
        }
        for name, value in positive.items():
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number above 0, not {value!r}")
        for name, value in (
            ("min_agreement", self.min_agreement),
            ("min_confirm", self.min_confirm),
        ):
            if not (math.isfinite(value) and 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1, not {value!r}")
        for name, count in (
            ("rate", self.rate),
            ("anchors", self.anchors),
            ("min_windows", self.min_windows),
            ("max_windows", self.max_windows),
        ):
            if not (isinstance(count, int) and count >= 1):
                raise ValueError(f"{name} must be a whole number of at least 1, not {count!r}")
        low, high = self.band_hz
        if not (0 < low < high < self.rate / 2):
            raise ValueError(f"band_hz must satisfy 0 < low < high < rate/2, not {self.band_hz}")
        if not (math.isfinite(self.silence_dbfs) and self.silence_dbfs < 0):
            raise ValueError("silence_dbfs must be a negative number of dB")
        if self.timeout_s is not None and not (
            math.isfinite(self.timeout_s) and self.timeout_s > 0
        ):
            raise ValueError("timeout_s must be a finite number of seconds above 0")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["band_hz"] = list(self.band_hz)
        return out


# ---------------------------------------------------------------- GCC-PHAT search


@dataclass(frozen=True)
class Peak:
    """The best lag of one window search. ``lag`` is the other file's sample where the window
    starts, refined to a fraction of a sample."""

    lag: float
    value: float  # |r| at the peak
    polarity: int  # +1, or -1 for an inverted microphone
    second: float  # the highest |r| outside the guard band
    at_edge: bool  # the peak is within the guard band of a search limit inside the other file

    @property
    def ratio(self) -> float:
        if self.second > 0:
            return min(self.value / self.second, RATIO_CAP)
        return RATIO_CAP if self.value > 0 else 0.0


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def parabolic(left: float | None, mid: float, right: float | None) -> float:
    """Fractional position (within ±0.5) of the vertex of the parabola through three samples
    around a maximum; 0 when a neighbour is missing or the samples do not curve down."""
    if left is None or right is None:
        return 0.0
    denom = left - 2.0 * mid + right
    if denom >= 0:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (left - right) / denom))


def gcc_phat_search(
    np: Any,
    window: Any,
    other: Any,
    lag_lo: int,
    lag_hi: int,
    *,
    rate: int,
    band_hz: tuple[float, float],
    guard_s: float,
    phat_eps: float,
) -> Peak:
    """Search ``other`` for ``window`` at every start lag in ``[lag_lo, lag_hi]``.

    ``r[k] = sum_j other[lag_lo + k + j] * window[j]``, PHAT-weighted inside ``band_hz``, in
    overlap-save blocks of one FFT size. The caller keeps ``lag_hi + len(window) <= len(other)``.
    """
    n_win = len(window)
    count = lag_hi - lag_lo + 1
    if n_win == 0 or count <= 0:
        raise ValueError("empty window or lag range")
    cell = max(1, round(CELL_S * rate))
    n_fft = _next_pow2(n_win + min(count, 2 * n_win) + 2)
    room = n_fft - n_win - 1  # lags per block whose right neighbour is also free of wrap-around
    step = count if count <= room else max(cell, room // cell * cell)
    k_lo = math.ceil(band_hz[0] * n_fft / rate)
    k_hi = math.floor(band_hz[1] * n_fft / rate)
    spec_win = np.conj(np.fft.rfft(window, n_fft))
    cells: list[Any] = []
    best_value = -1.0
    best = (0, 0.0, 1, None, None)  # g, value, polarity, left, right
    prev_last: float | None = None
    for start in range(0, count, step):
        first = lag_lo + start
        n_valid = min(step, count - start)
        spec = np.fft.rfft(other[first : first + n_fft], n_fft) * spec_win
        spec[:k_lo] = 0
        spec[k_hi + 1 :] = 0
        mag = np.abs(spec)
        # Relative to the typical in-band bin, not the largest: one loud tone would otherwise
        # raise the epsilon over every other bin and undo the whitening.
        eps = max(phat_eps * float(np.median(mag[k_lo : k_hi + 1])), _TINY)
        spec /= mag + eps
        r = np.fft.irfft(spec, n_fft)[: n_valid + 1]
        a = np.abs(r)
        k = int(np.argmax(a[:n_valid]))
        if a[k] > best_value:
            best_value = float(a[k])
            left = float(a[k - 1]) if k > 0 else prev_last
            best = (start + k, best_value, 1 if r[k] >= 0 else -1, left, float(a[k + 1]))
        prev_last = float(a[n_valid - 1])
        pad = (-n_valid) % cell
        valid = a[:n_valid] if not pad else np.concatenate([a[:n_valid], np.zeros(pad)])
        cells.append(valid.reshape(-1, cell).max(axis=1))
    grid = np.concatenate(cells)
    g, value, polarity, left, right = best
    guard = max(1, round(guard_s * rate))
    lo_cell = max(0, (g - guard) // cell)
    hi_cell = (g + guard) // cell
    outside = [part for part in (grid[:lo_cell], grid[hi_cell + 1 :]) if part.size]
    second = max((float(part.max()) for part in outside), default=0.0)
    # A peak within the guard band of a limit of the search range (not of the start or end of
    # the other file, where the window cannot go further) may be the flank or a side lobe of a
    # better peak just beyond it.
    at_edge = (g < guard and lag_lo > 0) or (count - 1 - g < guard and lag_hi < len(other) - n_win)
    frac = 0.0 if at_edge or value <= 0 else parabolic(left, value, right)
    return Peak(
        lag=lag_lo + g + frac,
        value=value,
        polarity=polarity,
        second=second,
        at_edge=at_edge,
    )


def _dbfs(np: Any, samples: Any, chunk: int = 1 << 20) -> float:
    """RMS level in dBFS, summed in float64 a chunk at a time (no file-sized temporary)."""
    n = len(samples)
    if n == 0:
        return -math.inf
    total = 0.0
    for i in range(0, n, chunk):
        part = samples[i : i + chunk].astype(np.float64)
        # einsum, not np.dot: dot goes through BLAS, whose worker threads spin on every core.
        total += float(np.einsum("i,i->", part, part))
    power = total / n
    return 10.0 * math.log10(power) if power > 0 else -math.inf


# ---------------------------------------------------------------- consensus


@dataclass(frozen=True)
class WindowResult:
    """One anchor or consistency window. ``offset_s`` is None when it was not searched."""

    ref_start_s: float
    length_s: float
    state: Literal["searched", "silent", "unsearchable"]
    offset_s: float | None = None
    ratio: float | None = None
    peak: float | None = None
    polarity: int | None = None
    at_edge: bool = False

    def confident(self, min_ratio: float) -> bool:
        return (
            self.state == "searched"
            and not self.at_edge
            and self.ratio is not None
            and self.ratio >= min_ratio
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ref_start_s": round(self.ref_start_s, 4), "state": self.state}
        if self.state == "searched":
            out.update(
                offset_s=round(self.offset_s or 0.0, 6),
                ratio=round(self.ratio or 0.0, 3),
                peak=round(self.peak or 0.0, 6),
                polarity=self.polarity,
            )
            if self.at_edge:
                out["at_edge"] = True
        return out


@dataclass(frozen=True)
class Candidate:
    """A cluster of window offsets within ``agree_s`` of each other."""

    offset_s: float
    members: tuple[int, ...]  # indices of the agreeing windows
    eligible: tuple[int, ...]  # indices of the windows that could have matched at this offset
    median_ratio: float

    @property
    def agreement(self) -> float:
        return len(self.members) / len(self.eligible) if self.eligible else 0.0

    def confident(self, settings: SyncSettings) -> bool:
        return (
            len(self.members) >= settings.min_windows
            and self.agreement >= settings.min_agreement
            and self.median_ratio >= settings.min_ratio
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_s": round(self.offset_s, 6),
            "windows_agreeing": len(self.members),
            "windows_eligible": len(self.eligible),
            "agreement": round(self.agreement, 3),
            "median_ratio": round(self.median_ratio, 3),
        }


def candidates(
    windows: list[WindowResult],
    *,
    agree_s: float,
    eligible_at: Any,
) -> list[Candidate]:
    """Cluster the searched windows' offsets, best candidate first.

    A candidate is every searched, non-edge window within ``agree_s`` of one window's offset;
    its offset is their median. ``eligible_at(offset)`` returns the indices of the windows that
    could have matched at that offset (not silent, inside the other file). Candidates are
    ranked by the number of agreeing windows, then their median ratio; each window belongs to
    the first (best) candidate that claims it.
    """
    usable = [
        i
        for i, w in enumerate(windows)
        if w.state == "searched" and not w.at_edge and w.offset_s is not None
    ]
    found: list[Candidate] = []
    remaining = set(usable)
    while remaining:
        best: tuple[tuple[int, float], tuple[int, ...]] | None = None
        for i in sorted(remaining):
            centre = windows[i].offset_s
            assert centre is not None
            members = tuple(
                j
                for j in sorted(remaining)
                if abs((windows[j].offset_s or 0.0) - centre) <= agree_s
            )
            score = (len(members), median(windows[j].ratio or 0.0 for j in members))
            if best is None or score > best[0]:
                best = (score, members)
        assert best is not None
        members = best[1]
        offset = median(windows[j].offset_s or 0.0 for j in members)
        eligible = tuple(sorted(set(eligible_at(offset)) | set(members)))
        found.append(
            Candidate(
                offset_s=offset,
                members=members,
                eligible=eligible,
                median_ratio=best[0][1],
            )
        )
        remaining -= set(members)
    return found


# ---------------------------------------------------------------- drift fit


@dataclass(frozen=True)
class LineFit:
    slope: float  # seconds of offset per second of reference time
    intercept: float  # offset at reference time 0
    inliers: tuple[int, ...]
    residual_rms_s: float

    def at(self, t: float) -> float:
        return self.intercept + self.slope * t


def theil_sen(t: list[float], y: list[float]) -> LineFit:
    """Robust line through (t, y): the median of pairwise slopes and the median intercept.

    Inliers are the points within max(3 robust sigma, 1 ms) of the line (sigma from the median
    absolute deviation); ``residual_rms_s`` is over them. Needs two distinct times.
    """
    n = len(t)
    slopes = [
        (y[j] - y[i]) / (t[j] - t[i]) for i in range(n) for j in range(i + 1, n) if t[j] != t[i]
    ]
    if not slopes:
        raise ValueError("a line needs at least two distinct times")
    slope = median(slopes)
    intercept = median(yi - slope * ti for ti, yi in zip(t, y, strict=True))
    residuals = [yi - (intercept + slope * ti) for ti, yi in zip(t, y, strict=True)]
    centre = median(residuals)
    mad = median(abs(r - centre) for r in residuals)
    band = max(3 * _MAD_SIGMA * mad, _INLIER_FLOOR_S)
    inliers = tuple(i for i, r in enumerate(residuals) if abs(r) <= band)
    rms = math.sqrt(sum(residuals[i] ** 2 for i in inliers) / len(inliers)) if inliers else 0.0
    return LineFit(slope=slope, intercept=intercept, inliers=inliers, residual_rms_s=rms)


# ---------------------------------------------------------------- frames


def to_frames(offset_s: float, fps: Fraction) -> tuple[int, float, float]:
    """``offset_s`` in frames at ``fps``: (nearest frame with halves rounding up, the exact
    frame count, the rounding error in ms, positive when the frame lands later)."""
    exact = Fraction(offset_s) * fps
    frames = math.floor(exact + Fraction(1, 2))
    error_ms = float((Fraction(frames) / fps - Fraction(offset_s)) * 1000)
    return frames, float(exact), error_ms


# ---------------------------------------------------------------- one pair


@dataclass
class _Audio:
    path: Path
    duration_s: float | None  # from ffprobe
    mono: Mono

    @property
    def decoded_s(self) -> float:
        return self.mono.duration_s

    def describe(self, cap_s: float) -> dict[str, Any]:
        truncated = self.duration_s is not None and self.duration_s > cap_s + 1e-6
        return {
            "file": str(self.path),
            "duration_s": None if self.duration_s is None else round(self.duration_s, 4),
            "decoded_s": round(self.decoded_s, 4),
            "truncated": truncated,
            "non_finite_samples": self.mono.non_finite,
        }


def _overlap(offset: float, ref_s: float, other_s: float) -> tuple[float, float]:
    """The overlap on the reference clock, ``[lo, hi]`` (``hi < lo`` when there is none)."""
    return max(0.0, offset), min(ref_s, offset + other_s)


def _anchor_starts(n_ref: int, n_win: int, count: int) -> list[int]:
    if count == 1 or n_ref == n_win:
        return [(n_ref - n_win) // 2]
    span = n_ref - n_win
    return sorted({round(i * span / (count - 1)) for i in range(count)})


def _coarse(np: Any, ref: Any, other: Any, s: SyncSettings) -> tuple[list[WindowResult], int]:
    rate = s.rate
    n_win = min(round(s.anchor_s * rate), len(ref), len(other))
    max_lag = round(s.max_offset_s * rate)
    results: list[WindowResult] = []
    for a0 in _anchor_starts(len(ref), n_win, s.anchors):
        win = ref[a0 : a0 + n_win]
        base = WindowResult(ref_start_s=a0 / rate, length_s=n_win / rate, state="silent")
        if _dbfs(np, win) < s.silence_dbfs:
            results.append(base)
            continue
        lag_lo = max(0, a0 - max_lag)
        lag_hi = min(len(other) - n_win, a0 + max_lag)
        if lag_hi < lag_lo:
            results.append(WindowResult(base.ref_start_s, base.length_s, "unsearchable"))
            continue
        peak = gcc_phat_search(
            np, win, other, lag_lo, lag_hi,
            rate=rate, band_hz=s.band_hz, guard_s=s.guard_s, phat_eps=s.phat_eps,
        )  # fmt: skip
        results.append(_searched(base, a0, peak, rate))
    return results, n_win


def _searched(base: WindowResult, a0: int, peak: Peak, rate: int) -> WindowResult:
    return WindowResult(
        ref_start_s=base.ref_start_s,
        length_s=base.length_s,
        state="searched",
        offset_s=(a0 - peak.lag) / rate,
        ratio=peak.ratio,
        peak=peak.value,
        polarity=peak.polarity,
        at_edge=peak.at_edge,
    )


def _eligible(windows: list[WindowResult], n_other_s: float, offset: float) -> list[int]:
    """Windows that are not silent and lie wholly inside the other file at ``offset``."""
    out = []
    for i, w in enumerate(windows):
        if w.state == "silent":
            continue
        start_in_other = w.ref_start_s - offset
        if start_in_other >= 0 and start_in_other + w.length_s <= n_other_s + 1e-9:
            out.append(i)
    return out


def _consistency(
    np: Any, ref: Any, other: Any, offset: float, s: SyncSettings
) -> list[WindowResult]:
    rate = s.rate
    n_win = round(s.window_s * rate)
    lo, hi = _overlap(offset, len(ref) / rate, len(other) / rate)
    if hi - lo < s.window_s:
        return []
    count = math.floor((hi - lo - s.window_s) / s.step_s + 1e-9) + 1
    picks = (
        range(count)
        if count <= s.max_windows
        else sorted({round(i * (count - 1) / (s.max_windows - 1)) for i in range(s.max_windows)})
        if s.max_windows > 1
        else [count // 2]
    )
    search = round(s.search_s * rate)
    results = []
    for j in picks:
        a0 = round((lo + j * s.step_s) * rate)
        a0 = min(a0, len(ref) - n_win)
        win = ref[a0 : a0 + n_win]
        base = WindowResult(ref_start_s=a0 / rate, length_s=n_win / rate, state="silent")
        if _dbfs(np, win) < s.silence_dbfs:
            results.append(base)
            continue
        predicted = a0 - offset * rate
        lag_lo = max(0, math.floor(predicted) - search)
        lag_hi = min(len(other) - n_win, math.ceil(predicted) + search)
        if lag_hi < lag_lo:
            results.append(WindowResult(base.ref_start_s, base.length_s, "unsearchable"))
            continue
        peak = gcc_phat_search(
            np, win, other, lag_lo, lag_hi,
            rate=rate, band_hz=s.band_hz, guard_s=s.guard_s, phat_eps=s.phat_eps,
        )  # fmt: skip
        results.append(_searched(base, a0, peak, rate))
    return results


def _round_level(level: float) -> float | None:
    return round(level, 1) if math.isfinite(level) else None


def _hint_low(s: SyncSettings) -> str:
    return (
        "no offset is trusted: check that both files carry audio of the same room, widen "
        f"--max-offset-s (now {s.max_offset_s:g} s) if they started further apart, or sync by "
        "hand starting from the candidates"
    )


def analyse_pair(
    np: Any, ref: Mono, other: Mono, *, fps: Fraction, ref_origin: int, settings: SyncSettings
) -> dict[str, Any]:
    """Offset of ``other`` against ``ref`` (both decoded at ``settings.rate``)."""
    s = settings
    ref_s, other_s = ref.duration_s, other.duration_s
    out: dict[str, Any] = {"status": "low-confidence"}
    shortest = min(ref_s, other_s)
    if shortest < s.min_overlap_s:
        which = "reference" if ref_s <= other_s else "other file"
        out.update(
            status="no-overlap",
            overlap_s=round(shortest, 4),
            hint=(
                f"the {which} has only {shortest:.2f} s of audio, less than the "
                f"{s.min_overlap_s:g} s of overlap needed"
            ),
        )
        return out
    levels = {"reference": _dbfs(np, ref.samples), "other file": _dbfs(np, other.samples)}
    out["level_dbfs"] = {
        "reference": _round_level(levels["reference"]),
        "other": _round_level(levels["other file"]),
    }
    silent = [name for name, level in levels.items() if level < s.silence_dbfs]
    if silent:
        out.update(
            candidates=[],
            reason=f"the {' and the '.join(silent)} is silent (below {s.silence_dbfs:g} dBFS)",
            hint=_hint_low(s),
        )
        return out
    anchors, _ = _coarse(np, ref.samples, other.samples, s)
    found = candidates(
        anchors,
        agree_s=s.agree_s,
        eligible_at=lambda off: _eligible(anchors, other_s, off),
    )
    coarse: dict[str, Any] = {
        "windows": [w.to_dict() for w in anchors],
        "silent": sum(w.state == "silent" for w in anchors),
        "at_edge": sum(w.at_edge for w in anchors),
    }
    out["coarse"] = coarse
    best = found[0] if found else None
    if best is None or not best.confident(s):
        out["candidates"] = [c.to_dict() for c in found[:_CANDIDATES_SHOWN]]
        if best is None:
            reason = (
                "the reference is silent in every anchor window"
                if coarse["silent"] == len(anchors)
                else "no anchor window found a usable peak inside the search range"
            )
        else:
            reason = (
                f"best candidate: {len(best.members)} of {len(best.eligible)} windows agree "
                f"(need {s.min_windows} and {s.min_agreement:.0%}), median ratio "
                f"{best.median_ratio:.2f} (need {s.min_ratio:g})"
            )
        out.update(reason=reason, hint=_hint_low(s))
        return out
    coarse.update(best.to_dict())
    offset = best.offset_s
    windows = _consistency(np, ref.samples, other.samples, offset, s)
    searched = [i for i, w in enumerate(windows) if w.state == "searched"]
    confident = [i for i in searched if windows[i].confident(s.min_ratio)]
    drift: dict[str, Any] = {
        "windows": len(windows),
        "silent": sum(w.state == "silent" for w in windows),
        "unsearchable": sum(w.state == "unsearchable" for w in windows),
        "low_confidence": len(searched) - len(confident),
        "measured": False,
    }
    inliers: list[int] = []
    if len(confident) >= 2:
        t = [windows[i].ref_start_s + windows[i].length_s / 2 for i in confident]
        y = [windows[i].offset_s or 0.0 for i in confident]
        fit = theil_sen(t, y)
        inliers = [confident[k] for k in fit.inliers]
        lo, hi = _overlap(offset, ref_s, other_s)
        offset = fit.at((lo + hi) / 2)
        drift.update(
            measured=True,
            ppm=round(fit.slope * 1e6, 3),
            residual_rms_ms=round(fit.residual_rms_s * 1000, 4),
            used=len(inliers),
            outliers=len(confident) - len(inliers),
        )
    elif len(confident) == 1:
        inliers = confident
        offset = windows[confident[0]].offset_s or offset
        drift["note"] = "one consistency window confirmed the offset: drift not measured"
    else:
        drift["note"] = "no consistency window could be compared: drift not measured"
    confirm = len(inliers) / len(searched) if searched else None
    drift["confirmed"] = None if confirm is None else round(confirm, 3)
    drift["detail"] = [w.to_dict() for w in windows]
    out["drift"] = drift
    lo, hi = _overlap(offset, ref_s, other_s)
    out["overlap_s"] = round(max(0.0, hi - lo), 4)
    out["overlap_ref_s"] = [round(lo, 4), round(hi, 4)]
    if confirm is not None and confirm < s.min_confirm:
        out["candidates"] = [{**best.to_dict(), "offset_s": round(offset, 6)}]
        out.update(
            reason=(
                f"only {len(inliers)} of {len(searched)} consistency windows confirm the "
                f"offset (need {s.min_confirm:.0%})"
            ),
            hint=_hint_low(s),
        )
        return out
    if hi - lo < s.min_overlap_s:
        out.update(
            status="no-overlap",
            hint=f"the files share {max(0.0, hi - lo):.2f} s of audio, less than "
            f"{s.min_overlap_s:g} s",
        )
        return out
    frames, exact, error_ms = to_frames(offset, fps)
    out.update(
        offset_s=round(offset, 6),
        offset_frames=frames,
        offset_frames_exact=round(exact, 4),
        rounding_error_ms=round(error_ms, 4),
        suggested_recording_origin=ref_origin + frames,
        confidence={
            "agreement": round(best.agreement, 3),
            "median_ratio": round(best.median_ratio, 3),
            "windows_agreeing": len(best.members),
            "windows_eligible": len(best.eligible),
            "confirmed": drift["confirmed"],
        },
    )
    frame_ms = float(1000 / fps)
    if drift["measured"]:
        moved_ms = abs(drift["ppm"]) * 1e-6 * (hi - lo) * 1000
        drift["over_overlap_ms"] = round(moved_ms, 3)
        if moved_ms > frame_ms:
            out.update(
                status="drift",
                hint=(
                    f"the offset moves {moved_ms:.1f} ms ({drift['ppm']:+.1f} ppm) across the "
                    f"{hi - lo:.0f} s overlap, more than one frame ({frame_ms:.1f} ms): "
                    "the suggested origin fits the middle of the overlap; retime the other "
                    "file or sync it in parts (aar does not correct drift)"
                ),
            )
            return out
    out["status"] = "ok"
    return out


# ---------------------------------------------------------------- files


def _load(path: Path, s: SyncSettings, what: str) -> _Audio:
    try:
        info = probe(path)
        if info.audio is None:
            raise AnalyzeError(
                "no-audio", f"{what} {path} has no audio stream", hint="sync needs audio"
            )
        duration = info.audio.duration_s or info.duration_s
        timeout = s.timeout_s or timeout_for(
            min(duration or s.max_duration_s, s.max_duration_s), per_media_s=0.5
        )
        mono = read_mono(
            path,
            rate=s.rate,
            max_duration_s=s.max_duration_s,
            timeout_s=timeout,
            band_hz=s.band_hz,
        )
    except MediaError as exc:
        raise from_media_error(exc, f"{what} {path}") from None
    return _Audio(path=path, duration_s=duration, mono=mono)


def sync_files(
    ref: str | os.PathLike[str],
    others: list[str | os.PathLike[str]],
    *,
    fps: Fraction,
    ref_origin: int = 0,
    settings: SyncSettings | None = None,
) -> dict[str, Any]:
    """Offsets of every file in ``others`` against ``ref``: a JSON-ready report.

    Raises ``AnalyzeError`` when the run cannot start (numpy, ffmpeg, a missing file). A file
    that exists but does not decode, or has no audio, gives a ``failed`` pair instead.
    """
    s = settings or SyncSettings()
    if not (isinstance(fps, Fraction) and fps > 0):
        raise ValueError("fps must be a positive Fraction")
    np = need_numpy()
    report: dict[str, Any] = {
        "ok": False,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "sign": SIGN_CONVENTION,
        "settings": s.to_dict(),
        "reference": {"file": str(ref), "recording_origin": ref_origin},
        "pairs": [],
    }
    ref_audio: _Audio | None = None
    ref_error: AnalyzeError | None = None
    try:
        ref_audio = _load(Path(ref), s, "reference")
        report["reference"].update(ref_audio.describe(s.max_duration_s))
    except AnalyzeError as exc:
        if exc.exit_code != 1:
            raise
        ref_error = exc
        report["reference"]["error"] = exc.to_dict()
    for path in others:
        pair: dict[str, Any] = {"file": str(path)}
        if ref_audio is None:
            assert ref_error is not None
            pair.update(status="failed", error=ref_error.to_dict())
            report["pairs"].append(pair)
            continue
        try:
            other = _load(Path(path), s, "file")
        except AnalyzeError as exc:
            if exc.exit_code != 1:
                raise
            pair.update(status="failed", error=exc.to_dict())
            report["pairs"].append(pair)
            continue
        described = other.describe(s.max_duration_s)
        pair.update({k: v for k, v in described.items() if k != "file"})
        pair.update(
            analyse_pair(np, ref_audio.mono, other.mono, fps=fps, ref_origin=ref_origin, settings=s)
        )
        report["pairs"].append(pair)
        del other  # one other file in memory at a time
    report["ok"] = bool(report["pairs"]) and all(p["status"] == "ok" for p in report["pairs"])
    return report


__all__ = [
    "SIGN_CONVENTION",
    "Candidate",
    "LineFit",
    "Peak",
    "SyncSettings",
    "WindowResult",
    "analyse_pair",
    "candidates",
    "gcc_phat_search",
    "parabolic",
    "sync_files",
    "theil_sen",
    "to_frames",
]
