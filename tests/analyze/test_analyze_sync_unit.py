"""The pure parts of ``analyze.sync`` on numpy arrays: GCC-PHAT search, peak ratio and guard
band, consensus, the drift fit, frame conversion and the confirm step. No ffmpeg."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from analyze_support import band_noise

from aar.analyze.audio import Mono
from aar.analyze.sync import (
    RATIO_CAP,
    Candidate,
    SyncSettings,
    WindowResult,
    analyse_pair,
    candidates,
    gcc_phat_search,
    parabolic,
    theil_sen,
    to_frames,
)

RATE = 8000
BAND = (150.0, 3500.0)


def noise(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return band_noise(int(seconds * RATE), RATE, rng, *BAND).astype(np.float32)


def search(window, other, lo, hi, guard_s=0.020):
    return gcc_phat_search(
        np, window, other, lo, hi, rate=RATE, band_hz=BAND, guard_s=guard_s, phat_eps=1e-3
    )


# ---------------------------------------------------------------- GCC-PHAT


def test_search_finds_the_lag_and_reports_a_high_ratio():
    other = noise(30.0, 1)
    window = other[80_000:120_000]  # 5 s starting at 10 s
    peak = search(window, other, 0, len(other) - len(window))
    assert peak.lag == pytest.approx(80_000, abs=0.01)
    assert peak.polarity == 1 and not peak.at_edge
    assert peak.ratio > 10
    flipped = search(-window, other, 0, len(other) - len(window))
    assert flipped.lag == pytest.approx(80_000, abs=0.01) and flipped.polarity == -1


def test_blocks_give_the_same_peak_as_one_big_correlation():
    # A 1 s window over 25 s of lags needs many overlap-save blocks; the peak must not move
    # at a block edge. The reference is brute-force GCC-PHAT in one FFT.
    other = noise(30.0, 2)
    window = other[123_457 : 123_457 + RATE].copy()
    lo, hi = 3, len(other) - len(window)
    peak = search(window, other, lo, hi)
    n = 1 << (len(other) + len(window)).bit_length()
    spec = np.fft.rfft(other, n) * np.conj(np.fft.rfft(window, n))
    r = np.abs(np.fft.irfft(spec / (np.abs(spec) + 1e-12), n))[lo : hi + 1]
    assert round(peak.lag) == lo + int(np.argmax(r)) == 123_457


def test_digital_silence_gives_no_peak():
    window = noise(5.0, 3)
    peak = search(window, np.zeros(20 * RATE, dtype=np.float32), 0, 15 * RATE)
    assert peak.value == 0 and peak.second == 0 and peak.ratio == 0
    loud = search(window, noise(20.0, 4), 0, 15 * RATE)  # unrelated noise: no match
    assert 1.0 <= loud.ratio < 1.5


@pytest.mark.parametrize(("echo_ms", "guarded"), [(10.0, True), (40.0, False)])
def test_the_guard_band_hides_near_echoes_only(echo_ms, guarded):
    # The other file carries the window twice with equal strength: echo_ms apart. Inside the
    # 20 ms guard the second copy is part of the main peak; outside it, it is a rival.
    base = noise(20.0, 5)
    shift = round(echo_ms / 1000 * RATE)
    other = base.copy()
    other[shift:] += base[:-shift]
    window = base[40_000:80_000]
    peak = search(window, other, 30_000, 50_000)
    assert round(peak.lag) in (40_000, 40_000 + shift)
    if guarded:
        assert peak.ratio > 5
    else:
        assert peak.ratio < 1.3


def test_a_peak_on_a_limit_of_the_range_is_flagged_but_not_on_the_file_ends():
    other = noise(20.0, 6)
    window = other[40_000:80_000]
    inside = search(window, other, 39_000, 41_000)
    assert not inside.at_edge and inside.lag == pytest.approx(40_000, abs=0.01)
    # The true lag is 1 sample past the range: the range limit catches a flank or a side lobe
    # of the peak, a few samples from the limit.
    limited = search(window, other, 38_000, 39_999)
    assert limited.at_edge and 39_999 - limited.lag < 0.020 * RATE
    # The same window found at the very start or end of the other file is a real match.
    first = search(other[:40_000], other, 0, 2_000)
    last = search(other[-40_000:], other, len(other) - 42_000, len(other) - 40_000)
    assert first.lag == 0 and not first.at_edge and first.ratio > 10
    assert last.lag == pytest.approx(len(other) - 40_000, abs=0.01)
    assert not last.at_edge and last.ratio > 10


@pytest.mark.parametrize("frac", [0.2, 0.4, 0.6])
def test_sub_sample_peaks_are_refined_with_a_parabola(frac):
    # The PHAT main lobe of a 150-3500 Hz band at 8 kHz is about two samples wide, so a
    # three-point parabola is biased towards the nearest sample: measured at most 0.23 sample
    # (29 us) here. Rounding to the sample would be off by up to 0.5 sample.
    rng = np.random.default_rng(7)
    fine = band_noise(20 * 48_000, 48_000, rng, *BAND)
    other = fine[::6].astype(np.float32)  # 8 kHz
    delayed = np.interp(np.arange(len(other)) * 6 + frac * 6, np.arange(len(fine)), fine)
    window = delayed[40_000:80_000].astype(np.float32)  # window[j] = other[40000 + frac + j]
    peak = search(window, other, 39_000, 41_000)
    error = abs(peak.lag - (40_000 + frac))
    assert error < 0.2 if frac != 0.4 else error < 0.25
    assert error < abs(round(peak.lag) - (40_000 + frac))  # better than the nearest sample


@pytest.mark.parametrize(
    ("left", "mid", "right", "expected"),
    [
        (1.0, 2.0, 1.0, 0.0),
        (1.0, 2.0, 1.5, 0.5 * (1.0 - 1.5) / (1.0 - 4.0 + 1.5)),
        (None, 2.0, 1.0, 0.0),
        (2.0, 2.0, 2.0, 0.0),  # flat: no vertex
        (0.0, 1.0, 1.0, 0.5),
    ],
)
def test_parabolic(left, mid, right, expected):
    assert parabolic(left, mid, right) == pytest.approx(expected)


# ---------------------------------------------------------------- consensus


def _window(offset: float | None, ratio: float = 10.0, *, start: float = 0.0, edge=False):
    if offset is None:
        return WindowResult(ref_start_s=start, length_s=10.0, state="silent")
    return WindowResult(
        ref_start_s=start,
        length_s=10.0,
        state="searched",
        offset_s=offset,
        ratio=ratio,
        peak=0.3,
        polarity=1,
        at_edge=edge,
    )


def test_candidates_cluster_within_agree_s_and_take_the_median():
    windows = [
        _window(2.00),
        _window(2.10),
        _window(1.90),
        _window(2.20, ratio=50),
        _window(-7.0, ratio=1.1),
        _window(9.0, edge=True),  # never clusters
        _window(None),
    ]
    found = candidates(windows, agree_s=0.25, eligible_at=lambda off: [0, 1, 2, 3, 4])
    best = found[0]
    assert best.members == (0, 1, 2, 3)
    assert best.offset_s == pytest.approx(2.05)  # median of 1.9, 2.0, 2.1, 2.2
    assert best.median_ratio == 10.0
    assert best.eligible == (0, 1, 2, 3, 4)
    assert best.agreement == pytest.approx(0.8)
    assert [c.members for c in found[1:]] == [(4,)]


def test_candidates_prefer_more_windows_then_higher_ratios():
    windows = [_window(1.0, 3), _window(1.1, 3), _window(5.0, 90), _window(5.1, 90)]
    found = candidates(windows, agree_s=0.25, eligible_at=lambda off: [0, 1, 2, 3])
    assert found[0].members == (2, 3)  # two each: the higher median ratio wins
    windows.append(_window(1.05, 2.5))
    found = candidates(windows, agree_s=0.25, eligible_at=lambda off: [0, 1, 2, 3, 4])
    assert found[0].members == (0, 1, 4)  # three beat two, whatever the ratio


@pytest.mark.parametrize(
    ("members", "eligible", "ratio", "confident"),
    [
        (2, 2, 2.0, True),  # every threshold met exactly
        (1, 1, 50.0, False),  # one window is not a consensus
        (3, 5, 9.0, True),  # 60 %
        (3, 6, 9.0, False),  # 50 %
        (4, 4, 1.99, False),  # agreeing windows, but no clear peak
    ],
)
def test_candidate_confidence_thresholds(members, eligible, ratio, confident):
    candidate = Candidate(
        offset_s=1.0,
        members=tuple(range(members)),
        eligible=tuple(range(eligible)),
        median_ratio=ratio,
    )
    assert candidate.confident(SyncSettings()) is confident


# ---------------------------------------------------------------- drift fit and frames


def test_theil_sen_ignores_a_wrong_peak():
    t = [float(i) for i in range(20)]
    y = [1.0 + 100e-6 * ti for ti in t]
    y[7] += 0.4  # a window that locked onto another peak
    fit = theil_sen(t, y)
    assert fit.slope == pytest.approx(100e-6, rel=1e-9)
    assert fit.intercept == pytest.approx(1.0, abs=1e-12)
    assert 7 not in fit.inliers and len(fit.inliers) == 19
    assert fit.residual_rms_s == pytest.approx(0.0, abs=1e-12)
    assert fit.at(10.0) == pytest.approx(1.001)
    with pytest.raises(ValueError):
        theil_sen([1.0, 1.0], [0.0, 1.0])


@pytest.mark.parametrize(
    ("offset", "fps", "frames", "exact", "error_ms"),
    [
        (2.0, Fraction(24000, 1001), 48, 47.952048, 2.0),
        (2.0, Fraction(25), 50, 50.0, 0.0),
        (1.23456, Fraction(25), 31, 30.864, 5.44),
        (1.23456, Fraction(24000, 1001), 30, 29.599840, 16.69),
        (-3.5, Fraction(24000, 1001), -84, -83.916084, -3.5),
        (-3.5, Fraction(25), -87, -87.5, 20.0),  # halves round up, towards +inf
        (0.5, Fraction(1), 1, 0.5, 500.0),
        (-0.5, Fraction(1), 0, -0.5, 500.0),
    ],
)
def test_to_frames(offset, fps, frames, exact, error_ms):
    got_frames, got_exact, got_error = to_frames(offset, fps)
    assert got_frames == frames
    assert got_exact == pytest.approx(exact, abs=1e-6)
    assert got_error == pytest.approx(error_ms, abs=0.01)
    assert got_error == pytest.approx((frames / float(fps) - offset) * 1000, abs=1e-9)


# ---------------------------------------------------------------- the confirm step


def test_anchors_that_agree_but_are_not_confirmed_are_low_confidence():
    # The other file shares only the three anchor spans with the reference; between them it is
    # unrelated noise. The anchors agree perfectly, but most consistency windows find nothing,
    # so the offset is not trusted.
    ref = noise(50.0, 8)
    other = noise(50.0, 9)
    settings = SyncSettings(anchors=3, anchor_s=5.0)
    for start in (0, int(22.5 * RATE), 45 * RATE):
        other[start : start + 5 * RATE] = ref[start : start + 5 * RATE]
    pair = analyse_pair(
        np, Mono(ref, RATE, 0.0, 0), Mono(other, RATE, 0.0, 0), fps=Fraction(25),
        ref_origin=0, settings=settings,
    )  # fmt: skip
    assert pair["coarse"]["windows_agreeing"] == 3 and pair["coarse"]["agreement"] == 1.0
    assert pair["status"] == "low-confidence"
    assert "consistency windows confirm" in pair["reason"]
    assert pair["drift"]["confirmed"] < 0.5
    assert "offset_s" not in pair and "suggested_recording_origin" not in pair
    whole = analyse_pair(
        np, Mono(ref, RATE, 0.0, 0), Mono(ref.copy(), RATE, 0.0, 0), fps=Fraction(25),
        ref_origin=0, settings=settings,
    )  # fmt: skip
    assert whole["status"] == "ok" and whole["drift"]["confirmed"] == 1.0


# ---------------------------------------------------------------- settings


@pytest.mark.parametrize(
    "bad",
    [
        {"max_offset_s": 0},
        {"max_offset_s": math.inf},
        {"min_agreement": 1.5},
        {"min_ratio": -1},
        {"anchors": 0},
        {"band_hz": (3500.0, 150.0)},
        {"band_hz": (150.0, 4000.0)},
        {"silence_dbfs": 0.0},
        {"timeout_s": math.nan},
    ],
)
def test_bad_settings_are_refused(bad):
    with pytest.raises(ValueError):
        SyncSettings(**bad)


def test_ratio_is_capped_for_json():
    window = noise(2.0, 10)
    other = np.zeros(4 * RATE, dtype=np.float32)
    other[RATE : 3 * RATE] = window
    peak = search(window, other, RATE, RATE)  # a single lag: nothing to compare against
    assert peak.ratio == RATIO_CAP and math.isfinite(peak.ratio)
