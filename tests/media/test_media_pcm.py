"""quiet_runs on synthetic PCM: sample-exact run lengths, levels, channels, edges."""

from __future__ import annotations

import math
import random
from array import array

import pytest

from aar.media import Pcm, quiet_runs

RATE = 48000


def pcm_from(samples: list[float], *, channels: int = 1, start_s: float = 0.0) -> Pcm:
    return Pcm(rate=RATE, channels=channels, start_s=start_s, samples=array("f", samples))


def tone(n: int, amp: float = 0.5, freq: float = 1000.0, phase0: int = 0) -> list[float]:
    return [amp * math.sin(2 * math.pi * freq * (i + phase0) / RATE) for i in range(n)]


def gap_signal(gap_samples: int, *, lead: int = 4800, tail: int = 4800) -> list[float]:
    return tone(lead) + [0.0] * gap_samples + tone(tail, phase0=lead + gap_samples)


@pytest.mark.parametrize(("gap_ms", "ok"), [(60, True), (59, False), (61, True), (30, False)])
def test_digital_gap_measures_its_own_length(gap_ms, ok):
    scan = quiet_runs(pcm_from(gap_signal(gap_ms * 48)), threshold_dbfs=-40.0)
    best = scan.longest
    assert best is not None
    # The run is the gap plus at most a few samples where the tone passes through zero.
    assert best.duration_s * 1000 == pytest.approx(gap_ms, abs=0.25)
    assert (best.duration_s * 1000 >= 60) is ok
    assert best.start_s == pytest.approx(4800 / RATE, abs=0.0002)
    assert scan.quietest_dbfs == -math.inf


def test_window_shorter_than_the_gap_only():
    scan = quiet_runs(pcm_from(gap_signal(5 * 48)), threshold_dbfs=-40.0, window_s=0.010)
    assert scan.runs == []
    assert scan.quietest_dbfs is not None and scan.quietest_dbfs > -40


def test_room_tone_below_threshold_is_quiet_and_above_is_not():
    rng = random.Random(7)

    def noise(rms_db: float, n: int) -> list[float]:
        sigma = 10 ** (rms_db / 20)
        return [rng.gauss(0.0, sigma) for _ in range(n)]

    quiet = quiet_runs(pcm_from(tone(2400) + noise(-46, 4800) + tone(2400)), threshold_dbfs=-40)
    assert quiet.longest is not None
    assert quiet.longest.duration_s == pytest.approx(0.1, abs=0.002)
    assert -48 < quiet.longest.level_dbfs < -40
    loud = quiet_runs(pcm_from(noise(-35, 9600)), threshold_dbfs=-40)
    assert loud.runs == []
    assert loud.quietest_dbfs == pytest.approx(-35, abs=1.5)


def test_level_is_the_loudest_channel():
    left = [0.0] * 4800
    right = tone(4800, amp=0.2)
    interleaved = [x for pair in zip(left, right, strict=True) for x in pair]
    assert quiet_runs(pcm_from(interleaved, channels=2), threshold_dbfs=-40).runs == []
    both_quiet = [0.0] * 9600
    scan = quiet_runs(pcm_from(both_quiet, channels=2), threshold_dbfs=-40)
    assert scan.longest is not None and scan.longest.duration_s == pytest.approx(0.1)


@pytest.mark.parametrize(("level_db", "quiet"), [(-39.999, False), (-40.001, True)])
def test_threshold_is_strict_to_a_thousandth_of_a_db(level_db, quiet):
    # Samples are float32, so "exactly -40" is not representable; 0.001 dB either side is.
    level = 10 ** (level_db / 20)
    scan = quiet_runs(pcm_from([level] * 4800), threshold_dbfs=-40.0)
    assert bool(scan.runs) is quiet
    assert scan.quietest_dbfs == pytest.approx(level_db, abs=0.0005)


def test_runs_are_reported_in_source_time_and_separately():
    signal = tone(2400) + [0.0] * 4800 + tone(2400, phase0=7200) + [0.0] * 2400 + tone(480)
    scan = quiet_runs(pcm_from(signal, start_s=10.0), threshold_dbfs=-40)
    assert len(scan.runs) == 2
    first, second = scan.runs
    assert first.start_s == pytest.approx(10.05, abs=0.0002)
    assert first.duration_s == pytest.approx(0.1, abs=0.0003)
    assert second.duration_s == pytest.approx(0.05, abs=0.0003)
    assert scan.longest is first


def test_too_little_audio_reports_nothing():
    scan = quiet_runs(pcm_from([0.0] * 100), threshold_dbfs=-40)
    assert scan.runs == [] and scan.quietest_dbfs is None and scan.longest is None
    empty = quiet_runs(pcm_from([]), threshold_dbfs=-40)
    assert empty.scanned_s == 0
