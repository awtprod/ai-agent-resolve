"""Exact rational time conversions of the FCPXML exporter's timebase."""

from __future__ import annotations

from fractions import Fraction

import pytest

from aar.plan.export.errors import EXPORT_CODES, ExportError
from aar.plan.export.timebase import SUPPORTED_RATES, Timebase, parse_time
from aar.plan.model import Fps

R2398 = Fps(num=24000, den=1001)
R24 = Fps(num=24)
R25 = Fps(num=25)
R2997 = Fps(num=30000, den=1001)
R5994 = Fps(num=60000, den=1001)


def tb(fps: Fps) -> Timebase:
    return Timebase.of(fps)


@pytest.mark.parametrize("rate", list(SUPPORTED_RATES))
def test_supported_rates_accepted(rate: Fraction) -> None:
    got = Timebase.of(Fps(num=rate.numerator, den=rate.denominator))
    assert got.rate == rate


@pytest.mark.parametrize(
    ("fps", "rate"),
    [(Fps(num=48000, den=2002), Fraction(24000, 1001)), (Fps(num=50, den=2), Fraction(25))],
)
def test_unreduced_rates(fps: Fps, rate: Fraction) -> None:
    assert Timebase.of(fps).rate == rate


@pytest.mark.parametrize(
    "fps",
    [Fps(num=23976, den=1000), Fps(num=12), Fps(num=24001, den=1001), Fps(num=120)],
)
def test_unsupported_rates_refused(fps: Fps) -> None:
    with pytest.raises(ExportError) as info:
        Timebase.of(fps)
    assert info.value.code == "fps_not_representable"
    assert info.value.code in EXPORT_CODES


@pytest.mark.parametrize(
    ("fps", "text"),
    [
        (R2398, "1001/24000s"),
        (R2997, "1001/30000s"),
        (R24, "1/24s"),
        (R25, "1/25s"),
        (Fps(num=30), "1/30s"),
        (R5994, "1001/60000s"),
    ],
)
def test_frame_duration_text(fps: Fps, text: str) -> None:
    from aar.plan.export.timebase import frame_duration_text

    assert frame_duration_text(tb(fps)) == text


@pytest.mark.parametrize(
    ("fps", "frames", "text"),
    [
        (R2398, 0, "0s"),
        (R2398, 1, "1001/24000s"),
        (R2398, 24, "24024/24000s"),
        (R2398, 1000, "1001000/24000s"),
        (R2398, 24000, "1001s"),
        (R2398, 86400, "86486400/24000s"),
        (R25, 50, "2s"),
        (R25, 1, "1/25s"),
        (R2997, 108000, "108108000/30000s"),
    ],
)
def test_time_text(fps: Fps, frames: int, text: str) -> None:
    assert tb(fps).time(frames) == text


def test_negative_frames_refused() -> None:
    with pytest.raises(ValueError, match="never negative"):
        tb(R24).time(-1)


@pytest.mark.parametrize("rate", list(SUPPORTED_RATES))
def test_no_drift(rate: Fraction) -> None:
    t = Timebase.of(Fps(num=rate.numerator, den=rate.denominator))
    for n in (0, 1, 7, 1001, 10 * 3600 * t.nominal + 13):
        text = t.time(n)
        assert t.frames(text) == n
        assert parse_time(text) == n * Fraction(rate.denominator, rate.numerator)


def test_frames_zero() -> None:
    assert tb(R24).frames("0s") == 0


@pytest.mark.parametrize("value", ["1/48s", "1.5s", "5", "-1s", "-1/24s", "garbage", "", "1/0s"])
def test_frames_refuses(value: str) -> None:
    with pytest.raises(ValueError):
        tb(R24).frames(value)


@pytest.mark.parametrize("value", [f"{2**63}/1s", f"1/{2**32}s", f"1/{2**32 + 5}s"])
def test_parse_time_limits(value: str) -> None:
    with pytest.raises(ValueError, match="limits"):
        parse_time(value)


def test_parse_time_at_limits() -> None:
    assert parse_time(f"{2**63 - 1}/{2**32 - 1}s") == Fraction(2**63 - 1, 2**32 - 1)


@pytest.mark.parametrize(
    ("fps", "timecode", "frames"),
    [
        (R2398, "01:00:00:00", 86400),
        (R24, "01:00:00:00", 86400),
        (R25, "01:00:00:00", 90000),
        (R2997, "01:00:00:00", 108000),
        (R24, "00:00:01:05", 29),
    ],
)
def test_timecode_frames(fps: Fps, timecode: str, frames: int) -> None:
    assert tb(fps).timecode_frames(timecode) == frames


@pytest.mark.parametrize("timecode", ["00:00:00:24", "00:60:00:00", "1:00:00:00", "01:00:00;00"])
def test_timecode_refuses(timecode: str) -> None:
    with pytest.raises(ValueError):
        tb(R24).timecode_frames(timecode)
