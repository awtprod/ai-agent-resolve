"""Exact rational time for FCPXML: integer frames <-> rational seconds, with no float anywhere.

FCPXML writes every time as a rational number of seconds (``1001/24000s``, ``5s``). A plan counts
integer frames at its ``fps``. One frame lasts ``den/num`` seconds, so ``n`` frames are
``n * den / num`` seconds, written over the frame duration's own denominator, the way Final Cut Pro
writes them (``24024/24000s``, ``86486400/24000s``) and reduced to whole seconds when exact.

Only the rates a Resolve timeline can be set to are accepted. A rate outside the table, such as
``23976/1000`` (which is not 23.976 NTSC), would be snapped by the importer and drift, so it is
refused rather than written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from aar.plan.export.errors import ExportError
from aar.plan.model import Fps

#: Timeline rates FCPXML export accepts, with the suffix Final Cut Pro uses in format names.
SUPPORTED_RATES: dict[Fraction, str] = {
    Fraction(24000, 1001): "2398",
    Fraction(24): "24",
    Fraction(25): "25",
    Fraction(30000, 1001): "2997",
    Fraction(30): "30",
    Fraction(48000, 1001): "4795",
    Fraction(48): "48",
    Fraction(50): "50",
    Fraction(60000, 1001): "5994",
    Fraction(60): "60",
}

_TIME = re.compile(r"(0|[1-9][0-9]*)(?:/([1-9][0-9]*))?s")
_TIMECODE = re.compile(r"([0-9]{2}):([0-9]{2}):([0-9]{2}):([0-9]{2})")
_MAX_NUMERATOR = 2**63 - 1
_MAX_DENOMINATOR = 2**32 - 1


def _rate_text(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}" if rate.denominator != 1 else str(rate.numerator)


@dataclass(frozen=True)
class Timebase:
    """One plan's frame rate, with exact conversions between frames and FCPXML time values."""

    rate: Fraction

    @classmethod
    def of(cls, fps: Fps) -> Timebase:
        """The timebase for a plan rate; refuses a rate the format can't represent exactly."""
        rate = Fraction(fps.num, fps.den)
        if rate not in SUPPORTED_RATES:
            known = ", ".join(_rate_text(r) for r in SUPPORTED_RATES)
            raise ExportError(
                "fps_not_representable",
                f"plan fps {fps.num}/{fps.den} is not a timeline rate FCPXML export supports",
                f"use one of {known} (NTSC rates are N*1000/1001, e.g. 24000/1001, not 23976/1000)",
                fps=f"{fps.num}/{fps.den}",
            )
        return cls(rate)

    @property
    def frame_duration(self) -> Fraction:
        """Seconds per frame, e.g. 1001/24000."""
        return 1 / self.rate

    @property
    def nominal(self) -> int:
        """Whole frames per second used for non-drop timecode (24 for 23.976)."""
        return round(self.rate)

    @property
    def suffix(self) -> str:
        """Final Cut Pro's format-name suffix, e.g. ``2398``."""
        return SUPPORTED_RATES[self.rate]

    @property
    def rate_text(self) -> str:
        return _rate_text(self.rate)

    def time(self, frames: int) -> str:
        """``frames`` as an FCPXML time value: ``N/Ds`` over the frame duration's denominator, or
        ``Ns`` when it is a whole number of seconds."""
        if frames < 0:
            raise ValueError(f"FCPXML times here are never negative (got {frames} frames)")
        step = self.frame_duration
        numerator = frames * step.numerator
        if numerator % step.denominator == 0:
            return f"{numerator // step.denominator}s"
        return f"{numerator}/{step.denominator}s"

    def frames(self, value: str) -> int:
        """Parse an FCPXML time value back to whole frames; ``ValueError`` when it is not a
        rational time or not on a frame boundary of this rate."""
        seconds = parse_time(value)
        count = seconds / self.frame_duration
        if count.denominator != 1:
            raise ValueError(
                f"time {value!r} is not a whole number of frames at {self.rate_text} fps"
            )
        return count.numerator

    def timecode_frames(self, timecode: str) -> int:
        """Non-drop ``HH:MM:SS:FF`` at the nominal rate, as a frame count."""
        match = _TIMECODE.fullmatch(timecode)
        if match is None:
            raise ValueError(f"timecode {timecode!r} is not HH:MM:SS:FF")
        hours, minutes, seconds, frames = (int(part) for part in match.groups())
        if minutes > 59 or seconds > 59 or frames >= self.nominal:
            raise ValueError(
                f"timecode {timecode!r} is out of range at {self.nominal} frames per second"
            )
        return ((hours * 60 + minutes) * 60 + seconds) * self.nominal + frames


def parse_time(value: str) -> Fraction:
    """An FCPXML rational time (``N/Ds`` or ``Ns``) as exact seconds; decimals are refused."""
    match = _TIME.fullmatch(value)
    if match is None:
        raise ValueError(f"{value!r} is not an FCPXML rational time such as '1001/24000s' or '5s'")
    numerator = int(match.group(1))
    denominator = int(match.group(2) or 1)
    if numerator > _MAX_NUMERATOR or denominator > _MAX_DENOMINATOR:
        raise ValueError(f"time {value!r} exceeds FCPXML's 64-bit/32-bit rational limits")
    return Fraction(numerator, denominator)


def frame_duration_text(timebase: Timebase) -> str:
    """The format's ``frameDuration``: one frame, e.g. ``1001/24000s`` or ``1/25s``."""
    return timebase.time(1)
