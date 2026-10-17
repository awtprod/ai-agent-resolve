"""Frame-rate arithmetic for appends across rates."""

from __future__ import annotations

import math
from fractions import Fraction

# Resolve reports NTSC-family rates as rounded floats (29.97, 23.976); these are the exact ones.
_EXACT = {
    23.976: Fraction(24000, 1001),
    29.97: Fraction(30000, 1001),
    47.952: Fraction(48000, 1001),
    59.94: Fraction(60000, 1001),
    119.88: Fraction(120000, 1001),
}


def exact_rate(fps: float) -> Fraction:
    """The exact rate behind a frame rate as Resolve reports it."""
    if fps <= 0:
        raise ValueError(f"frame rate must be positive, got {fps}")
    for nominal, exact in _EXACT.items():
        if abs(fps - nominal) < 1e-6:
            return exact
    return Fraction(str(fps))


def converted_length(frames: int, source_fps: float, timeline_fps: float) -> int:
    """Timeline frames that ``frames`` source frames should fill: the exact converted length,
    rounded half up. Equal rates convert one to one."""
    src, tl = exact_rate(source_fps), exact_rate(timeline_fps)
    if src == tl:
        return frames
    return math.floor(frames * tl / src + Fraction(1, 2))
