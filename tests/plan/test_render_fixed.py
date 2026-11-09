"""S18b round 4: exact fixed-point text for a Fraction, identical to Python 3.12's format(), and
the suite's guard that no Fraction gets a format spec (Python 3.11 has none)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from aar.plan.model import Fps
from aar.plan.render.fixed import decimal
from aar.plan.render.graph import _clock, seconds

EDGES = [
    Fraction(0),
    Fraction(-1, 10**9),  # rounds to zero, keeps its sign: "-0.000000"
    Fraction(1, 2),
    Fraction(-1, 2),
    Fraction(3, 2),
    Fraction(5, 2),
    Fraction(-5, 2),
    Fraction(1, 8),  # an exact half at 2 places: 0.125 -> 0.12
    Fraction(3, 8),  # 0.375 -> 0.38
    Fraction(-3, 8),
    Fraction(5, 16),  # an exact half at 4 places
    Fraction(1, 3),
    Fraction(-2, 3),
    Fraction(1001, 24000),
    Fraction(1001 * 72, 24000) - Fraction(1001, 48000),
    Fraction(10**40 + 1, 7),  # a huge numerator
    Fraction(-(10**30) - 5, 10),
    Fraction(999999, 1000000),  # carries into the units
    Fraction(-9999995, 10**7),  # an exact half that carries: -1.000000 or -0.999999?
    Fraction(12345678912345678912345, 10**12),
]


@pytest.mark.parametrize("places", [0, 1, 2, 3, 4, 6, 9, 12])
def test_decimal_is_python_312_format_on_the_edges(fraction_format_312: Any, places: int) -> None:
    for x in EDGES:
        assert decimal(x, places) == fraction_format_312(x, f".{places}f"), (x, places)
    for n in (0, 7, -7, 10**25, -(10**25)):
        assert decimal(n, places) == fraction_format_312(Fraction(n), f".{places}f")


def test_decimal_is_python_312_format_on_random_fractions(fraction_format_312: Any) -> None:
    rng = random.Random(20261109)
    for _ in range(20000):
        den = rng.choice([1, 2, 4, 8, 10, 16, 25, 1000, 1001, 24000, 30000, 48000]) * rng.randint(
            1, 50
        )
        num = rng.randint(-(10 ** rng.randint(1, 30)), 10 ** rng.randint(1, 30))
        places = rng.randint(0, 12)
        x = Fraction(num, den)
        assert decimal(x, places) == fraction_format_312(x, f".{places}f"), (x, places)
    for _ in range(5000):  # exact halves at the last place, both signs
        places = rng.randint(0, 9)
        x = Fraction(2 * rng.randint(-(10**12), 10**12) + 1, 2 * 10**places)
        assert decimal(x, places) == fraction_format_312(x, f".{places}f"), (x, places)


def test_decimal_refuses_negative_places() -> None:
    with pytest.raises(ValueError):
        decimal(Fraction(1, 3), -1)


def test_the_graph_clock_and_durations_need_no_format_spec() -> None:
    """The two Fraction sites CI on Python 3.11 failed at (``_clock`` and ``seconds``): the suite's
    guard emulates 3.11, so these run without any Fraction format spec."""
    assert _clock(Fraction(71571, 24000)) == "2.982125"
    assert _clock(Fraction(-1, 3)) == "0.000000"  # a seek is never before the file's start
    assert seconds(Fps(num=24000, den=1001), 15) == "0.625625"
    assert seconds(Fps(num=25, den=1), 0) == "0"


def test_the_suite_formats_fractions_as_python_311_does() -> None:
    """The guard is on: a format spec on a Fraction raises, as on 3.11; no spec is fine."""
    with pytest.raises(TypeError, match="Fraction"):
        f"{Fraction(1, 3):.3f}"
    assert f"{Fraction(1, 3)}" == "1/3" and format(Fraction(1, 3), "") == "1/3"
