"""Exact fixed-point text for a :class:`~fractions.Fraction` (S18b round 4).

Python 3.11's ``Fraction`` takes no format spec (``f"{x:.6f}"`` raises ``TypeError``; specs
arrived in 3.12) and the package supports 3.11, so the render writes every Fraction it puts in
an ffmpeg argument or a filter graph through :func:`decimal`: integer arithmetic only, byte for
byte what 3.12's ``format(x, ".Nf")`` gives (rounding half to even, and a negative value that
rounds to zero keeps its sign, ``-0.000``), so graphs and goldens are unchanged."""

from __future__ import annotations

from fractions import Fraction


def decimal(value: Fraction | int, places: int) -> str:
    """``value`` with ``places`` digits after the point, rounded half to even."""
    if places < 0:
        raise ValueError("places must be 0 or more")
    q = Fraction(value)
    whole, rest = divmod(abs(q.numerator) * 10**places, q.denominator)
    if 2 * rest > q.denominator or (2 * rest == q.denominator and whole % 2):
        whole += 1
    digits = str(whole).rjust(places + 1, "0")
    text = f"{digits[:-places]}.{digits[-places:]}" if places else digits
    return f"-{text}" if q < 0 else text
