"""Shared pytest configuration."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from fractions import Fraction
from typing import Any

import pytest

#: Python 3.12's own ``Fraction.__format__`` (None on 3.11, where Fraction takes no format spec).
_REAL_FRACTION_FORMAT: Callable[[Fraction, str], str] | None = Fraction.__dict__.get("__format__")


def _format_as_on_311(self: Fraction, spec: str) -> str:
    if spec:
        raise TypeError(
            "unsupported format string passed to Fraction.__format__ (Python 3.11's Fraction "
            "takes no format spec; write it with aar.plan.render.fixed.decimal)"
        )
    return str(self)


@pytest.fixture(autouse=True, scope="session")
def _fraction_format_as_on_311() -> Iterator[None]:
    """The whole suite runs with Fraction formatted as Python 3.11 does (round 4: a format spec on
    a Fraction raised on CI's 3.11 and nowhere here), so such a spec fails here too."""
    if _REAL_FRACTION_FORMAT is None:  # 3.11 itself
        yield
        return
    Fraction.__format__ = _format_as_on_311  # type: ignore[method-assign]
    try:
        yield
    finally:
        Fraction.__format__ = _REAL_FRACTION_FORMAT  # type: ignore[method-assign]


@pytest.fixture
def fraction_format_312() -> Any:
    """3.12's real Fraction formatter, for comparing against it (skips on 3.11)."""
    if _REAL_FRACTION_FORMAT is None:
        pytest.skip("Fraction takes format specs only from Python 3.12")
    return _REAL_FRACTION_FORMAT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``live`` unless ``AAR_LIVE=1`` is set."""
    if os.environ.get("AAR_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="needs a running DaVinci Resolve (set AAR_LIVE=1)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
