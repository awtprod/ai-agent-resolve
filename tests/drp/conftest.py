"""Fixtures for the ``aar.drp`` tests."""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"


@pytest.fixture
def fixture_path() -> Callable[[str], Path]:
    """Path of a bundled template, e.g. ``fixture_path("media-clip-h264.drp")``."""

    def get(name: str) -> Path:
        path = FIXTURES / name
        assert path.is_file(), f"missing fixture {name}"
        return path

    return get


@pytest.fixture
def make_zip(tmp_path: Path) -> Callable[..., Path]:
    """Write a zip with the given members (str or bytes) and return its path."""

    def build(name: str, members: dict[str, str | bytes]) -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for member, content in members.items():
                zf.writestr(member, content.encode() if isinstance(content, str) else content)
        return path

    return build
