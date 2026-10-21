"""The runtime dependency pins in pyproject.toml (S13a-f, validator finding F8)."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def runtime_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return [dep.replace(" ", "") for dep in data["project"]["dependencies"]]


def test_pydantic_is_pinned_below_the_next_major_version() -> None:
    """The plan models lean on pydantic 2 behaviour (error shapes, JSON Schema output)."""
    pins = [dep for dep in runtime_dependencies() if dep.lower().startswith("pydantic")]
    assert pins == ["pydantic>=2,<3"]


def test_the_installed_pydantic_is_inside_the_pin() -> None:
    assert int(version("pydantic").split(".")[0]) == 2
