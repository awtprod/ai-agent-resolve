"""Shared pytest configuration."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``live`` unless ``AAR_LIVE=1`` is set."""
    if os.environ.get("AAR_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="needs a running DaVinci Resolve (set AAR_LIVE=1)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
