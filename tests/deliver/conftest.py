"""Shared fixtures for the ``aar.deliver`` tests: real synthetic sources generated once."""

from __future__ import annotations

from pathlib import Path

import pytest
from deliver_support import make_source


@pytest.fixture(scope="session")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A read-only "edit PC" directory holding small real renders at several shapes.

    Generated once for the session (ffmpeg is slow). The fake edit host serves them by basename.
    """
    root = tmp_path_factory.mktemp("edit_pc")
    make_source(root / "render_1080p.mp4", width=1920, height=1080, duration=2.0, video_kbps=800)
    make_source(root / "render_720p.mp4", width=1280, height=720, duration=2.0, video_kbps=300)
    make_source(root / "render_portrait.mp4", width=1080, height=1920, duration=2.0, video_kbps=800)
    make_source(root / "render_360p.mp4", width=640, height=360, duration=2.0, video_kbps=1200)
    return root
