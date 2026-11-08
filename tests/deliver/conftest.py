"""Shared fixtures for the ``aar.deliver`` tests: real synthetic sources generated once."""

from __future__ import annotations

from pathlib import Path

import pytest
from deliver_support import make_source

from aar.deliver import service


@pytest.fixture(autouse=True)
def _allow_tmp_workdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the forbidden-workdir roots for the delivery tests, which use pytest's ``tmp_path`` as
    the work dir -- and that is under the system temp root (``/tmp``) with the default basetemp CI
    uses. Production keeps the real roots; the dedicated F8 tests restore them explicitly to assert
    the refusal."""
    monkeypatch.setattr(service, "FORBIDDEN_WORKDIR_ROOTS", ())


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
