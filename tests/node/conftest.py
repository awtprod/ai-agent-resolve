"""Node test configuration.

``AAR_TEST_DRAIN_DELAY=<seconds>`` makes every job-runner drain sleep that long before it
ticks the broker (the test-only hook ``aar.node.jobs.runner._drain_delay``). Set above the
production drain budget (0.5 s), it proves no test depends on a drain finishing inside the
wall-clock budget: the S7 determinism check runs the node suite with it set.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from aar.node.jobs import runner as job_runner


@pytest.fixture(autouse=True)
def _drain_delay_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    delay = os.environ.get("AAR_TEST_DRAIN_DELAY")
    if delay:
        monkeypatch.setattr(job_runner, "_drain_delay", float(delay))


@pytest.fixture(autouse=True)
def _close_job_worlds() -> Iterator[None]:
    """Close every JobWorld a test built: one live node runtime per node id (P5)."""
    from .jobs_support import LIVE_WORLDS

    yield
    while LIVE_WORLDS:
        LIVE_WORLDS.pop().close()
