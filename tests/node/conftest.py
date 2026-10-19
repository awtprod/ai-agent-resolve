"""Node test configuration.

``AAR_TEST_DRAIN_DELAY=<seconds>`` makes every job-runner drain sleep that long before it
ticks the broker (the test-only hook ``aar.node.jobs.runner._drain_delay``). Set above the
production drain budget (0.5 s), it proves no test depends on a drain finishing inside the
wall-clock budget: the S7 determinism check runs the node suite with it set.
"""

from __future__ import annotations

import os

import pytest

from aar.node.jobs import runner as job_runner


@pytest.fixture(autouse=True)
def _drain_delay_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    delay = os.environ.get("AAR_TEST_DRAIN_DELAY")
    if delay:
        monkeypatch.setattr(job_runner, "_drain_delay", float(delay))
