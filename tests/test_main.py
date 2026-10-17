"""Tests for the ``python -m aar`` entry point."""

from __future__ import annotations

import subprocess
import sys


def test_python_m_aar_runs_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aar", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "aar 0.0.1"
