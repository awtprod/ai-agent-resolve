"""Tests for the ``aar`` command line entry point."""

from __future__ import annotations

import pytest

from aar import __version__
from aar.cli import main


def test_version_flag_prints_version_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == f"aar {__version__}"
    assert __version__ == "0.0.1"


def test_no_args_prints_usage_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: aar")
    assert "--version" in out


def test_unknown_option_returns_nonzero_and_reports_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--no-such-flag"]) == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: --no-such-flag" in err
