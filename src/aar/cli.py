"""Command line entry point for the ``aar`` tool."""

from __future__ import annotations

import argparse

from aar import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Subcommand groups are registered here, one per subpackage, as those slices land.
    """
    parser = argparse.ArgumentParser(
        prog="aar",
        description="AI-Agent-Resolve: a companion app for DaVinci Resolve Studio.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits for --version, --help and usage errors; surface its code as a return value.
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    parser.print_help()
    return 0
