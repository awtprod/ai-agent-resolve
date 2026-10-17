"""Command line entry point for the ``aar`` tool."""

from __future__ import annotations

import argparse

from aar import __version__
from aar.drp import cli as drp_cli
from aar.hub import cli as hub_cli
from aar.plan import cli as plan_cli


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Subcommand groups are registered here, one per subpackage, as those slices land.
    """
    parser = argparse.ArgumentParser(
        prog="aar",
        description="AI-Agent-Resolve: a companion app for DaVinci Resolve Studio.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    groups = parser.add_subparsers(dest="group", metavar="GROUP")
    plan_cli.register(groups)
    drp_cli.register(groups)
    hub_cli.register(groups)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits for --version, --help and usage errors; surface its code as a return value.
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    func = getattr(args, "func", None)
    if func is not None:
        return int(func(args))
    parser.print_help()
    return 0
