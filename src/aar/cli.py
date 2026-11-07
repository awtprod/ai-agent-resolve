"""Command line entry point for the ``aar`` tool."""

from __future__ import annotations

import argparse

from aar import __version__, cli_io
from aar.analyze import cli as analyze_cli
from aar.drp import cli as drp_cli
from aar.host import cli as host_cli
from aar.hub import cli as hub_cli
from aar.node import cli as node_cli
from aar.plan import cli as plan_cli
from aar.qa import cli as qa_cli


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
    node_cli.register(groups)
    qa_cli.register(groups)
    host_cli.register(groups)
    analyze_cli.register(groups)
    # Keep the source-media store out of import-only/library paths; its CLI is needed
    # only while constructing the command inventory.
    from aar.media import cli as media_cli

    media_cli.register(groups)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse printed --version, --help or a usage error with bare print and is exiting;
        # flush the std streams safely so a full/closed/broken one gives 74/141 (never 120).
        code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
        return cli_io.finalize(code)
    func = getattr(args, "func", None)
    if func is not None:
        return int(func(args))
    parser.print_help()
    return cli_io.finalize(0)
