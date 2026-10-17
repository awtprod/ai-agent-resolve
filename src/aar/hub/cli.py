"""The ``aar hub`` subcommand group.

Other slices register their own groups the same way: expose ``register(subparsers)``
and let the top-level parser call it. The MCP SDK is only imported when the server is
actually started, so the core CLI keeps working without the optional ``hub`` extra.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``hub`` subcommand group on the top-level parser."""
    hub = subparsers.add_parser("hub", help="Run and inspect the project/event/work-graph hub.")
    hub_sub = hub.add_subparsers(dest="hub_command", required=True)

    serve = hub_sub.add_parser("serve", help="Serve the read-only hub over MCP (streamable HTTP).")
    serve.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
    serve.add_argument("--artifacts", required=True, help="Path to the artifact store directory.")
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1; the v1 hub is tailnet-only).",
    )
    serve.add_argument("--port", type=int, default=8787, help="Port to bind (default: 8787).")
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit binding to a non-loopback interface (required for any non-loopback --host).",
    )
    serve.set_defaults(func=_serve)


def _serve(args: argparse.Namespace) -> int:
    if args.host not in _LOOPBACK and not args.allow_remote:
        print(
            f"refusing to bind to {args.host!r} without --allow-remote: the v1 hub has no auth "
            "and is tailnet-only (default bind is 127.0.0.1)",
            file=sys.stderr,
        )
        return 2

    # Import here so the core CLI does not require the optional 'hub' extra (the MCP SDK).
    from aar.hub.artifacts import ArtifactStore
    from aar.hub.events import EventLog
    from aar.hub.mcp_server import build_server
    from aar.hub.registry import Registry

    # The read surface never writes, so open the log read-only and the artifact store
    # without creating anything (both must already exist; writers create them). A bad
    # --db or --artifacts is reported and exits 2 rather than raising a traceback or
    # leaving a stray empty file/directory behind.
    try:
        log = EventLog(args.db, read_only=True)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        # The exception message already names the path and the reason; don't re-prefix.
        print(f"aar hub serve: {exc}", file=sys.stderr)
        return 2
    try:
        artifacts = ArtifactStore(args.artifacts, log, create=False)
    except OSError as exc:
        print(f"aar hub serve: {exc}", file=sys.stderr)
        return 2
    registry = Registry(log)
    server = build_server(log, registry, artifacts)
    server.run(transport="streamable-http", host=args.host, port=args.port)
    return 0
