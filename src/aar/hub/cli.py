"""The ``aar hub`` subcommand group.

Other slices register their own groups the same way: expose ``register(subparsers)``
and let the top-level parser call it. The MCP SDK is only imported when the server is
actually started, so the core CLI keeps working without the optional ``hub`` extra.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_WILDCARD = frozenset({"0.0.0.0", "::"})


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

    console = hub_sub.add_parser(
        "console",
        help="Serve the human supervision console (reviews, leases, approvals, STOP).",
    )
    console.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
    console.add_argument("--artifacts", required=True, help="Path to the artifact store directory.")
    console.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1; reach it remotely via tailscale serve).",
    )
    console.add_argument("--port", type=int, default=8788, help="Port to bind (default: 8788).")
    console.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit binding to a non-loopback interface (required for any non-loopback --host).",
    )
    console.add_argument(
        "--operator",
        help="The human acting through this console. Without it (or trusted Tailscale "
        "headers) the console is read-only.",
    )
    console.add_argument(
        "--trust-tailscale-headers",
        action="store_true",
        help="Take the human from the Tailscale-User-Login header, honoured only for "
        "requests from loopback (tailscale serve). Any local process can send it.",
    )
    console.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="NAME",
        help="Another Host name the console answers to (for example the tailscale serve "
        "name). Repeatable. Loopback names are always allowed.",
    )
    console.set_defaults(func=_console)

    review = hub_sub.add_parser("review", help="Request, annotate and inspect human reviews.")
    review_sub = review.add_subparsers(dest="review_command", required=True)

    request = review_sub.add_parser("request", help="Open a review of a work-graph node.")
    _add_db(request)
    request.add_argument("--artifacts", required=True, help="Path to the artifact store.")
    request.add_argument("--project", required=True, help="Project id (prj_...).")
    request.add_argument("--subject", required=True, help="Work-graph node id under review.")
    request.add_argument(
        "--render",
        action="append",
        required=True,
        metavar="ARTIFACT_ID",
        help="Render artifact id (64 hex). Repeatable.",
    )
    request.add_argument("--title", required=True, help="What the human is reviewing.")
    request.add_argument("--timeline", help="Timeline name, for context.")
    request.add_argument("--fps", type=float, help="Frame rate of the render.")
    _add_actor(request)
    request.add_argument("--idempotency-key", help="Makes a retry return the same review.")
    request.set_defaults(func=_review_request)

    note = review_sub.add_parser("note", help="Add a note at a frame of a review's render.")
    _add_db(note)
    note.add_argument("--project", required=True, help="Project id (prj_...).")
    note.add_argument("--review", required=True, help="Review id.")
    note.add_argument("--frame", type=int, required=True, help="0-based frame in the render.")
    note.add_argument("--timecode", help="Label, e.g. 01:00:12:05 (derived from --fps if omitted).")
    note.add_argument("--text", required=True, help="The note.")
    _add_actor(note)
    note.add_argument("--idempotency-key", help="Makes a retry a no-op.")
    note.set_defaults(func=_review_note)

    listing = review_sub.add_parser("list", help="List reviews (open ones by default).")
    _add_db(listing)
    listing.add_argument("--project", help="Only this project.")
    listing.add_argument("--all", action="store_true", help="Include closed reviews.")
    listing.set_defaults(func=_review_list)

    show = review_sub.add_parser("show", help="Show one review with its notes and decision.")
    _add_db(show)
    show.add_argument("--project", required=True, help="Project id (prj_...).")
    show.add_argument("review_id", help="Review id.")
    show.set_defaults(func=_review_show)


def _add_db(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="Path to the SQLite event-log database.")


def _add_actor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor",
        required=True,
        metavar="KIND:ID",
        help="Who is acting, e.g. agent:agent-1 (kinds: human, agent, node).",
    )


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
        log.close()  # release the DB handle we just opened before bailing out
        print(f"aar hub serve: {exc}", file=sys.stderr)
        return 2
    registry = Registry(log)
    server = build_server(log, registry, artifacts)
    server.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


def _fail(code: str, message: str, status: int = 2) -> int:
    print(json.dumps({"error": {"code": code, "message": message}}), file=sys.stderr)
    return status


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


class _DatabaseUnavailable(Exception):
    """The hub database is missing or is not an aar event log."""


class _StoreUnavailable(Exception):
    """The artifact store directory is missing."""


# Everything EventLog raises for a path that is not a usable hub database: missing
# (FileNotFoundError), a directory or a non-SQLite file (OSError, or sqlite3.Error on
# the write path), ':memory:' (ValueError) and a SQLite file without the hub schema
# (RuntimeError).
_DB_OPEN_ERRORS = (OSError, ValueError, RuntimeError, sqlite3.Error)


def _open_existing(db: str, *, read_only: bool = False) -> Any:
    """Open an existing hub database, without creating or modifying anything else.

    A write open is preceded by a read-only one, so a SQLite file that is not a hub
    database is rejected by the schema check before the write open could switch its
    journal mode to WAL.
    """
    from aar.hub.events import EventLog

    path = Path(db)
    if db == ":memory:" or not path.is_file():
        what = "a directory" if path.is_dir() else "missing"
        raise _DatabaseUnavailable(f"no hub database at {db!r} ({what})")
    try:
        checked = EventLog(db, read_only=True)
        if read_only:
            return checked
        checked.close()
        return EventLog(db, create=False)
    except _DB_OPEN_ERRORS as exc:
        raise _DatabaseUnavailable(f"cannot open hub database {db!r}: {exc}") from exc


def _open_store(root: str, log: Any) -> Any:
    """Open an existing artifact store; a missing directory is an error, never created."""
    from aar.hub.artifacts import ArtifactStore

    if not Path(root).is_dir():
        raise _StoreUnavailable(f"no artifact store directory at {root!r}")
    return ArtifactStore(root, log)


def _parse_actor(value: str) -> Any:
    from aar.hub.events import Actor

    kind, sep, ident = value.partition(":")
    if not sep:
        raise ValueError(f"--actor must look like KIND:ID, got {value!r}")
    return Actor(kind, ident)


def _console(args: argparse.Namespace) -> int:
    if args.host not in _LOOPBACK and not args.allow_remote:
        return _fail(
            "remote_bind_refused",
            f"refusing to bind to {args.host!r} without --allow-remote: the console is "
            "tailnet-only (default bind is 127.0.0.1; use tailscale serve to reach it)",
        )

    from aar.hub.common import InvalidInput
    from aar.hub.console import ConsoleConfig, build_app

    allowed = list(args.allowed_host)
    if args.host not in _LOOPBACK and args.host not in _WILDCARD:
        allowed.append(args.host)
    try:
        config = ConsoleConfig(
            operator=args.operator,
            trust_tailscale_headers=args.trust_tailscale_headers,
            allowed_hosts=tuple(allowed),
        )
    except InvalidInput as exc:
        return _fail("invalid_input", str(exc))
    try:
        log = _open_existing(args.db)
    except _DatabaseUnavailable as exc:
        return _fail("database_unavailable", str(exc))
    try:
        store = _open_store(args.artifacts, log)
    except _StoreUnavailable as exc:
        log.close()
        return _fail("artifact_store_unavailable", str(exc))

    import uvicorn

    app = build_app(log, store, config)
    if config.operator is None and not config.trust_tailscale_headers:
        print("console is read-only: no --operator and no trusted headers", file=sys.stderr)
    # proxy_headers=False: the peer address must be the real TCP peer, because trusting
    # Tailscale headers depends on it being loopback.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=False,
        server_header=False,
        log_level="info",
    )
    return 0


def _with_review_errors(func: Any) -> Any:
    """Run a review command, turning expected failures into a JSON error and exit 2."""

    def run(args: argparse.Namespace) -> int:
        from aar.hub.common import HumanRequired
        from aar.hub.events import IdempotencyConflict
        from aar.hub.reviews import ReviewError

        try:
            return func(args)
        except _DatabaseUnavailable as exc:
            return _fail("database_unavailable", str(exc))
        except _StoreUnavailable as exc:
            return _fail("artifact_store_unavailable", str(exc))
        except HumanRequired as exc:
            return _fail("human_required", str(exc))
        except IdempotencyConflict as exc:
            return _fail("idempotency_conflict", str(exc))
        except ReviewError as exc:
            return _fail(type(exc).__name__, str(exc))
        except ValueError as exc:
            return _fail("invalid_input", str(exc))

    return run


@_with_review_errors
def _review_request(args: argparse.Namespace) -> int:
    from aar.hub.reviews import request_review

    actor = _parse_actor(args.actor)
    log = _open_existing(args.db)
    try:
        review = request_review(
            log,
            _open_store(args.artifacts, log),
            args.project,
            args.subject,
            args.render,
            actor,
            title=args.title,
            timeline=args.timeline,
            fps=args.fps,
            idempotency_key=args.idempotency_key,
        )
    finally:
        log.close()
    _print(review.to_dict())
    return 0


@_with_review_errors
def _review_note(args: argparse.Namespace) -> int:
    from aar.hub.reviews import add_note

    actor = _parse_actor(args.actor)
    log = _open_existing(args.db)
    try:
        note = add_note(
            log,
            args.project,
            args.review,
            args.frame,
            args.text,
            actor,
            timecode=args.timecode,
            idempotency_key=args.idempotency_key,
        )
    finally:
        log.close()
    _print(note.to_dict())
    return 0


@_with_review_errors
def _review_list(args: argparse.Namespace) -> int:
    from aar.hub.reviews import list_reviews

    log = _open_existing(args.db, read_only=True)
    try:
        found = list_reviews(log, args.project, pending_only=not args.all)
    finally:
        log.close()
    _print({"reviews": [review.to_dict() for review in found]})
    return 0


@_with_review_errors
def _review_show(args: argparse.Namespace) -> int:
    from aar.hub.reviews import get_review

    log = _open_existing(args.db, read_only=True)
    try:
        review = get_review(log, args.project, args.review_id)
    finally:
        log.close()
    if review is None:
        return _fail("review_not_found", f"no review {args.review_id!r} in {args.project!r}", 1)
    _print(review.to_dict())
    return 0
