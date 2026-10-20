"""The ``aar hub`` subcommand group.

Other slices register their own groups the same way: expose ``register(subparsers)``
and let the top-level parser call it. The MCP SDK is only imported when the server is
actually started, so the core CLI keeps working without the optional ``hub`` extra.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from aar import cli_io

_PROG = "aar hub"
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_WILDCARD = frozenset({"0.0.0.0", "::"})


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``hub`` subcommand group on the top-level parser."""
    hub = subparsers.add_parser("hub", help="Run and inspect the project/event/work-graph hub.")
    hub_sub = hub.add_subparsers(dest="hub_command", required=True)

    init = hub_sub.add_parser(
        "init",
        help="Create a new, empty hub database and artifact store (refuses to overwrite).",
    )
    init.add_argument("--db", required=True, help="Path for the new SQLite event-log database.")
    init.add_argument(
        "--artifacts", required=True, help="Path for the new artifact store directory."
    )
    init.set_defaults(func=_init)

    serve = hub_sub.add_parser(
        "serve",
        help="Serve the hub over MCP (streamable HTTP); read-only unless --agent-writes.",
    )
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
    serve.add_argument(
        "--agent-writes",
        action="store_true",
        help="Also serve the agent write tools (brief_create, recipe_create, plan_submit, "
        "node_set_status, facts_record, facts_retract) and their reads (facts_get, "
        "brief_get, plan_get), over a writable database. Agent ids are self-declared "
        "(attribution, not authentication): keep the hub tailnet-only. Human-only actions "
        "(review decisions, approvals, resume) are never tools.",
    )
    serve.set_defaults(func=_serve)

    console = hub_sub.add_parser(
        "console",
        help="Serve the human supervision console (reviews, leases, approvals, STOP).",
        description=(
            "Serve the human supervision console. Identity is network reachability: "
            "there are no accounts or passwords, so anyone who can reach the port acts "
            "as --operator. Keep it on loopback and reach it through tailscale serve "
            "(tailnet-only, DESIGN D2). A real credential is required before any "
            "exposure beyond the tailnet."
        ),
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
        "requests from loopback (tailscale serve). Any process on this host can send "
        "it and claim any login, so use it only where every local process is trusted.",
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

    withdraw = review_sub.add_parser(
        "withdraw", help="Withdraw an open review you requested (it becomes cancelled)."
    )
    _add_db(withdraw)
    withdraw.add_argument("--project", required=True, help="Project id (prj_...).")
    withdraw.add_argument("--review", required=True, help="Review id.")
    withdraw.add_argument("--reason", required=True, help="Why, for the audit trail.")
    _add_actor(withdraw)
    withdraw.add_argument("--idempotency-key", help="Makes a retry a no-op.")
    withdraw.set_defaults(func=_review_withdraw)

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

    node = hub_sub.add_parser(
        "node", help="Enrol, revoke and list the nodes allowed to call the node API."
    )
    node_sub = node.add_subparsers(dest="node_command", required=True)

    enroll = node_sub.add_parser(
        "enroll",
        help="Enrol a node and print its bearer token once (only its sha256 is stored).",
    )
    _add_db(enroll)
    enroll.add_argument("node_id", help="The node id (as the node reports it to the hub).")
    _add_actor(enroll)
    enroll.add_argument(
        "--rotate",
        action="store_true",
        help="Revoke the node's current token and enrol a new one, in one transaction.",
    )
    enroll.set_defaults(func=_node_enroll)

    revoke = node_sub.add_parser("revoke", help="Revoke a node's current token.")
    _add_db(revoke)
    revoke.add_argument("node_id", help="The node id to revoke.")
    _add_actor(revoke)
    revoke.add_argument("--reason", default="revoke", help="Why, for the audit trail.")
    revoke.set_defaults(func=_node_revoke)

    node_list = node_sub.add_parser(
        "list", help="List enrolled nodes with enrol/revoke times and a hash prefix only."
    )
    _add_db(node_list)
    node_list.set_defaults(func=_node_list)

    nodeapi = hub_sub.add_parser(
        "nodeapi",
        help="Serve the node-facing HTTP API (POST /node/v1/<op>, enrolled bearer token).",
    )
    nodeapi.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
    nodeapi.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1; tailnet-only, the token is a 2nd factor).",
    )
    nodeapi.add_argument("--port", type=int, default=8789, help="Port to bind (default: 8789).")
    nodeapi.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit binding to a non-loopback interface (required for any non-loopback --host).",
    )
    nodeapi.set_defaults(func=_nodeapi)

    job = hub_sub.add_parser("job", help="Submit and inspect durable Resolve node jobs.")
    job_sub = job.add_subparsers(dest="job_command", required=True)

    job_submit = job_sub.add_parser("submit", help="Submit a validated job to one named node.")
    _add_db(job_submit)
    _add_actor(job_submit)
    job_submit.add_argument("--kind", required=True, help="Registered JobRunner kind.")
    job_submit.add_argument("--params", required=True, help="Job params as a JSON object.")
    job_submit.add_argument("--project", required=True, help="ProjectLocator as a JSON object.")
    job_submit.add_argument(
        "--node", required=True, help="The only node initially allowed to claim."
    )
    job_submit.add_argument("--agent-id", required=True, help="Agent identity used by every op.")
    job_submit.add_argument("--priority", type=int, default=0)
    job_submit.add_argument("--approval-token")
    job_submit.add_argument("--idempotency-key", required=True)
    job_submit.set_defaults(func=_job_submit)

    job_list = job_sub.add_parser("list", help="List durable jobs and derived lease state.")
    _add_db(job_list)
    job_list.add_argument("--state")
    job_list.add_argument("--project-id")
    job_list.set_defaults(func=_job_list)

    job_show = job_sub.add_parser("show", help="Show one durable job.")
    _add_db(job_show)
    job_show.add_argument("job_id")
    job_show.set_defaults(func=_job_show)

    job_cancel = job_sub.add_parser("cancel", help="Request cancellation at the node's safe point.")
    _add_db(job_cancel)
    _add_actor(job_cancel)
    job_cancel.add_argument("job_id")
    job_cancel.add_argument("--reason", required=True)
    job_cancel.add_argument("--idempotency-key", required=True)
    job_cancel.set_defaults(func=_job_cancel)

    job_release = job_sub.add_parser(
        "release", help="Human-only release after checking Resolve; increments the fence."
    )
    _add_db(job_release)
    _add_actor(job_release)
    job_release.add_argument("job_id")
    job_release.add_argument(
        "--node", required=True, help="Node allowed to claim the new generation."
    )
    job_release.add_argument("--reason", required=True)
    job_release.add_argument("--idempotency-key", required=True)
    job_release.set_defaults(func=_job_release)

    from aar.hub import write_cli

    write_cli.register(hub_sub)


def _add_db(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="Path to the SQLite event-log database.")


def _add_actor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor",
        required=True,
        metavar="KIND:ID",
        help="Who is acting, e.g. agent:agent-1 (kinds: human, agent, node).",
    )


def _init(args: argparse.Namespace) -> int:
    # Import here so the core CLI does not require the optional 'hub' extra.
    from aar.hub.artifacts import ArtifactStore
    from aar.hub.events import EventLog

    db, artifacts = Path(args.db), Path(args.artifacts)
    # Both paths must be absent, and nothing is created until both are checked, so a refusal
    # never leaves a half-initialised hub behind.
    if db.exists():
        return _fail("db_exists", f"refusing to create a hub database over existing {args.db!r}")
    if artifacts.exists():
        return _fail(
            "artifacts_exist",
            f"refusing to create an artifact store over existing {args.artifacts!r}",
        )
    try:
        log = EventLog(args.db, create=True)
    except (OSError, sqlite3.Error) as exc:
        return _fail("db_create_failed", f"cannot create hub database {args.db!r}: {exc}")
    try:
        ArtifactStore(args.artifacts, log, create=True)
    except OSError as exc:
        log.close()
        return _fail(
            "artifacts_create_failed", f"cannot create artifact store {args.artifacts!r}: {exc}"
        )
    log.close()
    return _print({"db": str(db), "artifacts": str(artifacts)})


def _node_enroll(args: argparse.Namespace) -> int:
    from aar.hub import nodes

    try:
        actor = _parse_actor(args.actor)
    except ValueError as exc:
        return _fail("invalid_actor", str(exc))
    try:
        log = _open_existing(args.db)
    except _DatabaseUnavailable as exc:
        return _fail("database_unavailable", str(exc))
    try:
        node_id, token = nodes.enroll(log, args.node_id, actor, rotate=args.rotate)
    except nodes.NodeError as exc:
        return _fail(exc.code, str(exc))
    finally:
        log.close()
    # The token is printed here, once, and never stored or logged anywhere.
    return _print({"node_id": node_id, "token": token})


def _node_revoke(args: argparse.Namespace) -> int:
    from aar.hub import nodes

    try:
        actor = _parse_actor(args.actor)
    except ValueError as exc:
        return _fail("invalid_actor", str(exc))
    try:
        log = _open_existing(args.db)
    except _DatabaseUnavailable as exc:
        return _fail("database_unavailable", str(exc))
    try:
        nodes.revoke(log, args.node_id, actor, reason=args.reason)
    except nodes.NodeError as exc:
        return _fail(exc.code, str(exc))
    finally:
        log.close()
    return _print({"node_id": args.node_id, "revoked": True})


def _node_list(args: argparse.Namespace) -> int:
    from aar.hub import nodes

    try:
        log = _open_existing(args.db, read_only=True)
    except _DatabaseUnavailable as exc:
        return _fail("database_unavailable", str(exc))
    try:
        records = nodes.list_nodes(log)
    finally:
        log.close()
    return _print({"nodes": [record.to_dict() for record in records]})


def _nodeapi(args: argparse.Namespace) -> int:
    if args.host not in _LOOPBACK and not args.allow_remote:
        return _fail(
            "remote_bind_refused",
            f"refusing to bind to {args.host!r} without --allow-remote: the hub is tailnet-only "
            "(default bind is 127.0.0.1; the enrolled token is a second factor, not the only gate)",
        )

    # Import here so the core CLI does not require the optional 'hub' extra (Starlette/uvicorn).
    from aar.hub.nodeapi import build_app

    try:
        log = _open_existing(args.db)
    except _DatabaseUnavailable as exc:
        return _fail("database_unavailable", str(exc))

    import uvicorn

    app = build_app(log)
    # proxy_headers=False: the peer address must be the real TCP peer (loopback by default),
    # never a forwarded header a caller controls. server_header=False keeps the banner quiet.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=False,
        server_header=False,
        log_level="info",
    )
    return 0


def _job_command(action: Any, *, read_only: bool = False) -> Any:
    """CLI adapter: open exactly one existing log and render stable JSON errors."""

    def run(args: argparse.Namespace) -> int:
        from aar.hub import jobs
        from aar.hub.common import HumanRequired, InvalidInput
        from aar.hub.events import IdempotencyConflict

        try:
            log = _open_existing(args.db, read_only=read_only)
            try:
                value = action(jobs, log, args)
            finally:
                log.close()
            return _print(value)
        except _DatabaseUnavailable as exc:
            return _fail("database_unavailable", str(exc))
        except HumanRequired as exc:
            return _fail("human_required", str(exc))
        except InvalidInput as exc:
            return _fail("invalid_input", str(exc))
        except IdempotencyConflict as exc:
            return _fail("idempotency_conflict", str(exc))
        except jobs.JobError as exc:
            return _fail(exc.code, str(exc), 1 if isinstance(exc, jobs.JobNotFound) else 2)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return _fail("invalid_input", str(exc))

    return run


def _job_submit_action(jobs: Any, log: Any, args: argparse.Namespace) -> dict[str, Any]:
    from aar.node.locator import ProjectLocator

    params = json.loads(args.params)
    project = ProjectLocator.from_dict(json.loads(args.project))
    job, replayed = jobs.submit(
        log,
        _parse_actor(args.actor),
        kind=args.kind,
        params=params,
        project=project,
        agent_id=args.agent_id,
        assigned_node=args.node,
        priority=args.priority,
        approval_token=args.approval_token,
        idempotency_key=args.idempotency_key,
    )
    return {"job": job.to_dict(jobs.utcnow()), "replayed": replayed}


def _job_list_action(jobs: Any, log: Any, args: argparse.Namespace) -> dict[str, Any]:
    now = jobs.utcnow()
    found = [
        job
        for job in jobs.list_jobs(log)
        if args.project_id is None or job.target.project_id == args.project_id
    ]
    if args.state is not None:
        found = [job for job in found if job.state(now) == args.state]
    return {"jobs": [job.to_dict(now, include_payload=False) for job in found]}


def _job_show_action(jobs: Any, log: Any, args: argparse.Namespace) -> dict[str, Any]:
    job = jobs.get_job(log, args.job_id)
    if job is None:
        raise jobs.JobNotFound(f"no job {args.job_id!r}")
    return {"job": job.to_dict(jobs.utcnow())}


def _job_cancel_action(jobs: Any, log: Any, args: argparse.Namespace) -> dict[str, Any]:
    job, replayed = jobs.cancel(
        log,
        _parse_actor(args.actor),
        args.job_id,
        args.reason,
        idempotency_key=args.idempotency_key,
    )
    return {"job": job.to_dict(jobs.utcnow()), "replayed": replayed}


def _job_release_action(jobs: Any, log: Any, args: argparse.Namespace) -> dict[str, Any]:
    job, replayed = jobs.release(
        log,
        _parse_actor(args.actor),
        args.job_id,
        args.node,
        args.reason,
        idempotency_key=args.idempotency_key,
    )
    return {"job": job.to_dict(jobs.utcnow()), "replayed": replayed}


_job_submit = _job_command(_job_submit_action)
_job_list = _job_command(_job_list_action, read_only=True)
_job_show = _job_command(_job_show_action, read_only=True)
_job_cancel = _job_command(_job_cancel_action)
_job_release = _job_command(_job_release_action)


def _serve(args: argparse.Namespace) -> int:
    if args.host not in _LOOPBACK and not args.allow_remote:
        cli_io.warn(
            f"refusing to bind to {args.host!r} without --allow-remote: the v1 hub has no auth "
            "and is tailnet-only (default bind is 127.0.0.1)"
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
    # leaving a stray empty file/directory behind. --agent-writes opens the existing log
    # writable (create=False still checks the hub schema and never creates a database).
    try:
        log = _open_existing(args.db) if args.agent_writes else EventLog(args.db, read_only=True)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, _DatabaseUnavailable) as exc:
        # The exception message already names the path and the reason; don't re-prefix.
        cli_io.warn(f"aar hub serve: {exc}")
        return 2
    try:
        artifacts = ArtifactStore(args.artifacts, log, create=False)
    except OSError as exc:
        log.close()  # release the DB handle we just opened before bailing out
        cli_io.warn(f"aar hub serve: {exc}")
        return 2
    registry = Registry(log)
    server = build_server(log, registry, artifacts, agent_writes=args.agent_writes)
    if args.agent_writes:
        cli_io.warn(
            "aar hub serve: agent write tools are ON; agent ids are self-declared "
            "(attribution, not authentication). Keep this server tailnet-only."
        )
    server.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


def _fail(code: str, message: str, status: int = 2) -> int:
    cli_io.warn(json.dumps({"error": {"code": code, "message": message}}))
    return status


def _print(value: Any) -> int:
    """Write ``value`` as JSON to stdout; return 0 or an output-error code (74/141)."""
    return cli_io.emit(_PROG, json.dumps(value, indent=2, sort_keys=True) + "\n")


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

    The file must already exist (a write-mode SQLite connect would create it).
    ``EventLog(create=False)`` checks the hub schema before switching to WAL, so a
    SQLite file that is not a hub database is rejected and left byte-identical.
    """
    from aar.hub.events import EventLog

    path = Path(db)
    if db == ":memory:" or not path.is_file():
        what = "a directory" if path.is_dir() else "missing"
        raise _DatabaseUnavailable(f"no hub database at {db!r} ({what})")
    try:
        return EventLog(db, create=False, read_only=read_only)
    except _DB_OPEN_ERRORS as exc:
        raise _DatabaseUnavailable(f"cannot open hub database {db!r}: {exc}") from exc


def _open_store(root: str, log: Any) -> Any:
    """Open an existing artifact store; a missing directory is an error, never created."""
    from aar.hub.artifacts import ArtifactStore

    try:
        return ArtifactStore(root, log, create=False)
    except OSError as exc:
        raise _StoreUnavailable(f"no artifact store directory at {root!r}") from exc


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
        cli_io.warn("console is read-only: no --operator and no trusted headers")
    for warning in _console_warnings(args):
        cli_io.warn(f"WARNING: {warning}")
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


def _console_warnings(args: argparse.Namespace) -> list[str]:
    """Startup warnings for settings where identity is weaker than it may look."""
    warnings = []
    if args.trust_tailscale_headers:
        warnings.append(
            "--trust-tailscale-headers: any process on this host can send "
            "Tailscale-User-Login to loopback and act as any login. Use only where every "
            "local process is trusted."
        )
    if args.host not in _LOOPBACK:
        warnings.append(
            f"bound to {args.host}: anyone who can reach this address acts as "
            f"{args.operator or 'the trusted-header login'}. There is no other "
            "authentication; keep it on the tailnet."
        )
    return warnings


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
    return _print(review.to_dict())


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
    return _print(note.to_dict())


@_with_review_errors
def _review_withdraw(args: argparse.Namespace) -> int:
    from aar.hub.reviews import withdraw

    actor = _parse_actor(args.actor)
    log = _open_existing(args.db)
    try:
        withdrawal = withdraw(
            log,
            args.project,
            args.review,
            args.reason,
            actor,
            idempotency_key=args.idempotency_key,
        )
    finally:
        log.close()
    return _print(withdrawal.to_dict())


@_with_review_errors
def _review_list(args: argparse.Namespace) -> int:
    from aar.hub.reviews import list_reviews

    log = _open_existing(args.db, read_only=True)
    try:
        found = list_reviews(log, args.project, pending_only=not args.all)
    finally:
        log.close()
    return _print({"reviews": [review.to_dict() for review in found]})


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
    return _print(review.to_dict())
