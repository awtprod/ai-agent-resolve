"""``aar hub write ...`` and ``aar hub facts show``: the agent write surface without MCP.

Each write reads one JSON request object (from a file, or stdin with ``-``) whose keys are
the matching MCP tool's arguments other than ``agent_id`` and ``idempotency_key``, which
come from ``--actor`` and ``--idempotency-key``:

=============  ====================  ===============================================
command        MCP tool              request keys
=============  ====================  ===============================================
``brief``      ``brief_create``      ``brief``
``recipe``     ``recipe_create``     ``project_id``, ``recipe``
``plan``       ``plan_submit``       ``project_id``, ``derives_from``, ``plan``, ``title?``
``status``     ``node_set_status``   ``project_id``, ``node_id``, ``status``
``fact``       ``facts_record``      ``project_id``, ``key``, ``value``, ``source?``
``retract``    ``facts_retract``     ``project_id``, ``key``, ``reason``
=============  ====================  ===============================================

The result JSON goes to stdout. A refused write prints ``{"error": {code, message,
fix_hint, details}}`` on stderr and exits 2, like the other ``aar hub`` commands. A plan
that was stored but has lint errors (status ``blocked``) exits 1, as ``aar plan lint``
does for a plan with errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aar import cli_io

_PROG = "aar hub"
EXIT_OK = 0
EXIT_BLOCKED = 1  # a plan was recorded, but its lint found errors (status blocked)
EXIT_REFUSED = 2
MAX_REQUEST_BYTES = 2 * 1024 * 1024

_REQUEST_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # command: (required keys, optional keys)
    "brief": (frozenset({"brief"}), frozenset()),
    "recipe": (frozenset({"project_id", "recipe"}), frozenset()),
    "plan": (frozenset({"project_id", "derives_from", "plan"}), frozenset({"title"})),
    "status": (frozenset({"project_id", "node_id", "status"}), frozenset()),
    "fact": (frozenset({"project_id", "key", "value"}), frozenset({"source"})),
    "retract": (frozenset({"project_id", "key", "reason"}), frozenset()),
}
_HELP = {
    "brief": "Create a brief node (request {brief}).",
    "recipe": "Create a recipe node for a brief (request {project_id, recipe}).",
    "plan": "Lint and submit an EditPlan (request {project_id, derives_from, plan, title?}).",
    "status": "Move a node you created (request {project_id, node_id, status}).",
    "fact": "Record a project fact (request {project_id, key, value, source?}).",
    "retract": "Retract a project fact (request {project_id, key, reason}).",
}


def register(hub_sub: argparse._SubParsersAction) -> None:
    """Add ``write`` and ``facts`` to the ``aar hub`` group."""
    write = hub_sub.add_parser(
        "write",
        help="Agent writes without MCP: briefs, recipes, plans, own status, facts.",
        description=(
            "Each command reads one JSON request (a file, or - for stdin) with the same keys "
            "as the MCP tool's arguments, minus agent_id and idempotency_key. Exit 0 when "
            "the write is recorded (or replayed), 1 when a plan is recorded as blocked by "
            "lint errors, 2 when it is refused (JSON error on stderr). In v1 --actor is "
            "self-declared: attribution, not authentication."
        ),
    )
    write_sub = write.add_subparsers(dest="write_command", required=True)
    for command, help_text in _HELP.items():
        parser = write_sub.add_parser(command, help=help_text, description=help_text)
        parser.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
        if command == "plan":
            parser.add_argument("--artifacts", required=True, help="Artifact store directory.")
        parser.add_argument(
            "--actor",
            required=True,
            metavar="KIND:ID",
            help="Who is writing, e.g. agent:agent-1 (kinds: human, agent, node).",
        )
        parser.add_argument(
            "--idempotency-key",
            required=True,
            help="Names this request; retrying with the same key and request is a no-op.",
        )
        parser.add_argument(
            "request",
            nargs="?",
            default="-",
            help="JSON request file, or - (the default) for stdin.",
        )
        parser.set_defaults(func=_write, write_command=command)

    facts = hub_sub.add_parser("facts", help="Read per-project facts.")
    facts_sub = facts.add_subparsers(dest="facts_command", required=True)
    show = facts_sub.add_parser("show", help="Print a project's facts as JSON.")
    show.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
    show.add_argument("--project", required=True, help="Project id (prj_...).")
    show.add_argument("--prefix", help="Only keys that start with this, e.g. sync.offset.")
    show.set_defaults(func=_facts_show)


def _error(
    code: str,
    message: str,
    fix_hint: str,
    *,
    exit_status: int = EXIT_REFUSED,
    details: dict[str, Any] | None = None,
) -> int:
    body = {"code": code, "message": message, "fix_hint": fix_hint, "details": details or {}}
    cli_io.warn(json.dumps({"error": body}, sort_keys=True))
    return exit_status


def _print(value: Any) -> int:
    return cli_io.emit(_PROG, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _read_request(source: str) -> Any:
    """The request object from a file or stdin; ``ValueError`` with a reason if unusable."""
    if source == "-":
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    else:
        with Path(source).open("rb") as handle:
            raw = handle.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError(f"the request is over {MAX_REQUEST_BYTES} bytes")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_no_constants)
    except RecursionError:
        raise ValueError("the request is nested too deeply") from None
    except UnicodeDecodeError:
        raise ValueError("the request is not UTF-8") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"the request is not JSON: {exc}") from None


def _write(args: argparse.Namespace) -> int:
    from aar.hub.cli import _DatabaseUnavailable, _open_existing, _StoreUnavailable
    from aar.hub.writes import WriteError

    command = args.write_command
    try:
        request = _read_request(args.request)
    except (OSError, ValueError) as exc:
        return _error("invalid_input", str(exc), "Pass a JSON object (a file, or - for stdin).")
    required, optional = _REQUEST_KEYS[command]
    if not isinstance(request, dict):
        return _error(
            "invalid_input",
            f"the request must be a JSON object, got {type(request).__name__}",
            f"Send an object with {', '.join(sorted(required))}.",
        )
    missing = sorted(required - request.keys())
    unknown = sorted(request.keys() - required - optional)
    if missing or unknown:
        return _error(
            "invalid_input",
            f"request keys: missing {missing or 'none'}, unknown {unknown or 'none'}",
            f"Send exactly {', '.join(sorted(required))}"
            + (f" and optionally {', '.join(sorted(optional))}" if optional else "")
            + ".",
            details={"missing": missing, "unknown": unknown},
        )
    try:
        actor = _parse_actor(args.actor)
    except ValueError as exc:
        return _error("invalid_input", str(exc), "Use --actor KIND:ID, e.g. agent:agent-1.")
    try:
        log = _open_existing(args.db)
    except _DatabaseUnavailable as exc:
        return _error("database_unavailable", str(exc), "Pass --db an existing hub database.")
    try:
        try:
            result = _dispatch(command, args, log, actor, request)
        except _StoreUnavailable as exc:
            return _error(
                "artifact_store_unavailable", str(exc), "Pass --artifacts an existing store."
            )
        except WriteError as exc:
            return _error(exc.code, exc.message, exc.fix_hint, details=exc.details)
    finally:
        log.close()
    written = _print(result)
    if written != EXIT_OK:
        return written
    node = result.get("node") if command == "plan" else None
    if isinstance(node, dict) and node.get("status") == "blocked":
        return EXIT_BLOCKED
    return EXIT_OK


def _parse_actor(value: str) -> Any:
    from aar.hub.events import Actor

    kind, sep, ident = value.partition(":")
    if not sep:
        raise ValueError(f"--actor must look like KIND:ID, got {value!r}")
    return Actor(kind, ident)


def _dispatch(
    command: str, args: argparse.Namespace, log: Any, actor: Any, request: dict[str, Any]
) -> dict[str, Any]:
    from aar.hub import facts, writes
    from aar.hub.cli import _open_store

    key = args.idempotency_key
    if command == "brief":
        result = writes.create_brief(log, request["brief"], actor, idempotency_key=key)
    elif command == "recipe":
        result = writes.create_recipe(
            log, request["project_id"], request["recipe"], actor, idempotency_key=key
        )
    elif command == "plan":
        result = writes.submit_plan(
            log,
            _open_store(args.artifacts, log),
            request["project_id"],
            request["plan"],
            actor,
            derives_from=request["derives_from"],
            idempotency_key=key,
            title=request.get("title"),
        )
    elif command == "status":
        result = writes.set_own_status(
            log,
            request["project_id"],
            request["node_id"],
            request["status"],
            actor,
            idempotency_key=key,
        )
    elif command == "fact":
        result = facts.record_fact(
            log,
            request["project_id"],
            request["key"],
            request["value"],
            actor,
            idempotency_key=key,
            source=request.get("source"),
        )
    else:
        result = facts.retract_fact(
            log,
            request["project_id"],
            request["key"],
            request["reason"],
            actor,
            idempotency_key=key,
        )
    return result.to_dict()


def _facts_show(args: argparse.Namespace) -> int:
    from aar.hub.cli import _DatabaseUnavailable, _open_existing
    from aar.hub.facts import facts_view
    from aar.hub.registry import Registry

    try:
        log = _open_existing(args.db, read_only=True)
    except _DatabaseUnavailable as exc:
        return _error("database_unavailable", str(exc), "Pass --db an existing hub database.")
    try:
        if Registry(log).get(args.project) is None:
            return _error(
                "unknown_project",
                f"no project {args.project!r} in the hub registry",
                "Pass a project id (prj_...).",
                exit_status=1,
                details={"project_id": args.project},
            )
        view = facts_view(log, args.project).to_dict()
    finally:
        log.close()
    if args.prefix is not None:
        for section in ("facts", "retracted"):
            view[section] = [row for row in view[section] if row["key"].startswith(args.prefix)]
    return _print(view)
