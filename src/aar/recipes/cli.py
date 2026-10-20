"""The ``aar recipe`` subcommand group: a thin JSON wrapper over :mod:`aar.recipes.run`.

``aar recipe {list,show,start,status,complete,retry,cancel,worker}``. JSON in, JSON out. A refusal
prints ``{"error": {code, message, fix_hint, details}}`` on stderr and exits 2, as the other ``aar``
write surfaces do. Recipes are registered by modules named with ``--recipes`` (repeatable) or the
``AAR_RECIPES`` env, loaded before the command runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aar import cli_io

_PROG = "aar recipe"
EXIT_OK = 0
EXIT_REFUSED = 2
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``recipe`` group on the top-level parser."""
    recipe = subparsers.add_parser(
        "recipe", help="Run and inspect recipe runs (the cold recipe engine)."
    )
    sub = recipe.add_subparsers(dest="recipe_command", required=True)

    listing = sub.add_parser("list", help="List registered recipes.")
    _add_recipes(listing)
    listing.set_defaults(func=_list)

    show = sub.add_parser("show", help="Show one registered recipe's steps.")
    _add_recipes(show)
    show.add_argument("name", help="Recipe name.")
    show.set_defaults(func=_show)

    start = sub.add_parser("start", help="Start a recipe run (move it active and advance).")
    _add_db(start)
    _add_recipes(start)
    _add_actor(start)
    _add_key(start)
    start.add_argument("recipe_node", help="The recipe node id (recipe_create returns it).")
    start.set_defaults(func=_start)

    status = sub.add_parser("status", help="Print the recipe run status view (read-only).")
    _add_db(status)
    _add_recipes(status)
    status.add_argument("recipe_node", help="The recipe node id.")
    status.set_defaults(func=_status)

    complete = sub.add_parser("complete", help="Complete an agent step with a JSON outputs object.")
    _add_db(complete)
    complete.add_argument("--artifacts", required=True, help="Artifact store directory.")
    _add_recipes(complete)
    _add_actor(complete)
    _add_key(complete)
    complete.add_argument("--recipe", required=True, help="The recipe node id.")
    complete.add_argument("--step", required=True, help="The agent step id.")
    complete.add_argument("--attempt", type=int, help="The open attempt (optional).")
    complete.add_argument(
        "outputs", nargs="?", default="-", help="JSON outputs file, or - (default) for stdin."
    )
    complete.set_defaults(func=_complete)

    retry = sub.add_parser("retry", help="Retry a failed step that has attempts left.")
    _add_db(retry)
    _add_recipes(retry)
    _add_actor(retry)
    _add_key(retry)
    retry.add_argument("--recipe", required=True, help="The recipe node id.")
    retry.add_argument("--step", required=True, help="The failed step id.")
    retry.set_defaults(func=_retry)

    cancel = sub.add_parser("cancel", help="Cancel a run (cancel live ops, withdraw reviews).")
    _add_db(cancel)
    _add_recipes(cancel)
    _add_actor(cancel)
    _add_key(cancel)
    cancel.add_argument("--reason", default="recipe cancelled", help="Why, for the audit trail.")
    cancel.add_argument("recipe_node", help="The recipe node id.")
    cancel.set_defaults(func=_cancel)

    worker = sub.add_parser("worker", help="Run the cold worker on the hub host.")
    _add_db(worker)
    worker.add_argument("--artifacts", required=True, help="Artifact store directory.")
    _add_recipes(worker)
    _add_actor(worker)
    worker.add_argument("--once", action="store_true", help="Claim and run at most one step.")
    worker.add_argument("--poll-s", type=float, default=2.0, help="Idle poll interval (seconds).")
    worker.set_defaults(func=_worker)


def _add_db(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="Path to the SQLite event-log database.")


def _add_recipes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recipes",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import a module that registers recipes (repeatable; also reads AAR_RECIPES).",
    )


def _add_actor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor",
        required=True,
        metavar="KIND:ID",
        help="Who is acting, e.g. agent:agent-1 (kinds: human, agent, node).",
    )


def _add_key(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--idempotency-key", required=True, help="Names this request for retries.")


def _error(code: str, message: str, fix_hint: str, details: dict[str, Any] | None = None) -> int:
    body = {"code": code, "message": message, "fix_hint": fix_hint, "details": details or {}}
    cli_io.warn(json.dumps({"error": body}, sort_keys=True))
    return EXIT_REFUSED


def _print(value: Any) -> int:
    return cli_io.emit(_PROG, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_recipes(args: argparse.Namespace) -> None:
    from aar.recipes import registry

    registry.load_modules(registry.modules_from_env(getattr(args, "recipes", [])))


def _parse_actor(value: str) -> Any:
    from aar.hub.events import Actor

    kind, sep, ident = value.partition(":")
    if not sep:
        raise ValueError(f"--actor must look like KIND:ID, got {value!r}")
    return Actor(kind, ident)


def _read_json(source: str) -> Any:
    if source == "-":
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    else:
        with Path(source).open("rb") as handle:
            raw = handle.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError(f"the outputs are over {MAX_REQUEST_BYTES} bytes")
    return json.loads(raw.decode("utf-8"))


def _list(args: argparse.Namespace) -> int:
    from aar.recipes.run import recipe_list

    _load_recipes(args)
    return _print(recipe_list())


def _show(args: argparse.Namespace) -> int:
    from aar.recipes.run import recipe_show

    _load_recipes(args)
    return _print(recipe_show(args.name))


def _with_db(args: argparse.Namespace, func: Any, *, read_only: bool = False) -> int:
    from aar.hub.cli import _DatabaseUnavailable, _open_existing
    from aar.hub.writes import WriteError

    _load_recipes(args)
    try:
        log = _open_existing(args.db, read_only=read_only)
    except _DatabaseUnavailable as exc:
        return _error("database_unavailable", str(exc), "Pass --db an existing hub database.")
    try:
        try:
            return func(log)
        except WriteError as exc:
            return _error(exc.code, exc.message, exc.fix_hint, exc.details)
        except ValueError as exc:
            return _error("invalid_input", str(exc), "Check the arguments.")
    finally:
        log.close()


def _start(args: argparse.Namespace) -> int:
    from aar.recipes.run import recipe_start

    def run(log: Any) -> int:
        actor = _parse_actor(args.actor)
        return _print(
            recipe_start(log, args.recipe_node, actor, idempotency_key=args.idempotency_key)
        )

    return _with_db(args, run)


def _status(args: argparse.Namespace) -> int:
    from aar.recipes.run import recipe_status

    def run(log: Any) -> int:
        return _print(recipe_status(log, args.recipe_node))

    return _with_db(args, run, read_only=True)


def _complete(args: argparse.Namespace) -> int:
    from aar.hub.cli import _open_store
    from aar.recipes.run import step_complete

    try:
        outputs = _read_json(args.outputs)
    except (OSError, ValueError) as exc:
        return _error("invalid_input", str(exc), "Pass a JSON object (a file, or - for stdin).")
    if not isinstance(outputs, dict):
        return _error("invalid_input", "outputs must be a JSON object", "Send an object.")

    def run(log: Any) -> int:
        actor = _parse_actor(args.actor)
        store = _open_store(args.artifacts, log)
        return _print(
            step_complete(
                log,
                store,
                args.recipe,
                args.step,
                args.attempt,
                outputs,
                actor,
                idempotency_key=args.idempotency_key,
            )
        )

    return _with_db(args, run)


def _retry(args: argparse.Namespace) -> int:
    from aar.recipes.run import recipe_retry

    def run(log: Any) -> int:
        actor = _parse_actor(args.actor)
        return _print(
            recipe_retry(log, args.recipe, args.step, actor, idempotency_key=args.idempotency_key)
        )

    return _with_db(args, run)


def _cancel(args: argparse.Namespace) -> int:
    from aar.recipes.run import recipe_cancel

    def run(log: Any) -> int:
        actor = _parse_actor(args.actor)
        return _print(
            recipe_cancel(
                log,
                args.recipe_node,
                actor,
                idempotency_key=args.idempotency_key,
                reason=args.reason,
            )
        )

    return _with_db(args, run)


def _worker(args: argparse.Namespace) -> int:
    from aar.hub.cli import _DatabaseUnavailable, _open_existing, _open_store, _StoreUnavailable
    from aar.recipes.worker import run_worker

    _load_recipes(args)
    try:
        actor = _parse_actor(args.actor)
    except ValueError as exc:
        return _error("invalid_input", str(exc), "Use --actor KIND:ID.")
    try:
        log = _open_existing(args.db)
    except _DatabaseUnavailable as exc:
        return _error("database_unavailable", str(exc), "Pass --db an existing hub database.")
    try:
        store = _open_store(args.artifacts, log)
    except _StoreUnavailable as exc:
        log.close()
        return _error("artifact_store_unavailable", str(exc), "Pass --artifacts an existing store.")
    try:
        ran = run_worker(
            log, store, actor, once=args.once, poll_s=args.poll_s, install_signals=True
        )
    finally:
        log.close()
    return _print({"ok": True, "op": "recipe.worker", "ran": ran})
