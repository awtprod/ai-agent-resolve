"""``aar media``: ingest and resolve Linux-resident source media."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from aar import cli_io
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.media.store import (
    MAX_MEDIA_BYTES,
    InvalidMediaInput,
    MediaStore,
    MediaStoreError,
    MediaTooLarge,
    MediaUnavailable,
    UnknownProject,
)

_PROG = "aar media"
EXIT_OK = 0
EXIT_UNAVAILABLE = 1
EXIT_INPUT = 2


def register(subparsers: Any) -> None:
    """Add the ``media`` command group to the top-level parser."""
    media = subparsers.add_parser(
        "media",
        help="Ingest source media into the hub and resolve verified local copies.",
    )
    commands = media.add_subparsers(dest="media_command", metavar="COMMAND", required=True)

    ingest = commands.add_parser("ingest", help="Copy source media into the hub-owned store.")
    ingest.add_argument("source", type=Path, help="Linux-local source file to copy.")
    _add_hub(ingest)
    ingest.add_argument("--project", required=True, help="Registered project id.")
    ingest.add_argument("--actor", required=True, help="Actor as KIND:ID.")
    ingest.add_argument(
        "--max-bytes",
        type=_positive_bytes,
        help="Optional maximum source size, enforced while streaming.",
    )
    ingest.set_defaults(func=_cmd_ingest)

    availability = commands.add_parser(
        "availability", help="Verify whether a project media reference is locally available."
    )
    availability.add_argument("media_ref", help="Stable media:<sha256> reference.")
    _add_hub(availability)
    availability.add_argument("--project", required=True, help="Registered project id.")
    availability.set_defaults(func=_cmd_availability)

    resolve = commands.add_parser(
        "resolve", help="Return the verified Linux-local path for a project media reference."
    )
    resolve.add_argument("media_ref", help="Stable media:<sha256> reference.")
    _add_hub(resolve)
    resolve.add_argument("--project", required=True, help="Registered project id.")
    resolve.set_defaults(func=_cmd_resolve)


def _add_hub(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="Existing hub event-log database.")
    parser.add_argument("--artifacts", required=True, help="Existing hub artifact-store root.")


def _positive_bytes(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if not 0 < value <= MAX_MEDIA_BYTES:
        raise argparse.ArgumentTypeError(f"must be in 1..{MAX_MEDIA_BYTES}")
    return value


def _actor(raw: str) -> Actor:
    kind, separator, actor_id = raw.partition(":")
    if not separator:
        raise InvalidMediaInput(f"--actor must look like KIND:ID, got {raw!r}")
    try:
        return Actor(kind, actor_id)
    except ValueError as exc:
        raise InvalidMediaInput(str(exc)) from exc


def _open(args: argparse.Namespace, *, read_only: bool) -> tuple[EventLog, MediaStore]:
    db = Path(args.db)
    root = Path(args.artifacts)
    if not db.is_file():
        raise InvalidMediaInput(f"no hub database at {str(db)!r}")
    if not root.is_dir():
        raise InvalidMediaInput(f"no artifact store directory at {str(root)!r}")
    log = EventLog(db, create=False, read_only=read_only)
    try:
        return log, MediaStore(log, ArtifactStore(root, log, create=False))
    except BaseException:
        log.close()
        raise


def _emit(value: dict[str, Any], status: int = EXIT_OK) -> int:
    written = cli_io.emit(_PROG, json.dumps(value, indent=2, sort_keys=True) + "\n")
    return written or status


def _error(code: str, message: str, status: int) -> int:
    cli_io.warn(json.dumps({"ok": False, "error": {"code": code, "message": message}}))
    return status


def _run(args: argparse.Namespace, operation: Any, *, read_only: bool) -> int:
    log: EventLog | None = None
    try:
        log, store = _open(args, read_only=read_only)
        return int(operation(store))
    except MediaUnavailable as exc:
        return _error("media_unavailable", str(exc), EXIT_UNAVAILABLE)
    except UnknownProject as exc:
        return _error("unknown_project", str(exc), EXIT_INPUT)
    except MediaTooLarge as exc:
        return _error("media_too_large", str(exc), EXIT_INPUT)
    except InvalidMediaInput as exc:
        return _error("invalid_input", str(exc), EXIT_INPUT)
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        return _error("store_unavailable", str(exc), EXIT_INPUT)
    except MediaStoreError as exc:
        return _error("media_store_error", str(exc), EXIT_UNAVAILABLE)
    finally:
        if log is not None:
            log.close()


def _cmd_ingest(args: argparse.Namespace) -> int:
    def ingest(store: MediaStore) -> int:
        record = store.ingest(
            args.source,
            project_id=args.project,
            actor=_actor(args.actor),
            max_bytes=args.max_bytes,
        )
        return _emit(record.to_dict())

    return _run(args, ingest, read_only=False)


def _cmd_availability(args: argparse.Namespace) -> int:
    def availability(store: MediaStore) -> int:
        result = store.availability(args.media_ref, project_id=args.project)
        return _emit(result.to_dict(), EXIT_OK if result.available else EXIT_UNAVAILABLE)

    return _run(args, availability, read_only=True)


def _cmd_resolve(args: argparse.Namespace) -> int:
    def resolve(store: MediaStore) -> int:
        result = store.availability(args.media_ref, project_id=args.project)
        if not result.available or result.local_path is None:
            raise MediaUnavailable(result)
        return _emit(result.to_dict())

    return _run(args, resolve, read_only=True)
