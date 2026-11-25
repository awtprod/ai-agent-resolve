"""Pick up Resolve DRTs saved beside an exported round; retain every version.

Only runner-recorded export locations are read. Imports are proposals with explicit
reader limits, never replacements for the active plan or approvals of a review.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, Event, EventLog
from aar.media.settle import copy_settled
from aar.plan.export.errors import ExportError
from aar.plan.export.media import media_map_from_mapping
from aar.plan.model import EditPlan
from aar.plan.timeline_import import (
    TimelineFile,
    TimelineImportError,
    import_timeline,
    read_timeline,
)
from aar.recipes.ccn.scratch import work_root
from aar.recipes.events import EVENT_RESULT, RUNNER

EVENT_RETURN = "recipe.timeline_return"
MAX_DRT_BYTES = 64 * 1024 * 1024
MAX_EVENTS = 100_000
MAX_FILES = 100
SETTLE_SECONDS = 5.0


class ReturnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    recipe_node: str = Field(min_length=1, max_length=128)
    round: int = Field(ge=1, le=1000)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_map: dict[str, str] = Field(min_length=1, max_length=16)
    timeline: str = Field(min_length=1, max_length=200)


def _events(log: EventLog, types: list[str], project_id: str | None = None) -> Iterator[Event]:
    for count, event in enumerate(log.iter_events(types=types, project_id=project_id)):
        if count >= MAX_EVENTS:
            raise ValueError("desktop return history exceeds 100000 events; archive old runs")
        if event.actor == RUNNER:
            yield event


def returned_edits(log: EventLog, project_id: str, recipe_node: str) -> list[dict[str, Any]]:
    """Newest 20 received versions, each with immutable plan/diff/source artifacts."""
    found: deque[dict[str, Any]] = deque(maxlen=20)
    for event in _events(log, [EVENT_RETURN], project_id):
        if event.payload.get("recipe_node") == recipe_node:
            found.append(event.payload)
    return list(found)


def receive_one(
    log: EventLog,
    store: ArtifactStore,
    project_id: str,
    request: dict[str, Any],
    path: Path,
    actor: Actor,
    *,
    directory_fd: int,
) -> dict[str, Any] | None:
    """Snapshot one settled regular DRT, import it, and journal once per round + content."""
    req = ReturnRequest.model_validate(request)
    mapping = media_map_from_mapping(req.media_map)
    fd, temporary = tempfile.mkstemp(prefix="aar-return-", suffix=".drt", dir=work_root())
    snapshot = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as target:
            copied = copy_settled(
                path,
                target,
                max_bytes=MAX_DRT_BYTES,
                settle_seconds=SETTLE_SECONDS,
                what="returned timeline",
                dir_fd=directory_fd,
            )
        if copied is None:
            return None  # still being saved, or changed while copied; retry with a fresh snapshot
        key = (
            f"desktop-return:{project_id}:{req.recipe_node}:{req.round}:"
            f"{req.plan_sha256}:{copied.sha256}"
        )
        existing = log.get_by_idempotency_key(key)
        if existing is not None:
            return existing.payload
        drt = store.put(
            snapshot, actor=actor, project_id=project_id, meta={"role": "returned-timeline"}
        )
        result: dict[str, Any] = {
            "recipe_node": req.recipe_node,
            "round": req.round,
            "base_plan": req.plan_sha256,
            "drt": drt,
            "file": path.name,
            "status": "needs_review",
            "partial_reconstruction": True,
        }
        return record_import(
            log,
            store,
            project_id,
            recipe_node=req.recipe_node,
            plan_sha256=req.plan_sha256,
            mapping=mapping,
            read=lambda: read_timeline(snapshot),
            result=result,
            key=key,
            actor=actor,
        )
    finally:
        snapshot.unlink(missing_ok=True)


def record_import(
    log: EventLog,
    store: ArtifactStore,
    project_id: str,
    *,
    recipe_node: str,
    plan_sha256: str,
    mapping: Any,
    read: Callable[[], TimelineFile],
    result: dict[str, Any],
    key: str,
    actor: Actor,
) -> dict[str, Any] | None:
    """Import the timeline ``read`` gives against the plan ``plan_sha256`` (the plan the
    returned timeline was handed over from), store the edited plan and the diff, and journal
    ``result`` once under ``key`` as a ``recipe.timeline_return``. One implementation for a
    returned ``.drt`` and a library save (:mod:`aar.recipes.library_returns`): an import that
    fails is recorded ``failed`` with its error; pickup never replaces the active plan or
    decides a review. None when the run was cancelled meanwhile."""
    try:
        plan = EditPlan.model_validate_json(store.read_bytes(plan_sha256))
        timeline = read()
        imported = import_timeline(
            plan,
            timeline.timeline,
            timeline.media,
            media_map=mapping,
            reader_warnings=timeline.warnings,
        )
        result["plan"] = store.put(
            imported.plan.model_dump_json().encode(),
            actor=actor,
            project_id=project_id,
            meta={"role": "returned-plan"},
        )
        result["diff"] = store.put(
            json.dumps(imported.diff, allow_nan=False).encode(),
            actor=actor,
            project_id=project_id,
            meta={"role": "returned-timeline-diff"},
        )
        result["not_carried"] = imported.not_carried
        result["import_needs_attention"] = imported.needs_attention
        result["limits"] = imported.diff.get("limits", [])
    except (ValueError, OSError, TimelineImportError, ExportError) as exc:
        result.update(status="failed", error=str(exc)[:2000])
    # First writer wins if two workers process the same round/content concurrently.
    with log.transaction() as txn:
        from aar.recipes.state import load_state

        if load_state(log, recipe_node, project_id=project_id).recipe_status == "cancelled":
            return None
        existing = log.get_by_idempotency_key(key)
        if existing is not None:
            return existing.payload
        txn.append(EVENT_RETURN, RUNNER, result, project_id=project_id, idempotency_key=key)
    return result


def _scan_error(
    log: EventLog, event: Event, request: ReturnRequest, path: Path, exc: Exception
) -> None:
    payload = {
        "recipe_node": request.recipe_node,
        "round": request.round,
        "base_plan": request.plan_sha256,
        "file": path.name[:255],
        "status": "failed",
        "partial_reconstruction": True,
        "error": str(exc)[:2000],
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    from aar.recipes.state import load_state

    with log.transaction() as txn:
        if (
            load_state(log, request.recipe_node, project_id=event.project_id).recipe_status
            == "cancelled"
        ):
            return
        txn.append(
            EVENT_RETURN,
            RUNNER,
            payload,
            project_id=event.project_id,
            idempotency_key=f"desktop-return-error:{event.id}:{key}",
        )


@contextlib.contextmanager
def _return_directory(root: Path, folder: Path) -> Iterator[int]:
    """Pin every directory from the trusted export root without following share symlinks."""
    try:
        parts = folder.relative_to(root).parts
    except ValueError:
        raise ValueError("return folder is outside the export share") from None
    if any(part in (".", "..") for part in parts):
        raise ValueError("return folder contains a traversal")
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("desktop return pickup requires POSIX no-follow directory descriptors")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with contextlib.ExitStack() as stack:
        fd = os.open(root, flags)
        stack.callback(os.close, fd)
        for part in parts:
            fd = os.open(part, flags, dir_fd=fd)
            stack.callback(os.close, fd)
        yield fd


def _return_paths(directory_fd: int) -> list[Path]:
    with os.scandir(directory_fd) as entries:
        paths = []
        for n, entry in enumerate(entries):
            if n >= MAX_FILES:
                raise ValueError("return folder exceeds 100 entries; archive older returns")
            if entry.name.lower().endswith(".drt"):
                paths.append(Path(entry.name))
    return sorted(paths)


def scan_returns(log: EventLog, store: ArtifactStore, actor: Actor) -> int:
    """One bounded sweep of runner-owned export locations, including finished runs."""
    from aar.recipes.ccn.export_files import export_root
    from aar.recipes.state import load_state

    if not os.environ.get("AAR_EXPORT_DIR"):
        return 0
    root = export_root().resolve()
    count = 0
    for event in _events(log, [EVENT_RESULT]):
        out = event.payload.get("outputs", {})
        packet = out.get("desktop_return") if isinstance(out, dict) else None
        if not isinstance(packet, dict) or event.payload.get("outcome") != "done":
            continue
        request = ReturnRequest.model_validate(packet["request"])
        state = load_state(log, request.recipe_node, project_id=event.project_id)
        if state.recipe_status == "cancelled":
            continue
        folder = Path(packet["directory"])
        try:
            with _return_directory(root, folder) as directory_fd:
                for path in _return_paths(directory_fd):
                    try:
                        result = receive_one(
                            log,
                            store,
                            event.project_id,
                            packet["request"],
                            path,
                            actor,
                            directory_fd=directory_fd,
                        )
                        count += result is not None
                    except (OSError, ValueError) as exc:
                        _scan_error(log, event, request, path, exc)
        except FileNotFoundError:
            continue  # return folder removed; there is nothing to collect
        except (OSError, ValueError) as exc:
            _scan_error(log, event, request, folder, exc)
    return count
