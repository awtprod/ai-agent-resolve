"""Project-scoped source-media associations backed by the hub artifact store.

The bytes live only in :class:`aar.hub.artifacts.ArtifactStore`.  This module adds
the fact that a content hash is source media for a particular project, because the
artifact store deliberately records metadata only for the first copy it sees.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from aar.hub.artifacts import ArtifactStore, ArtifactTooLarge, CorruptArtifact
from aar.hub.events import Actor, Event, EventLog
from aar.hub.registry import Registry
from aar.node.branches import valid_agent_id

EVENT_MEDIA_INGESTED = "media.ingested"

MAX_ID_CHARS = 128
MAX_FILENAME_CHARS = 255
MAX_SOURCE_PATH_BYTES = 4096
MAX_MEDIA_BYTES = 2**63 - 1
_REF_RE = re.compile(r"media:([0-9a-f]{64})\Z")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class MediaStoreError(Exception):
    """Base class for typed media-store failures."""


class InvalidMediaInput(MediaStoreError, ValueError):
    """An ingest or lookup argument is malformed."""


class UnknownProject(MediaStoreError, LookupError):
    """The requested project is not registered in the hub."""


class MediaTooLarge(InvalidMediaInput):
    """The source exceeds the configured ingest limit."""


class MediaUnavailable(MediaStoreError, FileNotFoundError):
    """A project association exists, but its verified bytes are unavailable."""

    def __init__(self, availability: MediaAvailability) -> None:
        reason = availability.reason or "unavailable"
        super().__init__(f"{availability.media_ref} is unavailable: {reason}")
        self.availability = availability


@dataclass(frozen=True, slots=True)
class MediaAssociation:
    """The durable project-scoped fact recorded by an ingest."""

    media_ref: str
    sha256: str
    size: int
    project_id: str
    original_filename: str
    actor: Actor
    created: str


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """A completed ingest and its verified hub-local content path."""

    media_ref: str
    sha256: str
    size: int
    project_id: str
    original_filename: str
    local_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "media_ref": self.media_ref,
            "sha256": self.sha256,
            "size": self.size,
            "project_id": self.project_id,
            "original_filename": self.original_filename,
            "available": True,
            "local_path": str(self.local_path),
        }


@dataclass(frozen=True, slots=True)
class MediaAvailability:
    """Whether a project's media reference currently has verified local bytes."""

    media_ref: str
    project_id: str
    available: bool
    reason: str | None = None
    sha256: str | None = None
    size: int | None = None
    original_filename: str | None = None
    local_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.available,
            "media_ref": self.media_ref,
            "project_id": self.project_id,
            "available": self.available,
            "reason": self.reason,
            "sha256": self.sha256,
            "size": self.size,
            "original_filename": self.original_filename,
            "local_path": str(self.local_path) if self.local_path is not None else None,
        }


def _clean_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_ID_CHARS:
        raise InvalidMediaInput(f"{field} must be 1-{MAX_ID_CHARS} characters")
    if _CONTROL_RE.search(value):
        raise InvalidMediaInput(f"{field} must not contain control characters")
    return value


def _check_actor(actor: Any) -> Actor:
    if not isinstance(actor, Actor):
        raise InvalidMediaInput(f"actor must be an Actor, got {type(actor).__name__}")
    _clean_id(actor.id, "actor id")
    if actor.kind == "agent" and not valid_agent_id(actor.id):
        raise InvalidMediaInput("agent actor ids may contain only A-Z, a-z, 0-9, '.', '_' and '-'")
    return actor


def _check_max_bytes(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_MEDIA_BYTES:
        raise InvalidMediaInput(f"max_bytes must be an integer in 1..{MAX_MEDIA_BYTES}")
    return value


def _source_file(source: str | PathLike[str]) -> tuple[Path, str, int]:
    try:
        raw_path = os.fspath(source)
    except TypeError as exc:
        raise InvalidMediaInput("source must be a filesystem path") from exc
    if not isinstance(raw_path, str):
        raise InvalidMediaInput("source must be a text filesystem path")
    if (
        not raw_path
        or _CONTROL_RE.search(raw_path)
        or len(os.fsencode(raw_path)) > MAX_SOURCE_PATH_BYTES
    ):
        raise InvalidMediaInput("source path is empty, unsafe, or over 4096 encoded bytes")
    path = Path(raw_path)
    try:
        details = path.stat()
    except FileNotFoundError:
        raise InvalidMediaInput(f"source file not found: {path}") from None
    except OSError as exc:
        raise InvalidMediaInput(f"cannot inspect source file: {exc}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise InvalidMediaInput(f"source is not a regular file: {path}")
    filename = path.name
    if (
        filename in ("", ".", "..")
        or len(filename) > MAX_FILENAME_CHARS
        or _CONTROL_RE.search(filename)
    ):
        raise InvalidMediaInput("source filename is empty, unsafe, or over 255 characters")
    return path, filename, details.st_size


def _parse_ref(media_ref: Any) -> str:
    if not isinstance(media_ref, str):
        raise InvalidMediaInput("media_ref must be text")
    match = _REF_RE.fullmatch(media_ref)
    if match is None:
        raise InvalidMediaInput("media_ref must look like media:<64 lowercase sha256 hex>")
    return match.group(1)


def _association_key(project_id: str, sha256: str) -> str:
    project_hash = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    return f"media.association:{project_hash}:{sha256}"


def _association(event: Event, project_id: str, sha256: str) -> MediaAssociation:
    payload = event.payload
    expected_ref = f"media:{sha256}"
    size = payload.get("size") if isinstance(payload, dict) else None
    filename = payload.get("original_filename") if isinstance(payload, dict) else None
    valid = (
        event.type == EVENT_MEDIA_INGESTED
        and event.project_id == project_id
        and isinstance(payload, dict)
        and payload.get("media_ref") == expected_ref
        and payload.get("sha256") == sha256
        and payload.get("artifact_id") == sha256
        and payload.get("project_id") == project_id
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 <= size <= MAX_MEDIA_BYTES
        and isinstance(filename, str)
        and 0 < len(filename) <= MAX_FILENAME_CHARS
        and _CONTROL_RE.search(filename) is None
    )
    if not valid:
        raise MediaStoreError("the stored media association is malformed")
    return MediaAssociation(
        media_ref=expected_ref,
        sha256=sha256,
        size=size,
        project_id=project_id,
        original_filename=filename,
        actor=event.actor,
        created=event.ts,
    )


class MediaStore:
    """Ingest and resolve project source media through an existing artifact store."""

    def __init__(self, log: EventLog, artifacts: ArtifactStore) -> None:
        self._log = log
        self._artifacts = artifacts

    def _project(self, project_id: Any) -> str:
        project_id = _clean_id(project_id, "project_id")
        if Registry(self._log).get(project_id) is None:
            raise UnknownProject(f"unknown project: {project_id!r}")
        return project_id

    def ingest(
        self,
        source: str | PathLike[str],
        *,
        project_id: str,
        actor: Actor,
        max_bytes: int | None = None,
    ) -> MediaRecord:
        """Copy ``source`` into the hub store and record its project association."""
        project_id = self._project(project_id)
        actor = _check_actor(actor)
        max_bytes = _check_max_bytes(max_bytes)
        source_path, filename, known_size = _source_file(source)
        if max_bytes is not None and known_size > max_bytes:
            raise MediaTooLarge(f"source exceeds the configured limit of {max_bytes} bytes")

        try:
            sha256 = self._artifacts.put(
                source_path,
                actor=actor,
                meta={"kind": "source_media"},
                producing_op="media.ingest",
                max_bytes=max_bytes,
            )
        except ArtifactTooLarge as exc:
            raise MediaTooLarge(str(exc)) from exc

        media_ref = f"media:{sha256}"
        with self._artifacts.open(sha256) as handle:
            size = os.fstat(handle.fileno()).st_size
            local_path = Path(handle.name).resolve(strict=True)

        key = _association_key(project_id, sha256)
        with self._log.transaction() as transaction:
            event = transaction.find_by_idempotency_key(key)
            if event is None:
                event = transaction.append(
                    EVENT_MEDIA_INGESTED,
                    actor,
                    {
                        "media_ref": media_ref,
                        "artifact_id": sha256,
                        "sha256": sha256,
                        "size": size,
                        "project_id": project_id,
                        "original_filename": filename,
                    },
                    project_id=project_id,
                    idempotency_key=key,
                )
        saved = _association(event, project_id, sha256)
        if saved.size != size:
            raise MediaStoreError(
                f"stored media association size {saved.size} does not match verified size {size}"
            )
        return MediaRecord(
            media_ref=saved.media_ref,
            sha256=saved.sha256,
            size=saved.size,
            project_id=saved.project_id,
            original_filename=saved.original_filename,
            local_path=local_path,
        )

    def get_association(self, media_ref: str, *, project_id: str) -> MediaAssociation | None:
        """Return this project's association, without asserting byte availability."""
        project_id = self._project(project_id)
        sha256 = _parse_ref(media_ref)
        event = self._log.get_by_idempotency_key(_association_key(project_id, sha256))
        return None if event is None else _association(event, project_id, sha256)

    def availability(self, media_ref: str, *, project_id: str) -> MediaAvailability:
        """Verify associated content and describe why it is unavailable when it is not."""
        project_id = self._project(project_id)
        sha256 = _parse_ref(media_ref)
        event = self._log.get_by_idempotency_key(_association_key(project_id, sha256))
        if event is None:
            return MediaAvailability(media_ref, project_id, False, reason="not_associated")
        association = _association(event, project_id, sha256)
        base = {
            "media_ref": association.media_ref,
            "project_id": project_id,
            "sha256": sha256,
            "size": association.size,
            "original_filename": association.original_filename,
        }
        try:
            with self._artifacts.open(sha256) as handle:
                actual_size = os.fstat(handle.fileno()).st_size
                path = Path(handle.name).resolve(strict=True)
        except FileNotFoundError:
            return MediaAvailability(**base, available=False, reason="missing")
        except CorruptArtifact:
            return MediaAvailability(**base, available=False, reason="corrupt")
        except OSError:
            return MediaAvailability(**base, available=False, reason="unreadable")
        if actual_size != association.size:
            return MediaAvailability(**base, available=False, reason="size_mismatch")
        return MediaAvailability(**base, available=True, local_path=path)

    def resolve(self, media_ref: str, *, project_id: str) -> Path:
        """Return a verified hub-local path; never fall back to the original source."""
        availability = self.availability(media_ref, project_id=project_id)
        if not availability.available or availability.local_path is None:
            raise MediaUnavailable(availability)
        return availability.local_path
