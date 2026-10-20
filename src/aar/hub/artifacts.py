"""Content-addressed artifact store.

Artifacts (renders, contact sheets, transcripts, exported files) are stored by the
SHA-256 of their content under a directory, so identical content is stored once and
every reference is a stable hash id. The content lives on disk; the metadata (mime,
size, project, producing op, created) is recorded as an event on the shared log, so
provenance is part of the same history as everything else.

Ids are validated as 64-char lowercase hex before they touch the filesystem, so a
crafted id cannot escape the store. Content is verified against its hash on every
read, and ``put`` streams to a temp file, fsyncs and atomically renames, repairing a
corrupt existing copy when it has the bytes.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

from aar.hub.events import Actor, EventLog, IdempotencyConflict

EVENT_ARTIFACT_ADDED = "artifact.added"

_CHUNK = 1024 * 1024
_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class InvalidArtifactId(ValueError):
    """Raised when an artifact id is not a 64-char lowercase hex SHA-256."""


class CorruptArtifact(Exception):
    """Raised when stored content does not match the hash it is filed under."""


class ArtifactTooLarge(ValueError):
    """Raised when content exceeds a caller-supplied streaming size limit."""


@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    """Recorded facts about a stored artifact."""

    id: str
    size: int
    mime: str | None
    project_id: str | None
    producing_op: str | None
    created: str | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "size": self.size,
            "mime": self.mime,
            "project_id": self.project_id,
            "producing_op": self.producing_op,
            "created": self.created,
            "meta": self.meta,
        }


def _iter_source(source: bytes | bytearray | str | PathLike[str]) -> Iterator[bytes]:
    """Yield ``source`` as chunks: in-memory bytes in one piece, a file path streamed."""
    if isinstance(source, (bytes, bytearray)):
        yield bytes(source)
        return
    with open(source, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                return
            yield chunk


class ArtifactStore:
    """A content-addressed store rooted at a directory, with metadata on the event log."""

    def __init__(self, root: str | PathLike[str], log: EventLog, *, create: bool = True) -> None:
        self._root = Path(root)
        if create:
            self._root.mkdir(parents=True, exist_ok=True)
        elif not self._root.is_dir():
            # The read-only serve path passes create=False so a mistyped --artifacts
            # is a clear error, not a freshly created empty directory.
            raise FileNotFoundError(f"no artifact store directory at {self._root}")
        self._log = log

    def _path_for(self, artifact_id: str) -> Path:
        if not _ID_RE.fullmatch(artifact_id):
            raise InvalidArtifactId(
                f"artifact id must be 64 lowercase hex characters, got {artifact_id!r}"
            )
        return self._root / artifact_id[:2] / artifact_id[2:4] / artifact_id

    def _verify_file(self, path: Path, artifact_id: str) -> bool:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == artifact_id

    def put(
        self,
        source: bytes | bytearray | str | PathLike[str],
        *,
        actor: Actor,
        meta: dict[str, Any] | None = None,
        mime: str | None = None,
        project_id: str | None = None,
        producing_op: str | None = None,
        max_bytes: int | None = None,
    ) -> str:
        """Store content and return its SHA-256 id. Identical content is stored once (deduped).

        The bytes are streamed to a temp file while hashing, fsynced, then atomically
        renamed into place. If a copy already exists but is corrupt, it is repaired
        from these bytes. The first store's metadata is the recorded metadata. When
        ``max_bytes`` is set, no bytes beyond that limit are written.
        """
        if max_bytes is not None and (
            isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer or None")
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = self._root / f".incoming-{secrets.token_hex(8)}.tmp"
        hasher = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as out:
                for chunk in _iter_source(source):
                    next_size = size + len(chunk)
                    if max_bytes is not None and next_size > max_bytes:
                        raise ArtifactTooLarge(
                            f"source exceeds the configured limit of {max_bytes} bytes"
                        )
                    hasher.update(chunk)
                    size = next_size
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            artifact_id = hasher.hexdigest()
            path = self._path_for(artifact_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and self._verify_file(path, artifact_id):
                tmp.unlink()  # a good copy is already there: discard the temp (dedupe)
            else:
                os.replace(tmp, path)  # fresh write, or repair of a corrupt existing copy
            tmp = None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

        key = f"artifact:{artifact_id}"
        if self._log.get_by_idempotency_key(key) is None:
            payload = {
                "artifact_id": artifact_id,
                "size": size,
                "mime": mime,
                "project_id": project_id,
                "producing_op": producing_op,
                "meta": dict(meta or {}),
            }
            # If a concurrent put recorded the same content with different metadata,
            # the first writer wins -- which is exactly the dedupe contract.
            with contextlib.suppress(IdempotencyConflict):
                self._log.append(
                    EVENT_ARTIFACT_ADDED,
                    actor,
                    payload,
                    project_id=project_id,
                    idempotency_key=key,
                )
        return artifact_id

    def has(self, artifact_id: str) -> bool:
        """Return whether content for this id is present on disk."""
        return self._path_for(artifact_id).exists()

    def get_meta(self, artifact_id: str) -> ArtifactMeta | None:
        """Return the recorded metadata for an artifact, or ``None`` if it is unknown."""
        event = self._log.get_by_idempotency_key(f"artifact:{artifact_id}")
        if event is None:
            return None
        # The key is claimable by any writer, so read the payload defensively: a
        # malformed record yields empty fields rather than an exception on every read.
        payload = event.payload if isinstance(event.payload, dict) else {}
        size = payload.get("size")
        meta = payload.get("meta")

        def text(key: str) -> str | None:
            value = payload.get(key)
            return value if isinstance(value, str) else None

        return ArtifactMeta(
            id=artifact_id,
            size=size if isinstance(size, int) and not isinstance(size, bool) else 0,
            mime=text("mime"),
            project_id=text("project_id"),
            producing_op=text("producing_op"),
            created=event.ts,
            meta=meta if isinstance(meta, dict) else {},
        )

    def open(self, artifact_id: str) -> BinaryIO:
        """Open the stored content for reading, after verifying it against its hash."""
        path = self._path_for(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"no artifact content for {artifact_id!r}")
        if not self._verify_file(path, artifact_id):
            raise CorruptArtifact(f"stored content for {artifact_id!r} does not match its hash")
        return path.open("rb")

    def read_bytes(self, artifact_id: str) -> bytes:
        """Return the stored content as bytes, raising if it does not match its hash."""
        path = self._path_for(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"no artifact content for {artifact_id!r}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact_id:
            raise CorruptArtifact(f"stored content for {artifact_id!r} does not match its hash")
        return data
