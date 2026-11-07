"""Streaming size limits for the content-addressed artifact store."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import BinaryIO

import pytest

from aar.hub.artifacts import ArtifactStore, ArtifactTooLarge
from aar.hub.events import Actor, EventLog

ACTOR = Actor("agent", "bounded-writer")


def _store(tmp_path: Path) -> tuple[EventLog, ArtifactStore, Path]:
    log = EventLog(tmp_path / "hub.db")
    root = tmp_path / "artifacts"
    return log, ArtifactStore(root, log), root


@pytest.mark.parametrize("as_path", [False, True])
def test_exact_cap_succeeds_for_bytes_and_path(tmp_path: Path, as_path: bool) -> None:
    log, store, _ = _store(tmp_path)
    payload = b"exactly-at-cap"
    source: bytes | Path = payload
    if as_path:
        source = tmp_path / "source.bin"
        source.write_bytes(payload)

    artifact_id = store.put(source, actor=ACTOR, max_bytes=len(payload))

    assert store.read_bytes(artifact_id) == payload
    assert len(log.read(types=["artifact.added"])) == 1
    log.close()


@pytest.mark.parametrize("as_path", [False, True])
def test_over_cap_rejects_bytes_and_path_without_content_or_event(
    tmp_path: Path, as_path: bool
) -> None:
    log, store, root = _store(tmp_path)
    payload = b"one-byte-too-many"
    source: bytes | Path = payload
    if as_path:
        source = tmp_path / "source.bin"
        source.write_bytes(payload)

    with pytest.raises(ArtifactTooLarge):
        store.put(source, actor=ACTOR, max_bytes=len(payload) - 1)

    assert log.read(types=["artifact.added"]) == []
    assert list(root.glob("**/.incoming-*")) == []
    assert [path for path in root.rglob("*") if path.is_file()] == []
    if isinstance(source, Path):
        assert source.read_bytes() == payload
    log.close()


def test_source_open_failure_cleans_incoming_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log, store, root = _store(tmp_path)
    source = tmp_path / "unreadable.bin"
    source.write_bytes(b"source remains untouched")
    real_open = builtins.open

    def failing_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if Path(file) == source and mode == "rb":
            raise PermissionError("denied for test")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(PermissionError, match="denied for test"):
        store.put(source, actor=ACTOR, max_bytes=128)

    assert source.read_bytes() == b"source remains untouched"
    assert log.read(types=["artifact.added"]) == []
    assert list(root.glob("**/.incoming-*")) == []
    log.close()


def test_growing_path_is_rejected_with_bounded_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log, store, root = _store(tmp_path)
    source = tmp_path / "growing.bin"
    source.write_bytes(b"a" * 64)
    real_open = builtins.open
    read_sizes: list[int] = []

    class GrowingReader:
        def __init__(self, handle: BinaryIO) -> None:
            self._handle = handle
            self._grown = False

        def __enter__(self) -> GrowingReader:
            return self

        def __exit__(self, *args: object) -> None:
            self._handle.close()

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            if not self._grown:
                with real_open(source, "ab") as writer:
                    writer.write(b"b" * 4096)
                self._grown = True
            return self._handle.read(size)

    def tracking_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        handle = real_open(file, mode, *args, **kwargs)
        if Path(file) == source and mode == "rb":
            return GrowingReader(handle)
        return handle

    monkeypatch.setattr(builtins, "open", tracking_open)
    with pytest.raises(ArtifactTooLarge):
        store.put(source, actor=ACTOR, max_bytes=128)

    assert read_sizes and all(0 < size <= 1024 * 1024 for size in read_sizes)
    assert log.read(types=["artifact.added"]) == []
    assert list(root.glob("**/.incoming-*")) == []
    log.close()


@pytest.mark.parametrize("corrupt", [False, True])
def test_rejected_ingest_preserves_existing_valid_or_corrupt_content(
    tmp_path: Path, corrupt: bool
) -> None:
    log, store, _ = _store(tmp_path)
    payload = b"preexisting artifact bytes"
    artifact_id = store.put(payload, actor=ACTOR)
    path = store._path_for(artifact_id)
    if corrupt:
        path.write_bytes(b"corrupt but must remain untouched")
    before = path.read_bytes()
    events_before = log.read(types=["artifact.added"])

    with pytest.raises(ArtifactTooLarge):
        store.put(payload, actor=ACTOR, max_bytes=4)

    assert path.read_bytes() == before
    assert log.read(types=["artifact.added"]) == events_before
    assert list(path.parents[2].glob("**/.incoming-*")) == []
    log.close()


@pytest.mark.parametrize("bad_limit", [True, False, 0, -1, 1.5, "1"])
def test_max_bytes_requires_a_positive_integer(tmp_path: Path, bad_limit: object) -> None:
    log, store, root = _store(tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        store.put(b"data", actor=ACTOR, max_bytes=bad_limit)  # type: ignore[arg-type]

    assert list(root.iterdir()) == []
    assert log.read(types=["artifact.added"]) == []
    log.close()
