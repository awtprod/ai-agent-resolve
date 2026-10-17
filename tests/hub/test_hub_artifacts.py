"""Tests for the content-addressed artifact store."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aar.hub.artifacts import ArtifactStore, CorruptArtifact, InvalidArtifactId
from aar.hub.events import Actor, EventLog

ACTOR = Actor("agent", "renderer")


def _store(tmp_path: Path) -> tuple[EventLog, ArtifactStore]:
    log = EventLog(tmp_path / "hub.db")
    return log, ArtifactStore(tmp_path / "artifacts", log)


def test_put_returns_sha256_and_records_metadata_event(tmp_path: Path) -> None:
    log, store = _store(tmp_path)
    data = b"hello world"
    artifact_id = store.put(
        data, actor=ACTOR, mime="text/plain", project_id="prj_1", producing_op="op_render"
    )

    assert artifact_id == hashlib.sha256(data).hexdigest()
    assert store.has(artifact_id)

    events = log.read(types=["artifact.added"])
    assert len(events) == 1

    meta = store.get_meta(artifact_id)
    assert meta is not None
    assert meta.size == len(data)
    assert meta.mime == "text/plain"
    assert meta.project_id == "prj_1"
    assert meta.producing_op == "op_render"
    assert meta.created is not None


def test_identical_content_is_deduped(tmp_path: Path) -> None:
    log, store = _store(tmp_path)
    first = store.put(b"same bytes", actor=ACTOR, mime="text/plain", producing_op="op_a")
    second = store.put(b"same bytes", actor=ACTOR, mime="text/plain", producing_op="op_b")

    assert first == second
    assert len(log.read(types=["artifact.added"])) == 1
    # First writer wins: the recorded producing op is the original one.
    assert store.get_meta(first).producing_op == "op_a"


def test_put_from_path_streams_and_roundtrips(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    source = tmp_path / "input.bin"
    payload = b"\x00\x01\x02binary-payload" * 100_000  # a few MB, forcing multiple chunks
    source.write_bytes(payload)

    artifact_id = store.put(source, actor=ACTOR)
    assert artifact_id == hashlib.sha256(payload).hexdigest()
    assert store.read_bytes(artifact_id) == payload
    with store.open(artifact_id) as handle:
        assert handle.read() == payload


def test_put_leaves_no_temp_files(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    store.put(b"abc", actor=ACTOR)
    leftovers = list((tmp_path / "artifacts").glob("**/.incoming-*"))
    assert leftovers == []


def test_invalid_artifact_ids_are_rejected(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    for bad in (
        "/etc/passwd",  # absolute path
        "../../etc/passwd",  # traversal
        "../" + "a" * 61,  # traversal, right length otherwise
        "z" * 64,  # non-hex
        "ABCD" + "0" * 60,  # uppercase hex is not accepted
        "abc",  # too short
    ):
        with pytest.raises(InvalidArtifactId):
            store.has(bad)


def test_get_meta_for_unknown_returns_none_and_open_raises(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    assert store.get_meta("deadbeef") is None
    with pytest.raises(FileNotFoundError):
        store.open("0" * 64)


def test_corruption_is_detected_on_read(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    artifact_id = store.put(b"good bytes", actor=ACTOR)
    store._path_for(artifact_id).write_bytes(b"tampered")

    with pytest.raises(CorruptArtifact):
        store.read_bytes(artifact_id)
    with pytest.raises(CorruptArtifact):
        store.open(artifact_id)


def test_put_repairs_a_corrupt_existing_copy(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    artifact_id = store.put(b"good bytes", actor=ACTOR)
    store._path_for(artifact_id).write_bytes(b"tampered")

    again = store.put(b"good bytes", actor=ACTOR)
    assert again == artifact_id
    assert store.read_bytes(artifact_id) == b"good bytes"
