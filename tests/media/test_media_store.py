"""Linux-resident source media ingestion and verified project-scoped resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from media_support import needs_ffmpeg, render

from aar.cli import main
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.registry import Registry
from aar.media import decode_check, probe
from aar.media.store import (
    EVENT_MEDIA_INGESTED,
    InvalidMediaInput,
    MediaStore,
    MediaTooLarge,
    MediaUnavailable,
    UnknownProject,
)

HUMAN = Actor("human", "operator")
AGENT = Actor("agent", "cold-worker")


def hub(tmp_path: Path) -> tuple[Path, Path, EventLog, str, str, MediaStore]:
    db = tmp_path / "events.sqlite"
    artifacts = tmp_path / "artifacts"
    log = EventLog(db)
    registry = Registry(log)
    first = registry.create_project("First", HUMAN, project_id="prj_first").id
    second = registry.create_project("Second", HUMAN, project_id="prj_second").id
    store = MediaStore(log, ArtifactStore(artifacts, log))
    return db, artifacts, log, first, second, store


@needs_ffmpeg
def test_ingest_survives_source_removal_and_cold_tools_use_resolved_path(tmp_path: Path) -> None:
    db, artifacts, log, project, _, store = hub(tmp_path)
    source = render(tmp_path / "original review.mp4", frames=12)
    expected_size = source.stat().st_size

    record = store.ingest(source, project_id=project, actor=AGENT)
    assert record.media_ref == f"media:{record.sha256}"
    assert record.original_filename == source.name
    assert record.size == expected_size
    assert record.local_path.is_file()
    assert record.local_path != source

    source.unlink()
    log.close()
    reopened = EventLog(db, create=False)
    reopened_store = MediaStore(reopened, ArtifactStore(artifacts, reopened, create=False))
    resolved = reopened_store.resolve(record.media_ref, project_id=project)
    info = probe(resolved)
    assert info.size_bytes == expected_size and info.video is not None
    assert decode_check(resolved, timeout_s=30).ok
    reopened.close()


def test_same_project_is_idempotent_and_cross_project_associations_are_independent(
    tmp_path: Path,
) -> None:
    _, _, log, first, second, store = hub(tmp_path)
    source = tmp_path / "clip-one.mov"
    source.write_bytes(b"same source bytes")

    one = store.ingest(source, project_id=first, actor=AGENT)
    renamed = tmp_path / "renamed.mov"
    renamed.write_bytes(source.read_bytes())
    again = store.ingest(renamed, project_id=first, actor=AGENT)
    other = store.ingest(renamed, project_id=second, actor=AGENT)

    assert again == one
    assert other.media_ref == one.media_ref
    assert other.project_id == second
    associations = log.read(types=[EVENT_MEDIA_INGESTED])
    assert [(event.project_id, event.payload["original_filename"]) for event in associations] == [
        (first, "clip-one.mov"),
        (second, "renamed.mov"),
    ]
    assert len(log.read(types=["artifact.added"])) == 1
    log.close()


def test_missing_and_corrupt_content_are_explicitly_unavailable(tmp_path: Path) -> None:
    _, _, log, project, _, store = hub(tmp_path)
    missing_source = tmp_path / "missing.mov"
    corrupt_source = tmp_path / "corrupt.mov"
    missing_source.write_bytes(b"missing content")
    corrupt_source.write_bytes(b"corrupt content")
    missing = store.ingest(missing_source, project_id=project, actor=AGENT)
    corrupt = store.ingest(corrupt_source, project_id=project, actor=AGENT)

    missing.local_path.unlink()
    corrupt.local_path.write_bytes(b"tampered")
    assert store.availability(missing.media_ref, project_id=project).reason == "missing"
    assert store.availability(corrupt.media_ref, project_id=project).reason == "corrupt"
    with pytest.raises(MediaUnavailable):
        store.resolve(missing.media_ref, project_id=project)
    with pytest.raises(MediaUnavailable):
        store.resolve(corrupt.media_ref, project_id=project)
    log.close()


def test_project_actor_reference_filename_and_size_are_validated(tmp_path: Path) -> None:
    _, _, log, project, _, store = hub(tmp_path)
    source = tmp_path / "source.mov"
    source.write_bytes(b"123456789")

    with pytest.raises(UnknownProject):
        store.ingest(source, project_id="prj_unknown", actor=AGENT)
    with pytest.raises(InvalidMediaInput):
        store.ingest(source, project_id=project, actor=Actor("agent", "bad:id"))
    with pytest.raises(InvalidMediaInput):
        store.availability("not-a-reference", project_id=project)
    with pytest.raises(InvalidMediaInput):
        store.ingest(source, project_id=project, actor=AGENT, max_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(MediaTooLarge):
        store.ingest(source, project_id=project, actor=AGENT, max_bytes=8)
    assert log.read(types=[EVENT_MEDIA_INGESTED]) == []
    assert log.read(types=["artifact.added"]) == []
    assert source.read_bytes() == b"123456789"
    log.close()


def test_size_limit_rejects_source_that_grows_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, log, project, _, store = hub(tmp_path)
    source = tmp_path / "growing.mov"
    source.write_bytes(b"x" * 64)

    original = Path.stat

    def stale_stat(path: Path, *args: object, **kwargs: object):
        result = original(path, *args, **kwargs)
        if path == source:
            source.write_bytes(b"x" * 4096)
        return result

    monkeypatch.setattr(Path, "stat", stale_stat)
    with pytest.raises(MediaTooLarge):
        store.ingest(source, project_id=project, actor=AGENT, max_bytes=128)
    assert log.read(types=[EVENT_MEDIA_INGESTED]) == []
    assert log.read(types=["artifact.added"]) == []
    log.close()


def test_bounded_ingest_passes_original_source_directly_to_artifact_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, log, project, _, store = hub(tmp_path)
    source = tmp_path / "direct.mov"
    payload = b"single streaming copy"
    source.write_bytes(payload)
    original_put = ArtifactStore.put
    seen_calls: list[tuple[object, object]] = []

    def recording_put(self: ArtifactStore, put_source: object, **kwargs: object) -> str:
        seen_calls.append((put_source, kwargs.get("max_bytes")))
        return original_put(self, put_source, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArtifactStore, "put", recording_put)
    record = store.ingest(source, project_id=project, actor=AGENT, max_bytes=len(payload))

    assert seen_calls == [(source, len(payload))]
    assert record.local_path.read_bytes() == payload
    assert source.read_bytes() == payload
    log.close()


def test_cli_prints_json_and_uses_meaningful_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, artifacts, log, project, _, _ = hub(tmp_path)
    log.close()
    source = tmp_path / "cli.mov"
    source.write_bytes(b"cli source")
    base = ["--db", str(db), "--artifacts", str(artifacts), "--project", project]

    assert main(["media", "ingest", str(source), *base, "--actor", "agent:cli-worker"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["ok"] and Path(record["local_path"]).is_file()

    assert main(["media", "resolve", record["media_ref"], *base]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["available"] and resolved["local_path"] == record["local_path"]

    Path(record["local_path"]).unlink()
    assert main(["media", "availability", record["media_ref"], *base]) == 1
    unavailable = json.loads(capsys.readouterr().out)
    assert unavailable["available"] is False and unavailable["reason"] == "missing"

    unknown = ["media", "ingest", str(source), *base[:-1], "prj_unknown", "--actor", "agent:x"]
    assert main(unknown) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "unknown_project"

    too_large = [
        "media",
        "ingest",
        str(source),
        *base,
        "--actor",
        "agent:x",
        "--max-bytes",
        "1",
    ]
    assert main(too_large) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "media_too_large"
