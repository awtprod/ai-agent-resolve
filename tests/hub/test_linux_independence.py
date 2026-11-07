"""Cold Linux workflow with real generated audio, no network and no Resolve.

This covers media ownership, numeric audio measurements, plan lint, QA and review
retrieval. Transcription, face models, rendering in Resolve and Windows execution
are deliberately outside this acceptance test.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from starlette.testclient import TestClient

from aar.cli import main
from aar.hub import reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.console import ConsoleConfig, build_app
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.media import read_pcm
from aar.media.store import MediaStore
from aar.node.gateway import ScriptApiGateway
from aar.node.runtime import NodeRuntime
from aar.plan.lint import lint
from aar.plan.model import EditPlan
from aar.plan.profile import load_profile
from aar.qa import QAReport, check

FFMPEG = shutil.which(os.environ.get("AAR_FFMPEG", "").strip() or "ffmpeg")
FFPROBE = shutil.which(os.environ.get("AAR_FFPROBE", "").strip() or "ffprobe")
pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="Linux-owned media acceptance"),
    pytest.mark.skipif(
        not (FFMPEG and FFPROBE),
        reason=(
            "ffmpeg/ffprobe not found on PATH or via AAR_FFMPEG/AAR_FFPROBE; media tests need them"
        ),
    ),
]


def test_cold_linux_media_plan_qa_and_review_without_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")

    def windows_or_network(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("cold Linux workflow attempted Resolve, a node runtime, or network I/O")

    monkeypatch.setenv("AAR_LIVE", "0")
    monkeypatch.setattr(ScriptApiGateway, "__init__", windows_or_network)
    monkeypatch.setattr(NodeRuntime, "__init__", windows_or_network)
    monkeypatch.setitem(
        sys.modules, "DaVinciResolveScript", SimpleNamespace(scriptapp=windows_or_network)
    )
    monkeypatch.setattr(socket.socket, "connect", windows_or_network)
    monkeypatch.setattr(socket.socket, "connect_ex", windows_or_network)
    monkeypatch.setattr(socket, "create_connection", windows_or_network)

    source_dir = tmp_path / "removable-source"
    source_dir.mkdir()
    source = source_dir / "fixture.wav"
    assert FFMPEG is not None
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.1*sin(2*PI*1000*t)|0.1*sin(2*PI*1000*t):s=48000:d=2",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        timeout=15,
        stdin=subprocess.DEVNULL,
    )
    source_bytes = source.read_bytes()
    assert 0 < len(source_bytes) < 500_000
    digest = hashlib.sha256(source_bytes).hexdigest()
    db, artifact_root = tmp_path / "hub.db", tmp_path / "artifacts"
    actor = Actor("agent", "linux-fixture")
    log = EventLog(db)
    try:
        ArtifactStore(artifact_root, log)
        project = Registry(log).create_project("Linux fixture", actor)
    finally:
        log.close()

    hub_args = ["--db", str(db), "--artifacts", str(artifact_root), "--project", project.id]
    # The production CLI owns its connection and closes it after ingest.
    assert main(["media", "ingest", str(source), *hub_args, "--actor", "agent:linux-fixture"]) == 0
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["sha256"] == digest
    assert ingested["size"] == len(source_bytes)
    media_ref = ingested["media_ref"]
    source.unlink()
    source_dir.rmdir()
    assert not source.exists()

    assert main(["media", "resolve", media_ref, *hub_args]) == 0
    resolved = Path(json.loads(capsys.readouterr().out)["local_path"])
    assert resolved.is_absolute() and resolved.is_relative_to(artifact_root)
    assert resolved != source and resolved.read_bytes() == source_bytes

    log = EventLog(db, create=False)
    try:
        store = ArtifactStore(artifact_root, log, create=False)
        media = MediaStore(log, store)
        assert media.resolve(media_ref, project_id=project.id) == resolved
        association = media.get_association(media_ref, project_id=project.id)
        assert association is not None and association.original_filename == "fixture.wav"
        assert store.read_bytes(digest) == source_bytes

        # Real ffmpeg PCM decode plus independent numpy measurement after source removal.
        pcm = read_pcm(resolved, start_s=0, duration_s=2, channels=2, timeout_s=10)
        samples = np.asarray(pcm.samples, dtype=np.float64)
        assert samples.size == 2 * 48_000 * 2
        rms = float(np.sqrt(np.mean(samples**2)))
        assert math.isfinite(rms) and rms == pytest.approx(0.1 / math.sqrt(2), abs=0.0001)

        plan = EditPlan.model_validate(
            {
                "version": 1,
                "project": project.name,
                "timeline": "Linux cold fixture",
                "kind": "longform",
                "fps": "25",
                "resolution": {"width": 160, "height": 90},
                "profile": "ccn",
                "sources": [{"id": "linux-audio", "media": str(resolved), "duration": 50}],
                "segments": [{"track": 1, "source": "linux-audio", "src_in": 0, "src_out": 50}],
            }
        )
        profile = load_profile("ccn")
        linted = lint(plan, profile)
        assert linted.ok, linted.model_dump_json()
        assert linted.duration_frames == 50 and linted.duration_s == 2
        plan_id = store.put(
            plan.model_dump_json().encode(),
            actor=actor,
            project_id=project.id,
            mime="application/json",
        )
        report = check(resolved, kind="longform", profile=profile, plan=plan, timeout_s=10)
        assert report.check("decode").status == "pass", report.model_dump_json()
        for check_id, keys in {
            "loudness": ("integrated_lufs", "true_peak_dbtp"),
            "clipping": ("sample_peak_dbfs", "clipped_samples"),
            "longest-silence": ("silences", "longest_silence_s"),
            "duration": ("diff_frames",),
        }.items():
            entry = report.check(check_id)
            assert not entry.incomplete and entry.skip_reason is None
            for key in keys:
                value = entry.measured[key]
                assert type(value) in (int, float) and math.isfinite(value), (check_id, key, value)
        assert report.check("duration").status == "pass"
        assert report.check("loudness").measured["integrated_lufs"] == pytest.approx(-20, abs=0.2)
        # A -20 LUFS tone misses the -22 LUFS house target: a quality finding, not a tool failure.
        assert {entry.id for entry in report.checks if entry.status == "fail"} == {"loudness"}
        assert {
            entry.id: entry.skip_reason for entry in report.checks if entry.status == "skipped"
        } == {
            "size-cap": "not-applicable",
            "join-retranscribe": "no-transcript",
            "caption-coverage": "no-transcript",
        }
        qa_bytes = report.model_dump_json().encode()
        op = WorkGraph(log, project.id).add_node("op", actor, title="Linux fixture QA")
        qa_id = store.put(
            qa_bytes,
            actor=actor,
            project_id=project.id,
            producing_op=op.id,
            mime="application/json",
            meta={"kind": "qa", "plan_id": plan_id, "media_ref": media_ref, "rms": rms},
        )
        review = reviews.request_review(
            log, store, project.id, op.id, [qa_id], actor, title="Linux fixture QA review", fps=25
        )
        reviews.add_note(
            log, project.id, review.id, 0, "Synthetic tone: loudness finding expected.", actor
        )
    finally:
        log.close()

    # A new read-only hub and console retrieve persisted content, with no original or node.
    log = EventLog(db, create=False, read_only=True)
    try:
        store = ArtifactStore(artifact_root, log, create=False)
        assert Registry(log).get(project.id) is not None
        assert MediaStore(log, store).resolve(media_ref, project_id=project.id) == resolved
        saved_plan = EditPlan.model_validate_json(store.read_bytes(plan_id))
        assert saved_plan.sources[0].media == str(resolved)
        assert lint(saved_plan, profile).ok
        assert resolved.read_bytes() == source_bytes and not source.exists()
        saved_review = reviews.get_review(log, project.id, review.id)
        assert saved_review is not None and saved_review.renders == (qa_id,)
        assert saved_review.notes[0].text == "Synthetic tone: loudness finding expected."
        before = log.latest_seq()
        with TestClient(
            build_app(log, store, ConsoleConfig()), base_url="http://127.0.0.1:8788"
        ) as client:
            for _ in range(2):
                page = client.get(f"/reviews/{project.id}/{review.id}")
                assert page.status_code == 200 and review.title in page.text
                artifact = client.get(f"/artifacts/{qa_id}")
                assert artifact.status_code == 200 and artifact.content == qa_bytes
                assert QAReport.model_validate_json(artifact.content) == report
                assert client.get(f"/artifacts/{digest}").content == source_bytes
                plan_artifact = client.get(f"/artifacts/{plan_id}")
                assert plan_artifact.status_code == 200
                assert EditPlan.model_validate_json(plan_artifact.content) == saved_plan
        assert log.latest_seq() == before
    finally:
        log.close()
