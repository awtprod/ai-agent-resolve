"""Resume and idempotency: a re-run after a crash (at any step) makes no duplicate events.

The deterministic idempotency keys and the content-addressed store mean a full re-run after
success writes nothing new and does not transcode again. Partial states are reconstructed with the
same public writes and the library's own key builders, so a non-deterministic key would duplicate.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

from deliver_support import (
    AGENT,
    FakeEditHost,
    FakeTranscoder,
    Hub,
    make_hub,
    make_profile,
    qa_pass,
    render_output,
)

from aar.deliver.service import _delivery_dir, _review_keys, deliver_review
from aar.hub.facts import record_fact

MB = 1_000_000


def _review(hub: Hub, sources: Path, workdir: Path, *, source: str, cap_mb: float, transcoder):
    return deliver_review(
        project_id=hub.project_id,
        subject_id=hub.subject_id,
        deliverable="short-1",
        actor=AGENT,
        profile=make_profile(review_mp4_max_mb=cap_mb),
        title="Acme review",
        timeline="Main",
        fps=25.0,
        kind="short",
        render_output=render_output(sources / source, render_job_id="rj-1"),
        workdir=str(workdir),
        edit_host=FakeEditHost(sources),
        log=hub.log,
        store=hub.store,
        transcoder=transcoder,
        qa_check=qa_pass(),
    )


def _counts(hub: Hub) -> Counter[str]:
    return Counter(e.type for e in hub.log.iter_events())


def _ddir(hub: Hub, workdir: Path) -> Path:
    return Path(_delivery_dir(str(workdir), hub.project_id, "short-1", "rj-1"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- full re-run after success: nothing new ----------------------------------------------------


def test_full_rerun_use_as_is_makes_no_new_events(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    first = _review(
        hub, sources, work, source="render_720p.mp4", cap_mb=5.0, transcoder=FakeTranscoder([10])
    )
    before = _counts(hub)
    second_tc = FakeTranscoder([10])
    second = _review(hub, sources, work, source="render_720p.mp4", cap_mb=5.0, transcoder=second_tc)
    assert _counts(hub) == before  # no new events
    assert second.artifact_id == first.artifact_id
    assert second_tc.calls == []
    assert "pull" in second.resumed_steps


def test_full_rerun_transcoded_does_not_transcode_again(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    first = _review(
        hub,
        sources,
        work,
        source="render_1080p.mp4",
        cap_mb=0.5,
        transcoder=FakeTranscoder([int(0.4 * MB)]),
    )
    before = _counts(hub)
    second_tc = FakeTranscoder([int(0.4 * MB)])
    second = _review(
        hub, sources, work, source="render_1080p.mp4", cap_mb=0.5, transcoder=second_tc
    )
    assert _counts(hub) == before
    assert second.artifact_id == first.artifact_id
    assert second_tc.calls == []  # the review file was reused
    assert "transcode" in second.resumed_steps


# -- crash after pull: the local source is reused, never re-pulled -----------------------------


def test_resume_after_pull_skips_the_pull(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    ddir = _ddir(hub, work)
    shutil.copyfile(sources / "render_720p.mp4", ddir / "source.mp4")  # a crashed run left this
    host = FakeEditHost(sources)
    result = deliver_review(
        project_id=hub.project_id,
        subject_id=hub.subject_id,
        deliverable="short-1",
        actor=AGENT,
        profile=make_profile(review_mp4_max_mb=5.0),
        title="t",
        timeline=None,
        fps=None,
        kind="short",
        render_output=render_output(sources / "render_720p.mp4", render_job_id="rj-1"),
        workdir=str(work),
        edit_host=host,
        log=hub.log,
        store=hub.store,
        transcoder=FakeTranscoder([10]),
        qa_check=qa_pass(),
    )
    assert host.pull_calls == []  # never re-pulled
    assert host.hash_calls == [str(sources / "render_720p.mp4")]  # only the hash check
    assert "pull" in result.resumed_steps


# -- crash after transcode: the review file is reused ------------------------------------------


def test_resume_after_transcode_reuses_the_review_file(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    ddir = _ddir(hub, work)
    shutil.copyfile(sources / "render_1080p.mp4", ddir / "source.mp4")
    (ddir / "review.mp4").write_bytes(b"\x00" * int(0.3 * MB))  # the prior transcode output
    transcoder = FakeTranscoder([int(0.3 * MB)])
    result = _review(
        hub, sources, work, source="render_1080p.mp4", cap_mb=0.5, transcoder=transcoder
    )
    assert transcoder.calls == []
    assert "transcode" in result.resumed_steps


# -- crash after put: the artifact is not duplicated -------------------------------------------


def test_resume_after_put_does_not_duplicate_the_artifact(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    ddir = _ddir(hub, work)
    shutil.copyfile(sources / "render_720p.mp4", ddir / "source.mp4")
    # A crash after the store left the artifact (and its event) behind.
    artifact_id = hub.store.put(
        ddir / "source.mp4",
        actor=AGENT,
        meta={},
        mime="video/mp4",
        project_id=hub.project_id,
        producing_op="deliver.review",
    )
    assert len(list(hub.log.iter_events(types=["artifact.added"]))) == 1
    result = _review(
        hub, sources, work, source="render_720p.mp4", cap_mb=5.0, transcoder=FakeTranscoder([10])
    )
    assert result.artifact_id == artifact_id
    assert len(list(hub.log.iter_events(types=["artifact.added"]))) == 1  # not duplicated
    assert result.review_id is not None  # the review still got requested


# -- crash after facts: the facts are not duplicated (keys are deterministic) ------------------


def test_resume_after_facts_does_not_duplicate_facts(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    ddir = _ddir(hub, work)
    shutil.copyfile(sources / "render_720p.mp4", ddir / "source.mp4")
    source = ddir / "source.mp4"
    artifact_id = hub.store.put(
        source,
        actor=AGENT,
        meta={},
        mime="video/mp4",
        project_id=hub.project_id,
        producing_op="deliver.review",
    )
    size = source.stat().st_size
    keys = _review_keys("short-1", "rj-1", artifact_id)
    src = {"artifact": artifact_id}
    record_fact(
        hub.log,
        hub.project_id,
        "render.hash.short-1",
        {"sha256": artifact_id, "bytes": size},
        AGENT,
        idempotency_key=keys["hash"],
        source=src,
    )
    record_fact(
        hub.log,
        hub.project_id,
        "qa.size.short-1",
        {"value": round(size / MB, 6), "unit": "MB"},
        AGENT,
        idempotency_key=keys["qa_size"],
        source=src,
    )
    assert len(list(hub.log.iter_events(types=["facts.recorded"]))) == 2

    result = _review(
        hub, sources, work, source="render_720p.mp4", cap_mb=5.0, transcoder=FakeTranscoder([10])
    )
    # Same deterministic keys: the facts replay, no new events; the review still gets requested.
    assert len(list(hub.log.iter_events(types=["facts.recorded"]))) == 2
    assert "facts" in result.resumed_steps
    assert result.review_id is not None
