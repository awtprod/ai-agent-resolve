"""Regression pins for the round-2 mutants: boundaries, validation, facts, QA kind, mime."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import pytest
from deliver_support import (
    AGENT,
    FakeEditHost,
    FakeTranscoder,
    Hub,
    make_hub,
    make_profile,
    make_source,
    qa_fail,
    qa_pass,
    render_output,
)

from aar.deliver.errors import DeliverError
from aar.deliver.service import (
    _delivery_dir,
    _video_kbps,
    deliver_master,
    deliver_review,
)
from aar.hub.facts import facts_view

MB = 1_000_000


def _review(hub: Hub, sources, workdir, *, source, cap_mb, transcoder, qa=None, host=None, **kw):
    return deliver_review(
        project_id=kw.get("project_id", hub.project_id),
        subject_id=hub.subject_id,
        deliverable=kw.get("deliverable", "short-1"),
        actor=AGENT,
        profile=make_profile(review_mp4_max_mb=cap_mb),
        title="Acme review",
        timeline="Main",
        fps=25.0,
        kind=kw.get("kind", "short"),
        render_output=render_output(
            sources / source,
            duration=kw.get("duration", 6.0),
            render_job_id=kw.get("render_job_id", "rj-1"),
        ),
        workdir=str(workdir),
        edit_host=host or FakeEditHost(sources),
        log=kw.get("log", hub.log),
        store=kw.get("store", hub.store),
        transcoder=transcoder,
        qa_check=qa or qa_pass(),
    )


def _master(hub: Hub, sources, workdir, *, source="render_720p.mp4", qa=None, kind="longform"):
    return deliver_master(
        project_id=hub.project_id,
        deliverable="m-1",
        actor=AGENT,
        profile=make_profile(),
        kind=kind,
        render_output=render_output(sources / source, render_job_id="rj-9"),
        workdir=str(workdir),
        edit_host=FakeEditHost(sources),
        log=hub.log,
        store=hub.store,
        qa_check=qa or qa_pass(),
    )


def test_v12_file_exactly_at_cap_is_stored_as_is(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    size = (sources / "render_720p.mp4").stat().st_size
    cap_mb = size / MB
    while int(cap_mb * MB) < size:
        cap_mb = math.nextafter(cap_mb, math.inf)
    assert int(cap_mb * MB) == size
    tc = FakeTranscoder([10])
    result = _review(
        hub, sources, tmp_path / "w", source="render_720p.mp4", cap_mb=cap_mb, transcoder=tc
    )
    assert result.transcoded is False
    assert tc.calls == []


def test_v13_transcode_exactly_at_cap_succeeds(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    cap_bytes = 5 * MB
    tc = FakeTranscoder([cap_bytes])
    result = _review(
        hub, sources, tmp_path / "w", source="render_1080p.mp4", cap_mb=5.0, transcoder=tc
    )
    assert result.transcoded is True
    assert result.size == cap_bytes
    assert len(tc.calls) == 1


def test_v14_exactly_250_kbps_is_allowed(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    cap_mb = 0.3085
    assert _video_kbps(int(cap_mb * MB), 6.0) == 250
    tc = FakeTranscoder([1000])
    result = _review(
        hub, sources, tmp_path / "w", source="render_1080p.mp4", cap_mb=cap_mb, transcoder=tc
    )
    assert tc.calls[0]["video_kbps"] == 250
    assert result.transcoded is True


def test_v21_pull_never_overwrites(tmp_path: Path, sources: Path) -> None:
    seen: list[bool] = []

    class Spy(FakeEditHost):
        def pull(self, remote, local, *, overwrite=False):
            seen.append(overwrite)
            return super().pull(remote, local, overwrite=overwrite)

    hub = make_hub(tmp_path)
    _review(
        hub,
        sources,
        tmp_path / "w",
        source="render_720p.mp4",
        cap_mb=5.0,
        transcoder=FakeTranscoder([10]),
        host=Spy(sources),
    )
    assert seen == [False]


def test_v22_sha_is_lowercase(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    result = _review(
        hub,
        sources,
        tmp_path / "w",
        source="render_720p.mp4",
        cap_mb=5.0,
        transcoder=FakeTranscoder([10]),
    )
    assert result.sha256 == result.sha256.lower()
    meta = hub.store.get_meta(result.artifact_id)
    assert meta.meta["source_sha256"] == meta.meta["source_sha256"].lower()
    assert meta.meta["source_sha256"] != meta.meta["source_sha256"].upper()


def test_v24_hash_fact_bytes_is_the_review_size(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    result = _review(
        hub, sources, work, source="render_1080p.mp4", cap_mb=5.0, transcoder=FakeTranscoder([10])
    )
    review_file = work / hub.project_id / "short-1" / "rj-1" / "review.mp4"
    source_size = (sources / "render_1080p.mp4").stat().st_size
    review_size = review_file.stat().st_size
    assert review_size != source_size
    fact = facts_view(hub.log, hub.project_id).get("render.hash.review.short-1")
    assert fact.current.value["bytes"] == review_size
    assert result.size == review_size


def test_v26_post_qa_failure_reports_qa_as_the_last_step(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)

    class BoomStore:
        def put(self, *a, **k):
            raise DeliverError("boom", "store broke", "x")

    with pytest.raises(DeliverError) as info:
        _review(
            hub,
            sources,
            tmp_path / "w",
            source="render_1080p.mp4",
            cap_mb=5.0,
            transcoder=FakeTranscoder([10]),
            store=BoomStore(),
        )
    assert info.value.detail["completed_steps"] == ["pull", "transcode", "qa"]


def test_v28_rerun_reports_review_resumed(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    kw = dict(source="render_720p.mp4", cap_mb=5.0, transcoder=FakeTranscoder([10]))
    first = _review(hub, sources, tmp_path / "w", **kw)
    assert "review" not in first.resumed_steps
    second = _review(hub, sources, tmp_path / "w", **kw)
    assert "review" in second.resumed_steps


@pytest.mark.parametrize("field", ["project_id", "render_job_id", "deliverable"])
@pytest.mark.parametrize("bad", ["..", ".", "rj-1\n", "a/b", "a\\b", "../x", "x/.."])
def test_v29_v30_v31_hostile_ids_refused(tmp_path: Path, sources: Path, field: str, bad: str):
    hub = make_hub(tmp_path)
    kw = {"source": "render_720p.mp4", "cap_mb": 5.0, "transcoder": FakeTranscoder([10])}
    with pytest.raises(DeliverError) as info:
        _review(hub, sources, tmp_path / "w", **{field: bad}, **kw)
    assert info.value.code == "invalid_name"
    with pytest.raises(DeliverError) as info:
        deliver_master(
            project_id=bad if field == "project_id" else hub.project_id,
            deliverable=bad if field == "deliverable" else "m-1",
            actor=AGENT,
            profile=make_profile(),
            kind="longform",
            render_output=render_output(
                sources / "render_720p.mp4",
                render_job_id=bad if field == "render_job_id" else "rj-1",
            ),
            workdir=str(tmp_path / "w"),
            edit_host=FakeEditHost(sources),
            log=hub.log,
            store=hub.store,
            qa_check=qa_pass(),
        )
    assert info.value.code == "invalid_name"


@pytest.mark.parametrize(
    "workdir",
    [
        "relative/work",
        "work",
        os.sep + "tmp" + os.sep + "aar-w",
        os.sep + "var" + os.sep + "tmp" + os.sep + "aar-w",
        os.sep + "dev" + os.sep + "shm" + os.sep + "aar-w",
    ],
)
def test_v32_relative_and_temp_workdirs_refused(tmp_path: Path, sources: Path, workdir: str):
    hub = make_hub(tmp_path)
    with pytest.raises(DeliverError) as info:
        deliver_review(
            project_id=hub.project_id,
            subject_id=hub.subject_id,
            deliverable="short-1",
            actor=AGENT,
            profile=make_profile(),
            title="t",
            timeline=None,
            fps=None,
            kind="short",
            render_output=render_output(sources / "render_720p.mp4"),
            workdir=workdir,
            edit_host=FakeEditHost(sources),
            log=hub.log,
            store=hub.store,
            transcoder=FakeTranscoder([10]),
            qa_check=qa_pass(),
        )
    assert info.value.code == "invalid_workdir"


def test_v37_master_qa_keeps_the_content_kind(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    qa = qa_pass("short")
    _master(hub, sources, tmp_path / "w", qa=qa, kind="short")
    assert qa.calls[-1][1] == "short"


def test_v38_mov_master_mime_follows_suffix(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    edit = tmp_path / "edit"
    edit.mkdir()
    shutil.copyfile(sources / "render_720p.mp4", edit / "master.mov")
    result = _master(hub, edit, tmp_path / "w", source="master.mov")
    assert hub.store.get_meta(result.artifact_id).mime == "video/quicktime"


def _ddir(hub: Hub, workdir: Path) -> Path:
    d = Path(_delivery_dir(str(workdir), hub.project_id, "short-1", "rj-1"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_v16_over_cap_leftover_review_is_rebuilt(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ddir = _ddir(hub, work)
    shutil.copyfile(sources / "render_1080p.mp4", ddir / "source.mp4")
    make_source(ddir / "review.mp4", width=1280, height=720, duration=3.0, video_kbps=2000)
    cap_mb = 0.5
    assert (ddir / "review.mp4").stat().st_size > cap_mb * MB
    tc = FakeTranscoder([int(0.3 * MB)])
    result = _review(hub, sources, work, source="render_1080p.mp4", cap_mb=cap_mb, transcoder=tc)
    assert tc.calls != []
    assert "transcode" not in result.resumed_steps


def test_v17_resume_reuse_still_reports_video_kbps(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ddir = _ddir(hub, work)
    shutil.copyfile(sources / "render_1080p.mp4", ddir / "source.mp4")
    make_source(ddir / "review.mp4", width=1280, height=720, duration=1.0, video_kbps=200)
    tc = FakeTranscoder([int(0.3 * MB)])
    result = _review(hub, sources, work, source="render_1080p.mp4", cap_mb=0.5, transcoder=tc)
    assert "transcode" in result.resumed_steps
    assert tc.calls == []
    assert result.video_kbps is not None


def test_v23_review_qa_runs_on_the_review_file(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    qa = qa_pass()
    _review(
        hub,
        sources,
        work,
        source="render_1080p.mp4",
        cap_mb=5.0,
        transcoder=FakeTranscoder([10]),
        qa=qa,
    )
    ddir = Path(_delivery_dir(str(work), hub.project_id, "short-1", "rj-1"))
    assert qa.calls[-1][0] == str(ddir / "review.mp4")
    assert qa.calls[-1][0] != str(ddir / "source.mp4")


def test_v25_review_qa_failure_reports_truthful_completed_steps(
    tmp_path: Path, sources: Path
) -> None:
    hub = make_hub(tmp_path)
    with pytest.raises(DeliverError) as info:
        _review(
            hub,
            sources,
            tmp_path / "w",
            source="render_1080p.mp4",
            cap_mb=5.0,
            transcoder=FakeTranscoder([10]),
            qa=qa_fail(),
        )
    assert info.value.code == "review_qa_failed"
    assert info.value.detail["completed_steps"] == ["pull", "transcode"]


def test_v34_first_master_delivery_does_not_report_facts_resumed(
    tmp_path: Path, sources: Path
) -> None:
    hub = make_hub(tmp_path)
    result = _master(hub, sources, tmp_path / "w")
    assert "facts" not in result.resumed_steps


def test_v18_v19_passlogfile_is_per_delivery_and_absolute(tmp_path: Path, sources: Path) -> None:
    """The two-pass log path is per-delivery (inside the delivery dir) and absolute: never the
    shared workdir root, and never a bare name in the process cwd."""
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    tc = FakeTranscoder([int(0.3 * MB)])
    _review(hub, sources, work, source="render_1080p.mp4", cap_mb=0.5, transcoder=tc)
    passlog = tc.calls[0]["passlogfile"]
    assert os.path.isabs(passlog)  # V19: "ffpass" in the cwd would be relative
    ddir = _delivery_dir(str(work), hub.project_id, "short-1", "rj-1")
    assert os.path.dirname(passlog) == ddir  # V18: the workdir root is shared across deliveries
    assert os.path.basename(passlog) == "ffpass"
