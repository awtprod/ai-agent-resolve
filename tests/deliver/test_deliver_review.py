"""``deliver_review``: fit under the cap, QA, store, record facts, request review.

The source is a real synthetic render (for the probe of its dimensions); the transcoder and QA are
fakes so the cap, retry and QA paths are deterministic. The hub is real.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deliver_support import (
    AGENT,
    FakeEditHost,
    FakeTranscoder,
    Hub,
    make_hub,
    make_profile,
    qa_fail,
    qa_pass,
    render_output,
)

from aar.deliver.errors import DeliverError
from aar.deliver.service import (
    _box_for,
    _delivery_dir,
    _fit,
    _suffix,
    _video_kbps,
    deliver_review,
)
from aar.hub.facts import facts_view
from aar.hub.reviews import get_review

MB = 1_000_000


def _review(
    hub: Hub,
    sources: Path,
    workdir: Path,
    *,
    source: str,
    cap_mb: float,
    transcoder: FakeTranscoder,
    qa=None,
    kind: str = "short",
    duration: float = 6.0,
    deliverable: str = "short-1",
    render_job_id: str = "rj-1",
):
    return deliver_review(
        project_id=hub.project_id,
        subject_id=hub.subject_id,
        deliverable=deliverable,
        actor=AGENT,
        profile=make_profile(review_mp4_max_mb=cap_mb),
        title="Acme review",
        timeline="Main",
        fps=25.0,
        kind=kind,
        render_output=render_output(
            sources / source, duration=duration, render_job_id=render_job_id
        ),
        workdir=str(workdir),
        edit_host=FakeEditHost(sources),
        log=hub.log,
        store=hub.store,
        transcoder=transcoder,
        qa_check=qa or qa_pass(),
    )


# -- geometry (unit) ---------------------------------------------------------------------------


def test_box_is_chosen_by_aspect() -> None:
    assert _box_for(1920, 1080) == (1280, 720)
    assert _box_for(1080, 1920) == (720, 1280)
    assert _box_for(1000, 1000) == (1280, 720)  # square counts as landscape


def test_fit_never_upscales() -> None:
    assert _fit(640, 360, 1280, 720) == (640, 360)  # smaller than the box: left as is
    assert _fit(1000, 1000, 1280, 720) == (720, 720)  # bounded by the short side


def test_fit_downscales_and_keeps_even_dimensions() -> None:
    assert _fit(1920, 1080, 1280, 720) == (1280, 720)
    assert _fit(1080, 1920, 720, 1280) == (720, 1280)


def test_video_kbps_formula_and_cap() -> None:
    # (cap*0.92*8/duration - 128000)/1000, floored, capped at 4000.
    assert _video_kbps(500_000, 6.0) == 485
    assert _video_kbps(50 * MB, 6.0) == 4000  # capped


# -- happy paths -------------------------------------------------------------------------------


def test_use_as_is_when_it_fits_the_box_and_the_cap(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    transcoder = FakeTranscoder([10])
    result = _review(
        hub, sources, tmp_path / "work", source="render_720p.mp4", cap_mb=5.0, transcoder=transcoder
    )
    assert transcoder.calls == []  # it already fits: never transcoded
    assert result.transcoded is False
    assert result.video_kbps is None
    assert (result.width, result.height) == (1280, 720)
    assert result.qa_status == "pass"
    assert result.review_id is not None
    assert hub.store.has(result.artifact_id)


def test_box_follows_source_aspect_not_kind(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    transcoder = FakeTranscoder([200_000])
    # A landscape source delivered as a "short": the box must stay landscape (by aspect).
    _review(
        hub,
        sources,
        tmp_path / "work",
        source="render_1080p.mp4",
        cap_mb=5.0,
        transcoder=transcoder,
        kind="short",
    )
    assert transcoder.calls[0]["width"] == 1280
    assert transcoder.calls[0]["height"] == 720


def test_portrait_source_gets_the_portrait_box(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    transcoder = FakeTranscoder([200_000])
    _review(
        hub,
        sources,
        tmp_path / "work",
        source="render_portrait.mp4",
        cap_mb=5.0,
        transcoder=transcoder,
        kind="longform",
    )
    assert transcoder.calls[0]["width"] == 720
    assert transcoder.calls[0]["height"] == 1280


def test_facts_and_review_are_written(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    result = _review(
        hub,
        sources,
        tmp_path / "work",
        source="render_720p.mp4",
        cap_mb=5.0,
        transcoder=FakeTranscoder([10]),
    )
    facts = facts_view(hub.log, hub.project_id)
    hash_fact = facts.get("render.hash.short-1")
    assert hash_fact is not None
    assert hash_fact.current.value == {"sha256": result.artifact_id, "bytes": result.size}
    assert hash_fact.current.source == {"recipe_node": None, "artifact": result.artifact_id}
    size_fact = facts.get("qa.size.short-1")
    assert size_fact is not None
    assert size_fact.current.value["unit"] == "MB"
    assert size_fact.current.value["value"] == pytest.approx(result.size / MB, abs=1e-6)
    assert set(result.fact_keys) == {"render.hash.short-1", "qa.size.short-1"}
    review = get_review(hub.log, hub.project_id, result.review_id)
    assert review is not None
    assert review.renders == (result.artifact_id,)
    assert review.subject == hub.subject_id
    assert review.title == "Acme review"


# -- the cap: retry, over-cap, cannot-fit ------------------------------------------------------


def test_transcode_retries_once_at_a_lower_bitrate(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    cap_bytes = int(0.5 * MB)
    transcoder = FakeTranscoder([cap_bytes + 1000, cap_bytes - 1000])
    result = _review(
        hub,
        sources,
        tmp_path / "work",
        source="render_1080p.mp4",
        cap_mb=0.5,
        transcoder=transcoder,
        duration=6.0,
    )
    assert result.attempts == 2
    assert len(transcoder.calls) == 2
    first = transcoder.calls[0]["video_kbps"]
    assert first == 485
    assert transcoder.calls[1]["video_kbps"] == 412  # floor(0.85 * 485)
    assert result.transcoded is True
    assert result.video_kbps == 412


def test_over_cap_after_retry_is_refused(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    cap_bytes = int(0.5 * MB)
    transcoder = FakeTranscoder([cap_bytes + 5000, cap_bytes + 5000])
    with pytest.raises(DeliverError) as exc:
        _review(
            hub,
            sources,
            tmp_path / "work",
            source="render_1080p.mp4",
            cap_mb=0.5,
            transcoder=transcoder,
        )
    assert exc.value.code == "review_over_cap"
    assert exc.value.detail["size"] == cap_bytes + 5000
    assert exc.value.detail["completed_steps"] == ["pull"]
    # Nothing was stored or sent, and no over-cap review file was left behind.
    assert list(hub.log.iter_events(types=["artifact.added"])) == []
    assert not (tmp_path / "work" / hub.project_id / "short-1" / "rj-1" / "review.mp4").exists()
    assert not (
        tmp_path / "work" / hub.project_id / "short-1" / "rj-1" / "review.mp4.part"
    ).exists()


def test_cannot_fit_below_the_floor(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    transcoder = FakeTranscoder([1000])
    with pytest.raises(DeliverError) as exc:
        _review(
            hub,
            sources,
            tmp_path / "work",
            source="render_1080p.mp4",
            cap_mb=0.5,
            transcoder=transcoder,
            duration=30.0,
        )
    assert exc.value.code == "review_cannot_fit"
    assert exc.value.detail["duration_seconds"] == 30.0
    assert exc.value.detail["cap_mb"] == 0.5
    assert transcoder.calls == []  # refused before any ffmpeg ran


# -- safety: never overwrite a mismatch, remove a leftover .part -------------------------------


def test_local_mismatch_is_never_overwritten(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    ddir = _delivery_dir(str(work), hub.project_id, "short-1", "rj-1")
    suffix = _suffix(str(sources / "render_720p.mp4"))
    stale = Path(ddir) / f"source{suffix}"
    stale.write_bytes(b"a different render entirely")
    with pytest.raises(DeliverError) as exc:
        _review(
            hub,
            sources,
            work,
            source="render_720p.mp4",
            cap_mb=5.0,
            transcoder=FakeTranscoder([10]),
        )
    assert exc.value.code == "deliver_local_mismatch"
    assert stale.read_bytes() == b"a different render entirely"  # left untouched


def test_leftover_part_is_removed_before_transcoding(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "work"
    ddir = _delivery_dir(str(work), hub.project_id, "short-1", "rj-1")
    leftover = Path(ddir) / "review.mp4.part"
    leftover.write_bytes(b"crashed half-transcode")
    result = _review(
        hub,
        sources,
        work,
        source="render_1080p.mp4",
        cap_mb=0.5,
        transcoder=FakeTranscoder([int(0.4 * MB)]),
    )
    assert result.transcoded is True
    assert not leftover.exists()  # the stale .part was cleared
    assert (Path(ddir) / "review.mp4").exists()


# -- QA fail is never stored or sent -----------------------------------------------------------


def test_qa_fail_stores_nothing_and_sends_nothing(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    with pytest.raises(DeliverError) as exc:
        _review(
            hub,
            sources,
            tmp_path / "work",
            source="render_720p.mp4",
            cap_mb=5.0,
            transcoder=FakeTranscoder([10]),
            qa=qa_fail(),
        )
    assert exc.value.code == "review_qa_failed"
    assert exc.value.detail["failed_checks"] == ["decode"]
    assert exc.value.detail["completed_steps"] == ["pull", "fit"]
    assert list(hub.log.iter_events(types=["artifact.added"])) == []
    assert list(hub.log.iter_events(types=["facts.recorded"])) == []
    review_nodes = [
        e
        for e in hub.log.iter_events(types=["graph.node_added"], project_id=hub.project_id)
        if e.payload.get("kind") == "review"
    ]
    assert review_nodes == []
