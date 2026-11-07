"""One review delivered end to end with the real ``FfmpegTranscoder`` and the real ``aar.qa.check``.

A real 1080p source is transcoded to the 720p review box with a two-pass H.264 + AAC encode, QA'd
with a permissive profile (the synthetic sine is quiet, so loudness has a wide tolerance), stored,
and a review is requested.
"""

from __future__ import annotations

from pathlib import Path

from deliver_support import AGENT, FakeEditHost, make_hub, make_profile, make_source, render_output

from aar.deliver.service import deliver_review
from aar.deliver.transcode import FfmpegTranscoder
from aar.media import probe


def test_review_end_to_end_with_real_ffmpeg(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    # A longform source at ~-22.7 LUFS (volume 0.9), so the REAL QA holds the review to the
    # longform loudness target (-22 +/- a realistic 2 LU tolerance), not a hidden 60 LU one (F2).
    edit_pc = tmp_path / "pc"
    edit_pc.mkdir()
    make_source(edit_pc / "render_1080p.mp4", width=1920, height=1080, duration=2.0, volume=0.9)
    transcoder = FfmpegTranscoder()
    result = deliver_review(
        project_id=hub.project_id,
        subject_id=hub.subject_id,
        deliverable="short-1",
        actor=AGENT,
        profile=make_profile(review_mp4_max_mb=5.0, lufs_tolerance=2.0),
        title="Acme review",
        timeline="Main",
        fps=25.0,
        kind="longform",
        render_output=render_output(
            edit_pc / "render_1080p.mp4", duration=2.0, render_job_id="rj-e"
        ),
        workdir=str(tmp_path / "work"),
        edit_host=FakeEditHost(edit_pc),
        log=hub.log,
        store=hub.store,
        transcoder=transcoder,
        # qa_check defaults to the real aar.qa.check
    )
    assert result.transcoded is True
    assert result.attempts == 1
    assert (result.width, result.height) == (1280, 720)
    assert result.video_kbps == 4000  # capped
    assert result.qa_status in ("pass", "warn")
    assert result.review_id is not None

    # Two real ffmpeg passes ran (analysis + encode).
    assert len(transcoder.argv_log) == 2
    assert "+faststart" in transcoder.argv_log[1]

    # The stored review is a real, decodable, under-cap 720p mp4.
    assert hub.store.has(result.artifact_id)
    review_file = tmp_path / "work" / hub.project_id / "short-1" / "rj-e" / "review.mp4"
    assert result.size <= 5_000_000
    info = probe(str(review_file))
    assert info.video is not None
    assert (info.video.width, info.video.height) == (1280, 720)
