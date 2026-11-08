"""Round-3 validator-mutant pins (val_mutate_r2): lock, reuse, as-is gate, F8, keys, errors."""

from __future__ import annotations

import errno
import fcntl
import os
import sqlite3
from pathlib import Path

import pytest
from deliver_support import (
    AGENT,
    FakeEditHost,
    FakeTranscoder,
    make_hub,
    make_profile,
    make_source,
    qa_pass,
    render_output,
)

from aar.deliver import service
from aar.deliver.errors import DeliverError
from aar.deliver.model import RenderOutput
from aar.deliver.service import deliver_review
from aar.hub.writes import WriteError
from aar.media import run_tool

# Captured at import, before any fixture clears the roots (V51 catches a mutated constant).
PROD_ROOTS = tuple(service.FORBIDDEN_WORKDIR_ROOTS)
MB = 1_000_000


def _review(hub, sources, workdir, *, source, cap_mb, transcoder, qa=None, **kw):
    return deliver_review(
        project_id=hub.project_id,
        subject_id=kw.get("subject_id", hub.subject_id),
        deliverable="short-1",
        actor=AGENT,
        profile=make_profile(review_mp4_max_mb=cap_mb),
        title="Acme review",
        timeline="Main",
        fps=25.0,
        kind="short",
        render_output=render_output(sources / source, duration=kw.get("duration", 6.0)),
        workdir=str(workdir),
        edit_host=FakeEditHost(sources),
        log=hub.log,
        store=kw.get("store", hub.store),
        transcoder=transcoder,
        qa_check=qa or qa_pass(),
    )


def _ddir(hub, workdir: Path) -> Path:
    d = Path(service._delivery_dir(str(workdir), hub.project_id, "short-1", "rj-1"))
    return d


def _variant(path: Path, *, vcodec: str, acodec: str) -> Path:
    run_tool(
        "ffmpeg",
        [
            "-nostdin", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=1.0",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1.0",
            "-c:v", vcodec, "-pix_fmt", "yuv420p",
            "-c:a", acodec, "-shortest", str(path),
        ],
        timeout_s=120.0,
    )  # fmt: skip
    return path


@pytest.fixture
def variants(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("variants")
    _variant(root / "pcm.mov", vcodec="libx264", acodec="pcm_s16le")
    _variant(root / "aac.mkv", vcodec="libx264", acodec="aac")
    _variant(root / "mpeg4.mp4", vcodec="mpeg4", acodec="aac")
    _variant(root / "aac.mov", vcodec="libx264", acodec="aac")
    return root


def test_v20_stale_part_removed_before_cannot_fit(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ddir = _ddir(hub, work)
    part = ddir / "review.mp4.part"
    part.write_bytes(b"junk" * 100)
    with pytest.raises(DeliverError) as info:
        _review(
            hub, sources, work, source="render_1080p.mp4", cap_mb=0.05,
            transcoder=FakeTranscoder([10]),
        )  # fmt: skip
    assert info.value.code == "review_cannot_fit"
    assert not part.exists()


def test_v53_v54_held_lock_blocks_delivery(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    fd = service._acquire_delivery_lock(str(_ddir(hub, work)))
    try:
        with pytest.raises(DeliverError) as info:
            _review(
                hub, sources, work, source="render_720p.mp4", cap_mb=5.0,
                transcoder=FakeTranscoder([10]),
            )  # fmt: skip
        assert info.value.code == "delivery_in_progress"
    finally:
        service._release_delivery_lock(fd)


def test_v55_junk_reused_review_is_rebuilt(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ddir = _ddir(hub, work)
    (ddir / "review.mp4").write_bytes(os.urandom(300_000))
    tc = FakeTranscoder([int(0.3 * MB)])
    result = _review(hub, sources, work, source="render_1080p.mp4", cap_mb=1.0, transcoder=tc)
    assert result.transcoded is True
    assert "transcode" not in result.resumed_steps
    assert len(tc.calls) == 1


def test_v74_unprobeable_review_is_rebuilt(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ddir = _ddir(hub, work)
    (ddir / "review.mp4").write_bytes(b"0123456789")
    tc = FakeTranscoder([int(0.3 * MB)])
    result = _review(hub, sources, work, source="render_1080p.mp4", cap_mb=1.0, transcoder=tc)
    assert result.transcoded is True
    assert "transcode" not in result.resumed_steps
    assert len(tc.calls) == 1


@pytest.mark.parametrize("name", ["pcm.mov", "aac.mkv", "mpeg4.mp4"])
def test_v56_v57_v58_not_review_ready_is_transcoded(
    tmp_path: Path, variants: Path, name: str
) -> None:
    hub = make_hub(tmp_path)
    tc = FakeTranscoder([20_000])
    result = _review(hub, variants, tmp_path / "w", source=name, cap_mb=5.0, transcoder=tc)
    assert result.transcoded is True
    assert len(tc.calls) == 1


def test_v59_mov_as_is_keeps_quicktime_mime(tmp_path: Path, variants: Path) -> None:
    hub = make_hub(tmp_path)
    tc = FakeTranscoder([10])
    result = _review(hub, variants, tmp_path / "w", source="aac.mov", cap_mb=5.0, transcoder=tc)
    assert result.transcoded is False
    assert tc.calls == []
    assert hub.store.get_meta(result.artifact_id).mime == "video/quicktime"


def test_v63_role_is_in_the_idempotency_key(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    result = _review(
        hub, sources, tmp_path / "w", source="render_720p.mp4", cap_mb=5.0,
        transcoder=FakeTranscoder([10]),
    )  # fmt: skip
    rk = service._review_keys(hub.project_id, "short-1", "rj-1", result.sha256)
    mk = service._master_keys(hub.project_id, "short-1", "rj-1", result.sha256)
    assert rk["hash"].startswith("deliver:review:")
    assert mk["hash"].startswith("deliver:master:")
    assert hub.log.get_by_idempotency_key(rk["review"] + ":node") is not None


def test_v51_default_roots_forbid_tmp(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "FORBIDDEN_WORKDIR_ROOTS", PROD_ROOTS)
    hub = make_hub(tmp_path)
    with pytest.raises(DeliverError) as info:
        _review(
            hub, sources, os.sep + "tmp" + os.sep + "aar-x", source="render_720p.mp4",
            cap_mb=5.0, transcoder=FakeTranscoder([10]),
        )  # fmt: skip
    assert info.value.code == "invalid_workdir"


def test_v52_symlinked_workdir_is_resolved(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    link = other / "ln"
    link.symlink_to(root)
    monkeypatch.setattr(service, "FORBIDDEN_WORKDIR_ROOTS", (os.path.realpath(root),))
    hub = make_hub(tmp_path)
    with pytest.raises(DeliverError) as info:
        _review(
            hub, sources, link / "work", source="render_720p.mp4", cap_mb=5.0,
            transcoder=FakeTranscoder([10]),
        )  # fmt: skip
    assert info.value.code == "invalid_workdir"


@pytest.mark.parametrize(("duration", "expected"), [(6.0, 120.0), (40.0, 160.0)])
def test_v64_transcode_timeout_is_max_120_or_4x(
    tmp_path: Path, sources: Path, duration: float, expected: float
) -> None:
    hub = make_hub(tmp_path)
    tc = FakeTranscoder([int(0.3 * MB)])
    _review(
        hub, sources, tmp_path / "w", source="render_1080p.mp4", cap_mb=5.0,
        transcoder=tc, duration=duration,
    )  # fmt: skip
    assert tc.calls[0]["timeout_s"] == expected


def test_v69_negative_frame_count_refused() -> None:
    with pytest.raises(DeliverError) as info:
        RenderOutput.from_measured(
            {
                "output_path": "x.mp4",
                "frame_count": -5,
                "frame_rate": 25.0,
                "duration_seconds": 6.0,
                "render_job_id": "rj-1",
            }
        )
    assert info.value.code == "invalid_render_output"


class _FailingStore:
    def __init__(self, inner, exc: Exception) -> None:
        self._inner, self._exc = inner, exc

    def put(self, *a, **k):
        raise self._exc

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_v72_wrapped_error_keeps_completed_steps(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    store = _FailingStore(hub.store, WriteError("invalid_input", "no", "fix"))
    with pytest.raises(DeliverError) as info:
        _review(
            hub, sources, tmp_path / "w", source="render_1080p.mp4", cap_mb=5.0,
            transcoder=FakeTranscoder([10]), store=store,
        )  # fmt: skip
    assert info.value.code == "hub_write_failed"
    assert info.value.detail["completed_steps"] == ["pull", "transcode", "qa"]


def test_v73_unknown_subject_code(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    with pytest.raises(DeliverError) as info:
        _review(
            hub, sources, tmp_path / "w", source="render_720p.mp4", cap_mb=5.0,
            transcoder=FakeTranscoder([10]), subject_id="no-such-node",
        )  # fmt: skip
    assert info.value.code == "unknown_subject"


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (sqlite3.OperationalError("database is locked"), "hub_unavailable"),
        (PermissionError(errno.EACCES, "denied"), "store_unavailable"),
    ],
)
def test_r2_3_hub_and_store_unavailable(
    tmp_path: Path, sources: Path, exc: Exception, code: str
) -> None:
    hub = make_hub(tmp_path)
    with pytest.raises(DeliverError) as info:
        _review(
            hub, sources, tmp_path / "w", source="render_1080p.mp4", cap_mb=5.0,
            transcoder=FakeTranscoder([10]), store=_FailingStore(hub.store, exc),
        )  # fmt: skip
    assert info.value.code == code
    assert info.value.detail["completed_steps"] == ["pull", "transcode", "qa"]


def test_r2_5_flock_errno_distinguishes_unsupported_from_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raiser(code):
        def _flock(fd, op):
            raise OSError(code, os.strerror(code))

        return _flock

    monkeypatch.setattr(fcntl, "flock", raiser(errno.ENOLCK))
    with pytest.raises(DeliverError) as info:
        service._acquire_delivery_lock(str(tmp_path))
    assert info.value.code == "lock_unsupported"
    assert "ENOLCK" in info.value.message or "ENOLCK" in str(info.value.detail)

    monkeypatch.setattr(fcntl, "flock", raiser(errno.EWOULDBLOCK))
    with pytest.raises(DeliverError) as info:
        service._acquire_delivery_lock(str(tmp_path))
    assert info.value.code == "delivery_in_progress"


def test_v60_review_qa_uses_the_content_kind_targets(tmp_path: Path, sources: Path) -> None:
    """Real QA: a short at ~-15 LUFS passes the SHORT target; at ~-22.7 it fails. V60 puts QA back
    on the long-form 'review' target, which flips both outcomes."""
    from aar.deliver.transcode import FfmpegTranscoder

    hub = make_hub(tmp_path)
    pc = tmp_path / "pc"
    pc.mkdir()
    make_source(pc / "ok.mp4", width=1080, height=1920, duration=6.0, volume=2.14)
    make_source(pc / "quiet.mp4", width=1080, height=1920, duration=6.0, volume=0.9)
    prof = make_profile(review_mp4_max_mb=5.0, lufs_tolerance=1.5)

    def run(name: str) -> object:
        return deliver_review(
            project_id=hub.project_id,
            subject_id=hub.subject_id,
            deliverable="short-1",
            actor=AGENT,
            profile=prof,
            title="t",
            timeline="Main",
            fps=25.0,
            kind="short",
            render_output=render_output(pc / name, duration=6.0),
            workdir=str(tmp_path / f"w_{name}"),
            edit_host=FakeEditHost(pc),
            log=hub.log,
            store=hub.store,
            transcoder=FfmpegTranscoder(),
        )

    assert run("ok.mp4").qa_status in ("pass", "warn")
    with pytest.raises(DeliverError) as exc:
        run("quiet.mp4")
    assert exc.value.code == "review_qa_failed"


def test_r2_1_rebuild_with_new_bytes_supersedes_not_conflicts(
    tmp_path: Path, sources: Path
) -> None:
    """Re-delivering the same render with DIFFERENT review bytes is a new delivery that supersedes
    the render.hash fact through the facts store, never a hub idempotency_conflict (R2-1)."""
    from aar.hub.facts import facts_view

    hub = make_hub(tmp_path)
    work = tmp_path / "w"

    def run(tc: object, cap: float) -> object:
        return deliver_review(
            project_id=hub.project_id,
            subject_id=hub.subject_id,
            deliverable="short-1",
            actor=AGENT,
            profile=make_profile(review_mp4_max_mb=cap),
            title="t",
            timeline="Main",
            fps=25.0,
            kind="short",
            render_output=render_output(sources / "render_1080p.mp4", duration=6.0),
            workdir=str(work),
            edit_host=FakeEditHost(sources),
            log=hub.log,
            store=hub.store,
            transcoder=tc,
            qa_check=qa_pass(),
        )

    r1 = run(FakeTranscoder([400_000]), 0.5)
    ddir = Path(service._delivery_dir(str(work), hub.project_id, "short-1", "rj-1"))
    (ddir / "review.mp4").unlink()  # force a rebuild with different bytes
    r2 = run(FakeTranscoder([200_000]), 0.5)  # with the old content-free key this conflicted
    assert r2.artifact_id != r1.artifact_id
    fact = facts_view(hub.log, hub.project_id).get("render.hash.review.short-1")
    assert fact.current.value["sha256"] == r2.artifact_id  # superseded


def test_v34_master_facts_resumed_only_when_all_replay(tmp_path: Path, sources: Path) -> None:
    """Master 'facts' is reported resumed only when EVERY fact write replayed. Here the hash fact is
    pre-written (replays) but lufs/truepeak are new, so facts must NOT be reported resumed."""
    import shutil

    from deliver_support import QaStub, qa_report

    from aar.deliver.service import _master_keys, deliver_master
    from aar.hub.facts import record_fact
    from aar.qa.report import QACheck

    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ddir = Path(service._delivery_dir(str(work), hub.project_id, "m-1", "rj-9"))
    shutil.copyfile(sources / "render_720p.mp4", ddir / "master.mp4")
    art = hub.store.put(
        ddir / "master.mp4",
        actor=AGENT,
        meta={},
        mime="video/mp4",
        project_id=hub.project_id,
        producing_op="deliver.master",
    )
    keys = _master_keys(hub.project_id, "m-1", "rj-9", art)
    record_fact(
        hub.log,
        hub.project_id,
        "render.hash.master.m-1",
        {"sha256": art, "bytes": (ddir / "master.mp4").stat().st_size},
        AGENT,
        idempotency_key=keys["hash"],
        source={"artifact": art},
    )
    qa = QaStub(
        qa_report(
            [
                QACheck(
                    id="loudness",
                    status="pass",
                    message="x",
                    measured={"integrated_lufs": -22.3, "true_peak_dbtp": -1.5},
                )
            ],
            kind="longform",
        )
    )
    r = deliver_master(
        project_id=hub.project_id,
        deliverable="m-1",
        actor=AGENT,
        profile=make_profile(),
        kind="longform",
        render_output=render_output(sources / "render_720p.mp4", render_job_id="rj-9"),
        workdir=str(work),
        edit_host=FakeEditHost(sources),
        log=hub.log,
        store=hub.store,
        qa_check=qa,
    )
    assert "facts" not in r.resumed_steps


def test_key_base_depends_on_project_role_and_content() -> None:
    """The idempotency key base separates project, role and content (V62/V63/R2-1): any of them
    changing yields a different key, so deliveries never collide or wrongly replay."""
    kb = service._key_base
    base = kb("review", "p1", "d1", "r1", "sha-a")
    assert base.startswith("deliver:review:")
    assert kb("review", "p2", "d1", "r1", "sha-a") != base  # project matters (V62)
    assert kb("master", "p1", "d1", "r1", "sha-a") != base  # role matters (V63)
    assert kb("review", "p1", "d1", "r1", "sha-b") != base  # content matters (R2-1)
    assert kb("review", "p1", "d1", "r1", "sha-a") == base  # deterministic


def test_v61_composed_review_qa_includes_the_size_cap_check() -> None:
    """The composed review QA report carries a size-cap check (V61 drops it). The cap is also
    enforced structurally before QA, but the report must still surface it."""
    report = service._review_qa_report(
        qa_pass(), "review.mp4", "short", make_profile(review_mp4_max_mb=5.0), size_bytes=100
    )
    assert any(c.id == "size-cap" for c in report.checks)


def test_v75_non_yuv420p_h264_is_transcoded(tmp_path: Path, sources: Path) -> None:
    """An H.264 but yuv422p .mp4 that fits the box and cap must be transcoded, not stored as is
    (V75 drops the pixel-format gate and would keep an unplayable High 4:2:2 review)."""
    hub = make_hub(tmp_path)
    pc = tmp_path / "pc"
    pc.mkdir()
    run_tool(
        "ffmpeg",
        [
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv422p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(pc / "s422.mp4"),
        ],
        timeout_s=120.0,
    )
    r = deliver_review(
        project_id=hub.project_id,
        subject_id=hub.subject_id,
        deliverable="short-1",
        actor=AGENT,
        profile=make_profile(review_mp4_max_mb=20.0),
        title="t",
        timeline="Main",
        fps=25.0,
        kind="short",
        render_output=render_output(pc / "s422.mp4", duration=2.0),
        workdir=str(tmp_path / "w"),
        edit_host=FakeEditHost(pc),
        log=hub.log,
        store=hub.store,
        transcoder=FakeTranscoder([100_000]),
        qa_check=qa_pass(),
    )
    assert r.transcoded is True


def test_r3_2_redelivery_withdraws_the_prior_open_review(tmp_path: Path, sources: Path) -> None:
    """Each re-delivery with new bytes opens a new review and withdraws the prior open one for the
    same (project, deliverable, review role), so exactly one stays open (R3-2)."""
    from aar.hub.reviews import list_reviews

    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ids = []
    for size in (400_000, 300_000, 200_000):
        ddir = Path(service._delivery_dir(str(work), hub.project_id, "short-1", "rj-1"))
        rev = ddir / "review.mp4"
        if rev.exists():
            rev.unlink()
        r = deliver_review(
            project_id=hub.project_id,
            subject_id=hub.subject_id,
            deliverable="short-1",
            actor=AGENT,
            profile=make_profile(review_mp4_max_mb=0.5),
            title="t",
            timeline="Main",
            fps=25.0,
            kind="short",
            render_output=render_output(sources / "render_1080p.mp4", duration=6.0),
            workdir=str(work),
            edit_host=FakeEditHost(sources),
            log=hub.log,
            store=hub.store,
            transcoder=FakeTranscoder([size]),
            qa_check=qa_pass(),
        )
        ids.append(r.review_id)
    reviews = list_reviews(hub.log, hub.project_id)
    open_reviews = [rv for rv in reviews if rv.pending]
    assert [rv.id for rv in open_reviews] == [ids[-1]]  # exactly one, the newest
    cancelled = [rv for rv in reviews if rv.status == "cancelled"]
    assert len(cancelled) == 2
    for rv in cancelled:
        assert rv.withdrawal is not None
        assert "superseded by delivery" in rv.withdrawal.reason


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 0")
def test_r3_3_unreadable_source_is_local_io_error(tmp_path: Path, sources: Path) -> None:
    """An unreadable pre-pulled local source is a local_io_error, not store_unavailable (R3-3)."""
    import shutil

    hub = make_hub(tmp_path)
    work = tmp_path / "w"
    ddir = Path(service._delivery_dir(str(work), hub.project_id, "short-1", "rj-1"))
    shutil.copyfile(sources / "render_720p.mp4", ddir / "source.mp4")
    os.chmod(ddir / "source.mp4", 0)
    try:
        with pytest.raises(DeliverError) as exc:
            deliver_review(
                project_id=hub.project_id,
                subject_id=hub.subject_id,
                deliverable="short-1",
                actor=AGENT,
                profile=make_profile(review_mp4_max_mb=5.0),
                title="t",
                timeline="Main",
                fps=25.0,
                kind="short",
                render_output=render_output(sources / "render_720p.mp4", duration=2.0),
                workdir=str(work),
                edit_host=FakeEditHost(sources),
                log=hub.log,
                store=hub.store,
                transcoder=FakeTranscoder([10]),
                qa_check=qa_pass(),
            )
        assert exc.value.code == "local_io_error"
        assert exc.value.detail.get("completed_steps") == []
    finally:
        os.chmod(ddir / "source.mp4", 0o600)
