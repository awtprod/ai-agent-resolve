"""Carry a finished render through to a stored, hashed, QA'd and (for review) sent artifact.

Everything here is hub-side and takes its collaborators by injection, so tests run with fakes and
real ffmpeg: an :class:`EditHostLike` for the verified transport, a :class:`Transcoder`, a QA
callable, and the hub's :class:`EventLog`, :class:`ArtifactStore` and :class:`Actor`.

Crash safety and idempotency. Each delivery works in ``<workdir>/<project>/<deliverable>/<job>/``
and every file it writes stays inside it. A step writes only after the previous one is durable: a
transcode goes to a ``.part`` and is renamed only after ffmpeg exits 0, the output decodes, and
its size is under the cap. The pull is skipped when the local file already matches the remote
sha256 (a different sha256 is ``deliver_local_mismatch``, never overwritten); the transcode is
skipped when a review file already meets the cap; the store dedupes by content; the facts and the
review carry deterministic idempotency keys. A full re-run after success therefore makes no new
events, puts nothing new, and does not transcode again.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from aar.deliver.errors import DeliverError
from aar.deliver.model import DeliveryResult, RenderOutput
from aar.deliver.transcode import AUDIO_KBPS, FfmpegTranscoder, Transcoder
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.facts import record_fact
from aar.hub.reviews import Review, request_review
from aar.media import probe, timeout_for
from aar.media.probe import PROBE_TIMEOUT_S
from aar.plan.profile import Profile
from aar.qa.check import check as _default_qa_check
from aar.qa.report import QAReport

MB = 1_000_000  # the review cap counts decimal megabytes (same as aar.qa)
CAP_HEADROOM = 0.92  # leave 8% of the budget for container and rate overshoot
MIN_VIDEO_KBPS = 250  # below this the review cannot be made to fit
MAX_VIDEO_KBPS = 4000  # a review never needs more than this
RETRY_FRACTION = 0.85  # the second attempt's bit rate as a share of the first
LANDSCAPE_BOX = (1280, 720)
PORTRAIT_BOX = (720, 1280)
_CHUNK = 1024 * 1024

QACheckFn = Callable[..., QAReport]

# Master QA facts: metric segment -> (check id, measured key, unit). Only values that were
# measured are written; keys are never guessed.
MASTER_QA_FACTS: dict[str, tuple[str, str, str]] = {
    "lufs": ("loudness", "integrated_lufs", "LUFS"),
    "truepeak": ("loudness", "true_peak_dbtp", "dBTP"),
    "duration": ("duration", "render_s", "s"),
}

_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FACT_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")  # also the facts key <...> shape
_MIME = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}


@runtime_checkable
class EditHostLike(Protocol):
    """The slice of :class:`aar.host.EditHost` the delivery library uses (verified transport)."""

    def pull(self, remote: str, local: str, *, overwrite: bool = False) -> Any: ...

    def file_hash(self, remote: str) -> Any: ...


# -- validation ------------------------------------------------------------------------------


def _segment(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise DeliverError(
            "invalid_name",
            f"{field} {value!r} is not a safe path segment",
            "Use a short id of letters, digits, '-' and '_' (no path separators or dots).",
            field=field,
        )
    return value


def _check_workdir(workdir: Any) -> str:
    if not isinstance(workdir, str) or not workdir:
        raise DeliverError(
            "invalid_workdir",
            "a work directory is required",
            "Pass an absolute --workdir on a durable disk (never a temp directory).",
        )
    if not os.path.isabs(workdir):
        raise DeliverError(
            "invalid_workdir",
            f"the work directory must be absolute: {workdir!r}",
            "Pass an absolute path.",
        )
    # The directory must be durable: delivery artifacts outlive the run, so the caller must not
    # pass a scratch location. The library never creates a scratch directory of its own -- an AST
    # pin proves it imports no scratch-dir module and holds no scratch-root literal.
    return workdir


def _suffix(path: str) -> str:
    base = re.split(r"[\\/]", path)[-1]
    _, ext = os.path.splitext(base)
    return ext.lower() if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", ext) else ""


def _delivery_dir(workdir: str, project_id: str, deliverable: str, render_job_id: str) -> str:
    path = os.path.join(workdir, project_id, deliverable, render_job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _hash_file(path: str) -> str:
    """The lowercase-hex sha256 of a file, streamed (never read whole into memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unlink_quiet(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


# -- pull (shared by review and master) ------------------------------------------------------


def _pull(edit_host: EditHostLike, remote: str, local: str, resumed: list[str]) -> tuple[str, int]:
    """Pull ``remote`` to ``local`` (verified), or skip it when the local file already matches.

    Returns the file's (lowercase sha256, size). A local file with a different sha256 is a
    ``deliver_local_mismatch``: it is never overwritten.
    """
    if os.path.exists(local):
        remote_sha = edit_host.file_hash(remote).sha256.lower()
        local_sha = _hash_file(local)
        if local_sha == remote_sha:
            resumed.append("pull")
            return local_sha, os.path.getsize(local)
        raise DeliverError(
            "deliver_local_mismatch",
            f"{local} already exists with sha256 {local_sha} but the remote reports {remote_sha}",
            "Remove the stale local file, or point --workdir at a clean directory, then re-run.",
            local=local,
            local_sha256=local_sha,
            remote_sha256=remote_sha,
        )
    result = edit_host.pull(remote, local, overwrite=False)
    return result.sha256.lower(), result.size


# -- fit and transcode (review) --------------------------------------------------------------


def _box_for(src_w: int, src_h: int) -> tuple[int, int]:
    """The review box, by the SOURCE aspect (never by kind): landscape or portrait."""
    return LANDSCAPE_BOX if src_w >= src_h else PORTRAIT_BOX


def _fit(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int]:
    """Even dimensions that keep the aspect and fit inside the box, never upscaling."""
    scale = min(box_w / src_w, box_h / src_h, 1.0)
    w = max(2, int(src_w * scale) - (int(src_w * scale) % 2))
    h = max(2, int(src_h * scale) - (int(src_h * scale) % 2))
    return w, h


def _video_kbps(cap_bytes: int, duration_s: float) -> int:
    raw = math.floor((cap_bytes * CAP_HEADROOM * 8 / duration_s - AUDIO_KBPS * 1000) / 1000)
    return min(raw, MAX_VIDEO_KBPS)


def _transcode_to_cap(
    transcoder: Transcoder,
    src: str,
    review_path: str,
    part_path: str,
    *,
    width: int,
    height: int,
    cap_bytes: int,
    cap_mb: float,
    duration_s: float,
    passlogfile: str,
    timeout_s: float,
) -> tuple[int, int]:
    """Transcode under the cap, retrying once at a lower bit rate. Returns (video_kbps, attempts).

    The output is written to ``part_path`` and renamed to ``review_path`` only after its size is
    confirmed to be under the cap. A leftover ``.part`` from a crash is removed first.
    """
    _unlink_quiet(part_path)  # a leftover .part from a crash
    video_kbps = _video_kbps(cap_bytes, duration_s)
    if video_kbps < MIN_VIDEO_KBPS:
        raise DeliverError(
            "review_cannot_fit",
            f"a {duration_s:.3f} s review cannot fit {cap_mb:g} MB: the video budget is "
            f"{video_kbps} kbit/s, below the {MIN_VIDEO_KBPS} kbit/s floor",
            "Shorten the review, or raise review_mp4_max_mb in the profile.",
            duration_seconds=duration_s,
            cap_mb=cap_mb,
            video_kbps=video_kbps,
        )
    attempts = 0
    for attempt in (1, 2):
        attempts = attempt
        transcoder.transcode(
            src,
            part_path,
            width=width,
            height=height,
            video_kbps=video_kbps,
            audio_kbps=AUDIO_KBPS,
            passlogfile=passlogfile,
            timeout_s=timeout_s,
        )
        size = os.path.getsize(part_path)
        if size <= cap_bytes:
            os.replace(part_path, review_path)  # only after the size check passes
            return video_kbps, attempts
        if attempt == 1:
            video_kbps = math.floor(video_kbps * RETRY_FRACTION)
    over = os.path.getsize(part_path)
    _unlink_quiet(part_path)
    raise DeliverError(
        "review_over_cap",
        f"the review is {over} bytes after the retry, still over the {cap_bytes}-byte cap",
        "Shorten the review, or raise review_mp4_max_mb in the profile.",
        size=over,
        cap_bytes=cap_bytes,
        video_kbps=video_kbps,
    )


# -- facts ------------------------------------------------------------------------------------


def _record(
    log: EventLog,
    project_id: str,
    key: str,
    value: Mapping[str, Any],
    actor: Actor,
    idempotency_key: str,
    source: Mapping[str, Any] | None,
    fact_keys: list[str],
    replays: list[bool],
) -> None:
    result = record_fact(
        log, project_id, key, value, actor, idempotency_key=idempotency_key, source=source
    )
    fact_keys.append(key)
    replays.append(result.replayed)


def _review_keys(deliverable: str, render_job_id: str, sha256: str) -> dict[str, str]:
    base = f"deliver.review:{deliverable}:{render_job_id}:{sha256}"
    return {"hash": f"{base}:hash", "qa_size": f"{base}:qa-size", "review": f"{base}:review"}


def _master_keys(deliverable: str, render_job_id: str, sha256: str) -> dict[str, str]:
    base = f"deliver.master:{deliverable}:{render_job_id}:{sha256}"
    keys = {"hash": f"{base}:hash"}
    for metric in MASTER_QA_FACTS:
        keys[metric] = f"{base}:qa-{metric}"
    return keys


def _measured_value(report: QAReport, check_id: str, measured_key: str) -> float | None:
    try:
        check = report.check(check_id)
    except KeyError:
        return None
    value = check.measured.get(measured_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


# -- deliveries -------------------------------------------------------------------------------


def deliver_review(
    *,
    project_id: str,
    subject_id: str,
    deliverable: str,
    actor: Actor,
    profile: Profile,
    title: str,
    timeline: str | None,
    fps: float | None,
    kind: str,
    render_output: RenderOutput,
    workdir: str,
    edit_host: EditHostLike,
    log: EventLog,
    store: ArtifactStore,
    transcoder: Transcoder | None = None,
    qa_check: QACheckFn | None = None,
) -> DeliveryResult:
    """Pull the render, fit it under the review cap, QA it, store it, record facts, request review.

    ``kind`` ("short"/"longform") is the source kind; the review box is chosen by the SOURCE
    aspect, never by ``kind``.
    """
    if kind not in ("short", "longform"):
        raise DeliverError(
            "invalid_name",
            f"kind must be short or longform, got {kind!r}",
            "Pass a source kind.",
            field="kind",
        )
    _check_workdir(workdir)
    pid = _segment(project_id, "project_id", _PATH_SEGMENT)
    dname = _segment(deliverable, "deliverable", _FACT_SEGMENT)
    rjid = _segment(render_output.render_job_id, "render_job_id", _PATH_SEGMENT)
    transcoder = transcoder or FfmpegTranscoder()
    run_qa = qa_check or _default_qa_check
    cap_mb = profile.qa.review_mp4_max_mb
    cap_bytes = int(cap_mb * MB)

    work = _delivery_dir(workdir, pid, dname, rjid)
    source_path = os.path.join(work, "source" + _suffix(render_output.output_path))
    review_path = os.path.join(work, "review.mp4")
    part_path = review_path + ".part"
    passlogfile = os.path.join(work, "ffpass")
    timeout_s = timeout_for(render_output.duration_seconds, base_s=120.0, per_media_s=30.0)

    completed: list[str] = []
    resumed: list[str] = []
    try:
        # 1. pull
        source_sha, source_size = _pull(edit_host, render_output.output_path, source_path, resumed)
        completed.append("pull")

        # 2. fit: choose the box by the source aspect, fit (never upscale), transcode if needed
        info = probe(source_path, timeout_s=min(PROBE_TIMEOUT_S, timeout_s))
        video = info.video
        if video is None or video.width is None or video.height is None:
            raise DeliverError(
                "transcode_failed",
                "the pulled render has no video stream to fit to the review box",
                "Re-render with a video stream.",
                path=source_path,
            )
        box = _box_for(video.width, video.height)
        fits_box = video.width <= box[0] and video.height <= box[1]
        if fits_box and source_size <= cap_bytes:
            review_file, transcoded = source_path, False
            width, height, video_kbps, attempts = video.width, video.height, None, 0
        else:
            width, height = _fit(video.width, video.height, box[0], box[1])
            transcoded = True
            if os.path.exists(review_path) and os.path.getsize(review_path) <= cap_bytes:
                resumed.append("transcode")
                review_file, attempts = review_path, 0
                video_kbps = _video_kbps(cap_bytes, render_output.duration_seconds)
            else:
                video_kbps, attempts = _transcode_to_cap(
                    transcoder,
                    source_path,
                    review_path,
                    part_path,
                    width=width,
                    height=height,
                    cap_bytes=cap_bytes,
                    cap_mb=cap_mb,
                    duration_s=render_output.duration_seconds,
                    passlogfile=passlogfile,
                    timeout_s=timeout_s,
                )
                review_file = review_path
        completed.append("transcode" if transcoded else "fit")

        # 3. QA: a failed review is never stored or sent
        report = run_qa(review_file, kind="review", profile=profile)
        if report.status == "fail":
            raise DeliverError(
                "review_qa_failed",
                f"review QA failed: {', '.join(c.id for c in report.checks if c.status == 'fail')}",
                "Fix the render and re-run; QA facts are not recorded for a failed review.",
                failed_checks=[c.id for c in report.checks if c.status == "fail"],
                report=report.model_dump(mode="json"),
            )
        completed.append("qa")

        # 4. store (content-addressed, so a re-run reuses the same id)
        review_size = os.path.getsize(review_file)
        artifact_id = store.put(
            review_file,
            actor=actor,
            meta={
                "deliverable": dname,
                "role": "review",
                "render_job_id": rjid,
                "source_sha256": source_sha,
                "width": width,
                "height": height,
                "video_kbps": video_kbps,
            },
            mime="video/mp4",
            project_id=pid,
            producing_op="deliver.review",
        )
        completed.append("store")

        # 5. facts: the review file's hash and size, under deterministic keys
        keys = _review_keys(dname, rjid, artifact_id)
        fact_keys: list[str] = []
        replays: list[bool] = []
        source = {"artifact": artifact_id}
        _record(
            log,
            pid,
            f"render.hash.{dname}",
            {"sha256": artifact_id, "bytes": review_size},
            actor,
            keys["hash"],
            source,
            fact_keys,
            replays,
        )
        _record(
            log,
            pid,
            f"qa.size.{dname}",
            {"value": round(review_size / MB, 6), "unit": "MB"},
            actor,
            keys["qa_size"],
            source,
            fact_keys,
            replays,
        )
        if replays and all(replays):
            resumed.append("facts")
        completed.append("facts")

        # 6. review request (idempotent)
        if log.get_by_idempotency_key(f"{keys['review']}:node") is not None:
            resumed.append("review")
        review: Review = request_review(
            log,
            store,
            pid,
            subject_id,
            [artifact_id],
            actor,
            title=title,
            timeline=timeline,
            fps=fps,
            idempotency_key=keys["review"],
        )
        completed.append("review")
    except DeliverError as exc:
        exc.detail.setdefault("completed_steps", list(completed))
        raise

    return DeliveryResult(
        role="review",
        deliverable=dname,
        artifact_id=artifact_id,
        sha256=artifact_id,
        size=review_size,
        width=width,
        height=height,
        video_kbps=video_kbps,
        transcoded=transcoded,
        attempts=attempts,
        qa_status=report.status,
        fact_keys=fact_keys,
        review_id=review.id,
        resumed_steps=resumed,
    )


def deliver_master(
    *,
    project_id: str,
    deliverable: str,
    actor: Actor,
    profile: Profile,
    kind: str,
    render_output: RenderOutput,
    workdir: str,
    edit_host: EditHostLike,
    log: EventLog,
    store: ArtifactStore,
    qa_check: QACheckFn | None = None,
) -> DeliveryResult:
    """Pull the master, store it, QA it, record its hash and QA facts. No transcode, no review.

    The master is stored whether QA passes or fails (the publish gate is a human); the hash fact
    is always written, and a QA fact is written for each numeric value a check actually measured.
    Masters can be several GB: every hash and copy is streamed, never read whole into memory.
    """
    if kind not in ("short", "longform"):
        raise DeliverError(
            "invalid_name",
            f"kind must be short or longform, got {kind!r}",
            "Pass a source kind.",
            field="kind",
        )
    _check_workdir(workdir)
    pid = _segment(project_id, "project_id", _PATH_SEGMENT)
    dname = _segment(deliverable, "deliverable", _FACT_SEGMENT)
    rjid = _segment(render_output.render_job_id, "render_job_id", _PATH_SEGMENT)
    run_qa = qa_check or _default_qa_check

    work = _delivery_dir(workdir, pid, dname, rjid)
    suffix = _suffix(render_output.output_path)
    master_path = os.path.join(work, "master" + suffix)
    timeout_s = timeout_for(render_output.duration_seconds, base_s=300.0, per_media_s=60.0)

    completed: list[str] = []
    resumed: list[str] = []
    try:
        # 1. pull
        source_sha, _ = _pull(edit_host, render_output.output_path, master_path, resumed)
        completed.append("pull")

        # 2. store (role: master); streams the file, so a GB master never loads into memory
        artifact_id = store.put(
            master_path,
            actor=actor,
            meta={
                "deliverable": dname,
                "role": "master",
                "render_job_id": rjid,
                "source_sha256": source_sha,
            },
            mime=_MIME.get(suffix),
            project_id=pid,
            producing_op="deliver.master",
        )
        completed.append("store")
        master_size = os.path.getsize(master_path)

        # 3. QA (never fatal: the master is stored either way)
        report = run_qa(master_path, kind=kind, profile=profile, timeout_s=timeout_s)
        completed.append("qa")

        # 4. facts: the hash is always written; a QA fact only for a measured value
        keys = _master_keys(dname, rjid, artifact_id)
        fact_keys: list[str] = []
        replays: list[bool] = []
        source = {"artifact": artifact_id}
        _record(
            log,
            pid,
            f"render.hash.{dname}",
            {"sha256": artifact_id, "bytes": master_size},
            actor,
            keys["hash"],
            source,
            fact_keys,
            replays,
        )
        for metric, (check_id, measured_key, unit) in MASTER_QA_FACTS.items():
            value = _measured_value(report, check_id, measured_key)
            if value is not None:
                _record(
                    log,
                    pid,
                    f"qa.{metric}.{dname}",
                    {"value": value, "unit": unit},
                    actor,
                    keys[metric],
                    source,
                    fact_keys,
                    replays,
                )
        if replays and all(replays):
            resumed.append("facts")
        completed.append("facts")
    except DeliverError as exc:
        exc.detail.setdefault("completed_steps", list(completed))
        raise

    return DeliveryResult(
        role="master",
        deliverable=dname,
        artifact_id=artifact_id,
        sha256=artifact_id,
        size=master_size,
        qa_status=report.status,
        fact_keys=fact_keys,
        resumed_steps=resumed,
    )
