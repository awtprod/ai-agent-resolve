"""``check(...)``: run the media QA checks on a rendered file and return a structured report.

Checks (ids in report order): decode, loudness, clipping, longest-silence, size-cap, duration,
join-quiet, join-retranscribe, caption-coverage. Thresholds come from the profile's ``[qa]``
block (``QATargets``). docs/DESIGN.md "Media QA" describes each one.

A check that cannot run is ``skipped`` with a reason, never ``pass``. Any failure to measure
(ffmpeg error, timeout, unreadable file) is a ``fail`` with the tool's message.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aar.media import (
    AudioMeasurements,
    MediaError,
    MediaInfo,
    MediaInputError,
    MediaTimeout,
    count_clipped,
    decode_check,
    measure_audio,
    probe,
    quiet_runs,
    read_pcm,
    resolve_tool,
    timeout_for,
)
from aar.media.probe import PROBE_TIMEOUT_S
from aar.plan.inputs import Transcript, Word
from aar.plan.model import EditPlan, Source
from aar.plan.profile import Profile, QATargets
from aar.qa.planmap import Cut, KeptWord, cuts, kept_words, straddled_words
from aar.qa.report import CHECK_IDS, Kind, QACheck, QAReport, Status, build_report, skipped
from aar.qa.transcriber import Transcriber

MB = 1_000_000  # review_mp4_max_mb counts decimal megabytes
NO_LOUDNESS_LUFS = -70.0  # ebur128 reports this when no block passes the absolute gate
_EPS = 1e-9
_FRAME_EPS = 1e-3  # ffprobe prints durations to the microsecond
_LIST = 3  # examples named in a one-line message
_KINDS = ("short", "longform", "review")


# ---------------------------------------------------------------- small helpers


def _round(value: float | None, digits: int) -> float | None:
    """Round for the report; non-finite values (``-inf`` for digital silence) become None."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def clock(seconds: float) -> str:
    """``m:ss.ss`` for messages."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{int(seconds // 60)}:{seconds % 60:05.2f}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _examples(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_LIST])
    more = len(items) - _LIST
    return shown + (f" and {more} more" if more > 0 else "")


_TOKEN_STRIP = re.compile(r"[^\w' ]+")


def tokens(words: Sequence[str]) -> list[str]:
    """Lower-case word tokens with punctuation removed, for agreement scoring."""
    out: list[str] = []
    for text in words:
        cleaned = _TOKEN_STRIP.sub(" ", text.lower()).replace("_", " ")
        out.extend(t.strip("'") for t in cleaned.split() if t.strip("'"))
    return out


def agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """``difflib`` ratio of two token lists (1.0 when both are empty)."""
    return difflib.SequenceMatcher(None, list(a), list(b), autojunk=False).ratio()


# ---------------------------------------------------------------- pure evaluations


def evaluate_loudness(
    integrated_lufs: float,
    true_peak_dbtp: float | None,
    *,
    target_lufs: float,
    tolerance_lu: float,
    true_peak_max_dbtp: float,
) -> tuple[Status, str]:
    """Pass when ``|I - target| <= tolerance`` and true peak ``<= max`` (both inclusive)."""
    if not math.isfinite(integrated_lufs) or integrated_lufs <= NO_LOUDNESS_LUFS:
        return "fail", (
            f"No gated loudness (I = {integrated_lufs:.1f} LUFS): the audio is silent or shorter "
            "than 400 ms; check the mix."
        )
    problems: list[str] = []
    delta = integrated_lufs - target_lufs
    if abs(delta) > tolerance_lu + _EPS:
        way, fix = ("below", "raise") if delta < 0 else ("above", "lower")
        problems.append(
            f"integrated {integrated_lufs:.1f} LUFS is {abs(delta):.1f} LU {way} the "
            f"{target_lufs:g} target (± {tolerance_lu:g}): {fix} the gain by {abs(delta):.1f} dB"
        )
    if true_peak_dbtp is None:
        problems.append("true peak was not measured")
    elif true_peak_dbtp > true_peak_max_dbtp + _EPS:
        over = true_peak_dbtp - true_peak_max_dbtp
        problems.append(
            f"true peak {true_peak_dbtp:.1f} dBTP is above {true_peak_max_dbtp:g}: lower the "
            f"limiter ceiling or the gain by {over:.1f} dB"
        )
    if problems:
        return "fail", _cap("; ".join(problems)) + "."
    return "pass", (
        f"Integrated {integrated_lufs:.1f} LUFS (target {target_lufs:g} ± {tolerance_lu:g}) and "
        f"true peak {true_peak_dbtp:.1f} dBTP (max {true_peak_max_dbtp:g}) are within target."
    )


def evaluate_silence(longest_s: float, warn_s: float | None) -> Status:
    """Warn only when a warn threshold is set and the longest silence is strictly longer."""
    if warn_s is not None and longest_s > warn_s + _EPS:
        return "warn"
    return "pass"


def evaluate_size(size_bytes: int, max_mb: float) -> Status:
    """Pass when the file is at most ``max_mb`` decimal megabytes."""
    return "pass" if size_bytes <= max_mb * MB else "fail"


def evaluate_duration(diff_frames: float, tolerance_frames: int) -> Status:
    """Pass when the render is within ``tolerance_frames`` of the plan (inclusive)."""
    return "pass" if abs(diff_frames) <= tolerance_frames + _FRAME_EPS else "fail"


def evaluate_quiet_run(run_ms: float, min_ms: float) -> Status:
    """Pass when the longest quiet run is at least ``min_ms``."""
    return "pass" if run_ms >= min_ms - 1e-6 else "fail"


# ---------------------------------------------------------------- context


@dataclass
class _Ctx:
    path: Path
    kind: Kind
    qa: QATargets
    plan: EditPlan | None
    transcript: Transcript | None
    transcriber: Transcriber | None
    silence_db: float
    media_root: Path | None
    timeout_s: float | None
    info: MediaInfo | None = None
    sources: dict[str, tuple[Path | None, MediaInfo | None, str]] = field(default_factory=dict)

    def timeout(self, per_media_s: float, duration_s: float | None = None) -> float:
        if self.timeout_s is not None:
            return self.timeout_s
        if duration_s is None and self.info is not None:
            duration_s = self.info.duration_s
        return timeout_for(duration_s, per_media_s=per_media_s)

    @property
    def lufs_target(self) -> float:
        # A review render is a proxy of the long-form cut, so it is held to the long-form target.
        return self.qa.short_lufs if self.kind == "short" else self.qa.longform_lufs

    def render_duration(self) -> tuple[str, float | None]:
        """The render's length: video stream, else audio stream, else container."""
        info = self.info
        assert info is not None
        for name, stream in (("video stream", info.video), ("audio stream", info.audio)):
            if stream is not None and stream.duration_s is not None:
                return name, stream.duration_s
        return "container", info.duration_s

    def source(self, source: Source) -> tuple[Path | None, MediaInfo | None, str]:
        """Resolve and probe a plan source once: (path, info, why-unreadable)."""
        if source.id in self.sources:
            return self.sources[source.id]
        path = resolve_media(source.media, self.media_root)
        result: tuple[Path | None, MediaInfo | None, str]
        if path is None:
            result = (
                None,
                None,
                f"source {source.id!r} media {source.media!r} is not a local path",
            )
        else:
            try:
                info = probe(path, timeout_s=self._probe_timeout())
            except MediaError as exc:
                result = (path, None, f"source {source.id!r}: {exc.summary()}")
            else:
                if info.audio is None:
                    result = (path, None, f"source {source.id!r} has no audio stream")
                else:
                    result = (path, info, "")
        self.sources[source.id] = result
        return result

    def _probe_timeout(self) -> float:
        return PROBE_TIMEOUT_S if self.timeout_s is None else min(PROBE_TIMEOUT_S, self.timeout_s)


def resolve_media(media: str, media_root: Path | None) -> Path | None:
    """A plan source's media as a local path: absolute, or relative to ``media_root``.

    URIs (``scheme://...``) are not local paths. S13f adds the edit-PC to hub path map.
    """
    if not media or "://" in media:
        return None
    path = Path(media)
    if not path.is_absolute() and media_root is not None:
        path = media_root / path
    return path


# ---------------------------------------------------------------- checks


def _decode(ctx: _Ctx, error: MediaError | None) -> QACheck:
    if ctx.info is None:
        assert error is not None
        if isinstance(error, MediaTimeout):
            message = (
                f"ffprobe timed out after {error.timeout_s:g} s: the file is not readable media "
                "or sits on a stalled mount."
            )
        elif isinstance(error, MediaInputError):
            message = f"{_cap(str(error))}: point QA at the rendered file."
        else:
            message = f"Not readable media ({error.summary()}): re-render or re-export the file."
        return QACheck(
            id="decode",
            status="fail",
            message=message,
            measured={"readable": False, "exit_code": error.exit_code},
        )
    try:
        result = decode_check(ctx.path, timeout_s=ctx.timeout(3.0))
    except MediaError as exc:
        return QACheck(
            id="decode",
            status="fail",
            message=f"The full decode did not finish ({exc.summary()}): re-render the file.",
            measured={"readable": True, "exit_code": exc.exit_code},
        )
    measured = {
        "readable": True,
        "exit_code": result.exit_code,
        "error_lines": result.error_lines,
        "errors": result.errors,
    }
    if result.ok:
        return QACheck(
            id="decode",
            status="pass",
            message="Every video and audio stream decodes without an error.",
            measured=measured,
        )
    first = result.errors[0] if result.errors else f"exit status {result.exit_code}"
    return QACheck(
        id="decode",
        status="fail",
        message=(
            f"{result.error_lines} decode error line(s), first: {first!r}: re-render or "
            "re-export the file."
        ),
        measured=measured,
    )


def _audio(ctx: _Ctx) -> list[QACheck]:
    info = ctx.info
    assert info is not None
    loud_target = {
        "integrated_lufs": ctx.lufs_target,
        "tolerance_lu": ctx.qa.lufs_tolerance,
        "true_peak_max_dbtp": ctx.qa.true_peak_max_dbtp,
    }
    if info.audio is None:
        why = "the render has no audio stream"
        return [
            QACheck(
                id="loudness",
                status="fail",
                message=(
                    "The render has no audio stream: re-render with audio (a proxy-only or "
                    "offline clip renders silent)."
                ),
                measured={"audio_streams": 0},
                target=loud_target,
            ),
            skipped("clipping", "no-audio", why),
            skipped("longest-silence", "no-audio", why),
        ]
    try:
        measured = measure_audio(
            ctx.path,
            silence_db=ctx.silence_db,
            silence_min_s=ctx.qa.silence_min_s,
            timeout_s=ctx.timeout(2.0),
        )
    except MediaError as exc:
        message = f"Could not measure the audio ({exc.summary()}): check the audio stream decodes."
        return [
            QACheck(id=check_id, status="fail", message=message)
            for check_id in ("loudness", "clipping", "longest-silence")
        ]
    return [
        _loudness(ctx, measured, loud_target),
        _clipping(ctx, measured),
        _silence(ctx, measured),
    ]


def _loudness(ctx: _Ctx, m: AudioMeasurements, target: dict[str, Any]) -> QACheck:
    loud = m.loudness
    status, message = evaluate_loudness(
        loud.integrated_lufs,
        loud.true_peak_dbtp,
        target_lufs=ctx.lufs_target,
        tolerance_lu=ctx.qa.lufs_tolerance,
        true_peak_max_dbtp=ctx.qa.true_peak_max_dbtp,
    )
    return QACheck(
        id="loudness",
        status=status,
        message=message,
        measured={
            "integrated_lufs": _round(loud.integrated_lufs, 1),
            "true_peak_dbtp": _round(loud.true_peak_dbtp, 1),
            "sample_peak_dbfs": _round(loud.sample_peak_dbfs, 1),
            "lra_lu": _round(loud.lra_lu, 1),
        },
        target=target,
    )


def _clipping(ctx: _Ctx, m: AudioMeasurements) -> QACheck:
    info = ctx.info
    assert info is not None and info.audio is not None
    level_db = ctx.qa.clip_level_dbfs
    target = {"clipped_samples": 0, "clip_level_dbfs": level_db}
    peak = m.stats.peak_level_dbfs
    non_finite = m.stats.non_finite
    measured: dict[str, Any] = {"sample_peak_dbfs": _round(peak, 4), "non_finite_samples": 0}
    if non_finite:
        measured["non_finite_samples"] = non_finite
        return QACheck(
            id="clipping",
            status="fail",
            message=f"{non_finite} NaN or infinite sample(s): re-render the audio.",
            measured=measured,
            target=target,
        )
    if peak is None:
        return QACheck(
            id="clipping",
            status="fail",
            message="astats reported no sample peak: check the audio stream decodes.",
            measured=measured,
            target=target,
        )
    if peak < level_db:
        measured.update(clipped_samples=0, method="sample peak below the clip level")
        return QACheck(
            id="clipping",
            status="pass",
            message=f"No sample reaches {level_db:g} dBFS (peak {peak:.2f} dBFS).",
            measured=measured,
            target=target,
        )
    try:
        count = count_clipped(
            ctx.path,
            level=10.0 ** (level_db / 20.0),
            channels=info.audio.channels,
            sample_rate=info.audio.sample_rate,
            timeout_s=ctx.timeout(3.0),
        )
    except MediaError as exc:
        return QACheck(
            id="clipping",
            status="fail",
            message=f"Could not count clipped samples ({exc.summary()}).",
            measured=measured,
            target=target,
        )
    measured.update(
        clipped_samples=count.clipped,
        first_clip_s=_round(count.first_s, 4),
        scanned_samples=count.scanned,
        method="decoded float samples counted",
    )
    if count.clipped:
        where = f" (first at {clock(count.first_s)})" if count.first_s is not None else ""
        return QACheck(
            id="clipping",
            status="fail",
            message=(
                f"{count.clipped} clipped sample(s){where}: lower the gain and limit to "
                f"{ctx.qa.true_peak_max_dbtp:g} dBTP."
            ),
            measured=measured,
            target=target,
        )
    return QACheck(
        id="clipping",
        status="pass",
        message=f"No sample reaches {level_db:g} dBFS.",
        measured=measured,
        target=target,
    )


def _silence(ctx: _Ctx, m: AudioMeasurements) -> QACheck:
    info = ctx.info
    assert info is not None and info.audio is not None
    qa = ctx.qa
    stream_end = info.audio.duration_s if info.audio.duration_s is not None else info.duration_s
    target = {"silence_db": ctx.silence_db, "min_s": qa.silence_min_s}
    if qa.longest_silence_warn_s is not None:
        target["warn_s"] = qa.longest_silence_warn_s
    spans = [(s, s.duration(stream_end) or 0.0) for s in m.silences]
    longest = max(spans, key=lambda item: item[1], default=None)
    measured: dict[str, Any] = {
        "silences": len(spans),
        "longest_silence_s": 0.0,
        "log_truncated": m.log_truncated,
    }
    if longest is None:
        status: Status = "pass"
        message = f"No silence of {qa.silence_min_s:g} s or longer below {ctx.silence_db:g} dB."
    else:
        span, seconds = longest
        measured.update(
            longest_silence_s=round(seconds, 3),
            start_s=round(span.start_s, 3),
            end_s=_round(span.end_s if span.end_s is not None else stream_end, 3),
        )
        status = evaluate_silence(seconds, qa.longest_silence_warn_s)
        if status == "warn":
            message = (
                f"Longest silence {seconds:.2f} s at {clock(span.start_s)} is longer than "
                f"{qa.longest_silence_warn_s:g} s: trim the pause to a beat."
            )
        else:
            message = (
                f"Longest silence below {ctx.silence_db:g} dB is {seconds:.2f} s at "
                f"{clock(span.start_s)}."
            )
    if m.log_truncated:
        status = "warn" if status == "pass" else status
        message += " The silence log was truncated, so an earlier silence may be missing."
    return QACheck(
        id="longest-silence", status=status, message=message, measured=measured, target=target
    )


def _size_cap(ctx: _Ctx) -> QACheck:
    info = ctx.info
    assert info is not None
    max_mb = ctx.qa.review_mp4_max_mb
    if ctx.kind != "review":
        return skipped(
            "size-cap",
            "not-applicable",
            f"the size cap applies to review renders and this is a {ctx.kind} render",
            max_mb=max_mb,
        )
    size = info.size_bytes
    status = evaluate_size(size, max_mb)
    measured = {"size_bytes": size, "size_mb": round(size / MB, 3)}
    if status == "pass":
        message = f"Review render is {size / MB:.2f} MB (cap {max_mb:g} MB)."
    else:
        message = (
            f"Review render is {size / MB:.2f} MB, over the {max_mb:g} MB cap: re-encode it "
            "smaller (720p, two-pass, lower video bitrate)."
        )
    return QACheck(
        id="size-cap",
        status=status,
        message=message,
        measured=measured,
        target={"max_mb": max_mb, "max_bytes": int(max_mb * MB)},
    )


def _duration(ctx: _Ctx) -> QACheck:
    plan = ctx.plan
    if plan is None:
        return skipped("duration", "no-plan", "no plan was given")
    fps = plan.fps
    plan_frames = plan.total_frames()
    plan_s = fps.seconds(plan_frames)
    tolerance = ctx.qa.duration_tolerance_frames
    target = {"plan_frames": plan_frames, "plan_s": round(plan_s, 6), "tolerance_frames": tolerance}
    what, render_s = ctx.render_duration()
    if render_s is None:
        return QACheck(
            id="duration",
            status="fail",
            message="The render's duration is unknown: re-export it with a proper container.",
            target=target,
        )
    diff_frames = (render_s - plan_s) * fps.num / fps.den
    status = evaluate_duration(diff_frames, tolerance)
    measured = {
        "render_s": round(render_s, 6),
        "measured_from": what,
        "diff_frames": round(diff_frames, 3),
    }
    if status == "pass":
        message = (
            f"Render {render_s:.3f} s matches the plan's {plan_s:.3f} s "
            f"({diff_frames:+.2f} frames, tolerance {tolerance})."
        )
    else:
        way = "longer" if diff_frames > 0 else "shorter"
        message = (
            f"Render is {abs(diff_frames):.1f} frames {way} than the plan ({render_s:.3f} s vs "
            f"{plan_s:.3f} s): re-render the planned timeline range."
        )
    return QACheck(id="duration", status=status, message=message, measured=measured, target=target)


def _cut_ref(plan: EditPlan, cut: Cut) -> dict[str, Any]:
    return {
        "join": cut.number,
        "outgoing": cut.outgoing,
        "incoming": cut.incoming,
        "record_s": round(cut.record_s, 3),
        "record_tc": plan.fps.timecode(cut.record_frame),
        "source": cut.source_id,
        "src_in_s": round(cut.src_in_s, 4),
    }


def _join_quiet(ctx: _Ctx) -> QACheck:
    plan = ctx.plan
    qa = ctx.qa
    target = {
        "quiet_db": qa.join_quiet_db,
        "min_ms": qa.join_quiet_min_ms,
        "search_s": qa.join_quiet_search_s,
        "window_ms": qa.join_quiet_window_ms,
    }
    if plan is None:
        return skipped("join-quiet", "no-plan", "no plan was given", **target)
    cut_list = cuts(plan, qa.join_continuity_tolerance_frames)
    if not cut_list:
        return QACheck(
            id="join-quiet",
            status="pass",
            message="The plan has no cuts: every join continues the same recording.",
            measured={"cuts": 0},
            target=target,
        )
    sources = plan.source_map()
    details: list[dict[str, Any]] = []
    for cut in cut_list:
        entry = _cut_ref(plan, cut)
        path, src_info, why = ctx.source(sources[cut.source_id])
        if src_info is None or path is None or src_info.audio is None:
            entry.update(status="skipped", reason=why)
            details.append(entry)
            continue
        lo = max(0.0, cut.src_in_s - qa.join_quiet_search_s)
        hi = cut.src_in_s + qa.join_quiet_search_s
        end = src_info.audio.duration_s or src_info.duration_s
        if end is not None:
            hi = min(hi, end)
        entry["window_s"] = [round(lo, 4), round(hi, 4)]
        if hi <= lo:
            entry.update(status="fail", reason="src_in is at or past the end of the source audio")
            details.append(entry)
            continue
        try:
            pcm = read_pcm(
                path,
                start_s=lo,
                duration_s=hi - lo,
                channels=src_info.audio.channels or 1,
                timeout_s=ctx.timeout(1.0, duration_s=0.0),
            )
        except MediaError as exc:
            entry.update(status="fail", reason=f"could not read the source audio: {exc.summary()}")
            details.append(entry)
            continue
        scan = quiet_runs(
            pcm, threshold_dbfs=qa.join_quiet_db, window_s=qa.join_quiet_window_ms / 1000
        )
        best = scan.longest
        run_ms = best.duration_s * 1000 if best is not None else 0.0
        entry.update(
            status=evaluate_quiet_run(run_ms, qa.join_quiet_min_ms),
            longest_quiet_ms=round(run_ms, 2),
            run_start_s=_round(best.start_s, 4) if best else None,
            run_end_s=_round(best.end_s, 4) if best else None,
            run_level_dbfs=_round(best.level_dbfs, 1) if best else None,
            contains_cut=bool(best and best.start_s <= cut.src_in_s <= best.end_s),
            quietest_dbfs=_round(scan.quietest_dbfs, 1),
            scanned_s=round(scan.scanned_s, 4),
        )
        details.append(entry)
    return _summarise_joins(
        "join-quiet",
        details,
        target,
        fail_text=(
            f"lack a {qa.join_quiet_min_ms:g} ms run below {qa.join_quiet_db:g} dBFS within "
            f"±{qa.join_quiet_search_s:g} s of src_in: move them into a pause"
        ),
        pass_text=(
            f"have a {qa.join_quiet_min_ms:g} ms run below {qa.join_quiet_db:g} dBFS within "
            f"±{qa.join_quiet_search_s:g} s of src_in"
        ),
    )


def _summarise_joins(
    check_id: str,
    details: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    fail_text: str,
    pass_text: str,
) -> QACheck:
    total = len(details)
    failed = [d for d in details if d["status"] == "fail"]
    unread = [d for d in details if d["status"] == "skipped"]
    measured = {"cuts": total, "failed": len(failed), "unchecked": len(unread)}
    if unread and len(unread) == total:
        return QACheck(
            id=check_id,
            status="skipped",
            skip_reason="source-unreadable",
            message=f"Skipped: no source could be read ({unread[0]['reason']}).",
            measured=measured,
            target=target,
            details=details,
        )
    if failed:
        where = [f"join {d['join']} at {clock(d['record_s'])}" for d in failed]
        message = f"{len(failed)} of {total} cuts {fail_text} ({_examples(where)})."
        status: Status = "fail"
    elif unread:
        message = (
            f"{len(unread)} of {total} cuts could not be checked ({unread[0]['reason']}); "
            f"the rest {pass_text}."
        )
        status = "warn"
    else:
        message = f"All {total} cuts {pass_text}."
        status = "pass"
    return QACheck(
        id=check_id,
        status=status,
        message=message,
        measured=measured,
        target=target,
        details=details,
    )


def _in_region(start: float, end: float, lo: float, hi: float) -> bool:
    """A word belongs to a region when its midpoint lies in ``[lo, hi)``."""
    return lo <= (start + end) / 2 < hi


def _join_retranscribe(ctx: _Ctx) -> QACheck:
    plan, transcript, transcriber = ctx.plan, ctx.transcript, ctx.transcriber
    qa = ctx.qa
    target = {
        "windows": qa.join_windows,
        "window_s": qa.join_window_s,
        "step_s": qa.join_window_step_s,
        "agreement_min": qa.join_agreement_min,
        "straddle_s": qa.word_straddle_s,
    }
    if plan is None:
        return skipped("join-retranscribe", "no-plan", "no plan was given", **target)
    if transcript is None:
        return skipped("join-retranscribe", "no-transcript", "no source transcript", **target)
    if transcriber is None:
        return skipped("join-retranscribe", "no-transcriber", "no transcriber was given", **target)
    info = ctx.info
    if info is None or info.audio is None:
        return skipped("join-retranscribe", "no-audio", "the render has no audio", **target)
    cut_list = cuts(plan, qa.join_continuity_tolerance_frames)
    if not cut_list:
        return QACheck(
            id="join-retranscribe",
            status="pass",
            message="The plan has no cuts: every join continues the same recording.",
            measured={"cuts": 0},
            target=target,
        )
    _, render_end = ctx.render_duration()
    if render_end is None:
        render_end = info.duration_s or 0.0
    expected = kept_words(plan, transcript)
    details = [
        _retranscribe_one(ctx, plan, transcript, transcriber, cut, expected, render_end)
        for cut in cut_list
    ]
    return _summarise_joins(
        "join-retranscribe",
        details,
        target,
        fail_text=(
            "do not re-transcribe consistently with the plan: listen to them and re-cut where "
            "a word is clipped or doubled"
        ),
        pass_text="re-transcribe in agreement with each other and the plan",
    )


def _retranscribe_one(
    ctx: _Ctx,
    plan: EditPlan,
    transcript: Transcript,
    transcriber: Transcriber,
    cut: Cut,
    expected: list[KeptWord],
    render_end: float,
) -> dict[str, Any]:
    qa = ctx.qa
    n = qa.join_windows
    half = qa.join_window_s / 2
    offsets = [(k - (n - 1) / 2) * qa.join_window_step_s for k in range(n)]
    centre = cut.record_s
    lo = max(0.0, centre - half + max(offsets))
    hi = min(render_end, centre + half + min(offsets))
    entry = _cut_ref(plan, cut)
    entry["region_s"] = [round(lo, 3), round(hi, 3)]
    want = tokens(
        [w.text for w in expected if _in_region(w.record_start_s, w.record_end_s, lo, hi)]
    )
    entry["expected"] = " ".join(want)
    problems: list[str] = []
    straddled = straddled_words(plan, transcript, cut, qa.word_straddle_s)
    if straddled:
        entry["straddled"] = [
            {"side": s.side, "word": s.text, "start_s": s.start_s, "end_s": s.end_s}
            for s in straddled
        ]
        problems.append("the cut slices " + _examples([repr(s.text) for s in straddled]))
    heard: list[list[str]] = []
    windows: list[dict[str, Any]] = []
    for offset in offsets:
        start = max(0.0, centre - half + offset)
        end = min(render_end, centre + half + offset)
        window: dict[str, Any] = {"start_s": round(start, 3), "end_s": round(end, 3)}
        try:
            words: Sequence[Word] = transcriber.transcribe(ctx.path, start, end)
        except Exception as exc:  # the transcriber is foreign code; a crash fails the join
            window["error"] = f"{type(exc).__name__}: {exc}"
            windows.append(window)
            problems.append(f"the transcriber failed ({type(exc).__name__}: {exc})")
            continue
        got = tokens([w.text for w in words if _in_region(w.start_s, w.end_s, lo, hi)])
        window["text"] = " ".join(got)
        window["ratio_vs_plan"] = round(agreement(got, want), 3)
        heard.append(got)
        windows.append(window)
    entry["windows"] = windows
    pair_ratios = [
        round(agreement(heard[i], heard[j]), 3)
        for i in range(len(heard))
        for j in range(i + 1, len(heard))
    ]
    entry["pair_ratios"] = pair_ratios
    ratios = pair_ratios + [w["ratio_vs_plan"] for w in windows if "ratio_vs_plan" in w]
    entry["min_ratio"] = min(ratios) if ratios else None
    if ratios and min(ratios) < qa.join_agreement_min - _EPS:
        problems.append(f"agreement {min(ratios):.2f} is below {qa.join_agreement_min:g}")
    entry["status"] = "fail" if problems else "pass"
    if problems:
        entry["reason"] = "; ".join(problems)
    return entry


def _caption_coverage(ctx: _Ctx) -> QACheck:
    plan, transcript = ctx.plan, ctx.transcript
    if plan is None:
        return skipped("caption-coverage", "no-plan", "no plan was given")
    if transcript is None:
        return skipped("caption-coverage", "no-transcript", "no source transcript")
    if not plan.captions and ctx.kind != "short":
        return skipped(
            "caption-coverage", "no-captions", f"the {ctx.kind} plan has no captions to check"
        )
    fps = plan.fps
    spans = sorted((fps.seconds(c.record_in), fps.seconds(c.record_out)) for c in plan.captions)
    kept = kept_words(plan, transcript)
    uncovered = [w for w in kept if not any(a <= w.record_mid_s < b for a, b in spans)]
    measured = {"kept_words": len(kept), "captions": len(spans), "uncovered": len(uncovered)}
    details = [
        {
            "word": w.text,
            "segment": w.segment,
            "record_start_s": round(w.record_start_s, 3),
            "record_end_s": round(w.record_end_s, 3),
        }
        for w in uncovered
    ]
    if uncovered:
        where = [f"{w.text!r} at {clock(w.record_start_s)}" for w in uncovered]
        message = (
            f"{len(uncovered)} of {len(kept)} kept words have no caption: caption "
            f"{_examples(where)}."
        )
        status: Status = "fail"
    else:
        message = f"All {len(kept)} kept words are covered by a caption."
        status = "pass"
    return QACheck(
        id="caption-coverage",
        status=status,
        message=message,
        measured=measured,
        target={"uncovered": 0, "rule": "word midpoint inside a caption's record range"},
        details=details,
    )


# ---------------------------------------------------------------- entry point


def check(
    path: str | Path,
    *,
    kind: Kind,
    profile: Profile,
    plan: EditPlan | None = None,
    transcript: Transcript | None = None,
    transcriber: Transcriber | None = None,
    silence_db: float | None = None,
    media_root: str | Path | None = None,
    timeout_s: float | None = None,
) -> QAReport:
    """Check a rendered file and return the report.

    ``kind`` picks the loudness target (short, or long-form for ``longform`` and ``review``)
    and whether the review size cap applies. ``plan`` enables the duration, join and caption
    checks; ``transcript`` (source words, recording time) and ``transcriber`` enable the
    word checks. ``silence_db`` overrides the profile's silence threshold (a venue value).
    Relative source media paths in the plan resolve against ``media_root``. ``timeout_s``
    caps every ffmpeg call; by default the timeout grows with the media length.

    Raises ``MediaToolMissing`` when ffmpeg or ffprobe cannot be found: a report with every
    check skipped would read as harmless.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {', '.join(_KINDS)}")
    if timeout_s is not None and not timeout_s > 0:
        raise ValueError("timeout_s must be positive")
    qa = profile.qa
    level = qa.silence_db if silence_db is None else float(silence_db)
    if not (math.isfinite(level) and level < 0):
        raise ValueError("silence_db must be a negative dB value")
    resolve_tool("ffprobe")
    resolve_tool("ffmpeg")
    ctx = _Ctx(
        path=Path(path),
        kind=kind,
        qa=qa,
        plan=plan,
        transcript=transcript,
        transcriber=transcriber,
        silence_db=level,
        media_root=None if media_root is None else Path(media_root),
        timeout_s=timeout_s,
    )
    error: MediaError | None = None
    try:
        ctx.info = probe(ctx.path, timeout_s=ctx._probe_timeout())
    except MediaError as exc:
        error = exc
    checks = [_decode(ctx, error)]
    if ctx.info is None:
        why = "the render could not be read"
        for check_id in CHECK_IDS[1:6]:
            checks.append(skipped(check_id, "render-unreadable", why))
        checks.append(_join_quiet(ctx))
        checks.append(skipped("join-retranscribe", "render-unreadable", why))
        checks.append(_caption_coverage(ctx))
    else:
        checks.extend(_audio(ctx))
        checks.append(_size_cap(ctx))
        checks.append(_duration(ctx))
        checks.append(_join_quiet(ctx))
        checks.append(_join_retranscribe(ctx))
        checks.append(_caption_coverage(ctx))
    assert [c.id for c in checks] == list(CHECK_IDS)
    return build_report(
        file=str(path),
        kind=kind,
        profile=profile.name,
        media=None if ctx.info is None else ctx.info.to_dict(),
        checks=checks,
    )
