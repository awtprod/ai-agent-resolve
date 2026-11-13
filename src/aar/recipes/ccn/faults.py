"""Which ``render`` and ``export`` failures the agent's cut caused (S19f): the one table.

A failure is marked ``upstream`` (:class:`~aar.recipes.context.StepFailure`) when a different plan
can fix it: running the step again on the same plan fails the same way, so the way on is
``recipe_reopen cut``. Everything about the environment (media missing, changed or unreadable, a
missing tool, timeouts, disk, the export directory, params, the exporter's self-check) stays
unmarked: a retry, or a fix on the hub, is the way on. Each code is decided from the code that
raises it (``aar.plan.render``, ``aar.plan.export``, :mod:`aar.recipes.ccn.steps`); DESIGN.md
"CCN long-form recipe" carries the same table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aar.qa.check import NO_LOUDNESS_LUFS
from aar.qa.report import QACheck, QAReport

#: Plan layout codes the exporter raises, and the renderer passes through (it builds the
#: export's layout), that only the plan's content decides.
PLAN_LAYOUT_CODES = frozenset(
    {
        "lint_errors",  # the plan's own lint (with the transcript) has errors
        "empty_timeline",  # no segments and no outro
        "unknown_source",  # a segment names a source the plan does not have
        "zero_length_segment",  # src_out <= src_in
        "source_size_required",  # a reframe or transform on a source without a frame size
        "conflicting_source_durations",  # two plan sources of one file disagree on its length
        "conflicting_source_start_timecodes",  # ... or on its start timecode
        "outro_audio_track_missing",  # the outro's audio_track is not in audio_tracks
        "duplicate_transition",  # two transitions on one join
        "no_handle_for_transition",  # a transition longer than the ranges' handles
        "transitions_overlap",  # a clip too short for its two transitions
        "duplicate_keyframe_time",  # two reframe keyframes on one frame
        "non_finite_value",  # a gain or zoom that is not a finite number
        "unsafe_text",  # a title, marker or track name XML cannot carry
        "fade_too_long",  # fades longer than the sound under them
        "item_outside_timeline",  # a title or marker outside the record range
    }
)

#: Codes only the renderer raises that a different plan fixes.
RENDER_PLAN_CODES = frozenset(
    {
        "segment_past_media_end",  # a range (with its handles) past the end of the media
        "reframe_too_complex",  # too many reframe keyframes for the filter script
    }
)

#: ``render_too_large`` is the plan's only for this reason, and only when more than one picture
#: overlaps (``details["pictures"] > 1``): the cut can put a hard cut between them. One picture
#: alone over the budget costs the stub source's frame size, which no cut changes, and the budget
#: is the machine's free memory. Its other reasons (the encoder alone at the stub's resolution, a
#: budget that holds no sound of the mix) are the machine's memory, not the cut.
RENDER_TOO_LARGE_PLAN_REASON = "pictures"

#: Codes the ccn steps raise themselves that a different plan fixes: the plan's frame, sources
#: or names differ from the stub (``plan_not_from_stub``). ``qa_failed`` is decided per failure
#: by :func:`qa_failure_upstream`, not here.
STEP_PLAN_CODES = frozenset({"plan_not_from_stub"})

#: The review render's QA checks whose failure can be a judgement about the cut, each with the
#: measured key :mod:`aar.qa.check` writes only when the check ran on the content and judged it
#: (the value may be ``None``: digital silence measures ``-inf``, stored as ``None``). A check
#: that could not run fails without it: ``measure_audio`` or ``count_clipped`` raising a
#: ``MediaError`` (a timeout, an ffmpeg error), no audio stream, NaN samples. A loudness failure
#: on a gated loudness whose true peak was not measured (:func:`_true_peak_unmeasured`) is a
#: check that could not run either. ``decode`` (the renderer's output does not decode, or ffprobe
#: timed out), ``duration`` (the renderer already refuses output that does not hold the
#: timeline's frames, ``render_incomplete``, so a QA duration failure after it is the container
#: or the measurement, which no cut fixes), ``longest-silence`` (it can only warn, which the gate
#: allows: it fails the gate only when it could not measure or is incomplete) and ``size-cap``
#: (not applied to this review render) are never the cut's.
QA_JUDGED_BY = {
    "loudness": "integrated_lufs",
    "clipping": "clipped_samples",
}
#: ``join-quiet`` judges each join; a join entry carries this key only when its source audio was
#: read and judged (a join whose audio could not be read is ``unchecked``, or failed without it).
JOIN_JUDGED_BY = "run_at_cut"


def upstream_failure(code: str, details: Mapping[str, Any] | None = None) -> bool:
    """Whether the ``render``/``export`` step failure ``code`` (as the step records it:
    ``render_<code>``, ``export_<code>`` or a step code) is the cut's fault. ``details`` are the
    library error's details (only ``render_too_large`` reads them: ``reason`` and ``pictures``).
    ``qa_failed`` is never decided here (see :func:`qa_failure_upstream`)."""
    if code in STEP_PLAN_CODES:
        return True
    if code.startswith("export_"):
        return code.removeprefix("export_") in PLAN_LAYOUT_CODES
    if code.startswith("render_"):
        inner = code.removeprefix("render_")
        if inner == "render_too_large":  # the step records it as render_render_too_large
            found = details or {}
            pictures = found.get("pictures")
            return (
                found.get("reason") == RENDER_TOO_LARGE_PLAN_REASON
                and isinstance(pictures, int)
                and not isinstance(pictures, bool)
                and pictures > 1
            )
        return inner in PLAN_LAYOUT_CODES or inner in RENDER_PLAN_CODES
    return False


def qa_failure_upstream(report: QAReport, failing: Iterable[str]) -> bool:
    """Whether a ``qa_failed`` of the review render is the cut's: every failing required check
    (``failing``: the gate's failing check ids) ran on the content and judged it. A check that
    could not run, or ran incomplete, or is missing from the report, or was skipped, leaves the
    failure unmarked: a retry is the way on, since the plan may well be fine."""
    names = list(failing)
    if not names:
        return False
    by_id = {check.id: check for check in report.checks}
    unchecked = set(report.unchecked)
    for name in names:
        check = by_id.get(name)
        if check is None or check.incomplete or name in unchecked:
            return False
        if check.status not in ("fail", "warn"):
            return False  # skipped (not as not applicable): it did not run
        if not _judged(check):
            return False
    return True


def _judged(check: QACheck) -> bool:
    if check.id == "join-quiet":
        failed = [d for d in check.details if d.get("status") == "fail"]
        return bool(failed) and all(JOIN_JUDGED_BY in d for d in failed)
    key = QA_JUDGED_BY.get(check.id)
    if key is None or key not in check.measured:
        return False
    return not (check.id == "loudness" and _true_peak_unmeasured(check.measured))


def _true_peak_unmeasured(measured: Mapping[str, Any]) -> bool:
    """Whether a loudness check failed (also) because the true peak was not measured: the gated
    loudness is a number above :data:`~aar.qa.check.NO_LOUDNESS_LUFS` (so ``evaluate_loudness``
    went on to the true peak) and the true peak is missing. Digital silence (``-70`` LUFS or
    ``None``) fails on the loudness alone and stays judged."""
    lufs = measured.get("integrated_lufs")
    if not isinstance(lufs, (int, float)) or isinstance(lufs, bool) or lufs <= NO_LOUDNESS_LUFS:
        return False
    return measured.get("true_peak_dbtp") is None


__all__ = [
    "JOIN_JUDGED_BY",
    "PLAN_LAYOUT_CODES",
    "QA_JUDGED_BY",
    "RENDER_PLAN_CODES",
    "RENDER_TOO_LARGE_PLAN_REASON",
    "STEP_PLAN_CODES",
    "qa_failure_upstream",
    "upstream_failure",
]
