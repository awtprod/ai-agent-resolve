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

from collections.abc import Mapping
from typing import Any

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

#: ``render_too_large`` is the plan's only for this reason: too many overlapping pictures with no
#: hard cut between them. Its other reasons (the encoder alone at the stub's resolution, a budget
#: that holds no sound of the mix) are the machine's memory, not the cut.
RENDER_TOO_LARGE_PLAN_REASON = "pictures"

#: Codes the ccn steps raise themselves that a different plan fixes: the plan's frame, sources
#: or names differ from the stub (``plan_not_from_stub``), and the review render's QA gate.
STEP_PLAN_CODES = frozenset({"plan_not_from_stub", "qa_failed"})


def upstream_failure(code: str, details: Mapping[str, Any] | None = None) -> bool:
    """Whether the ``render``/``export`` step failure ``code`` (as the step records it:
    ``render_<code>``, ``export_<code>`` or a step code) is the cut's fault. ``details`` are the
    library error's details (only ``render_too_large`` reads them: its ``reason``)."""
    if code in STEP_PLAN_CODES:
        return True
    if code.startswith("export_"):
        return code.removeprefix("export_") in PLAN_LAYOUT_CODES
    if code.startswith("render_"):
        inner = code.removeprefix("render_")
        if inner == "render_too_large":  # the step records it as render_render_too_large
            return (details or {}).get("reason") == RENDER_TOO_LARGE_PLAN_REASON
        return inner in PLAN_LAYOUT_CODES or inner in RENDER_PLAN_CODES
    return False


__all__ = [
    "PLAN_LAYOUT_CODES",
    "RENDER_PLAN_CODES",
    "RENDER_TOO_LARGE_PLAN_REASON",
    "STEP_PLAN_CODES",
    "upstream_failure",
]
