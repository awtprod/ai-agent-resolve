"""Typed refusals for the FCPXML exporter.

An :class:`ExportError` is a :class:`~aar.plan.apply.errors.CompileError` with its own code set,
so the CLI prints it the same way (code, message, fix hint, details). A refusal is never a guess
or a silent drop: a plan feature FCPXML cannot carry goes in the export report instead, and a
plan the exporter cannot turn into a correct file is refused with one of these codes.
"""

from __future__ import annotations

from typing import ClassVar

from aar.plan.apply.errors import CompileError

#: Every code the exporter may refuse with. Kept here so a test can assert the set is stable.
EXPORT_CODES = frozenset(
    {
        "lint_errors",
        "fps_not_representable",
        "bad_timeline_start",
        "empty_timeline",
        "unknown_source",
        "zero_length_segment",
        "media_not_on_edit_pc",
        "source_size_required",
        "conflicting_source_durations",
        "pool_source_not_exportable",
        "bad_media_map",
        "outro_audio_track_missing",
        "duplicate_transition",
        "no_handle_for_transition",
        "transitions_overlap",
        "duplicate_keyframe_time",
        "non_finite_value",
        "unsafe_text",
        "fade_too_long",
        "item_outside_timeline",
        "self_check_failed",
        "output_exists",
        "output_conflict",
        "rollback_failed",
    }
)


class ExportError(CompileError):
    """A typed export refusal. ``code`` is one of :data:`EXPORT_CODES`."""

    codes: ClassVar[frozenset[str]] = EXPORT_CODES
