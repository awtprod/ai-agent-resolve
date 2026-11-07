"""Typed refusals for the ``plan.apply`` compiler.

Every refusal is one :class:`CompileError` with a stable ``code``, a human ``message`` and a
``fix_hint``, and optional ``details``. The codes are the compiler's contract: tests pin them,
and the CLI prints them. A refusal is never a guess or a silent drop.
"""

from __future__ import annotations

from typing import Any

#: Every code the compiler may refuse with. Kept here so a test can assert the set is stable.
CODES = frozenset(
    {
        "missing_template",
        "template_too_large",
        "template_size_unchecked",
        "template_name_mismatch",
        "batch_size_invalid",
        "segment_unmatched",
        "segment_double_matched",
        "missing_prepare_result",
        "missing_assemble_result",
        "phase_result_mismatch",
        "stale_phase_result",
        "forged_phase_result",
        "unresolved_reference",
        "no_handle_for_transition",
        "gain_out_of_range",
        "title_track_out_of_range",
        "title_too_many_lines",
        "too_many_steps",
        "zero_length_segment",
        "unknown_source",
        "duplicate_source",
        "source_not_found",
        "source_ambiguous",
        "title_template_missing",
        "title_template_ambiguous",
        "media_not_on_edit_pc",
        "outro_audio_track_missing",
        "end_fade_without_audio",
        "unknown_op",
        "lint_errors",
    }
)


class CompileError(Exception):
    """A typed compile refusal. ``code`` is one of :data:`CODES`."""

    def __init__(self, code: str, message: str, fix_hint: str, **details: Any) -> None:
        if code not in CODES:
            raise ValueError(f"unknown compile error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix_hint = fix_hint
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        """The refusal as a JSON-ready object (for ``--json`` and the hub error shape)."""
        return {
            "code": self.code,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "details": self.details,
        }
