"""Typed refusals for the hub render (S18b).

A :class:`RenderError` is a :class:`~aar.plan.apply.errors.CompileError`, so the CLI prints it the
same way (code, message, fix hint, details). Its codes include every FCPXML export code, because
the render builds the export's layout and an :class:`~aar.plan.export.ExportError` from it (a lint
error, a transition with no handle, a missing frame size) passes through under its own code.
"""

from __future__ import annotations

from typing import ClassVar

from aar.plan.apply.errors import CompileError
from aar.plan.export.errors import EXPORT_CODES, ExportError

#: Codes only the render raises.
RENDER_ONLY_CODES = frozenset(
    {
        "media_not_on_hub",
        "media_unreadable",
        "no_video_stream",
        "source_rate_mismatch",
        "source_vfr",
        "size_not_renderable",
        "frame_size_mismatch",
        "segment_past_media_end",
        "odd_resolution",
        "reframe_too_complex",
        "render_too_large",
        "bad_output_path",
        "tool_missing",
        "render_failed",
        "render_timeout",
        "render_incomplete",
    }
)

#: Every code the render may refuse with: its own plus the export's (from the shared layout).
RENDER_CODES = EXPORT_CODES | RENDER_ONLY_CODES

#: Codes for a render that was started and failed in ffmpeg (the CLI exits 3, not 2).
RUNTIME_CODES = frozenset({"render_failed", "render_timeout", "render_incomplete"})


class RenderError(CompileError):
    """A typed render refusal or failure. ``code`` is one of :data:`RENDER_CODES`."""

    codes: ClassVar[frozenset[str]] = RENDER_CODES

    @classmethod
    def from_export(cls, exc: ExportError) -> RenderError:
        """The same refusal, raised by the render (the layout is the export's)."""
        return cls(exc.code, exc.message, exc.fix_hint, **exc.details)


class RenderTerminated(KeyboardInterrupt):
    """The render got SIGTERM: like an interrupt, ffmpeg's process group was killed and the work
    dir removed on the way out (round 2)."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum
