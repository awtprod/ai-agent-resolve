"""Typed errors for the analysis workers, and the lazy imports of their optional extras.

numpy (extra ``analyze``), faster-whisper (extra ``whisper``) and OpenCV (extra ``faces``)
are imported only by the functions that need them, so ``import aar``, ``import aar.cli`` and
``import aar.analyze`` work without any of them. A command that needs a missing one raises
``AnalyzeError("missing-extra")`` whose hint names the extra to install.

Exit codes (the ``aar qa check`` split, docs/DESIGN.md): an error whose type is in
``INPUT_ERRORS`` means an input, a tool or a model could not be located or opened, or an
argument is bad (exit 2). ``output-error`` (the result could not be written) exits 74, as
``aar.cli_io`` does for a failing stdout. Every other type means the analysis ran and failed,
including media that exists but does not decode (exit 1).
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from aar.media import MediaError, MediaInputError, MediaTimeout, MediaToolMissing

EXIT_FAILED = 1
EXIT_INPUT = 2
EXIT_OUTPUT = 74  # as aar.cli_io: the output could not be written

INPUT_ERRORS = frozenset(
    {
        "missing-extra",
        "model-missing",
        "model-invalid",
        "model-outside-dir",
        "tool-missing",
        "input-missing",
        "invalid-argument",
        "invalid-plan",
        "input-too-long",
        "checkpoint-exists",
        "checkpoint-mismatch",
        "checkpoint-corrupt",
    }
)


class AnalyzeError(RuntimeError):
    """An analysis could not run or failed. ``type`` is a stable kebab-case id."""

    def __init__(self, type: str, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.type = type
        self.hint = hint

    @property
    def exit_code(self) -> int:
        if self.type == "output-error":
            return EXIT_OUTPUT
        return EXIT_INPUT if self.type in INPUT_ERRORS else EXIT_FAILED

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "message": str(self)}
        if self.hint:
            out["hint"] = self.hint
        return out


def from_media_error(exc: MediaError, what: str) -> AnalyzeError:
    """Map an ``aar.media`` error onto a typed analysis error about ``what``."""
    if isinstance(exc, MediaToolMissing):
        return AnalyzeError(
            "tool-missing", str(exc), hint="install ffmpeg or set AAR_FFMPEG / AAR_FFPROBE"
        )
    if isinstance(exc, MediaInputError):
        return AnalyzeError("input-missing", f"{what}: {exc}")
    if isinstance(exc, MediaTimeout):
        return AnalyzeError(
            "timeout",
            f"{what}: {exc}",
            hint="a slow or stalled mount, or a very long file: raise --timeout-s",
        )
    return AnalyzeError(
        "decode-failed", f"{what}: {exc.summary()}", hint="check the file plays; re-export it"
    )


def _require(module: str, extra: str, purpose: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError:
        raise AnalyzeError(
            "missing-extra",
            f"{purpose} needs the Python package {module!r}, which is not installed",
            hint=f"pip install 'ai-agent-resolve[{extra}]'",
        ) from None


def need_numpy() -> Any:
    """numpy, or ``AnalyzeError("missing-extra")`` naming the ``analyze`` extra."""
    return _require("numpy", "analyze", "audio analysis")


def need_cv2() -> Any:
    """cv2 (OpenCV), or ``AnalyzeError("missing-extra")`` naming the ``faces`` extra."""
    return _require("cv2", "faces", "face detection")


def need_faster_whisper() -> Any:
    """faster_whisper, or ``AnalyzeError("missing-extra")`` naming the ``whisper`` extra."""
    return _require("faster_whisper", "whisper", "transcription")
