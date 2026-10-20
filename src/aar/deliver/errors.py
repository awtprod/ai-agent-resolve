"""The typed error the delivery library raises. Every ``code`` is stable (listed in DESIGN)."""

from __future__ import annotations

from typing import Any

# Stable error codes. Keep in sync with docs/DESIGN.md "Delivery".
ERROR_CODES = (
    "invalid_render_output",  # from_measured got a missing or ill-typed key
    "invalid_workdir",  # the work directory is missing or not an absolute path
    "invalid_name",  # a project/deliverable/render-job id is not a safe path segment
    "deliver_local_mismatch",  # a local file exists with a different sha256 (never overwritten)
    "review_cannot_fit",  # the review cannot meet the cap even at the floor bit rate
    "review_over_cap",  # the transcode stayed over the cap after the retry
    "transcode_failed",  # ffmpeg exited non-zero or the output did not decode
    "review_qa_failed",  # QA failed on the review render (nothing stored or sent)
)


class DeliverError(Exception):
    """A delivery step failed. Carries a stable ``code``, a ``fix_hint`` and detail."""

    def __init__(self, code: str, message: str, fix_hint: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix_hint = fix_hint
        self.detail = dict(detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "fix_hint": self.fix_hint,
                "detail": self.detail,
            },
        }
