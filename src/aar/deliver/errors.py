"""The typed error the delivery library raises. Every ``code`` is stable (listed in DESIGN)."""

from __future__ import annotations

from typing import Any

# Stable error codes. Keep in sync with docs/DESIGN.md "Delivery".
ERROR_CODES = (
    "invalid_render_output",  # from_measured got a missing, ill-typed or non-finite key
    "invalid_workdir",  # the work dir is missing, not absolute, or under a temp root
    "invalid_name",  # a project/deliverable/render-job id is not a safe path segment
    "invalid_input",  # a value error after the first write (wrapped, carries completed_steps)
    "deliver_local_mismatch",  # a local file exists with a different sha256 (never overwritten)
    "delivery_in_progress",  # another delivery already holds the work dir's lock (transport exit)
    "lock_unsupported",  # flock failed with an errno other than EWOULDBLOCK/EAGAIN (bad fs)
    "hub_unavailable",  # the hub event log is locked, read-only or not a db (sqlite3 error)
    "store_unavailable",  # the artifact store is unwritable (OSError/PermissionError from put)
    "local_io_error",  # a local-filesystem read/write error not from the hub or artifact store
    "review_cannot_fit",  # the review cannot meet the cap even at the floor bit rate
    "review_over_cap",  # the transcode stayed over the cap after the retry
    "transcode_failed",  # ffmpeg exited non-zero or the output did not decode
    "review_qa_failed",  # QA failed on the review render (nothing stored or sent)
    "media_unreadable",  # ffprobe could not read a pulled render (wrapped after the first write)
    "unknown_subject",  # request_review got a --subject that is not a node (wrapped)
    "hub_write_failed",  # the hub refused a fact or review write (wrapped, carries completed_steps)
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
