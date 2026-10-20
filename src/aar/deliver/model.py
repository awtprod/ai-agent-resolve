"""The delivery library's input and output models.

``RenderOutput`` is the finished node ``render`` job's facts, carried to the hub. ``DeliveryResult``
is what a review or master delivery returns: the stored artifact, what was measured, and which
steps a resumed run skipped.
"""

from __future__ import annotations

from typing import Any

from aar.deliver.errors import DeliverError

_REQUIRED: tuple[tuple[str, type], ...] = (
    ("output_path", str),
    ("frame_count", int),
    ("frame_rate", (int, float)),  # type: ignore[assignment]
    ("duration_seconds", (int, float)),  # type: ignore[assignment]
    ("render_job_id", str),
)


class RenderOutput:
    """A finished render, built from the node ``render`` job's ``Done(measured)`` dict.

    ``from_measured`` refuses a missing or ill-typed key: the delivery library must never guess
    the output path, frame rate or duration.
    """

    __slots__ = ("duration_seconds", "frame_count", "frame_rate", "output_path", "render_job_id")

    def __init__(
        self,
        *,
        output_path: str,
        frame_count: int,
        frame_rate: float,
        duration_seconds: float,
        render_job_id: str,
    ) -> None:
        self.output_path = output_path
        self.frame_count = frame_count
        self.frame_rate = float(frame_rate)
        self.duration_seconds = float(duration_seconds)
        self.render_job_id = render_job_id

    @classmethod
    def from_measured(cls, measured: Any) -> RenderOutput:
        """Build from the render job's measured dict, refusing missing or ill-typed keys."""
        if not isinstance(measured, dict):
            raise DeliverError(
                "invalid_render_output",
                f"the render output must be an object, got {type(measured).__name__}",
                "Pass the node render job's measured dict (its Done(measured) keys).",
            )
        values: dict[str, Any] = {}
        for key, types in _REQUIRED:
            value = measured.get(key)
            # bool is an int subclass; a True frame_count or render boolean is ill-typed.
            if value is None or isinstance(value, bool) or not isinstance(value, types):
                want = types.__name__ if isinstance(types, type) else "number"
                raise DeliverError(
                    "invalid_render_output",
                    f"render output key {key!r} is missing or not a {want}: {value!r}",
                    "Give a render that finished: output_path, frame_count, frame_rate, "
                    "duration_seconds and render_job_id must all be present and well typed.",
                    key=key,
                )
            values[key] = value
        if values["output_path"] == "" or values["render_job_id"] == "":
            raise DeliverError(
                "invalid_render_output",
                "output_path and render_job_id must be non-empty",
                "Give a render that finished with a real output path and job id.",
            )
        if values["duration_seconds"] <= 0 or values["frame_rate"] <= 0:
            raise DeliverError(
                "invalid_render_output",
                "frame_rate and duration_seconds must be positive",
                "Give a render with a positive frame rate and duration.",
            )
        return cls(**values)


class DeliveryResult:
    """What a review or master delivery produced."""

    __slots__ = (
        "artifact_id",
        "attempts",
        "deliverable",
        "fact_keys",
        "height",
        "qa_status",
        "resumed_steps",
        "review_id",
        "role",
        "sha256",
        "size",
        "transcoded",
        "video_kbps",
        "width",
    )

    def __init__(
        self,
        *,
        role: str,
        deliverable: str,
        artifact_id: str,
        sha256: str,
        size: int,
        qa_status: str,
        fact_keys: list[str],
        width: int | None = None,
        height: int | None = None,
        video_kbps: int | None = None,
        transcoded: bool = False,
        attempts: int = 0,
        review_id: str | None = None,
        resumed_steps: list[str] | None = None,
    ) -> None:
        self.role = role
        self.deliverable = deliverable
        self.artifact_id = artifact_id
        self.sha256 = sha256
        self.size = size
        self.qa_status = qa_status
        self.fact_keys = fact_keys
        self.width = width
        self.height = height
        self.video_kbps = video_kbps
        self.transcoded = transcoded
        self.attempts = attempts
        self.review_id = review_id
        self.resumed_steps = resumed_steps or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "role": self.role,
            "deliverable": self.deliverable,
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "size": self.size,
            "width": self.width,
            "height": self.height,
            "video_kbps": self.video_kbps,
            "transcoded": self.transcoded,
            "attempts": self.attempts,
            "qa_status": self.qa_status,
            "fact_keys": list(self.fact_keys),
            "review_id": self.review_id,
            "resumed_steps": list(self.resumed_steps),
        }
