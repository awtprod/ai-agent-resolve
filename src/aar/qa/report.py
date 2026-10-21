"""The QA report: one entry per check, each with a status, measured values and a target."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["pass", "warn", "fail", "skipped"]
Kind = Literal["short", "longform", "review"]

# Why a check did not run. Skipped never means passed: the message says what is missing.
# The first four are inputs the caller chose not to give. render-unreadable and no-audio
# always come with a failed decode or loudness check; not-applicable is the size cap on a
# render that is not a review.
SkipReason = Literal[
    "no-plan",
    "no-transcript",
    "no-transcriber",
    "no-captions",
    "render-unreadable",
    "no-audio",
    "not-applicable",
]

# Why a check that ran could not check everything (``measured["unchecked_reason"]``).
UncheckedReason = Literal["source-unreadable", "transcript-does-not-cover", "log-truncated"]

CHECK_IDS = (
    "decode",
    "loudness",
    "clipping",
    "longest-silence",
    "size-cap",
    "duration",
    "join-quiet",
    "join-retranscribe",
    "caption-coverage",
)

_RANK = {"skipped": 0, "pass": 0, "warn": 1, "fail": 2}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def json_clean(value: Any) -> Any:
    """Replace non-finite floats (``-inf`` for digital silence, NaN) with None, recursively,
    so a report always serialises to strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    return value


class QACheck(_Model):
    id: str = Field(description="Check id, e.g. 'loudness'.")
    status: Status
    message: str = Field(description="One line: what to do, or why the check was skipped.")
    skip_reason: SkipReason | None = None
    measured: dict[str, Any] = Field(default_factory=dict)
    target: dict[str, Any] = Field(default_factory=dict)
    details: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-join or per-word entries, when the check has any."
    )
    incomplete: bool = Field(
        default=False,
        description=(
            "True when the check ran but could not check everything (unreadable sources, "
            "a transcript that does not cover every source, a truncated log); "
            "measured and details say what was left out."
        ),
    )

    @field_validator("measured", "target", "details", mode="after")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        return json_clean(value)

    @field_validator("message", mode="after")
    @classmethod
    def _one_line(cls, value: str) -> str:
        # File names and tool output can carry newlines; the message stays one line.
        return _CONTROL.sub(" ", value)


class QAReport(_Model):
    file: str
    kind: Kind
    profile: str
    status: Literal["pass", "warn", "fail"]
    ok: bool = Field(description="False when any check failed.")
    counts: dict[str, int]
    unchecked: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of checks that were skipped or could not check everything: a pass with an "
            "empty list is an all-clear, anything listed here was not (fully) checked."
        ),
    )
    media: dict[str, Any] | None = Field(default=None, description="ffprobe summary, if readable.")
    checks: list[QACheck]

    def check(self, check_id: str) -> QACheck:
        """The entry for ``check_id`` (KeyError if it is not in the report)."""
        for entry in self.checks:
            if entry.id == check_id:
                return entry
        raise KeyError(check_id)


def build_report(
    *, file: str, kind: Kind, profile: str, media: dict[str, Any] | None, checks: list[QACheck]
) -> QAReport:
    """Overall status is the worst check: fail over warn over pass; skipped counts as neither."""
    worst = max((_RANK[c.status] for c in checks), default=0)
    status: Literal["pass", "warn", "fail"] = ("pass", "warn", "fail")[worst]  # type: ignore[assignment]
    counts = {name: sum(1 for c in checks if c.status == name) for name in _RANK}
    unchecked = [c.id for c in checks if c.status == "skipped" or c.incomplete]
    return QAReport(
        file=file,
        kind=kind,
        profile=profile,
        status=status,
        ok=status != "fail",
        counts=counts,
        unchecked=unchecked,
        media=media,
        checks=checks,
    )


def skipped(check_id: str, reason: SkipReason, why: str, **target: Any) -> QACheck:
    return QACheck(
        id=check_id,
        status="skipped",
        skip_reason=reason,
        message=f"Skipped: {why}.",
        target=dict(target),
    )
