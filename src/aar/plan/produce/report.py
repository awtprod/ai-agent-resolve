"""What the producer did and what it warns about, JSON-ready."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aar.plan.lint import LintReport


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannerWarning(_Model):
    """Something the producer could not do as the profile asks; the plan is still lint clean."""

    code: str
    message: str
    at_s: float | None = Field(default=None, description="Recording time it concerns.")
    source: str | None = None
    hint: str = ""


class ProduceReport(_Model):
    ok: bool
    out: str | None = None
    kind: str
    segments: int
    cuts: int
    duration_s: float
    lint: LintReport
    warnings: list[PlannerWarning]
    log: dict[str, Any] = Field(
        default_factory=dict,
        description="snaps, trims, nudges, folds and the ending, in recording frames and seconds.",
    )
