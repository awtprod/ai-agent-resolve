"""The compiler's output types: a :class:`Step`, a :class:`SegmentPlacement` and a
:class:`FinishIntent`.

A :class:`Step` is one build op with its validated params and a stable name (the key an
``ops.sequence`` step uses). A :class:`SegmentPlacement` records where phase A puts each plan
segment on the timeline, so the dry-run can print per-track ranges and the property test can
check that every segment is appended exactly once, in record order, in batches within the cap.
A :class:`FinishIntent` is one phase-B action, renderable in the dry-run whether or not an
item id has been resolved from a phase-A result.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Phase = Literal["A", "B"]


class Step(BaseModel):
    """One compiled build op: its params already validate against the op's own model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: Phase
    op: str
    name: str
    """Stable step name; unique within a phase, used as the ``ops.sequence`` idempotency key."""
    params: dict[str, Any]
    """The op params, JSON-dumped from the op's validated model."""
    summary: str
    """A one-line human rendering of the key params, for the dry-run."""


class SegmentPlacement(BaseModel):
    """Where phase A appends one plan segment: its video track and record range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment: int
    track: int
    record_in: int
    record_out: int
    frames: int
    source: str
    append_step: str
    """Name of the ``timeline.append_ranges`` step whose batch carries this segment."""


class FinishIntent(BaseModel):
    """One phase-B action, for the dry-run. ``resolved`` says an item id was found."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    op: str
    target: str
    """Human description of the item(s) this addresses, e.g. 'segment 3 (video track 1 @120)'."""
    summary: str
    resolved: bool
