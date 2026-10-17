"""EditPlan: the typed, JSON-serialisable description of one Resolve build.

Timebase: every position and length is an integer number of frames at the plan's frame rate
(``fps``, a rational such as 24000/1001). Every range is half-open, ``[in, out)``: the ``out``
frame is not part of the range.

Two clocks appear in a plan:

* **source time**: frames inside one media file (``Segment.src_in`` / ``src_out``).
* **recording time**: frames since the start of one continuous recording. A recording can be
  captured by several cameras, each writing its own source file. ``Source.recording_origin`` is
  the recording time of that file's frame 0, so ``recording time = origin + source time``.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PLAN_VERSION = 1

ShotSize = Literal["close", "medium", "wide", "unknown"]
TargetKind = Literal["longform", "short"]
VideoTransitionType = Literal["cut", "dissolve", "dip", "wipe", "push", "other"]
TitleKind = Literal["title", "lower-third"]


class _Model(BaseModel):
    """Base for all plan models: unknown keys are errors, so agent typos surface early."""

    model_config = ConfigDict(extra="forbid")


class Fps(_Model):
    """A frame rate as a rational, e.g. ``{"num": 24000, "den": 1001}``.

    A string such as ``"24000/1001"`` or ``"25"`` is also accepted on input.
    """

    num: int = Field(gt=0, description="Numerator, e.g. 24000.")
    den: int = Field(default=1, gt=0, description="Denominator, e.g. 1001.")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: object) -> object:
        if isinstance(value, str):
            match = re.fullmatch(r"\s*(\d+)\s*(?:/\s*(\d+)\s*)?", value)
            if match is None:
                raise ValueError(f"fps {value!r} is not 'N' or 'N/D'")
            return {"num": int(match.group(1)), "den": int(match.group(2) or 1)}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"num": value, "den": 1}
        return value

    @property
    def nominal(self) -> int:
        """Whole frames per second used for timecode, e.g. 24 for 24000/1001."""
        return max(1, round(self.num / self.den))

    def seconds(self, frames: int) -> float:
        """Convert a frame count to seconds."""
        return frames * self.den / self.num

    def frames(self, seconds: float) -> int:
        """Convert seconds to the nearest frame count."""
        return round(seconds * self.num / self.den)

    def timecode(self, frames: int) -> str:
        """Format frames as non-drop ``HH:MM:SS:FF`` at the nominal rate."""
        sign = "-" if frames < 0 else ""
        total = abs(frames)
        base = self.nominal
        ff = total % base
        secs = total // base
        return f"{sign}{secs // 3600:02d}:{secs // 60 % 60:02d}:{secs % 60:02d}:{ff:02d}"


class Resolution(_Model):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Source(_Model):
    """One media file the plan cuts from."""

    id: str = Field(min_length=1, description="Unique id that segments refer to.")
    media: str = Field(min_length=1, description="Media reference: path, URI or media-pool name.")
    camera: str | None = Field(default=None, description="Camera / angle label, e.g. 'A'.")
    shot_size: ShotSize = "unknown"
    recording_id: str | None = Field(
        default=None,
        description="Which continuous recording this file belongs to. "
        "Files without one are treated as a recording of their own.",
    )
    recording_origin: int = Field(
        default=0,
        description="Recording time (frames) of this file's frame 0.",
    )
    duration: int | None = Field(
        default=None, gt=0, description="Media duration in frames, if known."
    )


class ReframeKey(_Model):
    """One reframe keyframe. Values are linearly interpolated between keyframes."""

    time: int = Field(ge=0, description="Frames from the segment's start.")
    x: float = Field(default=0.5, ge=0, le=1, description="Frame centre x, 0-1 of frame width.")
    y: float = Field(default=0.5, ge=0, le=1, description="Frame centre y, 0-1 of frame height.")
    zoom: float = Field(default=1.0, gt=0, description="Zoom factor, 1.0 = full frame.")


class Segment(_Model):
    """A range of one source placed on a video track. List order is record order."""

    track: int = Field(default=1, ge=1, description="Video track number.")
    source: str = Field(description="Id of a source in the plan.")
    src_in: int = Field(ge=0, description="First source frame (inclusive).")
    src_out: int = Field(ge=0, description="End source frame (exclusive).")
    record_in: int | None = Field(
        default=None,
        ge=0,
        description="Record position in frames. Omit to place it right after the previous "
        "segment on the same track.",
    )
    reframe: list[ReframeKey] = Field(default_factory=list)
    gain_db: float = Field(default=0.0, description="Audio gain in dB.")
    label: str | None = None
    notes: str | None = None

    @field_validator("reframe")
    @classmethod
    def _keys_in_time_order(cls, keys: list[ReframeKey]) -> list[ReframeKey]:
        times = [key.time for key in keys]
        if times != sorted(times):
            raise ValueError("reframe keyframes must be in non-decreasing time order")
        return keys

    @property
    def length(self) -> int:
        """Frame count of the range (zero or negative means the range is invalid)."""
        return self.src_out - self.src_in


class AudioTransition(_Model):
    type: str = Field(default="Cross Fade 0 dB", description="Resolve audio transition name.")
    frames: int = Field(ge=0, description="Length of the transition in frames.")


class Transition(_Model):
    """The join after segment ``after`` and the next segment on the same track."""

    after: int = Field(ge=0, description="Index of the outgoing segment.")
    video: VideoTransitionType = "cut"
    video_frames: int = Field(default=0, ge=0)
    audio: AudioTransition | None = None


class Title(_Model):
    kind: TitleKind = "title"
    lines: list[str] = Field(min_length=1)
    record_in: int = Field(ge=0)
    record_out: int = Field(ge=0, description="End frame (exclusive).")
    template: str | None = Field(default=None, description="Template / preset name.")


class CaptionStyle(_Model):
    font: str | None = None
    size: int | None = Field(default=None, gt=0)
    color: str | None = Field(default=None, description="Base text colour, '#RRGGBB' or a name.")
    active_word_color: str | None = Field(
        default=None, description="Colour of the word being spoken, '#RRGGBB' or a name."
    )


class Caption(_Model):
    lines: list[str] = Field(min_length=1, description="Text lines shown together.")
    record_in: int = Field(ge=0)
    record_out: int = Field(ge=0, description="End frame (exclusive).")
    style: CaptionStyle | None = Field(
        default=None, description="Overrides the plan's caption_style."
    )


class Marker(_Model):
    record: int = Field(ge=0)
    name: str = ""
    color: str | None = None
    note: str | None = None
    duration: int = Field(default=1, ge=1)


class EditPlan(_Model):
    """Everything needed to build one timeline, checked against a rule profile first."""

    version: Literal[1] = PLAN_VERSION
    project: str = Field(min_length=1, description="Resolve project name.")
    timeline: str = Field(min_length=1, description="Target timeline name.")
    kind: TargetKind
    fps: Fps
    resolution: Resolution
    profile: str = Field(min_length=1, description="Rule profile name, e.g. 'ccn'.")
    sources: list[Source]
    segments: list[Segment]
    transitions: list[Transition] = Field(default_factory=list)
    titles: list[Title] = Field(default_factory=list)
    captions: list[Caption] = Field(default_factory=list)
    caption_style: CaptionStyle | None = None
    markers: list[Marker] = Field(default_factory=list)

    def source_map(self) -> dict[str, Source]:
        """Sources by id. If an id is duplicated the first one wins (lint reports the clash)."""
        found: dict[str, Source] = {}
        for source in self.sources:
            found.setdefault(source.id, source)
        return found

    def recording_key(self, source: Source) -> str:
        """The recording a source belongs to; a source without one is its own recording."""
        return source.recording_id if source.recording_id is not None else f"source:{source.id}"

    def record_ranges(self) -> list[tuple[int, int]]:
        """Half-open record range of each segment, in frames.

        A segment without ``record_in`` starts where the previous segment on its track ended.
        """
        cursor: dict[int, int] = {}
        ranges: list[tuple[int, int]] = []
        for seg in self.segments:
            start = seg.record_in if seg.record_in is not None else cursor.get(seg.track, 0)
            end = start + max(seg.length, 0)
            cursor[seg.track] = end
            ranges.append((start, end))
        return ranges

    def total_frames(self) -> int:
        """Record end of the latest segment on any track."""
        return max((end for _, end in self.record_ranges()), default=0)

    def joins(self) -> list[tuple[int, int]]:
        """Pairs ``(outgoing, incoming)`` of segments that are neighbours on the same track."""
        last: dict[int, int] = {}
        pairs: list[tuple[int, int]] = []
        for index, seg in enumerate(self.segments):
            if seg.track in last:
                pairs.append((last[seg.track], index))
            last[seg.track] = index
        return pairs
