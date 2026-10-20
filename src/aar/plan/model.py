"""EditPlan: the typed, JSON-serialisable description of one Resolve build.

Timebase: every position and length is an integer number of frames at the plan's frame rate
(``fps``, a rational such as 24000/1001). Every range is half-open, ``[in, out)``: the ``out``
frame is not part of the range.

Two clocks appear in a plan:

* **source time**: frames inside one media file (``Segment.src_in`` / ``src_out``).
* **recording time**: frames since the start of one continuous recording. A recording can be
  captured by several cameras, each writing its own source file. ``Source.recording_origin`` is
  the recording time of that file's frame 0, so ``recording time = origin + source time``.

Versions: a plan says which format it uses in ``version``. Version 2 (the default for a plan
that does not say) adds the build-from-template fields: ``template``, ``head``, ``outro``,
``audio_tracks``, ``end_fade_frames``, ``Source.color`` and ``Segment.transform``. Everything
else is the same in both, so every valid version 1 plan is also valid as version 2. A plan that
says ``"version": 1`` may not use a version 2 field: giving one a value is a validation error
naming the field. Null (or an empty ``audio_tracks``) is not a use, so a full dump of a version
1 plan, which writes every field, still reads back as version 1.

The outro is not a content segment: it sits outside ``segments``, so the content rules
(chronological, dialogue splice, end on the laugh, cut density) see the last content segment
as the end. It does count towards the timeline's total length.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, InitErrorDetails, PydanticCustomError

PLAN_VERSION = 2


def _whole_number_version(value: object) -> object:
    """Refuse ``true`` and ``1.0``: pydantic would coerce them to 1, but the version is a label.

    ``Literal`` cannot be made strict, so the check is a before-validator. It raises the same
    ``literal_error`` that 0, 3 and ``"2"`` get, so every bad version reads alike, once, at the
    ``version`` loc.
    """
    if isinstance(value, (bool, float)):
        raise PydanticCustomError(
            "literal_error",
            "Input should be 1 or 2 (a whole number, not a boolean or a decimal)",
            {"expected": "1 or 2"},
        )
    return value


PlanVersion = Annotated[Literal[1, 2], BeforeValidator(_whole_number_version)]

#: Fields only a version 2 plan may use, per model.
V2_PLAN_FIELDS = ("template", "head", "outro", "audio_tracks", "end_fade_frames")
V2_SOURCE_FIELDS = ("color",)
V2_SEGMENT_FIELDS = ("transform",)

ShotSize = Literal["close", "medium", "wide", "unknown"]
TargetKind = Literal["longform", "short"]
VideoTransitionType = Literal["cut", "dissolve", "dip", "wipe", "push", "other"]
TitleKind = Literal["title", "lower-third"]


class _Model(BaseModel):
    """Base for all plan models: unknown keys are errors, so agent typos surface early."""

    model_config = ConfigDict(extra="forbid")


#: Unicode categories that draw nothing: format characters (zero-width space, joiners, BOM,
#: word joiner) and non-spacing combining marks (an accent with no letter under it).
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Mn"})


def _not_blank(value: str) -> str:
    """Refuse a text that shows nothing: empty, or only whitespace, format or combining marks.

    A single visible character is enough, so ``"!"``, ``"a"`` and an emoji joined by a
    zero-width joiner all pass, and so does an accented letter in either form.
    """
    if all(char.isspace() or unicodedata.category(char) in _INVISIBLE_CATEGORIES for char in value):
        raise ValueError(
            "must not be empty or blank (whitespace, zero-width and combining marks show nothing)"
        )
    return value


#: A name that must say something: empty and whitespace-only strings are rejected.
Name = Annotated[str, AfterValidator(_not_blank)]
Finite = Annotated[float, Field(allow_inf_nan=False)]

#: JSON Schema dialect the plan schema is written in (what pydantic emits).
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: JSON Schema pattern for the text form of a frame rate: 'N' or 'N/D', both above zero.
FPS_TEXT_PATTERN = r"^\s*0*[1-9][0-9]*\s*(/\s*0*[1-9][0-9]*\s*)?$"


class Fps(_Model):
    """A frame rate as a rational, e.g. ``{"num": 24000, "den": 1001}``.

    A string such as ``"24000/1001"`` or ``"25"`` (ASCII digits only) is also accepted on input,
    and so is a whole number such as ``25``. The input JSON Schema lists all three forms.
    """

    num: int = Field(gt=0, description="Numerator, e.g. 24000.")
    den: int = Field(default=1, gt=0, description="Denominator, e.g. 1001.")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """The input schema admits the 'N' / 'N/D' string and the whole-number shorthand too."""
        schema = handler.resolve_ref_schema(handler(core_schema))
        if handler.mode != "validation":
            return schema
        record = dict(schema)
        schema.clear()
        schema.update(
            title=record.pop("title"),
            description=record.pop("description"),
            anyOf=[
                record,
                {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "description": "Shorthand for a whole rate, e.g. 25.",
                },
                {
                    "type": "string",
                    "pattern": FPS_TEXT_PATTERN,
                    "description": "Shorthand 'N' or 'N/D' (both positive), e.g. '24000/1001'.",
                },
            ],
        )
        return schema

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: object) -> object:
        if isinstance(value, str):
            # ASCII only, as the schema's pattern is: \d would also read Arabic-Indic digits.
            match = (
                re.fullmatch(r"\s*(\d+)\s*(?:/\s*(\d+)\s*)?", value) if value.isascii() else None
            )
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


_RGB = ("red", "green", "blue")


def _rgb(name: str, values: list[float], *, minimum: float | None, strict: bool) -> list[float]:
    """Check one CDL triple: exactly 3 finite numbers, each at or above (or over) ``minimum``."""
    if len(values) != 3:
        raise ValueError(f"{name} must be 3 numbers (red, green, blue), not {len(values)}")
    for channel, value in zip(_RGB, values, strict=True):
        if not math.isfinite(value):
            raise ValueError(f"{name} {channel} must be finite, not {value!r}")
        if minimum is not None and (value <= minimum if strict else value < minimum):
            bound = f"more than {minimum:g}" if strict else f"at least {minimum:g}"
            raise ValueError(f"{name} {channel} must be {bound}, not {value!r}")
    return values


class Cdl(_Model):
    """An ASC CDL: per channel ``out = (in * slope + offset) ** power``, then saturation.

    Each triple is red, green, blue. Omitted parts are the identity.
    """

    slope: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        description="Red, green, blue; each at least 0.",
        json_schema_extra={"minItems": 3, "maxItems": 3},
    )
    offset: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Red, green, blue.",
        json_schema_extra={"minItems": 3, "maxItems": 3},
    )
    power: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        description="Red, green, blue; each more than 0.",
        json_schema_extra={"minItems": 3, "maxItems": 3},
    )
    saturation: Finite = Field(default=1.0, ge=0, description="At least 0; 1 leaves it as is.")

    @field_validator("slope")
    @classmethod
    def _slope(cls, values: list[float]) -> list[float]:
        return _rgb("slope", values, minimum=0.0, strict=False)

    @field_validator("offset")
    @classmethod
    def _offset(cls, values: list[float]) -> list[float]:
        return _rgb("offset", values, minimum=None, strict=False)

    @field_validator("power")
    @classmethod
    def _power(cls, values: list[float]) -> list[float]:
        return _rgb("power", values, minimum=0.0, strict=True)


class SourceColor(_Model):
    """Version 2. How the build colours the clips of one source. Every part is optional."""

    input_color_space: Name | None = Field(
        default=None,
        description="Resolve Input Color Space name, e.g. 'Rec.709 Gamma 2.4'. "
        "Never set it on BRAW media (house safety rule).",
    )
    cdl: Cdl | None = Field(default=None, description="A CDL applied on the clip's grade.")
    lut: Name | None = Field(default=None, description="LUT name as Resolve lists it.")


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
    color: SourceColor | None = Field(
        default=None,
        description="Version 2. Colour settings for every clip placed from this source.",
    )


class ReframeKey(_Model):
    """One reframe keyframe. Values are linearly interpolated between keyframes."""

    time: int = Field(ge=0, description="Frames from the segment's start.")
    x: float = Field(default=0.5, ge=0, le=1, description="Frame centre x, 0-1 of frame width.")
    y: float = Field(default=0.5, ge=0, le=1, description="Frame centre y, 0-1 of frame height.")
    zoom: float = Field(default=1.0, gt=0, description="Zoom factor, 1.0 = full frame.")


class Transform(_Model):
    """Version 2. A static transform for a whole segment, as in Resolve's Inspector.

    It is separate from the keyframed ``reframe``: the reframe moves within the transformed
    picture, so the effective zoom is ``transform.zoom`` times the reframe zoom.
    """

    zoom: Finite = Field(default=1.0, gt=0, description="Zoom factor, 1.0 = none (ZoomX/ZoomY).")
    pan: Finite = Field(
        default=0.0, description="Horizontal offset in pixels at the timeline resolution (Pan)."
    )
    tilt: Finite = Field(
        default=0.0, description="Vertical offset in pixels at the timeline resolution (Tilt)."
    )
    rotation_deg: Finite = Field(
        default=0.0, ge=-360, le=360, description="Rotation in degrees (RotationAngle)."
    )
    flip_h: bool = Field(default=False, description="Mirror left to right (FlipX).")
    flip_v: bool = Field(default=False, description="Mirror top to bottom (FlipY).")


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
    transform: Transform | None = Field(
        default=None, description="Version 2. A static transform for the whole segment."
    )
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


def _check_video_length(video: VideoTransitionType, frames: int) -> None:
    if video == "cut" and frames != 0:
        raise ValueError(
            f"a cut has no length, but video_frames is {frames}: "
            "set video_frames to 0 or choose a transition type"
        )
    if video != "cut" and frames == 0:
        raise ValueError(f"a {video} needs video_frames above 0")


class Head(_Model):
    """Version 2. How the first segment comes in from black.

    ``audio_fade_frames`` is deliberately uncapped: neither this model nor any lint rule limits
    the fade-in (the ``short-end-fade`` rule limits only a short's end fade), so a long fade-in
    is the author's call and is not an error or a finding.
    """

    video: VideoTransitionType = Field(
        default="cut", description="Video transition in from black; 'cut' starts hard."
    )
    video_frames: int = Field(default=0, ge=0, description="Length of the video transition.")
    audio_fade_frames: int = Field(
        default=0, ge=0, description="Audio fade-in from silence, in frames."
    )

    @model_validator(mode="after")
    def _length_matches_type(self) -> Head:
        _check_video_length(self.video, self.video_frames)
        return self


class OutroTransition(_Model):
    """Version 2. The transition from the last content segment into the outro."""

    video: VideoTransitionType = "cut"
    video_frames: int = Field(default=0, ge=0)
    audio: AudioTransition | None = None

    @model_validator(mode="after")
    def _length_matches_type(self) -> OutroTransition:
        _check_video_length(self.video, self.video_frames)
        return self


class Sting(_Model):
    """Version 2. Text shown over the start of the outro."""

    lines: list[Name] = Field(min_length=1, description="Text lines shown together.")
    start: int = Field(default=0, ge=0, description="Frames from the outro's first frame.")
    frames: int = Field(gt=0, description="How long the text shows, in frames.")
    template: Name | None = Field(default=None, description="Title template / preset name.")


class Outro(_Model):
    """Version 2. The outro clip appended after the last content segment.

    It starts where the content ends (the record end of the latest segment on any track).
    It is not a content segment, so content rules do not judge it.
    """

    source: Name = Field(description="Id of a source in the plan.")
    src_in: int = Field(default=0, ge=0, description="First source frame (inclusive).")
    src_out: int | None = Field(
        default=None,
        ge=0,
        description="End source frame (exclusive). Omit to play to the end of the source, "
        "whose duration must then be known.",
    )
    track: int = Field(default=1, ge=1, description="Video track number.")
    audio_track: Name | None = Field(
        default=None,
        description="Name of the audio_tracks entry that carries the outro's sound "
        "(e.g. the bus track).",
    )
    transition: OutroTransition = Field(
        default_factory=OutroTransition, description="How the content goes into the outro."
    )
    sting: Sting | None = Field(default=None, description="Text over the start of the outro.")


class AudioTrack(_Model):
    """Version 2. Settings for one audio track. List order is track order (A1 first)."""

    name: Name = Field(description="Track name; unique in the plan.")
    gain_db: Finite = Field(default=0.0, description="Track gain in dB.")
    mute: bool = Field(default=False, description="Muted tracks do not reach the mix.")
    bus: Name | None = Field(default=None, description="Name of the bus this track feeds.")


class Title(_Model):
    kind: TitleKind = "title"
    lines: list[Name] = Field(min_length=1)
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
    lines: list[Name] = Field(min_length=1, description="Text lines shown together.")
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

    model_config = ConfigDict(extra="forbid", json_schema_extra={"$schema": JSON_SCHEMA_DIALECT})

    version: PlanVersion = Field(
        default=PLAN_VERSION,
        description="Plan format version. 2 (the default) adds template, head, outro, "
        "audio_tracks, end_fade_frames, Source.color and Segment.transform; a plan that says "
        "1 may not use them.",
    )
    project: str = Field(min_length=1, description="Resolve project name.")
    timeline: str = Field(min_length=1, description="Target timeline name.")
    template: Name | None = Field(
        default=None,
        description="Version 2. Timeline the build starts from. The build branches it with "
        "timeline.branch (never timeline.duplicate), so the copy is a registered agent branch, "
        "and works in the branch.",
    )
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
    head: Head | None = Field(
        default=None, description="Version 2. How the first segment comes in from black."
    )
    outro: Outro | None = Field(
        default=None, description="Version 2. The outro appended after the last content segment."
    )
    audio_tracks: list[AudioTrack] = Field(
        default_factory=list,
        description="Version 2. Audio track settings, in track order (A1 first).",
    )
    end_fade_frames: int | None = Field(
        default=None,
        ge=0,
        description="Version 2. Audio fade-out at the very end of the timeline, in frames. "
        "On a short this is a de-click, never a fade to silence.",
    )

    @field_validator("audio_tracks")
    @classmethod
    def _track_names_unique(cls, tracks: list[AudioTrack]) -> list[AudioTrack]:
        first: dict[str, int] = {}
        for index, track in enumerate(tracks):
            if track.name in first:
                raise ValueError(
                    f"audio track name {track.name!r} is used by entries {first[track.name]} "
                    f"and {index}; names must be unique"
                )
            first[track.name] = index
        return tracks

    @model_validator(mode="after")
    def _version_and_references(self) -> EditPlan:
        problems = list(self._v2_fields_in_v1()) or list(self._outro_problems())
        if problems:
            raise ValidationError.from_exception_data(type(self).__name__, problems)
        return self

    def _v2_fields_in_v1(self) -> Iterable[InitErrorDetails]:
        """Version 2 fields in a plan that says version 1, one error per field."""
        if self.version != 1:
            return
        used: list[tuple[tuple[str | int, ...], object]] = [
            ((name,), getattr(self, name)) for name in V2_PLAN_FIELDS
        ]
        for kind, items, names in (
            ("sources", self.sources, V2_SOURCE_FIELDS),
            ("segments", self.segments, V2_SEGMENT_FIELDS),
        ):
            for index, item in enumerate(items):
                used.extend(((kind, index, name), getattr(item, name)) for name in names)
        for loc, value in used:
            if value is None or value == []:
                continue  # null or empty: the field is not used
            field = _dotted(loc)
            yield InitErrorDetails(
                type=PydanticCustomError(
                    "plan_version",
                    "{field} is a version 2 field, but this plan says version 1: "
                    'set "version": 2 or remove {field}',
                    {"field": field},
                ),
                loc=loc,
                input=_plain(value),
            )

    def _outro_problems(self) -> Iterable[InitErrorDetails]:
        """Cross-references of the outro: its source, range, audio track and sting."""
        outro = self.outro
        if outro is None:
            return

        def problem(loc: tuple[str, ...], message: str, value: object) -> InitErrorDetails:
            return InitErrorDetails(
                type=PydanticCustomError("plan_reference", message),
                loc=("outro", *loc),
                input=value,
            )

        source = self.source_map().get(outro.source)
        if source is None:
            known = ", ".join(repr(s.id) for s in self.sources) or "none"
            yield problem(
                ("source",),
                f"outro source {outro.source!r} is not one of the plan's sources ({known})",
                outro.source,
            )
        elif outro.src_out is None and source.duration is None:
            yield problem(
                ("src_out",),
                f"outro src_out is omitted, so the outro plays to the end of source "
                f"{source.id!r}, but that source has no duration: set src_out or the "
                "source's duration",
                None,
            )
        else:
            end = outro.src_out if outro.src_out is not None else source.duration
            assert end is not None
            if end <= outro.src_in:
                yield problem(
                    ("src_out",),
                    f"outro ends at source frame {end}, not after its start {outro.src_in}; "
                    "ranges are [in, out)",
                    end,
                )
            elif source.duration is not None and end > source.duration:
                yield problem(
                    ("src_out",),
                    f"outro ends at source frame {end}, past the end of source "
                    f"{source.id!r} ({source.duration} frames)",
                    end,
                )
            elif outro.sting is not None and outro.sting.start + outro.sting.frames > (
                length := end - outro.src_in
            ):
                yield problem(
                    ("sting",),
                    f"the sting runs to frame {outro.sting.start + outro.sting.frames} of the "
                    f"outro, past its end ({length} frames)",
                    outro.sting.model_dump(),
                )
        names = [track.name for track in self.audio_tracks]
        if outro.audio_track is not None and outro.audio_track not in names:
            known = ", ".join(repr(n) for n in names) or "none"
            yield problem(
                ("audio_track",),
                f"outro audio_track {outro.audio_track!r} is not one of the plan's "
                f"audio_tracks ({known})",
                outro.audio_track,
            )

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

    def content_frames(self) -> int:
        """Record end of the latest content segment on any track."""
        return max((end for _, end in self.record_ranges()), default=0)

    def outro_range(self) -> tuple[int, int] | None:
        """Half-open record range of the outro, which starts where the content ends."""
        if self.outro is None:
            return None
        source = self.source_map()[self.outro.source]
        end = self.outro.src_out if self.outro.src_out is not None else source.duration
        assert end is not None, "a validated plan knows the outro's length"
        start = self.content_frames()
        return start, start + max(end - self.outro.src_in, 0)

    def total_frames(self) -> int:
        """Record end of the timeline: the outro's end if there is one, else the content's."""
        outro = self.outro_range()
        return self.content_frames() if outro is None else outro[1]

    def joins(self) -> list[tuple[int, int]]:
        """Pairs ``(outgoing, incoming)`` of segments that are neighbours on the same track."""
        last: dict[int, int] = {}
        pairs: list[tuple[int, int]] = []
        for index, seg in enumerate(self.segments):
            if seg.track in last:
                pairs.append((last[seg.track], index))
            last[seg.track] = index
        return pairs


def _plain(value: object) -> object:
    """A field value with models dumped to dicts, for an error's ``input``."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _dotted(loc: tuple[str | int, ...]) -> str:
    """``("sources", 0, "color")`` as ``sources[0].color``."""
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else part)
    return text
