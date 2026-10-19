"""Write ops: duplicate a timeline, append source ranges, set gain and fades, copy grades,
set a clip's input colour space, add markers.

Each op checks every Resolve return value, reads what it wrote back, and reports the measured
values. Each is idempotent: a re-run finds its earlier work (``present``) instead of
repeating it. None of these ops calls a project-level colour setter.

Known, intentional, fail-closed behaviour (not bugs):

- ``timeline.duplicate`` interrupted after ``DuplicateTimeline`` but before its custom
  resolution and colour settings were applied leaves a timeline that does not match; a rerun
  with ``if_exists="reuse"`` reports ``reuse_mismatch`` and changes nothing, because the op
  cannot tell its own half-made copy from a human's timeline. So every failure after the
  copy was made carries the copy's unique id on the error, as
  ``detail["created_timeline_id"]`` (:data:`CREATED_TIMELINE_ID`): a caller that wants to
  undo the copy (``timeline.branch`` does) deletes exactly that id, never a timeline found by
  name. Cleaning up after a plain ``timeline.duplicate`` belongs to the S7 job runner.
- ``clip.set_input_color_space`` accepts only colour-space names it can see in use: the
  project's colour settings, clips in the listed items' bins, or ``OBSERVED_COLOR_SPACES``.
  There is no override; anything else is ``color_space_unknown``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from aar.node.ops import raw
from aar.node.ops.framework import (
    ItemRef,
    MediaRef,
    Note,
    OpEnv,
    OpError,
    OpModel,
    TimelineIdentity,
    TimelineRef,
    TrackType,
    as_op_error,
    op,
)
from aar.node.ops.timebase import converted_length

MarkerColor = Literal[
    "Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple", "Fuchsia",
    "Rose", "Lavender", "Sky", "Mint", "Lemon", "Sand", "Cocoa", "Cream",
]  # fmt: skip

# TimelineSettings keys that describe a timeline's colour pipeline, colour science first.
TIMELINE_COLOUR_KEYS: tuple[str, ...] = (
    "colorScienceMode",
    "isAutoColorManage",
    "rcmPresetMode",
    "colorSpaceTimeline",
    "colorSpaceOutput",
    "inputDRT",
    "outputDRT",
    "useInverseDRT",
    "useColorSpaceAwareGradingTools",
    "acesVersion",
    "separateColorSpaceAndGamma",
    "colorSpaceTimelineGamma",
    "colorSpaceOutputGamma",
    "colorAcesGamutCompressType",
    "colorAcesODT",
    "colorAcesMidGray",
    "colorAcesNodeLUTProcessingSpace",
    "use203NitsReference",
    "colorSpaceOutputGamutLimit",
    "colorSpaceOutputToneMapping",
    "colorSpaceOutputGamutMapping",
    "colorSpaceOutputGamutSaturationKnee",
    "colorSpaceOutputGamutSaturationMax",
    "colorSpaceOutputToneLuminanceMax",
    "outputDRTSatRolloffStart",
    "outputDRTSatRolloffLimit",
    "inputDRTSatRolloffStart",
    "inputDRTSatRolloffLimit",
    "useCATransform",
    "imageResizingGamma",
    "graphicsWhiteLevel",
    "timelineWorkingLuminance",
    "timelineWorkingLuminanceMode",
)


def _setting(value: Any) -> str | None:
    return None if value is None else str(value)


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _media_id(item: Any) -> str | None:
    mpi = item.GetMediaPoolItem()
    return str(mpi.GetUniqueId()) if mpi is not None else None


# -- timeline.duplicate ----------------------------------------------------------------------


class Resolution(OpModel):
    width: int = Field(strict=True, gt=0, le=16384)
    height: int = Field(strict=True, gt=0, le=16384)


class DuplicateParams(OpModel):
    source: TimelineRef | None = Field(default=None, description="Default: current timeline.")
    name: str = Field(min_length=1)
    if_exists: Literal["error", "reuse"] = Field(
        default="error",
        description="'reuse' picks up a timeline of that name left by an earlier attempt. It "
        "never changes it: it must already match the source's colour settings and the "
        "resolution asked for (else the op is refused), and it is never the source itself.",
    )
    resolution: Resolution | None = Field(
        default=None, description="Give the copy its own resolution (e.g. a vertical cut)."
    )


class SettingCheck(OpModel):
    key: str
    expected: str | None
    actual: str | None


class DuplicateResult(OpModel):
    source: TimelineIdentity
    timeline: TimelineIdentity
    created: bool
    timeline_count: int
    resolution: Resolution | None
    colour: list[SettingCheck]
    notes: list[Note]


def _set_resolution(
    source: Any, timeline: Any, res: Resolution
) -> tuple[list[SettingCheck], list[Note]]:
    trap = "custom-res-resets-colour-science"
    reference = source.GetSettings() or {}
    keep = {
        k: str(reference[k]) for k in TIMELINE_COLOUR_KEYS if reference.get(k) not in (None, "")
    }
    now = timeline.GetSettings() or {}
    wanted = (str(res.width), str(res.height))
    have = (
        _setting(now.get("timelineResolutionWidth")),
        _setting(now.get("timelineResolutionHeight")),
    )
    if now.get("useCustomSettings") != "1" or have != wanted:
        ok = timeline.SetSettings(
            {
                "useCustomSettings": "1",
                "timelineResolutionWidth": wanted[0],
                "timelineResolutionHeight": wanted[1],
            }
        )
        if not ok:
            raise OpError(
                "resolution_set_failed",
                f"Timeline.SetSettings(resolution {wanted[0]}x{wanted[1]}) returned False",
                "check the resolution is one Resolve accepts for a timeline",
            )
    after = timeline.GetSettings() or {}
    reset = {k: v for k, v in keep.items() if _setting(after.get(k)) != v}
    notes: list[Note] = []
    if reset:
        notes.append(
            Note(
                code="colour_settings_restored",
                message=f"the custom resolution reset {sorted(reset)}; set back from the source "
                "timeline",
                trap_id=trap,
            )
        )
        # Colour science first: the colour spaces and DRTs available depend on it.
        chunks = (
            [{"colorScienceMode": reset["colorScienceMode"]}] if "colorScienceMode" in reset else []
        )
        rest = {k: v for k, v in reset.items() if k != "colorScienceMode"}
        if rest:
            chunks.append(rest)
        for chunk in chunks:
            if not timeline.SetSettings(chunk):
                notes.append(
                    Note(
                        code="colour_settings_partial",
                        message=f"Timeline.SetSettings({sorted(chunk)}) returned False; checking "
                        "what was applied",
                        trap_id=trap,
                    )
                )
    final = timeline.GetSettings() or {}
    checks = [
        SettingCheck(key=k, expected=v, actual=_setting(final.get(k))) for k, v in keep.items()
    ]
    bad = [c for c in checks if c.actual != c.expected]
    if bad:
        raise OpError(
            "colour_settings_not_restored",
            f"after the custom resolution these colour settings differ from the source "
            f"timeline: {[c.key for c in bad]}",
            "set them by hand in the timeline settings, or drop the custom resolution",
            trap,
            mismatches=[c.model_dump() for c in bad],
        )
    return checks, notes


CREATED_TIMELINE_ID = "created_timeline_id"
"""``OpError`` detail key: the unique id of the copy ``timeline.duplicate`` made before it
failed. Set on every failure after ``DuplicateTimeline`` returned a timeline that was not in
the project before the call and is in it now, so it is provably this call's copy; absent
when no copy was made, or when what Resolve returned cannot be shown to be a new copy (then
nothing may be undone)."""


def _with_created(exc: Exception, created_id: str) -> OpError:
    """``exc`` as an ``OpError`` naming the copy this call made (:data:`CREATED_TIMELINE_ID`)."""
    error = as_op_error(exc)
    extra = {k: v for k, v in error.detail.items() if k not in ("fix_hint", "trap_id")}
    extra[CREATED_TIMELINE_ID] = created_id
    return OpError(error.code, error.message, error.fix_hint, error.trap_id, **extra)


@op(
    "timeline.duplicate",
    params=DuplicateParams,
    result=DuplicateResult,
    mutates=True,
    creates_timeline=True,
    traps=("custom-res-resets-colour-science",),
)
def timeline_duplicate(env: OpEnv, params: DuplicateParams) -> DuplicateResult:
    """Duplicate a timeline, optionally giving the copy its own resolution.

    Any failure after the copy is made names it on the error (:data:`CREATED_TIMELINE_ID`).
    """
    project = env.project()
    source = env.timeline(params.source)
    existing = [t for t in env.timelines() if t.GetName() == params.name]
    if existing:
        if any(t.GetUniqueId() == source.GetUniqueId() for t in existing):
            raise OpError(
                "reuse_is_source",
                f"{params.name!r} is the source timeline itself",
                "choose a new name for the copy",
            )
        if params.if_exists == "error" or len(existing) > 1:
            raise OpError(
                "timeline_exists",
                f"{len(existing)} timeline(s) named {params.name!r} already exist",
                "choose another name, or pass if_exists='reuse' to pick up an earlier attempt",
            )
        return _reuse(env, source, existing[0], params)
    before_ids = {str(t.GetUniqueId()) for t in env.timelines()}
    before = int(project.GetTimelineCount() or 0)
    made = source.DuplicateTimeline(params.name)
    if made is None:
        raise OpError(
            "duplicate_failed",
            f"DuplicateTimeline({params.name!r}) returned None",
            "check the name is unique and the source timeline is not locked",
        )
    uid = made.GetUniqueId()
    matches = [t for t in env.timelines() if t.GetUniqueId() == uid]
    # The copy is this call's own only if Resolve handed back a timeline that was not in the
    # project before the call and is in it now; anything else (the source itself, or a
    # timeline the project does not hold) is never named as created, so never undone.
    created_id = str(uid) if uid and str(uid) not in before_ids and matches else None
    try:
        return _finish_duplicate(env, project, source, params, matches, before)
    except Exception as exc:
        if created_id is None:
            raise
        raise _with_created(exc, created_id) from exc


def _finish_duplicate(
    env: OpEnv,
    project: Any,
    source: Any,
    params: DuplicateParams,
    matches: list[Any],
    before: int,
) -> DuplicateResult:
    """Verify the copy ``DuplicateTimeline`` returned, then give it its resolution."""
    count = int(project.GetTimelineCount() or 0)
    if len(matches) != 1 or matches[0].GetName() != params.name or count != before + 1:
        raise OpError(
            "duplicate_unverified",
            f"after DuplicateTimeline the project has {count} timelines (was {before}) and "
            f"{len(matches)} with the new id",
            "inspect the project with timeline.list",
        )
    dup, created = matches[0], True
    colour: list[SettingCheck] = []
    notes: list[Note] = []
    measured: Resolution | None = None
    if params.resolution is not None:
        colour, notes = _set_resolution(source, dup, params.resolution)
        settings = dup.GetSettings() or {}
        width = _int(settings.get("timelineResolutionWidth"))
        height = _int(settings.get("timelineResolutionHeight"))
        if (width, height) != (params.resolution.width, params.resolution.height):
            raise OpError(
                "resolution_unverified",
                f"the copy reads back as {width}x{height}",
                "set the resolution in the timeline settings by hand",
            )
        measured = Resolution(width=width, height=height)
    return DuplicateResult(
        source=env.identity(source),
        timeline=env.identity(dup),
        created=created,
        timeline_count=int(project.GetTimelineCount() or 0),
        resolution=measured,
        colour=colour,
        notes=notes,
    )


def _reuse(env: OpEnv, source: Any, dup: Any, params: DuplicateParams) -> DuplicateResult:
    """Pick up an existing timeline without changing it, if it is what the op would make."""
    # Unleased read of `dup` (its id is only known now, after submit): acceptable because it
    # only reads settings and never writes; a mismatch is refused with `dup` left unchanged.
    reference = source.GetSettings() or {}
    have = dup.GetSettings() or {}
    expected = {
        k: str(reference[k]) for k in TIMELINE_COLOUR_KEYS if reference.get(k) not in (None, "")
    }
    if params.resolution is not None:
        expected["timelineResolutionWidth"] = str(params.resolution.width)
        expected["timelineResolutionHeight"] = str(params.resolution.height)
    else:
        for key in ("timelineResolutionWidth", "timelineResolutionHeight"):
            if reference.get(key) not in (None, ""):
                expected[key] = str(reference[key])
    checks = [
        SettingCheck(key=k, expected=v, actual=_setting(have.get(k))) for k, v in expected.items()
    ]
    bad = [c for c in checks if c.actual != c.expected]
    if bad:
        raise OpError(
            "reuse_mismatch",
            f"timeline {params.name!r} exists but differs from what this op would make: "
            f"{[c.key for c in bad]}; it is left unchanged",
            "choose a new name, or delete the old attempt first",
            "custom-res-resets-colour-science" if params.resolution is not None else None,
            mismatches=[c.model_dump() for c in bad],
        )
    width = _int(have.get("timelineResolutionWidth"))
    height = _int(have.get("timelineResolutionHeight"))
    return DuplicateResult(
        source=env.identity(source),
        timeline=env.identity(dup),
        created=False,
        timeline_count=int(env.project().GetTimelineCount() or 0),
        resolution=(
            Resolution(width=width, height=height)
            if params.resolution is not None and width and height
            else None
        ),
        colour=[c for c in checks if c.key in TIMELINE_COLOUR_KEYS],
        notes=[Note(code="reused", message=f"timeline {params.name!r} already matched; unchanged")],
    )


# -- timeline.append_ranges --------------------------------------------------------------------


class AppendRange(OpModel):
    """One source range. ``src_out`` is exclusive, like ``aar.plan`` Segment ranges."""

    media: MediaRef
    src_in: int = Field(
        strict=True, ge=0, description="First source frame, in the source's frame rate."
    )
    src_out: int = Field(strict=True, ge=1, description="End source frame (exclusive).")
    track: int = Field(strict=True, default=1, ge=1)
    media_type: Literal["av", "video", "audio"] = "av"
    record_in: int | None = Field(
        strict=True,
        default=None,
        ge=0,
        description="Frames from the timeline start. Omit to follow the previous range on "
        "the same track (video for 'av'/'video', audio for 'audio').",
    )
    label: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> AppendRange:
        if self.src_out <= self.src_in:
            raise ValueError("src_out must be greater than src_in (ranges are [in, out))")
        return self


class AppendParams(OpModel):
    timeline: TimelineRef | None = Field(default=None, description="Default: current timeline.")
    ranges: list[AppendRange] = Field(min_length=1, max_length=5000)
    cursor: int = Field(
        strict=True, default=0, ge=0, description="Index of the first range to process."
    )
    max_ranges: int = Field(strict=True, default=25, ge=1, le=200, description="Ranges per call.")
    repair_short: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def _cursor_in_range(self) -> AppendParams:
        if self.cursor >= len(self.ranges):
            raise ValueError(f"cursor {self.cursor} is past the last range")
        return self


class PlacedItem(OpModel):
    item_id: str
    track_type: TrackType
    track: int
    start: int
    end: int
    duration: int
    record_in: int
    record_out: int
    source_start: int | None
    source_end: int | None


class AppendedRange(OpModel):
    index: int
    status: Literal["appended", "present", "repaired"]
    planned_record_in: int
    planned_record_out: int
    planned_frames: int
    end_frame_used: int
    video: PlacedItem | None
    audio: list[PlacedItem]
    notes: list[Note]


class TrackGap(OpModel):
    track_type: TrackType
    track: int
    start: int
    end: int
    frames: int
    record_in: int
    record_out: int


class AppendResult(OpModel):
    timeline: TimelineIdentity
    timeline_fps: float
    cursor: int
    next_cursor: int | None
    done: bool
    ranges: list[AppendedRange]
    gaps: list[TrackGap]
    """Gaps measured after this call on the tracks it touched, within the span it touched."""
    notes: list[Note]


@dataclass(frozen=True)
class _Planned:
    index: int
    rng: AppendRange
    kind: TrackType
    record_in: int
    frames: int


def _plan(
    ranges: list[AppendRange], facts: dict[str, dict[str, Any]], timeline_fps: float, upto: int
) -> list[_Planned]:
    ends: dict[tuple[str, int], int] = {}
    planned = []
    for index, rng in enumerate(ranges[:upto]):
        fps = facts[rng.media.unique_id]["fps"]
        if fps is None:
            raise OpError(
                "media_fps_unknown",
                f"range {index}: media {rng.media.unique_id!r} reports no FPS",
                "check the clip's attributes in the media pool",
            )
        kind: TrackType = "audio" if rng.media_type == "audio" else "video"
        frames = converted_length(rng.src_out - rng.src_in, fps, timeline_fps)
        record_in = rng.record_in if rng.record_in is not None else ends.get((kind, rng.track), 0)
        ends[(kind, rng.track)] = record_in + frames
        planned.append(_Planned(index, rng, kind, record_in, frames))
    return planned


def _preflight(timeline: Any, p: _Planned, facts: dict[str, Any]) -> None:
    rng = p.rng
    if facts["is_still"]:
        raise OpError(
            "still_refused",
            f"range {p.index}: {facts['name']!r} is a still image",
            "render the still to a ProRes 4444 MOV of the length wanted and append that",
            "append-still-ignores-range",
        )
    if facts["is_multicam"]:
        raise OpError(
            "multicam_refused",
            f"range {p.index}: {facts['name']!r} is a multicam clip",
            "flatten a scratch duplicate to measure each camera's offset, then append the "
            "camera files",
            "multicam-angles",
        )
    frames = facts["frames"]
    if frames is not None and rng.src_out > frames:
        raise OpError(
            "range_beyond_media",
            f"range {p.index}: src_out {rng.src_out} is past the media's {frames} frames",
            "clamp the range to the media",
        )
    kinds = {"av": ("video", "audio"), "video": ("video",), "audio": ("audio",)}[rng.media_type]
    for kind in kinds:
        if kind == "video" and not facts["has_video"]:
            raise OpError(
                "no_video", f"range {p.index}: {facts['name']!r} has no video", "use 'audio'"
            )
        if kind == "audio" and rng.media_type == "audio" and not facts["has_audio"]:
            raise OpError(
                "no_audio", f"range {p.index}: {facts['name']!r} has no audio", "use 'video'"
            )
        if kind == "audio" and not facts["has_audio"]:
            continue  # an 'av' append of silent media places video only
        if int(timeline.GetTrackCount(kind) or 0) < rng.track:
            raise OpError(
                "track_missing",
                f"range {p.index}: the timeline has no {kind} track {rng.track}",
                "add the track first",
            )


def _placed(item: Any, timeline_start: int) -> PlacedItem:
    row = raw.read_item(item, timeline_start)
    kind, track = item.GetTrackTypeAndIndex()
    return PlacedItem(
        item_id=row["item_id"],
        track_type=kind,
        track=int(track),
        start=row["start"],
        end=row["end"],
        duration=row["duration"],
        record_in=row["record_in"],
        record_out=row["record_out"],
        source_start=row["source_start"],
        source_end=row["source_end"],
    )


def _clips(timeline: Any, kind: str, track: int) -> list[Any]:
    """Items of a track, without transitions (listed as items of their own, across the cut)."""
    return [
        item
        for item in timeline.GetItemListInTrack(kind, track) or []
        if item.GetType() != "transition"
    ]


def _overlapping(timeline: Any, kind: str, track: int, start: int, end: int) -> list[Any]:
    return [
        item
        for item in _clips(timeline, kind, track)
        if int(item.GetStart()) < end and int(item.GetEnd()) > start
    ]


def _at(timeline: Any, kind: str, track: int, start: int, media_id: str) -> list[Any]:
    return [
        item
        for item in timeline.GetItemListInTrack(kind, track) or []
        if int(item.GetStart()) == start and _media_id(item) == media_id
    ]


class _Appender:
    def __init__(self, env: OpEnv, timeline: Any, params: AppendParams) -> None:
        self.env = env
        self.timeline = timeline
        self.params = params
        self.media_pool = env.media_pool()
        self.start = int(timeline.GetStartFrame())

    def append_once(self, p: _Planned, mpi: Any, end_frame: int) -> tuple[Any, list[Any]]:
        rng = p.rng
        record = self.start + p.record_in
        info: dict[str, Any] = {
            "mediaPoolItem": mpi,
            "startFrame": rng.src_in,
            # trap append-endframe-exclusive: the exclusive end goes straight through.
            "endFrame": end_frame,
            "trackIndex": rng.track,
            "recordFrame": record,
        }
        if rng.media_type != "av":
            info["mediaType"] = 1 if rng.media_type == "video" else 2
        returned = list(self.media_pool.AppendToTimeline([info]) or [])
        if not returned:
            raise OpError(
                "append_failed",
                f"range {p.index}: AppendToTimeline returned nothing",
                "check the slot is free and the track is not locked",
            )
        placed = _at(self.timeline, p.kind, rng.track, record, rng.media.unique_id)
        if len(placed) != 1:
            landed = [(str(i.GetUniqueId()), int(i.GetStart()), int(i.GetEnd())) for i in returned]
            raise OpError(
                "append_misplaced",
                f"range {p.index}: no single {p.kind} item of the media at record frame "
                f"{record}; returned items landed at {landed}",
                "inspect the timeline with timeline.describe and remove strays by hand",
                landed=landed,
            )
        return placed[0], returned

    def audio_items(self, p: _Planned) -> list[Any]:
        if p.rng.media_type != "av":
            return []
        record = self.start + p.record_in
        return _at(self.timeline, "audio", p.rng.track, record, p.rng.media.unique_id)

    def run(self, p: _Planned, mpi: Any, facts: dict[str, Any]) -> AppendedRange:
        rng = p.rng
        record = self.start + p.record_in
        notes: list[Note] = []
        status: Literal["appended", "present", "repaired"] = "appended"
        end_used = rng.src_out
        occupying = _overlapping(self.timeline, p.kind, rng.track, record, record + p.frames)
        if occupying:
            mine = _at(self.timeline, p.kind, rng.track, record, rng.media.unique_id)
            if len(occupying) != 1 or not mine:
                raise OpError(
                    "slot_occupied",
                    f"range {p.index}: {p.kind} track {rng.track} already has "
                    f"{len(occupying)} other item(s) in [{record}, {record + p.frames})",
                    "append to an empty agent branch timeline, or clear the slot first",
                    occupying=[str(i.GetUniqueId()) for i in occupying],
                )
            item, returned = mine[0], None  # placed by an earlier attempt
            status = "present"
        else:
            item, returned = self.append_once(p, mpi, end_used)
        audio = self.audio_items(p)
        if returned is not None and audio and not any(i.GetType() == "audio" for i in returned):
            notes.append(
                Note(
                    code="audio_found_by_walk",
                    message=f"AppendToTimeline returned only the video item; found {len(audio)} "
                    f"audio item(s) on audio track {rng.track}",
                    trap_id="av-append-video-only",
                )
            )
        duration = int(item.GetEnd()) - int(item.GetStart())
        if duration == p.frames - 1 and self.params.repair_short:
            frames = facts["frames"]
            if frames is not None and rng.src_out + 1 > frames:
                raise OpError(
                    "repair_beyond_media",
                    f"range {p.index}: landed 1 frame short and the media has no frame "
                    f"{rng.src_out} to extend into",
                    "shorten the range by one frame or pick a longer source",
                    "append-fps-short",
                )
            doomed = [item, *audio]
            if not self.timeline.DeleteClips(doomed, False):
                raise OpError(
                    "repair_delete_failed",
                    f"range {p.index}: DeleteClips on the short item returned False",
                    "delete the short item by hand and resume at this cursor",
                    "append-fps-short",
                )
            if _at(self.timeline, p.kind, rng.track, record, rng.media.unique_id):
                raise OpError(
                    "repair_delete_failed",
                    f"range {p.index}: the short item is still on the timeline",
                    "delete it by hand and resume at this cursor",
                    "append-fps-short",
                )
            end_used = rng.src_out + 1
            item, returned = self.append_once(p, mpi, end_used)
            audio = self.audio_items(p)
            status = "repaired"
            notes.append(
                Note(
                    code="repaired_short",
                    message=f"landed {duration} frames, 1 short of {p.frames}; re-appended with "
                    f"endFrame {end_used}",
                    trap_id="append-fps-short",
                )
            )
            duration = int(item.GetEnd()) - int(item.GetStart())
        if duration != p.frames:
            overshoot = status == "repaired" and duration == p.frames + 1
            raise OpError(
                "repair_overshoot" if overshoot else "length_mismatch",
                f"range {p.index}: the item is {duration} frames, planned {p.frames}",
                "the item is left in place for inspection; fix it by hand or adjust the "
                "range, then resume at this cursor",
                "append-fps-short" if overshoot or duration == p.frames - 1 else None,
                item_id=str(item.GetUniqueId()),
            )
        source_start = item.GetSourceStartFrame()
        if source_start is not None and int(source_start) != rng.src_in:
            if abs(int(source_start) - rng.src_in) > 1:
                raise OpError(
                    "source_mismatch",
                    f"range {p.index}: the item starts at source frame {source_start}, not "
                    f"{rng.src_in}",
                    "inspect the item with timeline.describe",
                    item_id=str(item.GetUniqueId()),
                )
            notes.append(
                Note(
                    code="source_readback_off_by_one",
                    message=f"source start reads back as {source_start} for in point {rng.src_in}",
                    trap_id="source-frame-readback-rounding",
                )
            )
        if facts["has_audio"] and rng.media_type == "av" and not audio:
            notes.append(
                Note(
                    code="audio_missing",
                    message=f"no audio item of the media at record frame {record} on audio "
                    f"track {rng.track}",
                    trap_id="av-append-video-only",
                )
            )
        for a in audio:
            if int(a.GetEnd()) - int(a.GetStart()) != duration:
                notes.append(
                    Note(
                        code="audio_length_differs",
                        message=f"audio item {a.GetUniqueId()} is "
                        f"{int(a.GetEnd()) - int(a.GetStart())} frames, video {duration}",
                    )
                )
        primary = _placed(item, self.start)
        return AppendedRange(
            index=p.index,
            status=status,
            planned_record_in=p.record_in,
            planned_record_out=p.record_in + p.frames,
            planned_frames=p.frames,
            end_frame_used=end_used,
            video=primary if p.kind == "video" else None,
            audio=[_placed(a, self.start) for a in audio] if p.kind == "video" else [primary],
            notes=notes,
        )


def _gaps(
    timeline: Any, start: int, touched: dict[tuple[str, int], tuple[int, int]]
) -> list[TrackGap]:
    out = []
    for (kind, track), (lo, hi) in sorted(touched.items()):
        spans = [
            (int(i.GetStart()), int(i.GetEnd()))
            for i in _clips(timeline, kind, track)
            if int(i.GetEnd()) > lo and int(i.GetStart()) < hi
        ]
        gaps, _ = raw.find_gaps(spans)
        out.extend(
            TrackGap(
                track_type=kind,
                track=track,
                start=g0,
                end=g1,
                frames=g1 - g0,
                record_in=g0 - start,
                record_out=g1 - start,
            )
            for g0, g1 in gaps
        )
    return out


@op(
    "timeline.append_ranges",
    params=AppendParams,
    result=AppendResult,
    mutates=True,
    needs_current_timeline=True,
    traps=(
        "append-endframe-exclusive",
        "append-fps-short",
        "append-still-ignores-range",
        "av-append-video-only",
        "multicam-angles",
        "append-ungraded",
        "source-frame-readback-rounding",
    ),
)
def timeline_append_ranges(env: OpEnv, params: AppendParams) -> AppendResult:
    """Append source ranges ([in, out)) to a timeline, verified and repaired item by item."""
    timeline = env.current_timeline(params.timeline)
    settings = timeline.GetSettings() or {}
    tl_fps = settings.get("timelineFrameRate")
    if tl_fps is None:
        raise OpError("timeline_fps_unknown", "timelineFrameRate is not set", "check settings")
    tl_fps = float(tl_fps)
    upto = min(len(params.ranges), params.cursor + params.max_ranges)
    found = env.media([r.media for r in params.ranges[:upto]])
    facts = {uid: raw.read_media(mpi) for uid, (mpi, _) in found.items()}
    planned = _plan(params.ranges, facts, tl_fps, upto)
    window = planned[params.cursor : upto]
    for p in window:
        _preflight(timeline, p, facts[p.rng.media.unique_id])

    appender = _Appender(env, timeline, params)
    done: list[AppendedRange] = []
    touched: dict[tuple[str, int], tuple[int, int]] = {}
    for p in window:
        uid = p.rng.media.unique_id
        try:
            done.append(appender.run(p, found[uid][0], facts[uid]))
        except OpError as exc:
            extra = {k: v for k, v in exc.detail.items() if k not in ("fix_hint", "trap_id")}
            raise OpError(
                exc.code,
                exc.message,
                exc.fix_hint,
                exc.trap_id,
                **extra,
                resume_cursor=p.index,
                completed=[r.model_dump() for r in done],
            ) from exc
        lo = appender.start + p.record_in
        kinds = [p.kind] + (["audio"] if p.rng.media_type == "av" else [])
        for kind in kinds:
            old = touched.get((kind, p.rng.track), (lo, lo + p.frames))
            touched[(kind, p.rng.track)] = (min(old[0], lo), max(old[1], lo + p.frames))
    gaps = _gaps(timeline, appender.start, touched)
    notes = [
        Note(
            code="items_ungraded",
            message="appended items have no grade; copy grades per camera with grades.copy",
            trap_id="append-ungraded",
        )
    ] if any(r.status != "present" and r.video is not None for r in done) else []  # fmt: skip
    next_cursor = upto if upto < len(params.ranges) else None
    return AppendResult(
        timeline=env.identity(timeline),
        timeline_fps=tl_fps,
        cursor=params.cursor,
        next_cursor=next_cursor,
        done=next_cursor is None,
        ranges=done,
        gaps=gaps,
        notes=notes,
    )


# -- item.set_audio_gain -----------------------------------------------------------------------


class GainTarget(OpModel):
    item: ItemRef
    gain_db: float = Field(
        strict=True, ge=-100.0, le=30.0, description="Clip gain in dB (AudioVolume)."
    )


class GainParams(OpModel):
    timeline: TimelineRef | None = Field(default=None, description="Default: current timeline.")
    targets: list[GainTarget] = Field(min_length=1, max_length=500)
    tolerance_db: float = Field(strict=True, default=0.05, gt=0, le=1.0)

    @model_validator(mode="after")
    def _once(self) -> GainParams:
        _no_repeats([t.item.item_id for t in self.targets], "item")
        return self


class GainChange(OpModel):
    item_id: str
    before_db: float
    after_db: float
    page_bounced: bool


class GainResult(OpModel):
    timeline: TimelineIdentity
    changes: list[GainChange]
    notes: list[Note]


def _no_repeats(keys: list[Any], what: str) -> None:
    seen: set[Any] = set()
    repeated = sorted({k for k in keys if k in seen or seen.add(k)}, key=str)
    if repeated:
        raise ValueError(f"each {what} may be listed once per request; repeated: {repeated}")


_JAM_HINT = (
    "Resolve's setters can jam after a Color -> Edit round trip: open another page and back "
    "by hand, then retry"
)


def _with_bounce(env: OpEnv, call: Callable[[], Any]) -> tuple[bool, bool]:
    """Run a Resolve setter; on False, bounce the page once and retry (trap
    ``setproperty-after-color``). Returns ``(succeeded, bounced)``."""
    if call():
        return True, False
    env.bounce_page()  # trap setproperty-after-color
    return bool(call()), True


def _set_properties(env: OpEnv, item: Any, props: dict[str, Any]) -> bool:
    """``SetProperties`` with the page-bounce retry. Returns whether a bounce was needed."""
    ok, bounced = _with_bounce(env, lambda: item.SetProperties(props))
    if ok:
        return bounced
    raise OpError(
        "set_properties_failed",
        f"SetProperties({sorted(props)}) on item {item.GetUniqueId()} returned False, also "
        "after a page bounce",
        "check the keys against TimelineItemProperties and that the item's track is not "
        f"locked; {_JAM_HINT}",
        "setproperty-after-color",
    )


@op(
    "item.set_audio_gain",
    params=GainParams,
    result=GainResult,
    mutates=True,
    traps=("volume-key", "setproperty-after-color"),
)
def item_set_audio_gain(env: OpEnv, params: GainParams) -> GainResult:
    """Set audio clip gain (AudioVolume, dB) on audio items, read back per item."""
    timeline = env.timeline(params.timeline)
    items = []
    for target in params.targets:
        if target.item.track_type != "audio":
            raise OpError(
                "not_audio_item",
                f"item {target.item.item_id} is on a {target.item.track_type} track",
                "set gain on the audio track items",
                "volume-key",
            )
        item = env.item(timeline, target.item)
        props = item.GetProperties() or {}
        if props.get("AudioVolume") is None:
            raise OpError(
                "no_audio_volume",
                f"item {target.item.item_id} has no AudioVolume property",
                "set gain on audio track items",
                "volume-key",
            )
        items.append((target, item, float(props["AudioVolume"])))
    changes = []
    notes: list[Note] = []
    for target, item, before in items:
        # trap volume-key: the key is AudioVolume; "Volume" is silently refused.
        try:
            bounced = _set_properties(env, item, {"AudioVolume": float(target.gain_db)})
        except OpError as exc:
            raise _with_completed(exc, changes) from exc
        if bounced:
            notes.append(
                Note(
                    code="page_bounced",
                    message=f"SetProperties returned False until a page bounce (item "
                    f"{target.item.item_id})",
                    trap_id="setproperty-after-color",
                )
            )
        after = (item.GetProperties() or {}).get("AudioVolume")
        if after is None or abs(float(after) - target.gain_db) > params.tolerance_db:
            raise OpError(
                "gain_unverified",
                f"item {target.item.item_id}: AudioVolume reads back as {after}, wanted "
                f"{target.gain_db}",
                "read the item with timeline.describe and set it by hand",
                "volume-key",
                completed=[c.model_dump() for c in changes],
            )
        changes.append(
            GainChange(
                item_id=target.item.item_id,
                before_db=before,
                after_db=float(after),
                page_bounced=bounced,
            )
        )
    return GainResult(timeline=env.identity(timeline), changes=changes, notes=notes)


# -- item.set_fades ---------------------------------------------------------------------------


class FadeTarget(OpModel):
    item: ItemRef
    fade_in: int | None = Field(strict=True, default=None, ge=0, description="Frames.")
    fade_out: int | None = Field(strict=True, default=None, ge=0, description="Frames.")

    @model_validator(mode="after")
    def _one(self) -> FadeTarget:
        if self.fade_in is None and self.fade_out is None:
            raise ValueError("give fade_in, fade_out or both")
        return self


class FadesParams(OpModel):
    timeline: TimelineRef | None = Field(default=None, description="Default: current timeline.")
    targets: list[FadeTarget] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _once(self) -> FadesParams:
        _no_repeats([t.item.item_id for t in self.targets], "item")
        return self


class FadeChange(OpModel):
    item_id: str
    before_in: int
    before_out: int
    after_in: int
    after_out: int


class FadesResult(OpModel):
    timeline: TimelineIdentity
    changes: list[FadeChange]
    notes: list[Note] = Field(default_factory=list)


def _fades(item: Any) -> tuple[int, int]:
    fades = item.GetFades() or {}
    return round(fades.get("FadeIn") or 0), round(fades.get("FadeOut") or 0)


@op("item.set_fades", params=FadesParams, result=FadesResult, mutates=True)
def item_set_fades(env: OpEnv, params: FadesParams) -> FadesResult:
    """Set fade-in/out lengths (frames) on timeline items, read back per item."""
    timeline = env.timeline(params.timeline)
    resolved = [(t, env.item(timeline, t.item)) for t in params.targets]
    planned = []
    for target, item in resolved:
        before = _fades(item)
        wanted = {
            k: v
            for k, v in (("FadeIn", target.fade_in), ("FadeOut", target.fade_out))
            if v is not None
        }
        expected = (wanted.get("FadeIn", before[0]), wanted.get("FadeOut", before[1]))
        duration = int(item.GetEnd()) - int(item.GetStart())
        if sum(expected) > duration:
            raise OpError(
                "fades_too_long",
                f"item {target.item.item_id}: fades {expected} add up to more than its "
                f"{duration} frames",
                "shorten the fades so fade-in plus fade-out fit inside the item",
            )
        planned.append((target, item, before, wanted, expected))
    changes: list[FadeChange] = []
    notes: list[Note] = []
    for target, item, before, wanted, expected in planned:
        if before != expected:
            ok, bounced = _with_bounce(env, lambda item=item, wanted=wanted: item.SetFades(wanted))
            if not ok:
                raise OpError(
                    "set_fades_failed",
                    f"SetFades({wanted}) on item {target.item.item_id} returned False, also "
                    "after a page bounce",
                    f"the fades fit the item, so the setter is refusing: {_JAM_HINT}",
                    "setproperty-after-color",
                    completed=[c.model_dump() for c in changes],
                )
            if bounced:
                notes.append(
                    Note(
                        code="page_bounced",
                        message=f"SetFades returned False until a page bounce (item "
                        f"{target.item.item_id})",
                        trap_id="setproperty-after-color",
                    )
                )
        after = _fades(item)
        if after != expected:
            raise OpError(
                "fades_unverified",
                f"item {target.item.item_id}: fades read back as {after}, wanted {expected}",
                "set them by hand",
                completed=[c.model_dump() for c in changes],
            )
        changes.append(
            FadeChange(
                item_id=target.item.item_id,
                before_in=before[0],
                before_out=before[1],
                after_in=after[0],
                after_out=after[1],
            )
        )
    return FadesResult(timeline=env.identity(timeline), changes=changes, notes=notes)


# -- grades.copy -------------------------------------------------------------------------------


class GradeGroup(OpModel):
    camera: str | None = Field(default=None, description="Label, e.g. the camera letter.")
    reference: ItemRef
    reference_timeline: TimelineRef | None = Field(
        default=None,
        description="Where the graded reference item is. Default: the target timeline.",
    )
    targets: list[ItemRef] = Field(min_length=1, max_length=500)


class GradesParams(OpModel):
    timeline: TimelineRef | None = Field(default=None, description="Default: current timeline.")
    groups: list[GradeGroup] = Field(min_length=1, max_length=50)
    reapply_luts: bool = Field(
        strict=True,
        default=True,
        description="Put back each target's per-node LUTs after the copy.",
    )
    require_graded_reference: bool = Field(
        strict=True,
        default=True,
        description="Refuse a reference whose nodes show no tools and no LUT (an ungraded "
        "item would wipe the targets' grades).",
    )

    @model_validator(mode="after")
    def _once(self) -> GradesParams:
        targets = [t.item_id for g in self.groups for t in g.targets]
        _no_repeats(targets, "target item")
        both = sorted({g.reference.item_id for g in self.groups} & set(targets))
        if both:
            raise ValueError(f"an item cannot be both a reference and a target: {both}")
        return self


class LutCheck(OpModel):
    node: int
    lut: str
    after: str


class GradedItem(OpModel):
    item_id: str
    nodes_before: int
    nodes_after: int
    luts_before: dict[int, str]
    luts_after: dict[int, str]
    reapplied: list[LutCheck]


class GradeGroupResult(OpModel):
    camera: str | None
    reference_item_id: str
    reference_nodes: int
    targets: list[GradedItem]


class GradesResult(OpModel):
    timeline: TimelineIdentity
    groups: list[GradeGroupResult]
    notes: list[Note]


def _luts(graph: Any) -> dict[int, str]:
    count = int(graph.GetNumNodes() or 0)
    return {n: str(graph.GetLUT(n)) for n in range(1, count + 1) if graph.GetLUT(n)}


def _grade_print(graph: Any) -> list[tuple[str, tuple[str, ...], str]]:
    """What a node graph holds, node by node: label, tools and LUT."""
    count = int(graph.GetNumNodes() or 0)
    return [
        (
            str(graph.GetNodeLabel(n) or ""),
            tuple(str(t) for t in (graph.GetToolsInNode(n) or [])),
            str(graph.GetLUT(n) or ""),
        )
        for n in range(1, count + 1)
    ]


def _with_completed(exc: OpError, done: list[Any]) -> OpError:
    """``exc`` with the work finished so far as ``completed`` (unless it already has it)."""
    if "completed" in exc.detail:
        return exc
    extra = {k: v for k, v in exc.detail.items() if k not in ("fix_hint", "trap_id")}
    return OpError(
        exc.code,
        exc.message,
        exc.fix_hint,
        exc.trap_id,
        **extra,
        completed=[d.model_dump() for d in done],
    )


@op(
    "grades.copy",
    params=GradesParams,
    result=GradesResult,
    mutates=True,
    traps=("append-ungraded", "setproperty-after-color"),
)
def grades_copy(env: OpEnv, params: GradesParams) -> GradesResult:
    """Copy a reference item's grade to target items, per camera, keeping per-clip LUTs."""
    results: list[GradeGroupResult] = []
    try:
        return _grades_copy(env, params, results)
    except OpError as exc:
        # Every error says which groups were copied (and verified) before it.
        raise _with_completed(exc, results) from exc


def _grades_copy(env: OpEnv, params: GradesParams, results: list[GradeGroupResult]) -> GradesResult:
    timeline = env.timeline(params.timeline)
    plan = []
    for group in params.groups:
        ref_tl = (
            timeline if group.reference_timeline is None else env.timeline(group.reference_timeline)
        )
        refs = [group.reference, *group.targets]
        if any(r.track_type != "video" for r in refs):
            raise OpError(
                "not_video_item",
                f"group {group.camera!r}: grades live on video items only",
                "use video track item refs",
            )
        reference = env.item(ref_tl, group.reference)
        print_ref = _grade_print(reference.GetNodeGraph())
        if params.require_graded_reference and not any(tools or lut for _, tools, lut in print_ref):
            raise OpError(
                "reference_ungraded",
                f"group {group.camera!r}: reference item {group.reference.item_id} shows no "
                f"tools and no LUT on its {len(print_ref)} node(s); copying it would wipe "
                f"the targets' grades",
                "pick a graded item of that camera as the reference (or pass "
                "require_graded_reference=false if its grade is not visible to the API)",
                "append-ungraded",
            )
        targets = []
        for ref in group.targets:
            item = env.item(timeline, ref)
            luts = _luts(item.GetNodeGraph())
            # Checked before anything is copied: CopyGrades replaces the whole graph, so a LUT
            # on a node the copied grade does not have could never be put back.
            beyond = {n: lut for n, lut in luts.items() if n > len(print_ref)}
            if params.reapply_luts and beyond:
                raise OpError(
                    "lut_node_missing",
                    f"item {ref.item_id} has LUT(s) {beyond} on node(s) the reference's "
                    f"{len(print_ref)}-node grade does not have; nothing was copied",
                    "add nodes to the reference grade, or move the LUTs to its nodes first",
                    "append-ungraded",
                )
            targets.append((item, int(item.GetNodeGraph().GetNumNodes() or 0), luts))
        plan.append((group, reference, print_ref, targets))

    notes: list[Note] = []
    for group, reference, print_ref, targets in plan:
        results.append(_copy_group(env, params, group, reference, print_ref, targets, notes))
    return GradesResult(timeline=env.identity(timeline), groups=results, notes=notes)


def _copy_group(
    env: OpEnv,
    params: GradesParams,
    group: GradeGroup,
    reference: Any,
    print_ref: list[tuple[str, tuple[str, ...], str]],
    targets: list[tuple[Any, int, dict[int, str]]],
    notes: list[Note],
) -> GradeGroupResult:
    items = [t for t, _, _ in targets]
    ok, bounced = _with_bounce(env, lambda: reference.CopyGrades(items))
    if not ok:
        raise OpError(
            "copy_grades_failed",
            f"group {group.camera!r}: CopyGrades returned False, also after a page bounce",
            "check the reference and targets are video items of open timelines; " + _JAM_HINT,
            "setproperty-after-color",
        )
    if bounced:
        notes.append(
            Note(
                code="page_bounced",
                message=f"CopyGrades returned False until a page bounce (group {group.camera!r})",
                trap_id="setproperty-after-color",
            )
        )
    graded = []
    for target, nodes_before, luts_before in targets:
        graph = target.GetNodeGraph()
        copied = _grade_print(graph)
        if copied != print_ref:
            raise OpError(
                "grade_unverified",
                f"item {target.GetUniqueId()} does not hold the reference's grade after "
                f"CopyGrades ({len(copied)} node(s), the reference {len(print_ref)})",
                "check the grade on the Color page",
                "append-ungraded",
                completed_targets=[g.model_dump() for g in graded],
            )
        reapplied = []
        if params.reapply_luts:
            # trap append-ungraded: CopyGrades replaced the node graph, so put the target's
            # own LUTs back on their nodes (checked to exist before the copy).
            for node, lut in sorted(luts_before.items()):
                ok, bounced = _with_bounce(
                    env, lambda node=node, lut=lut, graph=graph: graph.SetLUT(node, lut)
                )
                if not ok:
                    raise OpError(
                        "lut_reapply_failed",
                        f"SetLUT({node}, {lut!r}) on item {target.GetUniqueId()} returned "
                        "False, also after a page bounce",
                        "refresh the LUT list (Project.RefreshLUTList) so Resolve knows the "
                        f"LUT; {_JAM_HINT}",
                        "setproperty-after-color",
                        completed_targets=[g.model_dump() for g in graded],
                    )
                if bounced:
                    notes.append(
                        Note(
                            code="page_bounced",
                            message=f"SetLUT returned False until a page bounce (item "
                            f"{target.GetUniqueId()}, node {node})",
                            trap_id="setproperty-after-color",
                        )
                    )
                after = str(graph.GetLUT(node) or "")
                if after != lut:
                    raise OpError(
                        "lut_unverified",
                        f"node {node} of item {target.GetUniqueId()} reads back LUT {after!r}, "
                        f"not {lut!r}",
                        "set the LUT by hand",
                        "append-ungraded",
                        completed_targets=[g.model_dump() for g in graded],
                    )
                reapplied.append(LutCheck(node=node, lut=lut, after=after))
        graded.append(
            GradedItem(
                item_id=str(target.GetUniqueId()),
                nodes_before=nodes_before,
                nodes_after=len(copied),
                luts_before=luts_before,
                luts_after=_luts(target.GetNodeGraph()),
                reapplied=reapplied,
            )
        )
    return GradeGroupResult(
        camera=group.camera,
        reference_item_id=str(reference.GetUniqueId()),
        reference_nodes=len(print_ref),
        targets=graded,
    )


# -- clip.set_input_color_space -----------------------------------------------------------------

_SELECTOR_KEYS = frozenset({"bin", "bin_path", "folder", "folders", "recursive", "all", "selector"})

# Colour-space names seen in a live 21.1 project (clip 'Input Color Space' values and timeline
# colour spaces). A name is accepted if it is here, in the project's colour settings, or on a
# clip in the listed items' bins; anything else is refused before any write.
OBSERVED_COLOR_SPACES: frozenset[str] = frozenset(
    {"Rec.709 (Scene)", "Blackmagic Design Film Gen 5", "DaVinci WG/Intermediate"}
)
_PROJECT_COLOUR_KEYS = ("colorSpaceInput", "colorSpaceTimeline", "colorSpaceOutput")


class InputColourParams(OpModel):
    """An explicit list of media-pool items. Bin or recursive selectors are refused."""

    items: list[MediaRef] = Field(min_length=1, max_length=200)
    color_space: str = Field(min_length=1, description="e.g. 'Rec.709 (Scene)'.")

    @model_validator(mode="before")
    @classmethod
    def _no_selectors(cls, data: Any) -> Any:
        if isinstance(data, dict):
            used = sorted(_SELECTOR_KEYS & set(data))
            if used:
                raise ValueError(
                    f"selector keys {used} are refused: list the media-pool items explicitly "
                    "(trap rcm-input-colorspace: never sweep a bin)"
                )
        return data

    @model_validator(mode="after")
    def _unique(self) -> InputColourParams:
        ids = [m.unique_id for m in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("media items are listed more than once")
        if self.color_space != self.color_space.strip() or not self.color_space.strip():
            raise ValueError("color_space must be an exact name, without surrounding spaces")
        return self


def _known_colour_spaces(env: OpEnv, bins: set[tuple[str, ...]]) -> dict[str, list[str]]:
    """The colour-space names the op accepts, by where each comes from.

    Fail closed, with no override: a name that is in none of these is refused before any
    write, because a typo would otherwise be written to the clip.
    """
    settings = env.project().GetSettings() or {}
    project = {str(settings[k]) for k in _PROJECT_COLOUR_KEYS if settings.get(k)}
    clips: set[str] = set()
    for path in bins:
        for clip in env.bin(path).GetClipList() or []:
            value = (clip.GetClipProperty() or {}).get("Input Color Space")
            if value:
                clips.add(str(value))
    return {
        "project_colour_keys": sorted(project),
        "bin_clip_values": sorted(clips),
        "observed_names": sorted(OBSERVED_COLOR_SPACES),
    }


class ColourChange(OpModel):
    media_id: str
    name: str
    before: str
    after: str


class InputColourResult(OpModel):
    changes: list[ColourChange]


@op(
    "clip.set_input_color_space",
    params=InputColourParams,
    result=InputColourResult,
    mutates=True,
    traps=("rcm-input-colorspace",),
)
def clip_set_input_color_space(env: OpEnv, params: InputColourParams) -> InputColourResult:
    """Set 'Input Color Space' on listed media-pool items (never BRAW, never a bin sweep)."""
    trap = "rcm-input-colorspace"
    found = env.media(params.items)
    facts = [
        (ref, found[ref.unique_id][0], raw.read_media(found[ref.unique_id][0]))
        for ref in params.items
    ]
    braw = [f["media_id"] for _, _, f in facts if f["is_braw"]]
    if braw:
        raise OpError(
            "braw_refused",
            f"{len(braw)} listed item(s) are BRAW: {braw}",
            "BRAW colour comes from its RAW decode settings; leave it alone and list only the "
            "non-BRAW media",
            trap,
            braw=braw,
        )
    no_video = [
        f["media_id"]
        for _, _, f in facts
        if not f["has_video"] or f["type"] in ("Timeline", "Compound", "Multicam")
    ]
    if no_video:
        raise OpError(
            "not_video_media",
            f"listed item(s) have no own video colour space: {no_video}",
            "list video media files only",
            trap,
        )
    sources = _known_colour_spaces(env, {where for _, where in found.values()})
    known = {name for names in sources.values() for name in names}
    if params.color_space not in known:
        raise OpError(
            "color_space_unknown",
            f"{params.color_space!r} is not an accepted colour space. Accepted names come from "
            f"the project's colour settings ({', '.join(_PROJECT_COLOUR_KEYS)}), the 'Input "
            f"Color Space' of clips in the listed items' bins, and names observed on a live "
            f"Resolve; none has it",
            f"use an exact name from one of those sources: {sorted(known)}",
            trap,
            known=sorted(known),
            known_sources=sources,
        )
    changes = []
    for _, mpi, f in facts:
        before = f["input_color_space"]
        if before != params.color_space and not mpi.SetClipProperty(
            "Input Color Space", params.color_space
        ):
            raise OpError(
                "set_clip_property_failed",
                f"SetClipProperty('Input Color Space', {params.color_space!r}) on "
                f"{f['name']!r} returned False",
                "check the project is colour managed and the colour space name is exact",
                trap,
                completed=[c.model_dump() for c in changes],
            )
        after = str((mpi.GetClipProperty() or {}).get("Input Color Space") or "")
        if after != params.color_space:
            raise OpError(
                "colour_space_unverified",
                f"{f['name']!r} reads back {after!r}",
                "set it by hand in Clip Attributes",
                trap,
                completed=[c.model_dump() for c in changes],
            )
        changes.append(
            ColourChange(media_id=f["media_id"], name=f["name"], before=before, after=after)
        )
    return InputColourResult(changes=changes)


# -- markers.add ---------------------------------------------------------------------------------


class MarkerSpec(OpModel):
    frame: int = Field(strict=True, ge=0, description="Frames from the timeline start.")
    color: MarkerColor = "Blue"
    name: str = ""
    note: str = ""
    duration: int = Field(strict=True, default=1, ge=1)
    custom_data: str | None = None


class MarkersParams(OpModel):
    timeline: TimelineRef | None = Field(default=None, description="Default: current timeline.")
    markers: list[MarkerSpec] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _one_per_frame(self) -> MarkersParams:
        # A timeline holds one marker per frame: two in one request could never both land.
        _no_repeats([m.frame for m in self.markers], "marker frame")
        return self


class MarkerAdded(OpModel):
    frame: int
    status: Literal["added", "present"]
    color: str
    name: str
    note: str
    duration: int
    custom_data: str


class MarkersResult(OpModel):
    timeline: TimelineIdentity
    markers: list[MarkerAdded]


def _marker_matches(have: dict[str, Any], want: MarkerSpec) -> bool:
    return (
        have["color"] == want.color
        and have["name"] == want.name
        and have["note"] == want.note
        and have["duration"] == want.duration
        and have["custom_data"] == (want.custom_data or "")
    )


@op("markers.add", params=MarkersParams, result=MarkersResult, mutates=True)
def markers_add(env: OpEnv, params: MarkersParams) -> MarkersResult:
    """Add timeline markers; a marker already there with the same fields counts as done."""
    timeline = env.timeline(params.timeline)
    existing = {m["frame"]: m for m in raw.read_markers(timeline)}
    for spec in params.markers:
        have = existing.get(spec.frame)
        if have is not None and not _marker_matches(have, spec):
            raise OpError(
                "marker_conflict",
                f"a different marker is already at frame {spec.frame}: {have}",
                "pick another frame or remove the existing marker",
            )
    out = []
    for spec in params.markers:
        status: Literal["added", "present"] = "present"
        if spec.frame not in existing:
            args: list[Any] = [spec.frame, spec.color, spec.name, spec.note, spec.duration]
            if spec.custom_data is not None:
                args.append(spec.custom_data)
            if not timeline.AddMarker(*args):
                raise OpError(
                    "add_marker_failed",
                    f"AddMarker at frame {spec.frame} returned False",
                    "the frame must lie inside the timeline",
                    completed=[m.model_dump() for m in out],
                )
            status = "added"
        read = {m["frame"]: m for m in raw.read_markers(timeline)}.get(spec.frame)
        if read is None or not _marker_matches(read, spec):
            raise OpError(
                "marker_unverified",
                f"the marker at frame {spec.frame} reads back as {read}",
                "inspect markers with timeline.describe",
                completed=[m.model_dump() for m in out],
            )
        out.append(MarkerAdded(status=status, **read))
    return MarkersResult(timeline=env.identity(timeline), markers=out)
