"""Plan -> :class:`Layout`: every decision the FCPXML export makes, taken once, in plan terms.

The layout is the timeline the file will describe, in record and source frames: which clip sits
where (one per content segment, plus the outro), every transition with its alignment, the titles
and markers, and every plan feature FCPXML cannot carry (the report). The writer encodes a layout
as XML; the self-check decodes the XML and compares it with the layout built again from the plan.

Plan walking reuses the ``plan.apply`` compiler's helpers (S13h2): the first and last content
segment, the title track, the audio track index, the handle arithmetic and the "an unknown
handle is no handle" rule. Nothing here talks to Resolve or the network.

Conventions (see docs/DESIGN.md, "FCPXML export"):

* Plan video track ``N`` is lane ``N - 1``: V1 is the primary spine, V2 and up are connected clips.
* A content segment carries its own sound, which by the compiler's positional convention sits on
  audio track ``A{N}``; plan audio track ``A{k}`` for ``k >= 2`` maps to lane ``-(k - 1)`` for an
  audio-only clip (the outro's sound on its named track). FCPXML has no tracks, so a track's gain
  is folded into each clip's volume and its mute and bus are reported.
* A transition's alignment comes from the compiler's own rule
  (:func:`aar.plan.apply.compile.transition_alignment`): centred when the handles hold all its
  media (``d // 2`` frames before the cut), else it starts on the cut (``right``) or ends on it
  (``left``). FCPXML overlaps the clips under a transition element, so the media must exist.
* An audio-only crossfade is, by default, a split edit (``split``): the picture cuts at the
  plan's cut and each clip's sound runs on under the other's with fades; ``transition`` writes it
  as a spine transition holding only ``filter-audio``.
* Framing: ``Source.frame_size`` gives the source picture's aspect. A clip whose aspect differs
  from the timeline's, or that has a reframe or transform, gets ``adjust-conform type="fill"``
  and an ``adjust-transform`` computed in FCPXML's units from the source size; such a clip needs
  the size (``source_size_required``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from aar.plan.apply.compile import (
    _audio_index,
    _first_content_segment,
    _last_content_segment,
    incoming_head,
    outgoing_tail,
    title_track,
    transition_alignment,
)
from aar.plan.export.errors import ExportError
from aar.plan.export.media import MediaMap, MediaRef, media_ref
from aar.plan.export.timebase import Timebase
from aar.plan.model import EditPlan, ReframeKey, Segment, Source, Transform

DEFAULT_TIMELINE_START = "01:00:00:00"
# Characters XML 1.0 cannot carry, plus CR, which a parser turns into LF (so it would not
# round-trip): C0 controls other than tab and LF, U+FFFE/U+FFFF and lone surrogates.
_XML_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\ufffe\uffff\ud800-\udfff]")
DEFAULT_AUDIO_TRANSITION = "Cross Fade 0 dB"

Enable = Literal["all", "video", "audio"]
TransitionKind = Literal["head", "join", "outro"]
Alignment = Literal["right", "center", "left"]
#: How an audio-only crossfade is written: a J/L split edit, or a spine transition.
AudioCrossfade = Literal["split", "transition"]
AUDIO_CROSSFADE_FORMS: tuple[AudioCrossfade, ...] = ("split", "transition")


@dataclass(frozen=True)
class Keyframe:
    """One adjust-transform keyframe: frames from the segment's start, FCPXML units."""

    time: int
    position: tuple[float, float]
    scale: tuple[float, float]


@dataclass(frozen=True)
class TransformSpec:
    """A segment's adjust-transform: position (percent of the frame height, y up), scale (negative
    for a flip), rotation (degrees), and keyframes when the reframe moves."""

    position: tuple[float, float]
    scale: tuple[float, float]
    rotation: float
    keyframes: tuple[Keyframe, ...] = ()


@dataclass(frozen=True)
class ClipSpec:
    """One asset-clip, in plan terms (its record and source range before transition overlaps)."""

    location: str
    lane: int
    record_in: int
    record_out: int
    src_in: int
    src_out: int
    src: str
    enable: Enable
    audio_track: int | None
    volume_db: float = 0.0
    fade_in: int = 0
    fade_out: int = 0
    transform: TransformSpec | None = None
    name: str = ""
    note: str | None = None
    frame_size: tuple[int, int] | None = None
    conform: bool = False


@dataclass(frozen=True)
class TransitionSpec:
    """One spine transition: ``head_part`` of its ``frames`` lie before the ``cut``."""

    location: str
    kind: TransitionKind
    track: int
    cut: int
    frames: int
    head_part: int
    alignment: Alignment
    video: bool
    audio: bool
    form: AudioCrossfade = "transition"

    @property
    def window_start(self) -> int:
        return self.cut - self.head_part

    @property
    def tail_part(self) -> int:
        return self.frames - self.head_part


@dataclass(frozen=True)
class TitleSpec:
    location: str
    kind: str
    lane: int
    record_in: int
    record_out: int
    lines: tuple[str, ...]
    template: str | None
    name: str


@dataclass(frozen=True)
class MarkerSpec:
    location: str
    record: int
    duration: int
    value: str
    note: str | None


@dataclass(frozen=True)
class AssetSpec:
    src: str
    name: str
    duration: int


@dataclass(frozen=True)
class ReportItem:
    """A plan feature and what the export did with it, with its location in the plan."""

    feature: str
    location: str
    detail: str
    action: str

    def as_dict(self) -> dict[str, str]:
        return {
            "feature": self.feature,
            "location": self.location,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass
class Layout:
    plan: EditPlan
    timebase: Timebase
    tc_start: int
    timeline_start: str
    total_frames: int
    audio_crossfade: AudioCrossfade = "split"
    clips: list[ClipSpec] = field(default_factory=list)
    transitions: list[TransitionSpec] = field(default_factory=list)
    titles: list[TitleSpec] = field(default_factory=list)
    markers: list[MarkerSpec] = field(default_factory=list)
    media: dict[str, MediaRef] = field(default_factory=dict)
    assets: list[AssetSpec] = field(default_factory=list)
    not_expressible: list[ReportItem] = field(default_factory=list)
    notes: list[ReportItem] = field(default_factory=list)

    @property
    def title_lane(self) -> int:
        return title_track(self.plan) - 1

    def transition_at(self, cut: int) -> TransitionSpec | None:
        """The transition at record frame ``cut`` of video track 1, in either form (cuts are
        unique there)."""
        for spec in self.transitions:
            if spec.cut == cut:
                return spec
        return None

    def spine_transition_at(self, cut: int) -> TransitionSpec | None:
        """The transition at ``cut`` when it is written as a spine ``transition`` element."""
        spec = self.transition_at(cut)
        return spec if spec is not None and spec.form == "transition" else None

    def raw_range(self, clip: ClipSpec) -> tuple[int, int]:
        """The record frames a clip covers in the file: its plan range, widened under the
        transition elements on its ends (FCPXML overlaps both neighbours under one)."""
        if clip.lane != 0:
            return clip.record_in, clip.record_out
        into = self.spine_transition_at(clip.record_in)
        out = self.spine_transition_at(clip.record_out)
        return (
            clip.record_in - (into.head_part if into is not None else 0),
            clip.record_out + (out.tail_part if out is not None else 0),
        )

    def sound_range(self, clip: ClipSpec) -> tuple[int, int]:
        """The record frames a clip's sound plays: its plan range, widened only under an audio
        crossfade (in a transition element, or a split edit)."""
        if clip.lane != 0:
            return clip.record_in, clip.record_out
        into, out = self.transition_at(clip.record_in), self.transition_at(clip.record_out)
        return (
            clip.record_in - (into.head_part if into is not None and into.audio else 0),
            clip.record_out + (out.tail_part if out is not None and out.audio else 0),
        )

    def fades(self, clip: ClipSpec) -> tuple[int, int]:
        """The clip's sound fade-in and fade-out: the head and end fades, plus the fades of a
        split-edit crossfade on its ends (they never fall on the same end)."""
        fade_in, fade_out = clip.fade_in, clip.fade_out
        if clip.lane == 0 and clip.enable != "video":
            into, out = self.transition_at(clip.record_in), self.transition_at(clip.record_out)
            if into is not None and into.form == "split":
                fade_in += into.frames
            if out is not None and out.form == "split":
                fade_out += out.frames
        return fade_in, fade_out

    def audio_lane(self, track: int) -> int:
        """The lane of an audio-only clip on plan audio track ``A{track}`` (``track >= 2``)."""
        return -(track - 1)

    def sequence_note(self) -> str | None:
        """The sequence-level note: what the file cannot say about the audio topology."""
        lines = [
            f"{item.location}: {item.detail}"
            for item in self.not_expressible
            if item.feature in ("audio_mute", "audio_bus", "template")
        ]
        if not lines:
            return None
        return "aar export: not expressible in FCPXML (see the export report)\n" + "\n".join(lines)


# -- transforms --------------------------------------------------------------------------------


def conformed_size(frame_size: tuple[int, int], width: int, height: int) -> tuple[float, float]:
    """The source picture's size in timeline pixels under ``adjust-conform type="fill"``: scaled
    uniformly until it covers the whole frame."""
    fill = max(width / frame_size[0], height / frame_size[1])
    return frame_size[0] * fill, frame_size[1] * fill


def transform_spec(
    seg: Segment, frame_size: tuple[int, int], width: int, height: int
) -> TransformSpec:
    """A segment's static transform and reframe keyframes as one FCPXML adjust-transform, for a
    clip conformed with ``fill`` (DESIGN "FCPXML export"; never the timeline width for a source
    value).

    The reframe picks a crop window of the source: its centre ``(x, y)`` as shares of the source
    width and height, ``zoom`` 1.0 the fill window (the tallest crop the timeline's aspect allows).
    The static transform multiplies the zoom, rotates and flips about the frame centre, and pans
    and tilts in timeline pixels. In timeline pixels from the frame centre, y up, with the source
    filled to ``(cw, ch)``, a source point ``p`` lands at ``R(rot) * s * F * (p - c) + (pan,
    tilt)``, where ``c = ((x - 0.5) * cw, (0.5 - y) * ch)`` and ``s = zoom * reframe zoom``. So the
    FCPXML scale is ``s * F`` and the position ``(pan, tilt) - R(rot) * s * F * c``, in percent of
    the frame height.
    """
    tf = seg.transform or Transform()
    flip = (-1.0 if tf.flip_h else 1.0, -1.0 if tf.flip_v else 1.0)
    cw, ch = conformed_size(frame_size, width, height)
    angle = math.radians(tf.rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)

    def at(key: ReframeKey) -> tuple[tuple[float, float], tuple[float, float]]:
        zoom = key.zoom * tf.zoom
        ax = zoom * flip[0] * (key.x - 0.5) * cw
        ay = zoom * flip[1] * (0.5 - key.y) * ch
        qx, qy = cos * ax - sin * ay, sin * ax + cos * ay
        position = ((tf.pan - qx) / height * 100.0, (tf.tilt - qy) / height * 100.0)
        return position, (zoom * flip[0], zoom * flip[1])

    keys = list(seg.reframe) or [ReframeKey(time=0)]
    position, scale = at(keys[0])
    keyframes = tuple(Keyframe(key.time, *at(key)) for key in keys) if len(keys) > 1 else ()
    return TransformSpec(position, scale, tf.rotation_deg, keyframes)


# -- the build ---------------------------------------------------------------------------------


class _Builder:
    def __init__(
        self,
        plan: EditPlan,
        media_map: MediaMap | None,
        timeline_start: str,
        audio_crossfade: AudioCrossfade,
    ) -> None:
        if audio_crossfade not in AUDIO_CROSSFADE_FORMS:
            raise ValueError(f"audio_crossfade must be one of {AUDIO_CROSSFADE_FORMS}")
        self.plan = plan
        self.media_map = media_map
        self.timebase = Timebase.of(plan.fps)
        try:
            tc_start = self.timebase.timecode_frames(timeline_start)
        except ValueError as exc:
            raise ExportError(
                "bad_timeline_start",
                f"timeline start {timeline_start!r}: {exc}",
                "set the profile's [export] timeline_start to non-drop HH:MM:SS:FF, e.g. "
                f"{DEFAULT_TIMELINE_START}",
            ) from None
        total = plan.total_frames()
        if total <= 0:
            raise ExportError(
                "empty_timeline",
                "the plan places nothing on the timeline (no segments and no outro)",
                "add segments to the plan",
            )
        self.ranges = plan.record_ranges()
        self.layout = Layout(plan, self.timebase, tc_start, timeline_start, total, audio_crossfade)

    # -- report helpers ------------------------------------------------------------------------

    def unexpressible(self, feature: str, location: str, detail: str, action: str) -> None:
        self.layout.not_expressible.append(ReportItem(feature, location, detail, action))

    def note(self, feature: str, location: str, detail: str, action: str = "none") -> None:
        self.layout.notes.append(ReportItem(feature, location, detail, action))

    # -- walk ----------------------------------------------------------------------------------

    def build(self) -> Layout:
        self._texts()
        self._media()
        self._clips()
        self._fades()
        self._transitions()
        self._check_overlaps()
        self._check_fades()
        self._titles()
        self._markers()
        self._assets()
        self._report_plan_features()
        return self.layout

    def _texts(self) -> None:
        """Refuse text the file cannot carry exactly (XML 1.0), naming where it is."""
        plan = self.plan
        fields: list[tuple[str, str | None]] = [
            ("project", plan.project),
            ("timeline", plan.timeline),
            ("template", plan.template),
        ]
        for i, seg in enumerate(plan.segments):
            fields += [(f"segments[{i}].label", seg.label), (f"segments[{i}].notes", seg.notes)]
        for i, title in enumerate(plan.titles):
            fields += [(f"titles[{i}].lines[{j}]", line) for j, line in enumerate(title.lines)]
        if plan.outro is not None and plan.outro.sting is not None:
            fields += [
                (f"outro.sting.lines[{j}]", line) for j, line in enumerate(plan.outro.sting.lines)
            ]
        for i, marker in enumerate(plan.markers):
            fields += [(f"markers[{i}].name", marker.name), (f"markers[{i}].note", marker.note)]
        for i, track in enumerate(plan.audio_tracks):
            fields += [
                (f"audio_tracks[{i}].name", track.name),
                (f"audio_tracks[{i}].bus", track.bus),
            ]
        for location, value in fields:
            if value is not None and _XML_UNSAFE.search(value):
                raise ExportError(
                    "unsafe_text",
                    f"{location} holds a character FCPXML (XML 1.0) cannot carry exactly "
                    "(a control character, CR, U+FFFE/U+FFFF or a lone surrogate)",
                    "remove it from the text (use a plain line break, LF, if one is meant)",
                    location=location,
                )

    def _media(self) -> None:
        used = {seg.source for seg in self.plan.segments}
        if self.plan.outro is not None:
            used.add(self.plan.outro.source)
        for source in self.plan.sources:
            if source.id in used and source.id not in self.layout.media:
                self.layout.media[source.id] = media_ref(source, self.media_map)
        self._check_shared_durations()

    def _check_shared_durations(self) -> None:
        """Sources that share one media file may not disagree about its length (one asset)."""
        first: dict[str, tuple[str, int]] = {}
        for source in self.plan.sources:
            ref = self.layout.media.get(source.id)
            if ref is None or source.duration is None:
                continue
            seen = first.setdefault(ref.src, (source.id, source.duration))
            if seen[1] != source.duration:
                raise ExportError(
                    "conflicting_source_durations",
                    f"sources {seen[0]!r} and {source.id!r} are the same media file "
                    f"({ref.windows_path}) but give it different durations "
                    f"({seen[1]} and {source.duration} frames)",
                    "give both sources the same duration, or leave one unset, or merge them "
                    "into one source",
                    sources=[seen[0], source.id],
                    media=ref.windows_path,
                    durations=[seen[1], source.duration],
                )

    def _track_gain(self, track: int) -> float:
        tracks = self.plan.audio_tracks
        return tracks[track - 1].gain_db if 1 <= track <= len(tracks) else 0.0

    def _clips(self) -> None:
        sources = self.plan.source_map()
        for index, seg in enumerate(self.plan.segments):
            source = sources.get(seg.source)
            if source is None:
                raise ExportError(
                    "unknown_source",
                    f"segment {index} names source {seg.source!r}, which is not in the plan",
                    "add the source, or fix the segment's 'source' id",
                    segment=index,
                )
            if seg.length <= 0:
                raise ExportError(
                    "zero_length_segment",
                    f"segment {index} has a non-positive length ({seg.src_in}..{seg.src_out})",
                    "give src_out greater than src_in",
                    segment=index,
                )
            ref = self.layout.media[source.id]
            record_in, record_out = self.ranges[index]
            volume = seg.gain_db + self._track_gain(seg.track)
            if not math.isfinite(volume):
                raise ExportError(
                    "non_finite_value",
                    f"segment {index} gain {seg.gain_db!r} dB is not a finite number",
                    "give gain_db as a finite number of dB",
                    segment=index,
                )
            moved = seg.transform is not None or bool(seg.reframe)
            frame_size, conform = self._framing(f"segments[{index}]", source, moved)
            transform = None
            if moved:
                assert frame_size is not None
                transform = self._transform(index, seg, frame_size)
            self.layout.clips.append(
                ClipSpec(
                    location=f"segments[{index}]",
                    lane=seg.track - 1,
                    record_in=record_in,
                    record_out=record_out,
                    src_in=seg.src_in,
                    src_out=seg.src_out,
                    src=ref.src,
                    enable="all",
                    audio_track=seg.track,
                    volume_db=volume,
                    transform=transform,
                    name=seg.label or ref.name,
                    note=seg.notes,
                    frame_size=frame_size,
                    conform=conform,
                )
            )
        self._outro_clips()

    def _framing(
        self, location: str, source: Source, moved: bool
    ) -> tuple[tuple[int, int] | None, bool]:
        """The clip's source size and whether it is conformed with ``fill``: when its aspect
        differs from the timeline's, or it is moved (reframe or transform), which then needs the
        size (F1)."""
        res = self.plan.resolution
        if source.frame_size is None:
            if moved:
                index = self.plan.sources.index(source)
                raise ExportError(
                    "source_size_required",
                    f"{location} has a reframe or transform, so the export needs source "
                    f"{source.id!r}'s picture size to frame it, and the plan gives none",
                    f"set sources[{index}].frame_size to the media's [width, height] (aar plan "
                    "produce sets it from the face track)",
                    location=location,
                    source=source.id,
                )
            return None, False
        size = (source.frame_size[0], source.frame_size[1])
        differs = size[0] * res.height != size[1] * res.width
        return size, differs or moved

    def _transform(self, index: int, seg: Segment, frame_size: tuple[int, int]) -> TransformSpec:
        times = [key.time for key in seg.reframe]
        if len(set(times)) != len(times):
            raise ExportError(
                "duplicate_keyframe_time",
                f"segment {index} has two reframe keyframes at the same frame; FCPXML keyframes "
                "need distinct times",
                "drop or move one of the keyframes",
                segment=index,
            )
        if not all(math.isfinite(key.zoom) for key in seg.reframe):
            raise ExportError(
                "non_finite_value",
                f"segment {index} has a reframe zoom that is not a finite number",
                "give each reframe zoom as a finite factor",
                segment=index,
            )
        res = self.plan.resolution
        spec = transform_spec(seg, frame_size, res.width, res.height)
        for k, (a, b) in enumerate(zip(seg.reframe, seg.reframe[1:], strict=False)):
            if a.zoom != b.zoom and (a.x != b.x or a.y != b.y):
                self.unexpressible(
                    "reframe_interpolation",
                    f"segments[{index}].reframe[{k}..{k + 1}]",
                    "zoom and centre change together; FCPXML interpolates position and scale "
                    "separately, so between these keyframes the path differs slightly from the "
                    "plan's (exact at each keyframe)",
                    "check the move in Resolve, or add keyframes",
                )
        if seg.transform is not None and (seg.transform.flip_h or seg.transform.flip_v):
            self.note(
                "flip",
                f"segments[{index}].transform",
                "flips are written as a negative scale on that axis",
                "check the flip in Resolve's Inspector after import",
            )
        return spec

    def _outro_clips(self) -> None:
        outro = self.plan.outro
        if outro is None:
            return
        rng = self.plan.outro_range()
        assert rng is not None
        source = self.plan.source_map()[outro.source]
        src_out = outro.src_out if outro.src_out is not None else source.duration
        assert src_out is not None, "a validated plan knows the outro's length"
        ref = self.layout.media[source.id]
        audio = _audio_index(self.plan, outro.audio_track)
        if audio is None:
            # As the compiler does: the outro has sound, so it needs a named track (never A1 by
            # default, which CCN mutes).
            raise ExportError(
                "outro_audio_track_missing",
                f"the outro has no audio_track in audio_tracks (audio_track={outro.audio_track!r})",
                "set outro.audio_track to an audio_tracks entry (e.g. the A5 bus) so the outro's "
                "sound has a track; it must not fall back to A1",
                audio_track=outro.audio_track,
            )
        record_in, record_out = rng
        lane = outro.track - 1
        frame_size, conform = self._framing("outro", source, False)
        framing = {"frame_size": frame_size, "conform": conform}
        if audio == outro.track or audio == 1:
            if audio != outro.track:
                self.unexpressible(
                    "outro_audio_track",
                    "outro.audio_track",
                    f"the outro's sound is on A1 but its picture on V{outro.track}; an audio-only "
                    "clip cannot sit in the primary storyline, so the sound rides with the picture",
                    "move the outro's sound to A1 in Resolve after import",
                )
            self.layout.clips.append(
                ClipSpec(
                    "outro",
                    lane,
                    record_in,
                    record_out,
                    outro.src_in,
                    src_out,
                    ref.src,
                    "all",
                    # The sound plays on A{audio}: its own track, or A1 riding with a V2+
                    # picture; that track's gain applies (F2).
                    audio,
                    volume_db=self._track_gain(audio),
                    name=ref.name,
                    **framing,  # type: ignore[arg-type]
                )
            )
            return
        self.layout.clips.append(
            ClipSpec(
                "outro",
                lane,
                record_in,
                record_out,
                outro.src_in,
                src_out,
                ref.src,
                "video",
                None,
                name=ref.name,
                **framing,  # type: ignore[arg-type]
            )
        )
        self.layout.clips.append(
            ClipSpec(
                "outro.audio_track",
                self.layout.audio_lane(audio),
                record_in,
                record_out,
                outro.src_in,
                src_out,
                ref.src,
                "audio",
                audio,
                volume_db=self._track_gain(audio),
                name=ref.name,
            )
        )

    def _clip(self, location: str) -> int:
        for index, clip in enumerate(self.layout.clips):
            if clip.location == location:
                return index
        raise AssertionError(f"no clip at {location}")

    def _fades(self) -> None:
        head = self.plan.head
        first = _first_content_segment(self.plan)
        clips = self.layout.clips
        if head is not None and head.audio_fade_frames > 0 and first is not None:
            index = self._clip(f"segments[{first[0]}]")
            clips[index] = _replace(clips[index], fade_in=head.audio_fade_frames)
        end = self.plan.end_fade_frames or 0
        if end > 0:
            if self.plan.outro is not None:
                carrier = next(
                    i
                    for i, c in enumerate(clips)
                    if c.location.startswith("outro") and c.enable != "video"
                )
            else:
                last = _last_content_segment(self.plan)
                assert last is not None, "a non-empty timeline without an outro has segments"
                carrier = self._clip(f"segments[{last[0]}]")
            clips[carrier] = _replace(clips[carrier], fade_out=end)

    def _check_fades(self) -> None:
        """A clip's fades (head, end and split-edit crossfades) fit in the sound it plays."""
        for clip in self.layout.clips:
            if clip.enable == "video":
                continue
            fade_in, fade_out = self.layout.fades(clip)
            sound_in, sound_out = self.layout.sound_range(clip)
            if fade_in + fade_out > sound_out - sound_in:
                raise ExportError(
                    "fade_too_long",
                    f"{clip.location}: fade-in {fade_in} + fade-out {fade_out} frames exceed "
                    f"the {sound_out - sound_in} frames of sound it plays",
                    "shorten head.audio_fade_frames or end_fade_frames",
                    location=clip.location,
                )

    # -- transitions ---------------------------------------------------------------------------

    def _alignment(self, outgoing: int, frames: int, location: str) -> tuple[Alignment, int]:
        """The compiler's own alignment rule (F7), or a refusal when no alignment has the media
        (FCPXML overlaps the clips, or a split edit plays the sound, from the handles)."""
        chosen = transition_alignment(self.plan, outgoing, frames)
        if chosen is None:
            tail = outgoing_tail(self.plan, outgoing)
            head = incoming_head(self.plan, outgoing)
            raise ExportError(
                "no_handle_for_transition",
                f"{location}: no alignment fits a {frames}-frame transition (outgoing tail "
                f"{tail}, incoming head {head}); the media under it must exist",
                "shorten the transition, give the source ranges more handle, or set the source "
                "'duration' (an unknown duration counts as no handle)",
                location=location,
                outgoing_tail=tail,
                incoming_head=head,
                frames=frames,
            )
        alignment, before = chosen
        return alignment, before  # type: ignore[return-value]

    def _placeholder(self, location: str, video: str) -> None:
        if video != "dissolve":
            self.unexpressible(
                "transition_type",
                location,
                f"a {video!r} transition is written as a Cross Dissolve of the same length",
                f"replace it with the {video} transition in Resolve",
            )

    def _transitions(self) -> None:
        self._head_transition()
        self._join_transitions()
        self._outro_transition()
        cuts = [t.cut for t in self.layout.transitions]
        if len(cuts) != len(set(cuts)):
            raise ExportError(
                "duplicate_transition",
                "two transitions land on the same cut of video track 1",
                "keep one transition per join",
            )

    def _head_transition(self) -> None:
        head = self.plan.head
        if head is None or head.video == "cut":
            return
        first = _first_content_segment(self.plan)
        if first is None:
            return
        _, track, record_in = first
        if track != 1:
            self.unexpressible(
                "head_transition",
                "head",
                f"the first segment is on video track {track}; FCPXML transitions live in a "
                "storyline, and connected clips are not wrapped in one here",
                "add the transition from black in Resolve",
            )
            return
        self.layout.transitions.append(
            TransitionSpec("head", "head", 1, record_in, head.video_frames, 0, "right", True, False)
        )
        self._placeholder("head.video", head.video)

    def _join_transitions(self) -> None:
        incoming = dict(self.plan.joins())
        seen: set[int] = set()
        for number, tx in enumerate(self.plan.transitions):
            location = f"transitions[{number}]"
            if tx.after in seen:
                raise ExportError(
                    "duplicate_transition",
                    f"{location}: segment {tx.after} already has a transition",
                    "keep one transition per join",
                    location=location,
                )
            seen.add(tx.after)
            video = tx.video != "cut"
            audio_frames = tx.audio.frames if tx.audio is not None else 0
            if not video and audio_frames == 0:
                continue  # a cut with no audio transition: nothing to write
            nxt = incoming.get(tx.after)
            seg = self.plan.segments[tx.after] if tx.after < len(self.plan.segments) else None
            if nxt is None or seg is None:
                self.unexpressible(
                    "transition",
                    location,
                    f"segment {tx.after} has no following segment on its track",
                    "fix 'after' to the outgoing segment of a join",
                )
                continue
            if seg.track != 1:
                self.unexpressible(
                    "transition",
                    location,
                    f"the join is on video track {seg.track}; FCPXML transitions live in a "
                    "storyline, and connected clips are not wrapped in one here",
                    "add the transition in Resolve",
                )
                continue
            cut = self.ranges[tx.after][1]
            if self.ranges[nxt][0] != cut:
                self.unexpressible(
                    "transition",
                    location,
                    f"segments {tx.after} and {nxt} do not abut (gap from {cut} to "
                    f"{self.ranges[nxt][0]}); a transition needs both clips at the cut",
                    "close the gap, or add the transition in Resolve",
                )
                continue
            frames = tx.video_frames if video else audio_frames
            alignment, before = self._alignment(tx.after, frames, location)
            form: AudioCrossfade = "transition" if video else self.layout.audio_crossfade
            self.layout.transitions.append(
                TransitionSpec(
                    location, "join", 1, cut, frames, before, alignment, video, True, form
                )
            )
            if video:
                self._placeholder(f"{location}.video", tx.video)
                if audio_frames != frames:
                    plan_audio = f"{audio_frames} frames" if tx.audio is not None else "none"
                    self.unexpressible(
                        "audio_crossfade_length",
                        f"{location}.audio",
                        f"one FCPXML transition carries both picture and sound, so the audio "
                        f"crossfade is {frames} frames like the dissolve (plan: {plan_audio})",
                        "adjust the audio crossfade in Resolve if it must differ",
                    )
            if tx.audio is not None and tx.audio.type != DEFAULT_AUDIO_TRANSITION:
                self.unexpressible(
                    "audio_transition_type",
                    f"{location}.audio.type",
                    f"audio transition {tx.audio.type!r} is written as FCPXML's audio crossfade",
                    f"set it to {tx.audio.type!r} in Resolve",
                )

    def _outro_transition(self) -> None:
        outro = self.plan.outro
        if outro is None:
            return
        tx = outro.transition
        if tx.audio is not None:
            self.unexpressible(
                "outro_audio_transition",
                "outro.transition.audio",
                "the outro's sound sits on its own audio track, so there is no same-track sound "
                f"to cross-fade ({tx.audio.type}, {tx.audio.frames} frames)",
                "add fades on the two tracks in Resolve if wanted",
            )
        if tx.video == "cut":
            return
        last = _last_content_segment(self.plan)
        content_end = self.plan.content_frames()
        if (
            outro.track != 1
            or last is None
            or last[1] != 1
            or self.ranges[last[0]][1] != content_end
        ):
            self.unexpressible(
                "outro_transition",
                "outro.transition",
                "the outro and the last content segment are not neighbours on video track 1, "
                "and FCPXML transitions live in a storyline",
                "add the transition into the outro in Resolve",
            )
            return
        alignment, before = self._alignment(last[0], tx.video_frames, "outro.transition")
        self.layout.transitions.append(
            TransitionSpec(
                "outro.transition",
                "outro",
                1,
                content_end,
                tx.video_frames,
                before,
                alignment,
                True,
                False,
            )
        )
        self._placeholder("outro.transition.video", tx.video)

    def _check_overlaps(self) -> None:
        """Two transitions on one spine clip must not overlap: the part of the incoming one after
        its cut plus the part of the outgoing one before its cut fit in the clip."""
        by_cut = {t.cut: t for t in self.layout.transitions}
        for clip in self.layout.clips:
            if clip.lane != 0:
                continue
            into = by_cut.get(clip.record_in)
            out = by_cut.get(clip.record_out)
            used = (into.tail_part if into else 0) + (out.head_part if out else 0)
            if used > clip.record_out - clip.record_in:
                raise ExportError(
                    "transitions_overlap",
                    f"{clip.location}: its transitions need {used} frames of a "
                    f"{clip.record_out - clip.record_in}-frame clip, so they would overlap",
                    "shorten the transitions or lengthen the clip",
                    location=clip.location,
                )

    # -- titles and markers --------------------------------------------------------------------

    def _inside(self, location: str, record_in: int, record_out: int) -> None:
        if not 0 <= record_in < record_out <= self.layout.total_frames:
            raise ExportError(
                "item_outside_timeline",
                f"{location} spans {record_in}..{record_out}, outside the timeline "
                f"(0..{self.layout.total_frames})",
                "keep titles and markers inside the timeline's record range",
                location=location,
            )

    def _titles(self) -> None:
        lane = self.layout.title_lane
        for index, title in enumerate(self.plan.titles):
            location = f"titles[{index}]"
            self._inside(location, title.record_in, title.record_out)
            self._add_title(
                location,
                title.kind,
                title.record_in,
                title.record_out,
                title.lines,
                title.template,
                lane,
            )
        outro = self.plan.outro
        if outro is not None and outro.sting is not None:
            start = self.plan.content_frames() + outro.sting.start
            end = start + outro.sting.frames
            self._inside("outro.sting", start, end)
            self._add_title(
                "outro.sting", "sting", start, end, outro.sting.lines, outro.sting.template, lane
            )

    def _add_title(
        self,
        location: str,
        kind: str,
        record_in: int,
        record_out: int,
        lines: list[str],
        template: str | None,
        lane: int,
    ) -> None:
        self.layout.titles.append(
            TitleSpec(
                location,
                kind,
                lane,
                record_in,
                record_out,
                tuple(lines),
                template,
                " / ".join(lines),
            )
        )
        action = (
            f"replace it with template {template!r}" if template else "replace it with a template"
        )
        self.unexpressible(
            "title_template",
            location,
            f"exported as a Basic Title with its text; Resolve does not map it to a Fusion "
            f"template ({kind})",
            action,
        )

    def _markers(self) -> None:
        for index, marker in enumerate(self.plan.markers):
            location = f"markers[{index}]"
            self._inside(location, marker.record, marker.record + marker.duration)
            self.layout.markers.append(
                MarkerSpec(location, marker.record, marker.duration, marker.name, marker.note)
            )
            if marker.color is not None:
                self.unexpressible(
                    "marker_color",
                    f"{location}.color",
                    f"FCPXML markers have no colour ({marker.color})",
                    f"set the marker colour to {marker.color} in Resolve",
                )

    # -- assets --------------------------------------------------------------------------------

    def _assets(self) -> None:
        """One asset per distinct media URL, in first-use order. Its duration is the source's
        when known, else the furthest frame any clip reads (reported)."""
        sources = self.plan.source_map()
        by_cut = {t.cut: t for t in self.layout.transitions}
        reach: dict[str, int] = {}
        for clip in self.layout.clips:
            out = by_cut.get(clip.record_out) if clip.lane == 0 else None
            end = clip.src_out + (out.tail_part if out is not None else 0)
            reach[clip.src] = max(reach.get(clip.src, 0), end)
        for src in reach:
            users = [
                s
                for s in self.plan.sources
                if s.id in self.layout.media and self.layout.media[s.id].src == src
            ]
            known = [sources[s.id].duration for s in users if sources[s.id].duration is not None]
            duration = max(known) if known else reach[src]
            if not known:
                self.note(
                    "asset_duration",
                    ", ".join(f"sources[{self.plan.sources.index(s)}]" for s in users),
                    f"source duration unknown; the asset is declared {duration} frames (the "
                    "furthest frame used)",
                    "set the source's duration in the plan",
                )
            self.layout.assets.append(AssetSpec(src, self.layout.media[users[0].id].name, duration))

    # -- report --------------------------------------------------------------------------------

    def _report_plan_features(self) -> None:
        plan = self.plan
        if plan.template is not None:
            self.unexpressible(
                "template",
                "template",
                f"FCPXML builds a new timeline; it does not start from template {plan.template!r}, "
                "so its tracks, buses, Fusion titles and settings are not carried",
                "import the timeline, then bring its clips into a branch of the template",
            )
        used = {c.audio_track for c in self.layout.clips if c.audio_track is not None}
        for number, track in enumerate(plan.audio_tracks, start=1):
            location = f"audio_tracks[{number - 1}]"
            if track.mute:
                self.unexpressible(
                    "audio_mute",
                    f"{location}.mute",
                    f"A{number} {track.name!r} is muted in the plan; exported audible (the CCN "
                    "topology feeds the bus from muted tracks, so sound is never dropped)",
                    f"mute {track.name!r} in Resolve's Fairlight page if it must be silent",
                )
            if track.bus is not None:
                self.unexpressible(
                    "audio_bus",
                    f"{location}.bus",
                    f"A{number} {track.name!r} feeds bus {track.bus!r}; FCPXML has no buses or "
                    "Fairlight routing",
                    f"route {track.name!r} to {track.bus!r} in Fairlight",
                )
                target = next((t for t in plan.audio_tracks if t.name == track.bus), None)
                if target is not None and target.gain_db != 0.0:
                    self.unexpressible(
                        "bus_gain",
                        f"{location}.bus",
                        f"bus {track.bus!r}'s gain {target.gain_db:+g} dB is applied only to "
                        f"clips placed on {track.bus!r}, not to A{number} {track.name!r}'s clips "
                        "that feed it (FCPXML has no buses)",
                        f"set the gain on {track.bus!r} in Fairlight once the routing is in place",
                    )
            if track.gain_db != 0.0:
                self.note(
                    "audio_track_gain",
                    f"{location}.gain_db",
                    f"A{number} gain {track.gain_db:+g} dB is folded into each clip's volume on "
                    "that track (FCPXML has no track gain)",
                    "none, or move it to the track fader in Resolve",
                )
            if number not in used:
                self.unexpressible(
                    "audio_track_empty",
                    location,
                    f"no clip sits on A{number} {track.name!r}; FCPXML cannot declare an empty "
                    "track",
                    "add the track in Resolve after import if it is needed",
                )
        for index, source in enumerate(plan.sources):
            color = source.color
            if color is None:
                continue
            location = f"sources[{index}].color"
            if color.input_color_space is not None:
                self.unexpressible(
                    "input_color_space",
                    f"{location}.input_color_space",
                    f"input colour space {color.input_color_space!r} of source {source.id!r}",
                    "set it in Resolve (never on BRAW media)",
                )
            if color.cdl is not None:
                cdl = color.cdl
                self.unexpressible(
                    "cdl",
                    f"{location}.cdl",
                    f"CDL of source {source.id!r}: slope {list(cdl.slope)}, offset "
                    f"{list(cdl.offset)}, power {list(cdl.power)}, saturation {cdl.saturation:g}",
                    "apply it on the Color page",
                )
            if color.lut is not None:
                self.unexpressible(
                    "lut",
                    f"{location}.lut",
                    f"LUT {color.lut!r} of source {source.id!r}",
                    "apply it on the Color page",
                )
        if plan.captions:
            self.unexpressible(
                "captions",
                "captions",
                f"{len(plan.captions)} caption(s) are not exported",
                "captions are burned in with ffmpeg (S13e)",
            )
        if plan.caption_style is not None:
            self.unexpressible(
                "caption_style",
                "caption_style",
                "the caption style is not exported with the captions",
                "captions are burned in with ffmpeg (S13e)",
            )
        if plan.head is not None and plan.head.audio_fade_frames > 0:
            self.note(
                "audio_fade",
                "head.audio_fade_frames",
                f"{plan.head.audio_fade_frames}-frame fade-in on the first segment's volume",
            )
        if plan.end_fade_frames:
            self.note(
                "audio_fade",
                "end_fade_frames",
                f"{plan.end_fade_frames}-frame fade-out on the last clip's volume",
            )
        if any(t.kind == "outro" for t in self.layout.transitions):
            self.note(
                "outro_sound",
                "outro.transition",
                "the dissolve overlaps the last segment's picture into its tail; its sound still "
                "ends at the outro (audioStart/audioDuration)",
                "check it in Resolve",
            )
        for clip in self.layout.clips:
            if clip.enable == "video" or self.layout.sound_range(clip) == self.layout.raw_range(
                clip
            ):
                continue
            ends = (
                self.layout.spine_transition_at(clip.record_in),
                self.layout.spine_transition_at(clip.record_out),
            )
            for spec in ends:
                if spec is not None and spec.audio:
                    self.unexpressible(
                        "split_audio_crossfade",
                        spec.location,
                        f"{clip.location} needs split audio timing (audioStart/audioDuration) for "
                        "a video-only transition on its other end; Final Cut Pro ignores an "
                        "audio crossfade on such a clip (Resolve's import is unconfirmed)",
                        "check this audio crossfade after import and re-add it if missing",
                    )
        if any(t.form == "split" for t in self.layout.transitions):
            self.note(
                "audio_crossfade",
                "transitions",
                "audio-only crossfades are split edits: the picture cuts at the plan's cut, each "
                "clip's sound runs on under the other's (audioStart/audioDuration) and fades out "
                "or in over the crossfade (adjust-volume fadeIn/fadeOut)",
                "check them after import; --audio-crossfade transition writes spine transitions",
            )
        for source in plan.sources:
            if source.id in self.layout.media and source.frame_size is None:
                self.note(
                    "frame_size",
                    f"sources[{plan.sources.index(source)}].frame_size",
                    f"source {source.id!r} has no frame_size, so its clips get Resolve's default "
                    "framing (fit: letterboxed if its aspect differs from the timeline's)",
                    "set frame_size to frame it with fill",
                )
        self.note(
            "source_timecode",
            "sources",
            "each asset starts at 0s, so source frames count from each file's first frame",
            "check that clips land on the planned frames after import",
        )


def _replace(clip: ClipSpec, **changes: object) -> ClipSpec:
    from dataclasses import replace

    return replace(clip, **changes)  # type: ignore[arg-type]


def build_layout(
    plan: EditPlan,
    *,
    media_map: MediaMap | None = None,
    timeline_start: str = DEFAULT_TIMELINE_START,
    audio_crossfade: AudioCrossfade = "split",
) -> Layout:
    """Every export decision for ``plan``, or a typed :class:`ExportError`."""
    return _Builder(plan, media_map, timeline_start, audio_crossfade).build()
