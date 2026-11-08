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
* A transition of ``d`` frames at cut ``c`` is centred when the handles allow it (``d // 2``
  frames before the cut), else it starts on the cut (``right``) or ends on it (``left``), like
  the compiler. FCPXML overlaps the clips under it, so the handles are checked exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from aar.host.pathmap import HostPathMap
from aar.plan.apply.compile import (
    _audio_index,
    _first_content_segment,
    _fits,
    _last_content_segment,
    incoming_head,
    outgoing_tail,
    title_track,
)
from aar.plan.export.errors import ExportError
from aar.plan.export.media import MediaRef, media_ref
from aar.plan.export.timebase import Timebase
from aar.plan.model import EditPlan, ReframeKey, Segment, Transform

DEFAULT_TIMELINE_START = "01:00:00:00"
# Characters XML 1.0 cannot carry, plus CR, which a parser turns into LF (so it would not
# round-trip): C0 controls other than tab and LF, U+FFFE/U+FFFF and lone surrogates.
_XML_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\ufffe\uffff\ud800-\udfff]")
DEFAULT_AUDIO_TRANSITION = "Cross Fade 0 dB"

Enable = Literal["all", "video", "audio"]
TransitionKind = Literal["head", "join", "outro"]
Alignment = Literal["right", "center", "left"]


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
        """The spine transition at record frame ``cut`` (cuts are unique on track 1)."""
        for spec in self.transitions:
            if spec.cut == cut:
                return spec
        return None

    def raw_range(self, clip: ClipSpec) -> tuple[int, int]:
        """The record frames a clip covers in the file: its plan range, widened under the
        transitions on its ends (FCPXML overlaps both neighbours under a transition)."""
        if clip.lane != 0:
            return clip.record_in, clip.record_out
        into, out = self.transition_at(clip.record_in), self.transition_at(clip.record_out)
        return (
            clip.record_in - (into.head_part if into is not None else 0),
            clip.record_out + (out.tail_part if out is not None else 0),
        )

    def sound_range(self, clip: ClipSpec) -> tuple[int, int]:
        """The record frames a clip's sound plays: its plan range, widened only under a
        transition that carries an audio crossfade."""
        if clip.lane != 0:
            return clip.record_in, clip.record_out
        into, out = self.transition_at(clip.record_in), self.transition_at(clip.record_out)
        return (
            clip.record_in - (into.head_part if into is not None and into.audio else 0),
            clip.record_out + (out.tail_part if out is not None and out.audio else 0),
        )

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


def transform_spec(seg: Segment, width: int, height: int) -> TransformSpec:
    """A segment's static transform and reframe keyframes as one FCPXML adjust-transform.

    The reframe moves within the transformed picture (DESIGN, EditPlan v2): ``(x, y)`` is the
    point of the transformed picture shown at the frame centre and ``zoom`` scales about it. With
    ``w = ((x - 0.5) * width, (0.5 - y) * height)`` in pixels from the centre, y up, a picture
    point ``p`` lands at ``kz * (T(p) - w)``, so the position is ``kz * ((pan, tilt) - w)`` and the
    scale ``kz * zoom`` (negated per axis for a flip). Positions are percent of the frame height.
    """
    tf = seg.transform or Transform()
    flip = (-1.0 if tf.flip_h else 1.0, -1.0 if tf.flip_v else 1.0)

    def at(key: ReframeKey) -> tuple[tuple[float, float], tuple[float, float]]:
        kz = key.zoom
        wx = (key.x - 0.5) * width
        wy = (0.5 - key.y) * height
        position = (kz * (tf.pan - wx) / height * 100.0, kz * (tf.tilt - wy) / height * 100.0)
        scale = (kz * tf.zoom * flip[0], kz * tf.zoom * flip[1])
        return position, scale

    keys = list(seg.reframe) or [ReframeKey(time=0)]
    position, scale = at(keys[0])
    keyframes = tuple(Keyframe(key.time, *at(key)) for key in keys) if len(keys) > 1 else ()
    return TransformSpec(position, scale, tf.rotation_deg, keyframes)


# -- the build ---------------------------------------------------------------------------------


class _Builder:
    def __init__(
        self,
        plan: EditPlan,
        media_map: HostPathMap | None,
        timeline_start: str,
    ) -> None:
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
        self.layout = Layout(plan, self.timebase, tc_start, timeline_start, total)

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
            transform = None
            if seg.transform is not None or seg.reframe:
                transform = self._transform(index, seg)
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
                )
            )
        self._outro_clips()

    def _transform(self, index: int, seg: Segment) -> TransformSpec:
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
        spec = transform_spec(seg, res.width, res.height)
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
                    outro.track,
                    volume_db=self._track_gain(outro.track),
                    name=ref.name,
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
        for clip in clips:
            if clip.fade_in + clip.fade_out > clip.record_out - clip.record_in:
                raise ExportError(
                    "fade_too_long",
                    f"{clip.location}: fade-in {clip.fade_in} + fade-out {clip.fade_out} frames "
                    f"exceed the clip's {clip.record_out - clip.record_in} frames",
                    "shorten head.audio_fade_frames or end_fade_frames",
                    location=clip.location,
                )

    # -- transitions ---------------------------------------------------------------------------

    def _alignment(self, outgoing: int, frames: int, location: str) -> tuple[Alignment, int]:
        """The compiler's alignment order (centre, right, left) with FCPXML's exact handles: a
        centred transition puts ``frames // 2`` before the cut and the rest after it."""
        tail = outgoing_tail(self.plan, outgoing)
        head = incoming_head(self.plan, outgoing)
        before = frames // 2
        if _fits(tail, frames - before) and _fits(head, before):
            return "center", before
        if _fits(tail, frames):
            return "right", 0
        if _fits(head, frames):
            return "left", frames
        raise ExportError(
            "no_handle_for_transition",
            f"{location}: no alignment fits a {frames}-frame transition (outgoing tail {tail}, "
            f"incoming head {head}); FCPXML overlaps the clips, so the media must exist",
            "shorten the transition, give the source ranges more handle, or set the source "
            "'duration' (an unknown duration counts as no handle)",
            location=location,
            outgoing_tail=tail,
            incoming_head=head,
            frames=frames,
        )

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
            self.layout.transitions.append(
                TransitionSpec(location, "join", 1, cut, frames, before, alignment, video, True)
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
                    "the template provides it",
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
                self.layout.transition_at(clip.record_in),
                self.layout.transition_at(clip.record_out),
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
    media_map: HostPathMap | None = None,
    timeline_start: str = DEFAULT_TIMELINE_START,
) -> Layout:
    """Every export decision for ``plan``, or a typed :class:`ExportError`."""
    return _Builder(plan, media_map, timeline_start).build()
