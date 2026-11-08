"""The self-check: read an FCPXML file back and require it to equal the plan, without Resolve.

:func:`verify_fcpxml` decodes the file (:mod:`aar.plan.export.reader`, which also fails anything
the writer never emits), rebuilds the timeline in plan terms and compares it with the layout built
from the plan:

* the header: version 1.10, frame duration, tcStart, length, size, format name and names;
* the spine: elements contiguous from 0 to the plan's length, each transition overlapping its
  neighbours by exactly its length, with the writer's effect (name and uid);
* every clip (track, record in/out, source in/out, media URL, which streams, where its sound
  plays, volume, fades, conform, transform and keyframes), title, marker and asset (start,
  duration), and every clip's media inside its asset and its plan source;
* each moved clip's crop window, recomputed from the XML (conform, scale, rotation, position) and
  the source size, against the plan's reframe and transform: within 0.5 source pixels;
* and, straight from the plan, each segment's (track, record in/out, source in/out, media) and
  each exported transition's (cut, length): equal, not close. A split-edit crossfade is read back
  from the two clips' sound ranges and fades.

A transition's overlap in the file says where media plays but not where the plan's cut is
(FCPXML has no alignment), so the decode reads each window with the alignment the plan's handles
give (the compiler's rule). A wrong window or overlap leaves a mismatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from aar.plan.export.fcpxml import (
    AUDIO_CROSSFADE,
    BASIC_TITLE,
    CROSS_DISSOLVE,
    FCPXML_VERSION,
    format_name,
)
from aar.plan.export.layout import ClipSpec, Layout, TransformSpec, TransitionSpec
from aar.plan.export.reader import Decoded, RawClip, read_fcpxml
from aar.plan.export.timebase import frame_duration_text
from aar.plan.model import Transform

#: How far a recomputed crop window may sit from the plan's, in source pixels (F1).
WINDOW_TOLERANCE_PX = 0.5


@dataclass
class VerifyResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "problems": list(self.problems), "checked": dict(self.checked)}


@dataclass(frozen=True)
class ClipView:
    """A clip in plan terms, decoded or expected, for an exact comparison."""

    lane: int
    record_in: int
    record_out: int
    src_in: int
    src_out: int
    src: str | None
    enable: str
    sound: tuple[int, int] | None  # source frames where the sound plays; None for picture-only
    volume_db: float
    fade_in: int
    fade_out: int
    conform: bool
    transform: TransformSpec | None
    name: str
    note: str | None


def _expected_view(clip: ClipSpec, layout: Layout) -> ClipView:
    sound = None
    if clip.enable != "video":
        sound_in, sound_out = layout.sound_range(clip)
        shift = clip.src_in - clip.record_in  # record -> source frames for this clip
        sound = (sound_in + shift, sound_out + shift)
    fade_in, fade_out = layout.fades(clip) if clip.enable != "video" else (0, 0)
    return ClipView(
        clip.lane,
        clip.record_in,
        clip.record_out,
        clip.src_in,
        clip.src_out,
        clip.src,
        clip.enable,
        sound,
        clip.volume_db,
        fade_in,
        fade_out,
        clip.conform,
        clip.transform,
        clip.name,
        clip.note,
    )


def _decoded_view(raw: RawClip, head_in: int, tail_out: int) -> ClipView:
    length = raw.raw_out - raw.raw_in
    src_in = raw.start + head_in
    src_out = raw.start + length - tail_out
    sound = None
    if raw.enable != "video":
        sound = raw.sound if raw.sound is not None else (raw.start, raw.start + length)
    transform = raw.transform
    if transform is not None and transform.keyframes:
        # Keyframe times are clip-local source frames in the file; the plan counts from src_in.
        transform = TransformSpec(
            transform.position,
            transform.scale,
            transform.rotation,
            tuple(type(k)(k.time - src_in, k.position, k.scale) for k in transform.keyframes),
        )
    return ClipView(
        raw.lane,
        raw.raw_in + head_in,
        raw.raw_out - tail_out,
        src_in,
        src_out,
        raw.src,
        raw.enable,
        sound,
        raw.volume_db,
        raw.fade_in,
        raw.fade_out,
        raw.conform == "fill",
        transform,
        raw.name,
        raw.note,
    )


# -- crop windows (F1) -------------------------------------------------------------------------

Window = tuple[float, float, float, float]  # centre x, centre y, width, height, in source px


def _filled(frame_size: tuple[int, int], width: int, height: int) -> tuple[float, float]:
    fill = max(width / frame_size[0], height / frame_size[1])
    return frame_size[0] * fill, frame_size[1] * fill


def window_from_file(
    position: tuple[float, float],
    scale: tuple[float, float],
    rotation: float,
    frame_size: tuple[int, int],
    width: int,
    height: int,
) -> Window:
    """The source region a filled clip shows, from its FCPXML values: the frame centre pulled
    back through position, rotation and scale, then the frame's size divided by the scale."""
    cw, ch = _filled(frame_size, width, height)
    px, py = position[0] * height / 100.0, position[1] * height / 100.0
    angle = math.radians(rotation)
    # Undo the translation, then the rotation (R^-1 = R(-angle)), then the scale.
    rx = math.cos(angle) * -px + math.sin(angle) * -py
    ry = -math.sin(angle) * -px + math.cos(angle) * -py
    ux, uy = rx / scale[0], ry / scale[1]
    to_px = frame_size[0] / cw
    return (
        (ux / cw + 0.5) * frame_size[0],
        (0.5 - uy / ch) * frame_size[1],
        width / abs(scale[0]) * to_px,
        height / abs(scale[1]) * to_px,
    )


def window_from_plan(
    x: float,
    y: float,
    zoom: float,
    transform: Transform,
    frame_size: tuple[int, int],
    width: int,
    height: int,
) -> Window:
    """The source region the plan asks for: a crop of the fill window divided by the zoom,
    centred on ``(x, y)`` of the source, moved by the transform's pan and tilt (which shift the
    picture under the frame)."""
    cw, _ = _filled(frame_size, width, height)
    scale = zoom * transform.zoom
    flip = (-1.0 if transform.flip_h else 1.0, -1.0 if transform.flip_v else 1.0)
    angle = math.radians(transform.rotation_deg)
    # Pan/tilt in timeline px, taken back through the rotation and the (flipped) scale.
    dx = math.cos(angle) * transform.pan + math.sin(angle) * transform.tilt
    dy = -math.sin(angle) * transform.pan + math.cos(angle) * transform.tilt
    to_px = frame_size[0] / cw
    return (
        x * frame_size[0] - dx / (scale * flip[0]) * to_px,
        y * frame_size[1] + dy / (scale * flip[1]) * to_px,
        width / scale * to_px,
        height / scale * to_px,
    )


class _Check:
    def __init__(self, decoded: Decoded, layout: Layout) -> None:
        self.dec = decoded
        self.layout = layout
        self.problems: list[str] = list(decoded.problems)
        self.decoded_transitions: list[tuple[int, int, bool, bool]] = []
        self.views: list[ClipView] = []
        self.windows = 0

    def want(self, ok: bool, message: str) -> None:
        if not ok:
            self.problems.append(message)

    def header(self) -> None:
        dec, layout = self.dec, self.layout
        plan = layout.plan
        self.want(dec.version == FCPXML_VERSION, f"version is {dec.version!r}, not 1.10")
        want_fd = frame_duration_text(layout.timebase)
        self.want(
            dec.frame_duration == want_fd,
            f"frameDuration is {dec.frame_duration!r}, not {want_fd!r}",
        )
        self.want(
            dec.tc_start == layout.tc_start,
            f"tcStart is frame {dec.tc_start}, not {layout.tc_start}",
        )
        self.want(
            dec.duration == layout.total_frames,
            f"sequence duration is {dec.duration} frames, not {layout.total_frames}",
        )
        res = plan.resolution
        size = (res.width, res.height)
        self.want(
            (dec.width, dec.height) == size, f"format is {dec.width}x{dec.height}, not {size}"
        )
        name = format_name(layout.timebase, res.width, res.height)
        self.want(dec.format_name == name, f"format name is {dec.format_name!r}, not {name!r}")
        self.want(
            dec.project == plan.timeline, f"project is {dec.project!r}, not {plan.timeline!r}"
        )
        self.want(dec.event == plan.project, f"event is {dec.event!r}, not {plan.project!r}")

    def spine(self) -> None:
        """Contiguity, transition overlaps and effects, and the decode of each primary clip."""
        entries = self.dec.spine
        expected = {t.window_start: t for t in self.layout.transitions if t.form == "transition"}
        matched: dict[int, TransitionSpec] = {}
        for k, entry in enumerate(entries):
            if entry.kind != "transition":
                continue
            raw = self.dec.transitions[entry.index]
            spec = expected.get(raw.offset)
            if spec is None or spec.frames != raw.frames:
                self.problems.append(
                    f"transition at record {raw.offset} ({raw.frames} frames) is not in the plan"
                )
                continue
            matched[k] = spec
            nxt = entries[k + 1] if k + 1 < len(entries) else None
            prev = entries[k - 1] if k > 0 else None
            self.want(
                nxt is not None and nxt.kind == "clip" and nxt.raw_in == raw.offset,
                f"transition at record {raw.offset}: the next clip does not start under it",
            )
            if prev is None:
                self.want(
                    spec.kind == "head" and raw.offset == 0,
                    f"transition at record {raw.offset} opens the spine but is not a head at 0",
                )
            else:
                self.want(
                    prev.kind != "transition" and prev.raw_out == raw.offset + raw.frames,
                    f"transition at record {raw.offset}: the element before it does not end "
                    f"at {raw.offset + raw.frames} (FCPXML overlap)",
                )
            want_effect = (
                CROSS_DISSOLVE if spec.video else None,
                AUDIO_CROSSFADE if spec.audio else None,
            )
            self.want(
                (raw.video_effect, raw.audio_effect) == want_effect,
                f"transition at record {raw.offset}: effects {raw.video_effect}/"
                f"{raw.audio_effect}, not {want_effect} (name, uid)",
            )
            self.decoded_transitions.append(
                (raw.offset + spec.head_part, raw.frames, raw.video, raw.audio)
            )
        end = 0
        for k, entry in enumerate(entries):
            if entry.kind == "transition":
                continue
            if k == 0 or entries[k - 1].kind != "transition":
                self.want(
                    entry.raw_in == end,
                    f"spine {entry.kind} at record {entry.raw_in} does not follow the previous "
                    f"element's end {end}",
                )
            end = entry.raw_out
            if entry.kind != "clip":
                continue
            into = matched.get(k - 1)
            out = matched.get(k + 1)
            raw_clip = self.dec.clips[entry.index]
            self.views.append(
                _decoded_view(
                    raw_clip,
                    into.head_part if into is not None else 0,
                    out.tail_part if out is not None else 0,
                )
            )
        self.want(
            end == self.layout.total_frames,
            f"the spine ends at record {end}, not {self.layout.total_frames}",
        )
        for raw_clip in self.dec.clips:
            if raw_clip.lane != 0:
                self.views.append(_decoded_view(raw_clip, 0, 0))

    def split_crossfades(self) -> None:
        """Read each split-edit crossfade back from the two clips: the outgoing sound runs on by
        the part after the cut, the incoming starts early by the part before it, and both fade
        over the whole crossfade."""
        by_out = {v.record_out: v for v in self.views if v.lane == 0}
        by_in = {v.record_in: v for v in self.views if v.lane == 0}
        for spec in self.layout.transitions:
            if spec.form != "split":
                continue
            out, into = by_out.get(spec.cut), by_in.get(spec.cut)
            if out is None or into is None or out.sound is None or into.sound is None:
                self.problems.append(f"{spec.location}: no clips with sound meet at {spec.cut}")
                continue
            after = out.record_out + (out.sound[1] - out.src_out) - spec.cut
            before = spec.cut - (into.record_in + (into.sound[0] - into.src_in))
            frames = before + after
            self.want(
                out.fade_out == into.fade_in == frames and frames > 0,
                f"{spec.location}: the split crossfade's fades ({out.fade_out}, {into.fade_in}) "
                f"do not span its {frames} frames",
            )
            self.decoded_transitions.append((spec.cut, frames, False, True))

    def clips(self) -> None:
        want = sorted(
            (_expected_view(c, self.layout) for c in self.layout.clips),
            key=lambda v: (v.lane, v.record_in),
        )
        got = sorted(self.views, key=lambda v: (v.lane, v.record_in))
        want_keys = {(v.lane, v.record_in): v for v in want}
        got_keys = {(v.lane, v.record_in): v for v in got}
        self.want(len(got) == len(got_keys), "two clips share a lane and record start")
        for key, expected in want_keys.items():
            found = got_keys.get(key)
            if found is None:
                self.problems.append(f"no clip on lane {key[0]} at record {key[1]}")
            elif found != expected:
                diffs = [
                    f"{name} {getattr(found, name)!r} != {getattr(expected, name)!r}"
                    for name in ClipView.__dataclass_fields__
                    if getattr(found, name) != getattr(expected, name)
                ]
                self.problems.append(
                    f"clip on lane {key[0]} at record {key[1]}: " + "; ".join(diffs)
                )
        for key in got_keys.keys() - want_keys.keys():
            self.problems.append(f"unexpected clip on lane {key[0]} at record {key[1]}")
        want_transitions = sorted(
            (t.cut, t.frames, t.video, t.audio) for t in self.layout.transitions
        )
        self.want(
            sorted(self.decoded_transitions) == want_transitions,
            f"transitions {sorted(self.decoded_transitions)} != {want_transitions}",
        )

    def media_bounds(self) -> None:
        """Every clip reads only frames its asset declares, and its plan source has (F3)."""
        plan = self.layout.plan
        durations: dict[str, list[int]] = {}
        for source in plan.sources:
            ref = self.layout.media.get(source.id)
            if ref is not None and source.duration is not None:
                durations.setdefault(ref.src, []).append(source.duration)
        assets = {asset.src: asset for asset in self.dec.assets.values()}
        for raw in self.dec.clips:
            length = raw.raw_out - raw.raw_in
            reads = [(raw.start, raw.start + length)]
            if raw.sound is not None and raw.enable != "video":
                reads.append(raw.sound)
            first = min(a for a, _ in reads)
            last = max(b for _, b in reads)
            where = f"clip on lane {raw.lane} at record {raw.raw_in}"
            self.want(first >= 0, f"{where} reads before its media's first frame ({first})")
            asset = assets.get(raw.src or "")
            if asset is not None and asset.duration is not None:
                self.want(
                    last <= asset.duration,
                    f"{where} reads to frame {last}, past its asset's {asset.duration} frames",
                )
            for duration in durations.get(raw.src or "", []):
                self.want(
                    last <= duration,
                    f"{where} reads to frame {last}, past the plan source's {duration} frames",
                )

    def assets(self) -> None:
        got_assets = sorted((a.src, a.start, a.duration) for a in self.dec.assets.values())
        want_assets = sorted((a.src, 0, a.duration) for a in self.layout.assets)
        self.want(got_assets == want_assets, f"assets {got_assets} != {want_assets}")

    def crop_windows(self) -> None:
        """Each moved or filled clip's crop window, from the file, against the plan's (F1)."""
        plan = self.layout.plan
        res = plan.resolution
        views = {(v.lane, v.record_in): v for v in self.views}
        for clip in self.layout.clips:
            if not clip.conform or clip.frame_size is None:
                continue
            view = views.get((clip.lane, clip.record_in))
            if view is None:
                continue  # reported as a missing clip
            if not view.conform:
                self.problems.append(f"{clip.location}: no adjust-conform fill in the file")
                continue
            seg = None
            if clip.location.startswith("segments["):
                seg = plan.segments[int(clip.location[9:].rstrip("]"))]
            transform = (seg.transform if seg is not None else None) or Transform()
            keys = list(seg.reframe) if seg is not None and seg.reframe else []
            spec = view.transform
            if spec is None:
                file_keys = [(0, (0.0, 0.0), (1.0, 1.0), 0.0)]
            elif spec.keyframes:
                file_keys = [(k.time, k.position, k.scale, spec.rotation) for k in spec.keyframes]
            else:
                file_keys = [(0, spec.position, spec.scale, spec.rotation)]
            plan_keys = [(k.time, k.x, k.y, k.zoom) for k in keys] or [(0, 0.5, 0.5, 1.0)]
            if len(plan_keys) == 1 and len(file_keys) == 1:
                # One key is a static transform for the whole clip: its time does not matter.
                file_keys = [(plan_keys[0][0], *file_keys[0][1:])]
            if len(file_keys) != len(plan_keys):
                self.problems.append(
                    f"{clip.location}: {len(file_keys)} keyframe(s) in the file, "
                    f"{len(plan_keys)} in the plan"
                )
                continue
            for (time, position, scale, rotation), (ptime, x, y, zoom) in zip(
                file_keys, plan_keys, strict=True
            ):
                got = window_from_file(
                    position, scale, rotation, clip.frame_size, res.width, res.height
                )
                want = window_from_plan(
                    x, y, zoom, transform, clip.frame_size, res.width, res.height
                )
                self.windows += 1
                off = max(abs(a - b) for a, b in zip(got, want, strict=True))
                self.want(
                    time == ptime and off <= WINDOW_TOLERANCE_PX,
                    f"{clip.location} at frame {ptime}: crop window "
                    f"{tuple(round(v, 3) for v in got)} (frame {time}) is {off:.3f} source px "
                    f"from the plan's {tuple(round(v, 3) for v in want)}",
                )

    def titles_and_markers(self) -> None:
        # Title text is compared joined: FCPXML holds one text, the lines joined by LF.
        got_titles = sorted(
            (t.lane, t.record_in, t.record_in + t.frames, "\n".join(t.lines), t.name, t.effect)
            for t in self.dec.titles
        )
        want_titles = sorted(
            (t.lane, t.record_in, t.record_out, "\n".join(t.lines), t.name, BASIC_TITLE)
            for t in self.layout.titles
        )
        self.want(got_titles == want_titles, f"titles {got_titles} != {want_titles}")
        got_markers = sorted(
            (m.record, m.duration, m.value, m.note or "") for m in self.dec.markers
        )
        want_markers = sorted(
            (m.record, m.duration, m.value, m.note or "") for m in self.layout.markers
        )
        self.want(got_markers == want_markers, f"markers {got_markers} != {want_markers}")

    def against_plan(self) -> None:
        """The brief's check, straight from the plan: segments and exported transitions."""
        plan = self.layout.plan
        ranges = plan.record_ranges()
        have = {(v.lane, v.record_in, v.record_out, v.src_in, v.src_out, v.src) for v in self.views}
        for index, seg in enumerate(plan.segments):
            record_in, record_out = ranges[index]
            src = self.layout.media[seg.source].src
            row = (seg.track - 1, record_in, record_out, seg.src_in, seg.src_out, src)
            self.want(row in have, f"segment {index} {row} is not in the file")
        if plan.outro is not None:
            rng = plan.outro_range()
            assert rng is not None
            source = plan.source_map()[plan.outro.source]
            end = plan.outro.src_out if plan.outro.src_out is not None else source.duration
            src = self.layout.media[source.id].src
            row = (plan.outro.track - 1, rng[0], rng[1], plan.outro.src_in, end, src)
            self.want(row in have, f"outro {row} is not in the file")
        decoded = {(cut, frames) for cut, frames, _, _ in self.decoded_transitions}
        for spec in self.layout.transitions:
            cut, frames = _plan_transition(plan, spec)
            self.want(
                (cut, frames) in decoded,
                f"{spec.location}: the plan's cut {cut} ({frames} frames) is not in the file",
            )

    def result(self) -> VerifyResult:
        counts = {
            "clips": len(self.views),
            "transitions": len(self.decoded_transitions),
            "titles": len(self.dec.titles),
            "markers": len(self.dec.markers),
            "assets": len(self.dec.assets),
            "crop_windows": self.windows,
        }
        return VerifyResult(not self.problems, self.problems, counts)


def _plan_transition(plan: Any, spec: TransitionSpec) -> tuple[int, int]:
    """(cut, frames) of an exported transition, computed from the plan fields alone."""
    ranges = plan.record_ranges()
    if spec.kind == "head":
        first = min((ranges[i][0], seg.track, i) for i, seg in enumerate(plan.segments))
        return first[0], plan.head.video_frames
    if spec.kind == "outro":
        return plan.content_frames(), plan.outro.transition.video_frames
    number = int(spec.location.split("[")[1].rstrip("]"))
    tx = plan.transitions[number]
    frames = tx.video_frames if tx.video != "cut" else tx.audio.frames
    return ranges[tx.after][1], frames


def verify_fcpxml(text: str, layout: Layout) -> VerifyResult:
    """Read ``text`` back and compare it with ``layout`` (built from the plan)."""
    check = _Check(read_fcpxml(text), layout)
    if check.dec.timebase is None:
        check.want(False, "the file could not be decoded far enough to compare")
        return check.result()
    check.header()
    check.spine()
    check.split_crossfades()
    check.clips()
    check.media_bounds()
    check.assets()
    check.crop_windows()
    check.titles_and_markers()
    check.against_plan()
    return check.result()
