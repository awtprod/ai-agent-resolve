"""The self-check: read an FCPXML file back and require it to equal the plan, without Resolve.

:func:`verify_fcpxml` decodes the file (:mod:`aar.plan.export.reader`), rebuilds the timeline in
plan terms and compares it with the layout built from the plan:

* the header: version 1.10, frame duration, tcStart, length, size and names;
* the spine: elements contiguous from 0 to the plan's length, each transition overlapping its
  neighbours by exactly its length;
* every clip (track, record in/out, source in/out, media URL, which streams, where its sound
  plays, volume, fades, transform and keyframes), title, marker and asset;
* and, straight from the plan, each segment's (track, record in/out, source in/out, media) and
  each exported transition's (cut, length): equal, not close.

A transition's overlap in the file says where the media plays but not where the plan's cut is
(FCPXML has no alignment), so the decode reads each window with the alignment the plan's handles
give (the same rule the export uses). A wrong window or overlap leaves a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aar.plan.export.fcpxml import AUDIO_CROSSFADE, BASIC_TITLE, CROSS_DISSOLVE, FCPXML_VERSION
from aar.plan.export.layout import ClipSpec, Layout, TransformSpec, TransitionSpec
from aar.plan.export.reader import Decoded, RawClip, read_fcpxml
from aar.plan.export.timebase import frame_duration_text


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
    transform: TransformSpec | None
    name: str
    note: str | None


def _expected_view(clip: ClipSpec, layout: Layout) -> ClipView:
    sound = None
    if clip.enable != "video":
        sound_in, sound_out = layout.sound_range(clip)
        shift = clip.src_in - clip.record_in  # record -> source frames for this clip
        sound = (sound_in + shift, sound_out + shift)
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
        clip.fade_in,
        clip.fade_out,
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
        transform,
        raw.name,
        raw.note,
    )


class _Check:
    def __init__(self, decoded: Decoded, layout: Layout) -> None:
        self.dec = decoded
        self.layout = layout
        self.problems: list[str] = list(decoded.problems)
        self.decoded_transitions: list[tuple[int, int, bool, bool]] = []
        self.views: list[ClipView] = []

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
        size = (plan.resolution.width, plan.resolution.height)
        self.want(
            (dec.width, dec.height) == size, f"format is {dec.width}x{dec.height}, not {size}"
        )
        self.want(
            dec.project == plan.timeline, f"project is {dec.project!r}, not {plan.timeline!r}"
        )
        self.want(dec.event == plan.project, f"event is {dec.event!r}, not {plan.project!r}")

    def spine(self) -> None:
        """Contiguity, transition overlaps, and the decode of each primary clip."""
        entries = self.dec.spine
        expected = {t.window_start: t for t in self.layout.transitions}
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
                CROSS_DISSOLVE[0] if spec.video else None,
                AUDIO_CROSSFADE[0] if spec.audio else None,
            )
            self.want(
                (raw.video_effect, raw.audio_effect) == want_effect,
                f"transition at record {raw.offset}: effects {raw.video_effect!r}/"
                f"{raw.audio_effect!r}, not {want_effect}",
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

    def titles_and_markers(self) -> None:
        # Title text is compared joined: FCPXML holds one text, the lines joined by LF.
        got_titles = sorted(
            (t.lane, t.record_in, t.record_in + t.frames, "\n".join(t.lines), t.name, t.effect)
            for t in self.dec.titles
        )
        want_titles = sorted(
            (t.lane, t.record_in, t.record_out, "\n".join(t.lines), t.name, BASIC_TITLE[0])
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
        got_assets = sorted(self.dec.assets.values())
        want_assets = sorted(a.src for a in self.layout.assets)
        self.want(got_assets == want_assets, f"assets {got_assets} != {want_assets}")

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
    check.clips()
    check.titles_and_markers()
    check.against_plan()
    return check.result()
