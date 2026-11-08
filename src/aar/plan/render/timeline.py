"""The S18 layout as a list of pictures and sounds to render, in record frames (S18b).

Everything here is read off the FCPXML export's :class:`~aar.plan.export.layout.Layout` (built
with hub paths), so the render and the export agree on every placement: clip record ranges, the
handles each transition plays and its alignment, split-edit (J/L) sound ranges and fades, the head
and end fades, and each clip's volume (segment gain plus its audio track's gain).

* A **picture** is one clip's frames: its plan range, widened by the handle a dissolve plays
  (``head_part`` frames before the cut on the incoming clip, ``tail_part`` after it on the
  outgoing one). The head transition from black is a video fade on the first clip.
* A **sound** is one clip's sound: S18's ``sound_range`` and ``fades`` (a split edit's overlap),
  plus, for a dissolve that carries an audio crossfade, the same handles and a fade over the
  transition's window on each side. A muted track that feeds a bus stays audible (the CCN
  topology, as in S18); a muted track that feeds no bus is silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aar.plan.export.layout import ClipSpec, Layout, TitleSpec, TransitionSpec
from aar.plan.model import EditPlan, Segment
from aar.plan.render.errors import RenderError
from aar.plan.render.framing import Framing, framing_for
from aar.plan.render.media import HubSource


@dataclass(frozen=True)
class Picture:
    """One clip's frames: ``frames`` source frames from ``src_start``, at record ``start``."""

    location: str
    source_id: str
    input: int
    lane: int
    start: int
    frames: int
    src_start: int
    framing: Framing
    fade_in: int = 0  # the head's fade from black, in frames

    @property
    def end(self) -> int:
        return self.start + self.frames


@dataclass(frozen=True)
class Sound:
    """One clip's sound: record ``[start, end)`` from source frame ``src_start``."""

    location: str
    source_id: str
    input: int
    start: int
    end: int
    src_start: int
    gain_db: float
    fade_in: int
    fade_out: int


@dataclass(frozen=True)
class Dissolve:
    """A spine dissolve: ``frames`` long from record ``window_start``, at the plan's ``cut``."""

    location: str
    cut: int
    window_start: int
    frames: int


@dataclass
class Timeline:
    plan: EditPlan
    layout: Layout
    total_frames: int
    inputs: list[Path] = field(default_factory=list)
    pictures: list[Picture] = field(default_factory=list)
    sounds: list[Sound] = field(default_factory=list)
    dissolves: list[Dissolve] = field(default_factory=list)
    titles: list[TitleSpec] = field(default_factory=list)
    #: Plan features the timeline does not play, as (feature, location, detail).
    silent: list[tuple[str, str, str]] = field(default_factory=list)


def _segment(plan: EditPlan, clip: ClipSpec) -> Segment | None:
    if clip.location.startswith("segments["):
        return plan.segments[int(clip.location[len("segments[") : -1])]
    return None


def _source_id(plan: EditPlan, clip: ClipSpec) -> str:
    seg = _segment(plan, clip)
    if seg is not None:
        return seg.source
    assert plan.outro is not None and clip.location == "outro"
    return plan.outro.source


def _picture_ends(layout: Layout, clip: ClipSpec) -> tuple[int, int, int]:
    """Frames before and after the plan range the clip's picture plays (dissolve handles), and
    the head's fade from black on it."""
    if clip.lane != 0:
        return 0, 0, 0
    into = layout.spine_transition_at(clip.record_in)
    out = layout.spine_transition_at(clip.record_out)
    head = into.head_part if into is not None and into.kind != "head" else 0
    fade = into.frames if into is not None and into.kind == "head" else 0
    tail = out.tail_part if out is not None and out.kind != "head" else 0
    return head, tail, fade


def _with_audio(spec: TransitionSpec | None) -> bool:
    return spec is not None and spec.audio and spec.form == "transition"


def _sound_range(layout: Layout, clip: ClipSpec) -> tuple[int, int, int, int]:
    """Record range and fades of the clip's sound: S18's (split edits), widened by a dissolve's
    audio crossfade on either end."""
    start, end = layout.sound_range(clip)
    fade_in, fade_out = layout.fades(clip)
    if clip.lane == 0:
        into = layout.spine_transition_at(clip.record_in)
        out = layout.spine_transition_at(clip.record_out)
        if _with_audio(into):
            assert into is not None
            start -= into.head_part
            fade_in += into.frames
        if _with_audio(out):
            assert out is not None
            end += out.tail_part
            fade_out += out.frames
    return start, end, fade_in, fade_out


def _muted_without_bus(plan: EditPlan, track: int | None) -> bool:
    tracks = plan.audio_tracks
    if track is None or not 1 <= track <= len(tracks):
        return False
    return tracks[track - 1].mute and tracks[track - 1].bus is None


def build_timeline(layout: Layout, hub: dict[str, HubSource]) -> Timeline:
    """Pictures, sounds, dissolves and titles for the layout, or a typed refusal when a hub file
    is too short for what the timeline plays from it."""
    plan = layout.plan
    tl = Timeline(plan, layout, layout.total_frames, titles=list(layout.titles))
    index: dict[Path, int] = {}
    need: dict[str, int] = {}
    for clip in layout.clips:
        source_id = _source_id(plan, clip)
        media = hub[source_id]
        k = index.setdefault(media.path, len(index))
        seg = _segment(plan, clip)
        head, tail, fade = _picture_ends(layout, clip)
        frames = clip.record_out - clip.record_in + head + tail
        framing = framing_for(
            clip,
            keys=list(seg.reframe) if seg is not None else [],
            transform=seg.transform if seg is not None else None,
            display=media.display,
            resolution=plan.resolution,
            lead=head,
            frames=frames,
        )
        picture = Picture(
            clip.location,
            source_id,
            k,
            clip.lane,
            clip.record_in - head,
            frames,
            clip.src_in - head,
            framing,
            fade,
        )
        tl.pictures.append(picture)
        need[source_id] = max(need.get(source_id, 0), picture.src_start + frames)
        start, end, fade_in, fade_out = _sound_range(layout, clip)
        src_start = clip.src_in - (clip.record_in - start)
        if _muted_without_bus(plan, clip.audio_track):
            name = plan.audio_tracks[(clip.audio_track or 1) - 1].name
            tl.silent.append(
                (
                    "audio_mute",
                    clip.location,
                    f"its sound is on A{clip.audio_track} {name!r}, muted with no bus: silent",
                )
            )
        elif media.channels is None:
            tl.silent.append(("no_audio", clip.location, f"{media.path.name} has no audio stream"))
        else:
            tl.sounds.append(
                Sound(
                    clip.location,
                    source_id,
                    k,
                    start,
                    end,
                    src_start,
                    clip.volume_db,
                    fade_in,
                    fade_out,
                )
            )
            need[source_id] = max(need[source_id], src_start + end - start)
    tl.inputs = list(index)
    for spec in layout.transitions:
        if spec.form == "transition" and spec.kind != "head":
            tl.dissolves.append(Dissolve(spec.location, spec.cut, spec.window_start, spec.frames))
    for source_id, frames in need.items():
        media = hub[source_id]
        if media.frames is not None and frames > media.frames:
            raise RenderError(
                "media_too_short",
                f"source {source_id!r}: the timeline plays up to frame {frames} of "
                f"{media.path.name}, which holds {media.frames} frames",
                "check the source ranges and handles against the hub copy, or re-ingest it",
                source=source_id,
                needed_frames=frames,
                media_frames=media.frames,
            )
    return tl
