"""The render report: what was rendered, the QA results, and every plan feature not rendered
(S18b). It is written beside the file as ``<stem>.render.json``, as one pair with it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aar.plan.export.layout import Layout
from aar.plan.model import EditPlan
from aar.plan.render.fonts import FontChoice
from aar.plan.render.media import HubSource
from aar.plan.render.timeline import Timeline


@dataclass(frozen=True)
class Item:
    """A plan feature and what the render did with it, with its location in the plan."""

    feature: str
    location: str
    detail: str
    action: str = "none"

    def as_dict(self) -> dict[str, str]:
        return {
            "feature": self.feature,
            "location": self.location,
            "detail": self.detail,
            "action": self.action,
        }


#: S18 layout items the render shares (no transition in the layout means none in the render),
#: in render terms.
_FROM_LAYOUT = {
    "transition_type": (
        "rendered as a dissolve of the same length (the shared layout's placeholder)",
        "replace it in Resolve if the type matters",
    ),
    "audio_crossfade_length": (
        "the audio crossfade is rendered at the dissolve's length, as the export writes it",
        "adjust it in Resolve if it must differ",
    ),
    "audio_transition_type": (
        "rendered as a linear crossfade (Cross Fade 0 dB)",
        "set the type in Resolve",
    ),
    "outro_audio_transition": (
        "the outro's sound starts at its cut; the planned crossfade into it is not rendered",
        "none",
    ),
    "transition": ("not rendered: a cut", "fix the plan, or add the transition in Resolve"),
    "head_transition": (
        "not rendered: the first segment is not on V1, so it starts hard",
        "move the first segment to V1, or add it in Resolve",
    ),
    "outro_transition": (
        "not rendered: the outro and the last segment are not neighbours on V1, so it is a cut",
        "add it in Resolve",
    ),
}


def not_rendered(
    plan: EditPlan, layout: Layout, font: FontChoice, luts: Mapping[str, object]
) -> list[Item]:
    """Every plan feature the render does not play, with where it is and what to do."""
    items: list[Item] = []
    if plan.template is not None:
        items.append(
            Item(
                "template",
                "template",
                f"the render builds the plan's own timeline; template {plan.template!r} (its "
                "look, Fusion titles and buses) is not applied",
                "for the template's look, import the FCPXML export into a branch of it",
            )
        )
    used = {ref for ref in layout.media}
    for index, source in enumerate(plan.sources):
        color = source.color
        if color is None or source.id not in used:
            continue
        where = f"sources[{index}].color"
        if color.input_color_space is not None:
            items.append(
                Item(
                    "input_color_space",
                    f"{where}.input_color_space",
                    f"input colour space {color.input_color_space!r} of {source.id!r} is not "
                    "applied",
                    "grade in Resolve (never on BRAW)",
                )
            )
        if color.cdl is not None:
            items.append(
                Item(
                    "cdl",
                    f"{where}.cdl",
                    f"the CDL of {source.id!r} is not applied",
                    "apply it on the Color page",
                )
            )
        if color.lut is not None and source.id not in luts:
            items.append(
                Item(
                    "lut",
                    f"{where}.lut",
                    f"LUT {color.lut!r} of {source.id!r} is not a LUT file on the hub, so it is "
                    "not applied",
                    "give the LUT's hub path (.cube, .3dl) to render it with lut3d",
                )
            )
    tracks = plan.audio_tracks
    for number, track in enumerate(tracks, start=1):
        where = f"audio_tracks[{number - 1}]"
        if track.mute and track.bus is not None:
            items.append(
                Item(
                    "audio_mute",
                    f"{where}.mute",
                    f"A{number} {track.name!r} is muted but feeds bus {track.bus!r}: rendered "
                    "audible (the CCN topology), as the FCPXML export does",
                )
            )
        if track.bus is not None:
            items.append(
                Item(
                    "audio_bus",
                    f"{where}.bus",
                    f"bus routing is not rendered: A{number} {track.name!r} plays at each "
                    "segment's gain plus its own track's gain",
                    f"route {track.name!r} to {track.bus!r} in Fairlight",
                )
            )
            target = next((t for t in tracks if t.name == track.bus), None)
            if target is not None and target.gain_db != 0.0:
                items.append(
                    Item(
                        "bus_gain",
                        f"{where}.bus",
                        f"bus {track.bus!r}'s gain {target.gain_db:+g} dB is not applied to "
                        f"A{number} {track.name!r}'s clips (as the export)",
                    )
                )
    if plan.captions:
        items.append(
            Item(
                "captions",
                "captions",
                f"{len(plan.captions)} caption(s) are not burned in",
                "captions are burned in by the shorts compose (S13e)",
            )
        )
    if plan.caption_style is not None:
        items.append(Item("caption_style", "caption_style", "not applied (no captions drawn)"))
    for index, marker in enumerate(plan.markers):
        items.append(
            Item(
                "marker",
                f"markers[{index}]",
                f"marker {marker.name!r} at frame {marker.record}: an MP4 carries no markers",
                "see the FCPXML export report for the marker list",
            )
        )
    for title in layout.titles:
        if font.path is None:
            items.append(
                Item(
                    "title",
                    title.location,
                    f"{title.name!r} is not drawn: {font.reason}",
                    f"install {font.family} or set AAR_FONTS_DIR",
                )
            )
        else:
            template = f" ({title.template!r} template)" if title.template else ""
            items.append(
                Item(
                    "title_template",
                    title.location,
                    f"drawn as plain {font.used} text on its frames; the Resolve title"
                    f"{template} is not applied",
                    "replace it in Resolve for the final look",
                )
            )
    for item in layout.not_expressible:
        if item.feature in _FROM_LAYOUT:
            detail, action = _FROM_LAYOUT[item.feature]
            items.append(Item(item.feature, item.location, detail, action))
    return items


def notes(tl: Timeline, font: FontChoice, hub: Mapping[str, HubSource]) -> list[Item]:
    """What the render did that a reader should know, without a loss of a plan feature."""
    out = [Item(feature, location, detail) for feature, location, detail in tl.silent]
    for index, source in enumerate(tl.plan.sources):
        media = hub.get(source.id)
        offset = None if media is None else media.audio_start_s - media.video_start_s
        if media is not None and media.channels is not None and offset:
            out.append(
                Item(
                    "audio_offset",
                    f"sources[{index}]",
                    f"{media.path.name}'s audio starts {float(offset):+.6f} s from its video; the "
                    "sound is cut by time, so it stays in sync with the picture",
                )
            )
        if media is not None and media.channels is not None and media.channels > 2:
            out.append(
                Item(
                    "audio_channels",
                    f"sources[{index}]",
                    f"{media.path.name} has {media.channels} audio channels; the render plays "
                    "the first two as stereo",
                    "give a stereo or mono hub copy if another pair is meant",
                )
            )
    for pic in tl.pictures:
        if pic.framing.mode == "fit":
            out.append(
                Item(
                    "frame_size",
                    pic.location,
                    "no frame_size, and its aspect differs from the timeline's: fitted "
                    "(letterboxed), as Resolve's default for the FCPXML import",
                    "set the source's frame_size to fill",
                )
            )
    if font.fallback and font.path is not None and tl.titles:
        out.append(Item("font", "titles", font.reason))
    return out


def rendered(tl: Timeline, font: FontChoice, luts: Mapping[str, str]) -> dict[str, Any]:
    """What the render put on the timeline, in record frames."""
    fps = tl.plan.fps
    pictures = []
    for pic in tl.pictures:
        entry: dict[str, Any] = {
            "location": pic.location,
            "source": pic.source_id,
            "lane": pic.lane,
            "record": [pic.start, pic.end],
            "source_frames": [pic.src_start, pic.src_start + pic.frames],
            "framing": pic.framing.mode,
        }
        if pic.fade_in:
            entry["fade_from_black_frames"] = pic.fade_in
        plan = pic.framing.window
        if plan is not None:
            entry["window"] = {
                "moving": plan.moving,
                "first": [round(v, 3) for v in plan.first],
                "last": [round(v, 3) for v in plan.last],
                "pad": plan.pad,
                "box": list(plan.box),
                "units": "hub source pixels: centre x, centre y, width, height",
            }
        pictures.append(entry)
    return {
        "pictures": pictures,
        "dissolves": [
            {
                "location": d.location,
                "cut": d.cut,
                "window": [d.window_start, d.window_start + d.frames],
                "frames": d.frames,
            }
            for d in tl.dissolves
        ],
        "sounds": [
            {
                "location": s.location,
                "source": s.source_id,
                "record": [s.start, s.end],
                "gain_db": s.gain_db,
                "fade_in_frames": s.fade_in,
                "fade_out_frames": s.fade_out,
            }
            for s in tl.sounds
        ],
        "titles": [
            {
                "location": t.location,
                "record": [t.record_in, t.record_out],
                "lines": list(t.lines),
                "timecode": fps.timecode(t.record_in),
            }
            for t in tl.titles
        ]
        if font.path is not None
        else [],
        "luts": dict(sorted(luts.items())),
        "font": font.as_dict(),
    }
