"""Write a :class:`~aar.plan.export.layout.Layout` as one FCPXML 1.10 document (stdlib only).

Shape: ``fcpxml > resources (format, effects, assets with media-rep src) > library > event >
project > sequence > spine``. Geometry follows Apple's FCPXML reference (URLs in
aar-evidence/s18/PROGRESS.md):

* Spine offsets count from the sequence's ``tcStart`` (01:00:00:00 by default), in exact rational
  seconds. Holes on V1 are ``gap`` elements; the spine runs from 0 to the plan's total length.
* A ``transition`` overlaps the end of the element before it and the start of the one after: the
  outgoing clip is lengthened into its tail by the part after the cut, and the incoming clip
  starts earlier (offset and ``start``) by the part before it. A transition from black is the
  first spine element, or follows a gap that it overlaps.
* V2 and up, titles and audio-only clips are anchored to the spine element under their start,
  at a non-zero lane, with ``offset`` in that element's local time (``start`` + frames into it).
  Markers sit in the same element the same way.
* A clip whose sound must stop at a cut its picture runs past (the dissolve into the outro)
  carries ``audioStart``/``audioDuration``.

The output is deterministic: the same plan and inputs give the same bytes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal

from aar.plan.export.layout import ClipSpec, Layout, TitleSpec, TransformSpec
from aar.plan.export.timebase import Timebase, frame_duration_text

FCPXML_VERSION = "1.10"
HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'

# Effect uids as Final Cut Pro writes them (not on Apple's reference pages; see PROGRESS.md).
CROSS_DISSOLVE = ("Cross Dissolve", "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265")
AUDIO_CROSSFADE = ("Audio Crossfade", "FFAudioTransition")
BASIC_TITLE = (
    "Basic Title",
    ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti",
)
_STANDARD_SIZES = {(1280, 720): "720p", (1920, 1080): "1080p", (3840, 2160): "2160p"}
_MIXED_CONTENT = frozenset({"text", "text-style", "note"})


def number(value: float) -> str:
    """A float as the shortest exact decimal text, never in exponent form: ``-2.0`` -> ``-2``,
    ``1e-07`` -> ``0.0000001``. ``float(number(x)) == x`` for every finite ``x``."""
    text = format(Decimal(repr(float(value))), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _pair(values: tuple[float, float]) -> str:
    return f"{number(values[0])} {number(values[1])}"


def db(value: float) -> str:
    return f"{number(value)}dB"


@dataclass
class _SpineItem:
    """One primary-storyline element in record frames (``raw_*`` include transition overlaps)."""

    raw_in: int
    raw_out: int
    start: int  # local start frames: source frames for a clip, sequence frames for a gap
    element: ET.Element
    clip: ClipSpec | None


class _Writer:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.tb: Timebase = layout.timebase
        self.by_cut = {t.cut: t for t in layout.transitions}  # cuts are unique on track 1
        self.ids: dict[str, str] = {}
        self.asset_ids: dict[str, str] = {}
        self.styles = 0

    def t(self, frames: int) -> str:
        return self.tb.time(frames)

    def seq(self, record: int) -> str:
        """A record frame as a sequence time (counted from tcStart)."""
        return self.t(self.layout.tc_start + record)

    # -- document ------------------------------------------------------------------------------

    def document(self) -> ET.Element:
        layout = self.layout
        root = ET.Element("fcpxml", version=FCPXML_VERSION)
        resources = ET.SubElement(root, "resources")
        self._resources(resources)
        library = ET.SubElement(root, "library")
        event = ET.SubElement(library, "event", name=layout.plan.project)
        project = ET.SubElement(event, "project", name=layout.plan.timeline)
        sequence = ET.SubElement(
            project,
            "sequence",
            format=self.ids["format"],
            duration=self.t(layout.total_frames),
            tcStart=self.t(layout.tc_start),
            tcFormat="NDF",
            audioLayout="stereo",
            audioRate="48k",
        )
        note = layout.sequence_note()
        if note is not None:
            ET.SubElement(sequence, "note").text = note
        spine = ET.SubElement(sequence, "spine")
        items = self._spine(spine)
        self._anchor_all(items)
        return root

    def _resources(self, resources: ET.Element) -> None:
        layout = self.layout
        res = layout.plan.resolution
        self.ids["format"] = "r1"
        fmt = ET.SubElement(resources, "format", id="r1")
        size = _STANDARD_SIZES.get((res.width, res.height))
        if size is not None:
            fmt.set("name", f"FFVideoFormat{size}{self.tb.suffix}")
        fmt.set("frameDuration", frame_duration_text(self.tb))
        fmt.set("width", str(res.width))
        fmt.set("height", str(res.height))
        fmt.set("colorSpace", "1-1-1 (Rec. 709)")
        number_ = 2
        effects = []
        if any(t.video for t in layout.transitions):
            effects.append(("dissolve", CROSS_DISSOLVE))
        if any(t.audio for t in layout.transitions):
            effects.append(("crossfade", AUDIO_CROSSFADE))
        if layout.titles:
            effects.append(("title", BASIC_TITLE))
        for key, (name, uid) in effects:
            rid = f"r{number_}"
            number_ += 1
            self.ids[key] = rid
            ET.SubElement(resources, "effect", id=rid, name=name, uid=uid)
        for asset in layout.assets:
            rid = f"r{number_}"
            number_ += 1
            self.asset_ids[asset.src] = rid
            element = ET.SubElement(
                resources,
                "asset",
                id=rid,
                name=asset.name,
                start="0s",
                duration=self.t(asset.duration),
                hasVideo="1",
                hasAudio="1",
            )
            ET.SubElement(element, "media-rep", kind="original-media", src=asset.src)

    # -- spine ---------------------------------------------------------------------------------

    def _gap(self, spine: ET.Element, raw_in: int, raw_out: int) -> _SpineItem:
        start = self.layout.tc_start + raw_in
        element = ET.SubElement(
            spine,
            "gap",
            name="Gap",
            offset=self.seq(raw_in),
            start=self.t(start),
            duration=self.t(raw_out - raw_in),
        )
        return _SpineItem(raw_in, raw_out, start, element, None)

    def _transition(self, spine: ET.Element, spec_cut: int) -> None:
        spec = self.by_cut[spec_cut]
        name = CROSS_DISSOLVE[0] if spec.video else AUDIO_CROSSFADE[0]
        element = ET.SubElement(
            spine,
            "transition",
            name=name,
            offset=self.seq(spec.window_start),
            duration=self.t(spec.frames),
        )
        if spec.video:
            ET.SubElement(element, "filter-video", ref=self.ids["dissolve"], name=CROSS_DISSOLVE[0])
        if spec.audio:
            ET.SubElement(
                element, "filter-audio", ref=self.ids["crossfade"], name=AUDIO_CROSSFADE[0]
            )

    def _spine(self, spine: ET.Element) -> list[_SpineItem]:
        items: list[_SpineItem] = []
        cursor = 0
        primary = sorted((c for c in self.layout.clips if c.lane == 0), key=lambda c: c.record_in)
        for clip in primary:
            into = self.by_cut.get(clip.record_in)
            raw_in, raw_out = self.layout.raw_range(clip)
            if into is not None and into.kind == "head":
                if raw_in > cursor:
                    # The gap before a transition from black runs under it (FCPXML overlap).
                    items.append(self._gap(spine, cursor, raw_in + into.frames))
                self._transition(spine, clip.record_in)
            elif into is not None:
                self._transition(spine, clip.record_in)
            elif raw_in > cursor:
                items.append(self._gap(spine, cursor, raw_in))
            items.append(self._clip(spine, clip, raw_in, raw_out, self.seq(raw_in)))
            cursor = raw_out
        if cursor < self.layout.total_frames:
            items.append(self._gap(spine, cursor, self.layout.total_frames))
        return items

    # -- clips ---------------------------------------------------------------------------------

    def _clip(
        self, parent: ET.Element, clip: ClipSpec, raw_in: int, raw_out: int, offset: str
    ) -> _SpineItem:
        """An asset-clip; ``offset`` is a sequence time on the spine, else parent-local."""
        start = clip.src_in - (clip.record_in - raw_in)
        attrs = {"ref": self.asset_ids[clip.src]}
        if clip.lane != 0:
            attrs["lane"] = str(clip.lane)
        attrs["offset"] = offset
        attrs["name"] = clip.name
        attrs["start"] = self.t(start)
        attrs["duration"] = self.t(raw_out - raw_in)
        if clip.enable != "all":
            attrs["srcEnable"] = clip.enable
        if clip.enable != "video":
            sound_in, sound_out = self.layout.sound_range(clip)
            if (sound_in, sound_out) != (raw_in, raw_out):
                attrs["audioStart"] = self.t(start + sound_in - raw_in)
                attrs["audioDuration"] = self.t(sound_out - sound_in)
        element = ET.SubElement(parent, "asset-clip", attrs)
        if clip.note:
            ET.SubElement(element, "note").text = clip.note
        if clip.transform is not None:
            self._transform(element, clip, clip.transform)
        if clip.enable != "video" and (clip.volume_db != 0.0 or clip.fade_in or clip.fade_out):
            self._volume(element, clip)
        return _SpineItem(raw_in, raw_out, start, element, clip)

    def _transform(self, element: ET.Element, clip: ClipSpec, spec: TransformSpec) -> None:
        adjust = ET.SubElement(
            element,
            "adjust-transform",
            position=_pair(spec.position),
            scale=_pair(spec.scale),
            rotation=number(spec.rotation),
        )
        if not spec.keyframes:
            return
        for name in ("position", "scale"):
            param = ET.SubElement(adjust, "param", name=name)
            animation = ET.SubElement(param, "keyframeAnimation")
            for key in spec.keyframes:
                ET.SubElement(
                    animation,
                    "keyframe",
                    time=self.t(clip.src_in + key.time),
                    value=_pair(key.position if name == "position" else key.scale),
                    interp="linear",
                    curve="linear",
                )

    def _volume(self, element: ET.Element, clip: ClipSpec) -> None:
        adjust = ET.SubElement(element, "adjust-volume")
        if clip.volume_db != 0.0:
            adjust.set("amount", db(clip.volume_db))
        if clip.fade_in or clip.fade_out:
            param = ET.SubElement(adjust, "param", name="amount")
            if clip.fade_in:
                ET.SubElement(param, "fadeIn", type="linear", duration=self.t(clip.fade_in))
            if clip.fade_out:
                ET.SubElement(param, "fadeOut", type="linear", duration=self.t(clip.fade_out))

    # -- anchored items and markers ------------------------------------------------------------

    def _parent(self, items: list[_SpineItem], record: int) -> _SpineItem:
        """The spine element an item starting at ``record`` is anchored to: the last one that
        starts at or before it."""
        chosen = items[0]
        for item in items:
            if item.raw_in <= record:
                chosen = item
        return chosen

    def _local(self, parent: _SpineItem, record: int) -> str:
        return self.t(parent.start + record - parent.raw_in)

    def _anchor_all(self, items: list[_SpineItem]) -> None:
        anchored: list[tuple[int, int, int, ClipSpec | TitleSpec]] = []
        for clip in self.layout.clips:
            if clip.lane != 0:
                anchored.append((clip.record_in, -clip.lane, 0, clip))
        for title in self.layout.titles:
            anchored.append((title.record_in, -title.lane, 1, title))
        for record, _lane, _kind, obj in sorted(anchored, key=lambda a: a[:3]):
            parent = self._parent(items, record)
            if isinstance(obj, ClipSpec):
                offset = self._local(parent, record)
                self._clip(parent.element, obj, obj.record_in, obj.record_out, offset)
            else:
                self._title(parent, obj)
        for marker in sorted(self.layout.markers, key=lambda m: m.record):
            parent = self._parent(items, marker.record)
            element = ET.SubElement(
                parent.element,
                "marker",
                start=self._local(parent, marker.record),
                duration=self.t(marker.duration),
                value=marker.value,
            )
            if marker.note:
                element.set("note", marker.note)

    def _title(self, parent: _SpineItem, title: TitleSpec) -> None:
        self.styles += 1
        style_id = f"ts{self.styles}"
        element = ET.SubElement(
            parent.element,
            "title",
            ref=self.ids["title"],
            lane=str(title.lane),
            offset=self._local(parent, title.record_in),
            name=title.name,
            duration=self.t(title.record_out - title.record_in),
        )
        text = ET.SubElement(element, "text")
        ET.SubElement(text, "text-style", ref=style_id).text = "\n".join(title.lines)
        definition = ET.SubElement(element, "text-style-def", id=style_id)
        ET.SubElement(
            definition,
            "text-style",
            font="Helvetica",
            fontSize="60",
            fontColor="1 1 1 1",
            alignment="center",
        )


def _indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print, but never inside mixed content (title text, notes): whitespace there would
    become part of the text."""
    if element.tag in _MIXED_CONTENT or len(element) == 0:
        return
    pad = "\n" + "  " * (level + 1)
    element.text = pad
    for child in element:
        _indent(child, level + 1)
        child.tail = pad
    element[-1].tail = "\n" + "  " * level


def _reorder(root: ET.Element) -> None:
    """Put each clip's children in the DTD's order: note, adjustments, anchored items, markers."""
    rank = {
        "note": 0,
        "adjust-transform": 1,
        "adjust-volume": 2,
        "asset-clip": 3,
        "title": 3,
        "marker": 4,
    }
    for element in root.iter():
        if element.tag in ("asset-clip", "gap"):
            children = list(element)
            children.sort(key=lambda c: rank.get(c.tag, 9))  # stable: keeps anchored order
            element[:] = children


def write_fcpxml(layout: Layout) -> str:
    """The FCPXML document for ``layout``, as text (UTF-8 when encoded)."""
    root = _Writer(layout).document()
    _reorder(root)
    _indent(root)
    return HEADER + ET.tostring(root, encoding="unicode") + "\n"
