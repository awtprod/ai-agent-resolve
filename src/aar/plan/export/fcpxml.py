"""Write a :class:`~aar.plan.export.layout.Layout` as one FCPXML 1.10 document (stdlib only).

Shape: ``fcpxml > resources (format, effects, assets with media-rep src) > library > event >
project > sequence > spine``. Geometry follows Apple's FCPXML reference (URLs in
aar-evidence/s18/PROGRESS.md):

* Spine offsets count from the sequence's ``tcStart`` (01:00:00:00 by default), in exact rational
  seconds. Spine clips keep their plan record ranges and abut exactly; holes on V1 are ``gap``
  elements; the spine runs from 0 to the plan's total length. Resolve 21.1 "trims" overlapping
  spine clips and duplicated the outro when they overlapped (import test, 2026-11-08).
* A ``transition`` sits at the edit between two spine elements and plays over the media beyond
  the outgoing clip's out and the incoming clip's in (handles); neither clip is lengthened. Its
  ``offset`` is the cut minus its frames before the cut: the cut for one that starts on it (a
  transition from black, an outro dissolve with no head handle), the cut minus ``d // 2`` when
  centred, the cut minus ``d`` when it ends on it. Apple: the spine "contains elements ordered
  sequentially in time" and "only one story element is active at a given time, except when a
  transition is present"; the transition's "offset and duration attributes define the position
  and extent of the transition effect".
* Each ``asset`` starts at its media's start timecode (``Source.start_timecode``, ``0s`` when
  unknown), and a clip's times count from there: its ``start`` and ``audioStart`` (the asset's
  start + the source position) and its keyframe times. Resolve links a clip to its file only
  when the asset's start is the file's timecode (import test rerun, 2026-11-08).
* Connected clips (V2 and up, dense lanes) and titles are anchored to the spine element under
  their start, at a non-zero lane, with ``offset`` in that element's local time (``start`` +
  frames into it), so in the parent's timecode-based time: a title left at the un-shifted value
  imported an hour before the timeline start. Every clip keeps its own linked sound: no
  ``srcEnable`` (Resolve ignores it). No ``marker`` is written: Resolve imports none from
  FCPXML; the report lists them.
* A clip whose sound runs past a cut its picture stops at (a split-edit audio crossfade, a J/L
  cut) carries ``audioStart``/``audioDuration``; the DTD: "Use 'audioStart' and 'audioDuration'
  to define J/L cuts (i.e., split edits) on composite A/V clips". A split crossfade's fades are
  ``fadeIn`` and ``fadeOut`` inside ``adjust-volume``'s ``amount`` param.
* A clip framed for another aspect, or moved, carries ``adjust-conform type="fill"`` before its
  ``adjust-transform`` (the DTD: "The absence 'adjust-conform' element implies 'fit'").

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
#: Rates with a standard Final Cut Pro format name (FFVideoFormat1080p2398 etc.); 47.95 and 48
#: have none, so those formats, like vertical and other sizes, carry no name (it is optional).
_STANDARD_RATES = frozenset({"2398", "24", "25", "2997", "30", "50", "5994", "60"})
_MIXED_CONTENT = frozenset({"text", "text-style", "note"})


def format_name(timebase: Timebase, width: int, height: int) -> str | None:
    """Final Cut Pro's standard format name (``FFVideoFormat1080p2398``) where one exists for
    this size and rate, else ``None`` (the ``name`` attribute is optional; F12)."""
    size = _STANDARD_SIZES.get((width, height))
    if size is None or timebase.suffix not in _STANDARD_RATES:
        return None
    return f"FFVideoFormat{size}{timebase.suffix}"


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
        # Transitions written as spine elements, by cut (cuts are unique on video track 1).
        self.by_cut = {t.cut: t for t in layout.transitions if t.form == "transition"}
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
        name = format_name(self.tb, res.width, res.height)
        if name is not None:
            fmt.set("name", name)
        fmt.set("frameDuration", frame_duration_text(self.tb))
        fmt.set("width", str(res.width))
        fmt.set("height", str(res.height))
        fmt.set("colorSpace", "1-1-1 (Rec. 709)")
        number_ = 2
        effects = []
        if any(t.video for t in self.by_cut.values()):
            effects.append(("dissolve", CROSS_DISSOLVE))
        if any(t.audio for t in self.by_cut.values()):
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
                start=self.t(asset.start),
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
            if clip.record_in > cursor:
                items.append(self._gap(spine, cursor, clip.record_in))
            if clip.record_in in self.by_cut:
                self._transition(spine, clip.record_in)
            offset = self.seq(clip.record_in)
            items.append(self._clip(spine, clip, clip.record_in, clip.record_out, offset))
            cursor = clip.record_out
        if cursor < self.layout.total_frames:
            items.append(self._gap(spine, cursor, self.layout.total_frames))
        return items

    # -- clips ---------------------------------------------------------------------------------

    def _clip(
        self, parent: ET.Element, clip: ClipSpec, raw_in: int, raw_out: int, offset: str
    ) -> _SpineItem:
        """An asset-clip; ``offset`` is a sequence time on the spine, else parent-local.

        Its times count from its asset's start, the media's start timecode (round 4): ``start``
        and ``audioStart`` here, the keyframe times inside it, and the ``offset`` of every item
        anchored to it (``_local``). Resolve links the clip to its file only then."""
        start = self.layout.asset_start(clip.src) + clip.src_in
        attrs = {"ref": self.asset_ids[clip.src]}
        if clip.lane != 0:
            attrs["lane"] = str(clip.lane)
        attrs["offset"] = offset
        attrs["name"] = clip.name
        attrs["start"] = self.t(start)
        attrs["duration"] = self.t(raw_out - raw_in)
        # Never srcEnable: Resolve 21.1 ignores it and imports the clip's sound anyway.
        sound_in, sound_out = self.layout.sound_range(clip)
        if (sound_in, sound_out) != (raw_in, raw_out):
            attrs["audioStart"] = self.t(start + sound_in - raw_in)
            attrs["audioDuration"] = self.t(sound_out - sound_in)
        element = ET.SubElement(parent, "asset-clip", attrs)
        if clip.note:
            ET.SubElement(element, "note").text = clip.note
        if clip.conform:
            ET.SubElement(element, "adjust-conform", type="fill")
        if clip.transform is not None:
            self._transform(element, start, clip.transform)
        fade_in, fade_out = self.layout.fades(clip)
        if clip.volume_db != 0.0 or fade_in or fade_out:
            self._volume(element, clip, fade_in, fade_out)
        return _SpineItem(raw_in, raw_out, start, element, clip)

    def _transform(self, element: ET.Element, start: int, spec: TransformSpec) -> None:
        """``start`` is the clip's local start (keyframe times are clip-local)."""
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
                    time=self.t(start + key.time),
                    value=_pair(key.position if name == "position" else key.scale),
                    interp="linear",
                    curve="linear",
                )

    def _volume(self, element: ET.Element, clip: ClipSpec, fade_in: int, fade_out: int) -> None:
        adjust = ET.SubElement(element, "adjust-volume")
        if clip.volume_db != 0.0:
            adjust.set("amount", db(clip.volume_db))
        if fade_in or fade_out:
            param = ET.SubElement(adjust, "param", name="amount")
            if fade_in:
                ET.SubElement(param, "fadeIn", type="linear", duration=self.t(fade_in))
            if fade_out:
                ET.SubElement(param, "fadeOut", type="linear", duration=self.t(fade_out))

    # -- anchored items ------------------------------------------------------------------------

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
    """Put each clip's children in the DTD's order: note, adjustments, anchored items."""
    rank = {
        "note": 0,
        "adjust-conform": 1,
        "adjust-transform": 2,
        "adjust-volume": 3,
        "asset-clip": 4,
        "title": 4,
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
