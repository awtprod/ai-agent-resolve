"""Read an FCPXML file back without Resolve: structure checks and a raw decode of its timeline.

:func:`read_fcpxml` parses the text with the stdlib and returns a :class:`Decoded` timeline in
record and source frames: the spine elements as written (transition overlaps included), every
asset-clip with its lane and media URL, the transitions, titles and markers. It never needs the
plan; :mod:`aar.plan.export.verify` compares the decode with the plan.

Problems are collected, not raised, so one run reports them all:

* not well-formed XML, a DOCTYPE with an internal subset or an entity declaration (refused before
  parsing), or a file over :data:`MAX_BYTES`;
* an element, attribute or child order outside FCPXML 1.10 for the elements aar writes (the
  content models below are transcribed from Apple's DTD, see aar-evidence/s18/apple-docs);
* a duplicate id, or a ``ref``/``format`` that does not resolve to a resource of the right kind;
* a time that is not an exact rational on a frame boundary of the sequence's rate.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction

from aar.plan.export.layout import Keyframe, TransformSpec
from aar.plan.export.timebase import SUPPORTED_RATES, Timebase, parse_time

MAX_BYTES = 64 * 1024 * 1024

# -- FCPXML 1.10 content models for the elements aar reads (from Apple's DTD) -------------------

_ANCHOR = (
    r"(?:(?:audio|video|clip|title|caption|mc-clip|ref-clip|sync-clip|asset-clip|audition|spine) )*"
)
_MARKERS = r"(?:(?:marker|chapter-marker|rating|keyword|analysis-marker) )*"
_VIDEO = (
    r"(?:object-tracker )?(?:adjust-crop )?(?:adjust-corners )?(?:adjust-conform )?"
    r"(?:adjust-transform )?(?:adjust-blend )?(?:adjust-stabilization )?"
    r"(?:adjust-rollingShutter )?(?:adjust-360-transform )?(?:adjust-reorient )?"
    r"(?:adjust-orientation )?(?:adjust-cinematic )?"
)
_AUDIO = r"(?:adjust-volume )?(?:adjust-panner )?"
_VFILTERS = r"(?:(?:filter-video|filter-video-mask) )*"

CONTENT: dict[str, str] = {
    "fcpxml": r"(?:import-options )?(?:resources )?(?:library |(?:event )*)",
    "resources": r"(?:(?:asset|effect|format|media|locator) )*",
    "format": "",
    "effect": "",
    "asset": r"(?:media-rep )+(?:metadata )?",
    "media-rep": r"(?:bookmark )?",
    "library": r"(?:(?:event|smart-collection) )*",
    "event": r"(?:(?:clip|audition|mc-clip|ref-clip|sync-clip|asset-clip|collection-folder"
    r"|keyword-collection|smart-collection|project) )*",
    "project": r"sequence ",
    "sequence": r"(?:note )?spine (?:metadata )?",
    "spine": r"(?:(?:audio|video|clip|title|mc-clip|ref-clip|sync-clip|asset-clip|audition|gap"
    r"|transition) )*",
    "asset-clip": rf"(?:note )?(?:conform-rate )?(?:timeMap )?{_VIDEO}{_AUDIO}{_ANCHOR}"
    rf"{_MARKERS}(?:audio-channel-source )*{_VFILTERS}(?:filter-audio )*(?:metadata )?",
    "gap": rf"(?:note )?{_ANCHOR}{_MARKERS}(?:metadata )?",
    "title": rf"(?:param )*(?:text )*(?:text-style-def )*(?:note )?{_VIDEO}{_ANCHOR}{_MARKERS}"
    rf"{_VFILTERS}(?:metadata )?",
    "transition": rf"(?:filter-video )?(?:filter-audio )?{_MARKERS}(?:metadata )?(?:reserved )?",
    "filter-video": r"(?:data )*(?:param )*",
    "filter-audio": r"(?:data )?(?:param )*",
    "adjust-transform": r"(?:param )*",
    "adjust-volume": r"(?:param )*",
    "param": r"(?:fadeIn )?(?:fadeOut )?(?:keyframeAnimation )?(?:param )*",
    "keyframeAnimation": r"(?:keyframe )*",
    "keyframe": "",
    "fadeIn": "",
    "fadeOut": "",
    "marker": "",
    "text": r"(?:text-style )*",
    "text-style-def": r"text-style ",
    "text-style": r"(?:param )*",
    "note": "",
    "metadata": r"(?:md )*",
    "md": r"(?:array )?",
}

_FADE = frozenset({"linear", "easeIn", "easeOut", "easeInOut"})
_BOOL = frozenset({"0", "1"})
# element -> (allowed attributes, required attributes, enumerated values)
ATTRS: dict[str, tuple[frozenset[str], frozenset[str], dict[str, frozenset[str]]]] = {}


def _attrs(name: str, allowed: str, required: str = "", **enums: frozenset[str]) -> None:
    ATTRS[name] = (frozenset(allowed.split()), frozenset(required.split()), enums)


_attrs("fcpxml", "version", "version", version=frozenset({"1.10"}))
_attrs("resources", "")
_attrs(
    "format",
    "id name frameDuration fieldOrder width height paspH paspV colorSpace projection stereoscopic",
    "id",
)
_attrs("effect", "id name uid src", "id uid")
_attrs(
    "asset",
    "id name uid start duration hasVideo format hasAudio videoSources audioSources audioChannels "
    "audioRate customLUTOverride colorSpaceOverride projectionOverride stereoscopicOverride "
    "auxVideoFlags",
    "id",
)
_attrs(
    "media-rep",
    "kind sig src suggestedFilename",
    "src",
    kind=frozenset({"original-media", "proxy-media"}),
)
_attrs("library", "location colorProcessing")
_attrs("event", "name uid")
_attrs("project", "name uid id modDate")
_attrs(
    "sequence",
    "format duration tcStart tcFormat audioLayout audioRate renderFormat keywords",
    "format",
    tcFormat=frozenset({"DF", "NDF"}),
    audioLayout=frozenset({"mono", "stereo", "surround"}),
    audioRate=frozenset({"32k", "44.1k", "48k", "88.2k", "96k", "176.4k", "192k"}),
)
_attrs("spine", "lane offset name format")
_attrs(
    "asset-clip",
    "ref lane offset name start duration enabled srcEnable audioStart audioDuration format "
    "tcStart tcFormat modDate audioRole videoRole",
    "ref",
    srcEnable=frozenset({"all", "audio", "video"}),
    enabled=_BOOL,
    tcFormat=frozenset({"DF", "NDF"}),
)
_attrs("gap", "name offset start duration enabled", "duration", enabled=_BOOL)
_attrs("title", "ref lane offset name start duration enabled role", "ref duration", enabled=_BOOL)
_attrs("transition", "name offset duration", "duration")
_attrs("filter-video", "ref name enabled", "ref", enabled=_BOOL)
_attrs("filter-audio", "ref name enabled presetID", "ref", enabled=_BOOL)
_attrs("adjust-transform", "enabled position scale rotation anchor tracking", enabled=_BOOL)
_attrs("adjust-volume", "amount")
_attrs("param", "name key value enabled", "name", enabled=_BOOL)
_attrs("keyframeAnimation", "")
_attrs(
    "keyframe",
    "time value interp curve",
    "time value",
    interp=frozenset({"linear", "ease", "easeIn", "easeOut"}),
    curve=frozenset({"linear", "smooth"}),
)
_attrs("fadeIn", "type duration", "duration", type=_FADE)
_attrs("fadeOut", "type duration", "duration", type=_FADE)
_attrs("marker", "start duration value completed note", "start value")
_attrs("text", "display-style roll-up-height position placement alignment")
_attrs("text-style-def", "id name", "id")
_attrs(
    "text-style",
    "ref font fontSize fontFace fontColor backgroundColor bold italic strokeColor strokeWidth "
    "baseline shadowColor shadowOffset shadowBlurRadius kerning alignment lineSpacing tabStops "
    "baselineOffset underline",
    alignment=frozenset({"left", "center", "right", "justified"}),
)
_attrs("note", "")
_attrs("metadata", "")
_attrs("md", "key value editable type displayName description source", "key")

# Which resource kind each reference attribute must point at.
_REFS: dict[tuple[str, str], frozenset[str]] = {
    ("asset-clip", "ref"): frozenset({"asset"}),
    ("title", "ref"): frozenset({"effect"}),
    ("filter-video", "ref"): frozenset({"effect"}),
    ("filter-audio", "ref"): frozenset({"effect"}),
    ("sequence", "format"): frozenset({"format"}),
    ("asset", "format"): frozenset({"format"}),
    ("asset-clip", "format"): frozenset({"format"}),
    ("spine", "format"): frozenset({"format"}),
    ("text-style", "ref"): frozenset({"text-style-def"}),
}
_ENTITY = re.compile(r"<!ENTITY|<!DOCTYPE[^>]*\[")


# -- decoded timeline --------------------------------------------------------------------------


@dataclass
class RawClip:
    """An asset-clip as written: record frames of its extent, source frames of its start."""

    lane: int
    raw_in: int
    raw_out: int
    start: int
    src: str | None
    enable: str
    sound: tuple[int, int] | None  # source frames, when audioStart/audioDuration are given
    name: str
    note: str | None
    volume_db: float
    fade_in: int
    fade_out: int
    transform: TransformSpec | None  # keyframe times are clip-local source frames here


@dataclass
class RawTransition:
    offset: int
    frames: int
    video: bool
    audio: bool
    video_effect: str | None
    audio_effect: str | None


@dataclass
class RawTitle:
    lane: int
    record_in: int
    frames: int
    lines: tuple[str, ...]
    name: str
    effect: str | None


@dataclass
class RawMarker:
    record: int
    duration: int
    value: str
    note: str | None


@dataclass
class SpineEntry:
    """One spine child in order: ``kind`` is clip, gap or transition; ``index`` points into the
    matching list (clips or transitions); gaps keep their extent here."""

    kind: str
    index: int
    raw_in: int
    raw_out: int


@dataclass
class Decoded:
    problems: list[str] = field(default_factory=list)
    version: str | None = None
    timebase: Timebase | None = None
    frame_duration: str | None = None
    width: int | None = None
    height: int | None = None
    tc_start: int | None = None
    duration: int | None = None
    event: str | None = None
    project: str | None = None
    assets: dict[str, str] = field(default_factory=dict)  # asset id -> media-rep src
    effects: dict[str, str] = field(default_factory=dict)  # effect id -> name
    spine: list[SpineEntry] = field(default_factory=list)
    clips: list[RawClip] = field(default_factory=list)
    transitions: list[RawTransition] = field(default_factory=list)
    titles: list[RawTitle] = field(default_factory=list)
    markers: list[RawMarker] = field(default_factory=list)


class _Bad(ValueError):
    """A value that cannot be decoded; becomes a problem."""


# -- structure ---------------------------------------------------------------------------------


def check_structure(root: ET.Element) -> list[str]:
    """Element, attribute and child-order problems against the FCPXML 1.10 subset above, plus
    duplicate ids and references that do not resolve to the right kind of resource."""
    problems: list[str] = []
    ids: dict[str, str] = {}
    for element in root.iter():
        tag = element.tag
        if not isinstance(tag, str) or tag not in CONTENT:
            problems.append(f"element <{tag}> is outside the FCPXML subset aar reads")
            continue
        children = "".join(f"{child.tag} " for child in element)
        if not re.fullmatch(CONTENT[tag], children):
            problems.append(f"<{tag}> has children out of FCPXML 1.10 order: {children.strip()}")
        if tag not in ("text", "text-style", "note") and (element.text or "").strip():
            problems.append(f"<{tag}> has text content, which its model does not allow")
        allowed, required, enums = ATTRS[tag]
        for name, value in element.attrib.items():
            if name not in allowed:
                problems.append(f"<{tag}> has attribute {name!r}, not in FCPXML 1.10")
            elif name in enums and value not in enums[name]:
                problems.append(f"<{tag} {name}={value!r}> is not one of {sorted(enums[name])}")
        for name in sorted(required - set(element.attrib)):
            problems.append(f"<{tag}> is missing required attribute {name!r}")
        identifier = element.get("id")
        if identifier is not None:
            if identifier in ids:
                problems.append(f"id {identifier!r} is used twice")
            ids.setdefault(identifier, tag)
    for element in root.iter():
        for (tag, attr), kinds in _REFS.items():
            if element.tag != tag or attr not in element.attrib:
                continue
            target = ids.get(element.attrib[attr])
            if target not in kinds:
                problems.append(
                    f"<{tag} {attr}={element.attrib[attr]!r}> does not resolve to "
                    f"{' or '.join(sorted(kinds))} (found {target or 'nothing'})"
                )
    return problems


# -- decode ------------------------------------------------------------------------------------


def _floats(text: str, count: int, what: str) -> tuple[float, ...]:
    parts = text.split()
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        raise _Bad(f"{what} {text!r} is not {count} numbers") from None
    if len(values) != count:
        raise _Bad(f"{what} {text!r} is not {count} numbers")
    return values


def _db(text: str) -> float:
    if not text.endswith("dB"):
        raise _Bad(f"volume {text!r} is not in dB")
    try:
        return float(text[:-2])
    except ValueError:
        raise _Bad(f"volume {text!r} is not a number of dB") from None


class _Decoder:
    def __init__(self, root: ET.Element, out: Decoded) -> None:
        self.root = root
        self.out = out
        self.tb: Timebase | None = None
        self.tc = 0

    def frames(self, element: ET.Element, attr: str, default: str | None = None) -> int:
        value = element.get(attr, default)
        if value is None:
            raise _Bad(f"<{element.tag}> has no {attr}")
        assert self.tb is not None
        try:
            return self.tb.frames(value)
        except ValueError as exc:
            raise _Bad(f"<{element.tag} {attr}>: {exc}") from None

    def run(self) -> None:
        out = self.out
        root = self.root
        out.version = root.get("version")
        resources = root.find("resources")
        formats: dict[str, ET.Element] = {}
        if resources is not None:
            for child in resources:
                identifier = child.get("id", "")
                if child.tag == "format":
                    formats[identifier] = child
                elif child.tag == "effect":
                    out.effects[identifier] = child.get("name", "")
                elif child.tag == "asset":
                    rep = child.find("media-rep")
                    out.assets[identifier] = rep.get("src", "") if rep is not None else ""
        projects = root.findall("library/event/project")
        if len(projects) != 1:
            raise _Bad(f"expected one project, found {len(projects)}")
        event = root.find("library/event")
        out.event = event.get("name") if event is not None else None
        project = projects[0]
        out.project = project.get("name")
        sequence = project.find("sequence")
        if sequence is None:
            raise _Bad("the project has no sequence")
        fmt = formats.get(sequence.get("format", ""))
        if fmt is None:
            raise _Bad("the sequence's format does not resolve")
        out.frame_duration = fmt.get("frameDuration")
        if out.frame_duration is None:
            raise _Bad("the sequence's format has no frameDuration")
        try:
            rate = 1 / parse_time(out.frame_duration)
        except (ValueError, ZeroDivisionError) as exc:
            raise _Bad(f"frameDuration: {exc}") from None
        if Fraction(rate) not in SUPPORTED_RATES:
            raise _Bad(f"frameDuration {out.frame_duration!r} is not a supported rate")
        self.tb = out.timebase = Timebase(Fraction(rate))
        out.width = int(fmt.get("width", "0") or 0)
        out.height = int(fmt.get("height", "0") or 0)
        self.tc = out.tc_start = self.frames(sequence, "tcStart", "0s")
        out.duration = self.frames(sequence, "duration")
        spine = sequence.find("spine")
        if spine is None:
            raise _Bad("the sequence has no spine")
        self.spine(spine)

    def spine(self, spine: ET.Element) -> None:
        for child in spine:
            if child.tag == "transition":
                offset = self.frames(child, "offset", "0s") - self.tc
                frames = self.frames(child, "duration")
                fv = child.find("filter-video")
                fa = child.find("filter-audio")
                self.out.transitions.append(
                    RawTransition(
                        offset,
                        frames,
                        fv is not None,
                        fa is not None,
                        self.out.effects.get(fv.get("ref", "")) if fv is not None else None,
                        self.out.effects.get(fa.get("ref", "")) if fa is not None else None,
                    )
                )
                end = offset + frames
                self.out.spine.append(
                    SpineEntry("transition", len(self.out.transitions) - 1, offset, end)
                )
            elif child.tag in ("asset-clip", "gap"):
                raw_in = self.frames(child, "offset", "0s") - self.tc
                start = self.frames(child, "start", "0s")
                raw_out = raw_in + self.frames(child, "duration")
                if child.tag == "asset-clip":
                    self.out.clips.append(self.clip(child, 0, raw_in, start))
                    entry = SpineEntry("clip", len(self.out.clips) - 1, raw_in, raw_out)
                else:
                    entry = SpineEntry("gap", -1, raw_in, raw_out)
                self.out.spine.append(entry)
                self.children(child, raw_in, start, raw_out - raw_in)
            else:
                raise _Bad(f"the spine holds a <{child.tag}>, which aar does not write")

    def clip(self, element: ET.Element, lane: int, raw_in: int, start: int) -> RawClip:
        duration = self.frames(element, "duration")
        sound = None
        if "audioStart" in element.attrib or "audioDuration" in element.attrib:
            audio_start = self.frames(element, "audioStart", element.get("start", "0s"))
            audio_duration = self.frames(element, "audioDuration", element.get("duration"))
            sound = (audio_start, audio_start + audio_duration)
        volume, fade_in, fade_out = 0.0, 0, 0
        adjust = element.find("adjust-volume")
        if adjust is not None:
            volume = _db(adjust.get("amount", "0dB"))
            for param in adjust.findall("param"):
                if param.get("name") != "amount":
                    raise _Bad(f"adjust-volume param {param.get('name')!r} is not decoded")
                fi, fo = param.find("fadeIn"), param.find("fadeOut")
                fade_in = self.frames(fi, "duration") if fi is not None else 0
                fade_out = self.frames(fo, "duration") if fo is not None else 0
        note = element.find("note")
        return RawClip(
            lane=lane,
            raw_in=raw_in,
            raw_out=raw_in + duration,
            start=start,
            src=self.out.assets.get(element.get("ref", "")),
            enable=element.get("srcEnable", "all"),
            sound=sound,
            name=element.get("name", ""),
            note=note.text if note is not None else None,
            volume_db=volume,
            fade_in=fade_in,
            fade_out=fade_out,
            transform=self.transform(element.find("adjust-transform")),
        )

    def transform(self, adjust: ET.Element | None) -> TransformSpec | None:
        if adjust is None:
            return None
        position = _floats(adjust.get("position", "0 0"), 2, "position")
        scale = _floats(adjust.get("scale", "1 1"), 2, "scale")
        rotation = _floats(adjust.get("rotation", "0"), 1, "rotation")[0]
        tracks: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
        for param in adjust.findall("param"):
            name = param.get("name", "")
            if name not in ("position", "scale"):
                raise _Bad(f"adjust-transform param {name!r} is not decoded")
            keys = param.findall("keyframeAnimation/keyframe")
            tracks[name] = [
                (self.frames(k, "time"), _floats(k.get("value", ""), 2, name)) for k in keys
            ]
        keyframes: tuple[Keyframe, ...] = ()
        if tracks:
            pos, sca = tracks.get("position", []), tracks.get("scale", [])
            if [t for t, _ in pos] != [t for t, _ in sca]:
                raise _Bad("position and scale keyframes are at different times")
            keyframes = tuple(
                Keyframe(t, (p[0], p[1]), (s[0], s[1]))
                for (t, p), (_, s) in zip(pos, sca, strict=True)
            )
        return TransformSpec((position[0], position[1]), (scale[0], scale[1]), rotation, keyframes)

    def children(self, parent: ET.Element, raw_in: int, start: int, duration: int) -> None:
        """Anchored items and markers inside a spine element: record = parent's record start +
        (local time - parent's start)."""
        for child in parent:
            if child.tag not in ("asset-clip", "title", "marker"):
                continue
            local = self.frames(child, "start" if child.tag == "marker" else "offset", "0s")
            if not 0 <= local - start <= duration:
                raise _Bad(
                    f"<{child.tag}> in <{parent.tag}> is anchored outside its parent "
                    f"(local {local}, parent {start}..{start + duration})"
                )
            record = raw_in + local - start
            if child.tag == "marker":
                self.out.markers.append(
                    RawMarker(
                        record,
                        self.frames(child, "duration", self.tb.time(1) if self.tb else None),
                        child.get("value", ""),
                        child.get("note"),
                    )
                )
                continue
            lane = int(child.get("lane", "0"))
            if lane == 0:
                raise _Bad(f"an anchored <{child.tag}> has lane 0")
            if child.tag == "title":
                style = child.find("text/text-style")
                text = style.text if style is not None and style.text is not None else ""
                self.out.titles.append(
                    RawTitle(
                        lane,
                        record,
                        self.frames(child, "duration"),
                        tuple(text.split("\n")),
                        child.get("name", ""),
                        self.out.effects.get(child.get("ref", "")),
                    )
                )
                continue
            clip = self.clip(child, lane, record, self.frames(child, "start", "0s"))
            self.out.clips.append(clip)
            nested = [c.tag for c in child if c.tag in ("asset-clip", "title", "marker")]
            if nested:
                raise _Bad(f"a connected clip holds {nested}; aar writes none")


def read_fcpxml(text: str) -> Decoded:
    """Parse FCPXML text; problems are listed in ``Decoded.problems`` (empty when it is clean)."""
    out = Decoded()
    if len(text.encode("utf-8")) > MAX_BYTES:
        out.problems.append(f"the file is over {MAX_BYTES} bytes")
        return out
    if _ENTITY.search(text):
        out.problems.append("the file declares entities or a DOCTYPE internal subset (refused)")
        return out
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        out.problems.append(f"not well-formed XML: {exc}")
        return out
    if root.tag != "fcpxml":
        out.problems.append(f"the root element is <{root.tag}>, not <fcpxml>")
        return out
    out.problems.extend(check_structure(root))
    try:
        _Decoder(root, out).run()
    except (_Bad, ValueError) as exc:
        out.problems.append(str(exc))
    return out
