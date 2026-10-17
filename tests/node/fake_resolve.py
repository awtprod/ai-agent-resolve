"""An in-memory stand-in for the DaVinci Resolve scripting objects.

Method names and return conventions follow the 21.1 API stubs (``Resolve``, ``ProjectManager``,
``Project``, ``Timeline``, ``TimelineItem``, ``MediaPool``, ``MediaPoolItem``, ``Folder``,
``Graph``), so ``ScriptApiGateway`` and the op library run against it unchanged. Return shapes
follow what a read-only probe of a live 21.1 instance returned (ints for item positions,
floats for fades and ``FPS``, strings for most clip properties, ``""`` for the Project Manager
root).

Mutating calls (and Project Manager navigation and page changes) are recorded in
``FakeResolve.calls``. Render presets are shared by all projects, like Resolve's user presets.
``human_*`` helpers change state the way a person at the GUI would, without the gateway.

Every known trap of ``aar.node.traps`` is modelled here as it was observed on Resolve 21.1.
Each modelled behaviour carries a ``# trap: <id>`` comment. Where the observation does not
pin the exact numbers down, the comment says what the model chose.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

PAGES = ("media", "photo", "cut", "edit", "fusion", "color", "fairlight", "deliver")
_TIMECODE = re.compile(r"^\d{2}:\d{2}:\d{2}[:;]\d{2}$")
MARKER_COLORS = (
    "Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple", "Fuchsia",
    "Rose", "Lavender", "Sky", "Mint", "Lemon", "Sand", "Cocoa", "Cream",
)  # fmt: skip

# Colour keys of TimelineSettings that a per-timeline custom resolution resets.
COLOUR_KEYS = (
    "colorScienceMode",
    "isAutoColorManage",
    "rcmPresetMode",
    "colorSpaceTimeline",
    "colorSpaceOutput",
    "inputDRT",
    "outputDRT",
    "useInverseDRT",
    "useColorSpaceAwareGradingTools",
)
_PLAIN_YRGB = {
    "colorScienceMode": "davinciYRGB",
    "isAutoColorManage": "0",
    "rcmPresetMode": "Custom",
    "colorSpaceTimeline": "Rec.709 Gamma 2.4",
    "colorSpaceOutput": "Rec.709 Gamma 2.4",
    "inputDRT": "None",
    "outputDRT": "None",
    "useInverseDRT": "0",
    "useColorSpaceAwareGradingTools": "0",
}

_VIDEO_PROPS: dict[str, Any] = {
    "Pan": 0.0,
    "Tilt": 0.0,
    "ZoomX": 1.0,
    "ZoomY": 1.0,
    "Opacity": 100.0,
    "CompositeMode": 0,
}
_AUDIO_PROPS: dict[str, Any] = {
    "AudioVolumeEnabled": True,
    "AudioVolume": 0.0,
    "AudioPanEnabled": True,
    "AudioPan": 0.0,
}
_PROP_RANGES = {"AudioVolume": (-100.0, 30.0), "AudioPan": (-100.0, 100.0), "Opacity": (0.0, 100.0)}

# Resolve's nominal float rates and the exact rates behind them.
_EXACT_RATES = {
    23.976: Fraction(24000, 1001),
    29.97: Fraction(30000, 1001),
    47.952: Fraction(48000, 1001),
    59.94: Fraction(60000, 1001),
}


def _rate(fps: float) -> Fraction:
    for nominal, exact in _EXACT_RATES.items():
        if abs(fps - nominal) < 1e-6:
            return exact
    return Fraction(str(fps))


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class DeliverState:
    format: str = "mp4"
    codec: str = "H264"
    mode: int = 1
    settings: dict[str, Any] = field(
        default_factory=lambda: {"TargetDir": "D:/Renders", "CustomName": "master"}
    )

    def copy(self) -> DeliverState:
        return DeliverState(self.format, self.codec, self.mode, dict(self.settings))


# -- media pool ------------------------------------------------------------------------------

_KIND_TYPE = {
    "av": "Video + Audio",
    "video": "Video",
    "audio": "Audio",
    "still": "Still",  # model choice: the live probe found no stills, so this value is assumed
    "multicam": "Multicam",
    "compound": "Compound",
    "timeline": "Timeline",
}


class FakeMediaPoolItem:
    """A media-pool clip. ``kind`` is one of ``_KIND_TYPE``; ``braw`` marks Blackmagic RAW."""

    def __init__(
        self,
        app: FakeResolve,
        name: str,
        *,
        kind: str = "av",
        fps: float = 29.97,
        frames: int = 18000,
        braw: bool = False,
        input_color_space: str | None = None,
        online: str = "Online",
        proxy: str = "None",
    ) -> None:
        if kind not in _KIND_TYPE:
            raise ValueError(f"unknown clip kind {kind!r}")
        self._app = app
        self._name = name
        self._uid = f"mpi-{next(app.ids)}"
        self.kind = kind
        self.braw = braw
        self.fps = fps
        self.frames = frames
        has_video = kind in ("av", "video", "still", "multicam", "compound", "timeline")
        ext = "braw" if braw else {"audio": "wav", "still": "png"}.get(kind, "mov")
        if kind in ("multicam", "compound", "timeline"):
            ext = ""
        default_space = (
            "Blackmagic Design Film Gen 5" if braw else ("Rec.709 (Scene)" if has_video else "")
        )
        self.props: dict[str, Any] = {
            "Clip Name": name,
            "File Name": f"{name}.{ext}" if ext else "",
            "File Path": f"/media/{name}.{ext}" if ext else "",
            "Type": _KIND_TYPE[kind],
            "Format": "BRAW" if braw else {"audio": "Wave", "still": "PNG"}.get(kind, "QuickTime"),
            "Video Codec": "Blackmagic RAW" if braw else ("H.264 High L5.1" if has_video else ""),
            "FPS": float(fps),
            "Frames": str(frames),
            "Start": "0",
            "End": str(frames - 1),
            "Input Color Space": (
                input_color_space if input_color_space is not None else default_space
            ),
            # trap: render-uses-original-media: "Online Status" is the original media's; a clip
            # can read "Offline" while a linked proxy ("Proxy" is its size) still plays.
            "Online Status": online,
            "Proxy": proxy,
            "Audio Ch": "2" if kind in ("av", "audio") else "0",
        }  # fmt: skip

    @property
    def has_video(self) -> bool:
        return self.kind != "audio"

    @property
    def has_audio(self) -> bool:
        return self.kind in ("av", "audio", "multicam")

    def GetName(self) -> str:
        return self._name

    def GetUniqueId(self) -> str:
        return self._uid

    def GetMediaId(self) -> str:
        return f"media-{self._uid}"

    def GetClipProperty(self, propertyName: str | None = None) -> Any:
        if propertyName is None:
            return dict(self.props)
        return self.props.get(propertyName, "")

    def SetClipProperty(self, propertyName: str, propertyValue: str) -> bool:
        self._app.calls.append(("SetClipProperty", self._uid, propertyName, propertyValue))
        if propertyName != "Input Color Space":
            return False  # only the colour-space key is modelled
        if self.kind in ("audio", "multicam", "compound", "timeline"):
            return False
        project = self._app.current
        if project is None or "ColorManaged" not in project.settings.get("colorScienceMode", ""):
            return False
        # A BRAW clip accepts the change, which is the harm: its decode colour space is set
        # by the RAW settings, and overriding it breaks the grade. The guard is in the op.
        self.props["Input Color Space"] = propertyValue
        return True


class FakeFolder:
    def __init__(
        self,
        name: str,
        uid: str,
        subfolders: list[FakeFolder] | None = None,
        clips: list[FakeMediaPoolItem] | None = None,
    ) -> None:
        self._name = name
        self._uid = uid
        self.subfolders = subfolders or []
        self.clips = clips or []

    def GetName(self) -> str:
        return self._name

    def GetUniqueId(self) -> str:
        return self._uid

    def GetSubFolderList(self) -> list[FakeFolder]:
        return list(self.subfolders)

    def GetClipList(self) -> list[FakeMediaPoolItem]:
        return list(self.clips)

    def walk(self) -> list[FakeFolder]:
        out = [self]
        for sub in self.subfolders:
            out.extend(sub.walk())
        return out


class FakeMediaPool:
    def __init__(self, app: FakeResolve, project: FakeProject, root: FakeFolder) -> None:
        self._app = app
        self._project = project
        self._root = root
        self.current = root

    def GetRootFolder(self) -> FakeFolder:
        return self._root

    def GetCurrentFolder(self) -> FakeFolder:
        return self.current

    def SetCurrentFolder(self, folder: FakeFolder) -> bool:
        self._app.calls.append(("SetCurrentFolder", folder.GetName()))
        if not any(f is folder for f in self._root.walk()):
            return False
        self.current = folder
        return True

    def AppendToTimeline(self, clipInfos: list[dict[str, Any]]) -> list[FakeTimelineItem]:
        self._app.calls.append(
            ("AppendToTimeline", [_summarise_clip_info(info) for info in clipInfos])
        )
        timeline = self._project.current_timeline
        if timeline is None:
            return []
        returned: list[FakeTimelineItem] = []
        for info in clipInfos:
            made = timeline.append(info)
            if made is None:
                return []
            returned.extend(made)
        return returned

    # -- test helpers ----------------------------------------------------------------------

    def folder(self, *path: str) -> FakeFolder:
        node = self._root
        for name in path:
            node = next(f for f in node.subfolders if f.GetName() == name)
        return node

    def add_clip(self, path: tuple[str, ...], name: str, **kwargs: Any) -> FakeMediaPoolItem:
        clip = FakeMediaPoolItem(self._app, name, **kwargs)
        self.folder(*path).clips.append(clip)
        return clip

    def import_rec709(self, path: tuple[str, ...], name: str, **kwargs: Any) -> FakeMediaPoolItem:
        """A human imports a Rec.709 file into this (colour-managed) project."""
        # trap: rcm-input-colorspace: imported Rec.709 media inherits the project's camera-log
        # input default instead of reading as Rec.709.
        default = self._project.settings.get("colorSpaceInput", "Rec.709 (Scene)")
        return self.add_clip(path, name, input_color_space=default, **kwargs)


def _summarise_clip_info(info: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in info.items() if k != "mediaPoolItem"}
    mpi = info.get("mediaPoolItem")
    out["mediaPoolItem"] = mpi.GetUniqueId() if mpi is not None else None
    return out


# -- colour ---------------------------------------------------------------------------------


@dataclass
class FakeNode:
    label: str = ""
    lut: str = ""
    tools: tuple[str, ...] = ()


class FakeGraph:
    def __init__(self, app: FakeResolve, nodes: list[FakeNode] | None = None) -> None:
        self._app = app
        self.nodes = nodes if nodes is not None else [FakeNode()]

    def copy(self) -> FakeGraph:
        return FakeGraph(self._app, [FakeNode(n.label, n.lut, n.tools) for n in self.nodes])

    def GetNumNodes(self) -> int:
        return len(self.nodes)

    def GetLUT(self, nodeIndex: int) -> str:
        if not 1 <= nodeIndex <= len(self.nodes):
            return ""
        return self.nodes[nodeIndex - 1].lut

    def SetLUT(self, nodeIndex: int, lutPath: str) -> bool:
        self._app.calls.append(("SetLUT", nodeIndex, lutPath))
        if not 1 <= nodeIndex <= len(self.nodes) or not lutPath:
            return False
        if lutPath in self._app.unknown_luts:
            return False  # Resolve only applies LUTs it has discovered
        self.nodes[nodeIndex - 1].lut = lutPath
        return True

    def GetNodeLabel(self, nodeIndex: int) -> str:
        if not 1 <= nodeIndex <= len(self.nodes):
            return ""
        return self.nodes[nodeIndex - 1].label

    def GetToolsInNode(self, nodeIndex: int) -> list[str]:
        if not 1 <= nodeIndex <= len(self.nodes):
            return []
        return list(self.nodes[nodeIndex - 1].tools)


# -- timelines ------------------------------------------------------------------------------


class FakeTimelineItem:
    def __init__(
        self,
        timeline: FakeTimeline,
        track_type: str,
        mpi: FakeMediaPoolItem | None,
        start: int,
        end: int,
        src_start: int,
        src_end: int,
        *,
        graph: FakeGraph | None = None,
    ) -> None:
        app = timeline._app
        self._app = app
        self.timeline = timeline
        self.track_type = track_type
        self.mpi = mpi
        self.start = start
        self.end = end
        self.src_start = src_start
        self.src_end = src_end
        self._uid = f"ti-{next(app.ids)}"
        self.fades: dict[str, float] = {"FadeIn": 0.0, "FadeOut": 0.0}
        self.props: dict[str, Any] = dict(_AUDIO_PROPS if track_type == "audio" else _VIDEO_PROPS)
        self.graph = graph if graph is not None else FakeGraph(app)
        self.angle: int | None = 1 if mpi is not None and mpi.kind == "multicam" else None
        self.source_readback_skew = app.source_readback_skew
        self.item_type: str | None = None  # "transition" for a transition item

    def clone(self, timeline: FakeTimeline) -> FakeTimelineItem:
        twin = FakeTimelineItem(
            timeline,
            self.track_type,
            self.mpi,
            self.start,
            self.end,
            self.src_start,
            self.src_end,
            graph=self.graph.copy(),
        )
        twin.fades = dict(self.fades)
        twin.props = dict(self.props)
        twin.angle = self.angle
        return twin

    def GetType(self) -> str:
        # Live 21.1 lists transitions in GetItemListInTrack as items of type "transition",
        # without media, straddling the cut between two clips.
        return self.item_type or self.track_type

    def GetName(self) -> str:
        return self.mpi.GetName() if self.mpi is not None else "Generator"

    def GetUniqueId(self) -> str:
        return self._uid

    def GetStart(self, subframePrecision: bool = False) -> float:
        return float(self.start) if subframePrecision else self.start

    def GetEnd(self, subframePrecision: bool = False) -> float:
        return float(self.end) if subframePrecision else self.end

    def GetDuration(self, subframePrecision: bool = False) -> float:
        n = self.end - self.start
        return float(n) if subframePrecision else n

    def GetSourceStartFrame(self) -> int | None:
        if self.item_type == "transition":
            return None  # model: live transitions read back no numeric source frames
        # trap: source-frame-readback-rounding: live readback can sit one frame off the in
        # point that was asked for. ``source_readback_skew`` models it per item.
        return self.src_start + self.source_readback_skew

    def GetSourceEndFrame(self) -> int | None:
        return None if self.item_type == "transition" else self.src_end

    def GetMediaPoolItem(self) -> FakeMediaPoolItem | None:
        return self.mpi

    def GetTrackTypeAndIndex(self) -> list[str | int]:
        return [self.track_type, self.timeline.track_index_of(self)]

    def GetLinkedItems(self) -> list[FakeTimelineItem]:
        return []

    def GetClipEnabled(self) -> bool:
        return True

    def GetFades(self) -> dict[str, float]:
        return {} if self.item_type == "transition" else dict(self.fades)

    def SetFades(self, fades: dict[str, int]) -> bool:
        self._app.calls.append(("SetFades", self._uid, dict(fades)))
        if not fades or set(fades) - {"FadeIn", "FadeOut"}:
            return False
        new = dict(self.fades)
        for key, value in fades.items():
            if not isinstance(value, int) or value < 0:
                return False
            new[key] = float(value)
        if new["FadeIn"] + new["FadeOut"] > self.end - self.start:
            return False
        self.fades = new
        return True

    def GetProperties(self) -> dict[str, Any]:
        return dict(self.props)

    def SetProperties(self, properties: dict[str, Any]) -> bool:
        self._app.calls.append(("SetProperties", self._uid, dict(properties)))
        return self._set_props(properties)

    def SetProperty(self, propertyKey: Any, propertyValue: Any = None) -> bool:
        # Deprecated calling convention, still answered by 21.1.
        self._app.calls.append(("SetProperty", self._uid, propertyKey, propertyValue))
        if isinstance(propertyKey, dict):
            return self._set_props(propertyKey)
        return self._set_props({propertyKey: propertyValue})

    def GetProperty(self, propertyKey: str | None = None) -> Any:
        if propertyKey is None:
            return dict(self.props)
        return self.props.get(propertyKey)

    def _set_props(self, properties: dict[str, Any]) -> bool:
        if self._app.props_stuck:
            # trap: setproperty-after-color: after coming back to Edit from Color every
            # property setter returns False until the page is bounced.
            return False
        for key in properties:
            # trap: volume-key: keys are validated before anything is applied; the gain key
            # is AudioVolume, so "Volume" makes the whole call return False.
            if key not in self.props:
                return False
        for key, value in properties.items():
            low, high = _PROP_RANGES.get(key, (-math.inf, math.inf))
            self.props[key] = min(max(value, low), high) if isinstance(value, float) else value
        return True

    def GetNodeGraph(self, layerIdx: int | None = None) -> FakeGraph:
        return self.graph

    def CopyGrades(self, tgtTimelineItems: list[FakeTimelineItem]) -> bool:
        self._app.calls.append(("CopyGrades", self._uid, [t._uid for t in tgtTimelineItems]))
        if self.track_type != "video" or any(t.track_type != "video" for t in tgtTimelineItems):
            return False
        # trap: append-ungraded: CopyGrades works across timelines, and it replaces the
        # target's whole node graph, per-clip LUTs included.
        for target in tgtTimelineItems:
            target.graph = self.graph.copy()
        return True


@dataclass
class FakeTrack:
    name: str
    subtype: str = ""
    enabled: bool = True
    locked: bool = False
    items: list[FakeTimelineItem] = field(default_factory=list)


class FakeTimeline:
    def __init__(
        self,
        app: FakeResolve,
        name: str,
        uid: str,
        project: FakeProject | None = None,
        *,
        start_frame: int = 108000,
    ) -> None:
        self._app = app
        self._name = name
        self._uid = uid
        self.project = project
        self.timecode = "01:00:00:00"
        self.start_frame = start_frame
        self.tracks: dict[str, list[FakeTrack]] = {
            "video": [FakeTrack("Video 1")],
            "audio": [FakeTrack("Audio 1", "stereo")],
            "subtitle": [],
        }
        self.custom_settings: dict[str, Any] | None = None
        self.markers: dict[int, dict[str, Any]] = {}

    # -- Resolve API -----------------------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def GetUniqueId(self) -> str:
        return self._uid

    def GetStartTimecode(self) -> str:
        return "01:00:00:00"

    def GetCurrentTimecode(self) -> str:
        return self.timecode

    def SetCurrentTimecode(self, timecode: str) -> bool:
        self._app.calls.append(("SetCurrentTimecode", timecode))
        if not _TIMECODE.match(timecode):
            return False
        self.timecode = timecode
        return True

    def GetStartFrame(self) -> int:
        return self.start_frame

    def GetEndFrame(self) -> int:
        ends = [item.end for tracks in self.tracks.values() for t in tracks for item in t.items]
        return max(ends, default=self.start_frame)

    def GetTrackCount(self, trackType: str) -> int:
        return len(self.tracks.get(trackType, []))

    def _track(self, trackType: str, index: int) -> FakeTrack | None:
        tracks = self.tracks.get(trackType, [])
        return tracks[index - 1] if 1 <= index <= len(tracks) else None

    def GetItemListInTrack(self, trackType: str, index: int) -> list[FakeTimelineItem]:
        track = self._track(trackType, index)
        return sorted(track.items, key=lambda i: i.start) if track is not None else []

    def GetTrackName(self, trackType: str, trackIndex: int) -> str:
        track = self._track(trackType, trackIndex)
        return track.name if track is not None else ""

    def GetIsTrackEnabled(self, trackType: str, trackIndex: int) -> bool:
        track = self._track(trackType, trackIndex)
        return track is not None and track.enabled

    def GetIsTrackLocked(self, trackType: str, trackIndex: int) -> bool:
        track = self._track(trackType, trackIndex)
        return track is not None and track.locked

    def GetTrackSubType(self, trackType: str, trackIndex: int) -> str:
        track = self._track(trackType, trackIndex)
        return track.subtype if track is not None else ""

    def AddTrack(self, trackType: str, subTrackType: str | None = None) -> bool:
        self._app.calls.append(("AddTrack", trackType, subTrackType))
        if trackType not in self.tracks:
            return False
        n = len(self.tracks[trackType]) + 1
        label = {"video": "Video", "audio": "Audio", "subtitle": "Subtitle"}[trackType]
        self.tracks[trackType].append(FakeTrack(f"{label} {n}", subTrackType or ""))
        return True

    def GetSettings(self) -> dict[str, Any]:
        if self.custom_settings is None:
            assert self.project is not None
            return dict(self.project.settings)  # no useCustomSettings key, as observed live
        return dict(self.custom_settings)

    def SetSettings(self, settings: dict[str, Any]) -> bool:
        self._app.calls.append(("Timeline.SetSettings", self._uid, dict(settings)))
        ok = True
        if "useCustomSettings" in settings:
            if settings["useCustomSettings"] == "1" and self.custom_settings is None:
                assert self.project is not None
                self.custom_settings = {**self.project.settings, "useCustomSettings": "1"}
            elif settings["useCustomSettings"] == "0":
                self.custom_settings = None
        for key, value in settings.items():
            if key == "useCustomSettings":
                continue
            if self.custom_settings is None:
                ok = False  # only a custom timeline's settings can change
                continue
            self.custom_settings[key] = value
            if key in ("timelineResolutionWidth", "timelineResolutionHeight"):
                # trap: custom-res-resets-colour-science: a per-timeline custom resolution
                # drops the timeline back to plain DaVinci YRGB colour science.
                self.custom_settings.update(_PLAIN_YRGB)
        return ok

    def GetMarkers(self) -> dict[int, dict[str, Any]]:
        return {frame: dict(info) for frame, info in self.markers.items()}

    def AddMarker(
        self,
        frameId: int,
        color: str,
        name: str,
        note: str,
        duration: int,
        customData: str | None = None,
    ) -> bool:
        self._app.calls.append(("AddMarker", frameId, color, name, note, duration, customData))
        if frameId in self.markers or color not in MARKER_COLORS or duration < 1 or frameId < 0:
            return False
        self.markers[frameId] = {
            "color": color,
            "duration": duration,
            "note": note,
            "name": name,
            "customData": customData or "",
        }
        return True

    def DeleteClips(
        self, timelineItems: list[FakeTimelineItem], rippleDelete: bool = False
    ) -> bool:
        self._app.calls.append(("DeleteClips", [i.GetUniqueId() for i in timelineItems]))
        found = True
        for item in timelineItems:
            for tracks in self.tracks.values():
                for track in tracks:
                    if item in track.items:
                        track.items.remove(item)
                        break
                else:
                    continue
                break
            else:
                found = False
        return found

    def DuplicateTimeline(self, timelineName: str) -> FakeTimeline | None:
        self._app.calls.append(("DuplicateTimeline", self._name, timelineName))
        project = self.project
        assert project is not None
        if not timelineName or any(t.GetName() == timelineName for t in project.timelines):
            return None
        twin = FakeTimeline(
            self._app,
            timelineName,
            f"tl-{next(self._app.ids)}",
            project,
            start_frame=self.start_frame,
        )
        twin.tracks = {
            kind: [
                FakeTrack(t.name, t.subtype, t.enabled, t.locked, [i.clone(twin) for i in t.items])
                for t in tracks
            ]
            for kind, tracks in self.tracks.items()
        }
        twin.custom_settings = dict(self.custom_settings) if self.custom_settings else None
        twin.markers = {f: dict(m) for f, m in self.markers.items()}
        project.timelines.append(twin)
        return twin

    # -- the append model ------------------------------------------------------------------

    def fps(self) -> float:
        return float(self.GetSettings()["timelineFrameRate"])

    def track_index_of(self, item: FakeTimelineItem) -> int:
        for idx, track in enumerate(self.tracks[item.track_type], start=1):
            if item in track.items:
                return idx
        return 0

    def _free(self, track: FakeTrack, start: int, end: int) -> bool:
        return all(item.end <= start or item.start >= end for item in track.items)

    def append(self, info: dict[str, Any]) -> list[FakeTimelineItem] | None:
        """``MediaPool.AppendToTimeline`` for one ``AppendClipInfo``. ``None``: it failed."""
        mpi: FakeMediaPoolItem = info["mediaPoolItem"]
        media_type = info.get("mediaType")
        index = int(info.get("trackIndex", 1))
        if mpi.kind == "still":
            # trap: append-still-ignores-range: a still ignores startFrame/endFrame and lands
            # with a 150-frame default length and an offset. Model: 150 frames, placed one
            # frame after the record position asked for.
            src_start, src_end, length, offset = 0, 150, 150, 1
        else:
            # trap: append-endframe-exclusive: endFrame is exclusive, so [start, end) is used.
            src_start = int(info.get("startFrame", 0))
            src_end = int(info.get("endFrame", mpi.frames))
            if src_start < 0 or src_end > mpi.frames or src_end <= src_start:
                return None
            n = src_end - src_start
            tl_rate, src_rate = _rate(self.fps()), _rate(mpi.fps)
            # trap: append-fps-short: frames are in source fps; across a rate change the
            # timeline length is truncated, so e.g. 24 fps into 29.97 can land 1 frame short.
            length = n if tl_rate == src_rate else math.floor(n * tl_rate / src_rate)
            offset = 0
        want_video = mpi.has_video and media_type in (None, 1)
        want_audio = mpi.has_audio and media_type in (None, 2)
        if not want_video and not want_audio:
            return None
        made: list[FakeTimelineItem] = []
        for kind, wanted in (("video", want_video), ("audio", want_audio)):
            if not wanted:
                continue
            track = self._track(kind, index)
            if track is None:
                return None
            record = info.get("recordFrame")
            if record is None:
                record = max((i.end for i in track.items), default=self.start_frame)
            start = int(record) + offset
            if not self._free(track, start, start + length):
                return None  # model choice: an occupied slot makes the append fail
            # trap: append-ungraded: appended items come in with an ungraded node graph.
            item = FakeTimelineItem(self, kind, mpi, start, start + length, src_start, src_end)
            if mpi.kind == "multicam":
                # trap: multicam-angles: appended multicam clips lose their angle.
                item.angle = None
            track.items.append(item)
            made.append(item)
        if mpi.braw and media_type is None:
            # trap: av-append-video-only: an 'av' append of BRAW only returns the video item;
            # the audio item is on the timeline but not in the returned list.
            return [i for i in made if i.track_type == "video"]
        return made

    # -- test helpers ----------------------------------------------------------------------

    def place(
        self,
        track_type: str,
        index: int,
        mpi: FakeMediaPoolItem | None,
        start: int,
        length: int,
        src_start: int = 0,
    ) -> FakeTimelineItem:
        """Put an item on a track directly, as a human editing the timeline would."""
        while self.GetTrackCount(track_type) < index:
            self.tracks[track_type].append(FakeTrack(f"{track_type} {index}"))
        item = FakeTimelineItem(
            self, track_type, mpi, start, start + length, src_start, src_start + length
        )
        self.tracks[track_type][index - 1].items.append(item)
        return item

    def place_transition(
        self, track_type: str, index: int, cut: int, length: int
    ) -> FakeTimelineItem:
        """A transition centred on the cut at ``cut``, as live readback lists it."""
        start = cut - length // 2
        item = FakeTimelineItem(self, track_type, None, start, start + length, 0, 0)
        item.item_type = "transition"
        item.props = {}
        self.tracks[track_type][index - 1].items.append(item)
        return item

    def items(self, track_type: str, index: int = 1) -> list[FakeTimelineItem]:
        return [i for i in self.GetItemListInTrack(track_type, index) if i.item_type is None]


_PROJECT_SETTINGS: dict[str, Any] = {
    # Shapes and values as read from a live 21.1 project (Timeline.GetSettings() of a timeline
    # without custom settings), except colorSpaceInput, which the model assumes.
    "timelineFrameRate": 29.97,
    "timelineDropFrameTimecode": "0",
    "timelineResolutionWidth": "1920",
    "timelineResolutionHeight": "1080",
    "colorScienceMode": "davinciYRGBColorManagedv2",
    "isAutoColorManage": "0",
    "rcmPresetMode": "Custom",
    "colorSpaceTimeline": "DaVinci WG/Intermediate",
    "colorSpaceOutput": "Rec.709 (Scene)",
    "inputDRT": "None",
    "outputDRT": "Luminance Mapping",
    "useInverseDRT": "1",
    "useColorSpaceAwareGradingTools": "1",
    "colorSpaceInput": "Blackmagic Design Film Gen 5",
}


class FakeProject:
    def __init__(
        self, app: FakeResolve, name: str, timelines: list[str], folder: tuple[str, ...] = ()
    ) -> None:
        self._app = app
        self._name = name
        self._uid = f"proj-{next(app.ids)}"
        self.folder = tuple(folder)
        self.settings: dict[str, Any] = dict(_PROJECT_SETTINGS)
        self.timelines = [FakeTimeline(app, t, f"tl-{next(app.ids)}", self) for t in timelines]
        self.current_timeline: FakeTimeline | None = self.timelines[0] if self.timelines else None
        ids = app.ids
        root = FakeFolder(
            "Master",
            f"fold-{next(ids)}",
            [
                FakeFolder(
                    "Footage",
                    f"fold-{next(ids)}",
                    [
                        FakeFolder("Day 1", f"fold-{next(ids)}"),
                        FakeFolder("Day 2", f"fold-{next(ids)}"),
                    ],
                ),
                FakeFolder("Music", f"fold-{next(ids)}"),
            ],
        )
        self.media_pool = FakeMediaPool(app, self, root)
        self.deliver = DeliverState()
        self.rendering = False
        self.saves = 0

    def GetName(self) -> str:
        return self._name

    def GetUniqueId(self) -> str:
        return self._uid

    def GetMediaPool(self) -> FakeMediaPool:
        return self.media_pool

    def GetSettings(self) -> dict[str, Any]:
        return dict(self.settings)

    def SetSettings(self, settings: dict[str, Any]) -> bool:
        self._app.calls.append(("Project.SetSettings", dict(settings)))
        self.settings.update(settings)
        return True

    def GetTimelineCount(self) -> int:
        return len(self.timelines)

    def GetTimelineByIndex(self, idx: int) -> FakeTimeline | None:
        if 1 <= idx <= len(self.timelines):
            return self.timelines[idx - 1]
        return None

    def GetCurrentTimeline(self) -> FakeTimeline | None:
        return self.current_timeline

    def SetCurrentTimeline(self, timeline: FakeTimeline) -> bool:
        self._app.calls.append(("SetCurrentTimeline", timeline.GetName()))
        if not any(t is timeline for t in self.timelines):
            return False
        self.current_timeline = timeline
        return True

    def GetCurrentRenderFormatAndCodec(self) -> dict[str, str]:
        return {"format": self.deliver.format, "codec": self.deliver.codec}

    def SetCurrentRenderFormatAndCodec(self, format: str, codec: str) -> bool:
        self._app.calls.append(("SetCurrentRenderFormatAndCodec", format, codec))
        if not format or not codec:
            return False
        self.deliver.format, self.deliver.codec = format, codec
        return True

    def GetCurrentRenderMode(self) -> int:
        return self.deliver.mode

    def SetCurrentRenderMode(self, renderMode: int) -> bool:
        self._app.calls.append(("SetCurrentRenderMode", renderMode))
        if renderMode not in (0, 1):
            return False
        self.deliver.mode = renderMode
        return True

    def SetRenderSettings(self, settings: dict[str, Any]) -> bool:
        self._app.calls.append(("SetRenderSettings", dict(settings)))
        self.deliver.settings.update(settings)
        return True

    def GetRenderPresetList(self) -> list[str]:
        return list(self._app.presets)

    def SaveAsNewRenderPreset(self, presetName: str) -> bool:
        self._app.calls.append(("SaveAsNewRenderPreset", presetName))
        if not presetName or presetName in self._app.presets:
            return False
        self._app.presets[presetName] = self.deliver.copy()
        return True

    def LoadRenderPreset(self, presetName: str) -> bool:
        self._app.calls.append(("LoadRenderPreset", presetName))
        preset = self._app.presets.get(presetName)
        if preset is None:
            return False
        # trap: deliver-clobber: loading a preset replaces every Deliver setting.
        self.deliver = preset.copy()
        return True

    def DeleteRenderPreset(self, presetName: str) -> bool:
        self._app.calls.append(("DeleteRenderPreset", presetName))
        return self._app.presets.pop(presetName, None) is not None

    def IsRenderingInProgress(self) -> bool:
        return self.rendering

    def timeline(self, name: str) -> FakeTimeline:
        return next(t for t in self.timelines if t.GetName() == name)


class FakeProjectManager:
    """Project Manager with a folder tree. ``LoadProject`` only sees the current folder."""

    def __init__(self, app: FakeResolve) -> None:
        self._app = app

    def GetCurrentProject(self) -> FakeProject | None:
        return self._app.current

    def GetCurrentFolder(self) -> str:
        path = self._app.pm_path
        return path[-1] if path else ""  # the root reads as "", as observed live

    def GotoRootFolder(self) -> bool:
        self._app.calls.append(("GotoRootFolder",))
        self._app.pm_path = ()
        return True

    def GotoParentFolder(self) -> bool:
        self._app.calls.append(("GotoParentFolder",))
        if not self._app.pm_path:
            return False
        self._app.pm_path = self._app.pm_path[:-1]
        return True

    def OpenFolder(self, folderName: str) -> bool:
        self._app.calls.append(("OpenFolder", folderName))
        if folderName not in self._app.pm_children(self._app.pm_path):
            return False
        self._app.pm_path = (*self._app.pm_path, folderName)
        return True

    def GetFolderListInCurrentFolder(self) -> list[str]:
        return self._app.pm_children(self._app.pm_path)

    def GetProjectListInCurrentFolder(self) -> list[str]:
        return [p.GetName() for p in self._app.all_projects if p.folder == self._app.pm_path]

    def LoadProject(self, projectName: str) -> FakeProject | None:
        self._app.calls.append(("LoadProject", projectName))
        # trap: pm-folder-scoped-load: the name is looked up in the current folder only.
        project = next(
            (
                p
                for p in self._app.all_projects
                if p.GetName() == projectName and p.folder == self._app.pm_path
            ),
            None,
        )
        if project is None or projectName in self._app.missing:
            return None
        if projectName in self._app.load_silently_fails:
            return project  # claims success but leaves the old project open
        self._app.current = project
        return project

    def SaveProject(self) -> bool:
        self._app.calls.append(("SaveProject",))
        if self._app.current is None or self._app.save_fails:
            return False
        self._app.current.saves += 1
        return True


class FakeResolve:
    """The root ``resolve`` object. ``projects`` maps names to timeline names.

    Every project lives in Project Manager folder ``folder`` (default ``("CCN",)``) unless
    ``folders`` maps its name elsewhere. ``add_project`` adds more, including same-named
    projects in other folders. The Project Manager starts in the current project's folder.
    """

    def __init__(
        self,
        projects: dict[str, list[str]],
        current: str | None = None,
        *,
        folder: tuple[str, ...] = ("CCN",),
        folders: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.ids = itertools.count(1)
        self.calls: list[tuple[Any, ...]] = []
        self.all_projects: list[FakeProject] = []
        self.projects: dict[str, FakeProject] = {}
        self.pm_extra_folders: set[tuple[str, ...]] = set()
        for name, timelines in projects.items():
            self.add_project(name, timelines, (folders or {}).get(name, folder))
        first = current if current is not None else next(iter(projects), None)
        self.current: FakeProject | None = self.projects[first] if first is not None else None
        self.pm_path: tuple[str, ...] = self.current.folder if self.current else tuple(folder)
        self.page = "edit"
        self.props_stuck = False
        self.presets: dict[str, DeliverState] = {
            "YouTube 1080p": DeliverState(
                "mp4", "H265", 1, {"TargetDir": "C:/Exports", "CustomName": "yt"}
            )
        }
        self.missing: set[str] = set()
        self.load_silently_fails: set[str] = set()
        self.save_fails = False
        self.unknown_luts: set[str] = set()
        self.source_readback_skew = 0  # applied to items created from now on
        self._pm = FakeProjectManager(self)

    # -- Resolve API -----------------------------------------------------------------------

    def GetProjectManager(self) -> FakeProjectManager:
        return self._pm

    def GetCurrentPage(self) -> str:
        return self.page

    def OpenPage(self, pageName: str) -> bool:
        self.calls.append(("OpenPage", pageName))
        if pageName not in PAGES:
            return False
        self._change_page(pageName)
        return True

    def _change_page(self, page: str) -> None:
        # trap: setproperty-after-color: coming back to Edit from Color jams the property
        # setters; going to any other page (and back) clears it.
        if self.page == "color" and page == "edit":
            self.props_stuck = True
        elif page not in ("edit", "color"):
            self.props_stuck = False
        self.page = page

    # -- test helpers ----------------------------------------------------------------------

    @property
    def pm_folder(self) -> str:
        return self.pm_path[-1] if self.pm_path else ""

    def add_project(
        self, name: str, timelines: list[str], folder: tuple[str, ...] = ("CCN",)
    ) -> FakeProject:
        project = FakeProject(self, name, timelines, folder)
        self.all_projects.append(project)
        self.projects.setdefault(name, project)
        return project

    def add_pm_folder(self, path: tuple[str, ...]) -> None:
        self.pm_extra_folders.add(tuple(path))

    def pm_children(self, path: tuple[str, ...]) -> list[str]:
        paths = {p.folder for p in self.all_projects} | self.pm_extra_folders
        children = {
            full[len(path)] for full in paths if len(full) > len(path) and full[: len(path)] == path
        }
        return sorted(children)

    def project(self, name: str) -> FakeProject:
        return self.projects[name]

    def call_names(self) -> list[str]:
        return [call[0] for call in self.calls]

    def human_open_project(self, name: str) -> None:
        self.current = self.projects[name]
        self.pm_path = self.current.folder

    def human_set_page(self, page: str) -> None:
        self._change_page(page)

    def human_move_playhead(self, timecode: str) -> None:
        assert self.current is not None and self.current.current_timeline is not None
        self.current.current_timeline.timecode = timecode

    def human_browse_project_manager(self, path: tuple[str, ...]) -> None:
        self.pm_path = tuple(path)
