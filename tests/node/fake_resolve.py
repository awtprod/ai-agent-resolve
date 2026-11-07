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
pin the exact numbers down, the comment says what the model chose; behaviours the trap
registry lists as presumed are marked "(presumed)".

``FakeResolve.faults`` switches on lies Resolve's API can tell (a setter that returns True
without applying, a returned object that is not really there). They are not traps seen live;
they exist so the ops' read-back checks are proven to matter. See ``FAULTS``.
"""

from __future__ import annotations

import itertools
import math
import re
import string
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

PAGES = ("media", "photo", "cut", "edit", "fusion", "color", "fairlight", "deliver")

FAULTS = frozenset(
    {
        "SetProperties:no-op",  # SetProperties/SetProperty return True, apply nothing
        "SetFades:no-op",  # SetFades returns True, applies nothing
        "CopyGrades:no-op",  # CopyGrades returns True, copies nothing
        "SetLUT:no-op",  # Graph.SetLUT returns True, sets nothing
        "SetClipProperty:no-op",  # SetClipProperty returns True, sets nothing
        "DeleteClips:no-op",  # Timeline.DeleteClips returns True, deletes nothing
        "DuplicateTimeline:detached",  # returns a timeline that is not in the project
        "LoadRenderPreset:drops-codec",  # a preset round trip resets the codec to H264
        "OpenPage:keeps-jam",  # a page bounce does not clear the after-Color setter jam
        "CopyGrades:then-jam",  # CopyGrades works, then the setters jam (until a bounce)
        "CopyGrades:drops-label",  # the copy loses the first node's label
        "CopyGrades:drops-lut",  # the copy loses the LUT of the first node that has one
        "DuplicateTimeline:returns-source",  # the copy is made, but the source is returned
        "DeleteTimelines:fail",  # MediaPool.DeleteTimelines returns False, deletes nothing
        # S13h1 build ops
        "ImportMedia:partial",  # the last importable path is silently left out
        "ImportMedia:duplicate",  # the first importable path comes back as two items
        "AddSubFolder:fail",  # MediaPool.AddSubFolder returns None, makes nothing
        "SetCurrentFolder:ignored",  # SetCurrentFolder returns True, the folder stays
        "AddTrack:no-op",  # Timeline.AddTrack returns True, adds nothing
        "SetTrackName:no-op",  # SetTrackName returns True, renames nothing
        "SetTrackEnable:no-op",  # SetTrackEnable returns True, changes nothing
        "SetTrackLock:no-op",  # SetTrackLock returns True, changes nothing
        "AddTransition:none",  # TimelineItem.AddTransition returns None, adds nothing
        "AddTransition:short",  # the transition is made one frame shorter than asked
        "AddTransition:misplaced",  # the transition is made away from the edge asked for
        "SetCDL:false",  # TimelineItem.SetCDL returns False, sets nothing
        "SetInput:no-op",  # a Fusion tool's SetInput applies nothing
    }
)
# Audio track formats AddTrack accepts as a subtype (facts file, [I]: not stated by the stub).
AUDIO_FORMATS = frozenset(
    {"mono", "stereo", "5.1", "5.1film", "7.1", "7.1film"} | {f"adaptive{n}" for n in range(1, 37)}
)
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
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
    "RotationAngle": 0.0,
    "FlipX": False,
    "FlipY": False,
    # Stored only. Whether a ganged zoom mirrors ZoomX into ZoomY is not modelled: the op
    # writes and reads back both.
    "ZoomGang": True,
}
_AUDIO_PROPS: dict[str, Any] = {
    "AudioVolumeEnabled": True,
    "AudioVolume": 0.0,
    "AudioPanEnabled": True,
    "AudioPan": 0.0,
}
_PROP_RANGES = {
    "AudioVolume": (-100.0, 30.0),
    "AudioPan": (-100.0, 100.0),
    "Opacity": (0.0, 100.0),
    "ZoomX": (0.0, 100.0),
    "ZoomY": (0.0, 100.0),
    "RotationAngle": (-360.0, 360.0),
}

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


@dataclass
class FakeRenderJob:
    """A queued render job, as ``GetRenderJobList`` / ``GetRenderJobStatus`` would report it.

    ``outcome`` drives the progression once rendering: ``complete`` finishes after
    ``complete_after`` polls, ``fail`` ends with ``fail_error``, ``hang`` never finishes (for
    the watchdog). Each ``GetRenderJobStatus`` call while rendering advances the job.
    """

    job_id: str
    info: dict[str, Any]
    status: str = "Ready"
    completion: int = 0
    polls: int = 0
    time_ms: int = 0
    error: str = ""
    outcome: str = "complete"
    complete_after: int = 2
    result_time_ms: int = 5000
    fail_error: str = "render failed: frames offline"


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


def _cloud_sync_prop(value: int | float | str | None) -> str:
    """The 'Cloud Sync' clip property as Resolve returns it: the code as an integer-form string.

    Observed live 2026-10-17: every clip on a cloud library read "-1" (CLOUD_SYNC_DEFAULT); the
    string form of the other codes was not seen, so it is modelled as the integer form here.
    """
    if value is None:
        return "-1"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return str(int(value))
    return str(value)


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
        cloud_sync: int | str | None = None,
        file_path: str | None = None,
        title_text: str | None = None,
        title_tool: str = "TextPlus",
    ) -> None:
        if kind not in _KIND_TYPE:
            raise ValueError(f"unknown clip kind {kind!r}")
        # A Fusion title template: timeline items placed from it carry a Fusion comp whose
        # ``title_tool`` tool (a Text+ by default) holds ``title_text`` as its StyledText.
        self.title_text = title_text
        self.title_tool = title_tool
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
            # Blackmagic Cloud per-clip sync status (CloudSyncStatus), read back as a string
            # ("-1" = DEFAULT, observed live). The op labels it from the float CLOUD_SYNC_*
            # constants by matching numerically, so a string here against a float constant is
            # the real shape, not a mismatch.
            "Cloud Sync": _cloud_sync_prop(cloud_sync),
            "Audio Ch": "2" if kind in ("av", "audio") else "0",
        }  # fmt: skip
        if file_path is not None:
            # Imported media: the clip is named after its file, and "File Path" is the path as
            # the edit PC's filesystem spells it (model choice; Resolve's own spelling is [I]).
            base = re.split(r"[\\/]", file_path)[-1]
            self._name = base
            self.props.update({"Clip Name": base, "File Name": base, "File Path": file_path})

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
        if "SetClipProperty:no-op" in self._app.faults:
            return True
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
        stale: bool = False,
    ) -> None:
        self._name = name
        self._uid = uid
        self.subfolders = subfolders or []
        self.clips = clips or []
        self.stale = stale

    def GetName(self) -> str:
        return self._name

    def GetUniqueId(self) -> str:
        return self._uid

    def GetSubFolderList(self) -> list[FakeFolder]:
        return list(self.subfolders)

    def GetIsFolderStale(self) -> bool:
        # collaboration freshness (read-only); the write side is MediaPool.RefreshFolders.
        return self.stale

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
        if "SetCurrentFolder:ignored" in self._app.faults:
            return True
        self.current = folder
        return True

    def AddSubFolder(self, folder: FakeFolder, name: str) -> FakeFolder | None:
        self._app.calls.append(("AddSubFolder", folder.GetName(), name))
        if "AddSubFolder:fail" in self._app.faults:
            return None
        if not isinstance(name, str) or not name or not any(f is folder for f in self._root.walk()):
            return None
        made = FakeFolder(name, f"fold-{next(self._app.ids)}")
        folder.subfolders.append(made)
        return made

    def ImportMedia(self, clipInfos: list[Any]) -> list[FakeMediaPoolItem]:
        """Import edit-PC files (``FakeResolve.edit_files``) into the current folder.

        Model choices ([I] in the facts file): a path the edit PC does not have is left out of
        the returned list silently; a file already in the folder is imported again as a second
        item; the items come back in request order, minus the skipped ones. Paths are looked
        up the Windows way (either separator, ASCII case folded), and an item's "File Path" is
        the path as the filesystem spells it, which may differ from the request.
        """
        infos = [dict(i) if isinstance(i, dict) else {"FilePath": i} for i in clipInfos]
        self._app.calls.append(("ImportMedia", infos))
        found: list[tuple[str, dict[str, Any]]] = []
        for info in infos:
            hit = self._app.edit_file(str(info.get("FilePath") or ""))
            if hit is not None:
                found.append(hit)
        if "ImportMedia:partial" in self._app.faults and found:
            found = found[:-1]
        made: list[FakeMediaPoolItem] = []
        for index, (disk_path, spec) in enumerate(found):
            copies = 2 if index == 0 and "ImportMedia:duplicate" in self._app.faults else 1
            for _ in range(copies):
                clip = FakeMediaPoolItem(self._app, disk_path, file_path=disk_path, **spec)
                self.current.clips.append(clip)
                made.append(clip)
        return made

    def DeleteTimelines(self, timelines: list[Any]) -> bool:
        self._app.calls.append(("DeleteTimelines", [t.GetUniqueId() for t in timelines]))
        if "DeleteTimelines:fail" in self._app.faults:
            return False
        doomed = {id(t) for t in timelines}
        if not all(any(t is mine for mine in self._project.timelines) for t in timelines):
            return False
        self._project.timelines = [t for t in self._project.timelines if id(t) not in doomed]
        if (
            self._project.current_timeline is not None
            and id(self._project.current_timeline) in doomed
        ):
            self._project.current_timeline = (
                self._project.timelines[0] if self._project.timelines else None
            )
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
    cdl: dict[str, Any] | None = None
    """What ``TimelineItem.SetCDL`` last set on this node. Test-side only: 21.1 has no getter."""


class FakeGraph:
    def __init__(self, app: FakeResolve, nodes: list[FakeNode] | None = None) -> None:
        self._app = app
        self.nodes = nodes if nodes is not None else [FakeNode()]

    def copy(self) -> FakeGraph:
        return FakeGraph(
            self._app,
            [
                FakeNode(n.label, n.lut, n.tools, dict(n.cdl) if n.cdl is not None else None)
                for n in self.nodes
            ],
        )

    def GetNumNodes(self) -> int:
        return len(self.nodes)

    def GetLUT(self, nodeIndex: int) -> str:
        if not 1 <= nodeIndex <= len(self.nodes):
            return ""
        return self.nodes[nodeIndex - 1].lut

    def SetLUT(self, nodeIndex: int, lutPath: str) -> bool:
        self._app.calls.append(("SetLUT", nodeIndex, lutPath))
        if self._app.props_stuck:
            return False  # trap: setproperty-after-color (presumed for Graph.SetLUT)
        if not 1 <= nodeIndex <= len(self.nodes) or not lutPath:
            return False
        if lutPath in self._app.unknown_luts:
            return False  # Resolve only applies LUTs it has discovered
        if "SetLUT:no-op" in self._app.faults:
            return True
        # GetLUT returns a relative path [S]: a LUT given by its absolute path under the LUT
        # folder reads back relative to it (model choice: with Windows separators).
        self.nodes[nodeIndex - 1].lut = self._app.lut_relative(lutPath)
        return True

    def GetNodeLabel(self, nodeIndex: int) -> str:
        if not 1 <= nodeIndex <= len(self.nodes):
            return ""
        return self.nodes[nodeIndex - 1].label

    def GetToolsInNode(self, nodeIndex: int) -> list[str]:
        if not 1 <= nodeIndex <= len(self.nodes):
            return []
        return list(self.nodes[nodeIndex - 1].tools)


# -- Fusion ---------------------------------------------------------------------------------


class FakeFusionTool:
    """A Fusion tool (``Operator``): ``GetInput`` / ``SetInput`` on named inputs."""

    def __init__(self, app: FakeResolve, owner: str, tool_id: str, inputs: dict[str, Any]) -> None:
        self._app = app
        self._owner = owner
        self.tool_id = tool_id
        self.inputs = dict(inputs)

    def GetInput(self, id: str, time: float | None = None) -> Any:
        return self.inputs.get(id)

    def SetInput(self, id: str, value: Any, time: float | None = None) -> None:
        # Fusion's SetInput returns nothing (model choice), so only a read-back can tell.
        self._app.calls.append(("SetInput", self._owner, id, value))
        if "SetInput:no-op" in self._app.faults or id not in self.inputs:
            return
        self.inputs[id] = value


class FakeFusionComp:
    """A Fusion composition on a timeline item."""

    def __init__(self, app: FakeResolve, owner: str, tools: list[FakeFusionTool]) -> None:
        self._app = app
        self._owner = owner
        self.tools = tools

    def FindToolByID(self, id: str) -> FakeFusionTool | None:
        return next((t for t in self.tools if t.tool_id == id), None)

    def clone(self, owner: str) -> FakeFusionComp:
        return FakeFusionComp(
            self._app,
            owner,
            [FakeFusionTool(self._app, owner, t.tool_id, t.inputs) for t in self.tools],
        )


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
        self.transition_name: str | None = None  # the type AddTransition was given
        self.comps: list[FakeFusionComp] = []
        if mpi is not None and mpi.title_text is not None and track_type == "video":
            tool = FakeFusionTool(app, self._uid, mpi.title_tool, {"StyledText": mpi.title_text})
            self.comps = [FakeFusionComp(app, self._uid, [tool])]

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
        twin.item_type = self.item_type
        twin.comps = [c.clone(twin._uid) for c in self.comps]
        return twin

    def GetType(self) -> str:
        # Live 21.1 lists transitions in GetItemListInTrack as items of type "transition",
        # without media, straddling the cut between two clips.
        return self.item_type or self.track_type

    def GetName(self) -> str:
        if self.transition_name is not None:
            return self.transition_name  # model choice: a transition is named by its type
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
        if self._app.props_stuck:
            return False  # trap: setproperty-after-color (presumed for SetFades)
        if "SetFades:no-op" in self._app.faults:
            return True
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
        if "SetProperties:no-op" in self._app.faults:
            return True
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

    def SetCDL(self, CDL: dict[str, Any]) -> bool:
        self._app.calls.append(("SetCDL", self._uid, dict(CDL)))
        if self._app.props_stuck:
            return False  # trap: setproperty-after-color (presumed for SetCDL)
        if "SetCDL:false" in self._app.faults or self.track_type != "video":
            return False
        if set(CDL) != {"NodeIndex", "Slope", "Offset", "Power", "Saturation"}:
            return False  # model choice: every part is required
        node = CDL["NodeIndex"]
        if not isinstance(node, int) or not 1 <= node <= len(self.graph.nodes):
            return False
        parsed: dict[str, Any] = {}
        for key in ("Slope", "Offset", "Power"):
            parts = str(CDL[key]).split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                return False
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                return False
            parsed[key] = tuple(values)
        parsed["Saturation"] = float(CDL["Saturation"])
        self.graph.nodes[node - 1].cdl = parsed
        return True

    def cdl_on(self, node: int) -> dict[str, Any] | None:
        """Test-side accessor: the CDL last set on ``node`` (Resolve 21.1 has no getter)."""
        return self.graph.nodes[node - 1].cdl

    def GetLeftOffset(self, subframePrecision: bool = False) -> int:
        """The handle before the in point (model: source frames before ``src_start``)."""
        if self.item_type == "transition" or self.mpi is None:
            return 0
        return self.src_start

    def GetRightOffset(self, subframePrecision: bool = False) -> int:
        """The handle after the out point (model: source frames after ``src_end``)."""
        if self.item_type == "transition" or self.mpi is None:
            return 0
        return max(self.mpi.frames - self.src_end, 0)

    def AddTransition(self, transitionOptions: dict[str, Any]) -> FakeTimelineItem | None:
        options = dict(transitionOptions)
        self._app.calls.append(("AddTransition", self._uid, options))
        return self.timeline.add_transition(self, options)

    def GetFusionCompCount(self) -> int:
        return len(self.comps)

    def GetFusionCompByIndex(self, compIndex: int) -> FakeFusionComp | None:
        return self.comps[compIndex - 1] if 1 <= compIndex <= len(self.comps) else None

    def CopyGrades(self, tgtTimelineItems: list[FakeTimelineItem]) -> bool:
        self._app.calls.append(("CopyGrades", self._uid, [t._uid for t in tgtTimelineItems]))
        if self._app.props_stuck:
            return False  # trap: setproperty-after-color (presumed for CopyGrades)
        if self.track_type != "video" or any(t.track_type != "video" for t in tgtTimelineItems):
            return False
        if "CopyGrades:no-op" in self._app.faults:
            return True
        # trap: append-ungraded: CopyGrades works across timelines, and it replaces the
        # target's whole node graph, per-clip LUTs included.
        for target in tgtTimelineItems:
            target.graph = self.graph.copy()
            if "CopyGrades:drops-label" in self._app.faults and target.graph.nodes:
                target.graph.nodes[0].label = ""
            if "CopyGrades:drops-lut" in self._app.faults:
                lit = [n for n in target.graph.nodes if n.lut]
                if lit:
                    lit[0].lut = ""
        if "CopyGrades:then-jam" in self._app.faults:
            self._app.props_stuck = True
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
        if subTrackType is not None and (trackType != "audio" or subTrackType not in AUDIO_FORMATS):
            return False  # model choice: only an audio track takes a subtype, a known format
        if "AddTrack:no-op" in self._app.faults:
            return True
        n = len(self.tracks[trackType]) + 1
        label = {"video": "Video", "audio": "Audio", "subtitle": "Subtitle"}[trackType]
        # The subtype is stored as given; an audio track added without one gets "" (model
        # choice: Resolve's default channel format is not known).
        self.tracks[trackType].append(FakeTrack(f"{label} {n}", subTrackType or ""))
        return True

    def SetTrackName(self, trackType: str, trackIndex: int, name: str) -> bool:
        self._app.calls.append(("SetTrackName", trackType, trackIndex, name))
        track = self._track(trackType, trackIndex)
        if track is None or not isinstance(name, str) or not name:
            return False
        if track.locked:
            return False  # model choice: a locked track refuses edits
        if "SetTrackName:no-op" in self._app.faults:
            return True
        track.name = name
        return True

    def SetTrackEnable(self, trackType: str, trackIndex: int, enabled: bool) -> bool:
        self._app.calls.append(("SetTrackEnable", trackType, trackIndex, enabled))
        track = self._track(trackType, trackIndex)
        if track is None or not isinstance(enabled, bool):
            return False
        if track.locked:
            return False  # model choice: a locked track refuses edits
        if "SetTrackEnable:no-op" in self._app.faults:
            return True
        track.enabled = enabled
        return True

    def SetTrackLock(self, trackType: str, trackIndex: int, locked: bool) -> bool:
        self._app.calls.append(("SetTrackLock", trackType, trackIndex, locked))
        track = self._track(trackType, trackIndex)
        if track is None or not isinstance(locked, bool):
            return False
        if "SetTrackLock:no-op" in self._app.faults:
            return True
        track.locked = locked
        return True

    def GetSetting(self, settingName: str | None = None) -> Any:
        settings = self.GetSettings()
        if settingName is None:
            return settings
        return settings.get(settingName, "")  # model choice: "" for an unknown key

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

    def GetMarkerByCustomData(self, customData: str) -> dict[str, Any]:
        for frame in sorted(self.markers):
            info = self.markers[frame]
            if info.get("customData") == customData:
                return dict(info)
        return {}

    def GetMarkerCustomData(self, frameId: int) -> str:
        info = self.markers.get(frameId)
        return str(info.get("customData") or "") if info is not None else ""

    def UpdateMarkerCustomData(self, frameId: int, customData: str) -> bool:
        self._app.calls.append(("UpdateMarkerCustomData", frameId, customData))
        info = self.markers.get(frameId)
        if info is None:
            return False
        info["customData"] = customData
        return True

    def DeleteMarkerAtFrame(self, frameNum: int) -> bool:
        self._app.calls.append(("DeleteMarkerAtFrame", frameNum))
        return self.markers.pop(frameNum, None) is not None

    def DeleteMarkerByCustomData(self, customData: str) -> bool:
        self._app.calls.append(("DeleteMarkerByCustomData", customData))
        if not customData:
            return False  # live refuses an empty customData
        for frame in sorted(self.markers):
            if self.markers[frame].get("customData") == customData:
                del self.markers[frame]
                return True
        return False

    def DeleteMarkersByColor(self, color: str) -> bool:
        self._app.calls.append(("DeleteMarkersByColor", color))
        victims = [
            frame
            for frame, info in self.markers.items()
            if color == "All" or info.get("color") == color
        ]
        for frame in victims:
            del self.markers[frame]
        return bool(victims)

    def DeleteClips(
        self, timelineItems: list[FakeTimelineItem], rippleDelete: bool = False
    ) -> bool:
        self._app.calls.append(("DeleteClips", [i.GetUniqueId() for i in timelineItems]))
        if "DeleteClips:no-op" in self._app.faults:
            return True
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
        if "DuplicateTimeline:detached" in self._app.faults:
            return twin  # a timeline object the project does not hold
        project.timelines.append(twin)
        if "DuplicateTimeline:returns-source" in self._app.faults:
            return self
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

    # -- the transition model --------------------------------------------------------------

    def track_of(self, item: FakeTimelineItem) -> FakeTrack | None:
        index = self.track_index_of(item)
        return self.tracks[item.track_type][index - 1] if index else None

    @staticmethod
    def _handles_fit(
        before: FakeTimelineItem | None, after: FakeTimelineItem | None, alignment: str, n: int
    ) -> bool:
        """Whether an ``n``-frame transition at a cut fits the handles of the clips around it.

        Model ([I] in the facts file): ``left`` ends on the cut, ``right`` starts on it,
        ``center`` puts ``n // 2`` frames before it. The part before the cut shows the incoming
        clip early (its left handle), the part after it holds the outgoing clip on (its right
        handle). A side with no clip fades from or to black and needs nothing.
        """
        before_cut = {"left": n, "center": n // 2, "right": 0}[alignment]
        after_cut = n - before_cut
        fits_in = after is None or after.GetLeftOffset() >= before_cut
        fits_out = before is None or before.GetRightOffset() >= after_cut
        return fits_in and fits_out

    def add_transition(
        self, item: FakeTimelineItem, options: dict[str, Any]
    ) -> FakeTimelineItem | None:
        """``TimelineItem.AddTransition`` at one edge of ``item``; ``None`` when it fails."""
        app = self._app
        if "AddTransition:none" in app.faults:
            return None
        kind, category = options.get("type"), options.get("category")
        position, alignment = options.get("position"), options.get("alignment")
        n = options.get("duration")
        if not isinstance(kind, str) or not kind or item.item_type is not None:
            return None
        if category not in ("simple", "fusion", "ofx", "audio"):
            return None
        if (category == "audio") != (item.track_type == "audio"):
            return None  # model choice: audio transitions on audio items, the rest on video
        if position not in ("start", "end") or alignment not in ("left", "center", "right"):
            return None
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            return None  # model choice: the op always passes a duration
        track = self.track_of(item)
        if track is None or track.locked:
            return None
        edge = item.start if position == "start" else item.end
        if any(t.item_type == "transition" and t.start <= edge <= t.end for t in track.items):
            return None  # model choice: one transition per edge; it is never replaced
        clips = [c for c in track.items if c.item_type is None and c is not item]
        if position == "start":
            before, after = next((c for c in clips if c.end == edge), None), item
        else:
            before, after = item, next((c for c in clips if c.start == edge), None)
        if not self._handles_fit(before, after, alignment, n):
            # Too-short handles are not documented ([I]): the knob picks the behaviour.
            if app.transition_short_handles == "refuse":
                return None
            while n > 0 and not self._handles_fit(before, after, alignment, n):
                n -= 1
            if n == 0:
                return None
        if "AddTransition:short" in app.faults:
            n = n - 1 if n > 1 else n + 1
        start = {"left": edge - n, "center": edge - n // 2, "right": edge}[alignment]
        if "AddTransition:misplaced" in app.faults:
            start = edge + n + 1
        made = FakeTimelineItem(self, item.track_type, None, start, start + n, 0, 0)
        made.item_type = "transition"
        made.props = {}
        made.transition_name = kind
        track.items.append(made)
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
        # ProjectAttributes as ProjectManager.GetProjectAttributesInCurrentFolder returns them.
        self.attributes: dict[str, str] = {
            "liveCollaborationMode": "single_user",
            "lastModifiedDate": "2026-10-17T00:00:00+00:00",
            "creationDate": "2026-10-01T00:00:00+00:00",
            "notes": "",
        }
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
        # Render queue and the knobs a test uses to drive a render's outcome.
        self.render_jobs: list[FakeRenderJob] = []
        self.render_add_fails = False
        self.render_outcome = "complete"  # "complete" | "fail" | "hang"
        self.render_complete_after = 2
        self.render_time_ms = 5000
        self.render_error = "render failed: frames offline"
        # Live 21.1 appends the container extension to the output file name; set this to model it.
        self.render_output_extension = ""

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
        if "LoadRenderPreset:drops-codec" in self._app.faults:
            # fault: a preset round trip that loses the codec (presumed not to happen live;
            # the broker sets format and codec again after loading the preset anyway).
            self.deliver.codec = "H264"
        return True

    def DeleteRenderPreset(self, presetName: str) -> bool:
        self._app.calls.append(("DeleteRenderPreset", presetName))
        return self._app.presets.pop(presetName, None) is not None

    def IsRenderingInProgress(self) -> bool:
        return self.rendering

    # -- render queue ----------------------------------------------------------------------

    def AddRenderJob(self) -> str:
        self._app.calls.append(("AddRenderJob",))
        if self.render_add_fails:
            return ""  # Resolve returns an empty id when a job cannot be added
        settings = self.deliver.settings
        timeline = self.current_timeline
        fps = timeline.fps() if timeline is not None else None
        if settings.get("SelectAllFrames", True):
            span = (
                (timeline.GetEndFrame() - timeline.GetStartFrame()) if timeline is not None else 0
            )
            mark_in, mark_out = 0, max(span - 1, 0)
        else:
            mark_in = int(settings.get("MarkIn", 0))
            mark_out = int(settings.get("MarkOut", mark_in))
        job_id = f"rj-{next(self._app.ids)}"
        output_filename = str(settings.get("CustomName", "")) + self.render_output_extension
        info: dict[str, Any] = {
            "JobId": job_id,
            "RenderJobName": str(settings.get("CustomName", "")) or job_id,
            "TimelineName": timeline.GetName() if timeline is not None else "",
            "TargetDir": str(settings.get("TargetDir", "")),
            "OutputFilename": output_filename,
            "FormatWidth": int(settings.get("FormatWidth", 1920)),
            "FormatHeight": int(settings.get("FormatHeight", 1080)),
            "FrameRate": f"{fps:g}" if fps is not None else "",
            "MarkIn": mark_in,
            "MarkOut": mark_out,
            "VideoFormat": self.deliver.format,
            "VideoCodec": self.deliver.codec,
            "RenderMode": "Single clip" if self.deliver.mode == 1 else "Individual clips",
            "PresetName": "",
        }
        self.render_jobs.append(
            FakeRenderJob(
                job_id,
                info,
                outcome=self.render_outcome,
                complete_after=self.render_complete_after,
                result_time_ms=self.render_time_ms,
                fail_error=self.render_error,
            )
        )
        return job_id

    def GetRenderJobList(self) -> list[dict[str, Any]]:
        return [dict(job.info) for job in self.render_jobs]

    def StartRendering(self, jobIds: list[str], isInteractiveMode: bool = False) -> bool:
        self._app.calls.append(("StartRendering", list(jobIds), isInteractiveMode))
        ids = list(jobIds)
        targets = [job for job in self.render_jobs if job.job_id in ids]
        if len(targets) != len(set(ids)) or not targets:
            return False
        if any(job.status not in ("Ready", "Ready for background render") for job in targets):
            return False
        # Resolve renders one job at a time: starting a render while another is in progress
        # fails. The node must serialise renders per project.
        if any(job.status == "Rendering" and job.job_id not in ids for job in self.render_jobs):
            return False
        for job in targets:
            job.status = "Rendering"
            job.completion = 0
            job.polls = 0
        self.rendering = True
        return True

    def _settle_rendering(self) -> None:
        self.rendering = any(job.status == "Rendering" for job in self.render_jobs)

    def GetRenderJobStatus(self, jobId: str) -> dict[str, Any]:
        job = next((j for j in self.render_jobs if j.job_id == jobId), None)
        if job is None:
            return {}  # unknown job
        if job.status == "Rendering":
            job.polls += 1
            if job.outcome == "hang":
                job.completion = min(90, job.completion + 5)
            elif job.polls >= job.complete_after:
                if job.outcome == "fail":
                    job.status, job.error = "Failed", job.fail_error
                else:
                    job.status, job.completion, job.time_ms = "Complete", 100, job.result_time_ms
                self._settle_rendering()
            else:
                job.completion = min(95, int(100 * job.polls / max(job.complete_after, 1)))
        out: dict[str, Any] = {"JobStatus": job.status, "CompletionPercentage": job.completion}
        if job.status == "Complete":
            out["TimeTakenToRenderInMs"] = job.time_ms
        elif job.status == "Rendering":
            out["EstimatedTimeRemainingInMs"] = 1000
        elif job.status == "Failed" and job.error:
            out["Error"] = job.error
        return out

    def DeleteRenderJob(self, jobId: str) -> bool:
        self._app.calls.append(("DeleteRenderJob", jobId))
        before = len(self.render_jobs)
        self.render_jobs = [job for job in self.render_jobs if job.job_id != jobId]
        self._settle_rendering()
        return len(self.render_jobs) < before

    def DeleteAllRenderJobs(self) -> bool:
        self._app.calls.append(("DeleteAllRenderJobs",))
        self.render_jobs = []
        self.rendering = False
        return True

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

    def GetProjectAttributesInCurrentFolder(self) -> dict[str, dict[str, str]]:
        return {
            p.GetName(): dict(p.attributes)
            for p in self._app.all_projects
            if p.folder == self._app.pm_path
        }

    def GetCurrentDatabase(self) -> dict[str, str]:
        return dict(self._app.database)

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

    # resolve.CLOUD_SYNC_* status codes (CloudSyncStatus), observed live 2026-10-17 as floats.
    # The clip property reads them back as strings (e.g. "-1"), so the op matches numerically.
    # The sync-MODE codes (NONE 0.0, PROXY_ONLY 1.0, PROXY_AND_ORIG 2.0) collide with these, so
    # they are kept out of the status map; they are not modelled here (no op reads the mode).
    CLOUD_SYNC_DEFAULT = -1.0
    CLOUD_SYNC_DOWNLOAD_IN_QUEUE = 0.0
    CLOUD_SYNC_DOWNLOAD_IN_PROGRESS = 1.0
    CLOUD_SYNC_DOWNLOAD_SUCCESS = 2.0
    CLOUD_SYNC_DOWNLOAD_FAIL = 3.0
    CLOUD_SYNC_DOWNLOAD_NOT_FOUND = 4.0
    CLOUD_SYNC_UPLOAD_IN_QUEUE = 5.0
    CLOUD_SYNC_UPLOAD_IN_PROGRESS = 6.0
    CLOUD_SYNC_UPLOAD_SUCCESS = 7.0
    CLOUD_SYNC_UPLOAD_FAIL = 8.0
    CLOUD_SYNC_UPLOAD_NOT_FOUND = 9.0
    CLOUD_SYNC_SUCCESS = 10.0

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
        # ProjectManager.GetCurrentDatabase. Observed live 2026-10-17: a Blackmagic Cloud library
        # reports DbType "PostgreSQL" with an IpAddress (GetDatabaseList does not list it,
        # DESIGN F2). The name and IP here are placeholders ("198.51.100.10" is an RFC 5737
        # documentation address): no real host goes in the repo, and the op must drop IpAddress.
        # Tests set this to the local database to exercise the local_database note.
        self.database: dict[str, str] = {
            "DbType": "PostgreSQL",
            "DbName": "acme cloud library",
            "IpAddress": "198.51.100.10",
        }
        self.missing: set[str] = set()
        self.load_silently_fails: set[str] = set()
        self.save_fails = False
        self.unknown_luts: set[str] = set()
        # The LUT folder: SetLUT with an absolute path under it reads back relative to it.
        self.lut_root = "X:/acme/Support/LUT"
        # The edit PC's files, for MediaPool.ImportMedia: path -> FakeMediaPoolItem kwargs.
        self.edit_files: dict[str, dict[str, Any]] = {}
        # What AddTransition does when the handles are too short ([I]): make the longest
        # transition that fits ("shorten") or return None ("refuse").
        self.transition_short_handles = "shorten"
        self.faults: set[str] = set()  # names from FAULTS
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
        elif page not in ("edit", "color") and "OpenPage:keeps-jam" not in self.faults:
            self.props_stuck = False
        self.page = page

    # -- edit-PC filesystem and LUT folder --------------------------------------------------

    @staticmethod
    def _win_key(path: str) -> str:
        return path.replace("\\", "/").translate(_ASCII_LOWER)

    def edit_file(self, path: str) -> tuple[str, dict[str, Any]] | None:
        """The edit PC's file at ``path`` as ``(path as the filesystem spells it, spec)``."""
        key = self._win_key(path)
        for disk_path, spec in self.edit_files.items():
            if self._win_key(disk_path) == key:
                return disk_path, spec
        return None

    def lut_relative(self, lut_path: str) -> str:
        root = self._win_key(self.lut_root).rstrip("/") + "/"
        if self._win_key(lut_path).startswith(root):
            return lut_path.replace("\\", "/")[len(root) :].replace("/", "\\")
        return lut_path

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
