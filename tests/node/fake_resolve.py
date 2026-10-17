"""An in-memory stand-in for the DaVinci Resolve scripting objects.

Method names and return conventions follow the 21.1 API stubs (``Resolve``, ``ProjectManager``,
``Project``, ``Timeline``, ``MediaPool``, ``Folder``), so ``ScriptApiGateway`` itself runs
against it. Mutating calls are recorded in ``FakeResolve.calls``. Render presets are shared
by all projects, like Resolve's user presets. ``human_*`` helpers change state the way a
person at the GUI would, without going through the gateway.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

PAGES = ("media", "photo", "cut", "edit", "fusion", "color", "fairlight", "deliver")
_TIMECODE = re.compile(r"^\d{2}:\d{2}:\d{2}[:;]\d{2}$")


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


class FakeFolder:
    def __init__(self, name: str, uid: str, subfolders: list[FakeFolder] | None = None) -> None:
        self._name = name
        self._uid = uid
        self.subfolders = subfolders or []

    def GetName(self) -> str:
        return self._name

    def GetUniqueId(self) -> str:
        return self._uid

    def GetSubFolderList(self) -> list[FakeFolder]:
        return list(self.subfolders)

    def walk(self) -> list[FakeFolder]:
        out = [self]
        for sub in self.subfolders:
            out.extend(sub.walk())
        return out


class FakeMediaPool:
    def __init__(self, app: FakeResolve, root: FakeFolder) -> None:
        self._app = app
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

    def folder(self, *path: str) -> FakeFolder:
        node = self._root
        for name in path:
            node = next(f for f in node.subfolders if f.GetName() == name)
        return node


class FakeTimeline:
    def __init__(self, app: FakeResolve, name: str, uid: str) -> None:
        self._app = app
        self._name = name
        self._uid = uid
        self.timecode = "01:00:00:00"

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


class FakeProject:
    def __init__(self, app: FakeResolve, name: str, timelines: list[str]) -> None:
        self._app = app
        self._name = name
        self._uid = f"proj-{next(app.ids)}"
        self.timelines = [FakeTimeline(app, t, f"tl-{next(app.ids)}") for t in timelines]
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
        self.media_pool = FakeMediaPool(app, root)
        self.deliver = DeliverState()
        self.rendering = False
        self.saves = 0

    def GetName(self) -> str:
        return self._name

    def GetUniqueId(self) -> str:
        return self._uid

    def GetMediaPool(self) -> FakeMediaPool:
        return self.media_pool

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
        self.deliver = preset.copy()  # clobbers every Deliver setting, like Resolve does
        return True

    def DeleteRenderPreset(self, presetName: str) -> bool:
        self._app.calls.append(("DeleteRenderPreset", presetName))
        return self._app.presets.pop(presetName, None) is not None

    def IsRenderingInProgress(self) -> bool:
        return self.rendering

    def timeline(self, name: str) -> FakeTimeline:
        return next(t for t in self.timelines if t.GetName() == name)


class FakeProjectManager:
    def __init__(self, app: FakeResolve) -> None:
        self._app = app

    def GetCurrentProject(self) -> FakeProject | None:
        return self._app.current

    def GetCurrentFolder(self) -> str:
        return self._app.pm_folder

    def LoadProject(self, projectName: str) -> FakeProject | None:
        self._app.calls.append(("LoadProject", projectName))
        project = self._app.projects.get(projectName)
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
    """The root ``resolve`` object. ``projects`` maps names to timeline names."""

    def __init__(self, projects: dict[str, list[str]], current: str | None = None) -> None:
        self.ids = itertools.count(1)
        self.calls: list[tuple[Any, ...]] = []
        self.projects = {name: FakeProject(self, name, tls) for name, tls in projects.items()}
        first = current if current is not None else next(iter(projects), None)
        self.current: FakeProject | None = self.projects[first] if first is not None else None
        self.page = "edit"
        self.pm_folder = "CCN"
        self.presets: dict[str, DeliverState] = {
            "YouTube 1080p": DeliverState(
                "mp4", "H265", 1, {"TargetDir": "C:/Exports", "CustomName": "yt"}
            )
        }
        self.missing: set[str] = set()
        self.load_silently_fails: set[str] = set()
        self.save_fails = False
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
        self.page = pageName
        return True

    # -- test helpers ----------------------------------------------------------------------

    def project(self, name: str) -> FakeProject:
        return self.projects[name]

    def call_names(self) -> list[str]:
        return [call[0] for call in self.calls]

    def human_open_project(self, name: str) -> None:
        self.current = self.projects[name]

    def human_set_page(self, page: str) -> None:
        self.page = page

    def human_move_playhead(self, timecode: str) -> None:
        assert self.current is not None and self.current.current_timeline is not None
        self.current.current_timeline.timecode = timecode
