"""The broker's narrow view of a running DaVinci Resolve Studio.

``ResolveGateway`` is the minimal surface the session broker needs: read the current context
(project, timeline, page, playhead, Deliver settings, media-pool bin) and put it back.
``ScriptApiGateway`` implements it on the real scripting objects of Resolve 21.1
(``Resolve`` / ``ProjectManager`` / ``Project`` / ``Timeline`` / ``MediaPool`` / ``Folder``).

Every mutating call is verified: Resolve reports most failures as ``False`` or ``None``, and
the gateway turns those into a ``GatewayError`` with a stable ``code`` so callers can build
structured results instead of guessing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

# Page names accepted by ``Resolve.OpenPage`` in the 21.1 API stubs.
VALID_PAGES: frozenset[str] = frozenset(
    {"media", "photo", "cut", "edit", "fusion", "color", "fairlight", "deliver"}
)

# Fields of ``ResolveContext`` that are state rather than a human's working position: they are
# not restored and a change in them is not a sign of human activity.
_NON_POSITIONAL_FIELDS: frozenset[str] = frozenset({"rendering"})


class GatewayError(Exception):
    """A Resolve call failed or returned something other than what was asked for."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FolderRef:
    """A media-pool folder (bin), identified by unique id where Resolve provides one.

    ``Folder`` has no parent accessor in the API, so a path would need a walk of the whole
    bin tree on every poll. The id is enough to find the folder again on restore.
    """

    name: str
    unique_id: str | None = None


@dataclass(frozen=True)
class ResolveContext:
    """What a person sitting at Resolve would expect to find again after a switch.

    ``render_format``/``render_codec``/``render_mode`` are the parts of the Deliver page the
    API can read back. ``LoadRenderPreset`` clobbers the rest (target dir, file name, quality,
    ...), and the API has no ``GetRenderSettings``, so the broker additionally snapshots the
    full Deliver settings into a temporary render preset (see ``ResolveGateway``).
    """

    project: str | None
    project_id: str | None = None
    timeline: str | None = None
    timeline_id: str | None = None
    page: str | None = None
    playhead: str | None = None
    render_format: str | None = None
    render_codec: str | None = None
    render_mode: int | None = None
    folder: FolderRef | None = None
    project_manager_folder: str | None = None
    rendering: bool = False

    def position(self) -> dict[str, object]:
        """The restorable working position, i.e. every field except state like ``rendering``."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in _NON_POSITIONAL_FIELDS
        }

    def same_position(self, other: ResolveContext) -> bool:
        """True when ``other`` shows the same working position as this context."""
        return self.position() == other.position()

    def diff(self, other: ResolveContext) -> dict[str, tuple[object, object]]:
        """Position fields that differ, as ``{field: (self_value, other_value)}``."""
        mine, theirs = self.position(), other.position()
        return {k: (mine[k], theirs[k]) for k in mine if mine[k] != theirs[k]}


@runtime_checkable
class ResolveGateway(Protocol):
    """The minimal Resolve surface the broker needs. Mutators raise ``GatewayError`` on failure."""

    def get_context(self) -> ResolveContext:
        """Read the current context. Read-only."""
        ...

    def load_project(self, name: str) -> None:
        """Make ``name`` the current project (no-op if it already is)."""
        ...

    def save_project(self) -> None:
        """Save the current project."""
        ...

    def set_current_timeline(self, name: str, unique_id: str | None = None) -> None:
        """Make a timeline of the current project current, by unique id if given, else by name."""
        ...

    def open_page(self, page: str) -> None:
        """Switch Resolve to ``page`` (one of ``VALID_PAGES``)."""
        ...

    def set_playhead(self, timecode: str) -> None:
        """Move the current timeline's playhead to ``timecode``."""
        ...

    def set_render_format_and_codec(self, render_format: str, codec: str) -> None:
        """Set the Deliver page's format and codec."""
        ...

    def set_render_mode(self, mode: int) -> None:
        """Set the Deliver render mode: 0 individual clips, 1 single clip."""
        ...

    def set_current_folder(self, folder: FolderRef) -> None:
        """Select a media-pool folder of the current project."""
        ...

    def save_render_preset(self, name: str) -> None:
        """Save the current Deliver settings as a new render preset called ``name``."""
        ...

    def load_render_preset(self, name: str) -> None:
        """Load render preset ``name`` into the Deliver settings (clobbers them)."""
        ...

    def delete_render_preset(self, name: str) -> None:
        """Delete render preset ``name``."""
        ...


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


class ScriptApiGateway:
    """``ResolveGateway`` on the real scripting objects (``DaVinciResolveScript.scriptapp``).

    Scripting objects are fetched fresh on every call: a ``Project`` handle goes stale after
    ``LoadProject``. The raw ``resolve`` object stays available for op implementations.
    """

    def __init__(self, resolve: Any) -> None:
        if resolve is None:
            raise GatewayError("no_resolve", "the Resolve scripting object is None")
        self.resolve = resolve

    # -- helpers ---------------------------------------------------------------------------

    def _project_manager(self) -> Any:
        pm = self.resolve.GetProjectManager()
        if pm is None:
            raise GatewayError("no_project_manager", "Resolve.GetProjectManager() returned None")
        return pm

    def _project(self) -> Any:
        project = self._project_manager().GetCurrentProject()
        if project is None:
            raise GatewayError("no_project", "no project is open")
        return project

    def _timelines(self, project: Any) -> Iterator[Any]:
        count = int(project.GetTimelineCount() or 0)
        for idx in range(1, count + 1):  # GetTimelineByIndex is 1-based
            timeline = project.GetTimelineByIndex(idx)
            if timeline is not None:
                yield timeline

    @staticmethod
    def _walk_folders(root: Any) -> Iterator[Any]:
        stack, seen = [root], set()
        while stack:
            folder = stack.pop()
            key = folder.GetUniqueId() or id(folder)
            if key in seen:
                continue
            seen.add(key)
            yield folder
            stack.extend(reversed(folder.GetSubFolderList() or []))

    # -- ResolveGateway ----------------------------------------------------------------------

    def get_context(self) -> ResolveContext:
        pm = self._project_manager()
        page = _str_or_none(self.resolve.GetCurrentPage())
        pm_folder = _str_or_none(pm.GetCurrentFolder())
        project = pm.GetCurrentProject()
        if project is None:
            return ResolveContext(project=None, page=page, project_manager_folder=pm_folder)
        timeline = project.GetCurrentTimeline()
        fmt = project.GetCurrentRenderFormatAndCodec() or {}
        mode = project.GetCurrentRenderMode()
        media_pool = project.GetMediaPool()
        folder = media_pool.GetCurrentFolder() if media_pool is not None else None
        return ResolveContext(
            project=_str_or_none(project.GetName()),
            project_id=_str_or_none(project.GetUniqueId()),
            timeline=_str_or_none(timeline.GetName()) if timeline is not None else None,
            timeline_id=_str_or_none(timeline.GetUniqueId()) if timeline is not None else None,
            page=page,
            playhead=_str_or_none(timeline.GetCurrentTimecode()) if timeline is not None else None,
            render_format=_str_or_none(fmt.get("format")),
            render_codec=_str_or_none(fmt.get("codec")),
            render_mode=int(mode) if mode is not None else None,
            folder=(
                FolderRef(str(folder.GetName()), _str_or_none(folder.GetUniqueId()))
                if folder is not None
                else None
            ),
            project_manager_folder=pm_folder,
            rendering=bool(project.IsRenderingInProgress()),
        )

    def load_project(self, name: str) -> None:
        pm = self._project_manager()
        current = pm.GetCurrentProject()
        if current is not None and current.GetName() == name:
            return
        if pm.LoadProject(name) is None:
            raise GatewayError(
                "project_not_found",
                f"LoadProject({name!r}) returned None; is it in the current project-manager "
                f"folder {pm.GetCurrentFolder()!r}?",
            )
        current = pm.GetCurrentProject()
        actual = current.GetName() if current is not None else None
        if actual != name:
            raise GatewayError(
                "project_load_failed", f"asked for project {name!r} but {actual!r} is open"
            )

    def save_project(self) -> None:
        if not self._project_manager().SaveProject():
            raise GatewayError("save_failed", "ProjectManager.SaveProject() returned False")

    def set_current_timeline(self, name: str, unique_id: str | None = None) -> None:
        project = self._project()
        timelines = list(self._timelines(project))
        matches = []
        if unique_id is not None:
            matches = [t for t in timelines if t.GetUniqueId() == unique_id]
        if not matches:
            matches = [t for t in timelines if t.GetName() == name]
        if not matches:
            raise GatewayError(
                "timeline_not_found", f"no timeline {name!r} in project {project.GetName()!r}"
            )
        if len(matches) > 1:
            raise GatewayError(
                "timeline_ambiguous",
                f"{len(matches)} timelines named {name!r} in project {project.GetName()!r}",
            )
        target = matches[0]
        current = project.GetCurrentTimeline()
        if current is not None and current.GetUniqueId() == target.GetUniqueId():
            return
        if not project.SetCurrentTimeline(target):
            raise GatewayError("timeline_set_failed", f"SetCurrentTimeline({name!r}) failed")
        current = project.GetCurrentTimeline()
        if current is None or current.GetUniqueId() != target.GetUniqueId():
            raise GatewayError("timeline_set_failed", f"timeline {name!r} did not become current")

    def open_page(self, page: str) -> None:
        if page not in VALID_PAGES:
            raise GatewayError("page_invalid", f"{page!r} is not one of {sorted(VALID_PAGES)}")
        if self.resolve.GetCurrentPage() == page:
            return
        if not self.resolve.OpenPage(page):
            raise GatewayError("page_open_failed", f"OpenPage({page!r}) returned False")

    def set_playhead(self, timecode: str) -> None:
        timeline = self._project().GetCurrentTimeline()
        if timeline is None:
            raise GatewayError("no_timeline", "no current timeline to move the playhead on")
        if not timeline.SetCurrentTimecode(timecode):
            raise GatewayError("playhead_set_failed", f"SetCurrentTimecode({timecode!r}) failed")

    def set_render_format_and_codec(self, render_format: str, codec: str) -> None:
        if not self._project().SetCurrentRenderFormatAndCodec(render_format, codec):
            raise GatewayError(
                "render_format_failed",
                f"SetCurrentRenderFormatAndCodec({render_format!r}, {codec!r}) failed",
            )

    def set_render_mode(self, mode: int) -> None:
        if not self._project().SetCurrentRenderMode(mode):
            raise GatewayError("render_mode_failed", f"SetCurrentRenderMode({mode!r}) failed")

    def set_current_folder(self, folder: FolderRef) -> None:
        media_pool = self._project().GetMediaPool()
        if media_pool is None:
            raise GatewayError("no_media_pool", "Project.GetMediaPool() returned None")
        found = list(self._walk_folders(media_pool.GetRootFolder()))
        matches = []
        if folder.unique_id is not None:
            matches = [f for f in found if f.GetUniqueId() == folder.unique_id]
        if not matches:
            matches = [f for f in found if f.GetName() == folder.name]
        if not matches:
            raise GatewayError("folder_not_found", f"no media-pool folder {folder.name!r}")
        if len(matches) > 1:
            raise GatewayError(
                "folder_ambiguous", f"{len(matches)} media-pool folders named {folder.name!r}"
            )
        if not media_pool.SetCurrentFolder(matches[0]):
            raise GatewayError("folder_set_failed", f"SetCurrentFolder({folder.name!r}) failed")

    def save_render_preset(self, name: str) -> None:
        if not self._project().SaveAsNewRenderPreset(name):
            raise GatewayError("preset_save_failed", f"SaveAsNewRenderPreset({name!r}) failed")

    def load_render_preset(self, name: str) -> None:
        if not self._project().LoadRenderPreset(name):
            raise GatewayError("preset_load_failed", f"LoadRenderPreset({name!r}) failed")

    def delete_render_preset(self, name: str) -> None:
        if not self._project().DeleteRenderPreset(name):
            raise GatewayError("preset_delete_failed", f"DeleteRenderPreset({name!r}) failed")
