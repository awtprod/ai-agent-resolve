"""The broker's narrow view of a running DaVinci Resolve Studio.

``ResolveGateway`` is the minimal surface the session broker needs: read the current context
(project, timeline, page, playhead, Deliver settings, media-pool bin, Project Manager folder)
and put it back.
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

from aar.node.locator import ProjectLocator

# Page names accepted by ``Resolve.OpenPage`` in the 21.1 API stubs.
VALID_PAGES: frozenset[str] = frozenset(
    {"media", "photo", "cut", "edit", "fusion", "color", "fairlight", "deliver"}
)

# Fields of ``ResolveContext`` that are state rather than a human's working position: they are
# not restored and a change in them is not a sign of human activity.
_NON_POSITIONAL_FIELDS: frozenset[str] = frozenset({"rendering"})

# Fields that are only measured on request (reading them navigates the Project Manager). A
# ``None`` there means "not measured", which is never a difference.
_MEASURED_ON_REQUEST: frozenset[str] = frozenset({"project_manager_path"})

# Project Manager folders nest; anything deeper than this is treated as a broken walk.
MAX_PM_DEPTH = 64


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
    """The current Project Manager folder's name (``GetCurrentFolder``); the root is ``None``."""
    project_manager_path: tuple[str, ...] | None = None
    """The full Project Manager path from the root, ``()`` at the root. Only measured on request
    (``get_context(with_pm_path=True)``), because reading it walks the folder tree."""
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
        return not self.diff(other)

    def diff(self, other: ResolveContext) -> dict[str, tuple[object, object]]:
        """Position fields that differ, as ``{field: (self_value, other_value)}``.

        A field measured only on request (the Project Manager path) is compared only when
        both sides measured it.
        """
        mine, theirs = self.position(), other.position()
        return {
            k: (mine[k], theirs[k])
            for k in mine
            if mine[k] != theirs[k]
            and not (k in _MEASURED_ON_REQUEST and (mine[k] is None or theirs[k] is None))
        }


@dataclass(frozen=True)
class PMFolderListing:
    """One Project Manager folder: its path from the root, its projects and its subfolders."""

    path: tuple[str, ...]
    projects: tuple[str, ...]
    folders: tuple[str, ...]


@runtime_checkable
class ResolveGateway(Protocol):
    """The minimal Resolve surface the broker needs. Mutators raise ``GatewayError`` on failure."""

    def get_context(self, *, with_pm_path: bool = False) -> ResolveContext:
        """Read the current context. Read-only, except that ``with_pm_path`` walks the Project
        Manager up to the root and back down to measure ``project_manager_path``."""
        ...

    def load_project(self, locator: ProjectLocator) -> None:
        """Make the project at ``locator`` current (no-op if it already is).

        Navigates the Project Manager to ``locator.folder_path`` first, because ``LoadProject``
        only looks in the current folder, and verifies ``project_id`` after loading.
        """
        ...

    def get_project_manager_path(self) -> tuple[str, ...]:
        """The current Project Manager folder's full path from the root. Ends where it began."""
        ...

    def open_project_manager_path(self, path: tuple[str, ...]) -> None:
        """Navigate the Project Manager to ``path`` (from the root)."""
        ...

    def list_project_manager_folder(self) -> PMFolderListing:
        """Projects and subfolders of the current Project Manager folder. Read-only."""
        ...

    def find_project_folders(self, name: str) -> list[tuple[str, ...]]:
        """Paths of every Project Manager folder holding a project called ``name``.

        Walks the whole tree and returns to the starting folder.
        """
        ...

    def save_project(self) -> None:
        """Save the current project."""
        ...

    def set_current_timeline(
        self, name: str | None, unique_id: str | None = None, *, strict: bool = False
    ) -> None:
        """Make a timeline of the current project current, by unique id if given, else by name.

        ``strict``: the unique id is the identity; a name given with it must be that
        timeline's name (``timeline_ref_mismatch`` otherwise), and an unknown id never falls
        back to the name. Without ``strict`` (the broker's restore) the id wins over a stale
        name and an unknown id falls back to the name.
        """
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

    def _pm_folder_name(self, pm: Any) -> str:
        name = pm.GetCurrentFolder()
        return "" if name is None else str(name)

    def _open_child(self, pm: Any, name: str, where: tuple[str, ...]) -> None:
        if not pm.OpenFolder(name):
            raise GatewayError(
                "pm_open_failed", f"OpenFolder({name!r}) returned False in {'/'.join(where)!r}"
            )
        actual = self._pm_folder_name(pm)
        if actual != name:
            raise GatewayError(
                "pm_open_failed", f"OpenFolder({name!r}) left the Project Manager in {actual!r}"
            )

    def _walk_here(
        self, pm: Any, path: tuple[str, ...], out: list[PMFolderListing], max_folders: int
    ) -> None:
        listing = self.list_project_manager_folder()
        out.append(PMFolderListing(path, listing.projects, listing.folders))
        for name in listing.folders:
            if len(out) >= max_folders:
                raise GatewayError(
                    "pm_walk_too_large", f"more than {max_folders} Project Manager folders"
                )
            if len(path) + 1 > MAX_PM_DEPTH:
                raise GatewayError("pm_too_deep", f"folders nest deeper than {MAX_PM_DEPTH}")
            self._open_child(pm, name, path)
            self._walk_here(pm, (*path, name), out, max_folders)
            if not pm.GotoParentFolder():
                raise GatewayError(
                    "pm_parent_failed", f"GotoParentFolder() failed in {'/'.join((*path, name))!r}"
                )
            expected = path[-1] if path else ""
            if self._pm_folder_name(pm) != expected:
                raise GatewayError(
                    "pm_parent_failed", f"GotoParentFolder() did not return to {expected!r}"
                )

    def walk_project_manager(self, max_folders: int = 1000) -> list[PMFolderListing]:
        """Every Project Manager folder, depth first from the root. Ends where it began."""
        pm = self._project_manager()
        start = self.get_project_manager_path()
        out: list[PMFolderListing] = []
        try:
            if not pm.GotoRootFolder():
                raise GatewayError("pm_root_failed", "GotoRootFolder() returned False")
            self._walk_here(pm, (), out, max_folders)
        finally:
            # Always try to put the Project Manager back. If this fails too, its error is
            # raised with the walk's error (if any) chained as its context.
            self.open_project_manager_path(start)
        return out

    # -- ResolveGateway ----------------------------------------------------------------------

    def _descend(self, pm: Any, path: tuple[str, ...]) -> None:
        """From the root, open ``path`` one listed child at a time, verifying every level.

        Each segment must be listed in its parent (``GetFolderListInCurrentFolder``) and must
        become the current folder; so the whole path from the root is checked, not just its
        leaf. On a failure the error says how far the walk got (where the Project Manager is).
        """
        if self._pm_folder_name(pm) != "":
            raise GatewayError(
                "pm_root_failed",
                f"expected the Project Manager root, found {self._pm_folder_name(pm)!r}",
            )
        here: tuple[str, ...] = ()
        for name in path:
            listed = [str(f) for f in (pm.GetFolderListInCurrentFolder() or [])]
            where = "/".join(here) or "<root>"
            if name not in listed:
                raise GatewayError(
                    "pm_path_unverified",
                    f"folder {name!r} is not listed in {where!r}; the Project Manager is left "
                    f"in {where!r}",
                )
            try:
                self._open_child(pm, name, here)
            except GatewayError as exc:
                raise GatewayError(
                    exc.code,
                    f"{exc.message}; the walk to {'/'.join(path)!r} stopped in {where!r}",
                ) from exc
            here = (*here, name)

    def get_project_manager_path(self) -> tuple[str, ...]:
        # GetCurrentFolder() only returns the current folder's name ("" at the root), so the
        # path is read by walking up to the root, recording names, and walking back down.
        pm = self._project_manager()
        start = self._pm_folder_name(pm)
        if start == "":
            return ()
        names: list[str] = []
        for _ in range(MAX_PM_DEPTH):
            name = self._pm_folder_name(pm)
            if name == "":
                break
            if not pm.GotoParentFolder():
                # The root reads as "" (live 21.1), so a named folder has a parent: this is a
                # failure, not the root. Walk back down to where the walk started, then report.
                for child in reversed(names):
                    if not pm.OpenFolder(child):
                        break
                raise GatewayError(
                    "pm_parent_failed",
                    f"GotoParentFolder() returned False in {name!r} while reading the path of "
                    f"{start!r}; the Project Manager is in {self._pm_folder_name(pm)!r}",
                )
            names.append(name)
        else:
            raise GatewayError("pm_too_deep", f"folders nest deeper than {MAX_PM_DEPTH}")
        path = tuple(reversed(names))
        self._descend(pm, path)
        return path

    def open_project_manager_path(self, path: tuple[str, ...]) -> None:
        pm = self._project_manager()
        if not pm.GotoRootFolder():
            raise GatewayError("pm_root_failed", "GotoRootFolder() returned False")
        self._descend(pm, path)

    def list_project_manager_folder(self) -> PMFolderListing:
        pm = self._project_manager()
        projects = tuple(str(p) for p in (pm.GetProjectListInCurrentFolder() or []))
        folders = tuple(str(f) for f in (pm.GetFolderListInCurrentFolder() or []))
        return PMFolderListing((), projects, folders)

    def find_project_folders(self, name: str) -> list[tuple[str, ...]]:
        return [listing.path for listing in self.walk_project_manager() if name in listing.projects]

    def get_context(self, *, with_pm_path: bool = False) -> ResolveContext:
        pm = self._project_manager()
        pm_path = self.get_project_manager_path() if with_pm_path else None
        page = _str_or_none(self.resolve.GetCurrentPage())
        pm_folder = _str_or_none(pm.GetCurrentFolder())
        project = pm.GetCurrentProject()
        if project is None:
            return ResolveContext(
                project=None,
                page=page,
                project_manager_folder=pm_folder,
                project_manager_path=pm_path,
            )
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
            project_manager_path=pm_path,
            rendering=bool(project.IsRenderingInProgress()),
        )

    def load_project(self, locator: ProjectLocator) -> None:
        pm = self._project_manager()
        current = pm.GetCurrentProject()
        if current is not None and locator.matches(
            _str_or_none(current.GetName()), _str_or_none(current.GetUniqueId())
        ):
            return
        # LoadProject resolves the name in the current Project Manager folder only
        # (trap pm-folder-scoped-load), so go to the project's folder first.
        self.open_project_manager_path(locator.folder_path)
        name = locator.name
        if pm.LoadProject(name) is None:
            raise GatewayError(
                "project_not_found",
                f"LoadProject({name!r}) returned None in project-manager folder "
                f"{'/'.join(locator.folder_path) or '<root>'!r}",
            )
        current = pm.GetCurrentProject()
        actual = current.GetName() if current is not None else None
        if actual != name:
            raise GatewayError(
                "project_load_failed", f"asked for project {name!r} but {actual!r} is open"
            )
        actual_id = _str_or_none(current.GetUniqueId()) if current is not None else None
        if locator.project_id is not None and actual_id != locator.project_id:
            raise GatewayError(
                "project_id_mismatch",
                f"loaded {name!r} from {'/'.join(locator.folder_path) or '<root>'!r} but its id "
                f"is {actual_id!r}, not {locator.project_id!r}",
            )

    def save_project(self) -> None:
        if not self._project_manager().SaveProject():
            raise GatewayError("save_failed", "ProjectManager.SaveProject() returned False")

    def set_current_timeline(
        self, name: str | None, unique_id: str | None = None, *, strict: bool = False
    ) -> None:
        if name is None and unique_id is None:
            raise GatewayError("timeline_ref_empty", "a timeline needs a name or a unique id")
        project = self._project()
        timelines = list(self._timelines(project))
        label = name if name is not None else unique_id
        matches = []
        if unique_id is not None:
            matches = [t for t in timelines if t.GetUniqueId() == unique_id]
            if strict and matches and name is not None and matches[0].GetName() != name:
                raise GatewayError(
                    "timeline_ref_mismatch",
                    f"timeline id {unique_id!r} is {matches[0].GetName()!r}, not {name!r}",
                )
        if not matches and name is not None and not (strict and unique_id is not None):
            matches = [t for t in timelines if t.GetName() == name]
        if not matches:
            if unique_id is not None and (strict or name is None):
                unused = f"; the name {name!r} was not used to look it up" if name else ""
                message = (
                    f"no timeline with id {unique_id!r} in project {project.GetName()!r}{unused}"
                )
            else:
                message = f"no timeline {label!r} in project {project.GetName()!r}"
            raise GatewayError("timeline_not_found", message)
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
            raise GatewayError("timeline_set_failed", f"SetCurrentTimeline({label!r}) failed")
        current = project.GetCurrentTimeline()
        if current is None or current.GetUniqueId() != target.GetUniqueId():
            raise GatewayError("timeline_set_failed", f"timeline {label!r} did not become current")

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
