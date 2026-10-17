"""Read ops: find projects, list timelines, describe a timeline, find media.

None of these change Resolve, except that ``project.find`` with ``recursive=True`` walks the
Project Manager folder tree (and puts it back where it was).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aar.node.gateway import PMFolderListing, ScriptApiGateway
from aar.node.locator import ProjectLocator
from aar.node.ops import raw
from aar.node.ops.framework import (
    Note,
    OpEnv,
    OpError,
    OpModel,
    TimelineIdentity,
    TimelineRef,
    TrackType,
    op,
)

# -- project.find -----------------------------------------------------------------------------


class ProjectFindParams(OpModel):
    name: str | None = Field(default=None, min_length=1, description="Exact project name.")
    recursive: bool = Field(
        default=True,
        description="Walk the whole Project Manager tree (and restore it). False lists the "
        "current folder only; reading its path navigates only when it is not the root.",
    )
    max_folders: int = Field(default=500, ge=1, le=5000)


class FoundProject(OpModel):
    locator: ProjectLocator
    is_open: bool


class ProjectFindResult(OpModel):
    projects: list[FoundProject]
    folders_listed: int
    project_manager_path: tuple[str, ...]
    """Where the Project Manager was before (and is again after) the walk."""
    open_project: str | None
    open_project_id: str | None
    notes: list[Note] = Field(default_factory=list)


def _gateway(env: OpEnv) -> ScriptApiGateway:
    if not isinstance(env.gateway, ScriptApiGateway):
        raise OpError(
            "no_script_gateway",
            f"{type(env.gateway).__name__} cannot walk the Project Manager",
            "run on a ScriptApiGateway",
        )
    return env.gateway


@op(
    "project.find",
    params=ProjectFindParams,
    result=ProjectFindResult,
    mutates=False,
    traps=("pm-folder-scoped-load",),
)
def project_find(env: OpEnv, params: ProjectFindParams) -> ProjectFindResult:
    """Find projects in the Project Manager folder tree, as locators."""
    gw = _gateway(env)
    project = env.project()
    open_name, open_id = str(project.GetName()), str(project.GetUniqueId())
    path = gw.get_project_manager_path()
    if params.recursive:
        listings = gw.walk_project_manager(params.max_folders)
        after = gw.get_project_manager_path()
        if after != path:
            raise OpError(
                "pm_not_restored",
                f"the Project Manager ended in {'/'.join(after)!r}, not {'/'.join(path)!r}",
                "navigate the Project Manager back by hand",
            )
    else:
        here = gw.list_project_manager_folder()
        listings = [PMFolderListing(path, here.projects, here.folders)]
    hits = [
        (listing.path, name)
        for listing in listings
        for name in listing.projects
        if params.name is None or name == params.name
    ]
    notes: list[Note] = []
    same_name = [h for h in hits if h[1] == open_name]
    # The open project is somewhere in the tree. Only a full walk that finds its name exactly
    # once proves which entry it is; then that entry gets the id.
    open_entry = same_name[0] if params.recursive and len(same_name) == 1 else None
    if same_name and open_entry is None:
        notes.append(
            Note(
                code="open_project_ambiguous",
                message=f"{len(same_name)} listed project(s) are named like the open project "
                f"{open_name!r}"
                + ("" if params.recursive else " and only one folder was listed")
                + "; their ids are left unset",
                trap_id="pm-folder-scoped-load",
            )
        )
    projects = [
        FoundProject(
            locator=ProjectLocator(where, name, open_id if (where, name) == open_entry else None),
            is_open=(where, name) == open_entry,
        )
        for where, name in hits
    ]
    return ProjectFindResult(
        projects=projects,
        folders_listed=len(listings),
        project_manager_path=path,
        open_project=open_name,
        open_project_id=open_id,
        notes=notes,
    )


# -- timeline.list ------------------------------------------------------------------------------


class TimelineListParams(OpModel):
    pass


class TimelineSummary(OpModel):
    index: int
    name: str
    unique_id: str
    is_current: bool
    start_frame: int
    end_frame: int
    duration: int


class TimelineListResult(OpModel):
    project: str
    project_id: str
    current: TimelineIdentity | None
    timelines: list[TimelineSummary]


@op("timeline.list", params=TimelineListParams, result=TimelineListResult, mutates=False)
def timeline_list(env: OpEnv, params: TimelineListParams) -> TimelineListResult:
    """List the target project's timelines with their frame ranges."""
    project = env.project()
    current = project.GetCurrentTimeline()
    current_id = str(current.GetUniqueId()) if current is not None else None
    rows = []
    for index, timeline in enumerate(env.timelines(), start=1):
        start, end = int(timeline.GetStartFrame()), int(timeline.GetEndFrame())
        uid = str(timeline.GetUniqueId())
        rows.append(
            TimelineSummary(
                index=index,
                name=str(timeline.GetName()),
                unique_id=uid,
                is_current=uid == current_id,
                start_frame=start,
                end_frame=end,
                duration=end - start,
            )
        )
    return TimelineListResult(
        project=str(project.GetName()),
        project_id=str(project.GetUniqueId()),
        current=env.identity(current) if current is not None else None,
        timelines=rows,
    )


# -- timeline.describe -------------------------------------------------------------------------


class DescribeParams(OpModel):
    timeline: TimelineRef | None = Field(default=None, description="Default: current timeline.")
    track_types: tuple[TrackType, ...] = ("video", "audio", "subtitle")
    max_items_per_track: int = Field(default=1000, ge=1, le=20000)
    include_media: bool = True


class TimelineInfo(OpModel):
    name: str
    unique_id: str
    fps: float | None
    drop_frame: bool
    start_frame: int
    end_frame: int
    duration: int
    start_timecode: str
    width: int | None
    height: int | None


class ItemInfo(OpModel):
    """One timeline item. ``start``/``end`` are absolute timeline frames, ``end`` exclusive;
    ``record_in``/``record_out`` are the same, counted from the timeline start. Source frames
    are as Resolve reports them (trap ``source-frame-readback-rounding``)."""

    index: int
    item_id: str
    name: str
    type: str
    start: int
    end: int
    duration: int
    record_in: int
    record_out: int
    source_start: int | None
    source_end: int | None
    media_id: str | None
    fade_in: int | None
    fade_out: int | None
    audio_volume_db: float | None
    enabled: bool


class GapInfo(OpModel):
    start: int
    end: int
    frames: int
    record_in: int
    record_out: int


class TrackInfo(OpModel):
    type: TrackType
    index: int
    name: str
    enabled: bool
    locked: bool
    subtype: str
    item_count: int
    transitions: int
    """Transition items among the items read; they are not counted as gaps or overlaps."""
    truncated: bool
    items: list[ItemInfo]
    gaps: list[GapInfo]
    overlaps: int


class MarkerInfo(OpModel):
    frame: int
    color: str
    name: str
    note: str
    duration: int
    custom_data: str


class MediaInfo(OpModel):
    media_id: str
    name: str
    type: str
    format: str
    video_codec: str
    file_name: str
    file_path: str
    fps: float | None
    frames: int | None
    is_braw: bool
    is_still: bool
    is_multicam: bool
    has_video: bool
    has_audio: bool
    online_status: str
    proxy: str
    input_color_space: str
    bin_path: tuple[str, ...] | None = None


class DescribeResult(OpModel):
    timeline: TimelineInfo
    tracks: list[TrackInfo]
    markers: list[MarkerInfo]
    media: list[MediaInfo]
    notes: list[Note]
    text: str
    """The same facts as compact text, one line per track, item, gap and marker."""


def _media_notes(media: list[MediaInfo]) -> list[Note]:
    notes = []
    for m in media:
        if m.online_status and m.online_status != "Online":
            proxy = m.proxy not in ("", "None")
            notes.append(
                Note(
                    code="media_offline",
                    message=f"media {m.name!r} [{m.media_id}] is {m.online_status!r}"
                    + (" (it has a proxy, so it may still play)" if proxy else "")
                    + "; a render will fail on it",
                    trap_id="render-uses-original-media",
                )
            )
        if m.is_multicam:
            notes.append(
                Note(
                    code="multicam_item",
                    message=f"multicam clip {m.name!r} [{m.media_id}]: its angle cannot be "
                    "read or set through the API",
                    trap_id="multicam-angles",
                )
            )
    return notes


def _fmt_item(item: ItemInfo) -> str:
    parts = [f"  {item.record_in:>7} {item.record_out:>7} {item.duration:>6}f  {item.name}"]
    parts.append(f" src {item.source_start}-{item.source_end}")
    if item.fade_in or item.fade_out:
        parts.append(f" fade {item.fade_in or 0}/{item.fade_out or 0}")
    if item.audio_volume_db is not None and abs(item.audio_volume_db) > 1e-9:
        parts.append(f" vol {item.audio_volume_db:+.2f}dB")
    if not item.enabled:
        parts.append(" DISABLED")
    parts.append(f" [{item.item_id}]")
    return "".join(parts)


def describe_text(result_tl: TimelineInfo, tracks: list[TrackInfo], markers: list[MarkerInfo],
                  notes: list[Note]) -> str:  # fmt: skip
    """The compact text form of a timeline description."""
    t = result_tl
    rate = f"{t.fps:g}" if t.fps is not None else "?"
    lines = [
        f'timeline "{t.name}" [{t.unique_id}] {rate} fps{" DF" if t.drop_frame else ""}, '
        f"start {t.start_frame} ({t.start_timecode}), {t.duration} frames"
        + (f", {t.width}x{t.height}" if t.width and t.height else ""),
        "positions are frames from the timeline start, end exclusive",
    ]
    for track in tracks:
        letter = {"video": "V", "audio": "A", "subtitle": "S"}[track.type]
        flags = "".join(
            f" {flag}" for flag, on in (("off", not track.enabled), ("locked", track.locked)) if on
        )
        sub = f" {track.subtype}" if track.subtype else ""
        lines.append(
            f'{letter}{track.index} "{track.name}"{sub}{flags}: {track.item_count} item(s), '
            f"{len(track.gaps)} gap(s)"
            + (f", {track.overlaps} overlap(s)" if track.overlaps else "")
            + (" (truncated)" if track.truncated else "")
        )
        events: list[tuple[int, str]] = [(i.record_in, _fmt_item(i)) for i in track.items]
        events += [
            (g.record_in, f"  {g.record_in:>7} {g.record_out:>7} {g.frames:>6}f  GAP")
            for g in track.gaps
        ]
        lines.extend(text for _, text in sorted(events, key=lambda e: e[0]))
    if markers:
        lines.append(f"markers: {len(markers)}")
        for m in markers:
            label = " ".join(x for x in (m.name, m.note) if x)
            lines.append(f"  {m.frame:>7} {m.color} {m.duration}f {label}".rstrip())
    else:
        lines.append("markers: none")
    for note in notes:
        lines.append(
            f"! {note.code}: {note.message}" + (f" (trap {note.trap_id})" if note.trap_id else "")
        )
    return "\n".join(lines)


@op(
    "timeline.describe",
    params=DescribeParams,
    result=DescribeResult,
    mutates=False,
    traps=("source-frame-readback-rounding", "render-uses-original-media", "multicam-angles"),
)
def timeline_describe(env: OpEnv, params: DescribeParams) -> DescribeResult:
    """Describe a timeline: tracks, items, source ranges, gaps, markers and fades."""
    timeline = env.timeline(params.timeline)
    data = raw.read_timeline(
        timeline, params.track_types, params.max_items_per_track, params.include_media
    )
    info = TimelineInfo(**data["timeline"])
    tracks = [
        TrackInfo(
            **{k: v for k, v in t.items() if k not in ("items", "gaps")},
            items=[ItemInfo(index=i, **row) for i, row in enumerate(t["items"])],
            gaps=[GapInfo(**g) for g in t["gaps"]],
        )
        for t in data["tracks"]
    ]
    markers = [MarkerInfo(**m) for m in data["markers"]]
    media = [MediaInfo(**m) for m in data["media"]]
    notes = _media_notes(media)
    for track in tracks:
        if track.overlaps:
            notes.append(
                Note(
                    code="overlapping_items",
                    message=f"{track.type} track {track.index} has {track.overlaps} "
                    "overlapping item pair(s)",
                )
            )
    return DescribeResult(
        timeline=info,
        tracks=tracks,
        markers=markers,
        media=media,
        notes=notes,
        text=describe_text(info, tracks, markers, notes),
    )


# -- media.find -------------------------------------------------------------------------------


class MediaFindParams(OpModel):
    bin_path: tuple[str, ...] = Field(default=(), description="Bin below the media-pool root.")
    name: str | None = Field(default=None, min_length=1, description="Exact clip name.")
    name_contains: str | None = Field(default=None, min_length=1)
    path_contains: str | None = Field(
        default=None, min_length=1, description="Substring of the clip's file path."
    )
    recursive: bool = False
    limit: int = Field(default=200, ge=1, le=5000)


class MediaFindResult(OpModel):
    bin_path: tuple[str, ...]
    recursive: bool
    scanned: int
    truncated: bool
    media: list[MediaInfo]


def _media_matches(m: dict[str, Any], params: MediaFindParams) -> bool:
    if params.name is not None and m["name"] != params.name:
        return False
    if params.name_contains is not None and params.name_contains not in m["name"]:
        return False
    return params.path_contains is None or params.path_contains in m["file_path"]


@op("media.find", params=MediaFindParams, result=MediaFindResult, mutates=False)
def media_find(env: OpEnv, params: MediaFindParams) -> MediaFindResult:
    """Find media-pool clips by name or file path within a bin (non-recursive by default)."""
    found: list[MediaInfo] = []
    scanned = 0
    truncated = False
    for where, folder in env.walk_bins(params.bin_path, params.recursive):
        for clip in folder.GetClipList() or []:
            scanned += 1
            m = raw.read_media(clip)
            if not _media_matches(m, params):
                continue
            if len(found) >= params.limit:
                truncated = True
                break
            found.append(MediaInfo(**m, bin_path=where))
        if truncated:
            break
    return MediaFindResult(
        bin_path=params.bin_path,
        recursive=params.recursive,
        scanned=scanned,
        truncated=truncated,
        media=found,
    )
