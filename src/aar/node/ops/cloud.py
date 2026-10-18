"""Read-only Blackmagic Cloud ops: list projects, media sync status, folder freshness.

Blackmagic Cloud has no public API; everything here goes through a Resolve instance's
scripting API (DESIGN F1). None of these ops change Resolve. They are the read-only half of
S10; load, create, import, restore, ``RefreshFolders`` and setting ``liveCollaborationMode``
are the gated write half (human approval token, DESIGN R2/§5) and are not here.

What the 21.1 API truthfully exposes shapes these ops:

- There is no call that lists cloud projects as such. The four cloud functions
  (``LoadCloudProject``/``CreateCloudProject``/``ImportCloudProject``/``RestoreCloudProject``)
  all write, and ``GetDatabaseList`` does not list a cloud library (DESIGN F2). So
  ``cloud.projects`` lists the projects in the current Project Manager folder with
  ``GetProjectListInCurrentFolder`` + ``GetProjectAttributesInCurrentFolder`` (names,
  ``liveCollaborationMode``, dates, notes). A project's unique id is only readable once it is
  open (``Project.GetUniqueId``), so only the open project carries an id here.
- The project's cloud sync *mode* (``CloudSyncMode``, proxy-only vs proxy+original) is a
  write-only ``CloudSettings`` input with no getter. ``cloud.sync_status`` therefore reads the
  per-clip ``Cloud Sync`` status and flags proxy-only media from the readable per-clip
  evidence: a proxy is present while the original is offline (trap
  ``cloud-proxy-only-no-audio``; proxies have no audio in our setup, DESIGN R2).
- Freshness is ``Folder.GetIsFolderStale`` only; ``RefreshFolders`` is a write and is not
  called here.
"""

from __future__ import annotations

from pydantic import Field

from aar.node.ops import raw
from aar.node.ops.framework import Note, OpEnv, OpError, OpModel, op

# -- cloud.projects ---------------------------------------------------------------------------


class CloudProjectsParams(OpModel):
    pass


class CloudDatabase(OpModel):
    """The current database connection (``ProjectManager.GetCurrentDatabase``). The optional
    ``IpAddress`` of a PostgreSQL connection is deliberately not surfaced."""

    db_type: str
    db_name: str


class CloudProjectEntry(OpModel):
    name: str
    unique_id: str | None
    """Only the open project's id is readable without loading it (a gated write)."""
    is_open: bool
    live_collaboration_mode: str | None
    last_modified: str | None
    creation_date: str | None
    notes: str | None


class CloudProjectsResult(OpModel):
    current_folder: str
    """The current Project Manager folder's name (``""`` at the root). Reading the full path
    would navigate, so it is not measured here."""
    database: CloudDatabase
    projects: list[CloudProjectEntry]
    subfolders: tuple[str, ...]
    count: int
    by_collaboration: dict[str, int]
    open_project_id: str
    notes: list[Note]


def _opt(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


@op(
    "cloud.projects",
    params=CloudProjectsParams,
    result=CloudProjectsResult,
    mutates=False,
    traps=("pm-folder-scoped-load",),
)
def cloud_projects(env: OpEnv, params: CloudProjectsParams) -> CloudProjectsResult:
    """List the current library folder's projects with their collaboration attributes."""
    pm = env.project_manager()
    project = env.project()
    open_name, open_id = str(project.GetName()), str(project.GetUniqueId())
    current_folder = str(pm.GetCurrentFolder() or "")
    names = [str(n) for n in (pm.GetProjectListInCurrentFolder() or [])]
    subfolders = tuple(str(f) for f in (pm.GetFolderListInCurrentFolder() or []))
    attrs = pm.GetProjectAttributesInCurrentFolder() or {}
    db = pm.GetCurrentDatabase() or {}
    database = CloudDatabase(
        db_type=str(db.get("DbType") or ""), db_name=str(db.get("DbName") or "")
    )
    # Only a folder where the open name appears exactly once pins which entry is the open
    # project (pm-folder-scoped-load: names are not unique across folders).
    open_count = sum(1 for name in names if name == open_name)
    entries: list[CloudProjectEntry] = []
    by_collaboration: dict[str, int] = {}
    for name in names:
        a = attrs.get(name) or {}
        mode = _opt(a.get("liveCollaborationMode"))
        is_open = name == open_name and open_count == 1
        entries.append(
            CloudProjectEntry(
                name=name,
                unique_id=open_id if is_open else None,
                is_open=is_open,
                live_collaboration_mode=mode,
                last_modified=_opt(a.get("lastModifiedDate")),
                creation_date=_opt(a.get("creationDate")),
                notes=_opt(a.get("notes")),
            )
        )
        key = mode or "unknown"
        by_collaboration[key] = by_collaboration.get(key, 0) + 1
    notes: list[Note] = []
    if open_count > 1:
        notes.append(
            Note(
                code="open_project_ambiguous",
                message=f"{open_count} projects in this folder are named {open_name!r}; the "
                "open project's id is left unset on all of them",
                trap_id="pm-folder-scoped-load",
            )
        )
    if database.db_name == "Local Database":
        notes.append(
            Note(
                code="local_database",
                message="the current database is the local database; Blackmagic Cloud projects "
                "live in a cloud library, and no scripting call lists cloud libraries",
            )
        )
    return CloudProjectsResult(
        current_folder=current_folder,
        database=database,
        projects=entries,
        subfolders=subfolders,
        count=len(entries),
        by_collaboration=by_collaboration,
        open_project_id=open_id,
        notes=notes,
    )


# -- cloud.sync_status ------------------------------------------------------------------------


class CloudSyncParams(OpModel):
    bin_path: tuple[str, ...] = Field(default=(), description="Bin below the media-pool root.")
    recursive: bool = Field(default=True, strict=True)
    cursor: int = Field(
        strict=True,
        default=0,
        ge=0,
        description="Index of the first clip to read. cursor == total_clips is a legal resume "
        "point and returns an empty, done page; a cursor past that is an error.",
    )
    max_clips: int = Field(strict=True, default=200, ge=1, le=5000, description="Clips per call.")


class ClipSync(OpModel):
    media_id: str
    name: str
    bin_path: tuple[str, ...]
    type: str
    has_audio: bool
    cloud_sync: str
    """The resolve.CLOUD_SYNC_* status name, or 'unknown' if the raw value matched no code."""
    cloud_sync_raw: str
    proxy: str
    has_proxy: bool
    """A playable proxy (a resolution in 'Proxy'); 'Offline'/'None'/'' are not playable."""
    proxy_offline: bool
    """The proxy file itself is offline ('Proxy' == 'Offline'); not a playable proxy."""
    online_status: str
    original_offline: bool
    """The original media is offline ('Online Status' is neither 'Online' nor blank)."""
    proxy_only: bool
    """A playable proxy with the original offline: a build uses it and loses audio."""
    fully_offline: bool
    """The original is offline and there is no playable proxy: a build renders nothing."""


class CloudSyncResult(OpModel):
    bin_path: tuple[str, ...]
    recursive: bool
    total_clips: int
    cursor: int
    next_cursor: int | None
    done: bool
    scanned: int
    """Clips read on this call (the window ``[cursor, cursor + scanned)``)."""
    clips: list[ClipSync]
    status_counts: dict[str, int]
    """Per ``cloud_sync`` status totals for this page; sum across pages for the whole pool."""
    proxy_only: int
    fully_offline: int
    notes: list[Note]


@op(
    "cloud.sync_status",
    params=CloudSyncParams,
    result=CloudSyncResult,
    mutates=False,
    traps=("cloud-proxy-only-no-audio", "render-uses-original-media"),
)
def cloud_sync_status(env: OpEnv, params: CloudSyncParams) -> CloudSyncResult:
    """Per-clip Blackmagic Cloud sync status of the project's media, flagging proxy-only media."""
    labels = raw.cloud_sync_labels(env.resolve)
    flat: list[tuple[tuple[str, ...], object]] = []
    for where, folder in env.walk_bins(params.bin_path, params.recursive):
        for clip in folder.GetClipList() or []:
            flat.append((where, clip))
    total = len(flat)
    if params.cursor > total:
        raise OpError(
            "cursor_past_end",
            f"cursor {params.cursor} is past the last clip ({total} in scope)",
            "resume from a cursor <= total_clips from an earlier page, or start again at 0",
        )
    window = flat[params.cursor : params.cursor + params.max_clips]
    clips: list[ClipSync] = []
    status_counts: dict[str, int] = {}
    proxy_only = fully_offline = 0
    for where, clip in window:
        data = raw.read_cloud_sync(clip, labels)
        clips.append(
            ClipSync(
                media_id=data["media_id"],
                name=data["name"],
                bin_path=where,
                type=data["type"],
                has_audio=data["has_audio"],
                cloud_sync=data["cloud_sync"],
                cloud_sync_raw=data["cloud_sync_raw"],
                proxy=data["proxy"],
                has_proxy=data["has_proxy"],
                proxy_offline=data["proxy_offline"],
                online_status=data["online_status"],
                original_offline=data["original_offline"],
                proxy_only=data["proxy_only"],
                fully_offline=data["fully_offline"],
            )
        )
        status_counts[data["cloud_sync"]] = status_counts.get(data["cloud_sync"], 0) + 1
        if data["proxy_only"]:
            proxy_only += 1
        if data["fully_offline"]:
            fully_offline += 1
    reached = params.cursor + len(window)
    next_cursor = reached if reached < total else None
    notes: list[Note] = []
    if proxy_only:
        notes.append(
            Note(
                code="proxy_only",
                message=f"{proxy_only} clip(s) on this page are proxy-only (original offline, a "
                "playable proxy present); a build from them loses audio",
                trap_id="cloud-proxy-only-no-audio",
            )
        )
    if fully_offline:
        notes.append(
            Note(
                code="fully_offline",
                message=f"{fully_offline} clip(s) on this page have no playable media (original "
                "offline, no usable proxy); a render would produce nothing",
                trap_id="render-uses-original-media",
            )
        )
    return CloudSyncResult(
        bin_path=params.bin_path,
        recursive=params.recursive,
        total_clips=total,
        cursor=params.cursor,
        next_cursor=next_cursor,
        done=next_cursor is None,
        scanned=len(window),
        clips=clips,
        status_counts=status_counts,
        proxy_only=proxy_only,
        fully_offline=fully_offline,
        notes=notes,
    )


# -- cloud.freshness --------------------------------------------------------------------------


class CloudFreshnessParams(OpModel):
    bin_path: tuple[str, ...] = Field(default=(), description="Bin below the media-pool root.")
    recursive: bool = Field(default=True, strict=True)
    max_folders: int = Field(strict=True, default=2000, ge=1, le=20000)


class FolderFreshness(OpModel):
    bin_path: tuple[str, ...]
    name: str
    unique_id: str
    stale: bool


class CloudFreshnessResult(OpModel):
    bin_path: tuple[str, ...]
    recursive: bool
    folders: list[FolderFreshness]
    folder_count: int
    stale_count: int
    truncated: bool
    notes: list[Note]


@op(
    "cloud.freshness",
    params=CloudFreshnessParams,
    result=CloudFreshnessResult,
    mutates=False,
)
def cloud_freshness(env: OpEnv, params: CloudFreshnessParams) -> CloudFreshnessResult:
    """Report which media-pool folders are stale in collaboration mode (read-only; no refresh)."""
    folders: list[FolderFreshness] = []
    stale_count = 0
    truncated = False
    for where, folder in env.walk_bins(params.bin_path, params.recursive):
        if len(folders) >= params.max_folders:
            truncated = True
            break
        stale = bool(folder.GetIsFolderStale())
        folders.append(
            FolderFreshness(
                bin_path=where,
                name=str(folder.GetName()),
                unique_id=str(folder.GetUniqueId()),
                stale=stale,
            )
        )
        if stale:
            stale_count += 1
    notes: list[Note] = []
    if stale_count:
        notes.append(
            Note(
                code="stale_folders",
                message=f"{stale_count} media-pool folder(s) are stale in collaboration mode; a "
                "working human would RefreshFolders. This read-only op does not refresh.",
            )
        )
    return CloudFreshnessResult(
        bin_path=params.bin_path,
        recursive=params.recursive,
        folders=folders,
        folder_count=len(folders),
        stale_count=stale_count,
        truncated=truncated,
        notes=notes,
    )
