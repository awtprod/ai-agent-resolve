"""Build ops: the Resolve calls the ``plan.apply`` compiler (S13h2) needs to build an edit.

``media.import``, ``track.ensure``, ``transition.add``, ``item.set_color``,
``item.set_transform`` and ``title.place``. There is no compiler here; these are the ops it
compiles to.

Every op follows the write-op contract of :mod:`aar.node.ops.write`: it reads the current
state first and reports ``present`` without a call when that already matches; it reads back
after every write, and a write that returned success but reads back differently is an
``OpError`` with a stable code. Where 21.1 has no getter (the CDL), the result says so
(``verified: false``) instead of claiming a success it could not see. Every error raised once
some targets are done carries them as ``completed``.

Never destructive: no op deletes or replaces a track, clip, transition, node or media-pool item
it did not create in the same call, moves or relinks media, or changes the project or the
database. The one deletion is :func:`_undo_transition`: ``transition.add`` removing the
transition it has just made when its read-back is wrong. A test pins this with an AST scan.

Every op except ``media.import`` edits a ``timeline`` named by unique id, so the R3 branch
guard applies to it unchanged. ``media.import`` names no timeline and has a guard exemption
that says why it is safe (:data:`IMPORT_GUARD_EXEMPTION`).

Resolve behaviour that the 21.1 stubs do not state, and that the code relies on, is marked with
an ``[I]`` comment (see the S13 facts file); a live smoke must confirm each before it is
trusted.
"""

from __future__ import annotations

import itertools
import re
import string
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, model_validator

from aar.node.ops import raw
from aar.node.ops.framework import (
    ItemRef,
    MediaRef,
    Note,
    OpEnv,
    OpError,
    OpModel,
    TimelineIdentity,
    TimelineRef,
    TrackType,
    op,
)
from aar.node.ops.timebase import exact_rate
from aar.node.ops.write import (
    _JAM_HINT,
    AppendRange,
    _Appender,
    _clips,
    _int,
    _media_id,
    _no_repeats,
    _overlapping,
    _Planned,
    _preflight,
    _set_properties,
    _with_bounce,
    _with_completed,
)
from aar.plan.model import Name

MAX_TARGETS = 50
"""Every list an op takes holds 1 to this many entries."""

BinName = Annotated[Name, Field(max_length=128)]
TrackName = Annotated[Name, Field(max_length=64)]
TransitionName = Annotated[Name, Field(max_length=64)]
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _no_control(value: str) -> str:
    if _CONTROL.search(value):
        raise ValueError("must not contain NUL, line breaks or other control characters")
    return value


TitleLine = Annotated[Name, Field(max_length=200), AfterValidator(_no_control)]
LutPath = Annotated[Name, Field(max_length=1024), AfterValidator(_no_control)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def _partial(exc: OpError, done: list[Any], **detail: object) -> OpError:
    """``exc`` with the finished targets as ``completed`` and ``detail`` merged in (existing
    detail keys win, so an inner error's own ``completed`` is never overwritten)."""
    error = _with_completed(exc, done)
    extra = {k: v for k, v in error.detail.items() if k not in ("fix_hint", "trap_id")}
    for key, value in detail.items():
        extra.setdefault(key, value)
    return OpError(error.code, error.message, error.fix_hint, error.trap_id, **extra)


def _bounce_note(call: str, where: str) -> Note:
    return Note(
        code="page_bounced",
        message=f"{call} returned False until a page bounce ({where})",
        trap_id="setproperty-after-color",
    )


# -- media.import ------------------------------------------------------------------------------

IMPORT_GUARD_EXEMPTION = (
    "additive: only adds media-pool items to the named bin; never alters, moves, relinks or "
    "deletes existing items or any timeline"
)
EDIT_PATH_MAX = 1024
_DRIVE = re.compile(r"^[A-Za-z]:[\\/](?P<rest>.*)$", re.S)
_UNC = re.compile(r"^\\\\(?P<rest>.*)$", re.S)


def _edit_path(value: str) -> str:
    """An absolute edit-PC (Windows) file path: ``X:/...``, ``X:\\...`` or a UNC
    ``\\\\host\\share\\...``; never relative, with ``..``, or with control characters."""
    if len(value) > EDIT_PATH_MAX:
        raise ValueError(f"an edit-PC path is at most {EDIT_PATH_MAX} characters")
    if _CONTROL.search(value):
        raise ValueError("an edit-PC path must not contain NUL or control characters")
    drive, unc = _DRIVE.match(value), _UNC.match(value)
    if drive is None and unc is None:
        raise ValueError(
            "an edit-PC path must be absolute: X:/..., X:\\... or \\\\host\\share\\... "
            "(relative and drive-relative paths are refused)"
        )
    parts = re.split(r"[\\/]", (drive or unc).group("rest"))  # type: ignore[union-attr]
    if unc is not None and (len(parts) < 3 or parts[0] in ("?", ".")):
        raise ValueError("a UNC path needs \\\\host\\share\\file (device paths are refused)")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(
            "an edit-PC path must name a file, with no '.' or '..' segments and no empty "
            "segments (a doubled or trailing separator)"
        )
    return value


EditPath = Annotated[str, Field(min_length=1), AfterValidator(_edit_path)]


def path_key(path: str) -> str:
    """How two edit-PC paths are compared: ``\\`` as ``/``, ASCII case folded.

    Windows paths are case-insensitive. Only ASCII case folds, as in the S13f root bounds:
    under-folding is the safe direction (at worst a non-ASCII case variant is imported again
    and reported as unexpected), never a different file taken for an existing one.
    """
    return path.replace("\\", "/").translate(_ASCII_LOWER)


class ImportParams(OpModel):
    paths: list[EditPath] = Field(min_length=1, max_length=MAX_TARGETS)
    bin_path: list[BinName] = Field(
        min_length=1, max_length=8, description="Bin names from the media-pool root."
    )
    create_bins: bool = Field(default=True, strict=True)
    if_present: Literal["reuse", "error"] = Field(
        default="reuse",
        description="What to do with a path already in the bin (matched by 'File Path'): "
        "'reuse' reports it as present; 'error' refuses the whole request. Never re-imports.",
    )

    @model_validator(mode="after")
    def _once(self) -> ImportParams:
        _no_repeats([path_key(p) for p in self.paths], "path")
        return self


class ImportedMedia(OpModel):
    path: str
    status: Literal["imported", "present"]
    media: MediaRef
    file_path: str
    """The item's 'File Path' as Resolve reads it back."""


class ImportResult(OpModel):
    bin_path: tuple[str, ...]
    bins_created: list[str]
    media: list[ImportedMedia]
    notes: list[Note]


def _file_path(clip: Any) -> str:
    # [I] 'File Path' reads back the imported file's full path, in some Windows spelling
    # (separators and case may differ from the request): it is only ever compared by path_key.
    return str(clip.GetClipProperty("File Path") or "")


def _ensure_bin(pool: Any, path: tuple[str, ...], create: bool) -> tuple[Any, list[str]]:
    """The bin at ``path``, making the missing levels with ``AddSubFolder`` if ``create``."""
    folder = pool.GetRootFolder()
    if folder is None:
        raise OpError("no_root_folder", "MediaPool.GetRootFolder() returned None", "reopen it")
    created: list[str] = []
    for depth, name in enumerate(path):
        where = "/".join(path[: depth + 1])
        matches = [f for f in folder.GetSubFolderList() or [] if f.GetName() == name]
        if len(matches) > 1:
            raise OpError(
                "bin_ambiguous",
                f"{len(matches)} bins are named {where!r}",
                "rename one of them, or import into another bin",
                bins_created=created,
            )
        if not matches:
            if not create:
                raise OpError(
                    "bin_missing",
                    f"bin {where!r} does not exist",
                    "create it, or pass create_bins=true",
                    bin_path=list(path[: depth + 1]),
                )
            made = pool.AddSubFolder(folder, name)
            matches = [f for f in folder.GetSubFolderList() or [] if f.GetName() == name]
            if made is None or len(matches) != 1 or matches[0].GetUniqueId() != made.GetUniqueId():
                raise OpError(
                    "bin_create_failed",
                    f"AddSubFolder({name!r}) did not make bin {where!r}",
                    "create the bin by hand, then import again",
                    bins_created=created,
                )
            created.append(where)
        folder = matches[0]
    return folder, created


def _restore_folder(pool: Any, previous: Any) -> bool:
    """Make ``previous`` the current media-pool folder again; whether it is."""
    if previous is None:
        return True
    try:
        now = pool.GetCurrentFolder()
        if now is not None and now.GetUniqueId() == previous.GetUniqueId():
            return True
        back = pool.GetCurrentFolder() if pool.SetCurrentFolder(previous) else None
        return back is not None and back.GetUniqueId() == previous.GetUniqueId()
    except Exception:  # never mask the import's own outcome with a failed restore
        return False


def _import_into(pool: Any, folder: Any, paths: list[str]) -> tuple[list[Any], bool]:
    """``ImportMedia`` into ``folder``, which is made current for the call (the import goes
    to the current folder [S]); the folder that was current is put back in a ``finally``.
    Returns the items and whether the previous folder was restored."""
    previous = pool.GetCurrentFolder()
    items: list[Any] = []
    try:
        current = pool.GetCurrentFolder() if pool.SetCurrentFolder(folder) else None
        if current is None or current.GetUniqueId() != folder.GetUniqueId():
            raise OpError(
                "bin_select_failed",
                f"bin {folder.GetName()!r} did not become the current media-pool folder",
                "open the bin in the Media Pool by hand, then import again",
            )
        # [I] StartIndex/EndIndex apply to image sequences only and may be left out; only the
        # dict form is in the 21.1 stub, so the plain-string form is never used.
        items = list(pool.ImportMedia([{"FilePath": p} for p in paths]) or [])
    finally:
        restored = _restore_folder(pool, previous)
    return items, restored


@op(
    "media.import",
    params=ImportParams,
    result=ImportResult,
    mutates=True,
    traps=("rcm-input-colorspace",),
    guard_exemption=IMPORT_GUARD_EXEMPTION,
)
def media_import(env: OpEnv, params: ImportParams) -> ImportResult:
    """Import edit-PC media files into a media-pool bin, matched back by 'File Path'.

    A path already in the bin is never imported again: ``present`` (or ``media_present``
    under ``if_present="error"``). Existing items are never changed, moved or deleted.
    """
    pool = env.media_pool()
    where = tuple(params.bin_path)
    folder, bins_created = _ensure_bin(pool, where, params.create_bins)
    existing: dict[str, list[Any]] = {}
    for clip in folder.GetClipList() or []:
        key = path_key(_file_path(clip))
        if key:
            existing.setdefault(key, []).append(clip)
    present: list[ImportedMedia] = []
    missing: list[str] = []
    for path in params.paths:
        hits = existing.get(path_key(path), [])
        if len(hits) > 1:
            raise OpError(
                "media_ambiguous",
                f"bin {'/'.join(where)!r} holds {len(hits)} items for {path!r}",
                "remove the extra copy by hand (or move it to another bin), then import again",
                path=path,
                media_ids=[str(c.GetUniqueId()) for c in hits],
            )
        if hits:
            clip = hits[0]
            present.append(
                ImportedMedia(
                    path=path,
                    status="present",
                    media=MediaRef(unique_id=str(clip.GetUniqueId()), bin_path=where),
                    file_path=_file_path(clip),
                )
            )
        else:
            missing.append(path)
    if present and params.if_present == "error":
        raise OpError(
            "media_present",
            f"{len(present)} path(s) are already in bin {'/'.join(where)!r}; nothing was imported",
            "pass if_present='reuse' to use the existing items",
            present=[m.model_dump() for m in present],
        )
    if not missing:
        return ImportResult(bin_path=where, bins_created=bins_created, media=present, notes=[])
    try:
        returned, restored = _import_into(pool, folder, missing)
    except OpError as exc:
        raise _partial(exc, present, bins_created=bins_created) from exc
    imported, failed, unexpected = _match_imports(missing, returned)
    in_bin = {str(c.GetUniqueId()) for c in folder.GetClipList() or []}
    done = [
        *present,
        *(
            ImportedMedia(
                path=path,
                status="imported",
                media=MediaRef(unique_id=str(clip.GetUniqueId()), bin_path=where),
                file_path=_file_path(clip),
            )
            for path, clip in imported
        ),
    ]
    by_request = {path_key(m.path): m for m in done}
    done = [by_request[path_key(p)] for p in params.paths if path_key(p) in by_request]
    progress = {"bins_created": bins_created, "failed": failed}
    if unexpected:
        raise _partial(
            OpError(
                "import_unexpected",
                f"ImportMedia returned {len(unexpected)} item(s) that match no single requested "
                "path; they are left in the bin",
                "inspect the bin; remove the stray items by hand (nothing was deleted)",
                unexpected=unexpected,
            ),
            done,
            **progress,
        )
    misplaced = [str(c.GetUniqueId()) for _, c in imported if str(c.GetUniqueId()) not in in_bin]
    if misplaced:
        raise _partial(
            OpError(
                "import_unverified",
                f"{len(misplaced)} imported item(s) are not in bin {'/'.join(where)!r}",
                "find them with media.find and move them by hand",
                media_ids=misplaced,
            ),
            [m for m in done if m.media.unique_id not in misplaced],
            **progress,
        )
    if failed:
        raise _partial(
            OpError(
                "import_failed",
                f"{len(failed)} path(s) did not import: {failed}",
                "check each path exists on the edit PC and is a format Resolve reads; the "
                "paths that did import are in 'completed' and a rerun reuses them",
                paths=failed,
            ),
            done,
            bins_created=bins_created,
        )
    notes = [
        Note(
            code="input_color_space_default",
            message="imported media takes the project's input colour space default; set "
            "Rec.709 media with clip.set_input_color_space",
            trap_id="rcm-input-colorspace",
        )
    ]
    if not restored:
        notes.append(
            Note(
                code="folder_not_restored",
                message="the media-pool folder that was current could not be made current again",
            )
        )
    return ImportResult(bin_path=where, bins_created=bins_created, media=done, notes=notes)


def _match_imports(
    wanted: list[str], returned: list[Any]
) -> tuple[list[tuple[str, Any]], list[str], list[dict[str, str]]]:
    """Pair returned items with requested paths by 'File Path', never by position:
    ``(imported (path, item), failed paths, unexpected items)``.

    [I] ImportMedia leaves an unreadable path out silently and can return two items for one
    file, so the list's order and length say nothing about which path each item is.
    """
    keys = {path_key(p): p for p in wanted}
    by_key: dict[str, list[Any]] = {}
    for item in returned:
        by_key.setdefault(path_key(_file_path(item)), []).append(item)
    imported: list[tuple[str, Any]] = []
    unexpected: list[dict[str, str]] = []
    for key, items in by_key.items():
        if key in keys and len(items) == 1:
            imported.append((keys[key], items[0]))
            continue
        unexpected.extend(
            {"media_id": str(i.GetUniqueId()), "file_path": _file_path(i)} for i in items
        )
    failed = [p for k, p in keys.items() if k not in by_key]
    return imported, failed, unexpected


# -- track.ensure ------------------------------------------------------------------------------

AUDIO_FORMATS: tuple[str, ...] = (
    "mono",
    "stereo",
    "5.1",
    "5.1film",
    "7.1",
    "7.1film",
    *(f"adaptive{n}" for n in range(1, 37)),
)
"""[I] The audio channel formats ``AddTrack`` takes as a subtype (facts file)."""
AudioFormat = Literal[*AUDIO_FORMATS]  # type: ignore[valid-type]


class TrackSpec(OpModel):
    type: TrackType
    index: int = Field(strict=True, ge=1, le=16, description="1-based track index.")
    name: TrackName | None = None
    enabled: bool | None = Field(default=None, strict=True)
    locked: bool | None = Field(default=None, strict=True)
    audio_format: AudioFormat | None = Field(
        default=None,
        description="Audio only: the channel format the track must have (a format cannot be "
        "changed by script, so an existing track with another one is refused).",
    )

    @model_validator(mode="after")
    def _audio_only(self) -> TrackSpec:
        if self.audio_format is not None and self.type != "audio":
            raise ValueError("audio_format is for audio tracks only")
        return self


class TrackParams(OpModel):
    timeline: TimelineRef
    tracks: list[TrackSpec] = Field(min_length=1, max_length=MAX_TARGETS)

    @model_validator(mode="after")
    def _once(self) -> TrackParams:
        _no_repeats([(t.type, t.index) for t in self.tracks], "track")
        return self


class TrackState(OpModel):
    type: TrackType
    index: int
    created: bool
    name: str
    enabled: bool
    locked: bool
    audio_format: str | None
    changed: list[str]


class AddedTrack(OpModel):
    type: TrackType
    index: int
    audio_format: str | None


class TrackResult(OpModel):
    timeline: TimelineIdentity
    tracks: list[TrackState]
    added: list[AddedTrack]
    """Every track this call appended, including ones below a spec's index."""


def _track_count(timeline: Any, kind: str) -> int:
    return int(timeline.GetTrackCount(kind) or 0)


def _track_state(timeline: Any, spec: TrackSpec, created: bool, changed: list[str]) -> TrackState:
    kind, index = spec.type, spec.index
    return TrackState(
        type=kind,
        index=index,
        created=created,
        name=str(timeline.GetTrackName(kind, index) or ""),
        enabled=bool(timeline.GetIsTrackEnabled(kind, index)),
        locked=bool(timeline.GetIsTrackLocked(kind, index)),
        audio_format=str(timeline.GetTrackSubType(kind, index) or "") if kind == "audio" else None,
        changed=changed,
    )


def _writes_while_locked(timeline: Any, spec: TrackSpec) -> list[str]:
    """The fields ``spec`` would change on its (existing) track."""
    kind, index = spec.type, spec.index
    fields = []
    if spec.name is not None and str(timeline.GetTrackName(kind, index) or "") != spec.name:
        fields.append("name")
    if spec.enabled is not None and bool(timeline.GetIsTrackEnabled(kind, index)) != spec.enabled:
        fields.append("enabled")
    return fields


def _refuse_existing(timeline: Any, spec: TrackSpec) -> None:
    """Refusals for a track that exists, before anything is written."""
    kind, index = spec.type, spec.index
    if spec.audio_format is not None:
        # [I] GetTrackSubType returns the same format names AddTrack takes (AUDIO_FORMATS).
        have = str(timeline.GetTrackSubType(kind, index) or "")
        if have != spec.audio_format:
            raise OpError(
                "track_format_mismatch",
                f"audio track {index} is {have!r}, not {spec.audio_format!r}",
                "a track's channel format cannot be changed by script: use another track "
                "index, or change the format by hand",
                track=index,
                have=have,
                wanted=spec.audio_format,
            )
    if spec.locked is not False and bool(timeline.GetIsTrackLocked(kind, index)):
        fields = _writes_while_locked(timeline, spec)
        if fields:
            raise _locked(spec, fields)


def _locked(spec: TrackSpec, fields: list[str]) -> OpError:
    return OpError(
        "track_locked",
        f"{spec.type} track {spec.index} is locked; the spec would change {fields}",
        "pass locked=false in the same spec to unlock it first, or unlock it by hand",
        track_type=spec.type,
        track=spec.index,
        fields=fields,
    )


def _set_track(timeline: Any, spec: TrackSpec, field: str, value: Any) -> None:
    """One track setter, then its getter: a False return or a different read-back raises."""
    kind, index = spec.type, spec.index
    calls: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
        "name": (
            lambda: timeline.SetTrackName(kind, index, value),
            lambda: str(timeline.GetTrackName(kind, index) or ""),
        ),
        "enabled": (
            lambda: timeline.SetTrackEnable(kind, index, value),
            lambda: bool(timeline.GetIsTrackEnabled(kind, index)),
        ),
        "locked": (
            lambda: timeline.SetTrackLock(kind, index, value),
            lambda: bool(timeline.GetIsTrackLocked(kind, index)),
        ),
    }
    setter, getter = calls[field]
    if not setter():
        raise OpError(
            "track_set_failed",
            f"setting {field}={value!r} on {kind} track {index} returned False",
            f"check the track is not locked; {_JAM_HINT}",
            track_type=kind,
            track=index,
            field=field,
        )
    after = getter()
    if after != value:
        raise OpError(
            "track_not_applied",
            f"{kind} track {index}: {field} reads back {after!r} after setting {value!r}",
            "set it by hand in the timeline's track header",
            track_type=kind,
            track=index,
            field=field,
            wanted=value,
            actual=after,
        )


def _ensure_track(timeline: Any, spec: TrackSpec, added: list[AddedTrack]) -> TrackState:
    kind, index = spec.type, spec.index
    created = False
    count = _track_count(timeline, kind)
    while count < index:
        # Tracks are only ever appended, never deleted or reordered. The subtype is passed only
        # when there is one.
        args = (kind,) if spec.audio_format is None else (kind, spec.audio_format)
        ok = timeline.AddTrack(*args)
        after = _track_count(timeline, kind)
        got = str(timeline.GetTrackSubType(kind, count + 1) or "") if kind == "audio" else None
        if not ok or after != count + 1 or (spec.audio_format not in (None, got)):
            raise OpError(
                "track_add_failed",
                f"AddTrack{args} left {after} {kind} track(s) (was {count}), new track "
                f"format {got!r}",
                "add the track by hand, then run again",
                track_type=kind,
                track=count + 1,
                returned=bool(ok),
                audio_format=got,
            )
        count = after
        created = count == index
        added.append(AddedTrack(type=kind, index=count, audio_format=got))
    changed = ["created"] if created else []
    locked = bool(timeline.GetIsTrackLocked(kind, index))
    if locked and spec.locked is False:
        _set_track(timeline, spec, "locked", False)  # unlock first, then write
        changed.append("locked")
        locked = False
    fields = _writes_while_locked(timeline, spec) if locked else []
    if fields:
        raise _locked(spec, fields)
    if spec.name is not None and str(timeline.GetTrackName(kind, index) or "") != spec.name:
        _set_track(timeline, spec, "name", spec.name)
        changed.append("name")
    if spec.enabled is not None and bool(timeline.GetIsTrackEnabled(kind, index)) != spec.enabled:
        _set_track(timeline, spec, "enabled", spec.enabled)
        changed.append("enabled")
    if spec.locked is True and not locked:
        _set_track(timeline, spec, "locked", True)  # lock last, after the other writes
        changed.append("locked")
    return _track_state(timeline, spec, created, changed)


@op("track.ensure", params=TrackParams, result=TrackResult, mutates=True)
def track_ensure(env: OpEnv, params: TrackParams) -> TrackResult:
    """Make sure tracks exist (appending only) and set their name, enabled and lock, read back.

    There is no Fairlight mute, solo, bus or bounce call in 21.1: a disabled track is the
    nearest scriptable state to a mute.
    """
    timeline = env.timeline(params.timeline)
    for spec in params.tracks:
        if spec.index <= _track_count(timeline, spec.type):
            _refuse_existing(timeline, spec)
    added: list[AddedTrack] = []
    done: dict[tuple[str, int], TrackState] = {}
    # Lower indexes first, so a spec's own format is used for its track rather than the
    # fill-in format of a higher spec.
    try:
        for spec in sorted(params.tracks, key=lambda s: (s.type, s.index)):
            done[(spec.type, spec.index)] = _ensure_track(timeline, spec, added)
    except OpError as exc:
        finished = [done[(s.type, s.index)] for s in params.tracks if (s.type, s.index) in done]
        raise _partial(exc, finished, tracks_added=[a.model_dump() for a in added]) from exc
    return TrackResult(
        timeline=env.identity(timeline),
        tracks=[done[(s.type, s.index)] for s in params.tracks],
        added=added,
    )


# -- transition.add ----------------------------------------------------------------------------


class TransitionTarget(OpModel):
    item: ItemRef
    position: Literal["start", "end"]
    type: TransitionName = Field(
        default="Cross Dissolve", description="The transition's name, e.g. 'Cross Dissolve'."
    )
    category: Literal["simple", "audio"] = Field(
        default="simple", description="'simple' on a video item, 'audio' on an audio item."
    )
    alignment: Literal["left", "center", "right"] = Field(
        description="As on the Edit page: 'left' ends on the edit, 'center' straddles it, "
        "'right' starts on it."
    )
    frames: int = Field(strict=True, ge=1, le=240)

    @model_validator(mode="after")
    def _fits_item(self) -> TransitionTarget:
        if self.item.track_type not in ("video", "audio"):
            raise ValueError("a transition goes on a video or an audio item")
        if (self.category == "audio") != (self.item.track_type == "audio"):
            raise ValueError(
                "an 'audio' transition needs an audio item, a 'simple' one a video item"
            )
        return self


class TransitionParams(OpModel):
    timeline: TimelineRef
    targets: list[TransitionTarget] = Field(min_length=1, max_length=MAX_TARGETS)

    @model_validator(mode="after")
    def _once(self) -> TransitionParams:
        _no_repeats([(t.item.item_id, t.position) for t in self.targets], "item edge")
        return self


class HandleCheck(OpModel):
    side: Literal["outgoing", "incoming"]
    item_id: str
    needed: int
    available: int | None


class TransitionAdded(OpModel):
    item_id: str
    position: Literal["start", "end"]
    status: Literal["added", "present"]
    transition_id: str
    record_in: int
    record_out: int
    duration: int
    handles: list[HandleCheck]


class TransitionResult(OpModel):
    timeline: TimelineIdentity
    transitions: list[TransitionAdded]


def _transitions_at(timeline: Any, ref: ItemRef, edge: int) -> list[Any]:
    """Transition items on the item's track that touch the edit at ``edge``."""
    return [
        t
        for t in timeline.GetItemListInTrack(ref.track_type, ref.track) or []
        if t.GetType() == "transition" and int(t.GetStart()) <= edge <= int(t.GetEnd())
    ]


def _handles(timeline: Any, target: TransitionTarget, item: Any, edge: int) -> list[HandleCheck]:
    """What each clip at the edit needs and has, by the advisory rule: ``center`` needs
    ``frames // 2`` on each side, ``left``/``right`` all of it on one side.

    [I] The handle rules: the alignment meanings, that the part of the transition before
    the edit shows the incoming clip early (its left offset) and the part after it holds the
    outgoing clip on (its right offset), and that a side with no clip fades from or to black
    and needs nothing; and that GetLeftOffset/GetRightOffset count timeline frames. The
    read-back after AddTransition is the real check.
    """
    ref, uid = target.item, str(item.GetUniqueId())
    clips = [c for c in _clips(timeline, ref.track_type, ref.track) if str(c.GetUniqueId()) != uid]
    if target.position == "end":
        outgoing = item
        incoming = next((c for c in clips if int(c.GetStart()) == edge), None)
    else:
        outgoing = next((c for c in clips if int(c.GetEnd()) == edge), None)
        incoming = item
    f = target.frames
    need_out, need_in = {"center": (f // 2, f // 2), "left": (0, f), "right": (f, 0)}[
        target.alignment
    ]
    checks = []
    if outgoing is not None:
        checks.append(
            HandleCheck(
                side="outgoing",
                item_id=str(outgoing.GetUniqueId()),
                needed=need_out,
                available=_int(outgoing.GetRightOffset()),
            )
        )
    if incoming is not None:
        checks.append(
            HandleCheck(
                side="incoming",
                item_id=str(incoming.GetUniqueId()),
                needed=need_in,
                available=_int(incoming.GetLeftOffset()),
            )
        )
    return checks


def _present_or_conflict(
    timeline: Any, target: TransitionTarget, edge: int, start: int
) -> TransitionAdded | None:
    """The transition already at this edge with the asked length (``present``), ``None`` if
    there is none; a transition there of another length is never replaced."""
    found = _transitions_at(timeline, target.item, edge)
    if not found:
        return None
    if len(found) == 1 and int(found[0].GetDuration()) == target.frames:
        t = found[0]
        return TransitionAdded(
            item_id=target.item.item_id,
            position=target.position,
            status="present",
            transition_id=str(t.GetUniqueId()),
            record_in=int(t.GetStart()) - start,
            record_out=int(t.GetEnd()) - start,
            duration=int(t.GetDuration()),
            handles=[],
        )
    raise OpError(
        "transition_conflict",
        f"item {target.item.item_id} already has {len(found)} transition(s) at its "
        f"{target.position} ({[int(t.GetDuration()) for t in found]} frames), not one of "
        f"{target.frames}; it is never replaced",
        "remove or change the existing transition by hand, or ask for its length",
        existing=[
            {"transition_id": str(t.GetUniqueId()), "duration": int(t.GetDuration())} for t in found
        ],
    )


def _undo_transition(timeline: Any, ref: ItemRef, made: Any, reason: OpError) -> OpError:
    """Delete the transition this call has just made, whose read-back was wrong, and return
    ``reason`` to raise; ``transition_left_behind`` if it cannot be deleted. This is the only
    deletion in the build ops."""
    made_id = str(made.GetUniqueId())
    # [I] Timeline.DeleteClips deletes transition items too. Never a ripple delete.
    ok = timeline.DeleteClips([made], False)
    still = any(
        str(t.GetUniqueId()) == made_id
        for t in timeline.GetItemListInTrack(ref.track_type, ref.track) or []
    )
    if ok and not still:
        return reason
    extra = {k: v for k, v in reason.detail.items() if k not in ("fix_hint", "trap_id")}
    return OpError(
        "transition_left_behind",
        f"{reason.message}; deleting the new transition {made_id} failed, so it is still there",
        "delete that transition by hand (only it: nothing else was changed)",
        transition_id=made_id,
        cause=reason.code,
        **{k: v for k, v in extra.items() if k != "transition_id"},
    )


def _add_one(
    timeline: Any, target: TransitionTarget, item: Any, edge: int, start: int
) -> TransitionAdded:
    ref = target.item
    handles = _handles(timeline, target, item, edge)
    present = _present_or_conflict(timeline, target, edge, start)
    if present is not None:
        return present
    before = {
        str(t.GetUniqueId())
        for t in timeline.GetItemListInTrack(ref.track_type, ref.track) or []
        if t.GetType() == "transition"
    }
    made = item.AddTransition(
        {
            "type": target.type,
            "category": target.category,
            "position": target.position,
            "alignment": target.alignment,
            "duration": target.frames,
        }
    )
    if made is None:
        appeared = [str(t.GetUniqueId()) for t in _transitions_at(timeline, ref, edge)]
        appeared = [t for t in appeared if t not in before]
        raise OpError(
            "transition_refused",
            f"AddTransition({target.type!r}, {target.category}) at the {target.position} of "
            f"item {ref.item_id} returned None",
            "check the transition name and category, the item's handles and that the track "
            "is not locked",
            **({"possible_transitions": appeared} if appeared else {}),
        )
    made_id = str(made.GetUniqueId())
    on_track = any(
        str(t.GetUniqueId()) == made_id
        for t in timeline.GetItemListInTrack(ref.track_type, ref.track) or []
    )
    if str(made.GetType() or "") != "transition" or made_id in before or not on_track:
        # Not provably a transition this call made: never deleted.
        raise OpError(
            "transition_unverified",
            f"AddTransition returned {made.GetType()!r} item {made_id}, not a new transition "
            f"on {ref.track_type} track {ref.track}",
            "inspect the track with timeline.describe (nothing was deleted)",
            returned_id=made_id,
        )
    duration = int(made.GetDuration())
    t_start, t_end = int(made.GetStart()), int(made.GetEnd())
    if duration != target.frames:
        # [I] Too-short handles may give a shorter transition instead of None.
        raise _undo_transition(
            timeline,
            ref,
            made,
            OpError(
                "transition_short",
                f"the new transition at the {target.position} of item {ref.item_id} is "
                f"{duration} frames, not {target.frames}; it was deleted",
                "trim the source range so the clips have handles, or change the alignment",
                measured=duration,
                wanted=target.frames,
                handles=[h.model_dump() for h in handles],
            ),
        )
    if not t_start <= edge <= t_end:
        raise _undo_transition(
            timeline,
            ref,
            made,
            OpError(
                "transition_misplaced",
                f"the new transition spans [{t_start}, {t_end}), away from the edit at {edge}; "
                "it was deleted",
                "add it by hand on the Edit page",
                start=t_start,
                end=t_end,
                edge=edge,
            ),
        )
    return TransitionAdded(
        item_id=ref.item_id,
        position=target.position,
        status="added",
        transition_id=made_id,
        record_in=t_start - start,
        record_out=t_end - start,
        duration=duration,
        handles=handles,
    )


@op("transition.add", params=TransitionParams, result=TransitionResult, mutates=True)
def transition_add(env: OpEnv, params: TransitionParams) -> TransitionResult:
    """Add transitions at item edges, read back; R6 head/outro-only is plan.lint's and S13h2's.

    Policy: DESIGN R6 allows only the head and outro transitions. That rule is enforced by
    ``plan.lint`` and by the ``plan.apply`` compiler (S13h2), not by this op.

    An existing transition at an edge is never replaced: the same length is ``present``,
    another is ``transition_conflict``. A new transition whose read-back is wrong is deleted
    again (the only deletion the build ops make).
    """
    timeline = env.timeline(params.timeline)
    start = int(timeline.GetStartFrame())
    planned = []
    for target in params.targets:
        item = env.item(timeline, target.item)
        if str(item.GetType() or "") != target.item.track_type:
            raise OpError(
                "not_a_clip_item",
                f"item {target.item.item_id} is a {item.GetType()!r} item, not a "
                f"{target.item.track_type} clip",
                "put transitions on the video or audio clips at the edit",
            )
        edge = int(item.GetStart()) if target.position == "start" else int(item.GetEnd())
        _present_or_conflict(timeline, target, edge, start)
        short = [h for h in _handles(timeline, target, item, edge) if h.available is not None]
        short = [h for h in short if h.available < h.needed]
        if short and not _transitions_at(timeline, target.item, edge):
            raise OpError(
                "transition_needs_handles",
                f"a {target.alignment} transition of {target.frames} frames at the "
                f"{target.position} of item {target.item.item_id} needs "
                + ", ".join(
                    f"{h.needed} frames of {h.side} handle (has {h.available})" for h in short
                ),
                "trim the source range so the clip keeps handles, or change the alignment",
                needed={h.side: h.needed for h in short},
                available={h.side: h.available for h in short},
            )
        planned.append((target, item, edge))
    done: list[TransitionAdded] = []
    try:
        for target, item, edge in planned:
            done.append(_add_one(timeline, target, item, edge, start))
    except OpError as exc:
        raise _with_completed(exc, done) from exc
    return TransitionResult(timeline=env.identity(timeline), transitions=done)


# -- item.set_color ----------------------------------------------------------------------------

_NonNeg = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
_Pos = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]
NO_CDL_GETTER = "no-cdl-getter-21.1"


class CdlSpec(OpModel):
    """An ASC CDL with the fields and bounds of ``aar.plan.model.Cdl``: per channel
    ``out = (in * slope + offset) ** power``, then saturation; omitted parts are identity."""

    slope: tuple[_NonNeg, _NonNeg, _NonNeg] = (1.0, 1.0, 1.0)
    offset: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.0, 0.0)
    power: tuple[_Pos, _Pos, _Pos] = (1.0, 1.0, 1.0)
    saturation: _NonNeg = 1.0


class ColorTarget(OpModel):
    item: ItemRef
    node: int = Field(default=1, strict=True, ge=1, le=32, description="1-based node index.")
    cdl: CdlSpec | None = None
    lut: LutPath | None = Field(default=None, description="A LUT path as Resolve lists it.")

    @model_validator(mode="after")
    def _something_on_video(self) -> ColorTarget:
        if self.item.track_type != "video":
            raise ValueError("colour goes on video items")
        if self.cdl is None and self.lut is None:
            raise ValueError("give a cdl, a lut or both")
        return self


class ColorParams(OpModel):
    timeline: TimelineRef
    targets: list[ColorTarget] = Field(min_length=1, max_length=MAX_TARGETS)

    @model_validator(mode="after")
    def _once(self) -> ColorParams:
        _no_repeats([(t.item.item_id, t.node) for t in self.targets], "item node")
        return self


class CdlSent(OpModel):
    verified: bool
    reason: str
    sent: dict[str, Any]


class LutApplied(OpModel):
    status: Literal["set", "present"]
    requested: str
    read_back: str


class ColorApplied(OpModel):
    item_id: str
    node: int
    nodes: int
    status: Literal["applied", "present"]
    cdl: CdlSent | None
    lut: LutApplied | None
    page_bounced: bool


class ColorResult(OpModel):
    timeline: TimelineIdentity
    items: list[ColorApplied]
    notes: list[Note]


def _triple(values: tuple[float, float, float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def cdl_payload(node: int, cdl: CdlSpec) -> dict[str, Any]:
    """The ``SetCDL`` argument, formatted deterministically."""
    return {
        "NodeIndex": node,
        "Slope": _triple(cdl.slope),
        "Offset": _triple(cdl.offset),
        "Power": _triple(cdl.power),
        "Saturation": float(cdl.saturation),
    }


def _lut_parts(path: str) -> tuple[bool, list[str]]:
    absolute = bool(_DRIVE.match(path) or _UNC.match(path) or path.startswith(("/", "\\")))
    return absolute, [p for p in path_key(path).split("/") if p]


def same_lut(read_back: str, wanted: str) -> bool:
    """Whether ``GetLUT``'s read-back is the LUT asked for.

    ``GetLUT`` returns a path relative to the LUT folder [S] (which folder is [I]), while the
    request may be absolute. Separators and ASCII case are normalised; a relative request must
    equal the read-back, an absolute one must end with all of the read-back's components.
    """
    have_abs, have = _lut_parts(read_back)
    want_abs, want = _lut_parts(wanted)
    if not have or not want:
        return False
    if want_abs and not have_abs:
        return len(have) < len(want) and want[-len(have) :] == have
    return have == want


def _colour_one(
    env: OpEnv, target: ColorTarget, item: Any, graph: Any, nodes: int, notes: list[Note]
) -> ColorApplied:
    where = f"item {target.item.item_id}, node {target.node}"
    bounced = False
    cdl: CdlSent | None = None
    if target.cdl is not None:
        sent = cdl_payload(target.node, target.cdl)
        # No present check for the CDL: 21.1 has no CDL getter to compare with, and SetCDL sets
        # absolute values, so writing the same CDL again is harmless. [I] SetCDL may jam after
        # the Color page like the other setters, so it gets the same bounce and one retry.
        ok, was_bounced = _with_bounce(env, lambda: item.SetCDL(sent))
        if was_bounced:
            bounced = True
            notes.append(_bounce_note("SetCDL", where))
        if not ok:
            raise OpError(
                "cdl_refused",
                f"SetCDL on {where} returned False, also after a page bounce",
                f"check the node exists and the values are in range; {_JAM_HINT}",
                "setproperty-after-color",
                sent=sent,
            )
        cdl = CdlSent(verified=False, reason=NO_CDL_GETTER, sent=sent)
    lut: LutApplied | None = None
    if target.lut is not None:
        have = str(graph.GetLUT(target.node) or "")
        if same_lut(have, target.lut):
            lut = LutApplied(status="present", requested=target.lut, read_back=have)
        else:
            ok, was_bounced = _with_bounce(env, lambda: graph.SetLUT(target.node, target.lut))
            if was_bounced:
                bounced = True
                notes.append(_bounce_note("SetLUT", where))
            if not ok:
                raise OpError(
                    "lut_refused",
                    f"SetLUT({target.node}, {target.lut!r}) on {where} returned False, also "
                    "after a page bounce",
                    "give the LUT path as Resolve lists it and refresh the LUT list "
                    f"(Project.RefreshLUTList) so Resolve knows it; {_JAM_HINT}",
                    "setproperty-after-color",
                )
            after = str(graph.GetLUT(target.node) or "")
            if not same_lut(after, target.lut):
                raise OpError(
                    "lut_not_applied",
                    f"{where} reads back LUT {after!r} after SetLUT({target.lut!r})",
                    "set the LUT by hand on the Color page",
                    "setproperty-after-color",
                    read_back=after,
                )
            lut = LutApplied(status="set", requested=target.lut, read_back=after)
    status: Literal["applied", "present"] = (
        "present" if cdl is None and lut is not None and lut.status == "present" else "applied"
    )
    return ColorApplied(
        item_id=target.item.item_id,
        node=target.node,
        nodes=nodes,
        status=status,
        cdl=cdl,
        lut=lut,
        page_bounced=bounced,
    )


@op(
    "item.set_color",
    params=ColorParams,
    result=ColorResult,
    mutates=True,
    traps=("setproperty-after-color", "append-ungraded"),
)
def item_set_color(env: OpEnv, params: ColorParams) -> ColorResult:
    """Set a CDL and/or a LUT on a node of video items; the LUT is read back, the CDL cannot be.

    21.1 has no CDL getter, so the result says ``verified: false`` for the CDL. There is no
    add-node call: the node must exist. ``grades.copy`` replaces the whole node graph (trap
    ``append-ungraded``), so set colour after copying grades, not before.
    """
    timeline = env.timeline(params.timeline)
    planned = []
    for target in params.targets:
        item = env.item(timeline, target.item)
        graph = item.GetNodeGraph()
        nodes = int(graph.GetNumNodes() or 0) if graph is not None else 0
        if target.node > nodes:
            raise OpError(
                "node_out_of_range",
                f"item {target.item.item_id} has {nodes} node(s); node {target.node} does not "
                "exist",
                "21.1 has no call that adds a node: use a node that exists, or add nodes to "
                "the grade by hand",
                nodes=nodes,
                node=target.node,
            )
        planned.append((target, item, graph, nodes))
    done: list[ColorApplied] = []
    notes: list[Note] = []
    try:
        for target, item, graph, nodes in planned:
            done.append(_colour_one(env, target, item, graph, nodes, notes))
    except OpError as exc:
        raise _with_completed(exc, done) from exc
    return ColorResult(timeline=env.identity(timeline), items=done, notes=notes)


# -- item.set_transform ------------------------------------------------------------------------

FLOAT_TOLERANCE = 1e-3


class TransformTarget(OpModel):
    """A static Inspector transform, with the meaning of ``aar.plan.model.Transform``."""

    item: ItemRef
    zoom: float | None = Field(
        default=None, strict=True, allow_inf_nan=False, gt=0, le=100, description="ZoomX/ZoomY."
    )
    pan: FiniteFloat | None = Field(
        default=None, description="Pixels at the timeline resolution; |pan| <= 4 x width."
    )
    tilt: FiniteFloat | None = Field(
        default=None, description="Pixels at the timeline resolution; |tilt| <= 4 x height."
    )
    rotation_deg: float | None = Field(
        default=None, strict=True, allow_inf_nan=False, ge=-360, le=360
    )
    flip_h: bool | None = Field(default=None, strict=True, description="FlipX.")
    flip_v: bool | None = Field(default=None, strict=True, description="FlipY.")

    @model_validator(mode="after")
    def _something_on_video(self) -> TransformTarget:
        if self.item.track_type != "video":
            raise ValueError("a transform goes on a video item")
        if all(v is None for v in self.wanted().values()):
            raise ValueError("give at least one of zoom, pan, tilt, rotation_deg, flip_h, flip_v")
        return self

    def wanted(self) -> dict[str, float | bool | None]:
        """The Resolve properties this target sets (``None``: left alone)."""
        return {
            "ZoomX": self.zoom,
            "ZoomY": self.zoom,
            "Pan": self.pan,
            "Tilt": self.tilt,
            "RotationAngle": self.rotation_deg,
            "FlipX": self.flip_h,
            "FlipY": self.flip_v,
        }


class TransformParams(OpModel):
    timeline: TimelineRef
    targets: list[TransformTarget] = Field(min_length=1, max_length=MAX_TARGETS)

    @model_validator(mode="after")
    def _once(self) -> TransformParams:
        _no_repeats([t.item.item_id for t in self.targets], "item")
        return self


class TransformApplied(OpModel):
    item_id: str
    status: Literal["applied", "present"]
    values: dict[str, float | bool]
    """Every property the target names, as read back."""
    changed: list[str]
    page_bounced: bool


class TransformResult(OpModel):
    timeline: TimelineIdentity
    items: list[TransformApplied]
    notes: list[Note]


def _same_value(have: Any, want: float | bool) -> bool:
    """Floats within :data:`FLOAT_TOLERANCE`, bools exactly ([I]: FlipX/FlipY read back as
    bools; any other type fails closed)."""
    if isinstance(want, bool):
        return isinstance(have, bool) and have is want
    if isinstance(have, bool) or not isinstance(have, int | float):
        return False
    return abs(float(have) - want) <= FLOAT_TOLERANCE


def _resolution(timeline: Any) -> tuple[int, int]:
    width = _int(timeline.GetSetting("timelineResolutionWidth"))
    height = _int(timeline.GetSetting("timelineResolutionHeight"))
    if not width or not height or width <= 0 or height <= 0:
        raise OpError(
            "timeline_resolution_unknown",
            f"the timeline resolution reads {width}x{height}",
            "check the timeline settings",
        )
    return width, height


@op(
    "item.set_transform",
    params=TransformParams,
    result=TransformResult,
    mutates=True,
    traps=("setproperty-after-color",),
)
def item_set_transform(env: OpEnv, params: TransformParams) -> TransformResult:
    """Set zoom, pan, tilt, rotation and flips on video items, read back per property."""
    timeline = env.timeline(params.timeline)
    if any(t.pan is not None or t.tilt is not None for t in params.targets):
        width, height = _resolution(timeline)
        for t in params.targets:
            for axis, value, size, what in (
                ("pan", t.pan, width, "width"),
                ("tilt", t.tilt, height, "height"),
            ):
                if value is not None and abs(value) > 4 * size:
                    raise OpError(
                        "transform_out_of_range",
                        f"item {t.item.item_id}: {axis} {value} is beyond 4 x the timeline "
                        f"{what} ({4 * size})",
                        f"keep |{axis}| within 4 x {size} pixels",
                        axis=axis,
                        value=value,
                        limit=4 * size,
                    )
    planned = []
    for target in params.targets:
        item = env.item(timeline, target.item)
        wanted = {k: v for k, v in target.wanted().items() if v is not None}
        before = {k: item.GetProperty(k) for k in wanted}
        planned.append((target, item, wanted, before))
    done: list[TransformApplied] = []
    notes: list[Note] = []
    for target, item, wanted, before in planned:
        todo = {k: v for k, v in wanted.items() if not _same_value(before[k], v)}
        bounced = False
        after = before
        if todo:
            try:
                bounced = _set_properties(env, item, todo)
            except OpError as exc:
                raise _with_completed(exc, done) from exc
            if bounced:
                notes.append(_bounce_note("SetProperties", f"item {target.item.item_id}"))
            after = {k: item.GetProperty(k) for k in wanted}
            bad = [k for k, v in wanted.items() if not _same_value(after[k], v)]
            if bad:
                raise _with_completed(
                    OpError(
                        "property_not_applied",
                        f"item {target.item.item_id}: {bad[0]} reads back {after[bad[0]]!r}, "
                        f"wanted {wanted[bad[0]]!r}",
                        "set it by hand in the Inspector",
                        "setproperty-after-color",
                        property=bad[0],
                        properties=bad,
                    ),
                    done,
                )
        done.append(
            TransformApplied(
                item_id=target.item.item_id,
                status="applied" if todo else "present",
                values=after,
                changed=sorted(todo),
                page_bounced=bounced,
            )
        )
    return TransformResult(timeline=env.identity(timeline), items=done, notes=notes)


# -- title.place -------------------------------------------------------------------------------

MAX_RECORD_FRAME = 10_000_000


class TitleSpec(OpModel):
    template: MediaRef = Field(
        description="A media-pool item holding a Text+ title (a Fusion composition clip)."
    )
    track: int = Field(strict=True, ge=1, le=16, description="Video track.")
    record_in: int = Field(
        strict=True, ge=0, le=MAX_RECORD_FRAME, description="Frames from the timeline start."
    )
    record_out: int = Field(
        strict=True, ge=1, le=MAX_RECORD_FRAME, description="End frame (exclusive)."
    )
    lines: list[TitleLine] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _ordered(self) -> TitleSpec:
        if self.record_out <= self.record_in:
            raise ValueError("record_out must be greater than record_in (ranges are [in, out))")
        return self


class TitleParams(OpModel):
    timeline: TimelineRef
    titles: list[TitleSpec] = Field(min_length=1, max_length=MAX_TARGETS)

    @model_validator(mode="after")
    def _apart(self) -> TitleParams:
        _no_repeats([(t.track, t.record_in) for t in self.titles], "title slot")
        spans = sorted((t.track, t.record_in, t.record_out) for t in self.titles)
        for (track, _, end), (next_track, start, _) in itertools.pairwise(spans):
            if track == next_track and start < end:
                raise ValueError(f"two titles overlap on video track {track}")
        return self


class TitlePlaced(OpModel):
    status: Literal["placed", "present", "updated"]
    item: ItemRef
    record_in: int
    record_out: int
    text: str


class TitleResult(OpModel):
    timeline: TimelineIdentity
    titles: list[TitlePlaced]


def _title_text(item: Any, text: str) -> tuple[bool, str]:
    """Set the Text+ ``StyledText`` of the title item's first comp: ``(changed, read back)``."""
    uid = str(item.GetUniqueId())
    comp = item.GetFusionCompByIndex(1)  # [I] the template's Text+ lives in its first comp
    if comp is None:
        raise OpError(
            "title_has_no_comp",
            f"title item {uid} has no Fusion composition",
            "use a Fusion title template (a Fusion composition clip holding a Text+)",
            item_id=uid,
        )
    tool = comp.FindToolByID("TextPlus")  # [I] the Text+ registry id is "TextPlus"
    if tool is None:
        raise OpError(
            "title_has_no_text_tool",
            f"the composition of title item {uid} has no Text+ tool",
            "use a template whose composition holds a Text+ tool",
            item_id=uid,
        )
    have = tool.GetInput("StyledText")  # [I] the Text+ text input is "StyledText"
    if have == text:
        return False, text
    # [I] With no time, SetInput sets the constant value of an input that is not animated.
    # Its return value is not relied on: only the read-back counts.
    tool.SetInput("StyledText", text)
    after = tool.GetInput("StyledText")
    if after != text:
        raise OpError(
            "title_text_not_applied",
            f"title item {uid} reads back text {after!r}, not {text!r}",
            "set the text by hand in the Inspector; the item stays in place and a rerun reuses it",
            item_id=uid,
            read_back=after,
        )
    return True, after


@op(
    "title.place",
    params=TitleParams,
    result=TitleResult,
    mutates=True,
    needs_current_timeline=True,
    traps=("append-endframe-exclusive",),
)
def title_place(env: OpEnv, params: TitleParams) -> TitleResult:
    """Place Text+ title templates at exact record ranges and set their text, read back.

    ``InsertTitleIntoTimeline`` / ``InsertFusionTitleIntoTimeline`` land at the playhead with
    a default length that nothing can change afterwards, so the template is placed with
    ``AppendToTimeline`` (the ``timeline.append_ranges`` machinery). A range that holds
    anything else is refused, never overwritten; the same template exactly there is reused.
    """
    timeline = env.current_timeline(params.timeline)
    tl_fps = (timeline.GetSettings() or {}).get("timelineFrameRate")
    if tl_fps is None:
        raise OpError("timeline_fps_unknown", "timelineFrameRate is not set", "check settings")
    start = int(timeline.GetStartFrame())
    found = env.media([t.template for t in params.titles])
    facts = {uid: raw.read_media(mpi) for uid, (mpi, _) in found.items()}
    planned = []
    for index, spec in enumerate(params.titles):
        uid = spec.template.unique_id
        f, length = facts[uid], spec.record_out - spec.record_in
        if f["fps"] is None or exact_rate(float(f["fps"])) != exact_rate(float(tl_fps)):
            raise OpError(
                "template_rate_mismatch",
                f"title {index}: template {f['name']!r} runs at {f['fps']} fps, the timeline at "
                f"{tl_fps}",
                "use a title template made at the timeline's frame rate",
            )
        if f["frames"] is None or f["frames"] < length:
            raise OpError(
                "template_too_short",
                f"title {index}: template {f['name']!r} has {f['frames']} frames, the range "
                f"needs {length}",
                "use a longer template, or a shorter range",
                needed=length,
                available=f["frames"],
            )
        lo, hi = start + spec.record_in, start + spec.record_out
        occupying = _overlapping(timeline, "video", spec.track, lo, hi)
        reuse = None
        if occupying:
            only = occupying[0]
            exact = int(only.GetStart()) == lo and int(only.GetEnd()) == hi
            if len(occupying) == 1 and exact and _media_id(only) == uid:
                reuse = only
            else:
                raise OpError(
                    "track_occupied",
                    f"title {index}: video track {spec.track} already has {len(occupying)} "
                    f"item(s) in [{spec.record_in}, {spec.record_out}); a title never "
                    "overwrites",
                    "pick a free range or another track (track.ensure adds one)",
                    occupying=[str(i.GetUniqueId()) for i in occupying],
                )
        rng = AppendRange(
            media=spec.template,
            src_in=0,
            src_out=length,
            track=spec.track,
            media_type="video",
            record_in=spec.record_in,
        )
        p = _Planned(index, rng, "video", spec.record_in, length)
        if reuse is None:
            _preflight(timeline, p, f)
        planned.append((spec, p, reuse))
    # repair_short off: equal rates cannot land short, and a title never deletes an item.
    # [I] AppendToTimeline places a Fusion composition clip with its exact range like any
    # other media-pool item, and the timeline item carries a copy of the template's comp.
    appender = _Appender(env, timeline, repair_short=False)
    done: list[TitlePlaced] = []
    for spec, p, reuse in planned:
        text = "\n".join(spec.lines)  # [I] StyledText separates lines with "\n"
        placed_id = None
        try:
            if reuse is None:
                uid = spec.template.unique_id
                appended = appender.run(p, found[uid][0], facts[uid])
                if appended.video is None:  # pragma: no cover - a video append places video
                    raise OpError("append_failed", "no video item was placed", "retry")
                placed_id = appended.video.item_id
                item = env.item(
                    timeline, ItemRef(track_type="video", track=spec.track, item_id=placed_id)
                )
            else:
                item = reuse
            changed, read_back = _title_text(item, text)
        except OpError as exc:
            more = {"placed_item_id": placed_id} if placed_id is not None else {}
            raise _partial(exc, done, **more) from exc
        status: Literal["placed", "present", "updated"] = (
            "placed" if reuse is None else ("updated" if changed else "present")
        )
        done.append(
            TitlePlaced(
                status=status,
                item=ItemRef(track_type="video", track=spec.track, item_id=str(item.GetUniqueId())),
                record_in=int(item.GetStart()) - start,
                record_out=int(item.GetEnd()) - start,
                text=read_back,
            )
        )
    return TitleResult(timeline=env.identity(timeline), titles=done)
