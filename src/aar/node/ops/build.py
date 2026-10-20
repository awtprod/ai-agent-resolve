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

import re
import string
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, model_validator

from aar.node.ops.framework import (
    MediaRef,
    Note,
    OpEnv,
    OpError,
    OpModel,
    op,
)
from aar.node.ops.write import (
    _no_repeats,
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
    ``(imported (path, item), failed paths, unexpected items)``."""
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
