"""Write a new project: a copy of the seed changed by the closed list of operations.

:func:`build_project` takes a private copy of the seed (made by the caller in the staging
folder), the checked :class:`~aar.drp.libwrite.seed.Seed`, the media clips to add and the items
of the timeline, and changes the copy's rows in one SQLite transaction. Every change is one of
:data:`aar.drp.libwrite.ops.OPERATIONS`; the result names the ones used. The file stays in
rollback-journal mode, as Resolve keeps it, and no journal is left beside it.
"""

from __future__ import annotations

import sqlite3
import struct
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aar.drp.items.author import AuthoredItem
from aar.drp.items.rows import links_of
from aar.drp.library import SQL_ERRORS
from aar.drp.libwrite.errors import ProjectWriteError, refusing
from aar.drp.libwrite.seed import Seed
from aar.drp.mediapool.author import AuthoredClip
from aar.drp.mediapool.templates import ClipRows

__all__ = ["Built", "ItemSpec", "build_project", "operations_for", "sequence_length"]


@dataclass(frozen=True, order=True)
class ItemSpec:
    """One item of the cut, as the self-check reads it back: its track (``video``/``audio``),
    record frames counted from the timeline's start, source in and the file's edit-PC path."""

    track: str
    record_in: int
    record_out: int
    src_in: int
    path: str


@dataclass(frozen=True)
class Built:
    path: Path
    project_id: str
    operations: tuple[str, ...]


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@refusing("bad_input", "computing the sequence length", "this is a caller bug; report it")
def sequence_length(rate: float, start_seconds: float, ends: Sequence[int]) -> float:
    """``MediaExtents``' length (Resolve's formula, checked on 15 real sequences): the end of
    the last item, counted from frame 0, over the rate as a double, minus the start in
    seconds; 0 for an empty timeline."""
    if not ends:
        return 0.0
    return max(ends) / rate - start_seconds


@refusing("bad_input", "listing the operations", "this is a caller bug; report it")
def operations_for(
    seed: Seed, timeline_name: str | None, items: Sequence[AuthoredItem], *, media: bool
) -> list[str]:
    """The operations a project of these media and items uses, in the order they run."""
    ops = ["new_project_identity"]
    if timeline_name is not None and timeline_name != seed.timeline_name:
        ops.append("rename_timeline")
    if seed.items:
        ops.append("remove_items")
    if media:
        ops.append("add_media")
    by_id = {i.item_id: i for i in items}
    linked = picture_only = sound_only = split = False
    for item in items:
        if item.kind == "transition":
            continue
        partners = _partners(item)
        if not partners:
            if item.kind == "picture":
                picture_only = True
            else:
                sound_only = True
            continue
        linked = True
        for partner in partners:
            other = by_id.get(partner)
            if other is not None and any(
                other.item.get(c) != item.item.get(c) for c in ("Start", "Duration", "In")
            ):
                split = True
    if linked:
        ops.append("add_linked_items")
    if picture_only:
        ops.append("add_picture_only_items")
    if sound_only:
        ops.append("add_sound_only_items")
    if _gaps(seed, items):
        ops.append("leave_gaps")
    if split:
        ops.append("split_edit")
    if any(i.kind == "transition" for i in items):
        ops.append("add_sound_transitions")
    ops.append("set_extents")
    return ops


def _partners(item: AuthoredItem) -> tuple[str, ...]:
    return links_of(item.item)


def _gaps(seed: Seed, items: Sequence[AuthoredItem]) -> bool:
    for track in (seed.video_track, seed.audio_track):
        spans = sorted(
            (int(i.item["Start"]), int(i.item["Start"]) + int(i.item["Duration"]))
            for i in items
            if i.link["DbOwner"] == track and i.kind != "transition"
        )
        cursor = seed.start_frame
        for start, end in spans:
            if start != cursor:
                return True
            cursor = end
    return False


@refusing("bad_input", "writing the project's rows", "this is a caller bug; report it")
def build_project(
    staged: Path,
    seed: Seed,
    *,
    project_name: str,
    timeline_name: str | None,
    clips: Sequence[AuthoredClip],
    items: Sequence[AuthoredItem],
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> Built:
    """Change ``staged`` (a private copy of ``seed``) into the new project; see the module
    docstring. Raises :class:`ProjectWriteError` (``bad_input``, ``library_unwritable``)."""
    ops = operations_for(seed, timeline_name, items, media=bool(clips))
    project_id = new_id()
    if project_id in seed.taken:
        raise ProjectWriteError("bad_input", "the new project id is already used", "try again")
    _check_inputs(seed, clips, items)
    try:
        conn = sqlite3.connect(staged, isolation_level=None)
    except SQL_ERRORS as exc:
        raise _sql_refusal(exc, f"cannot open the staged copy {staged}") from None
    try:
        conn.execute("PRAGMA trusted_schema = OFF")
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "delete":
            raise ProjectWriteError(
                "bad_input",
                f"the seed copy is in journal mode {mode!r}, not Resolve's 'delete'",
                "use a seed saved by Resolve in a disk library",
            )
        conn.execute("BEGIN IMMEDIATE")
        _new_identity(conn, seed, project_id, project_name)
        if "rename_timeline" in ops:
            assert timeline_name is not None
            conn.execute(
                "UPDATE Sm2Timeline SET Name = ? WHERE Sm2Timeline_id = ?",
                (timeline_name, seed.timeline_id),
            )
            if seed.timeline_clip_id is not None:
                conn.execute(
                    "UPDATE Sm2MpMedia SET Name = ? WHERE Sm2MpMedia_id = ?",
                    (timeline_name, seed.timeline_clip_id),
                )
        if seed.items:
            marks = ", ".join("?" for _ in seed.items)
            conn.execute(f"DELETE FROM Sm2TiItem WHERE Sm2TiItem_id IN ({marks})", seed.items)
            conn.execute(
                f"DELETE FROM Sm2TiItem_Sm2TiTrack WHERE DbAssociate IN ({marks})", seed.items
            )
        for clip in clips:
            for table, rows in clip.rows.items():
                for row in rows:
                    _insert(conn, seed, table, row)
        for item in items:
            _insert(conn, seed, "Sm2TiItem", item.item)
            _insert(conn, seed, "Sm2TiItem_Sm2TiTrack", item.link)
        ends = [int(i.item["Start"]) + int(i.item["Duration"]) for i in items]
        rate = float(seed.rate)
        length = sequence_length(rate, seed.start_seconds, ends)
        conn.execute(
            "UPDATE Sm2Sequence SET MediaExtents = ? WHERE Sm2Sequence_id = ?",
            (struct.pack("<dd", seed.start_seconds, length), seed.sequence_id),
        )
        conn.execute("COMMIT")
        check = conn.execute("PRAGMA integrity_check").fetchall()
        if [str(r[0]) for r in check] != ["ok"]:
            raise ProjectWriteError(
                "self_check_failed",
                f"the written project fails SQLite's integrity check: {check[0][0]}",
                "this is a writer bug; report the seed",
            )
    except SQL_ERRORS as exc:
        raise _sql_refusal(exc, "writing the staged project failed") from None
    finally:
        conn.close()
    for side in ("-journal", "-wal", "-shm"):
        leftover = staged.with_name(staged.name + side)
        if leftover.exists():
            raise ProjectWriteError(
                "self_check_failed",
                f"{leftover.name} was left beside the written project",
                "this is a writer bug; report it",
            )
    return Built(staged, project_id, tuple(ops))


#: SQLite's error names that say the library (its disk, its share, its permissions) failed,
#: not the seed: anything else SQLite refuses while the copy's rows change is the seed's.
_STORAGE_ERRORS = (
    "SQLITE_FULL",
    "SQLITE_IOERR",
    "SQLITE_CANTOPEN",
    "SQLITE_READONLY",
    "SQLITE_BUSY",
    "SQLITE_LOCKED",
    "SQLITE_PERM",
    "SQLITE_NOLFS",
    "SQLITE_NOMEM",
)


#: SQLite's messages for those, for an error that carries no error name.
_STORAGE_TEXT = ("disk", "full", "i/o", "readonly", "read-only", "unable to open", "locked", "busy")


def _sql_refusal(exc: Exception, what: str) -> ProjectWriteError:
    """A storage failure is ``library_unwritable``; any other error SQLite raises on the copy
    (a constraint, a missing column, damage) is the seed's (``bad_seed``)."""
    name = str(getattr(exc, "sqlite_errorname", "") or "")
    text = str(exc).lower()
    storage = name.startswith(_STORAGE_ERRORS) or (
        not name and any(word in text for word in _STORAGE_TEXT)
    )
    if isinstance(exc, sqlite3.Error) and storage:
        return ProjectWriteError(
            "library_unwritable",
            f"{what}: {exc} ({name})",
            "check that the library is writable and has room",
        )
    return ProjectWriteError(
        "bad_seed",
        f"{what}: SQLite refused a change of the seed's copy ({type(exc).__name__}: {exc}"
        + (f", {name})" if name else ")"),
        "use a seed Resolve saved in a disk library",
    )


def _new_identity(conn: sqlite3.Connection, seed: Seed, project_id: str, name: str) -> None:
    """``new_project_identity``: the seed's project id, which occurs only in plain text columns,
    replaced in every one of them; and the project's new name."""
    for table, columns in seed.columns.items():
        for column in columns:
            conn.execute(
                f"UPDATE {_q(table)} SET {_q(column)} = ? WHERE typeof({_q(column)}) = 'text' "
                f"AND {_q(column)} = ?",
                (project_id, seed.project_id),
            )
    conn.execute("UPDATE SM_Project SET ProjectName = ?", (name,))


def _insert(conn: sqlite3.Connection, seed: Seed, table: str, row: dict[str, Any]) -> None:
    columns = seed.columns.get(table)
    if columns is None:
        raise ProjectWriteError(
            "bad_input", f"the seed has no {table} table", "use a seed saved by Resolve 21.1"
        )
    unknown = [c for c in row if c not in columns]
    if unknown:
        raise ProjectWriteError(
            "bad_input",
            f"the seed's {table} table has no column {unknown[0]} that a template row has",
            "use templates and a seed saved by the same Resolve version",
        )
    names = list(row)
    conn.execute(
        f"INSERT INTO {_q(table)} ({', '.join(_q(c) for c in names)}) "
        f"VALUES ({', '.join('?' for _ in names)})",
        [row[c] for c in names],
    )


def _check_inputs(seed: Seed, clips: Sequence[AuthoredClip], items: Sequence[AuthoredItem]) -> None:
    def bad(why: str) -> ProjectWriteError:
        return ProjectWriteError("bad_input", why, "this is a caller bug; report it")

    new_ids: set[str] = set()
    for clip in clips:
        ids = ClipRows.from_tables(clip.rows).ids()
        if ids & seed.taken or ids & new_ids:
            raise bad(
                f"a new media clip's id is already used ({sorted(ids & (seed.taken | new_ids))[0]})"
            )
        new_ids |= ids
    for item in items:
        if item.item_id in seed.taken or item.item_id in new_ids:
            raise bad(f"the item id {item.item_id} is already used")
        new_ids.add(item.item_id)
        if item.link["DbOwner"] not in (seed.video_track, seed.audio_track):
            raise bad("an item is placed on a track that is not the seed timeline's")
        if item.kind not in ("picture", "sound", "transition"):
            raise bad(f"an item is a {item.kind!r}")
        want = seed.video_track if item.kind == "picture" else seed.audio_track
        if item.link["DbOwner"] != want:
            raise bad(f"a {item.kind} item is placed on the other track")
    for track in (seed.video_track, seed.audio_track):
        # Record order, a transition before a clip that starts on the same frame (Resolve's).
        ordered = sorted(
            (
                int(i.item["Start"]),
                0 if i.kind == "transition" else 1,
                int(i.item["Start"]) + int(i.item["Duration"]),
                i.link["DbIndex"],
            )
            for i in items
            if i.link["DbOwner"] == track
        )
        cursor = seed.start_frame
        for index, (start, is_clip, end, db_index) in enumerate(ordered):
            if db_index != index:
                raise bad("the items' DbIndex is not their record order")
            if not is_clip:
                continue  # a transition lies over the cut it sits on (checked by the verifier)
            if start < cursor:
                raise bad("two items overlap on one track, or one starts before the timeline")
            cursor = end
