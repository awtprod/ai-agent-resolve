"""The seed: a project Resolve saved, chosen by the owner, that a new project is made from.

A seed carries the project settings, the bin, and exactly one timeline whose tracks are the
skeleton: one video track and one audio track (adding or removing tracks is not proven). Its
timeline is empty, or holds only plain picture and sound items (no grade, effect, Fusion
composition, marker or edit history: rows the writer would otherwise leave dangling), which are
removed (``remove_items``): the real saves show that an item's id is named only by its join row
and its linked items, and :func:`read_seed` checks that on the seed itself. The seed's own media
stay in the pool: removing media is not proven.

:func:`read_seed` reads a seed through the bounded library reader
(:func:`aar.drp.library.read_rows`, :func:`aar.drp.read_drp`) and refuses anything else with a
typed :class:`ProjectWriteError` (``bad_seed``).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.items.rows import ITEM_COLUMNS, ItemRows, shape_of
from aar.drp.library import SQL_ERRORS, LibraryError, LibraryOptions, read_rows
from aar.drp.libwrite.errors import ProjectWriteError, refusing
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.templates import (
    MEDIA_TABLES,
    ClipRows,
    clips_from_rows,
    project_frame_rate,
)
from aar.media.probe import STANDARD_RATES, snap_rate

__all__ = [
    "REPLACED_TABLES",
    "SEED_VERSIONS",
    "WRITER_CELLS",
    "Seed",
    "read_seed",
    "row_digest",
    "table_names",
]

#: Project versions a seed may have: the version the media-pool and item gates were run on.
SEED_VERSIONS = frozenset({17})
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_RESOLUTION_TEXT = re.compile(r"([0-9]{2,5}) x ([0-9]{2,5})(?: |\Z)")
_TABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_:.\-]*\Z")
_HINT = (
    "make the seed in Resolve: a new project with the settings you want, one empty timeline "
    "with one video and one audio track, saved in a disk library"
)
#: The most tables a seed may have (a Resolve 21.1 project has 146).
MAX_TABLES = 400
_JOIN = ("DbOwner", "DbAssociate", "DbPropertyName", "DbIndex")
#: The columns :func:`read_seed` and the writer read or write, per table; a seed whose table
#: lacks one is refused (``bad_seed``), never read as if the value were empty. The media
#: tables' own are :data:`aar.drp.mediapool.templates.REQUIRED_COLUMNS`, checked by
#: :func:`aar.drp.mediapool.templates.clips_from_rows`.
SEED_COLUMNS: dict[str, tuple[str, ...]] = {
    "SM_Project": ("SM_Project_id", "ProjectName"),
    "SM_Config": ("FieldsBlob",),
    "SM_Project_Sm2Timeline": _JOIN,
    "Sm2Timeline": ("Sm2Timeline_id", "Name", "Sequence", "MpTimelineClip"),
    "Sm2Sequence": ("Sm2Sequence_id", "FrameRate", "MediaExtents", "Resolution"),
    "Sm2SequenceContainer": ("Sm2SequenceContainer_id", "Sm2Sequence_id"),
    "Sm2SequenceContainer_Sm2TiTrack": _JOIN,
    "Sm2TiTrack": ("Sm2TiTrack_id", "Type"),
    "Sm2TiItem": ITEM_COLUMNS,
    "Sm2TiItem_Sm2TiTrack": _JOIN,
    "Sm2MpFolder": ("Sm2MpFolder_id", "MpFolder"),
    "BtLockableBlobMap": ("BtLockableBlobMap_id",),
}
#: The triggers a Resolve 21.1 project has (whitespace collapsed): its two deletes of a media
#: clip's metadata. The writer changes rows in every table (the project id), so a seed with
#: any other trigger is refused: a trigger would make the copy differ from what was written.
RESOLVE_TRIGGERS = frozenset(
    {
        (
            "Delete_BtVideoInfo_trigger",
            "CREATE TRIGGER Delete_BtVideoInfo_trigger AFTER DELETE ON BtVideoInfo BEGIN DELETE "
            "FROM BtLockableBlob WHERE 1 = (SELECT Enable FROM TriggerControl WHERE Name = "
            "'master_trigger') AND BlobOwner = OLD.BtVideoInfo_id; END",
        ),
        (
            "Delete_BtAudioInfo_trigger",
            "CREATE TRIGGER Delete_BtAudioInfo_trigger AFTER DELETE ON BtAudioInfo BEGIN DELETE "
            "FROM BtLockableBlob WHERE 1 = (SELECT Enable FROM TriggerControl WHERE Name = "
            "'master_trigger') AND BlobOwner = OLD.BtAudioInfo_id; END",
        ),
    }
)


#: The cells of a seed row the writer may change, each checked on its own by the verifier
#: (:func:`aar.drp.libwrite.verify.verify_written`): ``(table, id column, the Seed attribute
#: holding the row's id, column)``. Every other cell of every row of the seed's other tables
#: must come out of the writer as it went in (but for the project id, :func:`row_digest`).
WRITER_CELLS = (
    ("SM_Project", "SM_Project_id", "project_id", "ProjectName"),
    ("Sm2Timeline", "Sm2Timeline_id", "timeline_id", "Name"),
    ("Sm2MpMedia", "Sm2MpMedia_id", "timeline_clip_id", "Name"),
    ("Sm2Sequence", "Sm2Sequence_id", "sequence_id", "MediaExtents"),
)
#: The tables whose seed rows the writer removes and writes anew (the timeline's items and
#: their join rows): the verifier checks every row of them against the layout instead.
REPLACED_TABLES = ("Sm2TiItem", "Sm2TiItem_Sm2TiTrack")
_MASKED = "\x00writer cell"
_PROJECT = "\x00project id"


@refusing("bad_input", "digesting a row", "this is a writer bug; report it")
def row_digest(row: dict[str, Any], *, project_id: str, masked: Collection[str] = ()) -> str:
    """A digest of every column and value of ``row`` (type and value; a double by its bits),
    with a text equal to ``project_id`` and the ``masked`` columns standing for themselves, so
    the seed's row and the written one digest alike when nothing else changed."""
    h = hashlib.sha256()
    for column in sorted(row):
        value = row[column]
        if column in masked:
            value = _MASKED
        elif isinstance(value, str) and value == project_id:
            value = _PROJECT
        if isinstance(value, float):
            value = value.hex()
            kind = "float"
        else:
            kind = type(value).__name__
        h.update(repr((column, kind, value)).encode("utf-8", "backslashreplace"))
        h.update(b"\x00")
    return h.hexdigest()


def _masks(seed_ids: dict[str, str | None]) -> dict[str, tuple[str, str, str]]:
    """``table -> (id column, id, masked column)`` of :data:`WRITER_CELLS` for these ids."""
    out: dict[str, tuple[str, str, str]] = {}
    for table, id_column, attribute, column in WRITER_CELLS:
        value = seed_ids.get(attribute)
        if value is not None:
            out[table] = (id_column, value, column)
    return out


def fingerprint(
    tables: dict[str, list[dict[str, Any]]], *, project_id: str, ids: dict[str, str | None]
) -> tuple[dict[str, dict[str, int]], dict[str, object]]:
    """Per table (but :data:`REPLACED_TABLES`), how many rows of each :func:`row_digest` it
    holds; and the values of the :data:`WRITER_CELLS` (``"table.column"``)."""
    masks = _masks(ids)
    digests: dict[str, dict[str, int]] = {}
    cells: dict[str, object] = {}
    for table, rows in tables.items():
        if table in REPLACED_TABLES:
            continue
        mask = masks.get(table)
        counted: Counter[str] = Counter()
        for row in rows:
            masked: tuple[str, ...] = ()
            if mask is not None and row.get(mask[0]) == mask[1]:
                masked = (mask[2],)
                cells[f"{table}.{mask[2]}"] = row.get(mask[2])
            counted[row_digest(row, project_id=project_id, masked=masked)] += 1
        digests[table] = dict(counted)
    return digests, cells


@dataclass(frozen=True)
class Seed:
    path: Path
    version: int
    project_id: str
    project_name: str
    timeline_id: str
    timeline_name: str
    #: The timeline's media-pool clip (``Sm2MpTimelineClip``), whose Name is the timeline's.
    timeline_clip_id: str | None
    sequence_id: str
    video_track: str
    audio_track: str
    rate: Fraction
    rate_bytes: bytes
    start_seconds: float
    start_frame: int
    #: The timeline's resolution: its own, the project's when it follows the project's and the
    #: project stores it, or None when neither is stored.
    resolution: tuple[int, int] | None
    resolution_from: str | None
    #: The project's own frame rate (``SetupBA`` field 69), None when it is not stored.
    project_rate: Fraction | None
    #: The bin new media go in (the one holding the timeline's clip) and how many it holds.
    folder_id: str
    folder_count: int
    lockable_map_id: str | None
    #: The ids of the seed timeline's items, removed by ``remove_items``.
    items: tuple[str, ...]
    #: Every uuid text in the seed: a new id must not be one of them.
    taken: frozenset[str]
    #: The columns of each table.
    columns: dict[str, tuple[str, ...]]
    #: The seed's own media file clips (templates too).
    clips: tuple[ClipRows, ...]
    #: Per table (but :data:`REPLACED_TABLES`): how many rows of each :func:`row_digest`.
    digests: dict[str, dict[str, int]]
    #: The seed's values of the :data:`WRITER_CELLS`, by ``"table.column"``.
    cells: dict[str, object]


def _bad(message: str, hint: str = _HINT, **details: Any) -> ProjectWriteError:
    return ProjectWriteError("bad_seed", message, hint, **details)


@refusing("bad_seed", "listing the tables", _HINT)
def table_names(path: Path) -> list[str]:
    """The ordinary tables of ``path`` (see :func:`_tables`)."""
    return _tables(path)


def _tables(path: Path) -> list[str]:
    """The ordinary tables of ``path`` (an immutable, read-only connection; no schema code),
    after checking it has no trigger but Resolve's own (:data:`RESOLVE_TRIGGERS`)."""
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except SQL_ERRORS as exc:
        raise _bad(f"{path} cannot be opened: {exc}") from None
    try:
        conn.execute("PRAGMA trusted_schema = OFF")
        conn.execute("PRAGMA query_only = ON")
        names = [
            str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        triggers = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
    except SQL_ERRORS as exc:
        raise _bad(f"{path} cannot be read: {exc}") from None
    finally:
        conn.close()
    if len(names) > MAX_TABLES:
        raise _bad(f"{path} has {len(names)} tables, more than a project ({MAX_TABLES})")
    bad = [n for n in names if not _TABLE_NAME.match(n)]
    if bad:
        raise _bad(f"{path} has a table whose name is not an element name: {bad[0]!r}")
    for name, sql in triggers:
        if (str(name), " ".join(str(sql or "").split())) not in RESOLVE_TRIGGERS:
            raise _bad(
                f"{path} has a trigger Resolve does not make ({str(name)[:80]!r}): it would "
                "change the copy behind the writer's back"
            )
    return [n for n in names if not n.startswith("sqlite_")]


def _rows(lib: Any, table: str) -> list[dict[str, Any]]:
    columns = lib.columns[table]
    return [dict(zip(columns, values, strict=True)) for values in lib.rows[table]]


def _setup_resolution(config: list[dict[str, Any]]) -> tuple[int, int] | None:
    """The project's resolution when the project stores it: ``SetupBA`` field 1, text such as
    ``1920 x 1080 HD`` (one of the three Resolve 21.1 saves has it)."""
    if len(config) != 1 or not isinstance(config[0].get("FieldsBlob"), bytes):
        return None
    try:
        kd = blobs.decode_keyed_dict(config[0]["FieldsBlob"], strict=True)
        setup = kd.get("SetupBA")
        if not isinstance(setup, bytes):
            return None
        fields = blobs.decode_protobuf(
            blobs.unwrap_message(setup, max_output=1024 * 1024), nested_depth=0
        )
    except DrpError:
        return None
    texts = [f.value for f in fields if f.number == 1 and isinstance(f.value, bytes)]
    if len(texts) != 1:
        return None
    match = _RESOLUTION_TEXT.match(texts[0].decode("ascii", "replace"))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _one(rows: list[dict[str, Any]], what: str, hint: str = _HINT) -> dict[str, Any]:
    if len(rows) != 1:
        raise _bad(f"the seed has {len(rows)} {what}, not one", hint)
    return rows[0]


def _uuids(value: object, out: set[str]) -> None:
    if isinstance(value, str) and _UUID.match(value):
        out.add(value)


@refusing("bad_seed", "reading the seed", _HINT)
def read_seed(path: str | Path, *, options: LibraryOptions | None = None) -> Seed:
    """Read and check a seed (see the module docstring); a ``bad_seed`` refusal otherwise.
    The seed's ``LibraryError`` (busy, damaged, too large...) becomes ``bad_seed`` with the
    library's code in ``details``."""
    p = Path(path)
    try:
        return _read_seed(p, options or LibraryOptions())
    except LibraryError as exc:
        raise _bad(
            f"{p} cannot be read as a project: {exc.message} [{exc.code}]",
            exc.hint,
            library_code=exc.code,
        ) from None
    except MediaPoolError as exc:
        raise _bad(f"{p}: {exc.message} [{exc.code}]", exc.hint) from None


def _read_seed(p: Path, opts: LibraryOptions) -> Seed:
    # read_rows checks the file first (a regular file, SQLite, integrity, one project), so the
    # table listing below only ever opens a file that passed those checks.
    first = read_rows(p, ("SM_Project",), options=opts)
    if first.project_version not in SEED_VERSIONS:
        raise _bad(
            f"the seed's project version is {first.project_version}; the writer was checked on "
            f"version {sorted(SEED_VERSIONS)} only (Resolve 21.1)",
        )
    names = _tables(p)
    lib = read_rows(p, tuple(names), options=opts)
    tables = {t: _rows(lib, t) for t in names}
    needed = (
        "SM_Project",
        "SM_Config",
        "SM_Project_Sm2Timeline",
        "Sm2Timeline",
        "Sm2Sequence",
        "Sm2SequenceContainer",
        "Sm2SequenceContainer_Sm2TiTrack",
        "Sm2TiTrack",
        "Sm2TiItem",
        "Sm2TiItem_Sm2TiTrack",
        "Sm2MpFolder",
        "BtLockableBlobMap",
        *MEDIA_TABLES,
    )
    missing = [t for t in needed if t not in tables]
    if missing:
        raise _bad(f"the seed has no {', '.join(missing[:4])} table")
    for table, wanted in SEED_COLUMNS.items():
        absent = [c for c in wanted if c not in lib.columns[table]]
        if absent:
            raise _bad(f"the seed's {table} table has no column {', '.join(absent[:4])}")
    project = _one(tables["SM_Project"], "project rows")
    timeline = _one(
        tables["Sm2Timeline"],
        "timelines",
        "save a seed with exactly one timeline (delete the others in Resolve, or make a new "
        "project)",
    )
    timeline_id = str(timeline.get("Sm2Timeline_id"))
    handles = [r for r in tables["SM_Project_Sm2Timeline"] if r.get("DbAssociate") == timeline_id]
    if len(handles) != 1 or len(tables["SM_Project_Sm2Timeline"]) != 1:
        raise _bad("the seed's project does not list exactly its one timeline")
    sequence = _one(
        [r for r in tables["Sm2Sequence"] if r.get("Sm2Sequence_id") == timeline.get("Sequence")],
        "sequences for its timeline",
    )
    containers = [
        r
        for r in tables["Sm2SequenceContainer"]
        if r.get("Sm2Sequence_id") == sequence["Sm2Sequence_id"]
    ]
    container = _one(containers, "track containers for its timeline")
    links = [
        r
        for r in tables["Sm2SequenceContainer_Sm2TiTrack"]
        if r.get("DbOwner") == container["Sm2SequenceContainer_id"]
    ]
    by_prop: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        by_prop.setdefault(str(link.get("DbPropertyName")), []).append(link)
    counts = {k: len(v) for k, v in by_prop.items()}
    if counts != {"VideoTrackVec": 1, "AudioTrackVec": 1}:
        raise _bad(
            f"the seed timeline's tracks are {counts or 'none'}, not one video and one audio "
            "track (adding or removing tracks is not proven)",
            "save the seed with one video track and one audio track",
        )
    video_track = str(by_prop["VideoTrackVec"][0]["DbAssociate"])
    audio_track = str(by_prop["AudioTrackVec"][0]["DbAssociate"])
    track_rows = {str(r.get("Sm2TiTrack_id")): r for r in tables["Sm2TiTrack"]}
    for tid, want in ((video_track, 0), (audio_track, 1)):
        row = track_rows.get(tid)
        if row is None or row.get("Type") != want:
            raise _bad("a track the seed timeline lists is missing or of the other type")
    if len(track_rows) != 2:
        raise _bad(f"the seed has {len(track_rows)} tracks, not the timeline's two")
    rate_bytes = sequence.get("FrameRate")
    extents = sequence.get("MediaExtents")
    try:
        if not isinstance(rate_bytes, bytes) or not isinstance(extents, bytes):
            raise DrpError("missing")
        rate_value = blobs.decode_rate(rate_bytes)
        start_seconds, _ = blobs.decode_extents(extents)
        resolution = blobs.decode_resolution(sequence.get("Resolution") or b"")
    except DrpError:
        raise _bad(
            "the seed timeline's frame rate, extents or resolution does not decode"
        ) from None
    rate = snap_rate(Fraction(rate_value))
    if rate not in STANDARD_RATES or float(rate) != rate_value or any(rate_bytes[8:]):
        raise _bad(f"the seed timeline's frame rate {rate_value!r} is not a standard rate")
    start_frame = round(start_seconds * rate_value)
    if (
        start_frame / rate_value != start_seconds
        and abs(start_frame / rate_value - start_seconds) > 1e-6
    ):
        raise _bad(f"the seed timeline starts between frames ({start_seconds!r} s)")
    resolution_from: str | None = "timeline"
    res: tuple[int, int] | None = resolution
    if resolution == (0, 0):
        res = _setup_resolution(tables["SM_Config"])
        resolution_from = "project" if res is not None else None
    # Items: none, or plain ones that remove_items can delete.
    track_ids = {video_track, audio_track}
    items = [r for r in tables["Sm2TiItem"]]
    joins = tables["Sm2TiItem_Sm2TiTrack"]
    if any(r.get("Sm2TiTrack_id") not in track_ids for r in items) or any(
        j.get("DbOwner") not in track_ids for j in joins
    ):
        raise _bad("the seed has items outside its timeline's tracks (compound or nested clips)")
    clips_list, _ = clips_from_rows({t: tables[t] for t in MEDIA_TABLES})
    clips = {c.media_id: c for c in clips_list}
    by_item = {str(j.get("DbAssociate")): j for j in joins}
    if len(by_item) != len(joins) or set(by_item) != {str(r.get("Sm2TiItem_id")) for r in items}:
        raise _bad("the seed's items and their join rows do not match one to one")
    item_ids: list[str] = []
    for row in items:
        rows = ItemRows(row, by_item[str(row.get("Sm2TiItem_id"))])
        shape = shape_of(rows, clips)
        if shape.refused is not None or shape.tags:
            why = shape.refused or ", ".join(sorted(shape.tags))
            raise _bad(
                f"the seed timeline holds an item the writer cannot remove ({why}): removing it "
                "would leave the rows it names behind",
                "save a seed whose timeline is empty, or holds only plain cuts",
            )
        item_ids.append(rows.item_id)
    _check_item_ids_unnamed(tables, set(item_ids))
    # The bin for new media: the one that holds the timeline's own clip.
    clip_id = timeline.get("MpTimelineClip")
    timeline_clip = next(
        (r for r in tables["Sm2MpMedia"] if r.get("Sm2MpMedia_id") == clip_id), None
    )
    folders = {str(r.get("Sm2MpFolder_id")) for r in tables["Sm2MpFolder"]}
    folder_id = str(timeline_clip.get("Sm2MpFolder_id")) if timeline_clip else ""
    if folder_id not in folders:
        roots = [r for r in tables["Sm2MpFolder"] if r.get("MpFolder") in (None, "")]
        folder_id = str(_one(roots, "root bins").get("Sm2MpFolder_id"))
    in_folder = [r for r in tables["Sm2MpFolder_Sm2MpMedia"] if r.get("DbOwner") == folder_id]
    maps = tables["BtLockableBlobMap"]
    project_id = str(project.get("SM_Project_id"))
    timeline_clip_id = str(clip_id) if timeline_clip else None
    digests, cells = fingerprint(
        tables,
        project_id=project_id,
        ids={
            "project_id": project_id,
            "timeline_id": timeline_id,
            "timeline_clip_id": timeline_clip_id,
            "sequence_id": str(sequence["Sm2Sequence_id"]),
        },
    )
    taken: set[str] = set()
    for rows_ in tables.values():
        for row in rows_:
            for value in row.values():
                _uuids(value, taken)
    return Seed(
        path=p,
        version=first.project_version,
        project_id=project_id,
        project_name=str(project.get("ProjectName") or ""),
        timeline_id=timeline_id,
        timeline_name=str(timeline.get("Name") or ""),
        timeline_clip_id=timeline_clip_id,
        sequence_id=str(sequence["Sm2Sequence_id"]),
        video_track=video_track,
        audio_track=audio_track,
        rate=rate,
        rate_bytes=rate_bytes,
        start_seconds=start_seconds,
        start_frame=start_frame,
        resolution=res,
        resolution_from=resolution_from,
        project_rate=project_frame_rate(tables["SM_Config"]),
        folder_id=folder_id,
        folder_count=len(in_folder),
        lockable_map_id=str(maps[0]["BtLockableBlobMap_id"]) if len(maps) == 1 else None,
        items=tuple(item_ids),
        taken=frozenset(taken),
        columns={t: tuple(lib.columns[t]) for t in names},
        clips=tuple(clips_list),
        digests=digests,
        cells=cells,
    )


def _check_item_ids_unnamed(tables: dict[str, list[dict[str, Any]]], ids: set[str]) -> None:
    """Refuse a seed in which an item's id is named outside the item rows, their join rows and
    the link lists of the items themselves: removing the item would leave a dangling name."""
    if not ids:
        return
    needles = {i: i for i in ids}
    for table, rows in tables.items():
        if table == "Sm2TiItem_Sm2TiTrack":
            continue
        for row in rows:
            own = row.get("Sm2TiItem_id") if table == "Sm2TiItem" else None
            for column, value in row.items():
                if table == "Sm2TiItem" and column in ("Sm2TiItem_id", "FieldsBlob"):
                    continue
                try:
                    found = bt.contains_text(value, needles)
                except MediaPoolError:
                    found = None if not isinstance(value, bytes) else "?"
                if found is not None and found != own:
                    raise _bad(
                        f"an item of the seed timeline is named in {table}.{column}: removing "
                        "it is not proven to be safe",
                        "save a seed whose timeline is empty",
                    )
