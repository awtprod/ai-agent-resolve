"""Synthetic Resolve disk-library ``Project.db`` files for the S25a tests.

:func:`rows_from_members` turns the XML members of a synthetic ``.drp`` (the archives the other
``tests/drp`` tests build) into the rows a library holds, following the layout recorded in
DESIGN "Shared project library (S25)": one table per element type (``Sm2TiVideoClip`` and the
other item types share ``Sm2TiItem``, with the type in ``DbType``), the ``DbId`` as
``<Table>_id``, leaf elements as columns (hex blobs as raw bytes, empty elements as NULL),
an object inside a single-object element as a reference column plus an owner column
``<Parent>_id`` on the child, and an object inside a vector as a join table named after the
two tables in sorted order, with ``DbOwner``, ``DbAssociate``, ``DbPropertyName`` and
``DbIndex``. It is written from those facts, not from :mod:`aar.drp.library`, so a parity test
compares two independent layouts. Nothing here is product code.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from aar.drp.scan import Element, scan

JOIN = ["DbOwner", "DbAssociate", "DbPropertyName", "DbIndex"]

#: A subset of a real Project.db's tables and columns: what the reader needs, the owner
#: columns, and a few columns it does not read (so ignoring them is exercised too).
SCHEMA: dict[str, list[str]] = {
    "SM_Project": [
        "SM_Project_id",
        "DbType",
        "ProjectName",
        "ProjectVersion",
        "LastModTimeInSecs",
        "CurrentTimelineIndex",
        "CommonConfig",
        "LockableBlobMap",
        "FieldsBlob",
    ],
    "SM_Project_Sm2Timeline": JOIN,
    "SM_Config": ["SM_Config_id", "DbType", "Name", "SM_Project_id", "FieldsBlob"],
    "BtLockableBlobMap": ["BtLockableBlobMap_id", "DbType", "SM_Project_id", "FieldsBlob"],
    "BtLockableBlob": [
        "BtLockableBlob_id",
        "DbType",
        "BtLockableBlobMap_id",
        "FieldsBlob",
        "BlobOwner",
        "DbSavedTime",
    ],
    "Sm2MpFolder": ["Sm2MpFolder_id", "DbType", "Name", "MpFolder", "FieldsBlob", "ColorTag"],
    "Sm2MpFolder_Sm2MpMedia": JOIN,
    "Sm2MpMedia": [
        "Sm2MpMedia_id",
        "DbType",
        "Name",
        "MpFolder",
        "UniqueMediaPoolItemId",
        "Video",
        "EmbeddedAudio",
        "Left",
        "Right",
        "LeftEye",
        "RightEye",
        "Sequence",
        "TimelineSharedHandle",
        "Sm2MpFolder_id",
        "FieldsBlob",
    ],
    "BtVideoInfo": [
        "BtVideoInfo_id",
        "DbType",
        "Clip",
        "Time",
        "Geometry",
        "Radiometry",
        "Sm2MpMedia_id",
    ],
    "BtAudioInfo": ["BtAudioInfo_id", "DbType", "Clip", "TracksBA", "Sm2MpMedia_id"],
    "BtAudioInfo_Sm2MpMedia": JOIN,
    "Sm2Timeline": ["Sm2Timeline_id", "DbType", "Name", "Sequence", "Sm2MpMedia_id"],
    "Sm2Sequence": [
        "Sm2Sequence_id",
        "DbType",
        "MediaExtents",
        "FrameRate",
        "Resolution",
        "Sm2Timeline_id",
        "Sm2MpMedia_id",
        "FieldsBlob",
    ],
    "Sm2SequenceContainer": ["Sm2SequenceContainer_id", "DbType", "Sm2Sequence_id", "FieldsBlob"],
    "Sm2SequenceContainer_Sm2TiTrack": JOIN,
    "Sm2TiTrack": [
        "Sm2TiTrack_id",
        "DbType",
        "Type",
        "SubType",
        "Flags",
        "Sequence",
        "UserDefinedName",
        "FieldsBlob",
        "Sm2SequenceContainer_id",
    ],
    "Sm2TiItem_Sm2TiTrack": JOIN,
    "Sm2TiItem": [
        "Sm2TiItem_id",
        "DbType",
        "Name",
        "Start",
        "Duration",
        "In",
        "MarkersBA",
        "PrettyType",
        "MediaRef",
        "MediaStartTime",
        "MediaFilePath",
        "MediaTimemapBA",
        "MediaFrameRate",
        "MediaTrackIdx",
        "Sm2TiTrack_id",
        "FieldsBlob",
    ],
    "Gallery::GyGallery": ["Gallery::GyGallery_id", "DbType", "SM_Project_id"],
}
BLOB_COLUMNS = {
    "FieldsBlob",
    "Clip",
    "Time",
    "Geometry",
    "Radiometry",
    "TracksBA",
    "FrameRate",
    "Resolution",
    "MediaExtents",
    "MediaTimemapBA",
    "MediaFrameRate",
    "MarkersBA",
}
INTEGER_COLUMNS = {
    "ProjectVersion",
    "LastModTimeInSecs",
    "CurrentTimelineIndex",
    "Type",
    "SubType",
    "Flags",
    "MediaTrackIdx",
    "DbSavedTime",
}
#: Vectors of plain ids (not objects): the join table each one lives in.
TEXT_VECTORS = {("SM_Project", "TimelineHandleVec"): "SM_Project_Sm2Timeline"}
OLD = 1_000_000_000  # an mtime long past, so a fixture counts as settled at once

Tables = dict[str, list[dict[str, Any]]]


def table_of(tag: str) -> str | None:
    """The table an element type is stored in."""
    if tag.endswith("LockableBlob"):
        return "BtLockableBlob"
    if tag.startswith("Sm2Ti") and tag not in ("Sm2TiTrack", "Sm2Timeline"):
        return "Sm2TiItem"
    if tag.startswith("Sm2Mp") and tag.endswith("Clip"):
        return "Sm2MpMedia"
    if tag == "Sm2LockableBlobMap":
        return "BtLockableBlobMap"
    return tag if tag in SCHEMA else None


def _value(column: str, element: Element) -> object:
    text = element.text
    if not text:
        return None
    if column in BLOB_COLUMNS:
        return bytes.fromhex(text)
    if column in INTEGER_COLUMNS and text.strip().lstrip("-").isdigit():
        return int(text)
    return text


def rows_from_members(members: dict[str, str | bytes], *, project_version: int = 15) -> Tables:
    """The library rows of the project the XML ``members`` describe."""
    tables: Tables = {name: [] for name in SCHEMA}

    def walk(
        el: Element, parent: tuple[str, dict[str, Any]] | None, prop: str, index: int | None
    ) -> None:
        table = table_of(el.tag)
        if table is None:
            return
        rid = el.attrs["DbId"]
        row: dict[str, Any] = {f"{table}_id": rid, "DbType": el.tag}
        if parent is not None:
            ptable, prow = parent
            pid = prow[f"{ptable}_id"]
            if f"{ptable}_id" in SCHEMA[table]:
                row[f"{ptable}_id"] = pid
            if index is None:
                prow[prop] = rid
            else:
                join = "_".join(sorted((ptable, table)))
                if join in SCHEMA:
                    tables[join].append(
                        {
                            "DbOwner": pid,
                            "DbAssociate": rid,
                            "DbPropertyName": prop,
                            "DbIndex": index,
                        }
                    )
        tables[table].append(row)
        for child in el.children:
            if not child.children:
                if child.tag in SCHEMA[table]:
                    row[child.tag] = _value(child.tag, child)
                continue
            if all(c.tag == "Element" for c in child.children):
                for i, wrapper in enumerate(child.children):
                    if wrapper.children:
                        walk(wrapper.children[0], (table, row), child.tag, i)
                    elif (table, child.tag) in TEXT_VECTORS:
                        tables[TEXT_VECTORS[(table, child.tag)]].append(
                            {
                                "DbOwner": rid,
                                "DbAssociate": wrapper.text,
                                "DbPropertyName": child.tag,
                                "DbIndex": i,
                            }
                        )
            else:
                for obj in child.children:
                    walk(obj, (table, row), child.tag, None)

    for name, text in members.items():
        if not name.endswith(".xml"):
            continue
        data = text.encode() if isinstance(text, str) else text
        root = scan(data, name).root
        if root is not None:
            walk(root, None, "", None)
    for project in tables["SM_Project"]:
        project.setdefault("ProjectVersion", project_version)
    return tables


def write_db(
    path: Path,
    tables: Tables,
    *,
    schema: dict[str, list[str]] | None = None,
    settled: bool = True,
) -> Path:
    """Create ``path`` with ``schema`` (default :data:`SCHEMA`) and ``tables``' rows."""
    schema = SCHEMA if schema is None else schema
    conn = sqlite3.connect(path)
    try:
        for table, columns in schema.items():
            cols = ", ".join(f'"{c}"' for c in columns)
            conn.execute(f'CREATE TABLE "{table}" ({cols})')
            for row in tables.get(table, []):
                names = [c for c in row if c in columns]
                marks = ", ".join("?" for _ in names)
                quoted = ", ".join(f'"{c}"' for c in names)
                conn.execute(
                    f'INSERT INTO "{table}" ({quoted}) VALUES ({marks})', [row[c] for c in names]
                )
        conn.commit()
    finally:
        conn.close()
    if settled:
        os.utime(path, (OLD, OLD))
    return path


def library_db(
    path: Path, members: dict[str, str | bytes], edit: Callable[[Tables], None] | None = None
) -> Path:
    """A ``Project.db`` at ``path`` for the project ``members`` describe (``edit`` first)."""
    tables = rows_from_members(members)
    if edit is not None:
        edit(tables)
    return write_db(path, tables)


def find(tables: Tables, table: str, rid: str) -> dict[str, Any]:
    return next(r for r in tables[table] if r[f"{table}_id"] == rid)


def library_tree(root: Path, projects: Iterable[tuple[str, dict[str, str | bytes]]]) -> Path:
    """A disk library at ``root`` holding one project folder per (folder, members)."""
    base = root / "Resolve Projects" / "Users" / "guest" / "Projects"
    for folder, members in projects:
        (base / folder).mkdir(parents=True)
        library_db(base / folder / "Project.db", members)
    return root
