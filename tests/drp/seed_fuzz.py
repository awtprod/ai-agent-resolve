"""Damage a synthetic seed the ways a hostile or broken seed can differ from a Resolve save.

Each case starts from the plain seed of ``libwrite_support`` (a clip and a linked cut) and
applies one to three damages, chosen by a ``random.Random``:

* ``value``: a column of a row the seed reader reads set to another value or type (NULL, a
  number, text, bytes);
* ``fixed``: a fixed-layout blob (the sequence's resolution, rate and extents; an item's time
  map, rate and pre-conform extents) given another type, cut short, lengthened or a byte
  flipped;
* ``blob``: one byte of any blob flipped, or the blob cut short;
* ``schema``: a column the reader or the writer uses renamed, or a table dropped;
* ``bytes``: bits of the written file flipped (any page);
* ``page1``: a byte of a stored ``CREATE TABLE`` statement given its top bit (SQLite then
  quotes a byte that is not UTF-8 in its error message).

Nothing here is product code; every value is made up.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any

from libwrite_support import linked_pair, mp4_template, seed_tables, write_tables

__all__ = ["KINDS", "damaged_seed"]

KINDS = ("value", "fixed", "blob", "schema", "bytes", "page1")
CLIP = mp4_template()
_FIXED = {
    "Sm2Sequence": ("Resolution", "FrameRate", "MediaExtents"),
    "Sm2TiItem": ("MediaTimemapBA", "MediaFrameRate", "PreConformMediaExtents"),
}
_JOIN = ("DbOwner", "DbAssociate", "DbPropertyName", "DbIndex")
#: Columns the seed reader and the writer use (this module's own list, not the product's).
_SCHEMA = {
    "SM_Project": ("SM_Project_id", "ProjectName"),
    "SM_Project_Sm2Timeline": _JOIN,
    "Sm2Timeline": ("Sm2Timeline_id", "Name", "Sequence", "MpTimelineClip"),
    "Sm2Sequence": ("Sm2Sequence_id", "FrameRate", "MediaExtents", "Resolution"),
    "Sm2SequenceContainer": ("Sm2SequenceContainer_id", "Sm2Sequence_id"),
    "Sm2SequenceContainer_Sm2TiTrack": _JOIN,
    "Sm2TiTrack": ("Sm2TiTrack_id", "Type"),
    "Sm2TiItem": ("Sm2TiItem_id", "Start", "MediaFilePath", "FieldsBlob"),
    "Sm2TiItem_Sm2TiTrack": _JOIN,
    "Sm2MpFolder": ("Sm2MpFolder_id", "MpFolder"),
    "BtLockableBlobMap": ("BtLockableBlobMap_id",),
}
_VALUES: tuple[Any, ...] = (None, 0, -1, 7, 2**40, 1.5, "", "x", "7", b"", b"\x00", b"\xff" * 9)


def _tables() -> dict[str, list[dict[str, Any]]]:
    return seed_tables(
        clips=[CLIP], items=linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0)
    )


def _value(tables: dict[str, list[dict[str, Any]]], rng: random.Random) -> str:
    table = rng.choice(sorted(t for t, rows in tables.items() if rows))
    row = rng.choice(tables[table])
    column = rng.choice(sorted(row))
    row[column] = rng.choice(_VALUES)
    return f"value {table}.{column}"


def _fixed(tables: dict[str, list[dict[str, Any]]], rng: random.Random) -> str:
    table = rng.choice(sorted(_FIXED))
    row = rng.choice(tables[table])
    column = rng.choice(_FIXED[table])
    value = row.get(column)
    how = rng.randrange(5)
    if how == 0 or not isinstance(value, bytes) or not value:
        row[column] = rng.choice((7, 1.0, "16 bytes", None))
    elif how == 1:
        row[column] = value[: rng.randrange(len(value))]
    elif how == 2:
        row[column] = value + bytes(rng.randrange(1, 9))
    else:
        data = bytearray(value)
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        row[column] = bytes(data)
    return f"fixed {table}.{column} ({how})"


def _blob(tables: dict[str, list[dict[str, Any]]], rng: random.Random) -> str:
    blobs = [
        (table, row, column)
        for table, rows in sorted(tables.items())
        for row in rows
        for column, value in sorted(row.items())
        if isinstance(value, bytes) and value
    ]
    table, row, column = rng.choice(blobs)
    data = bytearray(row[column])
    if rng.random() < 0.7:
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
    else:
        del data[rng.randrange(len(data)) :]
    row[column] = bytes(data)
    return f"blob {table}.{column}"


def _schema(path: Path, rng: random.Random) -> str:
    table = rng.choice(sorted(_SCHEMA))
    conn = sqlite3.connect(path)
    try:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchall():
            return f"schema {table} already dropped"
        if rng.random() < 0.25:
            conn.execute(f'DROP TABLE "{table}"')
            done = f"schema drop {table}"
        else:
            column = rng.choice(_SCHEMA[table])
            conn.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{column}" TO "{column}_old"')
            done = f"schema rename {table}.{column}"
        conn.commit()
    finally:
        conn.close()
    return done


def _bytes(path: Path, rng: random.Random) -> str:
    data = bytearray(path.read_bytes())
    for _ in range(rng.randrange(1, 24)):
        data[rng.randrange(100, len(data))] ^= 1 << rng.randrange(8)
    path.write_bytes(bytes(data))
    return "bytes"


def _page1(path: Path, rng: random.Random) -> str:
    """A byte of a ``CREATE`` statement in the file given its top bit (the stored schema's
    text, as a damaged page leaves it): SQLite then quotes a byte that is not UTF-8 in its
    error message."""
    data = bytearray(path.read_bytes())
    starts = [i for i in range(len(data)) if data.startswith(b"CREATE TABLE", i)]
    at = rng.choice(starts) + rng.randrange(13, 120)
    data[min(at, len(data) - 1)] |= 0x80
    path.write_bytes(bytes(data))
    return "page1"


def damaged_seed(path: Path, number: int) -> list[str]:
    """Write damaged seed number ``number`` to ``path``; returns what was damaged."""
    rng = random.Random(number)
    kinds = [rng.choice(KINDS) for _ in range(rng.randint(1, 3))]
    tables = _tables()
    done = []
    for kind in kinds:
        if kind == "value":
            done.append(_value(tables, rng))
        elif kind == "fixed":
            done.append(_fixed(tables, rng))
        elif kind == "blob":
            done.append(_blob(tables, rng))
    write_tables(path, tables)
    done += [_schema(path, rng) for kind in kinds if kind == "schema"]
    done += [_bytes(path, rng) for kind in kinds if kind == "bytes"]  # last: the file as written
    done += [_page1(path, rng) for kind in kinds if kind == "page1"]
    return done
