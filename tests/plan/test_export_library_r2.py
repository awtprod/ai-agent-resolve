"""S25c2 fix round 2, the independent verifier made total (item N2): a value planted in the
staged project after the writer's own read-back, in any column of an item, in an added media
clip's rows, or in any other row of the seed, is refused before anything is placed.

The plants change the staged file between the writer's read-back and the verifier (as the
validator's did), so each one is seen by the verifier alone. Every value is made up."""

from __future__ import annotations

import sqlite3
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drp"))

import test_export_library as base

import aar.drp.libwrite as libwrite
from aar.drp.mediapool import blobtree as bt
from aar.plan.export.errors import ExportError

pytest.importorskip("zstandard")

_Plant = Callable[[sqlite3.Connection], None]


def _first(conn: sqlite3.Connection, kind: str) -> str:
    return str(
        conn.execute(
            "SELECT Sm2TiItem_id FROM Sm2TiItem WHERE DbType = ? ORDER BY CAST(Start AS INTEGER) "
            "LIMIT 1",
            (kind,),
        ).fetchone()[0]
    )


def _set_item(column: str, value: object, kind: str = "Sm2TiVideoClip") -> _Plant:
    def plant(conn: sqlite3.Connection) -> None:
        conn.execute(
            f'UPDATE Sm2TiItem SET "{column}" = ? WHERE Sm2TiItem_id = ?',
            (value, _first(conn, kind)),
        )

    return plant


def _added_clip(conn: sqlite3.Connection) -> str:
    """An added media clip the cut plays (one named in the plan)."""
    return str(
        conn.execute("SELECT Sm2MpMedia_id FROM Sm2MpMedia WHERE Name = 'camera-a.MP4'").fetchone()[
            0
        ]
    )


def _copy_row(
    conn: sqlite3.Connection, table: str, where: str, args: tuple[Any, ...], **change: Any
) -> None:
    columns = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    row = dict(
        zip(
            columns,
            conn.execute(f"SELECT * FROM {table} WHERE {where}", args).fetchone(),
            strict=True,
        )
    )
    row.update(change)
    conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        [row[c] for c in columns],
    )


def _extra_sound_join(conn: sqlite3.Connection) -> None:
    clip = _added_clip(conn)
    _copy_row(conn, "BtAudioInfo_Sm2MpMedia", "DbOwner = ?", (clip,), DbIndex=1)


def _extra_picture_info(conn: sqlite3.Connection) -> None:
    clip = _added_clip(conn)
    _copy_row(conn, "BtVideoInfo", "Sm2MpMedia_id = ?", (clip,), BtVideoInfo_id=str(uuid.uuid4()))


def _sound_names_another_file(conn: sqlite3.Connection) -> None:
    clip = _added_clip(conn)
    audio_id, blob = conn.execute(
        "SELECT BtAudioInfo_id, Clip FROM BtAudioInfo WHERE Sm2MpMedia_id = ?", (clip,)
    ).fetchone()

    def other(fields: Any) -> None:
        for field in fields:
            if field[0] == 2:
                field[2] = b"camera-z.MP4"

    conn.execute(
        "UPDATE BtAudioInfo SET Clip = ? WHERE BtAudioInfo_id = ?",
        (bt.edit_message(blob, "Clip", other), audio_id),
    )


def _fields_tail(conn: sqlite3.Connection) -> None:
    item = _first(conn, "Sm2TiVideoClip")
    (blob,) = conn.execute(
        "SELECT FieldsBlob FROM Sm2TiItem WHERE Sm2TiItem_id = ?", (item,)
    ).fetchone()

    def damage(fields: Any) -> None:
        for field in fields:
            if field[0] == 2:
                field[2] = field[2][:-1] + b"\x00"

    conn.execute(
        "UPDATE Sm2TiItem SET FieldsBlob = ? WHERE Sm2TiItem_id = ?",
        (bt.edit_message(blob, "FieldsBlob", damage), item),
    )


def _seed_media_renamed(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE Sm2MpMedia SET Name = 'renamed.MP4' WHERE Name = 'camera-b.MP4'")


def _extra_bin(conn: sqlite3.Connection) -> None:
    _copy_row(conn, "Sm2MpFolder", "1 = 1 LIMIT 1", (), Sm2MpFolder_id=str(uuid.uuid4()))


PLANTS: list[tuple[str, _Plant, str]] = [
    ("an effect blob", _set_item("EffectFiltersBA", b"\x01\x02\x03\x04\x05"), "EffectFiltersBA"),
    ("a grade table", _set_item("pLmVerTable", str(uuid.uuid4())), "pLmVerTable"),
    ("a Fusion composition", _set_item("CompositionTable", str(uuid.uuid4())), "CompositionTable"),
    ("markers", _set_item("MarkersBA", b""), "MarkersBA"),
    ("import metadata", _set_item("ImportExportMetadataBA", b"\x00"), "ImportExportMetadataBA"),
    ("wrong Flags", _set_item("Flags", 1, "Sm2TiAudioClip"), "Flags"),
    ("a wrong Name", _set_item("Name", "wrong"), "Name"),
    ("a Track", _set_item("Track", str(uuid.uuid4())), "Track"),
    ("a Group", _set_item("Group", str(uuid.uuid4())), "Group"),
    ("a reel number", _set_item("MediaReelNumber", "A001"), "MediaReelNumber"),
    (
        "PreConformMediaExtents",
        _set_item("PreConformMediaExtents", bytes(16)),
        "PreConformMediaExtents",
    ),
    (
        "a sound's PreConformMediaExtents",
        _set_item("PreConformMediaExtents", bytes(16), "Sm2TiAudioClip"),
        "PreConformMediaExtents",
    ),
    ("a damaged FieldsBlob tail", _fields_tail, "FieldsBlob"),
    ("an extra sound join of a used clip", _extra_sound_join, "BtAudioInfo_Sm2MpMedia"),
    ("an extra picture info of a used clip", _extra_picture_info, "BtVideoInfo"),
    ("a sound info naming another file", _sound_names_another_file, "Clip"),
    ("a seed's media row changed", _seed_media_renamed, "Sm2MpMedia"),
    ("a row the writer did not add", _extra_bin, "Sm2MpFolder"),
    (
        "the project's name",
        lambda c: c.execute("UPDATE SM_Project SET ProjectName = 'x'"),
        "SM_Project.ProjectName",
    ),
    (
        "the timeline's name",
        lambda c: c.execute("UPDATE Sm2Timeline SET Name = 'x'"),
        "Sm2Timeline.Name",
    ),
]


@pytest.mark.parametrize(("label", "plant", "field"), PLANTS, ids=[p[0] for p in PLANTS])
def test_a_value_planted_anywhere_the_writer_writes_is_refused_before_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, plant: _Plant, field: str
) -> None:
    paths = base._setup(tmp_path)
    real = libwrite.verify_written
    seen: list[str] = []

    def planted(path: Path, *args: Any, **kwargs: Any) -> Any:
        conn = sqlite3.connect(path)
        try:
            before = conn.total_changes
            plant(conn)
            assert conn.total_changes > before, f"the plant of {label} changed nothing"
            conn.commit()
        finally:
            conn.close()
        try:
            return real(path, *args, **kwargs)
        except Exception as exc:
            seen.append(getattr(exc, "details", {}).get("field", ""))
            raise

    monkeypatch.setattr(libwrite, "verify_written", planted)
    with pytest.raises(ExportError) as info:
        base._export(paths)
    assert info.value.code == "self_check_failed"
    assert seen == [field]
    assert base._projects(paths) == []


def test_without_a_plant_the_same_export_is_placed(tmp_path: Path) -> None:
    paths = base._setup(tmp_path)
    out = base._export(paths)
    assert base._projects(paths) == ["Show cut"]
    assert out.report["placement"]["cleanup"] == []
