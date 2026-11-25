"""Synthetic seeds, template items and libraries for the S25c2 tests (nothing here is product
code; every value is made up).

A seed is the project the owner saves in Resolve: one timeline with one video and one audio
track, a bin holding the timeline's own clip, the project's config and lockable map. Item rows
follow the field map in DESIGN "Library projects (S25c2)", written here from that map with this
module's own arithmetic (not :mod:`aar.drp.items`), so a test of the writer compares two
independent codings of it.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from mediapool_support import (
    BYTES,
    FOLDER,
    MAP,
    S,
    envelope,
    join,
    kd,
    mp4_facts,
    mp4_template,
    mp4_template_facts,
    pb,
    setup_ba,
    uid,
)

from aar.drp.items.rows import ITEM_COLUMNS
from aar.drp.library import COLUMNS
from aar.drp.mediapool.templates import REQUIRED_COLUMNS, ClipRows

__all__ = [
    "AUDIO_TRACK",
    "PROJECT",
    "TIMELINE",
    "VIDEO_TRACK",
    "facts_lines",
    "item_row",
    "linked_pair",
    "make_library",
    "mp4_facts",
    "mp4_template",
    "mp4_template_facts",
    "write_seed",
]

PROJECT = uid(0xA001)
TIMELINE = uid(0xA002)
SEQUENCE = uid(0xA003)
CONTAINER = uid(0xA004)
VIDEO_TRACK = uid(0xA005)
AUDIO_TRACK = uid(0xA006)
TIMELINE_CLIP = uid(0xA007)
CONFIG = uid(0xA008)
JOIN = ["DbOwner", "DbAssociate", "DbPropertyName", "DbIndex"]
#: The 16 constant bytes every picture item's FieldsBlob #2 holds (in the corpus too).
PICTURE_TAIL = bytes.fromhex("000000000000000600000000ffffffff")
PRECONFORM = bytes.fromhex("00000100000030c20000010000003042")
RATE_2398 = 24000 / 1001


def _rate16(value: float) -> bytes:
    return struct.pack("<d", value) + bytes(8)


def item_row(
    kind: str,
    item_id: str,
    track: str,
    clip: ClipRows,
    *,
    start: int,
    duration: int,
    src_in: int,
    partners: Sequence[str] = (),
    frames: int = 240,
    rate: float = RATE_2398,
    start_seconds: float = 86400 / RATE_2398,
    path: str = "X:\\Old\\Footage\\camera-b.MP4",
    selector: int = 4242,
) -> dict[str, Any]:
    """A picture or sound item of ``clip`` as Resolve stores it (a full-speed item of a picture
    file: the clip's frames over its rate give the time map)."""
    links = kd(*((str(i), S, p) for i, p in reversed(list(enumerate(partners)))))
    group = [(1, 2, pb((1, 2, links)))] if partners else []
    inner = pb(*group, (4, 0, 1))
    tail = (2, 2, PICTURE_TAIL) if kind == "picture" else (15, 0, 4)
    row = {c: None for c in ITEM_COLUMNS}
    row.update(
        {
            "Sm2TiItem_id": item_id,
            "DbType": "Sm2TiVideoClip" if kind == "picture" else "Sm2TiAudioClip",
            "Name": clip.media["Name"],
            "Start": str(start),
            "Duration": str(duration),
            "UiMemento": 0,
            "Flags": 0,
            "PriorityIndex": 0,
            "In": str(src_in) if src_in else None,
            "MediaRef": clip.media_id,
            "MediaStartTime": start_seconds,
            "MediaFilePath": path,
            "MediaTimemapBA": b"\x02" + struct.pack(">d", frames / rate - 1 / rate),
            "ThumbnailDirtyFlag": 1,
            "RenderTextEnabled": 1,
            "RenderTextGanged": 1,
            "RenderTextPrefixed": 1,
            "LastChangedTime": 0,
            "LastRenderedTime": 0,
            "IsMarkedForCaching": 0,
            "IsForceConformed": 1,
            "MatchConflictState": 0,
            "IsPreConformed": 0,
            "MediaFrameRate": _rate16(rate),
            "Sm2TiTrack_id": track,
            "MixedFrameRateAlignment": 0.0,
            "UseOppositeSrcForLeftEye": 0,
            "UseOppositeSrcForRightEye": 0,
            "WasDisbanded": 0,
            "CurrentSelectorIdx": selector if kind == "picture" else 0,
            "FieldsBlob": envelope(pb((1, 2, inner), tail)),
            "BlobLockSysId": "",
            "CompTableLockSysId": "",
            "VersionTableLockSysId": "",
        }
    )
    if kind == "picture":
        row["PreConformMediaExtents"] = PRECONFORM
    else:
        row["MediaTrackIdx"] = 0
        row["VirtualAudioTrackBA"] = kd(
            ("ChannelsBA", BYTES, struct.pack(">IIII", 2, 2, 0x4001, 0x4002)),
            ("AudioType", 2, 0),
        )
    return row


def linked_pair(
    clip: ClipRows, base: int, *, start: int, duration: int, src_in: int, index: int = 0
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """A linked picture and sound of ``clip`` on the seed's tracks: ``(item, join)`` twice."""
    pic, snd = uid(base), uid(base + 1)
    out = []
    for kind, me, other, track in (
        ("picture", pic, snd, VIDEO_TRACK),
        ("sound", snd, pic, AUDIO_TRACK),
    ):
        row = item_row(
            kind, me, track, clip, start=start, duration=duration, src_in=src_in, partners=(other,)
        )
        out.append((row, join(track, me, "Items", index)))
    return out


def _extents(start_frame: int, rate: float, length: float = 0.0) -> bytes:
    return struct.pack("<dd", start_frame / rate, length)


def _resolution(width: int, height: int) -> bytes:
    return struct.pack(">IIII", 0, width, 0, height)


def seed_tables(
    *,
    rate: float = RATE_2398,
    start_frame: int = 86400,
    resolution: tuple[int, int] = (3840, 2160),
    rate_text: str | None = "23.976",
    clips: Iterable[ClipRows] = (),
    items: Iterable[tuple[dict[str, Any], dict[str, Any]]] = (),
    timeline_name: str = "Timeline 1",
    project_name: str = "Seed",
    version: int = 17,
    extra_tracks: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """The rows of a seed project (see the module docstring)."""
    t: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
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
            "Sm2MpFolder_Sm2MpMedia",
            "Sm2MpMedia",
            "BtVideoInfo",
            "BtAudioInfo",
            "BtAudioInfo_Sm2MpMedia",
            "BtLockableBlob",
            "BtLockableBlobMap",
            "SmPreset",
        )
    }
    t["SM_Project"].append(
        {
            "SM_Project_id": PROJECT,
            "DbType": "SM_Project",
            "ProjectName": project_name,
            "ProjectVersion": version,
            "LastModTimeInSecs": 1_790_000_000,
            "CurrentTimelineIndex": 1,
            "CommonConfig": CONFIG,
            "LockableBlobMap": MAP,
        }
    )
    t["SM_Config"].append(
        {
            "SM_Config_id": CONFIG,
            "DbType": "SM_Config",
            "SM_Project_id": PROJECT,
            "FieldsBlob": setup_ba(rate_text) if rate_text else kd(("SetupBA", BYTES, b"")),
        }
    )
    t["BtLockableBlobMap"].append(
        {"BtLockableBlobMap_id": MAP, "DbType": "BtLockableBlobMap", "SM_Project_id": PROJECT}
    )
    t["SM_Project_Sm2Timeline"].append(join(PROJECT, TIMELINE, "TimelineHandleVec", 0))
    t["Sm2Timeline"].append(
        {
            "Sm2Timeline_id": TIMELINE,
            "DbType": "Sm2Timeline",
            "Name": timeline_name,
            "Sequence": SEQUENCE,
            "MpTimelineClip": TIMELINE_CLIP,
            "Sm2MpMedia_id": TIMELINE_CLIP,
        }
    )
    t["Sm2Sequence"].append(
        {
            "Sm2Sequence_id": SEQUENCE,
            "DbType": "Sm2Sequence",
            "MediaExtents": _extents(start_frame, rate),
            "FrameRate": _rate16(rate),
            "Resolution": _resolution(*resolution),
            "Sm2Timeline_id": TIMELINE,
            "FieldsBlob": kd(("SeqRef", S, SEQUENCE)),
        }
    )
    t["Sm2SequenceContainer"].append(
        {
            "Sm2SequenceContainer_id": CONTAINER,
            "DbType": "Sm2SequenceContainer",
            "Sm2Sequence_id": SEQUENCE,
        }
    )
    tracks = [(VIDEO_TRACK, 0, "VideoTrackVec", 0), (AUDIO_TRACK, 1, "AudioTrackVec", 0)]
    tracks += [(uid(0xA100 + n), 0, "VideoTrackVec", n + 1) for n in range(extra_tracks)]
    for tid, kind, prop, index in tracks:
        t["Sm2TiTrack"].append(
            {
                "Sm2TiTrack_id": tid,
                "DbType": "Sm2TiTrack",
                "Type": kind,
                "SubType": 0,
                "Flags": 0,
                "Sequence": SEQUENCE,
                "Sm2SequenceContainer_id": CONTAINER,
            }
        )
        t["Sm2SequenceContainer_Sm2TiTrack"].append(join(CONTAINER, tid, prop, index))
    t["Sm2MpFolder"].append(
        {"Sm2MpFolder_id": FOLDER, "DbType": "Sm2MpFolder", "Name": "Master", "MpFolder": None}
    )
    t["Sm2MpMedia"].append(
        {
            "Sm2MpMedia_id": TIMELINE_CLIP,
            "DbType": "Sm2MpTimelineClip",
            "Name": timeline_name,
            "MpFolder": FOLDER,
            "Sm2MpFolder_id": FOLDER,
            "UniqueMediaPoolItemId": uid(0xA009),
            "TimelineSharedHandle": TIMELINE,
            "FieldsBlob": envelope(pb((1, 2, pb((4, 0, 1))))),
        }
    )
    t["Sm2MpFolder_Sm2MpMedia"].append(join(FOLDER, TIMELINE_CLIP, "MediaVec", 0))
    for i, clip in enumerate(clips, start=1):
        for table, rows in clip.tables().items():
            for row in rows:
                row = dict(row)
                if table == "Sm2MpFolder_Sm2MpMedia":
                    row["DbIndex"] = i
                t[table].append(row)
    for row, link in items:
        t["Sm2TiItem"].append(row)
        t["Sm2TiItem_Sm2TiTrack"].append(link)
    return t


def write_tables(path: Path, tables: dict[str, list[dict[str, Any]]]) -> Path:
    """A ``Project.db`` holding ``tables``: each table has the columns the reader and the
    writers need plus every column a row names (as a Resolve table has all its columns)."""
    conn = sqlite3.connect(path)
    try:
        for table, rows in tables.items():
            names: list[str] = []
            for source in (
                COLUMNS.get(table, ()),
                REQUIRED_COLUMNS.get(table, ()),
                ITEM_COLUMNS if table == "Sm2TiItem" else (),
                JOIN
                if "_" in table and table.split("_")[0] != "SM" and table.count("_") == 1
                else (),
            ):
                names += [c for c in source if c not in names]
            for row in rows:
                names += [c for c in row if c not in names]
            if not names:
                names = [f"{table}_id"]
            conn.execute(
                f'CREATE TABLE "{table}" ({", ".join(chr(34) + c + chr(34) for c in names)})'
            )
            for row in rows:
                keys = list(row)
                conn.execute(
                    f'INSERT INTO "{table}" ({", ".join(chr(34) + c + chr(34) for c in keys)}) '
                    f"VALUES ({', '.join('?' for _ in keys)})",
                    [row[k] for k in keys],
                )
        conn.commit()
    finally:
        conn.close()
    import os

    os.utime(path, (1_000_000_000, 1_000_000_000))
    return path


def write_seed(path: Path, **kwargs: Any) -> Path:
    return write_tables(path, seed_tables(**kwargs))


def make_library(root: Path, user: str = "guest") -> Path:
    (root / "Resolve Projects" / "Users" / user / "Projects").mkdir(parents=True)
    return root


def facts_lines(path: Path, *facts: dict[str, Any]) -> Path:
    import json

    path.write_text("".join(json.dumps(f) + "\n" for f in facts), encoding="utf-8")
    return path
