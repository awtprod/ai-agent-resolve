"""Timeline items as rows: what one picture or sound item of a library project is.

A timeline item (docs/DESIGN.md, "Shared project library (S25)") is one ``Sm2TiItem`` row (its
type in ``DbType``) and one ``Sm2TiItem_Sm2TiTrack`` join row placing it on its track's
``Items`` vector (``DbIndex`` its position; the item's ``Sm2TiTrack_id`` names the same track).
Linked items (a picture and its own sound) name each other inside ``FieldsBlob``; item ids
occur nowhere else (checked on the real saves, :func:`item_references`).

:func:`read_items` reads every item of a saved project, with its media clip and track, through
:func:`aar.drp.library.read_rows` (the same private snapshot and budgets as every library
read). :func:`shape_of` says whether an item is of a shape the writer supports and which
later additions it carries; an item it cannot support is never repaired, only listed.
"""

from __future__ import annotations

import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.items.errors import ItemError
from aar.drp.library import LibraryError, LibraryOptions, read_rows
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool.errors import MediaPoolError, refusing
from aar.drp.mediapool.templates import (
    MEDIA_TABLES,
    TEMPLATE_TABLES,
    ClipRows,
    Row,
    clips_from_rows,
    project_frame_rate,
)

__all__ = [
    "ITEM_COLUMNS",
    "ITEM_CONSTANTS",
    "ITEM_TYPES",
    "ItemRows",
    "ItemSet",
    "Shape",
    "check_item",
    "constant_problem",
    "links_of",
    "read_items",
    "shape_of",
]

PICTURE = "Sm2TiVideoClip"
SOUND = "Sm2TiAudioClip"
#: The item types the writer knows: a picture and a sound of a media file.
ITEM_TYPES = {PICTURE: "picture", SOUND: "sound"}
_JOIN = ("DbOwner", "DbAssociate", "DbPropertyName", "DbIndex")
#: Every column of ``Sm2TiItem`` on the Resolve 21.1 saves (project version 17). A template
#: with any other column is refused: what a new column means is not known.
ITEM_COLUMNS = (
    "Sm2TiItem_id",
    "DbType",
    "Name",
    "Start",
    "Duration",
    "Track",
    "LinkedItemSync",
    "MarkersBA",
    "UiMemento",
    "Flags",
    "PriorityIndex",
    "In",
    "MediaRef",
    "MediaStartTime",
    "MediaFilePath",
    "MediaReelNumber",
    "MediaTimemapBA",
    "Group",
    "pLmVerTable",
    "pAuxLmVerTable",
    "Thumbnail",
    "ThumbnailDirtyFlag",
    "RenderTextEnabled",
    "RenderTextGanged",
    "RenderTextPrefixed",
    "TextRenderItemVec",
    "LastChangedTime",
    "LastRenderedTime",
    "IsMarkedForCaching",
    "IsForceConformed",
    "MatchConflictState",
    "MediaTrackIdx",
    "PrettyType",
    "Resolution",
    "EffectFiltersBA",
    "IsPreConformed",
    "PreConformMediaExtents",
    "ImportExportMetadataBA",
    "MediaFrameRate",
    "Sm2TiTrack_id",
    "VirtualAudioTrackBA",
    "MixedFrameRateAlignment",
    "UseOppositeSrcForLeftEye",
    "UseOppositeSrcForRightEye",
    "WasDisbanded",
    "CurrentSelectorIdx",
    "AlignmentType",
    "Position",
    "RenderCacheBA",
    "ClipGroup",
    "MediaMetadata",
    "FieldsBlob",
    "CompositionTable",
    "BlobLockSysId",
    "OriginalClip",
    "Sm2TiItem_Owner_id",
    "CompTableLockSysId",
    "VersionTableLockSysId",
)
#: Columns a later edit fills in (a grade, a Fusion composition, effects, import metadata,
#: markers, per-item metadata): a fresh item has them NULL, and the writer writes NULL.
LATER_COLUMNS = {
    "pLmVerTable": "graded",
    "CompositionTable": "fusion",
    "EffectFiltersBA": "effects",
    "ImportExportMetadataBA": "import-metadata",
    "MarkersBA": "markers",
    "MediaMetadata": "item-metadata",
}
#: The columns the item writer copies from a template that hold one value on every plain item
#: of the corpus (9,937 of the 9,938 picture and sound items of the two 21.1 saves; the other is
#: a title): a template holding anything else is refused (:func:`check_item`), never copied.
#: The verifier keeps its own table (:data:`aar.drp.libwrite.verify.ITEM_RULES`); a test
#: compares the two.
ITEM_CONSTANTS: dict[str, object] = {
    "Track": None,
    "LinkedItemSync": None,
    "UiMemento": 0,
    "Flags": 0,
    "PriorityIndex": 0,
    "MediaReelNumber": None,
    "Group": None,
    "pAuxLmVerTable": None,
    "Thumbnail": None,
    "ThumbnailDirtyFlag": 1,
    "RenderTextEnabled": 1,
    "RenderTextGanged": 1,
    "RenderTextPrefixed": 1,
    "TextRenderItemVec": None,
    "LastChangedTime": 0,
    "LastRenderedTime": 0,
    "IsMarkedForCaching": 0,
    "IsForceConformed": 1,
    "MatchConflictState": 0,
    "PrettyType": None,
    "Resolution": None,
    "IsPreConformed": 0,
    "MixedFrameRateAlignment": 0.0,
    "UseOppositeSrcForLeftEye": 0,
    "UseOppositeSrcForRightEye": 0,
    "WasDisbanded": 0,
    "AlignmentType": None,
    "Position": None,
    "RenderCacheBA": None,
    "ClipGroup": None,
    "BlobLockSysId": "",
    "OriginalClip": None,
    "Sm2TiItem_Owner_id": None,
    "CompTableLockSysId": "",
    "VersionTableLockSysId": "",
}
#: ``PreConformMediaExtents``, copied too, by kind: every picture item of the corpus holds
#: these 16 bytes (4,709 of 4,709), every sound item none.
PRECONFORM: dict[str, bytes | None] = {
    "picture": bytes.fromhex("00000100000030c20000010000003042"),
    "sound": None,
}
_FRAMES = re.compile(r"-?[0-9]{1,12}\Z")
_HINT = "use a template item from a project Resolve saved"
#: The tables :func:`read_items` reads.
ITEM_TABLES = (
    *TEMPLATE_TABLES,
    "Sm2TiItem",
    "Sm2TiItem_Sm2TiTrack",
    "Sm2TiTrack",
    "Sm2SequenceContainer_Sm2TiTrack",
    "Sm2Sequence",
    "Sm2Timeline",
)


@dataclass(frozen=True)
class ItemRows:
    """One item: its ``Sm2TiItem`` row and its ``Items`` join row."""

    item: Row
    link: Row

    @property
    def item_id(self) -> str:
        return str(self.item.get("Sm2TiItem_id"))

    @property
    def kind(self) -> str | None:
        """``picture``, ``sound``, or None for any other item type."""
        return ITEM_TYPES.get(str(self.item.get("DbType")))


@dataclass(frozen=True)
class Shape:
    """Whether an item is of a supported shape: ``refused`` says why not (None when it is);
    ``tags`` name the later additions it carries (``graded``, ``effects``, ``renamed``...)."""

    kind: str | None
    refused: str | None
    tags: frozenset[str] = frozenset()


@dataclass
class ItemSet:
    """Every item of one saved project, with the media clips and tracks they name."""

    path: Path
    project_version: int
    items: list[ItemRows]
    clips: dict[str, ClipRows]
    #: Track id -> (timeline id, ``Type``: 0 video, 1 audio).
    tracks: dict[str, tuple[str, int]]
    #: Timeline id -> (frame rate bytes, ``MediaExtents`` bytes), from its sequence.
    sequences: dict[str, tuple[object, object]]
    project_rate: Any = None
    lockable_map_id: str | None = None
    skipped: list[str] = field(default_factory=list)


def _bad(why: str) -> ItemError:
    return ItemError("bad_template", why, _HINT)


def frames_text(value: object) -> int | None:
    """A whole frame count stored as text (``Start``, ``Duration``, ``In``); None for anything
    else (a sub-frame ``N|hex`` value, a number stored as a number, text that is not digits)."""
    if not isinstance(value, str) or not _FRAMES.match(value):
        return None
    return int(value)


#: How far from a whole frame a stored fraction may be and still be float noise on a whole
#: frame (the bound :data:`aar.plan.timeline_import.FRAME_ROUNDOFF` uses on the same values).
WHOLE_FRAME_NOISE = 1e-9
#: The reason of an item whose range is whole frames written in Resolve's ``N|hex`` form with
#: float noise in the fraction: not authored and not compared yet (the writer writes plain
#: frame counts; whether Resolve's form matters is not known).
NOISY_WHOLE_FRAMES = "whole frames stored with a fraction (N|hex, float noise): not compared yet"


def _range_shape(item: Row) -> str | None:
    """Why an item's ``Start``, ``Duration`` or ``In`` keeps it from being supported, or None:
    a value that is not a frame count (``unreadable``), a true fraction of a frame
    (``sub-frame``), or, when every such value is a whole frame stored as ``N|hex`` with float
    noise in the fraction, :data:`NOISY_WHOLE_FRAMES`."""
    noisy = False
    for column in ("Start", "Duration", "In"):
        value = item.get(column)
        if (column == "In" and value is None) or frames_text(value) is not None:
            continue
        if not isinstance(value, str):
            return f"unreadable {column}"
        try:
            _, fraction = blobs.parse_frames(value)
        except DrpError:
            return f"unreadable {column}"
        if fraction is None:
            return f"unreadable {column}"
        if WHOLE_FRAME_NOISE < fraction < 1 - WHOLE_FRAME_NOISE:
            return f"sub-frame {column} (a true fraction of a frame)"
        noisy = True
    return NOISY_WHOLE_FRAMES if noisy else None


@refusing("bad_template", "reading an item's links", _HINT, error=ItemError)
def links_of(item: Row) -> tuple[str, ...]:
    """The ids of the items ``item`` is linked to (``FieldsBlob`` #1.#1.#1, keys ``0``..), in
    key order; an empty tuple for an unlinked item."""
    fields = _fields(item)
    first = bt.pb_get(fields, 1, "item FieldsBlob")
    inner = bt.pb_decode(first[2], "item FieldsBlob #1")
    groups = [f for f in inner if f[0] == 1]
    if not groups:
        return ()
    if len(groups) > 1:
        raise _bad("item FieldsBlob #1.#1 occurs twice")
    for number, _wire, value in bt.pb_decode(groups[0][2], "item FieldsBlob #1.#1"):
        if number == 1:
            entries = bt.kd_decode(value, "item links")
            keys = [e[0] for e in entries]
            if sorted(keys, key=int) != [str(i) for i in range(len(keys))]:
                raise _bad(f"item links have keys {keys[:6]}, not 0..{len(keys) - 1}")
            by_key = {e[0]: e[3] for e in entries}
            out = tuple(by_key[str(i)] for i in range(len(keys)))
            if not all(isinstance(v, str) for v in out):
                raise _bad("an item link is not an id")
            return out
    return ()


def _fields(item: Row) -> bt.Fields:
    _, payload = bt.message_of(item.get("FieldsBlob"), "item FieldsBlob")
    return bt.pb_decode(payload, "item FieldsBlob")


#: The fields of an item's ``FieldsBlob`` the writer knows, by path, with their wire types.
#: ``#1.#1.#1`` is the link list; ``#1.#1.#3`` (repeated), ``#1.#1.#21`` and ``#1.#9`` are
#: written by later edits; ``#1.#1.#2`` holds Fairlight clip effects.
KNOWN_FIELDS = {
    "#1": 2,
    "#1.#1": 2,
    "#1.#1.#1": 2,
    "#1.#1.#2": 2,
    "#1.#1.#3": 2,
    "#1.#1.#21": 0,
    "#1.#4": 0,
    "#1.#9": 0,
    "#2": 2,
    "#15": 0,
}
_KIND_TOP = {"picture": "#2", "sound": "#15"}


@refusing("bad_template", "reading an item's FieldsBlob", _HINT, error=ItemError)
def field_paths(item: Row) -> list[tuple[str, int]]:
    """``(path, wire)`` of every field of an item's ``FieldsBlob`` down to the known depth."""
    out: list[tuple[str, int]] = []
    for number, wire, value in _fields(item):
        path = f"#{number}"
        out.append((path, wire))
        if path == "#1" and wire == 2:
            for n2, w2, v2 in bt.pb_decode(value, "item FieldsBlob #1"):
                p2 = f"#1.#{n2}"
                out.append((p2, w2))
                if p2 == "#1.#1" and w2 == 2:
                    out.extend(
                        (f"#1.#1.#{n3}", w3) for n3, w3, _ in bt.pb_decode(v2, "FieldsBlob #1.#1")
                    )
    return out


@refusing("bad_template", "reading an item's shape", _HINT, error=ItemError)
def shape_of(rows: ItemRows, clips: dict[str, ClipRows]) -> Shape:
    """Whether ``rows`` is an item the writer can author (and use as a template), and the later
    additions it carries. Never raises: a value that does not decode refuses the item."""
    item = rows.item
    kind = rows.kind
    if kind is None:
        return Shape(None, f"a {item.get('DbType')}, not a picture or sound of a media file")
    try:
        return _shape(rows, kind, clips)
    except MediaPoolError as exc:  # ItemError is one; the boundary makes every damage one
        return Shape(kind, f"does not decode: {exc.message}"[:200])


@refusing("bad_template", "reading an item's shape", _HINT, error=ItemError)
def _shape(rows: ItemRows, kind: str, clips: dict[str, ClipRows]) -> Shape:
    item = rows.item
    unknown = [c for c in item if c not in ITEM_COLUMNS]
    if unknown:
        return Shape(kind, f"has columns this writer does not know: {unknown[:4]}")
    if item.get("PrettyType") is not None:
        return Shape(kind, "a title or generator (PrettyType)")
    range_problem = _range_shape(item)
    if range_problem is not None:
        return Shape(kind, range_problem)
    timemap = item.get("MediaTimemapBA")
    if not isinstance(timemap, bytes) or len(timemap) != 9 or timemap[0] != 2:
        return Shape(kind, "retimed (a time map other than full speed)")
    if item.get("MediaRef") not in clips:
        return Shape(kind, "its media is not a file clip of the project (compound, multicam, ...)")
    clip = clips[str(item["MediaRef"])]
    if kind == "picture" and clip.video is None:
        return Shape(kind, "a picture item of a sound file")
    if kind == "sound":
        index = item.get("MediaTrackIdx")
        if not isinstance(index, int) or isinstance(index, bool) or index != 0:
            return Shape(kind, f"plays audio stream {index!r}; only the first was seen")
        if not clip.audio:
            return Shape(kind, "a sound item of a file without sound")
    elif item.get("MediaTrackIdx") is not None:
        return Shape(kind, "a picture item with an audio stream index")
    tags = {tag for column, tag in LATER_COLUMNS.items() if item.get(column) is not None}
    paths = field_paths(item)
    for path, wire in paths:
        if KNOWN_FIELDS.get(path) != wire:
            return Shape(kind, f"FieldsBlob field {path} (wire {wire}) is not known")
    seen = [p for p, _ in paths]
    for path in seen:
        if seen.count(path) > 1 and path != "#1.#1.#3":
            return Shape(kind, f"FieldsBlob field {path} occurs twice")
    tops = {p for p in seen if p.count("#") == 1}
    if tops != {"#1", _KIND_TOP[kind]}:
        return Shape(kind, f"FieldsBlob holds {sorted(tops)}, not #1 and {_KIND_TOP[kind]}")
    if "#1.#4" not in seen:
        return Shape(kind, "FieldsBlob has no #1.#4")
    if "#1.#1" in seen and not any(p.startswith("#1.#1.") for p in seen):
        return Shape(
            kind, "FieldsBlob #1.#1 is an empty group (seen once; what leaves it is unknown)"
        )
    if "#1.#1.#2" in seen:
        tags.add("effects")
    if any(p in seen for p in ("#1.#1.#3", "#1.#1.#21", "#1.#9")):
        tags.add("history")
    if any(p in seen for p in ("#1.#1.#2", "#1.#1.#3", "#1.#1.#21", "#1.#9")):
        tags.add("blob-later")
    if "#1.#1" in seen and "#1.#1.#1" not in seen:
        tags.add("unlinked-group")
    links_of(item)
    return Shape(kind, None, frozenset(tags))


@refusing("bad_template", "checking the template item", _HINT, error=ItemError)
def check_item(rows: object, clips: dict[str, ClipRows]) -> ItemRows:
    """``rows`` if it is one item of a supported shape whose join row places it on its track;
    a ``bad_template`` refusal otherwise."""
    if not isinstance(rows, ItemRows):
        raise _bad("the template is not an ItemRows")
    if not isinstance(rows.item, dict) or not isinstance(rows.link, dict):
        raise _bad("the template's item or join row is not a row")
    link = rows.link
    if set(link) != set(_JOIN):
        raise _bad("the template's join row does not have the join columns")
    if (
        link["DbPropertyName"] != "Items"
        or link["DbAssociate"] != rows.item.get("Sm2TiItem_id")
        or link["DbOwner"] != rows.item.get("Sm2TiTrack_id")
    ):
        raise _bad("the template's join row does not place it on its own track")
    shape = shape_of(rows, clips)
    if shape.refused is not None:
        raise _bad(f"the template item is not supported: {shape.refused}")
    assert rows.kind is not None  # shape_of refuses any other type
    constants = {**ITEM_CONSTANTS, "PreConformMediaExtents": PRECONFORM[rows.kind]}
    problem = constant_problem(rows.item, constants)
    if problem is not None:
        raise _bad(f"the template item {problem}")
    return rows


@refusing("bad_template", "checking a template's constant columns", _HINT, error=ItemError)
def constant_problem(item: Row, constants: dict[str, object]) -> str | None:
    """Why ``item`` is not a template whose copied constant columns hold ``constants`` (the
    same type and value, a float bit for bit, as the verifier compares them), naming the
    first column that differs; None when every one holds its value."""
    for column, want in constants.items():
        got = item.get(column)
        same = type(got) is type(want) and (
            struct.pack("<d", got) == struct.pack("<d", want)
            if isinstance(got, float) and isinstance(want, float)
            else got == want
        )
        if not same:
            shown = f"{len(got)} bytes" if isinstance(got, bytes) else repr(got)[:60]
            return f"holds {column} {shown}, not {want!r} (the value Resolve writes there)"
    return None


def _rows(lib: Any, table: str) -> list[Row]:
    columns = lib.columns[table]
    return [dict(zip(columns, values, strict=True)) for values in lib.rows[table]]


@refusing("bad_template", "reading the items", _HINT, error=ItemError, passthrough=(LibraryError,))
def read_items(path: str | Path, *, options: LibraryOptions | None = None) -> ItemSet:
    """Every item of a saved ``Project.db`` with the media clips and tracks they name. Raises
    :class:`aar.drp.library.LibraryError` if the project cannot be read."""
    lib = read_rows(path, ITEM_TABLES, options=options)
    tables = {t: _rows(lib, t) for t in ITEM_TABLES}
    clips, skipped = clips_from_rows({t: tables[t] for t in MEDIA_TABLES})
    links: dict[object, list[Row]] = defaultdict(list)
    for row in tables["Sm2TiItem_Sm2TiTrack"]:
        links[row.get("DbAssociate")].append(row)
    items: list[ItemRows] = []
    notes = [f"{s.name or s.media_id}: {s.reason}" for s in skipped]
    for row in tables["Sm2TiItem"]:
        found = links.get(row.get("Sm2TiItem_id"), [])
        if len(found) != 1:
            notes.append(f"item {row.get('Sm2TiItem_id')}: {len(found)} join rows")
            continue
        items.append(ItemRows(row, found[0]))
    timelines = {r.get("Sequence"): str(r.get("Sm2Timeline_id")) for r in tables["Sm2Timeline"]}
    tracks = {
        str(r["Sm2TiTrack_id"]): (timelines.get(r.get("Sequence"), ""), int(r.get("Type") or 0))
        for r in tables["Sm2TiTrack"]
    }
    sequences = {
        timelines[r["Sm2Sequence_id"]]: (r.get("FrameRate"), r.get("MediaExtents"))
        for r in tables["Sm2Sequence"]
        if r.get("Sm2Sequence_id") in timelines
    }
    maps = tables["BtLockableBlobMap"]
    return ItemSet(
        path=Path(path),
        project_version=lib.project_version,
        items=items,
        clips={c.media_id: c for c in clips},
        tracks=tracks,
        sequences=sequences,
        project_rate=project_frame_rate(tables["SM_Config"]),
        lockable_map_id=str(maps[0]["BtLockableBlobMap_id"]) if len(maps) == 1 else None,
        skipped=notes,
    )
