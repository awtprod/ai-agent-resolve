"""Template clips: the complete row set of each media clip in a project Resolve saved.

A media clip (a ``Sm2MpMedia`` row of type ``Sm2MpVideoClip`` or ``Sm2MpAudioClip``) is these
rows, found on Resolve 21.1 saves (docs/DESIGN.md, "Media-pool rows (S25c1)"):

* ``Sm2MpMedia``: the clip; ``Video`` names its ``BtVideoInfo``.
* ``BtVideoInfo``: one for a picture (none for a sound file), owner ``Sm2MpMedia_id``.
* ``BtAudioInfo``: one per embedded audio stream, owner ``Sm2MpMedia_id``; its id is also inside
  the clip's ``FieldsBlob`` (``ChannelVecBA`` ``MediaRef``, UTF-16BE inside zstd).
* ``BtAudioInfo_Sm2MpMedia``: one ``EmbeddedAudioVec`` join row per audio info, ``DbIndex`` its
  position.
* ``Sm2MpFolder_Sm2MpMedia``: the ``MediaVec`` join row placing the clip in its bin.
* ``BtLockableBlob`` of type ``BtMetadataLockableBlob``: the clip's metadata, owned by the video
  info (by the first audio info for a sound file); none for some kinds.
* ``SmPreset`` (``SmPTZRPreset``): an empty pan/tilt/zoom preset some clips get once viewed.

Rows are read through :func:`aar.drp.library.read_rows` (private snapshot, budgets; the live
file is never opened for writing). A table without a column the writer reads is refused
(``bad_template``). A clip whose rows are incomplete or inconsistent is not a template: it is
listed in ``skipped`` with the reason, never repaired. :func:`check_clip` runs the same checks on
rows a caller assembled, before anything is authored from them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.library import LibraryError, LibraryOptions, LibraryRows, read_rows
from aar.drp.mediapool.errors import MediaPoolError, refusing
from aar.media.probe import snap_rate

__all__ = [
    "CLIP_TYPES",
    "MEDIA_TABLES",
    "REQUIRED_COLUMNS",
    "ClipRows",
    "Row",
    "Skipped",
    "TemplateSet",
    "check_clip",
    "clips_from_rows",
    "project_frame_rate",
    "read_templates",
]

Row = dict[str, Any]
#: The tables a media clip's rows live in, in the order they are written.
MEDIA_TABLES = (
    "Sm2MpMedia",
    "BtVideoInfo",
    "BtAudioInfo",
    "BtAudioInfo_Sm2MpMedia",
    "Sm2MpFolder_Sm2MpMedia",
    "BtLockableBlob",
    "SmPreset",
)
#: The tables :func:`read_templates` reads: the media tables plus the project's config (its
#: frame rate) and lockable map.
TEMPLATE_TABLES = (*MEDIA_TABLES, "SM_Config", "BtLockableBlobMap")
CLIP_TYPES = frozenset({"Sm2MpVideoClip", "Sm2MpAudioClip"})
METADATA_TYPE = "BtMetadataLockableBlob"
_MAX_AUDIO = 16
_JOIN = ("DbOwner", "DbAssociate", "DbPropertyName", "DbIndex")
#: The columns the writer reads (or writes) in each media table; a table without one is refused.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "Sm2MpMedia": (
        "Sm2MpMedia_id",
        "DbType",
        "Name",
        "MpFolder",
        "Sm2MpFolder_id",
        "Video",
        "UniqueMediaPoolItemId",
        "PTZRPreset",
        "FieldsBlob",
    ),
    "BtVideoInfo": (
        "BtVideoInfo_id",
        "Sm2MpMedia_id",
        "Clip",
        "Time",
        "Geometry",
        "Proxy",
        "VideoMetadata",
        "MediaMetadata",
    ),
    "BtAudioInfo": ("BtAudioInfo_id", "Sm2MpMedia_id", "Clip", "TracksBA", "MediaMetadata"),
    "BtAudioInfo_Sm2MpMedia": _JOIN,
    "Sm2MpFolder_Sm2MpMedia": _JOIN,
    "BtLockableBlob": (
        "BtLockableBlob_id",
        "DbType",
        "BtLockableBlobMap_id",
        "FieldsBlob",
        "BlobOwner",
        "DbSavedTime",
    ),
    "SmPreset": ("SmPreset_id", "Sm2MpMedia_id"),
}
_SCALARS = (type(None), int, float, str, bytes)
_HINT = "use a template clip from a project Resolve saved, of the same kind"
_RATE_TEXT = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,6})?\Z")


@dataclass(frozen=True)
class ClipRows:
    """Every row one media clip owns. ``audio`` and ``audio_links`` are in ``DbIndex`` order."""

    media: Row
    video: Row | None
    audio: tuple[Row, ...]
    audio_links: tuple[Row, ...]
    folder_link: Row
    lockables: tuple[Row, ...] = ()
    presets: tuple[Row, ...] = ()

    @property
    @refusing("bad_template", "reading the clip's id", _HINT)
    def media_id(self) -> str:
        return str(self.media["Sm2MpMedia_id"])

    @property
    def is_audio_clip(self) -> bool:
        return self.media.get("DbType") == "Sm2MpAudioClip"

    def tables(self) -> dict[str, list[Row]]:
        """The rows per table, in :data:`MEDIA_TABLES` order (empty tables left out)."""
        out: dict[str, list[Row]] = {
            "Sm2MpMedia": [self.media],
            "BtVideoInfo": [self.video] if self.video is not None else [],
            "BtAudioInfo": list(self.audio),
            "BtAudioInfo_Sm2MpMedia": list(self.audio_links),
            "Sm2MpFolder_Sm2MpMedia": [self.folder_link],
            "BtLockableBlob": list(self.lockables),
            "SmPreset": list(self.presets),
        }
        return {t: rows for t, rows in out.items() if rows}

    @classmethod
    @refusing("bad_template", "reading a clip from its rows per table", _HINT)
    def from_tables(cls, rows: dict[str, tuple[Row, ...]]) -> ClipRows:
        """A clip from its rows per table (as :func:`aar.drp.mediapool.author.author` returns
        them)."""
        return cls(
            media=rows["Sm2MpMedia"][0],
            video=rows["BtVideoInfo"][0] if "BtVideoInfo" in rows else None,
            audio=tuple(rows.get("BtAudioInfo", ())),
            audio_links=tuple(rows.get("BtAudioInfo_Sm2MpMedia", ())),
            folder_link=rows["Sm2MpFolder_Sm2MpMedia"][0],
            lockables=tuple(rows.get("BtLockableBlob", ())),
            presets=tuple(rows.get("SmPreset", ())),
        )

    @refusing("bad_template", "reading the ids of the clip's rows", _HINT)
    def ids(self) -> set[str]:
        """The ids of the rows the clip owns (not the bin, map or other rows it points to)."""
        found = {self.media_id, str(self.media.get("UniqueMediaPoolItemId") or "")}
        if self.video is not None:
            found.add(str(self.video["BtVideoInfo_id"]))
        found.update(str(a["BtAudioInfo_id"]) for a in self.audio)
        found.update(str(r["BtLockableBlob_id"]) for r in self.lockables)
        found.update(str(r["SmPreset_id"]) for r in self.presets)
        found.discard("")
        return found


@dataclass(frozen=True)
class Skipped:
    """A clip that is not a template, and why."""

    media_id: str
    name: str
    reason: str


@dataclass(frozen=True)
class TemplateSet:
    """The template clips of one saved project."""

    path: Path
    project_version: int
    clips: tuple[ClipRows, ...]
    skipped: tuple[Skipped, ...]
    #: The project's frame rate (``SM_Config`` ``SetupBA`` field 69), or None if it cannot be read.
    project_rate: Fraction | None
    lockable_map_id: str | None


def _rows(lib: LibraryRows, table: str) -> list[Row]:
    columns = lib.columns[table]
    return [dict(zip(columns, values, strict=True)) for values in lib.rows[table]]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def check_columns(table: str, columns: tuple[str, ...] | list[str]) -> None:
    """Refuse a media table that lacks a column the writer reads."""
    missing = [c for c in REQUIRED_COLUMNS.get(table, ()) if c not in columns]
    if missing:
        raise MediaPoolError(
            "bad_template",
            f"the {table} table has no column {missing[0]}",
            "use a project Resolve saved (disk library, version 17)",
        )


def _check_rows(table: str, rows: list[Row]) -> None:
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(k, str) for k in row):
            raise _bad(f"a {table} row is not a mapping of column names")
        check_columns(table, list(row))
        for column, value in row.items():
            if not isinstance(value, _SCALARS):
                raise _bad(f"{table}.{column} holds a {type(value).__name__}, not a column value")


@refusing("bad_template", "reading the clips of the media tables", _HINT)
def clips_from_rows(tables: dict[str, list[Row]]) -> tuple[list[ClipRows], list[Skipped]]:
    """Group the rows of :data:`MEDIA_TABLES` by clip, checking each clip's set is complete and
    consistent. Returns the templates and the clips that are not, with the reason. A missing
    table, a row without a column the writer reads or a value no database column holds is a
    ``bad_template`` refusal."""
    for table in MEDIA_TABLES:
        if table not in tables:
            raise MediaPoolError(
                "bad_template",
                f"the project has no {table} table",
                "use a project Resolve saved (disk library, version 17)",
            )
        _check_rows(table, tables[table])
    # Rows are grouped by the exact id value: an id that is not text names no clip.
    by_owner: dict[object, dict[str, list[Row]]] = defaultdict(lambda: defaultdict(list))
    for row in tables["BtVideoInfo"]:
        by_owner[row["Sm2MpMedia_id"]]["video"].append(row)
    for row in tables["BtAudioInfo"]:
        by_owner[row["Sm2MpMedia_id"]]["audio"].append(row)
    for row in tables["BtAudioInfo_Sm2MpMedia"]:
        by_owner[row["DbOwner"]]["audio_links"].append(row)
    for row in tables["Sm2MpFolder_Sm2MpMedia"]:
        by_owner[row["DbAssociate"]]["folder_links"].append(row)
    for row in tables["SmPreset"]:
        by_owner[row["Sm2MpMedia_id"]]["presets"].append(row)
    lockables: dict[object, list[Row]] = defaultdict(list)
    for row in tables["BtLockableBlob"]:
        lockables[row["BlobOwner"]].append(row)
    clips: list[ClipRows] = []
    skipped: list[Skipped] = []
    for media in tables["Sm2MpMedia"]:
        mid = _text(media.get("Sm2MpMedia_id")) or ""
        name = str(media.get("Name") or "")
        if media.get("DbType") not in CLIP_TYPES:
            skipped.append(Skipped(mid, name, f"a {media.get('DbType')}, not a media file clip"))
            continue
        try:
            clips.append(_clip(media, by_owner.get(mid, {}), lockables))
        except MediaPoolError as exc:
            skipped.append(Skipped(mid, name, exc.message))
    return clips, skipped


def _clip(media: Row, owned: dict[str, list[Row]], lockables: dict[object, list[Row]]) -> ClipRows:
    mid = _text(media.get("Sm2MpMedia_id"))
    if mid is None:
        raise _bad("the clip has no Sm2MpMedia_id")
    video_rows = owned.get("video", [])
    audio_rows = owned.get("audio", [])
    links = owned.get("audio_links", [])
    folders = owned.get("folder_links", [])
    if media.get("DbType") == "Sm2MpVideoClip":
        if len(video_rows) != 1:
            raise _bad(f"a video clip needs one BtVideoInfo row, it has {len(video_rows)}")
        video: Row | None = video_rows[0]
        if _text(video["BtVideoInfo_id"]) is None or media["Video"] != video["BtVideoInfo_id"]:
            raise _bad("Sm2MpMedia.Video does not name its BtVideoInfo row")
    else:
        if video_rows or media.get("Video") is not None:
            raise _bad("a sound clip has a BtVideoInfo row")
        video = None
        if not audio_rows:
            raise _bad("a sound clip has no BtAudioInfo row")
    if len(audio_rows) > _MAX_AUDIO:
        raise _bad(f"the clip has {len(audio_rows)} audio infos")
    if any(link.get("DbPropertyName") != "EmbeddedAudioVec" for link in links):
        raise _bad("a BtAudioInfo_Sm2MpMedia row has another property than EmbeddedAudioVec")
    by_id = {a.get("BtAudioInfo_id"): a for a in audio_rows}
    if len(by_id) != len(audio_rows) or None in by_id:
        raise _bad("two BtAudioInfo rows share an id, or one has none")
    if any(type(r.get("DbIndex")) is not int for r in links):
        raise _bad("an EmbeddedAudioVec join row's DbIndex is not a whole number")
    order = sorted(links, key=lambda r: r["DbIndex"])
    if [r["DbIndex"] for r in order] != list(range(len(audio_rows))) or len(links) != len(
        audio_rows
    ):
        raise _bad("the EmbeddedAudioVec join rows are missing, repeated or out of order")
    if {r.get("DbAssociate") for r in order} != set(by_id):
        raise _bad("the EmbeddedAudioVec join rows do not name the clip's audio infos")
    audio = tuple(by_id[r["DbAssociate"]] for r in order)
    if len(folders) != 1:
        raise _bad(f"the clip has {len(folders)} MediaVec join rows, not one")
    folder = folders[0]
    if (
        folder.get("DbPropertyName") != "MediaVec"
        or folder.get("DbOwner") != media.get("Sm2MpFolder_id")
        or media.get("MpFolder") != media.get("Sm2MpFolder_id")
    ):
        raise _bad("the clip's bin join row does not agree with its MpFolder")
    if any(_text(a["BtAudioInfo_id"]) is None for a in audio):
        raise _bad("a BtAudioInfo row's id is not a text")
    owner = video["BtVideoInfo_id"] if video is not None else audio[0]["BtAudioInfo_id"]
    found: list[Row] = []
    for key in {mid, *(a["BtAudioInfo_id"] for a in audio)} | (
        {video["BtVideoInfo_id"]} if video is not None else set()
    ):
        found.extend(lockables.get(key, []))
    if any(r.get("DbType") != METADATA_TYPE or r.get("BlobOwner") != owner for r in found):
        raise _bad("the clip owns a lockable blob this writer does not know")
    if len(found) > 1:
        raise _bad("the clip has more than one metadata blob")
    presets = owned.get("presets", [])
    preset_ref = media.get("PTZRPreset")
    if preset_ref is not None and preset_ref not in {p.get("SmPreset_id") for p in presets}:
        raise _bad("Sm2MpMedia.PTZRPreset names no preset of the clip")
    return ClipRows(
        media=media,
        video=video,
        audio=audio,
        audio_links=tuple(order),
        folder_link=folder,
        lockables=tuple(found),
        presets=tuple(presets),
    )


def _bad(why: str) -> MediaPoolError:
    return MediaPoolError("bad_template", why, "choose another template clip")


@refusing("bad_template", "checking the template clip", _HINT)
def check_clip(clip: object) -> ClipRows:
    """``clip`` if it is one complete, consistent clip (the checks :func:`clips_from_rows` makes,
    on exactly these rows, in this order); a ``bad_template`` refusal otherwise. Rows a caller
    assembled are checked with this before anything is authored from them."""
    if not isinstance(clip, ClipRows):
        raise _bad("the template is not a ClipRows")
    parts = (clip.audio, clip.audio_links, clip.lockables, clip.presets)
    if not all(isinstance(p, tuple) for p in parts):
        raise _bad("the template's rows are not tuples of rows")
    if not isinstance(clip.media, dict) or not isinstance(clip.folder_link, dict):
        raise _bad("the template's clip or bin join row is not a row")
    tables: dict[str, list[Row]] = {
        "Sm2MpMedia": [clip.media],
        "BtVideoInfo": [] if clip.video is None else [clip.video],
        "BtAudioInfo": list(clip.audio),
        "BtAudioInfo_Sm2MpMedia": list(clip.audio_links),
        "Sm2MpFolder_Sm2MpMedia": [clip.folder_link],
        "BtLockableBlob": list(clip.lockables),
        "SmPreset": list(clip.presets),
    }
    clips, skipped = clips_from_rows(tables)
    if skipped:
        raise _bad(skipped[0].reason)
    if len(clips) != 1 or clips[0] != clip:
        raise _bad("the template's rows are not exactly one clip's rows in their stored order")
    return clip


def project_frame_rate(config_rows: list[Row]) -> Fraction | None:
    """The project's timeline frame rate: ``SM_Config.FieldsBlob`` ``SetupBA`` field 69 (text
    such as ``29.97`` or ``23.976``), snapped to the standard rate it names. None when it cannot
    be read (checked on two Resolve 21.1 saves only)."""
    if len(config_rows) != 1 or not isinstance(config_rows[0].get("FieldsBlob"), bytes):
        return None
    try:
        kd = blobs.decode_keyed_dict(config_rows[0]["FieldsBlob"], strict=True)
        setup = kd.get("SetupBA")
        if not isinstance(setup, bytes):
            return None
        fields = blobs.decode_protobuf(
            blobs.unwrap_message(setup, max_output=1024 * 1024), nested_depth=0
        )
    except DrpError:
        return None
    texts = [f.value for f in fields if f.number == 69 and isinstance(f.value, bytes)]
    if len(texts) != 1:
        return None
    try:
        text = texts[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if not _RATE_TEXT.match(text):
        return None  # Fraction() would also read underscores, exponents and spaces
    return snap_rate(Fraction(text))


@refusing("bad_template", "reading the template clips", _HINT, passthrough=(LibraryError,))
def read_templates(path: str | Path, *, options: LibraryOptions | None = None) -> TemplateSet:
    """The template clips of a saved ``Project.db``. Raises :class:`aar.drp.library.LibraryError`
    if the project cannot be read, :class:`MediaPoolError` if it lacks the media tables."""
    lib = read_rows(path, TEMPLATE_TABLES, options=options)
    for table in MEDIA_TABLES:
        check_columns(table, lib.columns[table])
    tables = {t: _rows(lib, t) for t in TEMPLATE_TABLES}
    clips, skipped = clips_from_rows(tables)
    maps = tables["BtLockableBlobMap"]
    return TemplateSet(
        path=Path(path),
        project_version=lib.project_version,
        clips=tuple(clips),
        skipped=tuple(skipped),
        project_rate=project_frame_rate(tables["SM_Config"]),
        lockable_map_id=_text(maps[0].get("BtLockableBlobMap_id")) if len(maps) == 1 else None,
    )
