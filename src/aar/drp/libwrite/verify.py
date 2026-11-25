"""An independent read of a written project: is everything in the file what the writer meant?

:func:`verify_written` runs on the staged file after the writer's own read-back
(:func:`aar.drp.libwrite.check.check_written`) and before anything is placed. It does not use
the code that wrote the rows (:func:`aar.drp.items.author_item`,
:func:`aar.drp.items.media_view`, :func:`aar.drp.libwrite.cut.author_cut`,
:func:`aar.drp.libwrite.write.sequence_length`, :mod:`aar.drp.mediapool`'s authoring): from the
layout (the :class:`Cut` entries the caller authored the items from), from each media clip's own
rows *as written* (decoded here with the shared blob decoders only) and from the seed's own rows
(:attr:`Seed.digests`, taken when the seed was read), it recomputes what the file must hold, by
the field map of docs/DESIGN.md ("Timeline items (S25c2)"), and compares it with every row read
back:

* every column of every item (:data:`ITEM_RULES`, one rule per column of the writer's closed
  list :data:`aar.drp.items.rows.ITEM_COLUMNS`; a column outside it must be empty): recomputed
  from the layout and the clip's rows (type, track, place in the track's ``Items`` vector,
  ``Start``, ``Duration``, ``In``, ``Name``, ``MediaRef`` and the clip's file path, start time,
  time map, frame rate, stream index and channels; ``FieldsBlob`` byte for byte: the link to
  the other half of its layout entry and nothing else), empty as on an item Resolve has just
  made (a grade, Fusion, effects, markers, import metadata, item metadata), or a stated
  constant (every plain item of the corpus holds it);
* every column of every sound transition (:data:`TRANSITION_RULES`, one rule per column of the
  same closed list), recomputed from the cut's transitions (:class:`CutTransition`): its join
  (the outgoing entry's sound ends where exactly one other entry's sound starts, neither a
  split edit; two transitions on one clip do not overlap inside it, refused
  ``transitions_overlap``), its ``Start`` (the cut minus Resolve's centring,
  ``(length + 1) // 2`` frames), ``Duration``, ``AlignmentType`` and ``Position``,
  ``PrettyType`` (the kind's stored name,
  from the verifier's own table), ``FieldsBlob`` byte for byte (one field #2 holding the bytes
  every sound transition holds, which inflate to its 44-byte value, framed plain, version 2),
  its place in the track's ``Items`` vector (record order, a transition before a clip that
  starts on its frame), and that the two clips hold the media it plays (frames before the
  incoming clip's in point, frames of sound past the outgoing clip's out point), from the
  clips' rows as written;
* every media clip the writer adds: one media row, its picture info and its sound infos (each
  naming this clip, joined in order, as many as were authored), its one bin entry, and every
  ``Clip`` blob naming the same folder and file (the media's name);
* every other row of every table: what the seed held, with the project's new id, its name, the
  timeline's name (renamed or kept) and the sequence's ``MediaExtents`` (the seed's start, and
  the end of the last item over the rate minus the start) where the writer changes them, plus
  exactly the authored media rows; the same tables and columns as the seed, the same number of
  rows per table as the seed had plus what was added.

A difference is ``self_check_failed`` naming the field and the item or table.
"""

from __future__ import annotations

import math
import re
import struct
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.items.rows import ITEM_COLUMNS
from aar.drp.library import LibraryOptions, read_rows
from aar.drp.libwrite.cut import Cut, CutTransition
from aar.drp.libwrite.errors import ProjectWriteError, refusing
from aar.drp.libwrite.seed import (
    REPLACED_TABLES,
    WRITER_CELLS,
    Seed,
    row_digest,
    table_names,
)
from aar.drp.mediapool.templates import MEDIA_TABLES, ClipRows
from aar.drp.modify import encode
from aar.media.probe import timecode_frames

__all__ = [
    "ITEM_RULES",
    "TRANSITION_RULES",
    "Expected",
    "ExpectedTransition",
    "expected_items",
    "expected_transitions",
    "verify_written",
]

_HINT = "this is a writer bug: the project was not placed; report the seed and the plan"
_TABLES = (
    "Sm2TiItem",
    "Sm2TiItem_Sm2TiTrack",
    "Sm2Sequence",
    "Sm2MpMedia",
    "BtVideoInfo",
    "BtAudioInfo",
    "BtAudioInfo_Sm2MpMedia",
)
_TYPES = {"picture": "Sm2TiVideoClip", "sound": "Sm2TiAudioClip"}
_MAX_OUT = 1024 * 1024
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")

#: An item column recomputed from the layout and the clip's rows (in :func:`_derived`).
DERIVED = "derived"
#: An item column a later edit fills in (a grade, Fusion, effects, import metadata, markers,
#: item metadata): empty on an item Resolve has just made, and on every item written.
FRESH = "fresh"
#: ``FieldsBlob`` #2 of every plain picture item of the corpus (475 of 475), and the other
#: constants of an item's ``FieldsBlob``: ``#1.#4`` (all 667) and a sound's ``#15`` (192).
PICTURE_TAIL = bytes.fromhex("000000000000000600000000ffffffff")
_FIELDS_ONE_FOUR = 1
_SOUND_TAIL = 4
#: ``PreConformMediaExtents`` of every picture item of the corpus (4,434 of 4,434).
PICTURE_PRECONFORM = bytes.fromhex("00000100000030c20000010000003042")
#: What the verifier requires of each column of an item: :data:`DERIVED`, :data:`FRESH`, or a
#: constant (``("constant", value)``, the value every plain item of the corpus holds). One
#: entry per column of :data:`aar.drp.items.rows.ITEM_COLUMNS` (a test pins that).
ITEM_RULES: dict[str, object] = {
    "Sm2TiItem_id": DERIVED,
    "DbType": DERIVED,
    "Name": DERIVED,
    "Start": DERIVED,
    "Duration": DERIVED,
    "In": DERIVED,
    "MediaRef": DERIVED,
    "MediaStartTime": DERIVED,
    "MediaFilePath": DERIVED,
    "MediaTimemapBA": DERIVED,
    "MediaTrackIdx": DERIVED,
    "MediaFrameRate": DERIVED,
    "Sm2TiTrack_id": DERIVED,
    "VirtualAudioTrackBA": DERIVED,
    "PreConformMediaExtents": DERIVED,
    "FieldsBlob": DERIVED,
    "pLmVerTable": FRESH,
    "CompositionTable": FRESH,
    "EffectFiltersBA": FRESH,
    "ImportExportMetadataBA": FRESH,
    "MarkersBA": FRESH,
    "MediaMetadata": FRESH,
    "Track": ("constant", None),
    "LinkedItemSync": ("constant", None),
    "UiMemento": ("constant", 0),
    "Flags": ("constant", 0),
    "PriorityIndex": ("constant", 0),
    "MediaReelNumber": ("constant", None),
    "Group": ("constant", None),
    "pAuxLmVerTable": ("constant", None),
    "Thumbnail": ("constant", None),
    "ThumbnailDirtyFlag": ("constant", 1),
    "RenderTextEnabled": ("constant", 1),
    "RenderTextGanged": ("constant", 1),
    "RenderTextPrefixed": ("constant", 1),
    "TextRenderItemVec": ("constant", None),
    "LastChangedTime": ("constant", 0),
    "LastRenderedTime": ("constant", 0),
    "IsMarkedForCaching": ("constant", 0),
    "IsForceConformed": ("constant", 1),
    "MatchConflictState": ("constant", 0),
    "PrettyType": ("constant", None),
    "Resolution": ("constant", None),
    "IsPreConformed": ("constant", 0),
    "MixedFrameRateAlignment": ("constant", 0.0),
    "UseOppositeSrcForLeftEye": ("constant", 0),
    "UseOppositeSrcForRightEye": ("constant", 0),
    "WasDisbanded": ("constant", 0),
    "CurrentSelectorIdx": ("constant", 0),
    "AlignmentType": ("constant", None),
    "Position": ("constant", None),
    "RenderCacheBA": ("constant", None),
    "ClipGroup": ("constant", None),
    "BlobLockSysId": ("constant", ""),
    "OriginalClip": ("constant", None),
    "Sm2TiItem_Owner_id": ("constant", None),
    "CompTableLockSysId": ("constant", ""),
    "VersionTableLockSysId": ("constant", ""),
}


#: The verifier's own table of the sound transitions it accepts: Resolve's menu name -> the
#: ``PrettyType`` it stores (kept apart from the writer's on purpose; a test compares them).
TRANSITION_NAMES = {"Cross Fade 0 dB": "Cross Fade 0DB", "Cross Fade +3 dB": "Cross Fade +3DB"}
#: ``FieldsBlob`` #2 of every sound transition of the corpus (854 of 854), decompressed, and
#: as stored (``qCompress``: the length, then the zlib stream; the same bytes on all 854).
TRANSITION_PAYLOAD = bytes([3, 0, 1]) + bytes(41)
TRANSITION_FIELD = bytes.fromhex("0000002c") + bytes.fromhex("789c636660642016000000da0005")
_TRANSITION = "Sm2TiTransition"
#: What the verifier requires of each column of a sound transition, as :data:`ITEM_RULES`
#: does for a clip: :data:`DERIVED` (recomputed from the cut), or the constant every sound
#: transition of the corpus holds. One entry per column of ``ITEM_COLUMNS`` (a test pins it).
TRANSITION_RULES: dict[str, object] = {
    "Sm2TiItem_id": DERIVED,
    "DbType": DERIVED,
    "Start": DERIVED,
    "Duration": DERIVED,
    "Sm2TiTrack_id": DERIVED,
    "AlignmentType": DERIVED,
    "Position": DERIVED,
    "PrettyType": DERIVED,
    "FieldsBlob": DERIVED,
    "Name": ("constant", "Transition"),
    "Track": ("constant", None),
    "LinkedItemSync": ("constant", None),
    "MarkersBA": ("constant", None),
    "UiMemento": ("constant", 0),
    "Flags": ("constant", 0),
    "PriorityIndex": ("constant", 0),
    "In": ("constant", None),
    "MediaRef": ("constant", None),
    "MediaStartTime": ("constant", None),
    "MediaFilePath": ("constant", None),
    "MediaReelNumber": ("constant", None),
    "MediaTimemapBA": ("constant", None),
    "Group": ("constant", None),
    "pLmVerTable": ("constant", None),
    "pAuxLmVerTable": ("constant", None),
    "Thumbnail": ("constant", None),
    "ThumbnailDirtyFlag": ("constant", 1),
    "RenderTextEnabled": ("constant", 1),
    "RenderTextGanged": ("constant", 1),
    "RenderTextPrefixed": ("constant", 1),
    "TextRenderItemVec": ("constant", None),
    "LastChangedTime": ("constant", None),
    "LastRenderedTime": ("constant", None),
    "IsMarkedForCaching": ("constant", None),
    "IsForceConformed": ("constant", 1),
    "MatchConflictState": ("constant", 0),
    "MediaTrackIdx": ("constant", None),
    "Resolution": ("constant", None),
    "EffectFiltersBA": ("constant", None),
    "IsPreConformed": ("constant", 0),
    "PreConformMediaExtents": ("constant", None),
    "ImportExportMetadataBA": ("constant", None),
    "MediaFrameRate": ("constant", None),
    "VirtualAudioTrackBA": ("constant", None),
    "MixedFrameRateAlignment": ("constant", 0.0),
    "UseOppositeSrcForLeftEye": ("constant", 0),
    "UseOppositeSrcForRightEye": ("constant", 0),
    "WasDisbanded": ("constant", 0),
    "CurrentSelectorIdx": ("constant", 0),
    "RenderCacheBA": ("constant", None),
    "ClipGroup": ("constant", None),
    "MediaMetadata": ("constant", None),
    "CompositionTable": ("constant", None),
    "BlobLockSysId": ("constant", ""),
    "OriginalClip": ("constant", None),
    "Sm2TiItem_Owner_id": ("constant", None),
    "CompTableLockSysId": ("constant", ""),
    "VersionTableLockSysId": ("constant", ""),
}


@dataclass(frozen=True)
class ExpectedTransition:
    """One sound transition the cut asks for: its cut (record frame from the timeline's start),
    length, the frames before the cut, its stored name, and the two entries it joins."""

    number: int
    cut: int
    frames: int
    before: int
    pretty: str
    outgoing: int
    incoming: int


@refusing("self_check_failed", "listing the transitions a cut asks for", _HINT)
def expected_transitions(
    cuts: Sequence[Cut], transitions: Sequence[CutTransition]
) -> list[ExpectedTransition]:
    """The sound transitions ``transitions`` ask for, each on a join of two entries of
    ``cuts`` (recomputed here: the outgoing entry's sound ends where exactly one other entry's
    sound starts, and neither is a split edit), centred as Resolve centres one."""
    out: list[ExpectedTransition] = []
    for number, tx in enumerate(transitions):
        where = f"transition {number}"
        if not isinstance(tx, CutTransition) or not isinstance(tx.after, int):
            raise _fail(f"{where} is not a transition of the cut", field="transitions")
        if tx.alignment != "centre" or tx.name not in TRANSITION_NAMES:
            raise _fail(f"{where} is a {tx.name!r} aligned {tx.alignment!r}", field="PrettyType")
        if not 0 <= tx.after < len(cuts) or not isinstance(tx.frames, int) or tx.frames < 1:
            raise _fail(f"{where} names no cut entry or has no length", field="transitions")
        out_cut = cuts[tx.after]
        end = out_cut.record_in + out_cut.duration
        incoming = [
            i for i, c in enumerate(cuts) if i != tx.after and c.sound and c.record_in == end
        ]
        if not out_cut.sound or len(incoming) != 1:
            raise _fail(f"{where} is not on a join of two sounds", field="transitions")
        for c in (out_cut, cuts[incoming[0]]):
            if any(v is not None for v in (c.sound_record_in, c.sound_src_in, c.sound_duration)):
                raise _fail(f"{where} joins a split edit", field="transitions")
        out.append(
            ExpectedTransition(
                number,
                end,
                tx.frames,
                (tx.frames + 1) // 2,
                TRANSITION_NAMES[tx.name],
                tx.after,
                incoming[0],
            )
        )
    if len({t.cut for t in out}) != len(out):
        raise _fail("two transitions sit on one cut", field="transitions")
    # On each entry, the frames after the cut of the transition into it and before the cut of
    # the one out of it fit in its length (the plan export's rule and code).
    for index, cut in enumerate(cuts):
        into = sum(t.frames - t.before for t in out if t.incoming == index)
        out_of = sum(t.before for t in out if t.outgoing == index)
        if into and out_of and into + out_of > cut.duration:
            raise ProjectWriteError(
                "transitions_overlap",
                f"the transitions into and out of cut entry {index} need {into + out_of} "
                f"frames of its {cut.duration}, so they overlap",
                _HINT,
                field="transitions",
            )
    return out


@dataclass(frozen=True)
class Expected:
    """One item the layout asks for: entry ``group`` of the cut, its ``kind``, record frames
    from the timeline's start, source in, and the media clip it plays."""

    group: int
    kind: str
    record_in: int
    duration: int
    src_in: int
    media_id: str


@refusing("bad_input", "listing the items a cut asks for", "check the cut list")
def expected_items(cuts: Sequence[Cut]) -> list[Expected]:
    """The items a cut list asks for: a picture when the entry has one, a sound when it has
    one (with its own range for a split edit)."""
    out: list[Expected] = []
    for group, cut in enumerate(cuts):
        if cut.picture:
            out.append(
                Expected(
                    group, "picture", cut.record_in, cut.duration, cut.src_in, cut.clip.media_id
                )
            )
        if cut.sound:
            out.append(
                Expected(
                    group,
                    "sound",
                    cut.record_in if cut.sound_record_in is None else cut.sound_record_in,
                    cut.duration if cut.sound_duration is None else cut.sound_duration,
                    cut.src_in if cut.sound_src_in is None else cut.sound_src_in,
                    cut.clip.media_id,
                )
            )
    return out


def _fail(message: str, **details: object) -> ProjectWriteError:
    return ProjectWriteError("self_check_failed", message, _HINT, **details)


def _keyed(value: object, what: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, bytes):
        raise _fail(f"{what} is not a keyed dict")
    out: dict[str, Any] = {}
    for entry in blobs.decode_keyed_dict(value, strict=True).entries:
        if entry.key in out:
            raise _fail(f"{what} holds {entry.key!r} twice")
        out[entry.key] = entry.value
    return out


def _message(value: object, what: str) -> list[blobs.PbField]:
    if not isinstance(value, bytes):
        raise _fail(f"{what} is not a blob")
    payload = blobs.unwrap_message(value, max_output=_MAX_OUT)
    return blobs.decode_protobuf(payload, nested_depth=0)


def _one(fields: list[blobs.PbField], number: int, what: str) -> blobs.PbField:
    found = [f for f in fields if f.number == number]
    if len(found) != 1:
        raise _fail(f"{what} has {len(found)} field(s) #{number}, not one")
    return found[0]


@dataclass(frozen=True)
class _Media:
    """What an item of a media clip must hold, from the clip's rows as written."""

    path: str
    rate_bytes: bytes
    start: float
    timemap: bytes
    channels: int | None
    #: The picture's frames (None for a sound file).
    frames: int | None
    #: The first audio track's samples in frames at the item's rate, rounded up (None without
    #: sound): the most a sound item may read.
    sound_frames: int | None = None


def _media(media_id: str, tables: dict[str, list[dict[str, Any]]]) -> _Media:
    rows = [r for r in tables["Sm2MpMedia"] if r.get("Sm2MpMedia_id") == media_id]
    if len(rows) != 1:
        raise _fail(f"the media clip {media_id} is in the written project {len(rows)} times")
    media = rows[0]
    videos = [r for r in tables["BtVideoInfo"] if r.get("Sm2MpMedia_id") == media_id]
    links = sorted(
        (r for r in tables["BtAudioInfo_Sm2MpMedia"] if r.get("DbOwner") == media_id),
        key=lambda r: int(r.get("DbIndex") or 0),
    )
    audio_by_id = {r.get("BtAudioInfo_id"): r for r in tables["BtAudioInfo"]}
    audios = [
        audio_by_id[r.get("DbAssociate")] for r in links if r.get("DbAssociate") in audio_by_id
    ]
    if len(videos) > 1:
        raise _fail(
            f"the media clip {media_id} has {len(videos)} picture infos", field="BtVideoInfo"
        )
    if len(audios) != len(links) or (not videos and not audios):
        raise _fail(
            f"the media clip {media_id} does not have its video and audio rows",
            field="BtAudioInfo_Sm2MpMedia",
        )
    first_track: dict[str, Any] | None = None
    if audios:
        tracks = _keyed(audios[0].get("TracksBA"), "the clip's TracksBA")
        if len(tracks) != 1:
            raise _fail(f"the clip's first audio info has {len(tracks)} tracks, not one")
        first_track = _keyed(next(iter(tracks.values())), "the clip's TracksBA track")
    source = videos[0] if videos else audios[0]
    clip = _message(source.get("Clip"), "the clip's Clip")
    folder, name = _one(clip, 1, "Clip").value, _one(clip, 2, "Clip").value
    if not isinstance(folder, bytes) or not isinstance(name, bytes):
        raise _fail("the clip's Clip folder or name is not text")
    path = folder.decode("utf-8") + "\\" + name.decode("utf-8")
    channels: int | None = None
    sound: Fraction | None = None
    if first_track is not None:
        count = first_track.get("NumChannels")
        channels = count if isinstance(count, int) and not isinstance(count, bool) else None
        samples, sample_rate = first_track.get("Duration"), first_track.get("SampleRate")
        if (
            not isinstance(samples, int)
            or not isinstance(sample_rate, int)
            or isinstance(samples, bool)
            or isinstance(sample_rate, bool)
            or samples < 0
            or sample_rate <= 0
        ):
            raise _fail("the clip's first audio track has no length or sample rate")
        sound = Fraction(samples, sample_rate)

    def sound_frames(rate: float) -> int | None:
        # The rate as the standard rate it stands for (24000/1001, not the double's expansion).
        return None if sound is None else math.ceil(sound * Fraction(rate).limit_denominator(1001))

    if videos:
        time = _keyed(videos[0].get("Time"), "the clip's Time")
        rate_bytes = time.get("FrameRate")
        frames = time.get("NumFrames")
        timecode = time.get("Timecode")
        if not isinstance(rate_bytes, bytes) or len(rate_bytes) != 16:
            raise _fail("the clip's Time.FrameRate is not 16 bytes")
        if not isinstance(frames, int) or isinstance(frames, bool):
            raise _fail("the clip's Time.NumFrames is not a whole number")
        (rate,) = struct.unpack("<d", rate_bytes[:8])
        start = 0.0
        if isinstance(timecode, str):
            start = timecode_frames(timecode, Fraction(rate).limit_denominator(1001)) / rate
        last = frames / rate - 1 / rate
        timemap = b"\x02" + struct.pack(">d", last)
        return _Media(path, rate_bytes, start, timemap, channels, frames, sound_frames(rate))
    assert first_track is not None
    import_rate = _one(
        _sub(_message(media.get("FieldsBlob"), "the clip's FieldsBlob")),
        1,
        "the clip's FieldsBlob #2",
    )
    if import_rate.wire != 1 or not isinstance(import_rate.value, bytes):
        raise _fail("the clip's FieldsBlob #2.#1 is not a double")
    rate_bytes = import_rate.value + bytes(8)
    (rate,) = struct.unpack("<d", import_rate.value)
    samples, sample_rate = first_track["Duration"], first_track["SampleRate"]
    start = first_track.get("StartTime")
    if not isinstance(start, float):
        raise _fail("the clip's first audio track has no StartTime")
    last = (samples / sample_rate) * rate / rate - 1 / rate
    timemap = b"\x02" + struct.pack(">d", last)
    return _Media(path, rate_bytes, start, timemap, channels, None, sound_frames(rate))


def _sub(fields: list[blobs.PbField]) -> list[blobs.PbField]:
    inner = _one(fields, 2, "the clip's FieldsBlob")
    if inner.wire != 2 or not isinstance(inner.value, bytes):
        raise _fail("the clip's FieldsBlob #2 is not a message")
    return blobs.decode_protobuf(inner.value, nested_depth=0)


def _links(item: dict[str, Any]) -> list[str]:
    """The ids an item's ``FieldsBlob`` #1.#1.#1 names, by key ``0``, ``1``..."""
    top = _message(item.get("FieldsBlob"), "the item's FieldsBlob")
    first = _one(top, 1, "the item's FieldsBlob")
    if not isinstance(first.value, bytes):
        raise _fail("the item's FieldsBlob #1 is not a message")
    groups = [f for f in blobs.decode_protobuf(first.value, nested_depth=0) if f.number == 1]
    if not groups:
        return []
    if len(groups) != 1 or not isinstance(groups[0].value, bytes):
        raise _fail("the item's FieldsBlob #1.#1 is not one message")
    lists = [f for f in blobs.decode_protobuf(groups[0].value, nested_depth=0) if f.number == 1]
    if len(lists) != 1 or not isinstance(lists[0].value, bytes):
        raise _fail("the item's link list (#1.#1.#1) is not one keyed dict")
    entries = blobs.decode_keyed_dict(lists[0].value, strict=True).entries
    by_key = {e.key: e.value for e in entries}
    if sorted(by_key) != sorted(str(i) for i in range(len(entries))) or len(by_key) != len(entries):
        raise _fail(f"the item's link keys are {sorted(by_key)}, not 0..{len(entries) - 1}")
    return [by_key[str(i)] for i in range(len(entries))]


def _text_frames(value: object) -> int | None:
    if isinstance(value, str) and value.isascii() and value.lstrip("-").isdigit():
        return int(value)
    return None


@refusing("self_check_failed", "verifying the written project against its layout", _HINT)
def verify_written(
    path: Path,
    seed: Seed,
    cuts: Sequence[Cut],
    *,
    transitions: Sequence[CutTransition] = (),
    clips: Sequence[ClipRows] = (),
    project_name: str,
    timeline_name: str | None,
) -> int:
    """Check the project written at ``path`` from ``seed`` holds exactly the items ``cuts``
    asks for, the media clips ``clips`` the writer added, the project named ``project_name``
    and the timeline named ``timeline_name`` (None: the seed's), and otherwise the seed's rows
    (see the module docstring); returns how many items it checked, or raises
    ``self_check_failed`` naming the first difference."""
    unruled = sorted(set(ITEM_COLUMNS) ^ set(ITEM_RULES))
    unruled += sorted(set(ITEM_COLUMNS) ^ set(TRANSITION_RULES))
    if unruled:
        raise _fail(f"the verifier has no rule for item columns {unruled[:4]}", field="rules")
    tables = _read_all(path, seed)
    want = expected_items(cuts)
    fades = expected_transitions(cuts, transitions)
    rate = float(seed.rate)
    tracks = {"picture": seed.video_track, "sound": seed.audio_track}
    joins: dict[object, list[dict[str, Any]]] = {}
    for row in tables["Sm2TiItem_Sm2TiTrack"]:
        joins.setdefault(row.get("DbAssociate"), []).append(row)
    every = tables["Sm2TiItem"]
    count = len(want) + len(fades)
    if len(every) != count or len(tables["Sm2TiItem_Sm2TiTrack"]) != count:
        raise _fail(
            f"the written project holds {len(every)} items and "
            f"{len(tables['Sm2TiItem_Sm2TiTrack'])} join rows; the layout asks for {count}",
            field="items",
        )
    items = [r for r in every if r.get("DbType") != _TRANSITION]
    fade_rows = [r for r in every if r.get("DbType") == _TRANSITION]
    if len(fade_rows) != len(fades):
        raise _fail(
            f"the written project holds {len(fade_rows)} transitions; the cut asks for "
            f"{len(fades)}",
            field="DbType",
        )
    # Each track's Items vector: record order, a transition before a clip starting on its frame.
    order: dict[str, dict[tuple[str, int], int]] = {}
    for kind in tracks:
        keyed = [(e.record_in, 1, "clip", e.record_in) for e in want if e.kind == kind]
        if kind == "sound":
            keyed += [(t.cut - t.before, 0, "transition", t.cut) for t in fades]
        order[kind] = {(k[2], k[3]): i for i, k in enumerate(sorted(keyed))}
    written: dict[tuple[str, int], dict[str, Any]] = {}  # (kind, record start) -> item
    medias: dict[str, _Media] = {}
    for kind, track in tracks.items():
        mine = [r for r in items if r.get("Sm2TiTrack_id") == track]
        expected = sorted((e for e in want if e.kind == kind), key=lambda e: e.record_in)
        if len(mine) != len(expected):
            raise _fail(
                f"the {kind} track holds {len(mine)} items; the layout asks for {len(expected)}",
                field="Sm2TiTrack_id",
            )
        ordered = []
        for row in mine:
            link = joins.get(row.get("Sm2TiItem_id"), [])
            if len(link) != 1:
                raise _fail(f"a {kind} item has {len(link)} join rows", field="Items")
            ordered.append((link[0], row))
        ordered.sort(
            key=lambda lr: lr[0].get("DbIndex") if isinstance(lr[0].get("DbIndex"), int) else -1
        )
        for (link, row), exp in zip(ordered, expected, strict=True):
            where = f"the {kind} item at record frame {exp.record_in}"
            _same(where, "DbIndex", link.get("DbIndex"), order[kind][("clip", exp.record_in)])
            _same(where, "join DbOwner", link.get("DbOwner"), track)
            _same(where, "join DbPropertyName", link.get("DbPropertyName"), "Items")
            _same(where, "DbType", row.get("DbType"), _TYPES[kind])
            _same(where, "Start", row.get("Start"), str(seed.start_frame + exp.record_in))
            _same(where, "Duration", row.get("Duration"), str(exp.duration))
            _same(where, "In", row.get("In"), str(exp.src_in) if exp.src_in else None)
            _same(where, "MediaRef", row.get("MediaRef"), exp.media_id)
            media = medias.get(exp.media_id) or _media(exp.media_id, tables)
            medias[exp.media_id] = media
            _same(where, "MediaFilePath", row.get("MediaFilePath"), media.path)
            _same(where, "MediaFrameRate", row.get("MediaFrameRate"), media.rate_bytes)
            _same(where, "MediaTimemapBA", row.get("MediaTimemapBA"), media.timemap)
            _same_double(where, "MediaStartTime", row.get("MediaStartTime"), media.start)
            if kind == "picture":
                if media.frames is None:
                    raise _fail(f"{where} plays a sound file", field="MediaRef")
                if exp.src_in + exp.duration > media.frames:
                    raise _fail(f"{where} reads past the end of its media", field="In")
                _same(where, "MediaTrackIdx", row.get("MediaTrackIdx"), None)
            else:
                if media.sound_frames is None:
                    raise _fail(f"{where} plays a clip without sound", field="MediaRef")
                if exp.src_in + exp.duration > media.sound_frames:
                    raise _fail(f"{where} reads past the end of its clip's sound", field="In")
                _same(where, "MediaTrackIdx", row.get("MediaTrackIdx"), 0)
                _check_channels(where, row.get("VirtualAudioTrackBA"), media.channels)
            written[(kind, exp.record_in)] = row
        _check_order(kind, [r for _, r in ordered])
    by_group: dict[int, list[Expected]] = {}
    for e in want:
        by_group.setdefault(e.group, []).append(e)
    ids: set[object] = set()
    for exp in want:
        row = written[(exp.kind, exp.record_in)]
        others = [
            str(written[(o.kind, o.record_in)].get("Sm2TiItem_id"))
            for o in by_group[exp.group]
            if o is not exp
        ]
        where = f"the {exp.kind} item at record frame {exp.record_in}"
        links = _links(row)
        if sorted(links) != sorted(others) or len(set(links)) != len(links):
            raise _fail(
                f"{where} is linked to {links}; it must be linked to {others} (the other half "
                "of its layout entry) and nothing else",
                field="FieldsBlob #1.#1.#1",
            )
        item_id = row.get("Sm2TiItem_id")
        if (
            not isinstance(item_id, str)
            or not _UUID.match(item_id)
            or item_id in seed.taken
            or item_id in ids
        ):
            raise _fail(f"{where} has the id {item_id!r}, not a fresh one", field="Sm2TiItem_id")
        ids.add(item_id)
        derived = _derived(exp, row, medias[exp.media_id], tables, tracks, others)
        _check_columns(where, row, derived)
    _check_transitions(seed, fade_rows, joins, fades, cuts, medias, tables, order["sound"], ids)
    _check_clips(seed, tables, clips, cuts)
    _check_rows(seed, tables, clips, project_name=project_name, timeline_name=timeline_name)
    _check_extents(seed, tables, want, rate)
    return len(want) + len(fades)


def _check_transitions(
    seed: Seed,
    rows: list[dict[str, Any]],
    joins: dict[object, list[dict[str, Any]]],
    fades: list[ExpectedTransition],
    cuts: Sequence[Cut],
    medias: dict[str, _Media],
    tables: dict[str, list[dict[str, Any]]],
    order: dict[tuple[str, int], int],
    ids: set[object],
) -> None:
    """Every transition row against its rules (:data:`TRANSITION_RULES`), its join row, and
    the media its two clips hold beyond the cut."""
    by_start = {}
    for row in rows:
        start = _text_frames(row.get("Start"))
        if start is None or start in by_start:
            raise _fail("a transition has no whole-frame start, or two share one", field="Start")
        by_start[start] = row
    for fade in fades:
        where = f"the transition at record frame {fade.cut}"
        start = seed.start_frame + fade.cut - fade.before
        row = by_start.pop(start, None)
        if row is None:
            raise _fail(f"{where} is not in the written project at frame {start}", field="Start")
        link = joins.get(row.get("Sm2TiItem_id"), [])
        if len(link) != 1:
            raise _fail(f"{where} has {len(link)} join rows", field="Items")
        _same(where, "DbIndex", link[0].get("DbIndex"), order[("transition", fade.cut)])
        _same(where, "join DbOwner", link[0].get("DbOwner"), seed.audio_track)
        _same(where, "join DbPropertyName", link[0].get("DbPropertyName"), "Items")
        item_id = row.get("Sm2TiItem_id")
        if (
            not isinstance(item_id, str)
            or not _UUID.match(item_id)
            or item_id in seed.taken
            or item_id in ids
        ):
            raise _fail(f"{where} has the id {item_id!r}, not a fresh one", field="Sm2TiItem_id")
        ids.add(item_id)
        derived: dict[str, object] = {
            "Sm2TiItem_id": item_id,
            "DbType": _TRANSITION,
            "Start": str(start),
            "Duration": str(fade.frames),
            "Sm2TiTrack_id": seed.audio_track,
            "AlignmentType": 2,
            "Position": 2,
            "PrettyType": fade.pretty,
            "FieldsBlob": row.get("FieldsBlob"),  # checked by its structure below
        }
        _check_transition_fields(where, row.get("FieldsBlob"))
        for column, value in row.items():
            rule = TRANSITION_RULES.get(column)
            if rule is None:
                if value is not None:
                    raise _fail(f"{where} holds {column} {value!r}; it must be empty", field=column)
                continue
            expect = derived[column] if rule == DERIVED else rule[1]  # type: ignore[index]
            if not _equal(value, expect):
                shown = value if not isinstance(value, bytes) else f"{len(value)} bytes"
                raise _fail(
                    f"{where} holds {column} {shown!r}; it must hold {expect!r}", field=column
                )
        # The media it plays: frames past the outgoing clip's out point, before the incoming
        # clip's in point (the clips' rows as written).
        out_cut, in_cut = cuts[fade.outgoing], cuts[fade.incoming]
        after = fade.frames - fade.before
        if fade.before > out_cut.duration or after > in_cut.duration:
            raise _fail(f"{where} is longer than a clip it joins", field="Duration")
        media = medias.get(out_cut.clip.media_id) or _media(out_cut.clip.media_id, tables)
        if media.sound_frames is None or (
            out_cut.src_in + out_cut.duration + after > media.sound_frames
        ):
            raise _fail(
                f"{where} plays past the end of the outgoing clip's sound", field="Duration"
            )
        if in_cut.src_in < fade.before:
            raise _fail(f"{where} plays before the incoming clip's first frame", field="Start")
    if by_start:
        raise _fail(f"{len(by_start)} transition(s) the cut does not ask for", field="DbType")


def _check_transition_fields(where: str, value: object) -> None:
    """A sound transition's ``FieldsBlob``: version 2, framed plain, one field #2 whose
    qCompress'd value is :data:`TRANSITION_PAYLOAD`."""
    if not isinstance(value, bytes) or len(value) < 9:
        raise _fail(f"{where}'s FieldsBlob is not a blob", field="FieldsBlob")
    version, length = struct.unpack(">II", value[:8])
    if version != 2 or length != len(value) - 8 or value[8] != blobs.CODEC_PROTOBUF:
        raise _fail(f"{where}'s FieldsBlob is not a plain version 2 message", field="FieldsBlob")
    fields = blobs.decode_protobuf(value[9:], nested_depth=0)
    if len(fields) != 1 or fields[0].number != 2 or not isinstance(fields[0].value, bytes):
        raise _fail(f"{where}'s FieldsBlob holds other fields than #2", field="FieldsBlob")
    if encode.protobuf([(2, 2, fields[0].value)]) != value[9:]:
        raise _fail(f"{where}'s FieldsBlob is not in its canonical form", field="FieldsBlob")
    inner = fields[0].value
    if (
        inner != TRANSITION_FIELD
        or blobs.qcompress_decompress(inner, max_output=4096) != TRANSITION_PAYLOAD
    ):
        raise _fail(f"{where}'s FieldsBlob #2 is not the sound transition's value", field="#2")


def _read_all(path: Path, seed: Seed) -> dict[str, list[dict[str, Any]]]:
    """Every row of every table of the written file, which must have the seed's tables and
    columns."""
    try:
        names = table_names(path)
    except ProjectWriteError as exc:
        raise _fail(f"the written project's tables cannot be listed: {exc.message}") from None
    if sorted(names) != sorted(seed.columns):
        extra = sorted(set(names) - set(seed.columns))
        gone = sorted(set(seed.columns) - set(names))
        raise _fail(
            f"the written project's tables differ from the seed's (added {extra[:4]}, "
            f"gone {gone[:4]})",
            field="tables",
        )
    lib = read_rows(path, tuple(sorted(names)), options=LibraryOptions(settle_seconds=0.0))
    for table in names:
        if tuple(lib.columns[table]) != tuple(seed.columns[table]):
            raise _fail(f"the written {table} table has other columns than the seed's", field=table)
    return {
        t: [dict(zip(lib.columns[t], values, strict=True)) for values in lib.rows[t]] for t in names
    }


def _derived(
    exp: Expected,
    row: dict[str, Any],
    media: _Media,
    tables: dict[str, list[dict[str, Any]]],
    tracks: dict[str, str],
    partners: list[str],
) -> dict[str, object]:
    """The value of every :data:`DERIVED` column of the item ``exp`` asks for."""
    names = [r.get("Name") for r in tables["Sm2MpMedia"] if r.get("Sm2MpMedia_id") == exp.media_id]
    picture = exp.kind == "picture"
    return {
        "Sm2TiItem_id": row.get("Sm2TiItem_id"),  # its freshness is checked on its own
        "DbType": _TYPES[exp.kind],
        "Name": names[0] if len(names) == 1 else None,
        "Start": row.get("Start"),  # checked against the layout above
        "Duration": str(exp.duration),
        "In": str(exp.src_in) if exp.src_in else None,
        "MediaRef": exp.media_id,
        "MediaStartTime": media.start,
        "MediaFilePath": media.path,
        "MediaTimemapBA": media.timemap,
        "MediaTrackIdx": None if picture else 0,
        "MediaFrameRate": media.rate_bytes,
        "Sm2TiTrack_id": tracks[exp.kind],
        "VirtualAudioTrackBA": None if picture else _virtual_audio(media.channels),
        "PreConformMediaExtents": PICTURE_PRECONFORM if picture else None,
        "FieldsBlob": _fields_blob(exp.kind, partners),
    }


def _virtual_audio(channels: int | None) -> bytes:
    """A sound item's ``VirtualAudioTrackBA`` (every plain sound item of the corpus): its
    channels' ids ``0x4001``.. after a 2 and the count, then ``AudioType`` 1 for mono, 0 for
    stereo."""
    if channels not in (1, 2):
        raise _fail(f"a sound item plays a clip of {channels} channel(s)", field="channels")
    ids = b"".join(struct.pack(">I", 0x4001 + i) for i in range(channels))
    return encode.keyed_dict(
        [
            ("ChannelsBA", blobs.KD_BYTES, 0, struct.pack(">II", 2, channels) + ids),
            ("AudioType", blobs.KD_INT32, 0, 1 if channels == 1 else 0),
        ]
    )


def _fields_blob(kind: str, partners: list[str]) -> bytes:
    """An item's whole ``FieldsBlob``: ``#1`` holding the link list (keys ``0``.., highest
    first) when the item is linked and ``#4`` = 1, then a picture's ``#2`` or a sound's ``#15``
    constant; framed as Resolve frames a message of its length (version 2, plain protobuf up to
    64 bytes, else a raw zstd frame)."""
    inner: list[tuple[int, int, int | bytes]] = []
    if partners:
        entries = [(str(i), blobs.KD_STRING, 0, p) for i, p in enumerate(partners)]
        inner.append((1, 2, encode.protobuf([(1, 2, encode.keyed_dict(entries[::-1]))])))
    inner.append((4, 0, _FIELDS_ONE_FOUR))
    tail: tuple[int, int, int | bytes] = (
        (2, 2, PICTURE_TAIL) if kind == "picture" else (15, 0, _SOUND_TAIL)
    )
    payload = encode.protobuf([(1, 2, encode.protobuf(inner)), tail])
    if len(payload) <= 64:
        body = bytes([blobs.CODEC_PROTOBUF]) + payload
    else:
        body = bytes([blobs.CODEC_ZSTD]) + encode.zstd_raw_frame(payload)
    return struct.pack(">II", 2, len(body)) + body


def _check_columns(where: str, row: dict[str, Any], derived: dict[str, object]) -> None:
    """Every column of the item against its rule (:data:`ITEM_RULES`); a column outside the
    closed list must be empty."""
    for column, value in row.items():
        rule = ITEM_RULES.get(column)
        if rule is None:
            if value is not None:
                raise _fail(f"{where} holds {column} {value!r}; it must be empty", field=column)
            continue
        if rule == DERIVED:
            want = derived[column]
        elif rule == FRESH:
            want = None
        else:
            assert isinstance(rule, tuple)
            want = rule[1]
        if not _equal(value, want):
            shown = value if not isinstance(value, bytes) else f"{len(value)} bytes"
            raise _fail(
                f"{where} holds {column} {shown!r}; "
                + (
                    "it must be empty, as on an item Resolve has just made"
                    if rule == FRESH
                    else f"it must hold {want if not isinstance(want, bytes) else want.hex()!r}"
                ),
                field=column,
            )


def _equal(got: object, want: object) -> bool:
    if type(got) is not type(want):
        return False
    if isinstance(got, float) and isinstance(want, float):
        return struct.pack("<d", got) == struct.pack("<d", want)
    return got == want


def _clip_text(row: dict[str, Any], what: str) -> tuple[bytes, bytes]:
    fields = _message(row.get("Clip"), what)
    folder, name = _one(fields, 1, what).value, _one(fields, 2, what).value
    if not isinstance(folder, bytes) or not isinstance(name, bytes):
        raise _fail(f"{what}'s folder or file name is not text", field="Clip")
    return folder, name


def _check_clips(
    seed: Seed,
    tables: dict[str, list[dict[str, Any]]],
    clips: Sequence[ClipRows],
    cuts: Sequence[Cut],
) -> None:
    """Each media clip the writer added, as written: one media row of a file clip, its picture
    info (one for a picture file, none for a sound file) and its sound infos (each naming this
    clip, joined in ``EmbeddedAudioVec`` order from 0, as many as were authored), its one entry
    in the seed's bin, its lockable blobs and presets as many as were authored, and every
    ``Clip`` blob naming the same folder and file, the media's name. A cut may play only a
    clip of the seed or one added."""
    known = {c.media_id for c in seed.clips} | {c.media_id for c in clips}
    for cut in cuts:
        if cut.clip.media_id not in known:
            raise _fail(
                f"the cut plays the media clip {cut.clip.media_id}, neither the seed's nor added",
                field="MediaRef",
            )
    for clip in clips:
        mid = clip.media_id
        what = f"the added media clip {mid}"
        medias = [r for r in tables["Sm2MpMedia"] if r.get("Sm2MpMedia_id") == mid]
        if len(medias) != 1:
            raise _fail(f"{what} has {len(medias)} media rows", field="Sm2MpMedia")
        media = medias[0]
        picture = media.get("DbType") == "Sm2MpVideoClip"
        if media.get("DbType") not in ("Sm2MpVideoClip", "Sm2MpAudioClip"):
            raise _fail(f"{what} is a {media.get('DbType')!r}", field="DbType")
        videos = [r for r in tables["BtVideoInfo"] if r.get("Sm2MpMedia_id") == mid]
        if len(videos) != (1 if picture else 0) or len(videos) != (clip.video is not None):
            raise _fail(f"{what} has {len(videos)} picture infos", field="BtVideoInfo")
        if media.get("Video") != (videos[0].get("BtVideoInfo_id") if videos else None):
            raise _fail(f"{what}'s Video does not name its picture info", field="Video")
        audios = {
            r.get("BtAudioInfo_id"): r
            for r in tables["BtAudioInfo"]
            if r.get("Sm2MpMedia_id") == mid
        }
        links = [r for r in tables["BtAudioInfo_Sm2MpMedia"] if r.get("DbOwner") == mid]
        order = sorted(r.get("DbIndex") for r in links if isinstance(r.get("DbIndex"), int))
        named = [r.get("DbAssociate") for r in links]
        if (
            len(links) != len(clip.audio)
            or len(audios) != len(clip.audio)
            or order != list(range(len(links)))
            or sorted(map(str, named)) != sorted(map(str, audios))
            or len(set(named)) != len(named)
            or any(r.get("DbPropertyName") != "EmbeddedAudioVec" for r in links)
        ):
            raise _fail(
                f"{what} has {len(audios)} sound infos and {len(links)} sound joins "
                f"(DbIndex {order}); it was authored with {len(clip.audio)}",
                field="BtAudioInfo_Sm2MpMedia",
            )
        bins = [r for r in tables["Sm2MpFolder_Sm2MpMedia"] if r.get("DbAssociate") == mid]
        if (
            len(bins) != 1
            or bins[0].get("DbOwner") != seed.folder_id
            or bins[0].get("DbPropertyName") != "MediaVec"
            or media.get("Sm2MpFolder_id") != seed.folder_id
        ):
            raise _fail(f"{what} is not in the seed's bin once", field="Sm2MpFolder_Sm2MpMedia")
        owned = {mid, *map(str, audios)} | ({str(videos[0]["BtVideoInfo_id"])} if videos else set())
        lockables = [r for r in tables["BtLockableBlob"] if r.get("BlobOwner") in owned]
        presets = [r for r in tables["SmPreset"] if r.get("Sm2MpMedia_id") == mid]
        if len(lockables) != len(clip.lockables) or len(presets) != len(clip.presets):
            raise _fail(f"{what} has other metadata or preset rows than authored", field=what)
        files = {_clip_text(r, f"{what}'s picture Clip") for r in videos}
        files |= {_clip_text(r, f"{what}'s sound Clip") for r in audios.values()}
        if len(files) != 1:
            shown = sorted(
                f"{a.decode(errors='replace')}\\{b.decode(errors='replace')}" for a, b in files
            )
            raise _fail(
                f"{what}'s Clip blobs name {len(files)} files: {'; '.join(shown)}", field="Clip"
            )
        ((_, file_name),) = files
        if media.get("Name") != file_name.decode("utf-8", "replace"):
            raise _fail(f"{what} is named {media.get('Name')!r}, not its file's name", field="Name")
    folder = sorted(
        r.get("DbIndex")
        for r in tables["Sm2MpFolder_Sm2MpMedia"]
        if r.get("DbOwner") == seed.folder_id and r.get("DbPropertyName") == "MediaVec"
    )
    if clips and folder != list(range(len(folder))):
        raise _fail(f"the bin's media are not in order from 0 ({folder[:8]}...)", field="DbIndex")


def _check_rows(
    seed: Seed,
    tables: dict[str, list[dict[str, Any]]],
    clips: Sequence[ClipRows],
    *,
    project_name: str,
    timeline_name: str | None,
) -> None:
    """Every row of every table but the items': the seed's, but for the cells the writer
    changes (each checked here or in :func:`_check_extents`), plus exactly the added clips'
    rows."""
    projects = tables["SM_Project"]
    new_id = projects[0].get("SM_Project_id") if len(projects) == 1 else None
    if (
        not isinstance(new_id, str)
        or not _UUID.match(new_id)
        or new_id == seed.project_id
        or new_id in seed.taken
    ):
        raise _fail(f"the project's id {new_id!r} is not a fresh one", field="SM_Project_id")
    renamed = timeline_name is not None and timeline_name != seed.timeline_name
    ids: dict[str, str | None] = {
        "project_id": new_id,
        "timeline_id": seed.timeline_id,
        "timeline_clip_id": seed.timeline_clip_id,
        "sequence_id": seed.sequence_id,
    }
    wanted: dict[str, object] = {
        "SM_Project.ProjectName": project_name,
        "Sm2Timeline.Name": timeline_name if renamed else seed.cells.get("Sm2Timeline.Name"),
        "Sm2MpMedia.Name": timeline_name if renamed else seed.cells.get("Sm2MpMedia.Name"),
    }
    masks: dict[str, tuple[str, str, str]] = {}
    for table, id_column, attribute, column in WRITER_CELLS:
        value = ids[attribute]
        if value is None:
            continue
        masks[table] = (id_column, value, column)
        found = [r for r in tables[table] if r.get(id_column) == value]
        if len(found) != 1:
            raise _fail(f"the written {table} has {len(found)} rows of id {value}", field=table)
        key = f"{table}.{column}"
        if key in wanted and not _equal(found[0].get(column), wanted[key]):
            raise _fail(
                f"the written {key} is {found[0].get(column)!r}; it must be {wanted[key]!r}",
                field=key,
            )
    added: dict[str, Counter[str]] = {t: Counter() for t in MEDIA_TABLES}
    for clip in clips:
        for table, rows in clip.tables().items():
            columns = seed.columns.get(table, ())
            for row in rows:
                full = {c: row.get(c) for c in columns}
                added[table][row_digest(full, project_id=new_id)] += 1
    for table, rows in tables.items():
        if table in REPLACED_TABLES:
            continue
        mask = masks.get(table)
        got: Counter[str] = Counter()
        for row in rows:
            masked = (mask[2],) if mask is not None and row.get(mask[0]) == mask[1] else ()
            got[row_digest(row, project_id=new_id, masked=masked)] += 1
        had = Counter(seed.digests.get(table, {}))
        gone = had - got
        if gone:
            raise _fail(
                f"{sum(gone.values())} row(s) of the seed's {table} are changed or gone",
                field=table,
            )
        extra = got - had
        intended = added.get(table, Counter())
        if extra != intended:
            raise _fail(
                f"the written {table} holds {len(rows)} rows: the seed's "
                f"{sum(had.values())} and {sum(extra.values())} more, of which "
                f"{sum((extra - intended).values())} the writer did not add (it added "
                f"{sum(intended.values())})",
                field=table,
            )


def _same(where: str, field: str, got: object, want: object) -> None:
    if type(got) is not type(want) or got != want:
        raise _fail(
            f"{where} holds {field} {got!r}; its layout and media give {want!r}", field=field
        )


def _same_double(where: str, field: str, got: object, want: float) -> None:
    if not isinstance(got, float) or struct.pack("<d", got) != struct.pack("<d", want):
        raise _fail(f"{where} holds {field} {got!r}; its media gives {want!r}", field=field)


def _check_channels(where: str, value: object, channels: int | None) -> None:
    audio = _keyed(value, "the item's VirtualAudioTrackBA")
    data = audio.get("ChannelsBA")
    if channels is None or not isinstance(data, bytes) or len(data) != 8 + 4 * channels:
        raise _fail(
            f"{where} does not list its clip's {channels} channel(s)", field="VirtualAudioTrackBA"
        )
    if struct.unpack(">II", data[:8]) != (2, channels):
        raise _fail(
            f"{where} lists another channel count than its clip's", field="VirtualAudioTrackBA"
        )


def _check_order(kind: str, rows: list[dict[str, Any]]) -> None:
    end: int | None = None
    for row in rows:
        start, duration = _text_frames(row.get("Start")), _text_frames(row.get("Duration"))
        if start is None or duration is None or duration < 1:
            raise _fail(f"a {kind} item has no whole-frame range", field="Start")
        if end is not None and start < end:
            raise _fail(f"two {kind} items overlap or are out of record order", field="DbIndex")
        end = start + duration


def _check_extents(
    seed: Seed, tables: dict[str, list[dict[str, Any]]], want: list[Expected], rate: float
) -> None:
    rows = [r for r in tables["Sm2Sequence"] if r.get("Sm2Sequence_id") == seed.sequence_id]
    if len(rows) != 1 or not isinstance(rows[0].get("MediaExtents"), bytes):
        raise _fail("the timeline's sequence has no MediaExtents", field="MediaExtents")
    ends = [seed.start_frame + e.record_in + e.duration for e in want]
    length = (max(ends) / rate - seed.start_seconds) if ends else 0.0
    stored = rows[0]["MediaExtents"]
    expected = struct.pack("<dd", seed.start_seconds, length)
    if stored != expected:
        got = struct.unpack("<dd", stored) if len(stored) == 16 else stored
        raise _fail(
            f"the sequence's MediaExtents is {got!r}; the layout gives "
            f"{(seed.start_seconds, length)!r} (start, length in seconds)",
            field="MediaExtents",
        )
