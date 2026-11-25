"""An independent read of a written project: does every item hold what its layout asks for?

:func:`verify_written` runs on the staged file after the writer's own read-back
(:func:`aar.drp.libwrite.check.check_written`) and before anything is placed. It does not use
the code that wrote the rows (:func:`aar.drp.items.author_item`,
:func:`aar.drp.items.media_view`, :func:`aar.drp.libwrite.cut.author_cut`,
:func:`aar.drp.libwrite.write.sequence_length`): from the layout (the :class:`Cut` entries the
caller authored the items from) and from each media clip's own rows *as written*, decoded here
with the shared blob decoders only, it recomputes what every item and the sequence must hold, by
the field map of docs/DESIGN.md ("Timeline items (S25c2)"), and compares it with the rows read
back from the file:

* per item: its type and track, its place in the track's ``Items`` vector (record order, dense
  from 0), its record range (``Start``, ``Duration``), its source in (``In``), the media clip it
  plays (``MediaRef``), and that clip's file path, start time, time map, frame rate, stream
  index and channels (``MediaFilePath``, ``MediaStartTime``, ``MediaTimemapBA``,
  ``MediaFrameRate``, ``MediaTrackIdx``, ``VirtualAudioTrackBA``);
* its links: the other half of the same layout entry (a picture and its sound name each other)
  and nothing else; an entry with one half has no links;
* that the timeline holds those items and no others, and the sequence's ``MediaExtents``: the
  seed's start, and the end of the last item over the rate minus the start.

A difference is ``self_check_failed`` naming the field and the item.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.library import LibraryOptions, read_rows
from aar.drp.libwrite.cut import Cut
from aar.drp.libwrite.errors import ProjectWriteError, refusing
from aar.drp.libwrite.seed import Seed
from aar.media.probe import timecode_frames

__all__ = ["Expected", "expected_items", "verify_written"]

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
    if len(audios) != len(links) or len(videos) > 1 or (not videos and not audios):
        raise _fail(f"the media clip {media_id} does not have its video and audio rows")
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
    if first_track is not None:
        count = first_track.get("NumChannels")
        channels = count if isinstance(count, int) and not isinstance(count, bool) else None
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
        return _Media(path, rate_bytes, start, b"\x02" + struct.pack(">d", last), channels, frames)
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
    samples, sample_rate = first_track.get("Duration"), first_track.get("SampleRate")
    start = first_track.get("StartTime")
    if not isinstance(samples, int) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise _fail("the clip's first audio track has no length or sample rate")
    if not isinstance(start, float):
        raise _fail("the clip's first audio track has no StartTime")
    last = (samples / sample_rate) * rate / rate - 1 / rate
    return _Media(path, rate_bytes, start, b"\x02" + struct.pack(">d", last), channels, None)


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
def verify_written(path: Path, seed: Seed, cuts: Sequence[Cut]) -> int:
    """Check the project written at ``path`` from ``seed`` holds exactly the items ``cuts``
    asks for (see the module docstring); returns how many items it checked, or raises
    ``self_check_failed`` naming the first difference."""
    lib = read_rows(path, _TABLES, options=LibraryOptions(settle_seconds=0.0))
    tables = {
        t: [dict(zip(lib.columns[t], values, strict=True)) for values in lib.rows[t]]
        for t in _TABLES
    }
    want = expected_items(cuts)
    rate = float(seed.rate)
    tracks = {"picture": seed.video_track, "sound": seed.audio_track}
    joins: dict[object, list[dict[str, Any]]] = {}
    for row in tables["Sm2TiItem_Sm2TiTrack"]:
        joins.setdefault(row.get("DbAssociate"), []).append(row)
    items = tables["Sm2TiItem"]
    if len(items) != len(want) or len(tables["Sm2TiItem_Sm2TiTrack"]) != len(want):
        raise _fail(
            f"the written project holds {len(items)} items and "
            f"{len(tables['Sm2TiItem_Sm2TiTrack'])} join rows; the layout asks for {len(want)}",
            field="items",
        )
    written: dict[tuple[str, int], dict[str, Any]] = {}  # (kind, record start) -> item
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
        for index, ((link, row), exp) in enumerate(zip(ordered, expected, strict=True)):
            where = f"the {kind} item at record frame {exp.record_in}"
            _same(where, "DbIndex", link.get("DbIndex"), index)
            _same(where, "join DbOwner", link.get("DbOwner"), track)
            _same(where, "join DbPropertyName", link.get("DbPropertyName"), "Items")
            _same(where, "DbType", row.get("DbType"), _TYPES[kind])
            _same(where, "Start", row.get("Start"), str(seed.start_frame + exp.record_in))
            _same(where, "Duration", row.get("Duration"), str(exp.duration))
            _same(where, "In", row.get("In"), str(exp.src_in) if exp.src_in else None)
            _same(where, "MediaRef", row.get("MediaRef"), exp.media_id)
            media = _media(exp.media_id, tables)
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
                _same(where, "MediaTrackIdx", row.get("MediaTrackIdx"), 0)
                _check_channels(where, row.get("VirtualAudioTrackBA"), media.channels)
            written[(kind, exp.record_in)] = row
        _check_order(kind, [r for _, r in ordered])
    by_group: dict[int, list[Expected]] = {}
    for e in want:
        by_group.setdefault(e.group, []).append(e)
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
    _check_extents(seed, tables, want, rate)
    return len(want)


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
