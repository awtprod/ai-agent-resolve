"""Author the media-pool rows of a file Resolve has never imported.

:func:`author` takes the measured facts of one file and a template clip of the same kind (rows
Resolve saved), and returns the rows Resolve would have written had it imported the file:

* **identities** are fresh (row ids and the uuids that live only inside blobs);
* **derived** values are computed from the facts (:mod:`aar.drp.mediapool.formulas`);
* **project-local** values come from :class:`Placement` (bin, position, project frame rate,
  lockable map, date added) or are left as a fresh import has them (no playhead, no preset, no
  pins, no relink history);
* **constant and unexplained** values are copied from the template; the kind key guarantees they
  were equal across every file of the kind in the answer key.

It returns data only and writes nothing. Whatever it cannot justify is refused with a typed
:class:`MediaPoolError`, or named in the result's ``unverified``. Every blob of the template is
taken apart by a closed list of known fields: a field it does not know is a refusal, and after
authoring no id, file name or folder of the template may be left in any layer of any value.
"""

from __future__ import annotations

import struct
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.facts import MediaFacts
from aar.drp.mediapool.kinds import KINDS, KindKey, KindStatus, kind_of_facts, kind_of_rows, matches
from aar.drp.mediapool.templates import ClipRows, Row

__all__ = ["AuthoredClip", "Placement", "Template", "author"]

_UUID_TEXT = 36
#: Metadata entries (``BtMetadataLockableBlob``) derived from the facts, by key.
METADATA_RULES: dict[int, Callable[[MediaFacts], str]] = {
    215: lambda facts: facts.name,
}
_CLIP_FIELDS = frozenset({1, 2, 3, 5, 6, 7, 13, 14, 15, 16, 18, 26})
_CLIP_PROJECT_LOCAL = (14, 26)
_TIME_KEYS = frozenset({"UniqueId", "Timecode", "StartFrame", "NumFrames", "FrameRate", "DbType"})
_GEOMETRY_KEYS = frozenset(
    {"UniqueId", "Resolution", "IsResolutionTransposed", "FrameSize", "ScanType", "DbType"}
)
_PROXY_KEYS = frozenset({"UniqueId", "DbType", "DataManagerID"})
_VIDEO_METADATA_KEYS = frozenset({"UniqueId", "DbType", "ActualReelNumber"})
_TRACK_KEYS = frozenset(
    {
        "UniqueId",
        "StartTime",
        "SampleRate",
        "NumChannels",
        "IdxTrack",
        "Duration",
        "DbType",
        "CodecName",
        "ChannelLayout",
        "BitDepth",
    }
)
_EXTENTS_KEYS = frozenset(
    {"Type", "MediaExtents", "CodecName", "ChannelVecBA", "BitDepth", "SampleRate"}
)
#: ``Sm2MpMedia`` columns a fresh import leaves empty (UI state and history of the template's
#: own clip): the playhead, the pan/tilt/zoom preset, pins and per-clip metadata.
_MEDIA_CLEARED = ("CurPlayheadPosition", "PTZRPreset", "PinsBA", "VideoMetadata", "MediaMetadata")


@dataclass(frozen=True)
class Placement:
    """Where the new clip goes, and the project-local values a fresh import takes from the
    project it is imported into."""

    folder_id: str
    folder_index: int
    #: The project's timeline frame rate: a sound clip and a still store it.
    project_rate: Fraction
    #: The date the clip was added, in seconds since 1970.
    added_at: int
    #: The project's ``BtLockableBlobMap`` id (needed when the kind has a metadata blob).
    lockable_map_id: str | None = None
    #: ``BtLockableBlob.DbSavedTime`` of the metadata blob (the project's age in ms at the save).
    saved_age_ms: int = 0


@dataclass(frozen=True)
class Template:
    """A template clip: rows Resolve saved, and the facts of its file when they were measured."""

    rows: ClipRows
    facts: MediaFacts | None = None
    source: str = ""


@dataclass(frozen=True)
class AuthoredClip:
    """The rows of the new clip, per table, ready to insert (nothing is written here)."""

    kind: KindKey
    status: str
    media_id: str
    rows: dict[str, tuple[Row, ...]]
    unverified: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    template_media_id: str = ""


def _bad(what: str) -> MediaPoolError:
    return MediaPoolError(
        "bad_template", what, "use a template clip from a project Resolve saved, of the same kind"
    )


class _Ids:
    """Fresh ids: never one already in the project, never one used twice, never a template's."""

    def __init__(
        self, taken: Collection[str], template: set[str], new_id: Callable[[], str]
    ) -> None:
        self.taken = taken
        self.template = template
        self.new_id = new_id
        self.made: set[str] = set()

    def __call__(self) -> str:
        value = self.new_id()
        if not isinstance(value, str) or len(value) != _UUID_TEXT:
            raise MediaPoolError(
                "id_collision", f"the id source gave {value!r}, not a uuid text", "use uuid4 ids"
            )
        if value in self.made or value in self.taken or value in self.template:
            raise MediaPoolError(
                "id_collision",
                f"the new id {value} is already used in the project or by this clip",
                "author again with fresh ids",
            )
        self.made.add(value)
        return value


def _closed(entries: bt.KdEntries, known: frozenset[str], what: str) -> None:
    unknown = [e[0] for e in entries if e[0] not in known]
    if unknown:
        raise _bad(f"{what} holds keys this writer does not know: {unknown[:5]}")


def _rate_bytes(rate: Fraction) -> bytes:
    return struct.pack("<d", float(rate)) + bytes(8)


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


@dataclass
class _Derived:
    name: str
    folder: str
    mtime_text: str
    created_us: int
    rate: Fraction | None = None
    frames: int | None = None
    timecode: str | None = None
    start: float = 0.0
    resolution: tuple[int, int] | None = None
    frame_size: int | None = None
    codec_tag: str | None = None
    audio: list[dict[str, Any]] = field(default_factory=list)


def _derive(facts: MediaFacts, kind: KindKey, placement: Placement) -> _Derived:
    d = _Derived(
        name=facts.name,
        folder=facts.folder,
        mtime_text=fm.qt_local_time(facts.mtime_local),
        created_us=fm.creation_us(facts),
    )
    video = fm.video_stream(facts)
    if kind.media == "video":
        assert video is not None
        d.rate = fm.video_rate(facts)
        d.frames = fm.video_frames(facts, d.rate)
        d.timecode = fm.start_timecode(facts)
        d.start = fm.start_seconds(d.timecode, d.rate)
        w, h, _ = fm.resolution(video)
        d.resolution = (w, h)
        _, bpp = fm.picture_class(video)
        assert video.width is not None and video.height is not None
        d.frame_size = fm.frame_size(video.width, video.height, bpp)
        d.codec_tag = video.codec_tag_string
    elif kind.media == "still":
        assert video is not None
        if any(c.isdigit() for c in facts.name):
            # One still whose name holds a clip number has that number as its StartFrame: Resolve
            # seems to read it as a frame of a sequence. One sample does not establish the rule.
            raise MediaPoolError(
                "cannot_derive",
                f"the still's name {facts.name!r} holds a number, which Resolve may read as a "
                "frame number (StartFrame); that rule is not established",
                "rename the still without digits, or import it in Resolve by hand",
            )
        d.rate = placement.project_rate
        d.frames = 1
        w, h, _ = fm.resolution(video)
        d.resolution = (w, h)
        if facts.size < 1:
            raise MediaPoolError(
                "bad_facts", "the still is an empty file", "measure the file again"
            )
        d.frame_size = facts.size
    for stream in fm.audio_streams(facts):
        samples = fm.audio_samples(stream)
        rate = int(stream.sample_rate or 0)
        d.audio.append(
            {
                "codec": fm.audio_codec_name(stream),
                "bit_depth": fm.audio_bit_depth(stream),
                "channels": stream.channels,
                "sample_rate": rate,
                "samples": samples,
                "layout": fm.channel_layout(stream),
                "extent": samples / rate,
            }
        )
    return d


def author(
    facts: MediaFacts,
    template: Template,
    placement: Placement,
    *,
    taken: Collection[str] = frozenset(),
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
    kinds: dict[KindKey, KindStatus] | None = None,
) -> AuthoredClip:
    """The rows of a new media clip for ``facts``, built from ``template``. ``taken`` holds the
    ids already in the project the rows are for; ``new_id`` makes fresh ids (uuid4 text)."""
    registry = KINDS if kinds is None else kinds
    if not isinstance(template, Template):
        raise MediaPoolError(
            "no_template", "no template clip was given", "read templates from a saved project"
        )
    _check_picture_decoded(facts)
    file_kind = kind_of_facts(facts)
    look_note = matches(file_kind, kind_of_rows(template.rows), template.facts)
    status = registry.get(file_kind)
    if status is None:
        raise MediaPoolError(
            "unsupported_kind",
            f"no evidence yet that the rows of a {file_kind.label()} can be authored",
            "import the file in Resolve by hand, or add the kind to the corpus gate first",
        )
    if not isinstance(placement.folder_id, str) or len(placement.folder_id) != _UUID_TEXT:
        raise MediaPoolError(
            "bad_facts", "the placement's folder id is not a uuid text", "give the bin's id"
        )
    if not 0 <= placement.folder_index < 2**31 or placement.project_rate <= 0:
        raise MediaPoolError(
            "bad_facts",
            "the placement's index or frame rate is out of range",
            "check the placement",
        )
    derived = _derive(facts, file_kind, placement)
    writer = _Writer(facts, template.rows, placement, derived, taken, new_id)
    rows = writer.build()
    unverified = []
    if status.status != "verified":
        unverified.append(
            f"kind not verified: {status.note or 'a single file of this kind was checked'}"
        )
    if look_note is not None:
        unverified.append(look_note)
    return AuthoredClip(
        kind=file_kind,
        status=status.status if not unverified else "unverified",
        media_id=writer.media_id,
        rows=rows,
        unverified=tuple(unverified),
        notes=tuple(writer.notes),
        template_media_id=template.rows.media_id,
    )


class _Writer:
    def __init__(
        self,
        facts: MediaFacts,
        tpl: ClipRows,
        placement: Placement,
        derived: _Derived,
        taken: Collection[str],
        new_id: Callable[[], str],
    ) -> None:
        self.facts = facts
        self.tpl = tpl
        self.placement = placement
        self.d = derived
        self.notes: list[str] = []
        self.template_ids: set[str] = set(tpl.ids())
        self.ids = _Ids(taken, self.template_ids, new_id)
        self.media_id = self.ids()
        self.video_id = self.ids() if tpl.video is not None else None
        self.audio_ids = [self.ids() for _ in tpl.audio]
        if len(self.audio_ids) != len(derived.audio):
            raise MediaPoolError(
                "kind_mismatch",
                f"the template has {len(self.audio_ids)} audio streams, "
                f"the file {len(derived.audio)}",
                "give a template of the same kind",
            )

    # ------------------------------------------------------------------ fresh ids in blobs

    def _fresh(self, entries: bt.KdEntries, what: str) -> None:
        entry = bt.kd_get(entries, "UniqueId")
        if entry is None or not isinstance(entry[3], str):
            raise _bad(f"{what} has no UniqueId")
        self.template_ids.add(entry[3])
        entry[3] = self.ids()

    # ------------------------------------------------------------------ Clip blobs

    def _clip(self, blob: object, what: str, *, video: bool, audio: dict[str, Any] | None) -> bytes:
        def change(fields: bt.Fields) -> None:
            numbers = [f[0] for f in fields]
            unknown = sorted(set(numbers) - _CLIP_FIELDS)
            if unknown or len(numbers) != len(set(numbers)):
                raise _bad(f"{what} holds fields this writer does not know or repeats: {unknown}")
            for number in _CLIP_PROJECT_LOCAL:
                if bt.pb_remove(fields, number):
                    self.notes.append(
                        f"{what} field {number} (project-local) left out, as on a fresh import"
                    )
            for f in fields:
                number = f[0]
                if number == 1:
                    f[2] = self.d.folder.encode()
                elif number in (2, 6):
                    f[2] = self.d.name.encode()
                elif number == 3:
                    f[2] = self.d.mtime_text.encode()
                elif number == 5:
                    if video:
                        if self.d.codec_tag is None:
                            raise _bad(f"{what} has a codec field but the file has no codec tag")
                        f[2] = self.d.codec_tag.encode()
                    else:
                        assert audio is not None
                        f[2] = audio["codec"].encode()
                elif number == 7:
                    if not video:
                        raise _bad(f"{what}: an audio Clip with field 7")
                    if isinstance(f[2], bytes):
                        self.template_ids.add(f[2].decode("utf-8", "replace"))
                    f[2] = self.ids().encode()
                elif number == 13:
                    f[2] = self.d.created_us
            for number in (1, 2, 3, 13):
                bt.pb_get(fields, number, what)

        return bt.edit_message(blob, what, change)

    # ------------------------------------------------------------------ BtVideoInfo

    def _video(self) -> Row:
        tpl = self.tpl.video
        assert tpl is not None and self.video_id is not None
        row = dict(tpl)
        row["BtVideoInfo_id"] = self.video_id
        row["Sm2MpMedia_id"] = self.media_id
        row["MediaMetadata"] = None
        row["Clip"] = self._clip(tpl["Clip"], "BtVideoInfo.Clip", video=True, audio=None)

        def time(entries: bt.KdEntries) -> None:
            _closed(entries, _TIME_KEYS, "Time")
            self._fresh(entries, "Time")
            assert self.d.rate is not None and self.d.frames is not None
            if bt.kd_get(entries, "Timecode") is not None:
                if self.d.timecode is None:
                    raise MediaPoolError(
                        "kind_mismatch",
                        "the template has a timecode, the file none",
                        "give a template of the same kind",
                    )
                bt.kd_set(entries, "Timecode", self.d.timecode, "Time")
            start = bt.kd_get(entries, "StartFrame")
            if start is None or start[3] != 0:
                raise _bad("Time.StartFrame is not 0")
            bt.kd_set(entries, "NumFrames", self.d.frames, "Time")
            bt.kd_set(entries, "FrameRate", _rate_bytes(self.d.rate), "Time")

        row["Time"] = bt.edit_kd(tpl["Time"], "Time", time)

        def geometry(entries: bt.KdEntries) -> None:
            _closed(entries, _GEOMETRY_KEYS, "Geometry")
            self._fresh(entries, "Geometry")
            assert self.d.resolution is not None
            w, h = self.d.resolution
            bt.kd_set(entries, "Resolution", struct.pack(">IIII", 0, w, 0, h), "Geometry")
            bt.kd_set(entries, "FrameSize", self.d.frame_size, "Geometry")

        row["Geometry"] = bt.edit_kd(tpl["Geometry"], "Geometry", geometry)

        def proxy(entries: bt.KdEntries) -> None:
            _closed(entries, _PROXY_KEYS, "Proxy")
            self._fresh(entries, "Proxy")

        row["Proxy"] = bt.edit_kd(tpl["Proxy"], "Proxy", proxy)

        def video_metadata(entries: bt.KdEntries) -> None:
            _closed(entries, _VIDEO_METADATA_KEYS, "VideoMetadata")
            self._fresh(entries, "VideoMetadata")
            if bt.kd_get(entries, "ActualReelNumber") is not None:
                # Seen only on camera files whose metadata this writer refuses; one value ("1")
                # cannot tell the reel tag from the scene or shot tag.
                raise MediaPoolError(
                    "cannot_derive",
                    "the template stores a reel number, and no rule for it is proven",
                    "import the file in Resolve by hand",
                )

        row["VideoMetadata"] = bt.edit_kd(tpl["VideoMetadata"], "VideoMetadata", video_metadata)
        return row

    # ------------------------------------------------------------------ BtAudioInfo

    def _audio(self, index: int) -> Row:
        tpl = self.tpl.audio[index]
        info = self.d.audio[index]
        row = dict(tpl)
        row["BtAudioInfo_id"] = self.audio_ids[index]
        row["Sm2MpMedia_id"] = self.media_id
        row["MediaMetadata"] = None
        row["Clip"] = self._clip(tpl["Clip"], f"BtAudioInfo[{index}].Clip", video=False, audio=info)

        def tracks(entries: bt.KdEntries) -> None:
            if [e[0] for e in entries] != ["0"] or not isinstance(entries[0][3], bytes):
                raise _bad("TracksBA does not hold one track")

            def track(t: bt.KdEntries) -> None:
                _closed(t, _TRACK_KEYS, "TracksBA track")
                self._fresh(t, "TracksBA track")
                bt.kd_set(t, "StartTime", self.d.start, "TracksBA track")
                bt.kd_set(t, "SampleRate", info["sample_rate"], "TracksBA track")
                bt.kd_set(t, "NumChannels", info["channels"], "TracksBA track")
                bt.kd_set(t, "Duration", info["samples"], "TracksBA track")
                bt.kd_set(t, "CodecName", info["codec"], "TracksBA track")
                bt.kd_set(t, "BitDepth", info["bit_depth"], "TracksBA track")
                layout = bt.kd_get(t, "ChannelLayout")
                if (layout is None) != (info["layout"] is None):
                    raise MediaPoolError(
                        "kind_mismatch",
                        "the template's channel layout differs",
                        "give a template of the same kind",
                    )
                if layout is not None:
                    layout[3] = info["layout"]

            entries[0][3] = bt.edit_kd(entries[0][3], "TracksBA track", track)

        row["TracksBA"] = bt.edit_kd(tpl["TracksBA"], "TracksBA", tracks)
        return row

    # ------------------------------------------------------------------ Sm2MpMedia

    def _extents(self, blob: bytes) -> bytes:
        def change(entries: bt.KdEntries) -> None:
            if [e[0] for e in entries] != [str(i) for i in range(len(self.d.audio))]:
                raise _bad("the clip's audio extents do not list its audio streams in order")
            for i, entry in enumerate(entries):
                info = self.d.audio[i]
                old_ref = self.tpl.audio[i]["BtAudioInfo_id"]

                def one(
                    t: bt.KdEntries, info: dict[str, Any] = info, old_ref: str = old_ref, i: int = i
                ) -> None:
                    _closed(t, _EXTENTS_KEYS, "the clip's audio extents")
                    bt.kd_set(
                        t,
                        "MediaExtents",
                        struct.pack("<dd", self.d.start, info["extent"]),
                        "extents",
                    )
                    bt.kd_set(t, "CodecName", info["codec"], "extents")
                    if bt.kd_get(t, "BitDepth") is not None:
                        bt.kd_set(t, "BitDepth", info["bit_depth"], "extents")
                    if bt.kd_get(t, "SampleRate") is not None:
                        bt.kd_set(t, "SampleRate", info["sample_rate"], "extents")

                    def channels(c: bt.KdEntries) -> None:
                        keys = sorted(e[0] for e in c)
                        if keys != sorted(str(n) for n in range(info["channels"])):
                            raise _bad("ChannelVecBA does not list the stream's channels")

                        def ref(r: bt.KdEntries, n: int) -> None:
                            entry = bt.kd_get(r, "MediaRef")
                            if entry is None or entry[3] != old_ref:
                                raise _bad("a ChannelVecBA entry does not name its audio info")
                            entry[3] = self.audio_ids[i]
                            idx = bt.kd_get(r, "ChannelIdx")
                            if (idx is None) != (n == 0) or (idx is not None and idx[3] != n):
                                raise _bad("a ChannelVecBA entry has an unexpected ChannelIdx")
                            unknown = [e[0] for e in r if e[0] not in ("MediaRef", "ChannelIdx")]
                            if unknown:
                                raise _bad(f"a ChannelVecBA entry holds {unknown}")

                        for ch in c:
                            n = int(ch[0])
                            ch[3] = bt.edit_kd(
                                ch[3], "ChannelVecBA entry", lambda r, n=n: ref(r, n)
                            )

                    vec = bt.kd_get(t, "ChannelVecBA")
                    if vec is None:
                        raise _bad("the clip's audio extents have no ChannelVecBA")
                    vec[3] = bt.edit_kd(vec[3], "ChannelVecBA", channels)

                entry[3] = bt.edit_kd(entry[3], "the clip's audio extents", one)

        return bt.edit_kd(blob, "the clip's audio extents", change)

    def _media_fields(self, blob: object) -> bytes:
        is_audio = self.tpl.is_audio_clip

        def change(outer: bt.Fields) -> None:
            first = bt.pb_get(outer, 1, "Sm2MpMedia.FieldsBlob")
            others = [f for f in outer if f[0] != 1]
            if is_audio:
                rate = bt.pb_get(outer, 2, "Sm2MpMedia.FieldsBlob")
                inner = bt.pb_decode(rate[2], "the sound clip's frame rate")
                if [(f[0], f[1]) for f in inner] != [(1, 1)]:
                    raise _bad("the sound clip's frame rate field is not one double")
                inner[0][2] = struct.pack("<d", float(self.placement.project_rate))
                rate[2] = bt.pb_encode(inner)
                if len(others) != 1:
                    raise _bad("a sound clip's FieldsBlob holds fields this writer does not know")
                level = first
            else:
                if others:
                    if (
                        len(others) != 1
                        or others[0][0] != 2
                        or not _is_relink_history(others[0][2])
                    ):
                        raise _bad(
                            "a video clip's FieldsBlob holds fields this writer does not know"
                        )
                    bt.pb_remove(outer, 2)
                    self.notes.append(
                        "the template's relink history (FieldsBlob field 2) left out, "
                        "as on a fresh import"
                    )
                v = bt.pb_decode(first[2], "Sm2MpMedia.FieldsBlob #1")
                if bt.pb_remove(v, 3):
                    self.notes.append(
                        "FieldsBlob #1.#3 (project-local) left out, as on a fresh import"
                    )
                if sorted(f[0] for f in v) != [1, 2]:
                    raise _bad(
                        "a video clip's FieldsBlob #1 holds fields this writer does not know"
                    )
                level = bt.pb_get(v, 1, "FieldsBlob #1.#1")
                self._media_level(level)
                first[2] = bt.pb_encode(v)
                return
            self._media_level(level)

        return bt.edit_message(blob, "Sm2MpMedia.FieldsBlob", change)

    def _media_level(self, level: list[Any]) -> None:
        """The level holding the date added (``#1.#1.#2``) and the audio extents (``#5``)."""
        w = bt.pb_decode(level[2], "the clip's media fields")
        allowed = {1, 5} if self.d.audio else {1}
        if not {f[0] for f in w} <= allowed or len({f[0] for f in w}) != len(w):
            raise _bad("the clip's media fields hold fields this writer does not know")
        x = bt.pb_get(w, 1, "the clip's media fields")
        xf = bt.pb_decode(x[2], "the clip's date fields")
        if sorted(f[0] for f in xf) != [1, 6]:
            raise _bad("the clip's date fields are not the known pair")
        date = bt.pb_get(xf, 1, "the clip's date fields")
        df = bt.pb_decode(date[2], "the clip's date added")
        if [(f[0], f[1]) for f in df] != [(2, 0)]:
            raise _bad("the clip's date added is not one varint")
        df[0][2] = _zigzag(self.placement.added_at)
        date[2] = bt.pb_encode(df)
        x[2] = bt.pb_encode(xf)
        extents = [f for f in w if f[0] == 5]
        if self.d.audio:
            if len(extents) != 1:
                raise _bad("the clip has audio but its FieldsBlob lists no audio extents")
            extents[0][2] = self._extents(extents[0][2])
        level[2] = bt.pb_encode(w)

    def _media(self) -> Row:
        tpl = self.tpl.media
        row = dict(tpl)
        row["Sm2MpMedia_id"] = self.media_id
        row["UniqueMediaPoolItemId"] = self.ids()
        row["Name"] = self.d.name
        row["MpFolder"] = row["Sm2MpFolder_id"] = self.placement.folder_id
        row["Video"] = self.video_id
        for column in _MEDIA_CLEARED:
            if column in row:
                row[column] = None
        row["FieldsBlob"] = self._media_fields(tpl["FieldsBlob"])
        return row

    # ------------------------------------------------------------------ metadata

    def _metadata(self, tpl: Row, owner: str) -> Row:
        if self.placement.lockable_map_id is None:
            raise MediaPoolError(
                "cannot_derive",
                "the kind has a metadata blob and the placement names no BtLockableBlobMap",
                "give the project's lockable map id",
            )
        row = dict(tpl)
        row["BtLockableBlob_id"] = self.ids()
        row["BtLockableBlobMap_id"] = self.placement.lockable_map_id
        row["BlobOwner"] = owner
        row["DbSavedTime"] = self.placement.saved_age_ms

        def outer(entries: bt.KdEntries) -> None:
            if [e[0] for e in entries] != ["BlobData"]:
                raise _bad("the metadata blob is not one BlobData")

            def data(fields: bt.Fields) -> None:
                holder = bt.pb_get(fields, 2, "metadata BlobData")
                original, payload = bt.message_of(holder[2], "metadata entries")
                entries_pb = bt.pb_decode(payload, "metadata entries")
                for entry in entries_pb:
                    if entry[0] != 2 or entry[1] != 2:
                        raise _bad("a metadata entry is not field 2")
                    sub = bt.pb_decode(entry[2], "metadata entry")
                    if [(f[0], f[1]) for f in sub] != [(1, 0), (2, 2)]:
                        raise _bad("a metadata entry is not a key and a value")
                    key, value = sub[0][2], sub[1][2]
                    rule = METADATA_RULES.get(key)
                    if rule is not None:
                        sub[1][2] = rule(self.facts).encode()
                    elif value != b"":
                        raise MediaPoolError(
                            "cannot_derive",
                            f"metadata entry {key} holds a value in the template and no "
                            "rule derives it",
                            "import the file in Resolve by hand",
                        )
                    entry[2] = bt.pb_encode(sub)
                holder[2] = bt.rewrap(original, bt.pb_encode(entries_pb), "metadata entries")

            entries[0][3] = bt.edit_message(entries[0][3], "metadata BlobData", data)

        row["FieldsBlob"] = bt.edit_kd(tpl["FieldsBlob"], "the metadata blob", outer)
        return row

    # ------------------------------------------------------------------ all rows

    def build(self) -> dict[str, tuple[Row, ...]]:
        media = self._media()
        out: dict[str, list[Row]] = {"Sm2MpMedia": [media]}
        if self.tpl.video is not None:
            out["BtVideoInfo"] = [self._video()]
        out["BtAudioInfo"] = [self._audio(i) for i in range(len(self.tpl.audio))]
        out["BtAudioInfo_Sm2MpMedia"] = [
            dict(link, DbOwner=self.media_id, DbAssociate=self.audio_ids[i])
            for i, link in enumerate(self.tpl.audio_links)
        ]
        out["Sm2MpFolder_Sm2MpMedia"] = [
            dict(
                self.tpl.folder_link,
                DbOwner=self.placement.folder_id,
                DbAssociate=self.media_id,
                DbIndex=self.placement.folder_index,
            )
        ]
        owner = self.video_id if self.video_id is not None else self.audio_ids[0]
        out["BtLockableBlob"] = [self._metadata(r, owner) for r in self.tpl.lockables]
        if self.tpl.presets:
            self.notes.append("the template's pan/tilt/zoom preset left out, as on a fresh import")
        result = {t: tuple(rows) for t, rows in out.items() if rows}
        self._check_no_template_left(result)
        return result

    def _check_no_template_left(self, rows: dict[str, tuple[Row, ...]]) -> None:
        needles = {i: f"template id {i}" for i in self.template_ids if i}
        tpl_media = self.tpl.media
        for value, label in (
            (tpl_media.get("MpFolder"), "the template's bin id"),
            (
                self.tpl.lockables[0].get("BtLockableBlobMap_id") if self.tpl.lockables else None,
                "the template's lockable map id",
            ),
        ):
            if (
                isinstance(value, str)
                and value
                and value
                not in (
                    self.placement.folder_id,
                    self.placement.lockable_map_id,
                )
            ):
                needles[value] = label
        name = _clip_name(self.tpl)
        for text, label in (
            (name[0], "the template's folder"),
            (name[1], "the template's file name"),
        ):
            if text and text not in self.d.folder and text not in self.d.name:
                needles[text] = label
        for table, table_rows in rows.items():
            for row in table_rows:
                for column, value in row.items():
                    found = bt.contains_text(value, needles)
                    if found is not None:
                        raise _bad(
                            f"{found} is still in the new {table}.{column}: it is somewhere "
                            "this writer does not rewrite"
                        )


def _check_picture_decoded(facts: MediaFacts) -> None:
    """Refuse a file whose picture ffprobe does not decode (BRAW: no codec, size or pixel
    format; only its sound and timecode are measured)."""
    video = fm.video_stream(facts)
    if video is not None and (
        video.codec_name is None or video.width is None or video.pix_fmt is None
    ):
        raise MediaPoolError(
            "cannot_derive",
            f"ffprobe does not decode the picture of {facts.name} "
            "(no codec, size or pixel format), "
            "so its resolution, frame size and frame count cannot be derived",
            "import the file in Resolve by hand (BRAW), or author its proxy",
        )


def _clip_name(clip: ClipRows) -> tuple[str, str]:
    row = clip.video if clip.video is not None else clip.audio[0]
    _, payload = bt.message_of(row["Clip"], "Clip")
    fields = bt.pb_decode(payload, "Clip")
    folder = bt.pb_get(fields, 1, "Clip")[2]
    name = bt.pb_get(fields, 2, "Clip")[2]
    return folder.decode("utf-8", "replace"), name.decode("utf-8", "replace")


def _is_relink_history(value: object) -> bool:
    """A video clip's ``FieldsBlob`` field 2: a keyed dict of the infos the clip had before it
    was relinked (``VIDEO_INFO``, ``AUDIO_INFO_VEC``, ``AUDIO_TRACKS_EXTENTS``)."""
    if not isinstance(value, bytes):
        return False
    try:
        entries = bt.kd_decode(value, "relink history")
    except MediaPoolError:
        return False
    return {e[0] for e in entries} <= {"VIDEO_INFO", "AUDIO_INFO_VEC", "AUDIO_TRACKS_EXTENTS"}
