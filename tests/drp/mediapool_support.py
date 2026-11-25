"""Synthetic media clips and facts for the S25c1 tests (nothing here is product code).

The rows follow the layout recorded in DESIGN "Media-pool rows (S25c1)" and are built with the
project's own encoders (:mod:`aar.drp.modify.encode`); the facts are shaped like the JSON lines
the edit PC's facts script writes (``ffprobe -show_format -show_streams`` plus size and times).
Every value is made up.
"""

from __future__ import annotations

import copy
import sqlite3
import struct
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.mediapool.templates import ClipRows
from aar.drp.modify import encode

S, I32, I64, B, BYTES, DBL = (
    blobs.KD_STRING,
    blobs.KD_INT32,
    blobs.KD_INT64,
    blobs.KD_BOOL,
    blobs.KD_BYTES,
    blobs.KD_DOUBLE,
)
FOLDER = "f0000000-0000-4000-8000-000000000001"
MAP = "f0000000-0000-4000-8000-000000000002"
PC_FOLDER = "X:\\Show\\Sound"
WAV_NAME = "take-one.wav"
MP4_NAME = "camera-a.MP4"
PNG_NAME = "lower-third.png"


def uid(n: int) -> str:
    return str(uuid.UUID(int=n))


def counter(start: int) -> Iterator[str]:
    n = start
    while True:
        yield uid(n)
        n += 1


def new_ids(start: int = 0x9000):
    it = counter(start)
    return lambda: next(it)


def kd(*entries: tuple[str, int, Any]) -> bytes:
    return encode.keyed_dict([(k, t, 0, v) for k, t, v in entries])


def pb(*fields: tuple[int, int, Any]) -> bytes:
    return encode.protobuf(list(fields))


def envelope(payload: bytes, *, version: int = 2, zstd: bool = True) -> bytes:
    body = b"\x81" + encode.zstd_raw_frame(payload) if zstd else b"\x80" + payload
    return struct.pack(">II", version, len(body)) + body


def header(payload: bytes, version: int) -> bytes:
    """The codec-less framing a metadata blob's entries use."""
    return struct.pack(">II", version, len(payload)) + payload


def rate16(fps: float) -> bytes:
    return struct.pack("<d", fps) + bytes(8)


def clip_blob(
    folder: str,
    name: str,
    seen: str,
    codec: str | None,
    created: int,
    *,
    video: bool,
    clip_uuid: str | None = None,
    flags: int = 111,
    extra: tuple[tuple[int, int, Any], ...] = (),
) -> bytes:
    fields: list[tuple[int, int, Any]] = [
        (1, 2, folder.encode()),
        (2, 2, name.encode()),
        (3, 2, seen.encode()),
    ]
    if codec is not None:
        fields.append((5, 2, codec.encode()))
    if video:
        fields.append((6, 2, name.encode()))
        fields.append((7, 2, (clip_uuid or uid(0x7777)).encode()))
    fields += [(13, 0, created), (15, 0, 7), (16, 0, 70), (18, 0, flags), *extra]
    return envelope(pb(*fields))


def track(
    unique: str,
    *,
    start: float,
    rate: int,
    channels: int,
    samples: int,
    codec: str,
    depth: int,
    layout: int | None = None,
) -> bytes:
    entries: list[tuple[str, int, Any]] = [
        ("UniqueId", S, unique),
        ("StartTime", DBL, start),
        ("SampleRate", I32, rate),
        ("NumChannels", I32, channels),
        ("IdxTrack", I32, 0),
        ("Duration", I64, samples),
        ("DbType", S, "BtAudioTrack"),
        ("CodecName", S, codec),
    ]
    if layout is not None:
        entries.append(("ChannelLayout", I32, layout))
    entries.append(("BitDepth", I32, depth))
    return kd(("0", BYTES, kd(*entries)))


def channel_vec(audio_id: str, channels: int) -> bytes:
    # Resolve lists the channels highest first ("1" before "0"); the writer must not care.
    entries = []
    for n in reversed(range(channels)):
        ref: list[tuple[str, int, Any]] = [("MediaRef", S, audio_id)]
        if n:
            ref.append(("ChannelIdx", I32, n))
        entries.append((str(n), BYTES, kd(*ref)))
    return kd(*entries)


def extents(
    audio_id: str, *, channels: int, start: float, length: float, codec: str, depth: int
) -> bytes:
    one = kd(
        ("Type", I32, 0),
        ("MediaExtents", BYTES, struct.pack("<dd", start, length)),
        ("CodecName", S, codec),
        ("ChannelVecBA", BYTES, channel_vec(audio_id, channels)),
        ("BitDepth", I32, depth),
    )
    return kd(("0", BYTES, one))


def zigzag(n: int) -> int:
    return n << 1


def media_fields(
    *,
    added: int,
    video: bool,
    audio_extents: bytes | None,
    project_rate: float = 30000 / 1001,
    relink: bytes | None = None,
    later_age: int | None = None,
) -> bytes:
    date = pb((2, 0, zigzag(added)))
    x = pb((1, 2, date), (6, 0, 4))
    w_fields: list[tuple[int, int, Any]] = [(1, 2, x)]
    if audio_extents is not None:
        w_fields.append((5, 2, audio_extents))
    w = pb(*w_fields)
    if video:
        v_fields: list[tuple[int, int, Any]] = [(1, 2, w), (2, 2, bytes(range(16)))]
        if later_age is not None:
            v_fields.append((3, 0, later_age))
        outer: list[tuple[int, int, Any]] = [(1, 2, pb(*v_fields))]
        if relink is not None:
            outer.append((2, 2, relink))
    else:
        outer = [(1, 2, w), (2, 2, pb((1, 1, struct.pack("<d", project_rate))))]
    return envelope(pb(*outer))


def metadata(entries: list[tuple[int, str]], *, zstd: bool = False) -> bytes:
    body = pb(*((2, 2, pb((1, 0, k), (2, 2, v.encode()))) for k, v in entries))
    data = pb((2, 2, header(body, 1000001)))
    return kd(("BlobData", BYTES, envelope(data, version=10001, zstd=zstd)))


def media_row(
    mid: str, name: str, kind: str, *, video_id: str | None, fields: bytes, pool: str
) -> dict[str, Any]:
    return {
        "Sm2MpMedia_id": mid,
        "DbType": kind,
        "Name": name,
        "MpFolder": FOLDER,
        "CurPlayheadPosition": "17",
        "AudioSource": "AUDIO_SOURCE_EMBEDDED" if video_id else None,
        "PTZRPreset": None,
        "PTZRPresetType": 0,
        "PinsBA": b"",
        "Sm2MpFolder_id": FOLDER,
        "Video": video_id,
        "UniqueMediaPoolItemId": pool,
        "VideoType": 0,
        "ParentMpMediaId": "",
        "FieldsBlob": fields,
        "BlobLockSysId": "",
    }


def audio_row(aid: str, mid: str, clip: bytes, tracks: bytes) -> dict[str, Any]:
    return {
        "BtAudioInfo_id": aid,
        "DbType": "BtAudioInfo",
        "Clip": clip,
        "TracksBA": tracks,
        "MediaMetadata": b"",
        "Sm2MpMedia_id": mid,
        "Sm2Timeline_id": None,
        "FieldsBlob": None,
        "BlobLockSysId": "",
    }


def join(owner: str, associate: str, prop: str, index: int) -> dict[str, Any]:
    return {"DbOwner": owner, "DbAssociate": associate, "DbPropertyName": prop, "DbIndex": index}


def lockable(lid: str, owner: str, fields: bytes) -> dict[str, Any]:
    return {
        "BtLockableBlob_id": lid,
        "DbType": "BtMetadataLockableBlob",
        "BtLockableBlobMap_id": MAP,
        "FieldsBlob": fields,
        "BlobOwner": owner,
        "DbSavedTime": 123456,
    }


# --------------------------------------------------------------------------- WAV


def wav_facts(**change: Any) -> dict[str, Any]:
    """A stereo 24-bit WAV at 48 kHz, 10 s long, with no channel mask."""
    stream = {
        "index": 0,
        "codec_name": "pcm_s24le",
        "codec_type": "audio",
        "codec_tag_string": "[1][0][0][0]",
        "sample_fmt": "s32",
        "sample_rate": "48000",
        "channels": 2,
        "bits_per_sample": 24,
        "r_frame_rate": "0/0",
        "avg_frame_rate": "0/0",
        "time_base": "1/48000",
        "duration_ts": 480000,
        "duration": "10.000000",
        "bits_per_raw_sample": "24",
        "disposition": {"default": 0, "attached_pic": 0},
    }
    data: dict[str, Any] = {
        "path": f"{PC_FOLDER}\\{WAV_NAME}",
        "exists": True,
        "size": 2_880_044,
        "mtime_local": "Tue Mar 3 04:05:06 2026",
        "mtime_utc": "2026-03-03T09:05:06.1234567Z",
        "ctime_utc": "2026-03-03T09:01:02.7654321Z",
        "ffprobe": {
            "streams": [stream],
            "format": {
                "filename": f"{PC_FOLDER}\\{WAV_NAME}",
                "format_name": "wav",
                "duration": "10.000000",
            },
        },
    }
    for key, value in change.items():
        if key.startswith("stream_"):
            stream[key[len("stream_") :]] = value
        else:
            data[key] = value
    return data


def wav_template(
    *, base: int = 0x100, name: str = "older-take.wav", folder: str = "X:\\Old", rate: int = 48000
) -> ClipRows:
    """A stereo 24-bit WAV clip as Resolve saves it (5.0 s long)."""
    mid, aid, pool, lid = uid(base), uid(base + 1), uid(base + 2), uid(base + 3)
    seen = "Mon Oct  5 09:01:02 2026"
    clip = clip_blob(
        folder, name, seen, "Linear PCM", 1_791_000_000_000_000, video=False, flags=222
    )
    tracks = track(
        uid(base + 4), start=0.0, rate=rate, channels=2, samples=240000, codec="Linear PCM", depth=2
    )
    fields = media_fields(
        added=1_791_000_100,
        video=False,
        audio_extents=extents(aid, channels=2, start=0.0, length=5.0, codec="Linear PCM", depth=2),
    )
    return ClipRows(
        media=media_row(mid, name, "Sm2MpAudioClip", video_id=None, fields=fields, pool=pool),
        video=None,
        audio=(audio_row(aid, mid, clip, tracks),),
        audio_links=(join(mid, aid, "EmbeddedAudioVec", 0),),
        folder_link=join(FOLDER, mid, "MediaVec", 3),
        lockables=(lockable(lid, aid, metadata([(422, ""), (215, name), (16, "")])),),
    )


# --------------------------------------------------------------------------- MP4 (H.264 + PCM)


def mp4_facts(**change: Any) -> dict[str, Any]:
    """A 3840x2160 H.264 camera file at 23.976 with 16-bit PCM and a timecode on its metadata
    track; 125.125 s (3000 frames)."""
    video = {
        "index": 0,
        "codec_name": "h264",
        "codec_type": "video",
        "codec_tag_string": "avc1",
        "width": 3840,
        "height": 2160,
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "r_frame_rate": "24000/1001",
        "avg_frame_rate": "24000/1001",
        "time_base": "1/24000",
        "duration_ts": 3003000,
        "nb_frames": "3000",
        "disposition": {"default": 1, "attached_pic": 0},
        "tags": {"creation_time": "2026-03-20T11:22:33.000000Z"},
    }
    audio = {
        "index": 1,
        "codec_name": "pcm_s16be",
        "codec_type": "audio",
        "codec_tag_string": "twos",
        "sample_fmt": "s16",
        "sample_rate": "48000",
        "channels": 2,
        "bits_per_sample": 16,
        "r_frame_rate": "0/0",
        "avg_frame_rate": "0/0",
        "time_base": "1/48000",
        "duration_ts": 6006000,
    }
    data_track = {
        "index": 2,
        "codec_type": "data",
        "codec_tag_string": "rtmd",
        "r_frame_rate": "0/0",
        "avg_frame_rate": "0/0",
        "time_base": "1/24000",
        "duration_ts": 3003000,
        "tags": {"timecode": "02:10:20:12"},
    }
    data: dict[str, Any] = {
        "path": f"X:\\Show\\Footage\\{MP4_NAME}",
        "exists": True,
        "size": 1_234_567_890,
        "mtime_local": "Fri Mar 20 07:30:00 2026",
        "mtime_utc": "2026-03-20T11:30:00.0000000Z",
        "ctime_utc": "2026-03-21T12:00:00.5000000Z",
        "ffprobe": {
            "streams": [video, audio, data_track],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "125.125000",
                "tags": {"creation_time": "2026-03-20T11:22:33.000000Z"},
            },
        },
    }
    for key, value in change.items():
        if key.startswith("video_"):
            video[key[len("video_") :]] = value
        elif key.startswith("audio_"):
            audio[key[len("audio_") :]] = value
        elif key.startswith("data_"):
            data_track[key[len("data_") :]] = value
        else:
            data[key] = value
    return data


def mp4_template(
    *,
    base: int = 0x200,
    name: str = "camera-b.MP4",
    folder: str = "X:\\Old\\Footage",
    relink: bool = False,
    clip_extra: tuple[tuple[int, int, Any], ...] = ((14, 0, 4321), (26, 2, b"Old\\Footage")),
    radiometry: bytes | None = None,
    presets: bool = False,
) -> ClipRows:
    """A camera clip of the same kind (another file: 10 s, 01:00:00:00)."""
    mid, vid, aid, pool = uid(base), uid(base + 1), uid(base + 2), uid(base + 3)
    seen = "Tue Sep  1 08:00:00 2026"
    vclip = clip_blob(
        folder,
        name,
        seen,
        "avc1",
        1_788_000_000_000_000,
        video=True,
        clip_uuid=uid(base + 4),
        extra=clip_extra,
    )
    aclip = clip_blob(
        folder, name, seen, "Linear PCM", 1_788_000_000_000_000, video=False, extra=clip_extra
    )
    time = kd(
        ("UniqueId", S, uid(base + 5)),
        ("Timecode", S, "01:00:00:00"),
        ("StartFrame", I32, 0),
        ("NumFrames", I32, 240),
        ("FrameRate", BYTES, rate16(24000 / 1001)),
        ("DbType", S, "BtVideoTime"),
    )
    geometry = kd(
        ("UniqueId", S, uid(base + 6)),
        ("Resolution", BYTES, struct.pack(">IIII", 0, 3840, 0, 2160)),
        ("FrameSize", I32, 3840 * 2160 * 3 // 2 + 8192),
        ("DbType", S, "BtGeometry"),
    )
    proxy = kd(
        ("UniqueId", S, uid(base + 7)), ("DbType", S, "BtVideoProxy"), ("DataManagerID", I32, 5)
    )
    video_meta = kd(("UniqueId", S, uid(base + 8)), ("DbType", S, "BtVideoMetadata"))
    start = 86400 / (24000 / 1001)
    tracks = track(
        uid(base + 9),
        start=start,
        rate=48000,
        channels=2,
        samples=480480,
        codec="Linear PCM",
        depth=1,
    )
    history = (
        kd(
            ("VIDEO_INFO", BYTES, kd(("DbType", S, "BtVideoInfo"))),
            ("AUDIO_TRACKS_EXTENTS", BYTES, b""),
        )
        if relink
        else None
    )
    fields = media_fields(
        added=1_788_100_000,
        video=True,
        audio_extents=extents(
            aid, channels=2, start=start, length=10.01, codec="Linear PCM", depth=1
        ),
        relink=history,
        later_age=123456789012 if relink else None,
    )
    media = media_row(mid, name, "Sm2MpVideoClip", video_id=vid, fields=fields, pool=pool)
    preset_rows: tuple[dict[str, Any], ...] = ()
    if presets:
        media["PTZRPreset"] = uid(base + 10)
        preset_rows = (
            {
                "SmPreset_id": uid(base + 10),
                "DbType": "SmPTZRPreset",
                "Name": "",
                "Presets": b"{}",
                "Sm2MpMedia_id": mid,
            },
        )
    return ClipRows(
        media=media,
        video={
            "BtVideoInfo_id": vid,
            "DbType": "BtVideoInfo",
            "Clip": vclip,
            "Time": time,
            "Geometry": geometry,
            "Radiometry": radiometry
            if radiometry is not None
            else b"\x80" + pb((1, 0, 11), (2, 0, 22)),
            "Proxy": proxy,
            "VideoMetadata": video_meta,
            "MediaMetadata": None,
            "Type": "STANDARD_CLIP",
            "Eye": "SUB_TRACK_MONO",
            "Sm2MpMedia_id": mid,
            "FieldsBlob": None,
            "BlobLockSysId": "",
        },
        audio=(audio_row(aid, mid, aclip, tracks),),
        audio_links=(join(mid, aid, "EmbeddedAudioVec", 0),),
        folder_link=join(FOLDER, mid, "MediaVec", 0),
        presets=preset_rows,
    )


# --------------------------------------------------------------------------- PNG


def png_facts(**change: Any) -> dict[str, Any]:
    stream = {
        "index": 0,
        "codec_name": "png",
        "codec_type": "video",
        "codec_tag_string": "[0][0][0][0]",
        "width": 3840,
        "height": 2160,
        "pix_fmt": "rgba",
        "color_range": "pc",
        "color_space": "gbr",
        "r_frame_rate": "25/1",
        "avg_frame_rate": "25/1",
        "time_base": "1/25",
    }
    data: dict[str, Any] = {
        "path": f"X:\\Show\\Graphics\\{PNG_NAME}",
        "exists": True,
        "size": 40961,
        "mtime_local": "Wed Mar 4 10:00:00 2026",
        "mtime_utc": "2026-03-04T15:00:00.0000000Z",
        "ctime_utc": "2026-03-04T15:00:00.0000000Z",
        "ffprobe": {"streams": [stream], "format": {"format_name": "png_pipe"}},
    }
    for key, value in change.items():
        if key.startswith("stream_"):
            stream[key[len("stream_") :]] = value
        else:
            data[key] = value
    return data


def png_template(*, base: int = 0x300, name: str = "title-card.png") -> ClipRows:
    mid, vid, pool = uid(base), uid(base + 1), uid(base + 2)
    clip = clip_blob(
        "X:\\Old",
        name,
        "Tue Nov  3 10:00:00 2026",
        None,
        1_793_000_000_000_000,
        video=True,
        flags=333,
    )
    time = kd(
        ("UniqueId", S, uid(base + 3)),
        ("StartFrame", I32, 0),
        ("NumFrames", I32, 1),
        ("FrameRate", BYTES, rate16(24.0)),
        ("DbType", S, "BtVideoTime"),
    )
    geometry = kd(
        ("UniqueId", S, uid(base + 4)),
        ("ScanType", I32, 0),
        ("Resolution", BYTES, struct.pack(">IIII", 0, 1920, 0, 1080)),
        ("FrameSize", I32, 4000),
        ("DbType", S, "BtGeometry"),
    )
    return ClipRows(
        media=media_row(
            mid,
            name,
            "Sm2MpVideoClip",
            video_id=vid,
            fields=media_fields(added=1_793_000_000, video=True, audio_extents=None),
            pool=pool,
        ),
        video={
            "BtVideoInfo_id": vid,
            "DbType": "BtVideoInfo",
            "Clip": clip,
            "Time": time,
            "Geometry": geometry,
            "Radiometry": b"\x80" + pb((1, 0, 33), (6, 0, 44)),
            "Proxy": kd(
                ("UniqueId", S, uid(base + 5)),
                ("DbType", S, "BtVideoProxy"),
                ("DataManagerID", I32, 5),
            ),
            "VideoMetadata": kd(("UniqueId", S, uid(base + 6)), ("DbType", S, "BtVideoMetadata")),
            "MediaMetadata": None,
            "Type": "STANDARD_CLIP",
            "Eye": "SUB_TRACK_MONO",
            "Sm2MpMedia_id": mid,
            "FieldsBlob": None,
            "BlobLockSysId": "",
        },
        audio=(),
        audio_links=(),
        folder_link=join(FOLDER, mid, "MediaVec", 1),
    )


# --------------------------------------------------------------------------- helpers


def replace_row(clip: ClipRows, role: str, **values: Any) -> ClipRows:
    """A copy of ``clip`` with columns of one row changed (``media``, ``video``, ``audio0``...)."""
    clip = copy.deepcopy(clip)
    if role.startswith("audio"):
        rows = list(clip.audio)
        rows[int(role[5:])] = {**rows[int(role[5:])], **values}
        return ClipRows(
            clip.media,
            clip.video,
            tuple(rows),
            clip.audio_links,
            clip.folder_link,
            clip.lockables,
            clip.presets,
        )
    if role == "media":
        return ClipRows(
            {**clip.media, **values},
            clip.video,
            clip.audio,
            clip.audio_links,
            clip.folder_link,
            clip.lockables,
            clip.presets,
        )
    if role == "video":
        assert clip.video is not None
        return ClipRows(
            clip.media,
            {**clip.video, **values},
            clip.audio,
            clip.audio_links,
            clip.folder_link,
            clip.lockables,
            clip.presets,
        )
    if role == "lockable":
        rows = [{**clip.lockables[0], **values}]
        return ClipRows(
            clip.media,
            clip.video,
            clip.audio,
            clip.audio_links,
            clip.folder_link,
            tuple(rows),
            clip.presets,
        )
    raise ValueError(role)


def tables_of(*clips: ClipRows) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        t: []
        for t in (
            "Sm2MpMedia",
            "BtVideoInfo",
            "BtAudioInfo",
            "BtAudioInfo_Sm2MpMedia",
            "Sm2MpFolder_Sm2MpMedia",
            "BtLockableBlob",
            "SmPreset",
        )
    }
    for clip in clips:
        for table, rows in clip.tables().items():
            out[table].extend(copy.deepcopy(rows))
    return out


def setup_ba(rate_text: str) -> bytes:
    return kd(("SetupBA", BYTES, envelope(pb((69, 2, rate_text.encode())), version=3)))


def write_project(
    path: Path, *clips: ClipRows, rate_text: str = "29.97", version: int = 17
) -> Path:
    """A ``Project.db`` holding ``clips`` (and a project, config and lockable map row)."""
    tables = tables_of(*clips)
    tables["SM_Project"] = [
        {"SM_Project_id": uid(0xA), "ProjectName": "Synthetic", "ProjectVersion": version}
    ]
    tables["SM_Config"] = [{"SM_Config_id": uid(0xB), "FieldsBlob": setup_ba(rate_text)}]
    tables["BtLockableBlobMap"] = [{"BtLockableBlobMap_id": MAP}]
    columns: dict[str, list[str]] = {}
    for table, rows in tables.items():
        names: list[str] = []
        for row in rows:
            names += [c for c in row if c not in names]
        columns[table] = names or [f"{table}_id"]
    conn = sqlite3.connect(path)
    try:
        for table, names in columns.items():
            conn.execute(
                f'CREATE TABLE "{table}" ({", ".join(chr(34) + c + chr(34) for c in names)})'
            )
            for row in tables[table]:
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
