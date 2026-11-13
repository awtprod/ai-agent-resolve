"""Helpers for the timeline import tests (S20): a plan's S18 layout as ``aar.drp`` timeline items,
simple edits on them, and a small synthetic ``.drt`` built with the ``tests/drp`` builders.

:func:`rows_from_layout` is the written form of the first S20 assumption: after Resolve imports
aar's FCPXML, layout lane N sits on video track N+1, each clip's linked sound on audio track N+1
(S18's ``imported_audio_track``), each title on its lane's track and each spine transition on V1
over its window. Nothing here is product code.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from aar.drp.model import AudioTrackInfo, MediaClip, Timeline, TimelineItem, Track
from aar.plan.export.layout import Layout

_DRP_TESTS = Path(__file__).resolve().parent.parent / "drp"
if str(_DRP_TESTS) not in sys.path:
    sys.path.insert(0, str(_DRP_TESTS))

from drp_helpers import envelope, kd, pb_len, qstr, xml  # noqa: E402

TC_START = 108000  # 01:00:00:00 at a nominal 30


@dataclass
class Row:
    """One timeline item before it becomes an ``aar.drp`` item (record frames from the start)."""

    track: str
    kind: str
    record_in: int
    duration: int
    src_in: int | None = None
    path: str | None = None
    name: str = ""
    #: The audio track of the clip's linked sound (picture rows), moved with the picture.
    sound: str | None = None
    widen_in: int = 0
    widen_out: int = 0
    speed: float | None = 1.0
    fractions: dict[str, float] = field(default_factory=dict)
    role: str = ""

    @property
    def record_out(self) -> int:
        return self.record_in + self.duration


def rows_from_layout(layout: Layout) -> list[Row]:
    """The timeline Resolve is assumed to hold after importing the layout's FCPXML."""
    paths = {ref.src: ref.windows_path for ref in layout.media.values()}
    rows: list[Row] = []
    for clip in layout.clips:
        sound_in, sound_out = layout.sound_range(clip)
        rows.append(
            Row(
                track=f"V{clip.lane + 1}",
                kind="video",
                record_in=clip.record_in,
                duration=clip.record_out - clip.record_in,
                src_in=clip.src_in,
                path=paths[clip.src],
                name=clip.name,
                sound=f"A{layout.imported_audio_track(clip.lane)}",
                widen_in=clip.record_in - sound_in,
                widen_out=sound_out - clip.record_out,
                role="outro" if clip.location == "outro" else "",
            )
        )
    for title in layout.titles:
        rows.append(
            Row(f"V{title.lane + 1}", "title", title.record_in, title.record_out - title.record_in)
        )
    for spec in layout.transitions:
        if spec.form == "transition":
            rows.append(Row("V1", "transition", spec.window_start, spec.frames))
    return rows


def build_timeline(
    rows: list[Row], *, rate: float, start: int = TC_START, media_rate: float | None = None
) -> tuple[Timeline, dict[str, MediaClip]]:
    """``aar.drp`` model objects for ``rows``: picture rows bring their linked sound."""
    media: dict[str, MediaClip] = {}
    tracks: dict[str, list[TimelineItem]] = {}
    count = 0

    def clip_id(path: str) -> str:
        found = next((c.id for c in media.values() if c.path == path), None)
        if found is not None:
            return found
        cid = f"clip{len(media) + 1}"
        media[cid] = MediaClip(
            id=cid,
            element="Sm2MpVideoClip",
            type="video",
            name=path.rsplit("\\", 1)[-1],
            bin_id="bin1",
            pool_item_id=None,
            path=path,
            path_source="media_pool_blob",
            blob_path=path,
            timeline_paths=[],
            codec="prores",
            frames=None,
            start_frame=None,
            frame_rate=media_rate if media_rate is not None else rate,
            resolution=(1920, 1080),
            audio_tracks=[AudioTrackInfo("0", 48000, 2, "lpcm", None)],
            sequence_id=None,
            timeline_id=None,
        )
        return cid

    def add(track: str, kind: str, rin: int, dur: int, src: int | None, row: Row) -> None:
        nonlocal count
        count += 1
        retimed = row.speed is not None and row.speed != 1.0
        tracks.setdefault(track, []).append(
            TimelineItem(
                id=f"item{count}",
                element={"video": "Sm2TiVideoClip", "audio": "Sm2TiAudioClip"}.get(
                    kind, "Sm2TiTransition" if kind == "transition" else "Sm2TiVideoClip"
                ),
                kind=kind,
                name=row.name or None,
                pretty_type="Fusion Title" if kind == "title" else None,
                record_start=start + rin,
                record_end=start + rin + dur,
                duration=dur,
                source_in=src,
                source_out=None if src is None or retimed else src + dur,
                media_ref=None,
                media_clip_id=clip_id(row.path) if row.path else None,
                media_path=row.path,
                path_source="media_pool_blob" if row.path else "none",
                media_file_path=None,
                media_track_index=None,
                speed=row.speed if row.path else None,
                timemap_form=("retimed" if retimed else "identity") if row.path else None,
                fractions=dict(row.fractions),
            )
        )

    for row in rows:
        add(row.track, row.kind, row.record_in, row.duration, row.src_in, row)
        if row.kind == "video" and row.sound is not None:
            assert row.src_in is not None
            add(
                row.sound,
                "audio",
                row.record_in - row.widen_in,
                row.duration + row.widen_in + row.widen_out,
                row.src_in - row.widen_in,
                replace(row, fractions={}),
            )
    out: list[Track] = []
    for prefix, kind in (("V", "video"), ("A", "audio")):
        used = [int(t[1:]) for t in tracks if t.startswith(prefix)]
        for number in range(1, max(used, default=0) + 1):
            items = sorted(tracks.get(f"{prefix}{number}", []), key=lambda i: i.record_start)
            out.append(Track(f"t{prefix}{number}", kind, number, None, None, None, None, items))
    timeline = Timeline(
        id="tl1",
        kind="timeline",
        name="Edit",
        media_clip_id=None,
        sequence_id="seq1",
        container_id="cont1",
        member=None,
        frame_rate=rate,
        resolution=(1920, 1080),
        start_seconds=None,
        duration_seconds=None,
        start_frame=start,
        end_frame=None,
        tracks=out,
    )
    return timeline, media


# -- edits on the primary track ----------------------------------------------------------------


def content(rows: list[Row]) -> list[Row]:
    """The V1 content clips (not the outro), in record order."""
    return sorted(
        (r for r in rows if r.track == "V1" and r.kind == "video" and r.role != "outro"),
        key=lambda r: r.record_in,
    )


def ripple(rows: list[Row], clips: list[Row]) -> list[Row]:
    """``clips`` (the new V1 content, in order) packed from frame 0, the outro after them, and
    every other row kept. V1 transitions are dropped (an edit script moves cuts under them)."""
    others = [
        r for r in rows if not (r.track == "V1" and (r.kind == "transition" or (r.kind == "video")))
    ]
    out: list[Row] = []
    cursor = 0
    for clip in clips:
        out.append(replace(clip, record_in=cursor, widen_in=0, widen_out=0))
        cursor += clip.duration
    for r in rows:
        if r.role == "outro":
            out.append(replace(r, record_in=cursor))
    return out + others


# -- a small synthetic .drt (the tests/drp builders) -------------------------------------------

RATE_25 = struct.pack("<d", 25.0) + bytes(8)
RES_1080 = struct.pack(">IIII", 0, 1920, 0, 1080)


def _rate_bytes(rate: float) -> bytes:
    return struct.pack("<d", rate) + bytes(8)


def _bytes_entry(key: str, value: bytes) -> tuple[str, int, bytes]:
    return (key, 0x0C, struct.pack(">I", len(value)) + value)


def _clip_blob(directory: str, filename: str) -> str:
    message = pb_len(1, directory.encode()) + pb_len(2, filename.encode()) + pb_len(5, b"prores")
    return envelope(2, 0x80, message).hex()


@dataclass
class DrtClip:
    """One V1 picture item of a synthetic .drt: a media file (``D:\\dir\\name``) and ranges."""

    media: str
    record_in: int
    duration: int
    src_in: int


def drt_members(
    clips: list[DrtClip],
    *,
    timelines: tuple[str, ...] = ("Edit",),
    rate: float = 25.0,
    media_rate: float | None = None,
    start_seconds: float = 3600.0,
    media_frames: int = 10_000,
) -> dict[str, str | bytes]:
    """The members of a .drt holding ``timelines`` (each with the same V1 items), built as the
    ``tests/drp`` synthetic archives are."""
    start = round(start_seconds * rate)
    files = sorted({c.media for c in clips})
    mrate = _rate_bytes(media_rate if media_rate is not None else rate)
    pool_clips = ""
    for n, path in enumerate(files, start=1):
        directory, name = path.rsplit("\\", 1)
        time_kd = kd(
            [("NumFrames", 0x02, struct.pack(">i", media_frames)), _bytes_entry("FrameRate", mrate)]
        ).hex()
        pool_clips += (
            f' <Element><Sm2MpVideoClip DbId="clip{n}"><Name>{name}</Name><MpFolder>bin1</MpFolder>'
            f'<Video><BtVideoInfo DbId="vi{n}"><Clip>{_clip_blob(directory, name)}</Clip>'
            f"<Time>{time_kd}</Time></BtVideoInfo></Video></Sm2MpVideoClip></Element>\n"
        )
    extents = struct.pack("<dd", start_seconds, 60.0).hex()
    tl_handles = ""
    members: dict[str, str | bytes] = {}
    for t, tl_name in enumerate(timelines, start=1):
        seq_fb = kd([("SeqRef", 0x0A, qstr(f"cont{t}"))]).hex()
        pool_clips += (
            f' <Element><Sm2MpTimelineClip DbId="tc{t}"><Name>{tl_name}</Name>'
            f'<MpFolder>bin1</MpFolder><TimelineSharedHandle><Sm2Timeline DbId="tl{t}">'
            f'<Name>{tl_name}</Name><Sequence><Sm2Sequence DbId="seq{t}">'
            f"<FieldsBlob>{seq_fb}</FieldsBlob><MediaExtents>{extents}</MediaExtents>"
            f"<FrameRate>{_rate_bytes(rate).hex()}</FrameRate>"
            f"<Resolution>{RES_1080.hex()}</Resolution></Sm2Sequence></Sequence></Sm2Timeline>"
            "</TimelineSharedHandle></Sm2MpTimelineClip></Element>\n"
        )
        tl_handles += f"<Element>tl{t}</Element>"
        items = ""
        for n, clip in enumerate(clips, start=1):
            ref = f"clip{files.index(clip.media) + 1}"
            name = clip.media.rsplit("\\", 1)[1]
            items += (
                f'  <Element><Sm2TiVideoClip DbId="t{t}i{n}"><Name>{name}</Name>'
                f"<Start>{start + clip.record_in}</Start><Duration>{clip.duration}</Duration>"
                f"<In>{clip.src_in}</In><MediaRef>{ref}</MediaRef></Sm2TiVideoClip></Element>\n"
            )
        members[f"SeqContainer/cont{t}.xml"] = xml(
            f'<Sm2SequenceContainer DbId="cont{t}">\n'
            f' <VideoTrackVec><Element><Sm2TiTrack DbId="v{t}"><Type>0</Type><SubType>0</SubType>'
            f"<Flags>0</Flags><Sequence>seq{t}</Sequence><Items>\n{items}"
            " </Items></Sm2TiTrack></Element></VideoTrackVec>\n"
            " <AudioTrackVec/>\n"
            "</Sm2SequenceContainer>\n"
        )
    members["project.xml"] = xml(
        '<SM_Project DbId="p1">\n <ProjectName>Synth</ProjectName>\n'
        f" <TimelineHandleVec>{tl_handles}</TimelineHandleVec>\n"
        " <CurrentTimelineIndex>0</CurrentTimelineIndex>\n</SM_Project>\n"
    )
    members["MediaPool/Master/MpFolder.xml"] = xml(
        f'<Sm2MpFolder DbId="bin1"><FieldsBlob/><Name>Master</Name><MediaVec>\n{pool_clips}'
        "</MediaVec></Sm2MpFolder>\n"
    )
    return members


def plan_for(clips: list[DrtClip], **overrides: Any) -> dict[str, Any]:
    """A 25 fps plan whose V1 segments are ``clips`` in order (sources ``s1``.. per file)."""
    files = sorted({c.media for c in clips})
    data: dict[str, Any] = {
        "version": 2,
        "project": "Synth",
        "timeline": "Edit",
        "kind": "longform",
        "fps": 25,
        "resolution": {"width": 1920, "height": 1080},
        "profile": "ccn",
        "sources": [
            {"id": f"s{n}", "media": path, "duration": 10_000, "start_timecode": 0}
            for n, path in enumerate(files, start=1)
        ],
        "segments": [
            {
                "source": f"s{files.index(c.media) + 1}",
                "src_in": c.src_in,
                "src_out": c.src_in + c.duration,
            }
            for c in clips
        ],
    }
    data.update(overrides)
    return data
