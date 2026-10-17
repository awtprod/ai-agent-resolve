"""Reader tests on synthetic archives built byte by byte, for cases the templates lack."""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

import pytest
from drp_helpers import ZSTD_REASON, envelope, kd, pb_len, pb_var, qstr, xml

from aar.drp import blobs
from aar.drp.describe import describe
from aar.drp.errors import ContainerError
from aar.drp.reader import read_drp

MakeZip = Callable[[str, dict[str, str | bytes]], Path]

RATE_25 = struct.pack("<d", 25.0) + bytes(8)
RES_1080 = struct.pack(">IIII", 0, 1920, 0, 1080)
EXTENTS = struct.pack("<dd", 3600.0, 10.0)
HALF = struct.pack("<d", 0.5).hex()


def bytes_entry(key: str, value: bytes) -> tuple[str, int, bytes]:
    return (key, 0x0C, struct.pack(">I", len(value)) + value)


def marker(frame: int, color: int, name: str, note: str = "") -> bytes:
    detail = (
        pb_var(1, color) + pb_len(3, note.encode()) + pb_len(3, b"1") + pb_len(3, name.encode())
    )
    inner = pb_len(1, detail)
    body = struct.pack(">II", 2, len(inner)) + inner
    return pb_len(1, pb_var(1, frame) + pb_len(2, body))


def markers_blob(*entries: bytes) -> str:
    blob_data = envelope(10001, 0x80, pb_len(2, b"".join(entries)))
    return kd([bytes_entry("BlobData", blob_data)]).hex()


def clip_blob(directory: str, filename: str) -> str:
    message = pb_len(1, directory.encode()) + pb_len(2, filename.encode()) + pb_len(5, b"prores")
    return envelope(2, 0x80, message).hex()


def members(
    *,
    seq_fields: bool = True,
    track_sequence: str = "seq1",
    frame_rate_hex: str | None = None,
    extra_item: str = "",
) -> dict[str, str | bytes]:
    time_kd = kd(
        [("NumFrames", 0x02, struct.pack(">i", 250)), bytes_entry("FrameRate", RATE_25)]
    ).hex()
    seq_fb = kd([("SeqRef", 0x0A, qstr("cont1"))]).hex() if seq_fields else ""
    rate = frame_rate_hex if frame_rate_hex is not None else RATE_25.hex()
    clip = clip_blob("D:\\Footage", "a.mov")
    project = xml(
        '<SM_Project DbId="p1">\n'
        " <ProjectName>Synth &amp; Co</ProjectName>\n"
        " <TimelineHandleVec><Element>tl1</Element></TimelineHandleVec>\n"
        " <CurrentTimelineIndex>0</CurrentTimelineIndex>\n"
        ' <LockableBlobMap><Sm2LockableBlobMap DbId="lm"><LocableBlobSet>\n'
        '  <Element><Sm2SequenceLockableBlob DbId="lb1">'
        f"<FieldsBlob>{markers_blob(marker(5, 32, 'Laugh', 'big'), marker(40, 8, 'End'))}"
        "</FieldsBlob><BlobOwner>seq1</BlobOwner></Sm2SequenceLockableBlob></Element>\n"
        '  <Element><Sm2TiItemLockableBlob DbId="lb2">'
        f"<FieldsBlob>{markers_blob(marker(3, 2, 'Clip mark'))}</FieldsBlob>"
        "<BlobOwner>i1</BlobOwner></Sm2TiItemLockableBlob></Element>\n"
        " </LocableBlobSet></Sm2LockableBlobMap></LockableBlobMap>\n"
        "</SM_Project>\n"
    )
    pool = xml(
        '<Sm2MpFolder DbId="bin1"><FieldsBlob/><Name>Master</Name><MediaVec>\n'
        ' <Element><Sm2MpVideoClip DbId="clip1"><Name>a.mov</Name><MpFolder>bin1</MpFolder>'
        '<Video><BtVideoInfo DbId="vi">'
        f"<Clip>{clip}</Clip><Time>{time_kd}</Time>"
        "</BtVideoInfo></Video></Sm2MpVideoClip></Element>\n"
        ' <Element><Sm2MpTimelineClip DbId="tc1"><Name>Edit</Name><MpFolder>bin1</MpFolder>'
        '<TimelineSharedHandle><Sm2Timeline DbId="tl1"><Name>Edit</Name><Sequence>'
        f'<Sm2Sequence DbId="seq1"><FieldsBlob>{seq_fb}</FieldsBlob>'
        f"<MediaExtents>{EXTENTS.hex()}</MediaExtents><FrameRate>{rate}</FrameRate>"
        f"<Resolution>{RES_1080.hex()}</Resolution></Sm2Sequence></Sequence></Sm2Timeline>"
        "</TimelineSharedHandle></Sm2MpTimelineClip></Element>\n"
        "</MediaVec></Sm2MpFolder>\n"
    )
    seq = xml(
        '<Sm2SequenceContainer DbId="cont1">\n'
        ' <VideoTrackVec><Element><Sm2TiTrack DbId="t1"><Type>0</Type><SubType>0</SubType>'
        f"<Flags>0</Flags><Sequence>{track_sequence}</Sequence><Items>\n"
        '  <Element><Sm2TiVideoClip DbId="i1"><Name>a.mov</Name><Start>90000</Start>'
        "<Duration>48</Duration><In>10</In><MediaRef>clip1</MediaRef>"
        "<MediaFilePath>E:\\old\\a.mov</MediaFilePath><UnknownThingBA>13370042</UnknownThingBA>"
        "</Sm2TiVideoClip></Element>\n"
        '  <Element><Sm2TiVideoClip DbId="i2"><Name>gone.mov</Name>'
        f"<Start>90048|{HALF}</Start><Duration>24</Duration><In/><MediaRef>missing</MediaRef>"
        "</Sm2TiVideoClip></Element>\n"
        f"{extra_item}"
        " </Items><UserDefinedName>Main</UserDefinedName></Sm2TiTrack></Element></VideoTrackVec>\n"
        " <AudioTrackVec/>\n"
        ' <SubtitleTrackVec><Element><Sm2TiTrack DbId="t2"><Type>2</Type><SubType>0</SubType>'
        f"<Flags>0</Flags><Sequence>{track_sequence}</Sequence><Items/></Sm2TiTrack></Element>"
        "</SubtitleTrackVec>\n"
        "</Sm2SequenceContainer>\n"
    )
    return {
        "project.xml": project,
        "Gallery.xml": xml('<Gallery::GyGallery DbId="g"/>'),
        "MediaPool/Master/MpFolder.xml": pool,
        "SeqContainer/cont1.xml": seq,
    }


def codes(drp) -> set[str]:  # type: ignore[no-untyped-def]
    return {i.code for i in drp.issues}


def test_synthetic_project_model(make_zip: MakeZip) -> None:
    drp = read_drp(make_zip("synth.drp", members()), check_blobs=True)
    assert drp.project is not None and drp.project.name == "Synth & Co"
    assert [b.path for b in drp.bins] == ["Master"]
    clip = drp.media_by_id()["clip1"]
    assert clip.path == "D:\\Footage\\a.mov" and clip.path_source == "media_pool_blob"
    assert clip.codec == "prores" and clip.frames == 250 and clip.frame_rate == 25.0
    assert clip.timeline_paths == ["E:\\old\\a.mov"]

    [tl] = drp.timelines
    assert (tl.id, tl.name, tl.frame_rate, tl.resolution) == ("tl1", "Edit", 25.0, (1920, 1080))
    assert (tl.start_frame, tl.end_frame) == (90000, 90250)
    assert [(t.label, t.name) for t in tl.tracks] == [("V1", "Main"), ("S1", None)]
    first, second = tl.tracks[0].items
    assert (first.record_start, first.record_end, first.source_in, first.source_out) == (
        90000,
        90048,
        10,
        58,
    )
    assert first.media_path == "D:\\Footage\\a.mov"
    assert second.media_clip_id is None and second.media_path is None
    assert second.record_start == 90048 and second.fractions == {"record_start": 0.5}
    assert second.source_in == 0

    assert [(m.frame, m.color, m.name, m.note) for m in tl.markers] == [
        (5, "Red", "Laugh", "big"),
        (40, "Green", "End", ""),
    ]
    assert all(m.frame_space == "timeline-relative" for m in tl.markers)
    assert [(m.frame, m.color, m.name) for m in first.markers] == [(3, "Blue", "Clip mark")]
    assert {"unresolved-media", "stale-media-file-path"} <= codes(drp)
    assert drp.blob_stats.failures == {}
    assert drp.blob_stats.undecoded == {"UnknownThingBA": 1}


def test_describe_lists_items_markers_and_problems(make_zip: MakeZip) -> None:
    text = describe(read_drp(make_zip("synth.drp", members())))
    assert 'Timeline "Edit": 25 fps, 1920x1080, start 01:00:00:00 (frame 90000)' in text
    assert '01:00:00:00-01:00:01:23 video "a.mov" src 00:00:00:10-00:00:02:08' in text
    assert "UNRESOLVED media missing" in text
    assert "frame 5 (01:00:00:05) Red 'Laugh' big" in text
    assert "1 timeline item(s) reference media missing from this file" in text
    assert 'project "Synth & Co"' in text


def test_join_falls_back_to_track_sequence_reference(make_zip: MakeZip) -> None:
    drp = read_drp(make_zip("synth.drp", members(seq_fields=False)))
    assert [(t.name, t.container_id) for t in drp.timelines] == [("Edit", "cont1")]
    assert "orphan-timeline" not in codes(drp)


def test_container_without_media_pool_sequence_is_kept_as_orphan(make_zip: MakeZip) -> None:
    drp = read_drp(make_zip("synth.drp", members(seq_fields=False, track_sequence="nope")))
    assert "orphan-timeline" in codes(drp)
    orphan = next(t for t in drp.timelines if t.container_id == "cont1")
    assert orphan.name is None and len(orphan.tracks) == 2
    edit = next(t for t in drp.timelines if t.name == "Edit")
    assert edit.tracks == [] and "missing-container" in codes(drp)


def test_compressed_clip_blob_without_zstd_falls_back_to_timeline_path(
    make_zip: MakeZip, monkeypatch: pytest.MonkeyPatch
) -> None:
    zstandard = pytest.importorskip("zstandard", reason=ZSTD_REASON)
    message = pb_len(1, b"D:\\Footage") + pb_len(2, b"a.mov") + pb_len(3, b"padding " * 20)
    compressed = envelope(2, 0x81, zstandard.ZstdCompressor().compress(message)).hex()
    m = members()
    m["MediaPool/Master/MpFolder.xml"] = str(m["MediaPool/Master/MpFolder.xml"]).replace(
        clip_blob("D:\\Footage", "a.mov"), compressed
    )
    assert compressed in str(m["MediaPool/Master/MpFolder.xml"])
    with_lib = read_drp(make_zip("with.drp", m)).media_by_id()["clip1"]
    assert (with_lib.path, with_lib.path_source) == ("D:\\Footage\\a.mov", "media_pool_blob")

    monkeypatch.setattr(blobs, "_zstd_backend", lambda: (None, None))
    drp = read_drp(make_zip("without.drp", m))
    clip = drp.media_by_id()["clip1"]
    assert (clip.path, clip.path_source) == ("E:\\old\\a.mov", "timeline_item")
    assert "zstd-unavailable" in codes(drp)


def test_truncated_hex_is_reported_and_reading_continues(make_zip: MakeZip) -> None:
    drp = read_drp(make_zip("synth.drp", members(frame_rate_hex="0000000000003")))
    [tl] = drp.timelines
    assert tl.frame_rate is None and tl.start_frame is None
    bad = [i for i in drp.issues if i.code == "bad-hex"]
    assert len(bad) == 1 and "odd hex length" in bad[0].message
    assert bad[0].where is not None and bad[0].where.member == "MediaPool/Master/MpFolder.xml"
    assert len(tl.tracks[0].items) == 2


def test_malformed_xml_and_bad_numbers_do_not_raise(make_zip: MakeZip) -> None:
    item = (
        '  <Element><Sm2TiGenerator DbId="g1"><Name>Bad</Name><Start>soon</Start>'
        "<Duration>5</Duration><Stray></Element>\n"
    )
    drp = read_drp(make_zip("synth.drp", members(extra_item=item)))
    items = drp.timelines[0].tracks[0].items
    assert [i.kind for i in items] == ["video", "video", "generator"]
    assert items[2].record_start is None and items[2].duration == 5
    assert {"bad-frame-count", "scan-unclosed-element"} <= codes(drp)


def test_missing_project_xml_is_an_error_issue(make_zip: MakeZip) -> None:
    m = members()
    del m["project.xml"]
    drp = read_drp(make_zip("synth.drp", m))
    assert drp.project is None
    assert any(i.code == "missing-member" and i.severity == "error" for i in drp.issues)
    assert len(drp.media) == 2 and drp.timelines[0].name == "Edit"


def test_kind_is_sniffed_without_a_known_suffix(make_zip: MakeZip) -> None:
    m = members()
    del m["Gallery.xml"]
    m["MediaManagement.dat"] = b"\x00\x01"
    drp = read_drp(make_zip("export.bin", m))
    assert (drp.kind, drp.kind_source) == ("drt", "sniffed")
    assert "MediaManagement.dat" in drp.raw_members


def test_suffix_and_contents_disagree(make_zip: MakeZip) -> None:
    drp = read_drp(make_zip("named-wrong.drt", members()))
    assert drp.kind == "drt" and drp.sniffed_kind == "drp"
    assert any("contents look like drp" in i.message for i in drp.issues)


def test_member_size_limit(make_zip: MakeZip) -> None:
    drp = read_drp(make_zip("synth.drp", members()), max_member_bytes=200)
    assert "unreadable-member" in codes(drp)
    assert drp.timelines == [] and drp.media == []


def test_non_zip_bin_export_is_listed_best_effort(tmp_path: Path) -> None:
    path = tmp_path / "bin.drb"
    path.write_bytes(b"SQLite format 3\x00" + bytes(100))
    drp = read_drp(path)
    assert (drp.kind, drp.is_zip, drp.signature) == ("drb", False, "sqlite")
    assert drp.members == [] and "not-zip" in codes(drp)
    assert "not a zip archive" in describe(drp)


def test_zip_bin_export_parses_media_pool_folders(make_zip: MakeZip) -> None:
    pool = members()["MediaPool/Master/MpFolder.xml"]
    drp = read_drp(make_zip("bin.drb", {"MediaPool/Master/MpFolder.xml": pool}))
    assert drp.kind == "drb" and drp.sniffed_kind == "drb"
    assert "drb-best-effort" in codes(drp)
    assert drp.media_by_id()["clip1"].path == "D:\\Footage\\a.mov"


def test_missing_file_raises_container_error(tmp_path: Path) -> None:
    with pytest.raises(ContainerError, match="cannot read"):
        read_drp(tmp_path / "nope.drp")
