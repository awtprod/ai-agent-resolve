"""Probes from the S2 review: edits that an earlier version reported as identical.

Each test makes one edit that the diff must notice, and pins exactly what it reports. The
``M..`` labels name the review's mutation that each probe kills.
"""

from __future__ import annotations

import re
import shutil
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from diff_testing import (
    CLIP,
    H264,
    MARKER_BLOB_ID,
    POOL,
    SEQ_MEMBER,
    SEQUENCE,
    V1,
    V1_ITEM,
    assert_mirrors,
    diff,
    edited,
    marker_file,
    only_change,
)
from drp_edit import (
    add_carrier,
    after,
    carrier,
    element_span,
    flip_hex_byte,
    kd_bytes,
    kd_string,
    leaf_text,
    marker_entry,
    markers_fields_blob,
    once,
    rewrite_clip_blobs,
    set_fields,
)
from drp_helpers import envelope, kd, needs_zstandard, pb_len, pb_var

from aar.drp.reader import read_drp

FIXTURES = H264.parent
R19 = FIXTURES / "media-clip-r19.drp"
R19_POOL = "MediaPool/Master/MpFolder.xml"
R19_ARCHIVE = "MediaPool/Master/000_Archive/MpFolder.xml"
R19_CLIP = "98e443dd-f712-4ef0-83cc-3b18cfe5f957"
R19_MASTER_ID = "653a17d4-c214-4e7c-8854-8ada21b71530"
R19_ARCHIVE_ID = "7fb9ee45-9553-40cf-8f54-41d4a5132216"
TRACK_V1 = b'<Sm2TiTrack DbId="c24d0fcd-746d-4416-9516-07ad8dfd2243">'
RATE = b"<FrameRate>00000000000038400000000000000000</FrameRate>"
NEW_ITEM = "11111111-2222-3333-4444-555555555555"


# --------------------------------------------------------------------------- mutation probes


def test_m01_first_unchecked_frame_rate_byte(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {POOL: once(RATE, RATE.replace(b"3840000000", b"3840010000"))})
    assert read_drp(b).timelines[0].frame_rate == 24.0
    c = only_change(H264, b)
    assert (c.kind, c.byte_count) == ("opaque-changed", 1)
    assert c.field_name is not None and c.field_name.endswith("Sm2Sequence/FrameRate")
    assert "from byte 8" in c.detail


def test_m04_nested_dbid_attribute_value(tmp_path: Path) -> None:
    old = b'<BtVideoInfo DbId="3f5420f0-f4e4-4aa1-83a0-61b726cbf29d">'
    b = edited(tmp_path, "b.drp", {POOL: once(old, old.replace(b"cbf29d", b"cbf29e"))})
    c = only_change(H264, b)
    assert (c.kind, c.object_id) == ("opaque-changed", CLIP)
    assert c.field_name == "Video/BtVideoInfo" and "attribute(s) DbId changed" in c.detail


def test_m08_new_marker_blob_with_an_undecodable_list(tmp_path: Path) -> None:
    garbage = envelope(10001, 0x80, pb_var(3, 7))  # a protobuf that is not a marker list
    blob = kd([kd_bytes("BlobData", garbage)]).hex()
    b = edited(
        tmp_path, "b.drp", {"project.xml": add_carrier(carrier(MARKER_BLOB_ID, SEQUENCE, blob))}
    )
    assert read_drp(b).timelines[0].markers == []
    d = assert_mirrors(H264, b)
    assert [(c.section, c.kind, c.object_kind) for c in d.changes] == [
        ("opaque", "opaque-added", "markers")
    ]


def test_m11_text_between_child_elements(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: after(V1, b"<PrettyType/>", b"<PrettyType/>stray")})
    c = only_change(H264, b)
    assert (c.kind, c.object_id, c.byte_count) == ("opaque-changed", V1_ITEM, 5)
    assert "text outside child elements" in c.detail


def test_m12_m29_element_wrapper_of_a_unit(tmp_path: Path) -> None:
    wrapper = f'<Element>\n      <Sm2TiVideoClip DbId="{V1_ITEM}"'.encode()
    b = edited(
        tmp_path, "b.drp", {SEQ_MEMBER: once(wrapper, wrapper.replace(b">\n", b">junk\n", 1))}
    )
    c = only_change(H264, b)
    assert (c.kind, c.object_kind) == ("opaque-changed", "track")
    assert "text inside the <Element> wrapper" in c.detail and c.byte_count == 4
    b2 = edited(
        tmp_path,
        "b2.drp",
        {SEQ_MEMBER: once(wrapper, wrapper.replace(b"<Element>", b'<Element x="1">'))},
    )
    c2 = only_change(H264, b2)
    assert c2.kind == "opaque-changed" and "attribute(s) x changed" in c2.detail


def test_m13_xml_prolog(tmp_path: Path) -> None:
    b = edited(
        tmp_path, "b.drp", {SEQ_MEMBER: once(b'<?xml version="1.0"', b'<?xml version="1.1"')}
    )
    c = only_change(H264, b)
    assert (c.kind, c.field_name) == ("opaque-changed", SEQ_MEMBER) and "prolog" in c.detail


def test_m14_content_after_the_root(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: lambda d: d + b"<!-- appended -->\n"})
    c = only_change(H264, b)
    assert (c.kind, c.field_name) == ("opaque-changed", SEQ_MEMBER)
    assert "after the root element" in c.detail


def test_m19_marker_duration_spelling(tmp_path: Path) -> None:
    a = marker_file(tmp_path, "a.drp", marker_entry(48, 32, "Laugh"))
    b = marker_file(tmp_path, "b.drp", marker_entry(48, 32, "Laugh", duration=b"01"))
    assert read_drp(b).timelines[0].markers[0].duration == 1  # same decoded value
    c = only_change(a, b)
    assert (c.kind, c.field_name, c.object_kind) == ("opaque-changed", "FieldsBlob", "markers")


def test_m23_undecoded_entry_of_a_keyed_blob(tmp_path: Path) -> None:
    # The last entry of the video Time dict is DbType, which the reader does not use.
    b = edited(tmp_path, "b.drp", {POOL: flip_hex_byte(b"<BtVideoInfo DbId", b"Time", 266)})
    clip_a = read_drp(H264).media_by_id()[CLIP]
    clip_b = read_drp(b).media_by_id()[CLIP]
    assert (clip_a.frames, clip_a.frame_rate) == (clip_b.frames, clip_b.frame_rate)
    c = only_change(H264, b)
    assert (c.kind, c.field_name, c.byte_count) == ("opaque-changed", "Video/BtVideoInfo/Time", 1)


def settings_file(tmp_path: Path, name: str, other: str, hint: bytes) -> Path:
    setup = envelope(1, 0x80, pb_len(1, b"keep") + pb_len(69, hint))
    blob = kd([kd_string("Other", other), kd_bytes("SetupBA", setup)]).hex().encode()

    def run(data: bytes) -> bytes:
        start, end = leaf_text(data, b"<SM_Config", b"FieldsBlob")
        return data[:start] + blob + data[end:]

    return edited(tmp_path, name, {"project.xml": run})


def test_m24_settings_blob_entries_besides_setup(tmp_path: Path) -> None:
    a = settings_file(tmp_path, "a.drp", "x", b"24")
    assert read_drp(a).project.frame_rate_hint == "24"  # type: ignore[union-attr]
    c = only_change(a, settings_file(tmp_path, "b.drp", "y", b"24"))
    assert c.kind == "opaque-changed" and c.field_name == "CommonConfig/SM_Config/FieldsBlob"
    c2 = only_change(a, settings_file(tmp_path, "b2.drp", "x", b"25"))
    assert (c2.section, c2.field_name, c2.old, c2.new) == ("header", "frame_rate_hint", "24", "25")


def test_m26_every_removed_marker_is_reported(tmp_path: Path) -> None:
    a = marker_file(tmp_path, "a.drp", marker_entry(48, 32, "One"), marker_entry(96, 8, "Two"))
    b = marker_file(tmp_path, "b.drp")
    d = assert_mirrors(a, b)
    assert sorted((c.kind, c.old["name"]) for c in d.changes) == [
        ("removed", "One"),
        ("removed", "Two"),
    ]


def test_m28_order_of_child_elements(tmp_path: Path) -> None:
    def swap(data: bytes) -> bytes:
        at = data.index(TRACK_V1)
        m = re.compile(rb"(<FieldsBlob>[0-9a-f]*</FieldsBlob>)(\s*)(<Type>0</Type>)").search(
            data, at
        )
        assert m is not None
        return data[: m.start()] + m.group(3) + m.group(2) + m.group(1) + data[m.end() :]

    c = only_change(H264, edited(tmp_path, "b.drp", {SEQ_MEMBER: swap}))
    assert (c.kind, c.object_kind) == ("opaque-changed", "track")
    assert "order of child elements changed" in c.detail


# --------------------------------------------------------------------------- containment, order


def clip_to_archive(tmp_path: Path, name: str, *, follow: bool) -> Path:
    with zipfile.ZipFile(R19) as z:
        master, archive = z.read(R19_POOL), z.read(R19_ARCHIVE)
    start, end = element_span(master, R19_CLIP)
    piece = master[start:end]
    ref = f"<MpFolder>{R19_MASTER_ID}</MpFolder>".encode()
    assert piece.count(ref) == 1
    if follow:
        piece = piece.replace(ref, f"<MpFolder>{R19_ARCHIVE_ID}</MpFolder>".encode())
    moved_archive = archive.replace(b"<MediaVec/>", b"<MediaVec>\n  " + piece + b"\n </MediaVec>")
    return edited(
        tmp_path,
        name,
        {R19_POOL: lambda d: master[:start] + master[end:], R19_ARCHIVE: lambda d: moved_archive},
        src=R19,
    )


def test_clip_moved_between_bins_with_its_back_reference_unchanged(tmp_path: Path) -> None:
    b = clip_to_archive(tmp_path, "b.drp", follow=False)
    clip = read_drp(b).media_by_id()[R19_CLIP]
    assert clip.bin_id == R19_MASTER_ID  # the <MpFolder> back-reference still says Master
    c = only_change(R19, b)
    assert (c.section, c.kind, c.object_id, c.old, c.new) == (
        "media",
        "moved",
        R19_CLIP,
        "Master",
        "Master/Archive",
    )
    assert "MpFolder reference names" in c.detail


def test_clip_moved_between_bins_with_its_back_reference(tmp_path: Path) -> None:
    c = only_change(R19, clip_to_archive(tmp_path, "b.drp", follow=True))
    assert (c.kind, c.old, c.new) == ("moved", "Master", "Master/Archive")
    assert "MpFolder" not in c.detail


def two_items(tmp_path: Path, name: str, *, swapped: bool) -> Path:
    def run(data: bytes) -> bytes:
        start, end = element_span(data, V1_ITEM)
        piece = data[start:end]
        copy = piece.replace(V1_ITEM.encode(), NEW_ITEM.encode()).replace(
            b"<Start>86400<", b"<Start>90064<"
        )
        first, second = (copy, piece) if swapped else (piece, copy)
        return data[:start] + first + b"\n     " + second + data[end:]

    return edited(tmp_path, name, {SEQ_MEMBER: run})


def test_order_of_items_in_a_track_is_reported(tmp_path: Path) -> None:
    a = two_items(tmp_path, "a.drp", swapped=False)
    b = two_items(tmp_path, "b.drp", swapped=True)
    c = only_change(a, b)
    assert (c.section, c.kind, c.object_kind, c.field_name) == (
        "timelines",
        "reordered",
        "track",
        "item_order",
    )
    assert (c.old, c.new) == ([V1_ITEM, NEW_ITEM], [NEW_ITEM, V1_ITEM])


MEDIA_BLOB = "fcf30d1a-d0f2-4670-856a-da986df1912a"
ITEM_BLOB = "22222222-3333-4444-5555-666666666666"


def item_blob(media: bytes) -> bytes:
    return media.replace(MEDIA_BLOB.encode(), ITEM_BLOB.encode()).replace(
        b"Sm2MediaPoolLockableBlob", b"Sm2MpItemLockableBlob"
    )


def lockable_set(order: str) -> Callable[[bytes], bytes]:
    """Rebuild the project's LocableBlobSet from parts in ``order``.

    M = the template's media-pool blob, I = an item lockable blob, C = a marker blob for the
    timeline (a unit), N = <Note>, O = <Other> (the last two without a DbId); lowercase m, i
    are M, I without their <Element> wrapper.
    """

    def run(data: bytes) -> bytes:
        start, end = element_span(data, MEDIA_BLOB)
        media = data[start:end]
        markers = markers_fields_blob(marker_entry(48, 32, "Laugh"))
        parts = {
            "M": media,
            "I": item_blob(media),
            "C": carrier(MARKER_BLOB_ID, SEQUENCE, markers).strip(),
            "N": b"<Note>x</Note>",
            "O": b"<Other>y</Other>",
        }
        # m, i: the same two blobs without their <Element> wrapper (DbId on the child itself)
        parts["m"] = parts["M"].split(b"<Element>", 1)[1].rsplit(b"</Element>", 1)[0].strip()
        parts["i"] = parts["I"].split(b"<Element>", 1)[1].rsplit(b"</Element>", 1)[0].strip()
        return data[:start] + b"\n    ".join(parts[k] for k in order) + data[end:]

    return run


@needs_zstandard
def test_a_set_with_its_dbid_children_reordered_is_identical(tmp_path: Path) -> None:
    # What Resolve does on its own round trip: a mixed-tag set rewritten in hash order.
    a = edited(tmp_path, "a.drp", {"project.xml": lockable_set("MICN")})
    b = edited(tmp_path, "b.drp", {"project.xml": lockable_set("CNIM")})
    assert [m.name for m in read_drp(b).timelines[0].markers] == ["Laugh"]
    d = diff(a, b, gate=True)
    assert d.changes == [] and d.identical and d.gate.passed, d.to_text()
    # The same for DbId children written straight into the set, without wrappers.
    x = edited(tmp_path, "x.drp", {"project.xml": lockable_set("miN")})
    y = edited(tmp_path, "y.drp", {"project.xml": lockable_set("Nim")})
    assert diff(x, y, gate=True).changes == []


def test_entries_without_dbid_keep_their_order_inside_a_set(tmp_path: Path) -> None:
    a = edited(tmp_path, "a.drp", {"project.xml": lockable_set("NMIO")})
    b = edited(tmp_path, "b.drp", {"project.xml": lockable_set("OIMN")})
    c = only_change(a, b)
    assert c.kind == "opaque-changed" and "order of child elements changed" in c.detail
    assert c.field_name is not None and c.field_name.endswith("LocableBlobSet")


def test_a_dbid_child_leaving_the_set_is_a_move(tmp_path: Path) -> None:
    a = edited(tmp_path, "a.drp", {"project.xml": lockable_set("MI")})

    def out_of_set(data: bytes) -> bytes:
        data = lockable_set("M")(data)
        start, end = element_span(data, MEDIA_BLOB)
        moved = b"</LocableBlobSet>\n   " + item_blob(data[start:end])
        return data.replace(b"</LocableBlobSet>", moved, 1)

    b = edited(tmp_path, "b.drp", {"project.xml": out_of_set})
    d = assert_mirrors(a, b)
    assert sorted(c.kind for c in d.changes) == ["opaque-added", "opaque-removed"]
    tail = "under another parent element on the other side"
    assert all(c.detail.endswith(tail) for c in d.changes)
    assert not diff(a, b, gate=True).gate.passed


def test_a_unit_under_another_parent_without_a_reported_move(tmp_path: Path) -> None:
    # The reader finds marker blobs anywhere in project.xml, so moving one out of its set
    # changes no marker; only the byte-level parent check can see it.
    element = carrier(MARKER_BLOB_ID, SEQUENCE, markers_fields_blob(marker_entry(48, 32, "Laugh")))
    a = edited(tmp_path, "a.drp", {"project.xml": add_carrier(element)})
    b = edited(
        tmp_path,
        "b.drp",
        {"project.xml": once(b"</LocableBlobSet>", b"</LocableBlobSet>\n   " + element.strip())},
    )
    assert [m.name for m in read_drp(b).timelines[0].markers] == ["Laugh"]
    d = assert_mirrors(a, b)
    assert sorted(c.kind for c in d.changes) == ["opaque-added", "opaque-removed"]
    assert all("under another parent element" in c.detail for c in d.changes)


def test_duplicate_zip_member_name(tmp_path: Path) -> None:
    b = tmp_path / "dup.drp"
    shutil.copy(H264, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about the duplicate name
        with zipfile.ZipFile(b, "a") as z:
            z.writestr("Gallery.xml", z.read("Gallery.xml"))
    d = assert_mirrors(H264, b)
    assert [(c.kind, c.field_name, c.old, c.new) for c in d.changes] == [
        ("opaque-changed", "Gallery.xml", 1, 2)
    ]
    assert not d.gate.passed
    assert "B: duplicate zip member name(s): Gallery.xml" in d.gate.reasons


# --------------------------------------------------------------------------- minor findings


def extents_file(tmp_path: Path, name: str, start_hex: bytes) -> Path:
    old = b"<MediaExtents>000000000020ac405055555555156340</MediaExtents>"
    new = b"<MediaExtents>" + start_hex + b"5055555555156340</MediaExtents>"
    return edited(tmp_path, name, {POOL: once(old, new)})


def test_negative_zero_is_not_zero(tmp_path: Path) -> None:
    a = extents_file(tmp_path, "a.drp", b"0000000000000000")
    b = extents_file(tmp_path, "b.drp", b"0000000000000080")
    c = only_change(a, b)
    assert (c.section, c.field_name) == ("timelines", "start")
    assert "seconds 0.0 → -0.0" in c.detail


def test_a_list_leaf_is_explained_only_by_its_own_value(tmp_path: Path) -> None:
    handle = b"<Element>f75533e4-a99c-4cd9-8280-1f756f0d3bef</Element>"
    spaced = b"<Element>f75533e4-a99c-4cd9-8280-1f756f0d3bef </Element>"
    extra = b"\n  <Element>99999999-0000-0000-0000-000000000000</Element>"
    b = edited(tmp_path, "b.drp", {"project.xml": once(handle, spaced + extra)})
    d = assert_mirrors(H264, b)
    found = sorted((c.section, c.kind, c.field_name) for c in d.changes)
    assert found == [
        ("header", "changed", "timeline_ids"),
        ("opaque", "opaque-changed", "TimelineHandleVec/Element"),
    ]


def test_same_path_split_differently_is_opaque(tmp_path: Path) -> None:
    split_a = set_fields({1: b"/Volumes/Media", 2: b"sub/a.mp4"})
    split_b = set_fields({1: b"/Volumes/Media/sub", 2: b"a.mp4"})
    a = edited(tmp_path, "a.drp", {POOL: rewrite_clip_blobs(split_a)})
    b = edited(tmp_path, "b.drp", {POOL: rewrite_clip_blobs(split_b)})
    path = "/Volumes/Media/sub/a.mp4"
    assert read_drp(a).media_by_id()[CLIP].path == read_drp(b).media_by_id()[CLIP].path == path
    d = assert_mirrors(a, b)
    assert [c.kind for c in d.changes] == ["opaque-changed", "opaque-changed"]
    assert all(c.field_name.endswith("/Clip") for c in d.changes)


def test_marker_order_and_padded_varints(tmp_path: Path) -> None:
    one, two = marker_entry(48, 32, "One"), marker_entry(96, 8, "Two")
    c = only_change(
        marker_file(tmp_path, "a.drp", one, two), marker_file(tmp_path, "b.drp", two, one)
    )
    assert (c.section, c.kind, c.field_name, c.old, c.new) == (
        "markers",
        "reordered",
        "marker_order",
        [48, 96],
        [96, 48],
    )
    padded = marker_file(tmp_path, "p.drp", marker_entry(48, 32, "One", frame_raw=b"\xb0\x00"))
    assert read_drp(padded).timelines[0].markers[0].frame == 48
    c = only_change(marker_file(tmp_path, "c.drp", one), padded)
    assert (c.kind, c.field_name) == ("opaque-changed", "FieldsBlob")


def test_relinking_only_the_second_stream(tmp_path: Path) -> None:
    audio = set_fields({1: b"/Volumes/Audio", 2: b"ext.wav"})
    b = edited(tmp_path, "b.drp", {POOL: rewrite_clip_blobs(audio, which={1})})
    c = only_change(H264, b)
    assert (c.section, c.kind, c.field_name) == ("media", "changed", "linked_paths")
    assert (c.old, c.new) == ([], ["/Volumes/Audio/ext.wav"])


def test_hex_letter_case_is_reported_with_a_byte_count(tmp_path: Path) -> None:
    def upper(data: bytes) -> bytes:
        start, end = leaf_text(data, V1, b"FieldsBlob")
        return data[:start] + data[start:end].upper() + data[end:]

    c = only_change(H264, edited(tmp_path, "b.drp", {SEQ_MEMBER: upper}))
    assert c.kind == "opaque-changed" and c.byte_count and c.byte_count > 0
    assert "hex text differs" in c.detail


def test_added_marker_with_undecoded_fields(tmp_path: Path) -> None:
    odd = marker_entry(48, 32, "Laugh", extra=pb_var(9, 1))
    d = assert_mirrors(H264, marker_file(tmp_path, "b.drp", odd))
    assert sorted((c.section, c.kind) for c in d.changes) == [
        ("markers", "added"),
        ("opaque", "opaque-added"),
    ]
    plain = marker_file(tmp_path, "a.drp", marker_entry(10, 32, "First"))
    both = marker_file(tmp_path, "c.drp", marker_entry(10, 32, "First"), odd)
    d = assert_mirrors(plain, both)
    assert sorted((c.section, c.kind) for c in d.changes) == [
        ("markers", "added"),
        ("opaque", "opaque-changed"),
    ]


@needs_zstandard
def test_json_has_a_schema_version() -> None:
    data = diff(H264, H264).to_dict()
    assert data["schema_version"] == 2
    assert data["gate"] == {"passed": True, "reasons": [], "requested": False}


# --------------------------------------------------------------------------- round 3


R19_SEQ = "SeqContainer/277c0f51-503d-4bf4-9ad7-2e96c74447c6.xml"
UNIT_ROOTS = [
    (
        "h264-container",
        H264,
        SEQ_MEMBER,
        "Sm2SequenceContainer",
        "106d9f2b-2951-4ab9-b961-0aa934df20e9",
    ),
    ("h264-timeline-clip", H264, POOL, "Sm2MpTimelineClip", "6d692bf3-f02a-4943-ac2f-74d3f0b9fe1f"),
    ("r19-container", R19, R19_SEQ, "Sm2SequenceContainer", "277c0f51-503d-4bf4-9ad7-2e96c74447c6"),
    (
        "r19-timeline-clip",
        R19,
        R19_POOL,
        "Sm2MpTimelineClip",
        "7d029318-2b54-4cac-bc7d-2b439dd74a99",
    ),
]


@pytest.mark.parametrize(
    ("src", "member", "tag", "dbid"), [r[1:] for r in UNIT_ROOTS], ids=[r[0] for r in UNIT_ROOTS]
)
def test_h3_a_unit_root_dbid_flip_is_never_identical(
    src: Path, member: str, tag: str, dbid: str, tmp_path: Path
) -> None:
    start_tag = f'<{tag} DbId="{dbid}">'.encode()
    for i, ch in enumerate(dbid):
        if ch == "-":
            continue
        flipped = dbid[:i] + ("1" if ch == "0" else "0") + dbid[i + 1 :]
        b = edited(
            tmp_path,
            f"b{i}.drp",
            {member: once(start_tag, start_tag.replace(dbid.encode(), flipped.encode()))},
            src=src,
        )
        d = diff(src, b, gate=True)
        assert not d.identical and not d.gate.passed, (i, d.to_text())
        assert any(
            c.kind == "opaque-changed" and "attribute(s) DbId changed" in c.detail
            for c in d.changes
        ), i


def test_h1_prolog_bytes_and_header_comment_attributes(tmp_path: Path) -> None:
    probes = {
        "encoding": once(b'encoding="UTF-8"', b'encoding="UTF-9"'),
        "whitespace": once(b"?>\n<!--", b"?>\r\n<!--"),
        "comment attribute": once(b'DbPrjVer="17"-->', b'DbPrjVer="17" Extra="1"-->'),
        "after the root": lambda d: d + b"\n",
    }
    for name, edit in probes.items():
        b = edited(tmp_path, f"{name}.drp", {SEQ_MEMBER: edit})
        d = assert_mirrors(H264, b)
        assert [(c.kind, c.field_name) for c in d.changes] == [("opaque-changed", SEQ_MEMBER)], name
        assert not diff(H264, b, gate=True).gate.passed, name


def test_h1_a_version_stamp_change_stays_a_header_change(tmp_path: Path) -> None:
    stamp = (b'DbAppVer="21.0.0.0048"', b'DbAppVer="21.0.0.0049"')
    with zipfile.ZipFile(H264) as z:
        xml = [n for n in z.namelist() if n.endswith(".xml")]
    b = edited(tmp_path, "b.drp", {n: once(*stamp) for n in xml})
    d = assert_mirrors(H264, b)
    # Gallery.xml is not interpreted, so its bytes are compared whole.
    assert sorted((c.section, c.field_name) for c in d.changes) == [
        ("header", "DbAppVer"),
        ("opaque", "Gallery.xml"),
    ]


def test_tag_or_wrapper_change_under_the_same_parent_is_not_a_move(tmp_path: Path) -> None:
    a = edited(tmp_path, "a.drp", {"project.xml": lockable_set("MI")})

    def retag(data: bytes) -> bytes:
        data = lockable_set("MI")(data)
        start, end = element_span(data, ITEM_BLOB)
        piece = data[start:end].replace(b"Sm2MpItemLockableBlob", b"Sm2MpOtherLockableBlob")
        return data[:start] + piece + data[end:]

    for name, edit in (("retag", retag), ("unwrap", lockable_set("Mi"))):
        b = edited(tmp_path, f"{name}.drp", {"project.xml": edit})
        d = assert_mirrors(a, b)
        assert sorted(c.kind for c in d.changes) == ["opaque-added", "opaque-removed"], name
        assert not any("; moved:" in c.detail for c in d.changes), name


# Pins for the review's surviving mutants.


def test_m29_an_added_unit_whose_wrapper_holds_more(tmp_path: Path) -> None:
    def run(data: bytes) -> bytes:
        start, end = element_span(data, V1_ITEM)
        copy = (
            data[start:end]
            .replace(V1_ITEM.encode(), NEW_ITEM.encode())
            .replace(b"<Start>86400<", b"<Start>90064<")
            .replace(b"<Element>", b'<Element x="1">', 1)
        )
        return data[:end] + b"\n     " + copy + data[end:]

    d = assert_mirrors(H264, edited(tmp_path, "b.drp", {SEQ_MEMBER: run}))
    found = sorted((c.section, c.kind, c.object_id) for c in d.changes)
    assert found[0][:2] == ("opaque", "opaque-added") and found[1] == (
        "timelines",
        "added",
        NEW_ITEM,
    )
    assert len(found) == 2
    assert "the <Element> wrapper holds more than the object" in next(
        c.detail for c in d.changes if c.section == "opaque"
    )


def test_n20_unit_order_counts_outside_the_semantic_vectors(tmp_path: Path) -> None:
    # (a) next to a non-unit sibling inside Items, a unit keeps its place
    extra = b"<Extra/>\n     "
    v1_wrapper = f'<Element>\n      <Sm2TiVideoClip DbId="{V1_ITEM}"'.encode()

    def extra_last(data: bytes) -> bytes:
        end = element_span(data, V1_ITEM)[1]
        return data[:end] + b"\n     <Extra/>" + data[end:]

    a = edited(tmp_path, "a.drp", {SEQ_MEMBER: extra_last})
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: once(v1_wrapper, extra + v1_wrapper)})
    c = only_change(a, b)
    assert "order of child elements changed" in c.detail and c.field_name.endswith("Items")
    # (b) two units (marker blobs) under a parent that is neither a set nor a semantic vector
    timeline_blob = carrier(
        MARKER_BLOB_ID, SEQUENCE, markers_fields_blob(marker_entry(48, 32, "T"))
    )
    item_blob_ = carrier(
        "0c0c0c0c-1111-2222-3333-444444444444",
        V1_ITEM,
        markers_fields_blob(marker_entry(3, 2, "I")),
        tag="Sm2TiItemLockableBlob",
    )

    def two_units(first: bytes, second: bytes) -> Callable[[bytes], bytes]:
        moved = b"</LocableBlobSet>\n   " + first.strip() + b"\n   " + second.strip()
        return once(b"</LocableBlobSet>", moved)

    x = edited(tmp_path, "x.drp", {"project.xml": two_units(timeline_blob, item_blob_)})
    y = edited(tmp_path, "y.drp", {"project.xml": two_units(item_blob_, timeline_blob)})
    c = only_change(x, y)
    assert "order of child elements changed" in c.detail
    assert c.field_name.endswith("Sm2LockableBlobMap")


def test_n21_dbid_element_order_counts_outside_a_set(tmp_path: Path) -> None:
    def outside(order: str) -> Callable[[bytes], bytes]:
        def run(data: bytes) -> bytes:
            data = lockable_set("")(data)
            with zipfile.ZipFile(H264) as z:
                project = z.read("project.xml")
            start, end = element_span(project, MEDIA_BLOB)
            parts = {"M": project[start:end], "I": item_blob(project[start:end])}
            body = b"\n   ".join(parts[k] for k in order)
            return data.replace(b"</LocableBlobSet>", b"</LocableBlobSet>\n   " + body, 1)

        return run

    a = edited(tmp_path, "a.drp", {"project.xml": outside("MI")})
    b = edited(tmp_path, "b.drp", {"project.xml": outside("IM")})
    c = only_change(a, b)
    assert "order of child elements changed" in c.detail
    assert c.field_name.endswith("Sm2LockableBlobMap")


@needs_zstandard
def test_s03_units_inside_a_set_are_unordered(tmp_path: Path) -> None:
    timeline_blob = carrier(
        MARKER_BLOB_ID, SEQUENCE, markers_fields_blob(marker_entry(48, 32, "T"))
    )
    item_blob_ = carrier(
        "0c0c0c0c-1111-2222-3333-444444444444",
        V1_ITEM,
        markers_fields_blob(marker_entry(3, 2, "I")),
        tag="Sm2TiItemLockableBlob",
    )
    a = edited(tmp_path, "a.drp", {"project.xml": add_carrier(timeline_blob + item_blob_)})
    b = edited(tmp_path, "b.drp", {"project.xml": add_carrier(item_blob_ + timeline_blob)})
    d = diff(a, b, gate=True)
    assert d.changes == [] and d.gate.passed, d.to_text()


def test_s08_a_set_dbid_key_includes_the_tag(tmp_path: Path) -> None:
    a = edited(tmp_path, "a.drp", {"project.xml": lockable_set("mi")})

    def retag(data: bytes) -> bytes:
        data = lockable_set("mi")(data)
        return data.replace(b"<Sm2MpItemLockableBlob", b"<Sm2MpOtherLockableBlob").replace(
            b"</Sm2MpItemLockableBlob>", b"</Sm2MpOtherLockableBlob>"
        )

    d = assert_mirrors(a, edited(tmp_path, "b.drp", {"project.xml": retag}))
    assert sorted((c.kind, c.field_name.rsplit("/", 1)[-1][:22]) for c in d.changes) == [
        ("opaque-added", "Sm2MpOtherLockableBlob"),
        ("opaque-removed", "Sm2MpItemLockableBlob["),
    ]


def test_s09_two_copies_against_one_is_reported(tmp_path: Path) -> None:
    a = edited(tmp_path, "a.drp", {"project.xml": lockable_set("MM")})
    b = edited(tmp_path, "b.drp", {"project.xml": lockable_set("M")})
    d = assert_mirrors(a, b)
    assert [c.kind for c in d.changes] == ["opaque-removed"]
    assert not diff(a, b, gate=True).gate.passed
