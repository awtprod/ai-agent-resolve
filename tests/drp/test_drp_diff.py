"""Tests for ``aar.drp.diff``: properties, real template pairs and synthetic single edits.

The synthetic tests copy a bundled template into a temp dir, make one edit to the XML inside
the zip (``drp_edit``), and require exactly the expected change and nothing else.
"""

from __future__ import annotations

import json
import shutil
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from drp_edit import (
    add_carrier,
    after,
    carrier,
    chain,
    edit_zip,
    every,
    flip_hex_byte,
    leaf_text,
    marker_entry,
    markers_fields_blob,
    member_named,
    once,
    reindent,
    rewrite_clip_blobs,
    set_fields,
)
from drp_helpers import pb_var

from aar.drp.diff import Change, DiffError, ProjectDiff, diff_projects
from aar.drp.diff_model import REPORTED_FIELDS
from aar.drp.reader import read_drp

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"
ALL = ["empty-project.drp", "empty-project-r19.drp", "media-clip-h264.drp", "media-clip-r19.drp"]
H264 = FIXTURES / "media-clip-h264.drp"

# DbIds inside media-clip-h264.drp
V1_ITEM = "ad1fd562-da22-4b1f-b79f-f3106f9c2ac5"
A1_ITEM = "78f4666d-9ffc-4c26-a9cd-70bf093a36fa"
TIMELINE = "f75533e4-a99c-4cd9-8280-1f756f0d3bef"
SEQUENCE = "12d57950-0b88-4ccb-8cc5-f7638e046ebd"
CLIP = "2cd8eb8a-ae1b-4fbf-877a-c8dcda96e32b"
SEQ_MEMBER = member_named(H264, "SeqContainer/")
POOL = "MediaPool/Master/MpFolder.xml"
V1 = f'DbId="{V1_ITEM}"'.encode()

SWAP = {"added": "removed", "removed": "added", "opaque-added": "opaque-removed"}
SWAP |= {v: k for k, v in SWAP.items()}


def diff(a: Path, b: Path) -> ProjectDiff:
    return diff_projects(read_drp(a), read_drp(b))


def sig(c: Change) -> tuple[Any, ...]:
    """Everything structural about a change (not the wording, which names the newer side)."""
    return (
        c.section,
        c.kind,
        c.object_kind,
        c.object_id,
        c.match,
        c.timeline_id,
        c.field_name,
        json.dumps(c.to_dict()["old"], sort_keys=True),
        json.dumps(c.to_dict()["new"], sort_keys=True),
        tuple(sorted((c.delta or {}).items())),
        c.where_a,
        c.where_b,
        c.byte_count,
    )


def mirror_sig(c: Change) -> tuple[Any, ...]:
    """What the same change must look like when A and B are swapped (written independently)."""
    return (
        c.section,
        SWAP.get(c.kind, c.kind),
        c.object_kind,
        c.object_id,
        c.match,
        c.timeline_id,
        c.field_name,
        json.dumps(c.to_dict()["new"], sort_keys=True),
        json.dumps(c.to_dict()["old"], sort_keys=True),
        tuple(sorted((k, -v) for k, v in (c.delta or {}).items())),
        c.where_b,
        c.where_a,
        c.byte_count,
    )


def assert_mirrors(a: Path, b: Path) -> ProjectDiff:
    ab, ba = diff(a, b), diff(b, a)
    assert sorted(map(sig, ba.changes), key=repr) == sorted(map(mirror_sig, ab.changes), key=repr)
    return ab


def edited(tmp_path: Path, name: str, edits: dict[str, Callable[[bytes], bytes]]) -> Path:
    return edit_zip(H264, tmp_path / name, edits)


def only_change(a: Path, b: Path) -> Change:
    d = assert_mirrors(a, b)
    assert len(d.changes) == 1, d.to_text()
    return d.changes[0]


# --------------------------------------------------------------------------- properties


@pytest.mark.parametrize("name", ALL)
def test_diff_of_a_file_with_itself_is_empty(name: str, tmp_path: Path) -> None:
    src = FIXTURES / name
    assert diff(src, src).changes == []
    # Same members in another zip layout: the zip container itself is not content.
    rezipped = edit_zip(src, tmp_path / "rezipped.drp", reverse=True, stored=True)
    assert rezipped.read_bytes() != src.read_bytes()
    assert diff(src, rezipped).changes == []
    # Every interpreted member re-indented, so none is byte-identical and every element is
    # walked. (Members the reader does not interpret are compared as bytes; see below.)
    xml = [m.name for m in read_drp(src).members if m.interpreted]
    indented = edit_zip(src, tmp_path / "indented.drp", dict.fromkeys(xml, reindent))
    with zipfile.ZipFile(src) as z1, zipfile.ZipFile(indented) as z2:
        assert all(z1.read(n) != z2.read(n) for n in xml)
    d = diff(src, indented)
    assert d.identical and d.changes == [], d.to_text()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("empty-project.drp", "media-clip-h264.drp"),
        ("media-clip-r19.drp", "media-clip-h264.drp"),
        ("empty-project-r19.drp", "empty-project.drp"),
    ],
)
def test_real_pairs_mirror(a: str, b: str) -> None:
    d = assert_mirrors(FIXTURES / a, FIXTURES / b)
    assert d.changes


def by_kind(d: ProjectDiff) -> dict[tuple[str, str, str], list[Change]]:
    out: dict[tuple[str, str, str], list[Change]] = {}
    for c in d.changes:
        out.setdefault((c.section, c.object_kind, c.kind), []).append(c)
    return out


# --------------------------------------------------------------------------- real pairs


def test_empty_project_vs_media_clip() -> None:
    d = diff(FIXTURES / "empty-project.drp", FIXTURES / "media-clip-h264.drp")
    k = by_kind(d)
    header = [c for c in d.changes if c.section == "header"]
    assert not any(c.field_name in ("DbAppVer", "DbPrjVer") for c in header)  # both 21.0/17
    [rename] = k[("header", "project", "renamed")]
    assert (rename.old, rename.new) == ("_drt_template_src", "_media_tmpl")
    assert any(c.field_name == "id" for c in header)  # different project DbIds
    [clip] = k[("media", "media", "added")]
    assert clip.object_id == CLIP and clip.text.startswith('Media "sample.mp4" (DbId 2cd8eb8a…)')
    [old_tl] = k[("timelines", "timeline", "removed")]
    [new_tl] = k[("timelines", "timeline", "added")]
    assert old_tl.old == "Timeline 1" and "0 item(s)" in old_tl.detail
    assert new_tl.new == "MediaTemplate" and new_tl.object_id == TIMELINE
    assert "2 track(s), 2 item(s)" in new_tl.detail
    gallery = [c for c in d.changes if c.field_name == "Gallery.xml"]
    assert len(gallery) == 1 and gallery[0].kind == "opaque-changed"
    assert any(c.field_name == "SeqContainer/" and c.kind == "opaque-added" for c in d.changes)
    assert any("no DbId is shared" in n for n in d.notes)
    assert d.heuristic_count() == 0


def test_r19_vs_h264() -> None:
    d = diff(FIXTURES / "media-clip-r19.drp", FIXTURES / "media-clip-h264.drp")
    header = {c.field_name: (c.old, c.new) for c in d.changes if c.object_kind == "file"}
    assert header == {"DbAppVer": ("19.1.3.0007", "21.0.0.0048"), "DbPrjVer": (14, 17)}
    project = {c.field_name: (c.old, c.new) for c in d.changes if c.object_kind == "project"}
    assert project["project_version"] == ("14", "17")
    k = by_kind(d)
    assert sorted(c.old for c in k[("media", "bin", "removed")]) == ["Master", "Master/Archive"]
    assert [c.new for c in k[("media", "bin", "added")]] == ["Master"]
    assert [c.label.split(" (")[0] for c in k[("media", "media", "removed")]] == [
        'Media "r19_template_media.mp4"'
    ]
    [gone] = k[("timelines", "timeline", "removed")]
    [new] = k[("timelines", "timeline", "added")]
    assert "17 track(s)" in gone.detail and "2 track(s)" in new.detail
    assert d.counts()["markers"] == 0
    assert d.to_text().splitlines()[-1].startswith(f"summary: {len(d.changes)} changes (header 6")


# --------------------------------------------------------------------------- synthetic edits


def test_tail_trim(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: after(V1, b"<Duration>3664<", b"<Duration>3652<")})
    c = only_change(H264, b)
    assert (c.section, c.kind, c.object_kind, c.object_id, c.match) == (
        "timelines",
        "trimmed",
        "item",
        V1_ITEM,
        "dbid",
    )
    assert c.delta == {"head": 0, "tail": -12, "source_in": 0, "source_out": -12}
    assert c.timeline_id == TIMELINE
    assert c.text == (
        'Timeline "MediaTemplate" V1 item "sample.mp4" (DbId ad1fd562…): '
        "tail trimmed -12f (01:02:32:16 → 01:02:32:04); "
        "source out -12f (00:02:32:16 → 00:02:32:04)"
    )


def test_head_trim(tmp_path: Path) -> None:
    b = edited(
        tmp_path,
        "b.drp",
        {
            SEQ_MEMBER: chain(
                after(V1, b"<Start>86400<", b"<Start>86412<"),
                after(V1, b"<Duration>3664<", b"<Duration>3652<"),
                after(V1, b"<In/>", b"<In>12</In>"),
            )
        },
    )
    c = only_change(H264, b)
    assert c.kind == "trimmed"
    assert c.delta == {"head": 12, "tail": 0, "source_in": 12, "source_out": 0}
    assert "head trimmed +12f (01:00:00:00 → 01:00:00:12)" in c.detail


def test_move(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: after(V1, b"<Start>86400<", b"<Start>86424<")})
    c = only_change(H264, b)
    assert (c.kind, c.object_id, c.delta) == (
        "moved",
        V1_ITEM,
        {"record_start": 24, "record_end": 24},
    )
    assert c.detail == "moved +24f (01:00:00:00 → 01:00:01:00)"


def test_slip(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: after(V1, b"<In/>", b"<In>5</In>")})
    c = only_change(H264, b)
    assert (c.kind, c.delta) == ("slipped", {"source_in": 5, "source_out": 5})


def test_rename_timeline(tmp_path: Path) -> None:
    # Resolve writes the name twice: on the media-pool entry and on the timeline itself.
    b = edited(tmp_path, "b.drp", {POOL: every(b"<Name>MediaTemplate<", b"<Name>Cut 2<", 2)})
    c = only_change(H264, b)
    assert (c.section, c.kind, c.object_kind, c.object_id) == (
        "timelines",
        "renamed",
        "timeline",
        TIMELINE,
    )
    assert (c.old, c.new) == ("MediaTemplate", "Cut 2")
    assert c.text == 'Timeline "Cut 2" (DbId f75533e4…): renamed "MediaTemplate" → "Cut 2"'


def test_relink(tmp_path: Path) -> None:
    old = read_drp(H264).media_by_id()[CLIP].path
    new_dir, new_file = b"/Volumes/Media/relinked", b"sample.mp4"
    b = edited(tmp_path, "b.drp", {POOL: rewrite_clip_blobs(set_fields({1: new_dir, 2: new_file}))})
    with zipfile.ZipFile(H264) as z1, zipfile.ZipFile(b) as z2:
        assert z1.read(POOL).count(b"<Clip>") == 2 and z1.read(POOL) != z2.read(POOL)
    c = only_change(H264, b)
    assert (c.section, c.kind, c.object_kind, c.object_id) == ("media", "relinked", "media", CLIP)
    assert (c.old, c.new) == (old, "/Volumes/Media/relinked/sample.mp4")


def test_other_clip_blob_field_is_opaque_not_a_relink(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {POOL: rewrite_clip_blobs(set_fields({3: b"tampered"}))})
    d = assert_mirrors(H264, b)
    assert {(c.section, c.kind, c.field_name) for c in d.changes} == {
        ("opaque", "opaque-changed", "Video/BtVideoInfo/Clip"),
        ("opaque", "opaque-changed", "Audio/BtAudioInfo/Clip"),
    } or all(c.kind == "opaque-changed" and c.field_name.endswith("/Clip") for c in d.changes)
    assert len(d.changes) == 2
    assert all(c.object_id == CLIP and c.where_a.member == POOL for c in d.changes)


def marker_file(tmp_path: Path, name: str, *entries: bytes) -> Path:
    element = carrier(
        "0b0b0b0b-1111-2222-3333-444444444444", SEQUENCE, markers_fields_blob(*entries)
    )
    return edited(tmp_path, name, {"project.xml": add_carrier(element)})


def test_add_first_marker(tmp_path: Path) -> None:
    b = marker_file(tmp_path, "b.drp", marker_entry(48, 32, "Laugh", "big"))
    assert [(m.frame, m.name) for m in read_drp(b).timelines[0].markers] == [(48, "Laugh")]
    c = only_change(H264, b)
    assert (c.section, c.kind, c.object_kind, c.match) == (
        "markers",
        "added",
        "marker",
        "heuristic",
    )
    assert c.object_id == TIMELINE
    assert c.new == {
        "frame": 48,
        "color": "Red",
        "name": "Laugh",
        "note": "big",
        "duration": 1,
        "custom_data": "",
    }
    assert c.text == (
        'Timeline "MediaTemplate" marker "Laugh" at 01:00:02:00 (frame 48): added (Red, note "big")'
    )


def test_add_marker_to_existing_list(tmp_path: Path) -> None:
    a = marker_file(tmp_path, "a.drp", marker_entry(48, 32, "Laugh"))
    b = marker_file(tmp_path, "b.drp", marker_entry(48, 32, "Laugh"), marker_entry(96, 8, "End"))
    c = only_change(a, b)
    assert (c.kind, c.new["frame"], c.new["name"], c.new["color"]) == ("added", 96, "End", "Green")


def test_marker_moved_and_recoloured(tmp_path: Path) -> None:
    a = marker_file(tmp_path, "a.drp", marker_entry(48, 32, "Laugh"))
    b = marker_file(tmp_path, "b.drp", marker_entry(60, 32, "Laugh"))
    c = only_change(a, b)
    assert (c.kind, c.delta) == ("moved", {"frame": 12})
    b2 = marker_file(tmp_path, "b2.drp", marker_entry(48, 2, "Laugh"))
    c2 = only_change(a, b2)
    assert (c2.kind, c2.field_name) == ("changed", "color")
    assert 'color "Red" → "Blue"' in c2.detail


def test_remove_last_marker_from_its_blob(tmp_path: Path) -> None:
    a = marker_file(tmp_path, "a.drp", marker_entry(48, 32, "Laugh"))
    b = marker_file(tmp_path, "b.drp")
    assert read_drp(b).timelines[0].markers == []
    c = only_change(a, b)
    assert (c.section, c.kind, c.old["name"]) == ("markers", "removed", "Laugh")


def test_undecoded_field_inside_a_marker_is_opaque(tmp_path: Path) -> None:
    a = marker_file(tmp_path, "a.drp", marker_entry(48, 32, "Laugh", extra=pb_var(9, 1)))
    b = marker_file(tmp_path, "b.drp", marker_entry(48, 32, "Laugh", extra=pb_var(9, 2)))
    assert read_drp(a).timelines[0].markers[0].name == "Laugh"
    c = only_change(a, b)
    assert (c.kind, c.field_name, c.object_kind) == ("opaque-changed", "FieldsBlob", "markers")
    assert c.where_a is not None and c.where_a.member == "project.xml"


def test_flip_bytes_in_undecoded_blob(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: flip_hex_byte(V1, b"FieldsBlob", 40)})
    c = only_change(H264, b)
    assert (c.section, c.kind, c.object_kind, c.object_id) == (
        "opaque",
        "opaque-changed",
        "item",
        V1_ITEM,
    )
    assert c.field_name == "FieldsBlob" and c.byte_count == 1
    with zipfile.ZipFile(H264) as z:
        start, end = leaf_text(z.read(SEQ_MEMBER), V1, b"FieldsBlob")
    assert c.where_a is not None and c.where_b is not None
    assert (c.where_a.member, c.where_a.start, c.where_a.end) == (SEQ_MEMBER, start, end)
    assert (c.where_b.start, c.where_b.end) == (start, end)
    assert "1 of " in c.detail and "from byte 40" in c.detail


def test_same_decoded_value_with_different_text_is_not_equal(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: after(V1, b"<Start>86400<", b"<Start>086400<")})
    assert read_drp(b).timelines[0].tracks[0].items[0].record_start == 86400
    c = only_change(H264, b)
    assert (c.kind, c.field_name, c.object_id) == ("opaque-changed", "Start", V1_ITEM)
    assert "same decoded value" in c.detail


def test_undecoded_half_of_frame_rate_is_compared(tmp_path: Path) -> None:
    rate = b"<FrameRate>00000000000038400000000000000000</FrameRate>"
    b = edited(tmp_path, "b.drp", {POOL: once(rate, rate.replace(b"0000</", b"0001</"))})
    assert read_drp(b).timelines[0].frame_rate == 24.0
    c = only_change(H264, b)
    assert c.kind == "opaque-changed" and c.field_name.endswith("Sm2Sequence/FrameRate")
    assert c.byte_count == 1


def test_frame_rate_change_is_semantic_only(tmp_path: Path) -> None:
    rate = b"<FrameRate>00000000000038400000000000000000</FrameRate>"
    new = struct.pack("<d", 25.0).hex().encode() + b"0" * 16
    b = edited(tmp_path, "b.drp", {POOL: once(rate, b"<FrameRate>" + new + b"</FrameRate>")})
    c = only_change(H264, b)
    assert (c.kind, c.field_name, c.old, c.new) == ("changed", "frame_rate", 24.0, 25.0)


def test_unknown_element_added_inside_an_item(tmp_path: Path) -> None:
    b = edited(
        tmp_path,
        "b.drp",
        {SEQ_MEMBER: after(V1, b"<WasDisbanded>", b"<NewThingBA>00ff</NewThingBA><WasDisbanded>")},
    )
    c = only_change(H264, b)
    assert (c.kind, c.field_name, c.byte_count, c.object_id) == (
        "opaque-added",
        "NewThingBA",
        2,
        V1_ITEM,
    )


def test_gallery_and_members_compared_raw(tmp_path: Path) -> None:
    b = edited(tmp_path, "b.drp", {"Gallery.xml": lambda d: d + b"\n<!-- x -->"})
    c = only_change(H264, b)
    assert (c.kind, c.field_name, c.object_kind) == ("opaque-changed", "Gallery.xml", "member")
    extra = tmp_path / "extra.drp"
    shutil.copy(H264, extra)
    with zipfile.ZipFile(extra, "a") as z:
        z.writestr("LUT/look.cube", b"LUT_3D_SIZE 2\n")
    c = only_change(H264, extra)
    assert (c.kind, c.field_name, c.byte_count) == ("opaque-added", "LUT/look.cube", 14)


def test_heuristic_matching_without_dbids(tmp_path: Path) -> None:
    strip = chain(
        once(f' DbId="{V1_ITEM}"'.encode(), b""), once(f' DbId="{A1_ITEM}"'.encode(), b"")
    )
    a = edited(tmp_path, "a.drp", {SEQ_MEMBER: strip})
    b = edited(
        tmp_path,
        "b.drp",
        {SEQ_MEMBER: chain(after(V1, b"<Duration>3664<", b"<Duration>3600<"), strip)},
    )
    items = read_drp(a).timelines[0].tracks[0].items
    assert items[0].id == ""
    c = only_change(a, b)
    assert (c.kind, c.match, c.delta["tail"]) == ("trimmed", "heuristic", -64)
    assert c.object_id is not None and c.object_id.startswith("h:")
    assert "(no DbId)" in c.text


def test_every_reader_span_is_reported_or_kept_opaque() -> None:
    """A span key the reader adds later must be classified here, not silently trusted."""
    known_opaque = {"media": {"blob_path", "linked_paths", "codec"}, "item": {"speed"}}
    for name in ALL:
        drp = read_drp(FIXTURES / name)
        found: dict[str, set[str]] = {}
        if drp.project:
            found["project"] = {k.split("[")[0] for k in drp.project.spans}
        found["bin"] = {k for x in drp.bins for k in x.spans}
        found["media"] = {k.split("[")[0] for x in drp.media for k in x.spans}
        found["timeline"] = {k for t in drp.timelines for k in t.spans}
        found["track"] = {k for t in drp.timelines for tr in t.tracks for k in tr.spans}
        found["item"] = {
            k for t in drp.timelines for tr in t.tracks for i in tr.items for k in i.spans
        }
        for kind, keys in found.items():
            extra = keys - REPORTED_FIELDS[kind] - known_opaque.get(kind, set())
            assert not extra, (name, kind, extra)


# --------------------------------------------------------------------------- renderings


def test_text_and_json_renderings(tmp_path: Path) -> None:
    b = edited(
        tmp_path,
        "b.drp",
        {
            SEQ_MEMBER: chain(
                after(V1, b"<Start>86400<", b"<Start>86424<"),
                flip_hex_byte(V1, b"FieldsBlob", 3),
            )
        },
    )
    d = diff(H264, b)
    text = d.to_text().splitlines()
    assert text[0].startswith("a: ") and "DRP, Resolve 21.0.0.0048, DbPrjVer 17" in text[0]
    assert text[1].startswith("b: ")
    assert text[2].endswith("moved +24f (01:00:00:00 → 01:00:01:00)")
    assert "opaque <FieldsBlob> changed" in text[3]
    assert text[-1] == (
        "summary: 2 changes (header 0, media 0, timelines 1, markers 0, opaque 1); "
        "0 heuristic matches"
    )
    data = json.loads(json.dumps(d.to_dict()))
    assert data["identical"] is False and data["summary"]["changes"] == 2
    assert data["summary"]["by_section"]["opaque"] == 1
    moved = data["changes"][0]
    assert moved["kind"] == "moved" and moved["object_id"] == V1_ITEM
    assert moved["where_a"]["member"] == SEQ_MEMBER and moved["delta"]["record_start"] == 24
    only = d.only(["opaque"])
    assert [c.section for c in only.changes] == ["opaque"]
    assert only.to_text().splitlines()[-1].startswith("summary: 1 change (opaque 1)")
    with pytest.raises(ValueError, match="unknown section"):
        d.only(["nope"])


def test_file_changed_after_reading_is_an_error(tmp_path: Path) -> None:
    a = tmp_path / "a.drp"
    shutil.copy(H264, a)
    drp = read_drp(a)
    edit_zip(H264, tmp_path / "x.drp", {"Gallery.xml": lambda d: d + b" "})
    shutil.copy(tmp_path / "x.drp", a)
    with pytest.raises(DiffError, match="changed on disk"):
        diff_projects(drp, read_drp(H264))


def test_non_zip_files_are_compared_whole(tmp_path: Path) -> None:
    x, y = tmp_path / "x.drb", tmp_path / "y.drb"
    x.write_bytes(b"SQLite format 3\x00" + bytes(64))
    y.write_bytes(b"SQLite format 3\x00" + bytes(60) + b"\x01\x02\x00\x00")
    assert diff(x, x).changes == []
    c = only_change(x, y)
    assert (c.kind, c.object_kind, c.byte_count) == ("opaque-changed", "file", 2)
