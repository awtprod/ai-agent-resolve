"""S8b: follow-ups from the S8 validation (findings F-A to F-J and the listed test gaps).

Each test names the finding it pins. They run on the bundled templates only.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from diff_testing import CLIP, H264, POOL, SEQUENCE, TIMELINE
from drp_edit import add_carrier, carrier, edit_zip, marker_entry, markers_fields_blob
from drp_helpers import UnseekableSink, kd, needs_zstandard, pb_len, zstd_raw_frame

from aar.drp import blobs
from aar.drp.modify import compare_round_trip, encode, parse_plan, zipraw
from aar.drp.modify.encode import EncodeError
from aar.drp.modify.markers import COLOR_BITS
from aar.drp.modify.write import SIDECAR_SUFFIX, edit_file
from aar.drp.modify.write import leftovers as stale_temp_files
from aar.drp.reader import read_drp

MAC_DIR = "/Users/samuelgursky/Desktop"
BLUE = 2


def plan(*edits: dict[str, Any], **top: Any) -> Any:
    return parse_plan(json.dumps({"edits": list(edits), **top}))


def run(src: Path, tmp_path: Path, *edits: dict[str, Any], **kw: Any) -> Any:
    out = tmp_path / "out.drp"
    return edit_file(src, plan(*edits), out, **kw), out


# --------------------------------------------------------------------------- F-A


def _split_frames(payload: bytes) -> bytes:
    """``payload`` as two concatenated zstd frames. With zstandard the reader decodes both (the
    content is the same); without a library only the first frame is read."""
    half = len(payload) // 2
    return encode.zstd_raw_frame(payload[:half]) + encode.zstd_raw_frame(payload[half:])


def _clip_in_two_frames(data: bytes) -> bytes:
    def repl(m: re.Match[bytes]) -> bytes:
        raw = bytes.fromhex(m.group(1).decode())
        env = blobs.parse_envelope(raw)
        assert env is not None and env.codec == "zstd" and env.version is not None
        body = b"\x81" + _split_frames(blobs.unwrap_message(raw))
        new = struct.pack(">II", env.version, len(body)) + body
        return b"<Clip>" + new.hex().encode() + b"</Clip>"

    new, n = re.subn(rb"<Clip>([0-9a-f]+)</Clip>", repl, data)
    assert n >= 1
    return new


@needs_zstandard
def test_fa_path_map_refuses_a_clip_blob_in_two_zstd_frames(tmp_path: Path) -> None:
    src = edit_zip(H264, tmp_path / "two-frames.drp", {POOL: _clip_in_two_frames})
    drp = read_drp(src)
    assert not [i for i in drp.issues if i.severity != "info"]
    assert drp.media_by_id()[CLIP].path == read_drp(H264).media_by_id()[CLIP].path
    table = [{"windows": "E:", "mac": MAC_DIR}]
    outcome, out = run(
        src, tmp_path, {"op": "path_map", "direction": "mac_to_windows", "table": table}
    )
    assert outcome.status == "refused"
    [refusal] = outcome.refusals
    assert refusal.code == "cannot-encode" and "one zstd frame" in refusal.message
    assert not out.exists()


def _two_frame_markers(*entries: bytes) -> str:
    body = b"\x81" + _split_frames(pb_len(2, b"".join(entries)))
    blob_data = struct.pack(">II", 10001, len(body)) + body
    return kd([("BlobData", 0x0C, struct.pack(">I", len(blob_data)) + blob_data)]).hex()


@needs_zstandard
def test_fa_marker_edit_refuses_blob_data_in_two_zstd_frames(tmp_path: Path) -> None:
    element = carrier(
        "0b0b0b0b-1111-2222-3333-444444444444",
        SEQUENCE,
        _two_frame_markers(marker_entry(10, BLUE, "Intro"), marker_entry(20, BLUE, "Outro")),
    )
    src = edit_zip(H264, tmp_path / "markers.drp", {"project.xml": add_carrier(element)})
    assert [m.name for m in read_drp(src).timelines[0].markers] == ["Intro", "Outro"]
    edit = {
        "op": "marker_update",
        "owner": "timeline",
        "owner_id": TIMELINE,
        "frame": 10,
        "set": {"name": "X"},
    }
    outcome, out = run(src, tmp_path, edit)
    assert [r.code for r in outcome.refusals] == ["no-marker-list"]
    assert "one zstd frame" in outcome.refusals[0].message
    assert not out.exists()


@pytest.mark.parametrize(
    ("frame", "why"),
    [
        (zstd_raw_frame(b"abc") + b"\x00", "trailing"),
        (zstd_raw_frame(b"abc") + zstd_raw_frame(b"d"), "trailing"),
        (zstd_raw_frame(b"abc")[:-1], "truncated"),
        (zstd_raw_frame(b"abc")[:8], "^truncated zstd block header"),  # 2 of 3 header bytes
        (b"\x50\x2a\x4d\x18" + b"\x00" * 8, "^not a zstd frame"),  # a skippable frame
        (zstd_raw_frame(b"abc")[:4] + bytes([0x21]) + zstd_raw_frame(b"abc")[5:], "dictionary"),
        (zstd_raw_frame(b"abc")[:4] + bytes([0x28]) + zstd_raw_frame(b"abc")[5:], "reserved bit"),
        (zstd_raw_frame(b"abc")[:6] + (7 << 1 | 1).to_bytes(3, "little"), "reserved block"),
    ],
    ids=[
        "byte-after",
        "second-frame",
        "truncated",
        "short-block-header",
        "skippable",
        "dictionary",
        "reserved-header-bit",
        "reserved-block",
    ],
)
def test_fa_check_single_frame_refuses(frame: bytes, why: str) -> None:
    with pytest.raises(EncodeError, match=why):
        encode.check_single_frame(frame)


def test_fa_check_single_frame_sizes() -> None:
    for payload in (b"", b"x" * 300, b"y" * 140_000):
        frame = encode.zstd_raw_frame(payload)
        assert encode.check_single_frame(frame) == len(frame)
    rle = b"\x28\xb5\x2f\xfd\x20\x10" + ((16 << 3) | (1 << 1) | 1).to_bytes(3, "little") + b"z"
    assert encode.check_single_frame(rle) == len(rle)
    with_checksum = (
        b"\x28\xb5\x2f\xfd\x24\x01" + ((1 << 3) | 1).to_bytes(3, "little") + b"q" + b"\0" * 4
    )
    assert encode.check_single_frame(with_checksum) == len(with_checksum)
    with pytest.raises(EncodeError, match="truncated"):
        encode.check_single_frame(with_checksum[:-1])


@needs_zstandard
def test_fa_compressed_frames_are_walked_by_block_size() -> None:
    import zstandard

    payload = b"compressible " * 5000
    frame = zstandard.ZstdCompressor(level=3).compress(payload)
    assert encode.check_single_frame(frame) == len(frame)
    with pytest.raises(EncodeError, match="trailing"):
        encode.check_single_frame(frame + frame)


# --------------------------------------------------------------------------- F-G


def _settings_edit(change: Callable[[bytes, blobs.KeyedDict], bytes]) -> Callable[[bytes], bytes]:
    """Rewrite SM_Config's FieldsBlob keyed dict with ``change(kd_bytes, decoded)``."""

    def run_edit(data: bytes) -> bytes:
        start = data.index(b"<FieldsBlob>", data.index(b"<SM_Config")) + len(b"<FieldsBlob>")
        end = data.index(b"</FieldsBlob>", start)
        kd_bytes = blobs.parse_hex(data[start:end].decode())
        new = change(kd_bytes, blobs.decode_keyed_dict(kd_bytes, strict=True))
        return data[:start] + new.hex().encode() + data[end:]

    return run_edit


def _new_save_date(kd_bytes: bytes, kd_: blobs.KeyedDict, *, pad: int | None = None) -> bytes:
    """SetupBA with field 124 set to another date; ``pad`` re-encodes that varint field padded."""
    entry = kd_.entry("SetupBA")
    assert entry is not None and isinstance(entry.value, bytes)
    payload = blobs.unwrap_message(entry.value)
    fields = blobs.decode_protobuf(payload, nested_depth=0)
    date = next(f for f in fields if f.number == 124)
    payload = encode.replace_field(payload, date, b"Jan 19 2026 00:17:51")
    if pad is not None:
        f = next(x for x in blobs.decode_protobuf(payload, nested_depth=0) if x.number == pad)
        assert f.wire == 0 and isinstance(f.value, int) and f.value < 128
        padded = encode.pb_key(f.number, 0) + bytes([f.value | 0x80, 0x00])
        payload = payload[: f.offset] + padded + payload[f.end :]
    return encode.replace_kd_bytes(kd_bytes, entry, encode.envelope(entry.value, payload))


def _header_word_2(kd_bytes: bytes, kd_: blobs.KeyedDict) -> bytes:
    data = _new_save_date(kd_bytes, kd_)
    return struct.pack(">I", 2) + data[4:]


def _setup_null_flag(kd_bytes: bytes, kd_: blobs.KeyedDict) -> bytes:
    data = _new_save_date(kd_bytes, kd_)
    entry = blobs.decode_keyed_dict(data, strict=True).entry("SetupBA")
    assert entry is not None and entry.subtype == 0
    (klen,) = struct.unpack_from(">I", data, entry.offset)
    flag_at = entry.offset + 4 + klen + 4
    return data[:flag_at] + b"\x01" + data[flag_at + 1 :]


def _pad_field_8(kd_bytes: bytes, kd_: blobs.KeyedDict) -> bytes:
    return _new_save_date(kd_bytes, kd_, pad=8)


def _pad_date_length(kd_bytes: bytes, kd_: blobs.KeyedDict) -> bytes:
    """The new date, in a field 124 whose length prefix takes two bytes (20 = 0x94 0x00)."""
    entry = kd_.entry("SetupBA")
    assert entry is not None and isinstance(entry.value, bytes)
    payload = blobs.unwrap_message(entry.value)
    date = next(f for f in blobs.decode_protobuf(payload, nested_depth=0) if f.number == 124)
    padded = encode.pb_key(124, 2) + b"\x94\x00" + b"Jan 19 2026 00:17:51"
    assert len(b"Jan 19 2026 00:17:51") == 20
    payload = payload[: date.offset] + padded + payload[date.end :]
    return encode.replace_kd_bytes(kd_bytes, entry, encode.envelope(entry.value, payload))


@needs_zstandard
@pytest.mark.parametrize(
    "change",
    [_header_word_2, _setup_null_flag, _pad_field_8, _pad_date_length],
    ids=["kd-header-word", "setupba-null-flag", "setupba-padded-varint", "date-padded-length"],
)
def test_fg_settings_rule_covers_the_whole_blob(tmp_path: Path, change: Any) -> None:
    ok = edit_zip(H264, tmp_path / "ok.drp", {"project.xml": _settings_edit(_new_save_date)})
    assert compare_round_trip(H264, ok).ok
    bad = edit_zip(H264, tmp_path / "bad.drp", {"project.xml": _settings_edit(change)})
    for result in (compare_round_trip(H264, bad), compare_round_trip(bad, H264)):
        assert [r.code for r in result.refusals] == ["unexplained-opaque"]
        assert "settings-save-date" not in [rule for rule, _ in result.allowed]


def test_fg_bin_rule_compares_field_bytes(tmp_path: Path) -> None:
    def blob(
        stamp: int, field_5: bytes = encode.pb_varint(5, 1), stamp_field: bytes | None = None
    ) -> bytes:
        payload = (encode.pb_varint(4, stamp) if stamp_field is None else stamp_field) + field_5
        return (struct.pack(">II", 2, len(payload) + 1) + b"\x80" + payload).hex().encode()

    def with_blob(b: bytes) -> Callable[[bytes], bytes]:
        return lambda d: d.replace(b"<FieldsBlob/>", b"<FieldsBlob>" + b + b"</FieldsBlob>", 1)

    a = edit_zip(H264, tmp_path / "a.drp", {POOL: with_blob(blob(300))})
    (tmp_path / "b").mkdir()
    b = edit_zip(H264, tmp_path / "b" / "b.drp", {POOL: with_blob(blob(301))})
    assert compare_round_trip(a, b).ok
    padded_5 = encode.pb_key(5, 0) + b"\x81\x00"  # field 5 = 1, written in two bytes
    (tmp_path / "c").mkdir()
    c = edit_zip(H264, tmp_path / "c" / "c.drp", {POOL: with_blob(blob(301, padded_5))})
    assert blobs.decode_protobuf(padded_5, nested_depth=0)[0].value == 1
    assert [r.code for r in compare_round_trip(a, c).refusals] == ["unexplained-opaque"]
    # The stamp itself written in three bytes (301 = 0xad 0x02, padded to 0xad 0x82 0x00).
    padded_4 = encode.pb_key(4, 0) + b"\xad\x82\x00"
    assert blobs.decode_protobuf(padded_4, nested_depth=0)[0].value == 301
    (tmp_path / "d").mkdir()
    d = edit_zip(H264, tmp_path / "d" / "d.drp", {POOL: with_blob(blob(301, stamp_field=padded_4))})
    for old_side, new_side in ((a, d), (d, a)):
        assert [r.code for r in compare_round_trip(old_side, new_side).refusals] == [
            "unexplained-opaque"
        ]


# --------------------------------------------------------------------------- F-F


@pytest.mark.parametrize(
    ("row", "path", "mapped"),
    [
        ("E:\\Straße", "E:\\STRASSE\\a.mov", None),  # full case folding would match these
        ("E:\\STRASSE", "e:\\straße\\a.mov", None),
        ("E:\\ﬁles", "E:\\FILES\\a.mov", None),  # the fi ligature folds to "fi"
        ("E:\\\u212a", "E:\\k\\a.mov", None),  # KELVIN SIGN folds to "k"; NTFS keeps it apart
        ("E:\\Straße", "E:\\STRAßE\\a.mov", "/Volumes/M/a.mov"),  # ß has no simple uppercase
        ("E:\\Ärger", "e:\\äRGER\\a.mov", "/Volumes/M/a.mov"),
        ("E:\\Σίσυφος", "E:\\ΣΊΣΥΦΟΣ\\a.mov", "/Volumes/M/a.mov"),
    ],
    ids=["sharp-s-row", "sharp-s-path", "ligature", "kelvin", "sharp-s-kept", "umlaut", "greek"],
)
def test_ff_windows_names_compare_like_ntfs(row: str, path: str, mapped: str | None) -> None:
    from aar.drp.modify.paths import PathMapper

    result = PathMapper("windows_to_mac", [(row, "/Volumes/M")]).map(path)
    assert result.new == mapped
    if mapped is None:
        assert result.reason == "no-mapping"


def test_ff_rows_that_differ_only_under_full_folding_are_distinct() -> None:
    from aar.drp.modify.paths import PathMapper

    m = PathMapper("windows_to_mac", [("E:\\Straße", "/Volumes/A"), ("E:\\STRASSE", "/Volumes/B")])
    assert m.map("E:\\straße\\x").new == "/Volumes/A/x"
    assert m.map("E:\\strasse\\x").new == "/Volumes/B/x"


# L3: str.upper() EXPANDS the Greek iota-subscript letters (full case mapping), so the per-char
# rule left them unfolded. Their Unicode simple uppercase is a single precomposed capital, a
# collision NTFS $UpCase really has; ntfs_upper must fold onto it.


@pytest.mark.parametrize(
    ("small", "capital"),
    [("\u1f80", "\u1f88"), ("\u1fb3", "\u1fbc"), ("\u1fc3", "\u1fcc"), ("\u1ff3", "\u1ffc")],
    ids=["alpha-psili-ypo", "alpha-ypo", "eta-ypo", "omega-ypo"],
)
def test_ff_greek_iota_subscript_folds_onto_its_capital_like_ntfs(small: str, capital: str) -> None:
    from aar.drp.modify.paths import PathMapper, ntfs_upper

    assert len(small.upper()) > 1  # Python's full uppercase expands it (base + capital iota)
    assert ntfs_upper(small) == ntfs_upper(capital)  # NTFS simple uppercase folds them together
    mapped = PathMapper("windows_to_mac", [("E:\\" + small, "/Volumes/M")]).map(
        "E:\\" + capital + "\\a.mov"
    )
    assert mapped.new == "/Volumes/M/a.mov"


@pytest.mark.parametrize(
    ("ch", "expansion"),
    [("\u00df", "SS"), ("\ufb01", "FI"), ("\u0149", "\u02bcN")],
    ids=["sharp-s", "fi-ligature", "apostrophe-n"],
)
def test_ff_full_expansions_keep_their_spelling(ch: str, expansion: str) -> None:
    from aar.drp.modify.paths import ntfs_upper

    assert ch.upper() == expansion  # Python's full uppercase does change the length
    assert ntfs_upper(ch) == ch  # simple uppercase has no single code point, so it is kept
    assert ntfs_upper(ch) != ntfs_upper(expansion)  # never equal to the expanded form


def test_ff_simple_uppercase_table_is_the_unicode_set() -> None:
    # The override table is exactly the code points whose full uppercase expands yet whose simple
    # (single-code-point) uppercase exists. Re-derived from the running Unicode database so it can
    # never silently drift from Python's data.
    from aar.drp.modify.paths import _SIMPLE_UPPER_EXPANDING

    derived = {}
    for cp in range(0x110000):
        c = chr(cp)
        title = c.title()
        if len(c.upper()) > 1 and len(title) == 1 and title != c:
            derived[c] = title
    assert derived == _SIMPLE_UPPER_EXPANDING
    assert len(_SIMPLE_UPPER_EXPANDING) == 27  # the Greek ypogegrammeni set


# --------------------------------------------------------------------------- F-C

RENAME = {"op": "rename", "target": "project", "new_name": "x"}


def _interrupted_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """What a hard kill between placing the label and placing the file leaves behind."""
    outcome, out = run(H264, tmp_path, RENAME)
    assert outcome.status == "written"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    out.unlink()  # the file was never moved into place
    stale = tmp_path / ".out.drp.k3j9x_2a.tmp.drp"  # its verified temp copy stayed behind
    stale.write_bytes(b"partial")
    return out, sidecar, stale


def test_fc_orphan_sidecar_gets_a_specific_refusal(tmp_path: Path) -> None:
    out, sidecar, stale = _interrupted_run(tmp_path)
    outcome = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in outcome.refusals] == ["orphan-sidecar"]
    details = outcome.refusals[0].details
    assert details["remove"] == [str(sidecar), str(stale)]
    assert str(sidecar) in outcome.refusals[0].message
    assert sidecar.exists() and stale.exists() and not out.exists()  # nothing deleted
    sidecar.unlink()
    stale.unlink()
    assert edit_file(H264, plan(RENAME), out).status == "written"


def test_fc_a_sidecar_we_did_not_write_is_plain_out_exists(tmp_path: Path) -> None:
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    for text in ("{}", "not json", json.dumps({"tool": "aar 0", "label": "unverified"})):
        sidecar.write_text(text)
        outcome = edit_file(H264, plan(RENAME), out)
        assert [r.code for r in outcome.refusals] == ["out-exists"], text
        # Refused up front, not after a try to write: the pre-check names the label's path.
        assert "pass --force" in outcome.refusals[0].message, text
        assert outcome.refusals[0].details == {"paths": [str(sidecar)]}, text
        assert sidecar.read_text() == text
        assert [p.name for p in tmp_path.iterdir()] == [sidecar.name]  # no temporary file


def test_fc_leftover_temp_files_are_reported_not_deleted(tmp_path: Path) -> None:
    stale = [
        tmp_path / ".out.drp.abcd1234.tmp.drp",
        tmp_path / ".out.drp.aar.json.zz_9yy1q.tmp.json",
    ]
    unrelated = [
        tmp_path / ".out.drp.notours.txt",
        tmp_path / ".other.drp.abcd1234.tmp.drp",
        tmp_path / ".out.drp.abcd1234.tmp.drp.keep",  # starts like a temporary file, is not one
        tmp_path / ".out.drp.aar.json.zz_9yy1q.tmp.json.bak",
    ]
    for p in stale + unrelated:
        p.write_bytes(b"x")
    outcome, _ = run(H264, tmp_path, RENAME)
    assert outcome.status == "written"
    assert outcome.to_dict()["leftovers"] == sorted(str(p) for p in stale)
    assert all(p.exists() for p in stale + unrelated)


# --------------------------------------------------------------------------- F-B


@pytest.mark.parametrize(
    "edit",
    [
        {"op": "rename", "target": "project", "new_name": "_media_tmpl"},
        {"op": "rename", "target": "clip", "id": CLIP, "new_name": "sample.mp4"},
        {"op": "rename", "target": "timeline", "id": TIMELINE, "new_name": "MediaTemplate"},
    ],
    ids=["project", "clip", "timeline"],
)
def test_fb_renaming_to_the_current_name_is_a_reported_no_op(
    tmp_path: Path, edit: dict[str, Any]
) -> None:
    outcome, out = run(H264, tmp_path, edit)
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    assert outcome.changes == [] and outcome.expected == []
    [report] = outcome.report
    assert report["unchanged"] is True and report["old"] == report["new"] == edit["new_name"]
    assert out.read_bytes() == H264.read_bytes()


def test_fb_a_no_op_rename_beside_a_real_one(tmp_path: Path) -> None:
    outcome, _ = run(
        H264,
        tmp_path,
        {"op": "rename", "target": "project", "new_name": "_media_tmpl"},
        {"op": "rename", "target": "clip", "id": CLIP, "new_name": "Interview"},
    )
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    assert [(c["object_kind"], c["new"]) for c in outcome.changes] == [("media", "Interview")]
    assert [r.get("unchanged", False) for r in outcome.report] == [True, False]


# --------------------------------------------------------------------------- F-H

UNDECODED = struct.pack("B", 7 << 3) + b"\x05"  # a varint field 7 the reader does not decode


def _odd_marker_src(tmp_path: Path, extra: bytes = UNDECODED, **kw: Any) -> Path:
    entries = [marker_entry(10, BLUE, "Intro", extra=extra, **kw), marker_entry(20, BLUE, "B")]
    element = carrier(
        "0b0b0b0b-1111-2222-3333-444444444444", SEQUENCE, markers_fields_blob(*entries)
    )
    return edit_zip(H264, tmp_path / "odd.drp", {"project.xml": add_carrier(element)})


def _marker(edit: dict[str, Any]) -> dict[str, Any]:
    return {"owner": "timeline", "owner_id": TIMELINE, **edit}


def test_fh_deleting_a_marker_with_undecoded_fields_is_refused_before_the_gate(
    tmp_path: Path,
) -> None:
    src = _odd_marker_src(tmp_path)
    outcome, out = run(src, tmp_path, _marker({"op": "marker_delete", "frame": 10}))
    assert [r.code for r in outcome.refusals] == ["marker-undecoded"]
    assert outcome.gate is None  # refused at apply time; nothing was written or diffed
    assert not out.exists()
    plain, _ = run(src, tmp_path, _marker({"op": "marker_delete", "frame": 20}))
    assert plain.status == "written", plain.to_dict()["refusals"]


def test_fh_an_update_that_would_rewrite_undecoded_bytes_is_refused(tmp_path: Path) -> None:
    src = _odd_marker_src(tmp_path, duration=b"01")  # "01" is kept raw: not Resolve's spelling
    assert read_drp(src).timelines[0].markers[0].duration == 1
    outcome, out = run(
        src, tmp_path, _marker({"op": "marker_update", "frame": 10, "set": {"duration": 24}})
    )
    assert [r.code for r in outcome.refusals] == ["marker-undecoded"]
    assert not out.exists()


def test_fh_a_non_canonical_encoding_alone_is_undecoded(tmp_path: Path) -> None:
    # No undecoded field here: the frame's varint is written in two bytes (10 = 0x8a 0x00). It
    # reads as 10 but is not how Resolve writes it, and the gate compares those bytes raw, so no
    # edit may delete this entry or rewrite the bytes of its frame.
    src = _odd_marker_src(tmp_path, extra=b"", frame_raw=b"\x8a\x00")
    assert read_drp(src).timelines[0].markers[0].frame == 10
    outcome, out = run(src, tmp_path, _marker({"op": "marker_delete", "frame": 10}))
    assert [r.code for r in outcome.refusals] == ["marker-undecoded"]
    assert not out.exists()
    moved, out = run(
        src, tmp_path, _marker({"op": "marker_update", "frame": 10, "set": {"frame": 12}})
    )
    assert moved.status == "refused" and moved.refusals and not out.exists()
    # The padded bytes stay as they are under an edit that does not touch them.
    renamed, _ = run(
        src, tmp_path, _marker({"op": "marker_update", "frame": 10, "set": {"name": "Intro 2"}})
    )
    assert renamed.status == "written", renamed.to_dict()["refusals"]
    plain, _ = run(src, tmp_path, _marker({"op": "marker_delete", "frame": 20}), force=True)
    assert plain.status == "written", plain.to_dict()["refusals"]


def _fields_blob_of(archive: Path) -> str:
    with zipfile.ZipFile(archive) as zf:
        found = re.findall(
            rb'<Sm2SequenceLockableBlob DbId="0b0b0b0b[^>]*>\s*<FieldsBlob>([0-9a-f]+)</Fields',
            zf.read("project.xml"),
        )
    assert len(found) == 1
    return found[0].decode()


def test_fh_multi_field_update_keeps_undecoded_fields(tmp_path: Path) -> None:
    src = _odd_marker_src(tmp_path)
    change = {"name": "Big", "note": "n", "color": "Red", "custom_data": "agent:7", "frame": 12}
    outcome, out = run(src, tmp_path, _marker({"op": "marker_update", "frame": 10, "set": change}))
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    [m, _] = read_drp(out).timelines[0].markers
    assert (m.frame, m.name, m.note, m.color, m.custom_data) == (12, "Big", "n", "Red", "agent:7")
    # The whole list, byte for byte: the colour, note and name re-encoded in place, the frame
    # moved, the custom data appended, and field 7 (undecoded) still sitting between them.
    expected = markers_fields_blob(
        marker_entry(
            12,
            COLOR_BITS["Red"],
            "Big",
            note="n",
            extra=UNDECODED + pb_len(6, b"agent:7"),
        ),
        marker_entry(20, BLUE, "B"),
    )
    assert _fields_blob_of(out) == expected


# --------------------------------------------------------------------------- F-E


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o600, 0o444])
def test_fe_output_and_sidecar_take_the_input_mode(tmp_path: Path, mode: int) -> None:
    src = tmp_path / "in.drp"
    src.write_bytes(H264.read_bytes())
    src.chmod(mode)
    (tmp_path / "o").mkdir()
    out = tmp_path / "o" / "out.drp"
    outcome = edit_file(src, plan(RENAME), out)
    assert outcome.status == "written"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    assert (out.stat().st_mode & 0o777, sidecar.stat().st_mode & 0o777) == (mode, mode)


def _fail_placing(monkeypatch: pytest.MonkeyPatch, target_suffix: str) -> None:
    from aar.drp.modify import write

    real = write._place

    def place(tmp: Path, dest: Path, force: bool) -> None:
        if dest.name.endswith(target_suffix):
            raise OSError(28, "No space left on device")
        real(tmp, dest, force)

    monkeypatch.setattr(write, "_place", place)


def test_fe_force_keeps_the_old_sidecar_when_placing_the_file_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, out = run(H264, tmp_path, RENAME)
    assert first.status == "written"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    old_label, old_file = sidecar.read_bytes(), out.read_bytes()
    _fail_placing(monkeypatch, ".drp")
    with pytest.raises(OSError, match="No space"):
        edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert sidecar.read_bytes() == old_label and out.read_bytes() == old_file
    assert stale_temp_files(out) == []


def test_fe_a_failed_placement_removes_the_new_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_placing(monkeypatch, ".drp")
    with pytest.raises(OSError, match="No space"):
        run(H264, tmp_path, RENAME)
    assert list(tmp_path.iterdir()) == []


# Placement failures that really happen (no patched internals): the output name is taken by a
# directory or a dangling symlink, or a name is too long for the filesystem.


def _directory_in_the_way(tmp_path: Path) -> Path:
    out = tmp_path / "out.drp"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    return out


def test_fe_force_over_a_directory_restores_the_old_label(tmp_path: Path) -> None:
    first, out = run(H264, tmp_path, RENAME)
    assert first.status == "written"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    old_label = sidecar.read_bytes()
    out.unlink()
    _directory_in_the_way(tmp_path)
    with pytest.raises(IsADirectoryError):
        edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert sidecar.read_bytes() == old_label  # the label being replaced is back
    assert (out / "keep.txt").read_text() == "keep"
    assert stale_temp_files(out) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.drp", "out.drp.aar.json"]


def test_fe_force_over_a_directory_leaves_no_label_when_there_was_none(tmp_path: Path) -> None:
    out = _directory_in_the_way(tmp_path)
    with pytest.raises(IsADirectoryError):
        edit_file(H264, plan(RENAME), out, force=True)
    assert [p.name for p in tmp_path.iterdir()] == ["out.drp"]
    assert (out / "keep.txt").read_text() == "keep"


def test_fe_a_dangling_symlink_at_the_output_is_not_replaced(tmp_path: Path) -> None:
    out = tmp_path / "out.drp"
    out.symlink_to(tmp_path / "nowhere.drp")  # exists() is False; link() still refuses it
    outcome = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in outcome.refusals] == ["out-exists"]
    assert "appeared while writing" in outcome.refusals[0].message
    assert outcome.status == "refused" and outcome.output is None
    assert out.is_symlink() and os.readlink(out) == str(tmp_path / "nowhere.drp")
    assert [p.name for p in tmp_path.iterdir()] == ["out.drp"]  # no label, no temporary file


def test_fe_a_name_too_long_for_the_label_temp_file_leaves_nothing(tmp_path: Path) -> None:
    limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    # The output's own temp name still fits, the label's (".<name>.aar.json.<8>.tmp.json") does not.
    out = tmp_path / ("a" * (limit - 27 - len(".drp")) + ".drp")
    with pytest.raises(OSError) as raised:
        edit_file(H264, plan(RENAME), out)
    assert raised.value.errno == errno.ENAMETOOLONG
    assert list(tmp_path.iterdir()) == []  # the output's temporary file was removed again


# --------------------------------------------------------------------------- F-D, F-J


def test_fd_stdout_closed_after_a_write_exits_74_with_the_file_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"edits": [RENAME]}))
    out = tmp_path / "out.drp"
    monkeypatch.setattr("sys.stdout", None)
    code = main(["drp", "edit", str(H264), "--plan", str(plan_file), "--out", str(out)])
    assert code == 74
    assert "stdout is closed" in capsys.readouterr().err
    label = json.loads(Path(str(out) + SIDECAR_SUFFIX).read_text())
    assert label["status"] == "unverified" and read_drp(out).project.name == "x"


CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"


def _edit_in_a_child(
    tmp_path: Path, **streams: Any
) -> tuple[subprocess.CompletedProcess[bytes], Path]:
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"edits": [RENAME]}))
    out = tmp_path / "out.drp"
    args = [sys.executable, "-c", CLI, "drp", "edit", str(H264), "--plan", str(plan_file)]
    proc = subprocess.run(
        [*args, "--out", str(out)], stderr=subprocess.PIPE, check=False, **streams
    )
    return proc, out


def _assert_edit_completed(out: Path) -> None:
    label = json.loads(Path(str(out) + SIDECAR_SUFFIX).read_text())
    assert label["status"] == "unverified" and label["output"]["size"] == out.stat().st_size
    assert read_drp(out).project.name == "x"
    assert stale_temp_files(out) == []


def test_fd_a_really_closed_stdout_exits_74_with_the_file_in_place(tmp_path: Path) -> None:
    proc, out = _edit_in_a_child(
        tmp_path, stdout=subprocess.DEVNULL, preexec_fn=lambda: os.close(1)
    )
    assert proc.returncode == 74, proc.stderr
    assert b"stdout is closed" in proc.stderr and b"Traceback" not in proc.stderr
    _assert_edit_completed(out)


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_fd_a_full_stdout_exits_74_with_the_file_in_place(tmp_path: Path) -> None:
    with open("/dev/full", "wb") as full:
        proc, out = _edit_in_a_child(tmp_path, stdout=full)
    assert proc.returncode == 74, proc.stderr
    assert b"cannot write output" in proc.stderr and b"Traceback" not in proc.stderr
    _assert_edit_completed(out)


def test_fd_a_closed_pipe_exits_141_quietly_with_the_file_in_place(tmp_path: Path) -> None:
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        proc, out = _edit_in_a_child(tmp_path, stdout=write_end)
    finally:
        os.close(write_end)
    assert proc.returncode == 141 and proc.stderr == b""
    _assert_edit_completed(out)


def test_fj_path_map_report_states_its_scope(tmp_path: Path) -> None:
    table = [{"windows": "E:", "mac": MAC_DIR}]
    outcome = edit_file(
        H264, plan({"op": "path_map", "direction": "mac_to_windows", "table": table}), dry_run=True
    )
    [report] = outcome.report
    assert "project settings" in report["scope"] and "GalleryPath" in report["scope"]


# --------------------------------------------------------------------------- test gaps


def _descriptor_zip() -> bytes:
    stream = UnseekableSink()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.xml", b"a" * 64)
        zf.writestr("b.xml", b"b" * 64)
    return bytes(stream.buf)


def _patched(data: bytes, at: int, fmt: str, value: int) -> bytes:
    out = bytearray(data)
    struct.pack_into(fmt, out, at, value)
    return bytes(out)


def test_gap_zip_data_descriptor_that_disagrees_is_refused() -> None:
    data = _descriptor_zip()
    first = zipraw.parse_layout(data)[0]
    assert first.flags & 0x8 and first.descriptor_sig
    crc_at = first.data_end + 4  # after the PK\x07\x08 signature
    bad = _patched(data, crc_at, "<I", 0x12345678)
    with pytest.raises(zipraw.ZipLayoutError, match="data descriptor disagrees"):
        zipraw.parse_layout(bad)


def test_gap_zip_data_descriptor_without_its_signature_is_kept_that_way() -> None:
    data = _descriptor_zip()
    first, second = zipraw.parse_layout(data)
    end = data.rfind(b"PK\x05\x06")
    cd_off = struct.unpack_from("<I", data, end + 16)[0]
    cut = data[: first.data_end] + data[first.data_end + 4 :]  # drop the optional signature
    cut = _patched(cut, second.central_start - 4 + 42, "<I", second.local_start - 4)
    cut = _patched(cut, end - 4 + 16, "<I", cd_off - 4)
    entry = zipraw.parse_layout(cut)[0]
    assert entry.flags & 0x8 and not entry.descriptor_sig
    assert zipraw.rewrite(cut, {}) == cut
    out = zipraw.rewrite(cut, {"a.xml": b"new"})
    assert not zipraw.parse_layout(out)[0].descriptor_sig
    with zipfile.ZipFile(io.BytesIO(out)) as zf:
        assert (zf.read("a.xml"), zf.read("b.xml")) == (b"new", b"b" * 64)


def test_gap_zip_local_flags_that_disagree_are_refused() -> None:
    data = _descriptor_zip()
    with pytest.raises(zipraw.ZipLayoutError, match="local and central headers disagree"):
        zipraw.parse_layout(_patched(data, 6, "<H", 0))  # local header: no descriptor flag


def _central_at(data: bytes, index: int) -> int:
    return zipraw.parse_layout(data)[index].central_start


def test_gap_zip_encrypted_members_are_refused() -> None:
    data = _descriptor_zip()
    flags = zipraw.parse_layout(data)[0].flags | 0x1
    both = _patched(_patched(data, 6, "<H", flags), _central_at(data, 0) + 8, "<H", flags)
    with pytest.raises(zipraw.ZipLayoutError, match="encrypted members"):
        zipraw.parse_layout(both)


def test_gap_zip_member_names_that_do_not_decode_are_refused() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("\u00e9.xml", b"x" * 64)  # a non-ASCII name sets the UTF-8 flag (0x800)
    data = bytearray(buf.getvalue())
    assert zipraw.parse_layout(bytes(data))[0].name == "\u00e9.xml"
    first = data.index(b"\xc3\xa9")
    data[first] = 0xFF  # the local header's name
    data[data.rindex(b"\xc3\xa9")] = 0xFF  # and the central record's: still consistent
    with pytest.raises(zipraw.ZipLayoutError, match="cannot be decoded"):
        zipraw.parse_layout(bytes(data))


@pytest.mark.parametrize(("field", "at"), [("crc", 0), ("size", 4), ("length", 8)])
def test_gap_zip_each_data_descriptor_field_is_checked(field: str, at: int) -> None:
    data = _descriptor_zip()
    first = zipraw.parse_layout(data)[0]
    value = struct.unpack_from("<I", data, first.data_end + 4 + at)[0]
    bad = _patched(data, first.data_end + 4 + at, "<I", value ^ 1)
    with pytest.raises(zipraw.ZipLayoutError, match="data descriptor disagrees"):
        zipraw.parse_layout(bad)


def test_gap_zip_a_data_descriptor_past_the_end_is_refused() -> None:
    data = _descriptor_zip()
    # The local header of a streamed member holds zeros, so only the central record claims a size.
    huge = _patched(data, _central_at(data, 1) + 20, "<I", len(data) + 1000)
    with pytest.raises(zipraw.ZipLayoutError, match="truncated data descriptor"):
        zipraw.parse_layout(huge)


@pytest.mark.parametrize(
    ("method", "number"), [(zipfile.ZIP_BZIP2, 12), (zipfile.ZIP_LZMA, 14)], ids=["bzip2", "lzma"]
)
def test_gap_zip_other_compression_methods_are_refused(method: int, number: int) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        zf.writestr("a.xml", b"a" * 64)
    with pytest.raises(zipraw.ZipLayoutError, match=f"compression method {number}"):
        zipraw.parse_layout(buf.getvalue())


def test_gap_sidecar_records_the_output_size_and_hash(tmp_path: Path) -> None:
    outcome, out = run(H264, tmp_path, RENAME)
    label = json.loads(Path(str(out) + SIDECAR_SUFFIX).read_text())
    data = out.read_bytes()
    assert label["output"]["size"] == len(data) == outcome.output["size"]
    assert label["output"]["sha256"] == hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- refusal paths
# Added after the S8b mutation pass: each test below fails when one specific check is removed.


def _orphan_label(tmp_path: Path) -> tuple[Path, Path]:
    out, sidecar, _stale = _interrupted_run(tmp_path)
    return out, sidecar


def test_fc_force_replaces_over_an_orphan_label(tmp_path: Path) -> None:
    out, sidecar = _orphan_label(tmp_path)
    outcome = edit_file(H264, plan(RENAME), out, force=True)
    assert outcome.status == "written" and not outcome.refusals
    label = json.loads(sidecar.read_text())
    assert label["output"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def _rewrite_label(sidecar: Path, change: Callable[[dict[str, Any]], None]) -> None:
    record = json.loads(sidecar.read_text())
    change(record)
    sidecar.write_text(json.dumps(record))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="refused"),
        lambda r: r.update(label="verified"),
        lambda r: r.update(tool="other 1"),
        lambda r: r["output"].update(path="/elsewhere/other.drp"),
        lambda r: r["output"].update(sidecar="/elsewhere/other.drp.aar.json"),
        lambda r: r.update(output=None),
    ],
    ids=["status", "label", "tool", "other-output", "other-sidecar", "no-output"],
)
def test_fc_a_label_that_is_not_ours_for_this_output_is_plain_out_exists(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    out, sidecar = _orphan_label(tmp_path)
    _rewrite_label(sidecar, change)
    before = sidecar.read_bytes()
    outcome = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in outcome.refusals] == ["out-exists"]
    assert sidecar.read_bytes() == before and not out.exists()


def test_fc_the_label_size_cap_is_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aar.drp.modify import write

    out, sidecar = _orphan_label(tmp_path)
    size = sidecar.stat().st_size
    monkeypatch.setattr(write, "_MAX_SIDECAR_BYTES", size)  # at the cap: still read and judged
    assert [r.code for r in edit_file(H264, plan(RENAME), out).refusals] == ["orphan-sidecar"]
    monkeypatch.setattr(write, "_MAX_SIDECAR_BYTES", size - 1)  # over it: never read
    assert [r.code for r in edit_file(H264, plan(RENAME), out).refusals] == ["out-exists"]


def test_fc_a_dangling_symlink_at_the_output_hides_an_orphan_label(tmp_path: Path) -> None:
    out, sidecar = _orphan_label(tmp_path)
    out.symlink_to(tmp_path / "nowhere.drp")  # out.exists() is False, yet a name is taken
    outcome = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in outcome.refusals] == ["out-exists"]
    assert out.is_symlink() and sidecar.exists()


def test_fe_a_mode_keeps_no_execute_bits(tmp_path: Path) -> None:
    src = tmp_path / "in.drp"
    src.write_bytes(H264.read_bytes())
    src.chmod(0o755)
    (tmp_path / "o").mkdir()
    out = tmp_path / "o" / "out.drp"
    assert edit_file(src, plan(RENAME), out).status == "written"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    assert (out.stat().st_mode & 0o777, sidecar.stat().st_mode & 0o777) == (0o644, 0o644)


def test_fe_a_missing_output_directory_is_a_refusal_not_a_crash(tmp_path: Path) -> None:
    out = tmp_path / "nowhere" / "out.drp"
    assert stale_temp_files(out) == []  # the scan of a directory that does not exist
    outcome = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in outcome.refusals] == ["no-output-dir"]
    assert list(tmp_path.iterdir()) == []


def test_fe_force_with_a_directory_at_the_label_leaves_no_temporary_file(tmp_path: Path) -> None:
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    sidecar.mkdir()
    (sidecar / "keep.txt").write_text("keep")
    with pytest.raises(IsADirectoryError):
        edit_file(H264, plan(RENAME), out, force=True)
    assert stale_temp_files(out) == []
    assert [p.name for p in tmp_path.iterdir()] == [sidecar.name]
    assert (sidecar / "keep.txt").read_text() == "keep"


def test_gap_zip_zero_local_sizes_need_the_descriptor_flag() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("a.xml", b"a" * 64)
    data = buf.getvalue()
    assert not zipraw.parse_layout(data)[0].flags & 0x8  # a seekable write: no descriptor
    zeroed = data[:14] + bytes(12) + data[26:]  # the local CRC, sizes zeroed; central intact
    with pytest.raises(zipraw.ZipLayoutError, match="local CRC or sizes disagree"):
        zipraw.parse_layout(zeroed)
    # The same zeros are fine once the member says a descriptor follows (what streaming writes).
    streamed = _descriptor_zip()
    assert streamed[14:26] == bytes(12) and zipraw.parse_layout(streamed)[0].flags & 0x8


def _cli_edit(tmp_path: Path, out: Path) -> list[str]:
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"edits": [RENAME]}))
    return ["drp", "edit", str(H264), "--plan", str(plan_file), "--out", str(out)]


def test_fe_cli_exits_74_when_the_label_cannot_be_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    (tmp_path / "o").mkdir()
    limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    out = tmp_path / "o" / ("a" * (limit - 27 - len(".drp")) + ".drp")  # as in the test above
    code = main(_cli_edit(tmp_path, out))
    captured = capsys.readouterr()
    assert code == 74
    result = json.loads(captured.out)
    assert result["status"] == "error"
    assert [r["code"] for r in result["refusals"]] == ["output-error"]
    assert "cannot write the output" in captured.err
    assert list((tmp_path / "o").iterdir()) == []


def test_fd_cli_internal_error_exits_2_and_is_never_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An internal error cannot be produced on purpose, so the engine is made to fail with an
    # exception no check expects; what is tested is how the command reports it.
    from aar.cli import main

    def explode(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("engine fell over")

    monkeypatch.setattr("aar.drp.modify.cli.edit_file", explode)
    code = main(_cli_edit(tmp_path, tmp_path / "out.drp"))
    captured = capsys.readouterr()
    assert code == 2  # 1 means "refused", which this is not
    result = json.loads(captured.out)
    assert result["status"] == "error"
    assert [r["code"] for r in result["refusals"]] == ["internal-error"]
    assert "RuntimeError: engine fell over" in result["refusals"][0]["message"]
    assert "Traceback" in captured.err and "internal error" in captured.err
    assert not (tmp_path / "out.drp").exists()


# --------------------------------------------------------------------------- DESIGN.md claims

DESIGN = (Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md").read_text(encoding="utf-8")


def _design_modify_section() -> str:
    start = DESIGN.index("- **Modify (S8).**")
    return DESIGN[start : DESIGN.index("- **Create (S9).**")]


def test_design_names_the_deferred_edits_the_plan_refuses() -> None:
    from aar.drp.modify.plan import DEFERRED_OPS

    section = _design_modify_section().lower()
    assert set(DEFERRED_OPS) == {"metadata", "fader_keyframes"}
    assert "cannot be verified" in section and "no keyframe writer exists" in section
    # Neither deferred edit is listed as built, and keyframes are not "ported from drp_tools".
    assert "ported from `drp_tools`" not in DESIGN
    assert "  - metadata\n" not in section and "  - fader keyframes" not in section
    assert "fader keyframes do not come from it" in DESIGN


def test_design_states_the_round_trip_allow_list_as_the_code_has_it() -> None:
    from aar.drp.modify.roundtrip import ROUND_TRIP_ALLOW_LIST

    ids = [rule.id for rule in ROUND_TRIP_ALLOW_LIST]
    assert sorted(ids) == sorted(
        [
            "bin-save-stamp",
            "db-saved-time",
            "last-modified",
            "playhead",
            "project-age",
            "ptzr-preset-dbid",
            "settings-save-date",
        ]
    )
    section = _design_modify_section()
    assert "exactly **11 opaque lines**" in section
    assert "root bin's fields blob" in section and "3 playhead values" in section
    assert "`LastModTimeInSecs`" in section
    # The counts come from corpus pair 002 -> 003; test_drp_modify_corpus pins them.


def test_design_states_the_input_has_warnings_exclusion_and_the_licence_status() -> None:
    assert "16 files, all `blob-partial`" in _design_modify_section()
    assert "the other 48 can be edited" in _design_modify_section()
    prior = DESIGN[DESIGN.index("- **Prior art.**") :].split("\n- ")[0]
    assert "`samuelgursky/drp-format` returns 404" in prior and "unconfirmed" in prior
    assert "(MIT, JavaScript)" not in prior and "keyframe writer, zstd" not in prior
