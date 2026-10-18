"""S8b: follow-ups from the S8 validation (findings F-A to F-J and the listed test gaps).

Each test names the finding it pins. They run on the bundled templates only.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from diff_testing import CLIP, H264, POOL, SEQUENCE, TIMELINE
from drp_edit import add_carrier, carrier, edit_zip, marker_entry
from drp_helpers import kd, needs_zstandard, pb_len, zstd_raw_frame

from aar.drp import blobs
from aar.drp.modify import compare_round_trip, encode, parse_plan
from aar.drp.modify.encode import EncodeError
from aar.drp.modify.write import SIDECAR_SUFFIX, edit_file
from aar.drp.reader import read_drp

MAC_DIR = "/Users/samuelgursky/Desktop"
BLUE = 2


def plan(*edits: dict[str, Any], **top: Any) -> Any:
    return parse_plan(json.dumps({"edits": list(edits), **top}))


def run(src: Path, tmp_path: Path, *edits: dict[str, Any], **kw: Any) -> Any:
    out = tmp_path / "out.drp"
    return edit_file(src, plan(*edits), out, **kw), out


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


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
        (b"\x50\x2a\x4d\x18" + b"\x00" * 8, "not a zstd frame"),  # a skippable frame
        (zstd_raw_frame(b"abc")[:4] + bytes([0x21]) + zstd_raw_frame(b"abc")[5:], "dictionary"),
        (zstd_raw_frame(b"abc")[:6] + (7 << 1 | 1).to_bytes(3, "little"), "reserved"),
    ],
    ids=["byte-after", "second-frame", "truncated", "skippable", "dictionary", "reserved-block"],
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


@needs_zstandard
@pytest.mark.parametrize(
    "change",
    [_header_word_2, _setup_null_flag, _pad_field_8],
    ids=["kd-header-word", "setupba-null-flag", "setupba-padded-varint"],
)
def test_fg_settings_rule_covers_the_whole_blob(tmp_path: Path, change: Any) -> None:
    ok = edit_zip(H264, tmp_path / "ok.drp", {"project.xml": _settings_edit(_new_save_date)})
    assert compare_round_trip(H264, ok).ok
    bad = edit_zip(H264, tmp_path / "bad.drp", {"project.xml": _settings_edit(change)})
    result = compare_round_trip(H264, bad)
    assert [r.code for r in result.refusals] == ["unexplained-opaque"]
    assert "settings-save-date" not in [rule for rule, _ in result.allowed]


def test_fg_bin_rule_compares_field_bytes(tmp_path: Path) -> None:
    def blob(stamp: int, field_5: bytes = encode.pb_varint(5, 1)) -> bytes:
        payload = encode.pb_varint(4, stamp) + field_5
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
        assert sidecar.read_text() == text


def test_fc_leftover_temp_files_are_reported_not_deleted(tmp_path: Path) -> None:
    stale = [
        tmp_path / ".out.drp.abcd1234.tmp.drp",
        tmp_path / ".out.drp.aar.json.zz_9yy1q.tmp.json",
    ]
    unrelated = [tmp_path / ".out.drp.notours.txt", tmp_path / ".other.drp.abcd1234.tmp.drp"]
    for p in stale + unrelated:
        p.write_bytes(b"x")
    outcome, _ = run(H264, tmp_path, RENAME)
    assert outcome.status == "written"
    assert outcome.to_dict()["leftovers"] == sorted(str(p) for p in stale)
    assert all(p.exists() for p in stale + unrelated)
