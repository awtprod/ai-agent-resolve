"""Tests for the hex-blob codecs in ``aar.drp.blobs``."""

from __future__ import annotations

import struct
import zlib

import pytest
from drp_helpers import envelope, kd, pb_len, pb_var, qstr, varint, zstd_raw_frame

from aar.drp import blobs
from aar.drp.errors import BlobError, ZstdUnavailableError

# Values copied from the bundled media-clip-h264.drp template.
TIMELINE_FRAME_RATE_24 = "00000000000038400000000000000000"
MEDIA_FRAME_RATE_2997 = "286b55e253f83d400000000000000000"
RESOLUTION_1080P = "00000000000007800000000000000438"
EXTENTS = "000000000020ac405055555555156340"
TIMEMAP_IDENTITY = (
    "02406314e147ae147b00000000000000004063149d495182aa0000000000000000406314e147ae147b"
)
TRACK_FIELDS = "000000010000000100000012004e0075006d004c00610079006500720073000000020000000000"


@pytest.fixture
def no_zstd_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend neither zstandard nor compression.zstd is installed."""
    monkeypatch.setattr(blobs, "_zstd_backend", lambda: (None, None))


def test_parse_hex_rejects_odd_length_and_non_hex() -> None:
    assert blobs.parse_hex(" 00ff ") == b"\x00\xff"
    with pytest.raises(BlobError, match="odd hex length 3"):
        blobs.parse_hex("abc")
    with pytest.raises(BlobError, match="non-hex"):
        blobs.parse_hex("zz")


def test_parse_frames_plain_and_with_subframe_remainder() -> None:
    assert blobs.parse_frames("86400") == (86400, None)
    frames, fraction = blobs.parse_frames("108243|00002085eb51e83f")
    assert frames == 108243
    assert fraction == pytest.approx(0.76, abs=1e-3)
    with pytest.raises(BlobError, match="outside"):
        blobs.parse_frames("1|000000000000f03f")  # 1.0 is not a remainder
    with pytest.raises(BlobError, match="not a frame count"):
        blobs.parse_frames("abc")


def test_fixed_size_values_from_template() -> None:
    assert blobs.decode_rate(bytes.fromhex(TIMELINE_FRAME_RATE_24)) == 24.0
    assert blobs.decode_rate(bytes.fromhex(MEDIA_FRAME_RATE_2997)) == pytest.approx(30000 / 1001)
    assert blobs.decode_resolution(bytes.fromhex(RESOLUTION_1080P)) == (1920, 1080)
    start, duration = blobs.decode_extents(bytes.fromhex(EXTENTS))
    assert start == 3600.0
    assert duration * 24 == pytest.approx(3664)
    with pytest.raises(BlobError, match="16 bytes"):
        blobs.decode_rate(bytes(8))
    with pytest.raises(BlobError, match="padding"):
        blobs.decode_resolution(bytes.fromhex("00000001000007800000000000000438"))


def test_keyed_dict_from_template_track_fields() -> None:
    d = blobs.decode_keyed_dict(bytes.fromhex(TRACK_FIELDS), strict=True)
    assert d.complete
    assert [(e.key, e.type, e.value) for e in d.entries] == [("NumLayers", 2, 0)]


def test_keyed_dict_all_value_types() -> None:
    data = kd(
        [
            ("s", 0x0A, qstr("hé")),
            ("b", 0x0C, struct.pack(">I", 2) + b"\x01\x02"),
            ("null", 0x0C, b"\xff\xff\xff\xff"),
            ("t", 0x01, b"\x01"),
            ("i", 0x02, struct.pack(">i", -5)),
            ("u", 0x03, struct.pack(">I", 44100)),
            ("l", 0x04, struct.pack(">q", -(2**40))),
            ("ul", 0x05, struct.pack(">Q", 2**63)),
            ("d", 0x06, struct.pack(">d", 2.5)),
            ("f", 0x87, struct.pack(">d", 1.0)),
            ("c", 0x07, struct.pack(">H", ord("x"))),
            ("date", 0x10, bytes(range(9))),
            (
                "list",
                0x09,
                struct.pack(">I", 2)
                + struct.pack(">IBi", 2, 0, 7)
                + struct.pack(">IB", 10, 0)
                + qstr("a"),
            ),
            ("strs", 0x0B, struct.pack(">I", 2) + qstr("x") + qstr("y")),
            ("map", 0x08, struct.pack(">I", 1) + qstr("k") + struct.pack(">IBi", 2, 0, 9)),
        ]
    )
    d = blobs.decode_keyed_dict(data, strict=True)
    got = {e.key: e.value for e in d.entries}
    assert got == {
        "s": "hé",
        "b": b"\x01\x02",
        "null": None,
        "t": True,
        "i": -5,
        "u": 44100,
        "l": -(2**40),
        "ul": 2**63,
        "d": 2.5,
        "f": 1.0,
        "c": "x",
        "date": bytes(range(9)),
        "list": [7, "a"],
        "strs": ["x", "y"],
        "map": {"k": 9},
    }
    assert d.entries[0].offset == 8
    assert d.entries[-1].end == len(data)


def test_keyed_dict_unknown_type_keeps_prefix_and_reports() -> None:
    data = kd([("ok", 0x02, struct.pack(">i", 1)), ("odd", 0x55, b"\x00" * 4)])
    d = blobs.decode_keyed_dict(data)
    assert [e.key for e in d.entries] == ["ok"]
    assert d.error is not None and "0x55" in d.error
    assert not d.complete
    assert d.consumed == d.entries[0].end
    with pytest.raises(BlobError, match="0x55"):
        blobs.decode_keyed_dict(data, strict=True)


def test_keyed_dict_truncated_byte_array() -> None:
    data = kd([("blob", 0x0C, struct.pack(">I", 100) + b"\x00" * 10)])
    with pytest.raises(BlobError, match="runs past the end"):
        blobs.decode_keyed_dict(data, strict=True)


def test_protobuf_wire_types_and_nesting() -> None:
    inner = pb_var(1, 300) + pb_len(2, b"hello")
    data = (
        pb_var(1, 150)
        + varint(2 << 3 | 1)
        + struct.pack("<d", 1.5)
        + varint(3 << 3 | 5)
        + struct.pack("<f", 0.5)
        + pb_len(4, inner)
    )
    fields = blobs.decode_protobuf(data)
    assert [(f.number, f.wire) for f in fields] == [(1, 0), (2, 1), (3, 5), (4, 2)]
    assert fields[0].value == 150
    assert fields[1].as_double == 1.5
    assert fields[2].as_float == 0.5
    assert fields[3].children is not None
    assert fields[3].children[0].value == 300
    assert fields[3].children[1].as_text == "hello"
    # Offsets of nested fields are absolute within the outer buffer.
    nested = fields[3].children[1]
    assert data[nested.offset : nested.end] == pb_len(2, b"hello")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (pb_len(1, b"abc")[:-1], "runs past the end"),
        (varint(1 << 3 | 3), "unsupported wire type 3"),
        (b"\x00\x01", "field number 0"),
        (b"\x08\xff", "truncated varint"),
    ],
)
def test_protobuf_rejects_malformed(data: bytes, message: str) -> None:
    with pytest.raises(BlobError, match=message):
        blobs.decode_protobuf(data)


def test_envelope_protobuf_and_flag_only_forms() -> None:
    payload = pb_var(1, 7)
    env = blobs.parse_envelope(envelope(2, 0x80, payload))
    assert env is not None
    assert (env.version, env.codec, env.payload) == (2, "protobuf", payload)
    flagged = blobs.parse_envelope(b"\x80" + payload)
    assert flagged is not None and flagged.version is None and flagged.codec == "protobuf"
    plain = blobs.parse_envelope(struct.pack(">II", 2, len(payload)) + payload)
    assert plain is not None and plain.codec == "none" and plain.payload == payload
    assert blobs.parse_envelope(b"\x00\x00\x00\x02\x00\x00\x00\x09\x80") is None


def test_raw_block_zstd_decodes_without_a_library(no_zstd_library: None) -> None:
    payload = pb_len(2, pb_len(1, pb_var(1, 42)))
    data = envelope(10001, 0x81, zstd_raw_frame(payload))
    assert blobs.zstd_backend() is None
    assert blobs.unwrap_message(data) == payload


def test_compressed_zstd_without_a_library_says_what_to_install(no_zstd_library: None) -> None:
    zstandard = pytest.importorskip("zstandard")
    frame = zstandard.ZstdCompressor().compress(b"resolve " * 64)
    with pytest.raises(ZstdUnavailableError, match="zstandard"):
        blobs.zstd_decompress(frame)
    result = blobs.check("FieldsBlob", envelope(2, 0x81, frame))
    assert not result.ok
    assert result.error is not None and result.error.startswith("zstd-unavailable")


def test_compressed_zstd_with_library_and_output_bound() -> None:
    zstandard = pytest.importorskip("zstandard")
    frame = zstandard.ZstdCompressor().compress(b"x" * 5000)
    assert blobs.zstd_decompress(frame) == b"x" * 5000
    with pytest.raises(BlobError, match="limit"):
        blobs.zstd_decompress(frame, max_output=100)


def test_zstd_output_bound_when_size_is_not_declared() -> None:
    zstandard = pytest.importorskip("zstandard")
    cctx = zstandard.ZstdCompressor(write_content_size=False)
    frame = cctx.compress(b"y" * 5000)
    with pytest.raises(BlobError, match="exceeds"):
        blobs.zstd_decompress(frame, max_output=100)


def test_qcompress_roundtrip_and_checks() -> None:
    raw = b"AdrDatabase" * 50
    data = struct.pack(">I", len(raw)) + zlib.compress(raw)
    assert blobs.classify("FLStudioModelBA", data) == blobs.BlobKind.QCOMPRESS
    assert blobs.qcompress_decompress(data) == raw
    with pytest.raises(BlobError, match="mismatch"):
        blobs.qcompress_decompress(struct.pack(">I", 3) + zlib.compress(raw))
    with pytest.raises(BlobError, match="limit"):
        blobs.qcompress_decompress(data, max_output=10)


def test_timemap_identity_from_template_and_retimed_keyed_form() -> None:
    tm = blobs.decode_timemap(bytes.fromhex(TIMEMAP_IDENTITY))
    assert tm.form == "identity" and tm.speed == 1.0
    assert tm.seconds[0] == pytest.approx(152.6525)
    assert len(tm.seconds) == 5
    retimed = kd(
        [
            ("XMax", 0x06, struct.pack(">d", 10.0)),
            ("LastValidYOffset", 0x06, struct.pack(">d", 5.0)),
            ("DbType", 0x0A, qstr("Sm2TimeMap")),
        ]
    )
    tm = blobs.decode_timemap(retimed)
    assert (tm.form, tm.speed) == ("retimed", 0.5)
    with pytest.raises(BlobError, match="unexpected length"):
        blobs.decode_timemap(b"\x02\x00\x00")


def test_classify_shapes() -> None:
    c = blobs.classify
    assert c("FieldsBlob", b"") == blobs.BlobKind.EMPTY
    assert c("FrameRate", bytes.fromhex(TIMELINE_FRAME_RATE_24)) == blobs.BlobKind.FIXED16
    assert c("FieldsBlob", bytes.fromhex(TRACK_FIELDS)) == blobs.BlobKind.KEYED_DICT
    assert c("Clip", envelope(2, 0x81, zstd_raw_frame(b"\x08\x01"))) == blobs.BlobKind.ENVELOPE_ZSTD
    assert c("EffectFiltersBA", envelope(2, 0x80, b"\x08\x01")) == blobs.BlobKind.ENVELOPE_PROTOBUF
    assert c("Radiometry", b"\x80\x08\x01") == blobs.BlobKind.FLAGGED_PROTOBUF
    assert c("Body", b"\x81" + zstd_raw_frame(b"\x08\x01")) == blobs.BlobKind.FLAGGED_ZSTD
    assert c("MediaTimemapBA", bytes.fromhex(TIMEMAP_IDENTITY)) == blobs.BlobKind.TIMEMAP_COMPACT
    assert c("Buffer", b"\xff\xd8\xff\xe0rest") == blobs.BlobKind.JPEG
    assert c("Presets", b"{}") == blobs.BlobKind.TEXT
    assert c("SomethingBA", b"\x13\x37\x00\xfe") == blobs.BlobKind.UNKNOWN
    assert blobs.is_blob_tag("MarkersBA") and blobs.is_blob_tag("Clip")
    assert not blobs.is_blob_tag("DbSavedTime")


def test_check_reports_nested_failures_with_the_key() -> None:
    bad_inner = envelope(1, 0x80, b"\x08")  # truncated varint inside the message
    data = kd([("SetupBA", 0x0C, struct.pack(">I", len(bad_inner)) + bad_inner)])
    result = blobs.check("FieldsBlob", data)
    assert not result.ok
    assert result.error is not None and result.error.startswith("SetupBA:")
    good_inner = envelope(1, 0x80, pb_var(1, 1))
    data = kd([("SetupBA", 0x0C, struct.pack(">I", len(good_inner)) + good_inner)])
    assert blobs.check("FieldsBlob", data).ok
    undecoded = blobs.check("PreConformMediaExtents", bytes(16))
    assert undecoded.ok and undecoded.undecoded
