"""Encoders and the raw zip rewriter behind ``aar drp edit`` (S8).

Every encoder is checked against the decoder in :mod:`aar.drp.blobs` it inverts, and the zip
rewriter against :mod:`zipfile` and against the input bytes it must leave alone.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import Path

import pytest
from drp_helpers import kd, needs_zstandard, pb_len, pb_var, qstr, zstd_raw_frame

from aar.drp import blobs
from aar.drp.modify import encode, zipraw
from aar.drp.modify.encode import EncodeError

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"
ALL = sorted(FIXTURES.glob("*.drp"))


# --------------------------------------------------------------------------- protobuf


@pytest.mark.parametrize("n", [0, 1, 127, 128, 300, 2**32, 2**63, 2**64 - 1])
def test_varint_round_trips_through_the_decoder(n: int) -> None:
    [f] = blobs.decode_protobuf(encode.pb_varint(5, n), nested_depth=0)
    assert (f.number, f.wire, f.value) == (5, 0, n)


@pytest.mark.parametrize("n", [-1, 2**64])
def test_varint_refuses_out_of_range(n: int) -> None:
    with pytest.raises(EncodeError):
        encode.varint(n)


def test_pb_key_refuses_bad_numbers_and_wires() -> None:
    for number, wire in ((0, 2), (1 << 29, 2), (1, 3), (1, 4)):
        with pytest.raises(EncodeError):
            encode.pb_key(number, wire)


def test_replace_field_changes_only_that_field() -> None:
    payload = pb_var(1, 7) + pb_len(2, b"old") + pb_len(3, b"tail")
    fields = blobs.decode_protobuf(payload, nested_depth=0)
    out = encode.replace_field(payload, fields[1], b"a longer value")
    assert out == pb_var(1, 7) + pb_len(2, b"a longer value") + pb_len(3, b"tail")
    out = encode.replace_field(payload, fields[0], 300)
    assert out == pb_var(1, 300) + pb_len(2, b"old") + pb_len(3, b"tail")


def test_replace_field_refuses_non_canonical_and_wrong_types() -> None:
    padded = b"\x08\x81\x00" + pb_len(2, b"x")  # field 1 = 1 written as a padded varint
    fields = blobs.decode_protobuf(padded, nested_depth=0)
    assert not encode.canonical(fields[0], padded)
    with pytest.raises(EncodeError, match="not canonically"):
        encode.replace_field(padded, fields[0], 2)
    with pytest.raises(EncodeError, match="cannot write"):
        encode.replace_field(padded, fields[1], 5)
    long_len = b"\x12\x81\x00x"  # field 2, length 1 written in two bytes
    [f] = blobs.decode_protobuf(long_len, nested_depth=0)
    with pytest.raises(EncodeError):
        encode.replace_field(long_len, f, b"y")


def test_canonical_fixed_width_fields() -> None:
    data = encode.pb_key(4, 1) + b"\x00" * 8 + encode.pb_key(5, 5) + b"\x01\x02\x03\x04"
    for f in blobs.decode_protobuf(data, nested_depth=0):
        assert encode.canonical(f, data)


# --------------------------------------------------------------------------- zstd and envelopes


@pytest.mark.parametrize("size", [0, 1, 255, 256, 65791, 65792, 131072, 131073, 300_000])
def test_zstd_raw_frame_decodes_without_a_library(size: int) -> None:
    payload = bytes((i * 7) & 0xFF for i in range(size))
    frame = encode.zstd_raw_frame(payload)
    assert blobs._zstd_raw_only(frame, len(payload) + 1) == payload
    _, declared = blobs._frame_header(frame)
    assert declared == size


@needs_zstandard
@pytest.mark.parametrize("size", [0, 5, 256, 70000, 300_000])
def test_zstd_raw_frame_decodes_with_zstandard(size: int) -> None:
    payload = bytes((i * 13) & 0xFF for i in range(size))
    assert blobs.zstd_decompress(encode.zstd_raw_frame(payload)) == payload


def test_zstd_raw_frame_matches_resolve_small_frames() -> None:
    """Resolve writes a single raw block with a one-byte content size for small payloads."""
    for payload in (b"", b"x", bytes(range(200))):
        assert encode.zstd_raw_frame(payload) == zstd_raw_frame(payload)


def test_envelope_keeps_version_and_codec() -> None:
    body = pb_len(1, b"/media") + pb_len(2, b"a.mov")
    plain = struct.pack(">II", 7, len(body) + 1) + b"\x80" + body
    assert encode.envelope(plain, body) == plain
    z = struct.pack(">II", 2, 1 + len(zstd_raw_frame(body))) + b"\x81" + zstd_raw_frame(body)
    out = encode.envelope(z, body + b"\x18\x01")
    env = blobs.parse_envelope(out)
    assert env is not None and (env.version, env.codec) == (2, "zstd")
    assert blobs.unwrap_message(out) == body + b"\x18\x01"
    none = struct.pack(">II", 2, len(body)) + body
    assert encode.envelope(none, body) == none
    flagged = b"\x80" + body
    assert encode.envelope(flagged, body + b"\x18\x02") == b"\x80" + body + b"\x18\x02"
    with pytest.raises(EncodeError):
        encode.envelope(b"", body)


def test_zstd_raw_frame_refuses_huge_payloads() -> None:
    class Huge(bytes):
        def __len__(self) -> int:
            return 2**32

    with pytest.raises(EncodeError):
        encode.zstd_raw_frame(Huge(b""))


# --------------------------------------------------------------------------- keyed dicts, hex, text


def test_replace_kd_bytes_rewrites_only_one_value() -> None:
    blob = kd(
        [
            ("A", 0x0A, qstr("keep")),
            ("BlobData", 0x0C, struct.pack(">I", 3) + b"abc"),
            ("Z", 0x02, struct.pack(">i", 5)),
        ]
    )
    dec = blobs.decode_keyed_dict(blob, strict=True)
    entry = dec.entry("BlobData")
    assert entry is not None
    # An entry decoded from another blob does not fit this one, and is refused.
    short = kd([("BlobData", 0x0C, struct.pack(">I", 3) + b"abc")])
    longer = kd([("BlobData", 0x0C, struct.pack(">I", 5) + b"abcde")])
    short_entry = blobs.decode_keyed_dict(short, strict=True).entry("BlobData")
    assert short_entry is not None
    with pytest.raises(EncodeError, match="layout"):
        encode.replace_kd_bytes(longer, short_entry, b"x")
    with pytest.raises(EncodeError, match="not inside"):
        encode.replace_kd_bytes(b"\0" * 8, entry, b"x")
    out = encode.replace_kd_bytes(blob, entry, b"0123456789")
    again = blobs.decode_keyed_dict(out, strict=True)
    assert [(e.key, e.value) for e in again.entries] == [
        ("A", "keep"),
        ("BlobData", b"0123456789"),
        ("Z", 5),
    ]
    assert encode.replace_kd_bytes(blob, entry, b"abc") == blob
    a_entry = dec.entry("A")
    assert a_entry is not None
    with pytest.raises(EncodeError):
        encode.replace_kd_bytes(blob, a_entry, b"x")


def test_hex_like_keeps_case_and_whitespace() -> None:
    assert encode.hex_like(b"00ab", b"\x01\xcd") == b"01cd"
    assert encode.hex_like(b"00AB", b"\x01\xcd") == b"01CD"
    assert encode.hex_like(b"\n  00ab \n", b"\xff") == b"\n  ff \n"
    assert encode.hex_like(b"0011", b"\xab") == b"ab"  # no letters to follow: lowercase
    with pytest.raises(EncodeError, match="mixes"):
        encode.hex_like(b"aB", b"\x00")


def test_xml_text_escapes_what_resolve_escapes() -> None:
    assert encode.xml_text("A & B <c> 'q' \"d\" é") == "A &amp; B &lt;c&gt; 'q' \"d\" é".encode()


@pytest.mark.parametrize(
    "bad", ["", "a\x00b", "tab\there", "line\nbreak", "del\x7f", "nc " + chr(0xFFFE)]
)
def test_check_text_refuses_what_resolve_never_writes(bad: str) -> None:
    with pytest.raises(EncodeError):
        encode.check_text(bad, "name", max_chars=100)


def test_check_text_limits() -> None:
    encode.check_text("x" * 100, "name", max_chars=100)
    with pytest.raises(EncodeError, match="longer"):
        encode.check_text("x" * 101, "name", max_chars=100)
    with pytest.raises(EncodeError, match="Unicode"):
        encode.check_text("lone \ud800 surrogate", "name", max_chars=100)


# --------------------------------------------------------------------------- zip rewriter


@pytest.mark.parametrize("path", ALL, ids=[p.name for p in ALL])
def test_rewrite_without_replacements_is_the_identity(path: Path) -> None:
    data = path.read_bytes()
    assert zipraw.rewrite(data, {}) == data


@pytest.mark.parametrize("path", ALL, ids=[p.name for p in ALL])
def test_rewrite_replaces_one_member_and_copies_the_rest(path: Path) -> None:
    data = path.read_bytes()
    entries = zipraw.parse_layout(data)
    target = "project.xml"
    new = b"<?xml?>\n<SM_Project/>\n" * 50
    out = zipraw.rewrite(data, {target: new})
    with zipfile.ZipFile(path) as za, zipfile.ZipFile(_bytes_file(out)) as zb:
        assert zb.testzip() is None
        assert za.namelist() == zb.namelist()
        for name in za.namelist():
            assert zb.read(name) == (new if name == target else za.read(name))
    after = {e.name: e for e in zipraw.parse_layout(out)}
    for e in entries:
        if e.name == target:
            continue
        copy = after[e.name]
        assert out[copy.local_start : copy.local_end] == data[e.local_start : e.local_end]
        a_rec = bytearray(data[e.central_start : e.central_end])
        b_rec = bytearray(out[copy.central_start : copy.central_end])
        a_rec[42:46] = b_rec[42:46] = b"\0\0\0\0"  # only the offset may differ
        assert a_rec == b_rec


def _bytes_file(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)


def _zip(members: list[tuple[str, bytes]], **kw: object) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", kw.get("method", zipfile.ZIP_DEFLATED)) as zf:  # type: ignore[arg-type]
        for name, data in members:
            zf.writestr(name, data)
        if "comment" in kw:
            zf.comment = kw["comment"]  # type: ignore[assignment]
    return buf.getvalue()


def test_rewrite_stored_members_and_comments() -> None:
    data = _zip([("a.xml", b"aaa"), ("b.xml", b"bbb")], method=zipfile.ZIP_STORED, comment=b"hi")
    out = zipraw.rewrite(data, {"a.xml": b"replaced"})
    with zipfile.ZipFile(_bytes_file(out)) as zf:
        assert zf.read("a.xml") == b"replaced"
        assert zf.read("b.xml") == b"bbb"
        assert zf.comment == b"hi"
        assert zf.getinfo("a.xml").compress_type == zipfile.ZIP_STORED


class _Unseekable(io.RawIOBase):
    """A write-only stream that cannot seek, so zipfile writes data descriptors (as Resolve)."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b: object) -> int:
        data = bytes(b)  # type: ignore[call-overload]
        self.buf += data
        return len(data)


def test_rewrite_keeps_the_data_descriptor_form() -> None:
    stream = _Unseekable()
    with zipfile.ZipFile(stream, "w") as zf:
        zf.writestr("a.xml", b"streamed content" * 20)
        zf.writestr("b.xml", b"other")
    data = bytes(stream.buf)
    entry = zipraw.parse_layout(data)[0]
    assert entry.flags & 0x8 and entry.descriptor_sig
    assert zipraw.rewrite(data, {}) == data
    out = zipraw.rewrite(data, {"a.xml": b"new"})
    after = zipraw.parse_layout(out)[0]
    assert after.flags & 0x8 and after.descriptor_sig
    with zipfile.ZipFile(_bytes_file(out)) as zf:
        assert zf.read("a.xml") == b"new"
        assert zf.read("b.xml") == b"other"
    crc, _csize, usize = struct.unpack_from("<III", out, after.local_end - 12)
    assert (crc, usize) == (zlib.crc32(b"new"), 3)


def test_rewrite_refuses_an_unknown_member() -> None:
    with pytest.raises(zipraw.ZipLayoutError, match="no member"):
        zipraw.rewrite(_zip([("a.xml", b"a")]), {"b.xml": b"b"})


def _patch(data: bytes, at: int, fmt: str, value: int) -> bytes:
    out = bytearray(data)
    struct.pack_into(fmt, out, at, value)
    return bytes(out)


def test_layout_refusals() -> None:
    good = _zip([("a.xml", b"a" * 50), ("b.xml", b"b" * 50)])
    end = good.rfind(b"PK\x05\x06")
    cd_off = struct.unpack_from("<I", good, end + 16)[0]
    cases = {
        "prefix bytes": b"JUNK" + good,
        "trailing bytes": good + b"x",
        "no end record": good[:end],
        "zip64 count": _patch(good, end + 10, "<H", 0xFFFF),
        "multi-disk": _patch(good, end + 4, "<H", 1),
        "bad cd signature": _patch(good, cd_off, "<I", 0x12345678),
        "encrypted": _patch(good, cd_off + 8, "<H", 1),
        "method": _patch(good, cd_off + 10, "<H", 12),
        "cd size": _patch(good, end + 12, "<I", end - cd_off - 1),
    }
    for why, data in cases.items():
        with pytest.raises(zipraw.ZipLayoutError):
            zipraw.parse_layout(data)
        assert why


def test_layout_refuses_duplicate_names_and_gaps() -> None:
    import warnings

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.xml", b"1")
            zf.writestr("a.xml", b"2")
    with pytest.raises(zipraw.ZipLayoutError, match="duplicate"):
        zipraw.parse_layout(buf.getvalue())
    good = _zip([("a.xml", b"a" * 50)])
    end = good.rfind(b"PK\x05\x06")
    cd_off = struct.unpack_from("<I", good, end + 16)[0]
    gap = good[:cd_off] + b"GAP!" + good[cd_off:]
    gap = _patch(gap, end + 4 + 16, "<I", cd_off + 4)
    with pytest.raises(zipraw.ZipLayoutError, match="between"):
        zipraw.parse_layout(gap)


def test_layout_refuses_a_gap_between_members() -> None:
    good = _zip([("a.xml", b"a" * 50), ("b.xml", b"b" * 50)])
    first, second = zipraw.parse_layout(good)
    end = good.rfind(b"PK\x05\x06")
    cd_off = struct.unpack_from("<I", good, end + 16)[0]
    junk = b"JUNK"
    data = good[: first.local_end] + junk + good[first.local_end :]
    shift = len(junk)
    data = _patch(data, second.central_start + shift + 42, "<I", second.local_start + shift)
    data = _patch(data, end + shift + 16, "<I", cd_off + shift)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:  # zipfile itself reads past the junk
        assert zf.read("b.xml") == b"b" * 50
    with pytest.raises(zipraw.ZipLayoutError, match="gap or overlap"):
        zipraw.parse_layout(data)


def test_layout_refuses_local_sizes_that_disagree() -> None:
    good = _zip([("a.xml", b"a" * 50)])
    with pytest.raises(zipraw.ZipLayoutError, match="CRC or sizes"):
        zipraw.parse_layout(_patch(good, 14, "<I", 0x12345678))


@pytest.mark.parametrize("path", ALL, ids=[p.name for p in ALL])
def test_rewrite_updates_the_local_header_of_a_replaced_member(path: Path) -> None:
    out = zipraw.rewrite(path.read_bytes(), {"project.xml": b"<x/>"})
    entry = next(e for e in zipraw.parse_layout(out) if e.name == "project.xml")
    crc, csize, usize = struct.unpack_from("<III", out, entry.local_start + 14)
    assert (crc, usize) == (zlib.crc32(b"<x/>"), 4)
    assert csize == entry.data_end - entry.data_start


def test_layout_refuses_mismatched_local_headers() -> None:
    good = _zip([("a.xml", b"a" * 50)])
    with pytest.raises(zipraw.ZipLayoutError, match="signature"):
        zipraw.parse_layout(_patch(good, 0, "<I", 0))
    with pytest.raises(zipraw.ZipLayoutError, match="disagree"):
        zipraw.parse_layout(_patch(good, 8, "<H", 0))  # local method stored, central deflate
    renamed = bytearray(good)
    renamed[30] = ord("b")
    with pytest.raises(zipraw.ZipLayoutError, match="names differ"):
        zipraw.parse_layout(bytes(renamed))
