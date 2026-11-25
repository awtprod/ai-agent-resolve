"""Encoders for the blob layouts :mod:`aar.drp.blobs` decodes, used to re-encode edited blobs.

Each encoder is the inverse of a decoder in :mod:`aar.drp.blobs` and is checked against it in the
tests (decode -> encode is the identity on every corpus blob of that kind):

* protobuf: :func:`varint`, :func:`pb_key`, :func:`pb_len`, :func:`replace_field`
* message envelope ``[u32 version][u32 length][codec][payload]``: :func:`envelope`
* zstd frames made of raw blocks only: :func:`zstd_raw_frame`; an original payload must be
  exactly one frame (:func:`check_single_frame`). Resolve itself writes raw-block
  frames for small payloads; any conforming zstd decoder reads them. Writing raw blocks keeps the
  writer stdlib-only and its output deterministic (no compressor version in the loop).
* keyed dict byte-array values: :func:`replace_kd_bytes`; whole keyed dicts: :func:`keyed_dict`
* whole protobuf messages from a field list: :func:`protobuf`
* hex text: :func:`hex_like` keeps the letter case of the text it replaces
* XML character data: :func:`xml_text`

Every function either returns bytes or raises :class:`EncodeError`; nothing is guessed.
"""

from __future__ import annotations

import re
import struct

from aar.drp import blobs
from aar.drp.errors import BlobError, DrpError

__all__ = [
    "EncodeError",
    "check_envelope_payload",
    "check_single_frame",
    "check_text",
    "envelope",
    "hex_like",
    "keyed_dict",
    "pb_key",
    "pb_len",
    "pb_varint",
    "protobuf",
    "replace_field",
    "replace_kd_bytes",
    "varint",
    "xml_text",
    "zstd_raw_frame",
]

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_RAW_BLOCK_MAX = 128 * 1024  # zstd Block_Maximum_Size
_U32_MAX = 0xFFFFFFFF
# C0 controls, DEL and the two non-characters XML 1.0 forbids; tab/LF/CR are refused too because
# Resolve's names and paths never contain them and the reader's text() keeps them verbatim.
_BAD_TEXT = re.compile("[\x00-\x1f\x7f￾￿]")


class EncodeError(DrpError):
    """A value cannot be encoded the way the original was, so the edit is refused."""


def varint(n: int) -> bytes:
    """Protobuf base-128 varint of a non-negative integer (at most 64 bits)."""
    if n < 0 or n >= 1 << 64:
        raise EncodeError(f"varint out of range: {n}")
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def pb_key(number: int, wire: int) -> bytes:
    if number <= 0 or number >= 1 << 29 or wire not in (0, 1, 2, 5):
        raise EncodeError(f"bad protobuf key {number}/{wire}")
    return varint(number << 3 | wire)


def pb_len(number: int, payload: bytes) -> bytes:
    """A length-delimited field."""
    return pb_key(number, 2) + varint(len(payload)) + payload


def pb_varint(number: int, value: int) -> bytes:
    return pb_key(number, 0) + varint(value)


def canonical(f: blobs.PbField, payload: bytes) -> bool:
    """Whether ``f`` is written in its shortest form (key, length and varint value)."""
    raw = payload[f.offset : f.end]
    if f.wire == 0 and isinstance(f.value, int):
        return raw == pb_varint(f.number, f.value)
    if f.wire == 2 and isinstance(f.value, bytes):
        return raw == pb_len(f.number, f.value)
    if isinstance(f.value, bytes):
        return raw == pb_key(f.number, f.wire) + f.value
    return False


def replace_field(payload: bytes, f: blobs.PbField, new: bytes | int) -> bytes:
    """``payload`` with field ``f`` (from ``decode_protobuf(payload)``) given a new value.

    Only the field's own bytes change. A field written in a non-canonical form is refused: the
    re-encoded field would be canonical, which changes bytes the edit did not ask to change.
    """
    if payload[f.offset : f.end] == b"" or f.end > len(payload):
        raise EncodeError("field span is not inside the payload")
    if not canonical(f, payload):
        raise EncodeError(f"protobuf field {f.number} is not canonically encoded")
    if f.wire == 2 and isinstance(new, bytes):
        enc = pb_len(f.number, new)
    elif f.wire == 0 and isinstance(new, int) and not isinstance(new, bool):
        enc = pb_varint(f.number, new)
    else:
        raise EncodeError(f"cannot write {type(new).__name__} into wire type {f.wire}")
    return payload[: f.offset] + enc + payload[f.end :]


def zstd_raw_frame(payload: bytes) -> bytes:
    """A single-segment zstd frame holding ``payload`` in raw blocks, with no checksum."""
    size = len(payload)
    if size < 256:
        fhd, fcs = 0x20, struct.pack("<B", size)
    elif size < 65536 + 256:
        fhd, fcs = 0x60, struct.pack("<H", size - 256)
    elif size <= _U32_MAX:
        fhd, fcs = 0xA0, struct.pack("<I", size)
    else:
        raise EncodeError("payload too large for a zstd frame written here")
    out = bytearray(ZSTD_MAGIC + bytes([fhd]) + fcs)
    pos = 0
    while True:
        chunk = payload[pos : pos + _RAW_BLOCK_MAX]
        pos += len(chunk)
        last = pos >= size
        out += ((len(chunk) << 3) | int(last)).to_bytes(3, "little")
        out += chunk
        if last:
            return bytes(out)


def check_single_frame(frame: bytes) -> int:
    """``len(frame)`` when ``frame`` is exactly one zstd frame; raises :class:`EncodeError` else.

    The frame is walked by its block headers (no decompression): raw blocks carry their size,
    RLE blocks one byte, compressed blocks their compressed size, then the optional checksum.
    A second frame, a skippable frame, trailing bytes, a dictionary or a reserved bit or block
    type are refused. Decoders differ on what follows a first frame (``zstandard`` reads on into
    the next frame, the stdlib-only path stops after the first), so a re-encode of such a
    payload would change or drop data that the gate cannot see.
    """
    if len(frame) < 6 or frame[:4] != ZSTD_MAGIC:
        raise EncodeError("not a zstd frame (magic 28b52ffd missing)")
    fhd = frame[4]
    if fhd & 0x08:
        raise EncodeError("the zstd frame header has its reserved bit set")
    if fhd & 0x03:
        raise EncodeError("the zstd frame uses a dictionary")
    try:
        off, _ = blobs._frame_header(frame)
    except BlobError as exc:
        raise EncodeError(f"truncated zstd frame header: {exc}") from exc
    while True:
        if off + 3 > len(frame):
            raise EncodeError("truncated zstd block header")
        header = int.from_bytes(frame[off : off + 3], "little")
        off += 3
        last, btype, bsize = header & 1, (header >> 1) & 3, header >> 3
        if btype == 3:
            raise EncodeError("a zstd block has the reserved block type")
        off += 1 if btype == 1 else bsize
        if off > len(frame):
            raise EncodeError("truncated zstd block")
        if last:
            break
    if fhd & 0x04:
        off += 4
        if off > len(frame):
            raise EncodeError("truncated zstd checksum")
    if off != len(frame):
        raise EncodeError(f"{len(frame) - off} trailing byte(s) after the first zstd frame")
    return off


def check_envelope_payload(original: bytes) -> None:
    """Refuse an envelope whose zstd payload is not exactly one zstd frame."""
    env = blobs.parse_envelope(original)
    if env is None or env.codec != "zstd":
        return
    try:
        check_single_frame(env.payload)
    except EncodeError as exc:
        raise EncodeError(f"the blob's payload must be exactly one zstd frame: {exc}") from exc


def envelope(original: bytes, payload: bytes) -> bytes:
    """Re-frame ``payload`` (protobuf bytes) with the framing of the ``original`` blob.

    Keeps the version and the codec: ``protobuf`` (0x80) stores the payload as is, ``zstd`` (0x81)
    stores it in a raw-block zstd frame, and ``none`` (header without a codec byte) as is. The
    flag-only form (no header) keeps its codec byte.
    """
    env = blobs.parse_envelope(original)
    if env is None:
        raise EncodeError("the original blob is not an envelope")
    check_envelope_payload(original)
    if env.codec == "zstd":
        body = bytes([blobs.CODEC_ZSTD]) + zstd_raw_frame(payload)
    elif env.codec == "protobuf":
        body = bytes([blobs.CODEC_PROTOBUF]) + payload
    else:
        body = payload
    if env.version is None:
        return body
    if len(body) > _U32_MAX:
        raise EncodeError("envelope payload too large")
    return struct.pack(">II", env.version, len(body)) + body


def replace_kd_bytes(data: bytes, entry: blobs.KdEntry, new: bytes) -> bytes:
    """A keyed dict with one byte-array entry (``type 0x0C``) given a new value.

    The key, type and null flag bytes are kept; only the length word and the value change.
    """
    if entry.type != blobs.KD_BYTES or not isinstance(entry.value, bytes):
        raise EncodeError(f"keyed-dict entry {entry.key!r} is not a byte array")
    try:
        (klen,) = struct.unpack_from(">I", data, entry.offset)
        value_at = entry.offset + 4 + klen + 5
        (old_len,) = struct.unpack_from(">I", data, value_at)
    except struct.error as exc:
        raise EncodeError(f"keyed-dict entry {entry.key!r} is not inside this blob") from exc
    if old_len != len(entry.value) or value_at + 4 + old_len != entry.end:
        raise EncodeError(f"keyed-dict entry {entry.key!r} does not have the expected layout")
    if len(new) >= _U32_MAX:
        raise EncodeError("byte array too large")
    return data[:value_at] + struct.pack(">I", len(new)) + new + data[entry.end :]


def hex_like(original_text: bytes, data: bytes) -> bytes:
    """Hex text for ``data``, in the letter case of ``original_text``, keeping its surrounding
    whitespace. Mixed-case originals are refused: their exact spelling cannot be kept."""
    stripped = original_text.strip()
    lead = original_text[: len(original_text) - len(original_text.lstrip())]
    trail = original_text[len(original_text.rstrip()) :]
    letters = bytes(c for c in stripped if chr(c).isalpha())
    if letters and letters.lower() != letters and letters.upper() != letters:
        raise EncodeError("hex text mixes letter cases")
    text = data.hex().encode("ascii")
    if letters and letters.upper() == letters:
        text = text.upper()
    return lead + text + trail


def check_text(value: str, what: str, *, max_chars: int) -> None:
    """Refuse text that Resolve never writes or that XML cannot carry."""
    if not value:
        raise EncodeError(f"{what} is empty")
    if len(value) > max_chars:
        raise EncodeError(f"{what} is longer than {max_chars} characters")
    bad = _BAD_TEXT.search(value)
    if bad is not None:
        raise EncodeError(f"{what} contains the control character U+{ord(bad.group()):04X}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EncodeError(f"{what} is not valid Unicode text: {exc.reason}") from exc


def xml_text(value: str) -> bytes:
    """Character data the way Resolve writes it: ``&``, ``<`` and ``>`` escaped, UTF-8."""
    escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return escaped.encode("utf-8")


# --------------------------------------------------------------------------- whole structures

_KD_FIXED = {
    blobs.KD_INT32: ">i",
    blobs.KD_UINT32: ">I",
    blobs.KD_INT64: ">q",
    blobs.KD_UINT64: ">Q",
    blobs.KD_DOUBLE: ">d",
    blobs.KD_FLOAT: ">d",
}
_MAX_KD_DEPTH = 32


def _kd_string(value: str | None) -> bytes:
    if value is None:
        return b"\xff\xff\xff\xff"
    data = value.encode("utf-16-be")
    if len(data) >= _U32_MAX:
        raise EncodeError("keyed-dict string too long")
    return struct.pack(">I", len(data)) + data


def _kd_value(vtype: int, value: object, depth: int) -> bytes:
    if depth > _MAX_KD_DEPTH:
        raise EncodeError("keyed dict nested too deeply")
    try:
        if vtype == blobs.KD_STRING:
            if value is not None and not isinstance(value, str):
                raise EncodeError("a keyed-dict string must be str or None")
            return _kd_string(value)
        if vtype == blobs.KD_BYTES:
            if value is None:
                return b"\xff\xff\xff\xff"
            if not isinstance(value, bytes) or len(value) >= _U32_MAX:
                raise EncodeError("a keyed-dict byte array must be bytes under 4 GiB")
            return struct.pack(">I", len(value)) + value
        if vtype == blobs.KD_BOOL:
            if not isinstance(value, bool):
                raise EncodeError("a keyed-dict bool must be bool")
            return b"\x01" if value else b"\x00"
        if vtype in _KD_FIXED:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise EncodeError(f"keyed-dict type 0x{vtype:x} needs a number")
            if vtype not in (blobs.KD_DOUBLE, blobs.KD_FLOAT) and not isinstance(value, int):
                raise EncodeError(f"keyed-dict type 0x{vtype:x} needs an integer")
            return struct.pack(_KD_FIXED[vtype], value)
        if vtype == blobs.KD_CHAR:
            if not isinstance(value, str) or len(value) != 1 or ord(value) > 0xFFFF:
                raise EncodeError("a keyed-dict char must be one BMP character")
            return struct.pack(">H", ord(value))
        if vtype == blobs.KD_DATE:
            if not isinstance(value, bytes) or len(value) != 9:
                raise EncodeError("a keyed-dict date must be its 9 raw bytes")
            return value
    except struct.error as exc:
        raise EncodeError(f"keyed-dict value out of range for type 0x{vtype:x}: {exc}") from None
    raise EncodeError(f"keyed-dict type 0x{vtype:x} is not written here")


def keyed_dict(entries: list[tuple[str, int, int, object]] | tuple, *, header: int = 1) -> bytes:
    """A keyed dict (the inverse of :func:`aar.drp.blobs.decode_keyed_dict`) from
    ``(key, type, null flag, value)`` entries, in their order.

    The scalar types, strings, byte arrays and raw 9-byte dates are written; lists and maps
    (types 0x08, 0x09, 0x0B) are refused, as no media row has them at the top of a blob."""
    if not 0 <= header <= _U32_MAX:
        raise EncodeError("keyed-dict header out of range")
    out = bytearray(struct.pack(">II", header, len(entries)))
    for entry in entries:
        key, vtype, null, value = entry
        if not isinstance(key, str) or not key:
            raise EncodeError("a keyed-dict key must be a non-empty str")
        if null not in (0, 1):
            raise EncodeError(f"keyed-dict null flag must be 0 or 1, not {null!r}")
        out += _kd_string(key) + struct.pack(">IB", vtype, null) + _kd_value(vtype, value, 0)
    return bytes(out)


def protobuf(fields: list[tuple[int, int, int | bytes]] | tuple) -> bytes:
    """A protobuf message, canonically encoded, from ``(number, wire, value)`` fields in their
    order: varints (wire 0) take an int, fixed64 (1) 8 bytes, length-delimited (2) bytes and
    fixed32 (5) 4 bytes. The inverse of ``aar.drp.blobs.decode_protobuf(nested_depth=0)`` on
    every canonical message."""
    out = bytearray()
    for number, wire, value in fields:
        if wire == 0:
            if isinstance(value, bool) or not isinstance(value, int):
                raise EncodeError(f"protobuf field {number}: a varint needs an int")
            out += pb_varint(number, value)
        elif wire == 2:
            if not isinstance(value, bytes):
                raise EncodeError(f"protobuf field {number}: a length-delimited value needs bytes")
            out += pb_len(number, value)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            if not isinstance(value, bytes) or len(value) != size:
                raise EncodeError(f"protobuf field {number}: wire {wire} needs {size} bytes")
            out += pb_key(number, wire) + value
        else:
            raise EncodeError(f"protobuf field {number}: wire type {wire} is not written")
    return bytes(out)
