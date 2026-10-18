"""Encoders for the blob layouts :mod:`aar.drp.blobs` decodes, used to re-encode edited blobs.

Each encoder is the inverse of a decoder in :mod:`aar.drp.blobs` and is checked against it in the
tests (decode -> encode is the identity on every corpus blob of that kind):

* protobuf: :func:`varint`, :func:`pb_key`, :func:`pb_len`, :func:`replace_field`
* message envelope ``[u32 version][u32 length][codec][payload]``: :func:`envelope`
* zstd frames made of raw blocks only: :func:`zstd_raw_frame`. Resolve itself writes raw-block
  frames for small payloads; any conforming zstd decoder reads them. Writing raw blocks keeps the
  writer stdlib-only and its output deterministic (no compressor version in the loop).
* keyed dict byte-array values: :func:`replace_kd_bytes`
* hex text: :func:`hex_like` keeps the letter case of the text it replaces
* XML character data: :func:`xml_text`

Every function either returns bytes or raises :class:`EncodeError`; nothing is guessed.
"""

from __future__ import annotations

import re
import struct

from aar.drp import blobs
from aar.drp.errors import DrpError

__all__ = [
    "EncodeError",
    "check_text",
    "envelope",
    "hex_like",
    "pb_key",
    "pb_len",
    "pb_varint",
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


def envelope(original: bytes, payload: bytes) -> bytes:
    """Re-frame ``payload`` (protobuf bytes) with the framing of the ``original`` blob.

    Keeps the version and the codec: ``protobuf`` (0x80) stores the payload as is, ``zstd`` (0x81)
    stores it in a raw-block zstd frame, and ``none`` (header without a codec byte) as is. The
    flag-only form (no header) keeps its codec byte.
    """
    env = blobs.parse_envelope(original)
    if env is None:
        raise EncodeError("the original blob is not an envelope")
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
