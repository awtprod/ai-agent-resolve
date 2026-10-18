"""Codecs for the binary state Resolve stores as hex text inside its XML.

Everything here is stdlib-only except zstd frames with compressed blocks, which need the
optional ``zstandard`` package (``pip install 'ai-agent-resolve[drp]'``) or Python 3.14's
``compression.zstd``. Raw and RLE zstd blocks, which Resolve writes for small payloads, decode
without either.

Layouts, all measured against the bundled templates and the private corpus (see
``tests/drp``):

* keyed dict: ``[u32 BE 1][u32 BE count]`` then ``count`` entries of
  ``[u32 keyLen][UTF-16BE key][u32 type][u8 subtype][payload]`` (``Time``, ``Geometry``,
  ``TracksBA``, small ``FieldsBlob`` values, ...).
* message envelope: ``[u32 BE version][u32 BE length of the rest][codec][payload]`` where the
  codec byte is ``0x80`` (payload is protobuf) or ``0x81`` (payload is a zstd frame holding
  protobuf). Large ``FieldsBlob``, ``Clip`` and ``EffectFiltersBA`` values use it; nested marker
  messages use the same header without a codec byte.
* flagged message: ``[codec][payload]`` with the same codec byte and no header (``Body`` of
  ``ListMgt::LmVersion``, ``Radiometry``, ``KeyframesBA``).
* qCompress: ``[u32 BE uncompressed size][zlib stream]`` (``RenderCacheBA``, ``CompositionBA``,
  ``FLStudioModelBA``).
* fixed 16-byte values: ``FrameRate``/``MediaFrameRate`` = little-endian double + 8 zero bytes;
  ``Resolution`` = ``[u32 0][u32 BE width][u32 0][u32 BE height]``; ``MediaExtents`` = two
  little-endian doubles (start seconds, duration seconds).
* ``MediaTimemapBA``: either ``[u8 kind][big-endian doubles]`` (1x speed) or a keyed dict
  ``Sm2TimeMap`` (retimed).

Protobuf blobs are decoded at the wire level only; Resolve's ``.proto`` schema is private, so
field numbers have no names unless a caller maps them.
"""

from __future__ import annotations

import io
import math
import re
import struct
import zlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from aar.drp.errors import BlobError, BudgetExceededError, ZstdUnavailableError
from aar.drp.scan import parse_int

__all__ = [
    "DEFAULT_MAX_OUTPUT",
    "BlobCheck",
    "BlobKind",
    "DecodeBudget",
    "Envelope",
    "KdEntry",
    "KeyedDict",
    "PbField",
    "TimeMap",
    "check",
    "classify",
    "decode_extents",
    "decode_keyed_dict",
    "decode_protobuf",
    "decode_rate",
    "decode_resolution",
    "decode_timemap",
    "is_blob_tag",
    "parse_envelope",
    "parse_frames",
    "parse_hex",
    "qcompress_decompress",
    "unwrap_message",
    "zstd_backend",
    "zstd_decompress",
]

DEFAULT_MAX_OUTPUT = 64 * 1024 * 1024
"""Upper bound for any single decompressed blob (bytes)."""

DEFAULT_MAX_TOTAL_OUTPUT = 400 * 1024 * 1024
"""Default budget for all decompression in one file (bytes).

The largest total in the private corpus is 20.8 MB (reader plus ``--blobs``); this is about
20 times that.
"""


@dataclass(slots=True)
class DecodeBudget:
    """Bytes that all decompressions for one file may still produce.

    Every zstd or qCompress decode charges what it produced, including a decode that failed
    after producing output. Once the budget is used up, further decodes raise
    :class:`~aar.drp.errors.BudgetExceededError` without decompressing anything, and
    ``refused`` counts them.
    """

    limit: int = DEFAULT_MAX_TOTAL_OUTPUT
    used: int = 0
    refused: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def cap(self, max_output: int) -> tuple[int, bool]:
        """The output limit for one decode, and whether the budget (not the blob limit) sets it."""
        if self.remaining <= 0:
            self.refused += 1
            raise BudgetExceededError(f"decompression budget of {self.limit} bytes used up")
        if self.remaining < max_output:
            return self.remaining, True
        return max_output, False

    def refuse(self) -> BudgetExceededError:
        self.refused += 1
        return BudgetExceededError(f"decompression budget of {self.limit} bytes used up")


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
CODEC_PROTOBUF = 0x80
CODEC_ZSTD = 0x81

# Tags whose text is hex-encoded binary even though the name does not end in BA/Blob.
_EXTRA_BLOB_TAGS = frozenset(
    {
        "Body",
        "Buffer",
        "Clip",
        "FrameRate",
        "Geometry",
        "MediaExtents",
        "MediaFrameRate",
        "PreConformMediaExtents",
        "Presets",
        "Proxy",
        "Radiometry",
        "Resolution",
        "Time",
        "VideoMetadata",
    }
)
_FIXED16_TAGS = frozenset(
    {"FrameRate", "MediaFrameRate", "Resolution", "MediaExtents", "PreConformMediaExtents"}
)
_HEX_RE = re.compile(r"[0-9a-fA-F]*")


def is_blob_tag(tag: str) -> bool:
    """Whether an element with this tag holds hex-encoded binary.

    Decimal fields like ``DbSavedTime`` also look like hex, so this is decided by name.
    """
    return tag.endswith(("BA", "Blob")) or tag in _EXTRA_BLOB_TAGS


def parse_frames(text: str) -> tuple[int, float | None]:
    """Parse a frame count written as ``123`` or ``123|<16 hex digits>``.

    Resolve 20+ appends a sub-frame remainder to some ``Start``/``Duration``/``In`` values: the
    hex is a little-endian double in ``[0, 1)``. Raises :class:`BlobError` on anything else.
    """
    whole, sep, frac = text.strip().partition("|")
    frames = parse_int(whole)
    if frames is None:
        raise BlobError(f"not a frame count: {text.strip()[:40]!r}")
    if not sep:
        return frames, None
    try:
        raw = bytes.fromhex(frac)
    except ValueError as exc:
        raise BlobError(f"sub-frame part is not hex: {frac[:40]!r}") from exc
    if len(raw) != 8:
        raise BlobError(f"sub-frame part must be 8 bytes, got {len(raw)}")
    (fraction,) = struct.unpack("<d", raw)
    if not 0.0 <= fraction < 1.0:
        raise BlobError(f"sub-frame remainder {fraction!r} is outside [0, 1)")
    return frames, fraction


def parse_hex(text: str) -> bytes:
    """Decode a hex blob strictly. Raises :class:`BlobError` on odd length or non-hex text."""
    stripped = text.strip()
    if not _HEX_RE.fullmatch(stripped):
        raise BlobError("not hex: contains non-hex characters")
    if len(stripped) % 2:
        raise BlobError(f"odd hex length {len(stripped)} (truncated blob?)")
    return bytes.fromhex(stripped)


# --------------------------------------------------------------------------- keyed dict
#
# A keyed dict is a header word (always 1 so far) followed by what looks like Qt's QDataStream
# serialization of a QVariantMap in the Qt 4 stream format: a count, then per entry a QString
# key ([u32 byte length][UTF-16BE]) and a QVariant ([u32 type id][u8 is-null][value]). The Qt 4
# reading is inferred from two details that match it and nothing else: type 0x87 (135) is
# QMetaType::Float in Qt 4 and holds an 8-byte double, and dates are 9 bytes (Qt 4 QDateTime).
# Every keyed dict in the templates and the corpus decodes to its last byte with these types.

KD_BOOL = 0x01
KD_INT32 = 0x02
KD_UINT32 = 0x03
KD_INT64 = 0x04
KD_UINT64 = 0x05
KD_DOUBLE = 0x06
KD_CHAR = 0x07
KD_MAP = 0x08
KD_LIST = 0x09
KD_STRING = 0x0A
KD_STRING_LIST = 0x0B
KD_BYTES = 0x0C
KD_DATE = 0x10
KD_FLOAT = 0x87
_MAX_VARIANT_DEPTH = 32

KdValue = Any
"""str, int, float, bool, bytes (byte arrays and 9-byte dates), None, list or dict."""


@dataclass(frozen=True, slots=True)
class KdEntry:
    """One keyed-dict entry. ``offset``/``end`` are byte offsets inside the blob.

    ``value`` is a Python value: ``str``, ``int``, ``float``, ``bool``, ``bytes`` (byte arrays
    and raw 9-byte dates), ``None`` (null), ``list`` (lists) or ``dict`` (nested maps).
    """

    key: str
    type: int
    subtype: int
    value: KdValue
    offset: int
    end: int


@dataclass(slots=True)
class KeyedDict:
    """A decoded keyed dict.

    ``error`` is set when decoding stopped early, and ``consumed`` says where it stopped.
    """

    header: int
    count: int
    entries: list[KdEntry]
    consumed: int
    size: int
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.error is None and self.consumed == self.size and len(self.entries) == self.count

    def get(self, key: str, default: Any = None) -> Any:
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        return default

    def entry(self, key: str) -> KdEntry | None:
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None


def _qstring(data: bytes, off: int) -> tuple[str | None, int]:
    (n,) = struct.unpack_from(">I", data, off)
    off += 4
    if n == 0xFFFFFFFF:
        return None, off
    if n % 2 or off + n > len(data):
        raise BlobError(f"bad string length {n} at {off - 4}")
    return data[off : off + n].decode("utf-16-be"), off + n


def _qvariant(data: bytes, off: int, depth: int) -> tuple[int, int, KdValue, int]:
    """Decode one QVariant at ``off``: returns (type, null flag, value, new offset)."""
    if depth > _MAX_VARIANT_DEPTH:
        raise BlobError("keyed dict nested too deeply")
    vtype, null = struct.unpack_from(">IB", data, off)
    off += 5
    value: KdValue
    if vtype == KD_STRING:
        value, off = _qstring(data, off)
    elif vtype == KD_BYTES:
        (n,) = struct.unpack_from(">I", data, off)
        off += 4
        if n == 0xFFFFFFFF:
            value = None
        else:
            if off + n > len(data):
                raise BlobError(f"byte array of {n} bytes runs past the end of the blob")
            value, off = data[off : off + n], off + n
    elif vtype == KD_BOOL:
        if off + 1 > len(data):
            raise BlobError("truncated bool")
        value, off = data[off] != 0, off + 1
    elif vtype == KD_INT32:
        (value,) = struct.unpack_from(">i", data, off)
        off += 4
    elif vtype == KD_UINT32:
        (value,) = struct.unpack_from(">I", data, off)
        off += 4
    elif vtype == KD_INT64:
        (value,) = struct.unpack_from(">q", data, off)
        off += 8
    elif vtype == KD_UINT64:
        (value,) = struct.unpack_from(">Q", data, off)
        off += 8
    elif vtype in (KD_DOUBLE, KD_FLOAT):
        (value,) = struct.unpack_from(">d", data, off)
        off += 8
    elif vtype == KD_CHAR:
        (code,) = struct.unpack_from(">H", data, off)
        value, off = chr(code), off + 2
    elif vtype == KD_DATE:
        if off + 9 > len(data):
            raise BlobError("truncated date")
        value, off = data[off : off + 9], off + 9
    elif vtype in (KD_LIST, KD_STRING_LIST, KD_MAP):
        (count,) = struct.unpack_from(">I", data, off)
        off += 4
        if count > len(data):
            raise BlobError(f"implausible element count {count}")
        if vtype == KD_MAP:
            mapping: dict[str, KdValue] = {}
            for _ in range(count):
                key, off = _qstring(data, off)
                _, _, item, off = _qvariant(data, off, depth + 1)
                mapping[key or ""] = item
            value = mapping
        else:
            items: list[KdValue] = []
            for _ in range(count):
                if vtype == KD_STRING_LIST:
                    text, off = _qstring(data, off)
                    items.append(text)
                else:
                    _, _, item, off = _qvariant(data, off, depth + 1)
                    items.append(item)
            value = items
    else:
        raise BlobError(f"unknown keyed-dict value type 0x{vtype:x}")
    return vtype, null, value, off


def decode_keyed_dict(data: bytes, *, strict: bool = False) -> KeyedDict:
    """Decode a keyed dict.

    With ``strict=False`` decoding stops at the first problem and the decoded prefix is
    returned with ``error`` set, so nothing is silently dropped: bytes from ``consumed`` on are
    still in ``data``. With ``strict=True`` any problem raises :class:`BlobError`.
    """
    size = len(data)
    if size < 8:
        raise BlobError(f"keyed dict too short ({size} bytes)")
    header, count = struct.unpack_from(">II", data, 0)
    kd = KeyedDict(header=header, count=count, entries=[], consumed=8, size=size)
    off = 8
    key = None
    try:
        for _ in range(count):
            start = off
            (klen,) = struct.unpack_from(">I", data, off)
            if klen % 2 or klen > 4096 or off + 4 + klen > size:
                raise BlobError(f"bad key length {klen} at {off}")
            key, off = _qstring(data, off)
            vtype, null, value, off = _qvariant(data, off, 0)
            kd.entries.append(KdEntry(key or "", vtype, null, value, start, off))
            kd.consumed = off
    except BlobError as exc:
        kd.error = f"{exc} (near key {key!r})" if key is not None else str(exc)
    except (struct.error, UnicodeDecodeError) as exc:
        kd.error = f"truncated or malformed near key {key!r}: {exc}"
    if kd.error is None and kd.consumed != size:
        kd.error = f"{size - kd.consumed} trailing bytes after {count} entries"
    if strict and kd.error is not None:
        raise BlobError(f"keyed dict: {kd.error}")
    return kd


# --------------------------------------------------------------------------- protobuf


@dataclass(slots=True)
class PbField:
    """One protobuf field at the wire level.

    ``value`` is an ``int`` for varints, ``bytes`` for fixed64/fixed32/length-delimited fields.
    ``children`` is set when a length-delimited value itself parses cleanly as protobuf; that
    is a guess (strings can look like messages), so the raw ``value`` is always kept too.
    """

    number: int
    wire: int
    offset: int
    end: int
    value: int | bytes
    children: list[PbField] | None = None

    @property
    def as_double(self) -> float | None:
        if self.wire == 1 and isinstance(self.value, bytes):
            return struct.unpack("<d", self.value)[0]
        return None

    @property
    def as_float(self) -> float | None:
        if self.wire == 5 and isinstance(self.value, bytes):
            return struct.unpack("<f", self.value)[0]
        return None

    @property
    def as_text(self) -> str | None:
        if self.wire != 2 or not isinstance(self.value, bytes):
            return None
        try:
            text = self.value.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return text if text.isprintable() else None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"field": self.number, "wire": self.wire}
        if isinstance(self.value, int):
            out["varint"] = self.value
        elif self.wire == 1:
            out["double"] = self.as_double
            out["hex"] = self.value.hex()
        elif self.wire == 5:
            out["float"] = self.as_float
            out["hex"] = self.value.hex()
        elif self.children is not None:
            out["message"] = [c.to_json() for c in self.children]
        elif self.as_text is not None:
            out["text"] = self.as_text
        else:
            out["hex"] = self.value.hex()
        return out


def _varint(data: bytes, off: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if off >= len(data):
            raise BlobError("protobuf: truncated varint")
        byte = data[off]
        off += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, off
        shift += 7
        if shift > 63:
            raise BlobError("protobuf: varint longer than 10 bytes")


def decode_protobuf(
    data: bytes, *, nested_depth: int = 6, max_fields: int = 200_000, _base: int = 0
) -> list[PbField]:
    """Decode protobuf wire format strictly into a field list (raises :class:`BlobError`).

    Length-delimited values are tried as nested messages up to ``nested_depth`` levels.
    ``max_fields`` bounds the work done on hostile input.
    """
    fields: list[PbField] = []
    off = 0
    size = len(data)
    while off < size:
        start = off
        key, off = _varint(data, off)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise BlobError(f"protobuf: field number 0 at {_base + start}")
        value: int | bytes
        if wire == 0:
            value, off = _varint(data, off)
        elif wire == 1:
            if off + 8 > size:
                raise BlobError("protobuf: truncated fixed64")
            value, off = data[off : off + 8], off + 8
        elif wire == 5:
            if off + 4 > size:
                raise BlobError("protobuf: truncated fixed32")
            value, off = data[off : off + 4], off + 4
        elif wire == 2:
            length, off = _varint(data, off)
            if off + length > size:
                raise BlobError("protobuf: length-delimited field runs past the end")
            value, off = data[off : off + length], off + length
        else:
            raise BlobError(f"protobuf: unsupported wire type {wire} at {_base + start}")
        pb = PbField(number, wire, _base + start, _base + off, value)
        if wire == 2 and nested_depth > 0 and value:
            try:
                pb.children = decode_protobuf(
                    value,
                    nested_depth=nested_depth - 1,
                    max_fields=max_fields,
                    _base=_base + off - len(value),
                )
            except BlobError:
                pb.children = None
        fields.append(pb)
        if len(fields) > max_fields:
            raise BlobError(f"protobuf: more than {max_fields} fields")
    return fields


# --------------------------------------------------------------------------- compression


def zstd_backend() -> str | None:
    """Name of the zstd library that will be used, or ``None`` when only raw blocks decode."""
    return _zstd_backend()[0]


def _zstd_backend() -> tuple[str | None, Any]:
    try:
        import zstandard

        return "zstandard", zstandard
    except ImportError:
        pass
    try:
        from compression import zstd  # type: ignore[import-not-found]  # Python 3.14+

        return "compression.zstd", zstd
    except ImportError:
        return None, None


def _frame_header(frame: bytes) -> tuple[int, int | None]:
    """Return (offset of the first block, declared content size or None)."""
    if len(frame) < 6 or frame[:4] != ZSTD_MAGIC:
        raise BlobError("not a zstd frame (magic 28b52ffd missing)")
    fhd = frame[4]
    single = (fhd >> 5) & 1
    fcs_code = fhd >> 6
    dict_code = fhd & 3
    off = 5 + (0 if single else 1) + (0, 1, 2, 4)[dict_code]
    fcs_len = (1 if single else 0, 2, 4, 8)[fcs_code]
    if off + fcs_len > len(frame):
        raise BlobError("zstd: truncated frame header")
    size: int | None = None
    if fcs_len:
        size = int.from_bytes(frame[off : off + fcs_len], "little")
        if fcs_len == 2:
            size += 256
    return off + fcs_len, size


def _zstd_raw_only(frame: bytes, max_output: int) -> bytes:
    """Decode a zstd frame made only of raw/RLE blocks (what Resolve writes for tiny payloads)."""
    off, _ = _frame_header(frame)
    if frame[4] & 3:
        raise ZstdUnavailableError("zstd frame uses a dictionary")
    out = bytearray()
    while True:
        if off + 3 > len(frame):
            raise BlobError("zstd: truncated block header")
        header = int.from_bytes(frame[off : off + 3], "little")
        off += 3
        last, btype, bsize = header & 1, (header >> 1) & 3, header >> 3
        if btype == 0:
            if off + bsize > len(frame):
                raise BlobError("zstd: truncated raw block")
            out += frame[off : off + bsize]
            off += bsize
        elif btype == 1:
            if off >= len(frame):
                raise BlobError("zstd: truncated RLE block")
            out += frame[off : off + 1] * bsize
            off += 1
        else:
            raise ZstdUnavailableError(
                "this zstd blob has compressed blocks; install the optional dependency "
                "'zstandard' (pip install 'ai-agent-resolve[drp]') to decode it"
            )
        if len(out) > max_output:
            raise BlobError(f"zstd output exceeds {max_output} bytes")
        if last:
            return bytes(out)


def zstd_decompress(
    frame: bytes, *, max_output: int = DEFAULT_MAX_OUTPUT, budget: DecodeBudget | None = None
) -> bytes:
    """Decompress one zstd frame, refusing to produce more than ``max_output`` bytes.

    With a ``budget``, the output is also limited to what the budget has left, and everything
    produced is charged to it, also when the frame turns out to be corrupt or too large.
    """
    _, declared = _frame_header(frame)
    if declared is not None and declared > max_output:
        raise BlobError(f"zstd frame declares {declared} bytes, over the {max_output} limit")
    blob_limit = max_output
    by_budget = False
    if budget is not None:
        if declared is not None and declared > budget.remaining:
            raise budget.refuse()
        max_output, by_budget = budget.cap(max_output)
    produced = [0]
    try:
        out = _zstd_decode(frame, max_output, produced)
    finally:
        if budget is not None:
            budget.used += min(produced[0], max_output)
    if len(out) > max_output:
        if by_budget and budget is not None:
            raise budget.refuse()
        raise BlobError(f"zstd output exceeds {blob_limit} bytes")
    return out


def _zstd_decode(frame: bytes, max_output: int, produced: list[int]) -> bytes:
    """Decode at most ``max_output + 1`` bytes of ``frame``; ``produced[0]`` counts the output."""
    name, lib = _zstd_backend()
    if name == "zstandard":
        reader = lib.ZstdDecompressor().stream_reader(io.BytesIO(frame), read_across_frames=False)
        chunks: list[bytes] = []
        try:
            while produced[0] <= max_output:
                chunk = reader.read(min(1 << 20, max_output + 1 - produced[0]))
                if not chunk:
                    break
                chunks.append(chunk)
                produced[0] += len(chunk)
        except lib.ZstdError as exc:
            raise BlobError(f"zstd: {exc}") from exc
        finally:
            reader.close()
        return b"".join(chunks)
    if name == "compression.zstd":
        dec = lib.ZstdDecompressor()
        try:
            out = dec.decompress(frame, max_length=max_output + 1)
        except lib.ZstdError as exc:
            produced[0] = max_output  # amount unknown: charge the most it could have been
            raise BlobError(f"zstd: {exc}") from exc
        produced[0] = len(out)
        return bytes(out)
    out = _zstd_raw_only(frame, max_output)
    produced[0] = len(out)
    return out


def qcompress_decompress(
    data: bytes, *, max_output: int = DEFAULT_MAX_OUTPUT, budget: DecodeBudget | None = None
) -> bytes:
    """Decode Qt ``qCompress`` output: ``[u32 BE size][zlib]``, bounded by ``max_output``.

    With a ``budget``, the output is also limited to what the budget has left, and charged to it.
    """
    if len(data) < 6:
        raise BlobError("qCompress blob too short")
    (declared,) = struct.unpack_from(">I", data, 0)
    if declared > max_output:
        raise BlobError(f"qCompress declares {declared} bytes, over the {max_output} limit")
    blob_limit = max_output
    by_budget = False
    if budget is not None:
        if declared > budget.remaining:
            raise budget.refuse()
        max_output, by_budget = budget.cap(max_output)
    dec = zlib.decompressobj()
    src = memoryview(data)[4:]
    buf = bytearray()
    try:
        # Feed the input in pieces so the output produced so far is known even on error.
        for i in range(0, len(src), 1 << 16):
            buf += dec.decompress(src[i : i + (1 << 16)], max_output + 1 - len(buf))
            if len(buf) > max_output:
                break
        else:
            buf += dec.flush()
    except zlib.error as exc:
        raise BlobError(f"qCompress: {exc}") from exc
    finally:
        if budget is not None:
            budget.used += min(len(buf), max_output)
    out = bytes(buf)
    if len(out) > max_output:
        if by_budget and budget is not None:
            raise budget.refuse()
        raise BlobError(f"qCompress output exceeds {blob_limit} bytes")
    if len(out) != declared:
        raise BlobError(f"qCompress size mismatch: declared {declared}, got {len(out)}")
    return out


# --------------------------------------------------------------------------- envelopes


@dataclass(frozen=True, slots=True)
class Envelope:
    """Framing of a serialized Resolve message.

    ``version`` and ``declared_length`` are ``None`` for the flag-only form. ``codec`` is
    ``"protobuf"`` (0x80), ``"zstd"`` (0x81) or ``"none"`` (header without a codec byte).
    ``payload`` is the bytes after the framing, still compressed when ``codec == "zstd"``.
    """

    version: int | None
    declared_length: int | None
    codec: str
    payload_offset: int
    payload: bytes


def parse_envelope(data: bytes) -> Envelope | None:
    """Recognise the two framings; ``None`` if ``data`` matches neither."""
    if len(data) >= 9:
        version, length = struct.unpack_from(">II", data, 0)
        if length == len(data) - 8:
            codec_byte = data[8]
            if codec_byte == CODEC_ZSTD and data[9:13] == ZSTD_MAGIC:
                return Envelope(version, length, "zstd", 9, data[9:])
            if codec_byte == CODEC_PROTOBUF:
                return Envelope(version, length, "protobuf", 9, data[9:])
            return Envelope(version, length, "none", 8, data[8:])
    if len(data) >= 1:
        if data[0] == CODEC_ZSTD and data[1:5] == ZSTD_MAGIC:
            return Envelope(None, None, "zstd", 1, data[1:])
        if data[0] == CODEC_PROTOBUF:
            return Envelope(None, None, "protobuf", 1, data[1:])
    return None


def unwrap_message(
    data: bytes, *, max_output: int = DEFAULT_MAX_OUTPUT, budget: DecodeBudget | None = None
) -> bytes:
    """Return the protobuf bytes inside an envelope, decompressing zstd when needed."""
    env = parse_envelope(data)
    if env is None:
        raise BlobError("not an envelope or flagged message")
    if env.codec == "zstd":
        return zstd_decompress(env.payload, max_output=max_output, budget=budget)
    return env.payload


# --------------------------------------------------------------------------- fixed-size values


def _fixed16(data: bytes, what: str) -> None:
    if len(data) != 16:
        raise BlobError(f"{what} blob must be 16 bytes, got {len(data)}")


def decode_rate(data: bytes) -> float:
    """``FrameRate`` / ``MediaFrameRate``: little-endian double, then 8 more bytes.

    The second half is zero in every timeline and timeline-item rate seen so far; two media
    ``Time`` rates had its last byte set to 1 (meaning unknown), so it is not checked.
    """
    _fixed16(data, "frame rate")
    (fps,) = struct.unpack_from("<d", data, 0)
    if not math.isfinite(fps) or fps < 0:
        raise BlobError(f"frame rate blob decodes to an implausible value {fps!r}")
    return fps


def decode_resolution(data: bytes) -> tuple[int, int]:
    """``Resolution``: ``[u32 0][u32 BE width][u32 0][u32 BE height]``."""
    _fixed16(data, "resolution")
    pad_w, width, pad_h, height = struct.unpack(">IIII", data)
    if pad_w or pad_h:
        raise BlobError("resolution blob has non-zero padding words")
    return width, height


def decode_extents(data: bytes) -> tuple[float, float]:
    """``MediaExtents``: little-endian doubles (start seconds, duration seconds)."""
    _fixed16(data, "extents")
    start, duration = struct.unpack("<dd", data)
    if not (math.isfinite(start) and math.isfinite(duration)):
        raise BlobError("extents blob decodes to non-finite values")
    return start, duration


# --------------------------------------------------------------------------- timemap


@dataclass(frozen=True, slots=True)
class TimeMap:
    """A decoded ``MediaTimemapBA``.

    ``form`` is ``"identity"`` (1x, compact) or ``"retimed"`` (``Sm2TimeMap`` keyed dict).
    For the retimed form ``speed`` is ``source_seconds / record_seconds`` (negative values are
    not produced here; reverse maps are not decoded yet).
    """

    form: str
    kind: int | None = None
    seconds: tuple[float, ...] = ()
    speed: float | None = None
    record_seconds: float | None = None
    source_seconds: float | None = None


def decode_timemap(data: bytes) -> TimeMap:
    """Decode ``MediaTimemapBA``; raises :class:`BlobError` when neither form fits."""
    if len(data) >= 8 and data[:4] == b"\x00\x00\x00\x01":
        kd = decode_keyed_dict(data, strict=True)
        if kd.get("DbType") != "Sm2TimeMap":
            raise BlobError(f"keyed timemap has DbType {kd.get('DbType')!r}")
        record = kd.get("XMax")
        source = kd.get("LastValidYOffset")
        speed = None
        if isinstance(record, float) and isinstance(source, float) and record > 0:
            # A non-finite ratio (an infinite/NaN XMax or LastValidYOffset, or an overflow of
            # the division) would reach the model as a float that json.dumps writes as the
            # invalid token Infinity/NaN. Reject it like decode_rate/decode_extents do, so the
            # reader turns it into an issue and leaves speed null. See DESIGN R1 and blobs.py.
            if not (math.isfinite(record) and math.isfinite(source)):
                raise BlobError("retimed timemap has a non-finite XMax or LastValidYOffset")
            speed = source / record
            if not math.isfinite(speed):
                raise BlobError(f"retimed timemap speed {speed!r} is not finite")
        return TimeMap(
            form="retimed",
            speed=speed,
            record_seconds=record if isinstance(record, float) else None,
            source_seconds=source if isinstance(source, float) else None,
        )
    if len(data) < 9 or (len(data) - 1) % 8:
        raise BlobError(f"compact timemap has unexpected length {len(data)}")
    values = struct.unpack(f">{(len(data) - 1) // 8}d", data[1:])
    return TimeMap(form="identity", kind=data[0], seconds=tuple(values), speed=1.0)


# --------------------------------------------------------------------------- classification


class BlobKind(StrEnum):
    """What a blob's bytes look like, judged from its tag and content."""

    EMPTY = "empty"
    INVALID_HEX = "invalid-hex"
    KEYED_DICT = "keyed-dict"
    ENVELOPE_PROTOBUF = "envelope-protobuf"
    ENVELOPE_ZSTD = "envelope-zstd"
    ENVELOPE_PLAIN = "envelope-plain"
    FLAGGED_PROTOBUF = "flagged-protobuf"
    FLAGGED_ZSTD = "flagged-zstd"
    QCOMPRESS = "qcompress"
    FIXED16 = "fixed16"
    TIMEMAP_COMPACT = "timemap-compact"
    JPEG = "jpeg"
    PNG = "png"
    TEXT = "text"
    UNKNOWN = "unknown"


def _looks_like_qcompress(data: bytes) -> bool:
    if len(data) < 6:
        return False
    cmf, flg = data[4], data[5]
    return cmf & 0x0F == 8 and (cmf << 8 | flg) % 31 == 0


def _looks_like_keyed_dict(data: bytes) -> bool:
    """Header 1 and a plausible first key; malformed bodies are then reported by decoding."""
    if len(data) < 12 or data[:4] != b"\x00\x00\x00\x01":
        return False
    (count, klen) = struct.unpack_from(">II", data, 4)
    if count == 0:
        return len(data) == 8
    if klen == 0 or klen % 2 or 12 + klen > len(data):
        return False
    try:
        key = data[12 : 12 + klen].decode("utf-16-be")
    except UnicodeDecodeError:
        return False
    return key.isprintable()


def classify(tag: str, data: bytes) -> BlobKind:
    """Classify a decoded blob by shape. Cheap: no decompression."""
    if not data:
        return BlobKind.EMPTY
    if tag in _FIXED16_TAGS and len(data) == 16:
        return BlobKind.FIXED16
    if data[:3] == b"\xff\xd8\xff":
        return BlobKind.JPEG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return BlobKind.PNG
    if _looks_like_keyed_dict(data):
        return BlobKind.KEYED_DICT
    env = parse_envelope(data)
    if env is not None:
        if env.version is None:
            return BlobKind.FLAGGED_ZSTD if env.codec == "zstd" else BlobKind.FLAGGED_PROTOBUF
        return {
            "zstd": BlobKind.ENVELOPE_ZSTD,
            "protobuf": BlobKind.ENVELOPE_PROTOBUF,
        }.get(env.codec, BlobKind.ENVELOPE_PLAIN)
    if _looks_like_qcompress(data):
        return BlobKind.QCOMPRESS
    if tag == "MediaTimemapBA" and len(data) >= 9 and (len(data) - 1) % 8 == 0:
        return BlobKind.TIMEMAP_COMPACT
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return BlobKind.UNKNOWN
    return BlobKind.TEXT if text.isprintable() else BlobKind.UNKNOWN


_MESSAGE_KINDS = frozenset(
    {
        BlobKind.ENVELOPE_PROTOBUF,
        BlobKind.ENVELOPE_ZSTD,
        BlobKind.ENVELOPE_PLAIN,
        BlobKind.FLAGGED_PROTOBUF,
        BlobKind.FLAGGED_ZSTD,
    }
)


@dataclass(frozen=True, slots=True)
class BlobCheck:
    """Result of fully decoding one blob (used for statistics).

    ``ok`` is false when the layout was recognised but the bytes did not decode (``error``
    says why). ``undecoded`` is true when the layout is not known well enough to decode at all;
    such blobs are carried raw, which is not a failure.
    """

    kind: BlobKind
    ok: bool
    error: str | None = None
    undecoded: bool = False


_MAX_NESTING = 4


def check(
    tag: str,
    data: bytes,
    *,
    max_output: int = DEFAULT_MAX_OUTPUT,
    budget: DecodeBudget | None = None,
) -> BlobCheck:
    """Classify and then fully decode a blob as far as its kind allows.

    Keyed dicts are decoded strictly, and their byte values that are themselves keyed dicts,
    messages or qCompress data are decoded too (up to four levels deep).
    """
    kind = classify(tag, data)
    try:
        if not _check(tag, kind, data, max_output, 0, budget):
            return BlobCheck(kind, True, undecoded=True)
    except BudgetExceededError as exc:
        return BlobCheck(kind, False, f"budget-exceeded: {exc}")
    except ZstdUnavailableError as exc:
        return BlobCheck(kind, False, f"zstd-unavailable: {exc}")
    except BlobError as exc:
        return BlobCheck(kind, False, str(exc))
    return BlobCheck(kind, True)


def _check(
    tag: str,
    kind: BlobKind,
    data: bytes,
    max_output: int,
    depth: int,
    budget: DecodeBudget | None = None,
) -> bool:
    """Decode ``data`` as ``kind``; False when the layout is not interpreted. Raises BlobError."""
    if kind in _MESSAGE_KINDS:
        payload = unwrap_message(data, max_output=max_output, budget=budget)
        decode_protobuf(payload, nested_depth=0)
    elif kind == BlobKind.QCOMPRESS:
        qcompress_decompress(data, max_output=max_output, budget=budget)
    elif kind == BlobKind.TIMEMAP_COMPACT:
        decode_timemap(data)
    elif kind == BlobKind.FIXED16:
        if tag in ("FrameRate", "MediaFrameRate"):
            decode_rate(data)
        elif tag == "Resolution":
            decode_resolution(data)
        elif tag == "MediaExtents":
            decode_extents(data)
        else:
            return False
    elif kind == BlobKind.KEYED_DICT:
        if tag == "MediaTimemapBA":
            decode_timemap(data)
        kd = decode_keyed_dict(data, strict=True)
        if depth < _MAX_NESTING:
            for entry in kd.entries:
                if not isinstance(entry.value, bytes) or not entry.value:
                    continue
                inner = classify(entry.key, entry.value)
                if inner in _NESTED_CHECKED:
                    try:
                        _check(entry.key, inner, entry.value, max_output, depth + 1, budget)
                    except ZstdUnavailableError:
                        raise
                    except BlobError as exc:
                        raise BlobError(f"{entry.key}: {exc}") from exc
    elif kind == BlobKind.UNKNOWN:
        return False
    return True


_NESTED_CHECKED = _MESSAGE_KINDS | {BlobKind.KEYED_DICT, BlobKind.QCOMPRESS}
