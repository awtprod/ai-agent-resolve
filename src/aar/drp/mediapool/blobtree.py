"""Edit the blobs of a media clip's rows: decode one layer, change values, encode it again.

Decoding is :mod:`aar.drp.blobs`; encoding is :mod:`aar.drp.modify.encode` (one implementation of
each). A blob that does not decode the way the template's kind says is a ``bad_template``
refusal, never a guess. :func:`layers` walks every decodable layer of a value (envelopes, zstd,
keyed dicts, protobuf) within fixed bounds; the writer uses it to prove that no template id is
left anywhere in what it wrote.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.modify import encode

__all__ = [
    "Fields",
    "KdEntries",
    "kd_decode",
    "kd_encode",
    "kd_get",
    "kd_set",
    "layers",
    "message_of",
    "pb_decode",
    "pb_encode",
    "pb_get",
    "pb_remove",
    "rewrap",
]

#: A protobuf message as mutable ``[number, wire, value]`` lists, in order.
Fields = list[list[Any]]
#: A keyed dict as mutable ``[key, type, null flag, value]`` lists, in order.
KdEntries = list[list[Any]]

#: The most a blob of a media row may decompress to (the largest seen is under 4 KiB).
MAX_BLOB_OUTPUT = 1024 * 1024
_MAX_LAYERS = 10
_MAX_WALKED = 4096


def _bad(what: str, why: str) -> MediaPoolError:
    return MediaPoolError(
        "bad_template",
        f"{what}: {why}",
        "use a template clip from a project Resolve saved, of the same kind",
    )


def pb_decode(data: bytes, what: str) -> Fields:
    try:
        fields = blobs.decode_protobuf(data, nested_depth=0, max_fields=10_000)
    except DrpError as exc:
        raise _bad(what, f"not a protobuf message ({exc})") from None
    return [[f.number, f.wire, f.value] for f in fields]


def pb_encode(fields: Fields) -> bytes:
    return encode.protobuf([(n, w, v) for n, w, v in fields])


def pb_get(fields: Fields, number: int, what: str) -> list[Any]:
    """The one field ``number``; a missing or repeated field is a ``bad_template`` refusal."""
    hits = [f for f in fields if f[0] == number]
    if len(hits) != 1:
        raise _bad(what, f"expected one field {number}, found {len(hits)}")
    return hits[0]


def pb_remove(fields: Fields, number: int) -> int:
    """Remove every field ``number``; returns how many there were."""
    before = len(fields)
    fields[:] = [f for f in fields if f[0] != number]
    return before - len(fields)


def kd_decode(data: bytes, what: str) -> KdEntries:
    try:
        kd = blobs.decode_keyed_dict(data, strict=True)
    except DrpError as exc:
        raise _bad(what, f"not a keyed dict ({exc})") from None
    if kd.header != 1:
        raise _bad(what, f"keyed dict header {kd.header}, expected 1")
    return [[e.key, e.type, e.subtype, e.value] for e in kd.entries]


def kd_encode(entries: KdEntries) -> bytes:
    return encode.keyed_dict([(k, t, n, v) for k, t, n, v in entries])


def kd_get(entries: KdEntries, key: str) -> list[Any] | None:
    hits = [e for e in entries if e[0] == key]
    if len(hits) > 1:
        raise _bad(f"key {key!r}", "appears twice in one keyed dict")
    return hits[0] if hits else None


def kd_set(entries: KdEntries, key: str, value: Any, what: str) -> None:
    """Give an existing entry a new value of its own type."""
    entry = kd_get(entries, key)
    if entry is None:
        raise _bad(what, f"has no {key!r}")
    entry[3] = value


def message_of(blob: object, what: str) -> tuple[bytes, bytes]:
    """``(original blob, protobuf payload)`` of an enveloped message (zstd or plain)."""
    if not isinstance(blob, bytes):
        raise _bad(what, "is not a blob")
    env = blobs.parse_envelope(blob)
    if env is None or env.codec not in ("zstd", "protobuf", "none") or env.version is None:
        raise _bad(what, "is not an enveloped message")
    try:
        return blob, blobs.unwrap_message(blob, max_output=MAX_BLOB_OUTPUT)
    except DrpError as exc:
        raise _bad(what, f"does not decode ({exc})") from None


def rewrap(original: bytes, payload: bytes, what: str) -> bytes:
    """``payload`` framed the way ``original`` was (same version and codec)."""
    try:
        return encode.envelope(original, payload)
    except DrpError as exc:
        raise _bad(what, f"cannot be framed again ({exc})") from None


def edit_message(blob: object, what: str, change: Callable[[Fields], None]) -> bytes:
    """Decode an enveloped protobuf message, let ``change`` edit its fields, frame it again."""
    original, payload = message_of(blob, what)
    fields = pb_decode(payload, what)
    change(fields)
    return rewrap(original, pb_encode(fields), what)


def edit_kd(blob: object, what: str, change: Callable[[KdEntries], None]) -> bytes:
    if not isinstance(blob, bytes):
        raise _bad(what, "is not a blob")
    entries = kd_decode(blob, what)
    change(entries)
    return kd_encode(entries)


def layers(value: object) -> Iterator[bytes]:
    """Every layer of ``value`` as bytes: the value itself, and, as far as they decode, the
    payloads of envelopes (zstd decompressed, at most 1 MiB each), the values of keyed dicts and
    the length-delimited fields of protobuf messages, at most 10 layers deep and 4096 layers in
    all. Text is yielded UTF-8 encoded."""
    count = 0
    stack: list[tuple[bytes, int]] = []
    if isinstance(value, str):
        stack.append((value.encode("utf-8", "surrogatepass"), 0))
    elif isinstance(value, bytes):
        stack.append((value, 0))
    seen: set[bytes] = set()
    while stack:
        data, depth = stack.pop()
        if data in seen:
            continue
        seen.add(data)
        count += 1
        if count > _MAX_WALKED:
            raise _bad("a blob", f"has more than {_MAX_WALKED} layers")
        yield data
        if depth >= _MAX_LAYERS or len(data) < 2:
            continue
        stack.extend((child, depth + 1) for child in _children(data))


def _children(data: bytes) -> list[bytes]:
    out: list[bytes] = []
    env = blobs.parse_envelope(data)
    if env is not None:
        # Bytes that only look like an envelope are not decoded: they are still searched raw.
        with contextlib.suppress(DrpError):
            out.append(blobs.unwrap_message(data, max_output=MAX_BLOB_OUTPUT))
    if len(data) >= 8 and data[:4] == b"\x00\x00\x00\x01":
        try:
            kd = blobs.decode_keyed_dict(data, strict=True)
        except DrpError:
            kd = None
        if kd is not None:
            for entry in kd.entries:
                if isinstance(entry.value, bytes):
                    out.append(entry.value)
                elif isinstance(entry.value, str):
                    out.append(entry.value.encode("utf-8", "surrogatepass"))
    try:
        fields = blobs.decode_protobuf(data, nested_depth=0, max_fields=4096)
    except DrpError:
        fields = []
    out.extend(f.value for f in fields if f.wire == 2 and isinstance(f.value, bytes) and f.value)
    return out


def contains_text(value: object, needles: dict[str, str]) -> str | None:
    """The first of ``needles`` (text -> label) found in any layer of ``value``, in UTF-8,
    UTF-16BE or UTF-16LE; None if none is."""
    encoded = [
        (form, label)
        for text, label in needles.items()
        for form in (text.encode(), text.encode("utf-16-be"), text.encode("utf-16-le"))
    ]
    for layer in layers(value):
        for form, label in encoded:
            if form in layer:
                return label
    return None
