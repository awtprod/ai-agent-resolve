"""Edit the blobs of a media clip's rows: decode one layer, change values, encode it again.

Decoding is :mod:`aar.drp.blobs`; encoding is :mod:`aar.drp.modify.encode` (one implementation of
each). A blob that does not decode the way the template's kind says is a ``bad_template``
refusal, never a guess: every decoder here takes only bytes, and an encoding error (a value the
template's field type cannot hold, a keyed-dict type the encoder does not write) is a typed
refusal too. :func:`layers` walks every decodable layer of a value (envelopes, zstd, keyed dicts,
protobuf) within fixed bounds; the writer uses it to prove that no template id is left anywhere
in what it wrote, and a layer it cannot open (a zstd frame that does not decompress) is a
refusal, never skipped.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.modify import encode
from aar.drp.modify.encode import EncodeError

__all__ = [
    "Fields",
    "KdEntries",
    "frame_codec",
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
#: The values each scalar keyed-dict type holds: ``kd_set`` refuses a value outside them, so a
#: derived number too large for the template's field is a refusal, not an encoder crash.
_KD_INT_RANGES = {
    blobs.KD_INT32: (-(2**31), 2**31 - 1),
    blobs.KD_UINT32: (0, 2**32 - 1),
    blobs.KD_INT64: (-(2**63), 2**63 - 1),
    blobs.KD_UINT64: (0, 2**64 - 1),
}
_KD_FLOATS = frozenset({blobs.KD_DOUBLE, blobs.KD_FLOAT})


def _bad(what: str, why: str) -> MediaPoolError:
    return MediaPoolError(
        "bad_template",
        f"{what}: {why}",
        "use a template clip from a project Resolve saved, of the same kind",
    )


def _bytes(data: object, what: str) -> bytes:
    if not isinstance(data, bytes):
        raise _bad(what, f"is {type(data).__name__}, not a blob")
    return data


def _encode_refused(what: str, exc: EncodeError) -> MediaPoolError:
    return _bad(what, f"cannot be encoded again ({exc})")


def pb_decode(data: object, what: str) -> Fields:
    data = _bytes(data, what)
    try:
        fields = blobs.decode_protobuf(data, nested_depth=0, max_fields=10_000)
    except DrpError as exc:
        raise _bad(what, f"not a protobuf message ({exc})") from None
    return [[f.number, f.wire, f.value] for f in fields]


def pb_encode(fields: Fields, what: str = "a protobuf message") -> bytes:
    try:
        return encode.protobuf([(n, w, v) for n, w, v in fields])
    except EncodeError as exc:
        raise _encode_refused(what, exc) from None


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


def kd_decode(data: object, what: str) -> KdEntries:
    data = _bytes(data, what)
    try:
        kd = blobs.decode_keyed_dict(data, strict=True)
    except DrpError as exc:
        raise _bad(what, f"not a keyed dict ({exc})") from None
    if kd.header != 1:
        raise _bad(what, f"keyed dict header {kd.header}, expected 1")
    return [[e.key, e.type, e.subtype, e.value] for e in kd.entries]


def kd_encode(entries: KdEntries, what: str = "a keyed dict") -> bytes:
    try:
        return encode.keyed_dict([(k, t, n, v) for k, t, n, v in entries])
    except EncodeError as exc:
        raise _encode_refused(what, exc) from None


def kd_get(entries: KdEntries, key: str) -> list[Any] | None:
    hits = [e for e in entries if e[0] == key]
    if len(hits) > 1:
        raise _bad(f"key {key!r}", "appears twice in one keyed dict")
    return hits[0] if hits else None


def kd_set(entries: KdEntries, key: str, value: Any, what: str) -> None:
    """Give an existing entry a new value of its own type. A template entry whose type cannot
    hold that kind of value is a ``bad_template`` refusal; a derived number outside the range of
    the entry's type is ``cannot_derive``."""
    entry = kd_get(entries, key)
    if entry is None:
        raise _bad(what, f"has no {key!r}")
    vtype = entry[1]
    if isinstance(value, bool):
        fits = vtype == blobs.KD_BOOL
    elif isinstance(value, int):
        fits = vtype in _KD_INT_RANGES
        low, high = _KD_INT_RANGES.get(vtype, (0, -1))
        if fits and not low <= value <= high:
            raise MediaPoolError(
                "cannot_derive",
                f"{what}.{key}: the derived value {value} does not fit the template's field "
                f"(type 0x{vtype:x}, {low}..{high})",
                "check the file's facts; a value this large is not one Resolve stores",
            )
    elif isinstance(value, float):
        fits = vtype in _KD_FLOATS
    elif isinstance(value, str):
        fits = vtype == blobs.KD_STRING
    elif isinstance(value, bytes):
        fits = vtype == blobs.KD_BYTES
    else:
        fits = False
    if not fits:
        raise _bad(
            what, f"{key!r} is of type 0x{vtype:x}, which cannot hold {type(value).__name__}"
        )
    entry[3] = value


def message_of(blob: object, what: str) -> tuple[bytes, bytes]:
    """``(original blob, protobuf payload)`` of an enveloped message (zstd or plain)."""
    blob = _bytes(blob, what)
    env = blobs.parse_envelope(blob)
    if env is None or env.codec not in ("zstd", "protobuf", "none") or env.version is None:
        raise _bad(what, "is not an enveloped message")
    try:
        return blob, blobs.unwrap_message(blob, max_output=MAX_BLOB_OUTPUT)
    except DrpError as exc:
        raise _bad(what, f"does not decode ({exc})") from None


#: Resolve frames a message of at most this many bytes as plain protobuf (codec 0x80) and a
#: longer one as zstd (0x81): every envelope of the answer key's media rows follows it (plain up
#: to 64 bytes, zstd from 66; no message of 65 bytes was seen, so that length is refused).
PLAIN_MAX = 64
_UNSEEN_LENGTHS = frozenset({65})


def frame_codec(original_codec: str, payload: bytes, what: str) -> str:
    """The codec Resolve gives a message of ``payload``'s length, for an envelope whose
    template had ``original_codec``; ``none`` (the codec-less framing) is kept as it is."""
    if original_codec not in ("zstd", "protobuf"):
        return original_codec
    if len(payload) in _UNSEEN_LENGTHS:
        raise MediaPoolError(
            "cannot_derive",
            f"{what}: a message of {len(payload)} bytes, between the longest Resolve was seen "
            f"to store plain ({PLAIN_MAX}) and the shortest it compressed",
            "import the file in Resolve by hand; its framing for this length is not known",
        )
    return "protobuf" if len(payload) <= PLAIN_MAX else "zstd"


def rewrap(original: bytes, payload: bytes, what: str) -> bytes:
    """``payload`` framed the way ``original`` was (same version and form), with the codec
    Resolve uses for its length (:func:`frame_codec`)."""
    env = blobs.parse_envelope(original)
    if env is None:
        raise _bad(what, "is not an envelope")
    codec = frame_codec(env.codec, payload, what)
    try:
        return encode.envelope(
            original, payload, codec=codec if env.codec in ("zstd", "protobuf") else None
        )
    except DrpError as exc:  # EncodeError is a DrpError
        raise _bad(what, f"cannot be framed again ({exc})") from None


def edit_message(blob: object, what: str, change: Callable[[Fields], None]) -> bytes:
    """Decode an enveloped protobuf message, let ``change`` edit its fields, frame it again."""
    original, payload = message_of(blob, what)
    fields = pb_decode(payload, what)
    change(fields)
    return rewrap(original, pb_encode(fields, what), what)


def edit_kd(blob: object, what: str, change: Callable[[KdEntries], None]) -> bytes:
    entries = kd_decode(blob, what)
    change(entries)
    return kd_encode(entries, what)


def layers(value: object) -> Iterator[bytes]:
    """Every layer of ``value`` as bytes: the value itself, and, as far as they decode, the
    payloads of envelopes (zstd decompressed, at most 1 MiB each), the values of keyed dicts and
    the length-delimited fields of protobuf messages, at most 10 layers deep and 4096 layers in
    all. Text is yielded UTF-8 encoded. More layers, or a zstd layer that does not decompress,
    is a ``bad_template`` refusal: a layer that cannot be searched could hide a template id."""
    # Breadth first, so every value is first reached at the shallowest depth it occurs at and
    # dedupe by content cannot leave a value unexpanded that a shallower path would expand.
    count = 0
    queue: deque[tuple[bytes, int]] = deque()
    if isinstance(value, str):
        queue.append((value.encode("utf-8", "surrogatepass"), 0))
    elif isinstance(value, bytes):
        queue.append((value, 0))
    seen: set[bytes] = set()
    while queue:
        data, depth = queue.popleft()
        if data in seen:
            continue
        seen.add(data)
        count += 1
        if count > _MAX_WALKED:
            raise _bad("a blob", f"has more than {_MAX_WALKED} layers")
        yield data
        if len(data) < 2:
            continue
        children = [c for c in _children(data) if c not in seen]
        if children and depth >= _MAX_LAYERS:
            # A layer that would open further cannot be searched: refused, never skipped.
            raise _bad("a blob", f"has layers more than {_MAX_LAYERS} deep")
        queue.extend((child, depth + 1) for child in children)


def _children(data: bytes) -> list[bytes]:
    out: list[bytes] = []
    env = blobs.parse_envelope(data)
    if env is not None:
        # Only zstd can fail to unwrap (the other framings hold their payload as is); a layer
        # that cannot be opened cannot be searched, so it is refused rather than skipped.
        # The zstd backend can return nothing for a truncated frame instead of failing, so a
        # frame must first be exactly one complete frame.
        try:
            if env.codec == "zstd":
                encode.check_single_frame(env.payload)
            out.append(blobs.unwrap_message(data, max_output=MAX_BLOB_OUTPUT))
        except DrpError as exc:  # EncodeError is a DrpError
            raise _bad("a blob", f"holds a zstd layer that does not decompress ({exc})") from None
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
        # Bounded by the data itself (a field takes at least two bytes): a field cap here would
        # leave a message with more fields unwalked; the walk's own bound refuses it instead.
        fields = blobs.decode_protobuf(data, nested_depth=0, max_fields=len(data))
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
        for form in (
            text.encode("utf-8", "surrogatepass"),
            text.encode("utf-16-be", "surrogatepass"),
            text.encode("utf-16-le", "surrogatepass"),
        )
    ]
    for layer in layers(value):
        for form, label in encoded:
            if form in layer:
                return label
    return None
