"""Edit a marker list blob: the ``BlobData`` of a ``Sm2SequenceLockableBlob`` (timeline markers)
or ``Sm2TiItemLockableBlob`` (clip markers) in ``project.xml``.

Layout (as decoded by :func:`aar.drp.reader._decode_markers`, after drp-format's
timeline-markers-blob.js, MIT): ``FieldsBlob`` keyed dict -> ``BlobData`` byte array -> envelope
-> protobuf ``{2: {1: entry, 1: entry, ...}}``; an entry is ``{1: frame, 2: body}``, the body
``[u32 version][u32 length] {1: detail}``, the detail ``{1: colour bit, 3: note, 3: duration,
3: name, 6: custom data}``.

Edits keep every byte they do not have to change: an updated entry has only its changed fields
re-encoded (undecoded fields stay in place), a deleted entry is cut out, and a new entry copies
the framing (body version) of the entries already in the list. Every enclosing length, the
envelope and the keyed dict are then re-encoded around the new bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from aar.drp import blobs
from aar.drp.diff_opaque import _marker_entry_residue
from aar.drp.modify import encode
from aar.drp.modify.encode import EncodeError
from aar.drp.reader import MARKER_COLORS
from aar.drp.scan import parse_int

__all__ = [
    "COLOR_BITS",
    "DEFAULT_BODY_VERSION",
    "MarkerEntry",
    "MarkerList",
    "props",
    "undecoded_part",
]

COLOR_BITS = {name: bit for bit, name in MARKER_COLORS.items()}
# Every marker body in the corpus (57 of 57, Resolve 19-21) has version 2.
DEFAULT_BODY_VERSION = 2


def props(
    frame: int, color: str | None, name: str, note: str, duration: int | None, custom: str
) -> dict[str, Any]:
    """A marker's decoded values in the order the diff compares them."""
    return {
        "frame": frame,
        "color": color,
        "name": name,
        "note": note,
        "duration": duration,
        "custom_data": custom,
    }


def _first(fields: list[blobs.PbField], number: int, wire: int) -> blobs.PbField | None:
    return next((f for f in fields if f.number == number and f.wire == wire), None)


@dataclass(slots=True)
class MarkerEntry:
    """One entry: its encoded bytes (the value of list field 1) and its decoded values."""

    raw: bytes
    values: dict[str, Any]
    body_version: int
    odd: tuple[Any, ...] | None = None
    """What the S2 diff keeps of this entry beyond its decoded values (undecoded fields,
    non-standard spellings), or ``None`` when it holds nothing else. The gate compares it raw,
    so an edit must leave it as it is."""

    @classmethod
    def decode(cls, raw: bytes) -> MarkerEntry:
        parts = blobs.decode_protobuf(raw, nested_depth=0)
        frame = _first(parts, 1, 0)
        body = _first(parts, 2, 2)
        if frame is None or body is None or not isinstance(body.value, bytes):
            raise EncodeError("marker entry without frame or body")
        env = blobs.parse_envelope(body.value)
        if env is None or env.version is None or env.codec != "none":
            raise EncodeError("marker body is not [u32 version][u32 length] protobuf")
        detail = _first(blobs.decode_protobuf(env.payload, nested_depth=0), 1, 2)
        if detail is None or not isinstance(detail.value, bytes):
            raise EncodeError("marker body without a detail message")
        fields = blobs.decode_protobuf(detail.value, nested_depth=0)
        color = _first(fields, 1, 0)
        strings = [f for f in fields if f.number == 3 and isinstance(f.value, bytes)]
        custom = _first(fields, 6, 2)
        texts = [s.value.decode("utf-8", "replace") for s in strings if isinstance(s.value, bytes)]
        texts += ["", "", ""]
        parsed = parse_int(texts[1])
        bit = color.value if color is not None else None
        values = props(
            frame.value if isinstance(frame.value, int) else 0,
            MARKER_COLORS.get(bit) if isinstance(bit, int) else None,
            texts[2],
            texts[0],
            parsed if parsed is not None and parsed >= 0 else None,
            custom.value.decode("utf-8", "replace")
            if custom is not None and isinstance(custom.value, bytes)
            else "",
        )
        return cls(raw, values, env.version, undecoded_part(raw))

    @classmethod
    def new(cls, values: dict[str, Any], body_version: int) -> MarkerEntry:
        detail = (
            encode.pb_varint(1, COLOR_BITS[values["color"]])
            + encode.pb_len(3, values["note"].encode("utf-8"))
            + encode.pb_len(3, str(values["duration"]).encode("ascii"))
            + encode.pb_len(3, values["name"].encode("utf-8"))
        )
        if values["custom_data"]:
            detail += encode.pb_len(6, values["custom_data"].encode("utf-8"))
        inner = encode.pb_len(1, detail)
        body = struct.pack(">II", body_version, len(inner)) + inner
        raw = encode.pb_varint(1, values["frame"]) + encode.pb_len(2, body)
        return cls(raw, dict(values), body_version)

    def updated(self, changes: dict[str, Any]) -> MarkerEntry:
        """This entry with some values changed; every other byte is kept."""
        raw = self.raw
        detail_changes = {k: v for k, v in changes.items() if k != "frame"}
        if detail_changes:
            parts = blobs.decode_protobuf(raw, nested_depth=0)
            body = _first(parts, 2, 2)
            assert body is not None and isinstance(body.value, bytes)
            env = blobs.parse_envelope(body.value)
            assert env is not None
            inner = env.payload
            detail_field = _first(blobs.decode_protobuf(inner, nested_depth=0), 1, 2)
            assert detail_field is not None and isinstance(detail_field.value, bytes)
            detail = _update_detail(detail_field.value, detail_changes)
            inner = encode.replace_field(inner, detail_field, detail)
            new_body = encode.envelope(body.value, inner)
            raw = encode.replace_field(raw, body, new_body)
        if "frame" in changes:
            parts = blobs.decode_protobuf(raw, nested_depth=0)
            frame = _first(parts, 1, 0)
            assert frame is not None
            raw = encode.replace_field(raw, frame, changes["frame"])
        values = dict(self.values)
        values.update(changes)
        return MarkerEntry(raw, values, self.body_version, undecoded_part(raw))


def undecoded_part(raw: bytes) -> tuple[Any, ...] | None:
    """The part of a marker entry the gate compares raw (see :attr:`MarkerEntry.odd`).

    Uses the diff's own residue (``aar.drp.diff_opaque``), so the modify path refuses exactly
    what the post-write gate would.
    """
    [entry] = blobs.decode_protobuf(encode.pb_len(1, raw), nested_depth=0)
    residue = _marker_entry_residue(entry)
    rest_entry, _strings, _version, _codec, rest_inner, rest_detail, noncanonical = residue
    return residue if rest_entry or rest_inner or rest_detail or noncanonical else None


def _update_detail(detail: bytes, changes: dict[str, Any]) -> bytes:
    """Re-encode only the changed fields of a detail message, back to front."""
    fields = blobs.decode_protobuf(detail, nested_depth=0)
    strings = [f for f in fields if f.number == 3 and isinstance(f.value, bytes)]
    if len(strings) < 3:
        raise EncodeError("marker detail has fewer than three strings (note, duration, name)")
    targets: list[tuple[blobs.PbField, bytes | int]] = []
    if "color" in changes:
        color = _first(fields, 1, 0)
        if color is None:
            raise EncodeError("marker detail has no colour field to change")
        targets.append((color, COLOR_BITS[changes["color"]]))
    for key, index in (("note", 0), ("duration", 1), ("name", 2)):
        if key in changes:
            value = changes[key]
            text = str(value) if key == "duration" else value
            targets.append((strings[index], text.encode("utf-8")))
    append = b""
    if "custom_data" in changes:
        custom = _first(fields, 6, 2)
        data = changes["custom_data"].encode("utf-8")
        if custom is not None:
            targets.append((custom, data))
        elif data:
            append = encode.pb_len(6, data)
    for f, value in sorted(targets, key=lambda t: t[0].offset, reverse=True):
        detail = encode.replace_field(detail, f, value)
    return detail + append


@dataclass(slots=True)
class MarkerList:
    """A decoded marker list with enough of its framing to write it back."""

    kd_bytes: bytes
    kd_entry: blobs.KdEntry
    blob_data: bytes
    payload: bytes
    holder: blobs.PbField
    entries: list[MarkerEntry]

    @classmethod
    def decode(cls, fields_blob: bytes, budget: blobs.DecodeBudget) -> MarkerList:
        kd = blobs.decode_keyed_dict(fields_blob, strict=True)
        entry = kd.entry("BlobData")
        if entry is None or not isinstance(entry.value, bytes):
            raise EncodeError("the marker blob has no BlobData (no marker list yet)")
        blob_data = entry.value
        if blobs.parse_envelope(blob_data) is None:
            raise EncodeError("BlobData is not an envelope")
        encode.check_envelope_payload(blob_data)
        payload = blobs.unwrap_message(blob_data, budget=budget)
        top = blobs.decode_protobuf(payload, nested_depth=0)
        if len(top) != 1 or top[0].number != 2 or not isinstance(top[0].value, bytes):
            raise EncodeError("the marker list does not have exactly one holder field 2")
        holder = top[0]
        assert isinstance(holder.value, bytes)
        entries = []
        for f in blobs.decode_protobuf(holder.value, nested_depth=0):
            if f.number != 1 or f.wire != 2 or not isinstance(f.value, bytes):
                raise EncodeError(f"unexpected field {f.number} in the marker list")
            if not encode.canonical(f, holder.value):
                raise EncodeError("a marker entry is not canonically encoded")
            entries.append(MarkerEntry.decode(f.value))
        return cls(fields_blob, entry, blob_data, payload, holder, entries)

    def body_version(self) -> int:
        versions = {e.body_version for e in self.entries}
        if len(versions) > 1:
            raise EncodeError(f"marker bodies disagree on their version: {sorted(versions)}")
        return versions.pop() if versions else DEFAULT_BODY_VERSION

    def encode(self) -> bytes:
        """The whole ``FieldsBlob`` keyed dict with the current entries."""
        content = b"".join(encode.pb_len(1, e.raw) for e in self.entries)
        payload = encode.replace_field(self.payload, self.holder, content)
        blob_data = encode.envelope(self.blob_data, payload)
        return encode.replace_kd_bytes(self.kd_bytes, self.kd_entry, blob_data)

    def values(self) -> list[dict[str, Any]]:
        return [dict(e.values) for e in self.entries]
