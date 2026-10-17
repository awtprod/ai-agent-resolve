"""Byte-level builders shared by the ``aar.drp`` tests (encoders written from the spec)."""

from __future__ import annotations

import struct

HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<!--DbAppVer="{app}" DbPrjVer="{prj}"-->\n'


def xml(body: str, app: str = "20.0.0.0049", prj: int = 15) -> str:
    """An XML member with Resolve's two header lines."""
    return HEADER.format(app=app, prj=prj) + body


def qstr(text: str) -> bytes:
    raw = text.encode("utf-16-be")
    return struct.pack(">I", len(raw)) + raw


def kd(entries: list[tuple[str, int, bytes]]) -> bytes:
    """Encode a keyed dict from (key, type, encoded value payload) triples."""
    out = struct.pack(">II", 1, len(entries))
    for key, vtype, payload in entries:
        out += qstr(key) + struct.pack(">IB", vtype, 0) + payload
    return out


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def pb_len(field: int, payload: bytes) -> bytes:
    return varint(field << 3 | 2) + varint(len(payload)) + payload


def pb_var(field: int, value: int) -> bytes:
    return varint(field << 3) + varint(value)


def zstd_raw_frame(payload: bytes) -> bytes:
    """A zstd frame made of one raw block (what Resolve writes for small payloads)."""
    assert len(payload) <= 255
    block = ((len(payload) << 3) | 1).to_bytes(3, "little")
    return b"\x28\xb5\x2f\xfd" + bytes([0x20, len(payload)]) + block + payload


def envelope(version: int, codec: int, payload: bytes) -> bytes:
    body = bytes([codec]) + payload
    return struct.pack(">II", version, len(body)) + body
