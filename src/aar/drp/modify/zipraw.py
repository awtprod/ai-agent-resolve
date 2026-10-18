"""Rewrite a zip archive with some members replaced, copying every other byte as it is.

:mod:`zipfile` cannot copy a member without recompressing it, so this module works on the raw
archive: untouched members (local header, compressed data, data descriptor) and the central
directory records are copied byte for byte, and only the replaced members are re-deflated. A
rewrite that replaces nothing returns the input bytes exactly.

Only the plain layout Resolve writes is accepted, so that "every other byte is copied" holds:
local entries back to back from offset 0, the central directory right after them, then the end
record (with its comment, if any). Zip64, multi-disk archives, encrypted members, compression
methods other than stored and deflate, duplicate names and anything between the parts are
refused with :class:`ZipLayoutError`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from aar.drp.errors import DrpError

__all__ = ["ZipEntry", "ZipLayoutError", "parse_layout", "rewrite"]

_LOCAL = struct.Struct("<IHHHHHIIIHH")  # 30 bytes
_CENTRAL = struct.Struct("<IHHHHHHIIIHHHHHII")  # 46 bytes
_END = struct.Struct("<IHHHHIIH")  # 22 bytes
_LOCAL_SIG = 0x04034B50
_CENTRAL_SIG = 0x02014B50
_END_SIG = 0x06054B50
_DESC_SIG = b"PK\x07\x08"
_ZIP64_LOCATOR = b"PK\x06\x07"
_STORED, _DEFLATED = 0, 8
_FLAG_ENCRYPTED = 0x1
_FLAG_DESCRIPTOR = 0x8
_DEFLATE_LEVEL = 6


class ZipLayoutError(DrpError):
    """The archive does not have the plain layout this writer can rewrite byte for byte."""


@dataclass(frozen=True, slots=True)
class ZipEntry:
    """One member: where its local entry and its central record sit in the archive."""

    name: str
    name_raw: bytes
    local_start: int
    data_start: int
    data_end: int
    local_end: int
    central_start: int
    central_end: int
    method: int
    flags: int
    descriptor_sig: bool


@dataclass(frozen=True, slots=True)
class _Layout:
    entries: list[ZipEntry]
    central_start: int
    end_start: int


def parse_layout(data: bytes) -> list[ZipEntry]:
    """The archive's members in file order; raises :class:`ZipLayoutError` on anything else."""
    return _parse(data).entries


def _parse(data: bytes) -> _Layout:
    end = data.rfind(struct.pack("<I", _END_SIG), max(0, len(data) - 22 - 0xFFFF))
    if end < 0 or end + 22 > len(data):
        raise ZipLayoutError("no end-of-central-directory record")
    _sig, disk, cd_disk, here, total, cd_size, cd_off, clen = _END.unpack_from(data, end)
    if end + 22 + clen != len(data):
        raise ZipLayoutError("bytes after the end record (or a bad comment length)")
    if disk or cd_disk or here != total:
        raise ZipLayoutError("multi-disk archives are not supported")
    if 0xFFFF in (here, total) or 0xFFFFFFFF in (cd_size, cd_off):
        raise ZipLayoutError("zip64 archives are not supported")
    if end >= 20 and data[end - 20 : end - 16] == _ZIP64_LOCATOR:
        raise ZipLayoutError("zip64 archives are not supported")
    if cd_off + cd_size != end:
        raise ZipLayoutError("the central directory does not end at the end record")
    entries: list[ZipEntry] = []
    pos = cd_off
    names: set[bytes] = set()
    for _ in range(total):
        if pos + 46 > end:
            raise ZipLayoutError("truncated central directory")
        rec = _CENTRAL.unpack_from(data, pos)
        (csig, _made, _need, flags, method, _t, _d, crc, csize, usize, nlen, xlen, mlen) = rec[:13]
        disk_start, offset = rec[13], rec[16]
        if csig != _CENTRAL_SIG:
            raise ZipLayoutError(f"bad central directory signature at {pos}")
        if 0xFFFFFFFF in (csize, usize, offset) or disk_start:
            raise ZipLayoutError("zip64 or multi-disk member")
        if flags & _FLAG_ENCRYPTED:
            raise ZipLayoutError("encrypted members are not supported")
        if method not in (_STORED, _DEFLATED):
            raise ZipLayoutError(f"compression method {method} is not supported")
        name_raw = data[pos + 46 : pos + 46 + nlen]
        central_end = pos + 46 + nlen + xlen + mlen
        if central_end > end:
            raise ZipLayoutError("truncated central directory record")
        if name_raw in names:
            raise ZipLayoutError(f"duplicate member name {name_raw!r}")
        names.add(name_raw)
        entries.append(
            _local(data, offset, name_raw, (flags, method, crc, csize, usize), (pos, central_end))
        )
        pos = central_end
    if pos != end:
        raise ZipLayoutError("central directory size does not match its records")
    ordered = sorted(entries, key=lambda e: e.local_start)
    expect = 0
    for e in ordered:
        if e.local_start != expect:
            raise ZipLayoutError(f"gap or overlap before member {e.name!r}")
        expect = e.local_end
    if expect != cd_off:
        raise ZipLayoutError("bytes between the last member and the central directory")
    return _Layout(ordered, cd_off, end)


def _local(
    data: bytes,
    offset: int,
    name_raw: bytes,
    central: tuple[int, int, int, int, int],
    central_span: tuple[int, int],
) -> ZipEntry:
    """The local entry at ``offset``, checked against its central record's fields."""
    flags, method, crc, csize, usize = central
    if offset + 30 > len(data):
        raise ZipLayoutError(f"local header of {name_raw!r} is past the end")
    lsig, _need, lflags, lmethod, _t, _d, _crc, _cs, _us, nlen, xlen = _LOCAL.unpack_from(
        data, offset
    )
    if lsig != _LOCAL_SIG:
        raise ZipLayoutError(f"bad local header signature for {name_raw!r}")
    if data[offset + 30 : offset + 30 + nlen] != name_raw:
        raise ZipLayoutError(f"local and central names differ for {name_raw!r}")
    if (lflags, lmethod) != (flags, method):
        raise ZipLayoutError(f"local and central headers disagree for {name_raw!r}")
    data_start = offset + 30 + nlen + xlen
    data_end = data_start + csize
    local_end = data_end
    descriptor_sig = False
    if flags & _FLAG_DESCRIPTOR:
        descriptor_sig = data[data_end : data_end + 4] == _DESC_SIG
        body = data_end + (4 if descriptor_sig else 0)
        if body + 12 > len(data):
            raise ZipLayoutError(f"truncated data descriptor for {name_raw!r}")
        if struct.unpack_from("<III", data, body) != (crc, csize, usize):
            raise ZipLayoutError(
                f"data descriptor disagrees with the central record of {name_raw!r}"
            )
        local_end = body + 12
    try:
        name = name_raw.decode("utf-8" if flags & 0x800 else "cp437")
    except UnicodeDecodeError as exc:
        raise ZipLayoutError(f"member name {name_raw!r} cannot be decoded") from exc
    return ZipEntry(
        name=name,
        name_raw=name_raw,
        local_start=offset,
        data_start=data_start,
        data_end=data_end,
        local_end=local_end,
        central_start=central_span[0],
        central_end=central_span[1],
        method=method,
        flags=flags,
        descriptor_sig=descriptor_sig,
    )


def _compress(method: int, content: bytes) -> bytes:
    if method == _STORED:
        return content
    comp = zlib.compressobj(_DEFLATE_LEVEL, zlib.DEFLATED, -15)
    return comp.compress(content) + comp.flush()


def rewrite(data: bytes, replacements: dict[str, bytes]) -> bytes:
    """``data`` with the named members' contents replaced; every other byte copied as it is."""
    layout = _parse(data)
    by_name = {e.name: e for e in layout.entries}
    missing = sorted(set(replacements) - set(by_name))
    if missing:
        raise ZipLayoutError(f"no member named {missing[0]!r}")
    out = bytearray()
    new_offset: dict[str, int] = {}
    new_sizes: dict[str, tuple[int, int, int]] = {}
    for e in layout.entries:
        new_offset[e.name] = len(out)
        content = replacements.get(e.name)
        if content is None:
            out += data[e.local_start : e.local_end]
            continue
        packed = _compress(e.method, content)
        crc = zlib.crc32(content) & 0xFFFFFFFF
        if max(len(packed), len(content)) >= 0xFFFFFFFF:
            raise ZipLayoutError(f"member {e.name!r} would need zip64")
        new_sizes[e.name] = (crc, len(packed), len(content))
        header = bytearray(data[e.local_start : e.data_start])
        struct.pack_into("<III", header, 14, crc, len(packed), len(content))
        out += header
        out += packed
        if e.flags & _FLAG_DESCRIPTOR:
            out += (_DESC_SIG if e.descriptor_sig else b"") + struct.pack(
                "<III", crc, len(packed), len(content)
            )
    cd_start = len(out)
    for e in sorted(layout.entries, key=lambda x: x.central_start):
        rec = bytearray(data[e.central_start : e.central_end])
        if e.name in new_sizes:
            struct.pack_into("<III", rec, 16, *new_sizes[e.name])
        offset = new_offset[e.name]
        if offset >= 0xFFFFFFFF:
            raise ZipLayoutError("the archive would need zip64")
        struct.pack_into("<I", rec, 42, offset)
        out += rec
    cd_size = len(out) - cd_start
    end = bytearray(data[layout.end_start :])
    struct.pack_into("<II", end, 12, cd_size, cd_start)
    out += end
    return bytes(out)
