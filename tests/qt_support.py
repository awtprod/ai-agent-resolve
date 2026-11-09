"""QuickTime files built byte by byte for the S18a timecode-reader tests, and seekable views.

Nothing here uses aar.media, so a bug in the reader cannot hide in the fixtures. The layout
copies the real CCN BRAW (aar-evidence/s18a-braw, never committed): ``wide``, then ``mdat``
(optionally with a 64-bit size), then ``moov`` at the end with five tracks (vide, tmcd, soun
and two meta), each with a data-handler ``hdlr`` (``alis``) in ``minf`` besides the media
``hdlr`` in ``mdia``, ``co64`` chunk offsets, and the tmcd sample (a big-endian u32 frame
number) at offset 4096.
"""

from __future__ import annotations

import bisect
import io
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

TMCD_AT = 4096  # where the real BRAW keeps its tmcd sample
NTSC_SCALE, NTSC_DURATION = 30000, 1001


def atom(kind: bytes, *parts: bytes, wide: bool = False) -> bytes:
    """One atom; ``wide`` writes size 1 and the 64-bit size after the type."""
    body = b"".join(parts)
    if wide:
        return struct.pack(">I4sQ", 1, kind, 16 + len(body)) + body
    return struct.pack(">I4s", 8 + len(body), kind) + body


def hdlr(subtype: bytes, component: bytes = b"mhlr") -> bytes:
    """A QuickTime handler: version/flags, component type, subtype, manufacturer, flags, mask,
    an empty Pascal name."""
    return atom(b"hdlr", bytes(4), component, subtype, bytes(12), b"\x00")


def tmcd_entry(
    flags: int = 0x2,
    timescale: int = NTSC_SCALE,
    duration: int = NTSC_DURATION,
    per_second: int = 30,
    *,
    kind: bytes = b"tmcd",
    size: int | None = None,
) -> bytes:
    body = bytes(6) + struct.pack(">H", 1) + bytes(4)
    body += struct.pack(">IIIBB", flags, timescale, duration, per_second, 0)
    return struct.pack(">I4s", size if size is not None else 8 + len(body), kind) + body


def stsd(*entries: bytes, count: int | None = None) -> bytes:
    n = len(entries) if count is None else count
    return atom(b"stsd", bytes(4), struct.pack(">I", n), *entries)


def stsc(*entries: tuple[int, int, int], count: int | None = None) -> bytes:
    """Sample-to-chunk: (first_chunk, samples_per_chunk, sample_description_index) rows."""
    n = len(entries) if count is None else count
    return atom(
        b"stsc", bytes(4), struct.pack(">I", n), *(struct.pack(">III", *e) for e in entries)
    )


def co64(*offsets: int, count: int | None = None) -> bytes:
    n = len(offsets) if count is None else count
    return atom(b"co64", bytes(4), struct.pack(">I", n), *(struct.pack(">Q", o) for o in offsets))


def stco(*offsets: int, count: int | None = None) -> bytes:
    n = len(offsets) if count is None else count
    return atom(b"stco", bytes(4), struct.pack(">I", n), *(struct.pack(">I", o) for o in offsets))


def elst(*entries: tuple[int, int, int, int], version: int = 0, count: int | None = None) -> bytes:
    """An edit list: (segment_duration, media_time, rate_integer, rate_fraction) rows, media_time
    signed (32-bit in version 0, 64-bit in version 1)."""
    n = len(entries) if count is None else count
    row = ">Iihh" if version == 0 else ">Qqhh"
    rows = (struct.pack(row, *e) for e in entries)
    return atom(b"elst", struct.pack(">B3x", version), struct.pack(">I", n), *rows)


def trak(
    handler: bytes,
    *stbl_children: bytes,
    minf_extra: Sequence[bytes] = (),
    edts: bytes | None = None,
    media_scale: int = NTSC_SCALE,
) -> bytes:
    """A track whose ``mdia/hdlr`` is ``handler``; ``minf`` carries the ``alis`` data handler
    first, as BRAW's does, so a reader that takes any ``hdlr`` sees ``alis``."""
    tkhd = atom(b"tkhd", bytes(84))
    mdhd = atom(b"mdhd", bytes(4), bytes(8), struct.pack(">II", media_scale, 1576576), bytes(4))
    minf = atom(
        b"minf",
        *minf_extra,
        hdlr(b"alis", b"dhlr"),
        atom(b"dinf", atom(b"dref", bytes(4), struct.pack(">I", 1), atom(b"alis", b"\0\0\0\1"))),
        atom(b"stbl", *stbl_children),
    )
    edits = [edts] if edts is not None else []
    return atom(b"trak", tkhd, *edits, atom(b"mdia", mdhd, hdlr(handler), minf))


@dataclass
class Movie:
    """The parts of a BRAW-like file, to change one at a time."""

    frame: int = 521420
    mdat_size: int = 8192  # bytes of mdat payload (the sample sits inside it)
    mdat_wide: bool = False  # size 1 and a 64-bit size (BRAW's 3.6 GB mdat fits 32 bits)
    tmcd_stbl: list[bytes] = field(default_factory=list)
    tracks_before: list[bytes] = field(default_factory=list)
    tracks_after: list[bytes] = field(default_factory=list)
    with_tmcd: bool = True
    tmcd_edts: bytes | None = None  # an edts atom for the tmcd track (none by default)
    tmcd_scale: int = NTSC_SCALE  # the tmcd track's mdhd timescale

    def tmcd_track(self, offset: int) -> bytes:
        children = self.tmcd_stbl or [
            stsd(tmcd_entry()),
            atom(b"stts", bytes(4), struct.pack(">III", 1, 1, 1576576)),
            atom(b"stsc", bytes(4), struct.pack(">IIII", 1, 1, 1, 1)),
            atom(b"stsz", bytes(4), struct.pack(">II", 4, 1)),
            co64(offset),
        ]
        return trak(
            b"tmcd",
            *children,
            minf_extra=[atom(b"gmhd", atom(b"gmin", bytes(16)))],
            edts=self.tmcd_edts,
            media_scale=self.tmcd_scale,
        )

    def moov(self, offset: int = TMCD_AT) -> bytes:
        video = trak(b"vide", stsd(atom(b"brvl", bytes(78))), atom(b"skip", b"\0\0"), co64(4100))
        sound = trak(b"soun", stsd(atom(b"in24", bytes(28))), co64(4100 + 2476032))
        meta = trak(b"meta", stsd(atom(b"mebx", bytes(16))), co64(6000))
        tracks = [*self.tracks_before, video]
        if self.with_tmcd:
            tracks.append(self.tmcd_track(offset))
        tracks += [sound, meta, meta, *self.tracks_after]
        return atom(b"moov", atom(b"mvhd", bytes(100)), *tracks, atom(b"meta", bytes(64)))

    def build(self) -> bytes:
        """``wide`` + ``mdat`` (the sample at :data:`TMCD_AT`) + ``moov``."""
        mdat_body = bytearray(self.mdat_size)
        at = TMCD_AT - 8 - (16 if self.mdat_wide else 8)  # after wide and the mdat header
        mdat_body[at : at + 4] = struct.pack(">I", self.frame)
        mdat = atom(b"mdat", bytes(mdat_body), wide=self.mdat_wide)
        return atom(b"wide") + mdat + self.moov()

    def big(self, size: int, sample_at: int) -> tuple[Sparse, int]:
        """The same movie with an mdat of ``size`` bytes (a 64-bit size) and the tmcd sample at
        ``sample_at``, as a :class:`Sparse` view (nothing that large is written anywhere)."""
        head = atom(b"wide") + struct.pack(">I4sQ", 1, b"mdat", 16 + size)
        moov = self.moov(sample_at)
        end = len(head) + size
        segments = {0: head, sample_at: struct.pack(">I", self.frame), end: moov}
        return Sparse(end + len(moov), segments), end + len(moov)


class Sparse(io.RawIOBase):
    """A read-only seekable view of a file of ``size`` bytes of which only ``segments``
    (offset -> bytes) are known. Reading any other byte raises ``LookupError`` (not an
    ``OSError``, so the reader under test cannot swallow it), unless ``fill`` gives zeros."""

    def __init__(self, size: int, segments: dict[int, bytes], *, fill: bool = False) -> None:
        self.size = size
        self.segments = sorted(segments.items())
        self.starts = [offset for offset, _ in self.segments]
        self.fill = fill
        self.pos = 0
        self.reads: list[tuple[int, int]] = []

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        start = self.pos
        end = self.size if n < 0 else min(self.size, start + n)
        self.reads.append((start, max(0, end - start)))
        out = bytearray()
        at = start
        while at < end:
            i = bisect.bisect_right(self.starts, at) - 1  # the last segment starting at or before
            if i >= 0 and at < self.starts[i] + len(self.segments[i][1]):
                o, d = self.segments[i]
                take = d[at - o : min(len(d), end - o)]
                out += take
                at += len(take)
                continue
            if not self.fill:
                raise LookupError(f"read of unknown byte {at}")
            nxt = self.starts[i + 1] if i + 1 < len(self.starts) else end
            out += bytes(min(end, nxt) - at)
            at = min(end, nxt)
        self.pos = end
        return bytes(out)

    @property
    def bytes_read(self) -> int:
        return sum(n for _, n in self.reads)


def retime_tmcd(path: Path, timescale: int, duration: int, per_second: int | None = None) -> None:
    """Rewrite the tmcd sample description of an ffmpeg-made file (the validator's dev.py
    repro): timescale, frameDuration and, if given, numberOfFrames. The entry is found as 'tmcd',
    6 reserved bytes, data reference 1."""
    data = bytearray(path.read_bytes())
    at = data.find(b"tmcd" + bytes(6) + b"\x00\x01")
    assert at > 0 and data.find(b"tmcd" + bytes(6) + b"\x00\x01", at + 1) == -1
    data[at + 20 : at + 28] = struct.pack(">II", timescale, duration)
    if per_second is not None:
        data[at + 28] = per_second
    path.write_bytes(bytes(data))


def edit_tmcd_elst(path: Path, media_time: int, rate: tuple[int, int] = (1, 0)) -> None:
    """Rewrite the first edit of the tmcd track's version-0 elst in an ffmpeg-made file (ffmpeg
    writes one edit, media_time 0, rate 1, on every track). Walks the atoms itself."""
    data = bytearray(path.read_bytes())

    def kids(start: int, end: int):  # type: ignore[no-untyped-def]
        while end - start >= 8:
            size, kind = struct.unpack(">I4s", data[start : start + 8])
            yield kind, start + 8, start + size
            start += size

    moov = next((b, e) for k, b, e in kids(0, len(data)) if k == b"moov")
    for kind, body, end in kids(*moov):
        if kind != b"trak":
            continue
        children = {k: (b, e) for k, b, e in kids(body, end)}
        mdia = dict((k, (b, e)) for k, b, e in kids(*children[b"mdia"]))
        if data[mdia[b"hdlr"][0] + 8 : mdia[b"hdlr"][0] + 12] != b"tmcd":
            continue
        elst_body = next(b for k, b, _ in kids(*children[b"edts"]) if k == b"elst")
        assert data[elst_body] == 0, "version 0 only"
        first = elst_body + 8
        data[first + 4 : first + 12] = struct.pack(">ihh", media_time, *rate)
        path.write_bytes(bytes(data))
        return
    raise AssertionError("no tmcd track")
