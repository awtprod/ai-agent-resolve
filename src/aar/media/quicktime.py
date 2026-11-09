"""Read a QuickTime or MP4 file's start timecode from its ``tmcd`` track, without ffprobe.

BRAW (the CCN camera format) is a plain QuickTime container: ``wide`` and ``mdat`` at the
start, ``moov`` at the end, a ``tmcd`` track whose first sample is the start timecode as a
big-endian signed 32-bit frame number. :func:`read_timecode` finds it by seeking: it walks the
top-level atoms to ``moov`` (wherever it is), takes the first ``trak`` whose ``mdia/hdlr``
subtype is ``tmcd``, reads the ``stsd`` entry that ``stsc``'s first row gives chunk 1 (entry 1
when there is one entry and no ``stsc``) for flags, timescale, frameDuration and
numberOfFrames, the first chunk offset from ``stco`` or ``co64`` (the first sample starts its
first chunk), and the four bytes there. It never reads the whole file: a few hundred bytes for a
normal movie.

Every size and offset is checked against its parent atom and the file length, ``moov`` may be
at most :data:`MAX_MOOV_BYTES`, and at most :data:`MAX_ATOMS` atom headers are read, so a
malformed file ends in a typed :class:`TimecodeUnreadable` (``code`` says why), never another
exception and never a hang. Stdlib only.

The sample is a signed 32-bit frame count (QuickTime lets flag bit 2 allow negatives), used as
given: for drop-frame (flag bit 0) it counts real frames (01:00:00;00 at 29.97 is 107892). A
negative count is refused (``negative_timecode``) whatever the flags, since no FCPXML asset can
start before zero; read unsigned, -1 would wrap into an unrelated positive timecode. The rate is
timescale/frameDuration. A track whose numberOfFrames is not that rate rounded is refused
(``unsupported``) rather than rescaled, and so is drop-frame at a rate other than 29.97 or
59.94. With the 24-hour flag (bit 1) a count past 24 hours wraps, as ffmpeg's timecode string
does; without it, it is refused. A tmcd track whose first edit (``edts/elst``) is not trivial
(media_time 0 at rate exactly 1: a trimmed start, an empty edit, another rate) is refused as
``unsupported`` rather than applied; the video track's edits are not read.
"""

from __future__ import annotations

import os
import stat
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import IO, Any, Literal

from aar.media.probe import DROP_FRAME_RATES, frames_timecode, snap_rate
from aar.media.tools import MediaError, require_regular_file

#: The largest ``moov`` read; a longer one is refused as ``too_large``.
MAX_MOOV_BYTES = 64 * 1024 * 1024
#: The most atom headers one read visits (top level and inside ``moov`` together).
MAX_ATOMS = 100_000
#: The most top-level atoms one walk of the file visits. A real movie has about ten (ftyp/wide,
#: mdat, moov, free); each costs a seek, and a cold disk reads a page per seek, so this bounds
#: physical I/O to a few MB however the atoms are spread.
MAX_TOP_LEVEL_ATOMS = 1000
#: The most bytes one read takes from the file; past it the read is refused as ``too_large``.
#: Reads are unbuffered, so this bounds what reaches the disk, not just what is parsed.
MAX_READ_BYTES = 8 * 1024 * 1024
#: Suffixes the plan producer reads with this module when ffprobe gives no timecode.
QUICKTIME_SUFFIXES = (".braw", ".mov", ".mp4")

#: tmcd flags (QuickTime File Format, timecode sample description).
FLAG_DROP_FRAME = 0x1
FLAG_24_HOUR = 0x2
_TMCD_ENTRY = 34  # the tmcd sample description: 34 bytes before any child atoms

Reason = Literal[
    "not_quicktime",
    "no_timecode",
    "malformed",
    "unsupported",
    "negative_timecode",
    "too_large",
    "io",
]
_TOOL = "quicktime"


class TimecodeUnreadable(MediaError):
    """The file has no start timecode this reader can use. ``code`` is one of:
    ``not_quicktime`` (no ``moov`` atom), ``no_timecode`` (no tmcd track, or one without a
    sample), ``malformed`` (a size or offset that does not fit), ``unsupported`` (a tmcd track
    this reader will not guess at), ``negative_timecode`` (the signed sample is below zero),
    ``too_large`` (``moov`` past :data:`MAX_MOOV_BYTES`) or ``io`` (the file could not be
    read)."""

    def __init__(self, message: str, *, code: Reason) -> None:
        super().__init__(message, tool=_TOOL)
        self.code: Reason = code


@dataclass(frozen=True)
class QuickTimeTimecode:
    """A tmcd track's start timecode: ``frames`` from 00:00:00:00 at ``rate``."""

    frames: int
    rate: Fraction  # timescale / frameDuration, reduced
    drop_frame: bool
    wraps_24h: bool
    flags: int
    timescale: int
    frame_duration: int
    frames_per_second: int  # the track's numberOfFrames
    sample_offset: int  # where the first tmcd sample is in the file

    @property
    def timecode(self) -> str:
        """``HH:MM:SS:FF``, or ``HH:MM:SS;FF`` for drop-frame."""
        return frames_timecode(self.frames, self.rate, drop_frame=self.drop_frame)

    def to_dict(self) -> dict[str, Any]:
        """The read as JSON. ``stub`` is what to put in a plan stub's source: the frame and the
        tmcd rate snapped to a standard rate (2997/100 is 30000/1001), as the producer would."""
        standard = snap_rate(self.rate) or self.rate
        rate = {"num": standard.numerator, "den": standard.denominator}
        return {
            "frames": self.frames,
            "rate": f"{self.rate.numerator}/{self.rate.denominator}",
            "fps": round(float(self.rate), 6),
            "drop_frame": self.drop_frame,
            "timecode": self.timecode,
            "wraps_24h": self.wraps_24h,
            "tmcd": {
                "flags": self.flags,
                "timescale": self.timescale,
                "frame_duration": self.frame_duration,
                "frames_per_second": self.frames_per_second,
                "sample_offset": self.sample_offset,
            },
            "stub": {"start_timecode": self.frames, "start_timecode_rate": rate},
        }


@dataclass(frozen=True)
class _Atom:
    kind: bytes
    start: int  # the header's offset
    body: int  # the body's offset
    end: int  # one past the atom

    @property
    def name(self) -> str:
        return repr(self.kind.decode("latin-1"))


class _Source:
    """Bounded, seek-based reads of a file of known length."""

    def __init__(self, stream: IO[bytes], size: int) -> None:
        self.stream = stream
        self.size = size
        self.atoms = 0
        self.bytes_read = 0

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or offset + length > self.size:
            raise TimecodeUnreadable(
                f"a read of {length} bytes at {offset} runs past the end of the "
                f"{self.size}-byte file",
                code="malformed",
            )
        self.bytes_read += length
        if self.bytes_read > MAX_READ_BYTES:
            raise TimecodeUnreadable(
                f"more than {MAX_READ_BYTES} bytes read before the timecode", code="too_large"
            )
        try:
            self.stream.seek(offset)
            data = self.stream.read(length) or b""  # a raw read gives None only when it would block
        except OSError as exc:
            raise TimecodeUnreadable(
                f"cannot read the file: {exc.strerror or exc}", code="io"
            ) from None
        if len(data) != length:
            raise TimecodeUnreadable(
                f"the file ended {len(data)} bytes into a {length}-byte read at {offset} "
                "(was it truncated while being read?)",
                code="io",
            )
        return data

    def children(
        self, start: int, end: int, parent: str, *, limit: int | None = None
    ) -> Iterator[_Atom]:
        """The atoms from ``start`` to ``end``, at most ``limit`` of them. A remainder under 8
        bytes is ignored, as ffmpeg does; size 0 runs to ``end``; size 1 takes the 64-bit size
        after the type."""
        offset = start
        seen = 0
        while end - offset >= 8:
            self.atoms += 1
            seen += 1
            if self.atoms > MAX_ATOMS:
                raise TimecodeUnreadable(
                    f"more than {MAX_ATOMS} atoms before the timecode", code="malformed"
                )
            if limit is not None and seen > limit:
                raise TimecodeUnreadable(
                    f"more than {limit} atoms in {parent} before the timecode", code="malformed"
                )
            size, kind = struct.unpack(">I4s", self.read(offset, 8))
            body = offset + 8
            if size == 1:
                if end - offset < 16:
                    raise TimecodeUnreadable(
                        f"atom {kind.decode('latin-1')!r} at {offset} in {parent}: its 64-bit "
                        "size is cut off",
                        code="malformed",
                    )
                (size,) = struct.unpack(">Q", self.read(offset + 8, 8))
                body = offset + 16
            elif size == 0:
                size = end - offset
            atom = _Atom(kind, offset, body, offset + size)
            if size < body - offset:
                raise TimecodeUnreadable(
                    f"atom {atom.name} at {offset} in {parent} claims {size} bytes, less than "
                    f"its {body - offset}-byte header",
                    code="malformed",
                )
            if size > end - offset:
                raise TimecodeUnreadable(
                    f"atom {atom.name} at {offset} in {parent} claims {size} bytes, past the "
                    f"end of {parent} ({end - offset} left)",
                    code="malformed",
                )
            yield atom
            offset = atom.end

    def child(self, parent: _Atom, kind: bytes) -> _Atom | None:
        return next(
            (a for a in self.children(parent.body, parent.end, parent.name) if a.kind == kind), None
        )

    def body(self, atom: _Atom, length: int, *, at: int = 0) -> bytes:
        """``length`` bytes of ``atom``'s body from ``at``; one that is too short is malformed."""
        if atom.end - atom.body < at + length:
            raise TimecodeUnreadable(
                f"atom {atom.name} at {atom.start} is {atom.end - atom.body} bytes, too short "
                f"for its fields",
                code="malformed",
            )
        return self.read(atom.body + at, length)


def _movie(src: _Source) -> _Atom:
    for atom in src.children(0, src.size, "the file", limit=MAX_TOP_LEVEL_ATOMS):
        if atom.kind == b"moov":
            if atom.end - atom.body > MAX_MOOV_BYTES:
                raise TimecodeUnreadable(
                    f"the moov atom is {atom.end - atom.body} bytes, over the "
                    f"{MAX_MOOV_BYTES}-byte limit",
                    code="too_large",
                )
            return atom
    raise TimecodeUnreadable(
        "no moov atom: not a QuickTime or MP4 movie, or an unfinished recording",
        code="not_quicktime",
    )


def _timecode_track(src: _Source, moov: _Atom) -> tuple[_Atom, _Atom]:
    """The first ``trak`` whose ``mdia/hdlr`` subtype is ``tmcd`` (``minf`` has a data-handler
    ``hdlr`` too, which is not the track's kind), and its ``mdia``."""
    for trak in src.children(moov.body, moov.end, "moov"):
        if trak.kind != b"trak":
            continue
        mdia = src.child(trak, b"mdia")
        hdlr = src.child(mdia, b"hdlr") if mdia is not None else None
        if mdia is not None and hdlr is not None and src.body(hdlr, 4, at=8) == b"tmcd":
            return trak, mdia
    raise TimecodeUnreadable("the movie has no timecode (tmcd) track", code="no_timecode")


def _require(src: _Source, parent: _Atom, kind: bytes) -> _Atom:
    found = src.child(parent, kind)
    if found is None:
        raise TimecodeUnreadable(
            f"the tmcd track has no {kind.decode('latin-1')!r} atom", code="malformed"
        )
    return found


def _first_chunk(src: _Source, stbl: _Atom) -> int:
    for atom in src.children(stbl.body, stbl.end, "stbl"):
        if atom.kind not in (b"stco", b"co64"):
            continue
        width = 4 if atom.kind == b"stco" else 8
        (count,) = struct.unpack(">I", src.body(atom, 4, at=4))
        if count == 0:
            raise TimecodeUnreadable("the tmcd track has no samples", code="no_timecode")
        if count * width > atom.end - atom.body - 8:
            raise TimecodeUnreadable(
                f"the tmcd {atom.name} lists {count} chunks but holds "
                f"{(atom.end - atom.body - 8) // width}",
                code="malformed",
            )
        raw = src.body(atom, width, at=8)
        return struct.unpack(">I" if width == 4 else ">Q", raw)[0]
    raise TimecodeUnreadable("the tmcd track has no chunk offsets (stco or co64)", code="malformed")


def _require_in_mdat(src: _Source, offset: int) -> None:
    """The sample's four bytes must lie in the payload of a top-level ``mdat``: an offset into
    ``moov``, a header or past the media is not a timecode sample."""
    for atom in src.children(0, src.size, "the file", limit=MAX_TOP_LEVEL_ATOMS):
        if atom.start > offset:
            break
        if atom.kind == b"mdat" and atom.body <= offset and offset + 4 <= atom.end:
            return
    raise TimecodeUnreadable(
        f"the tmcd sample at {offset} is not inside an mdat atom's media", code="malformed"
    )


def _entries(src: _Source, stsd: _Atom) -> int:
    (entries,) = struct.unpack(">I", src.body(stsd, 4, at=4))
    if entries == 0:
        raise TimecodeUnreadable("the tmcd track's stsd has no entries", code="malformed")
    if entries * 8 > stsd.end - stsd.body - 8:
        raise TimecodeUnreadable(
            f"the tmcd track's stsd lists {entries} sample descriptions but holds at most "
            f"{(stsd.end - stsd.body - 8) // 8}",
            code="malformed",
        )
    return entries


def _description_index(src: _Source, stbl: _Atom, stsd: _Atom) -> int:
    """Which sample description (1-based) the first sample uses: the sample_description_index
    of stsc's first row, which must start at chunk 1 (the first sample opens the first chunk).
    A track with one description and no stsc uses it, as cameras write (the CCN BRAW has both);
    with several, stsc is required."""
    entries = _entries(src, stsd)
    found = src.child(stbl, b"stsc")
    if found is None:
        if entries == 1:
            return 1
        raise TimecodeUnreadable(
            f"the tmcd track has {entries} sample descriptions and no stsc to choose one",
            code="malformed",
        )
    (rows,) = struct.unpack(">I", src.body(found, 4, at=4))
    if rows == 0:
        raise TimecodeUnreadable("the tmcd track's stsc is empty", code="malformed")
    if rows * 12 > found.end - found.body - 8:
        raise TimecodeUnreadable(
            f"the tmcd track's stsc lists {rows} rows but holds "
            f"{(found.end - found.body - 8) // 12}",
            code="malformed",
        )
    first_chunk, per_chunk, index = struct.unpack(">III", src.body(found, 12, at=8))
    if first_chunk != 1:
        raise TimecodeUnreadable(
            f"the tmcd track's stsc starts at chunk {first_chunk}, not 1", code="malformed"
        )
    if per_chunk < 1:
        raise TimecodeUnreadable(
            f"the tmcd track's stsc gives chunk 1 {per_chunk} samples per chunk",
            code="malformed",
        )
    if not 1 <= index <= entries:
        raise TimecodeUnreadable(
            f"the tmcd track's stsc gives chunk 1 sample description {index}, not 1-{entries}",
            code="malformed",
        )
    return index


def _description(src: _Source, stsd: _Atom, index: int) -> int:
    """The file offset of sample description ``index`` (1-based), walked by the entries' size
    fields, each bounded by stsd; the chosen one must be a 'tmcd' of at least 34 bytes."""

    def header(at: int, number: int, least: int) -> tuple[int, bytes]:
        left = stsd.end - at
        if left < 8:
            raise TimecodeUnreadable(
                f"the tmcd track's stsd lists {index} or more sample descriptions but holds "
                f"{number - 1}",
                code="malformed",
            )
        size, kind = struct.unpack(">I4s", src.read(at, 8))
        if not least <= size <= left:
            raise TimecodeUnreadable(
                f"the tmcd track's sample description {number} claims {size} bytes (at least "
                f"{least}, at most {left} fit)",
                code="malformed",
            )
        return size, kind

    at = stsd.body + 8
    for number in range(1, index):  # the descriptions before the chosen one
        size, _ = header(at, number, 8)
        at += size
    _, kind = header(at, index, _TMCD_ENTRY)
    if kind != b"tmcd":
        raise TimecodeUnreadable(
            f"the tmcd track's sample description {index} is {kind.decode('latin-1')!r}, "
            "not 'tmcd'",
            code="malformed",
        )
    return at


def _require_unedited(src: _Source, trak: _Atom) -> None:
    """The tmcd track's first edit (``edts/elst``) must be trivial: media_time 0 at rate exactly
    1, so the track starts at its first sample. A trimmed start (media_time > 0), an empty edit
    (-1: the track starts after the movie), any other negative time or another rate is refused
    as ``unsupported``: how Resolve maps an edited tmcd is unverified, and camera files (the
    CCN BRAW: one edit, 0, rate 1) do not need it. Later edits do not move the start. No edts,
    no elst or no entries means no edit."""
    edts = src.child(trak, b"edts")
    found = src.child(edts, b"elst") if edts is not None else None
    if found is None:
        return
    version, rows = struct.unpack(">B3xI", src.body(found, 8))
    if version not in (0, 1):
        raise TimecodeUnreadable(
            f"the tmcd track's elst version {version} is not 0 or 1, so its edit is unknown",
            code="unsupported",
        )
    width = 12 if version == 0 else 20
    if rows * width > found.end - found.body - 8:
        raise TimecodeUnreadable(
            f"the tmcd track's elst lists {rows} edits but holds "
            f"{(found.end - found.body - 8) // width}",
            code="malformed",
        )
    if rows == 0:
        return
    row = ">Iihh" if version == 0 else ">Qqhh"
    duration, media_time, rate, fraction = struct.unpack(row, src.body(found, width, at=8))
    if media_time == 0 and (rate, fraction) == (1, 0):
        return
    time = f"media_time {media_time}"
    if media_time == -1:
        time += " (an empty edit: the track starts after the movie does)"
    elif media_time > 0:
        time += " (a trimmed start)"
    elif media_time < 0:
        time += " (a negative media time)"
    speed = f"rate {rate}+{fraction}/65536" + ("" if (rate, fraction) == (1, 0) else " (not 1)")
    raise TimecodeUnreadable(
        f"the tmcd track's first edit (duration {duration}, {time}, {speed}) does not start at "
        "its first sample at rate 1, and an edited timecode track is not supported",
        code="unsupported",
    )


def _read(src: _Source) -> QuickTimeTimecode:
    trak, mdia = _timecode_track(src, _movie(src))
    _require_unedited(src, trak)
    stbl = _require(src, _require(src, mdia, b"minf"), b"stbl")
    stsd = _require(src, stbl, b"stsd")
    entry = _description(src, stsd, _description_index(src, stbl, stsd))
    flags, timescale, duration, per_second = struct.unpack(">IIIB", src.read(entry + 20, 13))
    if timescale == 0 or duration == 0 or per_second == 0:
        raise TimecodeUnreadable(
            f"the tmcd track's timescale {timescale}, frameDuration {duration} and "
            f"numberOfFrames {per_second} must all be positive",
            code="malformed",
        )
    rate = Fraction(timescale, duration)
    drop = bool(flags & FLAG_DROP_FRAME)
    wraps = bool(flags & FLAG_24_HOUR)
    if per_second != round(rate):
        raise TimecodeUnreadable(
            f"the tmcd track counts {per_second} frames a second at {float(rate):g} fps, so its "
            "frame number is ambiguous",
            code="unsupported",
        )
    if drop and rate not in DROP_FRAME_RATES:
        raise TimecodeUnreadable(
            f"the tmcd track is drop-frame at {float(rate):g} fps; drop-frame exists only at "
            "29.97 and 59.94",
            code="unsupported",
        )
    offset = _first_chunk(src, stbl)
    if offset + 4 > src.size:
        raise TimecodeUnreadable(
            f"the tmcd sample at {offset} is past the end of the {src.size}-byte file",
            code="malformed",
        )
    _require_in_mdat(src, offset)
    (frames,) = struct.unpack(">i", src.read(offset, 4))  # signed: flag 0x4 allows negatives
    if frames < 0:
        raise TimecodeUnreadable(
            f"the tmcd sample is {frames}, before 00:00:00:00 (flags {flags:#x}); a negative "
            "media start cannot be an FCPXML asset start",
            code="negative_timecode",
        )
    day = frames_per_day(rate, drop_frame=drop)
    if frames >= day:
        if not wraps:
            raise TimecodeUnreadable(
                f"the tmcd sample {frames} is past 24 hours at {float(rate):g} fps and the track "
                "does not wrap at 24 hours",
                code="unsupported",
            )
        frames %= day
    return QuickTimeTimecode(
        frames, rate, drop, wraps, flags, timescale, duration, per_second, offset
    )


def frames_per_day(rate: Fraction, *, drop_frame: bool) -> int:
    """How many frames 24 hours of timecode label at ``rate``."""
    nominal = round(rate)
    if not drop_frame:
        return 86400 * nominal
    skip = nominal // 15
    return 144 * (600 * nominal - 9 * skip)  # 144 ten-minute blocks


def read_timecode_from(stream: IO[bytes], size: int) -> QuickTimeTimecode:
    """The start timecode of the ``size``-byte movie ``stream`` (seekable, binary)."""
    if size < 8:
        raise TimecodeUnreadable(f"{size} bytes is too short for a movie", code="not_quicktime")
    return _read(_Source(stream, size))


def read_timecode(path: str | os.PathLike[str]) -> QuickTimeTimecode:
    """The start timecode of the QuickTime or MP4 file at ``path``. Raises ``MediaInputError``
    when it is missing or not a regular file, ``TimecodeUnreadable`` when it has no usable
    timecode."""
    p = require_regular_file(path, tool=_TOOL)
    try:
        fd = os.open(p, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    except OSError as exc:
        raise TimecodeUnreadable(f"cannot open {p}: {exc.strerror or exc}", code="io") from None
    with os.fdopen(fd, "rb", buffering=0) as stream:  # unbuffered: 8-byte reads stay 8 bytes
        try:
            info = os.fstat(stream.fileno())
        except OSError as exc:  # pragma: no cover - fstat on an open descriptor
            raise TimecodeUnreadable(f"cannot stat {p}: {exc}", code="io") from None
        if not stat.S_ISREG(info.st_mode):  # replaced by a FIFO or device since the check
            raise TimecodeUnreadable(f"not a regular file: {p}", code="io")
        return read_timecode_from(stream, info.st_size)


__all__ = [
    "MAX_ATOMS",
    "MAX_MOOV_BYTES",
    "MAX_READ_BYTES",
    "MAX_TOP_LEVEL_ATOMS",
    "QUICKTIME_SUFFIXES",
    "QuickTimeTimecode",
    "TimecodeUnreadable",
    "frames_per_day",
    "read_timecode",
    "read_timecode_from",
]
