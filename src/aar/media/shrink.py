"""``aar media shrink``: a small working copy of a camera proxy, the same frames (S19b).

The hub edits from copies of the camera proxies, and a 90-minute proxy (about 5.5 GB) leaves room
for one camera. The timeline the hub exports links the edit PC's own full-size copy, so the hub's
copy only has to transcribe, sync, find faces and make a review render, from the same frames.

:func:`shrink_file` re-encodes the video (H.264 8-bit 4:2:0 at :data:`VIDEO_ARGS`, a keyframe at
least every :data:`KEYFRAME_MAX_S`, one output frame per input frame) and keeps every audio stream
lossless (:data:`AUDIO_CODECS`: FLAC in MP4, ALAC in QuickTime), with the container's metadata, the
start timecode (a tmcd track when the source has one, none when it has none) and a mark
(:data:`MARK_KEY`, with :data:`SETTINGS_VERSION`) saying it is a shrunk copy. docs/DESIGN.md
"Shrunk working copies" says why these settings.

Every output is verified before it gets its final name: same frame size, pixel aspect, rotation,
frame count, constant rate, stream starts (:data:`START_TOLERANCE`), start timecode (ffprobe's tag
and :func:`aar.media.quicktime.read_timecode`), audio streams, channels, channel layout, sample
rate and sample count, a few frames of picture (:func:`picture_differences`) and the decoded audio
of every stream hashed equal; the video is H.264 yuv420p, every frame lasts one frame and no two
keyframes are further apart than :data:`KEYFRAME_MAX_S`. Any difference removes the temporary file
and raises :class:`ShrinkError` (``verify_failed``) naming it. A stop signal or Ctrl-C ends the
run the same way as a timeout (:func:`stop_signals`).

Modes: ``--out DST`` writes a new file (a file to a file, or the video files directly in a folder
to the same names in another folder) and never overwrites; a destination that already verifies as
a shrink of its source is skipped, so a rerun resumes a batch. ``--replace`` writes beside the
source, verifies, and atomically renames the small file over the source; it is the only mode that
removes anything, and it never runs on a file already marked as shrunk.

Stdlib only, so it runs on the edit PC from a copy of ``src`` with no ``aar`` installed (with
``PYTHONPATH`` set to that ``src``)::

    python -m aar.media.shrink E:\\Show\\Proxy --out Z:\\incoming\\Show\\Proxy

ffmpeg and ffprobe come from ``AAR_FFMPEG``/``AAR_FFPROBE`` or ``PATH`` (:mod:`aar.media.tools`).
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import secrets
import shutil
import signal
import sys
import threading
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from aar.media.probe import parse_rational, probe
from aar.media.quicktime import TimecodeUnreadable, read_timecode
from aar.media.tools import (
    MediaError,
    MediaInputError,
    MediaTimeout,
    MediaToolMissing,
    input_url,
    require_regular_file,
    run_tool,
    timeout_for,
)

# --- The settings, in one place (docs/DESIGN.md "Shrunk working copies") -----------------------

#: Bumped whenever an encoder setting below changes; written into every output's mark.
SETTINGS_VERSION = 2
#: The video encode: H.264 8-bit 4:2:0 at constant quality.
VIDEO_PRESET = "medium"
VIDEO_CRF = 30
VIDEO_ARGS: tuple[str, ...] = (
    *("-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p"),
)
#: The longest stretch between two keyframes (review renders seek per segment).
KEYFRAME_MAX_S = Fraction(2)
#: Container (by file extension) and its lossless audio codec. The QuickTime muxer refuses FLAC,
#: so ``.mov`` takes ALAC (about 1.6 % larger on the pilot's audio).
CONTAINERS: dict[str, str] = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov"}
AUDIO_CODECS: dict[str, str] = {"mp4": "flac", "mov": "alac"}
#: The container metadata key that marks a shrunk copy.
MARK_KEY = "aar_shrink"
MARK_PREFIX = "aar-shrink v"
#: Before encoding, the destination folder must have this fraction of the source's size free.
MIN_FREE_FRACTION = 0.5
#: At most this many files in one folder batch.
MAX_BATCH = 64
#: The encode's timeout: ``ENCODE_TIMEOUT_BASE_S + ENCODE_TIMEOUT_PER_MEDIA_S * duration``.
ENCODE_TIMEOUT_BASE_S = 300.0
ENCODE_TIMEOUT_PER_MEDIA_S = 4.0
#: The packet scan's and the audio hash's timeouts, the same way.
READ_TIMEOUT_BASE_S = 120.0
READ_TIMEOUT_PER_MEDIA_S = 1.0
#: How far a frame step or the average rate may be from one constant frame (the render's rule).
VFR_TOLERANCE = Fraction(1, 1000)
VFR_STEP_TOLERANCE = Fraction(1, 8)
#: Audio sample formats the lossless codecs keep exactly (integers of at most 24 bits).
_INT16 = frozenset({"u8", "u8p", "s16", "s16p"})
_INT32 = frozenset({"s32", "s32p"})
#: The channel layouts each container's lossless codec carries through unchanged (measured on
#: ffmpeg 6.1: FLAC in MP4 stores no channel mask, so 5.1 comes back as 5.1(side); ALAC in
#: QuickTime keeps a ``chan`` atom). A source with another named layout is refused before
#: encoding; one whose layout is unset ("unknown", as the camera proxies have it) takes the
#: codec's default for its channel count.
KEPT_LAYOUTS: dict[str, frozenset[str]] = {
    "mp4": frozenset({"mono", "stereo", "3.0", "quad", "5.0(side)", "5.1(side)", "7.1"}),
    "mov": frozenset({"mono", "stereo", "3.0", "4.0", "5.0", "5.1"}),
}
#: How far a stream's start in the copy may be from the source's (the video, every audio
#: stream). Far under a frame (33 ms) and under what anyone hears as sound off picture; above
#: the rounding of a timestamp in a 1/30000 or 1/48000 time base.
START_TOLERANCE = Fraction(1, 1000)
#: The picture check: :data:`PICTURE_RUN` consecutive frames at :data:`PICTURE_WINDOWS` places
#: (the first and the last frames included), decoded from both files at :data:`PICTURE_SIZE` in
#: 8-bit gray and compared by their mean absolute difference in gray levels (0-255).
PICTURE_WINDOWS = 5
PICTURE_RUN = 3
PICTURE_SIZE = (128, 72)
#: A copy's frame may differ from the source's same frame by at most this much on average.
PICTURE_MAX_DIFF = 6.0
#: The shift test, decided over all windows together: the copy is shifted by a frame when its
#: frames are closer to the source's frames one later (or one earlier) than to the same frames
#: by more than this share of how much the source's own neighbouring frames differ there
#: (summed over the windows; the share is between -1 and 1, since no frame can be closer to a
#: neighbour than that neighbour is to the same frame). Measured (docs/DESIGN.md): true copies
#: at most 0.006 (static grain) and -0.05 to -0.99 wherever the picture moves; copies shifted
#: by one frame 0.10 (a noisy exposure ramp) to 0.99 wherever it moves.
PICTURE_SHIFT_MARGIN = 0.05
#: A window whose source frames differ from their neighbours by less than this (mean gray
#: levels) is too still to tell a shift: it counts neither way.
PICTURE_MIN_MOTION = 0.05
#: One picture window (a seek and the decode of at most one keyframe interval) may take this.
PICTURE_TIMEOUT_S = 120.0

CODES = frozenset(
    {
        "bad_arguments",
        "unreadable",
        "unsupported_container",
        "no_video",
        "vfr",
        "odd_size",
        "negative_start",
        "audio_unsupported",
        "already_shrunk",
        "symlink",
        "destination_exists",
        "not_writable",
        "low_space",
        "timeout",
        "encode_failed",
        "verify_failed",
        "source_changed",
        "too_many_files",
        "tool_missing",
        "interrupted",
        "io",
    }
)


def mark_value() -> str:
    """The mark written into each output's metadata under :data:`MARK_KEY`."""
    return (
        f"{MARK_PREFIX}{SETTINGS_VERSION} libx264 preset={VIDEO_PRESET} crf={VIDEO_CRF} "
        f"keyframe<={KEYFRAME_MAX_S}s audio=lossless"
    )


def mark_version(mark: str | None) -> int | None:
    """The settings version in a mark, None when ``mark`` is not one of ours."""
    if not mark or not mark.startswith(MARK_PREFIX):
        return None
    digits = mark[len(MARK_PREFIX) :].split(" ", 1)[0]
    return int(digits) if digits.isdigit() else None


class ShrinkError(RuntimeError):
    """A typed refusal or failure: ``code`` (one of :data:`CODES`), ``message`` and a fix hint."""

    def __init__(self, code: str, message: str, hint: str = "", **details: Any) -> None:
        if code not in CODES:
            raise ValueError(f"unknown shrink error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fix_hint": self.hint,
            "details": self.details,
        }


# --- Reading a file -------------------------------------------------------------------------


@dataclass
class _StreamScan:
    kind: str = "unknown"
    time_base: Fraction | None = None
    sample_rate: int | None = None
    sample_fmt: str | None = None
    bits: int | None = None
    layout: str | None = None
    packets: int = 0
    ticks: int = 0  # the packets' durations, summed, in time_base
    untimed: int = 0  # packets without a duration
    duration_ts: int | None = None  # the stream's own duration, in time_base
    pts: array[int] = field(default_factory=lambda: array("q"))
    keys: array[int] = field(default_factory=lambda: array("q"))


class _ScanSink:
    """Parses ffprobe's compact packet/stream/format lines as they arrive (flat memory)."""

    def __init__(self) -> None:
        self.rest = b""
        self.streams: dict[int, _StreamScan] = {}
        self.mark: str | None = None

    def feed(self, chunk: bytes) -> None:
        data = self.rest + chunk
        lines = data.split(b"\n")
        self.rest = lines.pop()
        for line in lines:
            self.line(line.decode("utf-8", errors="replace").rstrip("\r"))

    def line(self, text: str) -> None:
        section, _, rest = text.partition("|")
        values = dict(item.partition("=")[::2] for item in rest.split("|") if item)
        if section == "packet":
            index = _int(values.get("stream_index"))
            if index is None:
                return
            scan = self.streams.setdefault(index, _StreamScan())
            scan.packets += 1
            duration = _int(values.get("duration"))
            if duration is None:
                scan.untimed += 1
            scan.ticks += duration or 0
            pts = _int(values.get("pts"))
            if pts is not None:
                scan.pts.append(pts)
                if values.get("flags", "").startswith("K"):
                    scan.keys.append(pts)
        elif section == "stream":
            index = _int(values.get("index"))
            if index is None:
                return
            scan = self.streams.setdefault(index, _StreamScan())
            scan.kind = values.get("codec_type") or "unknown"
            scan.time_base = parse_rational(values.get("time_base"))
            scan.sample_rate = _int(values.get("sample_rate"))
            scan.sample_fmt = values.get("sample_fmt") or None
            scan.bits = _int(values.get("bits_per_raw_sample"))
            scan.layout = values.get("channel_layout") or None
            scan.duration_ts = _int(values.get("duration_ts"))
        elif section == "format":
            mark = values.get(f"tag:{MARK_KEY}")
            self.mark = mark or None


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_timeout(duration_s: float | None) -> float:
    return timeout_for(duration_s, base_s=READ_TIMEOUT_BASE_S, per_media_s=READ_TIMEOUT_PER_MEDIA_S)


def _scan(path: Path, duration_s: float | None) -> _ScanSink:
    """Every packet's stream, timestamp, duration and keyframe flag, each stream's time base and
    sample format, and the mark: one ffprobe pass, demuxing only."""
    sink = _ScanSink()
    run_tool(
        "ffprobe",
        [
            *("-hide_banner", "-v", "error", "-show_entries"),
            "stream=index,codec_type,time_base,sample_rate,sample_fmt,bits_per_raw_sample,"
            "channel_layout,duration_ts"
            f":packet=stream_index,pts,duration,flags:format_tags={MARK_KEY}",
            *("-of", "compact", input_url(path)),
        ],
        timeout_s=_read_timeout(duration_s),
        stdout_sink=sink.feed,
    )
    if sink.rest:
        sink.line(sink.rest.decode("utf-8", errors="replace"))
        sink.rest = b""
    return sink


def _audio_hashes(path: Path, duration_s: float | None) -> tuple[str, ...]:
    """The SHA-256 of every audio stream's decoded samples (as 32-bit integers), in order."""
    result = run_tool(
        "ffmpeg",
        [
            *("-hide_banner", "-nostdin", "-nostats", "-v", "error"),
            *("-i", input_url(path), "-map", "0:a", "-c:a", "pcm_s32le"),
            *("-f", "streamhash", "-hash", "sha256", "-"),
        ],
        timeout_s=_read_timeout(duration_s),
        stdout_limit=1024 * 1024,
    )
    lines = result.stdout.decode("ascii", errors="replace").split()
    return tuple(line.rsplit("=", 1)[-1] for line in lines if line.strip())


@dataclass(frozen=True)
class AudioFacts:
    """One audio stream: its channels, layout, sample rate, sample count and start (seconds)."""

    index: int
    channels: int | None
    sample_rate: int | None
    samples: int | None
    layout: str | None = None
    start: Fraction | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "channels": self.channels,
            "layout": self.layout,
            "sample_rate": self.sample_rate,
            "samples": self.samples,
        }


def _start(scan: _StreamScan) -> Fraction | None:
    """A stream's first timestamp in seconds (its earliest packet)."""
    if scan.time_base is None or not scan.pts:
        return None
    return min(scan.pts) * scan.time_base


def layout_named(layout: str | None) -> bool:
    """Whether ffprobe names a channel layout (not unset, "unknown" or "N channels")."""
    return bool(layout) and layout != "unknown" and not str(layout).endswith("channels")


@dataclass(frozen=True)
class Facts:
    """What a source and its shrunk copy must share, read from one file."""

    path: str
    size_bytes: int
    duration_s: float | None
    video_index: int
    codec: str | None
    pix_fmt: str | None
    width: int | None
    height: int | None
    sample_aspect: Fraction | None
    rotation: int
    base_fps: Fraction | None
    avg_fps: Fraction | None
    frames: int
    video_start: Fraction | None  # the first frame's timestamp, seconds
    #: The first frame step that is not one frame: (seconds into the video, length in frames).
    uneven: tuple[float, float] | None
    max_keyframe_gap_s: float | None
    timecode_tag: str | None
    timecode: tuple[int, Fraction, bool] | None  # the tmcd track as aar.media.quicktime reads it
    audio: tuple[AudioFacts, ...]
    audio_formats: tuple[tuple[int, str | None, int | None], ...]  # index, sample_fmt, bits
    mark: str | None

    def summary(self) -> dict[str, Any]:
        rate = self.base_fps
        return {
            "frames": self.frames,
            "rate": None if rate is None else f"{rate.numerator}/{rate.denominator}",
            "size": [self.width, self.height],
            "start_timecode": self.timecode_tag,
            "audio": [a.to_dict() for a in self.audio],
            "bytes": self.size_bytes,
        }


def _rate_text(rate: Fraction | None) -> str:
    return "unknown" if rate is None else f"{rate.numerator}/{rate.denominator}"


def read_facts(path: str | os.PathLike[str]) -> Facts:
    """Probe and scan ``path`` (:class:`Facts`); ``ShrinkError`` when it is not a video file."""
    p = Path(path)
    try:
        info = probe(p)
    except MediaToolMissing as exc:
        raise _tool_missing(exc) from None
    except MediaError as exc:
        raise ShrinkError(
            "unreadable",
            f"{str(p)!r} cannot be read as media: {exc.summary()}",
            "check the file plays; copy it again if it was cut short",
            path=str(p),
        ) from None
    video = info.video
    if video is None:
        raise ShrinkError(
            "no_video",
            f"{str(p)!r} has no video stream",
            "give the camera's video file (the proxy)",
            path=str(p),
        )
    try:
        scan = _scan(p, info.duration_s)
    except MediaToolMissing as exc:
        raise _tool_missing(exc) from None
    except MediaError as exc:
        raise ShrinkError(
            "unreadable",
            f"ffprobe cannot list the packets of {str(p)!r}: {exc.summary()}",
            "check the file plays; copy it again if it was cut short",
            path=str(p),
        ) from None
    vscan = scan.streams.get(video.index, _StreamScan())
    uneven, gap = _video_timing(vscan, video.base_fps)
    audio = []
    formats = []
    for stream in info.streams:
        if stream.kind != "audio":
            continue
        ascan = scan.streams.get(stream.index, _StreamScan())
        samples = None
        if ascan.time_base is not None and stream.sample_rate:
            # The packets' durations, summed; when a packet has none (the MP4 muxer leaves the
            # last FLAC packet of a stream that starts late without one), the stream's own
            # duration.
            ticks = ascan.ticks
            if ascan.untimed and ascan.duration_ts is not None:
                ticks = ascan.duration_ts
            samples = round(ticks * ascan.time_base * stream.sample_rate)
        audio.append(
            AudioFacts(
                stream.index,
                stream.channels,
                stream.sample_rate,
                samples,
                ascan.layout,
                _start(ascan),
            )
        )
        formats.append((stream.index, ascan.sample_fmt, ascan.bits))
    return Facts(
        path=str(p),
        size_bytes=info.size_bytes,
        duration_s=info.duration_s,
        video_index=video.index,
        codec=video.codec,
        pix_fmt=video.pix_fmt,
        width=video.width,
        height=video.height,
        sample_aspect=video.sample_aspect,
        rotation=video.rotation,
        base_fps=video.base_fps,
        avg_fps=video.fps,
        frames=vscan.packets,
        video_start=_start(vscan),
        uneven=uneven,
        max_keyframe_gap_s=gap,
        timecode_tag=info.timecode,
        timecode=_tmcd(p),
        audio=tuple(audio),
        audio_formats=tuple(formats),
        mark=scan.mark,
    )


def _video_timing(
    scan: _StreamScan, rate: Fraction | None
) -> tuple[tuple[float, float] | None, float | None]:
    """The first frame step that is not one frame (seconds in, frames long), and the longest
    stretch without a keyframe in seconds."""
    tb = scan.time_base
    if not tb or not rate or not scan.pts:
        return None, None
    pts = sorted(scan.pts)
    frame = 1 / (tb * rate)  # ticks per frame
    slack = frame * VFR_STEP_TOLERANCE
    uneven = None
    for a, b in itertools.pairwise(pts):
        if abs((b - a) - frame) > slack:
            uneven = (float((a - pts[0]) * tb), float((b - a) / frame))
            break
    keys = sorted(scan.keys)
    gap = None
    if keys:
        end = pts[-1] + frame
        edges = [*keys, end]
        gap = float(max(b - a for a, b in itertools.pairwise(edges)) * tb)
        if keys[0] > pts[0]:  # no keyframe at the start
            gap = max(gap, float((keys[0] - pts[0]) * tb))
    return uneven, gap


def _tmcd(path: Path) -> tuple[int, Fraction, bool] | None:
    try:
        found = read_timecode(path)
    except TimecodeUnreadable:
        return None
    return (found.frames, found.rate, found.drop_frame)


def _tool_missing(exc: MediaError) -> ShrinkError:
    return ShrinkError(
        "tool_missing",
        str(exc),
        "install ffmpeg and ffprobe on PATH, or set AAR_FFMPEG and AAR_FFPROBE",
    )


# --- Checks ---------------------------------------------------------------------------------


def check_source(facts: Facts, fmt: str | None = None) -> None:
    """Refuse a source this command cannot shrink exactly: an already shrunk copy, a variable
    frame rate (the stub builder's wording), an odd frame size (H.264 4:2:0 needs even ones),
    audio the lossless codecs cannot keep, or (given the destination's muxer ``fmt``) a channel
    layout that muxer's codec does not carry."""
    path = facts.path
    if facts.mark is not None:
        raise ShrinkError(
            "already_shrunk",
            f"{path!r} is already a shrunk copy ({facts.mark})",
            "shrink the original, not the shrunk copy",
            path=path,
        )
    why = _vfr_reason(facts)
    if why is not None:
        raise ShrinkError(
            "vfr",
            f"{path!r} has a variable frame rate: {why}",
            "conform the file to a constant frame rate first",
            path=path,
        )
    starts = [("video", facts.video_start)]
    starts += [(f"audio stream {a.index}", a.start) for a in facts.audio]
    for what, at in starts:
        if at is not None and at < 0:
            raise ShrinkError(
                "negative_start",
                f"{path!r}: its {what} starts at {float(at):.3f} s, before the file does: "
                "frames or samples hidden by an edit list (a cut made without re-encoding), "
                "which the copy cannot keep hidden",
                "give the camera's own proxy, or cut the file again starting at a keyframe",
                path=path,
            )
    if (facts.width or 0) % 2 or (facts.height or 0) % 2:
        raise ShrinkError(
            "odd_size",
            f"{path!r} is {facts.width}x{facts.height}; H.264 in 4:2:0 needs an even width and "
            "height",
            "give the camera's proxy, or scale the file to an even size first",
            path=path,
        )
    for index, sample_fmt, bits in facts.audio_formats:
        exact = sample_fmt in _INT16 or (
            sample_fmt in _INT32 and bits is not None and 0 < bits <= 24
        )
        if not exact:
            raise ShrinkError(
                "audio_unsupported",
                f"{path!r}: audio stream {index} is {sample_fmt or 'unknown'} "
                f"({bits or 'unknown'} bits); FLAC and ALAC keep integer samples of at most "
                "24 bits exactly",
                "give the camera's proxy (24-bit or 16-bit PCM)",
                path=path,
                stream=index,
            )
    if fmt is None:
        return
    for audio in facts.audio:
        if layout_named(audio.layout) and audio.layout not in KEPT_LAYOUTS[fmt]:
            others = [f for f in sorted(KEPT_LAYOUTS) if audio.layout in KEPT_LAYOUTS[f]]
            raise ShrinkError(
                "audio_unsupported",
                f"{path!r}: audio stream {audio.index} has the channel layout {audio.layout}, "
                f"which {AUDIO_CODECS[fmt].upper()} in .{fmt} does not keep",
                f"name the destination .{others[0]} (that container keeps it)"
                if others
                else "give the camera's proxy (mono or stereo audio)",
                path=path,
                stream=audio.index,
                layout=audio.layout,
            )


def _vfr_reason(facts: Facts) -> str | None:
    base = facts.base_fps
    if base is None:
        return "it states no frame rate"
    if facts.uneven is not None:
        at, step = facts.uneven
        return f"{at:.3f} s into its video a frame lasts {step:.3f} frames, not one"
    avg = facts.avg_fps
    if avg is not None and abs(avg / base - 1) > VFR_TOLERANCE:
        return f"its average rate {_rate_text(avg)} is not its base rate {_rate_text(base)}"
    return None


def differences(source: Facts, out: Facts) -> list[str]:
    """Each way ``out`` is not a shrink of ``source`` (empty when it is)."""
    found: list[str] = []

    def same(what: str, a: Any, b: Any) -> None:
        if a != b:
            found.append(f"{what}: source {a}, copy {b}")

    same("frame size", (source.width, source.height), (out.width, out.height))
    same("pixel aspect", source.sample_aspect, out.sample_aspect)
    same("rotation", source.rotation, out.rotation)
    same("frame count", source.frames, out.frames)
    same("frame rate", _rate_text(source.base_fps), _rate_text(out.base_fps))
    _same_start(found, "video", source.video_start, out.video_start)
    why = _vfr_reason(out)
    if why is not None:
        found.append(f"constant frame rate: the copy {why}")
    same("start timecode", source.timecode_tag, out.timecode_tag)
    same("tmcd start timecode", source.timecode, out.timecode)
    same("audio streams", len(source.audio), len(out.audio))
    for n, (a, b) in enumerate(zip(source.audio, out.audio, strict=False)):
        same(f"audio stream {n} channels", a.channels, b.channels)
        same(f"audio stream {n} sample rate", a.sample_rate, b.sample_rate)
        same(f"audio stream {n} sample count", a.samples, b.samples)
        if layout_named(a.layout):
            same(f"audio stream {n} channel layout", a.layout, b.layout)
        _same_start(found, f"audio stream {n}", a.start, b.start)
    same("video codec", "h264", out.codec)
    same("pixel format", "yuv420p", out.pix_fmt)
    gap = out.max_keyframe_gap_s
    if gap is None or gap > float(KEYFRAME_MAX_S) + 1e-9:
        found.append(f"keyframe interval: the copy's longest is {gap} s, over {KEYFRAME_MAX_S} s")
    if mark_version(out.mark) is None:
        found.append(f"mark: the copy has no {MARK_KEY} mark")
    return found


def _same_start(found: list[str], what: str, a: Fraction | None, b: Fraction | None) -> None:
    if (a is None) != (b is None) or (
        a is not None and b is not None and abs(a - b) > START_TOLERANCE
    ):
        shown = ["none" if x is None else f"{float(x):.6f} s" for x in (a, b)]
        found.append(
            f"{what} start: source {shown[0]}, copy {shown[1]} (tolerance "
            f"{float(START_TOLERANCE) * 1000:g} ms)"
        )


# --- The picture check ----------------------------------------------------------------------


def picture_windows(frames: int) -> list[int]:
    """The first frame of each picture window: the first and the last :data:`PICTURE_RUN`
    frames and evenly between, :data:`PICTURE_WINDOWS` places at most."""
    last = frames - min(PICTURE_RUN, frames)
    if last <= 0:
        return [0]
    steps = PICTURE_WINDOWS - 1
    return sorted({round(last * k / steps) for k in range(steps + 1)})


def _gray_run(path: Path, facts: Facts, first: int) -> list[bytes]:
    """Up to :data:`PICTURE_RUN` frames from frame ``first``, as small gray pictures. The seek
    is by the frame's own timestamp (``-seek_timestamp``), half a frame early, and exact."""
    width, height = PICTURE_SIZE
    args = ["-hide_banner", "-nostdin", "-nostats", "-v", "error", "-noautorotate"]
    if first > 0 and facts.base_fps:
        at = (facts.video_start or 0) + (Fraction(first) - Fraction(1, 2)) / facts.base_fps
        args += ["-seek_timestamp", "1", "-ss", f"{float(at):.6f}"]
    args += [
        *("-i", input_url(path), "-map", f"0:{facts.video_index}", "-fps_mode", "passthrough"),
        *("-frames:v", str(PICTURE_RUN), "-an", "-sn", "-dn"),
        *("-vf", f"scale={width}:{height}:flags=area,format=gray", "-f", "rawvideo", "-"),
    ]
    size = width * height
    raw = run_tool(
        "ffmpeg", args, timeout_s=PICTURE_TIMEOUT_S, stdout_limit=PICTURE_RUN * size
    ).stdout
    return [raw[i : i + size] for i in range(0, len(raw) - size + 1, size)]


def _mean_difference(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / max(1, len(a))


def shift_difference(runs: Sequence[tuple[Sequence[bytes], Sequence[bytes]]]) -> str | None:
    """The shift test over every window's ``(copy frames, source frames)``, decided together.
    For a shift s of +1 or -1 and each frame k with a neighbour k + s in its window, the gain
    is how much closer the copy's frame k is to the source's frame k + s than to the source's
    frame k, and the motion how far the source's frame k is from its frame k + s. A window
    whose motion is under :data:`PICTURE_MIN_MOTION` per frame is too still to tell and counts
    neither way. The copy is shifted by s when its gains summed over the windows exceed
    :data:`PICTURE_SHIFT_MARGIN` of their motion summed: noise and the encoder's own errors
    point either way from window to window and cancel, a shift points the same way in every
    window that moves. A shift where nothing moves cannot be seen, and changes nothing there."""
    for shift in (1, -1):
        gain = motion = 0.0
        windows = 0
        for ours, theirs in runs:
            ks = [k for k in range(min(len(ours), len(theirs))) if 0 <= k + shift < len(theirs)]
            moved = sum(_mean_difference(theirs[k], theirs[k + shift]) for k in ks)
            if not ks or moved < PICTURE_MIN_MOTION * len(ks):
                continue
            windows += 1
            motion += moved
            gain += sum(
                _mean_difference(ours[k], theirs[k]) - _mean_difference(ours[k], theirs[k + shift])
                for k in ks
            )
        if windows and gain > PICTURE_SHIFT_MARGIN * motion:
            return (
                f"picture: over {windows} window(s) the copy's frames match the source's frames "
                f"{shift:+d} better than the same frames, by {gain / motion:.2f} of the source's "
                f"own change from frame to frame (at most {PICTURE_SHIFT_MARGIN:g}): the picture "
                "is shifted by a frame"
            )
    return None


def picture_differences(source: Facts, out: Facts, out_path: Path) -> list[str]:
    """How the copy's picture is not the source's, from :func:`picture_windows`: a window that
    decodes another number of frames, a frame further than :data:`PICTURE_MAX_DIFF` from the
    source's same frame (another picture), or frames shifted by one (:func:`shift_difference`,
    over the other windows together)."""
    found: list[str] = []
    runs: list[tuple[list[bytes], list[bytes]]] = []
    for first in picture_windows(source.frames):
        theirs = _gray_run(Path(source.path), source, first)
        ours = _gray_run(out_path, out, first)
        if not theirs or len(ours) != len(theirs):
            found.append(
                f"picture: from frame {first} the source decodes {len(theirs)} frames, "
                f"the copy {len(ours)}"
            )
            continue
        worst = max(range(len(ours)), key=lambda k: _mean_difference(ours[k], theirs[k]))
        own = _mean_difference(ours[worst], theirs[worst])
        if own > PICTURE_MAX_DIFF:
            found.append(
                f"picture: frame {first + worst} differs from the source's by {own:.2f} gray "
                f"levels on average (at most {PICTURE_MAX_DIFF:g}): another picture"
            )
            continue
        runs.append((ours, theirs))
    shifted = shift_difference(runs)
    if shifted is not None:
        found.append(shifted)
    return found


def verify(source: Facts, out_path: Path, *, name: Path | None = None) -> Facts:
    """Read ``out_path`` and compare it with ``source``, the decoded audio included; raise
    ``verify_failed`` naming every difference. ``name`` is the file's final name (the error names
    it rather than the temporary one, and says the partial copy was removed)."""
    label = str(name or out_path)
    hint = (
        "the copy was removed; check the source plays, then run again"
        if name is not None
        else "move the destination aside (it is not this source's shrink), then run again"
    )
    try:
        out = read_facts(out_path)
    except ShrinkError as exc:
        raise ShrinkError(
            "verify_failed",
            f"the copy {label!r} cannot be read back: {exc.message}",
            hint,
            path=label,
        ) from None
    found = differences(source, out)
    if not found:
        try:
            found += picture_differences(source, out, out_path)
        except MediaError as exc:
            raise ShrinkError(
                "verify_failed",
                f"the picture of {label!r} could not be decoded and compared: {exc.summary()}",
                hint,
                path=label,
            ) from None
    if not found and source.audio:
        try:
            want = _audio_hashes(Path(source.path), source.duration_s)
            got = _audio_hashes(out_path, out.duration_s)
        except MediaError as exc:
            raise ShrinkError(
                "verify_failed",
                f"the audio of {label!r} could not be decoded and compared: {exc.summary()}",
                hint,
                path=label,
            ) from None
        for n in range(max(len(want), len(got))):
            if n >= len(want) or n >= len(got) or want[n] != got[n]:
                found.append(f"audio stream {n} decoded samples differ")
    if found:
        raise ShrinkError(
            "verify_failed",
            f"{label!r} is not a shrink of {source.path!r}: " + "; ".join(found),
            hint,
            path=label,
            differences=found,
        )
    return out


# --- Encoding -------------------------------------------------------------------------------


def container_of(path: Path) -> str:
    """The muxer for ``path``'s extension; ``unsupported_container`` otherwise."""
    fmt = CONTAINERS.get(path.suffix.lower())
    if fmt is None:
        raise ShrinkError(
            "unsupported_container",
            f"{str(path)!r}: only {', '.join(sorted(CONTAINERS))} files are shrunk",
            "give the camera proxy (.mp4 or .mov)",
            path=str(path),
        )
    return fmt


def encode_args(facts: Facts, source: Path, target: Path, fmt: str) -> list[str]:
    """The ffmpeg arguments that shrink ``source`` into ``target`` (muxer ``fmt``)."""
    rate = facts.base_fps or Fraction(30)
    gop = max(1, math.floor(KEYFRAME_MAX_S * rate))
    args = [
        # -y: the temporary name was reserved (created empty) by this run; -noautorotate keeps
        # the picture as stored and its display matrix, as the source has them.
        *("-hide_banner", "-nostdin", "-nostats", "-v", "error", "-y", "-noautorotate"),
        *("-i", input_url(source), "-map", f"0:{facts.video_index}"),
    ]
    for audio in facts.audio:
        args += ["-map", f"0:{audio.index}"]
    args += [
        *VIDEO_ARGS,
        # passthrough: one output frame per input frame. enc_time_base demux: the encoder keeps
        # the source's timestamps, so a video that does not start on the frame grid keeps its
        # start (and its offset to the sound).
        *("-g", str(gop), "-fps_mode", "passthrough", "-enc_time_base", "demux"),
        *("-c:a", AUDIO_CODECS[fmt]),
        *("-map_metadata", "0", "-movflags", "+use_metadata_tags"),
        *("-metadata", f"{MARK_KEY}={mark_value()}"),
        *("-f", fmt, input_url(target)),
    ]
    return args


Status = Literal["shrunk", "skipped", "failed"]


@dataclass
class FileResult:
    """One file's outcome, as one JSON line."""

    source: str
    destination: str
    mode: str
    status: Status = "failed"
    reason: str | None = None
    source_bytes: int | None = None
    output_bytes: int | None = None
    encode_s: float | None = None
    duration_s: float | None = None
    output: dict[str, Any] | None = None
    error: ShrinkError | None = None

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    def to_dict(self) -> dict[str, Any]:
        ratio = None
        if self.source_bytes and self.output_bytes is not None:
            ratio = round(self.output_bytes / self.source_bytes, 4)
        speed = None
        if self.encode_s and self.duration_s:
            speed = round(self.duration_s / self.encode_s, 2)
        return {
            "ok": self.ok,
            "status": self.status,
            "mode": self.mode,
            "source": self.source,
            "destination": self.destination,
            "reason": self.reason,
            "source_bytes": self.source_bytes,
            "output_bytes": self.output_bytes,
            "ratio": ratio,
            "encode_s": None if self.encode_s is None else round(self.encode_s, 2),
            "speed_x_realtime": speed,
            "verified": self.output,
            "error": None if self.error is None else self.error.to_dict(),
        }


def _temp_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.aar-shrink-{secrets.token_hex(4)}.part")


def _unlink(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _reserve(temp: Path, path: str) -> None:
    """Create ``temp`` empty and exclusively, so a folder that cannot be written is named as
    such before ffmpeg starts (and no other file can take the name)."""
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    except OSError as exc:
        raise ShrinkError(
            "not_writable",
            f"cannot create a file in {str(temp.parent)!r}: {exc.strerror or exc}",
            "give a folder this user can write to (--replace writes beside the source, so "
            "its folder must be writable)",
            path=path,
            folder=str(temp.parent),
        ) from None
    os.close(fd)


def _flush_file(path: Path) -> None:
    """Make ``path``'s data durable before it is renamed over anything (os.fsync)."""
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _flush_folder(folder: Path) -> None:
    """Make a rename in ``folder`` durable where the platform lets a folder be opened and
    synced (POSIX); elsewhere (Windows, where a folder cannot be opened as a file, and NTFS
    journals the rename itself) nothing to do."""
    try:
        fd = os.open(folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):  # some file systems refuse fsync on a folder
            os.fsync(fd)
    finally:
        os.close(fd)


def _fingerprint(st: os.stat_result) -> tuple[int, ...]:
    """What tells a file changed: size, modification and change time (on Windows the
    creation time), and identity (device and file number, a swap for another file)."""
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev)


def _check_unchanged(src: Path, before: os.stat_result) -> None:
    try:
        now = src.stat()
    except OSError:
        now = None
    if now is None or _fingerprint(now) != _fingerprint(before):
        raise ShrinkError(
            "source_changed",
            f"{str(src)!r} changed while it was being shrunk; it was left as it is and the "
            "copy was removed",
            "run again once nothing else writes the file",
            path=str(src),
        )


def _no_clobber_rename(temp: Path, final: Path) -> None:
    """Give ``temp`` the name ``final`` without ever replacing a file that appeared there."""
    if os.name != "posix":
        os.rename(temp, final)  # Windows: refuses (FileExistsError) when final exists
        return
    try:
        os.link(temp, final)  # refuses (FileExistsError) when final exists
    except FileExistsError:
        raise
    except OSError:  # a file system without hard links
        if final.exists():
            raise FileExistsError(str(final)) from None
        os.rename(temp, final)
        return
    temp.unlink()


def _check_space(directory: Path, need: float, path: str) -> None:
    free = shutil.disk_usage(directory).free
    if free < need:
        raise ShrinkError(
            "low_space",
            f"{str(directory)!r} has {free / 1e9:.2f} GB free; shrinking {path!r} needs "
            f"{need / 1e9:.2f} GB ({MIN_FREE_FRACTION:g} of the source's size) before it starts",
            "free space on the destination, or shrink into another folder",
            free_bytes=free,
            need_bytes=int(need),
        )


def _encode(
    facts: Facts,
    source: Path,
    final: Path,
    *,
    min_free_fraction: float,
    encode_timeout_s: float | None,
) -> tuple[Path, Facts, float]:
    """Encode ``source`` to a temporary name beside ``final`` and verify it. Returns the
    temporary path (verified, not yet renamed), what it holds and the encode's seconds. The
    temporary file is removed on any failure, timeout or interrupt."""
    fmt = container_of(final)
    _check_space(final.parent, min_free_fraction * facts.size_bytes, facts.path)
    temp = _temp_path(final)
    timeout_s = encode_timeout_s or timeout_for(
        facts.duration_s, base_s=ENCODE_TIMEOUT_BASE_S, per_media_s=ENCODE_TIMEOUT_PER_MEDIA_S
    )
    kept = False
    try:
        _reserve(temp, facts.path)
        started = time.monotonic()
        try:
            run_tool("ffmpeg", encode_args(facts, source, temp, fmt), timeout_s=timeout_s)
        except MediaTimeout as exc:
            raise ShrinkError(
                "timeout",
                f"ffmpeg did not finish shrinking {facts.path!r} in {exc.timeout_s:g} s; "
                "it was stopped and its partial file removed",
                "check the machine is not overloaded, then run again",
                path=facts.path,
            ) from None
        except MediaToolMissing as exc:
            raise _tool_missing(exc) from None
        except MediaError as exc:
            raise ShrinkError(
                "encode_failed",
                f"ffmpeg could not shrink {facts.path!r}: {exc.summary()}",
                "check the source plays and ffmpeg has libx264, then run again",
                path=facts.path,
            ) from None
        elapsed = time.monotonic() - started
        out = verify(facts, temp, name=final)
        kept = True
        return temp, out, elapsed
    finally:
        if not kept:
            _unlink(temp)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def shrink_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str] | None = None,
    *,
    replace: bool = False,
    min_free_fraction: float = MIN_FREE_FRACTION,
    encode_timeout_s: float | None = None,
) -> FileResult:
    """Shrink one file: to ``destination`` (never overwritten; skipped when it already verifies
    as a shrink of ``source``), or over ``source`` itself with ``replace``. Raises
    :class:`ShrinkError`; ``KeyboardInterrupt`` stops ffmpeg and removes the partial file."""
    if replace == (destination is not None):
        raise ShrinkError(
            "bad_arguments",
            "give either a destination (--out) or --replace",
            "aar media shrink SRC --out DST, or aar media shrink SRC --replace",
        )
    src = Path(os.path.abspath(source))
    final = src if destination is None else Path(os.path.abspath(destination))
    result = FileResult(str(src), str(final), "replace" if replace else "out")
    if replace and os.path.islink(src):
        raise ShrinkError(
            "symlink",
            f"{str(src)!r} is a symbolic link; --replace would replace the link and leave the "
            "full-size file it points to",
            "run --replace on the file the link points to, or use --out",
            path=str(src),
        )
    try:
        require_regular_file(src, tool="ffmpeg")
    except MediaInputError as exc:
        raise ShrinkError(
            "unreadable", str(exc), "give an existing video file", path=str(src)
        ) from None
    container_of(src)
    fmt = container_of(final)
    if not replace and _same_file(src, final):
        raise ShrinkError(
            "destination_exists",
            f"the destination {str(final)!r} is the source itself",
            "use --replace to shrink a file in place",
            path=str(final),
        )
    if not final.parent.is_dir():
        raise ShrinkError(
            "bad_arguments",
            f"the destination folder {str(final.parent)!r} does not exist",
            "create it first",
            path=str(final),
        )
    before = src.stat()
    facts = read_facts(src)
    result.source_bytes = facts.size_bytes
    result.duration_s = facts.duration_s
    if replace and facts.mark is not None:
        raise ShrinkError(
            "already_shrunk",
            f"{str(src)!r} is already a shrunk copy ({facts.mark}); --replace never runs twice",
            "nothing to do: the file is already small",
            path=str(src),
        )
    if not replace and os.path.lexists(final):
        check_source(facts, fmt)
        try:
            out = verify(facts, final)
        except ShrinkError as exc:
            raise ShrinkError(
                "destination_exists",
                f"{str(final)!r} already exists and is not a shrink of {str(src)!r}; it was left "
                f"as it is ({exc.message})",
                "move it aside or choose another destination; nothing is ever overwritten",
                path=str(final),
                differences=exc.details.get("differences", []),
            ) from None
        result.status = "skipped"
        result.reason = "the destination already verifies as a shrink of the source"
        result.output_bytes = out.size_bytes
        result.output = out.summary()
        return result
    check_source(facts, fmt)
    temp, out_facts, elapsed = _encode(
        facts,
        src,
        final,
        min_free_fraction=min_free_fraction,
        encode_timeout_s=encode_timeout_s,
    )
    try:
        _flush_file(temp)
        _check_unchanged(src, before)
        if replace:
            os.replace(temp, src)
        else:
            try:
                _no_clobber_rename(temp, final)
            except FileExistsError:
                raise ShrinkError(
                    "destination_exists",
                    f"{str(final)!r} appeared while shrinking; it was not overwritten",
                    "check what else writes there, then run again",
                    path=str(final),
                ) from None
        _flush_folder(final.parent)
    finally:
        _unlink(temp)
    result.status = "shrunk"
    result.encode_s = elapsed
    result.output_bytes = out_facts.size_bytes
    result.output = out_facts.summary()
    return result


def video_files(directory: Path) -> list[Path]:
    """The video files directly in ``directory`` (by :data:`CONTAINERS` extension, not hidden),
    sorted by name; ``too_many_files`` past :data:`MAX_BATCH`."""
    found = sorted(
        (
            entry
            for entry in directory.iterdir()
            if entry.suffix.lower() in CONTAINERS
            and not entry.name.startswith(".")
            and entry.is_file()
        ),
        key=lambda entry: entry.name,
    )
    if len(found) > MAX_BATCH:
        raise ShrinkError(
            "too_many_files",
            f"{str(directory)!r} holds {len(found)} video files; one batch takes at most "
            f"{MAX_BATCH}",
            "split the folder",
            path=str(directory),
        )
    return found


def _plan(source: Path, out: Path | None, replace: bool) -> list[tuple[Path, Path | None]]:
    if source.is_dir():
        if out is not None:
            if out.exists() and not out.is_dir():
                raise ShrinkError(
                    "bad_arguments",
                    f"{str(source)!r} is a folder, so --out must be a folder, not a file",
                    "give a folder for --out",
                )
            out.mkdir(parents=True, exist_ok=True)
        return [(f, None if out is None else out / f.name) for f in video_files(source)]
    if out is not None and out.is_dir():
        raise ShrinkError(
            "bad_arguments",
            f"{str(source)!r} is a file, so --out must name the new file, not a folder",
            f"give --out a file name, e.g. {str(out / source.name)!r}",
        )
    return [(source, out)]


def shrink(
    source: str | os.PathLike[str],
    out: str | os.PathLike[str] | None = None,
    *,
    replace: bool = False,
    emit: Any = None,
    **options: Any,
) -> list[FileResult]:
    """Shrink a file or each video file directly in a folder, carrying on past a failed file.
    ``emit`` (optional) gets each result as it finishes."""
    src = Path(os.path.abspath(source))
    dst = None if out is None else Path(os.path.abspath(out))
    if replace == (dst is not None):
        raise ShrinkError(
            "bad_arguments",
            "give either --out DST or --replace",
            "aar media shrink SRC --out DST, or aar media shrink SRC --replace",
        )
    if not src.exists():
        raise ShrinkError("unreadable", f"{str(src)!r} does not exist", "check the path")
    batch = src.is_dir()
    results: list[FileResult] = []
    for file, target in _plan(src, dst, replace):
        try:
            result = shrink_file(file, target, replace=replace, **options)
        except ShrinkError as exc:
            if batch and replace and exc.code == "already_shrunk":
                result = FileResult(str(file), str(file), "replace", "skipped", exc.message)
            else:
                result = FileResult(
                    str(file), str(target or file), "replace" if replace else "out", error=exc
                )
                result.reason = exc.code
        except OSError as exc:
            error = ShrinkError("io", f"{str(file)!r}: {exc}", "check the paths and space")
            result = FileResult(
                str(file), str(target or file), "replace" if replace else "out", error=error
            )
            result.reason = "io"
        except KeyboardInterrupt as exc:
            # ffmpeg was killed and the partial file removed on the way out (run_tool's and
            # _encode's finally blocks); report the file, then stop the batch.
            error = interrupted_error(exc, str(file))
            result = FileResult(
                str(file), str(target or file), "replace" if replace else "out", error=error
            )
            result.reason = error.code
            results.append(result)
            if emit is not None:
                emit(result)
            raise
        results.append(result)
        if emit is not None:
            emit(result)
    return results


# --- Stopping ---------------------------------------------------------------------------------


class Stopped(KeyboardInterrupt):
    """A stop signal (SIGTERM, SIGHUP, SIGBREAK) or Ctrl-C, raised in the main thread so every
    ``finally`` on the way out runs: ffmpeg's process group is killed and the temporary file
    removed, exactly as on a timeout."""

    def __init__(self, signal_name: str) -> None:
        super().__init__(signal_name)
        self.signal_name = signal_name


#: The signals turned into :class:`Stopped`, where the platform has them: Ctrl-C, ``kill``, a
#: closed terminal (SIGHUP, POSIX) and Ctrl-Break (SIGBREAK, Windows).
STOP_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK")


@contextlib.contextmanager
def stop_signals() -> Any:
    """While the block runs, turn :data:`STOP_SIGNALS` into :class:`Stopped`. Only the first
    signal raises (later ones are ignored, so a second Ctrl-C cannot cut the cleanup short). A
    signal this process already ignores (``nohup``) stays ignored. Outside the main thread, or
    where a signal does not exist, nothing changes."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    stopping = False

    def handler(signum: int, _frame: Any) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        raise Stopped(signal.Signals(signum).name)

    previous: dict[int, Any] = {}
    for name in STOP_SIGNALS:
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            if signal.getsignal(signum) == signal.SIG_IGN:
                continue
            previous[signum] = signal.signal(signum, handler)
        except (OSError, ValueError, RuntimeError):  # not settable here
            continue
    try:
        yield
    finally:
        for signum, old in previous.items():
            with contextlib.suppress(OSError, ValueError, RuntimeError, TypeError):
                signal.signal(signum, old)


def interrupted_error(exc: BaseException, path: str | None = None) -> ShrinkError:
    """The typed error for a run stopped by a signal or Ctrl-C."""
    name = getattr(exc, "signal_name", "SIGINT")
    where = f" while shrinking {path!r}" if path else ""
    return ShrinkError(
        "interrupted",
        f"stopped by {name}{where}; ffmpeg was stopped and the partial copy removed",
        "run again: files already shrunk are verified and skipped",
        signal=name,
        **({"path": path} if path else {}),
    )


# --- Command line ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAILED = 2


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``shrink`` arguments, shared by ``aar media shrink`` and ``python -m``."""
    parser.add_argument("source", type=Path, help="A proxy file, or a folder of them.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--out",
        type=Path,
        help="New file (for a file) or folder (for a folder; the same names). Never overwritten; "
        "one that already verifies as a shrink of its source is skipped.",
    )
    where.add_argument(
        "--replace",
        action="store_true",
        help="Write beside the source, verify, then put the small file in its place.",
    )
    parser.add_argument("--report", type=Path, help="Also write a JSON report of the run here.")
    parser.set_defaults(func=run_args)


def build_parser(prog: str = "python -m aar.media.shrink") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=DESCRIPTION)
    add_arguments(parser)
    return parser


DESCRIPTION = (
    "Make a small working copy of a camera proxy: the same frames, frame rate, start timecode "
    f"and lossless audio, H.264 at a much lower bitrate (settings v{SETTINGS_VERSION}). Every "
    "copy is verified before it gets its name. Prints one JSON line per file; exits 0 when "
    "every file was shrunk or already done, 2 otherwise."
)


def _print(line: str, stream: Any) -> None:
    try:
        stream.write(line + "\n")
        stream.flush()
    except (OSError, ValueError):
        pass


def run_args(args: argparse.Namespace) -> int:
    """Run ``shrink`` for parsed arguments; print one JSON line per file; 0 or 2. A stop signal
    or Ctrl-C (:func:`stop_signals`) ends the run with the typed ``interrupted`` error, its JSON
    line and the report, exit 2."""
    failed = False
    results: list[FileResult] = []
    interrupted: ShrinkError | None = None

    def emit(result: FileResult) -> None:
        nonlocal failed
        failed = failed or not result.ok
        results.append(result)
        _print(json.dumps(result.to_dict(), sort_keys=True), sys.stdout)
        if result.error is not None:
            _print(f"aar media shrink: {result.error} (fix: {result.error.hint})", sys.stderr)

    with stop_signals():
        try:
            shrink(args.source, args.out, replace=bool(args.replace), emit=emit)
        except ShrinkError as exc:
            _print(json.dumps({"ok": False, "error": exc.to_dict()}, sort_keys=True), sys.stdout)
            _print(f"aar media shrink: {exc} (fix: {exc.hint})", sys.stderr)
            return EXIT_FAILED
        except KeyboardInterrupt as exc:
            failed = True
            interrupted = interrupted_error(exc)
            if not results or results[-1].reason != "interrupted":  # stopped between files
                line = {"ok": False, "error": interrupted.to_dict()}
                _print(json.dumps(line, sort_keys=True), sys.stdout)
                _print(f"aar media shrink: {interrupted} (fix: {interrupted.hint})", sys.stderr)
        return _write_report(args, results, failed, interrupted)


def _write_report(
    args: argparse.Namespace,
    results: list[FileResult],
    failed: bool,
    interrupted: ShrinkError | None,
) -> int:
    if args.report is not None:
        report = {
            "ok": not failed,
            "settings_version": SETTINGS_VERSION,
            "mark": mark_value(),
            "interrupted": None if interrupted is None else interrupted.to_dict(),
            "files": [r.to_dict() for r in results],
        }
        try:
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        except OSError as exc:
            _print(f"aar media shrink: cannot write the report: {exc}", sys.stderr)
            return EXIT_FAILED
    return EXIT_FAILED if failed else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m aar.media.shrink``: the same command without the rest of ``aar``."""
    args = build_parser().parse_args(argv)
    return run_args(args)


if __name__ == "__main__":
    sys.exit(main())
