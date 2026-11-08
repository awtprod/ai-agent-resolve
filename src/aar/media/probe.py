"""``probe(path)``: duration, size, streams and start timecode of a media file, from ffprobe's
JSON; :func:`timecode_frames` counts a SMPTE timecode in frames and :func:`frames_timecode`
labels a frame count."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any

from aar.media.tools import MediaError, input_url, require_regular_file, run_tool

PROBE_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Stream:
    """One stream. ``kind`` is ffprobe's codec_type: video, audio, subtitle, data, attachment."""

    index: int
    kind: str
    codec: str | None
    duration_s: float | None
    fps: Fraction | None = None
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    attached_pic: bool = False
    pix_fmt: str | None = None
    rotation: int = 0  # display rotation in degrees, 0-359 (video)
    sample_aspect: Fraction | None = None  # pixel aspect; None when unknown or not video
    timecode: str | None = None  # the stream's ``timecode`` tag (a video or tmcd stream)

    @property
    def display_size(self) -> tuple[int, int] | None:
        """The picture as it is shown: the pixel aspect applied, then a 90/270 rotation."""
        if not self.width or not self.height:
            return None
        return display_size(self.width, self.height, self.sample_aspect, self.rotation)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fps"] = None if self.fps is None else f"{self.fps.numerator}/{self.fps.denominator}"
        aspect = self.sample_aspect
        data["sample_aspect"] = (
            None if aspect is None else f"{aspect.numerator}/{aspect.denominator}"
        )
        size = self.display_size
        data["display_size"] = None if size is None else list(size)
        return data


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe says about a file. ``size_bytes`` comes from the file system."""

    path: str
    format: str | None
    duration_s: float | None
    size_bytes: int
    bit_rate: int | None
    streams: tuple[Stream, ...] = field(default_factory=tuple)
    format_timecode: str | None = None  # the container's ``timecode`` tag

    @property
    def video(self) -> Stream | None:
        """The first real video stream (cover art is skipped)."""
        return next((s for s in self.streams if s.kind == "video" and not s.attached_pic), None)

    @property
    def audio(self) -> Stream | None:
        """The first audio stream: the one QA measures."""
        return next((s for s in self.streams if s.kind == "audio"), None)

    @property
    def timecode(self) -> str | None:
        """The media's start timecode tag, as written: the video stream's, else the first other
        stream's that has one (a QuickTime tmcd track), else the container's."""
        video = self.video
        if video is not None and video.timecode:
            return video.timecode
        tagged = next((s.timecode for s in self.streams if s.timecode), None)
        return tagged or self.format_timecode

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "duration_s": self.duration_s,
            "size_bytes": self.size_bytes,
            "bit_rate": self.bit_rate,
            "streams": [s.to_dict() for s in self.streams],
            "format_timecode": self.format_timecode,
            "timecode": self.timecode,
        }


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_rational(text: Any) -> Fraction | None:
    """``"30000/1001"`` to a Fraction; ``"0/0"``, empty or junk to None."""
    if not isinstance(text, str) or not text.strip():
        return None
    num, _, den = text.strip().partition("/")
    try:
        n, d = int(num), int(den or "1")
    except ValueError:
        return None
    if n <= 0 or d <= 0:
        return None
    return Fraction(n, d)


def sample_aspect(text: Any) -> Fraction | None:
    """ffprobe's ``sample_aspect_ratio`` (``"4:3"``; ``"0:1"`` when unknown) as a Fraction."""
    return parse_rational(text.replace(":", "/")) if isinstance(text, str) else None


def stream_rotation(raw: dict[str, Any]) -> int:
    """A stream's display rotation in degrees, 0-359: its Display Matrix side data, else the
    older ``rotate`` tag."""
    rotation = 0
    for side in raw.get("side_data_list") or []:
        if isinstance(side, dict) and _num(side.get("rotation")) is not None:
            rotation = round(float(side["rotation"]))
    tags = raw.get("tags")
    if rotation == 0 and isinstance(tags, dict) and _num(tags.get("rotate")) is not None:
        rotation = round(float(tags["rotate"]))
    return rotation % 360


def _timecode_tag(raw: Any) -> str | None:
    """A stream's or the container's ``timecode`` tag, in any case: Matroska's is ``TIMECODE``
    (on the container), QuickTime's and MXF's ``timecode``."""
    tags = raw.get("tags") if isinstance(raw, dict) else None
    if not isinstance(tags, dict):
        return None
    for key, value in tags.items():
        named = isinstance(key, str) and key.lower() == "timecode"
        if named and isinstance(value, str) and value.strip():
            return value.strip()
    return None


_TIMECODE = re.compile(r"(\d{2}):(\d{2}):(\d{2})([:;.])(\d{2,3})")
#: The only rates drop-frame timecode exists at: 29.97 and 59.94.
DROP_FRAME_RATES = (Fraction(30000, 1001), Fraction(60000, 1001))


def timecode_frames(text: str, fps: Fraction) -> int:
    """A SMPTE timecode as the frame count it labels from 00:00:00:00 at ``fps``.

    ``HH:MM:SS:FF`` is non-drop: frames count at the nominal rate (24 for 23.976, 30 for 29.97),
    so 01:00:00:00 is 86400 at 24 and 108000 at 29.97. ``HH:MM:SS;FF`` (or ``.``) is
    drop-frame, which exists only at 29.97 and 59.94: frame numbers 0-1 (0-3 at 59.94) are
    skipped at the start of every minute except each tenth, so 01:00:00;00 is 107892 at 29.97.
    Raises ``ValueError`` saying what is wrong."""
    match = _TIMECODE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"{text!r} is not a HH:MM:SS:FF timecode")
    hours, minutes, seconds, frame = (int(match.group(n)) for n in (1, 2, 3, 5))
    nominal = round(fps)
    for value, limit, unit in (
        (hours, 24, "hour"),
        (minutes, 60, "minute"),
        (seconds, 60, "second"),
        (frame, nominal, "frame"),
    ):
        if value >= limit:
            raise ValueError(f"{text!r} has {unit} {value} (at {fps} fps the limit is {limit - 1})")
    count = ((hours * 60 + minutes) * 60 + seconds) * nominal + frame
    if match.group(4) == ":":
        return count
    if fps not in DROP_FRAME_RATES:
        raise ValueError(
            f"{text!r} is drop-frame, which exists only at 29.97 and 59.94 fps, not {fps}"
        )
    skip = nominal // 15  # 2 frame numbers at 29.97, 4 at 59.94
    if seconds == 0 and frame < skip and minutes % 10:
        raise ValueError(f"{text!r} names a frame number drop-frame timecode skips (dropped)")
    total_minutes = hours * 60 + minutes
    return count - skip * (total_minutes - total_minutes // 10)


def is_drop_frame(text: str) -> bool:
    """Whether a ``HH:MM:SS;FF`` label is drop-frame: ``;`` or ``.`` before the frames."""
    match = _TIMECODE.fullmatch(text.strip())
    return match is not None and match.group(4) != ":"


def frames_timecode(frames: int, fps: Fraction, *, drop_frame: bool = False) -> str:
    """The SMPTE label of frame ``frames`` (counted from 00:00:00:00) at ``fps``, the inverse of
    :func:`timecode_frames`: ``HH:MM:SS:FF`` at the nominal rate, or ``HH:MM:SS;FF`` for
    drop-frame (29.97 and 59.94 only), which skips frame numbers 0-1 (0-3 at 59.94) at the start
    of every minute except each tenth. Hours are not wrapped. Raises ``ValueError`` for a negative
    count, a rate under 0.5 fps or drop-frame at another rate."""
    if frames < 0:
        raise ValueError(f"frame {frames} is before 00:00:00:00")
    nominal = round(fps)
    if nominal < 1:
        raise ValueError(f"{fps} fps rounds to no frames a second")
    separator = ":"
    if drop_frame:
        if fps not in DROP_FRAME_RATES:
            raise ValueError(f"drop-frame exists only at 29.97 and 59.94 fps, not {fps}")
        skip = nominal // 15  # 2 frame numbers at 29.97, 4 at 59.94
        tens, rest = divmod(frames, 600 * nominal - 9 * skip)  # ten-minute blocks
        frames += 9 * skip * tens
        if rest > skip:
            frames += skip * ((rest - skip) // (60 * nominal - skip))
        separator = ";"
    seconds, ff = divmod(frames, nominal)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}{separator}{ff:02d}"


def display_size(
    width: int, height: int, aspect: Fraction | None, rotation: int
) -> tuple[int, int]:
    """Width and height as shown (and as ffmpeg emits them): the pixel aspect applied to the
    width, then swapped by a 90 or 270 degree rotation."""
    shown = max(1, round(width * aspect)) if aspect else width
    return (height, shown) if rotation % 180 == 90 else (shown, height)


def parse_probe_json(data: dict[str, Any], path: str, size_bytes: int) -> MediaInfo:
    """Build a MediaInfo from ffprobe ``-show_format -show_streams -of json`` output."""
    fmt = data.get("format") or {}
    streams: list[Stream] = []
    for raw in data.get("streams") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("codec_type") or "unknown")
        fps = None
        if kind == "video":
            fps = parse_rational(raw.get("avg_frame_rate")) or parse_rational(
                raw.get("r_frame_rate")
            )
        disposition = raw.get("disposition") or {}
        streams.append(
            Stream(
                index=_int(raw.get("index")) or 0,
                kind=kind,
                codec=raw.get("codec_name"),
                duration_s=_num(raw.get("duration")),
                fps=fps,
                width=_int(raw.get("width")),
                height=_int(raw.get("height")),
                sample_rate=_int(raw.get("sample_rate")),
                channels=_int(raw.get("channels")),
                attached_pic=bool(disposition.get("attached_pic")),
                pix_fmt=(str(raw["pix_fmt"]) if raw.get("pix_fmt") else None),
                rotation=stream_rotation(raw) if kind == "video" else 0,
                sample_aspect=(
                    sample_aspect(raw.get("sample_aspect_ratio")) if kind == "video" else None
                ),
                timecode=_timecode_tag(raw),
            )
        )
    return MediaInfo(
        path=path,
        format=fmt.get("format_name"),
        duration_s=_num(fmt.get("duration")),
        size_bytes=size_bytes,
        bit_rate=_int(fmt.get("bit_rate")),
        streams=tuple(streams),
        format_timecode=_timecode_tag(fmt),
    )


def probe(path: str | os.PathLike[str], *, timeout_s: float = PROBE_TIMEOUT_S) -> MediaInfo:
    """Probe a local file. Raises ``MediaError`` (or a subclass) if it is not readable media."""
    p = require_regular_file(path, tool="ffprobe")
    size = p.stat().st_size
    if size == 0:
        raise MediaError(f"empty file: {p}", tool="ffprobe")
    result = run_tool(
        "ffprobe",
        [
            "-hide_banner",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            input_url(p),
        ],
        timeout_s=timeout_s,
        stdout_limit=4 * 1024 * 1024,
    )
    try:
        data = json.loads(result.stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe printed invalid JSON: {exc}", tool="ffprobe") from None
    if not isinstance(data, dict):
        raise MediaError("ffprobe printed JSON that is not an object", tool="ffprobe")
    info = parse_probe_json(data, str(p), size)
    if not info.streams:
        raise MediaError(f"no streams found in {p}", tool="ffprobe")
    return info
