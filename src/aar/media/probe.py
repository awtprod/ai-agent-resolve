"""``probe(path)``: duration, size and streams of a media file, from ffprobe's JSON."""

from __future__ import annotations

import json
import math
import os
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

    @property
    def video(self) -> Stream | None:
        """The first real video stream (cover art is skipped)."""
        return next((s for s in self.streams if s.kind == "video" and not s.attached_pic), None)

    @property
    def audio(self) -> Stream | None:
        """The first audio stream: the one QA measures."""
        return next((s for s in self.streams if s.kind == "audio"), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "duration_s": self.duration_s,
            "size_bytes": self.size_bytes,
            "bit_rate": self.bit_rate,
            "streams": [s.to_dict() for s in self.streams],
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
            )
        )
    return MediaInfo(
        path=path,
        format=fmt.get("format_name"),
        duration_s=_num(fmt.get("duration")),
        size_bytes=size_bytes,
        bit_rate=_int(fmt.get("bit_rate")),
        streams=tuple(streams),
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
