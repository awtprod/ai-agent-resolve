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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fps"] = None if self.fps is None else f"{self.fps.numerator}/{self.fps.denominator}"
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
