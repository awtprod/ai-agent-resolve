"""What the writer must know about one media file on the edit PC, checked at the boundary.

The facts are measured on another machine (``ffprobe -show_format -show_streams`` plus the file's
size and modification time, gathered by a script on the edit PC), so everything here is treated
as untrusted: types are strict (a number sent as a string is refused), every text, list and
mapping is bounded, and the timestamps must parse. Only the ffprobe fields the writer reads are
typed; the others ffprobe prints vary by version and are ignored. Digits are matched as ASCII
only (``[0-9]``, never ``\\d``, and never Python's ``int()`` on unchecked text, which reads other
scripts' digits): a timecode or rotation tag in other digits is refused, not read. The path is
checked by the edit PC's one path validator (:func:`aar.host.winpath.check_file_path`) and the
hub's check for characters a reader cannot see (:func:`aar.hub.lessons.hidden_character`).
Derived values are computed in :mod:`aar.drp.mediapool.formulas`, never here.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from aar.drp.mediapool.errors import MediaPoolError

__all__ = [
    "MAX_FACTS_BYTES",
    "MediaFacts",
    "ProbeFormat",
    "ProbeStream",
    "facts_from_json",
]

#: The largest JSON text accepted for one file's facts.
MAX_FACTS_BYTES = 1024 * 1024
MAX_STREAMS = 64
MAX_TAGS = 256
MAX_TEXT = 4096
MAX_PATH = 1024
MAX_SIDE_DATA = 32

_Text = Annotated[StrictStr, Field(max_length=MAX_TEXT)]
_Tags = Annotated[
    dict[Annotated[StrictStr, Field(max_length=256)], _Text], Field(max_length=MAX_TAGS)
]
_Count = Annotated[StrictInt, Field(ge=0, lt=2**62)]
_Pixels = Annotated[StrictInt, Field(ge=1, le=65536)]
_DIGITS = re.compile(r"[0-9]{1,9}\Z")
_RATIONAL = re.compile(r"-?[0-9]{1,12}/-?[0-9]{1,12}\Z")
# PowerShell's (and C's ctime) local time text: "Fri Mar 13 04:05:06 2026" (one space or two
# before a one-digit day).
_LOCAL = re.compile(
    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +"
    r"([0-9]{1,2}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) ([0-9]{4})\Z"
)
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# .NET's round-trip UTC text: "2026-03-13T09:05:06.1234567Z" (up to 7 fractional digits).
_UTC = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?Z\Z"
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
#: A ``timecode`` tag (any case), as ffprobe prints it: ASCII digits only.
TIMECODE_TAG = re.compile(r"[0-9]{2}:[0-9]{2}:[0-9]{2}[:;.][0-9]{2,3}\Z")
#: A ``rotate`` tag: whole degrees in ASCII digits.
_ROTATE_TAG = re.compile(r"-?[0-9]{1,4}\Z")
#: The most a display rotation may say, in degrees (ffprobe gives -360..360).
_MAX_ROTATION = 360.0


def _no_control(value: str | None) -> str | None:
    if value is not None and _CONTROL.search(value):
        raise ValueError("text holds a control character")
    return value


def _checked_tags(tags: dict[str, str]) -> dict[str, str]:
    """The tags the writer reads, checked: a ``timecode`` tag (any case) must be ``HH:MM:SS:FF``
    in ASCII digits and a ``rotate`` tag whole degrees in ASCII digits."""
    for key, value in tags.items():
        lower = key.lower()
        if lower == "timecode" and not TIMECODE_TAG.match(value):
            raise ValueError(f"the {key} tag {value[:40]!r} is not HH:MM:SS:FF in ASCII digits")
        if lower == "rotate" and not _ROTATE_TAG.match(value):
            raise ValueError(f"the rotate tag {value[:40]!r} is not whole degrees in ASCII digits")
    return tags


class ProbeStream(BaseModel):
    """One ffprobe stream: only the fields the writer reads are typed."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    index: Annotated[StrictInt, Field(ge=0, le=1023)]
    codec_type: _Text | None = None
    codec_name: _Text | None = None
    codec_tag_string: _Text | None = None
    profile: _Text | None = None
    width: _Pixels | None = None
    height: _Pixels | None = None
    pix_fmt: _Text | None = None
    color_space: _Text | None = None
    color_transfer: _Text | None = None
    color_primaries: _Text | None = None
    color_range: _Text | None = None
    sample_fmt: _Text | None = None
    sample_rate: _Text | None = None
    channels: Annotated[StrictInt, Field(ge=1, le=64)] | None = None
    channel_layout: _Text | None = None
    bits_per_sample: Annotated[StrictInt, Field(ge=0, le=64)] | None = None
    bits_per_raw_sample: _Text | None = None
    r_frame_rate: _Text | None = None
    avg_frame_rate: _Text | None = None
    time_base: _Text | None = None
    duration_ts: _Count | None = None
    nb_frames: _Text | None = None
    disposition: Annotated[
        dict[Annotated[StrictStr, Field(max_length=64)], StrictInt], Field(max_length=64)
    ] = {}
    tags: _Tags = {}
    side_data_list: Annotated[list[dict[str, Any]], Field(max_length=MAX_SIDE_DATA)] = []

    @field_validator(
        "codec_type", "codec_name", "codec_tag_string", "pix_fmt", "sample_fmt", "channel_layout"
    )
    @classmethod
    def _plain(cls, value: str | None) -> str | None:
        return _no_control(value)

    @field_validator("sample_rate", "bits_per_raw_sample", "nb_frames")
    @classmethod
    def _digits(cls, value: str | None) -> str | None:
        if value is not None and not _DIGITS.match(value):
            raise ValueError(f"expected decimal digits, got {value[:40]!r}")
        return value

    @field_validator("r_frame_rate", "avg_frame_rate", "time_base")
    @classmethod
    def _rational(cls, value: str | None) -> str | None:
        if value is not None and not _RATIONAL.match(value):
            raise ValueError(f"expected a rational like 30000/1001, got {value[:40]!r}")
        return value

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: dict[str, str]) -> dict[str, str]:
        return _checked_tags(value)

    @field_validator("side_data_list")
    @classmethod
    def _side(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in value:
            if "rotation" in item:
                rotation = item["rotation"]
                if (
                    isinstance(rotation, bool)
                    or not isinstance(rotation, int | float)
                    or not math.isfinite(rotation)
                    or abs(rotation) > _MAX_ROTATION
                ):
                    raise ValueError(f"a side data rotation of {str(rotation)[:40]!r} degrees")
            if len(item) > 64:
                raise ValueError("a side data entry has more than 64 fields")
            for key, field in item.items():
                if not isinstance(key, str) or len(key) > 64:
                    raise ValueError("a side data key is not a short text")
                if isinstance(field, str) and len(field) > MAX_TEXT:
                    raise ValueError("a side data text is too long")
                if not isinstance(field, str | int | float | bool | type(None)):
                    raise ValueError("a side data value is not a scalar")
        return value

    @property
    def is_attached_picture(self) -> bool:
        return self.disposition.get("attached_pic", 0) == 1


class ProbeFormat(BaseModel):
    """ffprobe's ``format`` section: only the fields the writer reads are typed."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    format_name: _Text | None = None
    duration: _Text | None = None
    tags: _Tags = {}

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: dict[str, str]) -> dict[str, str]:
        return _checked_tags(value)


class Probe(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    format: ProbeFormat
    streams: Annotated[list[ProbeStream], Field(max_length=MAX_STREAMS)]

    @field_validator("streams")
    @classmethod
    def _unique(cls, value: list[ProbeStream]) -> list[ProbeStream]:
        indexes = [s.index for s in value]
        if len(set(indexes)) != len(indexes):
            raise ValueError("two streams have the same index")
        return value


def check_pc_path(value: str) -> str:
    """An edit-PC file path as Resolve will store it. The edit PC's path rules are
    :func:`aar.host.winpath.check_file_path` (one implementation: a drive or UNC root, no empty,
    ``.`` or ``..`` component, so no doubled separator; no reserved device name; no character
    Windows forbids in a name, the colon among them; no trailing dot or space; no control
    character or lone surrogate). The writer adds three: no character a reader cannot see
    (:func:`aar.hub.lessons.hidden_character`: bidirectional, zero-width, C1 control and other
    format characters, which would be written into the clip unseen); backslashes only (Resolve's
    folder and name are split at the last backslash, and how it stores a ``/`` was not seen);
    and a folder under the root (how Resolve writes the folder of a file at a drive's or share's
    root was not seen)."""
    from aar.host.errors import PathRefused
    from aar.host.winpath import check_file_path
    from aar.hub.lessons import hidden_character

    try:
        win = check_file_path(value)
    except PathRefused as exc:
        raise ValueError(f"the path is not a valid edit-PC file path ({exc.reason})") from None
    hidden = hidden_character(value)
    if hidden is not None:
        raise ValueError(f"the path holds a character a reader cannot see (U+{ord(hidden):04X})")
    if "/" in value:
        raise ValueError("the path must use backslashes only, as Resolve stores it")
    if len(win.parts) < 2:
        raise ValueError("the path must name a file in a folder, not at the root of a drive")
    return value


def parse_local_time(text: str) -> datetime:
    """``"Fri Mar 13 04:05:06 2026"`` (the edit PC's wall clock) as a naive datetime; the
    weekday must agree with the date."""
    match = _LOCAL.match(text)
    if match is None:
        raise ValueError(f"expected a time like 'Fri Mar 13 04:05:06 2026', got {text[:40]!r}")
    day, month = int(match.group(3)), _MONTHS.index(match.group(2)) + 1
    hour, minute, second, year = (int(match.group(n)) for n in (4, 5, 6, 7))
    value = datetime(year, month, day, hour, minute, second)  # raises on a bad date
    if _DAYS[value.weekday()] != match.group(1):
        raise ValueError(f"{text!r}: the weekday does not match the date")
    return value


def parse_utc_ns(text: str) -> int:
    """``"2026-03-13T09:05:06.1234567Z"`` as nanoseconds since 1970 (fraction kept exactly)."""
    match = _UTC.match(text)
    if match is None:
        raise ValueError(
            f"expected a UTC time like 2026-03-13T09:05:06.1234567Z, got {text[:40]!r}"
        )
    whole = datetime(*(int(match.group(n)) for n in range(1, 7)), tzinfo=UTC)
    fraction = (match.group(7) or "0").ljust(9, "0")
    return int(whole.timestamp()) * 1_000_000_000 + int(fraction)


class MediaFacts(BaseModel):
    """One file on the edit PC: its path as Resolve stores it, size, modification time (the PC's
    wall clock and UTC) and ffprobe's format and streams."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[StrictStr, Field(min_length=4, max_length=MAX_PATH)]
    size: Annotated[StrictInt, Field(ge=0, lt=2**53)]
    mtime_local: datetime
    mtime_utc_ns: Annotated[StrictInt, Field(ge=0, lt=2**63)]
    ctime_utc_ns: Annotated[StrictInt, Field(ge=0, lt=2**63)] | None = None
    probe: Probe

    @field_validator("path")
    @classmethod
    def _windows_path(cls, value: str) -> str:
        return check_pc_path(value)

    @property
    def folder(self) -> str:
        """The folder part as Resolve stores it (``F:\\Show\\Sound``; no trailing backslash)."""
        return self.path.rpartition("\\")[0]

    @property
    def name(self) -> str:
        return self.path.rpartition("\\")[2]

    @property
    def extension(self) -> str:
        """The lower-case extension without the dot, or ``""``."""
        stem, dot, ext = self.name.rpartition(".")
        return ext.lower() if dot and stem else ""

    @property
    def streams(self) -> list[ProbeStream]:
        return self.probe.streams


def facts_from_json(data: object) -> MediaFacts:
    """Facts from one collected record: ``{"path", "exists", "size", "mtime_local", "mtime_utc",
    "ctime_utc", "ffprobe": {"format", "streams"}}`` (or a JSON text of it, at most
    :data:`MAX_FACTS_BYTES`). Raises :class:`MediaPoolError` (``bad_facts``) saying what is wrong.
    """
    try:
        if isinstance(data, bytes | str):
            size = len(data if isinstance(data, bytes) else data.encode("utf-8", "surrogatepass"))
            if size > MAX_FACTS_BYTES:
                raise ValueError(f"the facts are longer than {MAX_FACTS_BYTES} bytes")
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError("the facts are not a JSON object")
        # A lone surrogate (JSON "\\ud800") is text no file name, tag or blob can hold.
        json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        known = {"path", "exists", "size", "mtime_local", "mtime_utc", "ctime_utc", "ffprobe"}
        extra = sorted(str(k)[:40] for k in data if k not in known)
        if extra:
            raise ValueError(f"unknown keys {extra[:5]}")
        if data.get("exists") is not True:
            raise ValueError("the file did not exist when it was measured")
        for key in ("mtime_local", "mtime_utc"):
            if not isinstance(data.get(key), str):
                raise ValueError(f"{key} must be a text")
        ctime = data.get("ctime_utc")
        if ctime is not None and not isinstance(ctime, str):
            raise ValueError("ctime_utc must be a text")
        return MediaFacts(
            path=data.get("path"),  # type: ignore[arg-type]
            size=data.get("size"),  # type: ignore[arg-type]
            mtime_local=parse_local_time(data["mtime_local"]),
            mtime_utc_ns=parse_utc_ns(data["mtime_utc"]),
            ctime_utc_ns=None if ctime is None else parse_utc_ns(ctime),
            probe=Probe.model_validate(data.get("ffprobe")),
        )
    except MediaPoolError:
        raise
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "facts"
        raise MediaPoolError(
            "bad_facts",
            f"the facts are malformed at {where}: {first.get('msg', 'invalid')}",
            "measure the file again with the facts script on the edit PC",
        ) from None
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise MediaPoolError(
            "bad_facts",
            f"the facts are malformed: {str(exc)[:200]}",
            "measure the file again with the facts script on the edit PC",
        ) from None
