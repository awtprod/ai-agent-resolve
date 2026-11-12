"""The ``ccn-longform`` recipe's params (``extra="forbid"``), validated at the boundary.

Every hub path must be absolute, name an existing regular file, and lie under ``AAR_MEDIA_ROOT``
when that is set (resolved with ``realpath``, so a symlink cannot lead out of it). The media map
(hub root -> edit-PC root, the FCPXML export's own) must parse as the export parses it and must
cover every source, so a run never finds out at the export step that the PC cannot link a clip.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aar.hub.writes import SEGMENT_RE, TitleText
from aar.plan.export.errors import ExportError
from aar.plan.export.media import media_map_from_mapping, media_ref
from aar.plan.model import ShotSize, Source
from aar.plan.stub import first_duplicate

MEDIA_ROOT_ENV = "AAR_MEDIA_ROOT"
MAX_SOURCES = 8
MAX_MAP_ENTRIES = 16
MAX_PATH = 1024

PauseTrim = Literal["auto", "on", "off"]
Camera = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,15}$")]
Segment = Annotated[str, Field(pattern=SEGMENT_RE)]
#: A Whisper model name, never a path: models are looked up under AAR_MODEL_DIR only.
ModelName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]


def media_root() -> str | None:
    """The allowed media root (``AAR_MEDIA_ROOT``), resolved, or None when unset."""
    root = os.environ.get(MEDIA_ROOT_ENV, "").strip()
    return os.path.realpath(root) if root else None


def check_hub_file(value: str) -> str:
    """``value`` if it is an absolute path to an existing regular file inside the media root."""
    if "\x00" in value or len(value) > MAX_PATH:
        raise ValueError("not a usable path")
    if not os.path.isabs(value):
        raise ValueError(f"{value!r} is not an absolute hub path")
    if any(part in (".", "..") for part in value.split("/")):
        raise ValueError(f"{value!r} has a '.' or '..' component")
    real = os.path.realpath(value)
    root = media_root()
    if root is not None and not (real == root or real.startswith(root.rstrip("/") + "/")):
        raise ValueError(f"{value!r} is not under the media root ({MEDIA_ROOT_ENV})")
    if not os.path.isfile(real):
        raise ValueError(f"{value!r} does not exist or is not a regular file")
    return value


class SourceParam(BaseModel):
    """One camera file on the hub (the proxy), with its labels."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=MAX_PATH)
    camera: Camera | None = None
    shot_size: ShotSize = "unknown"
    id: Segment | None = None

    @field_validator("path")
    @classmethod
    def _hub_file(cls, value: str) -> str:
        return check_hub_file(value)


class CcnLongformParams(BaseModel):
    """What one ``ccn-longform`` run needs. The first source is the reference: the transcript,
    the laughs and the sync all measure against it."""

    model_config = ConfigDict(extra="forbid")

    sources: list[SourceParam] = Field(min_length=1, max_length=MAX_SOURCES)
    recording_id: Segment
    timeline: TitleText
    kind: Literal["longform", "short"] = "longform"
    media_map: dict[str, str] = Field(min_length=1, max_length=MAX_MAP_ENTRIES)
    whisper_model: ModelName = "medium.en"
    whisper_threads: int = Field(default=0, ge=0, le=64)
    normalize_audio: bool = True
    pause_trim: PauseTrim = "auto"
    laughs: bool = False
    allow_unknown_timecode: bool = False

    @field_validator("media_map")
    @classmethod
    def _media_map(cls, value: dict[str, str]) -> dict[str, str]:
        try:
            media_map_from_mapping(value)
        except ExportError as exc:
            raise ValueError(exc.message) from None
        return value

    @model_validator(mode="after")
    def _distinct(self) -> CcnLongformParams:
        dup = first_duplicate([source.path for source in self.sources])
        if dup is not None:
            n = dup[1]
            raise ValueError(
                f"sources[{n}] {self.sources[n].path!r} is a file already given (sources"
                f"[{dup[0]}]: the same path, a symlink or a hardlink)"
            )
        return self

    @model_validator(mode="after")
    def _covered(self) -> CcnLongformParams:
        mapping = media_map_from_mapping(self.media_map)
        for n, source in enumerate(self.sources):
            try:
                media_ref(Source(id=source.id or f"source-{n}", media=source.path), mapping)
            except ExportError as exc:
                raise ValueError(
                    f"sources[{n}] {source.path!r}: the media map does not give the edit PC's "
                    f"path for it ({exc.message})"
                ) from None
        return self

    def pause_trim_on(self) -> bool:
        """``auto`` is off for one camera (room audio: trims cut real material, the S19 pilot)
        and on for several (a board feed)."""
        if self.pause_trim == "auto":
            return len(self.sources) > 1
        return self.pause_trim == "on"
