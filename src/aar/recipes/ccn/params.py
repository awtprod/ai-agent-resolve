"""The ``ccn-longform`` recipe's params (``extra="forbid"``), validated at the boundary.

Every hub path must be absolute, name an existing regular file, and lie under ``AAR_MEDIA_ROOT``
when that is set (resolved with ``realpath``, so a symlink cannot lead out of it). The media map
(hub root -> edit-PC root, the FCPXML export's own) must parse as the export parses it and must
cover every source, so a run never finds out at the export step that the PC cannot link a clip;
for the same reason two sources that would be one file on the PC (names that differ only in
case) are refused here.

**A camera of several files (S24a).** Sources that share a ``camera`` label and give ``part``
1..n are that camera's consecutive files. Only the reference camera (the first source's) may be
several files for now: another camera of several files is refused (placing those by audio sync
against the reference timeline is S24b). ``part_placement`` picks how the reference camera's
parts are placed on the recording clock (``timecode``, the default when every part's start
timecode is read, or ``contiguous``); see :mod:`aar.plan.stub`. ``laughs`` is refused for a
reference camera of several files: the laughs analysis is one paid call per file and its maps
are not merged yet.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aar.hub.writes import SEGMENT_RE, TitleText
from aar.plan.export.errors import ExportError
from aar.plan.export.media import (
    collision_text,
    media_map_from_mapping,
    media_ref,
    windows_collision,
)
from aar.plan.model import PartPlacement, ShotSize, Source
from aar.plan.stub import MAX_SOURCES as STUB_MAX_SOURCES
from aar.plan.stub import first_duplicate

MEDIA_ROOT_ENV = "AAR_MEDIA_ROOT"
#: Three cameras of about 15 files each (the stub's own bound).
MAX_SOURCES = STUB_MAX_SOURCES
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
    #: Part N (from 1) of this camera's consecutive files (module docstring).
    part: int | None = Field(default=None, ge=1, le=MAX_SOURCES)

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
    #: Camera label -> how its parts are placed; unset: by timecode when every part has one.
    part_placement: dict[Camera, PartPlacement] = Field(default_factory=dict, max_length=1)

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
    def _parts(self) -> CcnLongformParams:
        groups: dict[str, list[int]] = {}
        for n, source in enumerate(self.sources):
            if source.part is not None and source.camera is None:
                raise ValueError(f"sources[{n}] gives part {source.part} but no camera label")
            if source.camera is not None:
                groups.setdefault(source.camera, []).append(n)
        reference = self.sources[0].camera
        for camera, members in groups.items():
            parts = [self.sources[n].part for n in members]
            if len(members) > 1 and None in parts:
                raise ValueError(
                    f"camera {camera!r} is given {len(members)} times (sources {members}); "
                    "give each file of one camera its part (1, 2, ...), or each camera its own "
                    "label"
                )
            if None not in parts and sorted(parts) != list(range(1, len(members) + 1)):  # type: ignore[type-var]
                raise ValueError(
                    f"camera {camera!r} has parts {parts}; its files must be parts 1 to "
                    f"{len(members)}, each once"
                )
            if len(members) > 1 and camera != reference:
                raise ValueError(
                    f"camera {camera!r} is several files; only the reference camera (the first "
                    f"source's, {reference!r}) may be several files for now: give that camera "
                    "as one file, or leave it out (other cameras of several files, placed by "
                    "audio sync, are slice S24b)"
                )
        for camera in self.part_placement:
            if len(groups.get(camera, [])) < 2:
                raise ValueError(
                    f"part_placement names camera {camera!r}, which is not several files"
                )
        if self.laughs and self.reference_parts() > 1:
            raise ValueError(
                "laughs is not supported yet for a reference camera of several files (one paid "
                "analysis per file, and the maps are not merged): set laughs to false"
            )
        return self

    def reference_parts(self) -> int:
        """How many files the reference camera (the first source's) is."""
        camera = self.sources[0].camera
        if camera is None:
            return 1
        return sum(1 for source in self.sources if source.camera == camera)

    def cameras(self) -> int:
        """How many cameras the sources are: a camera of several files counts once, and an
        unlabelled source is a camera of its own."""
        return len({source.camera or f"#{n}" for n, source in enumerate(self.sources)})

    @model_validator(mode="after")
    def _covered(self) -> CcnLongformParams:
        mapping = media_map_from_mapping(self.media_map)
        refs = []
        for n, source in enumerate(self.sources):
            try:
                refs.append(
                    media_ref(Source(id=source.id or f"source-{n}", media=source.path), mapping)
                )
            except ExportError as exc:
                raise ValueError(
                    f"sources[{n}] {source.path!r}: the media map does not give the edit PC's "
                    f"path for it ({exc.message})"
                ) from None
        clash = windows_collision(refs)
        if clash is not None:
            a, b = (next(n for n, ref in enumerate(refs) if ref is found) for found in clash)
            raise ValueError(
                f"sources[{a}] and sources[{b}]: {collision_text(*clash)}; rename one of them "
                "on the hub so the names differ by more than case"
            )
        return self

    def pause_trim_on(self) -> bool:
        """``auto`` is off for one camera (room audio: trims cut real material, the S19 pilot)
        and on for several (a board feed). Cameras are counted, not files: one camera of several
        files is still one camera's room audio."""
        if self.pause_trim == "auto":
            return self.cameras() > 1
        return self.pause_trim == "on"
