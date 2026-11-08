"""The producer's input files, validated at the boundary.

* **Plan stub**: an EditPlan whose ``segments`` and ``transitions`` are empty. Its ``sources``
  are the source list (media, camera, recording, origin, duration); every other field (project,
  timeline, kind, fps, resolution, template, head, outro, audio tracks) passes through.
* **Keep list** (``KeepList``): the recording-time ranges to keep, in seconds, chronological.
* **Camera schedule** (``CameraSchedule``): recording-time spans, each naming the camera on
  screen. A camera is a ``Source.camera`` label, or a source id when no label matches.
* **Sync report**: ``aar analyze sync`` output; each ``ok`` pair sets the matching source's
  ``recording_origin``.
* **Face track**: ``aar analyze faces`` output (``aar.faces/1``), bound to the source whose
  media is the file it was measured on.

Transcripts and laugh maps use ``aar.plan.inputs``. Times are recording-time seconds unless a
name says ``file``.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aar.plan.inputs import LaughMap, Transcript, load_laughs, load_transcript
from aar.plan.model import EditPlan
from aar.plan.produce.errors import ProduceInputError


class _Own(BaseModel):
    """A format this module defines: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _Foreign(BaseModel):
    """Another tool's output: the fields used are checked, the rest is ignored."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class KeepRange(_Own):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    label: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> KeepRange:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"keep range {self.start_s:g}-{self.end_s:g} s does not end after it starts"
            )
        return self


class KeepList(_Own):
    """The ranges of one recording to keep, chronological and not overlapping."""

    recording: str | None = None
    ranges: list[KeepRange] = Field(min_length=1)

    @model_validator(mode="after")
    def _chronological(self) -> KeepList:
        for prev, cur in zip(self.ranges, self.ranges[1:], strict=False):
            if cur.start_s < prev.end_s:
                raise ValueError(
                    f"keep range {cur.start_s:g}-{cur.end_s:g} s starts before the previous one "
                    f"ends ({prev.end_s:g} s): keep ranges are chronological and never overlap"
                )
        return self


class CameraSpan(_Own):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    camera: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> CameraSpan:
        if self.end_s <= self.start_s:
            raise ValueError(f"camera span {self.start_s:g}-{self.end_s:g} s is empty or reversed")
        return self


class CameraSchedule(_Own):
    """Which camera is on screen when. Spans are chronological and never overlap."""

    recording: str | None = None
    spans: list[CameraSpan] = Field(min_length=1)

    @model_validator(mode="after")
    def _chronological(self) -> CameraSchedule:
        for prev, cur in zip(self.spans, self.spans[1:], strict=False):
            if cur.start_s < prev.end_s:
                raise ValueError(
                    f"camera span at {cur.start_s:g} s starts before the previous one ends "
                    f"({prev.end_s:g} s)"
                )
        return self


class Face(_Foreign):
    box: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _normalised(self) -> Face:
        w, h = self.box[2], self.box[3]
        if not all(0 <= v <= 1 for v in self.box) or w <= 0 or h <= 0:
            raise ValueError("a face box is [x, y, w, h] in 0-1 with a positive size")
        return self

    @property
    def centre_x(self) -> float:
        return self.box[0] + self.box[2] / 2

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]


class FaceFrame(_Foreign):
    file_s: float = Field(ge=0)
    faces: list[Face]

    @model_validator(mode="after")
    def _largest_first(self) -> FaceFrame:
        # The faces worker writes them largest first, but the order is not trusted: the camera
        # follows ``faces[0]``, so it is made the largest here (stable for equal sizes).
        self.faces = sorted(self.faces, key=lambda face: -face.area)
        return self


class FaceInput(_Foreign):
    path: str = Field(min_length=1)


class FaceHeader(_Foreign):
    frame_size: list[int] = Field(min_length=2, max_length=2)
    input: FaceInput
    recording: str | None = None

    @model_validator(mode="after")
    def _size(self) -> FaceHeader:
        if any(v <= 0 for v in self.frame_size):
            raise ValueError("frame_size must be two positive numbers")
        return self


class FaceTrack(_Foreign):
    """The parts of an ``aar.faces/1`` document the camera path uses."""

    schema_id: Literal["aar.faces/1"] = Field(alias="schema")
    header: FaceHeader
    frames: list[FaceFrame]

    @model_validator(mode="after")
    def _increasing(self) -> FaceTrack:
        times = [frame.file_s for frame in self.frames]
        if any(b <= a for a, b in itertools.pairwise(times)):
            raise ValueError("face frames must be in strictly increasing file_s order")
        return self

    @property
    def aspect(self) -> float:
        """Source width over height (the sampled frames keep the display aspect)."""
        return self.header.frame_size[0] / self.header.frame_size[1]

    def samples(self) -> list[tuple[float, float]]:
        """(file seconds, centre x) of the largest face in every frame that has one."""
        return [(f.file_s, f.faces[0].centre_x) for f in self.frames if f.faces]


class SyncReference(_Foreign):
    file: str = Field(min_length=1)
    recording_origin: int


class SyncPair(_Foreign):
    file: str = Field(min_length=1)
    status: str
    suggested_recording_origin: int | None = None


class SyncReport(_Foreign):
    fps: str
    reference: SyncReference
    pairs: list[SyncPair]


M = TypeVar("M", bound=BaseModel)


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProduceInputError(
            "input-missing", f"cannot read the {what} {path}: {exc.strerror or exc}"
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProduceInputError("invalid-json", f"the {what} {path} is not JSON: {exc}") from None


def _details(exc: ValidationError) -> list[Any]:
    return json.loads(exc.json(include_url=False, include_context=False, include_input=False))


def _validate(model: type[M], data: Any, path: Path, what: str, code: str) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ProduceInputError(
            code, f"the {what} {path} failed validation", details=_details(exc)
        ) from None


def _wrap(data: Any, key: str) -> Any:
    return {key: data} if isinstance(data, list) else data


def load_keep(path: str | os.PathLike[str]) -> KeepList:
    """A keep list: ``{"recording"?, "ranges": [{start_s, end_s, label?}]}`` or the bare list."""
    p = Path(path)
    data = _wrap(_read_json(p, "keep list"), "ranges")
    return _validate(KeepList, data, p, "keep list", "invalid-keep")


def load_schedule(path: str | os.PathLike[str]) -> CameraSchedule:
    """A camera schedule: ``{"recording"?, "spans": [{start_s, end_s, camera}]}`` or the list."""
    p = Path(path)
    data = _wrap(_read_json(p, "camera schedule"), "spans")
    return _validate(CameraSchedule, data, p, "camera schedule", "invalid-schedule")


def load_faces(path: str | os.PathLike[str]) -> FaceTrack:
    """An ``aar.faces/1`` document; its ``header.input.path`` resolves against its folder."""
    p = Path(path)
    track = _validate(FaceTrack, _read_json(p, "face track"), p, "face track", "invalid-faces")
    media = Path(track.header.input.path)
    if not media.is_absolute():
        track.header.input.path = str(p.parent / media)
    return track


def load_sync(path: str | os.PathLike[str]) -> SyncReport:
    """An ``aar analyze sync`` report; relative file names resolve against its folder."""
    p = Path(path)
    report = _validate(SyncReport, _read_json(p, "sync report"), p, "sync report", "invalid-sync")
    for item in (report.reference, *report.pairs):
        if not Path(item.file).is_absolute():
            item.file = str(p.parent / item.file)
    return report


def load_stub(path: str | os.PathLike[str]) -> EditPlan:
    """A plan stub: an EditPlan with no segments and no transitions."""
    p = Path(path)
    plan = _validate(EditPlan, _read_json(p, "plan stub"), p, "plan stub", "invalid-stub")
    if plan.segments or plan.transitions:
        raise ProduceInputError(
            "stub-has-segments",
            f"the plan stub {p} already has segments or transitions",
            hint="give a stub with empty segments and transitions; the producer writes them",
        )
    return plan


def _input(loader: Callable[[Path], M], path: Path, what: str) -> M:
    try:
        return loader(path)
    except ValidationError as exc:
        raise ProduceInputError(
            f"invalid-{what}", f"the {what} {path} failed validation", details=_details(exc)
        ) from None
    except OSError as exc:
        raise ProduceInputError(
            "input-missing", f"cannot read the {what} {path}: {exc.strerror or exc}"
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProduceInputError("invalid-json", f"the {what} {path} is not JSON: {exc}") from None


def read_transcript(path: str | os.PathLike[str]) -> Transcript:
    """A transcript (``aar.plan.inputs``), with its errors typed."""
    return _input(load_transcript, Path(path), "transcript")


def read_laughs(path: str | os.PathLike[str]) -> LaughMap:
    """A laugh map (``aar.plan.inputs``), with its errors typed."""
    return _input(load_laughs, Path(path), "laughs")


def same_file(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    """True when two paths name one file: the same inode when both exist, else the same
    normalised absolute path."""
    pa, pb = Path(a), Path(b)
    try:
        return os.path.samefile(pa, pb)
    except OSError:
        return os.path.normpath(pa.absolute()) == os.path.normpath(pb.absolute())
