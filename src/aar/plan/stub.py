"""``aar.plan.stub``: the plan stub ``aar plan produce`` takes, built from hub media files (S19a).

A stub is an EditPlan with sources and no segments. :func:`build_stub` makes one from media files
on the hub, so neither a person nor an agent writes it by hand. Per source it records the hub path
as ``media``, the video stream's frame count as ``duration``, the display size as ``frame_size``
and the start timecode (with its rate), all read with code that already exists:

- the frame count, the display size and the constant-rate check are the hub render's own
  (:func:`aar.plan.render.media.resolve_source`: one timing probe, every video packet's
  timestamp), so a stub source is a file the render can cut exactly;
- the frame rate is the one the producer counts timecodes at
  (:func:`aar.plan.produce.build.media_timecode_rate`: the video rate snapped to a standard one);
- the start timecode is the producer's (:func:`~aar.plan.produce.build.probe_sources` and
  :func:`~aar.plan.produce.build.source_start_timecodes`: ffprobe's tag, else the tmcd track read
  by :mod:`aar.media.quicktime`).

Refusals are :class:`StubError` with a stable ``code`` (the CLI exits 2): ``unreadable`` (missing,
not a regular readable file, or ffprobe cannot read it), ``no_video``, ``no_audio``, ``vfr`` (a
variable frame rate), ``rates_differ`` (the sources' rates differ from each other),
``timecode_unknown`` (no start timecode, unless ``allow_unknown_timecode``, the exporter's own
opt-out), ``timecode_ambiguous`` (the media's timecode rate cannot be told),
``duplicate_source_id``, ``duplicate_media`` (one file twice, directly or through a symlink),
``resolution_required`` (a short with no resolution), ``bad_base`` and
``invalid_stub``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.media import MediaError, probe
from aar.plan.model import EditPlan, Fps, ShotSize, Source, TargetKind
from aar.plan.produce.build import (
    Probed,
    media_timecode_rate,
    probe_sources,
    source_start_timecodes,
)
from aar.plan.produce.errors import ProduceInputError
from aar.plan.render.errors import RenderError
from aar.plan.render.media import resolve_source

#: The keys of a base JSON that pass through into the stub.
BASE_KEYS = ("template", "head", "outro", "audio_tracks")
#: At most this many media files in one stub.
MAX_SOURCES = 16

CODES = frozenset(
    {
        "unreadable",
        "no_video",
        "no_audio",
        "vfr",
        "rates_differ",
        "timecode_unknown",
        "timecode_ambiguous",
        "duplicate_source_id",
        "duplicate_media",
        "resolution_required",
        "bad_base",
        "invalid_stub",
    }
)


class StubError(ValueError):
    """A typed refusal: ``code`` (one of :data:`CODES`), ``message``, ``hint`` and details."""

    def __init__(self, code: str, message: str, hint: str = "", **details: Any) -> None:
        if code not in CODES:
            raise ValueError(f"unknown stub error code {code!r}")
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


@dataclass(frozen=True)
class StubSource:
    """One media file for the stub: its hub path and the optional labels a person gives."""

    media: str | os.PathLike[str]
    camera: str | None = None
    shot_size: ShotSize = "unknown"
    id: str | None = None


@dataclass(frozen=True)
class StubResult:
    """The stub and a JSON-ready report of what was read for each source."""

    plan: EditPlan
    report: dict[str, Any]


def _rate_text(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


def _cameras(sources: Sequence[StubSource]) -> list[str]:
    """Each source's camera label: its own, else A, B, C... by position (the profile's
    ``default_camera`` is A, so one unlabelled camera is on screen without a schedule)."""
    return [source.camera or chr(ord("A") + n) for n, source in enumerate(sources)]


def _source_ids(sources: Sequence[StubSource], cameras: Sequence[str]) -> list[str]:
    ids: list[str] = []
    for source, camera in zip(sources, cameras, strict=True):
        if source.id:
            ids.append(source.id)
        else:
            ids.append("cam-" + "".join(c if c.isalnum() else "-" for c in camera.lower()))
    seen: set[str] = set()
    for sid in ids:
        if sid in seen:
            raise StubError(
                "duplicate_source_id",
                f"two sources have the id {sid!r}",
                "give each source its own id (or camera label)",
                source=sid,
            )
        seen.add(sid)
    return ids


def _regular_file(media: str | os.PathLike[str], sid: str) -> Path:
    path = Path(os.path.abspath(media))
    try:
        str(path).encode("utf-8")
    except UnicodeEncodeError:
        raise StubError(
            "unreadable",
            f"source {sid!r}: the path {str(path)!r} is not valid UTF-8; a plan cannot name it",
            "rename the file (or its folder) to a UTF-8 name",
            source=sid,
        ) from None
    if not path.is_file() or not os.access(path, os.R_OK):
        what = "does not exist" if not path.exists() else "is not a readable regular file"
        raise StubError(
            "unreadable",
            f"source {sid!r}: {str(path)!r} {what}",
            "give the hub path of a media file (in the media root's incoming folder)",
            source=sid,
            path=str(path),
        )
    return path


def _probe_rate(path: Path, sid: str) -> tuple[Fraction, tuple[int, int] | None]:
    """The rate a stub plan runs at for this file (produce's own rule) and its display size."""
    try:
        info = probe(path)
    except MediaError as exc:
        raise StubError(
            "unreadable",
            f"source {sid!r}: ffprobe cannot read {str(path)!r}: {exc.summary()}",
            "check the file (aar media availability), or copy it to the hub again",
            source=sid,
            path=str(path),
        ) from None
    video = info.video
    if video is None:
        raise StubError(
            "no_video",
            f"source {sid!r}: {str(path)!r} has no video stream",
            "give the camera's video file (the proxy), not an audio-only file",
            source=sid,
            path=str(path),
        )
    if info.audio is None:
        raise StubError(
            "no_audio",
            f"source {sid!r}: {str(path)!r} has no audio stream; the transcript, the joins and "
            "the render need its sound",
            "give a file with the camera's sound",
            source=sid,
            path=str(path),
        )
    seen = Probed(video.display_size, info.timecode, video.fps, None, None, video.base_fps)
    rate, _ = media_timecode_rate(seen)
    if rate is None:
        raise StubError(
            "unreadable",
            f"source {sid!r}: {str(path)!r} states no video frame rate",
            "re-wrap the file (ffmpeg -c copy) and try again",
            source=sid,
            path=str(path),
        )
    return rate, video.display_size


_RENDER_CODES = {
    "source_vfr": ("vfr", "conform the file to a constant frame rate first"),
    "source_rate_mismatch": ("vfr", "conform the file to a constant, standard frame rate first"),
    "no_video_stream": ("no_video", "give the camera's video file"),
}


def _frames(path: Path, sid: str, fps: Fraction) -> tuple[int, tuple[int, int]]:
    """The video stream's frame count and display size, through the hub render's own checks."""
    probe_source = Source(id=sid, media=str(path))
    try:
        hub = resolve_source(
            probe_source,
            Fps(num=fps.numerator, den=fps.denominator),
            media_map=None,
            media_root=None,
        )
    except RenderError as exc:
        code, hint = _RENDER_CODES.get(exc.code, ("unreadable", exc.fix_hint))
        raise StubError(code, exc.message, hint, source=sid, path=str(path)) from None
    return hub.frames, hub.display


def _base_parts(base: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[Any], list[str]]:
    """The base's pass-through keys, the outro's source (when the base has an outro), and the keys
    that were ignored."""
    if base is None:
        return {}, [], []
    if not isinstance(base, Mapping):
        raise StubError("bad_base", "the base must be a JSON object", "give an EditPlan object")
    kept = {key: base[key] for key in BASE_KEYS if key in base and base[key] is not None}
    ignored = sorted(k for k in base if k not in (*BASE_KEYS, "sources"))
    extra: list[Any] = []
    outro = kept.get("outro")
    if outro is not None:
        name = outro.get("source") if isinstance(outro, Mapping) else None
        sources = base.get("sources")
        found = [
            s
            for s in (sources if isinstance(sources, list) else [])
            if isinstance(s, Mapping) and s.get("id") == name
        ]
        if len(found) != 1:
            raise StubError(
                "bad_base",
                f"the base's outro names source {name!r}, which its sources do not hold once",
                "give the outro clip's source in the base's sources",
                source=name,
            )
        extra.append(dict(found[0]))
    return kept, extra, ignored


def build_stub(
    sources: Sequence[StubSource],
    *,
    recording_id: str,
    project: str,
    timeline: str,
    kind: TargetKind,
    profile: str,
    resolution: tuple[int, int] | None = None,
    base: Mapping[str, Any] | None = None,
    allow_unknown_timecode: bool = False,
) -> StubResult:
    """Build a plan stub from hub media files (module docstring); a :class:`StubError` on refusal.

    Every source belongs to ``recording_id`` at origin 0 (``aar analyze sync`` sets the others'
    origins in ``aar plan produce --sync``). A source without a camera label gets A, B, C... by
    position, and without an id ``cam-<label>``. The plan runs at the first source's rate; the
    resolution defaults to its display size for a long-form."""
    if not sources:
        raise StubError("invalid_stub", "give at least one media file", "pass a media path")
    if len(sources) > MAX_SOURCES:
        raise StubError(
            "invalid_stub",
            f"{len(sources)} media files; a stub holds at most {MAX_SOURCES}",
            "split the show",
        )
    cameras = _cameras(sources)
    ids = _source_ids(sources, cameras)
    kept, extra, ignored = _base_parts(base)
    for item in extra:
        if item.get("id") in ids:
            raise StubError(
                "duplicate_source_id",
                f"the base's outro source id {item.get('id')!r} is also a media source's id",
                "rename one of them",
                source=item.get("id"),
            )
    paths = [_regular_file(source.media, sid) for source, sid in zip(sources, ids, strict=True)]
    seen_files: dict[str, str] = {}
    for path, sid in zip(paths, ids, strict=True):
        real = os.path.realpath(path)
        if real in seen_files:
            raise StubError(
                "duplicate_media",
                f"sources {seen_files[real]!r} and {sid!r} are the same file ({real!r})",
                "give each camera's own file once",
                sources=[seen_files[real], sid],
            )
        seen_files[real] = sid
    rates: list[Fraction] = []
    for path, sid in zip(paths, ids, strict=True):
        rates.append(_probe_rate(path, sid)[0])
    fps = rates[0]
    for rate, sid in zip(rates, ids, strict=True):
        if rate != fps:
            raise StubError(
                "rates_differ",
                f"source {sid!r} runs at {_rate_text(rate)} fps, source {ids[0]!r} at "
                f"{_rate_text(fps)}: one plan has one frame rate",
                "conform the files to one rate, or make a stub per rate",
                rates={i: _rate_text(r) for i, r in zip(ids, rates, strict=True)},
            )
    measured = [_frames(path, sid, fps) for path, sid in zip(paths, ids, strict=True)]
    if resolution is None:
        if kind != "longform":
            raise StubError(
                "resolution_required",
                "a short needs its resolution (e.g. 1080x1920)",
                "pass the resolution",
            )
        resolution = measured[0][1]
    stub_sources: list[dict[str, Any]] = [
        {
            "id": sid,
            "media": str(path),
            "camera": camera,
            "shot_size": source.shot_size,
            "recording_id": recording_id,
            "recording_origin": 0,
            "duration": frames,
            "frame_size": [display[0], display[1]],
        }
        for source, camera, sid, path, (frames, display) in zip(
            sources, cameras, ids, paths, measured, strict=True
        )
    ]
    data: dict[str, Any] = {
        "project": project,
        "timeline": timeline,
        "kind": kind,
        "fps": {"num": fps.numerator, "den": fps.denominator},
        "resolution": {"width": resolution[0], "height": resolution[1]},
        "profile": profile,
        "sources": stub_sources + extra,
        "segments": [],
        **kept,
    }
    plan = _validated(data)
    probed, warnings = probe_sources(plan, {}, None)
    try:
        timecodes, tc_warnings = source_start_timecodes(plan, probed, used=set(ids))
    except ProduceInputError as exc:
        raise StubError("timecode_ambiguous", exc.message, exc.hint) from None
    warnings += tc_warnings
    missing = [sid for sid in ids if sid not in timecodes]
    if missing and not allow_unknown_timecode:
        raise StubError(
            "timecode_unknown",
            "no start timecode could be read for " + ", ".join(repr(s) for s in missing),
            "check the file with aar media timecode FILE; without a timecode Resolve links clips "
            "only if the file starts at 00:00:00:00 (opt out with allow_unknown_timecode, as the "
            "export's --allow-unknown-timecode)",
            sources=missing,
        )
    report_sources = []
    for entry in stub_sources:
        found = timecodes.get(entry["id"])
        if found is not None:
            rate = found.rate or fps
            entry["start_timecode"] = found.frames
            entry["start_timecode_rate"] = {"num": rate.numerator, "den": rate.denominator}
        report_sources.append(
            {
                "id": entry["id"],
                "media": entry["media"],
                "camera": entry["camera"],
                "duration": entry["duration"],
                "frame_size": entry["frame_size"],
                "start_timecode": None if found is None else found.to_dict(fps),
            }
        )
    plan = _validated(data)
    report = {
        "fps": _rate_text(fps),
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "recording_id": recording_id,
        "sources": report_sources,
        "unknown_timecode": missing,
        "base_keys": sorted(kept),
        "base_ignored": ignored,
        "warnings": [w.model_dump(mode="json") for w in warnings],
    }
    return StubResult(plan, report)


def _validated(data: dict[str, Any]) -> EditPlan:
    try:
        return EditPlan.model_validate(data)
    except ValidationError as exc:
        raise StubError(
            "invalid_stub",
            f"the stub does not validate as an EditPlan ({exc.error_count()} problem(s))",
            "check the labels, names and the base JSON",
            errors=[
                {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]}
                for e in exc.errors()[:20]
            ],
        ) from None


__all__ = ["BASE_KEYS", "CODES", "StubError", "StubResult", "StubSource", "build_stub"]
