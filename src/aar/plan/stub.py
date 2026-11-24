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

**A camera of several files (S24a).** Sources that share a camera label and give ``part`` 1..n
are one camera's consecutive files (a camera that splits its recording, or an operator who
stopped and started). The reference camera (the camera of the first source) may have parts:
they go first in the stub, in part order, with ids ``cam-<label>-p<n>``, and each part's
``recording_origin`` is placed on the recording clock by ``placement``: ``timecode`` (the default
when every part's start timecode was read: the part's start timecode less the first part's, a
midnight wrap counted as the next day) or ``contiguous`` (each part starts where the previous
one ends; the default without timecodes). Refusals: ``bad_parts`` (a part without a camera
label, a label repeated without parts, part numbers that are not 1..n), ``parts_unsupported``
(another camera of several files: S24b), ``parts_overlap`` (a part starts more than
:data:`PART_OVERLAP_FRAMES` before the previous one ends), ``parts_out_of_order`` (a part's
timecode is before the previous part's, and not a midnight wrap) and ``parts_rate`` (a
timecode counted at another rate than the plan's). Each part's origin, duration, gap to the
previous part and placement are in the report's ``parts``.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.media import MediaError, probe
from aar.media.probe import DROP_FRAME_RATES
from aar.plan.model import EditPlan, Fps, PartPlacement, ShotSize, Source, TargetKind
from aar.plan.produce.build import (
    Probed,
    media_timecode_rate,
    probe_sources,
    source_start_timecodes,
)
from aar.plan.produce.errors import ProduceInputError
from aar.plan.produce.report import PlannerWarning
from aar.plan.render.errors import RenderError
from aar.plan.render.media import resolve_source

#: The keys of a base JSON that pass through into the stub.
BASE_KEYS = ("template", "head", "outro", "audio_tracks")
#: At most this many media files in one stub (three cameras of about 15 files each).
MAX_SOURCES = 48
#: A part may start this many frames before the previous part ends (timecode rounding); more is
#: an overlap and refused.
PART_OVERLAP_FRAMES = 2
#: A part whose timecode is before the previous part's is a midnight wrap when the wrapped gap
#: is at most this long; otherwise it is out of order.
WRAP_MAX_GAP_S = 12 * 3600

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
        "bad_parts",
        "parts_unsupported",
        "parts_overlap",
        "parts_out_of_order",
        "parts_rate",
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
    part: int | None = None


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


def _source_ids(
    sources: Sequence[StubSource], cameras: Sequence[str], parted: Collection[str] = ()
) -> list[str]:
    """Each source's id: its own, else ``cam-<label>``, with ``-p<n>`` for a part of a camera of
    several files (``parted``)."""
    ids: list[str] = []
    for source, camera in zip(sources, cameras, strict=True):
        if source.id:
            ids.append(source.id)
        else:
            sid = "cam-" + "".join(c if c.isalnum() else "-" for c in camera.lower())
            ids.append(f"{sid}-p{source.part}" if camera in parted else sid)
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


def first_duplicate(paths: Sequence[str | os.PathLike[str]]) -> tuple[int, int] | None:
    """``(i, j)``, ``i < j``, for the first path ``j`` that names the same file as an earlier
    path ``i``: the same resolved path (a symlink) or the same ``(st_dev, st_ino)`` (a hardlink).
    None when every path names its own file. A path that cannot be stat'ed is compared by its
    resolved path alone; the callers refuse such a path on their own existence check."""
    seen: dict[object, int] = {}
    for j, path in enumerate(paths):
        keys: list[object] = [os.path.realpath(path)]
        with contextlib.suppress(OSError):
            st = os.stat(path)
            keys.append((st.st_dev, st.st_ino))
        for key in keys:
            if key in seen:
                return seen[key], j
        for key in keys:
            seen.setdefault(key, j)
    return None


def _probe_rate(
    path: Path, sid: str, *, require_audio: bool = True
) -> tuple[Fraction, tuple[int, int] | None]:
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
    if require_audio and info.audio is None:
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


def _parts(sources: Sequence[StubSource], cameras: Sequence[str]) -> dict[str, list[int]]:
    """The cameras of several files (labels with ``part`` 1..n), each with its sources' indexes
    in part order; a :class:`StubError` for parts that are not well formed (module docstring)."""
    groups: dict[str, list[int]] = {}
    for n, (source, camera) in enumerate(zip(sources, cameras, strict=True)):
        if source.part is not None and not source.camera:
            raise StubError(
                "bad_parts",
                f"media file {n} gives part {source.part} but no camera label",
                "give every part its camera's label, the same for all of that camera's files",
            )
        groups.setdefault(camera, []).append(n)
    parted: dict[str, list[int]] = {}
    for camera, members in groups.items():
        numbers = [sources[n].part for n in members]
        if all(number is None for number in numbers):
            if len(members) > 1:
                raise StubError(
                    "bad_parts",
                    f"camera {camera!r} is given {len(members)} times without part numbers",
                    "give each file of one camera its part (1, 2, ...), or each camera its own "
                    "label",
                    camera=camera,
                )
            continue
        if any(number is None for number in numbers) or sorted(numbers) != list(  # type: ignore[type-var]
            range(1, len(members) + 1)
        ):
            raise StubError(
                "bad_parts",
                f"camera {camera!r} has parts {numbers}; its files must be parts 1 to "
                f"{len(members)}, each once",
                "number a camera's files 1, 2, ... in the order the camera wrote them",
                camera=camera,
            )
        if len(members) > 1:
            parted[camera] = sorted(members, key=lambda n: sources[n].part or 0)
    reference = cameras[0]
    others = sorted(set(parted) - {reference})
    if others:
        raise StubError(
            "parts_unsupported",
            f"camera {others[0]!r} is several files; only the reference camera (the first "
            f"source's, {reference!r}) may be several files for now",
            "give that camera as one file, or leave it out (other cameras of several files, "
            "placed by audio sync, are slice S24b)",
            camera=others[0],
        )
    return parted


def _day_frames(rate: Fraction, drop_frame: bool) -> int:
    """The frames between two midnights at ``rate``: the timecode labels of one day."""
    nominal = round(rate)
    if drop_frame and rate in DROP_FRAME_RATES:
        skip = nominal // 15
        return 86400 * nominal - skip * (1440 - 144)
    return 86400 * nominal


def place_parts(
    parts: Sequence[dict[str, Any]],
    timecodes: Mapping[str, Any],
    fps: Fraction,
    method: PartPlacement,
) -> list[dict[str, Any]]:
    """Set each part's ``recording_origin`` (frames; the first part at 0) in the stub source
    entries ``parts`` (in part order) by ``method``, and return what was placed per part: its
    origin, duration, gap to the previous part (frames, negative for a small overlap) and the
    method. ``timecodes`` maps a source id to its :class:`StartTimecode`."""
    placed: list[dict[str, Any]] = []
    first: int | None = None
    day = wraps = 0
    start = end = 0  # where the previous part starts and ends, in recording frames
    wrap_max = math.ceil(WRAP_MAX_GAP_S * fps)
    for number, entry in enumerate(parts):
        sid = entry["id"]
        prev = parts[number - 1]["id"] if number else None
        if method == "contiguous":
            origin = end
        else:
            found = timecodes[sid]
            rate = found.rate or fps
            if rate != fps:
                raise StubError(
                    "parts_rate",
                    f"part {sid!r}: its start timecode counts at {_rate_text(rate)} fps, the "
                    f"plan runs at {_rate_text(fps)}; its place cannot be read from it",
                    "give the camera's placement as contiguous",
                    source=sid,
                )
            if first is None:
                first, day = found.frames, _day_frames(rate, bool(found.drop_frame))
            origin = found.frames - first + wraps * day
            if number and origin < start:  # the timecode went back: midnight, or misordered
                if origin + day - end > wrap_max:
                    raise StubError(
                        "parts_out_of_order",
                        f"part {sid!r} starts at timecode frame {found.frames}, before part "
                        f"{prev!r} (frame {timecodes[prev].frames})",
                        "number the parts in the order the camera wrote them (their timecodes "
                        "rise), or give the placement as contiguous",
                        sources=[prev, sid],
                    )
                wraps += 1
                origin += day
            if number and origin < end - PART_OVERLAP_FRAMES:
                raise StubError(
                    "parts_overlap",
                    f"part {sid!r} starts {end - origin} frame(s) before part {prev!r} ends: "
                    "by their timecodes the two files overlap",
                    "check the files are one camera's consecutive parts; or give the "
                    "placement as contiguous",
                    sources=[prev, sid],
                    overlap_frames=end - origin,
                )
        entry["recording_origin"] = origin
        entry["part_placement"] = method
        placed.append(
            {
                "id": sid,
                "part": entry["part"],
                "recording_origin": origin,
                "duration": entry["duration"],
                "gap_frames": origin - end if number else 0,
                "placement": method,
            }
        )
        start, end = origin, origin + entry["duration"]
    return placed


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
    placement: Mapping[str, PartPlacement] | None = None,
    extra_sources: Sequence[StubSource] = (),
) -> StubResult:
    """Build a plan stub from hub media files (module docstring); a :class:`StubError` on refusal.

    Every source in ``sources`` belongs to ``recording_id`` at origin 0 (``aar analyze sync`` sets
    the others' origins in ``aar plan produce --sync``), except the parts of a reference camera
    of several files, which are placed by ``placement`` (camera label -> ``timecode`` or
    ``contiguous``; module docstring). A source without a camera label gets A, B, C... by
    position, and without an id ``cam-<label>`` (``cam-<label>-p<n>`` for a part). Optional
    ``extra_sources`` (intros, outros, alpha graphic ``.mov`` overlays) are probed at the same
    frame rate, may omit audio, have no ``recording_id`` (so audio sync skips them) and are never
    parts. The plan runs at the first source's
    rate; the resolution defaults to its display size for a long-form."""
    if not sources:
        raise StubError("invalid_stub", "give at least one media file", "pass a media path")
    total_sources = len(sources) + len(extra_sources)
    if total_sources > MAX_SOURCES:
        raise StubError(
            "invalid_stub",
            f"{total_sources} media files; a stub holds at most {MAX_SOURCES}",
            "split the show",
        )
    cameras = _cameras(sources)
    parted = _parts(sources, cameras)
    for camera in placement or {}:
        if camera not in parted:
            raise StubError(
                "bad_parts",
                f"a placement is given for camera {camera!r}, which is not several files",
                "give a placement only for a camera whose files have parts",
                camera=camera,
            )
    ref_parts = parted.get(cameras[0], [])
    order = [*ref_parts, *(n for n in range(len(sources)) if n not in ref_parts)]
    sources = [sources[n] for n in order]
    cameras = [cameras[n] for n in order]
    for n, source in enumerate(extra_sources):
        if source.part is not None:
            raise StubError(
                "bad_parts",
                f"extra source {n} gives part {source.part}; only a camera's files have parts",
                "leave part out of extra sources (graphics, intros, outros)",
            )
    cam_ids = _source_ids(sources, cameras, parted)
    extra_ids: list[str] = []
    seen_ids = set(cam_ids)
    for n, source in enumerate(extra_sources):
        sid = source.id or f"extra-{n + 1}"
        if sid in seen_ids:
            raise StubError(
                "duplicate_source_id",
                f"two sources have the id {sid!r}",
                "give each source its own id (or camera label)",
                source=sid,
            )
        seen_ids.add(sid)
        extra_ids.append(sid)
    ids = [*cam_ids, *extra_ids]
    kept, extra, ignored = _base_parts(base)
    for item in extra:
        if item.get("id") in seen_ids:
            raise StubError(
                "duplicate_source_id",
                f"the base's outro source id {item.get('id')!r} is also a media source's id",
                "rename one of them",
                source=item.get("id"),
            )
    all_sources = [*sources, *extra_sources]
    paths = [_regular_file(source.media, sid) for source, sid in zip(all_sources, ids, strict=True)]
    dup = first_duplicate(paths)
    if dup is not None:
        first, again = ids[dup[0]], ids[dup[1]]
        raise StubError(
            "duplicate_media",
            f"sources {first!r} and {again!r} are the same file ({str(paths[dup[1]])!r}, a "
            "link or a second name of it)",
            "give each camera's own file once",
            sources=[first, again],
        )
    cam_count = len(sources)
    rates: list[Fraction] = []
    for idx, (path, sid) in enumerate(zip(paths, ids, strict=True)):
        rates.append(_probe_rate(path, sid, require_audio=idx < cam_count)[0])
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
            sources, cameras, cam_ids, paths[:cam_count], measured[:cam_count], strict=True
        )
    ]
    for source, sid, path, (frames, display) in zip(
        extra_sources, extra_ids, paths[cam_count:], measured[cam_count:], strict=True
    ):
        extra_entry: dict[str, Any] = {
            "id": sid,
            "media": str(path),
            "shot_size": source.shot_size,
            "duration": frames,
            "frame_size": [display[0], display[1]],
        }
        if source.camera is not None:
            extra_entry["camera"] = source.camera
        stub_sources.append(extra_entry)
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
    for entry in stub_sources:
        found = timecodes.get(entry["id"])
        if found is not None:
            rate = found.rate or fps
            entry["start_timecode"] = found.frames
            entry["start_timecode_rate"] = {"num": rate.numerator, "den": rate.denominator}
    parts: list[dict[str, Any]] = []
    method: PartPlacement | None = None
    if ref_parts:
        entries = stub_sources[: len(ref_parts)]
        unread = [e["id"] for e in entries if e["id"] not in timecodes]
        method = (placement or {}).get(cameras[0])
        if method == "timecode" and unread:
            raise StubError(
                "timecode_unknown",
                f"camera {cameras[0]!r} is placed by timecode, but no start timecode could be "
                "read for " + ", ".join(repr(s) for s in unread),
                "give the camera's placement as contiguous",
                sources=unread,
            )
        if method is None:
            method = "contiguous" if unread else "timecode"
            if unread:
                warnings.append(
                    PlannerWarning(
                        code="parts-contiguous",
                        message=f"camera {cameras[0]!r}: no start timecode for "
                        + ", ".join(repr(s) for s in unread)
                        + ", so its parts are placed one after another, with no gaps",
                        hint="check the files with aar media timecode FILE",
                    )
                )
        for entry, source in zip(entries, sources, strict=False):
            entry["part"] = source.part
        parts = place_parts(entries, timecodes, fps, method)
    report_sources = []
    for entry in stub_sources:
        found = timecodes.get(entry["id"])
        report_sources.append(
            {
                "id": entry["id"],
                "media": entry["media"],
                "camera": entry.get("camera"),
                "duration": entry["duration"],
                "frame_size": entry["frame_size"],
                "start_timecode": None if found is None else found.to_dict(fps),
                "part": entry.get("part"),
                "recording_origin": entry.get("recording_origin", 0),
            }
        )
    plan = _validated(data)
    report = {
        "fps": _rate_text(fps),
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "recording_id": recording_id,
        "sources": report_sources,
        "cameras": list(dict.fromkeys(cameras)),
        "reference_camera": cameras[0],
        "parts": parts,
        "part_placement": method,
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


__all__ = [
    "BASE_KEYS",
    "CODES",
    "MAX_SOURCES",
    "PART_OVERLAP_FRAMES",
    "StubError",
    "StubResult",
    "StubSource",
    "build_stub",
    "first_duplicate",
    "place_parts",
]
