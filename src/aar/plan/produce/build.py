"""``produce``: inputs to an EditPlan v2 that lints clean, through the five transforms.

Order: keep list (edges moved out of words) -> camera schedule -> pause trim -> sliver fold ->
end on the laugh -> sliver fold -> joins in quiet (with a fold after each pass) -> framing ->
the two invariants -> the plan -> ``aar.plan.lint`` under the profile. A plan with a lint error
is never returned (``LintGateError``); warnings are allowed and listed, next to the producer's
own warnings. Everything but the media access in ``audio`` is pure and deterministic.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.media import MediaError, probe
from aar.media.probe import (
    DROP_FRAME_RATES,
    frames_timecode,
    is_drop_frame,
    parse_rational,
    snap_rate,
    timecode_frames,
)
from aar.media.quicktime import (
    QUICKTIME_SUFFIXES,
    QuickTimeTimecode,
    TimecodeUnreadable,
    read_timecode,
)
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import LintReport, all_rules, lint, validate_profile
from aar.plan.model import EditPlan, Fps, Source
from aar.plan.produce.audio import JoinJudge, ScanStats, SourceAudio, open_audio, pause_runs
from aar.plan.produce.camera import crop_width, face_x, framing_levels, reframe_keys
from aar.plan.produce.ending import Ending, end_on_laugh
from aar.plan.produce.errors import InvariantError, LintGateError, ProduceError, ProduceInputError
from aar.plan.produce.inputs import CameraSchedule, FaceTrack, KeepList, SyncReport, same_file
from aar.plan.produce.joins import Judge, Nudge, place_joins, reach_frames
from aar.plan.produce.pauses import Trim, find_trims, kept_spans
from aar.plan.produce.report import PlannerWarning, ProduceReport
from aar.plan.produce.slivers import Fold, fold_slivers
from aar.plan.produce.timeline import (
    Coverage,
    KeepFrames,
    Piece,
    ScheduleFrames,
    Snap,
    Words,
    assert_chronological,
    assert_no_splice,
    is_cut,
    keep_frames,
    min_frames,
    schedule_frames,
    split,
)
from aar.plan.profile import ProduceSettings, Profile, ProfileError
from aar.qa.check import resolve_media

#: Rounds of (joins in quiet, sliver fold); a fold can change a cut's incoming source.
_JOIN_ROUNDS = 4


@dataclass(frozen=True)
class RuleView:
    """The lint thresholds the producer must satisfy, read from the profile's rules."""

    word_tolerance_s: float
    end_window_s: tuple[float, float]
    min_strength: float
    audio_type: str
    max_pan_speed: float | None
    handle_frames: int  # media each side of a cut that the crossfade plays


def _rule(profile: Profile, rule_id: str) -> dict[str, Any] | None:
    config = profile.rules.get(rule_id)
    if config is None or not config.enabled:
        return None
    return all_rules()[rule_id].resolve(config.params)


def check_profile(profile: Profile, kind: str) -> RuleView:
    """Check ``[produce]`` against the rules it must satisfy; a mismatch is a ``ProfileError``.

    The producer must never make a plan its own profile refuses, so: the ending window holds
    ``end_after_ms``, punch-ins and framing presets change the zoom by at least
    ``no-same-camera-jump``'s ``min_zoom_delta``, the crossfade fits ``audio-crossfade``, and a
    short's camera path has ``no-whip-pan`` limits to clamp to.
    """
    validate_profile(profile)
    s = profile.produce
    splice = _rule(profile, "no-dialogue-splice")
    tolerance = float(splice["word_tolerance_ms"]) / 1000 if splice else 0.0
    end_after = s.end_after_ms / 1000
    window = (end_after, end_after)
    min_strength = 0.0
    laugh = _rule(profile, "end-on-laugh")
    if laugh is not None:
        min_strength = float(laugh["min_strength"])
        if laugh["anchor"] == "onset":
            low, high = float(laugh["min_after_ms"]), float(laugh["max_after_ms"])
            if not low <= s.end_after_ms <= high:
                raise ProfileError(
                    f"profile {profile.name!r}: produce.end_after_ms {s.end_after_ms:g} is "
                    f"outside rules.end-on-laugh's window ({low:g}-{high:g} ms)"
                )
            window = (low / 1000, high / 1000)
    jump = _rule(profile, "no-same-camera-jump")
    if jump is not None:
        delta = float(jump["min_zoom_delta"])
        if kind == "longform" and abs(s.punch_in_zoom - 1.0) < delta:
            raise ProfileError(
                f"profile {profile.name!r}: produce.punch_in_zoom {s.punch_in_zoom:g} changes "
                f"the zoom by less than rules.no-same-camera-jump min_zoom_delta {delta:g}"
            )
        if kind == "short":
            zooms = [preset.zoom for preset in s.framing]
            pairs = list(zip(zooms, zooms[1:] + zooms[:1], strict=True))
            if len(zooms) < 2 or any(abs(a - b) < delta for a, b in pairs):
                raise ProfileError(
                    f"profile {profile.name!r}: produce.framing needs at least two presets whose "
                    f"neighbouring zooms (cyclically) differ by rules.no-same-camera-jump "
                    f"min_zoom_delta {delta:g}; got {zooms}"
                )
    audio_type = "Cross Fade 0 dB"
    handle = math.ceil(s.crossfade_frames / 2)  # a crossfade is centred on its cut
    fade = _rule(profile, "audio-crossfade")
    if fade is not None:
        audio_type = str(fade["audio_type"])
        handle = math.ceil(s.crossfade_frames * float(fade["handle_fraction"]))
        if s.crossfade_frames > int(fade["max_frames"]):
            raise ProfileError(
                f"profile {profile.name!r}: produce.crossfade_frames {s.crossfade_frames} is "
                f"more than rules.audio-crossfade max_frames {fade['max_frames']}"
            )
    pan = _rule(profile, "no-whip-pan")
    max_pan = float(pan["max_pan_speed"]) if pan is not None else None
    if kind == "short" and max_pan is None:
        raise ProfileError(
            f"profile {profile.name!r}: a short's camera path is clamped to rules.no-whip-pan; "
            "enable that rule"
        )
    return RuleView(tolerance, window, min_strength, audio_type, max_pan, handle)


def _matching(plan: EditPlan, path: str, media_root: Path | None) -> list[Source]:
    found = []
    for source in plan.sources:
        local = resolve_media(source.media, media_root)
        if local is not None and same_file(local, path):
            found.append(source)
    return found


def apply_sync(plan: EditPlan, report: SyncReport, media_root: Path | None) -> EditPlan:
    """The stub with each ``ok`` sync pair's ``suggested_recording_origin`` on its source."""
    fps = parse_rational(report.fps)
    if fps != Fraction(plan.fps.num, plan.fps.den):
        raise ProduceInputError(
            "sync-fps-mismatch",
            f"the sync report is at {report.fps} fps, the stub at {plan.fps.num}/{plan.fps.den}",
            hint="re-run aar analyze sync with the plan's --fps",
        )
    refs = _matching(plan, report.reference.file, media_root)
    if len(refs) != 1:
        raise ProduceInputError(
            "sync-reference-unknown",
            f"the sync reference {report.reference.file} is "
            f"{'not a source' if not refs else 'more than one source'} of the stub",
        )
    ref = refs[0]
    if ref.recording_origin != report.reference.recording_origin:
        raise ProduceInputError(
            "sync-origin-mismatch",
            f"the sync report measured against origin {report.reference.recording_origin} for "
            f"source {ref.id!r}, but the stub gives it {ref.recording_origin}",
            hint="re-run aar analyze sync with --ref-origin set to the stub's origin",
        )
    origins: dict[str, int] = {}
    for pair in report.pairs:
        matched = _matching(plan, pair.file, media_root)
        if not matched:
            raise ProduceInputError(
                "sync-file-unknown", f"the sync pair {pair.file} is not a source of the stub"
            )
        if pair.status != "ok" or pair.suggested_recording_origin is None:
            raise ProduceInputError(
                "sync-not-ok",
                f"the sync pair {pair.file} has status {pair.status!r}, not 'ok'",
                hint="fix the sync (see the report's hint) or set recording_origin by hand "
                "and leave out --sync",
            )
        for source in matched:
            if plan.recording_key(source) != plan.recording_key(ref):
                raise ProduceInputError(
                    "sync-recording-mismatch",
                    f"source {source.id!r} is synced to {ref.id!r} but names another recording",
                )
            origins[source.id] = pair.suggested_recording_origin
    data = plan.model_dump(mode="json")
    for item in data["sources"]:
        if item["id"] in origins:
            item["recording_origin"] = origins[item["id"]]
    return EditPlan.model_validate(data)


def bind_faces(
    plan: EditPlan, tracks: Sequence[FaceTrack], media_root: Path | None
) -> dict[str, FaceTrack]:
    """Each face track on the source whose media is the file it was measured on."""
    bound: dict[str, FaceTrack] = {}
    for track in tracks:
        matched = _matching(plan, track.header.input.path, media_root)
        if len(matched) != 1:
            raise ProduceInputError(
                "faces-unmatched",
                f"the face track of {track.header.input.path} matches "
                f"{len(matched)} sources of the stub, not one",
            )
        source = matched[0]
        if source.id in bound:
            raise ProduceInputError(
                "faces-duplicate", f"two face tracks are for source {source.id!r}"
            )
        if not _applies(track.header.recording, [source]):
            raise ProduceInputError(
                "recording-mismatch",
                f"the face track is for recording {track.header.recording!r}, not source "
                f"{source.id!r}'s",
            )
        bound[source.id] = track
    return bound


def _applies(name: str | None, sources: Sequence[Source]) -> bool:
    """An input naming ``name`` covers every source (the linter's ``applies``)."""
    return name is None or all(name in (s.recording_id, s.id) for s in sources)


Size = tuple[int, int]


def _aspect_differs(a: Size, b: Size) -> bool:
    """Two picture sizes disagree when their aspects differ by more than 1 %: a face track keeps
    the display aspect but rounds its sampled size to even pixels."""
    return abs(a[0] * b[1] - b[0] * a[1]) > 0.01 * b[0] * a[1]


def source_frame_sizes(
    stub: EditPlan, probed: Mapping[str, Size | None], faces: Mapping[str, FaceTrack]
) -> tuple[dict[str, Size], list[PlannerWarning]]:
    """Each source's ``frame_size`` (S18): the stub's own, else the media probe's display size,
    else the face track's sampled size. An audio-only source (no probed video, no face track)
    gets none. When the sources of a size disagree on the aspect, a warning names them; the
    stub's explicit size still wins, and the probe wins over the face track."""
    sizes: dict[str, Size] = {}
    warnings: list[PlannerWarning] = []
    for source in stub.sources:
        found: list[tuple[str, Size]] = []
        if source.frame_size is not None:
            found.append(("stub", (source.frame_size[0], source.frame_size[1])))
        probe_size = probed.get(source.id)
        if probe_size is not None:
            found.append(("media probe", probe_size))
        track = faces.get(source.id)
        if track is not None:
            found.append(("face track", (track.header.frame_size[0], track.header.frame_size[1])))
        if not found:
            continue
        (origin, chosen), others = found[0], found[1:]
        sizes[source.id] = chosen
        for name, size in others:
            if _aspect_differs(chosen, size):
                warnings.append(
                    PlannerWarning(
                        code="frame-size-mismatch",
                        message=(
                            f"source {source.id!r}: the {origin} gives a {chosen[0]}x{chosen[1]} "
                            f"picture, the {name} {size[0]}x{size[1]} (another aspect); the "
                            f"{origin}'s size is used"
                        ),
                        source=source.id,
                        hint="check the media, the stub's frame_size and the face track",
                    )
                )
    return sizes, warnings


@dataclass(frozen=True)
class Probed:
    """What the media says about one source's file: its display size, its start timecode tag as
    ffprobe reads it, and its video rate (the average; None without video) and ``base_fps``
    (r_frame_rate), from the probe; and its tmcd track as :mod:`aar.media.quicktime` reads it
    (S18a), for a ``.braw``, ``.mov`` or ``.mp4`` file. ``tmcd_problem`` says why that read
    failed and ``tmcd_code`` is the reader's code, unless the file simply has no tmcd track."""

    frame_size: Size | None
    timecode: str | None = None
    fps: Fraction | None = None
    tmcd: QuickTimeTimecode | None = None
    tmcd_problem: str | None = None
    base_fps: Fraction | None = None
    tmcd_code: str | None = None


def probe_frame_sizes(
    stub: EditPlan, audio: Mapping[str, SourceAudio], media_root: Path | None
) -> tuple[dict[str, Size | None], list[PlannerWarning]]:
    """The display size of every source the producer can read (see :func:`probe_sources`)."""
    probed, warnings = probe_sources(stub, audio, media_root)
    return {key: value.frame_size for key, value in probed.items()}, warnings


def read_tmcd(path: Path) -> tuple[QuickTimeTimecode | None, str | None, str | None]:
    """The tmcd track of a QuickTime-family file (``QUICKTIME_SUFFIXES``), with the reason and
    the reader's code when the read failed; ``(None, None, None)`` for another kind of file or
    one with no tmcd track."""
    if path.suffix.lower() not in QUICKTIME_SUFFIXES:
        return None, None, None
    try:
        return read_timecode(path), None, None
    except TimecodeUnreadable as exc:
        if exc.code == "no_timecode":
            return None, None, None
        return None, str(exc), exc.code
    except MediaError as exc:  # gone or replaced since it was probed
        return None, exc.summary(), "io"


def probe_sources(
    stub: EditPlan, audio: Mapping[str, SourceAudio], media_root: Path | None
) -> tuple[dict[str, Probed], list[PlannerWarning]]:
    """The probe of every source the producer can read: the cut's sources from the probe
    ``open_audio`` already made, the rest (an outro clip, say) probed when their media is a local
    file. A file that cannot be probed is a warning, not a failure: the cut does not use it.
    Every QuickTime-family file also has its tmcd track read (S18a), so BRAW's start timecode
    comes even when ffprobe cannot open the file."""
    found: dict[str, Probed] = {}
    for key, value in audio.items():
        tmcd, problem, code = read_tmcd(value.path)
        found[key] = Probed(
            value.frame_size, value.timecode, value.fps, tmcd, problem, value.base_fps, code
        )
    warnings: list[PlannerWarning] = []
    for source in stub.sources:
        if source.id in found:
            continue
        path = resolve_media(source.media, media_root)
        if path is None or not path.is_file():
            continue
        tmcd, problem, code = read_tmcd(path)
        try:
            info = probe(path)
        except MediaError as exc:
            warnings.append(
                PlannerWarning(
                    code="frame-size-unread",
                    message=f"source {source.id!r}: its picture size could not be read "
                    f"({exc.summary()})",
                    source=source.id,
                    hint="set the source's frame_size and start_timecode in the stub",
                )
            )
            if tmcd is not None:  # BRAW where ffprobe cannot open it: the timecode still counts
                found[source.id] = Probed(None, tmcd=tmcd)
            continue
        video = info.video
        found[source.id] = Probed(
            video.display_size if video is not None else None,
            info.timecode,
            video.fps if video is not None else None,
            tmcd,
            problem,
            video.base_fps if video is not None else None,
            code,
        )
    return found, warnings


@dataclass(frozen=True)
class StartTimecode:
    """A source's start timecode: ``frames`` from 00:00:00:00 counted at ``rate`` (the media's
    own frame rate; None when unknown, meaning the plan's). ``origin`` says what supplied the
    frames (``stub``, ``media probe`` or ``tmcd reader``) and ``rate_origin`` the rate when that
    differs; ``drop_frame`` is the media's flag (None when the stub gave it). Equality compares
    only the frames and the rate."""

    frames: int
    rate: Fraction | None = None
    origin: str = field(default="stub", compare=False)
    rate_origin: str | None = field(default=None, compare=False)
    drop_frame: bool | None = field(default=None, compare=False)

    def to_dict(self, plan_fps: Fraction) -> dict[str, Any]:
        rate = self.rate or plan_fps
        try:
            label: str | None = frames_timecode(self.frames, rate, drop_frame=bool(self.drop_frame))
        except ValueError:
            label = None
        return {
            "frames": self.frames,
            "rate": None if self.rate is None else f"{rate.numerator}/{rate.denominator}",
            "timecode": label,
            "drop_frame": self.drop_frame,
            "from": self.origin,
            "rate_from": self.rate_origin or (self.origin if self.rate is not None else "plan fps"),
        }


def _rates(rates: Sequence[Fraction | None]) -> str:
    return " or ".join(f"{r.numerator}/{r.denominator}" for r in rates if r is not None)


def _clean_rate(rate: Fraction | None) -> bool:
    """A rate ffprobe can mean exactly: whole or NTSC-style (n/1001), 1 to 1000 fps."""
    return rate is not None and rate.denominator in (1, 1001) and 1 <= rate <= 1000


def media_timecode_rate(seen: Probed) -> tuple[Fraction | None, str | None]:
    """The rate a source's start timecode counts at, and where it comes from (S18a): the video
    stream's average, else its r_frame_rate, snapped to a standard rate
    (``aar.media.probe.snap_rate``: BRAW's average 2953125/98536 is 30000/1001); when neither
    snaps (a 15, 120 or 240 fps clip), the stream's exact rate, its r_frame_rate when that is a
    clean rational (whole or n/1001), else its average, as dev did. With no video, the tmcd
    track's rate, snapped when it snaps. ``(None, None)`` with neither."""
    streams = [rate for rate in (seen.fps, seen.base_fps) if rate is not None]
    if streams:
        snapped = next((r for r in map(snap_rate, streams) if r is not None), None)
        if snapped is not None:
            return snapped, "media probe"
        return (seen.base_fps if _clean_rate(seen.base_fps) else streams[0]), "media probe"
    if seen.tmcd is not None:
        return snap_rate(seen.tmcd.rate) or seen.tmcd.rate, "tmcd reader"
    return None, None


def _media_timecodes(
    source_id: str, seen: Probed, stub_rate: Fraction | None, plan_fps: Fraction
) -> tuple[list[tuple[StartTimecode, str]], list[PlannerWarning], str | None]:
    """What the media says a source's start timecode is, best first, each with how it reads
    (``tag 01:00:00:00``): ffprobe's tag, then the tmcd track's sample, both counted at
    :func:`media_timecode_rate`. The third item, when set, says why neither can be used: a tmcd
    track that disagrees with that rate (its numberOfFrames is not the rate rounded, as a 25 fps
    tmcd on a 50 fps stream; drop-frame at a rate without it) or one the reader will not guess
    at. With no tmcd track there is nothing to disagree with. A tag that cannot be parsed, or
    one the tmcd track gives another frame for, is a warning; the tag still wins."""
    tmcd, tag = seen.tmcd, seen.timecode
    found: list[tuple[StartTimecode, str]] = []
    warnings: list[PlannerWarning] = []
    if seen.tmcd_code == "unsupported":  # a tmcd track whose rate the reader will not guess
        return (
            found,
            warnings,
            f"its tmcd track cannot be read without guessing ({seen.tmcd_problem})",
        )
    if tag is None and tmcd is None:
        return found, warnings, None
    rate, rate_from = media_timecode_rate(seen)
    what = "video" if rate_from == "media probe" else "tmcd track"
    if tmcd is not None and rate is not None:  # the tmcd track must count at that rate
        if tmcd.frames_per_second != round(rate):
            return (
                found,
                warnings,
                f"its tmcd track counts {tmcd.frames_per_second} frames a second (at "
                f"{_rates([tmcd.rate])}) but its {what} runs at {_rates([rate])} fps",
            )
        if tmcd.drop_frame and rate not in DROP_FRAME_RATES:
            return (
                found,
                warnings,
                f"its tmcd track is drop-frame but its {what} runs at {_rates([rate])} fps",
            )
    if tag is not None:
        try:
            read = timecode_frames(tag, rate or stub_rate or plan_fps)
        except ValueError as exc:
            warnings.append(
                PlannerWarning(
                    code="start-timecode-unread",
                    message=f"source {source_id!r}: its media's start timecode tag cannot "
                    f"be used ({exc})",
                    source=source_id,
                    hint="set the source's start_timecode in the stub",
                )
            )
        else:
            if tmcd is not None and read != tmcd.frames:
                warnings.append(
                    PlannerWarning(
                        code="start-timecode-reader-mismatch",
                        message=f"source {source_id!r}: ffprobe reads its start timecode as "
                        f"{tag} (frame {read}), its tmcd track as {tmcd.timecode} (frame "
                        f"{tmcd.frames}); ffprobe's is used",
                        source=source_id,
                        hint="check the media, or set start_timecode in the stub",
                    )
                )
            rate_origin = rate_from if rate is not None and rate_from != "media probe" else None
            found.append(
                (
                    StartTimecode(
                        read, rate or stub_rate, "media probe", rate_origin, is_drop_frame(tag)
                    ),
                    f"tag {tag}",
                )
            )
    if tmcd is not None:
        rate_origin = rate_from if rate_from != "tmcd reader" else None
        found.append(
            (
                StartTimecode(tmcd.frames, rate, "tmcd reader", rate_origin, tmcd.drop_frame),
                f"tmcd track {tmcd.timecode}",
            )
        )
    return found, warnings, None


def source_start_timecodes(
    stub: EditPlan, probed: Mapping[str, Probed], used: Collection[str] | None = None
) -> tuple[dict[str, StartTimecode], list[PlannerWarning]]:
    """Each source's ``start_timecode`` and ``start_timecode_rate`` (S18 round 4, S18a), as
    ``frame_size``: the stub's own win; else the media's timecode tag (ffprobe); else the tmcd
    track's first sample (``aar.media.quicktime``, for BRAW, which ffprobe may not open).

    The rate (S18a rounds 2-3, :func:`media_timecode_rate`) is the video stream's, snapped to a
    standard rate (BRAW's average 2953125/98536 is 30000/1001; a 29.97 stream with a tmcd track
    at 2997/100 or 30/1 is 30000/1001; a 23.976 stream with a tmcd track at 24/1 is 24000/1001),
    else its exact rate (a 120 fps clip counts at 120), and frames count at it, so a timecode
    lands exactly on a frame of a plan at that rate. With no video it is the tmcd track's, else
    the plan's. When a tmcd track disagrees with that rate (see :func:`_media_timecodes`) and
    the stub gives no start_timecode, the producer refuses (``timecode-rate-ambiguous``) rather
    than guess.

    Only the sources in ``used`` (the cut's and the outro's; None means every source) are
    worked out: a source the plan does not play keeps the stub's values and can neither warn
    nor refuse.

    A tag that cannot be read, a tmcd track that cannot be read when nothing else gives a
    timecode, a stub value the media disagrees with, and a tag the tmcd track disagrees with are
    warnings; a source with none gets none (the export then refuses or reports it). A stub that
    gives ``start_timecode`` without its rate, for media whose timecode runs at another rate
    than the plan's, is refused (``start-timecode-rate-required``): the frame count is
    ambiguous, and the export would place the asset at the wrong time."""
    plan_fps = Fraction(stub.fps.num, stub.fps.den)
    timecodes: dict[str, StartTimecode] = {}
    warnings: list[PlannerWarning] = []
    for source in stub.sources:
        if used is not None and source.id not in used:
            continue
        seen = probed.get(source.id) or Probed(None)
        given = source.start_timecode_rate
        stub_rate = Fraction(given.num, given.den) if given is not None else None
        media, media_warnings, ambiguous = _media_timecodes(source.id, seen, stub_rate, plan_fps)
        warnings += media_warnings
        snapped, _ = media_timecode_rate(seen)
        media_rate = snapped or seen.fps or seen.base_fps or (seen.tmcd.rate if seen.tmcd else None)
        if source.start_timecode is not None:
            if stub_rate is None and media_rate is not None and media_rate != plan_fps:
                raise ProduceInputError(
                    "start-timecode-rate-required",
                    f"source {source.id!r}: the stub gives start_timecode "
                    f"{source.start_timecode} without start_timecode_rate, and its media runs "
                    f"at {float(media_rate):g} fps, not the plan's {float(plan_fps):g}",
                    hint="give start_timecode_rate (the media's rate) with start_timecode, or "
                    "leave both unset to read them from the media",
                )
            chosen = StartTimecode(source.start_timecode, stub_rate)
            timecodes[source.id] = chosen
            if media:
                best, label = media[0]
                media_s = Fraction(best.frames) / (best.rate or stub_rate or plan_fps)
                if media_s != Fraction(chosen.frames) / (chosen.rate or plan_fps):
                    warnings.append(
                        PlannerWarning(
                            code="start-timecode-mismatch",
                            message=f"source {source.id!r}: the stub gives start timecode "
                            f"frame {source.start_timecode}, the media's {label} (frame "
                            f"{best.frames}); the stub's is used",
                            source=source.id,
                            hint="check the media and the stub's start_timecode",
                        )
                    )
        elif ambiguous is not None:
            raise ProduceInputError(
                "timecode-rate-ambiguous",
                f"source {source.id!r}: the rate of its start timecode is ambiguous: {ambiguous}",
                hint="set start_timecode and start_timecode_rate in the stub (aar media timecode "
                "FILE prints the tmcd track's frame and rate)",
            )
        elif media:
            timecodes[source.id] = media[0][0]
        elif seen.tmcd_problem is not None:
            warnings.append(
                PlannerWarning(
                    code="start-timecode-unread",
                    message=f"source {source.id!r}: its media's tmcd track cannot be read "
                    f"({seen.tmcd_problem})",
                    source=source.id,
                    hint="set the source's start_timecode in the stub (aar media timecode FILE "
                    "prints it for a readable file)",
                )
            )
    return timecodes, warnings


@dataclass
class ProduceResult:
    plan: EditPlan
    lint: LintReport
    warnings: list[PlannerWarning]
    snaps: list[Snap] = field(default_factory=list)
    trims: list[Trim] = field(default_factory=list)
    nudges: list[Nudge] = field(default_factory=list)
    folds: list[Fold] = field(default_factory=list)
    ending: Ending | None = None
    start_timecodes: dict[str, StartTimecode] = field(default_factory=dict)

    @property
    def cuts(self) -> int:
        plan = self.plan
        ranges = [(s.source, s.src_in, s.src_out) for s in plan.segments]
        origins = {s.id: s.recording_origin for s in plan.sources}
        count = 0
        for (a, _, a_out), (b, b_in, _) in itertools.pairwise(ranges):
            count += origins[a] + a_out != origins[b] + b_in
        return count

    def log(self) -> dict[str, Any]:
        fps = self.plan.fps

        def sec(frame: int) -> float:
            return round(fps.seconds(frame), 3)

        return {
            "snaps": [asdict(s) for s in self.snaps],
            "trims": [
                {
                    "keep": t.keep,
                    "pause_s": [round(t.pause_s[0], 3), round(t.pause_s[1], 3)],
                    "out_s": sec(t.out_frame),
                    "in_s": sec(t.in_frame),
                    "removed_s": sec(t.removed_frames),
                }
                for t in self.trims
            ],
            "nudges": [
                {**asdict(n), "delta_ms": round(fps.seconds(n.to_frame - n.from_frame) * 1000, 1)}
                for n in self.nudges
            ],
            "folds": [asdict(f) for f in self.folds],
            "ending": None
            if self.ending is None
            else {**asdict(self.ending), "out_s": sec(self.ending.to_frame)},
            "start_timecodes": {
                key: value.to_dict(Fraction(fps.num, fps.den))
                for key, value in self.start_timecodes.items()
            },
        }

    def report(self, out: str | None = None) -> ProduceReport:
        return ProduceReport(
            ok=self.lint.ok,
            out=out,
            kind=self.plan.kind,
            segments=len(self.plan.segments),
            cuts=self.cuts,
            duration_s=self.lint.duration_s,
            lint=self.lint,
            warnings=self.warnings,
            log=self.log(),
        )


def assemble(
    stub: EditPlan,
    pieces: Sequence[Piece],
    keeps: Sequence[KeepFrames],
    coverage: dict[str, Coverage],
    faces: dict[str, FaceTrack],
    profile: Profile,
    rules: RuleView,
    *,
    frame_sizes: Mapping[str, tuple[int, int]] | None = None,
    start_timecodes: Mapping[str, StartTimecode] | None = None,
) -> tuple[EditPlan, list[PlannerWarning]]:
    """Transform 5 and the plan: framing, the camera path, segments and transitions.

    ``frame_sizes`` sets each source's ``frame_size`` (from :func:`source_frame_sizes`); without
    it the stub's own sizes and the face tracks' are used. ``start_timecodes`` sets each
    source's ``start_timecode`` and ``start_timecode_rate`` (from
    :func:`source_start_timecodes`); without it the stub's own stay."""
    s = profile.produce
    fps = stub.fps
    sources = stub.source_map()
    short = stub.kind == "short"
    levels = framing_levels(pieces, sources, len(s.framing) if short else 2)
    warnings: list[PlannerWarning] = []
    segments: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        origin = coverage[piece.source].origin
        seg: dict[str, Any] = {
            "track": 1,
            "source": piece.source,
            "src_in": piece.rec_in - origin,
            "src_out": piece.rec_out - origin,
        }
        if keeps[piece.keep].label is not None:
            seg["label"] = keeps[piece.keep].label
        if not short:
            if levels[index]:
                seg["transform"] = {"zoom": s.punch_in_zoom}
        else:
            preset = s.framing[levels[index]]
            track = faces.get(piece.source)
            face = None
            width = 1.0
            if track is not None:
                times = [fps.seconds(seg["src_in"] + k) for k in range(piece.length)]
                face = face_x(track.samples(), times)
                aspect = stub.resolution.width / stub.resolution.height
                width = crop_width(aspect, track.aspect, preset.zoom)
            assert rules.max_pan_speed is not None
            keys, margin = reframe_keys(
                face,
                preset,
                width,
                fps.num / fps.den,
                s.steady,
                max_pan_speed=rules.max_pan_speed,
            )
            seg["reframe"] = [key.model_dump() for key in keys]
            if margin < s.steady.warn_margin:
                warnings.append(
                    PlannerWarning(
                        code="face-near-edge",
                        message=(
                            f"segment {index}: the face comes within {max(margin, 0):.3f} source "
                            f"widths of the crop edge (warn under {s.steady.warn_margin:g})"
                        ),
                        at_s=round(fps.seconds(piece.rec_in), 3),
                        source=piece.source,
                        hint="the subject moves faster than a steady crop can follow here",
                    )
                )
        segments.append(seg)
    transitions = [
        {
            "after": index - 1,
            "video": "cut",
            "video_frames": 0,
            "audio": {"type": rules.audio_type, "frames": s.crossfade_frames},
        }
        for index in range(1, len(pieces))
        if is_cut(pieces[index - 1], pieces[index])
    ]
    data = stub.model_dump(mode="json")
    if frame_sizes is None:
        frame_sizes, _ = source_frame_sizes(stub, {}, faces)
    for item in data["sources"]:
        if item["id"] in frame_sizes:
            item["frame_size"] = list(frame_sizes[item["id"]])
        if start_timecodes is not None and item["id"] in start_timecodes:
            timecode = start_timecodes[item["id"]]
            item["start_timecode"] = timecode.frames
            if timecode.rate is not None:
                item["start_timecode_rate"] = {
                    "num": timecode.rate.numerator,
                    "den": timecode.rate.denominator,
                }
    data.update(version=2, segments=segments, transitions=transitions)
    try:
        return EditPlan.model_validate(data), warnings
    except ValidationError as exc:  # pragma: no cover - the transforms make valid plans
        raise InvariantError("invalid-plan", f"the produced plan is not valid: {exc}") from None


#: ``runs_for(parts)``: the quiet runs (recording seconds) over one keep range's parts.
RunsFor = Callable[[Sequence[Piece]], list[tuple[float, float]]]


@dataclass
class Cut:
    """The pure core's result: the final pieces and what each transform did."""

    pieces: list[Piece]
    trims: list[Trim]
    nudges: list[Nudge]
    folds: list[Fold]
    ending: Ending | None
    warnings: list[PlannerWarning]


def cut(
    keeps: Sequence[KeepFrames],
    parts: Sequence[Sequence[Piece]],
    sched: ScheduleFrames,
    coverage: dict[str, Coverage],
    words: Words,
    fps: Fps,
    *,
    settings: ProduceSettings,
    rules: RuleView,
    runs_for: RunsFor,
    judge: Judge,
    laughs: LaughMap | None,
) -> Cut:
    """Transforms 1-4 and the invariants, with the media access injected (``runs_for``,
    ``judge``): pure and deterministic for pure arguments."""
    s = settings
    need = min_frames(s.min_shot_s, fps)
    nudge_frames = reach_frames(s.join_nudge_ms, fps)
    pieces: list[Piece] = []
    trims: list[Trim] = []
    for index, k in enumerate(keeps):
        found = find_trims(
            index,
            k.start,
            k.end,
            runs_for(parts[index]),
            words,
            fps,
            min_pause_s=s.min_pause_s,
            beat_s=s.beat_s,
            beat_lead_s=s.beat_lead_s,
        )
        trims += found
        for a, b in kept_spans(k.start, k.end, found):
            pieces += split(a, b, index, sched, coverage, fps)
    warnings: list[PlannerWarning] = []
    pieces, folds = fold_slivers(pieces, coverage, fps, need=need)
    ending = None
    if laughs is not None:
        pieces, ending, ended = end_on_laugh(
            pieces,
            laughs,
            keeps,
            words,
            coverage,
            fps,
            end_after_s=s.end_after_ms / 1000,
            window_s=rules.end_window_s,
            min_strength=rules.min_strength,
            need=need,
        )
        warnings += ended
        pieces, more = fold_slivers(pieces, coverage, fps, need=need)
        folds += more
    nudges: list[Nudge] = []
    for _ in range(_JOIN_ROUNDS):
        pieces, nudged, join_warnings = place_joins(
            pieces,
            judge,
            words,
            coverage,
            keeps,
            fps,
            nudge_frames=nudge_frames,
            need=need,
            handle=rules.handle_frames,
        )
        nudges += nudged
        pieces, more = fold_slivers(pieces, coverage, fps, need=need)
        folds += more
        if not more:
            break
    else:  # pragma: no cover - each round only merges, so this needs > 3 cascading folds
        raise ProduceError("joins-unstable", "joins and sliver folds did not settle")
    warnings += join_warnings
    assert_chronological(pieces, fps)
    assert_no_splice(pieces, words)
    return Cut(pieces, trims, nudges, folds, ending, warnings)


def produce(
    stub: EditPlan,
    *,
    transcript: Transcript,
    keep: KeepList,
    profile: Profile,
    schedule: CameraSchedule | None = None,
    sync: SyncReport | None = None,
    faces: Sequence[FaceTrack] = (),
    laughs: LaughMap | None = None,
    venue: str | None = None,
    media_root: Path | None = None,
) -> ProduceResult:
    """Make the plan. Raises ``ProduceInputError``/``ProfileError`` for bad inputs (exit 2),
    ``ProduceError``/``InvariantError``/``LintGateError`` when no clean plan results (exit 1)."""
    if stub.segments or stub.transitions:
        raise ProduceInputError("stub-has-segments", "the plan stub already has segments")
    rules = check_profile(profile, stub.kind)
    s = profile.produce
    silence_db = s.silence_for(venue)
    if sync is not None:
        stub = apply_sync(stub, sync, media_root)
    fps = stub.fps
    coverage = {src.id: Coverage(src.recording_origin, src.duration) for src in stub.sources}
    origins = {key: value.origin for key, value in coverage.items()}
    if transcript.recording is None:
        raise ProduceInputError(
            "transcript-unnamed",
            "the transcript does not name its recording, so its clock cannot be checked "
            "against the sources",
            hint='set "recording" to the sources\' recording_id (aar analyze transcript '
            "--recording)",
        )
    words = Words(transcript.words, rules.word_tolerance_s, fps, s.min_word_ms / 1000)
    keeps, snaps, keep_warnings = keep_frames(keep, words, fps, min_shot_s=s.min_shot_s)
    sched = schedule_frames(stub, schedule, s.default_camera)
    parts = [split(k.start, k.end, i, sched, coverage, fps) for i, k in enumerate(keeps)]
    used = list(dict.fromkeys(p.source for group in parts for p in group))
    sources = stub.source_map()
    content = [sources[i] for i in used]
    if len({stub.recording_key(src) for src in content}) != 1:
        raise ProduceInputError(
            "mixed-recordings",
            f"the scheduled cameras ({', '.join(used)}) are not one recording",
            hint="give every camera of the cut the same recording_id",
        )
    for what, name in (
        ("transcript", transcript.recording),
        ("keep list", keep.recording),
        ("camera schedule", schedule.recording if schedule else None),
        ("laugh map", laughs.recording if laughs else None),
    ):
        if not _applies(name, content):
            raise ProduceInputError(
                "recording-mismatch",
                f"the {what} is for recording {name!r}, not the one the cut comes from",
            )
    tracks = bind_faces(stub, faces, media_root)
    warnings: list[PlannerWarning] = list(keep_warnings)
    if tracks and stub.kind != "short":
        warnings.append(
            PlannerWarning(
                code="faces-ignored",
                message="face tracks steer only a short's camera path; a long-form ignores them",
            )
        )
    audio = open_audio(stub, used, media_root)

    def runs_for(group: Sequence[Piece]) -> list[tuple[float, float]]:
        return pause_runs(
            group,
            audio,
            origins,
            fps,
            threshold_db=silence_db,
            window_s=s.scan_window_ms / 1000,
            rate=s.scan_rate_hz,
            chunk_s=s.scan_chunk_s,
            stats=stats,
        )

    stats: dict[str, ScanStats] = {}
    done = cut(
        keeps,
        parts,
        sched,
        coverage,
        words,
        fps,
        settings=s,
        rules=rules,
        runs_for=runs_for,
        judge=JoinJudge(audio, profile.qa, fps, s.out_quiet_preroll_ms),
        laughs=laughs,
    )
    warnings += done.warnings
    for source_id, seen in stats.items():
        length = audio[source_id].duration_s or seen.scanned_s
        if length >= s.no_pause_warn_s and seen.pauses == 0:
            warnings.append(
                PlannerWarning(
                    code="no-pauses",
                    message=(
                        f"source {source_id!r} ({length:.1f} s) holds no quiet stretch below "
                        f"{silence_db:g} dB in the {seen.scanned_s:.1f} s kept: the audio may be "
                        "silent, or the venue's silence threshold wrong"
                    ),
                    source=source_id,
                    hint="check the source's audio and the --venue",
                )
            )
    probed, probe_warnings = probe_sources(stub, audio, media_root)
    sizes = {key: value.frame_size for key, value in probed.items()}
    frame_sizes, mismatches = source_frame_sizes(stub, sizes, tracks)
    played = {piece.source for piece in done.pieces} | (
        {stub.outro.source} if stub.outro else set()
    )
    start_timecodes, timecode_warnings = source_start_timecodes(stub, probed, played)
    warnings += probe_warnings + mismatches + timecode_warnings
    plan, framing_warnings = assemble(
        stub,
        done.pieces,
        keeps,
        coverage,
        tracks,
        profile,
        rules,
        frame_sizes=frame_sizes,
        start_timecodes=start_timecodes,
    )
    warnings += framing_warnings
    report = lint(plan, profile, transcript=transcript, laughs=laughs)
    if not report.ok:
        errors = [f for f in report.findings if f.severity == "error"]
        raise LintGateError(
            f"the produced plan has {len(errors)} lint error(s) under profile {profile.name!r}",
            hint="the inputs ask for something the profile forbids; see the findings",
            details=[f.model_dump() for f in errors],
            report=report,
        )
    return ProduceResult(
        plan,
        report,
        warnings,
        snaps,
        done.trims,
        done.nudges,
        done.folds,
        done.ending,
        start_timecodes,
    )
