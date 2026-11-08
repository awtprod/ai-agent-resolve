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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.media.probe import parse_rational
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import LintReport, all_rules, lint, validate_profile
from aar.plan.model import EditPlan, Fps, Source
from aar.plan.produce.audio import JoinJudge, ScanStats, open_audio, pause_runs
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
) -> tuple[EditPlan, list[PlannerWarning]]:
    """Transform 5 and the plan: framing, the camera path, segments and transitions."""
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
        judge=JoinJudge(audio, profile.qa, fps),
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
    plan, framing_warnings = assemble(stub, done.pieces, keeps, coverage, tracks, profile, rules)
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
        plan, report, warnings, snaps, done.trims, done.nudges, done.folds, done.ending
    )
