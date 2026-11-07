"""Property tests of the producer's pure core, over seeded random shows (no media).

Each case invents a transcript, quiet runs that only roughly match its gaps (the audio and
Whisper disagree by up to 0.15 s, and some runs pass over a mumbled word), a keep list whose
edges may fall inside words, a two-camera schedule with sliver flashes, a QA-like join judge and
sometimes a laugh map. ``cut`` and ``assemble`` must then give a plan that is chronological,
splits no word, has no shot under ``min_shot_s``, starts every cut in quiet or names it, keeps
every reframe move under the no-whip-pan limits, and lints with no error under ``ccn``.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

import pytest

from aar.plan.inputs import Laugh, LaughMap, Transcript, Word
from aar.plan.lint import lint
from aar.plan.model import EditPlan, Fps
from aar.plan.produce.build import assemble, check_profile, cut
from aar.plan.produce.inputs import CameraSchedule, FaceTrack, KeepList
from aar.plan.produce.timeline import (
    Coverage,
    Piece,
    Words,
    cut_points,
    is_cut,
    keep_frames,
    min_frames,
    schedule_frames,
    split,
)
from aar.plan.profile import load_profile

PROFILE = load_profile("ccn")
CASES = 120


def _words(rng: random.Random, until: float) -> list[Word]:
    out: list[Word] = []
    t = 0.4
    while t < until:
        dur = rng.uniform(0.05, 0.6)
        gap = rng.uniform(0.02, 0.15) if rng.random() < 0.7 else rng.uniform(0.2, 2.6)
        out.append(Word(text=f"w{len(out)}", start_s=round(t, 3), end_s=round(t + dur, 3)))
        t += dur + gap
    return out


def _runs(rng: random.Random, words: list[Word]) -> list[tuple[float, float]]:
    """The quiet audio: word gaps, each edge off by up to 0.15 s, some merged over a word."""
    runs: list[tuple[float, float]] = []
    for a, b in itertools.pairwise(words):
        lo = a.end_s + rng.uniform(-0.15, 0.15)
        hi = b.start_s + rng.uniform(-0.15, 0.15)
        if hi - lo > 0.03:
            if runs and rng.random() < 0.1 and lo - runs[-1][1] < 0.7:
                runs[-1] = (runs[-1][0], hi)  # a mumble too quiet to break the run
            elif not runs or lo > runs[-1][1]:
                runs.append((lo, hi))
    return runs


def _keep(rng: random.Random, start: float, end: float) -> KeepList:
    points = sorted(rng.uniform(start, end) for _ in range(2 * rng.randint(1, 4)))
    ranges = []
    for a, b in zip(points[::2], points[1::2], strict=True):
        if b - a >= 1.2 and (not ranges or a >= ranges[-1]["end_s"]):
            ranges.append({"start_s": round(a, 3), "end_s": round(b, 3)})
    if not ranges:
        ranges = [{"start_s": start, "end_s": start + 3.0}]
    return KeepList.model_validate({"ranges": ranges})


def _schedule(rng: random.Random, end: float) -> CameraSchedule:
    spans, t, cam = [], 0.0, "A"
    while t < end + 5:
        length = rng.choice([rng.uniform(0.1, 0.9), rng.uniform(1.0, 9.0)])
        spans.append({"start_s": round(t, 3), "end_s": round(t + length, 3), "camera": cam})
        t = round(t + length, 3)
        cam = "B" if cam == "A" else "A"
    return CameraSchedule.model_validate({"spans": spans})


def _stub(kind: str, fps: Fps) -> EditPlan:
    short = kind == "short"
    return EditPlan.model_validate(
        {
            "project": "p",
            "timeline": "t",
            "kind": kind,
            "fps": {"num": fps.num, "den": fps.den},
            "resolution": {"width": 1080, "height": 1920}
            if short
            else {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {
                    "id": "a",
                    "media": "a.wav",
                    "camera": "A",
                    "shot_size": "close",
                    "recording_id": "r",
                    "recording_origin": 0,
                    "duration": 4000,
                },
                {
                    "id": "b",
                    "media": "b.wav",
                    "camera": "B",
                    "shot_size": "wide",
                    "recording_id": "r",
                    "recording_origin": -30,
                    "duration": 4000,
                },
            ],
            "segments": [],
        }
    )


def _faces(rng: random.Random, until: float) -> FaceTrack:
    frames, x = [], rng.uniform(0.3, 0.7)
    for i in range(int(until * 2) + 2):
        if rng.random() < 0.15:
            x = rng.uniform(0.05, 0.95)  # the subject lunges: faster than any allowed pan
        else:
            x = min(0.95, max(0.05, x + rng.uniform(-0.03, 0.03)))
        faces = [] if rng.random() < 0.1 else [{"box": [max(0.0, x - 0.05), 0.2, 0.1, 0.3]}]
        frames.append({"file_s": i / 2, "faces": faces})
    return FaceTrack.model_validate(
        {
            "schema": "aar.faces/1",
            "header": {"frame_size": [640, 360], "input": {"path": "a.wav"}},
            "frames": frames,
        }
    )


def _judge(runs: list[tuple[float, float]], fps: Fps, origins: dict[str, int]) -> Any:
    """QA's verdict on the synthetic audio: src_in inside a run of at least 60 ms (10 ms slack)."""

    def judge(source: str, src_in: int) -> bool:
        at = fps.seconds(src_in + origins[source])
        return any(b - a >= 0.06 and a - 0.01 <= at <= b + 0.01 for a, b in runs)

    return judge


def _case(seed: int, kind: str) -> dict[str, Any]:
    rng = random.Random(seed)
    fps = Fps(num=30000, den=1001) if seed % 3 else Fps(num=25)
    span = 30.0 if kind == "short" else 70.0
    words = _words(rng, span + 4)
    keep = _keep(rng, 0.6, span)
    laughs = None
    if rng.random() < 0.7:
        last = keep.ranges[-1]
        onset = round(rng.uniform(last.start_s, last.end_s), 3)
        laughs = LaughMap(laughs=[Laugh(start_s=onset, end_s=onset + 1.0, strength=1.0)])
    return {
        "rng": rng,
        "fps": fps,
        "words": words,
        "runs": _runs(rng, words),
        "keep": keep,
        "schedule": _schedule(rng, span + 4),
        "laughs": laughs,
        "stub": _stub(kind, fps),
        "faces": _faces(rng, span + 4) if kind == "short" else None,
    }


def _run(case: dict[str, Any], kind: str) -> tuple[list[Piece], Any, Any, EditPlan, Words]:
    fps, stub = case["fps"], case["stub"]
    rules = check_profile(PROFILE, kind)
    settings = PROFILE.produce
    coverage = {s.id: Coverage(s.recording_origin, s.duration) for s in stub.sources}
    origins = {k: v.origin for k, v in coverage.items()}
    words = Words(case["words"], rules.word_tolerance_s, fps)
    keeps, _ = keep_frames(case["keep"], words, fps, min_shot_s=settings.min_shot_s)
    sched = schedule_frames(stub, case["schedule"], settings.default_camera)
    parts = [split(k.start, k.end, i, sched, coverage, fps) for i, k in enumerate(keeps)]
    runs = case["runs"]
    judge = _judge(runs, fps, origins)
    done = cut(
        keeps,
        parts,
        sched,
        coverage,
        words,
        fps,
        settings=settings,
        rules=rules,
        runs_for=lambda group: [r for r in runs if r[1] > fps.seconds(group[0].rec_in)],
        judge=judge,
        laughs=case["laughs"],
    )
    faces = {"a": case["faces"], "b": case["faces"]} if case["faces"] else {}
    plan, _ = assemble(stub, done.pieces, keeps, coverage, faces, PROFILE, rules)
    return done.pieces, done, (keeps, judge), plan, words


@pytest.mark.parametrize("kind", ["longform", "short"])
def test_random_shows_give_plans_that_keep_every_house_rule(kind: str) -> None:
    seen = {"trims": 0, "nudges": 0, "folds": 0, "endings": 0, "warned": 0}
    for seed in range(CASES):
        case = _case(seed, kind)
        pieces, done, (keeps, judge), plan, words = _run(case, kind)
        fps = case["fps"]
        need = min_frames(PROFILE.produce.min_shot_s, fps)
        where = f"seed {seed}"
        # chronological, and every shot at least min_shot_s
        for a, b in itertools.pairwise(pieces):
            assert b.rec_in >= a.rec_out, where
        assert all(p.length >= need for p in pieces), where
        # no edit point inside a word, judged exactly as no-dialogue-splice judges it
        for _, _, frame in cut_points(pieces):
            assert words.inside(frame) is None, where
        # no pause trim swallowed a word: within a keep range, gaps hold no speech
        for a, b in itertools.pairwise(pieces):
            if a.keep == b.keep and is_cut(a, b):
                lo, hi = fps.seconds(a.rec_out), fps.seconds(b.rec_in)
                assert words.speech_between(lo, hi) is None, where
        # every cut starts in quiet, or a planner warning names it
        warned = {w.at_s for w in done.warnings if w.code == "join-not-quiet"}
        for a, b in itertools.pairwise(pieces):
            if is_cut(a, b):
                origin = plan.source_map()[b.source].recording_origin
                ok = judge(b.source, b.rec_in - origin)
                assert ok or round(fps.seconds(b.rec_in), 3) in warned, where
        # nothing outside the keep list but a join nudge (an in point moved up to
        # join_nudge_ms earlier, into word-free audio) and the ending (up to the laugh window)
        nudge = fps.frames(PROFILE.produce.join_nudge_ms / 1000)
        for piece in pieces:
            keep = keeps[piece.keep]
            assert keep.start - nudge <= piece.rec_in < piece.rec_out, where
            assert piece.rec_out <= keep.end + fps.frames(0.4), where
        # the plan lints clean under ccn, reframe speeds included
        transcript = Transcript(words=case["words"])
        report = lint(plan, PROFILE, transcript=transcript, laughs=case["laughs"])
        errors = [f.message for f in report.findings if f.severity == "error"]
        assert errors == [], f"{where}: {errors}"
        for seg in plan.segments:
            for k0, k1 in itertools.pairwise(seg.reframe):
                speed = abs(k1.x - k0.x) / fps.seconds(k1.time - k0.time)
                assert speed < 0.3 and k1.zoom == k0.zoom, where
        seen["trims"] += len(done.trims)
        seen["nudges"] += len(done.nudges)
        seen["folds"] += len(done.folds)
        seen["endings"] += done.ending is not None
        seen["warned"] += len(warned)
    # the generator really exercises every transform
    assert all(count > 0 for count in seen.values()), seen
