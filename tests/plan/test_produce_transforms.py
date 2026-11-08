"""Unit tests of the plan producers' pure transforms (no media, no ffmpeg).

Frame rate 25 fps throughout, so frame f is exactly f * 0.04 s and every expected frame below
can be checked by hand.
"""

from __future__ import annotations

import itertools
import math

import pytest

from aar.plan.inputs import Laugh, LaughMap, Word
from aar.plan.model import EditPlan, Fps
from aar.plan.produce.camera import (
    clamp_inside,
    crop_width,
    face_x,
    framing_levels,
    gaussian,
    reframe_keys,
    slew_limit,
    steady,
)
from aar.plan.produce.ending import end_on_laugh
from aar.plan.produce.errors import InvariantError, ProduceError, ProduceInputError
from aar.plan.produce.inputs import CameraSchedule, KeepList
from aar.plan.produce.joins import place_joins
from aar.plan.produce.pauses import find_trims, kept_spans
from aar.plan.produce.slivers import fold_slivers
from aar.plan.produce.timeline import (
    Coverage,
    KeepFrames,
    Piece,
    Words,
    assert_chronological,
    assert_no_splice,
    coalesce,
    cut_points,
    keep_frames,
    min_frames,
    schedule_frames,
    split,
)
from aar.plan.profile import FramingPreset, SteadySettings

FPS = Fps(num=25)
TOL = 0.01


def words(*rows: tuple[str, float, float]) -> Words:
    return Words([Word(text=t, start_s=a, end_s=b) for t, a, b in rows], TOL, FPS)


NO_WORDS = words()
WIDE = {"a": Coverage(0, None), "b": Coverage(0, None)}


def stub(*cams: tuple[str, str | None]) -> EditPlan:
    return EditPlan.model_validate(
        {
            "project": "p",
            "timeline": "t",
            "kind": "longform",
            "fps": 25,
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {"id": i, "media": f"{i}.wav", "camera": c, "recording_id": "r"} for i, c in cams
            ],
            "segments": [],
        }
    )


# ---------------------------------------------------------------- timeline


def test_min_frames_rounds_up_like_the_min_shot_rule() -> None:
    assert min_frames(1.0, FPS) == 25
    assert min_frames(1.0, Fps(num=30000, den=1001)) == 30  # 29.97 -> 30 frames to last 1 s
    assert min_frames(0.41, FPS) == 11


def test_keep_edges_become_frames() -> None:
    keeps, snaps, warnings = keep_frames(
        KeepList.model_validate({"ranges": [{"start_s": 1.0, "end_s": 3.0, "label": "x"}]}),
        NO_WORDS,
        FPS,
        min_shot_s=1.0,
    )
    assert [(k.start, k.end, k.label) for k in keeps] == [(25, 75, "x")]
    assert snaps == [] and warnings == []


def test_a_keep_edge_inside_a_word_moves_inward_and_leaves_the_partial_word_out() -> None:
    w = words(("hello", 0.9, 1.3), ("world", 2.8, 3.2))
    keeps, snaps, _ = keep_frames(
        KeepList.model_validate({"ranges": [{"start_s": 1.0, "end_s": 3.0}]}),
        w,
        FPS,
        min_shot_s=1.0,
    )
    # in 1.00 s is inside "hello" (core 0.91-1.29): forward to frame 33 (1.32 s); out 3.00 s is
    # inside "world" (2.81-3.19): back to frame 70 (2.80 s). The range only ever shrinks.
    assert [(k.start, k.end) for k in keeps] == [(33, 70)]
    assert [(s.edge, s.word, s.from_frame, s.to_frame) for s in snaps] == [
        ("in", "hello", 25, 33),
        ("out", "world", 75, 70),
    ]
    assert all(w.inside(f) is None for f in (33, 70))


def test_a_long_word_at_an_edge_never_pulls_in_excluded_material() -> None:
    w = words(("noise", 2.5, 10.5))  # an 8 s "word" across the out edge
    keeps, _, _ = keep_frames(
        KeepList.model_validate({"ranges": [{"start_s": 0.4, "end_s": 3.0}]}),
        w,
        FPS,
        min_shot_s=1.0,
    )
    assert [(k.start, k.end) for k in keeps] == [(10, 62)]  # 2.48 s, outside the 2.51 core


def test_touching_keep_ranges_merge_and_keep_a_word_across_the_seam() -> None:
    w = words(("seam", 2.9, 3.1))
    keeps, snaps, _ = keep_frames(
        KeepList.model_validate(
            {"ranges": [{"start_s": 1.0, "end_s": 3.0}, {"start_s": 3.0, "end_s": 5.0}]}
        ),
        w,
        FPS,
        min_shot_s=1.0,
    )
    assert [(k.start, k.end) for k in keeps] == [(25, 125)] and snaps == []


def test_a_range_inside_one_word_is_dropped_with_a_warning() -> None:
    w = words(("long", 0.9, 1.6))
    keeps, _, warnings = keep_frames(
        KeepList.model_validate(
            {"ranges": [{"start_s": 1.0, "end_s": 1.5}, {"start_s": 2.0, "end_s": 4.0}]}
        ),
        w,
        FPS,
        min_shot_s=0.4,
    )
    assert [(k.start, k.end) for k in keeps] == [(50, 100)]
    assert [x.code for x in warnings] == ["keep-dropped"] and warnings[0].at_s == 1.0
    with pytest.raises(ProduceInputError) as exc:
        keep_frames(
            KeepList.model_validate({"ranges": [{"start_s": 1.0, "end_s": 1.5}]}),
            w,
            FPS,
            min_shot_s=0.4,
        )
    assert exc.value.code == "keep-empty"


def test_a_range_that_shrinks_under_min_shot_is_refused() -> None:
    w = words(("word", 0.9, 1.5))
    with pytest.raises(ProduceInputError) as exc:
        keep_frames(
            KeepList.model_validate({"ranges": [{"start_s": 1.0, "end_s": 2.2}]}),
            w,
            FPS,
            min_shot_s=1.0,
        )
    assert exc.value.code == "keep-too-short" and "partial word" in exc.value.message


@pytest.mark.parametrize(("end_s", "ok"), [(1.96, False), (2.0, True)])
def test_a_keep_range_must_last_min_shot(end_s: float, ok: bool) -> None:
    keep = KeepList.model_validate({"ranges": [{"start_s": 1.0, "end_s": end_s}]})
    if ok:
        keep_frames(keep, NO_WORDS, FPS, min_shot_s=1.0)
        return
    with pytest.raises(ProduceInputError) as exc:
        keep_frames(keep, NO_WORDS, FPS, min_shot_s=1.0)
    assert exc.value.code == "keep-too-short"


def test_the_schedule_splits_a_range_at_camera_changes() -> None:
    plan = stub(("a", "A"), ("b", "B"))
    sched = schedule_frames(
        plan,
        CameraSchedule.model_validate(
            {
                "spans": [
                    {"start_s": 0, "end_s": 2, "camera": "A"},
                    {"start_s": 2, "end_s": 9, "camera": "b"},
                ]
            }
        ),
        "A",
    )
    assert split(25, 100, 3, sched, WIDE, FPS) == [Piece("a", 25, 50, 3), Piece("b", 50, 100, 3)]


def test_no_schedule_means_the_default_camera_everywhere() -> None:
    plan = stub(("a", "A"), ("b", "B"))
    sched = schedule_frames(plan, None, "B")
    assert split(0, 10, 0, sched, WIDE, FPS) == [Piece("b", 0, 10, 0)]


def test_a_schedule_gap_inside_a_keep_range_is_refused() -> None:
    plan = stub(("a", "A"), ("b", "B"))
    sched = schedule_frames(
        plan,
        CameraSchedule.model_validate(
            {
                "spans": [
                    {"start_s": 0, "end_s": 2, "camera": "A"},
                    {"start_s": 2.4, "end_s": 9, "camera": "B"},
                ]
            }
        ),
        "A",
    )
    with pytest.raises(ProduceInputError) as exc:
        split(25, 100, 0, sched, WIDE, FPS)
    assert exc.value.code == "schedule-gap"
    with pytest.raises(ProduceInputError):  # a range that starts before the schedule does
        split(0, 40, 0, schedule_frames(plan, CameraSchedule.model_validate(
            {"spans": [{"start_s": 0.5, "end_s": 9, "camera": "A"}]}), "A"), WIDE, FPS)  # fmt: skip


def test_a_camera_without_media_there_is_refused() -> None:
    plan = stub(("a", "A"))
    sched = schedule_frames(plan, None, "A")
    with pytest.raises(ProduceInputError) as exc:
        split(25, 100, 0, sched, {"a": Coverage(30, None)}, FPS)
    assert exc.value.code == "camera-does-not-cover"
    with pytest.raises(ProduceInputError):
        split(25, 100, 0, sched, {"a": Coverage(0, 99)}, FPS)
    assert split(25, 100, 0, sched, {"a": Coverage(25, 75)}, FPS) == [Piece("a", 25, 100, 0)]


def test_coalesce_merges_unbroken_playback_of_one_source_in_one_keep_range() -> None:
    assert coalesce([Piece("a", 0, 10, 0), Piece("a", 10, 20, 0), Piece("b", 20, 30, 0)]) == [
        Piece("a", 0, 20, 0),
        Piece("b", 20, 30, 0),
    ]
    two_keeps = [Piece("a", 0, 10, 0), Piece("a", 10, 20, 1)]
    assert coalesce(two_keeps) == two_keeps
    gap = [Piece("a", 0, 10, 0), Piece("a", 11, 20, 0)]
    assert coalesce(gap) == gap


def test_chronological_invariant() -> None:
    assert_chronological([Piece("a", 0, 10, 0), Piece("b", 10, 20, 0)], FPS)
    with pytest.raises(InvariantError) as exc:
        assert_chronological([Piece("a", 0, 10, 0), Piece("b", 9, 20, 0)], FPS)
    assert exc.value.code == "not-chronological"
    with pytest.raises(InvariantError) as exc:
        assert_chronological([Piece("a", 5, 5, 0)], FPS)
    assert exc.value.code == "empty-piece"


def test_no_splice_invariant_checks_every_edit_point_but_not_a_camera_switch() -> None:
    w = words(("word", 0.9, 1.3))
    switch = [Piece("a", 0, 27, 0), Piece("b", 27, 50, 0)]  # 1.08 s: inside, but no edit
    assert cut_points(switch) == [(0, "in", 0), (1, "out", 50)]
    assert_no_splice(switch, w)
    with pytest.raises(InvariantError) as exc:
        assert_no_splice([Piece("a", 0, 27, 0), Piece("b", 40, 50, 0)], w)
    assert exc.value.code == "dialogue-splice"
    with pytest.raises(InvariantError):
        assert_no_splice([Piece("a", 27, 50, 0)], w)  # the first in point


# ---------------------------------------------------------------- pause trim

TRIM = {"min_pause_s": 0.9, "beat_s": 0.45, "beat_lead_s": 0.25}


def test_a_long_pause_inside_the_range_shrinks_to_the_beat() -> None:
    trims = find_trims(0, 0, 250, [(2.0, 3.5)], NO_WORDS, FPS, **TRIM)
    # 2.25 s -> frame 56.25 -> 56; 3.30 s -> 82.5, which Fps.frames rounds half to even: 82
    assert [(t.out_frame, t.in_frame) for t in trims] == [(56, 82)]
    assert trims[0].removed_frames == 26
    assert kept_spans(0, 250, trims) == [(0, 56), (82, 250)]


@pytest.mark.parametrize(("run", "count"), [((2.0, 2.9), 1), ((2.0, 2.88), 0)])
def test_min_pause_is_inclusive(run: tuple[float, float], count: int) -> None:
    assert len(find_trims(0, 0, 250, [run], NO_WORDS, FPS, **TRIM)) == count


def test_silence_at_a_keep_range_edge_is_the_humans_choice() -> None:
    assert find_trims(0, 50, 250, [(2.0, 3.5)], NO_WORDS, FPS, **TRIM) == []  # starts at the in
    assert find_trims(0, 0, 100, [(2.5, 4.0)], NO_WORDS, FPS, **TRIM) == []  # ends at the out
    assert len(find_trims(0, 49, 101, [(2.0, 3.5)], NO_WORDS, FPS, **TRIM)) == 1


def test_a_transcribed_word_inside_a_quiet_run_is_never_swallowed() -> None:
    mumble = words(("um", 2.6, 2.8))
    # 2.0-3.5 less "um" leaves 0.6 + 0.69 s: neither half is a pause.
    assert find_trims(0, 0, 250, [(2.0, 3.5)], mumble, FPS, **TRIM) == []
    # In a longer run the half after the word still qualifies, and keeps the word.
    trims = find_trims(0, 0, 250, [(2.0, 4.5)], mumble, FPS, **TRIM)
    assert len(trims) == 1
    t = trims[0]
    assert (
        FPS.seconds(t.out_frame) > 2.79
        and mumble.speech_between(FPS.seconds(t.out_frame), FPS.seconds(t.in_frame)) is None
    )


def test_a_word_whisper_stretched_into_the_pause_shortens_it() -> None:
    stretched = words(("on", 1.5, 2.6))
    trims = find_trims(0, 0, 250, [(2.0, 4.0)], stretched, FPS, **TRIM)
    assert [(t.out_frame, t.in_frame) for t in trims] == [(71, 95)]  # from 2.59 s + 0.25


def test_trims_never_cut_into_any_word() -> None:
    w = words(("a", 1.0, 1.9), ("b", 2.1, 2.3), ("c", 3.6, 4.0))
    for run in [(1.9, 3.6), (1.95, 3.55), (0.5, 4.5), (2.3, 3.6)]:
        for trim in find_trims(0, 0, 250, [run], w, FPS, **TRIM):
            lo, hi = FPS.seconds(trim.out_frame), FPS.seconds(trim.in_frame)
            assert w.speech_between(lo, hi) is None
            assert w.inside(trim.out_frame) is None and w.inside(trim.in_frame) is None


# ---------------------------------------------------------------- joins in quiet


class FakeJudge:
    """Quiet exactly at the listed source frames; records every question."""

    def __init__(self, ins: tuple[int, ...] = (), outs: tuple[int, ...] = ()) -> None:
        self.ins, self.outs = set(ins), set(outs)
        self.calls: list[tuple[str, str, int, int]] = []

    def in_quiet(self, source: str, src_in: int, *, handle: int) -> bool:
        self.calls.append(("in", source, src_in, handle))
        return src_in in self.ins

    def out_quiet(self, source: str, src_out: int, *, handle: int) -> bool:
        self.calls.append(("out", source, src_out, handle))
        return src_out in self.outs


ONE_KEEP = [KeepFrames(0, 1000, None)]
TWO_KEEPS = [KeepFrames(0, 50, None), KeepFrames(100, 1000, None)]


def joins(pieces, judge, w=NO_WORDS, cover=WIDE, keeps=ONE_KEEP):
    return place_joins(pieces, judge, w, cover, keeps, FPS, nudge_frames=3, need=25, handle=1)


def test_a_cut_already_in_quiet_stays() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 0)]
    out, nudges, warnings = joins(pieces, FakeJudge(ins=(100,)))
    assert out == pieces and nudges == [] and warnings == []


def test_inside_a_keep_range_the_in_point_moves_either_way_later_first() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 0)]
    out, nudges, _ = joins(pieces, FakeJudge(ins=(98, 102)))
    assert out[1].rec_in == 102
    assert [(n.join, n.side, n.from_frame, n.to_frame) for n in nudges] == [(1, "in", 100, 102)]
    out, _, _ = joins(pieces, FakeJudge(ins=(98, 103)))
    assert out[1].rec_in == 98  # a trimmed pause may come back a little


def test_at_a_human_cut_the_in_point_only_moves_inward_with_its_handle() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 1)]
    judge = FakeJudge(ins=(98, 102), outs=(50,))
    out, _, warnings = joins(pieces, judge, keeps=TWO_KEEPS)
    assert out[1].rec_in == 102 and warnings == []
    assert ("in", "a", 100, 1) in judge.calls  # the incoming handle must be quiet too
    out, _, warnings = joins(pieces, FakeJudge(ins=(98,), outs=(50,)), keeps=TWO_KEEPS)
    assert out[1].rec_in == 100  # 98 would play material the human excluded
    assert [x.code for x in warnings] == ["join-not-quiet"] and "human cut" in warnings[0].message


def test_a_trim_cut_judges_no_handle_and_no_out_side() -> None:
    judge = FakeJudge(ins=(100,))
    joins([Piece("a", 0, 50, 0), Piece("a", 100, 200, 0)], judge)
    assert judge.calls == [("in", "a", 100, 0)]


def test_the_out_side_of_a_human_cut_moves_earlier_into_quiet() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 1)]
    out, nudges, warnings = joins(pieces, FakeJudge(ins=(100,), outs=(48, 52)), keeps=TWO_KEEPS)
    assert out[0].rec_out == 48 and warnings == []
    assert [(n.side, n.from_frame, n.to_frame) for n in nudges] == [("out", 50, 48)]
    out, _, warnings = joins(pieces, FakeJudge(ins=(100,), outs=(52,)), keeps=TWO_KEEPS)
    assert out[0].rec_out == 50  # never later, into the human's cut
    assert [x.code for x in warnings] == ["join-out-not-quiet"] and "join 1" in warnings[0].message


def test_an_out_side_nudge_never_drops_a_word() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 1)]
    w = words(("tail", 1.9, 1.95))  # frames 47.5-48.75 occupy 1.905-1.945 s (40 ms at least)
    out, _, warnings = joins(pieces, FakeJudge(ins=(100,), outs=(47,)), w, keeps=TWO_KEEPS)
    assert out[0].rec_out == 50 and [x.code for x in warnings] == ["join-out-not-quiet"]


def test_the_judge_gets_source_frames_not_recording_frames() -> None:
    judge = FakeJudge()
    joins(
        [Piece("a", 0, 50, 0), Piece("b", 100, 200, 0)],
        judge,
        cover={**WIDE, "b": Coverage(30, None)},
    )
    assert judge.calls[0] == ("in", "b", 70, 0)


def test_a_nudge_never_adds_or_drops_a_word() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 0)]
    # "x" sits in 3.93-3.99 s (frames 98.25-99.75): moving the in point to 98 would add it.
    w = words(("x", 3.92, 4.0), ("y", 4.04, 4.2))
    out, nudges, warnings = joins(pieces, FakeJudge(ins=(98, 102)), w)
    assert out[1].rec_in == 100 and nudges == []
    assert [x.code for x in warnings] == ["join-not-quiet"]


def test_a_nudge_keeps_the_cut_a_cut() -> None:
    pieces = [Piece("a", 0, 99, 0), Piece("a", 100, 200, 0)]
    out, _, warnings = joins(pieces, FakeJudge(ins=(99, 98)))
    assert out[1].rec_in == 100 and warnings  # 99 would be unbroken playback, 98 overlaps


def test_a_nudge_may_not_strand_an_unfoldable_sliver() -> None:
    keeps = [KeepFrames(0, 50, None), KeepFrames(100, 126, None)]
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 126, 1)]
    out, _, warnings = joins(pieces, FakeJudge(ins=(102,), outs=(50,)), keeps=keeps)
    assert out[1].rec_in == 100 and warnings  # 102-126 is 24 frames, keep range 1 has no other
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 126, 0)]
    out, _, _ = joins(pieces, FakeJudge(ins=(102,)))
    assert out[1].rec_in == 102  # same keep range: a later fold can absorb it


def test_a_camera_switch_in_unbroken_playback_is_not_judged() -> None:
    judge = FakeJudge()
    _, _, warnings = joins([Piece("a", 0, 50, 0), Piece("b", 50, 100, 0)], judge)
    assert judge.calls == [] and warnings == []


def test_no_quiet_frame_leaves_the_cut_and_names_the_join() -> None:
    keeps = [KeepFrames(0, 120, None), KeepFrames(200, 300, None)]
    pieces = [Piece("a", 0, 50, 0), Piece("a", 60, 120, 0), Piece("b", 200, 300, 1)]
    out, _, warnings = joins(pieces, FakeJudge(ins=(60,), outs=(120,)), keeps=keeps)
    assert out == pieces
    assert len(warnings) == 1 and "join 2" in warnings[0].message
    assert warnings[0].source == "b" and warnings[0].at_s == 8.0


def test_no_incoming_handle_shifts_the_in_point_one_frame_or_fails() -> None:
    cover = {"a": Coverage(0, None), "b": Coverage(100, None)}  # b's media starts at the cut
    pieces = [Piece("a", 0, 50, 0), Piece("b", 100, 200, 1)]
    out, nudges, _ = joins(pieces, FakeJudge(ins=(1,), outs=(50,)), cover=cover, keeps=TWO_KEEPS)
    assert out[1].rec_in == 101 and [(n.side, n.to_frame) for n in nudges] == [("in", 101)]
    with pytest.raises(ProduceError) as exc:
        joins(pieces, FakeJudge(ins=(0, 2), outs=(50,)), cover=cover, keeps=TWO_KEEPS)
    assert exc.value.code == "no-handle" and "cut 1" in exc.value.message


def test_no_outgoing_handle_shifts_the_out_point_one_frame_or_fails() -> None:
    cover = {"a": Coverage(0, 50), "b": Coverage(0, None)}  # a's media ends at the cut
    pieces = [Piece("a", 0, 50, 0), Piece("b", 100, 200, 1)]
    out, _, _ = joins(pieces, FakeJudge(ins=(100,), outs=(49,)), cover=cover, keeps=TWO_KEEPS)
    assert out[0].rec_out == 49
    with pytest.raises(ProduceError) as exc:
        joins(pieces, FakeJudge(ins=(100,), outs=(50,)), cover=cover, keeps=TWO_KEEPS)
    assert exc.value.code == "no-handle" and "after its out" in exc.value.message


# ---------------------------------------------------------------- sliver fold


def test_a_sliver_folds_into_the_previous_neighbour_on_a_tie() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("b", 50, 60, 0), Piece("a", 60, 120, 0)]
    out, folds = fold_slivers(pieces, WIDE, FPS, need=25)
    assert out == [Piece("a", 0, 120, 0)]  # folded into the previous, then coalesced
    assert [(f.into, f.readded_frames) for f in folds] == [("previous", 0)]


def test_a_sliver_folds_toward_the_smaller_gap() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 70, 80, 0), Piece("a", 85, 150, 0)]
    out, folds = fold_slivers(pieces, WIDE, FPS, need=25)
    assert out == [Piece("a", 0, 50, 0), Piece("a", 70, 150, 0)]
    assert [(f.into, f.readded_frames) for f in folds] == [("next", 5)]


def test_a_sliver_never_folds_across_a_human_cut() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 60, 70, 1), Piece("a", 80, 150, 1)]
    out, folds = fold_slivers(pieces, WIDE, FPS, need=25)
    assert out == [Piece("a", 0, 50, 0), Piece("a", 60, 150, 1)]
    assert folds[0].into == "next"


def test_a_fold_needs_the_neighbours_media() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("b", 50, 60, 0), Piece("a", 60, 120, 0)]
    cover = {"a": Coverage(0, 55), "b": Coverage(0, None)}
    with pytest.raises(ProduceError):  # neither a's media covers 0-60 nor 50-120
        fold_slivers(pieces, cover, FPS, need=25)
    cover = {"a": Coverage(55, None), "b": Coverage(0, None)}
    pieces = [Piece("b", 0, 50, 0), Piece("a", 50, 60, 0), Piece("a", 70, 120, 0)]
    out, _folds = fold_slivers(pieces, cover, FPS, need=25)
    assert out == [Piece("b", 0, 60, 0), Piece("a", 70, 120, 0)]


def test_an_unfoldable_sliver_is_an_error() -> None:
    with pytest.raises(ProduceError) as exc:
        fold_slivers([Piece("a", 0, 50, 0), Piece("a", 60, 70, 1)], WIDE, FPS, need=25)
    assert exc.value.code == "unfoldable-sliver"


def test_after_folding_every_piece_lasts_min_shot() -> None:
    pieces = [Piece("a", i * 10, i * 10 + 9, 0) for i in range(12)]
    out, _ = fold_slivers(pieces, WIDE, FPS, need=25)
    assert all(p.length >= 25 for p in out)
    assert out[0].rec_in == 0 and out[-1].rec_out == 119


# ---------------------------------------------------------------- end on the laugh

KEEPS = [KeepFrames(0, 100, None), KeepFrames(150, 400, None)]


def ending(pieces, laughs, w=NO_WORDS, cover=WIDE, window=(0.0, 0.4), strength=0.0):
    return end_on_laugh(
        pieces,
        LaughMap(laughs=[Laugh(start_s=a, end_s=b, strength=s) for a, b, s in laughs]),
        KEEPS,
        w,
        cover,
        FPS,
        end_after_s=0.2,
        window_s=window,
        min_strength=strength,
        need=25,
    )


def test_the_content_ends_end_after_past_the_final_laughs_onset() -> None:
    pieces = [Piece("a", 0, 100, 0), Piece("a", 150, 300, 1), Piece("a", 320, 400, 1)]
    out, end, warnings = ending(pieces, [(4.0, 5.0, 1.0), (10.0, 11.0, 1.0)])
    assert out == [Piece("a", 0, 100, 0), Piece("a", 150, 255, 1)]  # 10.2 s; 320-400 dropped
    assert end is not None and (end.from_frame, end.to_frame, end.dropped_frames) == (400, 255, 125)
    assert warnings == []


def test_the_ending_may_grow_into_a_trimmed_pause_but_never_past_the_keep_out() -> None:
    pieces = [Piece("a", 150, 300, 1), Piece("a", 320, 400, 1)]
    out, _, _ = ending(pieces, [(12.3, 13.0, 1.0)])
    assert out[-1] == Piece("a", 150, 312, 1)  # 12.5 s = frame 312.5 -> 312, in the trimmed 300-320
    out, end, _ = ending(pieces, [(15.95, 17.0, 1.0)])
    assert out[-1] == Piece("a", 320, 400, 1)  # 16.15 s is past the keep out: 16.0 s is nearest
    assert end is not None and end.to_frame == 400


def test_a_laugh_at_the_final_ranges_start_never_replays_the_human_cut() -> None:
    """The validator's repro (finding 1): keeps [0, 300) and [900, 1200) with a laugh 1 ms
    after frame 900. The only frame that fits stretched [0, 300) across the cut to 900."""
    fps = Fps(num=30000, den=1001)
    keeps = [KeepFrames(0, 300, None), KeepFrames(900, 1200, None)]
    pieces = [Piece("a", 0, 300, 0), Piece("a", 900, 1200, 1)]
    laughs = LaughMap.model_validate(
        {"laughs": [{"start_s": fps.seconds(900) + 0.001, "end_s": fps.seconds(940)}]}
    )
    out, end, warnings = end_on_laugh(
        pieces,
        laughs,
        keeps,
        Words([], 0.01, fps),
        {"a": Coverage(0, None)},
        fps,
        end_after_s=0.2,
        window_s=(0.0, 0.4),
        min_strength=0.0,
        need=30,
    )
    assert out == pieces and end is None
    assert [w.code for w in warnings] == ["end-not-on-laugh"]


def test_dropping_kept_speech_after_the_laugh_is_a_warning() -> None:
    pieces = [Piece("a", 150, 400, 1)]
    talk = words(("and", 11.0, 11.3), ("more", 12.0, 12.4))
    out, end, warnings = ending(pieces, [(10.0, 10.6, 1.0)], talk)
    assert out[-1].rec_out == 255 and end is not None
    assert [w.code for w in warnings] == ["end-dropped-speech"]
    assert "5.800 s" in warnings[0].message and "2 transcribed word(s)" in warnings[0].message
    _, _, quiet = ending(pieces, [(15.0, 15.5, 1.0)], talk)
    assert quiet == []  # nothing spoken after 15.2 s


def test_only_a_laugh_in_the_final_keep_range_counts() -> None:
    pieces = [Piece("a", 0, 100, 0), Piece("a", 150, 400, 1)]
    out, end, warnings = ending(pieces, [(2.0, 3.0, 1.0), (17.0, 18.0, 1.0)])
    assert out == pieces and end is None
    assert [w.code for w in warnings] == ["no-final-laugh"]


def test_weak_laughs_do_not_count() -> None:
    pieces = [Piece("a", 150, 400, 1)]
    out, _, _ = ending(pieces, [(8.0, 9.0, 1.0), (10.0, 11.0, 0.2)], strength=0.5)
    assert out[-1].rec_out == 205


def test_the_ending_steps_around_a_word_inside_the_window() -> None:
    pieces = [Piece("a", 150, 400, 1)]
    talk = words(("over", 10.1, 10.3))  # covers the 10.2 s target
    out, _, _ = ending(pieces, [(10.0, 11.0, 1.0)], talk)
    # 253-257 (10.12-10.28 s) are inside "over" (core 10.11-10.29); 252 and 258 tie, earlier wins
    assert out[-1].rec_out == 252
    assert talk.inside(252) is None and talk.inside(253) is not None


def test_no_clean_frame_in_the_window_keeps_the_ending() -> None:
    pieces = [Piece("a", 150, 400, 1)]
    talk = words(("over", 9.9, 10.6))
    out, end, warnings = ending(pieces, [(10.0, 11.0, 1.0)], talk)
    assert out == pieces and end is None and [w.code for w in warnings] == ["end-not-on-laugh"]


def test_the_ending_never_leaves_an_unfoldable_last_shot() -> None:
    pieces = [Piece("a", 0, 100, 0), Piece("a", 150, 400, 1)]
    out, end, warnings = ending(pieces, [(6.1, 7.0, 1.0)])  # 6.3 s: only 8 frames of keep 1
    assert out == pieces and end is None and warnings


# ---------------------------------------------------------------- framing and camera path


def test_framing_steps_on_same_camera_cuts_and_resets_elsewhere() -> None:
    sources = stub(("a", "A"), ("a2", "A"), ("b", "B")).source_map()
    pieces = [
        Piece("a", 0, 50, 0),
        Piece("a", 60, 100, 0),  # same source, cut -> 1
        Piece("a2", 110, 150, 0),  # same camera label, cut -> 2 % 3
        Piece("b", 150, 200, 0),  # another camera -> 0
        Piece("b", 250, 300, 1),  # cut -> 1
        Piece("a", 300, 350, 1),  # unbroken switch -> 0
        Piece("a", 360, 400, 1),  # cut -> 1
        Piece("b", 420, 460, 1),  # cut to another camera -> 0
    ]
    assert framing_levels(pieces, sources, 3) == [0, 1, 2, 0, 1, 0, 1, 0]
    assert framing_levels(pieces, sources, 2) == [0, 1, 0, 0, 1, 0, 1, 0]


def test_gaussian_and_box_keep_a_constant() -> None:
    assert gaussian([0.3] * 40, 5.0) == pytest.approx([0.3] * 40)
    assert steady([0.4] * 30, 0.3, 25.0, SteadySettings()) == pytest.approx([0.4] * 30)


def test_the_camera_holds_while_the_face_sways_inside_the_dead_zone() -> None:
    sway = [0.5 + (0.03 if i % 10 < 5 else -0.03) for i in range(200)]  # dead zone 0.25*0.32=0.08
    path = steady(sway, 0.32, 25.0, SteadySettings())
    assert max(path) - min(path) < 1e-9


def test_the_camera_follows_a_walk_and_keeps_the_face_inside() -> None:
    walk = [0.3 + 0.4 * min(1.0, max(0.0, (i - 50) / 50)) for i in range(250)]
    path = steady(walk, 0.32, 25.0, SteadySettings())
    assert path[0] == pytest.approx(0.3, abs=1e-9) and path[-1] == pytest.approx(0.62, abs=0.01)
    assert all(abs(f - c) <= 0.16 for f, c in zip(walk, path, strict=True))


def test_the_safety_pass_eases_the_path_toward_a_face_near_the_edge() -> None:
    s = SteadySettings()
    off = SteadySettings(safety_passes=0)
    jump = [0.3] * 60 + [0.5] * 140  # the glide lags a 0.2 jump past the 0.16 - 0.057 limit
    with_safety, without = steady(jump, 0.32, 25.0, s), steady(jump, 0.32, 25.0, off)
    assert min(0.16 - abs(f - c) for f, c in zip(jump, with_safety, strict=True)) > min(
        0.16 - abs(f - c) for f, c in zip(jump, without, strict=True)
    )


def test_clamp_and_slew_limit() -> None:
    assert clamp_inside([0.0, 0.5, 1.0], 0.4) == [0.2, 0.5, 0.8]
    assert clamp_inside([0.1, 0.9], 1.2) == [0.5, 0.5]
    out = slew_limit([0.0, 1.0, 1.0, 0.0, 0.0], 0.1)
    # forward first, so the path starts where the camera is and never moves before the face
    assert out == pytest.approx([0.0, 0.1, 0.2, 0.1, 0.0])
    assert all(abs(b - a) <= 0.1 + 1e-12 for a, b in itertools.pairwise(out))


def test_face_x_interpolates_and_holds_the_ends() -> None:
    samples = [(1.0, 0.2), (2.0, 0.4)]
    assert face_x(samples, [0.0, 1.0, 1.5, 2.0, 9.0]) == pytest.approx([0.2, 0.2, 0.3, 0.4, 0.4])
    assert face_x([], [0.0]) is None


def test_crop_width_for_a_vertical_crop_of_a_wide_source() -> None:
    assert crop_width(9 / 16, 16 / 9, 1.0) == pytest.approx(0.31640625)
    assert crop_width(9 / 16, 16 / 9, 1.25) == pytest.approx(0.253125)


def test_without_a_face_the_short_holds_the_centre() -> None:
    preset = FramingPreset(name="B", zoom=1.25, y=0.45)
    keys, margin = reframe_keys(None, preset, 0.3, 25.0, SteadySettings(), max_pan_speed=0.3)
    assert [(k.time, k.x, k.y, k.zoom) for k in keys] == [(0, 0.5, 0.45, 1.25)]
    assert margin == math.inf


def test_keys_cover_the_segment_and_never_whip() -> None:
    face = [0.2 if (i // 30) % 2 else 0.8 for i in range(301)]  # teleports every 1.2 s
    preset = FramingPreset(name="A", zoom=1.0)
    keys, _ = reframe_keys(face, preset, 0.32, 25.0, SteadySettings(), max_pan_speed=0.3)
    assert keys[0].time == 0 and keys[-1].time == 300
    assert {k.zoom for k in keys} == {1.0} and {k.y for k in keys} == {0.5}
    for a, b in itertools.pairwise(keys):
        assert abs(b.x - a.x) / ((b.time - a.time) / 25) <= 0.3 * 0.9 + 1e-3
        assert 0.16 - 1e-4 <= a.x <= 0.84 + 1e-4


def test_the_latest_laugh_in_the_final_keep_range_wins() -> None:
    out, end, _ = ending([Piece("a", 150, 400, 1)], [(8.0, 9.0, 1.0), (10.0, 11.0, 1.0)])
    assert out[-1].rec_out == 255 and end is not None and end.onset_s == 10.0


@pytest.mark.parametrize(
    ("at", "inside"),
    [(1.005, False), (1.009, False), (1.011, True), (1.989, True), (1.991, False)],
)
def test_the_splice_test_allows_the_word_tolerance_at_each_edge(at: float, inside: bool) -> None:
    from aar.plan.rules.speech import word_at

    word = Word(text="w", start_s=1.0, end_s=2.0)
    assert (word_at([word], at, 0.01) is not None) is inside


# ---------------------------------------------------------------- round 2 pins


def test_a_sliver_grows_into_a_trimmed_pause_that_stays_trimmed() -> None:
    """F7: folding re-runs the trim instead of bringing the whole pause back."""
    out, folds = fold_slivers([Piece("a", 0, 18, 0), Piece("a", 60, 150, 0)], WIDE, FPS, need=25)
    assert out == [Piece("a", 0, 25, 0), Piece("a", 60, 150, 0)]
    assert [(f.into, f.readded_frames, f.retrimmed) for f in folds] == [("next", 7, True)]
    out, folds = fold_slivers([Piece("a", 0, 100, 0), Piece("a", 140, 150, 0)], WIDE, FPS, need=25)
    assert out == [Piece("a", 0, 100, 0), Piece("a", 125, 150, 0)]
    assert folds[0].retrimmed and folds[0].into == "previous"


def test_a_pause_too_short_to_regrow_comes_back_whole() -> None:
    out, folds = fold_slivers([Piece("a", 0, 18, 0), Piece("a", 22, 150, 0)], WIDE, FPS, need=25)
    assert out == [Piece("a", 0, 150, 0)]
    assert [(f.into, f.readded_frames, f.retrimmed) for f in folds] == [("next", 4, False)]


def test_both_regrowths_tie_toward_the_previous_side() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 110, 0), Piece("a", 160, 250, 0)]
    out, folds = fold_slivers(pieces, WIDE, FPS, need=25)
    assert out[1] == Piece("a", 85, 110, 0) and folds[0].into == "previous"


def test_the_nudge_reach_rounds_down() -> None:
    """V07: 150 ms is 4.5 frames at 29.97 fps, so a nudge reaches 4 frames, not 5."""
    from aar.plan.produce.joins import reach_frames

    assert reach_frames(150, Fps(num=30000, den=1001)) == 4
    assert reach_frames(150, FPS) == 3 and reach_frames(160, FPS) == 4 and reach_frames(0, FPS) == 0


def test_a_nudge_never_starts_before_the_media() -> None:
    """V06: with no crossfade handle to keep, the media's first frame still bounds a nudge."""
    cover = {"a": Coverage(0, None), "b": Coverage(100, None)}
    pieces = [Piece("a", 0, 50, 0), Piece("b", 101, 200, 0)]
    out, _, warnings = place_joins(
        pieces, FakeJudge(ins=(-1, 0)), NO_WORDS, cover, ONE_KEEP, FPS,
        nudge_frames=3, need=25, handle=0,
    )  # fmt: skip
    assert out[1].rec_in == 100  # source frame 0, never -1
    out, _, warnings = place_joins(
        pieces, FakeJudge(ins=(-1,)), NO_WORDS, cover, ONE_KEEP, FPS,
        nudge_frames=3, need=25, handle=0,
    )  # fmt: skip
    assert out[1].rec_in == 101 and [w.code for w in warnings] == ["join-not-quiet"]


def test_a_laugh_exactly_at_the_final_keep_start_counts() -> None:
    """V11: the onset may equal the range's first frame (then no clean frame is long enough)."""
    _, end, warnings = ending([Piece("a", 150, 400, 1)], [(6.0, 7.0, 1.0)])
    assert end is None and [w.code for w in warnings] == ["end-not-on-laugh"]


def test_the_ending_never_runs_past_the_media() -> None:
    """V13: target 255 (10.2 s) is past the media's last frame 253, so 253 it is."""
    out, end, _ = ending(
        [Piece("a", 150, 250, 1)], [(10.0, 11.0, 1.0)], cover={"a": Coverage(0, 253)}
    )
    assert end is not None and out[-1].rec_out == 253


def test_the_face_margin_is_the_distance_to_the_crop_edge() -> None:
    """V22: a face pinned at 0.95 under a 0.32-wide crop that clamps at 0.84 sits 0.05 inside."""
    preset = FramingPreset(name="A", zoom=1.0)
    _, margin = reframe_keys([0.95] * 60, preset, 0.32, 25.0, SteadySettings(), max_pan_speed=0.3)
    assert margin == pytest.approx(0.05)


def test_a_one_frame_schedule_gap_is_refused() -> None:
    """V28: 2.00-2.04 s is one frame (50 -> 51) with no camera."""
    plan = stub(("a", "A"), ("b", "B"))
    sched = schedule_frames(
        plan,
        CameraSchedule.model_validate(
            {"spans": [{"start_s": 0, "end_s": 2.0, "camera": "A"},
                       {"start_s": 2.04, "end_s": 9, "camera": "B"}]}
        ),
        "A",
    )  # fmt: skip
    with pytest.raises(ProduceInputError) as exc:
        split(25, 100, 0, sched, WIDE, FPS)
    assert exc.value.code == "schedule-gap"


def test_zero_length_and_15_ms_words_split_a_pause() -> None:
    """The validator's repro (finding 2): 'uh' (0 ms) and 'mm' (15 ms) in a 3 s quiet run were
    removed; with min_word_ms 40 every word splits the pause and stays."""
    fps = Fps(num=30000, den=1001)
    for text, a, b in (("uh", 11.5, 11.5), ("mm", 11.5, 11.515), ("um", 11.5, 11.7)):
        w = Words(
            [Word(text="so", start_s=9.0, end_s=10.0), Word(text=text, start_s=a, end_s=b),
             Word(text="then", start_s=13.0, end_s=13.5)],
            0.010, fps, 0.040,
        )  # fmt: skip
        trims = find_trims(
            0, fps.frames(8.0), fps.frames(15.0), [(10.0, 13.0)], w, fps,
            min_pause_s=0.9, beat_s=0.45, beat_lead_s=0.25,
        )  # fmt: skip
        assert len(trims) == 2, text
        assert not any(
            fps.seconds(t.out_frame) <= a and b <= fps.seconds(t.in_frame) for t in trims
        )


def test_a_word_occupies_at_least_min_word_around_its_middle() -> None:
    w = Words([Word(text="uh", start_s=1.0, end_s=1.0)], 0.01, FPS, 0.04)
    assert w.spans(0.0, 2.0) == [pytest.approx((0.98, 1.02))]
    assert w.speech_between(1.019, 2.0) is not None and w.speech_between(1.021, 2.0) is None
    assert Words([Word(text="uh", start_s=1.0, end_s=1.0)], 0.01, FPS).spans(0.0, 2.0) == []
    long = Words([Word(text="long", start_s=1.0, end_s=2.0)], 0.01, FPS, 0.04)
    assert long.spans(0.0, 3.0) == [pytest.approx((1.01, 1.99))]  # the core, as before
