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
    keeps, snaps = keep_frames(
        KeepList.model_validate({"ranges": [{"start_s": 1.0, "end_s": 3.0, "label": "x"}]}),
        NO_WORDS,
        FPS,
        min_shot_s=1.0,
    )
    assert [(k.start, k.end, k.label) for k in keeps] == [(25, 75, "x")]
    assert snaps == []


def test_a_keep_edge_inside_a_word_moves_out_and_keeps_the_word() -> None:
    w = words(("hello", 0.9, 1.3), ("world", 2.8, 3.2))
    keeps, snaps = keep_frames(
        KeepList.model_validate({"ranges": [{"start_s": 1.0, "end_s": 3.0}]}),
        w,
        FPS,
        min_shot_s=1.0,
    )
    # in 1.00 s is inside "hello" (0.91-1.29 core): back to frame 22 (0.88 s); out 3.00 s is inside
    # "world": forward to frame 80 (3.20 s, not strictly inside 2.81-3.19).
    assert [(k.start, k.end) for k in keeps] == [(22, 80)]
    assert [(s.edge, s.word, s.from_frame, s.to_frame) for s in snaps] == [
        ("in", "hello", 25, 22),
        ("out", "world", 75, 80),
    ]
    assert all(w.inside(f) is None for f in (22, 80))


def test_snapped_ranges_that_now_overlap_merge() -> None:
    w = words(("long", 2.9, 3.6))
    keeps, _ = keep_frames(
        KeepList.model_validate(
            {"ranges": [{"start_s": 1.0, "end_s": 3.0}, {"start_s": 3.5, "end_s": 5.0}]}
        ),
        w,
        FPS,
        min_shot_s=1.0,
    )
    assert [(k.start, k.end) for k in keeps] == [(25, 125)]


def test_touching_keep_ranges_stay_two_ranges() -> None:
    keeps, _ = keep_frames(
        KeepList.model_validate(
            {"ranges": [{"start_s": 1.0, "end_s": 3.0}, {"start_s": 3.0, "end_s": 5.0}]}
        ),
        NO_WORDS,
        FPS,
        min_shot_s=1.0,
    )
    assert [(k.start, k.end) for k in keeps] == [(25, 75), (75, 125)]


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


def judge_quiet_at(*frames: int):
    calls: list[tuple[str, int]] = []

    def judge(source: str, src_in: int) -> bool:
        calls.append((source, src_in))
        return src_in in frames

    judge.calls = calls  # type: ignore[attr-defined]
    return judge


def joins(pieces, judge, w=NO_WORDS, cover=WIDE):
    return place_joins(pieces, judge, w, cover, FPS, nudge_frames=3, need=25)


def test_a_cut_already_in_quiet_stays() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 1)]
    out, nudges, warnings = joins(pieces, judge_quiet_at(100))
    assert out == pieces and nudges == [] and warnings == []


def test_a_cut_moves_to_the_nearest_quiet_frame_later_first_on_a_tie() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 1)]
    out, nudges, _ = joins(pieces, judge_quiet_at(98, 102))
    assert out[1].rec_in == 102
    assert [(n.join, n.from_frame, n.to_frame) for n in nudges] == [(1, 100, 102)]
    out, _, _ = joins(pieces, judge_quiet_at(98, 103))
    assert out[1].rec_in == 98


def test_the_judge_gets_source_frames_not_recording_frames() -> None:
    judge = judge_quiet_at(70)
    joins(
        [Piece("a", 0, 50, 0), Piece("b", 100, 200, 1)],
        judge,
        cover={**WIDE, "b": Coverage(30, None)},
    )
    assert judge.calls == [("b", 70)]  # type: ignore[attr-defined]


def test_a_nudge_never_adds_or_drops_a_word() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 200, 1)]
    # "x" sits in 3.93-3.99 s (frames 98.25-99.75): moving the in point to 98 would add it.
    w = words(("x", 3.92, 4.0), ("y", 4.04, 4.2))
    out, nudges, warnings = joins(pieces, judge_quiet_at(98, 102), w)
    assert out[1].rec_in == 100 and nudges == []
    assert [x.code for x in warnings] == ["join-not-quiet"]


def test_a_nudge_keeps_the_cut_a_cut() -> None:
    pieces = [Piece("a", 0, 99, 0), Piece("a", 100, 200, 1)]
    out, _, warnings = joins(pieces, judge_quiet_at(99, 98))
    assert out[1].rec_in == 100 and warnings  # 99 would be unbroken playback, 98 overlaps


def test_a_nudge_may_not_strand_an_unfoldable_sliver() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 100, 126, 1)]
    out, _, warnings = joins(pieces, judge_quiet_at(102))
    assert out[1].rec_in == 100 and warnings  # 102-126 is 24 frames, keep range 1 has no other
    pieces = [Piece("a", 0, 50, 1), Piece("a", 100, 126, 1)]
    out, _, _ = joins(pieces, judge_quiet_at(102))
    assert out[1].rec_in == 102  # same keep range: a later fold can absorb it


def test_a_camera_switch_in_unbroken_playback_is_not_judged() -> None:
    judge = judge_quiet_at()
    _out, _, warnings = joins([Piece("a", 0, 50, 0), Piece("b", 50, 100, 0)], judge)
    assert judge.calls == [] and warnings == []  # type: ignore[attr-defined]


def test_no_quiet_frame_leaves_the_cut_and_names_the_join() -> None:
    pieces = [Piece("a", 0, 50, 0), Piece("a", 60, 120, 0), Piece("b", 200, 300, 1)]
    out, _, warnings = joins(pieces, judge_quiet_at(60))
    assert out == pieces
    assert len(warnings) == 1 and "join 2" in warnings[0].message
    assert warnings[0].source == "b" and warnings[0].at_s == 8.0


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


def test_the_ending_may_grow_into_a_trimmed_pause_or_past_the_keep_out() -> None:
    pieces = [Piece("a", 150, 300, 1), Piece("a", 320, 400, 1)]
    out, _, _ = ending(pieces, [(12.3, 13.0, 1.0)])
    assert out[-1] == Piece("a", 150, 312, 1)  # 12.5 s = frame 312.5 -> 312, in the trimmed 300-320
    out, _, _ = ending(pieces, [(15.95, 17.0, 1.0)])
    assert out[-1] == Piece("a", 320, 404, 1)  # 16.15 s: 4 frames past the keep out


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
    ]
    assert framing_levels(pieces, sources, 3) == [0, 1, 2, 0, 1, 0]
    assert framing_levels(pieces, sources, 2) == [0, 1, 0, 0, 1, 0]


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
