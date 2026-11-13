"""Timeline import (S20) fix round 3: the validator's round-3 findings A, B and F (M06), and the
rule that a timeline with items the plan cannot hold is never a clean import, each as a test that
fails on the code before the fix (aar-evidence/s20-val/r3/VERDICT.md)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_timeline_import import layout_rows, run, simple, statuses
from timeline_import_support import Row, content

from aar.plan.cli import _import_text
from aar.plan.model import EditPlan
from aar.plan.timeline_import import ImportResult

DESIGN = Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md"


def text(result: ImportResult) -> str:
    return _import_text(result.diff, Path("o.json"), Path("r.json"))


def plan_of(segments: list[dict[str, Any]], **extra: Any) -> EditPlan:
    return EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "fps": 25,
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {
                    "id": s,
                    "media": f"E:\\Show\\{s}.mov",
                    "duration": 9000,
                    "start_timecode": 0,
                    "frame_size": [1920, 1080],
                }
                for s in "ABC"
            ],
            "segments": segments,
            **extra,
        }
    )


def pieces(result: ImportResult) -> list[tuple[int, str, list[tuple[int, int, int]]]]:
    return [
        (
            r["index"],
            r["status"],
            [(n["src_in"], n["src_out"], n["record_in"]) for n in r["new"]],
        )
        for r in result.diff["segments"]
    ]


def added(result: ImportResult) -> list[tuple[str, int, int, int]]:
    return [(a["source"], a["src_in"], a["src_out"], a["record_in"]) for a in result.diff["added"]]


# -- A: a stand-in claims its original before the overlap matching -------------------------------


def answer_plan() -> EditPlan:
    """The validator's p_attack case a: A 1000..1100 with a label and a zoom, B, A 2000..2100."""
    return plan_of(
        [
            {
                "source": "A",
                "src_in": 1000,
                "src_out": 1100,
                "label": "answer",
                "reframe": [{"time": 0, "zoom": 1.6}],
            },
            {"source": "B", "src_in": 0, "src_out": 100, "label": "b"},
            {"source": "A", "src_in": 2000, "src_out": 2100, "label": "later"},
        ]
    )


def slowed_with_cutaway(
    cutaway_first: bool, in_point: bool = True
) -> tuple[EditPlan, list[Row], ImportResult]:
    plan = answer_plan()
    a0, b1, a2 = content(layout_rows(plan))
    path = a0.path
    slowed = replace(a0, speed=0.5, duration=200, sound=None)
    if not in_point:
        slowed = replace(slowed, src_in=None)
    cutaway = Row("V1", "video", 0, 30, 1050, path)
    if cutaway_first:
        rows = [
            cutaway,
            replace(slowed, record_in=30),
            replace(b1, record_in=230, sound=None),
            replace(a2, record_in=330, sound=None),
        ]
    else:
        rows = [
            slowed,
            replace(b1, record_in=200, sound=None),
            replace(a2, record_in=300, sound=None),
            replace(cutaway, record_in=400),
        ]
    return plan, rows, run(plan, rows)


@pytest.mark.parametrize(
    ("cutaway_first", "in_point", "kept_at", "cutaway_at"),
    [(False, True, 0, 400), (True, True, 30, 0), (False, False, 0, 400)],
    ids=["a_cutaway_after", "a2_cutaway_before", "a3_in_unreadable"],
)
def test_a_retimed_clip_keeps_its_original_against_an_overlapping_cutaway(
    cutaway_first: bool, in_point: bool, kept_at: int, cutaway_at: int
) -> None:
    """Validator A (cases a, a2, a3): the cutaway A 1050..1080 took segments[0] by overlap,
    with its label and zoom ("trimmed"), the retimed clip was not kept, nothing was listed."""
    plan, _, result = slowed_with_cutaway(cutaway_first, in_point)
    rows = result.diff["segments"]
    assert rows[0]["status"] == "kept" and rows[0]["kept_because"] == "retimed"
    assert rows[0]["new"][0]["record_in"] == kept_at
    assert added(result) == [("A", 1050, 1080, cutaway_at)]
    kept = next(s for s in result.plan.segments if s.label == "answer")
    assert kept == plan.segments[0].model_copy(update={"record_in": kept.record_in})
    cut = next(s for s in result.plan.segments if (s.src_in, s.src_out) == (1050, 1080))
    assert (cut.label, cut.reframe) == (None, [])
    [entry] = result.diff["uncertain_matches"]
    assert entry["assigned_to"] == 0 and entry["why"] == "contested"
    assert [(c["record_in"], c["carried"]) for c in entry["contested_by"]] == [(cutaway_at, True)]
    assert f"also a candidate: carried V1 {cutaway_at}..{cutaway_at + 30}" in text(result)


def test_b_identical_copies_one_bladed_one_retimed() -> None:
    """Validator A (case b): both rows said "trimmed", the retimed copy was not kept."""
    plan = plan_of(
        [
            {"source": "A", "src_in": 500, "src_out": 600},
            {"source": "B", "src_in": 0, "src_out": 100},
            {"source": "A", "src_in": 500, "src_out": 600},
        ]
    )
    t0, tb, t2 = content(layout_rows(plan))
    rows = [
        replace(t0, duration=40, sound=None),
        replace(t0, record_in=40, src_in=560, duration=40, sound=None),
        replace(tb, record_in=80, sound=None),
        replace(t2, record_in=180, speed=0.5, sound=None),
    ]
    result = run(plan, rows)
    assert pieces(result) == [
        (0, "split", [(500, 540, 0), (560, 600, 40)]),
        (1, "moved", [(0, 100, 80)]),
        (2, "kept", [(500, 600, 180)]),
    ]
    [entry] = result.diff["uncertain_matches"]
    assert entry["assigned_to"] == 2 and entry["why"] == "contested"
    assert [c["record_in"] for c in entry["contested_by"]] == [0, 40]


SEED_FILES = {
    "A": "E:\\Show\\CamA.mov",
    "B": "E:\\Show\\CamB.mov",
    "C": "F:\\Other\\CamC.MOV",
    "O": "E:\\Assets\\Outro.mov",
}


def seed_plan(fps: Any, segments: list[dict[str, Any]], **extra: Any) -> EditPlan:
    return EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "fps": fps,
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {
                    "id": s,
                    "media": SEED_FILES[s],
                    "duration": 4000,
                    "start_timecode": 0,
                    "frame_size": [1920, 1080],
                }
                for s in "ABC"
            ]
            + [{"id": "O", "media": SEED_FILES["O"], "duration": 90, "start_timecode": 0}],
            "segments": segments,
            "outro": {"source": "O", "audio_track": "A1"},
            "audio_tracks": [{"name": "A1"}, {"name": "A2"}],
            **extra,
        }
    )


def seed_rows(spec: list[tuple[str, int, int, int, str, float]]) -> list[Row]:
    rows = []
    for track, rin, dur, src, path, speed in spec:
        sound = f"A{track[1:]}" if speed == 1.0 else None
        role = "outro" if path == SEED_FILES["O"] else ""
        rows.append(Row(track, "video", rin, dur, src, path, sound=sound, speed=speed, role=role))
    return rows


def test_seed_5280_a_slowed_segment_is_kept_and_the_overlapping_clip_added() -> None:
    """Validator generator seed 5280 (gen3_realin.out, K4): the retimed segments[0] lost its
    original to an added clip of CamA that overlaps it."""
    plan = seed_plan(
        25,
        [
            {"source": "A", "src_in": 49, "src_out": 144, "gain_db": -3.0, "label": "L"},
            {"source": "B", "src_in": 1462, "src_out": 1519},
            {"source": "C", "src_in": 3470, "src_out": 3649},
        ],
    )
    a, c, o = SEED_FILES["A"], SEED_FILES["C"], SEED_FILES["O"]
    rows = seed_rows(
        [
            ("V1", 0, 95, 49, a, 0.5),
            ("V1", 117, 102, 3470, c, 1.0),
            ("V1", 219, 57, 29, a, 1.0),
            ("V1", 276, 90, 0, o, 1.0),
        ]
    )
    result = run(plan, rows)
    assert pieces(result) == [
        (0, "kept", [(49, 144, 0)]),
        (1, "removed", []),
        (2, "trimmed", [(3470, 3572, 117)]),
    ]
    assert added(result) == [("A", 29, 86, 219)]
    assert [u["assigned_to"] for u in result.diff["uncertain_matches"]] == [0]


def test_seed_11956_a_slowed_segment_is_kept_and_the_overlapping_clip_added() -> None:
    """Validator generator seed 11956 (gen3_realin.out, K4), at 29.97 fps with b-roll on V2."""
    plan = seed_plan(
        {"num": 30000, "den": 1001},
        [
            {"source": "A", "src_in": 911, "src_out": 955},
            {"source": "C", "track": 2, "src_in": 1943, "src_out": 1990, "record_in": 235},
            {"source": "B", "src_in": 1414, "src_out": 1601},
            {"source": "C", "src_in": 447, "src_out": 630, "gain_db": -3.0, "label": "L"},
            {"source": "C", "track": 2, "src_in": 2889, "src_out": 2903, "record_in": 380},
            {"source": "B", "src_in": 173, "src_out": 409},
        ],
    )
    a, b, c, o = (SEED_FILES[k] for k in "ABCO")
    rows = seed_rows(
        [
            ("V1", 0, 44, 911, a, 1.0),
            ("V1", 44, 124, 1414, b, 0.5),
            ("V1", 168, 183, 447, c, 1.0),
            ("V1", 351, 63, 10, "G:\\Stock\\CamA.mov", 1.0),
            ("V1", 414, 236, 173, b, 1.0),
            ("V1", 650, 68, 1541, b, 1.0),
            ("V1", 718, 90, 0, o, 1.0),
            ("V2", 380, 14, 2889, c, 1.0),
        ]
    )
    result = run(plan, rows, rate=30000 / 1001)
    assert statuses(result) == ["unchanged", "removed", "kept", "moved", "unchanged", "unchanged"]
    assert result.diff["segments"][2]["new"][0]["record_in"] == 44
    assert added(result) == [("B", 1541, 1609, 650)]
    [entry] = result.diff["uncertain_matches"]
    assert entry["assigned_to"] == 2 and entry["contested_by"][0]["record_in"] == 650


def test_an_exact_item_beats_a_stand_in_and_both_are_listed() -> None:
    """Whichever wins: the plan's exact clip of segments[0] takes it; a retimed clip whose in
    point lies inside segments[0] is a candidate too, so the exact clip's match is listed."""
    plan = answer_plan()
    a0, b1, a2 = content(layout_rows(plan))
    rows = [
        a0,
        replace(b1, sound=None),
        replace(a2, sound=None),
        replace(a0, record_in=300, src_in=1020, speed=0.5, duration=60, sound=None),
    ]
    result = run(plan, rows)
    assert statuses(result)[0] == "unchanged"
    [entry] = result.diff["uncertain_matches"]
    assert (entry["index"], entry["assigned_to"], entry["why"]) == (0, 0, "contested")
    assert [(c["index"], c["carried"]) for c in entry["contested_by"]] == [(None, False)]
    assert "also a candidate: not-carried V1 300..360" in text(result)


# -- not carried on the primary track: never a clean import -------------------------------------


def test_a_not_carried_item_on_the_primary_track_needs_attention() -> None:
    plan = simple()
    rows = layout_rows(plan)
    second = content(rows)[1]
    result = run(plan, [replace(r, speed=0.5, sound=None) if r is second else r for r in rows])
    assert result.diff["primary_not_carried"] == 1
    assert not result.has_errors and result.needs_attention
    first = text(result).splitlines()[0]
    assert first == (
        "The plan could not hold 1 item(s) on V1 (see not carried); check the segments around "
        "them by eye."
    )


def test_an_unedited_timeline_is_still_clean() -> None:
    plan = simple()
    result = run(plan, layout_rows(plan))
    assert result.diff["primary_not_carried"] == 0 and not result.needs_attention
    assert text(result).startswith("timeline ")


def test_a_not_carried_item_off_the_primary_track_is_not_counted() -> None:
    plan = plan_of(
        [
            {"source": "A", "src_in": 0, "src_out": 400},
            {"source": "C", "track": 2, "src_in": 100, "src_out": 150, "record_in": 100},
        ]
    )
    rows = layout_rows(plan)
    broll = next(r for r in rows if r.track == "V2")
    result = run(plan, [replace(r, speed=2.0, sound=None) if r is broll else r for r in rows])
    assert result.diff["segments"][1]["status"] == "kept"
    assert result.diff["primary_not_carried"] == 0 and not result.needs_attention


def test_design_states_the_exit_and_no_longer_says_the_plan_loses_nothing() -> None:
    design = DESIGN.read_text(encoding="utf-8")
    assert "so the plan loses nothing" not in design
    assert "`primary_not_carried`" in design and "`kept_other_length`" in design


# -- B: a kept segment at another length on the timeline -----------------------------------------


@pytest.mark.parametrize(("speed", "frames"), [(0.5, 200), (2.0, 50)])
def test_a_kept_segment_at_another_length_is_listed(speed: float, frames: int) -> None:
    """Validator B (p_overlap slow_down_50): B at 50 % fills 100..300 on the timeline; the plan
    kept its own 100 frames at 100 and the next segment at 300, a gap with no entry."""
    plan = plan_of(
        [
            {"source": "A", "src_in": 0, "src_out": 100},
            {"source": "B", "src_in": 0, "src_out": 100, "label": "speed me"},
            {"source": "A", "src_in": 500, "src_out": 600},
        ]
    )
    c0, c1, c2 = content(layout_rows(plan))
    rows = [
        c0,
        replace(c1, speed=speed, duration=frames, sound=None),
        replace(c2, record_in=100 + frames),
    ]
    result = run(plan, rows)
    assert statuses(result) == ["unchanged", "kept", "moved"]
    [entry] = result.diff["kept_other_length"]
    assert entry["location"] == "segments[1]"
    assert (entry["timeline_frames"], entry["plan_frames"]) == (frames, 100)
    line = next(x for x in text(result).splitlines() if x.startswith("kept at another length"))
    gap = f"a {frames - 100}-frame gap after it" if frames > 100 else f"{100 - frames} frame(s) of"
    assert f"segments[1] {frames} frames on the timeline, 100 in the plan ({gap}" in line


def test_a_kept_segment_at_its_own_length_is_not_listed() -> None:
    plan = simple()
    rows = layout_rows(plan)
    second = content(rows)[1]
    result = run(plan, [replace(r, speed=0.5, sound=None) if r is second else r for r in rows])
    assert result.diff["segments"][1]["status"] == "kept"
    assert result.diff["kept_other_length"] == []


# -- F (M06): the primary-track guard of the cross-track match -----------------------------------


def test_b_roll_cut_into_the_spine_stays_in_the_spine() -> None:
    """Validator p_cutin (mutant M06): the owner moved the V2 b-roll into V1 as a cut-in. Taken
    across tracks it came back on V2 and left a 50-frame hole in the spine."""
    plan = plan_of(
        [
            {"source": "A", "src_in": 0, "src_out": 200},
            {"source": "A", "src_in": 1000, "src_out": 1200},
            {
                "source": "C",
                "track": 2,
                "src_in": 100,
                "src_out": 150,
                "record_in": 100,
                "label": "broll",
            },
        ]
    )
    rows = layout_rows(plan)
    first, second = content(rows)
    broll = next(r for r in rows if r.track == "V2")
    edited = [
        replace(first, sound=None),
        replace(broll, track="V1", record_in=200, sound=None),
        replace(second, record_in=250, sound=None),
    ]
    result = run(plan, edited)
    assert [(s.track, s.source) for s in result.plan.segments] == [(1, "A"), (1, "C"), (1, "A")]
    assert result.plan.record_ranges() == [(0, 200), (200, 250), (250, 450)]
    assert result.diff["track_mismatch"] == []
    assert statuses(result) == ["unchanged", "moved", "removed"]
    assert added(result) == [("C", 100, 150, 200)]
