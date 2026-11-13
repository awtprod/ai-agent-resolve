"""Timeline import (S20) fix round 2: the validator's round-2 findings N1-N4, the "nothing
matched" refusal and the smaller ones, each as a test that fails on the code before the fix
(aar-evidence/s20-val/r2/VERDICT.md), plus tests for the surviving mutants it listed."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_timeline_import import codes, layout_rows, run, simple, simple_data, statuses
from test_timeline_import_cli import CLIPS, call, setup, write_drt
from test_timeline_import_fixes import broll_plan
from timeline_import_support import Row, build_timeline, content, drt_members, ripple, xml

from aar.cli import main
from aar.plan.cli import _import_text
from aar.plan.model import EditPlan
from aar.plan.timeline_import import (
    ASSUMPTIONS,
    TimelineImportError,
    assign_items,
    import_timeline,
)

DESIGN = Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md"


def text(result: Any) -> str:
    return _import_text(result.diff, Path("o.json"), Path("r.json"))


# -- N1: one source range used twice --------------------------------------------------------------


def teaser_plan(b_frames: int = 100) -> EditPlan:
    """The validator's p_twins plan: a teaser A 500..600 with a zoom, then B, then the main take
    A 500..600."""
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
                for s in "AB"
            ],
            "segments": [
                {
                    "source": "A",
                    "src_in": 500,
                    "src_out": 600,
                    "label": "teaser",
                    "reframe": [{"time": 0, "zoom": 2.0}],
                },
                {"source": "B", "src_in": 0, "src_out": b_frames, "label": "b"},
                {"source": "A", "src_in": 500, "src_out": 600, "label": "main"},
            ],
        }
    )


def test_a_ripple_deleted_teaser_leaves_the_main_take_and_says_it_chose() -> None:
    """Validator N1 (ripple_delete_teaser_no_tie): the main take got the teaser's label and zoom,
    and nothing was listed."""
    plan = teaser_plan(60)
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[1], c[2]]))
    assert statuses(result) == ["removed", "moved", "moved"]
    assert [(s.label, s.reframe) for s in result.plan.segments] == [("b", []), ("main", [])]
    assert result.diff["uncertain_matches"] == [
        {
            "index": 1,
            "assigned_to": 2,
            "why": "same_source_range",
            "also_overlaps": [0],
            "contested_by": [],
        }
    ]
    assert "uncertain matches (1), check them: [1] -> segments[2] (same source range as 0)" in (
        text(result)
    )


def test_a_teaser_moved_to_the_end_keeps_both_takes_fields() -> None:
    plan = teaser_plan()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[1], c[2], c[0]]))
    assert [s.label for s in result.plan.segments] == ["b", "main", "teaser"]
    assert [k.zoom for k in result.plan.segments[2].reframe] == [2.0]
    assert [u["index"] for u in result.diff["uncertain_matches"]] == [1, 2]


def test_a_lone_copy_with_no_context_is_placed_by_position_and_listed() -> None:
    plan = teaser_plan()
    rows = layout_rows(plan)
    result = run(plan, ripple(rows, [content(rows)[2]]))
    assert result.plan.segments[0].label == "teaser"  # position: no context tells
    assert result.diff["uncertain_matches"][0]["also_overlaps"] == [2]


def test_identical_copies_need_no_entry() -> None:
    data = teaser_plan().model_dump(mode="json", exclude_none=True)
    data["segments"][0] = dict(data["segments"][2])
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[1], c[2]]))
    assert result.diff["uncertain_matches"] == []


def test_assign_items_context_before_position() -> None:
    originals = [(0, "a", 0, 100, 0), (1, "b", 0, 100, 100), (2, "a", 0, 100, 200)]
    # the a item follows b's item: it is the original that followed b in the plan (2), not the
    # one at its position (0)
    got, also = assign_items([("b", 0, 100, 0), ("a", 0, 100, 100)], originals)
    assert got == {0: 1, 1: 2} and also == {1: [0]}
    # identical copies are not listed
    got, also = assign_items([("b", 0, 100, 0), ("a", 0, 100, 100)], originals, lambda a, b: True)
    assert also == {}


# -- L-a: only real contenders are listed ---------------------------------------------------------


def test_a_sliver_of_overlap_is_not_a_contender() -> None:
    originals = [(0, "a", 100, 300, 0), (1, "a", 250, 400, 200)]
    # trimmed copy of 0 (100..280) touches 1 by 30 frames: 1 is not in contention
    assert assign_items([("a", 100, 280, 0)], originals) == ({0: 0}, {})
    # an item mostly inside both is
    assert assign_items([("a", 250, 300, 0)], originals)[1] == {0: [1]}


# -- N2: titles paired by count -------------------------------------------------------------------


def titles_plan() -> EditPlan:
    """The validator's p_titles plan."""
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
                {"id": "A", "media": "E:\\Show\\A.mov", "duration": 9000, "start_timecode": 0}
            ],
            "segments": [{"source": "A", "src_in": 0, "src_out": 400}],
            "titles": [
                {"lines": ["Jane Doe, guest"], "record_in": 50, "record_out": 120},
                {"lines": ["Subscribe"], "record_in": 300, "record_out": 360},
            ],
        }
    )


def test_a_title_paired_by_order_is_kept_and_listed_as_uncertain() -> None:
    """Validator N2: the owner deleted "Subscribe" and added a title of their own; the plan put
    "Subscribe" at the new title's range as a plain move."""
    plan = titles_plan()
    rows = layout_rows(plan)
    title_rows = [r for r in rows if r.kind == "title"]
    owner = Row(title_rows[0].track, "title", 200, 40)
    result = run(plan, [r for r in rows if r is not title_rows[1]] + [owner])
    assert [(t.lines[0], t.record_in) for t in result.plan.titles] == [
        ("Jane Doe, guest", 50),
        ("Subscribe", 200),
    ]
    [entry] = result.diff["uncertain_titles"]
    assert (entry["location"], entry["new"]["record_in"]) == ("titles[1]", 200)
    assert [e["by"] for e in result.diff["repositioned"]] == ["order"]
    assert "uncertain titles (1), paired by order only" in text(result)
    assert result.diff["unchanged"] is False


def test_a_title_that_follows_its_content_is_certain() -> None:
    """Kills the validator's R09 (re-anchoring switched off: the order fallback would still pair
    it, but as uncertain)."""
    plan = titles_plan()
    data = plan.model_dump(mode="json", exclude_none=True)
    data["segments"] = [
        {"source": "A", "src_in": 0, "src_out": 100},
        {"source": "A", "src_in": 1000, "src_out": 1400},
    ]
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    titles = [r for r in rows if r.kind == "title"]
    shifted = [replace(t, record_in=t.record_in - 100) for t in titles if t.record_in >= 100]
    edited = [*ripple([r for r in rows if r.kind != "title"], [c[1]]), *shifted]
    result = run(plan, edited)
    assert [(t.lines[0], t.record_in) for t in result.plan.titles] == [("Subscribe", 200)]
    assert result.diff["uncertain_titles"] == []
    assert [(e["location"], e["by"]) for e in result.diff["repositioned"]] == [
        ("titles[1]", "content")
    ]


def test_the_order_fallback_needs_as_many_items_as_titles() -> None:
    """Kills R10: one plan title gone, two unknown title items: nothing is paired."""
    plan = titles_plan()
    rows = layout_rows(plan)
    track = next(r.track for r in rows if r.kind == "title")
    edited = [r for r in rows if r.kind != "title" or r.record_in == 50]
    edited += [Row(track, "title", 200, 20), Row(track, "title", 250, 20)]
    result = run(plan, edited)
    assert codes(result, "dropped") == ["title_removed"]
    assert codes(result).count("title_added") == 2
    assert result.diff["uncertain_titles"] == []


# -- N3: elements on another track than predicted -------------------------------------------------


def tracks_plan() -> EditPlan:
    """The validator's p_tracks plan: A, B on V1, b-roll G on V2 over B, a title over B."""
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
                {"id": s, "media": f"E:\\Show\\{s}.mov", "duration": 9000, "start_timecode": 0}
                for s in "ABG"
            ],
            "segments": [
                {"source": "A", "src_in": 0, "src_out": 200},
                {"source": "B", "src_in": 0, "src_out": 200},
                {"source": "G", "track": 2, "src_in": 0, "src_out": 50, "record_in": 250},
            ],
            "titles": [{"lines": ["name of B"], "record_in": 220, "record_out": 300}],
        }
    )


def moved_tracks(rows: list[Row], mapping: dict[str, str]) -> list[Row]:
    return [
        replace(r, track=mapping.get(r.track, r.track)) if r.kind != "audio" else r for r in rows
    ]


@pytest.mark.parametrize(
    ("mapping", "found"),
    [
        ({"V2": "V3", "V3": "V2"}, {"segments[2]": "V3", "titles[0]": "V2"}),
        ({"V2": "V3", "V3": "V4"}, {"segments[2]": "V3", "titles[0]": "V4"}),
        (
            {"V1": "V2", "V2": "V3", "V3": "V4"},
            {"segments[0]": "V2", "segments[1]": "V2", "segments[2]": "V3", "titles[0]": "V4"},
        ),
    ],
    ids=["swapped", "off_v1_up_one", "spine_up_one"],
)
def test_an_element_on_another_track_is_kept_with_one_warning(
    mapping: dict[str, str], found: dict[str, str]
) -> None:
    """Validator N3: b-roll and title deleted from the plan (removed + added entries)."""
    plan = tracks_plan()
    timeline, media = build_timeline(moved_tracks(layout_rows(plan), mapping), rate=25.0)
    result = import_timeline(plan, timeline, media)
    assert result.plan == plan
    assert set(statuses(result)) == {"unchanged"}
    assert {m["location"]: m["found_track"] for m in result.diff["track_mismatch"]} == found
    assert codes(result, "dropped") == [] and result.diff["added"] == []
    assert not [c for c in codes(result) if c in ("title_added", "other_picture_track")]
    assert result.diff["unchanged"] is False
    assert f"TRACK ASSUMPTION FAILED ({len(found)})" in text(result)


def test_two_titles_at_one_range_keep_their_own_tracks() -> None:
    """Two plan titles at one range on two lanes; the owner deletes the first: the second keeps
    its own item, the first is removed (a cross-track match came before the own-track one)."""
    data = titles_plan().model_dump(mode="json", exclude_none=True)
    data["titles"] = [
        {"lines": ["one"], "record_in": 50, "record_out": 120},
        {"lines": ["two"], "record_in": 50, "record_out": 120},
    ]
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    titles = [r for r in rows if r.kind == "title"]
    assert titles[0].track != titles[1].track
    result = run(plan, [r for r in rows if r is not titles[0]])
    assert [t.lines for t in result.plan.titles] == [["two"]]
    assert result.diff["track_mismatch"] == []
    assert codes(result, "dropped") == ["title_removed"]


def test_a_cross_track_match_never_takes_an_item_its_own_track_still_misses() -> None:
    """Seed 8591 of the validator's generator: title 0's content-follow range equals title 1's
    moved item on title 1's own track; title 1's item is not taken across tracks."""
    data = titles_plan().model_dump(mode="json", exclude_none=True)
    data["titles"] = [
        {"lines": ["one"], "record_in": 50, "record_out": 120},
        {"lines": ["two"], "record_in": 60, "record_out": 130},
    ]
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    t0, t1 = [r for r in rows if r.kind == "title"]
    edited = [r for r in rows if r.kind != "title"]
    edited += [replace(t0, record_in=10), replace(t1, record_in=50)]  # t1 sits at t0's range
    result = run(plan, edited)
    assert result.diff["track_mismatch"] == []
    assert [(t.lines[0], t.record_in) for t in result.plan.titles] == [("one", 10), ("two", 50)]


# -- N4: a not-carried item keeps the plan's own segment ----------------------------------------


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"speed": 0.5}, "retimed"),
        ({"fractions": {"record_start": 0.5}}, "sub_frame"),
        ({"path": "E:\\Show\\B.png"}, "still_image"),
    ],
)
def test_a_not_carried_item_keeps_its_segment_in_place(change: dict[str, Any], code: str) -> None:
    """Validator N4: the segment was removed, against DESIGN ("the plan keeps its own version")."""
    data = simple_data()
    data["titles"], data["markers"] = [], []
    if code == "still_image":
        data["sources"][1]["media"] = "E:\\Show\\B.png"
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    second = content(rows)[1]
    edited = [replace(r, sound=None, **change) if r is second else r for r in rows]
    result = run(plan, edited)
    assert code in codes(result)
    # camB is also the fourth segment's source: as a still file both are stills
    last = "kept" if code == "still_image" else "unchanged"
    assert statuses(result) == ["unchanged", "kept", "unchanged", last]
    assert result.diff["segments"][1]["kept_because"] == code
    assert result.plan.segments == plan.segments
    assert f"segments[1] kept ({code}: the plan keeps its own version)" in text(result)


def test_a_stand_in_is_chosen_by_its_in_point() -> None:
    """Two camB originals no item took; the retimed item reads camB 900.. (the fourth segment's
    range), so the fourth is kept, not the nearer second."""
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    retimed = replace(c[3], speed=2.0, sound=None)
    result = run(plan, ripple(rows, [c[0], retimed, c[2]]))
    assert statuses(result) == ["unchanged", "removed", "unchanged", "kept"]
    assert result.diff["uncertain_matches"] == []


def test_a_stand_in_without_evidence_is_kept_and_listed() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    # a retimed camB clip from elsewhere in the file, after the last clip
    stray = replace(c[3], src_in=3000, speed=2.0, sound=None)
    result = run(plan, ripple(rows, [c[0], c[2], stray]))
    [entry] = result.diff["uncertain_matches"]
    assert entry["why"] == "not_carried_item" and entry["also_overlaps"] == [1]
    assert "segments[" in text(result) and "uncertain matches (1)" in text(result)


# -- nothing matched ---------------------------------------------------------------------------


def test_a_timeline_whose_media_match_nothing_is_refused() -> None:
    plan = simple()
    rows = [
        replace(r, path=r.path.replace("E:\\", "X:\\")) if r.path else r for r in layout_rows(plan)
    ]
    with pytest.raises(TimelineImportError) as caught:
        run(plan, rows)
    assert caught.value.code == "nothing_matched"
    assert "media-map" in caught.value.fix_hint
    # one item that still matches is enough to import
    keep = layout_rows(plan)
    assert run(plan, [*rows[1:], keep[0]]).diff is not None


def test_nothing_matched_through_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    moved = [dataclasses.replace(c, media=c.media.replace("D:\\", "Q:\\")) for c in CLIPS]
    drt, plan = setup(tmp_path, moved)
    code, out, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (nothing_matched)" in err and out == ""
    assert not (tmp_path / "edited.plan.json").exists()


# -- the validator's surviving mutants (mutate_r2.out) -------------------------------------------


def test_an_info_level_reader_issue_that_can_hide_items_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R01: `unknown-track` is an info-level issue; it can hide a track's items."""
    members = drt_members(CLIPS)
    key = next(k for k in members if k.startswith("SeqContainer/"))
    body = members[key]
    assert isinstance(body, str)
    members[key] = body.replace(
        " <AudioTrackVec/>",
        ' <AudioTrackVec><Element><Sm2TiOddTrack DbId="odd"/></Element></AudioTrackVec>',
    )
    _, plan = setup(tmp_path)
    drt = write_drt(tmp_path / "odd.drt", members)
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (damaged_timeline)" in err and "unknown-track" in err


def test_the_cli_passes_reader_warnings_into_the_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R17: the reader's pass-through warnings must reach the diff and stdout."""
    members = drt_members(CLIPS)
    members["MediaPool/Stray/MpFolder.xml"] = xml(
        '<Sm2MpFolder DbId="bin9"><FieldsBlob/><MpFolder>nosuch</MpFolder><Name>Stray</Name>'
        "<MediaVec></MediaVec></Sm2MpFolder>\n"
    )
    _, plan = setup(tmp_path)
    drt = write_drt(tmp_path / "warn.drt", members)
    code, out, _ = call(tmp_path, capsys, drt, plan)
    assert code == 0 and "reader warnings (1): orphan-bin x1" in out
    diff = json.loads((tmp_path / "edited.diff.json").read_text())
    assert diff["reader_warnings"]["count"] == 1
    assert diff["reader_warnings"]["listed"][0]["code"] == "orphan-bin"


def test_a_missing_report_folder_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R21: only the --out folder was checked in the test before."""
    drt, plan = setup(tmp_path)
    missing = tmp_path / "gone"
    code = main(
        [
            "plan",
            "import-timeline",
            str(drt),
            "--plan",
            str(plan),
            "--out",
            str(tmp_path / "p.json"),
            "--report",
            str(missing / "r.json"),
        ]
    )
    err = capsys.readouterr().err
    assert code == 2 and "refused (output_folder_missing)" in err and str(missing) in err
    assert not (tmp_path / "p.json").exists()


def test_a_join_keeps_the_fields_of_the_member_that_holds_the_piece() -> None:
    """R22: the piece is the second member's, so the result keeps the second's fields."""
    data = simple_data()
    data["segments"] = [
        {"source": "camA", "src_in": 0, "src_out": 100, "label": "one"},
        {"source": "camA", "src_in": 100, "src_out": 200, "label": "two"},
        {"source": "camB", "src_in": 0, "src_out": 100},
    ]
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    joined = replace(c[0], src_in=60, duration=140)  # 40 of one, all of two
    result = run(plan, ripple(rows, [joined, c[2]]))
    assert statuses(result)[:2] == ["joined", "joined"]
    assert result.diff["segments"][0]["joined"] == {"segments": [0, 1], "fields_from": 1}
    assert result.plan.segments[0].label == "two"
    assert "fields of segments[1]" in text(result)


# -- docs agree with the code -------------------------------------------------------------------


def test_the_diffs_assumptions_are_the_design_table() -> None:
    section = DESIGN.read_text(encoding="utf-8").split("#### Timeline import (S20)")[1]
    table = section.split("**Assumed until a real round trip")[1].split("\n\n")[1]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    rows = rows[1:]  # the header
    assert len(rows) == len(ASSUMPTIONS)
    for row, assumption in zip(rows, ASSUMPTIONS, strict=True):
        assert f"`{assumption['code']}`" in row


def test_the_cli_help_does_not_say_only_v1_becomes_the_segments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with contextlib.suppress(SystemExit):
        main(["plan", "import-timeline", "--help"])
    help_text = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert "V1 items in record order become the segments" not in help_text
    assert "track_mismatch" in help_text and "even in a timeline --timeline does not pick" in (
        help_text
    )


def test_a_caption_ending_on_the_last_frame_is_kept() -> None:
    """R18: a caption over the outro's last frames ends exactly at the new end."""
    data = simple_data()
    data["titles"], data["markers"] = [], []
    data["captions"] = [{"lines": ["bye"], "record_in": 880, "record_out": 900}]
    plan = EditPlan.model_validate(data)
    assert plan.total_frames() == 900
    result = run(plan, layout_rows(plan))
    assert [(c.record_in, c.record_out) for c in result.plan.captions] == [(880, 900)]
    assert codes(result, "dropped") == []


def test_an_added_off_v1_clip_starting_with_a_v1_clip_follows_it() -> None:
    """R14: an added b-roll clip that starts on the frame a V1 clip starts is listed after it."""
    plan = broll_plan()
    rows = layout_rows(plan)
    broll = next(r for r in rows if r.track == "V2" and r.kind == "video")
    added = replace(broll, record_in=200, src_in=1000, duration=20, sound=None)
    result = run(plan, [*rows, added])
    assert [(s.source, s.src_in) for s in result.plan.segments] == [
        ("A", 0),
        ("B", 0),
        ("G", 0),
        ("G", 1000),
    ]


def test_a_caption_over_a_joined_neighbour_follows_the_join() -> None:
    """R08: the second of two joined neighbours has no piece of its own; its content is in the
    first's piece, so a caption over it stays."""
    data = simple_data()
    data["segments"] = [
        {"source": "camA", "src_in": 0, "src_out": 100, "label": "one"},
        {"source": "camA", "src_in": 100, "src_out": 200, "label": "two"},
        {"source": "camB", "src_in": 0, "src_out": 100},
    ]
    data["titles"], data["markers"] = [], []
    data["captions"] = [{"lines": ["in two"], "record_in": 150, "record_out": 170}]
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    joined = replace(c[0], duration=190)
    result = run(plan, ripple(rows, [joined, c[2]]))
    assert statuses(result)[:2] == ["joined", "joined"]
    assert [(x.record_in, x.record_out) for x in result.plan.captions] == [(150, 170)]


def test_a_lone_stand_in_whose_in_point_is_elsewhere_is_listed() -> None:
    """One camB original free and a retimed camB clip reading frames outside it: kept (the plan
    loses nothing) but listed, with no other candidate."""
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    stray = replace(c[3], src_in=3000, speed=2.0, sound=None, record_in=0)
    result = run(plan, ripple(rows, [c[0], c[2], c[3], stray]))
    assert statuses(result)[1] == "kept"
    assert result.diff["uncertain_matches"] == [
        {
            "index": 3,
            "assigned_to": 1,
            "why": "not_carried_item",
            "also_overlaps": [],
            "contested_by": [],
        }
    ]
    assert "[3] -> segments[1] (a not-carried item kept as it without evidence)" in text(result)
