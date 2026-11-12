"""Timeline import (S20): pure-function tests on timelines built from a fixture plan's layout."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest
from export_support import EXPORT_FIXTURES, MEDIA_MAP, load_plan, permissive
from timeline_import_support import Row, build_timeline, content, ripple, rows_from_layout

from aar.plan.export import media_map_from_mapping
from aar.plan.export.layout import build_layout
from aar.plan.model import EditPlan, ReframeKey
from aar.plan.timeline_import import (
    DROPPED_CODES,
    IMPORT_CODES,
    NOT_CARRIED_CODES,
    ImportResult,
    TimelineImportError,
    import_timeline,
    window_reframe,
)

RATE_2997 = 30000 / 1001
TC_2997 = EXPORT_FIXTURES / "tc_2997.plan.json"


def simple_data() -> dict[str, Any]:
    """Four V1 segments from two camera files and a b-roll file nobody placed, with an outro."""
    return {
        "version": 2,
        "project": "P",
        "timeline": "T",
        "kind": "longform",
        "fps": 25,
        "resolution": {"width": 1920, "height": 1080},
        "profile": "ccn",
        "sources": [
            {"id": "camA", "media": "E:\\Show\\A.mov", "duration": 5000, "start_timecode": 0},
            {
                "id": "camB",
                "media": "E:\\Show\\B.mov",
                "duration": 5000,
                "start_timecode": 0,
                "frame_size": [1920, 1080],
            },
            {"id": "broll", "media": "E:\\Show\\Crowd.mov", "duration": 400, "start_timecode": 0},
            {"id": "out", "media": "E:\\Assets\\Outro.mov", "duration": 100, "start_timecode": 0},
        ],
        "segments": [
            {"source": "camA", "src_in": 0, "src_out": 200, "label": "open"},
            {
                "source": "camB",
                "src_in": 200,
                "src_out": 400,
                "reframe": [
                    {"time": 0, "x": 0.2, "y": 0.5, "zoom": 1.0},
                    {"time": 100, "x": 0.6, "y": 0.5, "zoom": 2.0},
                ],
            },
            {"source": "camA", "src_in": 500, "src_out": 700, "gain_db": -2.0},
            {"source": "camB", "src_in": 900, "src_out": 1100, "notes": "close"},
        ],
        "titles": [{"lines": ["Hello"], "record_in": 500, "record_out": 590}],
        "markers": [{"record": 550, "name": "late"}],
        "outro": {"source": "out", "audio_track": "A1"},
        "audio_tracks": [{"name": "A1"}],
    }


def simple() -> EditPlan:
    return EditPlan.model_validate(simple_data())


def run(
    plan: EditPlan, rows: list[Row], rate: float = 25.0, media_map: Any = None, **kw: Any
) -> ImportResult:
    timeline, media = build_timeline(rows, rate=rate, **kw)
    return import_timeline(plan, timeline, media, media_map=media_map, profile=permissive())


def layout_rows(plan: EditPlan, media_map: Any = None) -> list[Row]:
    return rows_from_layout(build_layout(plan, media_map=media_map, allow_unknown_timecode=True))


def statuses(result: ImportResult) -> list[str]:
    return [row["status"] for row in result.diff["segments"]]


def codes(result: ImportResult, key: str = "not_carried") -> list[str]:
    return [entry["code"] for entry in result.diff[key]]


def ranges(result: ImportResult) -> list[tuple[str, int, int, int]]:
    plan = result.plan
    return [
        (seg.source, seg.src_in, seg.src_out, start)
        for seg, (start, _) in zip(plan.segments, plan.record_ranges(), strict=True)
    ]


# -- identity ----------------------------------------------------------------------------------


@pytest.mark.parametrize("make", [simple, lambda: load_plan(TC_2997)], ids=["simple", "tc_2997"])
def test_unedited_timeline_gives_an_empty_diff_and_the_same_plan(make: Any) -> None:
    plan = make()
    rate = plan.fps.num / plan.fps.den
    result = run(plan, layout_rows(plan), rate=rate)
    diff = result.diff
    assert diff["unchanged"] is True
    assert set(statuses(result)) == {"unchanged"}
    assert diff["added"] == diff["foreign"] == diff["not_carried"] == diff["dropped"] == []
    assert result.plan.segments == plan.segments
    assert result.plan == plan
    assert diff["media_matched_by"] == ["path"]
    assert diff["lint_errors"] is False and diff["lint"]["ok"] is True
    assert [row["new"][0]["index"] for row in diff["segments"]] == list(range(len(plan.segments)))


def test_the_diff_says_what_the_reader_cannot_see_and_what_is_assumed() -> None:
    plan = simple()
    diff = run(plan, layout_rows(plan)).diff
    assert any(limit.startswith("enabled_state:") for limit in diff["limits"])
    assert [a["id"] for a in diff["assumptions"]] == ["tracks", "source_frames", "media_path"]
    assert list(diff) == [
        "format",
        "unchanged",
        "timeline",
        "primary_track",
        "media_matched_by",
        "segments",
        "added",
        "outro",
        "foreign",
        "not_carried",
        "dropped",
        "limits",
        "assumptions",
        "lint",
        "lint_errors",
    ]


# -- edits on V1 -------------------------------------------------------------------------------


def test_head_trim_trims_the_segment_and_moves_the_rest() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    trimmed = replace(c[0], src_in=31, duration=169)
    result = run(plan, ripple(rows, [trimmed, *c[1:]]))
    assert statuses(result) == ["trimmed", "moved", "moved", "moved"]
    first = result.diff["segments"][0]
    assert (first["head_frames"], first["tail_frames"]) == (31, 0)
    assert result.diff["segments"][1]["old"]["record_in"] == 200
    assert result.diff["segments"][1]["new"][0]["record_in"] == 169
    assert ranges(result) == [
        ("camA", 31, 200, 0),
        ("camB", 200, 400, 169),
        ("camA", 500, 700, 369),
        ("camB", 900, 1100, 569),
    ]
    assert result.plan.segments[0].label == "open"
    assert result.diff["outro"]["status"] == "moved"
    assert result.plan.outro == plan.outro
    assert result.diff["not_carried"] == []


def test_tail_trim() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[0], c[1], c[2], replace(c[3], duration=150)]))
    assert statuses(result) == ["unchanged", "unchanged", "unchanged", "trimmed"]
    last = result.diff["segments"][3]
    assert (last["head_frames"], last["tail_frames"]) == (0, 50)
    assert result.plan.segments[3].src_out == 1050
    assert result.plan.segments[3].notes == "close"


def test_extending_into_the_handle_is_a_negative_trim() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[0], c[1], replace(c[2], src_in=480, duration=220), c[3]]))
    row = result.diff["segments"][2]
    assert (row["status"], row["head_frames"], row["tail_frames"]) == ("trimmed", -20, 0)


def test_head_trim_windows_reframe_keys_to_the_new_start() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[0], replace(c[1], src_in=250, duration=150), *c[2:]]))
    keys = result.plan.segments[1].reframe
    # the old key at 0 is cut away: a key at the new start holds the interpolated framing
    assert [(k.time, round(k.x, 6), round(k.zoom, 6)) for k in keys] == [
        (0, 0.4, 1.5),
        (50, 0.6, 2.0),
    ]
    assert run(plan, ripple(rows, c)).plan.segments[1].reframe == plan.segments[1].reframe


def test_ripple_delete_removes_one_and_moves_the_rest() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[0], c[2], c[3]]))
    assert statuses(result) == ["unchanged", "removed", "moved", "moved"]
    assert result.diff["segments"][1]["new"] == []
    assert [s.source for s in result.plan.segments] == ["camA", "camA", "camB"]
    # the title (500..590) and the marker (550) are still inside the shorter timeline
    assert result.plan.total_frames() == 700
    assert result.diff["dropped"] == []


def test_entries_past_the_new_end_are_dropped_and_reported() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[0], c[1]]))
    assert codes(result, "dropped") == ["title_past_end", "marker_past_end"]
    assert result.plan.titles == [] and result.plan.markers == []
    assert result.diff["dropped"][0]["location"] == "titles[0]"


def test_reorder() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[2], c[0], c[1], c[3]]))
    assert statuses(result) == ["moved", "moved", "moved", "unchanged"]
    assert [s.source for s in result.plan.segments] == ["camA", "camA", "camB", "camB"]
    assert result.plan.segments[0].gain_db == -2.0
    assert result.diff["segments"][2]["new"][0] == {
        "index": 0,
        "src_in": 500,
        "src_out": 700,
        "record_in": 0,
        "record_out": 200,
    }


def test_split_gives_two_pieces_that_inherit_the_segment() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    head = replace(c[0], duration=80)
    tail = replace(c[0], src_in=120, duration=80)
    result = run(plan, ripple(rows, [head, tail, *c[1:]]))
    row = result.diff["segments"][0]
    assert row["status"] == "split"
    assert [(n["src_in"], n["src_out"], n["record_in"]) for n in row["new"]] == [
        (0, 80, 0),
        (120, 200, 80),
    ]
    assert [s.label for s in result.plan.segments[:2]] == ["open", "open"]
    assert statuses(result)[1:] == ["moved", "moved", "moved"]


def test_split_windows_reframe_keys_per_piece() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    head, tail = replace(c[1], duration=50), replace(c[1], src_in=250, duration=150)
    result = run(plan, ripple(rows, [c[0], head, tail, *c[2:]]))
    first, second = result.plan.segments[1].reframe, result.plan.segments[2].reframe
    assert [(k.time, round(k.x, 6)) for k in first] == [(0, 0.2), (49, 0.396)]
    assert [(k.time, round(k.x, 6)) for k in second] == [(0, 0.4), (50, 0.6)]


def test_an_added_range_from_a_known_source() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    extra = Row("V1", "video", 0, 60, 30, "E:\\Show\\Crowd.mov", sound="A1")
    result = run(plan, ripple(rows, [c[0], extra, *c[1:]]))
    assert statuses(result) == ["unchanged", "moved", "moved", "moved"]
    assert result.diff["added"] == [
        {
            "index": 1,
            "source": "broll",
            "match": "path",
            "src_in": 30,
            "src_out": 90,
            "record_in": 200,
            "record_out": 260,
        }
    ]
    added = result.plan.segments[1]
    assert (added.source, added.src_in, added.src_out, added.track) == ("broll", 30, 90, 1)
    assert added.label is None and added.reframe == []


def test_a_foreign_item_is_reported_never_guessed() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    stranger = Row("V1", "video", 0, 40, 0, "F:\\Elsewhere\\Other.mov", sound="A1")
    result = run(plan, ripple(rows, [c[0], stranger, *c[1:]]))
    assert [(f["track"], f["record_in"], f["media"]) for f in result.diff["foreign"]] == [
        ("V1", 200, "F:\\Elsewhere\\Other.mov")
    ]
    assert [s.source for s in result.plan.segments] == ["camA", "camB", "camA", "camB"]
    assert result.plan.segments[1].record_in == 240  # the gap the foreign item leaves
    assert result.diff["added"] == []
    assert result.diff["unchanged"] is False


def test_a_retimed_item_is_not_carried() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[0], replace(c[1], speed=2.0, sound=None), *c[2:]]))
    assert codes(result) == ["retimed"]
    assert result.diff["not_carried"][0]["record_in"] == 200
    assert statuses(result) == ["unchanged", "removed", "unchanged", "unchanged"]
    assert result.plan.segments[1].record_in == 400


def test_an_item_at_a_sub_frame_position_is_not_carried() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    moved = replace(c[3], fractions={"record_start": 0.5})
    result = run(plan, ripple(rows, [*c[:3], moved]))
    assert "sub_frame" in codes(result)
    assert statuses(result)[3] == "removed"


def test_a_changed_audio_item_is_an_audio_only_edit() -> None:
    plan = simple()
    rows = layout_rows(plan)
    # the third clip's sound slipped 10 frames against its picture
    edited = [replace(r, sound=None) if r is content(rows)[2] else r for r in rows]
    slipped = Row("A1", "audio", 400, 200, 510, "E:\\Show\\A.mov")
    result = run(plan, [*edited, slipped])
    assert codes(result) == ["audio_only_edit", "audio_only_edit"]
    assert [e["track"] for e in result.diff["not_carried"]] == ["A1", "V1"]
    assert set(statuses(result)) == {"unchanged"}
    assert result.plan == plan


def test_an_extra_sound_with_no_picture_is_an_audio_only_edit() -> None:
    plan = simple()
    music = Row("A2", "audio", 0, 300, 0, "E:\\Music\\bed.wav")
    result = run(plan, [*layout_rows(plan), music])
    assert codes(result) == ["audio_only_edit"]
    assert result.diff["not_carried"][0]["media"] == "E:\\Music\\bed.wav"


# -- plan-level carry on the CCN fixture -------------------------------------------------------


def tc_rows() -> tuple[EditPlan, list[Row]]:
    plan = load_plan(TC_2997)
    return plan, layout_rows(plan)


def keep_transitions(rows: list[Row], edited: list[Row]) -> list[Row]:
    return edited + [r for r in rows if r.kind == "transition"]


def test_a_transition_whose_join_is_gone_is_dropped() -> None:
    plan, rows = tc_rows()
    c = content(rows)
    edited = ripple(rows, [c[0], c[2]])
    result = run(plan, edited, rate=RATE_2997)
    assert codes(result, "dropped") == ["transition_join_gone"]
    assert result.plan.transitions == []
    # the head and outro dissolves are still in the plan; the timeline here has none
    assert codes(result) == ["transition_missing", "transition_missing"]


def test_a_surviving_transition_is_reindexed() -> None:
    plan, rows = tc_rows()
    c = content(rows)
    extra = Row("V1", "video", 0, 30, 2000, "E:/Show/Late/A.mov", sound="A1")
    result = run(plan, ripple(rows, [extra, *c]), rate=RATE_2997)
    assert [t.after for t in result.plan.transitions] == [2]
    assert plan.transitions[0].after == 1
    assert result.diff["dropped"] == []


def test_a_transition_added_in_resolve_is_not_carried() -> None:
    plan, rows = tc_rows()
    dissolve = Row("V1", "transition", 290, 20)
    result = run(plan, [*rows, dissolve], rate=RATE_2997)
    assert codes(result) == ["transition_added"]
    assert result.plan == plan


def test_a_change_on_another_picture_track_is_not_carried() -> None:
    plan, rows = tc_rows()
    edited = [replace(r, record_in=r.record_in + 30) if r.track == "V2" else r for r in rows]
    result = run(plan, edited, rate=RATE_2997)
    assert codes(result) == ["other_picture_track", "other_picture_track"]
    assert result.diff["segments"][3]["status"] == "kept_off_primary"
    assert result.plan == plan


def test_the_outro_trim_is_carried() -> None:
    plan, rows = tc_rows()
    edited = [replace(r, duration=120) if r.role == "outro" else r for r in rows]
    result = run(plan, edited, rate=RATE_2997)
    assert result.diff["outro"]["status"] == "trimmed"
    assert result.plan.outro is not None and result.plan.outro.src_out == 120
    assert result.plan.outro.sting == plan.outro.sting  # type: ignore[union-attr]


def test_an_outro_cut_shorter_than_its_sting_is_kept() -> None:
    plan, rows = tc_rows()
    edited = [replace(r, duration=40) if r.role == "outro" else r for r in rows]
    result = run(plan, edited, rate=RATE_2997)
    assert "outro_changed" in codes(result)
    assert result.plan.outro == plan.outro


# -- media matching ----------------------------------------------------------------------------


def test_a_hub_source_matches_through_the_media_map() -> None:
    data = simple_data()
    data["sources"][0]["media"] = "/srv/media/day1/A.mov"
    plan = EditPlan.model_validate(data)
    media_map = media_map_from_mapping(MEDIA_MAP)
    rows = layout_rows(plan, media_map)
    assert content(rows)[0].path == "E:\\Media\\day1\\A.mov"
    upper = [replace(r, path=r.path.upper()) if r.path else r for r in rows]
    result = run(plan, upper, media_map=media_map)
    assert result.diff["unchanged"] is True and result.diff["media_matched_by"] == ["path"]
    # without the map the same timeline still matches, by the unique file name
    fallback = run(plan, upper)
    assert fallback.diff["media_matched_by"] == ["file-name", "path"]
    assert set(statuses(fallback)) == {"unchanged"}


def test_a_file_name_shared_by_two_media_files_is_foreign() -> None:
    data = simple_data()
    data["sources"].append({"id": "dup", "media": "E:\\Other\\A.mov", "duration": 5000})
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    moved = [replace(r, path="G:\\Copy\\A.mov") if r.path == "E:\\Show\\A.mov" else r for r in rows]
    result = run(plan, moved)
    assert len(result.diff["foreign"]) == 2
    assert statuses(result) == ["removed", "unchanged", "removed", "unchanged"]


def test_two_sources_on_one_file_take_the_one_the_segment_used() -> None:
    data = simple_data()
    data["sources"].insert(0, {"id": "alias", "media": "E:\\Show\\A.mov", "duration": 5000})
    data["segments"][2]["source"] = "alias"
    plan = EditPlan.model_validate(data)
    result = run(plan, layout_rows(plan))
    assert result.diff["unchanged"] is True
    assert [s.source for s in result.plan.segments] == ["camA", "camB", "alias", "camB"]


# -- refusals ----------------------------------------------------------------------------------


def refusal(plan: EditPlan, rows: list[Row], **kw: Any) -> TimelineImportError:
    with pytest.raises(TimelineImportError) as caught:
        run(plan, rows, **kw)
    assert caught.value.fix_hint
    return caught.value


def test_a_timeline_at_another_frame_rate_is_refused() -> None:
    plan = simple()
    assert refusal(plan, layout_rows(plan), rate=24.0).code == "fps_mismatch"


def test_media_at_another_frame_rate_is_refused() -> None:
    plan = simple()
    error = refusal(plan, layout_rows(plan), media_rate=30.0)
    assert error.code == "media_fps_mismatch" and error.details["track"] == "V1"


def test_a_source_range_past_the_source_is_refused() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    error = refusal(plan, ripple(rows, [*c[:3], replace(c[3], src_in=4950, duration=100)]))
    assert error.code == "source_range_outside" and error.details["source"] == "camB"


def test_codes_are_stable() -> None:
    assert {
        "unreadable_timeline",
        "no_timeline",
        "several_timelines",
        "timeline_not_found",
        "timeline_too_large",
        "fps_mismatch",
        "media_fps_mismatch",
        "source_range_outside",
        "output_exists",
    } == IMPORT_CODES
    assert set(NOT_CARRIED_CODES).isdisjoint(DROPPED_CODES)


# -- reframe windowing -------------------------------------------------------------------------


def keys(*spec: tuple[int, float]) -> list[ReframeKey]:
    return [ReframeKey(time=t, x=x) for t, x in spec]


def test_window_reframe() -> None:
    k = keys((0, 0.0), (100, 1.0))
    assert window_reframe([], 10, 20) == []
    assert window_reframe(k, 0, 101) == k
    assert [(w.time, w.x) for w in window_reframe(k, 50, 101)] == [(0, 0.5), (50, 1.0)]
    assert [(w.time, w.x) for w in window_reframe(k, 0, 51)] == [(0, 0.0), (50, 0.5)]
    assert [(w.time, w.x) for w in window_reframe(k, 25, 76)] == [(0, 0.25), (50, 0.75)]
    # an extended head shifts the keys later; the framing before them holds the first key
    assert [(w.time, w.x) for w in window_reframe(k, -10, 101)] == [(10, 0.0), (110, 1.0)]
    single = keys((0, 0.3))
    assert [(w.time, w.x) for w in window_reframe(single, 40, 80)] == [(0, 0.3)]
    assert copy.deepcopy(k) == k
