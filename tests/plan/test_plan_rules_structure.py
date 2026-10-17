"""ranges-valid, chronological and short-max-duration."""

from __future__ import annotations

from collections.abc import Callable

from plan_support import PlanData, only, refs

from aar.plan.lint import LintReport

RunLint = Callable[..., LintReport]


def test_clean_plan_passes_every_rule(clean_data: PlanData, run_lint: RunLint, transcript, laughs):
    report = run_lint(clean_data, transcript=transcript, laughs=laughs)
    assert report.findings == []
    assert report.ok
    assert report.counts == {"error": 0, "warn": 0, "info": 0}
    assert (report.duration_frames, report.duration_s) == (720, 30.03)


# ranges-valid


def test_ranges_valid_passes_for_the_clean_plan(clean_data: PlanData, run_lint: RunLint):
    assert only(run_lint(clean_data), "ranges-valid") == []


def test_ranges_valid_flags_an_empty_range(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["src_out"] = 1500  # equal to src_in: [in, out) is empty
    (finding,) = only(run_lint(clean_data), "ranges-valid")
    assert finding.severity == "error"
    assert "src_out 1500 <= src_in 1500" in finding.message
    assert refs(finding) == [
        (1, "in", "00:00:10:00", "00:01:02:12"),
        (1, "out", "00:00:10:00", "00:01:02:12"),
    ]


def test_ranges_valid_flags_a_range_past_the_end_of_the_media(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["sources"][2]["duration"] = 1700  # cam-c; segment 1 ends at 1740
    (finding,) = only(run_lint(clean_data), "ranges-valid")
    assert finding.severity == "error"
    assert "past the end of source 'cam-c' (1700 frames)" in finding.message
    assert finding.fix_hint == "Trim src_out to at most 1700."
    assert refs(finding) == [(1, "out", "00:00:20:00", "00:01:12:12")]


def test_ranges_valid_accepts_a_range_that_ends_exactly_at_the_media_end(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["sources"][2]["duration"] = 1740
    assert only(run_lint(clean_data), "ranges-valid") == []


def test_ranges_valid_flags_overlap_on_one_track(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["record_in"] = 200  # segment 0 plays until frame 240
    (finding,) = only(run_lint(clean_data), "ranges-valid")
    assert finding.severity == "error"
    assert "Segment 1 starts at record frame 200, before segment 0 ends at 240" in finding.message
    assert refs(finding) == [
        (0, "out", "00:00:10:00", "00:00:51:16"),
        (1, "in", "00:00:08:08", "00:01:02:12"),
    ]


def test_ranges_valid_allows_overlap_on_different_tracks(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["track"] = 2
    clean_data["segments"][1]["record_in"] = 0
    assert only(run_lint(clean_data), "ranges-valid") == []


def test_ranges_valid_flags_an_unknown_source_without_crashing_other_rules(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["segments"][2]["source"] = "cam-z"
    report = run_lint(clean_data)
    (finding,) = only(report, "ranges-valid")
    assert finding.severity == "error"
    assert "unknown source 'cam-z'" in finding.message
    assert refs(finding) == [(2, None, None, None)]
    assert not report.ok


def test_ranges_valid_flags_duplicate_source_ids(clean_data: PlanData, run_lint: RunLint):
    clean_data["sources"].append(dict(clean_data["sources"][0]))
    (finding,) = only(run_lint(clean_data), "ranges-valid")
    assert finding.message == "Source id 'cam-a' is defined more than once."


def test_ranges_valid_flags_a_keyframe_after_the_segment_ends(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["segments"][1]["reframe"] = [{"time": 0}, {"time": 241}]
    (finding,) = only(run_lint(clean_data), "ranges-valid")
    assert "keyframe at frame 241" in finding.message
    assert refs(finding) == [(1, None, None, None)]


def test_ranges_valid_flags_a_transition_with_nothing_after_it(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["transitions"].append({"after": 2, "video": "cut"})
    (finding,) = only(run_lint(clean_data), "ranges-valid")
    assert finding.severity == "error"
    assert "Transition 1 follows segment 2" in finding.message


def test_ranges_valid_flags_an_empty_caption_range(clean_data: PlanData, run_lint: RunLint):
    clean_data["captions"][1]["record_out"] = 600
    (finding,) = only(run_lint(clean_data), "ranges-valid")
    assert finding.message.startswith("Caption 1 ends at 600, not after its start 600")
    assert finding.refs[0].item == "caption[1]"
    assert finding.refs[0].record_tc == "00:00:25:00"


# chronological


def test_chronological_passes_for_the_clean_plan(clean_data: PlanData, run_lint: RunLint):
    assert only(run_lint(clean_data), "chronological") == []


def test_chronological_flags_going_back_in_time_using_recording_time(
    clean_data: PlanData, run_lint: RunLint
):
    # cam-c starts 48 frames into the recording, so source 800 is recording frame 848,
    # earlier than segment 0 (recording frame 1000).
    clean_data["segments"][1]["src_in"] = 800
    clean_data["segments"][1]["src_out"] = 1040
    (finding,) = only(run_lint(clean_data), "chronological")
    assert finding.severity == "error"
    assert "Segment 1 goes back in time" in finding.message
    assert "00:00:35:08" in finding.message  # recording frame 848
    assert refs(finding) == [
        (0, "out", "00:00:10:00", "00:00:51:16"),
        (1, "in", "00:00:10:00", "00:00:33:08"),
    ]


def test_chronological_flags_repeated_material(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["src_in"] = 1000  # recording frame 1048, inside segment 0
    clean_data["segments"][1]["src_out"] = 1240
    (finding,) = only(run_lint(clean_data), "chronological")
    assert finding.severity == "error"
    assert "repeats material already played" in finding.message


def test_chronological_ignores_other_recordings(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["src_in"] = 800
    clean_data["segments"][1]["src_out"] = 1040
    clean_data["sources"][2]["recording_id"] = "take-2"
    assert only(run_lint(clean_data), "chronological") == []


def test_chronological_treats_a_source_without_recording_id_as_its_own_recording(
    clean_data: PlanData, run_lint: RunLint
):
    for source in clean_data["sources"]:
        source["recording_id"] = None
    clean_data["segments"][1].update(source="cam-a", src_in=500, src_out=740)
    (finding,) = only(run_lint(clean_data), "chronological")
    assert "recording 'source:cam-a'" in finding.message
    assert [r.segment for r in finding.refs] == [0, 1]


def test_chronological_allows_a_gap_forward(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][2]["src_in"] = 9000  # skips ahead of segment 1, never back
    clean_data["segments"][2]["src_out"] = 9240
    assert only(run_lint(clean_data), "chronological") == []


# short-max-duration


def _stretch_last_segment(data: PlanData, total_frames: int) -> None:
    data["segments"][2]["src_out"] = data["segments"][2]["src_in"] + (total_frames - 480)


def test_short_max_duration_accepts_the_last_frame_under_the_limit(
    clean_data: PlanData, run_lint: RunLint
):
    _stretch_last_segment(clean_data, 887)  # 36.9 s at 23.976 fps
    assert only(run_lint(clean_data), "short-max-duration") == []


def test_short_max_duration_flags_a_short_that_reaches_the_limit(
    clean_data: PlanData, run_lint: RunLint
):
    _stretch_last_segment(clean_data, 888)  # 37.04 s
    (finding,) = only(run_lint(clean_data), "short-max-duration")
    assert finding.severity == "error"
    assert finding.message == "The short runs 37.037 s (888 frames); it must be under 37 s."
    assert finding.fix_hint == "Cut at least 1 frames so the total is at most 887 frames."
    assert [(r.item, r.record_frame, r.record_tc) for r in finding.refs] == [
        ("end", 888, "00:00:37:00")
    ]


def test_short_max_duration_does_not_apply_to_longform(clean_data: PlanData, run_lint: RunLint):
    _stretch_last_segment(clean_data, 5000)
    clean_data["kind"] = "longform"
    assert only(run_lint(clean_data), "short-max-duration") == []
