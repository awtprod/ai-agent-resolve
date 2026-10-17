"""no-same-camera-jump, timejump-close-to-wide, no-dissolve, audio-crossfade, no-whip-pan."""

from __future__ import annotations

from collections.abc import Callable

from plan_support import PlanData, only, refs

from aar.plan.lint import LintReport

RunLint = Callable[..., LintReport]

JOIN_0_1 = [(0, "out", "00:00:10:00", "00:00:51:16"), (1, "in", "00:00:10:00", "00:01:02:12")]


# no-same-camera-jump


def test_same_camera_jump_passes_when_the_cameras_differ(clean_data: PlanData, run_lint: RunLint):
    assert only(run_lint(clean_data), "no-same-camera-jump") == []


def test_same_camera_jump_flags_a_time_jump_on_one_camera(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1].update(source="cam-a", src_in=1500, src_out=1740)
    (finding,) = only(run_lint(clean_data), "no-same-camera-jump")
    assert finding.severity == "error"
    assert "same camera ('A')" in finding.message
    assert "zoom changes by only 0.05 (1.05 to 1)" in finding.message
    assert refs(finding) == JOIN_0_1


def test_same_camera_jump_allows_a_big_enough_punch_in(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1].update(source="cam-a", src_in=1500, src_out=1740)
    clean_data["segments"][1]["reframe"] = [{"time": 0, "zoom": 1.3}]  # 1.05 -> 1.3
    assert only(run_lint(clean_data), "no-same-camera-jump") == []


def test_same_camera_jump_flags_a_punch_in_that_is_too_small(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["segments"][1].update(source="cam-a", src_in=1500, src_out=1740)
    clean_data["segments"][1]["reframe"] = [{"time": 0, "zoom": 1.15}]  # 1.05 -> 1.15
    (finding,) = only(run_lint(clean_data), "no-same-camera-jump")
    assert "only 0.10" in finding.message


def test_same_camera_without_a_time_jump_is_not_a_jump_cut(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1].update(source="cam-a", src_in=1240, src_out=1480)
    assert only(run_lint(clean_data), "no-same-camera-jump") == []


# timejump-close-to-wide


def test_close_to_wide_passes_for_the_clean_plan(clean_data: PlanData, run_lint: RunLint):
    assert only(run_lint(clean_data), "timejump-close-to-wide") == []


def test_close_to_wide_warns_when_the_jump_goes_wide_to_close(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["sources"][0]["shot_size"] = "wide"  # cam-a, before the jump
    clean_data["sources"][2]["shot_size"] = "close"  # cam-c, after the jump
    (finding,) = only(run_lint(clean_data), "timejump-close-to-wide")
    assert finding.severity == "warn"
    assert "goes wide to close" in finding.message
    assert refs(finding) == JOIN_0_1


def test_close_to_wide_accepts_medium_to_wide(clean_data: PlanData, run_lint: RunLint):
    clean_data["sources"][0]["shot_size"] = "medium"
    assert only(run_lint(clean_data), "timejump-close-to-wide") == []


def test_close_to_wide_reports_unknown_shot_sizes_instead_of_passing(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["sources"][2]["shot_size"] = "unknown"
    (finding,) = only(run_lint(clean_data), "timejump-close-to-wide")
    assert finding.severity == "info"
    assert finding.message.startswith("Could not check 1 time jump(s)")
    assert refs(finding) == [(1, "in", "00:00:10:00", "00:01:02:12")]


def test_close_to_wide_ignores_joins_with_no_time_jump(clean_data: PlanData, run_lint: RunLint):
    # Segments 1 and 2 are continuous in recording time, so only the 0-1 jump is a candidate.
    clean_data["sources"][1]["shot_size"] = "close"  # cam-b after a wide cam-c
    assert only(run_lint(clean_data), "timejump-close-to-wide") == []


# no-dissolve


def test_no_dissolve_passes_for_hard_cuts(clean_data: PlanData, run_lint: RunLint):
    assert only(run_lint(clean_data), "no-dissolve") == []


def test_no_dissolve_flags_any_video_transition_that_is_not_a_cut(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["transitions"][0].update(video="dissolve", video_frames=12)
    (finding,) = only(run_lint(clean_data), "no-dissolve")
    assert finding.severity == "error"
    assert "'dissolve' (12 frames)" in finding.message
    assert refs(finding) == [(0, "out", "00:00:10:00", "00:00:51:16")]


# audio-crossfade


def test_audio_crossfade_passes_at_the_frame_limit(clean_data: PlanData, run_lint: RunLint):
    clean_data["transitions"][0]["audio"]["frames"] = 30
    assert only(run_lint(clean_data), "audio-crossfade") == []


def test_audio_crossfade_flags_a_wrong_type(clean_data: PlanData, run_lint: RunLint):
    clean_data["transitions"][0]["audio"]["type"] = "Constant Power"
    (finding,) = only(run_lint(clean_data), "audio-crossfade")
    assert finding.severity == "error"
    assert finding.message == (
        "Audio transition after segment 0 is 'Constant Power', not 'Cross Fade 0 dB'."
    )
    assert refs(finding) == JOIN_0_1


def test_audio_crossfade_flags_one_frame_too_long(clean_data: PlanData, run_lint: RunLint):
    clean_data["transitions"][0]["audio"]["frames"] = 31
    (finding,) = only(run_lint(clean_data), "audio-crossfade")
    assert finding.severity == "error"
    assert "is 31 frames; the maximum is 30" in finding.message
    assert refs(finding) == JOIN_0_1


def test_audio_crossfade_flags_a_join_with_no_audio_transition(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["transitions"] = []
    (finding,) = only(run_lint(clean_data), "audio-crossfade")
    assert finding.severity == "error"
    assert "between segments 0 and 1 has no audio transition" in finding.message
    assert refs(finding) == JOIN_0_1


def test_audio_crossfade_does_not_ask_for_a_fade_on_an_unbroken_join(
    clean_data: PlanData, run_lint: RunLint
):
    # Segments 1 and 2 play on without a gap in recording time and have no transition.
    assert clean_data["transitions"][0]["after"] == 0
    assert only(run_lint(clean_data), "audio-crossfade") == []


def test_audio_crossfade_flags_too_little_media_before_the_incoming_cut(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["segments"][1].update(src_in=4, src_out=244)  # a 12-frame fade needs 6
    # Keep segment 2 continuous with segment 1 (cam-b starts 24 frames into the recording
    # and cam-c 48), so the only audio edit is the 0-1 join.
    clean_data["segments"][2].update(src_in=268, src_out=508)
    (finding,) = only(run_lint(clean_data), "audio-crossfade")
    assert finding.severity == "error"
    assert "only 4 frames of media before its in point" in finding.message
    assert "needs 6" in finding.message
    assert refs(finding) == [
        (0, "out", "00:00:10:00", "00:00:51:16"),
        (1, "in", "00:00:10:00", "00:00:00:04"),
    ]


def test_audio_crossfade_flags_too_little_media_after_the_outgoing_cut(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["sources"][0]["duration"] = 1243  # cam-a, segment 0 ends at 1240
    (finding,) = only(run_lint(clean_data), "audio-crossfade")
    assert finding.severity == "error"
    assert "only 3 frames of media after its out point" in finding.message
    assert refs(finding) == JOIN_0_1


def test_audio_crossfade_accepts_exactly_enough_handle(clean_data: PlanData, run_lint: RunLint):
    clean_data["sources"][0]["duration"] = 1246  # 6 frames after the cut
    assert only(run_lint(clean_data), "audio-crossfade") == []


def test_audio_crossfade_says_so_when_the_media_duration_is_unknown(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["sources"][0]["duration"] = None
    (finding,) = only(run_lint(clean_data), "audio-crossfade")
    assert finding.severity == "info"
    assert finding.message.startswith("Skipped the outgoing handle check on 1 join(s)")
    assert refs(finding) == [(0, "out", "00:00:10:00", "00:00:51:16")]


# no-whip-pan


def test_whip_pan_passes_for_slow_reframing(clean_data: PlanData, run_lint: RunLint):
    # Segment 0 of the clean plan drifts 0.02 widths and 0.05 zoom over ten seconds.
    assert clean_data["segments"][0]["reframe"]
    assert only(run_lint(clean_data), "no-whip-pan") == []


def test_whip_pan_flags_a_fast_pan(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["reframe"] = [{"time": 0, "x": 0.2}, {"time": 12, "x": 0.8}]
    (finding,) = only(run_lint(clean_data), "no-whip-pan")
    assert finding.severity == "error"
    assert "0.600 frame widths in 0.50 s (1.20/s); the limit is 0.3/s" in finding.message
    assert refs(finding) == [
        (1, None, "00:00:10:00", "00:01:02:12"),
        (1, None, "00:00:10:12", "00:01:03:00"),
    ]


def test_whip_pan_measures_vertical_movement_too(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["reframe"] = [{"time": 0, "y": 0.1}, {"time": 24, "y": 0.9}]
    (finding,) = only(run_lint(clean_data), "no-whip-pan")
    assert "0.800 frame widths in 1.00 s (0.80/s)" in finding.message


def test_whip_pan_flags_a_fast_zoom(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["reframe"] = [{"time": 0, "zoom": 1.0}, {"time": 24, "zoom": 1.5}]
    (finding,) = only(run_lint(clean_data), "no-whip-pan")
    assert finding.severity == "error"
    assert "zoom changes 0.500 in 1.00 s (0.50/s); the limit is 0.25/s" in finding.message


def test_whip_pan_flags_an_instant_jump_between_two_keyframes(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["segments"][1]["reframe"] = [{"time": 5, "x": 0.2}, {"time": 5, "x": 0.8}]
    (finding,) = only(run_lint(clean_data), "no-whip-pan")
    assert "in 0.00 s (inf/s)" in finding.message


def test_whip_pan_accepts_two_identical_keyframes(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1]["reframe"] = [{"time": 5}, {"time": 5}]
    assert only(run_lint(clean_data), "no-whip-pan") == []
