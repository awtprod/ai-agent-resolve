"""S13a rules: end-on-laugh at the onset, dissolve exemptions for the head and the outro,
cut density, shortest shot, a short's de-click end fade, and what the outro means to the
content rules."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest
from plan_support import PlanData, only, refs

from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import LintReport, all_rules
from aar.plan.profile import Profile

RunLint = Callable[..., LintReport]

# At 25 fps the clean short's final out point (cam-b frame 2004, origin 24) is recording frame
# 2028 = 81.12 s exactly, and half a frame is 20 ms, so window edges can be hit exactly.
AT = 81.12


def at_25(data: PlanData) -> PlanData:
    data["fps"] = 25
    return data


def v2(data: PlanData) -> PlanData:
    """The clean short as a version 2 plan with a 302-frame outro source."""
    data["version"] = 2
    data["sources"].append({"id": "outro", "media": "media/outro.mov", "duration": 302})
    return data


def laugh_map(*items: tuple[float, float], recording: str | None = None) -> LaughMap:
    return LaughMap.model_validate(
        {"recording": recording, "laughs": [{"start_s": s, "end_s": e} for s, e in items]}
    )


def timeline(data: PlanData, lengths: Iterable[int], *, continuous: Iterable[int] = ()) -> PlanData:
    """Track-1 segments of the given lengths at 25 fps, all from camera A.

    Every join is a cut (a 100-frame jump in the recording) unless the incoming segment's
    index is in ``continuous``, in which case it resumes exactly where the previous one ended.
    """
    data["fps"] = 25
    keep = set(continuous)
    segments, src = [], 1000
    for index, length in enumerate(lengths):
        if index and index not in keep:
            src += 100
        segments.append({"source": "cam-a", "src_in": src, "src_out": src + length})
        src += length
    data["segments"] = segments
    data["transitions"] = []
    return data


# end-on-laugh: the onset anchor (ccn: onset, 0-400 ms)


@pytest.mark.parametrize(
    "onset",
    [
        AT,  # exactly 0 ms after the onset
        AT - 0.4,  # exactly 400 ms after
        AT - 0.42,  # 400 ms plus half a frame
        AT + 0.02,  # half a frame before the onset
    ],
    ids=["0ms", "400ms", "400ms+half-frame", "-half-frame"],
)
def test_end_on_laugh_onset_window_edges_pass(
    clean_data: PlanData, run_lint: RunLint, onset: float
):
    onset = round(onset, 3)
    laughs = laugh_map((onset, onset + 1.5))
    assert only(run_lint(at_25(clean_data), laughs=laughs), "end-on-laugh") == []


@pytest.mark.parametrize(
    ("onset", "how", "target"),
    [
        (80.68, "440 ms after the nearest laugh began (80.68 s)", 1993),  # one frame late
        (81.16, "40 ms before the nearest laugh begins (81.16 s)", 2005),  # one frame early
        (80.699, "421 ms after the nearest laugh began (80.70 s)", 1993),  # past the slack
        (81.141, "21 ms before the nearest laugh begins (81.14 s)", 2005),  # past the slack
    ],
    ids=["frame-late", "frame-early", "beyond-half-frame-late", "beyond-half-frame-early"],
)
def test_end_on_laugh_onset_window_flags_an_out_point_just_outside(
    clean_data: PlanData, run_lint: RunLint, onset: float, how: str, target: int
):
    laughs = laugh_map((onset, onset + 1.5))
    (finding,) = only(run_lint(at_25(clean_data), laughs=laughs), "end-on-laugh")
    assert finding.severity == "warn"
    assert finding.message == (
        f"The edit ends {how}; it should end within 400 ms after a laugh begins."
    )
    # The suggested frame is the onset itself, in the final segment's source frames.
    assert finding.fix_hint == (
        f"Set the final src_out to source frame {target} (the onset of that laugh)."
    )
    assert refs(finding) == [(2, "out", "00:00:28:20", "00:01:20:04")]


def test_end_on_laugh_accepts_a_laugh_shorter_than_the_window(
    clean_data: PlanData, run_lint: RunLint, ccn_v1: Profile
):
    # A 100 ms laugh that is already over: the out point is 220 ms after its onset.
    short = laugh_map((80.9, 81.0))
    assert only(run_lint(at_25(clean_data), laughs=short), "end-on-laugh") == []
    # Out 520 ms after a short laugh's onset is late, although only 420 ms after its end,
    # which the old end-anchored window (0-500 ms after the end) accepted.
    late = laugh_map((80.6, 80.7))
    (finding,) = only(run_lint(clean_data, laughs=late), "end-on-laugh")
    assert "520 ms after the nearest laugh began (80.60 s)" in finding.message
    assert only(run_lint(clean_data, laughs=late, profile=ccn_v1), "end-on-laugh") == []


def test_end_on_laugh_onset_anchor_does_not_wait_for_a_long_laugh(
    clean_data: PlanData, run_lint: RunLint, ccn_v1: Profile
):
    # Cut 220 ms into a 9 s laugh: right for ccn; the old end anchor wanted the laugh to end.
    laughs = laugh_map((80.9, 90.0))
    at_25(clean_data)
    assert only(run_lint(clean_data, laughs=laughs), "end-on-laugh") == []
    (finding,) = only(run_lint(clean_data, laughs=laughs, profile=ccn_v1), "end-on-laugh")
    assert "ends 8880 ms before the nearest laugh ends (90.00 s)" in finding.message


def test_end_on_laugh_window_can_start_after_the_anchor(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
):
    ccn.rules["end-on-laugh"].params.update(min_after_ms=100)
    at_25(clean_data)
    assert (
        only(run_lint(clean_data, laughs=laugh_map((81.02, 83.0)), profile=ccn), "end-on-laugh")
        == []
    )
    (finding,) = only(
        run_lint(clean_data, laughs=laugh_map((81.06, 83.0)), profile=ccn), "end-on-laugh"
    )
    assert finding.message == (
        "The edit ends 60 ms after the nearest laugh began (81.06 s); it should end between "
        "100 and 400 ms after a laugh begins."
    )
    assert finding.fix_hint == (
        "Set the final src_out to source frame 2005 (100 ms after the onset of that laugh)."
    )


def test_end_on_laugh_end_anchor_with_a_window(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
):
    ccn.rules["end-on-laugh"].params.update(anchor="end", min_after_ms=100, max_after_ms=500)
    at_25(clean_data)
    (finding,) = only(
        run_lint(clean_data, laughs=laugh_map((79.0, 81.05)), profile=ccn), "end-on-laugh"
    )
    assert finding.message == (
        "The edit ends 70 ms after the nearest laugh ended (81.05 s); it should end between "
        "100 and 500 ms after a laugh ends."
    )
    assert finding.fix_hint == (
        "Set the final src_out to source frame 2005 (100 ms after the end of that laugh)."
    )
    assert (
        only(run_lint(clean_data, laughs=laugh_map((79.0, 81.0)), profile=ccn), "end-on-laugh")
        == []
    )


def test_end_on_laugh_without_the_new_keys_is_the_old_end_window(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
):
    # Only min_strength set: the code defaults are anchor end, 0-500 ms, as before S13a.
    ccn.rules["end-on-laugh"].params = {"min_strength": 0.0}
    at_25(clean_data)
    assert (
        only(run_lint(clean_data, laughs=laugh_map((79.0, 80.62)), profile=ccn), "end-on-laugh")
        == []
    )
    (finding,) = only(
        run_lint(clean_data, laughs=laugh_map((79.0, 80.58)), profile=ccn), "end-on-laugh"
    )
    assert finding.message == (
        "The edit ends 540 ms after the nearest laugh ended (80.58 s); it should end within "
        "500 ms after a laugh."
    )
    assert finding.fix_hint == "Set the final src_out to source frame 1990 (the end of that laugh)."


def test_end_on_laugh_names_the_laugh_nearest_by_its_anchor(
    clean_data: PlanData, run_lint: RunLint
):
    laughs = laugh_map((78.0, 81.1), (81.6, 82.0))  # onsets 3.12 s before and 480 ms after
    (finding,) = only(run_lint(at_25(clean_data), laughs=laughs), "end-on-laugh")
    assert "480 ms before the nearest laugh begins (81.60 s)" in finding.message


# no-dissolve: the head and the outro (ccn: allowed, at most 30 frames)


def test_no_dissolve_allows_the_15_frame_start_and_outro_dissolves(
    clean_data: PlanData, run_lint: RunLint
):
    v2(clean_data)
    clean_data["head"] = {"video": "dissolve", "video_frames": 15, "audio_fade_frames": 15}
    clean_data["outro"] = {
        "source": "outro",
        "transition": {"video": "dissolve", "video_frames": 15},
    }
    assert only(run_lint(clean_data), "no-dissolve") == []


@pytest.mark.parametrize(("frames", "count"), [(30, 0), (31, 1)])
def test_no_dissolve_caps_the_head_transition(
    clean_data: PlanData, run_lint: RunLint, frames: int, count: int
):
    v2(clean_data)["head"] = {"video": "dissolve", "video_frames": frames}
    findings = only(run_lint(clean_data), "no-dissolve")
    assert len(findings) == count
    if count:
        (finding,) = findings
        assert finding.severity == "error"
        assert finding.message == (
            "The head transition (in from black) is a 31-frame dissolve; the maximum is 30 frames."
        )
        assert finding.fix_hint == "Shorten head.video_frames to 30 or fewer."
        (ref,) = finding.refs
        assert (ref.item, ref.record_frame, ref.record_tc) == ("head", 0, "00:00:00:00")


@pytest.mark.parametrize(("frames", "count"), [(30, 0), (31, 1)])
def test_no_dissolve_caps_the_outro_transition(
    clean_data: PlanData, run_lint: RunLint, frames: int, count: int
):
    v2(clean_data)["outro"] = {
        "source": "outro",
        "transition": {"video": "dip", "video_frames": frames},
    }
    findings = only(run_lint(clean_data), "no-dissolve")
    assert len(findings) == count
    if count:
        (finding,) = findings
        assert finding.message == (
            "The transition into the outro is a 31-frame dip; the maximum is 30 frames."
        )
        assert finding.fix_hint == "Shorten outro.transition.video_frames to 30 or fewer."
        (ref,) = finding.refs
        assert (ref.item, ref.record_frame, ref.record_tc) == ("outro", 720, "00:00:30:00")


def test_no_dissolve_still_errors_between_content_segments(clean_data: PlanData, run_lint: RunLint):
    v2(clean_data)
    clean_data["head"] = {"video": "dissolve", "video_frames": 15}
    clean_data["outro"] = {
        "source": "outro",
        "transition": {"video": "dissolve", "video_frames": 15},
    }
    clean_data["transitions"][0].update(video="dissolve", video_frames=15)
    (finding,) = only(run_lint(clean_data), "no-dissolve")
    assert finding.message.startswith("Video transition after segment 0 is 'dissolve'")


def test_no_dissolve_errors_on_head_and_outro_when_the_profile_does_not_allow_them(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile, ccn_v1: Profile
):
    v2(clean_data)
    clean_data["head"] = {"video": "dissolve", "video_frames": 15}
    clean_data["outro"] = {
        "source": "outro",
        "transition": {"video": "dissolve", "video_frames": 15},
    }
    ccn.rules["no-dissolve"].params.update(allow_head=False, allow_outro=False)
    for profile in (ccn, ccn_v1):  # ccn_v1 sets no keys: the code default allows only cuts
        head, outro = only(run_lint(clean_data, profile=profile), "no-dissolve")
        assert head.message == (
            "The head transition (in from black) is 'dissolve' (15 frames); this profile allows "
            "only hard cuts."
        )
        assert head.fix_hint == (
            'Set head.video to "cut" with video_frames 0, or allow it in the profile '
            "(allow_head = true)."
        )
        assert outro.message.startswith("The transition into the outro is 'dissolve'")
        assert "(allow_outro = true)" in outro.fix_hint


def test_no_dissolve_ignores_a_hard_start_with_an_audio_fade(
    clean_data: PlanData, run_lint: RunLint, ccn_v1: Profile
):
    v2(clean_data)["head"] = {"video": "cut", "audio_fade_frames": 15}
    clean_data["outro"] = {"source": "outro"}
    assert only(run_lint(clean_data, profile=ccn_v1), "no-dissolve") == []


# cut-density (ccn: more than 2 cuts in any 10 s window warns)


def test_cut_density_allows_exactly_two_cuts_in_a_window(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
):
    timeline(clean_data, [250, 100, 250])  # cuts at 10 s and 14 s
    assert only(run_lint(clean_data), "cut-density") == []
    # The same two cuts are one too many where the profile allows only one.
    ccn.rules["cut-density"].params["max_cuts"] = 1
    (finding,) = only(run_lint(clean_data, profile=ccn), "cut-density")
    assert finding.message.startswith("2 cuts in 4.00 s of record time")


def test_cut_density_warns_on_three_cuts_in_a_window(clean_data: PlanData, run_lint: RunLint):
    report = run_lint(timeline(clean_data, [250, 100, 100, 250]))  # cuts at 10, 14 and 18 s
    (finding,) = only(report, "cut-density")
    assert finding.severity == "warn"
    assert finding.message == (
        "3 cuts in 8.00 s of record time (00:00:10:00 to 00:00:18:00); at most 2 are allowed "
        "in any 10 s."
    )
    assert "drop or merge cuts" in finding.fix_hint
    assert [(r.segment, r.point, r.record_tc) for r in finding.refs] == [
        (1, "in", "00:00:10:00"),
        (2, "in", "00:00:14:00"),
        (3, "in", "00:00:18:00"),
    ]


def test_cut_density_window_is_half_open(clean_data: PlanData, run_lint: RunLint, ccn: Profile):
    # Cuts at frames 100, 225 and 350: exactly 10 s from first to last is not one window...
    timeline(clean_data, [100, 125, 125, 100])
    assert only(run_lint(clean_data), "cut-density") == []
    # ...but a window one frame longer holds all three.
    ccn.rules["cut-density"].params["window_s"] = 10.04
    (finding,) = only(run_lint(clean_data, profile=ccn), "cut-density")
    assert finding.message.startswith("3 cuts in 10.00 s of record time")


def test_cut_density_one_frame_inside_the_window_warns(clean_data: PlanData, run_lint: RunLint):
    # Cuts at frames 100, 225 and 349: 9.96 s from first to last.
    (finding,) = only(run_lint(timeline(clean_data, [100, 125, 124, 100])), "cut-density")
    assert finding.message.startswith("3 cuts in 9.96 s of record time")


def test_cut_density_merges_overlapping_windows_into_one_finding(
    clean_data: PlanData, run_lint: RunLint
):
    report = run_lint(timeline(clean_data, [25] * 7))  # six cuts one second apart
    (finding,) = only(report, "cut-density")
    assert finding.message.startswith(
        "6 cuts in 5.00 s of record time (00:00:01:00 to 00:00:06:00)"
    )
    assert len(finding.refs) == 6


def test_cut_density_reports_bursts_whose_windows_only_touch_separately(
    clean_data: PlanData, run_lint: RunLint
):
    # Bursts at 4, 6, 8 s and 16, 18, 20 s: the cuts at 6, 8 and 16 s span exactly 10 s, so
    # no window holds cuts of both bursts.
    findings = only(run_lint(timeline(clean_data, [100, 50, 50, 200, 50, 50, 100])), "cut-density")
    assert [f.message.split(";")[0] for f in findings] == [
        "3 cuts in 4.00 s of record time (00:00:04:00 to 00:00:08:00)",
        "3 cuts in 4.00 s of record time (00:00:16:00 to 00:00:20:00)",
    ]


def test_cut_density_merges_bursts_that_share_a_window(clean_data: PlanData, run_lint: RunLint):
    # One frame closer: the cuts at 6 s, 8 s and 15.96 s now fit one window, so it is one burst.
    findings = only(run_lint(timeline(clean_data, [100, 50, 50, 199, 50, 50, 100])), "cut-density")
    (finding,) = findings
    assert finding.message.startswith(
        "6 cuts in 15.96 s of record time (00:00:04:00 to 00:00:19:24)"
    )


def test_cut_density_merges_bursts_that_share_only_one_cut(clean_data: PlanData, run_lint: RunLint):
    # Cuts at 1, 2, 10, 12 and 19 s: 1-10 s and 10-19 s are both bursts (9 s each) and the
    # cuts at 2, 10 and 12 s are not (exactly 10 s), but the two bursts share the cut at 10 s.
    report = run_lint(timeline(clean_data, [25, 25, 200, 50, 175, 100]))
    (finding,) = only(report, "cut-density")
    assert finding.message.startswith(
        "5 cuts in 18.00 s of record time (00:00:01:00 to 00:00:19:00)"
    )


def test_cut_density_does_not_count_unbroken_joins(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
):
    # The join into segment 2 resumes the recording, so only two cuts remain.
    timeline(clean_data, [250, 100, 100, 250], continuous=[2])
    assert only(run_lint(clean_data), "cut-density") == []
    ccn.rules["cut-density"].params["max_cuts"] = 1
    (finding,) = only(run_lint(clean_data, profile=ccn), "cut-density")
    assert [(r.segment, r.record_tc) for r in finding.refs] == [
        (1, "00:00:10:00"),
        (3, "00:00:18:00"),
    ]


def test_cut_density_threshold_comes_from_the_profile(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
):
    ccn.rules["cut-density"].params["max_cuts"] = 3
    assert (
        only(run_lint(timeline(clean_data, [250, 100, 100, 250]), profile=ccn), "cut-density") == []
    )


# min-shot (ccn: under 1.0 s warns)


def test_min_shot_exactly_the_minimum_passes(clean_data: PlanData, run_lint: RunLint, ccn: Profile):
    timeline(clean_data, [250, 25, 250])  # exactly 1.00 s
    assert only(run_lint(clean_data), "min-shot") == []
    ccn.rules["min-shot"].params["min_seconds"] = 1.04  # one frame more is too short
    (finding,) = only(run_lint(clean_data, profile=ccn), "min-shot")
    assert finding.message.startswith("Segment 1 runs 1.000 s (25 frames)")


def test_min_shot_one_frame_under_the_minimum_warns(clean_data: PlanData, run_lint: RunLint):
    (finding,) = only(run_lint(timeline(clean_data, [250, 24, 250])), "min-shot")
    assert finding.severity == "warn"
    assert finding.message == "Segment 1 runs 0.960 s (24 frames); shots under 1 s read as slivers."
    assert finding.fix_hint == (
        "Fold it into a neighbour: extend segment 0 or segment 2 over its 24 frames, or "
        "lengthen it to at least 25 frames."
    )
    assert [(r.segment, r.point) for r in finding.refs] == [(1, "in"), (1, "out")]


def test_min_shot_at_a_fractional_rate(clean_data: PlanData, run_lint: RunLint):
    # At 24000/1001 one second is 23.976 frames: 24 frames pass, 23 do not.
    clean_data["segments"][1]["src_out"] = 1500 + 24
    assert only(run_lint(clean_data), "min-shot") == []
    clean_data["segments"][1]["src_out"] = 1500 + 23
    (finding,) = only(run_lint(clean_data), "min-shot")
    assert "0.959 s (23 frames)" in finding.message
    assert "at least 24 frames" in finding.fix_hint


def test_min_shot_names_the_only_neighbour_or_none(clean_data: PlanData, run_lint: RunLint):
    (first,) = only(run_lint(timeline(clean_data, [10, 250])), "min-shot")
    assert first.fix_hint.startswith("Fold it into a neighbour: extend segment 1 over its 10")
    (alone,) = only(run_lint(timeline(clean_data, [10])), "min-shot")
    assert alone.fix_hint == (
        "It has no neighbour on its track to fold into: lengthen it to at least 25 frames."
    )


def test_min_shot_skips_segments_that_ranges_valid_reports(clean_data: PlanData, run_lint: RunLint):
    clean_data["segments"][1].update(source="nope", src_out=1510)  # unknown source
    clean_data["segments"][2].update(src_out=1774)  # a usable 10-frame sliver
    (finding,) = only(run_lint(clean_data), "min-shot")
    assert finding.refs[0].segment == 2


# short-end-fade (ccn: a short's end fade is at most 1 frame)


def test_short_end_fade_summary_says_de_click():
    assert "de-click" in all_rules()["short-end-fade"].summary


@pytest.mark.parametrize(("frames", "count"), [(None, 0), (0, 0), (1, 0), (2, 1)])
def test_short_end_fade_allows_only_a_de_click(
    clean_data: PlanData, run_lint: RunLint, frames: int | None, count: int
):
    v2(clean_data)["end_fade_frames"] = frames
    findings = only(run_lint(clean_data), "short-end-fade")
    assert len(findings) == count
    if count:
        (finding,) = findings
        assert finding.severity == "warn"
        assert finding.message == (
            "The short's final audio fade is 2 frames (0.08 s); a short ends hard on the laugh, "
            "so the fade is only a de-click of at most 1 frame(s), never a fade to silence."
        )
        assert finding.fix_hint == "Set end_fade_frames to 1 or less."
        (ref,) = finding.refs
        assert (ref.item, ref.record_frame) == ("end", 720)


def test_short_end_fade_does_not_limit_a_longform(clean_data: PlanData, run_lint: RunLint):
    v2(clean_data).update(kind="longform", end_fade_frames=15)
    assert only(run_lint(clean_data), "short-end-fade") == []


# no-same-camera-jump: the static transform's zoom counts


def test_jump_cut_counts_the_static_transform_zoom(clean_data: PlanData, run_lint: RunLint):
    v2(clean_data)["segments"][1].update(source="cam-a", src_in=1500, src_out=1740)
    clean_data["segments"][1]["transform"] = {"zoom": 1.3}  # 1.05 -> 1.3: a real punch-in
    assert only(run_lint(clean_data), "no-same-camera-jump") == []
    clean_data["segments"][1]["transform"] = {"zoom": 1.1}  # 1.05 -> 1.1: too small
    (finding,) = only(run_lint(clean_data), "no-same-camera-jump")
    assert "zoom changes by only 0.05 (1.05 to 1.1)" in finding.message
    # The outgoing side's transform multiplies its last reframe key: 1.05 * 1.2 = 1.26.
    clean_data["segments"][1]["transform"] = None
    clean_data["segments"][0]["transform"] = {"zoom": 1.2}
    assert only(run_lint(clean_data), "no-same-camera-jump") == []


# the outro is not content


def test_content_rules_ignore_the_outro(
    clean_data: PlanData, run_lint: RunLint, transcript: Transcript, laughs: LaughMap
):
    # The outro plays camera A from frame 0, earlier in the recording than any segment, and
    # its range sits inside a word; none of that is judged, and the ending is still the laugh.
    v2(clean_data)["outro"] = {"source": "cam-a", "src_in": 0, "src_out": 48}
    words = Transcript.model_validate(
        {"words": [*transcript.model_dump()["words"], {"text": "x", "start_s": 0, "end_s": 2}]}
    )
    report = run_lint(clean_data, transcript=words, laughs=laughs)
    for rule in ("chronological", "no-dialogue-splice", "end-on-laugh", "cut-density"):
        assert only(report, rule) == [], rule
    assert (report.duration_frames, report.duration_s) == (768, 32.032)


@pytest.mark.parametrize(("outro_frames", "count"), [(167, 0), (168, 1)])
def test_short_max_duration_counts_the_outro(
    clean_data: PlanData, run_lint: RunLint, outro_frames: int, count: int
):
    # 720 content frames plus the outro: 887 frames is 36.995 s, 888 frames is 37.037 s.
    v2(clean_data)["outro"] = {"source": "outro", "src_out": outro_frames}
    findings = only(run_lint(clean_data), "short-max-duration")
    assert len(findings) == count
