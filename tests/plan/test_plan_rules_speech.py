"""no-dialogue-splice and end-on-laugh, which need a transcript or a laugh map."""

from __future__ import annotations

from collections.abc import Callable

from plan_support import PlanData, only, refs

from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import LintReport

RunLint = Callable[..., LintReport]


def words(*items: tuple[str, float, float], recording: str | None = None) -> Transcript:
    return Transcript.model_validate(
        {
            "recording": recording,
            "words": [{"text": t, "start_s": s, "end_s": e} for t, s, e in items],
        }
    )


def laugh_map(*items: tuple[float, float, float], recording: str | None = None) -> LaughMap:
    return LaughMap.model_validate(
        {
            "recording": recording,
            "laughs": [{"start_s": s, "end_s": e, "strength": k} for s, e, k in items],
        }
    )


# no-dialogue-splice


def test_dialogue_splice_passes_when_every_cut_sits_in_a_pause(
    clean_data: PlanData, run_lint: RunLint, transcript: Transcript
):
    assert only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice") == []


def test_dialogue_splice_reports_a_skip_without_a_transcript(
    clean_data: PlanData, run_lint: RunLint
):
    (finding,) = only(run_lint(clean_data), "no-dialogue-splice")
    assert finding.severity == "info"
    assert finding.message == "Skipped: this rule needs a transcript, but none was supplied."
    assert finding.refs == []


def test_dialogue_splice_errors_when_a_cut_lands_inside_a_word(
    clean_data: PlanData, run_lint: RunLint, transcript: Transcript
):
    # Source frame 988 of cam-a is 41.21 s, inside "anyway" (40.95-41.45 s).
    clean_data["segments"][0]["src_in"] = 988
    (finding,) = only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice")
    assert finding.severity == "error"
    assert 'falls inside the word "anyway" (40.95-41.45 s)' in finding.message
    assert "source frame 982 (before the word) or 994 (after it)" in finding.fix_hint
    assert refs(finding) == [(0, "in", "00:00:00:00", "00:00:41:04")]


def test_dialogue_splice_warns_when_the_pause_is_too_short(clean_data: PlanData, run_lint: RunLint):
    # Segment 0 ends at 51.72 s. The words around it leave a 100 ms gap; a 1 s pause follows.
    transcript = words(("one", 51.20, 51.66), ("two", 51.76, 52.00), ("three", 53.00, 53.30))
    (finding,) = only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice")
    assert finding.severity == "warn"
    assert 'sits in a 100 ms gap between "one" and "two"; at least 250 ms is wanted' in (
        finding.message
    )
    assert "The nearest is 52.00-53.00 s" in finding.fix_hint
    assert refs(finding) == [(0, "out", "00:00:10:00", "00:00:51:16")]


def test_dialogue_splice_accepts_a_cut_exactly_at_a_word_edge(
    clean_data: PlanData, run_lint: RunLint
):
    # The word ends where the cut is; the next word starts a full second later.
    edge = 1240 * 1001 / 24000
    transcript = words(("end", edge - 0.4, edge), ("next", edge + 1.0, edge + 1.3))
    assert only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice") == []


def test_dialogue_splice_does_not_check_an_unbroken_join(clean_data: PlanData, run_lint: RunLint):
    # Segments 1 and 2 join at recording 74.5 s, inside this word, but the audio plays on.
    transcript = words(("extraordinary", 74.0, 75.0))
    assert only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice") == []


def test_dialogue_splice_checks_the_join_once_it_becomes_a_cut(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["segments"][2].update(src_in=1769, src_out=2009)  # skip 5 frames ahead
    transcript = words(("extraordinary", 74.0, 75.0))
    findings = only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice")
    assert [(f.severity, refs(f)) for f in findings] == [
        ("error", [(1, "out", "00:00:20:00", "00:01:12:12")]),
        ("error", [(2, "in", "00:00:20:00", "00:01:13:17")]),
    ]


def test_dialogue_splice_reports_cuts_in_recordings_the_transcript_does_not_cover(
    clean_data: PlanData, run_lint: RunLint
):
    transcript = words(("hello", 1.0, 1.5), recording="take-2")
    (finding,) = only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice")
    assert finding.severity == "info"
    assert "4 cut point(s) in other recordings were not checked" in finding.message
    assert [(r.segment, r.point) for r in finding.refs] == [
        (0, "in"),
        (0, "out"),
        (1, "in"),
        (2, "out"),
    ]


def test_dialogue_splice_uses_recording_time_for_sources_with_an_origin(
    clean_data: PlanData, run_lint: RunLint
):
    # cam-c frame 0 is recording frame 48, so segment 1 starts at recording frame 1548 (64.56 s).
    # Without the origin the cut would look like 62.56 s, a word of its own.
    transcript = words(("before", 61.5, 62.9), ("inside", 64.0, 65.0), ("after", 66.5, 67.0))
    (finding,) = only(run_lint(clean_data, transcript=transcript), "no-dialogue-splice")
    assert finding.severity == "error"
    assert 'inside the word "inside"' in finding.message
    assert refs(finding) == [(1, "in", "00:00:10:00", "00:01:02:12")]


# end-on-laugh


def test_end_on_laugh_passes_when_the_edit_ends_just_after_a_laugh(
    clean_data: PlanData, run_lint: RunLint, laughs: LaughMap
):
    # The final out point is recording frame 2028 = 84.58 s; the laugh ends at 84.50 s.
    assert only(run_lint(clean_data, laughs=laughs), "end-on-laugh") == []


def test_end_on_laugh_reports_a_skip_without_a_laugh_map(clean_data: PlanData, run_lint: RunLint):
    (finding,) = only(run_lint(clean_data), "end-on-laugh")
    assert finding.severity == "info"
    assert finding.message == "Skipped: this rule needs a laugh map, but none was supplied."


def test_end_on_laugh_warns_when_the_edit_runs_on_after_the_laugh(
    clean_data: PlanData, run_lint: RunLint
):
    (finding,) = only(run_lint(clean_data, laughs=laugh_map((82.0, 83.0845, 0.9))), "end-on-laugh")
    assert finding.severity == "warn"
    assert "ends 1500 ms after the nearest laugh ended (83.08 s)" in finding.message
    assert "within 500 ms after a laugh" in finding.message
    assert "source frame 1968" in finding.fix_hint
    assert refs(finding) == [(2, "out", "00:00:30:00", "00:01:23:12")]


def test_end_on_laugh_warns_when_the_edit_cuts_the_laugh_off(
    clean_data: PlanData, run_lint: RunLint
):
    (finding,) = only(run_lint(clean_data, laughs=laugh_map((83.9, 85.1845, 0.9))), "end-on-laugh")
    assert finding.severity == "warn"
    assert "ends 600 ms before the nearest laugh ends (85.18 s)" in finding.message


def test_end_on_laugh_judges_the_last_segment_by_record_position(
    clean_data: PlanData, run_lint: RunLint
):
    # Segment 2 plays last; a laugh right after segment 0's out point must not matter.
    report = run_lint(clean_data, laughs=laugh_map((50.0, 51.7, 1.0)))
    (finding,) = only(report, "end-on-laugh")
    assert finding.refs[0].segment == 2


def test_end_on_laugh_ignores_laughs_below_the_profile_strength(
    clean_data: PlanData, ccn, run_lint: RunLint
):
    ccn.rules["end-on-laugh"].params["min_strength"] = 0.8
    report = run_lint(clean_data, laughs=laugh_map((83.9, 84.5, 0.5)), profile=ccn)
    (finding,) = only(report, "end-on-laugh")
    assert finding.severity == "warn"
    assert finding.message == "The laugh map has no laugh to end on."


def test_end_on_laugh_reports_a_laugh_map_for_another_recording(
    clean_data: PlanData, run_lint: RunLint
):
    laughs = laugh_map((83.9, 84.5, 0.9), recording="take-2")
    (finding,) = only(run_lint(clean_data, laughs=laughs), "end-on-laugh")
    assert finding.severity == "info"
    assert "the ending was not checked" in finding.message
    assert refs(finding) == [(2, "out", "00:00:30:00", "00:01:23:12")]
