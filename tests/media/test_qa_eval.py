"""Pure QA pieces: boundary evaluations, tokens and agreement, plan mapping, profile keys."""

from __future__ import annotations

import pytest

from aar.plan.inputs import Transcript, Word
from aar.plan.model import EditPlan
from aar.plan.profile import QATargets, load_profile, parse_profile
from aar.qa import (
    agreement,
    cuts,
    evaluate_duration,
    evaluate_loudness,
    evaluate_quiet_run,
    evaluate_silence,
    evaluate_size,
    kept_words,
    resolve_media,
    straddled_words,
    tokens,
)
from aar.qa.report import QACheck, build_report

SHORT = {"target_lufs": -14.5, "tolerance_lu": 0.5, "true_peak_max_dbtp": -1.0}
LONG = {"target_lufs": -22.0, "tolerance_lu": 0.5, "true_peak_max_dbtp": -1.0}


@pytest.mark.parametrize(
    ("integrated", "status"),
    [(-14.5, "pass"), (-14.0, "pass"), (-15.0, "pass"), (-13.9, "fail"), (-15.1, "fail")],
)
def test_short_loudness_tolerance_is_inclusive(integrated, status):
    assert evaluate_loudness(integrated, -3.0, **SHORT)[0] == status


@pytest.mark.parametrize(
    ("integrated", "status"),
    [(-22.0, "pass"), (-21.5, "pass"), (-22.5, "pass"), (-21.4, "fail"), (-22.6, "fail")],
)
def test_longform_loudness_tolerance_is_inclusive(integrated, status):
    assert evaluate_loudness(integrated, -3.0, **LONG)[0] == status


@pytest.mark.parametrize(("peak", "status"), [(-1.0, "pass"), (-1.1, "pass"), (-0.9, "fail")])
def test_true_peak_at_most_minus_one(peak, status):
    assert evaluate_loudness(-14.5, peak, **SHORT)[0] == status


def test_loudness_messages_say_what_to_do():
    status, message = evaluate_loudness(-16.0, -0.5, **SHORT)
    assert status == "fail"
    assert "raise the gain by 1.5 dB" in message
    assert "lower the limiter ceiling or the gain by 0.5 dB" in message
    assert "\n" not in message
    assert "lower the gain by 1.0 dB" in evaluate_loudness(-13.5, -2.0, **SHORT)[1]


@pytest.mark.parametrize("integrated", [-70.0, float("-inf"), float("nan")])
def test_no_gated_loudness_fails(integrated):
    status, message = evaluate_loudness(integrated, float("-inf"), **SHORT)
    assert status == "fail" and "silent" in message


def test_missing_true_peak_fails():
    assert evaluate_loudness(-14.5, None, **SHORT)[0] == "fail"


def test_silence_warn_is_off_by_default_and_strict():
    assert evaluate_silence(60.0, None) == "pass"
    assert evaluate_silence(2.0, 2.0) == "pass"
    assert evaluate_silence(2.001, 2.0) == "warn"


def test_size_cap_counts_decimal_megabytes():
    assert evaluate_size(20_000_000, 20) == "pass"
    assert evaluate_size(20_000_001, 20) == "fail"


def test_duration_within_one_frame_inclusive():
    assert evaluate_duration(1.0, 1) == "pass"
    assert evaluate_duration(-1.0, 1) == "pass"
    assert evaluate_duration(1.0004, 1) == "pass"  # ffprobe prints microseconds
    assert evaluate_duration(1.01, 1) == "fail"
    assert evaluate_duration(0.0, 0) == "pass"


def test_quiet_run_boundary():
    assert evaluate_quiet_run(60.0, 60) == "pass"
    assert evaluate_quiet_run(59.0, 60) == "fail"
    assert evaluate_quiet_run(59.99, 60) == "fail"


def test_tokens_and_agreement():
    assert tokens(["Hello,", "WORLD!", "don't", "'quoted'", "a_b", "--"]) == [
        "hello",
        "world",
        "don't",
        "quoted",
        "a",
        "b",
    ]
    assert agreement([], []) == 1.0
    assert agreement(["a"], []) == 0.0
    assert agreement(list("abcd"), list("abce")) == pytest.approx(0.75)


def _plan(segments, sources=None, captions=None, fps="50"):
    return EditPlan.model_validate(
        {
            "version": 1,
            "project": "P",
            "timeline": "T",
            "kind": "short",
            "fps": fps,
            "resolution": {"width": 160, "height": 90},
            "profile": "ccn",
            "sources": sources
            or [
                {"id": "a", "media": "a.wav", "recording_id": "rec"},
                {"id": "b", "media": "b.wav", "recording_id": "rec", "recording_origin": 50},
                {"id": "c", "media": "c.wav"},
            ],
            "segments": segments,
            "captions": captions or [],
        }
    )


def seg(source, src_in, src_out, **extra):
    return {"track": 1, "source": source, "src_in": src_in, "src_out": src_out, **extra}


def test_cuts_skip_continuous_joins_and_unusable_segments():
    plan = _plan(
        [
            seg("a", 0, 100),
            # b is 50 frames later on the recording clock: b's 50 is a's 100 -> continuous.
            seg("b", 50, 150),
            seg("a", 300, 400),  # a time jump in the same recording: a cut
            seg("c", 10, 60),  # another recording: a cut
            seg("c", 60, 60),  # zero length: not usable, so neither of its joins is checked
            seg("c", 61, 100),
        ]
    )
    found = cuts(plan, tolerance_frames=1)
    assert [(c.outgoing, c.incoming) for c in found] == [(1, 2), (2, 3)]
    first = found[0]
    assert first.number == 1
    assert first.record_frame == 200 and first.record_s == pytest.approx(4.0)
    assert first.src_in_frame == 300 and first.src_in_s == pytest.approx(6.0)
    assert first.source_id == "a" and first.out_source_id == "b" and first.src_out_frame == 150
    # The a -> b continuation is exact, so tolerance 0 still treats it as continuous.
    assert len(cuts(plan, tolerance_frames=0)) == 2
    loose = _plan([seg("a", 0, 100), seg("a", 102, 200)])
    assert len(cuts(loose, tolerance_frames=1)) == 1
    assert cuts(loose, tolerance_frames=2) == []


def test_kept_words_map_through_the_plan_by_midpoint():
    plan = _plan([seg("a", 50, 100), seg("a", 200, 250)])
    words = Transcript(
        words=[
            Word(text="before", start_s=0.5, end_s=0.9),
            Word(text="edge", start_s=0.95, end_s=1.1),  # midpoint 1.025 -> kept
            Word(text="one", start_s=1.2, end_s=1.4),
            Word(text="tail", start_s=1.9, end_s=2.2),  # midpoint 2.05 -> dropped
            Word(text="two", start_s=4.1, end_s=4.3),
        ]
    )
    kept = kept_words(plan, words)
    assert [w.text for w in kept] == ["edge", "one", "two"]
    assert kept[0].record_start_s == pytest.approx(-0.05)
    assert kept[1].record_start_s == pytest.approx(0.2)
    assert kept[2].record_start_s == pytest.approx(1.0 + 0.1)
    assert kept[2].segment == 1


def test_kept_words_respect_the_transcript_recording():
    plan = _plan([seg("a", 0, 100), seg("c", 0, 100)])
    words = [Word(text="x", start_s=0.5, end_s=0.6)]
    assert [w.segment for w in kept_words(plan, Transcript(words=words))] == [0, 1]
    assert [w.segment for w in kept_words(plan, Transcript(recording="rec", words=words))] == [0]
    assert [w.segment for w in kept_words(plan, Transcript(recording="c", words=words))] == [1]


def test_straddled_words_use_the_tolerance():
    plan = _plan([seg("a", 0, 100), seg("a", 200, 300)])
    cut = cuts(plan, 1)[0]
    transcript = Transcript(
        words=[
            Word(text="clean", start_s=1.5, end_s=1.97),  # ends 0.03 before the out point
            Word(text="sliced", start_s=1.9, end_s=2.2),  # crosses 2.0 by 0.1 / 0.2
            Word(text="edge", start_s=3.97, end_s=4.2),  # starts 0.03 before src_in 4.0
            Word(text="cut", start_s=3.9, end_s=4.1),
        ]
    )
    found = straddled_words(plan, transcript, cut, tolerance_s=0.03)
    assert [(s.side, s.text) for s in found] == [("out", "sliced"), ("in", "cut")]
    assert found[1].boundary_s == pytest.approx(4.0)


def test_resolve_media(tmp_path):
    assert resolve_media("clip.wav", tmp_path) == tmp_path / "clip.wav"
    assert resolve_media("/abs/clip.wav", tmp_path).as_posix() == "/abs/clip.wav"
    assert resolve_media("https://example.invalid/x.mov", tmp_path) is None
    assert resolve_media("", tmp_path) is None


def test_report_status_is_the_worst_check_and_skips_do_not_pass():
    def entry(status):
        return QACheck(id="x", status=status, message="m")

    assert (
        build_report(
            file="f", kind="short", profile="p", media=None, checks=[entry("pass"), entry("warn")]
        ).status
        == "warn"
    )
    failed = build_report(
        file="f", kind="short", profile="p", media=None, checks=[entry("skipped"), entry("fail")]
    )
    assert failed.status == "fail" and not failed.ok
    assert failed.counts == {"skipped": 1, "pass": 0, "warn": 0, "fail": 1}


def test_a_profile_without_the_new_qa_keys_still_loads():
    old = parse_profile(
        {
            "name": "old",
            "qa": {
                "longform_lufs": -16,
                "short_lufs": -14,
                "lufs_tolerance": 1,
                "review_mp4_max_mb": 50,
            },
            "rules": {},
        }
    )
    assert old.qa.true_peak_max_dbtp == -1.0
    assert old.qa.silence_db == -42.0
    assert old.qa.longest_silence_warn_s is None
    assert old.qa.join_quiet_min_ms == 60
    assert old.qa == QATargets(
        longform_lufs=-16, short_lufs=-14, lufs_tolerance=1, review_mp4_max_mb=50
    )


def test_ccn_qa_block_matches_the_defaults():
    ccn = load_profile("ccn").qa
    defaults = QATargets(
        longform_lufs=-22.0, short_lufs=-14.5, lufs_tolerance=0.5, review_mp4_max_mb=20
    )
    assert ccn == defaults


@pytest.mark.parametrize(
    "bad",
    [
        {"join_windows": 4},
        {"join_windows": 1},
        {"join_window_step_s": 2.0},  # 3 windows of 4 s stepped by 2 s do not all overlap
        {"join_window_edge_s": 1.6},  # the edges eat the whole overlap
        {"join_quiet_window_ms": 80},
        {"join_quiet_min_ms": 250},  # cannot fit in +-0.1 s
        {"silence_db": 0},
        {"longest_silence_warn_s": 0},
        {"join_agreement_min": 1.5},
        {"unknown_key": 1},
    ],
)
def test_bad_qa_values_are_rejected(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QATargets(
            longform_lufs=-22, short_lufs=-14.5, lufs_tolerance=0.5, review_mp4_max_mb=20, **bad
        )
