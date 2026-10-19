"""Pure QA pieces: boundary evaluations, tokens and agreement, plan mapping, profile keys."""

from __future__ import annotations

import pytest

from aar.media import QuietRun
from aar.plan.inputs import Transcript, Word
from aar.plan.model import EditPlan, Fps
from aar.plan.profile import ProfileError, QATargets, load_profile, parse_profile
from aar.qa import (
    agreement,
    cuts,
    evaluate_duration,
    evaluate_loudness,
    evaluate_quiet_run,
    evaluate_silence,
    evaluate_size,
    judge_join_quiet,
    kept_words,
    resolve_media,
    straddled_words,
    suggest_src_in,
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
        {"join_quiet_slack_ms": 50},  # 60 + 50 ms does not fit in 0.1 s
        {"join_quiet_slack_ms": -1},
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


@pytest.mark.parametrize("seed", range(5))
def test_kept_words_match_a_brute_force_mapping(seed):
    import random

    rng = random.Random(seed)
    starts = sorted(rng.uniform(0, 30) for _ in range(300))
    words = [
        Word(text=f"w{i}", start_s=s, end_s=s + rng.uniform(0.0, 0.9)) for i, s in enumerate(starts)
    ]
    transcript = Transcript(words=words)
    segments = []
    for _ in range(12):
        src_in = rng.randrange(0, 1400)
        segments.append(seg("a", src_in, src_in + rng.randrange(1, 200)))
    plan = _plan(segments)
    ranges = plan.record_ranges()
    expected = []
    for i, s in enumerate(plan.segments):
        r0, r1 = s.src_in / 50, s.src_out / 50
        for index, w in enumerate(words):
            if r0 <= (w.start_s + w.end_s) / 2 < r1:
                expected.append((ranges[i][0] / 50 + w.start_s - r0, index))
    expected.sort()
    got = kept_words(plan, transcript)
    assert [k.index for k in got] == [index for _, index in expected]
    assert [k.record_start_s for k in got] == pytest.approx([t for t, _ in expected])


# ---------------------------------------------------------------- round 2


def _run(start: float, end: float) -> QuietRun:
    return QuietRun(start_s=start, end_s=end, level_dbfs=-60.0)


@pytest.mark.parametrize(
    ("runs", "status", "at_cut", "nearest"),
    [
        ([(0.95, 1.05)], "pass", 0, None),  # contains src_in 1.0
        ([(0.93, 0.99)], "pass", 0, None),  # ends 10 ms before: inside the slack
        ([(0.92, 0.9899)], "fail", None, 0),  # ends 10.1 ms before
        ([(1.01, 1.07)], "pass", 0, None),  # starts 10 ms after
        ([(1.0101, 1.08)], "fail", None, 0),
        ([(0.98, 1.039)], "fail", 0, None),  # contains src_in but is 59 ms
        ([(0.98, 1.04)], "pass", 0, None),  # exactly 60 ms
        ([(0.80, 0.92), (1.06, 1.13)], "fail", None, 1),  # the nearer one (60 ms away)
        ([], "fail", None, None),
    ],
)
def test_judge_join_quiet(runs, status, at_cut, nearest):
    quiet = [_run(*r) for r in runs]
    judged = judge_join_quiet(quiet, 1.0, min_ms=60, slack_ms=10)
    assert judged.status == status
    assert judged.at_cut is (None if at_cut is None else quiet[at_cut])
    assert judged.nearest is (None if nearest is None else quiet[nearest])


def test_judge_join_quiet_prefers_the_longer_of_two_equally_near_pauses():
    a, b = _run(0.85, 0.95), _run(1.05, 1.20)
    assert judge_join_quiet([a, b], 1.0, min_ms=60, slack_ms=10).nearest is b
    assert judge_join_quiet([b, a], 1.0, min_ms=60, slack_ms=10).nearest is b


@pytest.mark.parametrize(
    ("fps", "frame"), [("24", 25), ("25", 26), ("30000/1001", 31), ("48", 50), ("60", 63)]
)
def test_suggest_src_in_is_the_frame_nearest_the_middle(fps, frame):
    plan_fps = Fps.model_validate(fps)
    assert suggest_src_in(_run(1.0, 1.1), plan_fps) == frame
    assert abs(plan_fps.seconds(frame) - 1.05) < plan_fps.seconds(1) / 2


@pytest.mark.parametrize(
    "toml",
    [
        "lufs_tolerance = inf",
        "short_lufs = nan",
        "longform_lufs = -inf",
        "review_mp4_max_mb = inf",
        "true_peak_max_dbtp = nan",
        "join_agreement_min = nan",
        "longest_silence_warn_s = inf",
    ],
)
def test_non_finite_qa_values_are_refused(tmp_path, toml):
    key = toml.split()[0]
    base = {
        "longform_lufs": "-22.0",
        "short_lufs": "-14.5",
        "lufs_tolerance": "0.5",
        "review_mp4_max_mb": "20",
    }
    base.pop(key, None)
    lines = ['name = "x"', "[qa]", *(f"{k} = {v}" for k, v in base.items()), toml]
    path = tmp_path / "x.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ProfileError, match=key):
        load_profile(path)


def test_non_finite_qa_values_are_refused_in_code():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QATargets(
            longform_lufs=-22, short_lufs=float("nan"), lufs_tolerance=0.5, review_mp4_max_mb=20
        )


def test_slack_and_minimum_must_fit_the_search_window():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="join_quiet_search_s must cover"):
        QATargets(
            longform_lufs=-22,
            short_lufs=-14.5,
            lufs_tolerance=0.5,
            review_mp4_max_mb=20,
            join_quiet_search_s=0.065,
        )
    QATargets(
        longform_lufs=-22,
        short_lufs=-14.5,
        lufs_tolerance=0.5,
        review_mp4_max_mb=20,
        join_quiet_search_s=0.07,
    )


def test_unchecked_lists_skipped_and_incomplete_checks():
    checks = [
        QACheck(id="a", status="pass", message="m"),
        QACheck(id="b", status="skipped", message="m", skip_reason="no-plan"),
        QACheck(id="c", status="warn", message="m", incomplete=True),
        QACheck(id="d", status="fail", message="m", incomplete=True),
        QACheck(id="e", status="warn", message="m"),
    ]
    report = build_report(file="f", kind="short", profile="p", media=None, checks=checks)
    assert report.unchecked == ["b", "c", "d"]
