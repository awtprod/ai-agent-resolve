"""Rule profiles: loading, validation against the registered rules, and overrides."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from plan_support import PlanData, only

from aar.plan.lint import LintReport, all_rules, lint
from aar.plan.model import EditPlan
from aar.plan.profile import Profile, ProfileError, load_profile, parse_profile

RunLint = Callable[..., LintReport]

CCN_RULES = [
    "chronological",
    "no-dialogue-splice",
    "no-same-camera-jump",
    "timejump-close-to-wide",
    "no-dissolve",
    "audio-crossfade",
    "short-max-duration",
    "end-on-laugh",
    "cut-density",
    "min-shot",
    "short-end-fade",
    "no-whip-pan",
    "caption-length",
    "caption-style",
    "ranges-valid",
]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "show.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_ccn_profile_carries_the_qa_targets(ccn: Profile):
    assert ccn.name == "ccn"
    qa = ccn.qa.model_dump()
    assert {key: qa[key] for key in ("longform_lufs", "short_lufs")} == {
        "longform_lufs": -22.0,
        "short_lufs": -14.5,
    }
    assert qa["lufs_tolerance"] == 0.5
    assert qa["review_mp4_max_mb"] == 20
    assert qa["true_peak_max_dbtp"] == -1.0
    assert qa["silence_db"] == -42.0
    assert qa["longest_silence_warn_s"] is None
    assert qa["join_quiet_db"] == -40.0
    assert qa["join_quiet_min_ms"] == 60


def test_ccn_profile_holds_the_thresholds(ccn: Profile):
    params = {rule_id: config.params for rule_id, config in ccn.rules.items()}
    assert params["no-dialogue-splice"]["min_pause_ms"] == 250
    assert params["no-same-camera-jump"]["min_zoom_delta"] == 0.2
    assert params["audio-crossfade"]["audio_type"] == "Cross Fade 0 dB"
    assert params["audio-crossfade"]["max_frames"] == 30
    assert params["short-max-duration"]["max_seconds"] == 37
    assert params["end-on-laugh"] == {
        "anchor": "onset",
        "min_after_ms": 0,
        "max_after_ms": 400,
        "min_strength": 0.0,
    }
    assert params["no-dissolve"] == {
        "allow_head": True,
        "max_head_frames": 30,
        "allow_outro": True,
        "max_outro_frames": 30,
    }
    assert params["cut-density"] == {
        "window_s": 10,
        "max_cuts": 2,
        "continuity_tolerance_frames": 1,
    }
    assert params["min-shot"] == {"min_seconds": 1.0}
    assert params["short-end-fade"] == {"max_frames": 1}
    assert params["no-whip-pan"] == {"max_pan_speed": 0.3, "max_zoom_speed": 0.25}
    assert params["caption-length"]["max_chars"] == 16
    assert params["caption-style"]["font"] == "Montserrat"


def test_ccn_profile_and_the_registry_list_the_same_rules(ccn: Profile):
    assert sorted(ccn.rules) == sorted(CCN_RULES)
    assert sorted(all_rules()) == sorted(CCN_RULES)


def test_an_unknown_builtin_profile_is_an_error():
    with pytest.raises(ProfileError, match="no built-in rule profile named 'nope'"):
        load_profile("nope")


def test_a_profile_path_that_does_not_exist_is_an_error(tmp_path: Path):
    with pytest.raises(ProfileError, match="profile file not found"):
        load_profile(tmp_path / "missing.toml")


def test_invalid_toml_is_an_error(tmp_path: Path):
    with pytest.raises(ProfileError, match="invalid TOML"):
        load_profile(write(tmp_path, "name = "))


def test_a_profile_without_qa_targets_is_an_error(tmp_path: Path):
    with pytest.raises(ProfileError, match="qa"):
        load_profile(write(tmp_path, 'name = "x"\n[rules.no-dissolve]\n'))


def test_another_show_can_ship_its_own_profile(
    tmp_path: Path, clean_data: PlanData, run_lint: RunLint
):
    path = write(
        tmp_path,
        """
name = "other"
[qa]
longform_lufs = -16.0
short_lufs = -14.0
lufs_tolerance = 1.0
review_mp4_max_mb = 50
[rules.caption-length]
max_chars = 10
""",
    )
    other = load_profile(path)
    assert other.name == "other"
    report = run_lint(clean_data, profile=other)
    assert {f.rule for f in report.findings} == {"caption-length"}
    assert [f.message for f in report.findings] == [
        "Caption 0 has a 14-character line 'I PARKED THERE'; the limit is 10.",
        "Caption 0 has a 14-character line 'FOR ONE MINUTE'; the limit is 10.",
        "Caption 1 has a 11-character line 'WAIT FOR IT'; the limit is 10.",
    ]


def test_only_the_rules_a_profile_lists_run(ccn: Profile, clean_data: PlanData, run_lint: RunLint):
    ccn.rules = {"no-dissolve": ccn.rules["no-dissolve"]}
    clean_data["transitions"][0]["video"] = "dissolve"
    clean_data["captions"][0]["lines"] = ["X" * 40]
    assert {f.rule for f in run_lint(clean_data, profile=ccn).findings} == {"no-dissolve"}


def test_a_rule_can_be_switched_off(ccn: Profile, clean_data: PlanData, run_lint: RunLint):
    ccn.rules["no-dissolve"].enabled = False
    clean_data["transitions"][0]["video"] = "dissolve"
    assert only(run_lint(clean_data, profile=ccn), "no-dissolve") == []


def test_a_rule_can_be_regraded_but_skipped_notices_keep_their_severity(
    ccn: Profile, clean_data: PlanData, run_lint: RunLint
):
    ccn.rules["no-dissolve"].severity = "warn"
    ccn.rules["end-on-laugh"].severity = "error"
    clean_data["transitions"][0]["video"] = "dissolve"
    report = run_lint(clean_data, profile=ccn)
    assert [f.severity for f in only(report, "no-dissolve")] == ["warn"]
    # end-on-laugh and no-dialogue-splice have no input here, so they only report a skip.
    assert [f.severity for f in only(report, "end-on-laugh")] == ["info"]
    assert report.ok  # a warning does not fail the plan


def test_a_threshold_comes_from_the_profile_not_the_code(
    ccn: Profile, clean_data: PlanData, run_lint: RunLint
):
    clean_data["transitions"][0]["audio"]["frames"] = 20
    assert only(run_lint(clean_data, profile=ccn), "audio-crossfade") == []
    ccn.rules["audio-crossfade"].params["max_frames"] = 18
    (finding,) = only(run_lint(clean_data, profile=ccn), "audio-crossfade")
    assert "is 20 frames; the maximum is 18" in finding.message


def test_missing_audio_transitions_can_be_allowed_by_the_profile(
    ccn: Profile, clean_data: PlanData, run_lint: RunLint
):
    clean_data["transitions"] = []
    ccn.rules["audio-crossfade"].params["require_on_cuts"] = False
    assert only(run_lint(clean_data, profile=ccn), "audio-crossfade") == []


def test_a_profile_naming_an_unknown_rule_is_rejected(ccn: Profile, clean_data: PlanData):
    profile = parse_profile({"name": "x", "qa": ccn.qa.model_dump(), "rules": {"no-such-rule": {}}})
    with pytest.raises(ProfileError, match="unknown rule 'no-such-rule'"):
        lint(EditPlan.model_validate(clean_data), profile)


def test_a_profile_with_a_missing_or_misspelt_threshold_is_rejected(
    ccn: Profile, clean_data: PlanData
):
    plan = EditPlan.model_validate(clean_data)
    missing = parse_profile(
        {"name": "x", "qa": ccn.qa.model_dump(), "rules": {"caption-length": {}}}
    )
    with pytest.raises(ProfileError, match=r"missing \['max_chars'\]"):
        lint(plan, missing)
    typo = parse_profile(
        {"name": "x", "qa": ccn.qa.model_dump(), "rules": {"caption-length": {"max_char": 16}}}
    )
    with pytest.raises(ProfileError, match=r"missing \['max_chars'\]"):
        lint(plan, typo)
    extra = parse_profile(
        {
            "name": "x",
            "qa": ccn.qa.model_dump(),
            "rules": {"caption-length": {"max_chars": 16, "max_words": 3}},
        }
    )
    with pytest.raises(ProfileError, match=r"unknown thresholds \['max_words'\]"):
        lint(plan, extra)
