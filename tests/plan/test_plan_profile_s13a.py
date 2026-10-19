"""S13a profile mechanics: optional thresholds with code defaults, value checks, the old
end-on-laugh key, and the CLI for EditPlan v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from plan_support import FIXTURES, V1_COMPAT, PlanData

from aar.cli import main
from aar.plan.lint import all_rules, lint, validate_profile
from aar.plan.model import EditPlan
from aar.plan.profile import Profile, ProfileError, parse_profile

CLEAN = str(FIXTURES / "short_clean.plan.json")


def one_rule(ccn: Profile, rule_id: str, params: dict[str, Any]) -> Profile:
    return parse_profile({"name": "x", "qa": ccn.qa.model_dump(), "rules": {rule_id: params}})


# end-on-laugh thresholds


def test_end_on_laugh_defaults_are_the_old_end_window():
    rule = all_rules()["end-on-laugh"]
    assert rule.params == ("min_strength",)
    assert rule.resolve({"min_strength": 0.0}) == {
        "anchor": "end",
        "min_after_ms": 0,
        "max_after_ms": 500,
        "min_strength": 0.0,
    }


def test_the_old_max_after_laugh_ms_key_is_the_end_anchored_maximum():
    rule = all_rules()["end-on-laugh"]
    assert rule.resolve({"min_strength": 0.0, "max_after_laugh_ms": 300}) == {
        "anchor": "end",
        "min_after_ms": 0,
        "max_after_ms": 300,
        "min_strength": 0.0,
    }


def test_the_v1_profile_still_loads_and_validates(ccn_v1: Profile):
    validate_profile(ccn_v1)
    assert ccn_v1.rules["end-on-laugh"].params == {"max_after_laugh_ms": 500, "min_strength": 0.0}
    assert "cut-density" not in ccn_v1.rules


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"max_after_laugh_ms": 500, "max_after_ms": 400},
            "set max_after_ms or its old name max_after_laugh_ms, not both",
        ),
        (
            {"max_after_laugh_ms": 500, "anchor": "onset"},
            'max_after_laugh_ms is the old name of max_after_ms with anchor "end"',
        ),
        ({"anchor": "middle"}, 'anchor must be "onset" or "end", not \'middle\''),
        ({"min_after_ms": 500, "max_after_ms": 400}, "min_after_ms (500) is more than"),
        ({"max_after_ms": "400"}, "max_after_ms must be a number, not '400'"),
        ({"min_after_ms": True}, "min_after_ms must be a number, not True"),
        ({"max_after_ms": float("inf")}, "max_after_ms must be finite"),
    ],
)
def test_bad_end_on_laugh_thresholds_are_profile_errors(
    ccn: Profile, clean_data: PlanData, params: dict[str, Any], message: str
):
    profile = one_rule(ccn, "end-on-laugh", {"min_strength": 0.0, **params})
    with pytest.raises(ProfileError, match="rule 'end-on-laugh': ") as raised:
        lint(EditPlan.model_validate(clean_data), profile)
    assert message in str(raised.value)


def test_a_negative_window_start_is_allowed(ccn: Profile, clean_data: PlanData):
    profile = one_rule(ccn, "end-on-laugh", {"min_strength": 0.0, "min_after_ms": -100})
    validate_profile(profile)


# no-dissolve thresholds


def test_no_dissolve_defaults_allow_only_cuts():
    assert all_rules()["no-dissolve"].resolve({}) == {
        "allow_head": False,
        "max_head_frames": None,
        "allow_outro": False,
        "max_outro_frames": None,
    }


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"allow_head": True}, "allow_head = true needs max_head_frames"),
        ({"allow_outro": True}, "allow_outro = true needs max_outro_frames"),
        ({"allow_head": "yes", "max_head_frames": 15}, "allow_head must be true or false"),
        ({"allow_head": True, "max_head_frames": -1}, "max_head_frames must be at least 0"),
        ({"allow_head": True, "max_head_frames": 1.5}, "max_head_frames must be a whole number"),
    ],
)
def test_bad_no_dissolve_thresholds_are_profile_errors(
    ccn: Profile, params: dict[str, Any], message: str
):
    with pytest.raises(ProfileError, match="rule 'no-dissolve': ") as raised:
        validate_profile(one_rule(ccn, "no-dissolve", params))
    assert message in str(raised.value)


def test_an_unknown_optional_threshold_is_still_rejected(ccn: Profile):
    with pytest.raises(ProfileError, match=r"unknown thresholds \['allow_middle'\]"):
        validate_profile(one_rule(ccn, "no-dissolve", {"allow_middle": True}))


# new rules' thresholds are required and checked


@pytest.mark.parametrize(
    ("rule_id", "params", "message"),
    [
        ("cut-density", {"window_s": 10, "continuity_tolerance_frames": 1}, "missing ['max_cuts']"),
        (
            "cut-density",
            {"window_s": 0, "max_cuts": 2, "continuity_tolerance_frames": 1},
            "window_s must be more than 0",
        ),
        (
            "cut-density",
            {"window_s": 10, "max_cuts": -1, "continuity_tolerance_frames": 1},
            "max_cuts must be at least 0",
        ),
        ("min-shot", {}, "missing ['min_seconds']"),
        ("min-shot", {"min_seconds": -1}, "min_seconds must be at least 0"),
        ("short-end-fade", {"max_frames": True}, "max_frames must be a whole number, not True"),
    ],
)
def test_new_rule_thresholds_are_required_and_checked(
    ccn: Profile, rule_id: str, params: dict[str, Any], message: str
):
    with pytest.raises(ProfileError) as raised:
        validate_profile(one_rule(ccn, rule_id, params))
    assert message in str(raised.value)


def test_a_bad_threshold_is_exit_two_on_the_cli(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    profile = tmp_path / "bad.toml"
    profile.write_text(
        'name = "bad"\n[qa]\nlongform_lufs = -22\nshort_lufs = -14.5\n'
        "lufs_tolerance = 0.5\nreview_mp4_max_mb = 20\n[rules.end-on-laugh]\n"
        'min_strength = 0.0\nanchor = "middle"\n',
        encoding="utf-8",
    )
    assert main(["plan", "lint", CLEAN, "--profile", str(profile)]) == 2
    assert 'anchor must be "onset" or "end"' in capsys.readouterr().err


# CLI: schema and version errors


def test_schema_is_the_version_2_schema(capsys: pytest.CaptureFixture[str]):
    assert main(["plan", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["properties"]["version"]["enum"] == [1, 2]
    assert schema["properties"]["version"]["default"] == 2
    for name in ("template", "head", "outro", "audio_tracks", "end_fade_frames"):
        assert name in schema["properties"], name
    assert "color" in schema["$defs"]["Source"]["properties"]
    assert "transform" in schema["$defs"]["Segment"]["properties"]


def test_a_v1_plan_using_a_v2_field_is_exit_two_naming_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    data = json.loads(Path(CLEAN).read_text(encoding="utf-8"))
    data["end_fade_frames"] = 1
    data["segments"][1]["transform"] = {"zoom": 1.25}
    path = tmp_path / "v1.plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["plan", "lint", str(path), "--json"]) == 2
    details = json.loads(capsys.readouterr().out)["error"]["details"]
    assert [(d["type"], d["loc"]) for d in details] == [
        ("plan_version", ["end_fade_frames"]),
        ("plan_version", ["segments", 1, "transform"]),
    ]
    assert main(["plan", "lint", str(path)]) == 2
    err = capsys.readouterr().err
    assert (
        "  segments.1.transform: segments[1].transform is a version 2 field, but this plan "
        'says version 1: set "version": 2 or remove segments[1].transform'
    ) in err


def test_a_v2_plan_lints_on_the_cli(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    data = json.loads(Path(CLEAN).read_text(encoding="utf-8"))
    data["version"] = 2
    data["template"] = "Example Template"
    data["sources"].append({"id": "outro", "media": "media/outro.mov", "duration": 48})
    data["head"] = {"video": "dissolve", "video_frames": 15, "audio_fade_frames": 15}
    data["outro"] = {"source": "outro", "transition": {"video": "dissolve", "video_frames": 15}}
    data["end_fade_frames"] = 1
    path = tmp_path / "v2.plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    laughs = str(FIXTURES / "short_clean.laughs.json")
    transcript = str(FIXTURES / "short_clean.transcript.json")
    assert main(["plan", "lint", str(path), "--transcript", transcript, "--laughs", laughs]) == 0
    assert capsys.readouterr().out.startswith(
        "profile ccn: 3 segment(s), 32.032 s (768 frames): 0 error, 0 warn, 0 info"
    )


def test_the_v1_profile_file_is_the_pre_s13a_ccn_profile():
    text = (V1_COMPAT / "ccn_v1.toml").read_text(encoding="utf-8")
    assert "max_after_laugh_ms = 500" in text
    assert "anchor" not in text and "cut-density" not in text
