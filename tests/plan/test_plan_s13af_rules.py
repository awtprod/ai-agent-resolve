"""S13a-f: the BRAW input-colour-space safety rule (validator finding F2)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from plan_support import PlanData, only

from aar.plan.lint import LintReport
from aar.plan.profile import Profile

RunLint = Callable[..., LintReport]

RULE = "no-braw-input-color-space"


def braw_source(data: PlanData, media: str, *, space: str | None = "Rec.709 Gamma 2.4") -> PlanData:
    """The clean short as a v2 plan whose first source is ``media``, with a colour override."""
    data["version"] = 2
    source = data["sources"][0]
    source["media"] = media
    if space is not None:
        source["color"] = {"input_color_space": space}
    return data


@pytest.mark.parametrize(
    "media",
    ["media/acme_take1.braw", "media/acme_take1.BRAW", "media/acme_take1.BrAw", "A001_C002.braw"],
)
def test_input_color_space_on_braw_media_is_an_error(
    clean_data: PlanData, run_lint: RunLint, media: str
) -> None:
    report = run_lint(braw_source(clean_data, media))
    [finding] = only(report, RULE)
    assert finding.severity == "error"
    assert report.ok is False
    assert report.counts["error"] >= 1
    assert "'cam-a'" in finding.message
    assert media in finding.message
    assert "Rec.709 Gamma 2.4" in finding.message
    assert "input_color_space" in finding.fix_hint
    assert "'cam-a'" in finding.fix_hint
    assert "BRAW decodes through its own RAW settings" in finding.fix_hint
    assert [r.source_id for r in finding.refs] == ["cam-a"]


def test_mov_with_a_colour_override_passes(clean_data: PlanData, run_lint: RunLint) -> None:
    report = run_lint(braw_source(clean_data, "media/acme_take1.mov"))
    assert only(report, RULE) == []


def test_braw_without_a_colour_override_passes(clean_data: PlanData, run_lint: RunLint) -> None:
    report = run_lint(braw_source(clean_data, "media/acme_take1.braw", space=None))
    assert only(report, RULE) == []


def test_braw_with_colour_but_no_input_space_passes(
    clean_data: PlanData, run_lint: RunLint
) -> None:
    data = braw_source(clean_data, "media/acme_take1.braw", space=None)
    data["sources"][0]["color"] = {"lut": "acme_look"}
    assert only(run_lint(data), RULE) == []


def test_braw_in_the_middle_of_the_name_is_not_braw_media(
    clean_data: PlanData, run_lint: RunLint
) -> None:
    """Only the end of the media string counts: it is an extension test, not a substring test."""
    for media in ("media/braw/acme_take1.mov", "media/acme.braw.mov", "acme_braw"):
        assert only(run_lint(braw_source(clean_data, media)), RULE) == []


def test_a_v1_plan_with_braw_media_passes(clean_data: PlanData, run_lint: RunLint) -> None:
    """Version 1 has no colour field, so there is nothing to flag."""
    clean_data["sources"][0]["media"] = "media/acme_take1.braw"
    assert clean_data["version"] == 1
    assert only(run_lint(clean_data), RULE) == []


def test_only_the_braw_sources_are_named(clean_data: PlanData, run_lint: RunLint) -> None:
    data = braw_source(clean_data, "media/acme_take1.braw")
    data["sources"][1]["color"] = {"input_color_space": "Rec.709 Gamma 2.4"}  # .mov: fine
    data["sources"][2]["media"] = "media/acme_take3.BRAW"
    data["sources"][2]["color"] = {"input_color_space": "Rec.709 Gamma 2.4"}
    findings = only(run_lint(data), RULE)
    assert [f.refs[0].source_id for f in findings] == ["cam-a", "cam-c"]
    assert "sources[0]" in findings[0].message
    assert "sources[2]" in findings[1].message


def test_a_profile_with_the_rule_off_passes(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
) -> None:
    ccn.rules[RULE].enabled = False
    report = run_lint(braw_source(clean_data, "media/acme_take1.braw"), profile=ccn)
    assert only(report, RULE) == []


def test_a_profile_can_regrade_the_rule(
    clean_data: PlanData, run_lint: RunLint, ccn: Profile
) -> None:
    ccn.rules[RULE].severity = "warn"
    report = run_lint(braw_source(clean_data, "media/acme_take1.braw"), profile=ccn)
    [finding] = only(report, RULE)
    assert finding.severity == "warn"


def test_the_ccn_profile_runs_the_rule_as_an_error(ccn: Profile) -> None:
    config = ccn.rules[RULE]
    assert config.enabled is True
    assert config.severity is None  # the rule's own severity, which is "error"
    assert config.params == {}
