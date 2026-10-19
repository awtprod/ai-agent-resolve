"""S13a-f validator follow-ups: the BRAW input-colour-space rule (F2), the plan version (F1) and
blank text lines (F5)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from plan_support import FIXTURES, PlanData, only
from pydantic import ValidationError

from aar.plan.lint import LintReport
from aar.plan.model import EditPlan
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


# --- F1: the plan version is a whole number ---------------------------------------------------


def version_errors(data: PlanData, *, as_json: bool = False) -> list[dict[str, object]]:
    """The errors validating ``data`` gives, through Python input or through JSON text."""
    with pytest.raises(ValidationError) as caught:
        if as_json:
            EditPlan.model_validate_json(json.dumps(data))
        else:
            EditPlan.model_validate(data)
    return [dict(error) for error in caught.value.errors()]


@pytest.mark.parametrize("version", [True, False, 1.0, 2.0, 0.0, 1.5, "2", "1", 0, 3, None, [2]])
@pytest.mark.parametrize("as_json", [False, True], ids=["python", "json"])
def test_a_version_that_is_not_the_whole_number_1_or_2_is_one_error_at_version(
    clean_data: PlanData, version: object, as_json: bool
) -> None:
    clean_data["version"] = version
    [error] = version_errors(clean_data, as_json=as_json)
    assert error["loc"] == ("version",)
    assert error["type"] == "literal_error"
    assert str(error["msg"]).startswith("Input should be 1 or 2")


@pytest.mark.parametrize("version", [True, 1.0, 2.0])
def test_the_error_for_a_boolean_or_decimal_version_says_what_is_wrong(
    clean_data: PlanData, version: object
) -> None:
    clean_data["version"] = version
    [error] = version_errors(clean_data)
    assert "whole number" in str(error["msg"])


@pytest.mark.parametrize("as_json", [False, True], ids=["python", "json"])
@pytest.mark.parametrize("version", [1, 2])
def test_the_whole_numbers_1_and_2_still_work(
    clean_data: PlanData, version: int, as_json: bool
) -> None:
    clean_data["version"] = version
    if as_json:
        plan = EditPlan.model_validate_json(json.dumps(clean_data))
    else:
        plan = EditPlan.model_validate(clean_data)
    assert plan.version == version
    assert type(plan.version) is int


@pytest.mark.parametrize("as_json", [False, True], ids=["python", "json"])
def test_an_omitted_version_is_still_2(clean_data: PlanData, as_json: bool) -> None:
    del clean_data["version"]
    if as_json:
        plan = EditPlan.model_validate_json(json.dumps(clean_data))
    else:
        plan = EditPlan.model_validate(clean_data)
    assert plan.version == 2


def test_a_bad_version_does_not_pile_a_second_error_on_a_v1_plan(clean_data: PlanData) -> None:
    """The version is wrong, so the v1/v2 check never runs: still one error."""
    clean_data["version"] = True
    clean_data["template"] = "Example Template"  # a v2 field; would be a plan_version error
    assert len(version_errors(clean_data)) == 1


def test_the_version_schema_is_still_the_integer_1_or_2() -> None:
    schema = EditPlan.model_json_schema()["properties"]["version"]
    assert schema["type"] == "integer"
    assert schema["enum"] == [1, 2]
    assert schema["default"] == 2


# --- F5: no blank text lines ------------------------------------------------------------------

BLANKS = ["", " ", "   ", "\t", "\n", " \u00a0 "]


def with_sting(data: PlanData, lines: list[str]) -> PlanData:
    """A version 2 plan with an outro whose sting has these lines."""
    data["version"] = 2
    data["sources"].append({"id": "outro", "media": "media/outro.mov", "duration": 302})
    data["outro"] = {"source": "outro", "sting": {"lines": lines, "start": 2, "frames": 100}}
    return data


def errors_at(data: PlanData) -> list[tuple[tuple[str | int, ...], str]]:
    """(loc, message) of every error validating ``data`` gives."""
    return [(tuple(e["loc"]), str(e["msg"])) for e in version_errors(data)]


@pytest.mark.parametrize("blank", BLANKS)
def test_a_blank_sting_line_is_one_error_at_that_line(clean_data: PlanData, blank: str) -> None:
    with_sting(clean_data, ["Pat Example", blank])
    [(loc, msg)] = errors_at(clean_data)
    assert loc == ("outro", "sting", "lines", 1)
    assert "blank" in msg


@pytest.mark.parametrize("blank", BLANKS)
def test_a_blank_title_line_is_one_error_at_that_line(clean_data: PlanData, blank: str) -> None:
    clean_data["titles"][0]["lines"] = [blank, "Stand-up comic"]
    [(loc, msg)] = errors_at(clean_data)
    assert loc == ("titles", 0, "lines", 0)
    assert "blank" in msg


@pytest.mark.parametrize("blank", BLANKS)
def test_a_blank_caption_line_is_one_error_at_that_line(clean_data: PlanData, blank: str) -> None:
    clean_data["captions"][1]["lines"] = ["WAIT FOR IT", blank]
    [(loc, msg)] = errors_at(clean_data)
    assert loc == ("captions", 1, "lines", 1)
    assert "blank" in msg


def test_every_blank_line_gets_its_own_error(clean_data: PlanData) -> None:
    clean_data["captions"][0]["lines"] = ["", "ONE", " "]
    assert [loc for loc, _ in errors_at(clean_data)] == [
        ("captions", 0, "lines", 0),
        ("captions", 0, "lines", 2),
    ]


@pytest.mark.parametrize("as_json", [False, True], ids=["python", "json"])
def test_a_blank_line_is_refused_through_json_too(clean_data: PlanData, as_json: bool) -> None:
    clean_data["captions"][0]["lines"] = ["  "]
    [error] = version_errors(clean_data, as_json=as_json)
    assert error["loc"] == ("captions", 0, "lines", 0)


def test_an_empty_list_of_lines_is_still_refused(clean_data: PlanData) -> None:
    clean_data["captions"][0]["lines"] = []
    [(loc, _)] = errors_at(clean_data)
    assert loc == ("captions", 0, "lines")


def test_ordinary_lines_still_validate_and_keep_their_text(clean_data: PlanData) -> None:
    """Padding inside a line is the author's business; only an all-blank line is refused."""
    with_sting(clean_data, ["Pat Example", " Stand-up comic "])
    clean_data["titles"][0]["lines"] = ["A", "B"]
    clean_data["captions"][0]["lines"] = ["I PARKED THERE", "FOR ONE MINUTE"]
    plan = EditPlan.model_validate(clean_data)
    assert plan.outro is not None and plan.outro.sting is not None
    assert plan.outro.sting.lines == ["Pat Example", " Stand-up comic "]
    assert plan.titles[0].lines == ["A", "B"]
    assert plan.captions[0].lines == ["I PARKED THERE", "FOR ONE MINUTE"]


def test_no_fixture_plan_has_a_blank_text_line() -> None:
    """The plan fixtures and goldens must keep validating: none of them uses a blank line."""
    plans = sorted(FIXTURES.rglob("*.plan.json"))
    assert plans
    for path in plans:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        texts = [
            line
            for item in [*data.get("titles", []), *data.get("captions", [])]
            for line in item["lines"]
        ]
        assert all(line.strip() for line in texts), path.name
