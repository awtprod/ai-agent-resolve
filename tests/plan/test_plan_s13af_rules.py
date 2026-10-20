"""S13a-f validator follow-ups: the BRAW input-colour-space rule (F2), the plan version (F1),
blank text lines (F5), what the cut rules count (F3) and the Head fade note (F6)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from plan_support import FIXTURES, PlanData, only
from pydantic import ValidationError

from aar.plan.lint import LintReport, all_rules
from aar.plan.model import EditPlan, Head
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


#: Real BRAW references that carry a URI query, a fragment or stray whitespace after ``.braw``.
DRESSED_BRAW = [
    "file:///acme/a.braw?x=1",
    "file:///acme/a.braw#t=1",
    "file:///acme/a.braw?x=1#t=1",
    "file:///acme/a.BRAW?x=1",
    "a.braw ",
    "a.BRAW \n",
    " a.braw",
    "\ta.braw\r\n",
    "a.braw ?x=1",
    "a.braw\t#t=1",
    "a.braw?",
    "a.braw#",
]


@pytest.mark.parametrize("media", DRESSED_BRAW, ids=repr)
def test_a_uri_query_fragment_or_stray_whitespace_does_not_hide_braw_media(
    clean_data: PlanData, run_lint: RunLint, media: str
) -> None:
    report = run_lint(braw_source(clean_data, media))
    [finding] = only(report, RULE)
    assert finding.severity == "error"
    assert [r.source_id for r in finding.refs] == ["cam-a"]


#: Media that is not BRAW: the ``.braw`` is not the end of the path.
NOT_BRAW = [
    "a.braw.mov",
    "a.brawx",
    "braw",
    "a.mov?f=a.braw",
    "a.mov#a.braw",
    "file:///acme/a.mov?f=a.braw#a.braw",
    "a.mov ",
    "a.braw/clip.mov",
]


@pytest.mark.parametrize("media", NOT_BRAW, ids=repr)
def test_media_whose_path_does_not_end_in_braw_still_passes(
    clean_data: PlanData, run_lint: RunLint, media: str
) -> None:
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


def test_the_rule_is_registered_as_an_error_rule_without_thresholds() -> None:
    registered = all_rules()[RULE]
    assert registered.severity == "error"
    assert registered.params == ()
    assert registered.needs == ()
    assert "BRAW" in registered.summary


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


@pytest.mark.parametrize("version", ["2", "1", 0, 3, None, [2]])
def test_any_other_bad_version_keeps_the_plain_message(
    clean_data: PlanData, version: object
) -> None:
    """The "not a boolean or a decimal" note is only for a boolean or a decimal."""
    clean_data["version"] = version
    [error] = version_errors(clean_data)
    assert error["msg"] == "Input should be 1 or 2"


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


# --- F3: the cut rules see same-track joins only ----------------------------------------------

DESIGN = FIXTURES.parents[2] / "docs" / "DESIGN.md"

#: The rules built on joins (neighbouring segments on one track), and the one that hints with them.
JOIN_RULES = (
    "cut-density",
    "no-same-camera-jump",
    "timejump-close-to-wide",
    "audio-crossfade",
    "min-shot",
)


def three_segments_two_cuts(data: PlanData) -> PlanData:
    """Track 1 only: segments of 250, 100 and 100 frames at 25 fps, cuts at 10 s and 14 s."""
    data["fps"] = 25
    data["transitions"] = []
    data["segments"] = [
        {"source": "cam-a", "src_in": 1000, "src_out": 1250},
        {"source": "cam-a", "src_in": 1350, "src_out": 1450},
        {"source": "cam-a", "src_in": 1550, "src_out": 1650},
    ]
    return data


def join_findings(report: LintReport) -> list[tuple[str, str, str]]:
    return [(f.rule, f.severity, f.message) for f in report.findings if f.rule in JOIN_RULES]


def test_the_design_doc_says_the_cut_rules_consider_same_track_joins_only() -> None:
    text = DESIGN.read_text(encoding="utf-8")
    [line] = [x for x in text.splitlines() if x.startswith("The join-based cut rules")]
    for rule_id in JOIN_RULES:
        assert f"`{rule_id}`" in line
    assert "same-track joins only" in line
    assert "overlay entry" in line
    assert "cut into the outro" in line


def test_an_overlay_entry_on_another_track_is_not_a_cut(
    clean_data: PlanData, run_lint: RunLint
) -> None:
    base = three_segments_two_cuts(clean_data)
    before = join_findings(run_lint(base))
    # Two cuts in 10 s is the most the profile allows, so a third would warn.
    assert not [f for f in before if f[0] == "cut-density"]
    clean_data["segments"].append(
        {"track": 2, "source": "cam-a", "src_in": 5000, "src_out": 5100, "record_in": 400}
    )
    plan = EditPlan.model_validate(clean_data)
    assert plan.joins() == [(0, 1), (1, 2)]
    assert join_findings(run_lint(clean_data)) == before


def test_the_same_segment_on_the_main_track_is_a_cut(
    clean_data: PlanData, run_lint: RunLint
) -> None:
    """The control for the overlay test: on track 1 it is a third cut, and it is counted."""
    three_segments_two_cuts(clean_data)
    clean_data["segments"].append({"source": "cam-a", "src_in": 5000, "src_out": 5100})
    [finding] = only(run_lint(clean_data), "cut-density")
    assert finding.message.startswith("3 cuts in ")


def test_the_cut_into_the_outro_is_not_a_cut(clean_data: PlanData, run_lint: RunLint) -> None:
    three_segments_two_cuts(clean_data)
    before = join_findings(run_lint(clean_data))
    clean_data["version"] = 2
    clean_data["sources"].append({"id": "outro", "media": "media/outro.mov", "duration": 302})
    # A 10-frame outro (0.4 s) is shorter than min-shot allows and follows the 2nd cut by 4 s.
    clean_data["outro"] = {"source": "outro", "src_out": 10}
    assert EditPlan.model_validate(clean_data).outro_range() == (450, 460)
    assert join_findings(run_lint(clean_data)) == before


# --- F6: the head's audio fade-in is uncapped, on purpose -------------------------------------


def test_a_long_head_audio_fade_in_is_valid_and_is_not_a_finding(
    clean_data: PlanData, run_lint: RunLint
) -> None:
    baseline = [(f.rule, f.severity, f.message) for f in run_lint(clean_data).findings]
    clean_data["version"] = 2
    clean_data["head"] = {"audio_fade_frames": 100_000}
    plan = EditPlan.model_validate(clean_data)
    assert plan.head is not None
    assert plan.head.audio_fade_frames == 100_000
    assert [(f.rule, f.severity, f.message) for f in run_lint(clean_data).findings] == baseline


def test_the_head_docstring_says_the_audio_fade_in_is_uncapped() -> None:
    doc = Head.__doc__ or ""
    assert "audio_fade_frames" in doc
    assert "uncapped" in doc


def test_a_negative_head_audio_fade_is_still_an_error(clean_data: PlanData) -> None:
    clean_data["version"] = 2
    clean_data["head"] = {"audio_fade_frames": -1}
    [error] = version_errors(clean_data)
    assert error["loc"] == ("head", "audio_fade_frames")
