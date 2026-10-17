"""caption-length and caption-style."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from plan_support import PlanData, only

from aar.plan.lint import LintReport

RunLint = Callable[..., LintReport]


def test_caption_length_passes_at_the_limit(clean_data: PlanData, run_lint: RunLint):
    clean_data["captions"][0]["lines"] = ["ABCDEFGHIJKLMNOP"]  # 16 characters
    assert only(run_lint(clean_data), "caption-length") == []


def test_caption_length_flags_one_character_over(clean_data: PlanData, run_lint: RunLint):
    clean_data["captions"][1]["lines"] = ["WAIT FOR IT", "ABCDEFGHIJKLMNOPQ"]  # 17 characters
    (finding,) = only(run_lint(clean_data), "caption-length")
    assert finding.severity == "error"
    assert finding.message == (
        "Caption 1 has a 17-character line 'ABCDEFGHIJKLMNOPQ'; the limit is 16."
    )
    assert [(r.item, r.record_frame, r.record_tc) for r in finding.refs] == [
        ("caption[1]", 600, "00:00:25:00")
    ]


def test_caption_length_counts_spaces(clean_data: PlanData, run_lint: RunLint):
    clean_data["captions"][0]["lines"] = ["SIXTEEN CHARS OK!", "SHORT"]  # first is 17 with spaces
    (finding,) = only(run_lint(clean_data), "caption-length")
    assert "17-character line" in finding.message


def test_caption_style_passes_for_montserrat_with_a_yellow_active_word(
    clean_data: PlanData, run_lint: RunLint
):
    assert only(run_lint(clean_data), "caption-style") == []


@pytest.mark.parametrize("color", ["yellow", "#FFE600", "#ffd700", "#FF0"])
def test_caption_style_accepts_yellows(clean_data: PlanData, run_lint: RunLint, color: str):
    clean_data["caption_style"]["active_word_color"] = color
    assert only(run_lint(clean_data), "caption-style") == []


@pytest.mark.parametrize("color", ["#00FF00", "#FFFFFF", "#FF8800", "chartreuse", "#FFE60"])
def test_caption_style_rejects_other_colours(clean_data: PlanData, run_lint: RunLint, color: str):
    clean_data["caption_style"]["active_word_color"] = color
    (finding,) = only(run_lint(clean_data), "caption-style")
    assert finding.severity == "info"
    assert finding.message.startswith("2 caption(s) do not highlight the active word in yellow")
    assert len(finding.refs) == 2


def test_caption_style_flags_the_wrong_font_only_where_it_applies(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["captions"][1]["style"] = {"font": "Arial", "active_word_color": "#FFE600"}
    (finding,) = only(run_lint(clean_data), "caption-style")
    assert finding.severity == "info"
    assert finding.message == "1 caption(s) do not use the font 'Montserrat'."
    assert [r.item for r in finding.refs] == ["caption[1]"]


def test_caption_style_compares_the_font_case_insensitively(
    clean_data: PlanData, run_lint: RunLint
):
    clean_data["caption_style"]["font"] = "montserrat"
    assert only(run_lint(clean_data), "caption-style") == []


def test_caption_style_flags_captions_with_no_style_at_all(clean_data: PlanData, run_lint: RunLint):
    clean_data["caption_style"] = None
    findings = only(run_lint(clean_data), "caption-style")
    assert [f.severity for f in findings] == ["info", "info"]
    assert [len(f.refs) for f in findings] == [2, 2]


def test_caption_style_caps_the_listed_refs(clean_data: PlanData, run_lint: RunLint):
    clean_data["caption_style"]["font"] = "Arial"
    clean_data["captions"] = [
        {"lines": ["HI"], "record_in": n * 10, "record_out": n * 10 + 5} for n in range(30)
    ]
    (finding,) = only(run_lint(clean_data), "caption-style")
    assert finding.message == (
        "30 caption(s) do not use the font 'Montserrat' (the first 20 are listed)."
    )
    assert len(finding.refs) == 20
