"""docs/DESIGN.md (Analysis workers, part 2) states what the S13c2 code does."""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.analyze import faces, gemini, laughs

DESIGN = Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md"


def paragraph() -> str:
    [line] = [
        x
        for x in DESIGN.read_text(encoding="utf-8").splitlines()
        if x.startswith("**Analysis workers, part 2 (S13c2).**")
    ]
    return line


@pytest.mark.parametrize(
    "phrase",
    [
        "`aar.faces/1`",
        "`aar.laughs/1`",
        "`$AAR_YUNET_MODEL`",
        "never downloaded",
        "frame on screen at `i / sample_fps`",
        "`truncated_tail`",
        "byte-identical",
        f"`{gemini.KEY_ENV}`",
        "sent only in a request header",
        "never written anywhere",
        "`--allow-paid`",
        "exits 2 before any network use",
        "`--from-raw`",
        "No redirects",
        "every request must go to the API's own host",
        "Applause never enters a laugh map",
        "tunable starting values",
        "`model-outside-dir`",
        "explicit model path is taken as given",
        "Laughs within 0.25 s merge",
    ],
)
def test_the_paragraph_names(phrase):
    assert phrase in paragraph()


def test_every_parser_drop_reason_is_documented():
    reasons = ("bad-item", "bare-number", "bad-time", "bad-level", "unknown-kind", "reversed")
    for reason in (*reasons, "out-of-range"):
        assert f"`{reason}`" in paragraph()
        assert f"``{reason}``" in laughs.__doc__


def test_the_documented_numbers_are_the_code_defaults():
    text = paragraph()
    settings = laughs.LaughSettings()
    assert f"`bump90 ≥ {settings.confirm_bump_db}` dB" in text
    assert f"`frac6 ≥ {settings.confirm_frac:.2f}`" in text
    assert f"{laughs.MERGE_GAP_S} s merge" in text
    assert f"{laughs.SUSTAIN_S * 1000:.0f} ms" in text
    assert f"{laughs.ONSET_DB:.0f} dB above the baseline" in text
    assert f"{laughs.SUSTAIN_DB:.0f} dB above it" in text
    assert f"{laughs.BASELINE_S:.0f} s before it" in text
    assert f"{laughs.REFINE_FRAME_S * 1000:.0f} ms frame" in text
    assert f"default {faces.FaceSettings().sample_fps}" in text
    assert "`LAUGH_PROMPT_V1`" in text and laughs.PROMPT_VERSION == "v1"
