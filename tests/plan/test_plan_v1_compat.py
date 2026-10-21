"""Version 1 plans validate and lint exactly as they did before EditPlan v2 (S13a).

``fixtures/v1_compat`` holds the ccn profile as it was at 464f0e4 (``ccn_v1.toml``, a byte
copy) and the JSON reports that the 464f0e4 code wrote for the two plan fixtures, without and
with the transcript and laugh map. Linting the same plans with the same profile must give the
same bytes today.
"""

from __future__ import annotations

import json

import pytest
from plan_support import FIXTURES, V1_COMPAT

from aar.cli import main
from aar.plan.lint import lint
from aar.plan.model import EditPlan

INPUTS = [
    "--transcript",
    str(FIXTURES / "short_clean.transcript.json"),
    "--laughs",
    str(FIXTURES / "short_clean.laughs.json"),
]
CASES = [
    ("short_clean", [], "short_clean.report.json", 0),
    ("short_clean", INPUTS, "short_clean.inputs.report.json", 0),
    ("short_broken", [], "short_broken.report.json", 1),
    ("short_broken", INPUTS, "short_broken.inputs.report.json", 1),
]


@pytest.mark.parametrize(("plan", "inputs", "golden", "code"), CASES)
def test_v1_fixtures_lint_byte_identically_to_the_v1_code(
    capsys: pytest.CaptureFixture[str], plan: str, inputs: list[str], golden: str, code: int
):
    profile = str(V1_COMPAT / "ccn_v1.toml")
    path = str(FIXTURES / f"{plan}.plan.json")
    assert main(["plan", "lint", path, *inputs, "--profile", profile, "--json"]) == code
    assert capsys.readouterr().out == (V1_COMPAT / golden).read_text(encoding="utf-8")


@pytest.mark.parametrize("plan", ["short_clean", "short_broken"])
def test_v1_fixtures_still_say_version_1_and_survive_a_full_dump(plan: str, ccn_v1, laughs):
    text = (FIXTURES / f"{plan}.plan.json").read_text(encoding="utf-8")
    assert json.loads(text)["version"] == 1
    parsed = EditPlan.model_validate_json(text)
    assert parsed.version == 1
    # A full dump writes every field, the version 2 ones as null or []; that is not a use.
    again = EditPlan.model_validate_json(parsed.model_dump_json())
    assert again == parsed
    assert again.version == 1
    assert lint(again, ccn_v1, laughs=laughs) == lint(parsed, ccn_v1, laughs=laughs)


@pytest.mark.parametrize("plan", ["short_clean", "short_broken"])
def test_v1_fixtures_lint_the_same_when_relabelled_version_2(plan: str, ccn, ccn_v1, laughs):
    data = json.loads((FIXTURES / f"{plan}.plan.json").read_text(encoding="utf-8"))
    as_v1 = EditPlan.model_validate(data)
    as_v2 = EditPlan.model_validate({**data, "version": 2})
    for profile in (ccn, ccn_v1):
        assert lint(as_v2, profile, laughs=laughs) == lint(as_v1, profile, laughs=laughs)
