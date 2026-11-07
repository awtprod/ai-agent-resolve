"""``aar plan apply`` end to end: golden dry-run text and JSON, lint refusal, exit codes, the
idempotency diff, and the ``--submit`` payload."""

from __future__ import annotations

import json

import pytest
from plan_support import FIXTURES

from aar.cli import main

LONGFORM = str(FIXTURES / "ccn_longform_v2.plan.json")
SHORT = str(FIXTURES / "short_v2.plan.json")


def _golden(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_longform_dry_run_matches_golden(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", LONGFORM]) == 0
    assert capsys.readouterr().out == _golden("ccn_longform_v2.apply.txt")


def test_short_dry_run_matches_golden(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", SHORT]) == 0
    assert capsys.readouterr().out == _golden("short_v2.apply.txt")


def test_longform_json_matches_golden(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "apply", LONGFORM, "--json"]) == 0
    assert capsys.readouterr().out == _golden("ccn_longform_v2.apply.json")


def test_json_dry_run_is_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["plan", "apply", SHORT, "--json"])
    body = json.loads(capsys.readouterr().out)
    assert body["plan"]["kind"] == "short"
    assert body["phase_a"]["steps"][0]["op"] == "timeline.branch"
    assert body["phase_b"]["resolved"] is False


def test_apply_refuses_a_plan_with_lint_errors(capsys: pytest.CaptureFixture[str]) -> None:
    # The pre-v2 broken short has range errors.
    broken = str(FIXTURES / "short_broken.plan.json")
    code = main(["plan", "apply", broken])
    assert code == 2
    err = capsys.readouterr().err
    assert "lint_errors" in err


def test_resolved_dry_run_shows_grouped_steps(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "plan",
            "apply",
            LONGFORM,
            "--phase-a-result",
            str(FIXTURES / "ccn_longform_v2.phase-a.json"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "resolved into 7 grouped step(s)" in out


def test_branch_describe_prints_idempotency(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "plan",
            "apply",
            LONGFORM,
            "--branch-describe",
            str(FIXTURES / "ccn_longform_v2.branch.json"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "IDEMPOTENCY" in out
    assert "appends: 2 present, 3 to add" in out  # 5 appends, 0 and 72 already present
    assert "markers: 1 present, 1 to add" in out


def test_submit_phase_a_refused_without_template_describe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["plan", "apply", LONGFORM, "--submit", "A"])
    assert code == 2
    assert "template_size_unchecked" in capsys.readouterr().err


def test_submit_phase_b_refused_without_phase_a_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["plan", "apply", LONGFORM, "--submit", "B"])
    assert code == 2
    assert "missing_phase_a_result" in capsys.readouterr().err


def test_submit_phase_b_builds_a_valid_ops_sequence(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "plan",
            "apply",
            LONGFORM,
            "--submit",
            "B",
            "--json",
            "--phase-a-result",
            str(FIXTURES / "ccn_longform_v2.phase-a.json"),
        ]
    )
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["kind"] == "ops.sequence"
    assert body["idempotency_key"].startswith("plan.apply:")
    assert body["idempotency_key"].endswith(":B")
    assert len(body["params"]["steps"]) == 7


def test_batch_size_over_ten_refused(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["plan", "apply", LONGFORM, "--batch-size", "11"])
    assert code == 2
    assert "batch_size_invalid" in capsys.readouterr().err
