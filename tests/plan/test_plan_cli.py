"""``aar plan schema`` and ``aar plan lint``: output and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from plan_support import FIXTURES

from aar.cli import main

CLEAN = str(FIXTURES / "short_clean.plan.json")
BROKEN = str(FIXTURES / "short_broken.plan.json")
TRANSCRIPT = str(FIXTURES / "short_clean.transcript.json")
LAUGHS = str(FIXTURES / "short_clean.laughs.json")


def test_schema_prints_the_json_schema(capsys: pytest.CaptureFixture[str]):
    assert main(["plan", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "EditPlan"
    assert "segments" in schema["properties"]
    assert schema["$defs"]["Segment"]["properties"]["src_out"]["description"].startswith(
        "End source frame (exclusive)"
    )


def test_lint_exits_zero_for_a_clean_plan(capsys: pytest.CaptureFixture[str]):
    code = main(["plan", "lint", CLEAN, "--transcript", TRANSCRIPT, "--laughs", LAUGHS])
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "profile ccn: 3 segment(s), 30.03 s (720 frames): 0 error, 0 warn, 0 info"
    )


def test_lint_exits_one_when_there_are_error_findings(capsys: pytest.CaptureFixture[str]):
    assert main(["plan", "lint", BROKEN]) == 1
    out = capsys.readouterr().out
    assert "9 error, 1 warn, 4 info" in out
    assert "ERROR chronological: Segment 1 goes back in time" in out
    assert "      at segment 1 in, record 00:00:10:00, source 00:00:41:16 (cam-a)" in out
    assert '      fix: Use audio type "Cross Fade 0 dB".' in out


def test_lint_json_output_is_the_report(capsys: pytest.CaptureFixture[str]):
    assert main(["plan", "lint", BROKEN, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["profile"] == "ccn"
    assert report["counts"] == {"error": 9, "warn": 1, "info": 4}
    assert [f["rule"] for f in report["findings"] if f["severity"] == "error"] == [
        "chronological",
        "no-same-camera-jump",
        "no-dissolve",
        "audio-crossfade",
        "audio-crossfade",
        "audio-crossfade",
        "short-max-duration",
        "no-whip-pan",
        "caption-length",
    ]
    first = report["findings"][0]
    assert set(first) == {"rule", "severity", "message", "fix_hint", "refs"}
    assert first["refs"][1] == {
        "segment": 1,
        "item": None,
        "point": "in",
        "source_id": "cam-a",
        "record_frame": 240,
        "record_tc": "00:00:10:00",
        "source_frame": 1000,
        "source_tc": "00:00:41:16",
    }


def test_warnings_alone_do_not_fail_the_plan(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    laughs = tmp_path / "laughs.json"
    laughs.write_text(json.dumps([{"start_s": 80.0, "end_s": 81.0}]), encoding="utf-8")
    code = main(["plan", "lint", CLEAN, "--transcript", TRANSCRIPT, "--laughs", str(laughs)])
    assert code == 0
    assert "0 error, 1 warn, 0 info" in capsys.readouterr().out


def test_profile_option_overrides_the_plans_profile(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    profile = tmp_path / "strict.toml"
    profile.write_text(
        'name = "strict"\n[qa]\nlongform_lufs = -22\nshort_lufs = -14.5\n'
        "lufs_tolerance = 0.5\nreview_mp4_max_mb = 20\n[rules.caption-length]\nmax_chars = 5\n",
        encoding="utf-8",
    )
    assert main(["plan", "lint", CLEAN, "--profile", str(profile)]) == 1
    assert capsys.readouterr().out.startswith("profile strict: 3 segment(s)")


def test_missing_plan_file_is_exit_two(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    assert main(["plan", "lint", str(tmp_path / "nope.json")]) == 2
    assert capsys.readouterr().err.startswith("error: [Errno 2] No such file or directory")


def test_malformed_json_is_exit_two(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["plan", "lint", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_an_invalid_plan_is_exit_two_with_structured_details(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"project": "x"}), encoding="utf-8")
    assert main(["plan", "lint", str(bad), "--json"]) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is False
    assert body["error"]["type"] == "invalid-input"
    assert {"loc": ["timeline"], "msg": "Field required", "type": "missing"} in body["error"][
        "details"
    ]


def test_an_invalid_plan_lists_each_problem_in_text_mode(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"project": "x"}), encoding="utf-8")
    assert main(["plan", "lint", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "error: input failed validation" in err
    assert "  timeline: Field required" in err


def test_an_unknown_profile_is_exit_two(capsys: pytest.CaptureFixture[str]):
    assert main(["plan", "lint", CLEAN, "--profile", "nope"]) == 2
    assert "no built-in rule profile named 'nope'" in capsys.readouterr().err


def test_a_bad_transcript_is_exit_two(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    transcript = tmp_path / "t.json"
    transcript.write_text(json.dumps([{"text": "a", "start_s": 3, "end_s": 1}]), encoding="utf-8")
    assert main(["plan", "lint", CLEAN, "--transcript", str(transcript)]) == 2
    assert "ends before it starts" in capsys.readouterr().err


def test_plan_without_a_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture[str]):
    assert main(["plan"]) == 2
    assert "the following arguments are required: COMMAND" in capsys.readouterr().err
