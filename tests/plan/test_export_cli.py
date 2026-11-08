"""``aar plan export`` end to end: write, refuse, media map, --verify, determinism."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from apply_builders import plan_data, seg, source
from export_support import EXPORT_FIXTURES, GOLDEN_PLANS
from plan_support import FIXTURES

from aar.cli import main
from aar.plan.cli import report_path

LONGFORM = GOLDEN_PLANS["ccn_longform_v2"]
OFFSET_OLD = 'offset="86558472/24000s"'
OFFSET_NEW = 'offset="86559473/24000s"'


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["plan", "export", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_plan(tmp_path: Path, media: str, name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(plan_data(sources=[source("a", media)], segments=[seg()])))
    return path


def test_report_path_unit() -> None:
    assert report_path(Path("a/b/x.fcpxml")) == Path("a/b/x.report.json")


def test_export_writes_fcpxml_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cut.fcpxml"
    code, stdout, _ = run(capsys, str(LONGFORM), "--out", str(out))
    assert code == 0
    assert out.is_file()
    report_file = tmp_path / "cut.report.json"
    assert report_file.is_file()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["file"] == "cut.fcpxml"
    assert report["self_check"]["ok"] is True
    assert report["fcpxml_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert stdout.startswith("exported FCPXML 1.10: ")
    lines = stdout.splitlines()
    assert any(line.startswith("self-check: ok") for line in lines)
    assert any(line.startswith("not expressible in FCPXML (") for line in lines)


def test_export_not_expressible_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, stdout, _ = run(capsys, str(LONGFORM), "--out", str(tmp_path / "cut.fcpxml"))
    assert any(line.startswith("not expressible in FCPXML (23):") for line in stdout.splitlines())


def test_export_json_matches_report_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cut.fcpxml"
    code, stdout, _ = run(capsys, str(LONGFORM), "--out", str(out), "--json")
    assert code == 0
    assert json.loads(stdout) == json.loads((tmp_path / "cut.report.json").read_text("utf-8"))


def test_existing_out_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cut.fcpxml"
    out.write_bytes(b"precious")
    code, _, stderr = run(capsys, str(LONGFORM), "--out", str(out))
    assert code == 2
    assert "refused (output_exists)" in stderr
    assert out.read_bytes() == b"precious"
    assert not (tmp_path / "cut.report.json").exists()


def test_existing_report_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cut.fcpxml"
    report = tmp_path / "cut.report.json"
    report.write_bytes(b"precious")
    code, _, stderr = run(capsys, str(LONGFORM), "--out", str(out))
    assert code == 2
    assert "refused (output_exists)" in stderr
    assert report.read_bytes() == b"precious"
    assert not out.exists()


def test_overwrite_replaces_both(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ccn_longform_v2.fcpxml"
    report = tmp_path / "ccn_longform_v2.report.json"
    out.write_bytes(b"old")
    report.write_bytes(b"old")
    code, _, _ = run(capsys, str(LONGFORM), "--out", str(out), "--overwrite")
    assert code == 0
    assert out.read_bytes() == (EXPORT_FIXTURES / "ccn_longform_v2.fcpxml").read_bytes()
    assert report.read_bytes() != b"old"
    assert json.loads(report.read_text(encoding="utf-8"))["file"] == "ccn_longform_v2.fcpxml"


def test_out_directory_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "adir"
    out.mkdir()
    code, _, stderr = run(capsys, str(LONGFORM), "--out", str(out))
    assert code == 2
    assert "output_exists" in stderr


def test_out_in_missing_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "missing" / "cut.fcpxml"
    code, _, stderr = run(capsys, str(LONGFORM), "--out", str(out))
    assert code == 2
    assert stderr.startswith("error:")


def test_lint_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = FIXTURES / "short_broken.plan.json"
    out = tmp_path / "cut.fcpxml"
    code, _, stderr = run(capsys, str(broken), "--out", str(out))
    assert code == 2
    assert "refused (lint_errors)" in stderr
    indented = [line for line in stderr.splitlines() if line.startswith("  ") and ": " in line]
    assert indented
    assert not out.exists()
    assert not (tmp_path / "cut.report.json").exists()


def test_lint_refusal_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = FIXTURES / "short_broken.plan.json"
    out = tmp_path / "cut.fcpxml"
    code, stdout, _ = run(capsys, str(broken), "--out", str(out), "--json")
    assert code == 2
    payload: dict[str, Any] = json.loads(stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "lint_errors"
    assert payload["error"]["details"]["errors"]
    assert not out.exists()


def test_lint_warnings_not_fatal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cut.fcpxml"
    code, _, stderr = run(capsys, str(GOLDEN_PLANS["ccn_2997"]), "--out", str(out))
    assert code == 0
    assert "warning timejump-close-to-wide" in stderr


def test_media_map_required_then_applied(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = write_plan(tmp_path, "/srv/media/ep1/a.mov")
    code, _, stderr = run(capsys, str(plan), "--out", str(tmp_path / "x.fcpxml"))
    assert code == 2
    assert "refused (media_not_on_edit_pc)" in stderr

    media_map = tmp_path / "map.json"
    media_map.write_text(json.dumps({"/srv/media": "E:\\Media"}))
    out = tmp_path / "y.fcpxml"
    code, _, _ = run(capsys, str(plan), "--out", str(out), "--media-map", str(media_map))
    assert code == 0
    report = json.loads((tmp_path / "y.report.json").read_text(encoding="utf-8"))
    assert report["media"][0]["rule"] == "media-map"
    assert report["media"][0]["src"] == "file:///E:/Media/ep1/a.mov"
    assert "file:///E:/Media/ep1/a.mov" in out.read_text(encoding="utf-8")


def test_bad_media_map(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = write_plan(tmp_path, "/srv/media/ep1/a.mov")
    media_map = tmp_path / "map.json"
    media_map.write_text("{not json")
    code, _, stderr = run(
        capsys, str(plan), "--out", str(tmp_path / "x.fcpxml"), "--media-map", str(media_map)
    )
    assert code == 2
    assert "refused (bad_media_map)" in stderr


def test_pool_source_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = write_plan(tmp_path, "Ep1 Multicam")
    code, _, stderr = run(capsys, str(plan), "--out", str(tmp_path / "x.fcpxml"))
    assert code == 2
    assert "refused (pool_source_not_exportable)" in stderr


def test_invalid_plan_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = tmp_path / "bad.json"
    plan.write_text('{"nope": 1}')
    code, _, stderr = run(capsys, str(plan), "--out", str(tmp_path / "x.fcpxml"))
    assert code == 2
    assert stderr.startswith("error:")


def test_missing_plan_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, _ = run(capsys, str(tmp_path / "nope.json"), "--out", str(tmp_path / "x.fcpxml"))
    assert code == 2


@pytest.mark.parametrize("name", sorted(GOLDEN_PLANS))
def test_verify_golden_ok(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    golden = EXPORT_FIXTURES / f"{name}.fcpxml"
    code, stdout, _ = run(capsys, str(GOLDEN_PLANS[name]), "--verify", str(golden))
    assert code == 0
    assert stdout.startswith("verify ")
    assert ": ok (" in stdout


@pytest.mark.parametrize("name", sorted(GOLDEN_PLANS))
def test_verify_golden_json(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    golden = EXPORT_FIXTURES / f"{name}.fcpxml"
    code, stdout, _ = run(capsys, str(GOLDEN_PLANS[name]), "--verify", str(golden), "--json")
    assert code == 0
    payload = json.loads(stdout)
    assert payload["ok"] is True
    assert payload["problems"] == []


def test_verify_tampered_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    text = (EXPORT_FIXTURES / "ccn_longform_v2.fcpxml").read_text(encoding="utf-8")
    assert OFFSET_OLD in text
    tampered = tmp_path / "tampered.fcpxml"
    tampered.write_text(text.replace(OFFSET_OLD, OFFSET_NEW, 1), encoding="utf-8")
    code, stdout, _ = run(capsys, str(LONGFORM), "--verify", str(tampered))
    assert code == 1
    assert "MISMATCH" in stdout


def test_verify_not_xml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junk = tmp_path / "junk.fcpxml"
    junk.write_text("this is not xml")
    code, stdout, _ = run(capsys, str(LONGFORM), "--verify", str(junk))
    assert code == 1
    assert "not well-formed" in stdout


def test_verify_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, _ = run(capsys, str(LONGFORM), "--verify", str(tmp_path / "nope.fcpxml"))
    assert code == 2


def test_out_and_verify_together_is_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(capsys, str(LONGFORM), "--out", str(tmp_path / "a"), "--verify", "b")
    assert code == 2 and "not allowed with argument" in err
    assert not (tmp_path / "a").exists()


def test_neither_out_nor_verify_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, str(LONGFORM))
    assert code == 2 and "one of the arguments --out --verify is required" in err


def test_format_otio_is_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, str(LONGFORM), "--out", str(tmp_path / "a"), "--format", "otio")
    assert code == 2 and "invalid choice: 'otio'" in err
    assert not (tmp_path / "a").exists()


def test_export_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outs = []
    for sub in ("one", "two"):
        directory = tmp_path / sub
        directory.mkdir()
        out = directory / "cut.fcpxml"
        assert run(capsys, str(LONGFORM), "--out", str(out))[0] == 0
        outs.append(out)
    assert outs[0].read_bytes() == outs[1].read_bytes()
    assert report_path(outs[0]).read_bytes() == report_path(outs[1]).read_bytes()
