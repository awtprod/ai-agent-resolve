"""S25c2 fix round 1, the export and its CLI: the FCPXML usage error as before the library
format (item 8), what a failed export leaves named in its text and JSON (item 3), a name
refused before any work (item 5), and the validator's surviving mutants of the export (the
import diff's keys, an audio-only source)."""

from __future__ import annotations

import argparse
import errno
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drp"))

import test_export_library as base
from libwrite_support import facts_lines, mp4_facts, mp4_template_facts
from mediapool_support import wav_facts

import aar.drp.libwrite.place as place
import aar.plan.timeline_import as timeline_import
from aar.plan.export.errors import ExportError

pytest.importorskip("zstandard")


# -- 8. the FCPXML export's usage error is the one it printed before the library format ---------


def _dev_parser() -> argparse.ArgumentParser:
    """``aar plan export`` as it was before the library format (origin/dev), argument for
    argument: the usage error the command must still print without --out or --verify."""
    export = argparse.ArgumentParser(prog="aar plan export")
    export.add_argument("plan", type=Path, help="EditPlan JSON file.")
    export.add_argument("--profile", help="Profile name or .toml path (default: the plan's).")
    export.add_argument("--format", choices=["fcpxml"], default="fcpxml", help="Output format.")
    target = export.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="FCPXML file to write (never overwritten).")
    target.add_argument(
        "--verify", type=Path, metavar="FILE", help="Read FILE back and compare it with the plan."
    )
    export.add_argument("--media-map", type=Path, help="JSON object.")
    export.add_argument("--transcript", type=Path, help="Transcript JSON, for the lint.")
    export.add_argument("--laughs", type=Path, help="Laugh map JSON, for the lint.")
    export.add_argument("--overwrite", action="store_true", help="Replace.")
    export.add_argument("--allow-unknown-timecode", action="store_true", help="Export anyway.")
    export.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return export


@pytest.mark.parametrize("extra", [[], ["--json"], ["--format", "fcpxml"]])
def test_the_fcpxml_export_without_a_target_prints_the_usage_error_it_always_did(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    stream = io.StringIO()
    old, sys.stderr = sys.stderr, stream
    try:
        with pytest.raises(SystemExit) as stopped:
            _dev_parser().parse_args(["plan.json", *extra])
    finally:
        sys.stderr = old
    assert stopped.value.code == 2
    expected = stream.getvalue()
    assert expected.endswith(
        "aar plan export: error: one of the arguments --out --verify is required\n"
    )
    code = base._main(["export", str(tmp_path / "missing-plan.json"), *extra])
    out, err = capsys.readouterr()
    assert (code, out, err) == (2, "", expected)


# -- 3. what a failed export leaves in the library is named in its text and its JSON ------------


def _failing_rmtree(exc: BaseException) -> Any:
    def rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        raise exc

    return rmtree


def test_a_refused_export_names_the_staging_it_could_not_remove(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = base._setup(tmp_path)
    monkeypatch.setattr(place.shutil, "rmtree", _failing_rmtree(OSError(errno.EBUSY, "busy")))
    assert base._main(base._cli(tmp_path, paths, "--project-name", "Left", "--json")) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "operation_not_opened"
    assert ".aar-staging" in error["details"]["residue"]
    assert error["details"]["residue"] in error["message"]
    assert base._main(base._cli(tmp_path, paths, "--project-name", "Left too")) == 2
    err = capsys.readouterr().err
    assert "left in the library" in err and ".aar-staging" in err


def test_an_interrupted_export_says_what_it_left_and_exits_130(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = base._setup(tmp_path)
    monkeypatch.setattr(place.shutil, "rmtree", _failing_rmtree(KeyboardInterrupt()))
    assert base._main(base._cli(tmp_path, paths, "--project-name", "Stopped", "--json")) == 130
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "interrupted"
    assert ".aar-staging" in error["details"]["residue"]
    assert "left in the library" in error["message"]


def test_an_export_placed_despite_a_late_rename_error_reports_it_placed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = base._setup(tmp_path)
    real = place._move

    def late(folder: Path, target: Path) -> None:
        real(folder, target)
        raise OSError(errno.EIO, "the share answered late")

    monkeypatch.setattr(place, "_move", late)
    out = base._export(paths, project_name="Late")
    assert base._projects(paths) == ["Late"]
    assert any("in place" in note for note in out.report["placement"]["cleanup"])


# -- 5. a name the share cannot hold is refused before any work ----------------------------------


def test_a_name_too_long_once_encoded_is_refused_before_the_facts_are_read(
    tmp_path: Path,
) -> None:
    paths = base._setup(tmp_path)
    facts_lines(paths["facts"])  # no facts at all: the name must be refused first
    with pytest.raises(ExportError) as info:
        base._export(paths, project_name="\U0001f600" * 70)
    assert info.value.code == "bad_project_name"
    assert sorted(p.name for p in (paths["library"] / "Resolve Projects").iterdir()) == ["Users"]


# -- 10. the validator's surviving mutants of the export -----------------------------------------


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"unchanged": False}, "reports changes"),
        ({"not_carried": [{}]}, "not_carried: 1"),
        ({"foreign": [{}]}, "foreign: 1"),
        ({"added": [{}]}, "added: 1"),
        ({"dropped": [{}]}, "dropped: 1"),
        ({"track_mismatch": [{}]}, "track_mismatch: 1"),
        ({"export_errors": [{}]}, "does not lay out"),
    ],
)
def test_an_import_that_does_not_report_the_plan_unchanged_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any], problem: str
) -> None:
    paths = base._setup(tmp_path)
    real = timeline_import.import_timeline

    def changed(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        result.diff.update(change)
        return result

    monkeypatch.setattr(timeline_import, "import_timeline", changed)
    with pytest.raises(ExportError) as info:
        base._export(paths)
    assert info.value.code == "self_check_failed"
    assert any(problem in p for p in info.value.details["problems"])
    assert base._projects(paths) == []


def test_an_audio_only_source_is_refused(tmp_path: Path) -> None:
    paths = base._setup(tmp_path)
    sound = "X:\\Show\\Sound\\take-one.wav"
    facts_lines(paths["facts"], mp4_facts(), wav_facts(path=sound), mp4_template_facts())
    plan = base._plan(
        sources=[
            {"id": "a", "media": base.CAM_A, "duration": 3000},
            {"id": "c", "media": sound, "duration": 3000},
        ],
        segments=[
            {"track": 1, "source": "a", "src_in": 0, "src_out": 48},
            {"track": 1, "source": "c", "src_in": 0, "src_out": 48},
        ],
    )
    with pytest.raises(ExportError) as info:
        base._export(paths, plan)
    assert info.value.code == "facts_mismatch" and "audio file" in info.value.message
