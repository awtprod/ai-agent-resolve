"""S25c2: ``aar plan export --format library``: an EditPlan as a project in a disk library.

Synthetic seeds, templates and facts (``tests/drp/libwrite_support.py``): a 23.976 seed at
01:00:00:00, 3840x2160, holding a template camera clip and a linked cut; two camera files of the
same verified kind as the plan's sources. Every refusal of the export, the report, the CLI and a
round-trip property (random cuts within the supported set, written, read back, imported: equal
to the plan)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drp"))

from libwrite_support import (
    facts_lines,
    linked_pair,
    make_library,
    mp4_facts,
    mp4_template,
    mp4_template_facts,
    write_seed,
)

from aar.cli import main as aar_main
from aar.drp.library import LibraryOptions
from aar.drp.mediapool.gate import load_facts
from aar.plan.export.errors import ExportError
from aar.plan.export.library import export_library
from aar.plan.model import EditPlan
from aar.plan.profile import parse_profile
from aar.plan.timeline_import import import_timeline, read_timeline

pytest.importorskip("zstandard")

OPTS = LibraryOptions(settle_seconds=0.0)
CLIP = mp4_template()
CAM_A = "X:\\Show\\Footage\\camera-a.MP4"
CAM_C = "X:\\Show\\Footage\\camera-c.MP4"
PROFILE = {
    "name": "export-test",
    "qa": {
        "longform_lufs": -22.0,
        "short_lufs": -14.5,
        "lufs_tolerance": 0.5,
        "review_mp4_max_mb": 20,
    },
    "rules": {"ranges-valid": {}},
    "export": {"timeline_start": "01:00:00:00"},
}


def _plan(**change: Any) -> EditPlan:
    data: dict[str, Any] = {
        "version": 2,
        "project": "Show",
        "timeline": "Cut 1",
        "kind": "longform",
        "fps": "24000/1001",
        "resolution": {"width": 3840, "height": 2160},
        "profile": "export-test",
        "sources": [
            {"id": "a", "media": CAM_A, "duration": 3000},
            {"id": "c", "media": CAM_C, "duration": 3000},
        ],
        "segments": [
            {"track": 1, "source": "a", "src_in": 0, "src_out": 48},
            {"track": 1, "source": "c", "src_in": 100, "src_out": 160},
            {"track": 1, "source": "a", "src_in": 500, "src_out": 524},
        ],
    }
    data.update(change)
    return EditPlan.model_validate(data)


def _setup(tmp_path: Path, **seed_kwargs: Any) -> dict[str, Path]:
    seeds = tmp_path / "seeds"
    seeds.mkdir(exist_ok=True)
    seed_kwargs.setdefault("clips", [CLIP])
    seed_kwargs.setdefault("items", linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0))
    seed = write_seed(seeds / "seed.db", **seed_kwargs)
    facts = facts_lines(
        seeds / "facts.jsonl",
        mp4_facts(),
        mp4_facts(path=CAM_C),
        mp4_template_facts(),
    )
    library = make_library(tmp_path / "lib")
    return {"seed": seed, "facts": facts, "library": library}


def _export(paths: dict[str, Path], plan: EditPlan | None = None, **kwargs: Any) -> Any:
    kwargs.setdefault("project_name", "Show cut")
    kwargs.setdefault("allow_unopened", True)
    return export_library(
        plan or _plan(),
        parse_profile(PROFILE),
        library=paths["library"],
        seed=paths["seed"],
        facts=load_facts(paths["facts"]),
        templates=kwargs.pop("templates", [paths["seed"]]),
        options=OPTS,
        **kwargs,
    )


def _projects(paths: dict[str, Path]) -> list[str]:
    return sorted(
        p.name
        for p in (paths["library"] / "Resolve Projects" / "Users" / "guest" / "Projects").iterdir()
    )


def test_a_plan_becomes_a_library_project_that_imports_back_unchanged(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    out = _export(paths)
    report = out.report
    assert report["format"] == "library"
    assert report["project"]["name"] == "Show cut" and _projects(paths) == ["Show cut"]
    assert report["self_check"]["import"]["unchanged"] is True
    assert report["exported"] == {
        "segments": 3,
        "outro": False,
        "picture_items": 3,
        "sound_items": 3,
    }
    assert [m["status"] for m in report["media"]] == ["verified", "verified"]
    assert [op["operation"] for op in report["operations"]] == [
        "new_project_identity",
        "rename_timeline",
        "remove_items",
        "add_media",
        "add_linked_items",
        "set_extents",
    ]
    assert report["operations_not_opened"] == ["rename_timeline", "add_media"]
    timeline = read_timeline(Path(report["project"]["path"]), library_options=OPTS)
    result = import_timeline(_plan(), timeline.timeline, timeline.media)
    assert result.diff["unchanged"] is True
    assert result.plan == _plan()


def test_unopened_operations_need_the_flag(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    with pytest.raises(ExportError) as info:
        _export(paths, allow_unopened=False)
    assert info.value.code == "operation_not_opened"
    assert "add_media" in info.value.details["operations"]
    assert _projects(paths) == []


def test_gaps_are_carried(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    plan = _plan(
        segments=[
            {"track": 1, "source": "a", "src_in": 0, "src_out": 48, "record_in": 10},
            {"track": 1, "source": "c", "src_in": 100, "src_out": 160, "record_in": 100},
        ]
    )
    out = _export(paths, plan)
    assert "leave_gaps" in out.report["operations_not_opened"]


@pytest.mark.parametrize(
    ("change", "feature"),
    [
        (
            {"transitions": [{"after": 0, "video": "dissolve", "video_frames": 6}]},
            "join transition",
        ),
        (
            {
                "titles": [
                    {"kind": "lower-third", "lines": ["Hi"], "record_in": 0, "record_out": 24}
                ]
            },
            "title",
        ),
        (
            {
                "segments": [
                    {"track": 1, "source": "a", "src_in": 0, "src_out": 48},
                    {"track": 2, "source": "c", "src_in": 0, "src_out": 24, "record_in": 0},
                ]
            },
            "off V1",
        ),
    ],
)
def test_a_feature_that_would_change_the_cut_is_refused(
    tmp_path: Path, change: dict[str, Any], feature: str
) -> None:
    paths = _setup(tmp_path)
    with pytest.raises(ExportError) as info:
        _export(paths, _plan(**change))
    assert info.value.code == "unsupported_plan_feature"
    assert any(feature in f["feature"] for f in info.value.details["features"])


def test_gain_and_markers_are_reported_as_not_carried(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    plan = _plan(
        segments=[{"track": 1, "source": "a", "src_in": 0, "src_out": 48, "gain_db": -3.0}],
        markers=[{"record": 10, "name": "check", "color": "Blue"}],
    )
    report = _export(paths, plan).report
    features = {(i["feature"], i["location"]) for i in report["not_carried"]}
    assert ("gain", "segments[0]") in features
    assert ("marker", "markers[0]") in features
    assert report["self_check"]["import"]["unchanged"] is True


def test_a_source_without_facts_is_refused(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    facts_lines(paths["facts"], mp4_facts(), mp4_template_facts())
    with pytest.raises(ExportError) as info:
        _export(paths)
    assert info.value.code == "facts_missing"
    assert info.value.details["sources"] == [{"source": "c", "path": CAM_C}]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        (
            {
                "video_r_frame_rate": "25/1",
                "video_avg_frame_rate": "25/1",
                "video_time_base": "1/25",
                "video_duration_ts": 3000,
            },
            "fps",
        ),
        (
            {"video_duration_ts": 100100, "audio_duration_ts": 200200, "data_duration_ts": 100100},
            "frames",
        ),
    ],
)
def test_facts_of_another_file_are_refused(
    tmp_path: Path, change: dict[str, Any], fragment: str
) -> None:
    paths = _setup(tmp_path)
    facts_lines(paths["facts"], mp4_facts(), mp4_facts(path=CAM_C, **change), mp4_template_facts())
    with pytest.raises(ExportError) as info:
        _export(paths)
    assert info.value.code == "facts_mismatch"
    assert fragment in info.value.message


def test_a_stated_start_timecode_must_match_the_facts(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    plan = _plan(
        sources=[
            {"id": "a", "media": CAM_A, "duration": 3000, "start_timecode": 86400},
            {"id": "c", "media": CAM_C, "duration": 3000},
        ]
    )
    with pytest.raises(ExportError) as info:
        _export(paths, plan)
    assert info.value.code == "facts_mismatch"


@pytest.mark.parametrize(
    ("seed_kwargs", "fragment"),
    [
        ({"rate": 25.0, "rate_text": "25"}, "fps"),
        ({"start_frame": 0}, "starts at frame 0"),
        ({"resolution": (1920, 1080)}, "1920x1080"),
        ({"resolution": (0, 0)}, "does not store"),
    ],
)
def test_a_seed_with_other_settings_is_refused(
    tmp_path: Path, seed_kwargs: dict[str, Any], fragment: str
) -> None:
    paths = _setup(tmp_path, **seed_kwargs)
    with pytest.raises(ExportError) as info:
        _export(paths)
    assert info.value.code == "seed_mismatch"
    assert fragment in info.value.message
    assert _projects(paths) == []


def test_a_stated_seed_resolution_is_checked_and_reported(tmp_path: Path) -> None:
    paths = _setup(tmp_path, resolution=(0, 0))
    with pytest.raises(ExportError) as info:
        _export(paths, seed_resolution="1920x1080")
    assert info.value.code == "seed_mismatch"
    out = _export(paths, seed_resolution="3840x2160")
    assert "stated" in out.report["timeline"]["resolution_checked_against"]


def test_media_that_would_not_be_verified_needs_the_flag(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    facts_lines(paths["facts"], mp4_facts(), mp4_facts(path=CAM_C))  # no template facts
    with pytest.raises(ExportError) as info:
        _export(paths)
    assert info.value.code == "media_refused"
    assert info.value.details["media_code"] == "unverified"
    out = _export(paths, allow_unverified_media=True)
    assert {m["status"] for m in out.report["media"]} == {"unverified"}
    assert all(m["unverified"] for m in out.report["media"])


def test_an_existing_project_name_is_refused(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _export(paths)
    with pytest.raises(ExportError) as info:
        _export(paths, project_name="SHOW CUT")
    assert info.value.code == "project_exists"


def test_a_plan_with_lint_errors_is_refused(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    plan = _plan(segments=[{"track": 1, "source": "a", "src_in": 0, "src_out": 4000}])
    with pytest.raises(ExportError) as info:
        _export(paths, plan)
    assert info.value.code == "lint_errors"


@pytest.mark.parametrize("seed_number", range(12))
def test_random_cuts_round_trip_through_a_library_project(tmp_path: Path, seed_number: int) -> None:
    rng = random.Random(seed_number)
    segments = []
    cursor = 0
    for _ in range(rng.randint(1, 6)):
        source = rng.choice(["a", "c"])
        length = rng.randint(1, 120)
        src_in = rng.randint(0, 3000 - length)
        seg: dict[str, Any] = {
            "track": 1,
            "source": source,
            "src_in": src_in,
            "src_out": src_in + length,
        }
        if rng.random() < 0.3:
            cursor += rng.randint(1, 30)
            seg["record_in"] = cursor
        segments.append(seg)
        cursor += length
    plan = _plan(segments=segments)
    paths = _setup(tmp_path)
    out = _export(paths, plan)
    timeline = read_timeline(Path(out.report["project"]["path"]), library_options=OPTS)
    result = import_timeline(plan, timeline.timeline, timeline.media)
    assert result.diff["unchanged"] is True
    assert result.plan == plan


# -- CLI -----------------------------------------------------------------------------------------


def _main(args: list[str]) -> int:
    return aar_main(["plan", *args])


def _cli(
    tmp_path: Path, paths: dict[str, Path], *extra: str, plan: EditPlan | None = None
) -> list[str]:
    plan_file = tmp_path / "plan.json"
    plan_file.write_text((plan or _plan()).model_dump_json(), encoding="utf-8")
    profile = tmp_path / "profile.toml"
    profile.write_text(
        'name = "export-test"\n[qa]\nlongform_lufs = -22.0\nshort_lufs = -14.5\n'
        "lufs_tolerance = 0.5\nreview_mp4_max_mb = 20\n[rules.ranges-valid]\n"
        '[export]\ntimeline_start = "01:00:00:00"\n',
        encoding="utf-8",
    )
    return [
        "export",
        str(plan_file),
        "--profile",
        str(profile),
        "--format",
        "library",
        "--library",
        str(paths["library"]),
        "--seed",
        str(paths["seed"]),
        "--facts",
        str(paths["facts"]),
        "--templates",
        str(paths["seed"]),
        "--max-read-seconds",
        "60",
        *extra,
    ]


def test_the_cli_writes_the_project_and_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    report = tmp_path / "out" / "cut.report.json"
    report.parent.mkdir()
    args = _cli(
        tmp_path,
        paths,
        "--project-name",
        "CLI cut",
        "--allow-unopened-operations",
        "--report",
        str(report),
        "--json",
    )
    assert _main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(report.read_text(encoding="utf-8")) == printed
    assert printed["project"]["name"] == "CLI cut"


def test_the_cli_text_names_the_unopened_operations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    assert (
        _main(_cli(tmp_path, paths, "--project-name", "Text cut", "--allow-unopened-operations"))
        == 0
    )
    out = capsys.readouterr().out
    assert "NOT YET OPENED IN RESOLVE: rename_timeline, add_media" in out
    assert "self-check: ok" in out


def test_the_cli_refuses_with_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _setup(tmp_path)
    assert _main(_cli(tmp_path, paths, "--project-name", "No flag", "--json")) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["error"]["code"] == "operation_not_opened"


@pytest.mark.parametrize(
    "extra",
    [
        [],  # no --project-name
        ["--project-name", "x", "--out", "cut.fcpxml"],
    ],
)
def test_the_cli_needs_the_library_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    paths = _setup(tmp_path)
    assert _main(_cli(tmp_path, paths, *extra)) == 2
    assert "--format library needs" in capsys.readouterr().err


def test_library_arguments_do_not_go_with_fcpxml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(_plan().model_dump_json(), encoding="utf-8")
    code = _main(
        [
            "export",
            str(plan_file),
            "--profile",
            "ccn",
            "--out",
            str(tmp_path / "x.fcpxml"),
            "--seed",
            "s.db",
        ]
    )
    assert code == 2
    assert "only go with --format library" in capsys.readouterr().err


def test_an_existing_report_path_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    report = tmp_path / "taken.json"
    report.write_text("{}", encoding="utf-8")
    code = _main(
        _cli(
            tmp_path,
            paths,
            "--project-name",
            "R",
            "--allow-unopened-operations",
            "--report",
            str(report),
        )
    )
    assert code == 2
    assert _projects(paths) == []
    assert report.read_text(encoding="utf-8") == "{}"
