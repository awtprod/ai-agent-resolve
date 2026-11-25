"""``aar plan import-timeline`` on a project saved in a Resolve disk library (S25a).

The same synthetic timeline is written as a ``.drt`` and as a library ``Project.db``
(``tests/drp/library_support.py``); the import must give the same plan and diff from both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_timeline_import_cli import CLIPS, call, write_drt, write_plan
from timeline_import_support import DrtClip, drt_members, plan_for

from library_support import Tables, find, library_db  # isort: skip  (tests/drp on sys.path)


def both(tmp_path: Path, clips: list[DrtClip] = CLIPS, **kw: Any) -> tuple[Path, Path, Path]:
    members = drt_members(clips, **kw)
    drt = write_drt(tmp_path / "edit.drt", members)
    db = library_db(tmp_path / "Project.db", members)
    return drt, db, write_plan(tmp_path / "plan.json", plan_for(CLIPS))


def run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: Path, plan: Path, *extra: str
) -> tuple[int, dict[str, Any], str, str]:
    out_dir = tmp_path / f"out-{source.suffix.lstrip('.')}-{len(list(tmp_path.glob('out-*')))}"
    out_dir.mkdir()
    code, out, err = call(out_dir, capsys, source, plan, *extra)
    diff = {}
    if (out_dir / "edited.diff.json").exists():
        diff = json.loads((out_dir / "edited.diff.json").read_text())
        timeline = diff.pop("timeline")
        diff["timeline"] = {k: v for k, v in timeline.items() if k not in ("file", "sha256")}
    plan_text = (out_dir / "edited.plan.json").read_text() if diff else ""
    return code, diff, plan_text, out + err


def test_the_same_timeline_imports_the_same_from_drt_and_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drt, db, plan = both(tmp_path)
    a = run(tmp_path, capsys, drt, plan)
    b = run(tmp_path, capsys, db, plan)
    assert a[0] == b[0] == 0
    assert a[1] == b[1] and a[1]["unchanged"] is True
    assert a[2] == b[2]


def test_an_edit_saved_in_the_library_gives_the_same_diff_as_the_drt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edited = [DrtClip("D:\\Show\\A.mov", 0, 80, 30), CLIPS[1], CLIPS[2]]
    drt, db, plan = both(tmp_path, edited)
    a = run(tmp_path, capsys, drt, plan)
    b = run(tmp_path, capsys, db, plan)
    assert a[1] == b[1] and a[2] == b[2] and a[0] == b[0]
    assert b[1]["segments"][0]["status"] == "trimmed"


def test_a_trim_made_in_the_database_is_imported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    members = drt_members(CLIPS)

    def trim(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "t1i1")["Duration"] = "90"

    db = library_db(tmp_path / "Project.db", members, trim)
    plan = write_plan(tmp_path / "plan.json", plan_for(CLIPS))
    code, diff, _, _ = run(tmp_path, capsys, db, plan)
    assert code == 0
    first = diff["segments"][0]
    assert first["status"] == "trimmed" and first["tail_frames"] == 10


def test_a_project_with_several_timelines_needs_one_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, db, plan = both(tmp_path, timelines=("Edit", "Edit v2"))
    code, _, _, text = run(tmp_path, capsys, db, plan)
    assert code == 2 and "refused (several_timelines)" in text and "--timeline" in text
    code, diff, _, _ = run(tmp_path, capsys, db, plan, "--timeline", "Edit v2")
    assert code == 0 and diff["timeline"]["name"] == "Edit v2"


def test_a_project_being_saved_is_refused_as_busy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, db, plan = both(tmp_path)
    (tmp_path / "Project.db-journal").write_bytes(b"")
    code, diff, _, text = run(tmp_path, capsys, db, plan, "--wait", "0")
    assert code == 2 and diff == {} and "refused (library_busy)" in text
    assert "Project.db-journal" in text and "wait until Resolve has finished saving" in text
    code, _, _, text = run(tmp_path, capsys, db, plan, "--wait", "-1")
    assert code == 2 and "--wait" in text


def test_a_refused_project_is_unreadable_with_the_library_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = write_plan(tmp_path / "plan.json", plan_for(CLIPS))

    def old(tables: Tables) -> None:
        tables["SM_Project"][0]["ProjectVersion"] = 14

    db = library_db(tmp_path / "Project.db", drt_members(CLIPS), old)
    code, _, _, text = run(tmp_path, capsys, db, plan, "--wait", "0")
    assert code == 2 and "refused (unreadable_timeline)" in text
    assert "[unsupported_version]" in text


def test_a_library_problem_refuses_the_import_as_damaged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def dangle(tables: Tables) -> None:
        tables["Sm2TiItem"] = [r for r in tables["Sm2TiItem"] if r["Sm2TiItem_id"] != "t1i2"]

    db = library_db(tmp_path / "Project.db", drt_members(CLIPS), dangle)
    plan = write_plan(tmp_path / "plan.json", plan_for(CLIPS))
    code, _, _, text = run(tmp_path, capsys, db, plan)
    assert code == 2 and "refused (damaged_timeline)" in text
    assert "library-dangling-link" in text and "save it again" in text


def test_the_library_budgets_are_raised_from_import_timeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, db, plan = both(tmp_path)
    code, diff, _, text = run(tmp_path, capsys, db, plan, "--wait", "0", "--max-rows", "5")
    assert code == 2 and diff == {} and "raise the limit with --max-rows" in text
    code, _, _, _ = run(tmp_path, capsys, db, plan, "--wait", "0", "--max-rows", "1000")
    assert code == 0
