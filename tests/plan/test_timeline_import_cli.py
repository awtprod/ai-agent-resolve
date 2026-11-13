"""``aar plan import-timeline`` on small synthetic .drt files (built with the tests/drp builders),
every refusal included."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from timeline_import_support import DrtClip, drt_members, plan_for

from aar.cli import main
from aar.plan.model import EditPlan

#: Two structural rules only: the synthetic plans need not follow every CCN house rule.
PROFILE = """name = "import-test"
[qa]
longform_lufs = -22.0
short_lufs = -14.5
lufs_tolerance = 0.5
review_mp4_max_mb = 20
[rules.ranges-valid]
[rules.chronological]
tolerance_frames = 0
"""

CLIPS = [
    DrtClip("D:\\Show\\A.mov", 0, 100, 10),
    DrtClip("D:\\Show\\B.mov", 100, 50, 300),
    DrtClip("D:\\Show\\A.mov", 150, 75, 500),
]


def write_drt(path: Path, members: dict[str, str | bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data.encode() if isinstance(data, str) else data)
    return path


def profile(tmp_path: Path) -> Path:
    path = tmp_path / "import-test.toml"
    if not path.exists():
        path.write_text(PROFILE, encoding="utf-8")
    return path


def write_plan(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    drt: Path,
    plan: Path,
    *extra: str,
) -> tuple[int, str, str]:
    code = main(
        [
            "plan",
            "import-timeline",
            str(drt),
            "--plan",
            str(plan),
            "--out",
            str(tmp_path / "edited.plan.json"),
            "--report",
            str(tmp_path / "edited.diff.json"),
            "--profile",
            str(profile(tmp_path)),
            *extra,
        ]
    )
    out, err = capsys.readouterr()
    return code, out, err


def setup(
    tmp_path: Path, clips: list[DrtClip] = CLIPS, plan: dict[str, Any] | None = None, **kw: Any
) -> tuple[Path, Path]:
    drt = write_drt(tmp_path / "edit.drt", drt_members(clips, **kw))
    data = plan if plan is not None else plan_for(CLIPS)
    return drt, write_plan(tmp_path / "plan.json", data)


def test_unedited_timeline_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    drt, plan = setup(tmp_path)
    code, out, err = call(tmp_path, capsys, drt, plan)
    assert (code, err) == (0, "")
    assert "no changes" in out and "3 unchanged" in out and "lint: ok" in out
    edited = EditPlan.model_validate_json((tmp_path / "edited.plan.json").read_text())
    original = EditPlan.model_validate_json(plan.read_text())
    assert edited.segments == original.segments and edited == original
    diff = json.loads((tmp_path / "edited.diff.json").read_text())
    assert diff["unchanged"] is True
    assert diff["timeline"]["name"] == "Edit" and diff["timeline"]["start_frame"] == 90000
    assert diff["timeline"]["file"] == str(drt) and len(diff["timeline"]["sha256"]) == 64


def test_a_head_trim_through_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    edited = [
        CLIPS[0],
        DrtClip("D:\\Show\\B.mov", 100, 19, 331),
        DrtClip("D:\\Show\\A.mov", 119, 75, 500),
    ]
    drt, plan = setup(tmp_path, edited)
    code, out, _ = call(tmp_path, capsys, drt, plan, "--json")
    assert code == 0
    diff = json.loads(out)
    assert diff == json.loads((tmp_path / "edited.diff.json").read_text())
    assert [r["status"] for r in diff["segments"]] == ["unchanged", "trimmed", "moved"]
    assert diff["segments"][1]["head_frames"] == 31
    assert diff["segments"][2]["new"][0]["record_in"] == 119


def test_text_output_names_the_changes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    drt, plan = setup(tmp_path, [CLIPS[0], DrtClip("D:\\Show\\A.mov", 100, 75, 500)])
    code, out, _ = call(tmp_path, capsys, drt, plan)
    assert code == 0
    assert "segments[1] removed: 100..150 -> gone" in out
    assert "segments[2] moved: 150..225 -> [1] 100..175" in out


def test_lint_errors_still_write_the_plan_and_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = plan_for(CLIPS)
    for source in data["sources"]:
        source["recording_id"] = "rec1"
    swapped = [DrtClip("D:\\Show\\A.mov", 0, 75, 500), DrtClip("D:\\Show\\A.mov", 75, 100, 10)]
    drt, plan = setup(tmp_path, swapped, data)
    code, out, _ = call(tmp_path, capsys, drt, plan)
    assert code == 1
    assert "lint: ERRORS" in out
    diff = json.loads((tmp_path / "edited.diff.json").read_text())
    assert diff["lint_errors"] is True
    assert any(f["rule"] == "chronological" for f in diff["lint"]["findings"])
    assert (tmp_path / "edited.plan.json").is_file()


def test_several_timelines_need_a_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    drt, plan = setup(tmp_path, timelines=("Edit", "Edit v2"))
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (several_timelines)" in err and "fix:" in err
    assert not (tmp_path / "edited.plan.json").exists()
    code, out, _ = call(tmp_path, capsys, drt, plan, "--timeline", "Edit v2")
    assert code == 0 and "'Edit v2'" in out
    (tmp_path / "edited.plan.json").unlink()
    (tmp_path / "edited.diff.json").unlink()
    code, _, err = call(tmp_path, capsys, drt, plan, "--timeline", "Nope")
    assert code == 2 and "refused (timeline_not_found)" in err


def test_an_unreadable_file_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, plan = setup(tmp_path)
    junk = tmp_path / "junk.drt"
    junk.write_bytes(b"not a zip at all")
    code, _, err = call(tmp_path, capsys, junk, plan)
    assert code == 2 and "refused (unreadable_timeline)" in err
    code, _, err = call(tmp_path, capsys, tmp_path / "missing.drt", plan)
    assert code == 2 and "refused (unreadable_timeline)" in err
    empty = write_drt(tmp_path / "empty.drt", drt_members(CLIPS, timelines=()))
    code, _, err = call(tmp_path, capsys, empty, plan)
    assert code == 2 and "refused (no_timeline)" in err


def test_another_frame_rate_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    drt, plan = setup(tmp_path, rate=24.0)
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (fps_mismatch)" in err
    code, out, _ = call(tmp_path, capsys, drt, plan, "--json")
    body = json.loads(out)
    assert code == 2 and body["ok"] is False and body["error"]["code"] == "fps_mismatch"


def test_media_at_another_rate_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drt, plan = setup(tmp_path, media_rate=30.0)
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (media_fps_mismatch)" in err


def test_a_range_past_the_source_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = plan_for(CLIPS)
    data["sources"][0]["duration"] = 560
    drt, plan = setup(tmp_path, plan=data)
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (source_range_outside)" in err


@pytest.mark.parametrize("existing", ["edited.plan.json", "edited.diff.json"])
def test_an_existing_output_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], existing: str
) -> None:
    drt, plan = setup(tmp_path)
    (tmp_path / existing).write_text("keep me")
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (output_exists)" in err
    assert (tmp_path / existing).read_text() == "keep me"
    other = "edited.diff.json" if existing == "edited.plan.json" else "edited.plan.json"
    assert not (tmp_path / other).exists()


def test_the_same_path_for_both_outputs_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drt, plan = setup(tmp_path)
    same = str(tmp_path / "x.json")
    code = main(
        ["plan", "import-timeline", str(drt), "--plan", str(plan), "--out", same, "--report", same]
    )
    assert code == 2 and "refused (output_exists)" in capsys.readouterr().err


def test_an_unreadable_plan_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drt, _ = setup(tmp_path)
    bad = write_plan(tmp_path / "bad.json", {"version": 2})
    code, _, err = call(tmp_path, capsys, drt, bad)
    assert code == 2 and "input failed validation" in err


# -- fix round (validator H3, M1, M2, L3, L6) --------------------------------------------------


def damaged(members: dict[str, str | bytes], how: str) -> dict[str, str | bytes]:
    key = next(k for k in members if k.startswith("SeqContainer/"))
    text = members[key]
    assert isinstance(text, str)
    if how == "missing":
        return {k: v for k, v in members.items() if k != key}
    if how == "cut_after_first_item":
        return {**members, key: text[: text.index("</Element>") + len("</Element>")]}
    return {**members, key: text[: len(text) // 2]}


@pytest.mark.parametrize("how", ["missing", "cut_after_first_item", "half"])
def test_a_damaged_timeline_is_refused_with_its_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], how: str
) -> None:
    """Validator H3: a truncated or missing sequence container read as 'clips removed', exit 0."""
    _, plan = setup(tmp_path)
    drt = write_drt(tmp_path / "damaged.drt", damaged(drt_members(CLIPS), how))
    code, out, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (damaged_timeline)" in err
    assert ("missing-container" if how == "missing" else "scan-truncated") in err
    assert out == "" and not (tmp_path / "edited.plan.json").exists()
    assert not (tmp_path / "edited.diff.json").exists()


@pytest.mark.parametrize(
    ("clip", "code"),
    [
        (DrtClip("D:\\Show\\A.mov", 0, 100, -5), "negative_source_in"),
        (DrtClip("D:\\Show\\A.mov", -10, 100, 10), "outside_timeline"),
        (DrtClip("D:\\Show\\A.mov", 0, 0, 10), "zero_length"),
    ],
)
def test_hostile_item_values_are_typed_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], clip: DrtClip, code: str
) -> None:
    """Validator M1: these gave a pydantic traceback and exit 1 with nothing written. Since fix
    round 3 an item on the primary track the plan cannot hold exits 1 (written, check it)."""
    drt, plan = setup(tmp_path, [clip, *CLIPS[1:]])
    status, out, err = call(tmp_path, capsys, drt, plan)
    assert (status, err) == (1, "")
    assert out.startswith("The plan could not hold 1 item(s) on V1")
    assert f"not carried (1): {code} x1 (V1 x1)" in out
    diff = json.loads((tmp_path / "edited.diff.json").read_text())
    assert [e["code"] for e in diff["not_carried"]] == [code]
    assert diff["segments"][0]["status"] == "removed"


def test_an_edited_plan_the_export_refuses_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validator M2: a 3-frame first clip under a 6-frame head dissolve linted clean, exit 0."""
    data = plan_for(CLIPS, head={"video": "dissolve", "video_frames": 6})
    edited = [DrtClip("D:\\Show\\A.mov", 0, 3, 10), *CLIPS[1:]]
    drt, plan = setup(tmp_path, edited, data)
    code, out, _ = call(tmp_path, capsys, drt, plan)
    assert code == 1
    assert "lint: ok" in out and "export check: REFUSED (transitions_overlap)" in out
    diff = json.loads((tmp_path / "edited.diff.json").read_text())
    assert (diff["lint_errors"], diff["export_errors"]) == (False, True)
    assert (tmp_path / "edited.plan.json").is_file()


def test_a_missing_output_folder_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drt, plan = setup(tmp_path)
    missing = tmp_path / "no" / "such"
    code = main(
        [
            "plan",
            "import-timeline",
            str(drt),
            "--plan",
            str(plan),
            "--out",
            str(missing / "p.json"),
            "--report",
            str(tmp_path / "r.json"),
        ]
    )
    err = capsys.readouterr().err
    assert code == 2 and "refused (output_folder_missing)" in err and str(missing) in err
    assert ".aar-lock" not in err and not (tmp_path / "r.json").exists()


def test_too_many_items_is_refused_through_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aar.plan.timeline_import.MAX_ITEMS", len(CLIPS))
    drt, plan = setup(tmp_path)
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert (code, err) == (0, "")  # exactly the limit is read
    (tmp_path / "edited.plan.json").unlink()
    (tmp_path / "edited.diff.json").unlink()
    one_more = [*CLIPS, DrtClip("D:\\Show\\B.mov", 225, 10, 400)]
    drt = write_drt(tmp_path / "more.drt", drt_members(one_more))
    code, _, err = call(tmp_path, capsys, drt, plan)
    assert code == 2 and "refused (timeline_too_large)" in err
    assert "has 4 items, over the 3 limit" in err


def test_a_locked_output_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from aar.node.jobs import file_lock
    from aar.plan.export.files import lock_path

    drt, plan = setup(tmp_path)
    fd = file_lock.acquire_file_lock(str(lock_path(tmp_path / "edited.plan.json")))
    try:
        code, _, err = call(tmp_path, capsys, drt, plan)
        assert code == 2 and "refused (output_locked)" in err
        assert not (tmp_path / "edited.plan.json").exists()
    finally:
        file_lock.release_file_lock(fd)
    code, _, _ = call(tmp_path, capsys, drt, plan)
    assert code == 0
