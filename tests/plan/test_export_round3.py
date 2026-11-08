"""S18 round 3 (the Resolve 21.1 import test): dense lanes, the export pair written atomically
(item 9), and the fallback DTD checker's IDREF rule (item 8)."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
from apply_builders import plan_data, seg, source
from dtd_support import load_dtd, validate
from export_support import GOLDEN_PLANS, ccn, export, load_plan

import aar.plan.export.files as files
from aar.cli import main
from aar.plan.export import ExportError, verify_plan_fcpxml
from aar.plan.export.files import write_pair
from aar.plan.export.layout import stack_lanes
from aar.plan.model import EditPlan

SIZE = [1920, 1080]


def _plan(**kwargs: Any) -> EditPlan:
    kwargs.setdefault("sources", [source(frame_size=SIZE)])
    return EditPlan.model_validate(plan_data(**kwargs))


# -- dense lanes -------------------------------------------------------------------------------


def test_lanes_are_dense_so_v3_without_v2_imports_as_v2() -> None:
    """Resolve creates no empty track for a missing lane (lane -4 landed on A2), so the lanes
    in the file are dense and the report says where each plan track lands."""
    plan = _plan(
        segments=[seg(1, a=0, b=96), seg(3, a=0, b=24, record_in=10)],
        titles=[{"lines": ["x"], "record_in": 0, "record_out": 24}],
    )
    result = export(plan)
    assert result.layout.lanes == {1: 0, 3: 1}
    clip = ET.fromstring(result.fcpxml).find("library/event/project/sequence/spine/asset-clip")
    assert clip is not None
    assert sorted((c.tag, c.get("lane")) for c in clip if c.get("lane")) == [
        ("asset-clip", "1"),
        ("title", "2"),
    ]
    lanes = {row["carries"]: (row["lane"], row["imported_video"]) for row in result.report["lanes"]}
    assert lanes["plan V3 (connected clips) with their sound"] == (1, "V2")
    assert lanes["titles: titles[0]"] == (2, "V3")


# -- fix-check R1: overlapping titles stack on free lanes ------------------------------------


def test_stack_lanes_uses_the_first_free_lane() -> None:
    assert stack_lanes([(0, 10), (20, 30)], 3) == [3, 3]  # no overlap: one lane
    assert stack_lanes([(0, 10), (10, 20)], 3) == [3, 3]  # abutting spans share it
    assert stack_lanes([(0, 30), (10, 20), (15, 40), (31, 50)], 1) == [1, 2, 3, 1]
    assert stack_lanes([(40, 60), (0, 50)], 1) == [2, 1]  # in record order, not plan order
    assert stack_lanes([], 1) == []


def _ccn_with_overlap() -> EditPlan:
    """The CCN golden (lower third 48-120, sting 312-360) plus a lower third at 300-340 that
    overlaps the sting."""
    data = load_plan(GOLDEN_PLANS["ccn_longform_v2"]).model_dump(mode="json")
    data["titles"].append(
        {
            "kind": "lower-third",
            "lines": ["Sam Roe", "Judge"],
            "record_in": 300,
            "record_out": 340,
            "template": "CCN Lower Third",
        }
    )
    return EditPlan.model_validate(data)


def test_a_lower_third_overlapping_the_sting_gets_the_lane_above() -> None:
    """Resolve trims items that overlap on one track: the sting moves to the first free lane
    above, the two lower thirds (which never overlap) share the first title lane, and the
    report says where each lands."""
    result = export(_ccn_with_overlap(), ccn())
    lanes = {t.location: t.lane for t in result.layout.titles}
    assert lanes == {"titles[0]": 1, "titles[1]": 1, "outro.sting": 2}
    root = ET.fromstring(result.fcpxml)
    written = sorted((t.get("name"), t.get("lane")) for t in root.iter("title"))
    assert written == [
        ("Baking Resistance", "2"),
        ("Jane Doe / Head Judge", "1"),
        ("Sam Roe / Judge", "1"),
    ]
    report = result.report
    assert [(t["location"], t["lane"], t["imported_video"]) for t in report["titles"]] == [
        ("titles[0]", 1, "V2"),
        ("titles[1]", 1, "V2"),
        ("outro.sting", 2, "V3"),
    ]
    assert [(row["lane"], row["carries"]) for row in report["lanes"]] == [
        (2, "titles: outro.sting"),
        (1, "titles: titles[0], titles[1]"),
        (0, "V1, the primary storyline, with its clips' sound"),
    ]
    assert verify_plan_fcpxml(result.fcpxml, _ccn_with_overlap(), ccn()).ok


def test_the_golden_titles_keep_one_lane() -> None:
    """No overlap (lower third 48-120, sting 312-360): both on the one title lane."""
    result = export(load_plan(GOLDEN_PLANS["ccn_longform_v2"]), ccn())
    assert {t.location: t.lane for t in result.layout.titles} == {
        "titles[0]": 1,
        "outro.sting": 1,
    }


def test_verify_rejects_titles_overlapping_on_one_lane() -> None:
    """The self-check holds the rule independently of the layout: titles sharing a lane never
    overlap, and title lanes sit above the content lanes."""
    plan = _ccn_with_overlap()
    text = export(plan, ccn()).fcpxml
    root = ET.fromstring(text)
    sting = next(t for t in root.iter("title") if t.get("lane") == "2")
    sting.set("lane", "1")
    problems = verify_plan_fcpxml(ET.tostring(root, encoding="unicode"), plan, ccn()).problems
    assert any("titles overlap on lane 1: 300..340 and 312..360" in p for p in problems)
    sting.set("lane", "-1")  # (lane 0 is refused by the reader already)
    problems = verify_plan_fcpxml(ET.tostring(root, encoding="unicode"), plan, ccn()).problems
    assert any("a title on lane -1 is not above the content lanes" in p for p in problems)


# -- item 9: the export pair, both or neither --------------------------------------------------


def _pair(folder: Path) -> list[tuple[Path, bytes]]:
    return [(folder / "cut.fcpxml", b"<new fcpxml/>"), (folder / "cut.report.json", b"{new}")]


def _old(folder: Path) -> None:
    (folder / "cut.fcpxml").write_bytes(b"<old fcpxml/>")
    (folder / "cut.report.json").write_bytes(b"{old}")


def _fail_on(name: str, call: int, monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    real = getattr(files, name)
    calls = {"n": 0}

    def flaky(*args: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == call:
            raise exc
        return real(*args)

    monkeypatch.setattr(files, name, flaky)


@pytest.mark.parametrize(("name", "call"), [("_write_temp", 2), ("_place_one", 2)])
def test_a_failed_fresh_export_leaves_nothing(
    name: str, call: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure writing the report, or on its rename after the FCPXML landed: nothing remains,
    not the FCPXML, not a temporary file."""
    _fail_on(name, call, monkeypatch, OSError(28, "No space left on device"))
    with pytest.raises(OSError, match="No space left"):
        write_pair(_pair(tmp_path), overwrite=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.parametrize(("name", "call"), [("_write_temp", 2), ("_place_one", 2)])
def test_a_failed_overwrite_keeps_the_previous_pair(
    name: str, call: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --overwrite, the previous FCPXML and report survive byte-identical: the FCPXML
    already replaced is restored from its backup; no temporary file or backup remains."""
    _old(tmp_path)
    _fail_on(name, call, monkeypatch, PermissionError(13, "Permission denied"))
    with pytest.raises(PermissionError):
        write_pair(_pair(tmp_path), overwrite=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.fcpxml", "cut.report.json"]
    assert (tmp_path / "cut.fcpxml").read_bytes() == b"<old fcpxml/>"
    assert (tmp_path / "cut.report.json").read_bytes() == b"{old}"


def test_an_overwrite_replaces_both_and_cleans_up(tmp_path: Path) -> None:
    _old(tmp_path)
    write_pair(_pair(tmp_path), overwrite=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.fcpxml", "cut.report.json"]
    assert (tmp_path / "cut.fcpxml").read_bytes() == b"<new fcpxml/>"
    assert (tmp_path / "cut.report.json").read_bytes() == b"{new}"


def test_a_report_appearing_mid_write_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another writer creates the report between the check and the placement: output_exists,
    the new FCPXML is removed again, the other writer's file is kept."""
    real = files._place_one

    def racing(tmp: Path, dest: Path, overwrite: bool) -> None:
        if dest.name == "cut.report.json":
            dest.write_bytes(b"{theirs}")
        real(tmp, dest, overwrite)

    monkeypatch.setattr(files, "_place_one", racing)
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path), overwrite=False)
    assert caught.value.code == "output_exists"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.report.json"]
    assert (tmp_path / "cut.report.json").read_bytes() == b"{theirs}"


def _other_writer_replaces_the_fcpxml(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the second placement (the report), another writer first replaces the FCPXML this
    call has just placed (a new inode), then the placement fails."""
    real = files._place_one

    def racing(tmp: Path, dest: Path, overwrite: bool) -> None:
        if dest.name == "cut.report.json":
            theirs = folder / "theirs.tmp"
            theirs.write_bytes(b"<their fcpxml/>")
            os.replace(theirs, folder / "cut.fcpxml")
            raise OSError(28, "No space left on device")
        real(tmp, dest, overwrite)

    monkeypatch.setattr(files, "_place_one", racing)


def test_rollback_leaves_another_writers_file_and_keeps_the_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--overwrite: the FCPXML is no longer the file this call placed, so the rollback does not
    restore the backup over it; the backup stays beside it, named, and the error says so."""
    _old(tmp_path)
    _other_writer_replaces_the_fcpxml(tmp_path, monkeypatch)
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path), overwrite=True)
    assert caught.value.code == "output_conflict"
    assert "No space left" in caught.value.message
    assert (tmp_path / "cut.fcpxml").read_bytes() == b"<their fcpxml/>"
    assert (tmp_path / "cut.report.json").read_bytes() == b"{old}"  # never replaced
    kept = [p for p in tmp_path.iterdir() if p.name.startswith(".aar-backup-cut.fcpxml.")]
    assert len(kept) == 1 and kept[0].read_bytes() == b"<old fcpxml/>"
    assert caught.value.details["conflicts"] == [
        {
            "path": str(tmp_path / "cut.fcpxml"),
            "kept_backup": str(kept[0]),
            "reason": "was replaced by another writer and left in place",
        }
    ]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted(["cut.fcpxml", "cut.report.json", kept[0].name])  # no temp left


def test_rollback_never_deletes_another_writers_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh export (no backup): the same identity rule keeps the other writer's FCPXML."""
    _other_writer_replaces_the_fcpxml(tmp_path, monkeypatch)
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path), overwrite=False)
    assert caught.value.code == "output_conflict"
    assert caught.value.details["conflicts"] == [
        {
            "path": str(tmp_path / "cut.fcpxml"),
            "kept_backup": None,
            "reason": "was replaced by another writer and left in place",
        }
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.fcpxml"]
    assert (tmp_path / "cut.fcpxml").read_bytes() == b"<their fcpxml/>"


def test_an_in_place_change_is_not_ours_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same inode, different size and mtime: another writer rewrote our file in place."""
    _old(tmp_path)
    real = files._place_one

    def rewriting(tmp: Path, dest: Path, overwrite: bool) -> None:
        if dest.name == "cut.report.json":
            with open(tmp_path / "cut.fcpxml", "ab") as handle:
                handle.write(b"<!-- edited -->")
            raise OSError(5, "Input/output error")
        real(tmp, dest, overwrite)

    monkeypatch.setattr(files, "_place_one", rewriting)
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path), overwrite=True)
    assert caught.value.code == "output_conflict"
    assert (tmp_path / "cut.fcpxml").read_bytes() == b"<new fcpxml/><!-- edited -->"


def _fail_after_placing(name: str, exc: BaseException, monkeypatch: pytest.MonkeyPatch) -> None:
    """The placement of ``name`` lands, then fails before returning (an interrupt, or a failure
    after the rename): the window between the file appearing and write_pair hearing of it."""
    real = files._place_one

    def landed_then_failed(tmp: Path, dest: Path, overwrite: bool) -> None:
        real(tmp, dest, overwrite)
        if dest.name == name:
            raise exc

    monkeypatch.setattr(files, "_place_one", landed_then_failed)


FAILURES = [OSError(5, "Input/output error"), KeyboardInterrupt()]


@pytest.mark.parametrize("exc", FAILURES, ids=["oserror", "interrupt"])
@pytest.mark.parametrize("name", ["cut.fcpxml", "cut.report.json"])
@pytest.mark.parametrize("overwrite", [False, True], ids=["fresh", "overwrite"])
def test_a_failure_inside_the_placement_still_rolls_back(
    overwrite: bool,
    name: str,
    exc: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix-check R3: each destination is recorded before its placement, so one that landed
    just before the failure is put back too: a fresh export leaves nothing, an overwrite
    leaves the previous pair byte-identical; no temporary file or backup remains."""
    if overwrite:
        _old(tmp_path)
    _fail_after_placing(name, exc, monkeypatch)
    with pytest.raises(type(exc)):
        write_pair(_pair(tmp_path), overwrite=overwrite)
    if overwrite:
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.fcpxml", "cut.report.json"]
        assert (tmp_path / "cut.fcpxml").read_bytes() == b"<old fcpxml/>"
        assert (tmp_path / "cut.report.json").read_bytes() == b"{old}"
    else:
        assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["cut.fcpxml", "cut.report.json"])
def test_a_failed_temp_unlink_after_the_link_rolls_back(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-clobber placement links the temporary to the destination, then unlinks the
    temporary. If that unlink fails, the destination (still the temporary's inode) is ours and
    is removed again, with the temporary."""
    real = files._place_one

    def link_then_fail(tmp: Path, dest: Path, overwrite: bool) -> None:
        if dest.name != name:
            real(tmp, dest, overwrite)
            return
        os.link(tmp, dest)
        raise PermissionError(13, "Permission denied", str(tmp))

    monkeypatch.setattr(files, "_place_one", link_then_fail)
    with pytest.raises(PermissionError):
        write_pair(_pair(tmp_path), overwrite=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_failed_restore_keeps_the_backup_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If putting the previous file back fails, the backup is kept under a clear name and the
    failure is output_conflict naming it, not a lost previous file."""
    _old(tmp_path)
    _fail_on("_place_one", 2, monkeypatch, OSError(28, "No space left on device"))
    real_replace = os.replace
    onto_fcpxml = {"n": 0}

    def failing_restore(src: Any, dst: Any) -> None:
        if Path(dst).name == "cut.fcpxml":
            onto_fcpxml["n"] += 1
            if onto_fcpxml["n"] == 2:  # the first is the placement, the second the restore
                raise OSError(30, "Read-only file system")
        real_replace(src, dst)

    monkeypatch.setattr(files.os, "replace", failing_restore)
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path), overwrite=True)
    assert caught.value.code == "output_conflict"
    kept = [p for p in tmp_path.iterdir() if p.name.startswith(".aar-backup-cut.fcpxml.")]
    assert len(kept) == 1 and kept[0].read_bytes() == b"<old fcpxml/>"
    [conflict] = caught.value.details["conflicts"]
    assert conflict["kept_backup"] == str(kept[0])
    assert conflict["reason"].startswith("could not be restored")


def test_the_cli_keeps_the_previous_pair_when_the_report_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cut.fcpxml"
    _old(tmp_path)
    _fail_on("_place_one", 2, monkeypatch, OSError(28, "No space left on device"))
    argv = ["plan", "export", str(GOLDEN_PLANS["ccn_longform_v2"]), "--out", str(out)]
    assert main([*argv, "--overwrite"]) == 2
    assert "No space left" in capsys.readouterr().err
    assert out.read_bytes() == b"<old fcpxml/>"
    assert (tmp_path / "cut.report.json").read_bytes() == b"{old}"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.fcpxml", "cut.report.json"]
    # And without the failure the pair is replaced together.
    monkeypatch.undo()
    assert main([*argv, "--overwrite"]) == 0
    report = json.loads((tmp_path / "cut.report.json").read_text(encoding="utf-8"))
    assert report["file"] == "cut.fcpxml"


# -- item 8: the fallback DTD checker's IDREF rule ---------------------------------------------

MINI_DTD = """
<!ELEMENT root (item*, link*)>
<!ELEMENT item EMPTY>
<!ATTLIST item id ID #REQUIRED>
<!ELEMENT link EMPTY>
<!ATTLIST link ref IDREF #REQUIRED>
<!ATTLIST link refs IDREFS #IMPLIED>
"""


def _check(link: str) -> list[str]:
    document = f'<root><item id="r1"/><item id="r2"/>{link}</root>'
    return validate(load_dtd(MINI_DTD), document)


def test_an_idref_holds_exactly_one_name() -> None:
    assert _check('<link ref="r1"/>') == []
    # Both ids exist, but an IDREF is one name: a list is a violation.
    assert _check('<link ref="r1 r2"/>') == ["<link ref='r1 r2'> is not a single IDREF name"]
    assert _check('<link ref=" r1"/>') == ["<link ref=' r1'> is not a single IDREF name"]
    assert _check('<link ref="r3"/>') == ["<link> IDREF 'r3' resolves to no ID"]


def test_idrefs_may_hold_a_list() -> None:
    assert _check('<link ref="r1" refs="r1 r2"/>') == []
    assert _check('<link ref="r1" refs="r1 r9"/>') == ["<link> IDREF 'r9' resolves to no ID"]


def _ids(item: str, link: str = "") -> list[str]:
    return validate(load_dtd(MINI_DTD), f"<root>{item}{link}</root>")


@pytest.mark.parametrize(
    "value", ["r1", "_x", ":x", "a-1.b", "x\u00b7y", "\u00e9t\u00e9", "\u4e2d1"]
)
def test_ids_that_are_xml_names_pass(value: str) -> None:
    assert _ids(f'<item id="{value}"/>', f'<link ref="{value}" refs="{value}"/>') == []


@pytest.mark.parametrize("value", ["1bad", "-x", ".x", "\u00b7x", "9"])
def test_an_id_that_is_not_an_xml_name_is_rejected(value: str) -> None:
    assert _ids(f'<item id="{value}"/>') == [f"<item id={value!r}> is not an XML Name"]


def test_idref_and_idrefs_tokens_must_be_xml_names() -> None:
    problems = _ids('<item id="r1"/>', '<link ref="1x" refs="r1 2x"/>')
    assert problems == [
        "<link ref='1x'> is not an XML Name",
        "<link refs> token '2x' is not an XML Name",
    ]
