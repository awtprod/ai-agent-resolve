"""``aar drp edit`` end to end on the bundled templates (S8, offline part).

Each test writes a real file, then checks it independently of the writer's own verification:
the file is read back, diffed against the input with the S2 gate, and the change list is
compared with what the edit should do. Refusal tests check that nothing is written.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest
from diff_testing import A1_ITEM, CLIP, H264, POOL, SEQUENCE, TIMELINE, V1_ITEM
from drp_edit import add_carrier, carrier, edit_zip, marker_entry, markers_fields_blob, once
from drp_helpers import needs_zstandard

from aar.cli import main
from aar.drp import blobs
from aar.drp.diff import diff_projects
from aar.drp.modify import engine, parse_plan, verify_write, write
from aar.drp.modify.write import SIDECAR_SUFFIX, edit_file
from aar.drp.reader import read_drp

R19 = H264.parent / "media-clip-r19.drp"
R19_ARCHIVE = "7fb9ee45-9553-40cf-8f54-41d4a5132216"
R19_MASTER = "653a17d4-c214-4e7c-8854-8ada21b71530"
H264_PROJECT = "0f98e18c-6c78-4468-805d-a6680602dd5c"
H264_POOL_TIMELINE = "6d692bf3-f02a-4943-ac2f-74d3f0b9fe1f"
MAC_PATH = "/Users/samuelgursky/Desktop/20260410/CKY - Bam and The Rental Car.mp4"
BLUE, RED, GREEN = 2, 32, 8


def plan(*edits: dict[str, Any], **top: Any) -> Any:
    return parse_plan(json.dumps({"edits": list(edits), **top}))


def run(src: Path, tmp_path: Path, *edits: dict[str, Any], name: str = "out.drp", **kw: Any) -> Any:
    out = tmp_path / name
    return edit_file(src, plan(*edits), out, **kw), out


def changes(a: Path, b: Path) -> list[tuple[Any, ...]]:
    """The gate-mode diff, read independently of the writer's verification."""
    d = diff_projects(read_drp(a), read_drp(b), gate=True)
    assert d.gate.passed, d.gate.reasons
    return sorted(
        (c.section, c.kind, c.object_kind, c.object_id, c.field_name, _j(c.old), _j(c.new))
        for c in d.changes
    )


def _j(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def members_differing(a: Path, b: Path) -> set[str]:
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        return {n for n in za.namelist() if za.read(n) != zb.read(n)}


def no_temp_files(directory: Path) -> bool:
    return not [p for p in directory.iterdir() if p.name.startswith(".") and ".tmp" in p.name]


# --------------------------------------------------------------------------- rename


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (
            {"op": "rename", "target": "project", "new_name": "Renamed"},
            ("header", "renamed", "project", H264_PROJECT, "name", "_media_tmpl", "Renamed"),
        ),
        (
            {"op": "rename", "target": "clip", "id": CLIP, "new_name": "Interview A"},
            ("media", "renamed", "media", CLIP, "name", "sample.mp4", "Interview A"),
        ),
        (
            {"op": "rename", "target": "clip", "name": "sample.mp4", "new_name": "By name"},
            ("media", "renamed", "media", CLIP, "name", "sample.mp4", "By name"),
        ),
        (
            {"op": "rename", "target": "item", "id": A1_ITEM, "new_name": "A1 piece"},
            ("timelines", "renamed", "item", A1_ITEM, "name", "sample.mp4", "A1 piece"),
        ),
        (
            {"op": "rename", "target": "timeline", "id": TIMELINE, "new_name": "Main Edit"},
            ("timelines", "renamed", "timeline", TIMELINE, "name", "MediaTemplate", "Main Edit"),
        ),
    ],
    ids=["project", "clip", "clip-by-name", "item", "timeline"],
)
def test_rename_writes_exactly_one_change(
    tmp_path: Path, edit: dict[str, Any], expected: tuple[str, ...]
) -> None:
    outcome, out = run(H264, tmp_path, edit)
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    section, kind, okind, oid, field, old, new = expected
    assert changes(H264, out) == [(section, kind, okind, oid, field, _j(old), _j(new))]
    assert [c["kind"] for c in outcome.changes] == [kind]


def test_rename_timeline_keeps_its_pool_entry_in_step(tmp_path: Path) -> None:
    outcome, out = run(
        H264, tmp_path, {"op": "rename", "target": "timeline", "id": TIMELINE, "new_name": "Cut"}
    )
    assert outcome.status == "written"
    drp = read_drp(out)
    assert drp.timelines[0].name == "Cut"
    assert drp.media_by_id()[H264_POOL_TIMELINE].name == "Cut"
    assert members_differing(H264, out) == {POOL}


def test_rename_bin(tmp_path: Path) -> None:
    outcome, out = run(
        R19, tmp_path, {"op": "rename", "target": "bin", "id": R19_ARCHIVE, "new_name": "Selects"}
    )
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    assert changes(R19, out) == [
        ("media", "renamed", "bin", R19_ARCHIVE, "name", _j("Archive"), _j("Selects"))
    ]
    assert [b.path for b in read_drp(out).bins] == ["Master", "Master/Selects"]


def test_rename_escapes_and_keeps_every_other_byte(tmp_path: Path) -> None:
    name = "Tom & Jerry <final> 'cut' é"
    outcome, out = run(
        H264, tmp_path, {"op": "rename", "target": "clip", "id": CLIP, "new_name": name}
    )
    assert outcome.status == "written"
    assert read_drp(out).media_by_id()[CLIP].name == name
    with zipfile.ZipFile(H264) as za, zipfile.ZipFile(out) as zb:
        old, new = za.read(POOL), zb.read(POOL)
    escaped = "Tom &amp; Jerry &lt;final&gt; 'cut' é".encode()
    assert new.replace(escaped, b"sample.mp4", 1) == old
    assert members_differing(H264, out) == {POOL}


def test_rename_self_closing_name(tmp_path: Path) -> None:
    src = edit_zip(
        H264,
        tmp_path / "empty-name.drp",
        {"project.xml": once(b"<ProjectName>_media_tmpl</ProjectName>", b"<ProjectName/>")},
    )
    outcome, out = run(
        src, tmp_path, {"op": "rename", "target": "project", "new_name": "Now named"}
    )
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    assert read_drp(out).project.name == "Now named"


# --------------------------------------------------------------------------- paths


@needs_zstandard
def test_path_map_mac_to_windows(tmp_path: Path) -> None:
    table = [
        {"windows": "E:\\Footage", "mac": "/Users/samuelgursky/Desktop"},
        {"windows": "f:", "mac": "/Media"},
    ]
    outcome, out = run(
        H264, tmp_path, {"op": "path_map", "direction": "mac_to_windows", "table": table}
    )
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    new_clip = "E:\\Footage\\20260410\\CKY - Bam and The Rental Car.mp4"
    assert changes(H264, out) == sorted(
        [
            ("media", "relinked", "media", CLIP, "path", _j(MAC_PATH), _j(new_clip)),
            (
                "timelines",
                "changed",
                "item",
                V1_ITEM,
                "media_file_path",
                _j("/Media/sample.mp4"),
                _j("F:\\sample.mp4"),
            ),
            (
                "timelines",
                "changed",
                "item",
                A1_ITEM,
                "media_file_path",
                _j("/Media/sample.mp4"),
                _j("F:\\sample.mp4"),
            ),
        ]
    )
    [report] = outcome.report
    assert report["unmapped"] == []
    assert {t["new"] for t in report["touched"]} == {new_clip, "F:\\sample.mp4"}
    clip = read_drp(out).media_by_id()[CLIP]
    assert clip.path == new_clip and clip.codec == read_drp(H264).media_by_id()[CLIP].codec


@needs_zstandard
def test_path_map_round_trip_restores_every_decoded_value(tmp_path: Path) -> None:
    table = [
        {"windows": "E:", "mac": "/Users/samuelgursky/Desktop"},
        {"windows": "F:", "mac": "/Media"},
    ]
    first, out1 = run(
        H264, tmp_path, {"op": "path_map", "direction": "mac_to_windows", "table": table}
    )
    assert first.status == "written"
    back, out2 = run(
        out1,
        tmp_path,
        {"op": "path_map", "direction": "windows_to_mac", "table": table},
        name="back.drp",
    )
    assert back.status == "written", back.to_dict()["refusals"]
    # Decoded values and every byte outside the re-framed zstd blobs are back to the original.
    assert diff_projects(read_drp(H264), read_drp(out2), gate=True).changes == []


@needs_zstandard
def test_path_map_reports_what_it_could_not_map(tmp_path: Path) -> None:
    table = [{"windows": "\\\\nas\\share", "mac": "/Users/samuelgursky/Desktop"}]
    outcome, out = run(
        H264, tmp_path, {"op": "path_map", "direction": "mac_to_windows", "table": table}
    )
    assert outcome.status == "written"
    [report] = outcome.report
    assert [t["new"] for t in report["touched"]] == [
        "\\\\nas\\share\\20260410\\CKY - Bam and The Rental Car.mp4"
    ] * len(report["touched"])
    assert sorted((u["id"], u["reason"]) for u in report["unmapped"]) == [
        (A1_ITEM, "no-mapping"),
        (V1_ITEM, "no-mapping"),
    ]
    back = plan(
        {
            "op": "path_map",
            "direction": "windows_to_mac",
            "table": [{"windows": "E:", "mac": "/Volumes/E"}],
        }
    )
    again = edit_file(out, back, dry_run=True)
    reasons = {u["reason"] for u in again.report[0]["unmapped"]}
    assert reasons == {"unc-not-mapped", "other-form"}


def test_path_map_refuses_a_bad_table(tmp_path: Path) -> None:
    table = [{"windows": "E:", "mac": "/Volumes/A"}, {"windows": "e:\\", "mac": "/Volumes/B"}]
    outcome, out = run(
        H264, tmp_path, {"op": "path_map", "direction": "windows_to_mac", "table": table}
    )
    assert outcome.status == "refused" and outcome.refusals[0].code == "bad-table"
    assert not out.exists()


# --------------------------------------------------------------------------- markers


def marker_src(
    tmp_path: Path, *entries: bytes, owner: str = SEQUENCE, tag: str = "Sm2SequenceLockableBlob"
) -> Path:
    element = carrier(
        "0b0b0b0b-1111-2222-3333-444444444444", owner, markers_fields_blob(*entries), tag
    )
    return edit_zip(H264, tmp_path / "markers.drp", {"project.xml": add_carrier(element)})


def props(
    frame: int, color: str, name: str, note: str = "", duration: int = 1, custom: str = ""
) -> dict[str, Any]:
    return {
        "frame": frame,
        "color": color,
        "name": name,
        "note": note,
        "duration": duration,
        "custom_data": custom,
    }


def test_marker_add_update_delete_on_a_timeline(tmp_path: Path) -> None:
    src = marker_src(
        tmp_path,
        marker_entry(10, BLUE, "Intro"),
        marker_entry(50, GREEN, "Laugh", "big one"),
        marker_entry(90, RED, "Cut here"),
    )
    edits = [
        {
            "op": "marker_add",
            "owner": "timeline",
            "owner_id": TIMELINE,
            "frame": 30,
            "color": "Yellow",
            "name": "New",
            "note": "n",
            "duration": 12,
            "custom_data": "agent:1",
        },
        {
            "op": "marker_update",
            "owner": "timeline",
            "owner_id": TIMELINE,
            "frame": 50,
            "set": {"name": "Big laugh", "color": "Pink"},
        },
        {"op": "marker_delete", "owner": "timeline", "owner_id": TIMELINE, "frame": 90},
    ]
    outcome, out = run(src, tmp_path, *edits)
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    label = json.loads(Path(str(out) + SIDECAR_SUFFIX).read_text())
    assert parse_plan(json.dumps(label["plan"])).model_dump() == plan(*edits).model_dump()
    old50 = props(50, "Green", "Laugh", "big one")
    assert changes(src, out) == sorted(
        [
            (
                "markers",
                "added",
                "marker",
                TIMELINE,
                None,
                _j(None),
                _j(props(30, "Yellow", "New", "n", 12, "agent:1")),
            ),
            (
                "markers",
                "changed",
                "marker",
                TIMELINE,
                "color,name",
                _j(old50),
                _j({**old50, "color": "Pink", "name": "Big laugh"}),
            ),
            (
                "markers",
                "removed",
                "marker",
                TIMELINE,
                None,
                _j(props(90, "Red", "Cut here")),
                _j(None),
            ),
        ]
    )
    markers = read_drp(out).timelines[0].markers
    assert [(m.frame, m.name) for m in markers] == [(10, "Intro"), (30, "New"), (50, "Big laugh")]
    assert members_differing(src, out) == {"project.xml"}


def test_marker_move_is_a_remove_and_an_add(tmp_path: Path) -> None:
    src = marker_src(tmp_path, marker_entry(10, BLUE, "Intro"))
    edit = {
        "op": "marker_update",
        "owner": "timeline",
        "owner_id": TIMELINE,
        "frame": 10,
        "set": {"frame": 15},
    }
    outcome, out = run(src, tmp_path, edit)
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    assert [c[1] for c in changes(src, out)] == ["added", "removed"]
    assert [m.frame for m in read_drp(out).timelines[0].markers] == [15]


def test_marker_update_keeps_undecoded_fields(tmp_path: Path) -> None:
    extra = struct.pack("B", 7 << 3 | 0) + b"\x05"  # an undecoded varint field 7 in the detail
    src = marker_src(tmp_path, marker_entry(10, BLUE, "Intro", extra=extra))
    edit = {
        "op": "marker_update",
        "owner": "timeline",
        "owner_id": TIMELINE,
        "frame": 10,
        "set": {"note": "kept"},
    }
    outcome, out = run(src, tmp_path, edit)
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    with zipfile.ZipFile(out) as zf:
        assert b"0705" in zf.read("project.xml")  # hex of field 7 = 5 is still there


def test_item_marker(tmp_path: Path) -> None:
    src = marker_src(
        tmp_path, marker_entry(3, BLUE, "Clip note"), owner=V1_ITEM, tag="Sm2TiItemLockableBlob"
    )
    edit = {
        "op": "marker_update",
        "owner": "item",
        "owner_id": V1_ITEM,
        "frame": 3,
        "set": {"duration": 24},
    }
    outcome, out = run(src, tmp_path, edit)
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    old = props(3, "Blue", "Clip note")
    assert changes(src, out) == [
        ("markers", "changed", "marker", V1_ITEM, "duration", _j(old), _j({**old, "duration": 24}))
    ]


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        ({"op": "marker_add", "frame": 10}, "marker-exists"),
        ({"op": "marker_update", "frame": 11, "set": {"name": "x"}}, "marker-not-found"),
        ({"op": "marker_delete", "frame": 99}, "marker-not-found"),
        ({"op": "marker_update", "frame": 10, "set": {"frame": 20}}, "marker-exists"),
        ({"op": "marker_add", "frame": 5, "name": "bad\x01"}, "bad-value"),
    ],
)
def test_marker_refusals(tmp_path: Path, edit: dict[str, Any], code: str) -> None:
    src = marker_src(tmp_path, marker_entry(10, BLUE, "A"), marker_entry(20, BLUE, "B"))
    outcome, out = run(src, tmp_path, {"owner": "timeline", "owner_id": TIMELINE, **edit})
    assert outcome.status == "refused" and outcome.refusals[0].code == code
    assert not out.exists()


def test_marker_refusals_ambiguous_and_missing_list(tmp_path: Path) -> None:
    src = marker_src(tmp_path, marker_entry(10, BLUE, "A"), marker_entry(10, RED, "B"))
    outcome, _ = run(
        src,
        tmp_path,
        {"op": "marker_delete", "owner": "timeline", "owner_id": TIMELINE, "frame": 10},
    )
    assert outcome.refusals[0].code == "marker-ambiguous"
    outcome, _ = run(
        H264, tmp_path, {"op": "marker_add", "owner": "timeline", "owner_id": TIMELINE, "frame": 1}
    )
    assert outcome.refusals[0].code == "no-marker-list"
    outcome, _ = run(
        H264, tmp_path, {"op": "marker_add", "owner": "item", "owner_id": "nope", "frame": 1}
    )
    assert outcome.refusals[0].code == "target-not-found"


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("edits", "code"),
    [
        ([{"op": "rename", "target": "bin", "id": R19_MASTER, "new_name": "x"}], "root-bin"),
        (
            [{"op": "rename", "target": "clip", "id": "missing", "new_name": "x"}],
            "target-not-found",
        ),
        (
            [{"op": "rename", "target": "clip", "id": H264_POOL_TIMELINE, "new_name": "x"}],
            "target-not-found",
        ),
        (
            [{"op": "rename", "target": "item", "name": "sample.mp4", "new_name": "x"}],
            "target-ambiguous",
        ),
        (
            [{"op": "rename", "target": "project", "new_name": "x", "expect_name": "other"}],
            "expect-mismatch",
        ),
        ([{"op": "rename", "target": "project", "new_name": "tab\there"}], "bad-value"),
        (
            [
                {"op": "rename", "target": "clip", "id": CLIP, "new_name": "a"},
                {"op": "rename", "target": "clip", "id": CLIP, "new_name": "b"},
            ],
            "conflicting-edits",
        ),
        (
            [
                {
                    "op": "path_map",
                    "direction": "windows_to_mac",
                    "table": [{"windows": "E:", "mac": "/Volumes/E"}],
                },
                {
                    "op": "path_map",
                    "direction": "windows_to_mac",
                    "table": [{"windows": "F:", "mac": "/Volumes/F"}],
                },
            ],
            "conflicting-edits",
        ),
    ],
)
def test_edit_refusals_write_nothing(
    tmp_path: Path, edits: list[dict[str, Any]], code: str
) -> None:
    src = R19 if code == "root-bin" else H264
    outcome, out = run(src, tmp_path, *edits)
    assert outcome.status == "refused"
    assert outcome.refusals[0].code == code
    assert not out.exists() and not Path(str(out) + SIDECAR_SUFFIX).exists()
    assert no_temp_files(tmp_path)


def test_unintended_change_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = engine.apply

    def sneaky(member_data: dict[str, bytes], splices: Any) -> dict[str, bytes]:
        out = real(member_data, splices)
        out["project.xml"] = out["project.xml"].replace(
            b"<ProjectAgeInMs>105243607", b"<ProjectAgeInMs>105243608"
        )
        return out

    monkeypatch.setattr(engine, "apply", sneaky)
    outcome, out = run(H264, tmp_path, {"op": "rename", "target": "project", "new_name": "x"})
    assert outcome.status == "refused"
    codes = [r.code for r in outcome.refusals]
    assert "gate-failed" in codes and "unintended-change" in codes
    assert not out.exists() and no_temp_files(tmp_path)


def test_a_passing_gate_alone_is_not_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decoded, DbId-matched change the plan did not ask for passes the gate but is refused."""
    real = engine.apply

    def extra_rename(member_data: dict[str, bytes], splices: Any) -> dict[str, bytes]:
        out = real(member_data, splices)
        out[POOL] = out[POOL].replace(b"<Name>MediaTemplate</Name>", b"<Name>Other</Name>", 1)
        return out

    monkeypatch.setattr(engine, "apply", extra_rename)
    outcome, out = run(
        H264, tmp_path, {"op": "rename", "target": "clip", "id": CLIP, "new_name": "x"}
    )
    assert outcome.gate == {"passed": True, "reasons": []}
    assert [r.code for r in outcome.refusals] == ["unintended-change"]
    [extra] = outcome.refusals[0].details["changes"]
    assert (extra["kind"], extra["new"]) == ("renamed", "Other")
    assert not out.exists()


def test_missing_change_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = engine.apply

    def lazy(member_data: dict[str, bytes], splices: Any) -> dict[str, bytes]:
        return real(member_data, [s for s in splices if s.member != "project.xml"])

    monkeypatch.setattr(engine, "apply", lazy)
    outcome, out = run(
        H264,
        tmp_path,
        {"op": "rename", "target": "project", "new_name": "x"},
        {"op": "rename", "target": "clip", "id": CLIP, "new_name": "y"},
    )
    assert [r.code for r in outcome.refusals] == ["missing-change"]
    assert outcome.refusals[0].details["changes"][0]["object_kind"] == "project"
    assert not out.exists()


def test_version_bump_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = write.zipraw.rewrite

    def bump(data: bytes, replacements: dict[str, bytes]) -> bytes:
        replacements = dict(replacements)
        replacements["SeqContainer/106d9f2b-2951-4ab9-b961-0aa934df20e9.xml"] = (
            zipfile.ZipFile(H264)
            .read("SeqContainer/106d9f2b-2951-4ab9-b961-0aa934df20e9.xml")
            .replace(b'DbPrjVer="17"', b'DbPrjVer="18"', 1)
        )
        return real(data, replacements)

    monkeypatch.setattr(write.zipraw, "rewrite", bump)
    outcome, out = run(H264, tmp_path, {"op": "rename", "target": "project", "new_name": "x"})
    refusal = next(r for r in outcome.refusals if r.code == "version-gate")
    assert refusal.details["members"] == ["SeqContainer/106d9f2b-2951-4ab9-b961-0aa934df20e9.xml"]
    assert not out.exists()


def test_version_gate_against_the_target_host(tmp_path: Path) -> None:
    outcome, out = (
        edit_file(
            R19,
            plan({"op": "rename", "target": "project", "new_name": "x"}, max_project_version=13),
            tmp_path / "o.drp",
        ),
        tmp_path / "o.drp",
    )
    assert [r.code for r in outcome.refusals] == ["version-gate"]
    ok = edit_file(
        R19,
        plan({"op": "rename", "target": "project", "new_name": "x"}, max_project_version=14),
        out,
    )
    assert ok.status == "written"
    assert read_drp(out).versions.project_version == 14


def test_input_sha_must_match(tmp_path: Path) -> None:
    edits = {"op": "rename", "target": "project", "new_name": "x"}
    wrong = edit_file(H264, plan(edits, input_sha256="0" * 64), tmp_path / "o.drp")
    assert [r.code for r in wrong.refusals] == ["input-sha-mismatch"]
    right = edit_file(H264, plan(edits, input_sha256=read_drp(H264).sha256), tmp_path / "o.drp")
    assert right.status == "written"


def test_partial_blob_input_is_refused_by_design(tmp_path: Path) -> None:
    """A keyed dict that decodes only partly makes the gate impossible to pass for that file, so
    every edit to it is refused up front, also an edit far from the partial blob."""
    data = zipfile.ZipFile(H264).read("project.xml")
    start = data.index(b"<FieldsBlob>", data.index(b"<SM_Config")) + len(b"<FieldsBlob>")
    end = data.index(b"</FieldsBlob>", start)
    broken = data[:end] + b"00" + data[end:]  # one trailing byte after the last entry
    src = edit_zip(H264, tmp_path / "partial.drp", {"project.xml": lambda _: broken})
    assert "blob-partial" in {i.code for i in read_drp(src).issues}
    outcome, out = run(
        src, tmp_path, {"op": "rename", "target": "clip", "id": CLIP, "new_name": "x"}
    )
    assert [r.code for r in outcome.refusals] == ["input-has-warnings"]
    assert outcome.refusals[0].details["codes"] == ["blob-partial"]
    assert not out.exists()
    assert start < end  # the rename would not have touched this member


def test_unsupported_zip_layout_is_refused(tmp_path: Path) -> None:
    src = tmp_path / "prefixed.drp"
    src.write_bytes(b"\0" * 16 + H264.read_bytes())  # zipfile still reads it; the writer will not
    assert read_drp(src).project is not None
    outcome, out = run(src, tmp_path, {"op": "rename", "target": "project", "new_name": "x"})
    assert [r.code for r in outcome.refusals] in (["zip-layout"], ["input-has-warnings"])
    assert not out.exists()


# --------------------------------------------------------------------------- write safety


def test_no_op_plan_writes_the_input_bytes(tmp_path: Path) -> None:
    outcome, out = run(H264, tmp_path)
    assert outcome.status == "written" and outcome.changes == []
    assert out.read_bytes() == H264.read_bytes()


def test_existing_output_and_force(tmp_path: Path) -> None:
    out = tmp_path / "out.drp"
    out.write_bytes(b"keep me")
    edit = {"op": "rename", "target": "project", "new_name": "x"}
    outcome = edit_file(H264, plan(edit), out)
    assert [r.code for r in outcome.refusals] == ["out-exists"]
    assert out.read_bytes() == b"keep me"
    forced = edit_file(H264, plan(edit), out, force=True)
    assert forced.status == "written" and read_drp(out).project.name == "x"
    again = edit_file(H264, plan(edit), tmp_path / "other.drp")
    assert again.status == "written"
    Path(str(tmp_path / "third.drp") + SIDECAR_SUFFIX).write_text("{}")
    blocked = edit_file(H264, plan(edit), tmp_path / "third.drp")
    assert [r.code for r in blocked.refusals] == ["out-exists"]


def test_never_writes_in_place(tmp_path: Path) -> None:
    src = tmp_path / "in.drp"
    src.write_bytes(H264.read_bytes())
    for force in (False, True):
        outcome = edit_file(
            src, plan({"op": "rename", "target": "project", "new_name": "x"}), src, force=force
        )
        assert [r.code for r in outcome.refusals] == ["out-is-input"]
    assert src.read_bytes() == H264.read_bytes()


def test_sidecar_labels_the_output_unverified(tmp_path: Path) -> None:
    outcome, out = run(H264, tmp_path, {"op": "rename", "target": "project", "new_name": "x"})
    assert outcome.status == "written"
    label = json.loads(Path(str(out) + SIDECAR_SUFFIX).read_text())

    assert label["status"] == "unverified" and label["label"] == "unverified"
    assert label["input"]["sha256"] == hashlib.sha256(H264.read_bytes()).hexdigest()
    assert label["output"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert label["plan"]["edits"][0]["new_name"] == "x"
    assert [c["field"] for c in label["changes"]] == ["name"]
    assert label["gate"] == {"passed": True, "reasons": []}
    assert parse_plan(json.dumps(label["plan"])).edits[0].new_name == "x"
    assert no_temp_files(tmp_path)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "out.drp"
    outcome = edit_file(
        H264, plan({"op": "rename", "target": "project", "new_name": "x"}), out, dry_run=True
    )
    assert outcome.status == "dry-run" and len(outcome.changes) == 1
    assert list(tmp_path.iterdir()) == []
    assert edit_file(H264, plan(), None).refusals[0].code == "no-output"


def test_verify_write_needs_the_exact_change_list(tmp_path: Path) -> None:
    outcome, out = run(H264, tmp_path, {"op": "rename", "target": "project", "new_name": "x"})
    assert outcome.status == "written"
    assert verify_write(H264, out, [], {"project.xml"}).refusals[0].code == "unintended-change"
    expected = [
        engine.Expected("header", "renamed", "project", H264_PROJECT, "name", "_media_tmpl", "x", 0)
    ]
    assert verify_write(H264, out, expected, {"project.xml"}).ok
    stray = verify_write(H264, out, expected, set())
    assert [r.code for r in stray.refusals] == ["unintended-change"]
    assert stray.refusals[0].details == {"members": ["project.xml"]}


# --------------------------------------------------------------------------- CLI


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(["drp", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "plan.json"
    good.write_text(json.dumps({"edits": [{"op": "rename", "target": "project", "new_name": "x"}]}))
    out = tmp_path / "out.drp"
    code, body = cli(capsys, "edit", str(H264), "--plan", str(good), "--out", str(out))
    assert code == 0 and body["status"] == "written" and body["label"] == "unverified"
    code, body = cli(capsys, "edit", str(H264), "--plan", str(good), "--out", str(out))
    assert code == 1 and body["refusals"][0]["code"] == "out-exists"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"edits": [{"op": "delete_timeline"}]}))
    code, body = cli(
        capsys, "edit", str(H264), "--plan", str(bad), "--out", str(tmp_path / "x.drp")
    )
    assert code == 1 and body["refusals"][0]["code"] == "edit-not-whitelisted"
    later = tmp_path / "later.json"
    later.write_text(json.dumps({"edits": [{"op": "fader_keyframes"}]}))
    code, body = cli(
        capsys, "edit", str(H264), "--plan", str(later), "--out", str(tmp_path / "x.drp")
    )
    assert code == 1 and body["refusals"][0]["code"] == "edit-deferred"
    code, body = cli(
        capsys, "edit", str(tmp_path / "missing.drp"), "--plan", str(good), "--dry-run"
    )
    assert code == 2 and body["refusals"][0]["code"] == "input-unreadable"
    code, body = cli(capsys, "edit", str(H264), "--plan", str(tmp_path / "nope.json"), "--dry-run")
    assert code == 2 and body["refusals"][0]["code"] == "plan-unreadable"
    assert not (tmp_path / "x.drp").exists()


def test_cli_plan_from_stdin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    text = json.dumps({"edits": [{"op": "rename", "target": "project", "new_name": "y"}]})
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(text.encode())))
    code, body = cli(capsys, "edit", str(H264), "--plan", "-", "--dry-run")
    assert code == 0 and body["status"] == "dry-run" and body["changes"][0]["new"] == "y"


def test_cli_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["drp", "edit-schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    ops = {
        v["properties"]["op"]["const"]
        for v in schema["$defs"].values()
        if "op" in v.get("properties", {})
    }
    assert ops == {"rename", "path_map", "marker_add", "marker_update", "marker_delete"}


def test_blob_hex_case_survives_an_edit(tmp_path: Path) -> None:
    """A marker blob written in upper-case hex is re-encoded in upper case."""
    entry = marker_entry(10, BLUE, "A")
    element = carrier(
        "0b0b0b0b-1111-2222-3333-444444444444", SEQUENCE, markers_fields_blob(entry).upper()
    )
    src = edit_zip(H264, tmp_path / "upper.drp", {"project.xml": add_carrier(element)})
    outcome, out = run(
        src,
        tmp_path,
        {
            "op": "marker_update",
            "owner": "timeline",
            "owner_id": TIMELINE,
            "frame": 10,
            "set": {"name": "B"},
        },
    )
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    with zipfile.ZipFile(out) as zf:
        data = zf.read("project.xml")
    start = data.index(b"<FieldsBlob>", data.index(b"0b0b0b0b-1111")) + len(b"<FieldsBlob>")
    text = data[start : data.index(b"</FieldsBlob>", start)]
    assert text == text.upper() and blobs.parse_hex(text.decode())


def test_output_suffix_must_match_the_input_kind(tmp_path: Path) -> None:
    outcome = edit_file(H264, plan(), tmp_path / "out.drt")
    assert [r.code for r in outcome.refusals] == ["out-suffix"]
    assert list(tmp_path.iterdir()) == []
