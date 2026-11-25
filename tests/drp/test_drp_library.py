"""S25a: projects read straight from a Resolve disk library (``Project.db``).

Every fixture is synthetic: :mod:`library_support` lays the XML members of a synthetic ``.drp``
out as library rows, so each project is read both ways and the two models must agree.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import ANY

import pytest
from drp_helpers import envelope, kd, pb_len, qstr, xml
from library_support import (
    OLD,
    SCHEMA,
    Tables,
    find,
    library_db,
    library_tree,
    rows_from_members,
    write_db,
)
from test_drp_synthetic import EXTENTS, RATE_25, RES_1080, clip_blob, marker, markers_blob, members

from aar.cli import main
from aar.drp import read_drp
from aar.drp.diff import diff_projects
from aar.drp.library import (
    LIBRARY_KIND,
    LibraryError,
    LibraryOptions,
    is_library_path,
    list_library,
)
from aar.drp.model import DrpFile
from aar.drp.modify import edit_file, parse_plan

MakeZip = Callable[[str, dict[str, str | bytes]], Path]
FAST = LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0)

# ---------------------------------------------------------------------------- fixtures


def _bytes_entry(key: str, value: bytes) -> tuple[str, int, bytes]:
    return (key, 0x0C, struct.pack(">I", len(value)) + value)


def _tracks_ba(rate: int, channels: int) -> str:
    inner = kd(
        [
            ("DbType", 0x0A, qstr("BtAudioTrack")),
            ("SampleRate", 0x02, struct.pack(">i", rate)),
            ("NumChannels", 0x02, struct.pack(">i", channels)),
            ("CodecName", 0x0A, qstr("lpcm")),
        ]
    )
    return kd([_bytes_entry("0", inner)]).hex()


def _sequence(sid: str, container: str | None, res: bytes = RES_1080) -> str:
    fb = kd([("SeqRef", 0x0A, qstr(container))]).hex() if container else ""
    return (
        f'<Sm2Sequence DbId="{sid}"><FieldsBlob>{fb}</FieldsBlob>'
        f"<MediaExtents>{EXTENTS.hex()}</MediaExtents><FrameRate>{RATE_25.hex()}</FrameRate>"
        f"<Resolution>{res.hex()}</Resolution></Sm2Sequence>"
    )


def _item(tag: str, iid: str, body: str) -> str:
    return f'<Element><{tag} DbId="{iid}">{body}</{tag}></Element>'


def rich_members() -> dict[str, str | bytes]:
    """Nested bins, embedded audio, multicam and compound clips, two timelines, a transition
    (no ``<In>``), a generator (``<In/>``), a title, audio channels and markers."""
    setup = envelope(1, 0x80, pb_len(69, b"25"))
    config_fb = kd([_bytes_entry("SetupBA", setup)]).hex()
    video_time = kd(
        [("NumFrames", 0x02, struct.pack(">i", 500)), _bytes_entry("FrameRate", RATE_25)]
    ).hex()
    geometry = kd([_bytes_entry("Resolution", RES_1080)]).hex()
    cam = clip_blob("D:\\Show", "cam.mov")
    room = clip_blob("E:\\Rec", "room.wav")
    music = clip_blob("D:\\Music", "music.wav")
    project = xml(
        '<SM_Project DbId="p1">\n'
        " <ProjectName>Rich &lt;one&gt;</ProjectName><ProjectVersion>15</ProjectVersion>\n"
        " <LastModTimeInSecs>1700000000</LastModTimeInSecs>\n"
        " <TimelineHandleVec><Element>tl1</Element><Element>tl2</Element></TimelineHandleVec>\n"
        " <CurrentTimelineIndex>1</CurrentTimelineIndex>\n"
        f' <CommonConfig><SM_Config DbId="cfg"><FieldsBlob>{config_fb}</FieldsBlob>'
        "</SM_Config></CommonConfig>\n"
        ' <LockableBlobMap><Sm2LockableBlobMap DbId="lm"><LocableBlobSet>\n'
        '  <Element><Sm2SequenceLockableBlob DbId="lb1">'
        f"<FieldsBlob>{markers_blob(marker(5, 32, 'Laugh', 'big'))}</FieldsBlob>"
        "<BlobOwner>seq1</BlobOwner></Sm2SequenceLockableBlob></Element>\n"
        '  <Element><Sm2TiItemLockableBlob DbId="lb2">'
        f"<FieldsBlob>{markers_blob(marker(3, 2, 'Clip mark'))}</FieldsBlob>"
        "<BlobOwner>v1i1</BlobOwner></Sm2TiItemLockableBlob></Element>\n"
        '  <Element><BtMetadataLockableBlob DbId="lb3"><FieldsBlob/>'
        "<BlobOwner>vi1</BlobOwner></BtMetadataLockableBlob></Element>\n"
        " </LocableBlobSet></Sm2LockableBlobMap></LockableBlobMap>\n"
        "</SM_Project>\n"
    )
    master = xml(
        '<Sm2MpFolder DbId="bin1"><Name>Master</Name><MediaVec>\n'
        ' <Element><Sm2MpVideoClip DbId="clip1"><Name>cam.mov</Name><MpFolder>bin1</MpFolder>'
        "<UniqueMediaPoolItemId>u1</UniqueMediaPoolItemId>"
        f'<Video><BtVideoInfo DbId="vi1"><Clip>{cam}</Clip>'
        f"<Time>{video_time}</Time><Geometry>{geometry}</Geometry></BtVideoInfo></Video>"
        '<EmbeddedAudioVec><Element><BtAudioInfo DbId="ai1">'
        f"<Clip>{cam}</Clip><TracksBA>{_tracks_ba(48000, 2)}"
        '</TracksBA></BtAudioInfo></Element><Element><BtAudioInfo DbId="ai2">'
        f"<Clip>{room}</Clip><TracksBA>{_tracks_ba(96000, 1)}"
        "</TracksBA></BtAudioInfo></Element></EmbeddedAudioVec></Sm2MpVideoClip></Element>\n"
        ' <Element><Sm2MpAudioClip DbId="clip2"><Name>music.wav</Name><MpFolder>bin1</MpFolder>'
        '<EmbeddedAudioVec><Element><BtAudioInfo DbId="ai3">'
        f"<Clip>{music}</Clip><TracksBA>{_tracks_ba(48000, 2)}"
        "</TracksBA></BtAudioInfo></Element></EmbeddedAudioVec></Sm2MpAudioClip></Element>\n"
        ' <Element><Sm2MpTimelineClip DbId="tc1"><Name>Cut</Name><MpFolder>bin1</MpFolder>'
        '<TimelineSharedHandle><Sm2Timeline DbId="tl1"><Name>Cut</Name><Sequence>'
        f"{_sequence('seq1', 'cont1')}</Sequence></Sm2Timeline></TimelineSharedHandle>"
        "</Sm2MpTimelineClip></Element>\n"
        ' <Element><Sm2MpTimelineClip DbId="tc2"><Name>Short</Name><MpFolder>bin1</MpFolder>'
        '<TimelineSharedHandle><Sm2Timeline DbId="tl2"><Name>Short</Name><Sequence>'
        f"{_sequence('seq2', None, bytes(16))}</Sequence></Sm2Timeline></TimelineSharedHandle>"
        "</Sm2MpTimelineClip></Element>\n"
        "</MediaVec></Sm2MpFolder>\n"
    )
    sub = xml(
        '<Sm2MpFolder DbId="bin2"><Name>Sub</Name><MpFolder>bin1</MpFolder><MediaVec>\n'
        ' <Element><Sm2MpMulticamClip DbId="mc1"><Name>Multi</Name><MpFolder>bin2</MpFolder>'
        f"<Sequence>{_sequence('seq3', 'cont3')}</Sequence></Sm2MpMulticamClip></Element>\n"
        ' <Element><Sm2MpCompoundClip DbId="cc1"><Name>Comp</Name><MpFolder>bin2</MpFolder>'
        f"<Sequence>{_sequence('seq4', None)}</Sequence></Sm2MpCompoundClip></Element>\n"
        "</MediaVec></Sm2MpFolder>\n"
    )
    half = struct.pack("<d", 0.5).hex()
    timemap = kd(
        [
            ("DbType", 0x0A, qstr("Sm2TimeMap")),
            ("XMax", 0x06, struct.pack(">d", 2.0)),
            ("LastValidYOffset", 0x06, struct.pack(">d", 4.0)),
        ]
    ).hex()
    v1 = (
        _item(
            "Sm2TiVideoClip",
            "v1i1",
            "<PrettyType/><Name>cam.mov</Name><Start>90000</Start><Duration>100</Duration>"
            "<In>25</In><MediaRef>clip1</MediaRef><MediaFilePath>D:\\Show\\cam.mov</MediaFilePath>",
        )
        + _item(
            "Sm2TiTransition",
            "v1t",
            "<Name>Cross Dissolve</Name><Start>90090</Start><Duration>20</Duration>",
        )
        + _item(
            "Sm2TiVideoClip",
            "v1i2",
            f"<Name>cam.mov</Name><Start>90100</Start><Duration>50|{half}</Duration>"
            f"<In>400</In><MediaRef>clip1</MediaRef><MediaTimemapBA>{timemap}</MediaTimemapBA>",
        )
    )
    v2 = _item(
        "Sm2TiGenerator",
        "v2g",
        "<Name>Solid Color</Name><Start>90000</Start><Duration>30</Duration><In/>",
    ) + _item(
        "Sm2TiVideoClip",
        "v2t",
        "<PrettyType>Fusion Title</PrettyType><Name>Text+</Name><Start>90040</Start>"
        "<Duration>40</Duration><In/>",
    )
    a1 = _item(
        "Sm2TiAudioClip",
        "a1i1",
        "<Name>cam.mov</Name><Start>90000</Start><Duration>100</Duration><In>25</In>"
        "<MediaRef>clip1</MediaRef><MediaTrackIdx>0</MediaTrackIdx>",
    ) + _item(
        "Sm2TiAudioClip",
        "a1i2",
        "<Name>room.wav</Name><Start>90000</Start><Duration>100</Duration><In/>"
        "<MediaRef>clip1</MediaRef><MediaTrackIdx>1</MediaTrackIdx>",
    )

    def track(tid: str, kind: int, seq: str, items: str, name: str = "") -> str:
        label = f"<UserDefinedName>{name}</UserDefinedName>" if name else "<UserDefinedName/>"
        return (
            f'<Element><Sm2TiTrack DbId="{tid}"><Type>{kind}</Type><SubType>0</SubType>'
            f"<Flags>0</Flags><Sequence>{seq}</Sequence><Items>{items}</Items>{label}"
            "</Sm2TiTrack></Element>"
        )

    cont1 = xml(
        '<Sm2SequenceContainer DbId="cont1">'
        f"<VideoTrackVec>{track('t1', 0, 'seq1', v1, 'Main')}{track('t2', 0, 'seq1', v2)}"
        f"</VideoTrackVec><AudioTrackVec>{track('t3', 1, 'seq1', a1)}</AudioTrackVec>"
        "<SubtitleTrackVec/></Sm2SequenceContainer>\n"
    )
    short = _item(
        "Sm2TiAudioClip",
        "s1",
        "<Name>music.wav</Name><Start>90000</Start><Duration>75</Duration><In>0</In>"
        "<MediaRef>clip2</MediaRef><MediaTrackIdx>0</MediaTrackIdx>",
    )
    cont2 = xml(
        '<Sm2SequenceContainer DbId="cont2"><VideoTrackVec/>'
        f"<AudioTrackVec>{track('t4', 1, 'seq2', short)}</AudioTrackVec></Sm2SequenceContainer>\n"
    )
    angle = _item(
        "Sm2TiVideoClip",
        "m1",
        "<Name>cam.mov</Name><Start>90000</Start><Duration>500</Duration><In/>"
        "<MediaRef>clip1</MediaRef>",
    )
    cont3 = xml(
        '<Sm2SequenceContainer DbId="cont3">'
        f"<VideoTrackVec>{track('t5', 0, 'seq3', angle, 'Angle 1')}</VideoTrackVec>"
        "</Sm2SequenceContainer>\n"
    )
    return {
        "project.xml": project,
        "Gallery.xml": xml('<Gallery::GyGallery DbId="g"/>'),
        "MediaPool/Master/MpFolder.xml": master,
        "MediaPool/Master/Sub/MpFolder.xml": sub,
        "SeqContainer/cont1.xml": cont1,
        "SeqContainer/cont2.xml": cont2,
        "SeqContainer/cont3.xml": cont3,
    }


# ---------------------------------------------------------------------------- comparing

_SOURCE_ONLY = (
    "path",
    "kind",
    "kind_source",
    "sniffed_kind",
    "size",
    "sha256",
    "is_zip",
    "signature",
    "head_hex",
    "members",
    "raw_members",
)


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip(v)
            for k, v in value.items()
            if k not in ("source", "spans", "where", "member")
        }
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def semantic(drp: DrpFile) -> dict[str, Any]:
    """The model without what only says where it came from (file, members, provenance) and
    without ``DbAppVer``, which a library does not store."""
    data = drp.to_dict()
    for key in _SOURCE_ONLY:
        data.pop(key)
    versions = data.pop("versions")
    data["project_version"] = versions["project_version"] if versions else None
    data["issues"] = [(i["severity"], i["code"], i["message"]) for i in data["issues"]]
    return _strip(data)


def versioned(project: dict[str, str | bytes]) -> dict[str, str | bytes]:
    """``project`` with the ``<ProjectVersion>`` every library project stores."""
    out = dict(project)
    out["project.xml"] = str(out["project.xml"]).replace(
        "</ProjectName>", "</ProjectName><ProjectVersion>15</ProjectVersion>", 1
    )
    return out


def both(
    make_zip: MakeZip, tmp_path: Path, project: dict[str, str | bytes]
) -> tuple[DrpFile, DrpFile]:
    a = read_drp(make_zip("project.drp", project))
    b = read_drp(library_db(tmp_path / "Project.db", project), library_options=FAST)
    return a, b


def codes(drp: DrpFile) -> set[str]:
    return {i.code for i in drp.issues}


def edited(
    tmp_path: Path,
    project: dict[str, str | bytes],
    edit: Callable[[Tables], None],
    name: str = "Project.db",
) -> Path:
    return library_db(tmp_path / name, project, edit)


def refusal(path: Path, options: LibraryOptions = FAST, **kw: Any) -> LibraryError:
    with pytest.raises(LibraryError) as info:
        read_drp(path, library_options=options, **kw)
    return info.value


# ---------------------------------------------------------------------------- parity


def test_synthetic_project_reads_the_same_from_zip_and_library(
    make_zip: MakeZip, tmp_path: Path
) -> None:
    a, b = both(make_zip, tmp_path, versioned(members()))
    assert semantic(a) == semantic(b)
    assert (b.kind, b.kind_source, b.is_zip, b.signature) == (
        LIBRARY_KIND,
        "sqlite",
        False,
        "sqlite",
    )
    assert b.members == [] and b.raw_members == []
    assert b.versions is not None and (b.versions.app_version, b.versions.project_version) == (
        None,
        15,
    )
    assert b.path == str(tmp_path / "Project.db")
    assert b.sha256 == hashlib.sha256((tmp_path / "Project.db").read_bytes()).hexdigest()
    # The synthetic project's own warnings come through unchanged.
    assert {"unresolved-media", "stale-media-file-path"} <= codes(b)


def test_rich_project_reads_the_same_from_zip_and_library(
    make_zip: MakeZip, tmp_path: Path
) -> None:
    a, b = both(make_zip, tmp_path, rich_members())
    assert semantic(a) == semantic(b)
    assert [x.path for x in b.bins] == ["Master", "Master/Sub"]
    cut = next(t for t in b.timelines if t.name == "Cut")
    v1, v2, a1 = cut.tracks
    clip, transition, retimed = v1.items
    assert (clip.source_in, transition.source_in, transition.kind) == (25, None, "transition")
    assert (retimed.timemap_form, retimed.speed, retimed.source_out, retimed.fractions) == (
        "retimed",
        2.0,
        None,
        {"duration": 0.5},
    )
    generator, title = v2.items
    assert (generator.kind, generator.source_in, title.kind) == ("generator", 0, "title")
    assert [i.media_track_index for i in a1.items] == [0, 1]
    assert [m.name for m in cut.markers] == ["Laugh"] and clip.markers[0].name == "Clip mark"
    media = b.media_by_id()
    assert [t.sample_rate for t in media["clip1"].audio_tracks] == [48000, 96000]
    assert media["clip1"].linked_paths == ["E:\\Rec\\room.wav"]
    assert b.project is not None and b.project.frame_rate_hint == "25"
    assert {t.kind for t in b.timelines} == {"timeline", "multicam", "compound"}


def test_library_provenance_names_rows_and_columns(tmp_path: Path) -> None:
    drp = read_drp(library_db(tmp_path / "Project.db", rich_members()), library_options=FAST)
    cut = next(t for t in drp.timelines if t.name == "Cut")
    item = cut.tracks[0].items[0]
    assert item.source is not None and (item.source.member, item.source.kind) == (
        "Sm2TiItem/v1i1",
        "row",
    )
    start = item.spans["record_start"]
    assert (start.member, start.start, start.end, start.kind) == (
        "Sm2TiItem/v1i1/Start",
        0,
        0,
        "column",
    )
    rate = cut.spans["frame_rate"]
    assert (rate.member, rate.kind, rate.blob_path) == (
        "Sm2Sequence/seq1/FrameRate",
        "blob",
        "FrameRate",
    )
    assert drp.project is not None
    assert (
        drp.project.spans["timeline_ids[1]"].member
        == "SM_Project_Sm2Timeline/p1/TimelineHandleVec/1"
    )
    assert [b.member for b in drp.bins] == ["Sm2MpFolder/bin1", "Sm2MpFolder/bin2"]
    assert cut.member == "Sm2SequenceContainer/cont1"
    assert drp.versions is not None and drp.versions.source is not None
    assert drp.versions.source.member == "SM_Project/p1/ProjectVersion"
    for marker_ in cut.markers:
        assert (
            marker_.source is not None and marker_.source.member == "BtLockableBlob/lb1/FieldsBlob"
        )


def test_diff_of_a_drp_and_its_library_project_reports_nothing(
    make_zip: MakeZip, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = rich_members()
    zipped = make_zip("project.drp", project)
    db = library_db(tmp_path / "Project.db", project)
    result = diff_projects(read_drp(zipped), read_drp(db, library_options=FAST))
    assert result.changes == [] and result.identical
    assert not result.gate.passed
    assert any("undecoded content was not compared" in n for n in result.notes)
    assert any("DbAppVer" in n for n in result.notes)
    assert main(["drp", "diff", str(zipped), str(db)]) == 0
    assert "undecoded content was not compared" in capsys.readouterr().out
    assert main(["drp", "diff", "--gate", str(zipped), str(db)]) == 1


def test_diff_skips_zero_resolution_only_for_the_library_side(
    make_zip: MakeZip, tmp_path: Path
) -> None:
    project = rich_members()

    def zero(tables: Tables) -> None:
        find(tables, "Sm2Sequence", "seq1")["Resolution"] = bytes(16)

    db = edited(tmp_path, project, zero)
    zipped = read_drp(make_zip("project.drp", project))
    lib = read_drp(db, library_options=FAST)
    cut = next(t for t in lib.timelines if t.name == "Cut")
    assert cut.resolution == (0, 0)  # what the library stores
    forward = diff_projects(zipped, lib)
    assert forward.changes == []
    assert any('"Cut"' in n and "0x0" in n for n in forward.notes)
    assert diff_projects(lib, zipped).changes == []
    # Between two library projects a resolution change is a change.
    other = read_drp(library_db(tmp_path / "Other.db", project), library_options=FAST)
    assert [c.field_name for c in diff_projects(other, lib).changes] == ["resolution"]


def test_diff_of_two_library_saves_reports_the_trim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = rich_members()
    before = library_db(tmp_path / "before.db", project)

    def trim(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1i1")["Duration"] = "90"

    after = edited(tmp_path, project, trim, "after.db")
    os.utime(before, (OLD, OLD))
    assert main(["drp", "diff", str(before), str(after)]) == 1
    out = capsys.readouterr().out
    assert "100 → 90" in out or "trimmed" in out
    result = diff_projects(
        read_drp(before, library_options=FAST), read_drp(after, library_options=FAST)
    )
    assert [(c.kind, c.field_name, c.old, c.new) for c in result.changes] == [
        ("trimmed", "timing", ANY, ANY)
    ]
    assert not result.gate.passed


def test_describe_and_inspect_say_it_is_a_library_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = library_db(tmp_path / "Project.db", rich_members())
    assert main(["drp", "describe", str(db)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "LIBRARY project (Resolve disk library, SQLite Project.db): project version 15"
    )
    assert "not a zip archive" not in out and 'Timeline "Cut"' in out
    assert main(["drp", "inspect", "--json", str(db)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "library" and data["members"] == []


def test_modify_refuses_a_library_project(tmp_path: Path) -> None:
    db = library_db(tmp_path / "Project.db", rich_members())
    before = db.read_bytes()
    plan = parse_plan(
        json.dumps({"edits": [{"op": "rename", "target": "project", "new_name": "x"}]})
    )
    outcome = edit_file(db, plan, tmp_path / "out.drp")
    assert outcome.status == "refused"
    assert outcome.refusals[0].code == "library-project"
    assert not (tmp_path / "out.drp").exists() and db.read_bytes() == before


# ---------------------------------------------------------------------------- which files


def test_only_db_files_and_sqlite_content_are_library_projects(tmp_path: Path) -> None:
    sqlite_drb = tmp_path / "bin.drb"
    sqlite_drb.write_bytes(b"SQLite format 3\x00" + bytes(100))
    other = tmp_path / "Project.sqlite"
    other.write_bytes(b"SQLite format 3\x00" + bytes(100))
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    assert not is_library_path(sqlite_drb) and not is_library_path(tmp_path / "x.drp")
    assert is_library_path(tmp_path / "missing.db") and is_library_path(other)
    assert not is_library_path(text)


# ---------------------------------------------------------------------------- refusals


def test_a_db_that_is_not_sqlite_is_not_a_project(tmp_path: Path) -> None:
    path = tmp_path / "Project.db"
    path.write_bytes(b"PK\x03\x04 not a database at all")
    os.utime(path, (OLD, OLD))
    assert refusal(path).code == "not_a_project"


def test_a_truncated_db_is_damaged(tmp_path: Path) -> None:
    full = library_db(tmp_path / "full.db", rich_members()).read_bytes()
    path = tmp_path / "Project.db"
    path.write_bytes(full[: len(full) // 2])
    os.utime(path, (OLD, OLD))
    error = refusal(path)
    assert error.code == "damaged" and "save it again" in error.hint


def test_a_corrupt_page_fails_the_integrity_check(tmp_path: Path) -> None:
    path = library_db(tmp_path / "Project.db", rich_members())
    data = bytearray(path.read_bytes())
    page = 4096 if len(data) > 8192 else 1024
    data[page * 2 + 10 : page * 2 + 40] = b"\xff" * 30
    path.write_bytes(bytes(data))
    os.utime(path, (OLD, OLD))
    assert refusal(path).code == "damaged"


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("Sm2TiItem", None),
        ("Sm2TiItem_Sm2TiTrack", None),
        ("Sm2TiItem", "Duration"),
        ("SM_Project", "ProjectVersion"),
    ],
)
def test_missing_tables_or_columns_are_not_a_project(
    tmp_path: Path, table: str, column: str | None
) -> None:
    schema = {t: list(c) for t, c in SCHEMA.items()}
    if column is None:
        del schema[table]
    else:
        schema[table].remove(column)
    path = write_db(tmp_path / "Project.db", rows_from_members(rich_members()), schema=schema)
    error = refusal(path)
    assert error.code == "not_a_project"
    assert (f"{table}.{column}" if column else table) in error.message


def test_a_view_in_place_of_a_table_is_not_a_project(tmp_path: Path) -> None:
    schema = {t: c for t, c in SCHEMA.items() if t != "Sm2TiItem"}
    path = write_db(tmp_path / "Project.db", rows_from_members(rich_members()), schema=schema)
    conn = sqlite3.connect(path)
    conn.execute('CREATE VIEW "Sm2TiItem" AS SELECT 1 AS "Sm2TiItem_id"')
    conn.commit()
    conn.close()
    os.utime(path, (OLD, OLD))
    assert refusal(path).code == "not_a_project"


@pytest.mark.parametrize("version", [16, 14, None, "15x"])
def test_unsupported_project_versions_are_refused(tmp_path: Path, version: object) -> None:
    def set_version(tables: Tables) -> None:
        tables["SM_Project"][0]["ProjectVersion"] = version

    error = refusal(edited(tmp_path, rich_members(), set_version))
    assert error.code == "unsupported_version" and ".drp" in error.hint


def test_version_17_is_read(tmp_path: Path) -> None:
    def v17(tables: Tables) -> None:
        tables["SM_Project"][0]["ProjectVersion"] = 17

    drp = read_drp(edited(tmp_path, rich_members(), v17), library_options=FAST)
    assert drp.versions is not None and drp.versions.project_version == 17


def test_project_rows_must_be_exactly_one(tmp_path: Path) -> None:
    def none(tables: Tables) -> None:
        tables["SM_Project"].clear()

    def two(tables: Tables) -> None:
        tables["SM_Project"].append({**tables["SM_Project"][0], "SM_Project_id": "p2"})

    def nameless(tables: Tables) -> None:
        tables["SM_Project"][0]["SM_Project_id"] = None

    assert refusal(edited(tmp_path, rich_members(), none, "a.db")).code == "not_a_project"
    assert refusal(edited(tmp_path, rich_members(), two, "b.db")).code == "damaged"
    assert refusal(edited(tmp_path, rich_members(), nameless, "c.db")).code == "damaged"


# ---------------------------------------------------------------------------- hostile values


@pytest.mark.parametrize(
    ("start", "duration", "issue"),
    [
        (None, "100", None),
        ("90000", None, None),
        ("abc", "100", "bad-frame-count"),
        (90000.0, "100", "bad-frame-count"),
        ("90000", "1e3", "bad-frame-count"),
        (b"\xff\xfe", "100", "bad-frame-count"),
    ],
)
def test_null_and_non_numeric_frames_are_read_as_the_reader_reads_them(
    tmp_path: Path, start: object, duration: object, issue: str | None
) -> None:
    def frames(tables: Tables) -> None:
        row = find(tables, "Sm2TiItem", "v1i1")
        row["Start"], row["Duration"] = start, duration

    drp = read_drp(edited(tmp_path, rich_members(), frames), library_options=FAST)
    item = next(t for t in drp.timelines if t.name == "Cut").tracks[0].items[0]
    assert item.record_end is None
    assert (issue in codes(drp)) if issue else ("bad-frame-count" not in codes(drp))


def test_null_in_is_an_empty_in_except_on_transitions(tmp_path: Path) -> None:
    def nulls(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1i1")["In"] = None
        find(tables, "Sm2TiItem", "v1t")["In"] = "7"  # a stored value is always read

    drp = read_drp(edited(tmp_path, rich_members(), nulls), library_options=FAST)
    clip, transition, _ = next(t for t in drp.timelines if t.name == "Cut").tracks[0].items
    assert clip.source_in == 0 and transition.source_in == 7


def test_text_that_is_not_utf8_is_read_with_an_issue(tmp_path: Path) -> None:
    def bad(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1i1")["Name"] = b"caf\xe9 <&>"

    drp = read_drp(edited(tmp_path, rich_members(), bad), library_options=FAST)
    item = next(t for t in drp.timelines if t.name == "Cut").tracks[0].items[0]
    assert item.name == "caf\ufffd <&>" and "scan-invalid-utf8" in codes(drp)


def test_a_link_to_a_missing_row_is_reported_and_refuses_the_import(tmp_path: Path) -> None:
    def dangle(tables: Tables) -> None:
        tables["Sm2TiItem"] = [r for r in tables["Sm2TiItem"] if r["Sm2TiItem_id"] != "v1i2"]
        find(tables, "Sm2MpMedia", "tc2")["TimelineSharedHandle"] = "nowhere"

    path = edited(tmp_path, rich_members(), dangle)
    drp = read_drp(path, library_options=FAST)
    dangling = [i for i in drp.issues if i.code == "library-dangling-link"]
    assert len(dangling) == 2 and drp.issues[: len(dangling)] == dangling
    assert sorted(i.where.member for i in dangling if i.where is not None) == [
        "Sm2MpMedia/tc2/TimelineSharedHandle",
        "Sm2TiItem_Sm2TiTrack/t1/Items/2",
    ]
    assert [i.id for i in next(t for t in drp.timelines if t.name == "Cut").tracks[0].items] == [
        "v1i1",
        "v1t",
    ]


def test_duplicate_ids_are_reported_and_the_first_row_wins(tmp_path: Path) -> None:
    def dup(tables: Tables) -> None:
        tables["Sm2TiItem"].append({**find(tables, "Sm2TiItem", "v1i1"), "Duration": "1"})
        tables["Sm2MpMedia"].append({**find(tables, "Sm2MpMedia", "clip2"), "Name": "other"})

    drp = read_drp(edited(tmp_path, rich_members(), dup), library_options=FAST)
    assert len([i for i in drp.issues if i.code == "duplicate-id"]) == 2
    item = next(t for t in drp.timelines if t.name == "Cut").tracks[0].items[0]
    assert item.duration == 100 and drp.media_by_id()["clip2"].name == "music.wav"


def test_a_row_linked_twice_is_read_once(tmp_path: Path) -> None:
    def twice(tables: Tables) -> None:
        tables["Sm2TiItem_Sm2TiTrack"].append(
            {"DbOwner": "t2", "DbAssociate": "v1i1", "DbPropertyName": "Items", "DbIndex": 9}
        )

    drp = read_drp(edited(tmp_path, rich_members(), twice), library_options=FAST)
    assert "library-link-conflict" in codes(drp)
    cut = next(t for t in drp.timelines if t.name == "Cut")
    assert sum(i.id == "v1i1" for t in cut.tracks for i in t.items) == 1


def test_owner_column_and_join_disagreeing_is_a_conflict(tmp_path: Path) -> None:
    def owner(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1i1")["Sm2TiTrack_id"] = "t3"

    drp = read_drp(edited(tmp_path, rich_members(), owner), library_options=FAST)
    [conflict] = [i for i in drp.issues if i.code == "library-link-conflict"]
    assert "Sm2TiTrack_id names t3" in conflict.message


def test_an_unlinked_row_is_reported(tmp_path: Path) -> None:
    def unlink(tables: Tables) -> None:
        tables["Sm2TiItem_Sm2TiTrack"] = [
            r for r in tables["Sm2TiItem_Sm2TiTrack"] if r["DbAssociate"] != "v1i2"
        ]

    drp = read_drp(edited(tmp_path, rich_members(), unlink), library_options=FAST)
    [issue] = [i for i in drp.issues if i.code == "library-unlinked-row"]
    assert "1 Sm2TiItem row(s)" in issue.message


def test_join_order_follows_dbindex_and_ties_are_conflicts(tmp_path: Path) -> None:
    def reorder(tables: Tables) -> None:
        for row in tables["Sm2TiItem_Sm2TiTrack"]:
            if row["DbOwner"] == "t1":
                row["DbIndex"] = {"v1i1": 2, "v1t": 0, "v1i2": 1}[row["DbAssociate"]]
        tables["Sm2TiItem_Sm2TiTrack"].reverse()

    drp = read_drp(edited(tmp_path, rich_members(), reorder), library_options=FAST)
    cut = next(t for t in drp.timelines if t.name == "Cut")
    assert [i.id for i in cut.tracks[0].items] == ["v1t", "v1i2", "v1i1"]
    assert "library-link-conflict" not in codes(drp)

    def tie(tables: Tables) -> None:
        for row in tables["Sm2TiItem_Sm2TiTrack"]:
            if row["DbOwner"] == "t1":
                row["DbIndex"] = 0

    tied = read_drp(edited(tmp_path, rich_members(), tie, "tie.db"), library_options=FAST)
    assert "library-link-conflict" in codes(tied)


def test_a_cycle_in_bin_parents_is_read_and_reported(tmp_path: Path) -> None:
    def cycle(tables: Tables) -> None:
        find(tables, "Sm2MpFolder", "bin1")["MpFolder"] = "bin2"

    drp = read_drp(edited(tmp_path, rich_members(), cycle), library_options=FAST)
    assert "bin-cycle" in codes(drp) and len(drp.bins) == 2


def test_a_type_that_is_not_an_element_name_is_not_read(tmp_path: Path) -> None:
    def bad(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1t")["DbType"] = 'Sm2TiTransition DbId="x"><evil'

    drp = read_drp(edited(tmp_path, rich_members(), bad), library_options=FAST)
    assert "library-bad-type" in codes(drp)
    cut = next(t for t in drp.timelines if t.name == "Cut")
    assert [i.id for i in cut.tracks[0].items] == ["v1i1", "v1i2"]


def test_names_with_markup_survive(tmp_path: Path) -> None:
    def markup(tables: Tables) -> None:
        find(tables, "Sm2Timeline", "tl1")["Name"] = '</Name><Name>x & "y"'

    drp = read_drp(edited(tmp_path, rich_members(), markup), library_options=FAST)
    assert '</Name><Name>x & "y"' in [t.name for t in drp.timelines]


# ---------------------------------------------------------------------------- budgets


def test_an_oversized_value_is_too_large(tmp_path: Path) -> None:
    def big(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1i1")["MediaTimemapBA"] = b"\x00" * 50_000

    path = edited(tmp_path, rich_members(), big)
    assert refusal(path, max_member_bytes=20_000).code == "too_large"


def test_row_file_and_xml_budgets(tmp_path: Path) -> None:
    path = library_db(tmp_path / "Project.db", rich_members())
    assert (
        refusal(path, LibraryOptions(settle_seconds=0, deadline_seconds=0, max_rows=10)).code
        == "too_large"
    )
    assert (
        refusal(path, LibraryOptions(settle_seconds=0, deadline_seconds=0, max_db_bytes=1000)).code
        == "too_large"
    )
    assert refusal(path, max_total_xml_bytes=2_000).code == "too_large"


def test_a_query_past_its_time_budget_is_too_large(tmp_path: Path) -> None:
    def many(tables: Tables) -> None:
        base = find(tables, "Sm2TiItem", "v1i1")
        tables["Sm2TiItem"].extend({**base, "Sm2TiItem_id": f"x{n}"} for n in range(5000))

    path = edited(tmp_path, rich_members(), many)
    ticks = iter(range(0, 10**9, 1000))
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=0, max_read_seconds=1, clock=lambda: float(next(ticks))
    )
    error = refusal(path, options)
    assert error.code == "too_large" and "longer than" in error.message


# ---------------------------------------------------------------------------- reading safely


def _snapshots(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_nothing_is_left_beside_the_file_or_in_the_snapshot_folder(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    db = library_db(project / "Project.db", rich_members())
    before = db.read_bytes()
    stat = db.stat()
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, snapshot_dir=snaps)
    read_drp(db, library_options=options)
    assert _snapshots(project) == ["Project.db"] and _snapshots(snaps) == []
    assert db.read_bytes() == before and db.stat().st_mtime_ns == stat.st_mtime_ns
    # Refusals clean up too.
    (project / "Project.db").write_bytes(before[:5000])
    os.utime(db, (OLD, OLD))
    with pytest.raises(LibraryError):
        read_drp(db, library_options=options)
    assert _snapshots(project) == ["Project.db"] and _snapshots(snaps) == []


def test_a_journal_beside_the_file_means_busy(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    db = library_db(project / "Project.db", rich_members())
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    for side in ("Project.db-journal", "Project.db-wal"):
        (project / side).write_bytes(b"")
        error = refusal(
            db, LibraryOptions(settle_seconds=0, deadline_seconds=0, snapshot_dir=snaps)
        )
        assert error.code == "busy" and side in error.message and "wait" in error.hint
        (project / side).unlink()
        assert _snapshots(snaps) == []


class _Clock:
    """A fake monotonic clock; ``sleep`` advances it and runs ``on_sleep``."""

    def __init__(self, on_sleep: Callable[[int], None] = lambda n: None) -> None:
        self.t = 1000.0
        self.sleeps = 0
        self.on_sleep = on_sleep

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.t += max(seconds, 0.001)
        self.on_sleep(self.sleeps)


def _touch(path: Path, n: int) -> None:
    os.utime(path, (OLD + n, OLD + n))


def test_a_file_that_keeps_changing_is_busy(tmp_path: Path) -> None:
    db = library_db(tmp_path / "Project.db", rich_members())
    fake = _Clock(lambda n: _touch(db, n))
    options = LibraryOptions(
        settle_seconds=2,
        deadline_seconds=10,
        poll_seconds=1,
        clock=fake.clock,
        sleep=fake.sleep,
        now=lambda: OLD + 1.0,  # the file's mtime always looks fresh
    )
    error = refusal(db, options)
    assert error.code == "busy" and fake.sleeps >= 9


def test_a_file_read_once_it_stops_changing(tmp_path: Path) -> None:
    db = library_db(tmp_path / "Project.db", rich_members())
    fake = _Clock(lambda n: _touch(db, n) if n <= 3 else None)
    options = LibraryOptions(
        settle_seconds=2,
        deadline_seconds=30,
        poll_seconds=0.5,
        clock=fake.clock,
        sleep=fake.sleep,
        now=lambda: OLD + 1.0,  # mtime looks fresh: only watching can settle it (clock skew)
    )
    drp = read_drp(db, library_options=options)
    assert drp.project is not None and 6 <= fake.sleeps <= 10


def test_a_file_changed_during_the_copy_is_copied_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    from aar.media import settle

    db = library_db(tmp_path / "Project.db", rich_members())
    real_open = settle.open_regular
    opened = {"n": 0}

    @contextlib.contextmanager
    def open_and_touch(path: Path, *, dir_fd: int | None = None):  # type: ignore[no-untyped-def]
        # The real descriptor; on the first copy another program "saves" while it is read.
        with real_open(path, dir_fd=dir_fd) as stream:
            opened["n"] += 1
            if opened["n"] == 1:
                read = stream.read

                def read_then_touch(size: int = -1) -> bytes:
                    data = read(size)
                    os.utime(path, (OLD + 50, OLD + 50))
                    return data

                stream.read = read_then_touch  # type: ignore[method-assign]
            yield stream

    monkeypatch.setattr(settle, "open_regular", open_and_touch)
    fake = _Clock()
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=5, poll_seconds=1, clock=fake.clock, sleep=fake.sleep
    )
    assert read_drp(db, library_options=options).project is not None
    assert (opened["n"], fake.sleeps) == (2, 1)


def test_symlinks_and_snapshot_folders_inside_the_project_are_refused(tmp_path: Path) -> None:
    db = library_db(tmp_path / "real.db", rich_members())
    link = tmp_path / "Project.db"
    link.symlink_to(db)
    assert refusal(link).code == "unreadable"
    inside = tmp_path / "inside"
    inside.mkdir()
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, snapshot_dir=inside)
    assert refusal(db, options).code == "unreadable"
    assert refusal(tmp_path / "missing.db").code == "unreadable"


def test_cli_reports_a_refusal_with_its_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "Project.db"
    db.write_bytes(b"not a database")
    os.utime(db, (OLD, OLD))
    assert main(["drp", "describe", str(db)]) == 1
    err = capsys.readouterr().err
    assert "[not_a_project]" in err and "give the Project.db of a project" in err
    assert main(["drp", "diff", str(db), str(db)]) == 2
    assert "[not_a_project]" in capsys.readouterr().err


# ---------------------------------------------------------------------------- listing


def test_library_listing_lists_every_project_and_every_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = library_tree(tmp_path / "lib", [("Rich", rich_members()), ("Small", members())])
    base = root / "Resolve Projects" / "Users" / "guest" / "Projects"
    (base / "Busy").mkdir()
    library_db(base / "Busy" / "Project.db", members())
    (base / "Busy" / "Project.db-journal").write_bytes(b"")
    (base / "Folder" / "Junk").mkdir(parents=True)
    (base / "Folder" / "Junk" / "Project.db").write_bytes(b"not sqlite")
    os.utime(base / "Folder" / "Junk" / "Project.db", (OLD, OLD))

    listing = list_library(root, options=FAST)
    by_folder = {p.folder: p for p in listing.projects}
    assert sorted(by_folder) == ["Busy", "Folder/Junk", "Rich", "Small"]
    rich = by_folder["Rich"]
    assert (rich.name, rich.id, rich.project_version) == ("Rich <one>", "p1", 15)
    assert rich.timelines == ["Short", "Cut"] or rich.timelines == ["Cut", "Short"]
    assert rich.last_modified == "2023-11-14T22:13:20+00:00" and rich.refusal is None
    assert by_folder["Busy"].refusal is not None and by_folder["Busy"].refusal["code"] == "busy"
    assert by_folder["Folder/Junk"].refusal is not None
    assert by_folder["Folder/Junk"].refusal["code"] == "not_a_project"
    assert "unresolved-media" in by_folder["Small"].warnings

    assert main(["drp", "library", str(root), "--wait", "0", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert {p["folder"]: (p["refusal"] or {}).get("code") for p in data["projects"]} == {
        "Busy": "busy",
        "Folder/Junk": "not_a_project",
        "Rich": None,
        "Small": None,
    }
    assert main(["drp", "library", str(root / "Resolve Projects"), "--wait", "0"]) == 1
    text = capsys.readouterr().out
    assert "projects: 4 (2 refused)" in text and '"guest/Rich": "Rich <one>" version 15' in text
    assert '"guest/Busy": REFUSED busy:' in text and "INCOMPLETE" not in text


def test_a_library_of_readable_projects_exits_0_and_a_non_library_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = library_tree(tmp_path / "lib", [("Rich", rich_members())])
    assert main(["drp", "library", str(root), "--wait", "0"]) == 0
    capsys.readouterr()
    assert main(["drp", "library", str(tmp_path)]) == 1
    assert "not a Resolve disk library" in capsys.readouterr().err
    assert main(["drp", "library", str(root), "--wait", "-1"]) == 2


def test_zip_reading_is_unchanged_for_a_drp_with_sqlite_suffix_rules(make_zip: MakeZip) -> None:
    path = make_zip("project.drp", members())
    with zipfile.ZipFile(path) as zf:
        assert "project.xml" in zf.namelist()
    drp = read_drp(path)
    assert drp.kind == "drp" and drp.is_zip and drp.members
