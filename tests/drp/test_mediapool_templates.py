"""S25c1: template clips from a saved project, the encoders, the field compare and the gate."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

import mediapool_support as ms
import pytest

from aar.drp import blobs
from aar.drp.library import LibraryError, LibraryOptions, read_rows
from aar.drp.mediapool import facts_from_json, gate
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.fields import EXCLUDED, compare, flatten
from aar.drp.mediapool.templates import (
    ClipRows,
    clips_from_rows,
    project_frame_rate,
    read_templates,
)
from aar.drp.modify import encode
from aar.drp.modify.encode import EncodeError

FAST = LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0)


# --------------------------------------------------------------------------- encoders


def test_keyed_dict_encoder_is_the_inverse_of_the_decoder() -> None:
    entries = [
        ("s", blobs.KD_STRING, 0, "text é"),
        ("n", blobs.KD_STRING, 0, None),
        ("i", blobs.KD_INT32, 0, -5),
        ("u", blobs.KD_UINT32, 0, 7),
        ("l", blobs.KD_INT64, 0, 2**40),
        ("q", blobs.KD_UINT64, 0, 2**63),
        ("d", blobs.KD_DOUBLE, 0, 29.97),
        ("f", blobs.KD_FLOAT, 1, 1.5),
        ("b", blobs.KD_BOOL, 0, True),
        ("y", blobs.KD_BYTES, 0, b"\x00\x01"),
        ("z", blobs.KD_BYTES, 0, None),
        ("c", blobs.KD_CHAR, 0, "x"),
        ("t", blobs.KD_DATE, 0, bytes(9)),
    ]
    data = encode.keyed_dict(entries)
    decoded = blobs.decode_keyed_dict(data, strict=True)
    assert [(e.key, e.type, e.subtype, e.value) for e in decoded.entries] == entries
    assert encode.keyed_dict([(e.key, e.type, e.subtype, e.value) for e in decoded.entries]) == data


@pytest.mark.parametrize(
    "entry",
    [
        ("", blobs.KD_STRING, 0, "x"),
        ("k", blobs.KD_INT32, 2, 1),
        ("k", blobs.KD_INT32, 0, 2**31),
        ("k", blobs.KD_INT32, 0, 1.5),
        ("k", blobs.KD_BOOL, 0, 1),
        ("k", blobs.KD_STRING, 0, b"x"),
        ("k", blobs.KD_DATE, 0, b"short"),
        ("k", blobs.KD_MAP, 0, {}),
        ("k", 0x99, 0, 1),
    ],
)
def test_keyed_dict_encoder_refuses_what_it_cannot_write_faithfully(entry: tuple[Any, ...]) -> None:
    with pytest.raises(EncodeError):
        encode.keyed_dict([entry])


def test_protobuf_encoder_is_the_inverse_of_the_decoder() -> None:
    fields = [(1, 0, 300), (2, 2, b"abc"), (3, 1, bytes(8)), (4, 5, bytes(4)), (1, 0, 0)]
    data = encode.protobuf(fields)
    assert [
        (f.number, f.wire, f.value) for f in blobs.decode_protobuf(data, nested_depth=0)
    ] == fields
    for bad in (
        [(1, 0, -1)],
        [(1, 0, True)],
        [(1, 2, "x")],
        [(1, 1, b"short")],
        [(0, 0, 1)],
        [(1, 3, b"")],
    ):
        with pytest.raises(EncodeError):
            encode.protobuf(bad)


# --------------------------------------------------------------------------- templates


def tables(*clips: ClipRows) -> dict[str, list[dict[str, Any]]]:
    return ms.tables_of(*clips)


def test_complete_clips_are_templates_and_other_media_rows_are_listed() -> None:
    t = tables(ms.wav_template(), ms.mp4_template(), ms.png_template())
    t["Sm2MpMedia"].append(
        {
            **t["Sm2MpMedia"][0],
            "Sm2MpMedia_id": ms.uid(5),
            "DbType": "Sm2MpTimelineClip",
            "Name": "Timeline 1",
        }
    )
    clips, skipped = clips_from_rows(t)
    assert sorted(c.media["Name"] for c in clips) == [
        "camera-b.MP4",
        "older-take.wav",
        "title-card.png",
    ]
    assert [s.reason for s in skipped] == ["a Sm2MpTimelineClip, not a media file clip"]


def test_a_project_without_a_media_table_is_refused() -> None:
    t = tables(ms.wav_template())
    del t["BtAudioInfo_Sm2MpMedia"]
    with pytest.raises(MediaPoolError) as caught:
        clips_from_rows(t)
    assert caught.value.code == "bad_template"


def skipped_reason(edit) -> str:
    t = tables(ms.mp4_template())
    edit(t)
    clips, skipped = clips_from_rows(t)
    assert clips == [] and len(skipped) == 1
    return skipped[0].reason


@pytest.mark.parametrize(
    ("edit", "says"),
    [
        (lambda t: t["BtAudioInfo_Sm2MpMedia"].clear(), "EmbeddedAudioVec join rows"),
        (lambda t: t["BtAudioInfo_Sm2MpMedia"][0].update(DbIndex=1), "EmbeddedAudioVec join rows"),
        (
            lambda t: t["BtAudioInfo_Sm2MpMedia"][0].update(DbPropertyName="Other"),
            "another property",
        ),
        (
            lambda t: t["BtAudioInfo_Sm2MpMedia"].append(
                dict(t["BtAudioInfo_Sm2MpMedia"][0], DbIndex=1)
            ),
            "EmbeddedAudioVec join rows",
        ),
        (lambda t: t["Sm2MpFolder_Sm2MpMedia"].clear(), "MediaVec join rows"),
        (lambda t: t["Sm2MpFolder_Sm2MpMedia"][0].update(DbOwner=ms.uid(77)), "bin join row"),
        (lambda t: t["BtVideoInfo"].clear(), "one BtVideoInfo row"),
        (lambda t: t["Sm2MpMedia"][0].update(Video=ms.uid(77)), "does not name its BtVideoInfo"),
        (lambda t: t["BtAudioInfo"].append(dict(t["BtAudioInfo"][0])), "share an id"),
        (lambda t: t["Sm2MpMedia"][0].update(PTZRPreset=ms.uid(77)), "PTZRPreset names no preset"),
        (
            lambda t: t["BtLockableBlob"].append(
                {
                    "BtLockableBlob_id": ms.uid(78),
                    "DbType": "Sm2MpItemLockableBlob",
                    "BtLockableBlobMap_id": ms.MAP,
                    "FieldsBlob": b"",
                    "BlobOwner": t["Sm2MpMedia"][0]["Sm2MpMedia_id"],
                    "DbSavedTime": 0,
                }
            ),
            "lockable blob this writer does not know",
        ),
    ],
)
def test_an_incomplete_or_inconsistent_clip_is_not_a_template(edit, says: str) -> None:
    assert says in skipped_reason(edit)


def test_a_sound_clip_with_a_video_info_is_not_a_template() -> None:
    t = tables(ms.wav_template(), ms.png_template())
    t["BtVideoInfo"][0]["Sm2MpMedia_id"] = t["Sm2MpMedia"][0]["Sm2MpMedia_id"]
    _, skipped = clips_from_rows(t)
    assert any("sound clip has a BtVideoInfo" in s.reason for s in skipped)


def test_templates_are_read_through_a_snapshot_and_the_live_file_is_untouched(
    tmp_path: Path,
) -> None:
    (tmp_path / "lib").mkdir()
    db = ms.write_project(
        tmp_path / "lib" / "Project.db", ms.wav_template(), ms.mp4_template(), rate_text="23.976"
    )
    before = (db.read_bytes(), db.stat().st_mtime_ns)
    snapshots = tmp_path / "snap"
    snapshots.mkdir()
    tset = read_templates(
        db, options=LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0, snapshot_dir=snapshots)
    )
    assert len(tset.clips) == 2 and tset.project_version == 17
    assert tset.project_rate == Fraction(24000, 1001)
    assert tset.lockable_map_id == ms.MAP
    assert (db.read_bytes(), db.stat().st_mtime_ns) == before
    assert sorted(p.name for p in db.parent.iterdir()) == ["Project.db"]
    assert list(snapshots.iterdir()) == []


def test_reading_rows_is_bounded_and_refuses_what_is_not_a_project(tmp_path: Path) -> None:
    db = ms.write_project(tmp_path / "Project.db", ms.wav_template(), ms.mp4_template())
    with pytest.raises(LibraryError) as caught:
        read_rows(db, ("NoSuchTable",), options=FAST)
    assert caught.value.code == "not_a_project"
    with pytest.raises(LibraryError) as caught:
        read_rows(
            db,
            ("Sm2MpMedia", "BtAudioInfo"),
            options=LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0, max_rows=3),
        )
    assert caught.value.code == "too_large"
    with pytest.raises(LibraryError) as caught:
        read_rows(
            db,
            ("BtAudioInfo",),
            options=LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0, max_load_bytes=200),
        )
    assert caught.value.code == "too_large"
    with pytest.raises(ValueError):
        read_rows(db, ('Sm2MpMedia"; DROP TABLE x; --',), options=FAST)
    old = ms.write_project(tmp_path / "old.db", ms.wav_template(), version=12)
    with pytest.raises(LibraryError) as caught:
        read_rows(old, ("Sm2MpMedia",), options=FAST)
    assert caught.value.code == "unsupported_version"


def test_text_that_is_not_utf8_is_refused_not_decoded(tmp_path: Path) -> None:
    db = ms.write_project(tmp_path / "Project.db", ms.wav_template())
    conn = sqlite3.connect(db)
    conn.execute("UPDATE Sm2MpMedia SET Name = CAST(X'fffe41' AS TEXT)")
    conn.commit()
    conn.close()
    os.utime(db, (1_000_000_000, 1_000_000_000))
    with pytest.raises(LibraryError) as caught:
        read_rows(db, ("Sm2MpMedia",), options=FAST)
    assert caught.value.code == "damaged"


def test_the_project_rate_is_read_from_setup_field_69_or_not_at_all() -> None:
    assert project_frame_rate([{"FieldsBlob": ms.setup_ba("29.97")}]) == Fraction(30000, 1001)
    assert project_frame_rate([{"FieldsBlob": ms.setup_ba("25")}]) == Fraction(25)
    assert project_frame_rate([{"FieldsBlob": ms.setup_ba("12.5")}]) is None
    assert project_frame_rate([{"FieldsBlob": ms.setup_ba("fast")}]) is None
    assert project_frame_rate([{"FieldsBlob": b"junk"}]) is None
    assert project_frame_rate([]) is None


# --------------------------------------------------------------------------- compare


def test_flatten_names_rows_by_role_and_ids_by_the_row_they_name() -> None:
    f = flatten(ms.wav_template())
    assert f["JA0.DbAssociate"] == ("ref", "<A0>")
    assert f["A0.TracksBA.0.UniqueId"] == ("ref", "<uuid>")
    assert f["L0.FieldsBlob.BlobData>#2>meta{215}"] == ("text", "older-take.wav")


def test_compare_sees_a_changed_value_and_a_wrong_reference() -> None:
    a = ms.wav_template()
    b = ms.replace_row(
        a,
        "audio0",
        TracksBA=ms.track(
            ms.uid(1), start=0.0, rate=48000, channels=2, samples=1, codec="Linear PCM", depth=2
        ),
    )
    _, diffs, _ = compare(a, b, "audio")
    assert [d.path for d in diffs] == ["A0.TracksBA.0.Duration"]
    c = copy.deepcopy(a)
    links = (dict(c.audio_links[0], DbAssociate=c.media_id),)
    c = ClipRows(c.media, c.video, c.audio, links, c.folder_link, c.lockables)
    _, diffs, _ = compare(a, c, "audio")
    assert [d.path for d in diffs] == ["JA0.DbAssociate"]


def _two_block_frame(payload: bytes) -> bytes:
    """The same payload in a zstd frame of two raw blocks (the encoder writes one): another
    frame, as Resolve's compressed frames are, that decodes to the same bytes."""
    half = len(payload) // 2
    out = encode.ZSTD_MAGIC + bytes([0x20, len(payload)])
    for chunk, last in ((payload[:half], 0), (payload[half:], 1)):
        out += ((len(chunk) << 3) | last).to_bytes(3, "little") + chunk
    return out


def test_compare_ignores_metadata_order_and_a_zstd_frame_but_not_the_codec() -> None:
    a = ms.wav_template()
    # metadata entries in another order: equal (Resolve's order differs between saves)
    b = ms.replace_row(
        a, "lockable", FieldsBlob=ms.metadata([(16, ""), (215, "older-take.wav"), (422, "")])
    )
    assert compare(a, b, "audio")[1] == []
    # a missing entry is a difference
    c = ms.replace_row(a, "lockable", FieldsBlob=ms.metadata([(16, ""), (215, "older-take.wav")]))
    missing = {d.path for d in compare(a, c, "audio")[1]}
    prefix = "L0.FieldsBlob.BlobData>#2>"
    assert prefix + "meta{422}" in missing and prefix + "~order" in missing
    assert all(p.startswith((prefix + "meta{422}", prefix + "~order")) for p in missing)
    # the same message framed with the other codec: the envelope differs
    d = ms.replace_row(
        a,
        "lockable",
        FieldsBlob=ms.metadata([(422, ""), (215, "older-take.wav"), (16, "")], zstd=True),
    )
    assert [x.path for x in compare(a, d, "audio")[1]] == ["L0.FieldsBlob.BlobData"]
    # a zstd envelope whose frame differs but decodes to the same message: equal
    clip = a.audio[0]["Clip"]
    env = blobs.parse_envelope(clip)
    assert env is not None and env.codec == "zstd"
    body = b"\x81" + _two_block_frame(blobs.unwrap_message(clip))
    other = struct.pack(">II", env.version, len(body)) + body
    assert other != clip
    assert compare(a, ms.replace_row(a, "audio0", Clip=other), "audio")[1] == []


def test_excluded_fields_are_reported_when_they_differ() -> None:
    a = ms.wav_template()
    b = ms.replace_row(a, "media", CurPlayheadPosition="0", PinsBA=None)
    b = ClipRows(
        b.media, b.video, b.audio, b.audio_links, dict(b.folder_link, DbIndex=9), b.lockables
    )
    compared, diffs, hidden = compare(a, b, "audio")
    assert diffs == [] and compared > 50
    assert sorted(d.path for _, d in hidden) == ["JF0.DbIndex", "M.CurPlayheadPosition", "M.PinsBA"]
    assert all(e.reason and e.category == "project-local" for e in EXCLUDED)


def test_a_still_frame_rate_is_excluded_only_for_stills() -> None:
    a = ms.png_template()
    b = ms.replace_row(
        a,
        "video",
        Time=ms.kd(
            ("UniqueId", ms.S, ms.uid(9)),
            ("StartFrame", ms.I32, 0),
            ("NumFrames", ms.I32, 1),
            ("FrameRate", ms.BYTES, ms.rate16(25.0)),
            ("DbType", ms.S, "BtVideoTime"),
        ),
    )
    assert compare(a, b, "still")[1] == []
    assert [d.path for d in compare(a, b, "video")[1]] == ["V.Time.FrameRate"]


# ------------------------------------------------------------------ the gate, on synthetic projects


def facts_line(data: dict[str, Any]) -> str:
    return json.dumps(data)


def project_with(tmp_path: Path, name: str, *clips: ClipRows) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    return ms.write_project(folder / "Project.db", *clips)


def answer(template: ClipRows, facts: dict[str, Any], base: int) -> ClipRows:
    """Resolve's rows for ``facts``, as the writer itself authors them (synthetic answer key)."""
    from aar.drp.mediapool import Placement, Template, author

    result = author(
        facts_from_json(facts),
        Template(template),
        Placement(ms.FOLDER, 0, Fraction(30000, 1001), 1, ms.MAP),
        new_id=ms.new_ids(base),
    )
    rows = result.rows
    return ClipRows(
        rows["Sm2MpMedia"][0],
        None,
        rows["BtAudioInfo"],
        rows["BtAudioInfo_Sm2MpMedia"],
        rows["Sm2MpFolder_Sm2MpMedia"][0],
        rows["BtLockableBlob"],
    )


def test_gate_scores_equal_stale_and_single_members(tmp_path: Path) -> None:
    first = ms.wav_facts()
    second = ms.wav_facts(path="X:\\Show\\Sound\\take-two.wav", stream_duration_ts=96000)
    clip_a = answer(ms.wav_template(), first, 0x5000)
    clip_b = answer(ms.wav_template(), second, 0x6000)
    db = project_with(tmp_path, "p", clip_a, clip_b)
    facts = {f["path"].lower(): facts_from_json(f) for f in (first, second)}
    board = gate.run([db], facts, options=FAST)
    [score] = board.kinds
    assert (score.status, score.checked, score.members) == ("equal", 2, 2)
    assert score.fields == score.equal > 100
    # a file whose modification time differs from the one Resolve recorded is stale
    facts[second["path"].lower()] = facts_from_json(
        dict(second, mtime_local="Mon Nov 9 08:00:00 2026")
    )
    board = gate.run([db], facts, options=FAST)
    assert (board.kinds[0].stale, board.kinds[0].checked) == (1, 1)
    # a single member is self-checked only
    db2 = project_with(tmp_path, "q", clip_a)
    board = gate.run([db2], {first["path"].lower(): facts_from_json(first)}, options=FAST)
    assert board.kinds[0].status == "single" and board.kinds[0].self_check.startswith("equal")


def test_gate_reports_a_field_the_facts_do_not_explain(tmp_path: Path) -> None:
    first = ms.wav_facts()
    second = ms.wav_facts(path="X:\\Show\\Sound\\take-two.wav")
    clip_a = answer(ms.wav_template(), first, 0x5000)
    clip_b = answer(ms.wav_template(), second, 0x6000)
    # Resolve "stored" another duration for the second file than its facts give
    tracks = ms.track(
        ms.uid(0x6FFF),
        start=0.0,
        rate=48000,
        channels=2,
        samples=479999,
        codec="Linear PCM",
        depth=2,
    )
    clip_b = ms.replace_row(clip_b, "audio0", TracksBA=tracks)
    db = project_with(tmp_path, "p", clip_a, clip_b)
    facts = {f["path"].lower(): facts_from_json(f) for f in (first, second)}
    board = gate.run([db], facts, options=FAST)
    assert board.kinds[0].status == "differs"
    assert board.kinds[0].differing == {"A0.TracksBA.0.Duration": 1}
    assert "479999" in board.kinds[0].examples["A0.TracksBA.0.Duration"]
    assert "differs" in gate.format_scoreboard(board)


def test_gate_facts_file_is_checked_line_by_line(tmp_path: Path) -> None:
    good = tmp_path / "facts.jsonl"
    good.write_text(
        facts_line(ms.wav_facts())
        + "\n"
        + json.dumps({"path": "X:\\gone.png", "exists": False})
        + "\n\n"
    )
    assert list(gate.load_facts(good)) == [ms.wav_facts()["path"].lower()]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(facts_line(ms.wav_facts(size="1")) + "\n")
    with pytest.raises(MediaPoolError) as caught:
        gate.load_facts(bad)
    assert caught.value.code == "bad_facts" and ":1:" in caught.value.message
    broken = tmp_path / "broken.jsonl"
    broken.write_text(facts_line(ms.wav_facts()) + "\n{not json\n")
    with pytest.raises(MediaPoolError) as caught:
        gate.load_facts(broken)
    assert caught.value.code == "bad_facts" and ":2 " in caught.value.message


def test_struct_layout_of_the_synthetic_extents_matches_the_reader() -> None:
    start, length = blobs.decode_extents(struct.pack("<dd", 1.5, 2.5))
    assert (start, length) == (1.5, 2.5)
