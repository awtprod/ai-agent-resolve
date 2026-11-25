"""S25c1 round 1: a total comparison, typed refusals, ASCII digits, envelopes, the unverified
opt-in, template facts checked against their rows, path rules, and the surviving mutants.

Each section names the validation item it pins. Everything is synthetic (``mediapool_support``).
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

import mediapool_support as ms
import pytest
from pydantic import ValidationError

from aar.drp import blobs
from aar.drp.library import LibraryOptions
from aar.drp.mediapool import (
    KINDS,
    MediaFacts,
    MediaPoolError,
    Placement,
    Template,
    author,
    facts_from_json,
    gate,
)
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool.author import _derive, _Writer
from aar.drp.mediapool.facts import MAX_FACTS_BYTES
from aar.drp.mediapool.fields import compare, flatten
from aar.drp.mediapool.kinds import (
    ENVELOPE_PROPERTIES,
    KindStatus,
    envelope_of,
    kind_of_facts,
)
from aar.drp.mediapool.templates import (
    ClipRows,
    clips_from_rows,
    project_frame_rate,
    read_templates,
)
from aar.drp.modify import encode
from aar.media.probe import timecode_frames

PLACE = Placement(ms.FOLDER, 4, Fraction(30000, 1001), 1_795_000_000, ms.MAP, 99)
FAST = LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0)


def run(
    facts: dict[str, Any],
    template: ClipRows,
    *,
    template_facts: dict[str, Any] | None = None,
    placement: Placement = PLACE,
    **kw: Any,
):
    return author(
        facts_from_json(facts),
        Template(template, facts_from_json(template_facts) if template_facts else None),
        placement,
        new_id=ms.new_ids(),
        **kw,
    )


def refused(code: str, *args: Any, **kw: Any) -> MediaPoolError:
    with pytest.raises(MediaPoolError) as caught:
        run(*args, **kw)
    assert caught.value.code == code, caught.value
    assert caught.value.hint
    return caught.value


def facts_refused(data: object) -> MediaPoolError:
    with pytest.raises(MediaPoolError) as caught:
        facts_from_json(data)
    assert caught.value.code == "bad_facts", caught.value
    return caught.value


def edit_track(clip: ClipRows, change) -> ClipRows:
    """``clip`` with its first audio track's keyed dict edited by ``change(entries)``."""
    tracks = bt.kd_decode(clip.audio[0]["TracksBA"], "TracksBA")
    inner = bt.kd_decode(tracks[0][3], "track")
    change(inner)
    tracks[0][3] = bt.kd_encode(inner)
    return ms.replace_row(clip, "audio0", TracksBA=bt.kd_encode(tracks))


def differing(a: ClipRows, b: ClipRows, media: str = "audio") -> list[str]:
    return [d.path for d in compare(a, b, media)[1]]


def with_column(clip: ClipRows, value: object) -> ClipRows:
    return ms.replace_row(clip, "media", Probe=value)


def raw_kd(*entries: tuple[str, int, int, bytes]) -> bytes:
    """A keyed dict written byte by byte (``(key, type, null flag, raw value bytes)``)."""
    out = struct.pack(">II", 1, len(entries))
    for key, vtype, null, raw in entries:
        name = key.encode("utf-16-be")
        out += struct.pack(">I", len(name)) + name + struct.pack(">IB", vtype, null) + raw
    return out


# =========================================================================== 1. a total comparison


def test_a_keyed_dict_entrys_type_tag_is_compared() -> None:
    a = ms.wav_template()

    def retype(entries: bt.KdEntries) -> None:
        entry = bt.kd_get(entries, "Duration")
        assert entry is not None
        entry[1] = blobs.KD_UINT64  # same value, another type

    assert differing(a, edit_track(a, retype)) == ["A0.TracksBA.0.Duration~kd"]


def test_a_keyed_dict_entrys_null_flag_is_compared() -> None:
    a = ms.wav_template()

    def null(entries: bt.KdEntries) -> None:
        entry = bt.kd_get(entries, "Duration")
        assert entry is not None
        entry[2] = 1

    assert differing(a, edit_track(a, null)) == ["A0.TracksBA.0.Duration~kd"]


def test_keyed_dict_key_order_is_compared() -> None:
    a = ms.wav_template()
    assert differing(a, edit_track(a, lambda entries: entries.reverse())) == ["A0.TracksBA.0~order"]


def test_protobuf_field_order_is_compared() -> None:
    a = ms.wav_template()
    _, payload = bt.message_of(a.audio[0]["Clip"], "Clip")
    fields = bt.pb_decode(payload, "Clip")
    reordered = bt.rewrap(a.audio[0]["Clip"], bt.pb_encode(fields[::-1]), "Clip")
    assert differing(a, ms.replace_row(a, "audio0", Clip=reordered)) == ["A0.Clip>~order"]


def test_the_envelope_codec_is_compared() -> None:
    a = ms.wav_template()
    _, payload = bt.message_of(a.audio[0]["Clip"], "Clip")
    plain = ms.envelope(payload, zstd=False)
    assert differing(a, ms.replace_row(a, "audio0", Clip=plain)) == ["A0.Clip"]


def test_two_blobs_that_do_not_decode_are_compared_byte_for_byte() -> None:
    def broken(tail: bytes) -> bytes:
        body = b"\x81" + encode.ZSTD_MAGIC + tail
        return struct.pack(">II", 2, len(body)) + body

    a = with_column(ms.wav_template(), broken(b"\x00\x01\x02"))
    b = with_column(ms.wav_template(), broken(b"\x00\x09\x09"))
    assert flatten(a)["M.Probe"][0] in ("bytes", "fixed16")
    assert differing(a, b) == ["M.Probe"]


def test_a_duplicate_key_is_kept_as_its_own_entry() -> None:
    first = encode.keyed_dict([("A", 2, 0, 1), ("A", 2, 0, 2)])
    second = encode.keyed_dict([("A", 2, 0, 3), ("A", 2, 0, 2)])
    a, b = with_column(ms.wav_template(), first), with_column(ms.wav_template(), second)
    assert flatten(a)["M.Probe.A[1]"] == ("int", 2)
    assert differing(a, b) == ["M.Probe.A"]


def test_a_bools_byte_and_a_nans_bits_are_compared() -> None:
    one = raw_kd(("B", blobs.KD_BOOL, 0, b"\x01"))
    two = raw_kd(("B", blobs.KD_BOOL, 0, b"\x02"))  # also True to the decoder
    assert differing(with_column(ms.wav_template(), one), with_column(ms.wav_template(), two)) == [
        "M.Probe.B"
    ]
    nan1 = raw_kd(("D", blobs.KD_DOUBLE, 0, bytes.fromhex("7ff8000000000001")))
    nan2 = raw_kd(("D", blobs.KD_DOUBLE, 0, bytes.fromhex("7ff8000000000002")))
    assert differing(
        with_column(ms.wav_template(), nan1), with_column(ms.wav_template(), nan2)
    ) == ["M.Probe.D"]


def test_a_message_that_does_not_re_encode_to_its_bytes_is_compared_as_bytes() -> None:
    canonical = ms.envelope(b"\x08\x01", zstd=False)
    overlong = ms.envelope(b"\x08\x81\x00", zstd=False)  # field 1 = 1, written in two bytes
    a, b = with_column(ms.wav_template(), canonical), with_column(ms.wav_template(), overlong)
    assert flatten(b)["M.Probe"][0] == "bytes"
    assert "M.Probe" in differing(a, b)


def test_a_key_that_is_not_a_plain_word_cannot_pose_as_a_path() -> None:
    dotted = encode.keyed_dict([("a.b", 2, 0, 1)])
    assert 'M.Probe."a.b"' in flatten(with_column(ms.wav_template(), dotted))


def test_an_excluded_field_is_left_out_of_its_messages_order() -> None:
    with_extra = ms.mp4_template()
    without = ms.mp4_template(clip_extra=())
    _, found, hidden = compare(with_extra, without, "video")
    assert found == []
    assert {d.path for _, d in hidden} >= {"V.Clip>#14", "V.Clip>#26"}


def test_resolves_codec_follows_the_message_length() -> None:
    assert bt.frame_codec("zstd", bytes(64), "x") == "protobuf"
    assert bt.frame_codec("protobuf", bytes(66), "x") == "zstd"
    assert bt.frame_codec("none", bytes(200), "x") == "none"
    with pytest.raises(MediaPoolError) as caught:
        bt.frame_codec("zstd", bytes(65), "x")
    assert caught.value.code == "cannot_derive"
    zstd = ms.envelope(bytes(10), zstd=True, version=7)
    short = blobs.parse_envelope(bt.rewrap(zstd, b"\x08\x01", "x"))
    assert short is not None and (short.codec, short.version) == ("protobuf", 7)
    plain = ms.envelope(b"\x08\x01", zstd=False)
    long = blobs.parse_envelope(bt.rewrap(plain, b"\x0a\x46" + bytes(70), "x"))
    assert long is not None and long.codec == "zstd"


# =========================================================================== 2. typed refusals


@pytest.mark.parametrize("tag", ["0000-01-01T00:00:00Z", "2026-13-45T25:61:61Z"])
def test_a_creation_time_that_is_no_date_is_refused(tag: str) -> None:
    facts = ms.mp4_facts()
    facts["ffprobe"]["format"]["tags"]["creation_time"] = tag
    refused("cannot_derive", facts, ms.mp4_template())


def test_a_frame_count_beyond_any_real_file_is_refused() -> None:
    refused("cannot_derive", ms.mp4_facts(video_time_base="99999999999/1"), ms.mp4_template())


def test_a_derived_number_too_large_for_the_templates_field_is_refused() -> None:
    entries = [["NumFrames", blobs.KD_INT32, 0, 1]]
    with pytest.raises(MediaPoolError) as caught:
        bt.kd_set(entries, "NumFrames", 2**31, "Time")
    assert caught.value.code == "cannot_derive"
    with pytest.raises(MediaPoolError) as caught:
        bt.kd_set(entries, "NumFrames", "5", "Time")
    assert caught.value.code == "bad_template"


@pytest.mark.parametrize("column", ["Time", "Geometry", "Clip", "Proxy", "VideoMetadata"])
def test_a_null_video_blob_in_the_template_is_refused(column: str) -> None:
    template = ms.replace_row(ms.mp4_template(), "video", **{column: None})
    refused("bad_template", ms.mp4_facts(), template)


def test_a_null_track_list_in_the_template_is_refused() -> None:
    refused(
        "bad_template", ms.wav_facts(), ms.replace_row(ms.wav_template(), "audio0", TracksBA=None)
    )


def test_a_sound_clip_without_audio_rows_is_refused() -> None:
    template = dataclasses.replace(ms.wav_template(), audio=(), audio_links=())
    refused("bad_template", ms.wav_facts(), template)


@pytest.mark.parametrize(
    "change",
    [
        {"added_at": 2**70},
        {"added_at": 1.5},
        {"added_at": -1},
        {"project_rate": float("nan")},
        {"project_rate": Fraction(10**30)},
        {"project_rate": 29.97},
        {"folder_index": True},
        {"saved_age_ms": 2**63},
        {"lockable_map_id": "x" * 36},
        {"folder_id": ms.FOLDER.upper()},
    ],
)
def test_a_placement_out_of_range_is_refused(change: dict[str, Any]) -> None:
    placement = dataclasses.replace(PLACE, **change)
    refused("bad_placement", ms.wav_facts(), ms.wav_template(), placement=placement)


def test_a_template_table_without_a_column_the_writer_reads_is_refused(tmp_path: Path) -> None:
    db = ms.write_project(tmp_path / "Project.db", ms.mp4_template(), ms.wav_template())
    conn = sqlite3.connect(db)
    conn.execute('ALTER TABLE "BtVideoInfo" DROP COLUMN "Geometry"')
    conn.commit()
    conn.close()
    with pytest.raises(MediaPoolError) as caught:
        read_templates(db, options=FAST)
    assert caught.value.code == "bad_template" and "Geometry" in caught.value.message


def test_rows_without_a_column_or_with_a_value_no_column_holds_are_refused() -> None:
    tables = ms.tables_of(ms.wav_template())
    del tables["BtAudioInfo"][0]["Clip"]
    with pytest.raises(MediaPoolError) as caught:
        clips_from_rows(tables)
    assert caught.value.code == "bad_template"
    tables = ms.tables_of(ms.wav_template())
    tables["BtAudioInfo"][0]["Clip"] = [1, 2]
    with pytest.raises(MediaPoolError):
        clips_from_rows(tables)


def test_a_template_whose_rows_are_not_one_clip_is_refused_before_authoring() -> None:
    template = ms.wav_template()
    shuffled = dataclasses.replace(template, folder_link=dict(template.folder_link, DbOwner="x"))
    refused("bad_template", ms.wav_facts(), shuffled)
    refused(
        "bad_template", ms.wav_facts(), dataclasses.replace(template, audio=list(template.audio))
    )


def test_a_template_with_a_list_inside_an_edited_keyed_dict_is_refused() -> None:
    template = ms.mp4_template()
    time = bt.kd_decode(template.video["Time"], "Time")
    raw = template.video["Time"] + b""
    # append a list entry (type 9, no items) by hand: the encoder does not write lists
    count = struct.unpack(">I", raw[4:8])[0] + 1
    key = "DbType".encode("utf-16-be")
    listed = (
        struct.pack(">II", 1, count)
        + raw[8:]
        + struct.pack(">I", len(key))
        + key
        + struct.pack(">IBI", blobs.KD_LIST, 0, 0)
    )
    assert len(bt.kd_decode(listed, "Time")) == len(time) + 1
    refused("bad_template", ms.mp4_facts(), ms.replace_row(template, "video", Time=listed))


def test_a_clip_field_of_another_wire_type_is_refused() -> None:
    clip = ms.clip_blob("X:\\Old", "older-take.wav", "x", None, 1, video=False)
    _, payload = bt.message_of(clip, "Clip")
    fields = [[1, 0, 5] if f[0] == 1 else f for f in bt.pb_decode(payload, "Clip")]
    odd = ms.envelope(bt.pb_encode(fields))
    refused("bad_template", ms.wav_facts(), ms.replace_row(ms.wav_template(), "audio0", Clip=odd))


def test_facts_with_a_lone_surrogate_are_refused() -> None:
    facts_refused(ms.wav_facts(stream_codec_tag_string="\ud800"))


# =========================================================================== 3. ASCII digits


def test_a_timecode_in_other_digits_is_refused_not_authored() -> None:
    arabic = "\u0660\u0662:\u0661\u0660:\u0662\u0660:\u0661\u0662"
    facts_refused(ms.mp4_facts(data_tags={"timecode": arabic}))
    with pytest.raises(ValueError):
        timecode_frames(arabic, Fraction(24))


@pytest.mark.parametrize("value", ["\u0669\u0660", "90.0", " 90"])
def test_a_rotate_tag_that_is_not_ascii_whole_degrees_is_refused(value: str) -> None:
    facts_refused(ms.mp4_facts(video_tags={"rotate": value}))


@pytest.mark.parametrize("value", ["90", True, float("nan"), 1e300])
def test_a_side_data_rotation_that_is_not_a_finite_number_of_degrees_is_refused(
    value: object,
) -> None:
    facts_refused(ms.mp4_facts(video_side_data_list=[{"rotation": value}]))


def test_a_codec_tag_in_other_letters_is_no_known_codec() -> None:
    kind = kind_of_facts(facts_from_json(ms.mp4_facts(video_codec_tag_string="\uff41vc1")))
    assert kind.video_codec == "?h264"


@pytest.mark.parametrize("text", ["2_9.97", "\u0662\u0669.97", "29.97 ", "1e1"])
def test_a_project_rate_that_is_not_plain_ascii_is_not_read(text: str) -> None:
    assert project_frame_rate([{"FieldsBlob": ms.setup_ba(text)}]) is None
    assert project_frame_rate([{"FieldsBlob": ms.setup_ba("29.97")}]) == Fraction(30000, 1001)


# =========================================================================== 4. envelopes


def test_every_verified_kind_records_its_whole_envelope() -> None:
    for kind, status in KINDS.items():
        if status.status == "verified":
            assert status.envelope is not None, kind.label()
            assert set(status.envelope) == set(ENVELOPE_PROPERTIES), kind.label()
            assert all(len(values) == 1 for values in status.envelope.values()), kind.label()


def test_a_file_outside_its_kinds_envelope_is_unverified_with_the_property_named() -> None:
    facts = ms.mp4_facts(video_width=1920, video_height=1080, video_r_frame_rate="25/1")
    facts["ffprobe"]["streams"][0].update(avg_frame_rate="25/1", time_base="1/25", duration_ts=3000)
    result = run(
        facts, ms.mp4_template(), template_facts=ms.mp4_template_facts(), allow_unverified=True
    )
    assert result.status == "unverified"
    joined = " ".join(result.unverified)
    assert "resolution 1920x1080" in joined and "frame rate 25/1" in joined
    error = refused("unverified", facts, ms.mp4_template(), template_facts=ms.mp4_template_facts())
    assert "resolution" in error.message


def test_the_envelope_of_a_file() -> None:
    values = envelope_of(facts_from_json(ms.mp4_facts()))
    assert values == {
        "resolution": "3840x2160",
        "coded_size": "3840x2160",
        "frame_rate": "24000/1001",
        "drop_frame": "non-drop",
        "bit_depth": "8",
        "chroma": "4:2:0",
        "video_profile": "High",
        "video_level": "51",
        "field_order": "progressive",
        "sample_aspect_ratio": "1:1",
        "display_aspect_ratio": "16:9",
        "has_b_frames": "1",
        "color_range": "tv",
        "chroma_location": "left",
        "video_raw_bits": "8",
        "video_time_base": "1/24000",
        "video_stream_rates": "24000/1001 24000/1001",
        "audio": "48000 Hz 2 ch",
        "audio_profile": "-",
        "audio_codec": "pcm_s16be",
        "audio_sample_format": "s16",
        "audio_bits_per_sample": "16",
        "audio_raw_bits": "-",
        "audio_channel_layout": "-",
        "audio_initial_padding": "0",
        "audio_time_base": "1/48000",
    }
    drop = ms.mp4_facts(data_tags={"timecode": "01:00:00;00"})
    drop["ffprobe"]["streams"][0].update(
        r_frame_rate="30000/1001", avg_frame_rate="30000/1001", time_base="1/30000"
    )
    assert envelope_of(facts_from_json(drop))["drop_frame"] == "drop-frame"


def test_prores_4444_is_not_verified_and_says_why() -> None:
    [status] = [s for k, s in KINDS.items() if k.video_codec == "ap4h"]
    assert status.status == "unverified" and "#1.#3" in status.note


def test_every_result_names_clip_field_26_as_an_open_question() -> None:
    result = run(ms.wav_facts(), ms.wav_template(), template_facts=ms.wav_template_facts())
    assert any("field 26" in note for note in result.notes)


# =========================================================================== 5. opt-in unverified


def test_an_unverified_kind_is_refused_unless_allowed() -> None:
    kind = kind_of_facts(facts_from_json(ms.wav_facts()))
    kinds = {kind: KindStatus("unverified", 1, "one file")}
    tfacts = ms.wav_template_facts()
    error = refused(
        "unverified", ms.wav_facts(), ms.wav_template(), kinds=kinds, template_facts=tfacts
    )
    assert "allow_unverified=True" in error.hint and "one file" in error.message
    result = run(
        ms.wav_facts(), ms.wav_template(), kinds=kinds, allow_unverified=True, template_facts=tfacts
    )
    assert result.status == "unverified" and result.unverified == ("kind not verified: one file",)


def test_a_picture_template_without_facts_is_refused_unless_allowed() -> None:
    error = refused("unverified", ms.mp4_facts(), ms.mp4_template())
    assert "Radiometry" in error.message
    assert run(ms.mp4_facts(), ms.mp4_template(), allow_unverified=True).status == "unverified"


def test_a_verified_result_needs_no_opt_in() -> None:
    result = run(ms.mp4_facts(), ms.mp4_template(), template_facts=ms.mp4_template_facts())
    assert (result.status, result.unverified) == ("verified", ())


# =========================================================================== 6. template facts


def test_facts_that_are_not_the_templates_own_are_not_trusted() -> None:
    # the target's own facts handed over as the template's: same kind, other file
    error = refused("unverified", ms.mp4_facts(), ms.mp4_template(), template_facts=ms.mp4_facts())
    assert "template's facts are not trusted" in error.message and "differs" in error.message
    result = run(
        ms.mp4_facts(),
        ms.mp4_template(),
        template_facts=ms.mp4_facts(),
        allow_unverified=True,
    )
    assert any("V.Clip>#2" in r or "#1" in r for r in result.unverified), result.unverified


def test_template_facts_measured_after_the_file_changed_are_not_trusted() -> None:
    stale = ms.mp4_template_facts(mtime_local="Wed Sep 2 08:00:00 2026")
    error = refused("unverified", ms.mp4_facts(), ms.mp4_template(), template_facts=stale)
    assert "Clip>#3" in error.message


# =========================================================================== 7. smaller items


@pytest.mark.parametrize(
    "path",
    [
        "X:\\Show\\\\take.wav",  # doubled separator
        "X:\\take.wav",  # directly under the drive root
        "\\\\server\\share\\take.wav",  # directly under a share
        "X:\\Show\\ta\u202eke.wav",  # bidirectional override
        "X:\\Show\\ta\u200bke.wav",  # zero-width space
        "X:\\Show\\ta\x85ke.wav",  # C1 control
        "X:\\Show\\CON.wav",  # reserved device name
        "X:\\Show\\take:two.wav",  # colon in the name
        "X:\\Show\\take.wav.",  # trailing dot
        "X:\\Show/take.wav",  # a forward slash
        "\\\\?\\X:\\Show\\take.wav",  # device namespace
    ],
)
def test_a_path_the_edit_pc_rules_refuse_is_bad_facts(path: str) -> None:
    facts_refused(ms.wav_facts(path=path))


def test_a_second_picture_is_refused() -> None:
    facts = ms.mp4_facts()
    facts["ffprobe"]["streams"].append(dict(facts["ffprobe"]["streams"][0], index=7))
    refused("unsupported_kind", facts, ms.mp4_template(), template_facts=ms.mp4_template_facts())
    # cover art is not a second picture
    facts["ffprobe"]["streams"][-1]["disposition"] = {"attached_pic": 1}
    run(facts, ms.mp4_template(), template_facts=ms.mp4_template_facts())


def test_an_absurd_sample_count_is_refused() -> None:
    tfacts = ms.wav_template_facts()
    for samples in (2**61, 48000 * 86400 + 1):
        facts = ms.wav_facts(stream_duration_ts=samples)
        refused("cannot_derive", facts, ms.wav_template(), template_facts=tfacts)
    run(ms.wav_facts(stream_duration_ts=48000 * 86400), ms.wav_template(), template_facts=tfacts)


def test_an_unreadable_recorded_modification_time_is_reported_and_compared(
    tmp_path: Path,
) -> None:
    first = ms.wav_facts()
    second = ms.wav_facts(path="X:\\Show\\Sound\\take-two.wav")
    clips = []
    for facts, base in ((first, 0x5000), (second, 0x6000)):
        result = author(
            facts_from_json(facts),
            Template(ms.wav_template(), facts_from_json(ms.wav_template_facts())),
            PLACE,
            new_id=ms.new_ids(base),
        )
        clips.append(ClipRows.from_tables(result.rows))
    clip = ms.clip_blob(
        ms.PC_FOLDER, "take-two.wav", "not a time", "Linear PCM", 1, video=False, flags=222
    )
    clips[1] = ms.replace_row(clips[1], "audio0", Clip=clip)
    db = ms.write_project(tmp_path / "Project.db", *clips)
    facts_map = {f["path"].lower(): facts_from_json(f) for f in (first, second)}
    [score] = gate.run([db], facts_map, options=FAST).kinds
    assert score.unreadable == 1 and score.stale == 0 and score.checked == 2
    assert score.status == "differs"  # the recorded time is compared, and differs
    assert "unreadable" in gate.format_scoreboard(gate.Scoreboard([score], [], []))


def test_a_layer_that_cannot_be_opened_cannot_hide_a_template_id() -> None:
    body = b"\x81" + encode.ZSTD_MAGIC + b"\x00\x01"
    broken = struct.pack(">II", 2, len(body)) + body
    with pytest.raises(MediaPoolError) as caught:
        bt.contains_text(broken, {"x": "x"})
    assert caught.value.code == "bad_template"


# =========================================================================== 8. surviving mutants


def test_a_video_shorter_than_one_frame_is_refused() -> None:
    refused("cannot_derive", ms.mp4_facts(video_duration_ts=0), ms.mp4_template())


def test_the_sample_rate_bounds() -> None:
    stream = facts_from_json(ms.wav_facts(stream_sample_rate="384001")).streams[0]
    with pytest.raises(MediaPoolError):
        fm.sample_rate(stream)
    assert fm.sample_rate(facts_from_json(ms.wav_facts(stream_sample_rate="384000")).streams[0])
    with pytest.raises(MediaPoolError):
        fm.sample_rate(facts_from_json(ms.wav_facts(stream_sample_rate="999")).streams[0])


@pytest.mark.parametrize("rate", ["0/1001", "-24000/1001"])
def test_a_rate_whose_numerator_is_not_positive_is_refused(rate: str) -> None:
    refused(
        "cannot_derive",
        ms.mp4_facts(video_r_frame_rate=rate, video_avg_frame_rate=rate, data_avg_frame_rate=rate),
        ms.mp4_template(),
    )


def test_a_template_whose_start_frame_is_not_zero_is_refused() -> None:
    time = ms.kd(
        ("UniqueId", ms.S, ms.uid(0x2F5)),
        ("Timecode", ms.S, "01:00:00:00"),
        ("StartFrame", ms.I32, 5),
        ("NumFrames", ms.I32, 240),
        ("FrameRate", ms.BYTES, ms.rate16(24000 / 1001)),
        ("DbType", ms.S, "BtVideoTime"),
    )
    error = refused(
        "bad_template", ms.mp4_facts(), ms.replace_row(ms.mp4_template(), "video", Time=time)
    )
    assert "StartFrame" in error.message


def test_the_writer_refuses_a_template_with_another_number_of_audio_streams() -> None:
    # unreachable through author() (the kind key holds the audio streams): the writer's own check
    facts = facts_from_json(ms.mp4_facts())
    kind = kind_of_facts(facts)
    derived = _derive(facts, kind, PLACE)
    derived.audio = []
    with pytest.raises(MediaPoolError) as caught:
        _Writer(facts, ms.mp4_template(), PLACE, derived, frozenset(), ms.new_ids())
    assert caught.value.code == "kind_mismatch"


@pytest.mark.parametrize("rate", [Fraction(0), Fraction(31), Fraction(-24)])
def test_a_project_rate_that_is_not_standard_is_refused(rate: Fraction) -> None:
    placement = dataclasses.replace(PLACE, project_rate=rate)
    refused("bad_placement", ms.wav_facts(), ms.wav_template(), placement=placement)


@pytest.mark.parametrize("degrees", [90, 270, -90])
def test_a_quarter_turn_either_way_is_transposed(degrees: int) -> None:
    facts = ms.mp4_facts(video_side_data_list=[{"rotation": degrees}])
    assert kind_of_facts(facts_from_json(facts)).transposed


def test_a_forward_slash_is_refused() -> None:
    facts_refused(ms.wav_facts(path="X:\\Show/Sound\\take.wav"))


def test_the_facts_model_forbids_unknown_fields() -> None:
    good = facts_from_json(ms.wav_facts()).model_dump()
    with pytest.raises(ValidationError):
        MediaFacts.model_validate({**good, "zzz": 1})


def test_facts_longer_than_the_limit_are_refused() -> None:
    limit = 1024 * 1024  # the documented limit, written out so the constant itself is pinned
    assert limit == MAX_FACTS_BYTES
    text = json.dumps(ms.wav_facts())
    assert facts_from_json(text.ljust(limit)).name == ms.WAV_NAME
    error = facts_refused(text.ljust(limit + 1))
    assert "longer than" in error.message
    error = facts_refused(text.ljust(limit + 1).encode())
    assert "longer than" in error.message
    # the limit counts bytes, not characters: these are fewer characters than the limit
    padded = text.ljust(limit // 2 + 1000, "\u00e9")
    assert len(padded) < limit < len(padded.encode())
    error = facts_refused(padded)
    assert "longer than" in error.message


def skipped(tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    clips, skips = clips_from_rows(tables)
    assert clips == []
    return [s.reason for s in skips]


def test_a_clip_in_two_bins_is_not_a_template() -> None:
    tables = ms.tables_of(ms.wav_template())
    tables["Sm2MpFolder_Sm2MpMedia"].append(dict(tables["Sm2MpFolder_Sm2MpMedia"][0], DbIndex=9))
    assert skipped(tables) == ["the clip has 2 MediaVec join rows, not one"]


def test_a_clip_with_two_metadata_blobs_is_not_a_template() -> None:
    tables = ms.tables_of(ms.wav_template())
    tables["BtLockableBlob"].append(dict(tables["BtLockableBlob"][0], BtLockableBlob_id=ms.uid(77)))
    assert skipped(tables) == ["the clip has more than one metadata blob"]


def test_audio_join_rows_naming_other_infos_are_not_a_template() -> None:
    tables = ms.tables_of(ms.wav_template())
    tables["BtAudioInfo_Sm2MpMedia"][0]["DbAssociate"] = ms.uid(0xBAD)
    assert skipped(tables) == ["the EmbeddedAudioVec join rows do not name the clip's audio infos"]


def test_a_clip_with_more_than_16_audio_infos_is_not_a_template() -> None:
    tables = ms.tables_of(ms.wav_template())
    info, link = tables["BtAudioInfo"][0], tables["BtAudioInfo_Sm2MpMedia"][0]
    for n in range(1, 17):
        tables["BtAudioInfo"].append(dict(info, BtAudioInfo_id=ms.uid(0x800 + n)))
        tables["BtAudioInfo_Sm2MpMedia"].append(
            dict(link, DbAssociate=ms.uid(0x800 + n), DbIndex=n)
        )
    assert skipped(tables) == ["the clip has 17 audio infos"]
    tables["BtAudioInfo"].pop()
    tables["BtAudioInfo_Sm2MpMedia"].pop()
    clips, _ = clips_from_rows(tables)
    assert len(clips) == 1 and len(clips[0].audio) == 16


def test_a_repeated_key_in_a_template_blob_is_refused() -> None:
    time = ms.kd(
        ("UniqueId", ms.S, ms.uid(0x2F5)),
        ("Timecode", ms.S, "01:00:00:00"),
        ("StartFrame", ms.I32, 0),
        ("NumFrames", ms.I32, 240),
        ("NumFrames", ms.I32, 241),
        ("FrameRate", ms.BYTES, ms.rate16(24000 / 1001)),
        ("DbType", ms.S, "BtVideoTime"),
    )
    error = refused(
        "bad_template", ms.mp4_facts(), ms.replace_row(ms.mp4_template(), "video", Time=time)
    )
    assert "twice" in error.message


def test_a_keyed_dict_with_another_header_is_refused() -> None:
    template = ms.wav_template()
    tracks = template.audio[0]["TracksBA"]
    other = b"\x00\x00\x00\x02" + tracks[4:]
    error = refused(
        "bad_template", ms.wav_facts(), ms.replace_row(template, "audio0", TracksBA=other)
    )
    assert "header 2" in error.message


def test_more_layers_than_the_bound_are_refused() -> None:
    fields = [(n % 500 + 1, 2, f"layer {n}".encode()) for n in range(4200)]
    many = ms.envelope(ms.pb(*fields), zstd=False)
    with pytest.raises(MediaPoolError) as caught:
        list(bt.layers(many))
    assert caught.value.code == "bad_template" and "4096" in caught.value.message
    few = ms.envelope(ms.pb(*fields[:4000]), zstd=False)
    assert len(list(bt.layers(few))) <= 4096
