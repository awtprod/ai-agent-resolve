"""S25c1: authoring a clip's rows from facts and a template; refusals rather than guesses."""

from __future__ import annotations

import copy
import struct
from datetime import UTC, datetime
from fractions import Fraction
from typing import Any

import mediapool_support as ms
import pytest

from aar.drp import blobs
from aar.drp.mediapool import Placement, Template, author, facts_from_json
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool.author import CLIP_26_NOTE
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.fields import flatten
from aar.drp.mediapool.kinds import KINDS, KindStatus, kind_of_facts, kind_of_rows
from aar.drp.mediapool.templates import ClipRows

PLACE = Placement(ms.FOLDER, 4, Fraction(30000, 1001), 1_795_000_000, ms.MAP, 99)

CREATED_US = int(datetime(2026, 3, 20, 11, 22, 33, tzinfo=UTC).timestamp()) * 1_000_000


def make(
    facts: dict[str, Any],
    template: ClipRows,
    *,
    with_facts: dict[str, Any] | None = None,
    **kw: Any,
):
    tfacts = facts_from_json(with_facts) if with_facts is not None else None
    # Most tests here are about one rule, not about the verdict: they see the rows even when the
    # result would be unverified, and check ``status`` where it matters. The refusal itself is
    # tested in "unverified results" below, without this.
    kw.setdefault("allow_unverified", True)
    return author(
        facts_from_json(facts),
        Template(template, tfacts),
        kw.pop("placement", PLACE),
        new_id=kw.pop("new_id", ms.new_ids()),
        **kw,
    )


def as_clip(rows: dict[str, tuple[dict[str, Any], ...]]) -> ClipRows:
    return ClipRows(
        media=rows["Sm2MpMedia"][0],
        video=rows["BtVideoInfo"][0] if "BtVideoInfo" in rows else None,
        audio=rows.get("BtAudioInfo", ()),
        audio_links=rows.get("BtAudioInfo_Sm2MpMedia", ()),
        folder_link=rows["Sm2MpFolder_Sm2MpMedia"][0],
        lockables=rows.get("BtLockableBlob", ()),
    )


def flat(result) -> dict[str, tuple[Any, ...]]:
    return flatten(as_clip(result.rows))


def refusal(code: str, *args: Any, **kw: Any) -> MediaPoolError:
    with pytest.raises(MediaPoolError) as caught:
        make(*args, **kw)
    assert caught.value.code == code, caught.value
    assert caught.value.hint
    return caught.value


# --------------------------------------------------------------------------- a sound file


def test_a_sound_file_is_authored_from_its_facts() -> None:
    result = make(ms.wav_facts(), ms.wav_template(), with_facts=ms.wav_template_facts())
    f = flat(result)
    assert result.status == "verified" and result.unverified == ()
    assert f["M.Name"] == ("text", ms.WAV_NAME)
    assert f["A0.Clip>#1"] == ("text", ms.PC_FOLDER)
    assert f["A0.Clip>#2"] == ("text", ms.WAV_NAME)
    assert f["A0.Clip>#3"] == ("text", "Tue Mar  3 04:05:06 2026")
    ctime_us = facts_from_json(ms.wav_facts()).ctime_utc_ns // 1000
    assert f["A0.Clip>#13"] == ("varint", ctime_us)
    assert f["A0.TracksBA.0.Duration"] == ("int", 480000)
    assert f["A0.TracksBA.0.SampleRate"] == ("int", 48000)
    assert f["A0.TracksBA.0.StartTime"] == ("float", (0.0).hex())
    extent = bytes.fromhex(f["M.FieldsBlob>#1.#5.0.MediaExtents"][1])
    assert struct.unpack("<dd", extent) == (0.0, 10.0)
    # the project's rate, and the date added, as a fresh import into that project
    assert f["M.FieldsBlob>#2.#1"] == ("f64", (30000 / 1001).hex())
    assert f["M.FieldsBlob>#1.#1.#1.#2"] == ("varint", 1_795_000_000 * 2)
    # the metadata blob: the file name rule, the empty entries kept
    assert f["L0.FieldsBlob.BlobData>#2>meta{215}"] == ("text", ms.WAV_NAME)
    assert f["L0.FieldsBlob.BlobData>#2>meta{16}"] == ("bytes", "")
    assert f["L0.BtLockableBlobMap_id"] == ("ref", "<uuid>")
    assert result.rows["BtLockableBlob"][0]["DbSavedTime"] == 99
    # references name the new rows
    assert f["M.FieldsBlob>#1.#5.0.ChannelVecBA.0.MediaRef"] == ("ref", "<A0>")
    assert f["M.FieldsBlob>#1.#5.0.ChannelVecBA.1.MediaRef"] == ("ref", "<A0>")
    assert f["L0.BlobOwner"] == ("ref", "<A0>")
    assert f["JA0.DbOwner"] == ("ref", "<M>") and f["JA0.DbAssociate"] == ("ref", "<A0>")
    assert result.rows["Sm2MpFolder_Sm2MpMedia"][0]["DbIndex"] == 4


def test_a_fresh_import_has_no_ui_state() -> None:
    media = make(ms.wav_facts(), ms.wav_template()).rows["Sm2MpMedia"][0]
    assert (
        media["CurPlayheadPosition"] is None
        and media["PinsBA"] is None
        and media["PTZRPreset"] is None
    )


def test_the_template_is_not_changed() -> None:
    template = ms.wav_template()
    before = copy.deepcopy(template)
    make(ms.wav_facts(), template)
    assert template == before


def test_ids_are_fresh_and_no_template_id_is_left() -> None:
    template = ms.wav_template()
    result = make(ms.wav_facts(), template)
    old = template.ids() | {ms.uid(0x104)}  # the track's UniqueId lives only inside TracksBA
    for rows in result.rows.values():
        for row in rows:
            for value in row.values():
                assert bt.contains_text(value, {i: i for i in old}) is None


# --------------------------------------------------------------------------- a camera file


def test_a_camera_file_is_authored_from_its_facts() -> None:
    result = make(ms.mp4_facts(), ms.mp4_template(), with_facts=ms.mp4_template_facts())
    f = flat(result)
    assert result.status == "verified"
    assert f["V.Time.Timecode"] == ("text", "02:10:20:12")
    assert f["V.Time.NumFrames"] == ("int", 3000)
    assert f["V.Time.FrameRate"] == ("fixed16", ms.rate16(24000 / 1001).hex())
    assert f["V.Geometry.Resolution"] == ("fixed16", struct.pack(">IIII", 0, 3840, 0, 2160).hex())
    assert f["V.Geometry.FrameSize"] == ("int", 3840 * 2160 * 3 // 2 + 8192)
    assert f["V.Clip>#5"] == ("text", "avc1") and f["A0.Clip>#5"] == ("text", "Linear PCM")
    assert f["V.Clip>#13"] == ("varint", CREATED_US)
    start = 187692 / (24000 / 1001)
    assert f["A0.TracksBA.0.StartTime"] == ("float", start.hex())
    extent = struct.unpack("<dd", bytes.fromhex(f["M.FieldsBlob>#1.#1.#5.0.MediaExtents"][1]))
    assert extent == (start, 6006000 / 48000)
    assert f["V.Clip>#7"] == ("ref", "<uuid>")
    # copied: unexplained, equal across the kind (a flag-only protobuf envelope)
    assert f["V.Radiometry"] == ("envelope", None, "protobuf")


def test_history_and_later_state_of_the_template_are_left_out() -> None:
    result = make(ms.mp4_facts(), ms.mp4_template(relink=True, presets=True))
    f = flat(result)
    assert not any(p.startswith("M.FieldsBlob>#2") for p in f)
    assert "M.FieldsBlob>#1.#3" not in f
    assert not any(p.endswith(("Clip>#14", "Clip>#26")) for p in f)
    assert "SmPreset" not in result.rows and result.rows["Sm2MpMedia"][0]["PTZRPreset"] is None
    assert len(result.notes) == 8 and result.notes[-1] == CLIP_26_NOTE


def test_without_the_templates_facts_its_look_is_unverified() -> None:
    result = make(ms.mp4_facts(), ms.mp4_template())
    assert result.status == "unverified"
    assert any("Radiometry" in reason for reason in result.unverified)


def test_a_template_whose_file_has_another_look_is_refused() -> None:
    other = ms.mp4_facts(video_color_transfer=None)
    refusal("kind_mismatch", ms.mp4_facts(), ms.mp4_template(), with_facts=other)


def test_an_unverified_kind_says_so() -> None:
    kind = kind_of_facts(facts_from_json(ms.wav_facts()))
    result = make(
        ms.wav_facts(),
        ms.wav_template(),
        with_facts=ms.wav_template_facts(),
        kinds={kind: KindStatus("unverified", 1, "one file")},
    )
    assert result.status == "unverified" and result.unverified == ("kind not verified: one file",)


# --------------------------------------------------------------------------- a still


def test_a_still_takes_its_size_and_the_projects_rate() -> None:
    place = Placement(ms.FOLDER, 0, Fraction(24000, 1001), 1, ms.MAP)
    f = flat(
        make(ms.png_facts(), ms.png_template(), placement=place, with_facts=ms.png_template_facts())
    )
    assert f["V.Geometry.FrameSize"] == ("int", 40961)
    assert f["V.Time.FrameRate"] == ("fixed16", ms.rate16(24000 / 1001).hex())
    assert f["V.Time.NumFrames"] == ("int", 1)
    assert "V.Clip>#5" not in f


def test_a_still_named_with_digits_is_refused() -> None:
    refusal("cannot_derive", ms.png_facts(path="X:\\Show\\card7.png"), ms.png_template())


# --------------------------------------------------------------------------- refusals


def test_no_template_and_a_template_of_another_kind_are_refused() -> None:
    with pytest.raises(MediaPoolError) as caught:
        author(facts_from_json(ms.wav_facts()), None, PLACE)  # type: ignore[arg-type]
    assert caught.value.code == "no_template"
    refusal("kind_mismatch", ms.wav_facts(), ms.mp4_template())
    refusal("kind_mismatch", ms.wav_facts(stream_channels=1), ms.wav_template())


def test_a_kind_without_evidence_is_refused() -> None:
    refusal(
        "unsupported_kind", ms.wav_facts(stream_sample_rate="96000"), ms.wav_template(rate=96000)
    )


def test_a_picture_ffprobe_cannot_decode_is_refused() -> None:
    braw = ms.mp4_facts(
        video_codec_name=None, video_width=None, video_height=None, video_pix_fmt=None
    )
    refusal("cannot_derive", braw, ms.mp4_template())


def test_variable_rate_without_timecode_track_is_refused() -> None:
    refusal(
        "vfr_without_timecode",
        ms.mp4_facts(video_r_frame_rate="30/1", video_avg_frame_rate="1234567/41152"),
        ms.mp4_template(),
    )


def test_a_metadata_entry_no_rule_derives_is_refused() -> None:
    template = ms.replace_row(
        ms.wav_template(), "lockable", FieldsBlob=ms.metadata([(215, "x"), (107, "47")])
    )
    refusal("cannot_derive", ms.wav_facts(), template)


def test_a_reel_number_is_refused() -> None:
    template = ms.mp4_template()
    meta = ms.kd(
        ("UniqueId", ms.S, ms.uid(0x2FF)),
        ("DbType", ms.S, "BtVideoMetadata"),
        ("ActualReelNumber", ms.S, "1"),
    )
    refusal("cannot_derive", ms.mp4_facts(), ms.replace_row(template, "video", VideoMetadata=meta))


def test_a_metadata_blob_needs_the_projects_lockable_map() -> None:
    place = Placement(ms.FOLDER, 0, Fraction(30000, 1001), 1, None)
    refusal("cannot_derive", ms.wav_facts(), ms.wav_template(), placement=place)


def test_a_bad_placement_is_refused() -> None:
    refusal(
        "bad_placement",
        ms.wav_facts(),
        ms.wav_template(),
        placement=Placement("bin", 0, Fraction(25), 1, ms.MAP),
    )
    refusal(
        "bad_placement",
        ms.wav_facts(),
        ms.wav_template(),
        placement=Placement(ms.FOLDER, -1, Fraction(25), 1, ms.MAP),
    )


# --------------------------------------------------------------------------- hostile templates


def test_a_clip_blob_with_a_field_this_writer_does_not_know_is_refused() -> None:
    template = ms.wav_template()
    clip = ms.clip_blob(
        "X:\\Old", "older-take.wav", "x", "Linear PCM", 1, video=False, extra=((99, 0, 1),)
    )
    refusal("bad_template", ms.wav_facts(), ms.replace_row(template, "audio0", Clip=clip))


def test_a_keyed_dict_with_an_unknown_key_is_refused() -> None:
    template = ms.mp4_template()
    time = ms.kd(
        ("UniqueId", ms.S, ms.uid(1)),
        ("Timecode", ms.S, "01:00:00:00"),
        ("StartFrame", ms.I32, 0),
        ("NumFrames", ms.I32, 1),
        ("FrameRate", ms.BYTES, ms.rate16(23.976)),
        ("Surprise", ms.I32, 1),
    )
    refusal("bad_template", ms.mp4_facts(), ms.replace_row(template, "video", Time=time))


@pytest.mark.parametrize("column", ["Clip", "TracksBA"])
def test_a_blob_that_does_not_decode_is_refused(column: str) -> None:
    template = ms.replace_row(ms.wav_template(), "audio0", **{column: b"\x00\x00\x00\x02garbage"})
    refusal("bad_template", ms.wav_facts(), template)


def test_a_truncated_zstd_blob_is_refused() -> None:
    template = ms.wav_template()
    clip = template.audio[0]["Clip"]
    refusal("bad_template", ms.wav_facts(), ms.replace_row(template, "audio0", Clip=clip[:-4]))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-be", "utf-16-le"])
def test_a_template_id_somewhere_unexpected_is_refused(encoding: str) -> None:
    template = ms.mp4_template()
    hidden = template.audio[0]["BtAudioInfo_id"].encode(encoding)
    radiometry = b"\x80" + ms.pb((1, 0, 11), (40, 2, hidden))
    error = refusal(
        "bad_template", ms.mp4_facts(), ms.replace_row(template, "video", Radiometry=radiometry)
    )
    assert "still in the new BtVideoInfo.Radiometry" in error.message


def test_a_template_file_name_left_in_a_blob_is_refused() -> None:
    template = ms.mp4_template()
    radiometry = b"\x80" + ms.pb((1, 0, 11), (40, 2, b"camera-b.MP4"))
    refusal(
        "bad_template", ms.mp4_facts(), ms.replace_row(template, "video", Radiometry=radiometry)
    )


def test_a_channel_entry_naming_another_row_is_refused() -> None:
    template = ms.wav_template()
    fields = ms.media_fields(
        added=1,
        video=False,
        audio_extents=ms.extents(
            ms.uid(0xDEAD), channels=2, start=0.0, length=5.0, codec="Linear PCM", depth=2
        ),
    )
    refusal("bad_template", ms.wav_facts(), ms.replace_row(template, "media", FieldsBlob=fields))


def test_a_video_clip_fields_blob_with_unknown_fields_is_refused() -> None:
    template = ms.mp4_template()
    _, payload = bt.message_of(template.media["FieldsBlob"], "x")
    odd = ms.envelope(payload + ms.pb((9, 0, 1)))
    refusal("bad_template", ms.mp4_facts(), ms.replace_row(template, "media", FieldsBlob=odd))


# --------------------------------------------------------------------------- ids


def test_an_id_already_in_the_project_is_a_collision() -> None:
    first = ms.uid(0x9000)
    refusal("id_collision", ms.wav_facts(), ms.wav_template(), taken={first})


def test_an_id_source_that_repeats_or_returns_junk_is_refused() -> None:
    refusal("id_collision", ms.wav_facts(), ms.wav_template(), new_id=lambda: ms.uid(1))
    refusal("id_collision", ms.wav_facts(), ms.wav_template(), new_id=lambda: "not-a-uuid")
    template = ms.wav_template()
    refusal("id_collision", ms.wav_facts(), template, new_id=lambda: template.media_id)


def test_rows_kind_and_facts_kind_agree_for_every_synthetic_kind() -> None:
    for facts, template in (
        (ms.wav_facts(), ms.wav_template()),
        (ms.mp4_facts(), ms.mp4_template()),
        (ms.png_facts(), ms.png_template()),
    ):
        kind = kind_of_facts(facts_from_json(facts))
        assert kind_of_rows(template).without_look() == kind.without_look()
        assert kind in KINDS


def test_rewritten_blobs_keep_their_version_and_take_resolves_codec_for_their_length() -> None:
    template = ms.mp4_template()
    result = make(ms.mp4_facts(), template)
    for column in ("Clip",):
        old = blobs.parse_envelope(template.video[column])
        new = blobs.parse_envelope(result.rows["BtVideoInfo"][0][column])
        assert old is not None and new is not None
        payload = blobs.unwrap_message(result.rows["BtVideoInfo"][0][column])
        assert old.version == new.version
        assert new.codec == ("zstd" if len(payload) > 64 else "protobuf")
