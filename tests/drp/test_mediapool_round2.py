"""S25c1 fix round 2: one typed boundary, a wider envelope, template facts required, dates.

Each section names the fix-check item it pins (A to E). Everything is synthetic
(``mediapool_support``); the aimed fuzz for item A is in ``test_mediapool_fuzz.py``.
"""

from __future__ import annotations

import inspect
import struct
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import mediapool_support as ms
import pytest

import aar.drp.mediapool as mediapool
from aar.drp import blobs
from aar.drp.errors import BlobError, DrpError
from aar.drp.library import LibraryError, LibraryOptions
from aar.drp.mediapool import (
    KINDS,
    MediaPoolError,
    Placement,
    Template,
    author,
    facts_from_json,
    kind_of_rows,
    read_templates,
)
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool import templates as tp
from aar.drp.mediapool.templates import ClipRows, check_clip, clips_from_rows
from aar.drp.modify import encode
from aar.drp.modify.encode import EncodeError

PLACE = Placement(ms.FOLDER, 4, Fraction(30000, 1001), 1_795_000_000, ms.MAP, 99)
FAST = LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0)
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def run(
    facts: dict[str, Any],
    template: ClipRows,
    template_facts: dict[str, Any] | None,
    **kw: Any,
):
    return author(
        facts_from_json(facts),
        Template(template, facts_from_json(template_facts) if template_facts else None),
        kw.pop("placement", PLACE),
        new_id=kw.pop("new_id", ms.new_ids()),
        **kw,
    )


def refused(code: str, *args: Any, **kw: Any) -> MediaPoolError:
    with pytest.raises(MediaPoolError) as caught:
        run(*args, **kw)
    assert caught.value.code == code, caught.value
    assert caught.value.hint
    return caught.value


def mp4(**change: Any) -> tuple[dict[str, Any], ClipRows, dict[str, Any]]:
    return ms.mp4_facts(**change), ms.mp4_template(), ms.mp4_template_facts()


def wav(**change: Any) -> tuple[dict[str, Any], ClipRows, dict[str, Any]]:
    return ms.wav_facts(**change), ms.wav_template(), ms.wav_template_facts()


# =========================================================================== A. one typed boundary


def _with_resolution(clip: ClipRows, value: bytes) -> ClipRows:
    assert clip.video is not None
    entries = [
        (e.key, e.type, e.subtype, value if e.key == "Resolution" else e.value)
        for e in blobs.decode_keyed_dict(clip.video["Geometry"], strict=True).entries
    ]
    return ms.replace_row(clip, "video", Geometry=encode.keyed_dict(entries))


@pytest.mark.parametrize("at", [0, 3, 8, 11])
def test_a_resolution_with_a_non_zero_padding_word_is_a_typed_refusal(at: int) -> None:
    value = bytearray(struct.pack(">IIII", 0, 1920, 0, 1080))
    value[at] = 1
    template = _with_resolution(ms.mp4_template(), bytes(value))
    with pytest.raises(MediaPoolError) as caught:
        kind_of_rows(template)
    assert caught.value.code == "bad_template" and "padding" in caught.value.message
    assert isinstance(caught.value.__cause__, BlobError)
    error = refused("bad_template", ms.mp4_facts(), template, ms.mp4_template_facts())
    assert "BlobError" in error.message and "padding" in error.message


def _exported_functions() -> list[tuple[str, Any]]:
    out = [
        (name, getattr(mediapool, name))
        for name in mediapool.__all__
        if inspect.isfunction(getattr(mediapool, name))
    ]
    out += [
        ("check_clip", check_clip),
        ("clips_from_rows", clips_from_rows),
        ("ClipRows.from_tables", ClipRows.from_tables),
        ("ClipRows.ids", ClipRows.ids),
        ("ClipRows.media_id", ClipRows.media_id.fget),
    ]
    return out


def test_every_public_entry_point_is_behind_the_boundary() -> None:
    names = [name for name, _ in _exported_functions()]
    assert {"author", "facts_from_json", "kind_of_facts", "kind_of_rows", "read_templates"} <= set(
        names
    )
    unguarded = [name for name, func in _exported_functions() if not hasattr(func, "__wrapped__")]
    assert unguarded == []


@pytest.mark.parametrize(
    "planted",
    [
        BlobError("planted damage"),
        EncodeError("planted damage"),
        struct.error("planted damage"),
        IndexError("planted damage"),
        KeyError("planted damage"),
        ZeroDivisionError("planted damage"),
        OverflowError("planted damage"),
        ValueError("planted damage"),
        TypeError("planted damage"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_an_error_a_shared_decoder_raises_becomes_a_typed_refusal(
    planted: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever a decoder raises, below every specific check, leaves the package typed and
    with its message kept: here the keyed-dict decoder itself raises."""

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise planted

    facts, template, template_facts = mp4()
    parsed, parsed_template = facts_from_json(facts), facts_from_json(template_facts)
    monkeypatch.setattr(blobs, "decode_keyed_dict", broken)
    for call in (
        lambda: kind_of_rows(template),
        lambda: author(parsed, Template(template, parsed_template), PLACE, new_id=ms.new_ids()),
    ):
        with pytest.raises(MediaPoolError) as caught:
            call()
        assert caught.value.code == "bad_template"
        assert "planted damage" in caught.value.message
        # a DrpError is caught by the specific check around the decoder; the rest at the boundary
        assert caught.value.__cause__ is planted or isinstance(planted, DrpError)


def test_the_template_readers_refuse_in_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(MediaPoolError) as caught:
        ClipRows.from_tables({"BtVideoInfo": ()})
    assert caught.value.code == "bad_template" and "KeyError" in caught.value.message
    db = ms.write_project(tmp_path / "Project.db", ms.wav_template())

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise IndexError("planted damage")

    monkeypatch.setattr(tp, "read_rows", broken)
    with pytest.raises(MediaPoolError) as caught:
        read_templates(db, options=FAST)
    assert caught.value.code == "bad_template" and "planted damage" in caught.value.message


def test_the_project_readers_own_typed_error_passes_unchanged(tmp_path: Path) -> None:
    db = tmp_path / "Project.db"
    db.write_bytes(b"not a database")
    with pytest.raises(LibraryError):
        read_templates(db, options=FAST)


def test_an_id_source_that_raises_is_a_typed_refusal() -> None:
    """``new_id`` is the caller's code, not untrusted input; what it raises is still typed, and
    named as the id source's failure rather than taken for a damaged template."""

    def failing() -> str:
        raise RuntimeError("boom")

    facts, template, template_facts = mp4()
    error = refused("id_collision", facts, template, template_facts, new_id=failing)
    assert "RuntimeError: boom" in error.message
    assert isinstance(error.__cause__, RuntimeError)


# =========================================================================== B. the envelope


@pytest.mark.parametrize(
    ("change", "prop"),
    [
        ({"video_field_order": "tt"}, "field order"),
        ({"video_field_order": None}, "field order"),  # absent is compared as absent
        ({"video_sample_aspect_ratio": "4:3"}, "sample aspect ratio"),
        ({"video_display_aspect_ratio": "4:3"}, "display aspect ratio"),
        ({"video_level": 52}, "video level"),
        ({"video_has_b_frames": 2}, "has b frames"),
        ({"video_coded_height": 2176}, "coded size"),
        ({"video_chroma_location": "center"}, "chroma location"),
        ({"video_bits_per_raw_sample": "10"}, "video raw bits"),
        # the rate comes from a QuickTime timecode track; the stream's own rates say otherwise
        (
            {
                "video_r_frame_rate": "25/1",
                "video_avg_frame_rate": "25/1",
                "data_codec_tag_string": "tmcd",
                "data_avg_frame_rate": "24000/1001",
            },
            "video stream rates",
        ),
        ({"audio_bits_per_sample": 24}, "audio bits per sample"),
        ({"audio_sample_fmt": "s32"}, "audio sample format"),
        ({"audio_initial_padding": 1024}, "audio initial padding"),
        ({"audio_codec_name": "pcm_s16le"}, "audio codec"),
    ],
)
def test_a_picture_file_outside_the_envelope_is_unverified(
    change: dict[str, Any], prop: str
) -> None:
    facts, template, template_facts = mp4(**change)
    error = refused("unverified", facts, template, template_facts)
    assert f"the file's {prop} " in error.message
    result = run(facts, template, template_facts, allow_unverified=True)
    assert result.status == "unverified"
    assert any(r.startswith(f"the file's {prop} ") for r in result.unverified), result.unverified


@pytest.mark.parametrize(
    ("change", "prop"),
    [
        ({"stream_bits_per_sample": 32}, "audio bits per sample"),
        ({"stream_bits_per_raw_sample": "20"}, "audio raw bits"),
        ({"stream_initial_padding": None}, "audio initial padding"),
        ({"stream_time_base": "1/96000"}, "audio time base"),
    ],
)
def test_a_sound_file_outside_the_envelope_is_unverified(change: dict[str, Any], prop: str) -> None:
    facts, template, template_facts = wav(**change)
    error = refused("unverified", facts, template, template_facts)
    assert f"the file's {prop} " in error.message


def test_a_file_inside_the_envelope_is_verified() -> None:
    for facts, template, template_facts in (mp4(), wav()):
        result = run(facts, template, template_facts)
        assert (result.status, result.unverified) == ("verified", ())


def test_every_verified_kind_records_every_envelope_property() -> None:
    for kind, status in KINDS.items():
        if status.status == "verified":
            assert status.envelope is not None, kind.label()
            assert set(status.envelope) == set(mediapool.kinds.ENVELOPE_PROPERTIES)
            assert all(len(values) == 1 for values in status.envelope.values()), kind.label()


# =========================================================================== D. template facts


@pytest.mark.parametrize("make", [mp4, wav])
def test_a_template_without_its_facts_is_never_verified(make: Any) -> None:
    facts, template, _ = make()
    error = refused("unverified", facts, template, None)
    assert "template's own facts are missing" in error.message
    result = run(facts, template, None, allow_unverified=True)
    assert result.status == "unverified"
    assert any("own facts are missing" in r for r in result.unverified)


def test_a_still_template_without_its_facts_is_never_verified() -> None:
    error = refused("unverified", ms.png_facts(), ms.png_template(), None)
    assert "template's own facts are missing" in error.message


# =========================================================================== E. dates


def _local(moment: datetime) -> str:
    return (
        f"{_DAYS[moment.weekday()]} {_MONTHS[moment.month - 1]} {moment.day} "
        f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d} {moment.year:04d}"
    )


@pytest.mark.parametrize(
    "moment",
    [
        datetime(1, 1, 1),
        datetime(1969, 12, 31, 23, 59, 59),
        datetime(fm.LAST_YEAR + 1, 1, 1),
        datetime(9999, 12, 31, 23, 59, 59),
    ],
    ids=str,
)
def test_a_modification_time_out_of_range_is_refused(moment: datetime) -> None:
    facts, template, template_facts = wav(mtime_local=_local(moment))
    error = refused("bad_facts", facts, template, template_facts, allow_unverified=True)
    assert f"year {moment.year}" in error.message and str(fm.LAST_YEAR) in error.message


@pytest.mark.parametrize(
    "moment", [datetime(1970, 1, 1), datetime(fm.LAST_YEAR, 12, 31, 23, 59, 59)], ids=str
)
def test_a_modification_time_in_range_is_written_with_a_four_digit_year(moment: datetime) -> None:
    facts, template, template_facts = wav(mtime_local=_local(moment))
    result = run(facts, template, template_facts, allow_unverified=True)
    _, payload = bt.message_of(result.rows["BtAudioInfo"][0]["Clip"], "Clip")
    text = bt.pb_get(bt.pb_decode(payload, "Clip"), 3, "Clip")[2].decode()
    assert text == fm.qt_local_time(moment) and text.endswith(f" {moment.year:04d}")


def test_creation_times_and_the_date_added_share_the_range() -> None:
    late = f"{fm.LAST_YEAR + 1}-01-01T00:00:00.0000000Z"
    last = f"{fm.LAST_YEAR}-12-31T23:59:59.0000000Z"
    # a sound file's creation time is the file's (no creation_time tag)
    facts, template, template_facts = wav(ctime_utc=late)
    refused("cannot_derive", facts, template, template_facts)
    facts, template, template_facts = wav(ctime_utc=last)
    run(facts, template, template_facts)
    # a picture file's is its creation_time tag
    facts, template, template_facts = mp4()
    facts["ffprobe"]["format"]["tags"]["creation_time"] = late.replace("0000000Z", "000000Z")
    refused("cannot_derive", facts, template, template_facts)
    end = int((datetime(fm.LAST_YEAR + 1, 1, 1) - datetime(1970, 1, 1)).total_seconds()) - 1
    assert end == fm.LATEST_SECONDS
    facts, template, template_facts = wav()
    ok = Placement(ms.FOLDER, 4, Fraction(30000, 1001), end, ms.MAP, 99)
    run(facts, template, template_facts, placement=ok)
    late_placement = Placement(
        ms.FOLDER, 4, Fraction(30000, 1001), end + int(timedelta(seconds=1).total_seconds()), ms.MAP
    )
    error = refused("bad_placement", facts, template, template_facts, placement=late_placement)
    assert str(fm.LAST_YEAR) in error.message
