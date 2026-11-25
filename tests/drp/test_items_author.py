"""S25c2: timeline items authored from a media clip, a template item and a placement.

The expected rows are written by ``libwrite_support.item_row`` from the field map with its own
arithmetic; every formula and every refusal has a test that fails if it changes."""

from __future__ import annotations

import copy
import struct
from typing import Any

import pytest
from libwrite_support import AUDIO_TRACK, VIDEO_TRACK, item_row, linked_pair
from mediapool_support import (
    BYTES,
    I64,
    S,
    envelope,
    kd,
    mp4_template,
    pb,
    replace_row,
    track,
    uid,
    wav_template,
)

from aar.drp.items import ItemError, ItemPlacement, ItemRows, author_item, links_of, shape_of
from aar.drp.items.fields import compare_items, item_excluded
from aar.drp.items.media import media_view, timemap
from aar.drp.items.rows import check_item

pytest.importorskip("zstandard")

CLIP = mp4_template()
CLIPS = {CLIP.media_id: CLIP}
RATE = 24000 / 1001


def _template(kind: str = "picture", *, linked: bool = True, **change: Any) -> ItemRows:
    pair = linked_pair(CLIP, 0xC000, start=86400, duration=48, src_in=0)
    row, link = pair[0] if kind == "picture" else pair[1]
    if not linked:
        row = item_row(
            kind,
            row["Sm2TiItem_id"],
            row["Sm2TiTrack_id"],
            CLIP,
            start=86400,
            duration=48,
            src_in=0,
        )
    row = {**row, **change}
    return ItemRows(row, link)


def _place(kind: str = "picture", **change: Any) -> ItemPlacement:
    values: dict[str, Any] = {
        "kind": kind,
        "track_id": VIDEO_TRACK if kind == "picture" else AUDIO_TRACK,
        "index": 2,
        "start": 86400 + 100,
        "duration": 50,
        "src_in": 25,
        "partners": (uid(0xD001 if kind == "picture" else 0xD000),),
    }
    values.update(change)
    return ItemPlacement(**values)


def _author(kind: str = "picture", template: ItemRows | None = None, **change: Any) -> ItemRows:
    item_id = uid(0xD000 if kind == "picture" else 0xD001)
    got = author_item(
        CLIP, template or _template(kind), CLIPS, _place(kind, **change), item_id=item_id
    )
    return got.rows()


def _expected(kind: str, **change: Any) -> ItemRows:
    place = _place(kind, **change)
    item_id = uid(0xD000 if kind == "picture" else 0xD001)
    row = item_row(
        kind,
        item_id,
        place.track_id,
        CLIP,
        start=place.start,
        duration=place.duration,
        src_in=place.src_in,
        partners=place.partners,
        selector=0,  # the field map: always 0, whatever the template holds
    )
    return ItemRows(
        row,
        {
            "DbOwner": place.track_id,
            "DbAssociate": item_id,
            "DbPropertyName": "Items",
            "DbIndex": place.index,
        },
    )


@pytest.mark.parametrize("kind", ["picture", "sound"])
def test_an_item_reproduces_the_field_map_in_every_field(kind: str) -> None:
    compared, diffs, hidden = compare_items(_expected(kind), _author(kind), {kind})
    assert compared > 40
    assert diffs == []
    assert hidden == []


def test_current_selector_idx_is_zero_whatever_the_template_holds() -> None:
    template = _template("picture")
    assert template.item["CurrentSelectorIdx"] == 4242  # a made-up template value
    assert _author("picture", template).item["CurrentSelectorIdx"] == 0
    assert _author("sound").item["CurrentSelectorIdx"] == 0


def test_the_time_map_is_the_frames_over_the_rate_minus_one_frame() -> None:
    got = _author("picture").item["MediaTimemapBA"]
    assert got == b"\x02" + struct.pack(">d", 240 / RATE - 1 / RATE)
    assert got.hex() == "024023efc3ece2a534"


def test_start_and_duration_are_text_and_in_is_null_from_frame_zero() -> None:
    item = _author("picture", src_in=0).item
    assert (item["Start"], item["Duration"], item["In"]) == ("86500", "50", None)
    assert _author("picture", src_in=7).item["In"] == "7"


def test_the_media_start_is_the_timecode_frames_over_the_rate() -> None:
    assert _author("picture").item["MediaStartTime"] == 86400 / RATE


def test_the_link_list_names_the_partners_highest_key_first() -> None:
    a, b = uid(0xE001), uid(0xE002)
    item = _author("picture", partners=(a, b)).item
    assert links_of(item) == (a, b)
    from aar.drp.mediapool import blobtree as bt

    _, payload = bt.message_of(item["FieldsBlob"], "x")
    inner = bt.pb_decode(bt.pb_decode(payload, "x")[0][2], "y")
    group = bt.pb_decode(inner[0][2], "z")
    keys = [e[0] for e in bt.kd_decode(group[0][2], "links")]
    assert keys == ["1", "0"]


def test_an_unlinked_item_has_no_link_group_and_is_framed_plain() -> None:
    item = _author("picture", partners=()).item
    assert links_of(item) == ()
    assert item["FieldsBlob"][8] == 0x80  # 64 bytes or less: plain
    linked = _author("picture").item
    assert linked["FieldsBlob"][8] == 0x81  # longer: zstd, as Resolve frames it


@pytest.mark.parametrize(
    ("channels", "ids", "audio_type"), [(1, [0x4001], 1), (2, [0x4001, 0x4002], 0)]
)
def test_a_sound_lists_one_channel_id_per_channel(
    channels: int, ids: list[int], audio_type: int
) -> None:
    clip = replace_row(
        CLIP,
        "audio0",
        TracksBA=track(
            uid(0x209),
            start=86400 / RATE,
            rate=48000,
            channels=channels,
            samples=480480,
            codec="Linear PCM",
            depth=1,
        ),
    )
    item = author_item(clip, _template("sound"), CLIPS, _place("sound"), item_id=uid(0xD001)).item
    from aar.drp.mediapool import blobtree as bt

    entries = {e[0]: e[3] for e in bt.kd_decode(item["VirtualAudioTrackBA"], "v")}
    assert entries["ChannelsBA"] == struct.pack(">II", 2, channels) + b"".join(
        struct.pack(">I", i) for i in ids
    )
    assert entries["AudioType"] == audio_type


def test_a_sound_of_six_channels_is_refused() -> None:
    clip = replace_row(
        CLIP,
        "audio0",
        TracksBA=track(
            uid(0x209),
            start=0.0,
            rate=48000,
            channels=6,
            samples=480480,
            codec="Linear PCM",
            depth=1,
        ),
    )
    with pytest.raises(ItemError) as info:
        author_item(clip, _template("sound"), CLIPS, _place("sound"), item_id=uid(0xD001))
    assert info.value.code == "unsupported_item"


def test_a_sound_file_takes_its_import_rate_and_its_samples_as_frames() -> None:
    wav = replace_row(
        wav_template(),
        "audio0",
        TracksBA=track(
            uid(0x104),
            start=0.0,
            rate=48000,
            channels=2,
            samples=2000082,
            codec="Linear PCM",
            depth=2,
        ),
    )
    view = media_view(wav)
    rate = 30000 / 1001
    assert view.rate_bytes == struct.pack("<d", rate) + bytes(8)
    # Resolve's value: the samples as frames at the rate (a double) over the rate, minus a frame.
    want = 2000082 / 48000 * rate / rate - 1 / rate
    assert view.last_seconds == want
    assert want != 2000082 / 48000 - 1 / rate  # the plain form differs in the last bit here
    assert timemap(view.last_seconds) == b"\x02" + struct.pack(">d", want)


def test_later_additions_of_the_template_are_not_copied() -> None:
    template = _template(
        "picture",
        pLmVerTable=uid(0xF001),
        EffectFiltersBA=b"\x00\x01",
        CompositionTable=uid(0xF002),
        ImportExportMetadataBA=kd(("X", S, "y")),
        MarkersBA=b"",
        MediaMetadata=b"",
    )
    item = _author("picture", template=template).item
    for column in (
        "pLmVerTable",
        "EffectFiltersBA",
        "CompositionTable",
        "ImportExportMetadataBA",
        "MarkersBA",
        "MediaMetadata",
    ):
        assert item[column] is None, column


def test_the_template_edit_history_is_left_out() -> None:
    links = kd(("0", S, uid(0xC001)))
    history = envelope(
        pb(
            (
                1,
                2,
                pb(
                    (1, 2, pb((1, 2, links), (3, 2, pb((1, 0, 1), (2, 0, 99))), (21, 0, 86400))),
                    (4, 0, 1),
                    (9, 0, 1),
                ),
            ),
            (2, 2, bytes.fromhex("000000000000000600000000ffffffff")),
        )
    )
    template = _template("picture", FieldsBlob=history)
    assert shape_of(template, CLIPS).tags >= {"history", "blob-later"}
    got = _author("picture", template=template)
    _, diffs, _ = compare_items(_expected("picture"), got, {"picture"})
    assert diffs == []


def test_a_template_id_left_in_a_copied_column_is_refused() -> None:
    template = _template("picture", Group=uid(0xC000))  # the template's own id
    with pytest.raises(ItemError) as info:
        _author("picture", template=template)
    assert info.value.code == "bad_template"
    assert "Group" in info.value.message


def test_a_template_of_the_other_kind_is_refused() -> None:
    with pytest.raises(ItemError) as info:
        _author("picture", template=_template("sound"))
    assert info.value.code == "kind_mismatch"


def test_a_linked_item_needs_a_linked_template() -> None:
    with pytest.raises(ItemError) as info:
        _author("picture", template=_template("picture", linked=False))
    assert info.value.code == "bad_template"


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"src_in": 200, "duration": 50}, "reads past"),
        ({"duration": 0}, "below one frame"),
        ({"start": -1}, "start"),
        ({"index": True}, "index"),
        ({"track_id": "x"}, "track id"),
        ({"partners": (uid(1), uid(1))}, "repeat"),
    ],
)
def test_a_bad_placement_is_refused(change: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ItemError) as info:
        _author("picture", **change)
    assert info.value.code == "bad_placement"
    assert fragment in info.value.message


def test_an_item_kind_other_than_picture_or_sound_is_refused() -> None:
    place = ItemPlacement("title", VIDEO_TRACK, 0, 86400, 10, 0)
    with pytest.raises(ItemError) as info:
        author_item(CLIP, _template(), CLIPS, place, item_id=uid(0xD000))
    assert info.value.code == "bad_placement"


def test_an_id_already_taken_is_refused() -> None:
    with pytest.raises(ItemError) as info:
        author_item(CLIP, _template(), CLIPS, _place(), item_id=uid(0xD000), taken={uid(0xD000)})
    assert info.value.code == "id_collision"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"Duration": "12|000000000000d03f"}, "sub-frame Duration (a true fraction"),
        ({"Duration": "12|7296ffffffffef3f"}, "whole frames stored with a fraction"),
        ({"In": "7|00"}, "unreadable In"),
        ({"Start": "x"}, "unreadable Start"),
        ({"MediaTimemapBA": kd(("DbType", S, "Sm2TimeMap"))}, "retimed"),
        ({"PrettyType": "Fusion Title"}, "title"),
        ({"MediaRef": uid(0x999)}, "not a file clip"),
        ({"MediaTrackIdx": 1}, "audio stream"),
        ({"NewColumn": 1}, "does not know"),
        ({"DbType": "Sm2TiTransition"}, "not a picture or sound"),
        (
            {"FieldsBlob": envelope(pb((1, 2, pb((4, 0, 1), (7, 0, 1))), (2, 2, bytes(16))))},
            "#1.#7",
        ),
        (
            {"FieldsBlob": envelope(pb((1, 2, pb((1, 2, b""), (4, 0, 1))), (2, 2, bytes(16))))},
            "empty group",
        ),
        ({"FieldsBlob": b"\x00\x01"}, "does not decode"),
    ],
)
def test_an_item_of_an_unsupported_shape_is_refused_with_its_reason(
    change: dict[str, Any], reason: str
) -> None:
    rows = _template("picture", **change)
    shape = shape_of(rows, CLIPS)
    assert shape.refused is not None and reason in shape.refused, shape
    with pytest.raises(ItemError):
        check_item(rows, CLIPS)


def test_exclusions_apply_only_to_the_tags_they_name() -> None:
    assert item_excluded("I.MediaStartTime", {"picture"}) is None
    assert item_excluded("I.MediaStartTime", {"picture", "exact-start"}) is not None
    assert item_excluded("I.CurrentSelectorIdx", {"sound"}) is None
    assert item_excluded("I.CurrentSelectorIdx", {"picture"}) is not None
    assert item_excluded("I.EffectFiltersBA>#1.#9", {"sound"}) is not None
    assert item_excluded("I.FieldsBlob", {"picture"}) is None
    assert item_excluded("I.FieldsBlob", {"picture", "blob-later"}) is not None


def test_a_damaged_media_clip_is_a_typed_refusal() -> None:
    clip = replace_row(CLIP, "video", Time=kd(("FrameRate", BYTES, b"\x00" * 3)))
    with pytest.raises(ItemError) as info:
        author_item(clip, _template(), CLIPS, _place(), item_id=uid(0xD000))
    assert info.value.code == "bad_media"


def test_the_template_item_is_not_changed() -> None:
    template = _template()
    before = copy.deepcopy(template)
    _author("picture", template=template)
    assert template == before


def test_a_samples_count_of_the_wrong_type_is_refused() -> None:
    wav = replace_row(
        wav_template(), "audio0", TracksBA=kd(("0", BYTES, kd(("Duration", I64, -1))))
    )
    with pytest.raises(ItemError):
        media_view(wav)


def _flip(value: bytes, rng: Any) -> bytes:
    data = bytearray(value)
    data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
    return bytes(data)


@pytest.mark.parametrize("seed_number", range(400))
def test_damaged_blobs_give_only_typed_refusals(seed_number: int) -> None:
    """Flip one bit in a blob the writer decodes: a template item's FieldsBlob, time map,
    VirtualAudioTrackBA or PreConformMediaExtents, or the media clip's Time (a fixed-layout
    frame rate among its values), TracksBA or Clip. Only a typed refusal or rows that differ
    from the expected ones only where the damage was copied may come out."""
    import random

    rng = random.Random(seed_number)
    kind = rng.choice(["picture", "sound"])
    template = _template(kind)
    clip = CLIP
    target = rng.choice(["item", "clip"])
    if target == "item":
        columns = [
            c
            for c in (
                "FieldsBlob",
                "MediaTimemapBA",
                "VirtualAudioTrackBA",
                "PreConformMediaExtents",
            )
            if isinstance(template.item.get(c), bytes)
        ]
        column = rng.choice(columns)
        template = ItemRows(
            {**template.item, column: _flip(template.item[column], rng)}, template.link
        )
    else:
        role, column = rng.choice(
            [("video", "Time"), ("video", "Clip"), ("audio0", "TracksBA"), ("audio0", "Clip")]
        )
        row = clip.video if role == "video" else clip.audio[0]
        assert row is not None
        clip = replace_row(clip, role, **{column: _flip(row[column], rng)})
    try:
        got = author_item(
            clip,
            template,
            {clip.media_id: clip},
            _place(kind),
            item_id=uid(0xD000 if kind == "picture" else 0xD001),
        )
    except ItemError:
        return
    assert got.item["Sm2TiItem_id"] == uid(0xD000 if kind == "picture" else 0xD001)


# -- S25e-f2: a damaged constant of a template item is refused when the template is read ------


def _constants(kind: str) -> dict[str, object]:
    from aar.drp.items.rows import ITEM_CONSTANTS, PRECONFORM

    return {**ITEM_CONSTANTS, "PreConformMediaExtents": PRECONFORM[kind]}


@pytest.mark.parametrize("kind", ["picture", "sound"])
@pytest.mark.parametrize("column", sorted(_constants("picture")))
def test_a_template_item_with_a_damaged_constant_is_refused_naming_the_column(
    kind: str, column: str
) -> None:
    """The item writer copies these columns from its template, and the verifier requires the
    one value each holds: a template holding another is refused ``bad_template`` naming the
    column, before anything is authored (not ``self_check_failed`` at placement)."""
    template = _template(kind)
    want = _constants(kind)[column]
    template.item[column] = 7 if want is None else "3|0000000000000000"
    with pytest.raises(ItemError) as info:
        _author(kind, template)
    assert info.value.code == "bad_template"
    assert column in info.value.message, info.value.message


def test_the_item_writer_and_the_verifier_state_the_same_constants() -> None:
    from aar.drp.items.rows import ITEM_CONSTANTS, PRECONFORM
    from aar.drp.libwrite.verify import ITEM_RULES, PICTURE_PRECONFORM

    stated = {c: r[1] for c, r in ITEM_RULES.items() if isinstance(r, tuple)}
    # CurrentSelectorIdx is written 0 whatever the template holds, so it is not copied.
    assert set(stated) - set(ITEM_CONSTANTS) == {"CurrentSelectorIdx"}
    for column, value in ITEM_CONSTANTS.items():
        assert value == stated[column] and type(value) is type(stated[column]), column
    assert PRECONFORM == {"picture": PICTURE_PRECONFORM, "sound": None}
    # The synthetic Resolve rows hold them too.
    for kind in ("picture", "sound"):
        row = _template(kind).item
        for column, value in _constants(kind).items():
            assert row[column] == value and type(row[column]) is type(value), (kind, column)
