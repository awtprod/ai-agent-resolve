"""S25e: sound transitions in a library project: the field map, the writer's refusals, the
joins they sit on, the verifier, the seed and a fuzz of damaged templates.

Templates are synthetic (``libwrite_support.transition_row``, written from DESIGN's field map,
not from the product). Every formula has a test that fails if it changes."""

from __future__ import annotations

import random
import sqlite3
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from libwrite_support import (
    AUDIO_TRACK,
    TRANSITION_FIELD,
    TRANSITION_VALUE,
    crossfade,
    linked_pair,
    make_library,
    mp4_template,
    qcompress,
    transition_row,
    write_seed,
)
from mediapool_support import envelope, join, pb, uid

from aar.drp.items.errors import ItemError
from aar.drp.items.rows import ITEM_COLUMNS, ItemRows, read_items
from aar.drp.items.transitions import (
    SOUND_KINDS,
    TRANSITION_CONSTANTS,
    TransitionPlacement,
    author_transition,
    check_transition,
    frames_before,
    transition_shape,
)
from aar.drp.library import LibraryOptions
from aar.drp.libwrite import Authored, ProjectWriteError, make_project
from aar.drp.libwrite.cut import Cut, CutTransition, author_cut, pick_item_templates
from aar.drp.libwrite.verify import (
    DERIVED,
    TRANSITION_NAMES,
    TRANSITION_PAYLOAD,
    TRANSITION_RULES,
    verify_written,
)
from aar.drp.mediapool.errors import MediaPoolError
from aar.plan.timeline_import import READER_WARNING_CODES, read_timeline

pytest.importorskip("zstandard")

OPTS = LibraryOptions(settle_seconds=0.0)
CLIP = mp4_template()  # 240 frames of picture and sound
RATE = Fraction(30000, 1001)
START = 108000
NEW = uid(0xE000)


def _template(**change: Any) -> ItemRows:
    row = transition_row(uid(0xC000), AUDIO_TRACK, start=1000, frames=2, **change)
    return ItemRows(row, join(AUDIO_TRACK, row["Sm2TiItem_id"], "Items", 1))


def _place(**change: Any) -> TransitionPlacement:
    data: dict[str, Any] = {
        "name": "Cross Fade 0 dB",
        "track_id": uid(0xD000),
        "index": 3,
        "cut": START + 48,
        "frames": 2,
    }
    data.update(change)
    return TransitionPlacement(**data)


# -- the field map ----------------------------------------------------------------------------


@pytest.mark.parametrize(("frames", "before"), [(2, 1), (3, 2), (4, 2), (15, 8), (16, 8)])
def test_centred_resolve_puts_the_odd_frame_before_the_cut(frames: int, before: int) -> None:
    assert frames_before("centre", frames) == before
    authored = author_transition(_template(), _place(frames=frames), transition_id=NEW)
    assert authored.item["Start"] == str(START + 48 - before)
    assert authored.item["Duration"] == str(frames)


@pytest.mark.parametrize("name", sorted(SOUND_KINDS))
def test_the_kind_is_pretty_type_alone(name: str) -> None:
    for template in (_template(pretty="Cross Fade 0DB"), _template(pretty="Cross Fade +3DB")):
        row = author_transition(template, _place(name=name), transition_id=NEW).item
        assert row["PrettyType"] == SOUND_KINDS[name]
        others = {c: v for c, v in row.items() if c != "PrettyType"}
        again = author_transition(_template(), _place(name=name), transition_id=NEW).item
        assert others == {c: v for c, v in again.items() if c != "PrettyType"}


def test_placed_identity_and_constant_columns() -> None:
    template = _template(history=True)
    authored = author_transition(template, _place(), transition_id=NEW)
    row, link = authored.item, authored.link
    assert authored.kind == "transition"
    assert row["Sm2TiItem_id"] == NEW
    assert (row["AlignmentType"], row["Position"]) == (2, 2)
    assert row["Sm2TiTrack_id"] == uid(0xD000)
    assert link == {
        "DbOwner": uid(0xD000),
        "DbAssociate": NEW,
        "DbPropertyName": "Items",
        "DbIndex": 3,
    }
    # FieldsBlob: the template's #1 (later-edit state) left out, #2 kept byte for byte.
    assert row["FieldsBlob"] == envelope(pb((2, 2, TRANSITION_FIELD)))
    copied = set(ITEM_COLUMNS) - {
        "Sm2TiItem_id",
        "Start",
        "Duration",
        "Sm2TiTrack_id",
        "AlignmentType",
        "Position",
        "PrettyType",
        "FieldsBlob",
    }
    for column in copied:
        assert row[column] == template.item[column], column
    assert set(row) == set(ITEM_COLUMNS)


def test_the_writer_and_the_verifier_name_the_same_kinds_and_value() -> None:
    import zlib

    from aar.drp.items.transitions import SOUND_FIELD, SOUND_PAYLOAD
    from aar.drp.libwrite.verify import TRANSITION_FIELD as VERIFIER_FIELD

    assert TRANSITION_NAMES == SOUND_KINDS
    assert TRANSITION_PAYLOAD == TRANSITION_VALUE == SOUND_PAYLOAD
    assert SOUND_FIELD == VERIFIER_FIELD == TRANSITION_FIELD
    assert zlib.decompress(SOUND_FIELD[4:]) == SOUND_PAYLOAD
    assert struct.unpack(">I", SOUND_FIELD[:4]) == (len(SOUND_PAYLOAD),)
    assert set(TRANSITION_RULES) == set(ITEM_COLUMNS)
    derived = {c for c, r in TRANSITION_RULES.items() if r == DERIVED}
    assert derived == {
        "Sm2TiItem_id",
        "DbType",
        "Start",
        "Duration",
        "Sm2TiTrack_id",
        "AlignmentType",
        "Position",
        "PrettyType",
        "FieldsBlob",
    }
    # Every constant the verifier states is what the synthetic Resolve row holds.
    row = transition_row(NEW, AUDIO_TRACK, start=10, frames=2)
    for column, rule in TRANSITION_RULES.items():
        if rule != DERIVED:
            assert row[column] == rule[1], column  # type: ignore[index]
            assert type(row[column]) is type(rule[1]), column  # type: ignore[index]


# -- refusals: placements ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "code", "words"),
    [
        ({"name": "Cross Fade -3 dB"}, "unsupported_item", "only"),
        ({"name": "Cross Dissolve"}, "unsupported_item", "only"),
        ({"alignment": "start"}, "unsupported_item", "aligned"),
        ({"alignment": "end"}, "unsupported_item", "aligned"),
        ({"alignment": "middle"}, "unsupported_item", "aligned"),
        ({"frames": 0}, "bad_placement", "below one frame"),
        ({"frames": -2}, "bad_placement", "whole number"),
        ({"frames": 1}, "unsupported_item", "lengths compared"),
        ({"frames": 17}, "unsupported_item", "lengths compared"),
        ({"frames": 10**9}, "bad_placement", "whole number"),
        ({"frames": True}, "bad_placement", "whole number"),
        ({"frames": 2.0}, "bad_placement", "whole number"),
        ({"rate": Fraction(24000, 1001)}, "unsupported_item", "29.97"),
        ({"rate": Fraction(25)}, "unsupported_item", "29.97"),
        ({"cut": 0}, "bad_placement", "before it"),
        ({"cut": -5}, "bad_placement", "whole number"),
        ({"index": -1}, "bad_placement", "whole number"),
        ({"track_id": "short"}, "bad_placement", "uuid"),
    ],
)
def test_a_placement_outside_the_evidence_is_a_typed_refusal(
    change: dict[str, Any], code: str, words: str
) -> None:
    with pytest.raises(ItemError) as info:
        author_transition(_template(), _place(**change), transition_id=NEW)
    assert info.value.code == code, info.value
    assert words in info.value.message


def test_an_id_that_is_taken_is_refused() -> None:
    with pytest.raises(ItemError) as info:
        author_transition(_template(), _place(), transition_id=NEW, taken={NEW})
    assert info.value.code == "id_collision"


# -- refusals: templates ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"PrettyType": "Smooth Cut"}, "not a kind"),
        ({"PrettyType": None}, "not a kind"),
        ({"AlignmentType": 1, "Position": 1}, "starts on the cut"),
        ({"AlignmentType": 3, "Position": 3}, "ends on the cut"),
        ({"AlignmentType": 2, "Position": 1}, "differ"),
        ({"AlignmentType": 7, "Position": 7}, "never seen"),
        ({"Duration": "2|0000000000c06e3d"}, "N|hex"),
        ({"Start": "abc"}, "unreadable"),
        ({"EffectFiltersBA": b"\x00"}, "holds"),
        ({"MediaRef": uid(5)}, "holds"),
        ({"In": "3"}, "holds"),
        ({"Duration": "40"}, "outside the lengths"),
        ({"FieldsBlob": envelope(pb((2, 2, qcompress(b"\x01" * 44))))}, "#2"),
        ({"FieldsBlob": envelope(pb((3, 0, 1), (2, 2, TRANSITION_FIELD)))}, "fields"),
        ({"FieldsBlob": envelope(pb((1, 2, pb((3, 0, 1))), (2, 2, b"x")))}, "#1 holds"),
        ({"FieldsBlob": b"\x00\x00\x00\x02\x00\x00"}, "decode"),
        ({"FieldsBlob": None}, "decode"),
        ({"Unknown": 1}, "does not know"),
    ],
)
def test_a_template_outside_the_field_map_is_refused(change: dict[str, Any], words: str) -> None:
    template = _template()
    template.item.update(change)
    with pytest.raises(ItemError) as info:
        author_transition(template, _place(), transition_id=NEW)
    assert info.value.code == "bad_template"
    assert words in info.value.message, info.value.message


def test_a_picture_transition_is_never_a_template() -> None:
    shape = transition_shape(_template(), 0)
    assert shape.kind == "picture" and "Smooth Cut" in str(shape.refused)
    assert transition_shape(_template(), 1).refused is None
    assert transition_shape(_template(), None).refused is not None


def test_a_template_whose_join_does_not_place_it_is_refused() -> None:
    template = _template()
    template.link["DbOwner"] = uid(0xFFFF)
    with pytest.raises(ItemError, match="own track"):
        check_transition(template)


def _flip(data: bytes, rng: random.Random) -> bytes:
    if not data:
        return data
    at = rng.randrange(len(data))
    return data[:at] + bytes([data[at] ^ (1 << rng.randrange(8))]) + data[at + 1 :]


@pytest.mark.parametrize("seed_number", range(4))
def test_fuzz_a_damaged_template_is_a_typed_refusal_or_a_valid_row(seed_number: int) -> None:
    """Aimed at the fixed-layout parts: the envelope's version and length, the qCompress
    length prefix, the zlib stream and the protobuf tags; and at random columns, constants
    among them. Each is a typed refusal or the row the untouched template gives, every column
    and the join row equal (DESIGN "Transitions and the latest save (S25e)")."""
    rng = random.Random(seed_number)
    good = envelope(pb((2, 2, TRANSITION_FIELD)))
    reference = author_transition(_template(), _place(), transition_id=NEW)
    junk = [None, 0, -1, "x", b"\xff", 2.5, 10**30, "3|0000000000000000", "", 1, True, 0.0]
    for _ in range(300):
        template = _template()
        target = rng.choice(["envelope", "prefix", "zlib", "tag", "column", "constant", "any"])
        blob = bytearray(good)
        if target == "envelope":
            blob[rng.randrange(9)] ^= 1 << rng.randrange(8)
        elif target == "prefix":
            blob[11 + rng.randrange(4)] ^= 1 << rng.randrange(8)
        elif target == "zlib":
            blob[15 + rng.randrange(len(good) - 15)] ^= 1 << rng.randrange(8)
        elif target == "tag":
            blob[9 + rng.randrange(2)] ^= 1 << rng.randrange(8)
        elif target == "any":
            blob = bytearray(_flip(good, rng))
        template.item["FieldsBlob"] = bytes(blob)
        if target in ("column", "constant"):
            columns = ITEM_COLUMNS if target == "column" else tuple(TRANSITION_CONSTANTS)
            template.item[rng.choice(list(columns))] = rng.choice(junk)
        try:
            authored = author_transition(template, _place(), transition_id=NEW)
        except ItemError:
            continue
        assert authored.item["FieldsBlob"] == good
        assert authored.item == reference.item and authored.link == reference.link


# -- joins in a cut ---------------------------------------------------------------------------


def _seed(tmp: Path, items: Any = None) -> Path:
    return write_seed(
        tmp / "seed.db",
        rate=float(RATE),
        start_frame=START,
        rate_text="29.97",
        resolution=(1920, 1080),
        clips=[CLIP],
        items=crossfade(CLIP, 0xB000, start=START) if items is None else items,
    )


def _make(tmp: Path, cuts_of: Any, transitions: Any, name: str = "Cut") -> Any:
    seed = tmp / "seed.db"
    if not seed.exists():
        _seed(tmp)
    picks = pick_item_templates([read_items(seed, options=OPTS)])

    def author(s: Any) -> Authored:
        cuts = cuts_of(s.clips[0])
        items, specs = author_cut(s, cuts, picks, transitions=transitions)
        return Authored([], items, specs, cuts=cuts, transitions=transitions)

    library = tmp / "lib"
    if not library.exists():
        make_library(library)
    return make_project(
        library,
        seed,
        project_name=name,
        timeline_name=None,
        author=author,
        allowed_issues=READER_WARNING_CODES,
        options=OPTS,
        allow_unopened=True,
    )


def three(clip: Any) -> list[Cut]:
    return [Cut(clip, 0, 24, 10), Cut(clip, 24, 24, 100), Cut(clip, 48, 30, 200)]


def test_crossfades_on_two_joins_are_written_verified_and_read_back(tmp_path: Path) -> None:
    tx = [CutTransition(0, 2, "Cross Fade 0 dB"), CutTransition(1, 4, "Cross Fade +3 dB")]
    result = _make(tmp_path, three, tx)
    assert "add_sound_transitions" in result.built.operations
    conn = sqlite3.connect(result.placed.path)
    rows = conn.execute(
        "SELECT i.Start, i.Duration, i.PrettyType, j.DbIndex FROM Sm2TiItem i JOIN "
        "Sm2TiItem_Sm2TiTrack j ON j.DbAssociate = i.Sm2TiItem_id "
        "WHERE i.DbType = 'Sm2TiTransition' ORDER BY j.DbIndex"
    ).fetchall()
    conn.close()
    # Sound track: clip 0, transition, clip 1, transition, clip 2.
    assert rows == [
        (str(START + 23), "2", "Cross Fade 0DB", 1),
        (str(START + 46), "4", "Cross Fade +3DB", 3),
    ]
    timeline = read_timeline(result.placed.path, library_options=OPTS)
    found = [
        (i.kind, i.record_start, i.duration) for t in timeline.timeline.tracks for i in t.items
    ]
    assert ("transition", START + 23, 2) in found and ("transition", START + 46, 4) in found


@pytest.mark.parametrize(
    ("cuts_of", "tx", "words"),
    [
        # longer than a neighbour: 16 frames over a 6-frame clip
        (
            lambda c: [Cut(c, 0, 6, 50), Cut(c, 6, 24, 100)],
            [CutTransition(0, 16, "Cross Fade 0 dB")],
            "longer than a clip",
        ),
        # longer than the handles: the incoming clip starts 1 frame into its file
        (
            lambda c: [Cut(c, 0, 24, 50), Cut(c, 24, 24, 1)],
            [CutTransition(0, 4, "Cross Fade 0 dB")],
            "ahead of the incoming",
        ),
        # the outgoing clip ends at the end of its file
        (
            lambda c: [Cut(c, 0, 24, 216), Cut(c, 24, 24, 10)],
            [CutTransition(0, 2, "Cross Fade 0 dB")],
            "past the outgoing",
        ),
        # a gap: not a join
        (
            lambda c: [Cut(c, 0, 24, 50), Cut(c, 30, 24, 100)],
            [CutTransition(0, 2, "Cross Fade 0 dB")],
            "not a join",
        ),
        # nothing follows
        (
            lambda c: [Cut(c, 0, 24, 50)],
            [CutTransition(0, 2, "Cross Fade 0 dB")],
            "not a join",
        ),
        # a split edit
        (
            lambda c: [Cut(c, 0, 24, 50, sound_src_in=60), Cut(c, 24, 24, 100)],
            [CutTransition(0, 2, "Cross Fade 0 dB")],
            "split edit",
        ),
        # two on one join
        (
            three,
            [CutTransition(0, 2, "Cross Fade 0 dB"), CutTransition(0, 2, "Cross Fade 0 dB")],
            "two transitions",
        ),
        (three, [CutTransition(7, 2, "Cross Fade 0 dB")], "does not exist"),
        (three, [CutTransition(-1, 2, "Cross Fade 0 dB")], "does not exist"),
        (three, [CutTransition(0, -4, "Cross Fade 0 dB")], "frames long"),
        (three, [CutTransition(0, 2, "Cross Fade 0 dB", "sideways")], "not known"),
    ],
)
def test_a_crossfade_that_is_not_on_a_join_with_its_media_is_refused(
    tmp_path: Path, cuts_of: Any, tx: Any, words: str
) -> None:
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, cuts_of, tx)
    assert info.value.code == "bad_transition", info.value
    assert words in info.value.message
    projects = tmp_path / "lib" / "Resolve Projects" / "Users" / "guest" / "Projects"
    assert list(projects.iterdir()) == []


@pytest.mark.parametrize(
    "tx",
    [
        CutTransition(0, 3, "Cross Fade 0 dB", "start"),
        CutTransition(0, 20, "Cross Fade 0 dB"),
        CutTransition(0, 2, "Cross Fade -3 dB"),
    ],
)
def test_an_unsupported_kind_alignment_or_length_is_refused_on_the_join(
    tmp_path: Path, tx: CutTransition
) -> None:
    with pytest.raises((ProjectWriteError, ItemError, MediaPoolError)) as info:
        _make(tmp_path, three, [tx])
    assert getattr(info.value, "code", "") in ("unsupported_item", "item_refused", "bad_input")


def test_a_timeline_of_another_rate_is_refused(tmp_path: Path) -> None:
    write_seed(
        tmp_path / "seed.db",
        rate=24000 / 1001,
        start_frame=86400,
        rate_text="23.976",
        clips=[CLIP],
        items=crossfade(CLIP, 0xB000, start=86400),
    )
    with pytest.raises((ProjectWriteError, MediaPoolError)) as info:
        _make(tmp_path, three, [CutTransition(0, 2, "Cross Fade 0 dB")])
    assert "29.97" in str(info.value)


def test_no_sound_transition_to_copy_is_no_template(tmp_path: Path) -> None:
    _seed(tmp_path, items=linked_pair(CLIP, 0xB000, start=START, duration=48, src_in=0))
    with pytest.raises((ProjectWriteError, MediaPoolError)) as info:
        _make(tmp_path, three, [CutTransition(0, 2, "Cross Fade 0 dB")])
    assert "no sound transition" in str(info.value)


# -- the seed ---------------------------------------------------------------------------------


def test_a_seed_whose_timeline_holds_a_plain_crossfade_is_cleared(tmp_path: Path) -> None:
    result = _make(tmp_path, three, [])
    assert "remove_items" in result.built.operations
    conn = sqlite3.connect(result.placed.path)
    kinds = [r[0] for r in conn.execute("SELECT DbType FROM Sm2TiItem")]
    conn.close()
    assert "Sm2TiTransition" not in kinds and len(kinds) == 6


def test_a_seed_whose_crossfade_carries_later_edits_is_refused(tmp_path: Path) -> None:
    items = crossfade(CLIP, 0xB000, start=START)
    row, _ = items[3]
    row["FieldsBlob"] = envelope(pb((1, 2, pb((21, 0, 5))), (2, 2, TRANSITION_FIELD)))
    _seed(tmp_path, items=items)
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, three, [])
    assert info.value.code == "bad_seed" and "history" in info.value.message


# -- the verifier on its own -------------------------------------------------------------------


def _written(tmp_path: Path) -> tuple[Path, Any, list[Cut], list[CutTransition]]:
    """A project with two crossfades, written and verified, then reopened for planting."""
    tx = [CutTransition(0, 2, "Cross Fade 0 dB"), CutTransition(1, 4, "Cross Fade +3 dB")]
    result = _make(tmp_path, three, tx)
    seed = result.seed
    cuts = list(result.authored.cuts)
    return result.placed.path, seed, cuts, tx


def _plant(path: Path, sql: str, *args: Any) -> None:
    conn = sqlite3.connect(path)
    conn.execute(sql, args)
    conn.commit()
    conn.close()


def _verify(path: Path, seed: Any, cuts: Any, tx: Any) -> None:
    verify_written(path, seed, cuts, transitions=tx, project_name="Cut", timeline_name=None)


def _other_value(value: object) -> object:
    if value is None:
        return 1
    if isinstance(value, (bool, int)):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, bytes):
        return value + b"\x00"
    return f"{value}x"


@pytest.mark.parametrize("column", sorted(ITEM_COLUMNS))
def test_every_column_of_a_transition_row_is_checked(tmp_path: Path, column: str) -> None:
    path, seed, cuts, tx = _written(tmp_path)
    _verify(path, seed, cuts, tx)  # as written: passes
    conn = sqlite3.connect(path)
    [(tid, value)] = conn.execute(
        f'SELECT Sm2TiItem_id, "{column}" FROM Sm2TiItem WHERE DbType = '
        "'Sm2TiTransition' ORDER BY CAST(Start AS INTEGER) LIMIT 1"
    ).fetchall()
    conn.close()
    if column == "Sm2TiItem_id":
        # an id of the seed's, or of another item, is not a fresh one
        value = seed.project_id
    _plant(
        path,
        f'UPDATE Sm2TiItem SET "{column}" = ? WHERE Sm2TiItem_id = ?',
        _other_value(value) if column != "Sm2TiItem_id" else value,
        tid,
    )
    if column == "Sm2TiItem_id":
        _plant(
            path,
            "UPDATE Sm2TiItem_Sm2TiTrack SET DbAssociate = ? WHERE DbAssociate = ?",
            value,
            tid,
        )
    with pytest.raises(ProjectWriteError) as info:
        _verify(path, seed, cuts, tx)
    assert info.value.code == "self_check_failed"


@pytest.mark.parametrize(
    ("sql", "words"),
    [
        (
            "UPDATE Sm2TiItem_Sm2TiTrack SET DbIndex = 9 WHERE DbIndex = 1 AND DbOwner = ?",
            "DbIndex",
        ),
        (
            "DELETE FROM Sm2TiItem WHERE Sm2TiItem_id IN (SELECT DbAssociate FROM "
            "Sm2TiItem_Sm2TiTrack WHERE DbIndex = 1 AND DbOwner = ?)",
            "items",
        ),
    ],
)
def test_a_transition_out_of_place_or_gone_is_refused(tmp_path: Path, sql: str, words: str) -> None:
    path, seed, cuts, tx = _written(tmp_path)
    _plant(path, sql, seed.audio_track)
    with pytest.raises(ProjectWriteError) as info:
        _verify(path, seed, cuts, tx)
    assert info.value.code == "self_check_failed"


def test_the_verifier_recomputes_the_join_and_the_media(tmp_path: Path) -> None:
    path, seed, cuts, tx = _written(tmp_path)
    # The cut the verifier is given asks for another crossfade length: refused.
    with pytest.raises(ProjectWriteError):
        _verify(path, seed, cuts, [tx[0], CutTransition(1, 2, "Cross Fade +3 dB")])
    # ... another kind
    with pytest.raises(ProjectWriteError):
        _verify(path, seed, cuts, [CutTransition(0, 2, "Cross Fade +3 dB"), tx[1]])
    # ... a transition the file does not hold
    with pytest.raises(ProjectWriteError):
        _verify(path, seed, cuts, [tx[0]])
    # ... a FieldsBlob #2 of another value, framed alike
    other = envelope(pb((2, 2, qcompress(bytes([3, 0, 2]) + bytes(41)))))
    _plant(
        path,
        "UPDATE Sm2TiItem SET FieldsBlob = ? WHERE DbType = 'Sm2TiTransition'",
        other,
    )
    with pytest.raises(ProjectWriteError, match="#2"):
        _verify(path, seed, cuts, tx)


def test_the_verifier_refuses_a_crossfade_without_its_media(tmp_path: Path) -> None:
    """The cut the file was written from read further than the clip has (a writer bug the
    verifier must catch on its own): a crossfade past the outgoing clip's sound."""
    path, seed, cuts, _tx = _written(tmp_path)
    short = [cuts[0], cuts[1], cuts[2]]
    bad = [CutTransition(0, 2, "Cross Fade 0 dB"), CutTransition(1, 16, "Cross Fade +3 dB")]
    with pytest.raises(ProjectWriteError):
        _verify(path, seed, short, bad)


def test_the_fields_blob_rule_is_byte_for_byte_on_its_framing(tmp_path: Path) -> None:
    path, seed, cuts, tx = _written(tmp_path)
    good = envelope(pb((2, 2, TRANSITION_FIELD)))
    zstd_framed = envelope(pb((2, 2, TRANSITION_FIELD)), zstd=True)
    version_one = struct.pack(">I", 1) + good[4:]
    for planted in (zstd_framed, version_one):
        _plant(
            path, "UPDATE Sm2TiItem SET FieldsBlob = ? WHERE DbType = 'Sm2TiTransition'", planted
        )
        with pytest.raises(ProjectWriteError):
            _verify(path, seed, cuts, tx)
    _plant(path, "UPDATE Sm2TiItem SET FieldsBlob = ? WHERE DbType = 'Sm2TiTransition'", good)
    _verify(path, seed, cuts, tx)


# -- S25e-f2: one rule at every door, early template refusals, the mutants that survived ------


def _overlapping(clip: Any) -> list[Cut]:
    """A 10-frame clip between two others: two 16-frame crossfades put 8 frames each in it."""
    return [Cut(clip, 0, 24, 50), Cut(clip, 24, 10, 100), Cut(clip, 34, 24, 150)]


OVERLAPPING = [CutTransition(0, 16, "Cross Fade 0 dB"), CutTransition(1, 16, "Cross Fade 0 dB")]


def test_two_crossfades_overlapping_inside_one_clip_are_refused_by_the_writer(
    tmp_path: Path,
) -> None:
    """Each side fits its own clip (8 of 24, 8 of 10, 8 of 10, 8 of 24), but the middle clip
    would hold 16 frames of crossfade in 10: the plan export's ``transitions_overlap``."""
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, _overlapping, OVERLAPPING)
    assert info.value.code == "transitions_overlap", info.value
    # The writer's own refusal, before anything is written (the verifier's names no transition).
    assert info.value.message.startswith(
        "cut entry 1: transition 0 and transition 1 need 16 frames of its 10"
    ), info.value.message
    projects = tmp_path / "lib" / "Resolve Projects" / "Users" / "guest" / "Projects"
    assert list(projects.iterdir()) == []


def test_two_crossfades_that_exactly_fill_a_clip_are_placed(tmp_path: Path) -> None:
    def exact(clip: Any) -> list[Cut]:
        return [Cut(clip, 0, 24, 50), Cut(clip, 24, 16, 100), Cut(clip, 40, 24, 150)]

    result = _make(tmp_path, exact, OVERLAPPING)
    assert "add_sound_transitions" in result.built.operations


def test_the_verifier_refuses_overlapping_crossfades_with_the_same_code(tmp_path: Path) -> None:
    path, seed, cuts, _ = _written(tmp_path)
    from aar.drp.libwrite.verify import expected_transitions

    clip = cuts[0].clip
    with pytest.raises(ProjectWriteError) as info:
        expected_transitions(_overlapping(clip), OVERLAPPING)
    assert info.value.code == "transitions_overlap"
    # ... and the whole check, before any row is compared
    with pytest.raises(ProjectWriteError) as info:
        _verify(path, seed, _overlapping(clip), OVERLAPPING)
    assert info.value.code == "transitions_overlap"


@pytest.mark.parametrize("column", sorted(TRANSITION_CONSTANTS))
def test_a_template_with_a_damaged_constant_is_refused_naming_the_column(column: str) -> None:
    """The validation's fuzz found rows copied with, for example, ``RenderTextEnabled``
    ``3|0000000000000000``: the verifier refused them at placement as a writer bug. The
    template is refused first, ``bad_template`` naming the column."""
    template = _template()
    # Another type than the constant's: a number where it is empty, else Resolve's N|hex text.
    template.item[column] = 7 if TRANSITION_CONSTANTS[column] is None else "3|0000000000000000"
    with pytest.raises(ItemError) as info:
        author_transition(template, _place(), transition_id=NEW)
    assert info.value.code == "bad_template"
    assert column in info.value.message, info.value.message


@pytest.mark.parametrize("value", [0, 2, 1.0, True, b"\x01", "1", "3|0000000000000000", None])
def test_a_constant_of_the_wrong_type_is_refused(value: object) -> None:
    template = _template()
    template.item["RenderTextEnabled"] = value
    with pytest.raises(ItemError, match="RenderTextEnabled"):
        author_transition(template, _place(), transition_id=NEW)


def test_the_writer_and_the_verifier_state_the_same_transition_constants() -> None:
    stated = {c: r[1] for c, r in TRANSITION_RULES.items() if r != DERIVED}  # type: ignore[index]
    copied = set(ITEM_COLUMNS) - {
        "Sm2TiItem_id",
        "DbType",
        "Start",
        "Duration",
        "Sm2TiTrack_id",
        "AlignmentType",
        "Position",
        "PrettyType",
        "FieldsBlob",
    }
    assert set(TRANSITION_CONSTANTS) == set(stated) == copied
    for column, value in TRANSITION_CONSTANTS.items():
        assert value == stated[column] and type(value) is type(stated[column]), column


def test_an_odd_crossfade_is_written_and_checked_by_the_verifiers_own_centring(
    tmp_path: Path,
) -> None:
    """A 3-frame crossfade: Resolve puts two frames before the cut. Written through
    ``make_project`` (so the verifier's own centring rule passes it), then planted one frame
    later (where ``frames // 2`` would put it): refused."""
    result = _make(tmp_path, three, [CutTransition(0, 3, "Cross Fade 0 dB")])
    conn = sqlite3.connect(result.placed.path)
    [(start, duration)] = conn.execute(
        "SELECT Start, Duration FROM Sm2TiItem WHERE DbType = 'Sm2TiTransition'"
    ).fetchall()
    conn.close()
    assert (start, duration) == (str(START + 24 - 2), "3")
    cuts = list(result.authored.cuts)
    tx = [CutTransition(0, 3, "Cross Fade 0 dB")]
    _verify(result.placed.path, result.seed, cuts, tx)
    _plant(
        result.placed.path,
        "UPDATE Sm2TiItem SET Start = ? WHERE DbType = 'Sm2TiTransition'",
        str(START + 24 - 1),
    )
    with pytest.raises(ProjectWriteError) as info:
        _verify(result.placed.path, result.seed, cuts, tx)
    assert info.value.code == "self_check_failed"


def test_the_verifier_checks_the_incoming_clips_handle_on_its_own(tmp_path: Path) -> None:
    """The incoming clip starts one frame into its file: a 2-frame crossfade (one frame before
    the cut) is written; the file then planted to hold a 4-frame one (two frames before the
    cut, one more than the file has) is refused by the verifier's own handle rule."""

    def late(clip: Any) -> list[Cut]:
        return [Cut(clip, 0, 24, 10), Cut(clip, 24, 24, 1), Cut(clip, 48, 30, 200)]

    result = _make(tmp_path, late, [CutTransition(0, 2, "Cross Fade 0 dB")])
    path, seed, cuts = result.placed.path, result.seed, list(result.authored.cuts)
    _plant(
        path,
        "UPDATE Sm2TiItem SET Start = ?, Duration = '4' WHERE DbType = 'Sm2TiTransition'",
        str(START + 24 - 2),
    )
    with pytest.raises(ProjectWriteError) as info:
        _verify(path, seed, cuts, [CutTransition(0, 4, "Cross Fade 0 dB")])
    assert info.value.code == "self_check_failed"
    assert "before the incoming clip's first frame" in info.value.message


def test_the_verifier_wants_the_stored_bytes_not_only_their_value(tmp_path: Path) -> None:
    """A ``FieldsBlob`` #2 of another zlib stream that inflates to the same 44 bytes (stored,
    not deflated) is not what Resolve writes: refused byte for byte."""
    import zlib

    path, seed, cuts, tx = _written(tmp_path)
    other = struct.pack(">I", len(TRANSITION_PAYLOAD)) + zlib.compress(TRANSITION_PAYLOAD, 0)
    assert other != TRANSITION_FIELD and zlib.decompress(other[4:]) == TRANSITION_PAYLOAD
    _plant(
        path,
        "UPDATE Sm2TiItem SET FieldsBlob = ? WHERE DbType = 'Sm2TiTransition'",
        envelope(pb((2, 2, other))),
    )
    with pytest.raises(ProjectWriteError, match="#2"):
        _verify(path, seed, cuts, tx)
