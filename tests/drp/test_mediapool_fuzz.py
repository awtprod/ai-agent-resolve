"""S25c1: randomly damaged facts and template rows give a typed refusal or a valid result.

Facts and templates are untrusted, so whatever is wrong with them must come out as a
:class:`MediaPoolError` (a code and a hint), never a ``ValueError``, ``TypeError``,
``IndexError``, ``KeyError``, ``EncodeError`` or any other exception. Each case starts from a
synthetic file and template (``mediapool_support``) and damages one to three things: a value of
the facts (a hostile number or text, a deleted or added key, a duplicated stream), a column of a
template row (NULL, another type, a deleted column, damaged bytes), or a value deep inside a
template blob (a keyed-dict entry's type, null flag, value or order; a protobuf field's number,
wire type, value or order; a frame's bytes). The damage is seeded, so a failure names its seed.

A second, aimed run damages one byte inside a fixed-layout value at any layer of a template blob
(a resolution, a frame rate, audio extents, a timemap, a fixed64 or fixed32 protobuf field), the
values the shared decoders read by position: a non-zero padding word there once escaped as an
untyped ``BlobError`` (one case in about 18,000 of the unaimed damage found it).
"""

from __future__ import annotations

import contextlib
import copy
import json
import random
from dataclasses import replace
from fractions import Fraction
from typing import Any

import mediapool_support as ms
import pytest

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.mediapool import AuthoredClip, Placement, Template, author, facts_from_json
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.kinds import kind_of_rows
from aar.drp.mediapool.templates import ClipRows, clips_from_rows
from aar.drp.modify import encode

#: Cases per test run; ``run_case`` is also driven at a far larger scale from outside the suite.
CASES = 600

HOSTILE: tuple[Any, ...] = (
    None,
    True,
    False,
    0,
    -1,
    1,
    2**31,
    2**61,
    2**63,
    2**70,
    -(2**70),
    1.5,
    -0.0,
    1e308,
    "",
    " ",
    "0",
    "-1",
    "1_0",
    "\u0660\u0661:\u0660\u0660:\u0660\u0660:\u0660\u0660",
    "01:00:00:00",
    "99:99:99:99",
    "0/0",
    "1/0",
    "-1/1",
    "99999999999/1",
    "30000/1001",
    "0000-01-01T00:00:00Z",
    "2026-13-45T25:61:61Z",
    "x" * 5000,
    "a\u202eb",
    "X:\\a\\\\b.wav",
    "X:\\b.wav",
    "X:\\a\\CON.wav",
    "X:\\a\\b:c.wav",
    [],
    {},
    [1, [2, [3]]],
    {"rotation": "90"},
)
PLACE = Placement(ms.FOLDER, 4, Fraction(30000, 1001), 1_795_000_000, ms.MAP, 99)
_SCALAR_TYPES = (
    blobs.KD_BOOL,
    blobs.KD_INT32,
    blobs.KD_UINT32,
    blobs.KD_INT64,
    blobs.KD_UINT64,
    blobs.KD_DOUBLE,
    blobs.KD_STRING,
    blobs.KD_BYTES,
)


# --------------------------------------------------------------------------- damaging facts


def _hostile(rng: random.Random) -> Any:
    return copy.deepcopy(rng.choice(HOSTILE))  # a fresh copy: a shared list could hold itself


def _containers(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Every path to a dict or list inside ``value``."""
    out: list[tuple[Any, ...]] = []
    if isinstance(value, dict | list):
        out.append(path)
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            out += _containers(item, (*path, key))
    return out


def _at(value: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


def damage_facts(rng: random.Random, data: dict[str, Any]) -> None:
    target = _at(data, rng.choice(_containers(data)))
    op = rng.randrange(4)
    if isinstance(target, list):
        if target and op == 0:
            del target[rng.randrange(len(target))]
        elif target and op == 1:
            target.append(copy.deepcopy(rng.choice(target)))
        else:
            target.append(_hostile(rng))
        return
    if not target or op == 0:
        key = rng.choice(["index", "tags", "timecode", "rotate", "zzz", "codec_type"])
        target[key] = _hostile(rng)
    elif op == 1:
        del target[rng.choice(list(target))]
    else:
        target[rng.choice(list(target))] = _hostile(rng)


# --------------------------------------------------------------------------- damaging blobs


def _bytes_damage(rng: random.Random, data: bytes) -> bytes:
    if not data:
        return bytes([rng.randrange(256)])
    op = rng.randrange(4)
    at = rng.randrange(len(data))
    if op == 0:
        return data[:at] + bytes([data[at] ^ (1 << rng.randrange(8))]) + data[at + 1 :]
    if op == 1:
        return data[:at]
    if op == 2:
        return data + bytes(rng.randrange(256) for _ in range(rng.randrange(1, 9)))
    return data[:at] + data[at:][::-1]


def _hostile_for(rng: random.Random, vtype: int) -> Any:
    if vtype == blobs.KD_STRING:
        texts = ["", "x" * 300, "\u0660\u0661", "01:00:00:00", ms.uid(rng.randrange(9999))]
        return rng.choice(texts)
    if vtype == blobs.KD_BYTES:
        return rng.choice([b"", b"\x00" * 16, bytes(range(40))])
    if vtype == blobs.KD_BOOL:
        return rng.choice([True, False])
    if vtype == blobs.KD_DOUBLE:
        return rng.choice([0.0, -1.0, 1e308, float("nan"), float("inf")])
    limits = {
        blobs.KD_INT32: (-(2**31), 2**31 - 1),
        blobs.KD_UINT32: (0, 2**32 - 1),
        blobs.KD_INT64: (-(2**63), 2**63 - 1),
        blobs.KD_UINT64: (0, 2**64 - 1),
    }
    low, high = limits[vtype]
    return rng.choice([low, high, 0, 1, -1 if low < 0 else 2])


def _kd_damage(rng: random.Random, entries: list[list[Any]], depth: int) -> None:
    if not entries:
        entries.append(["Extra", blobs.KD_INT32, 0, 1])
        return
    i = rng.randrange(len(entries))
    op = rng.randrange(8)
    if op == 0:
        vtype = rng.choice(_SCALAR_TYPES)
        entries[i] = [entries[i][0], vtype, entries[i][2], _hostile_for(rng, vtype)]
    elif op == 1:
        entries[i][2] = 1
    elif op == 2:
        entries[i][3] = _hostile_for(rng, entries[i][1]) if entries[i][1] in _SCALAR_TYPES else None
    elif op == 3:
        del entries[i]
    elif op == 4:
        entries.append(list(entries[i]))
    elif op == 5:
        entries.reverse()
    elif op == 6:
        entries[i][0] = rng.choice(["Surprise", "0", "UniqueId", "a.b", entries[0][0]])
    elif isinstance(entries[i][3], bytes) and entries[i][1] == blobs.KD_BYTES:
        entries[i][3] = damage_blob(rng, entries[i][3], depth + 1)
    else:
        entries[i][3] = _hostile_for(rng, entries[i][1]) if entries[i][1] in _SCALAR_TYPES else None


def _pb_damage(rng: random.Random, fields: list[list[Any]], depth: int) -> None:
    if not fields:
        fields.append([1, 0, 1])
        return
    i = rng.randrange(len(fields))
    op = rng.randrange(7)
    if op == 0:
        fields[i][0] = rng.choice([1, 2, 3, 7, 13, 99])
    elif op == 1:
        fields[i][1], fields[i][2] = rng.choice([(0, 2**64 - 1), (0, 0), (2, b""), (1, bytes(8))])
    elif op == 2:
        del fields[i]
    elif op == 3:
        fields.append(list(fields[i]))
    elif op == 4:
        fields.reverse()
    elif fields[i][1] == 2:
        fields[i][2] = damage_blob(rng, fields[i][2], depth + 1)
    else:
        fields[i][2] = rng.choice([0, 1, 2**63]) if fields[i][1] == 0 else fields[i][2]


def damage_blob(rng: random.Random, data: bytes, depth: int = 0) -> bytes:
    """``data`` damaged at a random layer: a keyed dict or an enveloped message is decoded and
    one entry or field changed (or a layer inside it damaged), then encoded again; anything
    else, or one time in four, has its bytes damaged."""
    if depth > 4 or rng.randrange(4) == 0:
        return _bytes_damage(rng, data)
    try:
        kd = blobs.decode_keyed_dict(data, strict=True)
    except DrpError:
        kd = None
    if kd is not None and data[:4] == b"\x00\x00\x00\x01":
        entries = [[e.key, e.type, e.subtype, e.value] for e in kd.entries]
        _kd_damage(rng, entries, depth)
        try:
            return encode.keyed_dict([tuple(e) for e in entries])  # type: ignore[misc]
        except (DrpError, ValueError):
            return _bytes_damage(rng, data)
    env = blobs.parse_envelope(data)
    if env is not None:
        try:
            payload = blobs.unwrap_message(data)
            fields = [
                [f.number, f.wire, f.value] for f in blobs.decode_protobuf(payload, nested_depth=0)
            ]
            _pb_damage(rng, fields, depth)
            return encode.envelope(data, encode.protobuf([tuple(f) for f in fields]))  # type: ignore[misc]
        except (DrpError, ValueError):
            return _bytes_damage(rng, data)
    return _bytes_damage(rng, data)


# --------------------------------------------------------------------------- aimed: fixed layouts

#: Keyed-dict entries whose value has a fixed byte layout, read by position in ``aar.drp.blobs``
#: (``decode_resolution``, ``decode_rate``, ``decode_extents``, ``decode_timemap``).
FIXED_KEYS = frozenset(
    {"Resolution", "FrameRate", "MediaFrameRate", "MediaExtents", "MediaTimemapBA"}
)


def _one_byte(rng: random.Random, value: bytes) -> bytes:
    """``value`` with one byte changed and its length kept (a padding word, a sign, an exponent)."""
    out = bytearray(value)
    at = rng.randrange(len(out))
    out[at] = rng.choice(
        [v for v in (0x00, 0x01, 0x7F, 0x80, 0xFF, rng.randrange(256)) if v != out[at]]
    )
    return bytes(out)


def _as_message(data: bytes) -> list[list[Any]] | None:
    """``data`` as a canonical protobuf message without an envelope, or None."""
    try:
        fields = [[f.number, f.wire, f.value] for f in blobs.decode_protobuf(data, nested_depth=0)]
        same = encode.protobuf([tuple(f) for f in fields]) == data  # type: ignore[misc]
    except (DrpError, ValueError):
        return None
    return fields if fields and same else None


def damage_fixed(rng: random.Random, data: bytes, depth: int = 0) -> bytes | None:
    """``data`` with one byte changed inside one fixed-layout value at any layer (an entry of
    :data:`FIXED_KEYS`, a fixed64 or fixed32 protobuf field), every layer around it encoded
    again; None if ``data`` holds no such value."""
    if depth > 6 or not data:
        return None
    try:
        kd = blobs.decode_keyed_dict(data, strict=True) if data[:4] == b"\x00\x00\x00\x01" else None
    except DrpError:
        kd = None
    if kd is not None:
        entries = [[e.key, e.type, e.subtype, e.value] for e in kd.entries]
        order = [i for i, e in enumerate(entries) if isinstance(e[3], bytes) and e[3]]
        rng.shuffle(order)
        for i in order:
            if entries[i][0] in FIXED_KEYS:
                entries[i][3] = _one_byte(rng, entries[i][3])
            else:
                inner = damage_fixed(rng, entries[i][3], depth + 1)
                if inner is None:
                    continue
                entries[i][3] = inner
            return encode.keyed_dict([tuple(e) for e in entries])  # type: ignore[misc]
        return None
    wrapped = blobs.parse_envelope(data) is not None
    if wrapped:
        try:
            fields = _as_message(blobs.unwrap_message(data))
        except DrpError:
            fields = None
    else:
        fields = _as_message(data)
    if fields is None:
        return None
    order = list(range(len(fields)))
    rng.shuffle(order)
    for i in order:
        _number, wire, value = fields[i]
        if wire in (1, 5):
            fields[i][2] = _one_byte(rng, value)
        elif wire == 2 and isinstance(value, bytes):
            inner = damage_fixed(rng, value, depth + 1)
            if inner is None:
                continue
            fields[i][2] = inner
        else:
            continue
        payload = encode.protobuf([tuple(f) for f in fields])  # type: ignore[misc]
        return encode.envelope(data, payload) if wrapped else payload
    return None


def damage_fixed_rows(rng: random.Random, clip: ClipRows) -> ClipRows | None:
    """``clip`` with one fixed-layout value inside one of its blobs damaged; None if none
    has one."""
    rows: dict[str, list[Any]] = {
        "media": [dict(clip.media)],
        "video": [dict(clip.video)] if clip.video is not None else [],
        "audio": [dict(r) for r in clip.audio],
        "audio_links": [dict(r) for r in clip.audio_links],
        "folder_link": [dict(clip.folder_link)],
        "lockables": [dict(r) for r in clip.lockables],
        "presets": [dict(r) for r in clip.presets],
    }
    cells = [
        (row, column)
        for group in rows.values()
        for row in group
        for column, value in row.items()
        if isinstance(value, bytes)
    ]
    rng.shuffle(cells)
    for row, column in cells:
        damaged = damage_fixed(rng, row[column])
        if damaged is not None:
            row[column] = damaged
            return _clip_of(rows, clip)
    return None


# --------------------------------------------------------------------------- damaging rows


def damage_rows(rng: random.Random, clip: ClipRows) -> ClipRows:
    rows = {
        "media": [clip.media],
        "video": [clip.video] if clip.video is not None else [],
        "audio": list(clip.audio),
        "audio_links": list(clip.audio_links),
        "folder_link": [clip.folder_link],
        "lockables": list(clip.lockables),
        "presets": list(clip.presets),
    }
    rows = copy.deepcopy(rows)
    group = rng.choice([g for g, r in rows.items() if r] + ["audio", "lockables"])
    if not rows[group] or rng.randrange(6) == 0:
        op = rng.randrange(2)
        if op == 0 and rows[group]:
            del rows[group][rng.randrange(len(rows[group]))]
        elif rows[group]:
            rows[group].append(copy.deepcopy(rng.choice(rows[group])))
        return _clip_of(rows, clip)
    row = rng.choice(rows[group])
    if not row:
        return _clip_of(rows, clip)
    column = rng.choice(list(row))
    op = rng.randrange(5)
    if op == 0:
        row[column] = None
    elif op == 1:
        del row[column]
    elif op == 2:
        row[column] = rng.choice([0, -1, 2**63, 1.5, "", "x", ms.uid(7), b"", b"\x80"])
    elif isinstance(row[column], bytes):
        row[column] = damage_blob(rng, row[column])
    else:
        row[column] = rng.choice([[], {}, None, 3, ms.uid(8)])
    return _clip_of(rows, clip)


def _clip_of(rows: dict[str, list[Any]], clip: ClipRows) -> ClipRows:
    return replace(
        clip,
        media=rows["media"][0] if rows["media"] else {},
        video=rows["video"][0] if rows["video"] else None,
        audio=tuple(rows["audio"]),
        audio_links=tuple(rows["audio_links"]),
        folder_link=rows["folder_link"][0] if rows["folder_link"] else {},
        lockables=tuple(rows["lockables"]),
        presets=tuple(rows["presets"]),
    )


# --------------------------------------------------------------------------- one case


_BASES = (
    (ms.wav_facts, ms.wav_template, ms.wav_template_facts),
    (ms.mp4_facts, ms.mp4_template, ms.mp4_template_facts),
    (ms.png_facts, ms.png_template, ms.png_template_facts),
)


def run_case(seed: int) -> str:
    """One damaged case: returns ``"ok"`` or ``"refused:<code>"``; raises anything else."""
    rng = random.Random(seed)
    make_facts, make_template, make_template_facts = rng.choice(_BASES)
    facts = make_facts()
    template = make_template()
    template_facts = make_template_facts() if make_template_facts is not None else None
    for _ in range(rng.randrange(1, 4)):
        where = rng.randrange(3)
        if where == 0:
            damage_facts(rng, facts)
        elif where == 1 or template_facts is None:
            template = damage_rows(rng, template)
        else:
            damage_facts(rng, template_facts)
    return _attempt(rng, facts, template, template_facts)


def run_fixed_case(seed: int) -> str:
    """One case with a fixed-layout value of the template damaged (and nothing else): returns
    ``"ok"`` or ``"refused:<code>"``; raises anything else, or if the template holds no
    fixed-layout value to damage."""
    rng = random.Random(seed)
    make_facts, make_template, make_template_facts = _BASES[seed % len(_BASES)]
    damaged = damage_fixed_rows(rng, make_template())
    assert damaged is not None, "every base template holds a fixed-layout value"
    return _attempt(rng, make_facts(), damaged, make_template_facts())


def _attempt(
    rng: random.Random,
    facts: dict[str, Any],
    template: ClipRows,
    template_facts: dict[str, Any] | None,
) -> str:
    """Every public entry point on the (damaged) inputs, then :func:`author`."""
    try:
        text = json.dumps(facts) if rng.randrange(2) else facts
        parsed = facts_from_json(text)
        parsed_template = facts_from_json(template_facts) if template_facts is not None else None
        kind_of_rows_or_refusal(template)
        clips_from_rows(
            {
                "Sm2MpMedia": [template.media],
                "BtVideoInfo": [template.video] if template.video is not None else [],
                "BtAudioInfo": list(template.audio),
                "BtAudioInfo_Sm2MpMedia": list(template.audio_links),
                "Sm2MpFolder_Sm2MpMedia": [template.folder_link],
                "BtLockableBlob": list(template.lockables),
                "SmPreset": list(template.presets),
            }
        )
        result = author(
            parsed,
            Template(template, parsed_template),
            PLACE,
            new_id=ms.new_ids(),
            allow_unverified=bool(rng.randrange(2)),
        )
    except MediaPoolError as exc:
        assert exc.code and exc.hint, exc
        return f"refused:{exc.code}"
    assert isinstance(result, AuthoredClip)
    assert result.status in ("verified", "unverified")
    assert (result.status == "unverified") == bool(result.unverified)
    return "ok"


def kind_of_rows_or_refusal(template: ClipRows) -> None:
    """The rows' kind is read only from a clip that passed its checks; reading a damaged one
    must still refuse in kind, never crash."""
    with contextlib.suppress(MediaPoolError):
        kind_of_rows(template)


@pytest.mark.parametrize("block", range(6))
def test_damaged_facts_and_templates_give_only_typed_refusals(block: int) -> None:
    outcomes: dict[str, int] = {}
    per = CASES // 6
    for seed in range(block * per, (block + 1) * per):
        try:
            outcome = run_case(seed)
        except Exception as exc:  # the assertion this test exists for
            raise AssertionError(f"seed {seed}: {type(exc).__name__}: {exc}") from exc
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    # the damage reaches deep enough to be refused for many reasons, and some cases survive
    assert len(outcomes) >= 3, outcomes


#: Aimed cases per test run: each base template in turn, one fixed-layout byte damaged.
FIXED_CASES = 600


@pytest.mark.parametrize("block", range(3))
def test_damaged_fixed_layout_values_give_only_typed_refusals(block: int) -> None:
    outcomes: dict[str, int] = {}
    per = FIXED_CASES // 3
    for seed in range(block * per, (block + 1) * per):
        try:
            outcome = run_fixed_case(seed)
        except Exception as exc:  # the assertion this test exists for
            raise AssertionError(f"fixed seed {seed}: {type(exc).__name__}: {exc}") from exc
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    # some damage is refused (a padding word, a non-finite rate), some is overwritten or copied
    assert "ok" in outcomes and any(o.startswith("refused:") for o in outcomes), outcomes


def test_the_aimed_damage_reaches_every_fixed_layout() -> None:
    """The aimed damage lands in a resolution's padding, a rate and the audio extents."""
    hit: set[str] = set()
    for seed in range(300):
        rng = random.Random(seed)
        for make_template in (ms.mp4_template, ms.wav_template, ms.png_template):
            before = make_template()
            after = damage_fixed_rows(rng, before)
            assert after is not None
            hit |= _changed_fixed(before, after)
    assert {"Resolution padding", "FrameRate", "FieldsBlob (extents or rate)"} <= hit, hit


def _changed_fixed(before: ClipRows, after: ClipRows) -> set[str]:
    out: set[str] = set()
    if before.video is not None and after.video is not None:
        for column, key in (("Geometry", "Resolution"), ("Time", "FrameRate")):
            old = blobs.decode_keyed_dict(before.video[column], strict=True).get(key)
            new = blobs.decode_keyed_dict(after.video[column], strict=True).get(key)
            if old != new:
                assert isinstance(old, bytes) and isinstance(new, bytes)
                padding = old[:4] != new[:4] or old[8:12] != new[8:12]
                out.add("Resolution padding" if key == "Resolution" and padding else key)
    if before.media["FieldsBlob"] != after.media["FieldsBlob"] and before.audio:
        out.add("FieldsBlob (extents or rate)")
    return out


def test_the_undamaged_cases_author() -> None:
    for make_facts, make_template, make_template_facts in _BASES:
        template_facts = make_template_facts() if make_template_facts is not None else None
        result = author(
            facts_from_json(make_facts()),
            Template(make_template(), facts_from_json(template_facts) if template_facts else None),
            PLACE,
            new_id=ms.new_ids(),
        )
        assert result.status == "verified"
