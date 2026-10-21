"""The Resolve round-trip allow-list on synthetic pairs built from a bundled template.

``compare_round_trip`` accepts exactly the values Resolve rewrites on every export, each
checked on the bytes; anything else still fails, and the writer's own check
(``verify_write``) never uses the allow-list. The real Resolve pair is in
``test_drp_modify_corpus``.
"""

from __future__ import annotations

import re
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from diff_testing import H264, POOL
from drp_edit import chain, edit_zip, once
from drp_helpers import needs_zstandard, pb_len, pb_var

from aar.drp import blobs
from aar.drp.modify import compare_round_trip, encode, verify_write
from aar.drp.modify.engine import Expected
from aar.drp.modify.roundtrip import ROUND_TRIP_ALLOW_LIST

Edit = Callable[[bytes], bytes]
PTZR = b'<SmPTZRPreset DbId="35b5b3bc-7552-4e66-ac5d-fb1141236c03">'


def regenerated_ptzr(data: bytes) -> bytes:
    return once(PTZR, b'<SmPTZRPreset DbId="11111111-2222-3333-4444-555555555555">')(data)


def resolve_rewrites() -> dict[str, Edit]:
    """What an export rewrites in this template, one value of each kind."""
    return {
        "project.xml": chain(
            once(
                b"<ProjectAgeInMs>105243607</ProjectAgeInMs>",
                b"<ProjectAgeInMs>99</ProjectAgeInMs>",
            ),
            once(b"<LastModTimeInSecs>1781956352<", b"<LastModTimeInSecs>1781959999<"),
            once(b"<DbSavedTime>105244836<", b"<DbSavedTime>105299999<"),
        ),
        POOL: chain(
            lambda d: d.replace(
                b"<CurPlayheadPosition/>", b"<CurPlayheadPosition>17532</CurPlayheadPosition>", 1
            ),
            regenerated_ptzr,
        ),
    }


def pair(tmp_path: Path, extra: dict[str, Edit] | None = None) -> Path:
    edits = resolve_rewrites()
    for member, edit in (extra or {}).items():
        edits[member] = chain(edits[member], edit) if member in edits else edit
    return edit_zip(H264, tmp_path / "exported.drp", edits)


def test_allow_list_accepts_what_resolve_rewrites(tmp_path: Path) -> None:
    exported = pair(tmp_path)
    result = compare_round_trip(H264, exported)
    assert result.ok, result.to_dict()
    assert sorted(rule for rule, _ in result.allowed) == [
        "db-saved-time",
        "last-modified",
        "playhead",
        "project-age",
        "ptzr-preset-dbid",
    ]


def test_writer_check_never_uses_the_allow_list(tmp_path: Path) -> None:
    exported = pair(tmp_path)
    check = verify_write(H264, exported, [], {"project.xml", POOL})
    assert not check.ok
    assert {r.code for r in check.refusals} == {"gate-failed", "unintended-change"}


@pytest.mark.parametrize(
    ("member", "edit"),
    [
        ("project.xml", lambda d: d.replace(b"<DbSavedTime>39549760<", b"<DbSavedTime>395x<", 1)),
        (POOL, lambda d: d.replace(b"<ID>100</ID>", b"<ID>101</ID>", 1)),
        (
            POOL,
            lambda d: d.replace(
                b'1111-2222-3333-4444-555555555555">',
                b'1111-2222-3333-4444-555555555555" X="1">',
                1,
            ),
        ),
        (
            "project.xml",
            once(b"<ProjectAgeInMs>99</ProjectAgeInMs>", b"<ProjectAgeInMs>-9</ProjectAgeInMs>"),
        ),
        ("project.xml", once(b"<Notes/>", b"<Notes>agent was here</Notes>")),
        ("project.xml", once(b"<DbSavedTime>105299999</DbSavedTime>", b"<DbSavedTime/>")),
    ],
    ids=[
        "saved-time-not-decimal",
        "preset-content",
        "preset-attribute",
        "age-negative",
        "other-leaf",
        "saved-time-emptied",
    ],
)
def test_anything_else_still_fails(tmp_path: Path, member: str, edit: Edit) -> None:
    exported = pair(tmp_path, {member: edit})
    result = compare_round_trip(H264, exported)
    assert not result.ok
    assert [r.code for r in result.refusals] == ["unexplained-opaque"]


def test_semantic_changes_must_be_expected(tmp_path: Path) -> None:
    exported = pair(
        tmp_path, {"project.xml": once(b"<ProjectName>_media_tmpl<", b"<ProjectName>Other<")}
    )
    result = compare_round_trip(H264, exported)
    assert [r.code for r in result.refusals] == ["unintended-change"]
    rename = Expected(
        "header",
        "renamed",
        "project",
        "0f98e18c-6c78-4468-805d-a6680602dd5c",
        "name",
        "_media_tmpl",
        "Other",
        0,
    )
    assert compare_round_trip(H264, exported, [rename]).ok
    (tmp_path / "plain").mkdir()
    missing = compare_round_trip(H264, pair(tmp_path / "plain"), [rename])
    assert [r.code for r in missing.refusals] == ["missing-change"]


def _setup_edit(change: Callable[[list[blobs.PbField], bytes], bytes]) -> Edit:
    """Re-encode SM_Config's SetupBA with ``change`` applied to its protobuf payload."""

    def run(data: bytes) -> bytes:
        start = data.index(b"<FieldsBlob>", data.index(b"<SM_Config")) + len(b"<FieldsBlob>")
        end = data.index(b"</FieldsBlob>", start)
        kd_bytes = blobs.parse_hex(data[start:end].decode())
        kd = blobs.decode_keyed_dict(kd_bytes, strict=True)
        entry = kd.entry("SetupBA")
        assert entry is not None and isinstance(entry.value, bytes)
        payload = blobs.unwrap_message(entry.value)
        new = change(blobs.decode_protobuf(payload, nested_depth=0), payload)
        blob = encode.replace_kd_bytes(kd_bytes, entry, encode.envelope(entry.value, new))
        return data[:start] + blob.hex().encode() + data[end:]

    return run


def _field(number: int, value: bytes | int) -> Callable[[list[blobs.PbField], bytes], bytes]:
    def change(fields: list[blobs.PbField], payload: bytes) -> bytes:
        f = next(x for x in fields if x.number == number)
        return encode.replace_field(payload, f, value)

    return change


@needs_zstandard
def test_settings_blob_only_its_save_date(tmp_path: Path) -> None:
    ok = pair(tmp_path, {"project.xml": _setup_edit(_field(124, b"Jan 19 2026 00:17:51"))})
    result = compare_round_trip(H264, ok)
    assert result.ok, result.to_dict()
    assert "settings-save-date" in [rule for rule, _ in result.allowed]
    (tmp_path / "b").mkdir()
    not_a_date = pair(tmp_path / "b", {"project.xml": _setup_edit(_field(124, b"tomorrow"))})
    assert not compare_round_trip(H264, not_a_date).ok
    (tmp_path / "c").mkdir()
    other = pair(tmp_path / "c", {"project.xml": _setup_edit(_field(8, 15))})
    assert not compare_round_trip(H264, other).ok


def _bin_blob(stamp: int, tail: bytes = b"\x0a\x01x") -> bytes:
    payload = pb_len(1, b"\x1a\x10" + bytes(16)) + pb_var(4, stamp) + pb_len(11, tail)
    body = b"\x80" + payload
    return (struct.pack(">II", 2, len(body)) + body).hex().encode()


def test_bin_save_stamp(tmp_path: Path) -> None:
    def with_blob(blob: bytes) -> Edit:
        return lambda d: re.sub(
            rb"<FieldsBlob/>", b"<FieldsBlob>" + blob + b"</FieldsBlob>", d, count=1
        )

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = edit_zip(H264, tmp_path / "a" / "a.drp", {POOL: with_blob(_bin_blob(3537590540))})
    b = edit_zip(
        a, tmp_path / "b" / "b.drp", {POOL: once(_bin_blob(3537590540), _bin_blob(3537599742))}
    )
    result = compare_round_trip(a, b)
    assert result.ok, result.to_dict()
    assert [rule for rule, _ in result.allowed] == ["bin-save-stamp"]
    (tmp_path / "c").mkdir()
    c = edit_zip(
        a,
        tmp_path / "c" / "c.drp",
        {POOL: once(_bin_blob(3537590540), _bin_blob(3537590540, b"\x0a\x01y"))},
    )
    assert not compare_round_trip(a, c).ok


def test_allow_list_is_documented() -> None:
    ids = [r.id for r in ROUND_TRIP_ALLOW_LIST]
    assert ids == [
        "last-modified",
        "project-age",
        "db-saved-time",
        "playhead",
        "ptzr-preset-dbid",
        "settings-save-date",
        "bin-save-stamp",
    ]
    assert all(r.what and r.check for r in ROUND_TRIP_ALLOW_LIST)


def test_unreadable_side_is_refused(tmp_path: Path) -> None:
    junk = tmp_path / "junk.drp"
    junk.write_bytes(b"not a zip")
    assert [r.code for r in compare_round_trip(H264, junk).refusals] == ["verify-unreadable"]
    assert zipfile.is_zipfile(H264)


def test_ptzr_rule_masks_only_the_dbid(tmp_path: Path) -> None:
    with_x = PTZR[:-1] + b' X="1">'
    a = edit_zip(H264, tmp_path / "a.drp", {POOL: once(PTZR, with_x)})
    new_id = b'<SmPTZRPreset DbId="11111111-2222-3333-4444-555555555555"'
    (tmp_path / "b").mkdir()
    same_x = edit_zip(a, tmp_path / "b" / "b.drp", {POOL: once(with_x, new_id + b' X="1">')})
    assert compare_round_trip(a, same_x).ok
    (tmp_path / "c").mkdir()
    other_x = edit_zip(a, tmp_path / "c" / "c.drp", {POOL: once(with_x, new_id + b' X="2">')})
    assert [r.code for r in compare_round_trip(a, other_x).refusals] == ["unexplained-opaque"]


def test_bin_rule_accepts_only_field_4(tmp_path: Path) -> None:
    def with_blob(blob: bytes) -> Edit:
        return lambda d: d.replace(b"<FieldsBlob/>", b"<FieldsBlob>" + blob + b"</FieldsBlob>", 1)

    def blob(stamp: int, other: int) -> bytes:
        payload = pb_var(4, stamp) + pb_var(5, other)
        return (struct.pack(">II", 2, len(payload) + 1) + b"\x80" + payload).hex().encode()

    a = edit_zip(H264, tmp_path / "a.drp", {POOL: with_blob(blob(1, 1))})
    (tmp_path / "b").mkdir()
    b = edit_zip(a, tmp_path / "b" / "b.drp", {POOL: once(blob(1, 1), blob(1, 2))})
    assert [r.code for r in compare_round_trip(a, b).refusals] == ["unexplained-opaque"]


def test_other_gate_reasons_still_fail(tmp_path: Path) -> None:
    """Both files read with a warning: no change at all, yet the gate cannot vouch for them."""
    bad = once(b"<CurrentTimelineIndex>0<", b"<CurrentTimelineIndex>x<")
    a = edit_zip(H264, tmp_path / "a.drp", {"project.xml": bad})
    (tmp_path / "b").mkdir()
    b = edit_zip(a, tmp_path / "b" / "b.drp", {"project.xml": once(b"<Notes/>", b"<Notes/>")})
    result = compare_round_trip(a, b)
    assert result.diff is not None and result.diff.changes == []
    assert [r.code for r in result.refusals] == ["gate-failed"]
    assert "bad-integer" in result.refusals[0].message
