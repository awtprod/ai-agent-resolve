"""The Resolve round-trip allow-list: what Resolve rewrites on every export.

When a file goes through Resolve (import, then ``ExportProject``), Resolve rewrites a few values
whatever the edit was: save times, the project's age, the playhead, regenerated DbIds of nested
pan/tilt presets, a save date inside the settings blob and a stamp in the root bin's fields
blob. The S8 sandbox check (blocked on approval D3) will compare the file we wrote with the file
Resolve exports from it; :func:`compare_round_trip` does that comparison and accepts exactly
these rewrites, each checked on the bytes, not just by name.

Use it ONLY for a Resolve-exported round trip. The writer's own output is checked by
:func:`aar.drp.modify.verify.verify_write`, which allows nothing: our writer must produce zero
unintended changes.

Evidence: corpus pair 002 -> 003 (``$AAR_DRP_CORPUS``; two exports of one project by Resolve
20.0.0.0049, one before and one after a multicam clip was made) differs on exactly the eleven
opaque lines these rules accept, plus the project's ``LastModTimeInSecs`` and the multicam
timeline itself. The settings blob there differs only in protobuf field 124 of ``SetupBA`` (a
save date such as ``Jan 19 2026 00:17:51``) once both are decompressed; the rest is zstd framing.
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.container import Container, open_container
from aar.drp.diff import diff_projects
from aar.drp.diff_model import Change, ProjectDiff
from aar.drp.errors import BlobError, ContainerError, DrpError
from aar.drp.model import Provenance
from aar.drp.modify import encode
from aar.drp.modify.encode import EncodeError
from aar.drp.modify.engine import Expected, Refusal, change_key
from aar.drp.reader import read_drp

__all__ = ["ROUND_TRIP_ALLOW_LIST", "AllowRule", "RoundTripResult", "compare_round_trip"]


@dataclass(frozen=True, slots=True)
class AllowRule:
    """One value Resolve rewrites on every export, and how a change to it is recognised."""

    id: str
    what: str
    check: str


ROUND_TRIP_ALLOW_LIST: tuple[AllowRule, ...] = (
    AllowRule(
        "last-modified",
        "project <LastModTimeInSecs>",
        "header change of the project's last_modified_secs (decoded)",
    ),
    AllowRule(
        "project-age",
        "project <ProjectAgeInMs>",
        "leaf directly under SM_Project; both sides are decimal integers",
    ),
    AllowRule(
        "db-saved-time",
        "<DbSavedTime> of any object",
        "leaf named DbSavedTime; both sides are decimal integers",
    ),
    AllowRule(
        "playhead",
        "<CurPlayheadPosition> of a media pool entry",
        "leaf named CurPlayheadPosition; each side is empty or a decimal integer",
    ),
    AllowRule(
        "ptzr-preset-dbid",
        "DbId of a nested <PTZRPreset>/<SmPTZRPreset>",
        "the start tags are identical once the DbId value is masked; children compared as usual",
    ),
    AllowRule(
        "settings-save-date",
        "SetupBA field 124 in project <CommonConfig>/<SM_Config>/<FieldsBlob>",
        "keyed-dict header, count and every entry byte-equal except SetupBA's value (its key, "
        "type and null flag equal too); SetupBA envelopes have the same version and codec; "
        "decompressed protobuf fields byte-equal except field 124, a canonically written date "
        "string on both sides",
    ),
    AllowRule(
        "bin-save-stamp",
        "field 4 of a bin's own <FieldsBlob>",
        "both are protobuf envelopes of the same version; fields byte-equal except varint "
        "field 4, written canonically on both sides",
    ),
)

_DECIMAL = re.compile(rb"^\s*\d+\s*$")
_DBID_ATTR = re.compile(rb'\bDbId\s*=\s*"[^"]*"')
_SAVE_DATE = re.compile(r"^[A-Z][a-z]{2} {1,2}\d{1,2} \d{4} \d{2}:\d{2}:\d{2}$")


@dataclass(slots=True)
class RoundTripResult:
    """``ok`` when every change is allow-listed or expected, and the rest of the gate holds."""

    allowed: list[tuple[str, Change]] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    diff: ProjectDiff | None = None

    @property
    def ok(self) -> bool:
        return not self.refusals and self.diff is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "allowed": [{"rule": r, "change": c.to_dict()} for r, c in self.allowed],
            "refusals": [r.to_dict() for r in self.refusals],
        }


class _Raw:
    """Raw member bytes of both files for the byte-level checks."""

    def __init__(self, a: Path, b: Path) -> None:
        self.c: dict[str, Container] = {"a": open_container(a), "b": open_container(b)}
        self.cache: dict[tuple[str, str], bytes] = {}

    def span(self, side: str, prov: Provenance | None) -> bytes | None:
        if prov is None or prov.kind not in ("text", "element"):
            return None
        key = (side, prov.member)
        if key not in self.cache:
            self.cache[key] = self.c[side].read(prov.member)
        return self.cache[key][prov.start : prov.end]

    def close(self) -> None:
        for c in self.c.values():
            c.close()


def _last(c: Change) -> str:
    return (c.field_name or "").rsplit("/", 1)[-1]


def _leaf_text(raw: bytes | None) -> bytes | None:
    """Text of a leaf span: inner text as is, a self-closing element as empty."""
    if raw is None:
        return None
    return b"" if raw.startswith(b"<") and raw.endswith(b"/>") else raw


def _decimal_both(c: Change, raw: _Raw, *, empty_ok: bool = False) -> bool:
    for side, prov in (("a", c.where_a), ("b", c.where_b)):
        text = _leaf_text(raw.span(side, prov))
        if text is None:
            return False
        if not (_DECIMAL.match(text) or (empty_ok and not text.strip())):
            return False
    return True


def _ptzr(c: Change, raw: _Raw) -> bool:
    tags = []
    for side, prov in (("a", c.where_a), ("b", c.where_b)):
        span = raw.span(side, prov)
        if span is None or prov is None or prov.kind != "element":
            return False
        tags.append(_DBID_ATTR.sub(b'DbId=""', span[: span.index(b">") + 1]))
    return tags[0] == tags[1]


def _hex(c: Change, raw: _Raw) -> tuple[bytes, bytes] | None:
    try:
        texts = [_leaf_text(raw.span(s, p)) for s, p in (("a", c.where_a), ("b", c.where_b))]
        if texts[0] is None or texts[1] is None:
            return None
        return blobs.parse_hex(texts[0].decode("ascii")), blobs.parse_hex(texts[1].decode("ascii"))
    except (BlobError, UnicodeDecodeError):
        return None


def _fields(payload: bytes) -> list[tuple[int, int, Any, bytes]]:
    """(number, wire, value, encoded bytes) of each field: rules compare the encoded bytes, so a
    re-spelled field (a padded varint, a longer length prefix) is not taken for an equal one."""
    return [
        (f.number, f.wire, f.value, payload[f.offset : f.end])
        for f in blobs.decode_protobuf(payload, nested_depth=0)
    ]


def _entry_head(data: bytes, e: blobs.KdEntry) -> bytes:
    """A keyed-dict entry's key, value type and null flag: everything before its value."""
    (klen,) = struct.unpack_from(">I", data, e.offset)
    return data[e.offset : e.offset + 4 + klen + 5]


def _settings_date(c: Change, raw: _Raw) -> bool:
    pair = _hex(c, raw)
    if pair is None:
        return False
    try:
        kds = [blobs.decode_keyed_dict(x, strict=True) for x in pair]
        # The header word and the entry count, then entry by entry.
        if pair[0][:8] != pair[1][:8]:
            return False
        if [e.key for e in kds[0].entries] != [e.key for e in kds[1].entries]:
            return False
        for ea, eb in zip(kds[0].entries, kds[1].entries, strict=True):
            if ea.key != "SetupBA":
                if pair[0][ea.offset : ea.end] != pair[1][eb.offset : eb.end]:
                    return False
                continue
            if _entry_head(pair[0], ea) != _entry_head(pair[1], eb):
                return False  # the SetupBA key, its value type or its null flag changed
            if not isinstance(ea.value, bytes) or not isinstance(eb.value, bytes):
                return False
            env_a, env_b = blobs.parse_envelope(ea.value), blobs.parse_envelope(eb.value)
            if env_a is None or env_b is None:
                return False
            if (env_a.version, env_a.codec) != (env_b.version, env_b.codec):
                return False
            fa = _fields(blobs.unwrap_message(ea.value))
            fb = _fields(blobs.unwrap_message(eb.value))
            if len(fa) != len(fb):
                return False
            for x, y in zip(fa, fb, strict=True):
                if x[3] == y[3]:
                    continue
                if x[:2] != (124, 2) or y[:2] != (124, 2):
                    return False
                if not all(_SAVE_DATE.match(v.decode("ascii")) for v in (x[2], y[2])):
                    return False
                if encode.pb_len(124, x[2]) != x[3] or encode.pb_len(124, y[2]) != y[3]:
                    return False  # the date's own field must be written canonically
        return "SetupBA" in {e.key for e in kds[0].entries}
    except (BlobError, EncodeError, UnicodeDecodeError, struct.error):
        return False


def _bin_stamp(c: Change, raw: _Raw) -> bool:
    pair = _hex(c, raw)
    if pair is None:
        return False
    try:
        env_a, env_b = blobs.parse_envelope(pair[0]), blobs.parse_envelope(pair[1])
        if env_a is None or env_b is None or env_a.version is None:
            return False
        if (env_a.version, env_a.codec) != (env_b.version, env_b.codec):
            return False
        fa = _fields(blobs.unwrap_message(pair[0]))
        fb = _fields(blobs.unwrap_message(pair[1]))
    except BlobError:
        return False
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb, strict=True):
        if x[3] == y[3]:
            continue
        if x[:2] != (4, 0) or y[:2] != (4, 0):
            return False
        if encode.pb_varint(4, x[2]) != x[3] or encode.pb_varint(4, y[2]) != y[3]:
            return False
    return True


def _rule_for(c: Change, raw: _Raw) -> str | None:
    """The allow-list rule that accounts for ``c``, or ``None``."""
    if c.section == "header":
        if (c.kind, c.object_kind, c.field_name) == ("changed", "project", "last_modified_secs"):
            ok = all(isinstance(v, int) and not isinstance(v, bool) for v in (c.old, c.new))
            return "last-modified" if ok else None
        return None
    if c.section != "opaque" or c.kind != "opaque-changed":
        return None
    checks: list[tuple[str, Callable[[], bool]]] = [
        (
            "project-age",
            lambda: (
                c.object_kind == "project"
                and c.field_name == "ProjectAgeInMs"
                and _decimal_both(c, raw)
            ),
        ),
        ("db-saved-time", lambda: _last(c) == "DbSavedTime" and _decimal_both(c, raw)),
        (
            "playhead",
            lambda: _last(c) == "CurPlayheadPosition" and _decimal_both(c, raw, empty_ok=True),
        ),
        (
            "ptzr-preset-dbid",
            lambda: (c.field_name or "").endswith("PTZRPreset/SmPTZRPreset") and _ptzr(c, raw),
        ),
        (
            "settings-save-date",
            lambda: (
                c.object_kind == "project"
                and c.field_name == "CommonConfig/SM_Config/FieldsBlob"
                and _settings_date(c, raw)
            ),
        ),
        (
            "bin-save-stamp",
            lambda: c.object_kind == "bin" and c.field_name == "FieldsBlob" and _bin_stamp(c, raw),
        ),
    ]
    for rule, check in checks:
        if check():
            return rule
    return None


def compare_round_trip(
    intended: str | Path, exported: str | Path, expected: Sequence[Expected] = ()
) -> RoundTripResult:
    """Compare a file we wrote (``intended``) with what Resolve exported from it.

    Every change must be accounted for by the allow-list or be one of ``expected`` (the
    sandbox check passes none: the exported file should match what we wrote). Changes left over
    must also pass the rest of the gate: decoded, DbId-matched, both files read cleanly.
    """
    result = RoundTripResult()
    a_path, b_path = Path(intended), Path(exported)
    try:
        a, b = read_drp(a_path), read_drp(b_path)
        d = diff_projects(a, b, gate=True)
        raw = _Raw(a_path, b_path)
    except DrpError as exc:
        result.refusals.append(Refusal("verify-unreadable", f"cannot compare the files: {exc}"))
        return result
    result.diff = d
    remaining: list[Change] = []
    try:
        for c in d.changes:
            try:
                rule = _rule_for(c, raw)
            except ContainerError:
                rule = None
            if rule is None:
                remaining.append(c)
            else:
                result.allowed.append((rule, c))
    finally:
        raw.close()
    opaque = [c for c in remaining if c.section == "opaque"]
    if opaque:
        result.refusals.append(
            Refusal(
                "unexplained-opaque",
                f"{len(opaque)} opaque change(s) are not on the round-trip allow-list",
                details={"changes": [c.to_dict() for c in opaque]},
            )
        )
    # The rest of the gate: the opaque-count reason is replaced by the check above; any other
    # reason (heuristic matches, duplicate DbIds, read warnings) still fails.
    n_opaque = sum(1 for c in d.changes if c.section == "opaque")
    count_reason = f"{n_opaque} opaque change(s): bytes differ that no decoded value explains"
    reasons = [r for r in d.gate.reasons if r != count_reason]
    if reasons:
        result.refusals.append(
            Refusal("gate-failed", "; ".join(reasons), details={"reasons": reasons})
        )
    want = Counter(e.key() for e in expected)
    have = Counter(change_key(c) for c in remaining if c.section != "opaque")
    if have - want:
        result.refusals.append(
            Refusal(
                "unintended-change",
                f"{sum((have - want).values())} change(s) are neither allow-listed nor expected",
                details={"changes": [repr(k) for k in sorted(have - want, key=repr)]},
            )
        )
    if want - have:
        result.refusals.append(
            Refusal(
                "missing-change",
                f"{sum((want - have).values())} expected change(s) do not show",
                details={"changes": [repr(k) for k in sorted(want - have, key=repr)]},
            )
        )
    return result
