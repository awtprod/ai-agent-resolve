"""Types shared by :mod:`aar.drp.diff` and :mod:`aar.drp.diff_opaque`, and the two renderings.

A :class:`ProjectDiff` is a list of :class:`Change` records, one per difference, in a stable
order. ``ProjectDiff.to_dict()`` is the JSON rendering (``schema_version`` says which layout);
``ProjectDiff.to_text()`` is compact text for an LLM: one line per change, grouped by
timeline, ending with a summary line.
"""

from __future__ import annotations

import dataclasses
import struct
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from aar.drp.errors import DrpError
from aar.drp.model import DrpFile, Provenance

__all__ = [
    "GATE_MATCHES",
    "MIRRORED_KINDS",
    "REPORTED_FIELDS",
    "SCHEMA_VERSION",
    "SECTIONS",
    "Change",
    "Ctx",
    "DiffError",
    "FileSummary",
    "Gate",
    "ProjectDiff",
    "same_value",
    "short_id",
    "short_id_pair",
]

SECTIONS: tuple[str, ...] = ("header", "media", "timelines", "markers", "opaque")

# Version of the JSON layout produced by ``ProjectDiff.to_dict``.
SCHEMA_VERSION = 2

# ``Change.match`` values the gate accepts: matched by DbId, by a key built on a DbId
# (a marker's owner DbId plus its frame), or not a matched object at all (file, member).
GATE_MATCHES = frozenset({"dbid", "owner+frame", "file", "member"})

# Kinds that swap when A and B swap. Every other kind keeps its name and flips its deltas.
MIRRORED_KINDS = {
    "added": "removed",
    "removed": "added",
    "opaque-added": "opaque-removed",
    "opaque-removed": "opaque-added",
}

# Decoded fields the semantic layer reports whenever their values differ, per object kind.
# The byte-level layer treats a changed text leaf as accounted for only when it is the source
# of one of these fields (see ``diff_opaque``); any other decoded value is still compared raw.
REPORTED_FIELDS: dict[str, frozenset[str]] = {
    "project": frozenset(
        {
            "name",
            "project_version",
            "current_timeline_index",
            "last_modified_secs",
            "timeline_ids",
            "frame_rate_hint",
        }
    ),
    "bin": frozenset({"name", "parent_id"}),
    "media": frozenset(
        {"name", "bin_id", "pool_item_id", "frames", "start_frame", "frame_rate", "resolution"}
    ),
    "timeline": frozenset({"name", "frame_rate", "resolution", "extents"}),
    "track": frozenset({"name", "type_code", "subtype", "flags"}),
    "item": frozenset(
        {
            "record_start",
            "duration",
            "source_in",
            "name",
            "pretty_type",
            "media_ref",
            "media_track_index",
            "media_file_path",
        }
    ),
}


class DiffError(DrpError):
    """The two files cannot be compared, for example one changed on disk after it was read."""


def short_id(value: str | None) -> str:
    """A DbId shortened for text output (the JSON keeps it whole)."""
    if not value:
        return "?"
    return value if len(value) <= 12 else value[:8] + "…"


def short_id_pair(a: str | None, b: str | None) -> tuple[str, str]:
    """Shorten two DbIds for text output so a difference between them stays visible.

    Each is shortened like :func:`short_id`, but when both would render identically (they
    share the leading 8 characters) a window from the first differing character is appended to
    each, so a one-character change does not read as ``0f98e18c… → 0f98e18c…``.
    """
    sa, sb = short_id(a), short_id(b)
    if not a or not b or sa != sb:
        return sa, sb
    # a and b share the leading 8 characters and both render as "{a[:8]}…". Diverge at the
    # first differing character, or at the end of the shorter when one merely extends the other
    # (a prefix extension), so the two windows differ whenever the ids do.
    limit = min(len(a), len(b))
    diff = next((i for i in range(limit) if a[i] != b[i]), limit)
    return f"{a[:8]}…{a[diff : diff + 4]}", f"{b[:8]}…{b[diff : diff + 4]}"


@dataclass(frozen=True, slots=True)
class Change:
    """One difference between file A and file B.

    ``kind`` is a stable verb: ``added``, ``removed``, ``renamed``, ``moved``, ``trimmed``,
    ``slipped``, ``timing-changed``, ``relinked``, ``changed``, ``opaque-changed``,
    ``opaque-added`` or ``opaque-removed``. ``object_id`` is the key the object was matched
    on: its DbId, or ``h:...`` for a heuristic key; ``match`` says how it was matched
    (``dbid``; ``owner+frame`` for markers, whose owner was matched by DbId; ``heuristic``; or
    ``file``/``member`` for things that are not model objects). ``old`` and
    ``new`` are the values in A and B; ``delta`` holds signed frame deltas (B minus A).
    ``where_a``/``where_b`` locate the source text in each file. ``byte_count`` is set for
    opaque changes: how many bytes differ (the larger side of the differing region; binary
    bytes for hex blobs).
    """

    section: str
    kind: str
    object_kind: str
    object_id: str | None
    match: str
    label: str
    detail: str
    timeline_id: str | None = None
    timeline_name: str | None = None
    field_name: str | None = None
    old: Any = None
    new: Any = None
    delta: dict[str, int] | None = None
    where_a: Provenance | None = None
    where_b: Provenance | None = None
    byte_count: int | None = None

    @property
    def text(self) -> str:
        """The one-line text rendering; heuristic matches say so at the end of the line."""
        tag = " [heuristic match]" if self.match == "heuristic" else ""
        return f"{self.label}: {self.detail}{tag}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "kind": self.kind,
            "object_kind": self.object_kind,
            "object_id": self.object_id,
            "match": self.match,
            "timeline_id": self.timeline_id,
            "timeline_name": self.timeline_name,
            "field": self.field_name,
            "old": _jsonable(self.old),
            "new": _jsonable(self.new),
            "delta": dict(self.delta) if self.delta is not None else None,
            "where_a": _prov(self.where_a),
            "where_b": _prov(self.where_b),
            "byte_count": self.byte_count,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class Ctx:
    """Who a change is about: the line prefix and the identity fields every change carries."""

    label: str
    object_kind: str
    object_id: str | None
    match: str
    timeline_id: str | None = None
    timeline_name: str | None = None
    obj_a: Any = field(default=None, compare=False, repr=False)
    obj_b: Any = field(default=None, compare=False, repr=False)

    def change(self, section: str, kind: str, detail: str, **values: Any) -> Change:
        """A change about this object; ``where_a``/``where_b`` default to the field's span."""
        name = values.get("field_name")
        values.setdefault("where_a", where(self.obj_a, name))
        values.setdefault("where_b", where(self.obj_b, name))
        return Change(
            section=section,
            kind=kind,
            object_kind=self.object_kind,
            object_id=self.object_id,
            match=self.match,
            label=self.label,
            detail=detail,
            timeline_id=self.timeline_id,
            timeline_name=self.timeline_name,
            **values,
        )


@dataclass(frozen=True, slots=True)
class FileSummary:
    """Identity of one side of the diff."""

    path: str
    sha256: str
    kind: str
    size: int
    app_version: str | None
    project_version: int | None

    @classmethod
    def of(cls, drp: DrpFile) -> FileSummary:
        v = drp.versions
        return cls(
            path=drp.path,
            sha256=drp.sha256,
            kind=drp.kind,
            size=drp.size,
            app_version=v.app_version if v else None,
            project_version=v.project_version if v else None,
        )

    def stamp(self) -> str:
        app = self.app_version or "?"
        return f"{self.kind.upper()}, Resolve {app}, DbPrjVer {self.project_version}"


@dataclass(frozen=True, slots=True)
class Gate:
    """Whether a diff can vouch for every difference it found (the S8 post-write gate).

    ``passed`` is true only when every change is semantic (decoded), every matched object was
    matched by DbId (or by a key built on one), both files read without warnings, and no DbId
    or zip member name is duplicated. ``reasons`` says what failed.
    """

    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class ProjectDiff:
    """The result of :func:`aar.drp.diff.diff_projects`.

    ``hidden`` counts, per section, changes removed by :meth:`only`; they still count for
    :attr:`identical`, so filtering the view never turns a difference into "identical".
    ``gate`` is computed on the whole diff; ``gate_mode`` says it was asked for (it then
    appears in the text rendering).
    """

    a: FileSummary
    b: FileSummary
    changes: list[Change]
    notes: list[str] = field(default_factory=list)
    sections: tuple[str, ...] = SECTIONS
    hidden: dict[str, int] = field(default_factory=dict)
    gate: Gate = field(default_factory=lambda: Gate(True))
    gate_mode: bool = False

    @property
    def identical(self) -> bool:
        """No differences at all, including any hidden by :meth:`only`."""
        return not self.changes and not sum(self.hidden.values())

    def counts(self) -> dict[str, int]:
        found = Counter(c.section for c in self.changes)
        return {s: found.get(s, 0) for s in self.sections}

    def heuristic_count(self) -> int:
        return sum(1 for c in self.changes if c.match == "heuristic")

    def only(self, sections: Iterable[str]) -> ProjectDiff:
        """The same diff showing only some sections; the rest are counted in ``hidden``."""
        wanted = tuple(dict.fromkeys(sections))
        unknown = [s for s in wanted if s not in SECTIONS]
        if unknown:
            raise ValueError(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
        keep = [c for c in self.changes if c.section in wanted]
        hidden = Counter(self.hidden)
        hidden.update(c.section for c in self.changes if c.section not in wanted)
        return dataclasses.replace(
            self,
            changes=keep,
            sections=wanted,
            hidden={s: hidden[s] for s in SECTIONS if hidden.get(s)},
        )

    def summary_line(self) -> str:
        counts = ", ".join(f"{s} {n}" for s, n in self.counts().items())
        n = len(self.changes)
        line = f"summary: {n} change{'s' if n != 1 else ''} ({counts})"
        total_hidden = sum(self.hidden.values())
        if total_hidden:
            parts = ", ".join(f"{s} {k}" for s, k in self.hidden.items())
            line += f"; {total_hidden} more hidden by --only ({parts})"
        h = self.heuristic_count()
        return line + f"; {h} heuristic match{'es' if h != 1 else ''}"

    def gate_line(self) -> str:
        if self.gate.passed:
            return "gate: PASS (every change is decoded and matched by DbId)"
        return "gate: FAIL: " + "; ".join(self.gate.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "a": dataclasses.asdict(self.a),
            "b": dataclasses.asdict(self.b),
            "identical": self.identical,
            "sections": list(self.sections),
            "summary": {
                "changes": len(self.changes),
                "by_section": self.counts(),
                "hidden": dict(self.hidden),
                "heuristic": self.heuristic_count(),
            },
            "gate": {
                "passed": self.gate.passed,
                "reasons": list(self.gate.reasons),
                "requested": self.gate_mode,
            },
            "notes": list(self.notes),
            "changes": [c.to_dict() for c in self.changes],
        }

    def to_text(self) -> str:
        lines = [f"a: {self.a.path} ({self.a.stamp()})", f"b: {self.b.path} ({self.b.stamp()})"]
        lines.extend(c.text for c in self.changes)
        lines.extend(f"note: {n}" for n in self.notes)
        if self.gate_mode:
            lines.append(self.gate_line())
        lines.append(self.summary_line())
        return "\n".join(lines) + "\n"


def _bits(value: Any) -> Any:
    """A comparison key that tells ``-0.0`` from ``0.0`` and NaN payloads apart."""
    if isinstance(value, float):
        return ("f", struct.pack("<d", value))
    if isinstance(value, tuple | list):
        return tuple(_bits(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _bits(v)) for k, v in value.items()))
    return value


def same_value(x: Any, y: Any) -> bool:
    """Equality of decoded values, with floats compared bit for bit."""
    return bool(_bits(x) == _bits(y))


# Changes reported under a name that is not a span key, and the span that holds the value.
# ``timing`` (record and source range) spans several fields, so it points at the whole item.
_SPAN_ALIASES = {"path": "blob_path", "start": "extents", "length": "extents"}


def where(obj: Any, name: str | None) -> Provenance | None:
    """The source span of ``obj``'s field ``name``, else of ``obj``'s whole element."""
    if obj is None:
        return None
    spans = getattr(obj, "spans", None) or {}
    if name:
        prov = spans.get(name) or spans.get(_SPAN_ALIASES.get(name, ""))
        if prov is None and name == "timeline_ids":
            prov = spans.get("timeline_ids[0]")
        if prov is not None:
            return prov
    source = getattr(obj, "source", None)
    return source if isinstance(source, Provenance) else None


def _prov(p: Provenance | None) -> dict[str, Any] | None:
    return dataclasses.asdict(p) if p is not None else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.hex()
    return value
