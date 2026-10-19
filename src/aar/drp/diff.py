"""Semantic diff of two ``.drp``/``.drt``/``.drb`` files.

::

    from aar.drp import read_drp
    from aar.drp.diff import diff_projects

    d = diff_projects(read_drp("v1.drp"), read_drp("v2.drp"))
    print(d.to_text())          # one line per change, grouped by timeline
    d.to_dict()                 # the same as JSON-ready data

:func:`diff_projects` returns a :class:`~aar.drp.diff_model.ProjectDiff`: a stably ordered list
of :class:`~aar.drp.diff_model.Change` records in five sections.

``header``     container kind, the DbAppVer/DbPrjVer stamp, and project-level values
``media``      bins and media clips: added, removed, renamed, moved, relinked, format changes
``timelines``  timelines, tracks and items: moves, trims, slips, source-media changes
``markers``    timeline and clip markers
``opaque``     everything the reader does not decode, compared byte for byte

Matching. Objects are matched by Resolve's ``DbId``. Only objects without one fall back to
(kind, name, parent) plus an ordinal, and those changes carry ``match="heuristic"``. Markers
have no DbId, so they are matched by owner and frame, then by a unique name, colour and note;
marker changes are always heuristic.

Nothing is called equal on the strength of what was decoded. :mod:`aar.drp.diff_opaque`
re-reads both files and compares every element of every member; a difference no decoded field
accounts for is reported as ``opaque-changed`` with member, byte span and byte count.

Unreadable inputs (a member that cannot be read, a ``.drp``/``.drt`` that is not a zip, a
missing ``project.xml``) raise :class:`~aar.drp.diff_model.DiffError`; they are never
reported as differences.

Gate (S8's post-write check). ``ProjectDiff.gate`` passes only when every change is semantic
(decoded) and matched by DbId (markers: owner DbId plus frame), no DbId or zip member name is
duplicated, and both files read without warnings. ``diff_projects(..., gate=True)`` also turns
off the heuristic pairing of markers by name, so a moved marker shows as removed plus added.
Known limits: blob content is compared after decoding where the reader decodes it, so (1) two
zstd frames that differ only in framing (for example the last-block bit) but decompress to the
same bytes, and (2) hex text that differs only in letter case inside a decoded blob
(``FrameRate``, ``Resolution``, ``Clip`` and the other blobs listed in
:mod:`aar.drp.diff_opaque`) pass as equal. Everything else, the XML prolog and header comment
included, is compared byte for byte.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from aar.drp.describe import frames_to_timecode
from aar.drp.diff_model import (
    GATE_MATCHES,
    SECTIONS,
    Change,
    Ctx,
    DiffError,
    FileSummary,
    Gate,
    ProjectDiff,
    same_value,
    short_id,
    short_id_pair,
)
from aar.drp.diff_opaque import OpaqueEngine, Source, Unit, member_root
from aar.drp.model import (
    Bin,
    DrpFile,
    Marker,
    MediaClip,
    ProjectInfo,
    Timeline,
    TimelineItem,
    Track,
)
from aar.drp.reader import DEFAULT_MAX_ELEMENTS, DEFAULT_MAX_TOTAL_XML_BYTES
from aar.drp.scan import MAX_ATTRS_PER_ELEMENT

__all__ = [
    "SECTIONS",
    "Change",
    "DiffError",
    "FileSummary",
    "Gate",
    "ProjectDiff",
    "diff_projects",
]

T = TypeVar("T")
Key = tuple[Any, ...]


def diff_projects(
    a: DrpFile,
    b: DrpFile,
    *,
    gate: bool = False,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    max_total_xml_bytes: int = DEFAULT_MAX_TOTAL_XML_BYTES,
) -> ProjectDiff:
    """Compare ``a`` (before) with ``b`` (after).

    Both files are reopened to compare what the model does not decode; they must not have
    changed since they were read (checked by SHA-256). Raises
    :class:`~aar.drp.diff_model.DiffError` when either file cannot be read whole, when they
    cannot be reopened, or when they have changed. ``gate=True`` asks for the S8 gate: no
    heuristic marker pairing, and the verdict shown in the text rendering (``ProjectDiff.gate``
    is computed either way). ``max_elements`` and ``max_total_xml_bytes`` bound the re-read and
    re-scan the opaque compare does, with the reader's defaults; a member past either raises
    :class:`~aar.drp.diff_model.DiffError`.
    """
    _check_readable("A", a)
    _check_readable("B", b)
    return _Differ(a, b, gate, max_elements, max_total_xml_bytes).run()


def _check_readable(side: str, drp: DrpFile) -> None:
    """An input the reader could not read whole is an error, not something to diff."""
    errors = [i for i in drp.issues if i.severity == "error"]
    if errors:
        found = "; ".join(f"{i.code}: {i.message}" for i in errors[:3])
        raise DiffError(f"{side} ({drp.path}) cannot be compared: {found}")
    if drp.kind in ("drp", "drt") and not drp.is_zip:
        raise DiffError(
            f"{side} ({drp.path}) is not a zip archive, so it is not a readable .{drp.kind}"
        )


# --------------------------------------------------------------------------- matching


@dataclass(slots=True)
class _Pair(Generic[T]):
    key: str
    a: T | None
    b: T | None
    match: str  # "dbid" or "heuristic"


def _match(
    a: Sequence[T],
    b: Sequence[T],
    dbid: Callable[[T], str | None],
    fallback: Callable[[T], tuple[Any, ...]],
    duplicates: list[str] | None = None,
    what: str = "object",
) -> list[_Pair[T]]:
    """Pair objects by DbId; objects without one by ``fallback`` plus an ordinal.

    An object with a DbId is never paired with one without. A DbId that occurs more than once
    on one side is paired by occurrence order, so those pairs are marked heuristic and the
    duplicate is recorded in ``duplicates``. Order: A's order, then B-only objects in B's order.
    """

    def keyed(objs: Sequence[T], side: str) -> dict[str, tuple[T, str]]:
        counts = Counter(i for i in (dbid(o) for o in objs) if i)
        for ident, n in sorted(counts.items()):
            if n > 1 and duplicates is not None:
                duplicates.append(f"{side}: DbId {ident} occurs {n} times among {what}s")
        out: dict[str, tuple[T, str]] = {}
        seen: Counter[tuple[Any, ...]] = Counter()
        for obj in objs:
            ident = dbid(obj)
            if ident:
                key, how = ident, ("dbid" if counts[ident] == 1 else "heuristic")
            else:
                fb = fallback(obj)
                seen[fb] += 1
                key, how = "h:" + "/".join(str(x) for x in fb) + f"#{seen[fb]}", "heuristic"
            if key in out:
                n = 2
                while f"{key}~{n}" in out:
                    n += 1
                key = f"{key}~{n}"
            out[key] = (obj, how)
        return out

    ka, kb = keyed(a, "A"), keyed(b, "B")
    pairs: list[_Pair[T]] = []
    for key, (obj, how) in ka.items():
        other = kb.get(key)
        if other is not None and other[1] == "heuristic":
            how = "heuristic"
        pairs.append(_Pair(key, obj, other[0] if other else None, how))
    for key, (obj, how) in kb.items():
        if key not in ka:
            pairs.append(_Pair(key, None, obj, how))
    return pairs


def _idtag(key: str, match: str) -> str:
    return f" (DbId {short_id(key)})" if match == "dbid" else " (no DbId)"


# --------------------------------------------------------------------------- formatting


def _q(text: str | None) -> str:
    return f'"{text}"' if text is not None else "none"


def _rate(fps: float | None) -> str:
    if fps is None:
        return "unknown fps"
    return f"{fps:.3f}".rstrip("0").rstrip(".") + " fps"


def _res(res: tuple[int, int] | None) -> str:
    return f"{res[0]}x{res[1]}" if res else "unknown resolution"


def _tc(fps: float | None) -> Callable[[int | None], str]:
    def tc(frame: int | None) -> str:
        return "?" if frame is None else frames_to_timecode(frame, fps)

    return tc


def _signed(n: int) -> str:
    return f"{n:+d}f"


def _plain(value: Any) -> Any:
    """A JSON-friendly copy of a model value (tuples become lists)."""
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------- values


def _project_values(p: ProjectInfo) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "project_version": p.project_version,
        "current_timeline_index": p.current_timeline_index,
        "last_modified_secs": p.last_modified_secs,
        "timeline_ids": tuple(p.timeline_ids),
        "frame_rate_hint": p.frame_rate_hint,
    }


def _media_values(m: MediaClip) -> dict[str, Any]:
    return {
        "element": m.element,
        "type": m.type,
        "name": m.name,
        "bin_id": m.bin_id,
        "pool_item_id": m.pool_item_id,
        "path": m.path,
        "path_source": m.path_source,
        "blob_path": m.blob_path,
        "linked_paths": tuple(m.linked_paths),
        "codec": m.codec,
        "frames": m.frames,
        "start_frame": m.start_frame,
        "frame_rate": m.frame_rate,
        "resolution": m.resolution,
    }


def _item_values(it: TimelineItem) -> dict[str, Any]:
    return {
        "element": it.element,
        "kind": it.kind,
        "name": it.name,
        "pretty_type": it.pretty_type,
        "record_start": (it.record_start, it.fractions.get("record_start")),
        "duration": (it.duration, it.fractions.get("duration")),
        "source_in": (it.source_in, it.fractions.get("source_in")),
        "media_ref": it.media_ref,
        "media_track_index": it.media_track_index,
        "media_file_path": it.media_file_path,
        "speed": it.speed,
        "timemap_form": it.timemap_form,
    }


def _diffs(va: dict[str, Any], vb: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Fields whose decoded values differ (floats bit for bit, so ``-0.0`` is not ``0.0``)."""
    return {k: (va[k], vb[k]) for k in va if not same_value(va[k], vb[k])}


_FIELD_LABELS = {
    "project_version": "ProjectVersion",
    "current_timeline_index": "current timeline index",
    "last_modified_secs": "last modified (epoch seconds)",
    "frame_rate_hint": "frame rate hint (inferred from the settings blob)",
    "pool_item_id": "UniqueMediaPoolItemId",
    "pretty_type": "PrettyType",
    "media_track_index": "source channel (MediaTrackIdx)",
    "timemap_form": "timemap form",
    "type_code": "track Type",
    "subtype": "track SubType",
    "flags": "track Flags",
    "element": "element type",
    "path_source": "path source",
}


@dataclass(slots=True)
class _Placed:
    track: Track
    item: TimelineItem


# --------------------------------------------------------------------------- the differ


class _Differ:
    def __init__(
        self,
        a: DrpFile,
        b: DrpFile,
        gate: bool = False,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
        max_total_xml_bytes: int = DEFAULT_MAX_TOTAL_XML_BYTES,
    ) -> None:
        self.a, self.b = a, b
        self.gate = gate
        self.max_elements = max_elements
        self.max_total_xml_bytes = max_total_xml_bytes
        self.duplicates: list[str] = []
        self.relocated: set[Key] = set()
        # media clip id -> the bin whose MediaVec lists it (its actual container)
        self.container = {
            "a": {c: x.id for x in a.bins for c in x.clip_ids},
            "b": {c: x.id for x in b.bins for c in x.clip_ids},
        }
        self.bin_key: dict[str, dict[str, str]] = {"a": {}, "b": {}}
        self.bin_ctx: dict[str, Ctx] = {}
        self.clip_key: dict[str, dict[str, str]] = {"a": {}, "b": {}}
        self.track_ctx: dict[tuple[str, str], Ctx] = {}
        self.out: list[Change] = []
        self.changed: set[tuple[Key, str]] = set()
        self.units: list[Unit] = []
        self.cover_objects: dict[str, list[tuple[Key, str, Any]]] = {"a": [], "b": []}
        self.owners: dict[str, dict[str, tuple[Key, Ctx]]] = {"a": {}, "b": {}}
        self.timeline_rank: dict[str, int] = {}
        self.bin_paths = {
            "a": {x.id: x.path for x in a.bins},
            "b": {x.id: x.path for x in b.bins},
        }
        self.media = {"a": a.media_by_id(), "b": b.media_by_id()}
        self.dbid_pairs = 0
        self.dbid_objects = 0

    # ------------------------------------------------------------------ driver

    def run(self) -> ProjectDiff:
        self._header()
        tl_pairs = _match(
            self.a.timelines,
            self.b.timelines,
            lambda t: t.id or t.sequence_id or t.media_clip_id,
            lambda t: (t.kind, t.name),
            self.duplicates,
            "timeline",
        )
        self._project(tl_pairs)
        bin_pairs = self._media(tl_pairs)
        for rank, pair in enumerate(tl_pairs):
            self.timeline_rank[pair.key] = rank
            self._timeline(pair)
        self._bin_orders(bin_pairs)
        budgets = {
            "max_elements": self.max_elements,
            "max_total_xml_bytes": self.max_total_xml_bytes,
        }
        sources = (Source(self.a, "A", **budgets), Source(self.b, "B", **budgets))
        try:
            engine = OpaqueEngine(
                sources,
                self.units,
                self.cover_objects,
                self.changed,
                self.owners,
                self.relocated,
            )
            self.out.extend(engine.run())
        finally:
            for s in sources:
                s.close()
        ordered = [c for _, c in sorted(enumerate(self.out), key=lambda ic: self._order(*ic))]
        return ProjectDiff(
            a=FileSummary.of(self.a),
            b=FileSummary.of(self.b),
            changes=ordered,
            notes=self._notes(),
            gate=self._gate(ordered),
            gate_mode=self.gate,
        )

    def _gate(self, changes: list[Change]) -> Gate:
        reasons = []
        opaque = sum(1 for c in changes if c.section == "opaque")
        if opaque:
            reasons.append(
                f"{opaque} opaque change(s): bytes differ that no decoded value explains"
            )
        loose = sum(1 for c in changes if c.match not in GATE_MATCHES)
        if loose:
            reasons.append(f"{loose} change(s) on objects not matched by DbId")
        reasons.extend(self.duplicates)
        for side, drp in (("A", self.a), ("B", self.b)):
            codes = sorted({i.code for i in drp.issues if i.severity in ("warning", "error")})
            if codes:
                reasons.append(f"{side}: read with warnings ({', '.join(codes)})")
            names = Counter(m.name for m in drp.members)
            dups = sorted(n for n, k in names.items() if k > 1)
            if dups:
                reasons.append(f"{side}: duplicate zip member name(s): {', '.join(dups)}")
        return Gate(not reasons, tuple(reasons))

    def _order(self, index: int, c: Change) -> tuple[int, int, int, int]:
        if c.section == "header":
            group = 0
        elif c.timeline_id is not None:
            group = 2
        elif c.object_kind in ("bin", "media"):
            group = 1
        elif c.object_kind in ("file", "project"):
            group = 0
        else:
            group = 3
        rank = self.timeline_rank.get(c.timeline_id or "", -1)
        section = {"opaque": 2, "markers": 1}.get(c.section, 0)
        return (group, rank, section, index)

    def _notes(self) -> list[str]:
        notes = [
            f"duplicate DbId: {d}; pairs by occurrence order (heuristic)" for d in self.duplicates
        ]
        if self.dbid_objects and not self.dbid_pairs:
            notes.append(
                "no DbId is shared between A and B: they look like different projects, so "
                "objects show as added and removed rather than changed"
            )
        for side, drp in (("A", self.a), ("B", self.b)):
            if drp.zstd_backend is None:
                notes.append(
                    f"{side}: no zstd decompressor is installed, so compressed blobs the model "
                    "would decode are compared as opaque bytes"
                )
            codes = Counter(i.code for i in drp.issues if i.severity in ("warning", "error"))
            if codes.get("marker-decode"):
                notes.append(
                    f"{side}: {codes['marker-decode']} marker blob(s) did not decode; their "
                    "bytes are compared as opaque, and their markers are missing from the model"
                )
            if codes.get("scan-attributes-truncated"):
                notes.append(
                    f"{side}: an element has more than {MAX_ATTRS_PER_ELEMENT} attributes; the "
                    "attributes past that point are compared as raw bytes, not by name"
                )
        return notes

    def _track_dbid(self, pair: _Pair[Any]) -> None:
        if pair.match == "dbid":
            self.dbid_objects += 1
            if pair.a is not None and pair.b is not None:
                self.dbid_pairs += 1

    def _covers(self, key: Key, kind: str, a: Any, b: Any) -> None:
        if a is not None:
            self.cover_objects["a"].append((key, kind, a))
        if b is not None:
            self.cover_objects["b"].append((key, kind, b))

    def _mark(self, key: Key, diffs: Iterable[str]) -> None:
        self.changed.update((key, f) for f in diffs)

    # ------------------------------------------------------------------ header

    def _header(self) -> None:
        a, b = self.a, self.b
        ctx = Ctx("Header", "file", None, "file")

        def emit(name: str, old: Any, new: Any, detail: str) -> None:
            self.out.append(
                ctx.change("header", "changed", detail, field_name=name, old=old, new=new)
            )

        if a.kind != b.kind:
            emit("kind", a.kind, b.kind, f"container kind {a.kind} → {b.kind}")
        if a.sniffed_kind != b.sniffed_kind:
            emit(
                "sniffed_kind",
                a.sniffed_kind,
                b.sniffed_kind,
                f"contents look like {a.sniffed_kind} → {b.sniffed_kind}",
            )
        if a.is_zip != b.is_zip:
            emit("is_zip", a.is_zip, b.is_zip, f"zip archive {a.is_zip} → {b.is_zip}")
        va, vb = a.versions, b.versions
        app_a, app_b = (va.app_version if va else None), (vb.app_version if vb else None)
        prj_a, prj_b = (va.project_version if va else None), (vb.project_version if vb else None)
        if app_a != app_b:
            emit("DbAppVer", app_a, app_b, f"DbAppVer {app_a} → {app_b}")
        if prj_a != prj_b:
            emit("DbPrjVer", prj_a, prj_b, f"DbPrjVer {prj_a} → {prj_b}")

    # ------------------------------------------------------------------ project

    def _project(self, tl_pairs: list[_Pair[Timeline]]) -> None:
        pa, pb = self.a.project, self.b.project
        if pa is None and pb is None:
            return
        name = (pb or pa).name if (pb or pa) else None
        key: Key = ("project",)
        # The project is the file's singleton, paired whatever its DbId; a DbId change is
        # reported as a change of the project rather than as a different object.
        same = pa is not None and pb is not None and pa.id == pb.id
        ctx = Ctx(
            f"Project {_q(name)}",
            "project",
            pa.id if same and pa else None,
            "dbid" if same else "file",
            obj_a=pa,
            obj_b=pb,
        )
        self.units.append(
            Unit(
                key,
                "project",
                ctx,
                pa.source if pa else None,
                pb.source if pb else None,
                dbid_a=pa.id if pa else None,  # a project DbId change is reported as "id"
                dbid_b=pb.id if pb else None,
            )
        )
        if pa is None or pb is None:
            kind = "added" if pa is None else "removed"
            self.out.append(ctx.change("header", kind, f"project.xml {kind}"))
            return
        self._covers(key, "project", pa, pb)
        diffs = _diffs(_project_values(pa), _project_values(pb))
        self._mark(key, diffs)
        for f, (old, new) in diffs.items():
            if f == "id":
                old_id, new_id = short_id_pair(old, new)
                detail = (
                    f"project DbId {old_id} → {new_id} "
                    "(a different DbId usually means a different project)"
                )
                self.out.append(
                    ctx.change("header", "changed", detail, field_name="id", old=old, new=new)
                )
            elif f == "name":
                self.out.append(
                    ctx.change(
                        "header",
                        "renamed",
                        f"renamed {_q(old)} → {_q(new)}",
                        field_name="name",
                        old=old,
                        new=new,
                    )
                )
            elif f == "timeline_ids":
                self._timeline_list(ctx, old, new, tl_pairs)
            else:
                self.out.append(
                    ctx.change(
                        "header",
                        "changed",
                        f"{_FIELD_LABELS.get(f, f)} {old} → {new}",
                        field_name=f,
                        old=old,
                        new=new,
                    )
                )

    def _timeline_list(
        self,
        ctx: Ctx,
        old: tuple[str, ...],
        new: tuple[str, ...],
        tl_pairs: list[_Pair[Timeline]],
    ) -> None:
        """The project's timeline list; additions and removals are reported per timeline."""
        removed = {p.a.id for p in tl_pairs if p.a is not None and p.b is None and p.a.id}
        added = {p.b.id for p in tl_pairs if p.b is not None and p.a is None and p.b.id}
        kept_a = [x for x in old if x not in removed]
        kept_b = [x for x in new if x not in added]
        if kept_a == kept_b:
            return
        self.out.append(
            ctx.change(
                "header",
                "changed",
                f"timeline list changed: {len(old)} → {len(new)} entries (order or membership "
                "beyond the timelines reported as added or removed)",
                field_name="timeline_ids",
                old=list(old),
                new=list(new),
            )
        )

    # ------------------------------------------------------------------ media pool

    def _media(self, tl_pairs: list[_Pair[Timeline]]) -> list[_Pair[Bin]]:
        bin_pairs = _match(
            self.a.bins,
            self.b.bins,
            lambda x: None if x.id == x.member else x.id,
            lambda x: ("bin", x.path),
            self.duplicates,
            "bin",
        )
        for pair in bin_pairs:
            if pair.a is not None:
                self.bin_key["a"][pair.a.id] = pair.key
            if pair.b is not None:
                self.bin_key["b"][pair.b.id] = pair.key
        for pair in bin_pairs:
            self._track_dbid(pair)
            self._bin(pair)
        owned = {
            "a": {p.a.media_clip_id for p in tl_pairs if p.a is not None and p.a.media_clip_id},
            "b": {p.b.media_clip_id for p in tl_pairs if p.b is not None and p.b.media_clip_id},
        }
        clips_a = [m for m in self.a.media if m.id not in owned["a"]]
        clips_b = [m for m in self.b.media if m.id not in owned["b"]]
        clip_pairs = _match(
            clips_a,
            clips_b,
            lambda m: m.id,
            lambda m: ("media", m.name, m.bin_id),
            self.duplicates,
            "media clip",
        )
        for pair in clip_pairs:
            self._track_dbid(pair)
            for side, clip in (("a", pair.a), ("b", pair.b)):
                if clip is not None:
                    self.clip_key[side][clip.id] = pair.key
            self._clip(pair)
        return bin_pairs

    def _bin_orders(self, bin_pairs: list[_Pair[Bin]]) -> None:
        """A bin's MediaVec order (the order of its contents) is part of the project."""
        for pair in bin_pairs:
            if pair.a is None or pair.b is None:
                continue
            ka = [self.clip_key["a"].get(c, f"?{c}") for c in pair.a.clip_ids]
            kb = [self.clip_key["b"].get(c, f"?{c}") for c in pair.b.clip_ids]
            seq_a = [k for k in ka if k in set(kb)]
            seq_b = [k for k in kb if k in set(ka)]
            if seq_a != seq_b:
                self.out.append(
                    self.bin_ctx[pair.key].change(
                        "media",
                        "reordered",
                        f"order of {len(seq_a)} item(s) in the bin changed",
                        field_name="clip_order",
                        old=seq_a,
                        new=seq_b,
                    )
                )

    def _bin(self, pair: _Pair[Bin]) -> None:
        x = pair.b or pair.a
        assert x is not None
        ctx = Ctx(
            f"Media pool bin {_q(x.path)}{_idtag(pair.key, pair.match)}",
            "bin",
            pair.key,
            pair.match,
            obj_a=pair.a,
            obj_b=pair.b,
        )
        key: Key = ("bin", pair.key)
        self.bin_ctx[pair.key] = ctx
        self.units.append(
            Unit(
                key,
                "bin",
                ctx,
                pair.a.source if pair.a else None,
                pair.b.source if pair.b else None,
            )
        )
        if pair.a is None or pair.b is None:
            kind = "added" if pair.a is None else "removed"
            self.out.append(
                ctx.change(
                    "media",
                    kind,
                    f"{kind} ({len(x.clip_ids)} item(s))",
                    old=None if pair.a is None else x.path,
                    new=x.path if pair.a is None else None,
                )
            )
            return
        self._covers(key, "bin", pair.a, pair.b)
        va = {"name": pair.a.name, "parent_id": pair.a.parent_id}
        vb = {"name": pair.b.name, "parent_id": pair.b.parent_id}
        diffs = _diffs(va, vb)
        self._mark(key, diffs)
        if "name" in diffs:
            old, new = diffs["name"]
            self.out.append(
                ctx.change(
                    "media",
                    "renamed",
                    f"renamed {_q(old)} → {_q(new)}",
                    field_name="name",
                    old=old,
                    new=new,
                )
            )
        if "parent_id" in diffs:
            old_parent = self.bin_paths["a"].get(pair.a.parent_id or "", pair.a.parent_id)
            new_parent = self.bin_paths["b"].get(pair.b.parent_id or "", pair.b.parent_id)
            self.out.append(
                ctx.change(
                    "media",
                    "moved",
                    f"moved from {_q(old_parent)} to {_q(new_parent)} "
                    f"({_q(pair.a.path)} → {_q(pair.b.path)})",
                    field_name="parent_id",
                    old=pair.a.path,
                    new=pair.b.path,
                )
            )

    def _clip(self, pair: _Pair[MediaClip]) -> None:
        x = pair.b or pair.a
        assert x is not None
        ctx = Ctx(
            f"Media {_q(x.name)}{_idtag(pair.key, pair.match)}",
            "media",
            pair.key,
            pair.match,
            obj_a=pair.a,
            obj_b=pair.b,
        )
        key: Key = ("media", pair.key)
        self.units.append(
            Unit(
                key,
                "media",
                ctx,
                pair.a.source if pair.a else None,
                pair.b.source if pair.b else None,
                pair.a,
                pair.b,
            )
        )
        if pair.a is None or pair.b is None:
            side = "b" if pair.a is None else "a"
            kind = "added" if pair.a is None else "removed"
            where = self.bin_paths[side].get(x.bin_id or "", x.bin_id)
            detail = f"{kind} in bin {_q(where)} ({x.type}; path {_q(x.path)})"
            self.out.append(
                ctx.change(
                    "media",
                    kind,
                    detail,
                    old=x.path if kind == "removed" else None,
                    new=x.path if kind == "added" else None,
                )
            )
            return
        self._covers(key, "media", pair.a, pair.b)
        diffs = _diffs(_media_values(pair.a), _media_values(pair.b))
        self._mark(key, diffs)
        self._media_changes(ctx, key, "media", pair.a, pair.b, diffs)

    def _media_changes(
        self,
        ctx: Ctx,
        key: Key,
        section: str,
        a: MediaClip,
        b: MediaClip,
        diffs: dict[str, tuple[Any, Any]],
        *,
        skip: frozenset[str] = frozenset(),
    ) -> None:
        def emit(kind: str, f: str, detail: str, old: Any = None, new: Any = None) -> None:
            self.out.append(
                ctx.change(section, kind, detail, field_name=f, old=_plain(old), new=_plain(new))
            )

        # Which bin's MediaVec lists the clip is its real place; <MpFolder> is a back-reference.
        cont_a, cont_b = self.container["a"].get(a.id), self.container["b"].get(b.id)
        path_a = self.bin_paths["a"].get(cont_a or "", cont_a)
        path_b = self.bin_paths["b"].get(cont_b or "", cont_b)
        moved = self.bin_key["a"].get(cont_a or "", cont_a) != self.bin_key["b"].get(
            cont_b or "", cont_b
        )
        refs_follow = a.bin_id == cont_a and b.bin_id == cont_b
        if moved:
            self.relocated.add(key)
            note = "" if b.bin_id == cont_b else f" (its MpFolder reference names {b.bin_id})"
            emit(
                "moved",
                "bin",
                f"moved from bin {_q(path_a)} to bin {_q(path_b)}{note}",
                path_a,
                path_b,
            )
        for f, (old, new) in diffs.items():
            if f in skip:
                continue
            if f == "name":
                emit("renamed", f, f"renamed {_q(old)} → {_q(new)}", old, new)
            elif f == "bin_id":
                if moved and refs_follow:
                    continue  # the move above, with the back-reference following it
                pa = self.bin_paths["a"].get(old or "", old)
                pb = self.bin_paths["b"].get(new or "", new)
                emit(
                    "changed",
                    f,
                    f"MpFolder reference {_q(pa)} → {_q(pb)} (listed in bin {_q(path_b)})",
                    pa,
                    pb,
                )
            elif f == "path":
                src = "" if b.path_source == "media_pool_blob" else f" [{b.path_source}]"
                emit("relinked", f, f"relinked {_q(old)} → {_q(new)}{src}", old, new)
            elif f == "blob_path":
                if "path" not in diffs:
                    emit("changed", f, f"media pool path {_q(old)} → {_q(new)}", old, new)
            elif f == "linked_paths":
                emit("changed", f, f"linked files {list(old)} → {list(new)}", old, new)
            elif f == "path_source":
                if "path" not in diffs:
                    emit("changed", f, f"path source {old} → {new}", old, new)
            elif f == "codec":
                emit("changed", f, f"codec {old} → {new}", old, new)
            elif f == "frame_rate":
                emit("changed", f, f"frame rate {_rate(old)} → {_rate(new)}", old, new)
            elif f == "resolution":
                emit("changed", f, f"resolution {_res(old)} → {_res(new)}", old, new)
            elif f == "frames":
                emit("changed", f, f"length {old} → {new} frames", old, new)
            elif f == "start_frame":
                emit("changed", f, f"start frame {old} → {new}", old, new)
            else:
                emit("changed", f, f"{_FIELD_LABELS.get(f, f)} {old} → {new}", old, new)

    # ------------------------------------------------------------------ timelines

    def _timeline(self, pair: _Pair[Timeline]) -> None:
        self._track_dbid(pair)
        tl = pair.b or pair.a
        assert tl is not None
        tkey = pair.key
        kind_word = tl.kind.capitalize()
        label = f"{kind_word} {_q(tl.name)}"
        ctx = Ctx(
            label + _idtag(tkey, pair.match),
            "timeline",
            tkey,
            pair.match,
            tkey,
            tl.name,
            obj_a=pair.a,
            obj_b=pair.b,
        )
        pool = {
            "a": self.media["a"].get(pair.a.media_clip_id or "") if pair.a else None,
            "b": self.media["b"].get(pair.b.media_clip_id or "") if pair.b else None,
        }
        pool_ctx = Ctx(
            f"{label} media pool entry{_idtag(tkey, pair.match)}",
            "timeline",
            tkey,
            pair.match,
            tkey,
            tl.name,
            obj_a=pool["a"],
            obj_b=pool["b"],
        )
        self.units.append(
            Unit(
                ("tlclip", tkey),
                "tlclip",
                pool_ctx,
                pool["a"].source if pool["a"] else None,
                pool["b"].source if pool["b"] else None,
                pool["a"],
                pool["b"],
            )
        )
        self.units.append(
            Unit(
                ("container", tkey),
                "container",
                Ctx(
                    f"{label} container{_idtag(tkey, pair.match)}",
                    "timeline",
                    tkey,
                    pair.match,
                    tkey,
                    tl.name,
                ),
                member_root(pair.a.member) if pair.a and pair.a.member else None,
                member_root(pair.b.member) if pair.b and pair.b.member else None,
            )
        )
        for side, t in (("a", pair.a), ("b", pair.b)):
            if t is not None and t.sequence_id:
                self.owners[side][t.sequence_id] = (
                    ("timeline", tkey),
                    Ctx(label, "timeline", tkey, pair.match, tkey, tl.name),
                )
        self._covers(("timeline", tkey), "timeline", pair.a, pair.b)
        self._covers(("tlclip", tkey), "media", pool["a"], pool["b"])

        if pair.a is None or pair.b is None:
            side = "b" if pair.a is None else "a"
            for track in tl.tracks:
                for item in track.items:
                    if item.id:
                        self.owners[side][item.id] = (("item", tkey, item.id), ctx)
            kind = "added" if pair.a is None else "removed"
            items = sum(len(t.items) for t in tl.tracks)
            markers = len(tl.markers) + sum(len(i.markers) for t in tl.tracks for i in t.items)
            detail = (
                f"{kind} ({_rate(tl.frame_rate)}, {_res(tl.resolution)}, "
                f"{len(tl.tracks)} track(s), "
                f"{items} item(s), {markers} marker(s))"
            )
            self.out.append(
                ctx.change(
                    "timelines",
                    kind,
                    detail,
                    old=tl.name if kind == "removed" else None,
                    new=tl.name if kind == "added" else None,
                )
            )
            return

        a, b = pair.a, pair.b
        tc_a, tc_b = _tc(a.frame_rate), _tc(b.frame_rate)
        va = {
            "name": a.name,
            "kind": a.kind,
            "frame_rate": a.frame_rate,
            "resolution": a.resolution,
            "extents": (a.start_seconds, a.duration_seconds),
        }
        vb = {
            "name": b.name,
            "kind": b.kind,
            "frame_rate": b.frame_rate,
            "resolution": b.resolution,
            "extents": (b.start_seconds, b.duration_seconds),
        }
        diffs = _diffs(va, vb)
        self._mark(("timeline", tkey), diffs)

        def emit(
            kind: str,
            f: str,
            detail: str,
            old: Any = None,
            new: Any = None,
            delta: dict[str, int] | None = None,
        ) -> None:
            self.out.append(
                ctx.change(
                    "timelines",
                    kind,
                    detail,
                    field_name=f,
                    old=_plain(old),
                    new=_plain(new),
                    delta=delta,
                )
            )

        for f, (old, new) in diffs.items():
            if f == "name":
                emit("renamed", f, f"renamed {_q(old)} → {_q(new)}", old, new)
            elif f == "frame_rate":
                emit("changed", f, f"frame rate {_rate(old)} → {_rate(new)}", old, new)
            elif f == "resolution":
                emit("changed", f, f"resolution {_res(old)} → {_res(new)}", old, new)
            elif f == "extents":
                if not same_value(a.start_seconds, b.start_seconds):
                    delta = None
                    if a.start_frame is not None and b.start_frame is not None:
                        delta = {"start": b.start_frame - a.start_frame}
                    emit(
                        "changed",
                        "start",
                        f"start {tc_a(a.start_frame)} → {tc_b(b.start_frame)}"
                        + (
                            f" ({_signed(delta['start'])})"
                            if delta and delta["start"]
                            else f" (seconds {a.start_seconds!r} → {b.start_seconds!r})"
                        ),
                        a.start_seconds,
                        b.start_seconds,
                        delta,
                    )
                if not same_value(a.duration_seconds, b.duration_seconds):
                    la = (
                        a.end_frame - a.start_frame
                        if a.end_frame is not None and a.start_frame is not None
                        else None
                    )
                    lb = (
                        b.end_frame - b.start_frame
                        if b.end_frame is not None and b.start_frame is not None
                        else None
                    )
                    delta = {"length": lb - la} if la is not None and lb is not None else None
                    emit(
                        "changed",
                        "length",
                        f"length {tc_a(la)} → {tc_b(lb)}"
                        + (
                            f" ({_signed(delta['length'])})"
                            if delta and delta["length"]
                            else f" (seconds {a.duration_seconds!r} → {b.duration_seconds!r})"
                        ),
                        a.duration_seconds,
                        b.duration_seconds,
                        delta,
                    )
            else:
                emit("changed", f, f"{_FIELD_LABELS.get(f, f)} {old} → {new}", old, new)

        for side, clip in (("a", pool["a"]), ("b", pool["b"])):
            if clip is not None:
                self.clip_key[side][clip.id] = f"timeline:{tkey}"
        self._pool_entry(pool_ctx, ("tlclip", tkey), pool["a"], pool["b"], a, b, "name" in diffs)
        track_keys = self._tracks(pair, label)
        self._items(pair, label, track_keys, tc_a, tc_b)
        mctx = Ctx(label, "timeline", tkey, pair.match, tkey, tl.name)
        self._markers(
            mctx, a.markers, b.markers, (a.start_frame or 0, tc_a), (b.start_frame or 0, tc_b)
        )

    def _pool_entry(
        self,
        ctx: Ctx,
        key: Key,
        ca: MediaClip | None,
        cb: MediaClip | None,
        ta: Timeline,
        tb: Timeline,
        renamed: bool,
    ) -> None:
        if ca is None and cb is None:
            return
        if ca is None or cb is None:
            kind = "added" if ca is None else "removed"
            self.out.append(ctx.change("timelines", kind, f"media pool entry {kind}"))
            return
        diffs = _diffs(_media_values(ca), _media_values(cb))
        self._mark(key, diffs)
        skip: frozenset[str] = frozenset()
        # The pool entry's name is the timeline's name; one rename is reported once.
        if renamed and "name" in diffs and (ca.name, cb.name) == (ta.name, tb.name):
            skip = frozenset({"name"})
        self._media_changes(ctx, key, "timelines", ca, cb, diffs, skip=skip)

    def _tracks(self, pair: _Pair[Timeline], label: str) -> dict[str, dict[int, str]]:
        assert pair.a is not None and pair.b is not None
        tkey = pair.key
        keys: dict[str, dict[int, str]] = {"a": {}, "b": {}}
        tpairs = _match(
            pair.a.tracks,
            pair.b.tracks,
            lambda t: t.id or None,
            lambda t: (t.type, t.index),
            self.duplicates,
            "track",
        )
        for tp in tpairs:
            self._track_dbid(tp)
            if tp.a is not None:
                keys["a"][id(tp.a)] = tp.key
            if tp.b is not None:
                keys["b"][id(tp.b)] = tp.key
            t = tp.b or tp.a
            assert t is not None
            ctx = Ctx(
                f"{label} track {t.label}{_idtag(tp.key, tp.match)}",
                "track",
                tp.key,
                tp.match,
                tkey,
                (pair.b or pair.a).name if (pair.b or pair.a) else None,
                obj_a=tp.a,
                obj_b=tp.b,
            )
            key: Key = ("track", tkey, tp.key)
            self.track_ctx[(tkey, tp.key)] = ctx
            self.units.append(
                Unit(
                    key, "track", ctx, tp.a.source if tp.a else None, tp.b.source if tp.b else None
                )
            )
            if tp.a is None or tp.b is None:
                kind = "added" if tp.a is None else "removed"
                self.out.append(
                    ctx.change(
                        "timelines",
                        kind,
                        f"{kind} ({len(t.items)} item(s))",
                        old=t.label if kind == "removed" else None,
                        new=t.label if kind == "added" else None,
                    )
                )
                continue
            self._covers(key, "track", tp.a, tp.b)
            va = {
                "name": tp.a.name,
                "type": tp.a.type,
                "label": tp.a.label,
                "type_code": tp.a.type_code,
                "subtype": tp.a.subtype,
                "flags": tp.a.flags,
            }
            vb = {
                "name": tp.b.name,
                "type": tp.b.type,
                "label": tp.b.label,
                "type_code": tp.b.type_code,
                "subtype": tp.b.subtype,
                "flags": tp.b.flags,
            }
            diffs = _diffs(va, vb)
            self._mark(key, diffs)
            if "type" in diffs:
                self.relocated.add(key)  # now in another track vector, reported as moved
            for f, (old, new) in diffs.items():
                if f == "label":
                    self.out.append(
                        ctx.change(
                            "timelines",
                            "moved",
                            f"moved {old} → {new}",
                            field_name=f,
                            old=old,
                            new=new,
                        )
                    )
                elif f == "name":
                    self.out.append(
                        ctx.change(
                            "timelines",
                            "renamed",
                            f"renamed {_q(old)} → {_q(new)}",
                            field_name=f,
                            old=old,
                            new=new,
                        )
                    )
                elif f == "type" and "label" in diffs:
                    continue
                else:
                    self.out.append(
                        ctx.change(
                            "timelines",
                            "changed",
                            f"{_FIELD_LABELS.get(f, f)} {old} → {new}",
                            field_name=f,
                            old=old,
                            new=new,
                        )
                    )
        return keys

    def _items(
        self,
        pair: _Pair[Timeline],
        label: str,
        track_keys: dict[str, dict[int, str]],
        tc_a: Callable[[int | None], str],
        tc_b: Callable[[int | None], str],
    ) -> None:
        assert pair.a is not None and pair.b is not None
        tkey = pair.key
        tname = pair.b.name
        placed_a = [_Placed(t, i) for t in pair.a.tracks for i in t.items]
        placed_b = [_Placed(t, i) for t in pair.b.tracks for i in t.items]
        index = {
            id(i): n for tl in (pair.a, pair.b) for t in tl.tracks for n, i in enumerate(t.items)
        }
        ipairs = _match(
            placed_a,
            placed_b,
            lambda p: p.item.id or None,
            lambda p: (p.item.kind, p.item.name, p.track.label),
            self.duplicates,
            "item",
        )
        stayed: dict[str, list[tuple[int, int, str, int | None, int | None]]] = {}
        for ip in ipairs:
            self._track_dbid(ip)
            p = ip.b or ip.a
            assert p is not None
            it = p.item
            ctx = Ctx(
                f"{label} {p.track.label} item {_q(it.name or it.pretty_type or it.kind)}"
                f"{_idtag(ip.key, ip.match)}",
                "item",
                ip.key,
                ip.match,
                tkey,
                tname,
                obj_a=ip.a.item if ip.a else None,
                obj_b=ip.b.item if ip.b else None,
            )
            key: Key = ("item", tkey, ip.key)
            for side, q in (("a", ip.a), ("b", ip.b)):
                if q is not None and q.item.id:
                    self.owners[side][q.item.id] = (key, ctx)
            self.units.append(
                Unit(
                    key,
                    "item",
                    ctx,
                    ip.a.item.source if ip.a else None,
                    ip.b.item.source if ip.b else None,
                )
            )
            if ip.a is None or ip.b is None:
                kind = "added" if ip.a is None else "removed"
                tc = tc_b if ip.a is None else tc_a
                media = self.media["b" if ip.a is None else "a"].get(it.media_ref or "")
                bits = [f"{kind} at {tc(it.record_start)}-{tc(it.record_end)}"]
                if (
                    it.source_in is not None
                    and it.source_out is not None
                    and it.kind != "transition"
                ):
                    bits.append(f"source {tc(it.source_in)}-{tc(it.source_out)}")
                if media is not None:
                    bits.append(f"media {_q(media.name)}")
                if it.markers:
                    bits.append(f"{len(it.markers)} marker(s)")
                rng = [it.record_start, it.record_end]
                self.out.append(
                    ctx.change(
                        "timelines",
                        kind,
                        ", ".join(bits),
                        old=rng if kind == "removed" else None,
                        new=rng if kind == "added" else None,
                    )
                )
                continue
            ia, ib = ip.a.item, ip.b.item
            self._covers(key, "item", ia, ib)
            diffs = _diffs(_item_values(ia), _item_values(ib))
            track_a = track_keys["a"].get(id(ip.a.track))
            track_b = track_keys["b"].get(id(ip.b.track))
            self._mark(key, diffs)
            if track_a == track_b and track_a is not None:
                stayed.setdefault(track_a, []).append(
                    (index[id(ia)], index[id(ib)], ip.key, ia.record_start, ib.record_start)
                )
            if track_a != track_b:
                self.relocated.add(key)
                self.out.append(
                    ctx.change(
                        "timelines",
                        "moved",
                        f"moved from track {ip.a.track.label} to {ip.b.track.label}",
                        field_name="track",
                        old=ip.a.track.label,
                        new=ip.b.track.label,
                    )
                )
            self._item_changes(ctx, ia, ib, diffs, tc_a, tc_b)
            self._markers(ctx, ia.markers, ib.markers, None, None)
        for track, rows in stayed.items():
            seq_a = [r[2] for r in sorted(rows, key=lambda r: r[0])]
            seq_b = [r[2] for r in sorted(rows, key=lambda r: r[1])]
            starts_a = [r[3] for r in sorted(rows, key=lambda r: r[0])]
            starts_b = [r[4] for r in sorted(rows, key=lambda r: r[1])]
            # Items written in timeline order reorder whenever one moves past another; that
            # is reported as the move. Only an order the positions do not imply is reported.
            if seq_a != seq_b and not (_ascending(starts_a) and _ascending(starts_b)):
                self.out.append(
                    self.track_ctx[(tkey, track)].change(
                        "timelines",
                        "reordered",
                        f"order of {len(seq_a)} item(s) in the track's Items changed",
                        field_name="item_order",
                        old=seq_a,
                        new=seq_b,
                    )
                )

    def _item_changes(
        self,
        ctx: Ctx,
        a: TimelineItem,
        b: TimelineItem,
        diffs: dict[str, tuple[Any, Any]],
        tc_a: Callable[[int | None], str],
        tc_b: Callable[[int | None], str],
    ) -> None:
        def emit(
            kind: str,
            f: str,
            detail: str,
            old: Any = None,
            new: Any = None,
            delta: dict[str, int] | None = None,
        ) -> None:
            self.out.append(
                ctx.change(
                    "timelines",
                    kind,
                    detail,
                    field_name=f,
                    old=_plain(old),
                    new=_plain(new),
                    delta=delta,
                )
            )

        timing = self._timing(a, b, tc_a, tc_b)
        if timing is not None:
            kind, detail, delta = timing
            emit(
                kind,
                "timing",
                detail,
                [a.record_start, a.record_end, a.source_in, a.source_out],
                [b.record_start, b.record_end, b.source_in, b.source_out],
                delta,
            )
        for f, (old, new) in diffs.items():
            if f in ("record_start", "duration", "source_in"):
                if not same_value(old[1], new[1]):
                    emit(
                        "changed",
                        f"{f}_fraction",
                        f"sub-frame remainder of {f} {old[1]} → {new[1]}",
                        old[1],
                        new[1],
                    )
                continue
            if f == "name":
                emit("renamed", f, f"renamed {_q(old)} → {_q(new)}", old, new)
            elif f == "media_ref":
                ma = self.media["a"].get(old or "")
                mb = self.media["b"].get(new or "")
                na = _q(ma.name) if ma else f"unresolved {old}"
                nb = _q(mb.name) if mb else f"unresolved {new}"
                emit("changed", f, f"source media {na} → {nb}", old, new)
            elif f == "media_file_path":
                emit("changed", f, f"timeline media path {_q(old)} → {_q(new)}", old, new)
            elif f == "speed":
                emit("changed", f, f"speed {old}x → {new}x", old, new)
            else:
                emit("changed", f, f"{_FIELD_LABELS.get(f, f)} {old} → {new}", old, new)

    @staticmethod
    def _timing(
        a: TimelineItem,
        b: TimelineItem,
        tc_a: Callable[[int | None], str],
        tc_b: Callable[[int | None], str],
    ) -> tuple[str, str, dict[str, int]] | None:
        """Classify a change of record and source range as moved, trimmed, slipped or other."""
        sa, sb, ea, eb = a.record_start, b.record_start, a.record_end, b.record_end
        src = all(v is not None for v in (a.source_in, b.source_in, a.source_out, b.source_out))
        if None in (sa, sb, ea, eb):
            same = (sa, ea, a.source_in) == (sb, eb, b.source_in)
            if same:
                return None
            return (
                "timing-changed",
                f"timing changed: record {tc_a(sa)}-{tc_a(ea)} → {tc_b(sb)}-{tc_b(eb)}, "
                f"source in {tc_a(a.source_in)} → {tc_b(b.source_in)}",
                {},
            )
        assert sa is not None and sb is not None and ea is not None and eb is not None
        ds, de = sb - sa, eb - ea
        dsi = dso = 0
        if src:
            assert a.source_in is not None and b.source_in is not None
            assert a.source_out is not None and b.source_out is not None
            dsi, dso = b.source_in - a.source_in, b.source_out - a.source_out
        elif a.source_in != b.source_in:
            return (
                "timing-changed",
                f"source in {tc_a(a.source_in)} → {tc_b(b.source_in)}",
                {"record_start": ds, "record_end": de},
            )
        if ds == de == dsi == dso == 0:
            return None
        if ds == de and dsi == dso == 0:
            return (
                "moved",
                f"moved {_signed(ds)} ({tc_a(sa)} → {tc_b(sb)})",
                {"record_start": ds, "record_end": de},
            )
        if (not src or (dsi == ds and dso == de)) and (ds or de):
            parts = []
            if ds:
                parts.append(f"head trimmed {_signed(ds)} ({tc_a(sa)} → {tc_b(sb)})")
            if de:
                parts.append(f"tail trimmed {_signed(de)} ({tc_a(ea)} → {tc_b(eb)})")
            delta = {"head": ds, "tail": de}
            if src:
                delta.update(source_in=dsi, source_out=dso)
                srcs = []
                if dsi:
                    srcs.append(f"in {_signed(dsi)} ({tc_a(a.source_in)} → {tc_b(b.source_in)})")
                if dso:
                    srcs.append(f"out {_signed(dso)} ({tc_a(a.source_out)} → {tc_b(b.source_out)})")
                parts.append("source " + ", ".join(srcs))
            return ("trimmed", "; ".join(parts), delta)
        if ds == de == 0 and dsi == dso:
            return (
                "slipped",
                f"slipped {_signed(dsi)}: source {tc_a(a.source_in)}-{tc_a(a.source_out)} → "
                f"{tc_b(b.source_in)}-{tc_b(b.source_out)}",
                {"source_in": dsi, "source_out": dso},
            )
        return (
            "timing-changed",
            f"timing changed: record {tc_a(sa)}-{tc_a(ea)} → {tc_b(sb)}-{tc_b(eb)} "
            f"(start {_signed(ds)}, end {_signed(de)}); "
            f"source in {_signed(dsi)}, out {_signed(dso)}",
            {"record_start": ds, "record_end": de, "source_in": dsi, "source_out": dso},
        )

    # ------------------------------------------------------------------ markers

    def _marker_order(
        self,
        owner: Ctx,
        ma: list[Marker],
        mb: list[Marker],
        pairs: list[tuple[Marker, Marker]],
        match: str,
    ) -> None:
        """Order of the markers kept on both sides, unless their frames imply it."""
        pos_a = {id(m): n for n, m in enumerate(ma)}
        pos_b = {id(m): n for n, m in enumerate(mb)}
        by_a = sorted(range(len(pairs)), key=lambda i: pos_a[id(pairs[i][0])])
        by_b = sorted(range(len(pairs)), key=lambda i: pos_b[id(pairs[i][1])])
        if by_a == by_b:
            return
        if _ascending([pairs[i][0].frame for i in by_a]) and _ascending(
            [pairs[i][1].frame for i in by_b]
        ):
            return
        self.out.append(
            Ctx(
                f"{owner.label} markers",
                "marker",
                owner.object_id,
                match,
                owner.timeline_id,
                owner.timeline_name,
            ).change(
                "markers",
                "reordered",
                f"order of {len(pairs)} marker(s) in the marker blob changed",
                field_name="marker_order",
                old=[pairs[i][0].frame for i in by_a],
                new=[pairs[i][1].frame for i in by_b],
            )
        )

    def _markers(
        self,
        owner: Ctx,
        ma: list[Marker],
        mb: list[Marker],
        base_a: tuple[int, Callable[[int | None], str]] | None,
        base_b: tuple[int, Callable[[int | None], str]] | None,
    ) -> None:
        """Markers of one owner present on both sides.

        Markers have no DbId: they are keyed by owner and frame (``match="owner+frame"`` when
        the owner was matched by DbId). Outside gate mode, a marker left over on each side
        with the same unique name, colour and note is paired as a move (``heuristic``).
        """
        if not ma and not mb:
            return
        keyed = "owner+frame" if owner.match == "dbid" else "heuristic"

        def where(m: Marker, base: tuple[int, Callable[[int | None], str]] | None) -> str:
            if base is None:
                return f"item frame {m.frame}"
            return f"{base[1](base[0] + m.frame)} (frame {m.frame})"

        def props(m: Marker) -> dict[str, Any]:
            return {
                "frame": m.frame,
                "color": m.color,
                "name": m.name,
                "note": m.note,
                "duration": m.duration,
                "custom_data": m.custom_data,
            }

        def ctx_for(
            m: Marker, base: tuple[int, Callable[[int | None], str]] | None, match: str = keyed
        ) -> Ctx:
            return Ctx(
                f"{owner.label} marker {_q(m.name)} at {where(m, base)}",
                "marker",
                owner.object_id,
                match,
                owner.timeline_id,
                owner.timeline_name,
            )

        def by_frame(ms: list[Marker]) -> dict[tuple[int, int], Marker]:
            out: dict[tuple[int, int], Marker] = {}
            seen: Counter[int] = Counter()
            for m in ms:
                seen[m.frame] += 1
                out[(m.frame, seen[m.frame])] = m
            return out

        fa, fb = by_frame(ma), by_frame(mb)
        pairs: list[tuple[Marker, Marker]] = [(fa[k], fb[k]) for k in fa if k in fb]
        left_a = [m for k, m in fa.items() if k not in fb]
        left_b = [m for k, m in fb.items() if k not in fa]

        def ident(m: Marker) -> tuple[str, str | None, str]:
            return (m.name, m.color, m.note)

        count_a = Counter(ident(m) for m in left_a)
        count_b = Counter(ident(m) for m in left_b)
        unique_b = {ident(m): m for m in left_b if count_b[ident(m)] == 1}
        moved: list[tuple[Marker, Marker]] = []
        for m in [] if self.gate else list(left_a):
            other = unique_b.get(ident(m))
            if count_a[ident(m)] == 1 and other is not None:
                moved.append((m, other))
                left_a.remove(m)
                left_b.remove(other)

        for m in left_a:
            self.out.append(
                ctx_for(m, base_a).change(
                    "markers",
                    "removed",
                    f"removed ({_marker_desc(m)})",
                    old=props(m),
                    where_a=m.source,
                    where_b=None,
                )
            )
        for m in left_b:
            self.out.append(
                ctx_for(m, base_b).change(
                    "markers",
                    "added",
                    f"added ({_marker_desc(m)})",
                    new=props(m),
                    where_a=None,
                    where_b=m.source,
                )
            )
        self._marker_order(owner, ma, mb, pairs + moved, keyed)
        heuristic = {id(x) for x, _ in moved}
        for x, y in pairs + moved:
            pa, pb = props(x), props(y)
            diffs = _diffs(pa, pb)
            if not diffs:
                continue
            bits = []
            delta = None
            kind = "changed"
            if "frame" in diffs:
                kind = "moved"
                delta = {"frame": y.frame - x.frame}
                bits.append(
                    f"moved {_signed(delta['frame'])} ({where(x, base_a)} → {where(y, base_b)})"
                )
            for f, (old, new) in diffs.items():
                if f != "frame":
                    bits.append(f"{f} {_shown(old)} → {_shown(new)}")
            match = "heuristic" if id(x) in heuristic else keyed
            self.out.append(
                ctx_for(y, base_b, match).change(
                    "markers",
                    kind,
                    "; ".join(bits),
                    field_name=",".join(diffs),
                    old=pa,
                    new=pb,
                    delta=delta,
                    where_a=x.source,
                    where_b=y.source,
                )
            )


def _ascending(values: list[int | None]) -> bool:
    nums = [-1 if v is None else v for v in values]
    return all(x <= y for x, y in itertools.pairwise(nums))


def _shown(value: Any) -> Any:
    return _q(value) if isinstance(value, str) else value


def _marker_desc(m: Marker) -> str:
    bits = [m.color or "no colour"]
    if m.note:
        bits.append(f"note {_q(m.note)}")
    if m.duration and m.duration != 1:
        bits.append(f"duration {m.duration}")
    return ", ".join(bits)
