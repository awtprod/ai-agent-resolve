"""Byte-level comparison behind :mod:`aar.drp.diff`: nothing is called equal unread.

The semantic layer only sees what the reader decodes. This module re-reads both files and
compares everything else, so a difference is never hidden because it is not decoded:

* Members the reader does not interpret (``Gallery.xml``, stills, ``.dat``) are compared
  whole. Members present on one side only are reported, directory entries included, and so is
  a zip member name that occurs more than once (only its last copy can be read).
* Every interpreted XML member that differs is scanned again, prolog and trailing content
  included. Model objects (project, bins, media clips, timeline containers, tracks, items,
  marker blobs) are "units": they are paired through the semantic layer's DbId matches, so an
  object that moved to another member is still compared with its counterpart. Inside a unit,
  children are aligned by unit, by the DbId of ``<Element>`` vector entries, otherwise by tag
  and position. Nested units are compared on their own, but where they sit is checked here:
  a unit under a different parent must have been reported as a move by the semantic layer,
  and an ``<Element>`` wrapper around a unit is compared too (attributes, stray text).
* A leaf whose bytes differ is accounted for only when (a) it is the text of a decoded field
  that the semantic layer reports and that leaf's own decoded value differs, or (b) it is a
  blob and the bytes left after removing exactly what the reader decoded are the same on both
  sides. Any other difference is ``opaque-changed`` with member, byte span and byte count.

Order: children are compared in order, except (1) inside a ``...Set`` element, every child
that carries a DbId (directly or in its ``<Element>`` wrapper) is matched by DbId and is
unordered, across tags: Resolve writes such sets in hash order, which differs between its own
exports. Children without a DbId keep their order relative to each other. (2) Units inside
``Items``, ``MediaVec`` and ``...TrackVec``, whose order the semantic layer reports (items and
bin contents reordered, track labels). An element with a DbId that leaves one parent for
another is reported on both sides as moved. Not compared, by design: whitespace between
elements and the zip container's own metadata.

Known limit: a blob is compared on its decoded content where the reader decodes it, so two
zstd frames that differ only in framing (for example the last-block bit) but decompress to the
same bytes are reported as equal.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aar.drp import blobs
from aar.drp.container import Container, open_container, parse_version_header
from aar.drp.diff_model import REPORTED_FIELDS, Change, Ctx, DiffError, same_value
from aar.drp.errors import BlobError, ContainerError
from aar.drp.model import DrpFile, MediaClip, Provenance
from aar.drp.reader import MARKER_COLORS, join_clip_path
from aar.drp.scan import Document, Element, scan

__all__ = ["Cover", "OpaqueEngine", "Source", "Unit", "member_root"]

_CARRIER_TAGS = frozenset({"Sm2SequenceLockableBlob", "Sm2TiItemLockableBlob"})
_CARRIER_CHILDREN = frozenset({"FieldsBlob", "BlobOwner", "DbSavedTime"})
_INFO_TAGS = frozenset({"BtVideoInfo", "BtAudioInfo"})
_INFO_SKIP = frozenset({"Sequence", "TimelineSharedHandle"})
_SEMANTIC_ORDER = frozenset({"Items", "MediaVec"})
_INDEX = re.compile(r"\[\d+\]$")
_INDEXED = re.compile(r"^(\w+)\[(\d+)\]$")
_FRAME_FIELDS = frozenset({"record_start", "duration", "source_in"})
_FRAME_RATE_HINT = re.compile(r"\d+(\.\d+)?")
_PATH_SEP = re.compile(r"[\\/]")


def member_root(member: str) -> Provenance:
    """Placeholder provenance meaning "the root element of ``member``"."""
    return Provenance(member, 0, 0, "member-root")


@dataclass(frozen=True, slots=True)
class Cover:
    """A decoded field whose source is one leaf element.

    ``mode`` says how the leaf was decoded: ``text`` (its whole text is the value), or a blob
    layout whose decoded parts can be masked out (``rate``, ``fixed``, ``keyed``, ``setup``,
    ``clip``, ``markers``). ``arg`` is mode-specific (the keyed-dict key, clip path list...).
    ``value`` is the value decoded from this very leaf (one entry of a list field).
    """

    unit: tuple[Any, ...]
    field: str
    mode: str
    arg: Any = None
    value: Any = None


@dataclass(slots=True)
class Unit:
    """A model object as an element on each side (``None`` where it does not exist)."""

    key: tuple[Any, ...]
    kind: str
    ctx: Ctx
    a: Provenance | None
    b: Provenance | None
    obj_a: Any = None
    obj_b: Any = None


@dataclass(slots=True)
class _Child:
    """A child element as aligned: a unit, an ``<Element>`` wrapping a unit, or plain."""

    el: Element
    unit: Unit | None
    wrapper: bool
    tag: str


class Source:
    """Raw member access for one side, checked against the hash the reader recorded.

    A member that cannot be read raises :class:`DiffError`: an unreadable input is an error,
    never a difference.
    """

    def __init__(self, drp: DrpFile, side: str = "A") -> None:
        self.drp = drp
        self.side = side
        self._raw: dict[str, bytes] = {}
        self._docs: dict[str, Document] = {}
        try:
            self.container: Container = open_container(drp.path)
        except ContainerError as exc:
            raise DiffError(f"{side}: cannot reopen {drp.path}: {exc}") from exc
        if self.container.sha256 != drp.sha256:
            self.container.close()
            raise DiffError(f"{side}: {drp.path} changed on disk after it was read; read it again")

    def data(self, name: str) -> bytes:
        if name not in self._raw:
            try:
                self._raw[name] = self.container.read(name)
            except ContainerError as exc:
                raise DiffError(f"{self.side}: member {name} cannot be read: {exc}") from exc
        return self._raw[name]

    def doc(self, name: str) -> Document:
        if name not in self._docs:
            self._docs[name] = scan(self.data(name), name)
        return self._docs[name]

    def whole(self) -> bytes:
        try:
            return Path(self.drp.path).read_bytes()
        except OSError as exc:
            raise DiffError(f"{self.side}: cannot read {self.drp.path}: {exc}") from exc

    def close(self) -> None:
        self.container.close()


Span = tuple[str, int, int]


def _span(member: str, el: Element) -> Span:
    return (member, el.start, el.end)


def _base_field(key: str) -> str:
    return _INDEX.sub("", key)


def _span_value(obj: Any, key: str) -> Any:
    """The decoded value that came from the span ``key`` of ``obj`` (one list entry if indexed)."""
    m = _INDEXED.match(key)
    if m:
        seq = getattr(obj, m.group(1), None)
        i = int(m.group(2))
        return seq[i] if isinstance(seq, list | tuple) and i < len(seq) else None
    value = getattr(obj, key, None)
    fractions = getattr(obj, "fractions", None)
    if key in _FRAME_FIELDS and isinstance(fractions, dict):
        return (value, fractions.get(key))
    return value


class OpaqueEngine:
    """Compare two sources member by member and unit by unit; see the module docstring.

    ``relocated`` holds the keys of units the semantic layer reported as moved to another
    parent (an item to another track, a clip to another bin); only those may change parent.
    """

    def __init__(
        self,
        sources: tuple[Source, Source],
        units: list[Unit],
        cover_objects: dict[str, list[tuple[tuple[Any, ...], str, Any]]],
        changed: set[tuple[tuple[Any, ...], str]],
        owners: dict[str, dict[str, tuple[tuple[Any, ...], Ctx]]],
        relocated: set[tuple[Any, ...]] | None = None,
    ) -> None:
        self.src = {"a": sources[0], "b": sources[1]}
        self.drp = {"a": sources[0].drp, "b": sources[1].drp}
        self.units = list(units)
        self.cover_objects = cover_objects
        self.changed = changed
        self.owners = owners
        self.relocated = relocated or set()
        self.out: list[Change] = []
        self.covers: dict[str, dict[Span, list[Cover]]] = {"a": {}, "b": {}}
        self.unit_at: dict[str, dict[Span, Unit]] = {"a": {}, "b": {}}
        self.root_units: dict[str, set[str]] = {"a": set(), "b": set()}
        self.elements: dict[str, dict[Span, Element]] = {"a": {}, "b": {}}
        self.skip: set[int] = set()
        self._same: dict[str, bool] = {}
        # DbId -> indexes in ``out`` of elements with that DbId present under one parent only
        self._lone_dbids: dict[str, dict[str, list[int]]] = {"a": {}, "b": {}}

    # ------------------------------------------------------------------ setup

    def run(self) -> list[Change]:
        a, b = self.drp["a"], self.drp["b"]
        if not a.is_zip or not b.is_zip:
            self._whole_file()
            return self.out
        self._duplicate_members()
        self._carriers()
        self._resolve_units()
        self._text_covers()
        self._members()
        for unit in self.units:
            if id(unit) in self.skip:
                continue
            if unit.a is not None and unit.b is not None:
                self._walk_unit(unit)
            elif unit.kind == "markers":
                self._lone_carrier(unit)
        self._mark_moves()
        return self.out

    def _mark_moves(self) -> None:
        """An element with a DbId that is gone from one parent and new under another moved."""
        for dbid in sorted(set(self._lone_dbids["a"]) & set(self._lone_dbids["b"])):
            for side in ("a", "b"):
                for i in self._lone_dbids[side][dbid]:
                    c = self.out[i]
                    self.out[i] = dataclasses.replace(
                        c,
                        detail=c.detail + "; moved: the same DbId sits under another parent "
                        "element on the other side",
                    )

    def _carriers(self) -> None:
        """Marker blobs live in ``project.xml``; pair them by the owner they belong to."""
        by_key: dict[tuple[Any, ...], Unit] = {}
        if self._identical("project.xml"):
            return  # byte-identical: no marker blob can differ
        for side in ("a", "b"):
            if not self._has("project.xml", side):
                continue
            doc = self.src[side].doc("project.xml")
            if doc.root is None:
                continue
            seen: Counter[tuple[Any, ...]] = Counter()
            for el in doc.root.iter():
                if el.tag not in _CARRIER_TAGS or el.find("BlobOwner") is None:
                    continue
                owner = (el.child_text("BlobOwner") or "").strip()
                found = self.owners[side].get(owner)
                if found is not None:
                    owner_key, owner_ctx = found
                    key: tuple[Any, ...] = ("markers", *owner_key)
                    ctx = Ctx(
                        f"{owner_ctx.label} marker blob",
                        "markers",
                        owner_ctx.object_id,
                        owner_ctx.match,
                        owner_ctx.timeline_id,
                        owner_ctx.timeline_name,
                    )
                else:
                    key = ("markers", "orphan", owner)
                    ctx = Ctx(f"Marker blob of unknown owner {owner}", "markers", owner, "dbid")
                seen[key] += 1
                if seen[key] > 1:
                    key = (*key, seen[key])
                prov = Provenance("project.xml", el.start, el.end, "element")
                unit = by_key.get(key)
                if unit is None:
                    unit = Unit(key, "markers", ctx, None, None)
                    by_key[key] = unit
                    self.units.append(unit)
                if side == "a":
                    unit.a = prov
                else:
                    unit.b = prov
                fb = el.find("FieldsBlob")
                if fb is not None and fb.inner_start is not None:
                    span = ("project.xml", fb.inner_start, fb.inner_end or fb.inner_start)
                    self.covers[side][span] = [Cover(key, "markers", "markers")]

    def _has(self, member: str, side: str) -> bool:
        return any(m.name == member for m in self.drp[side].members)

    def _identical(self, member: str) -> bool:
        """The member exists on both sides with the same bytes (nothing in it can differ)."""
        if member not in self._same:
            if not (self._has(member, "a") and self._has(member, "b")):
                self._same[member] = False
            else:
                self._same[member] = self.src["a"].data(member) == self.src["b"].data(member)
        return self._same[member]

    def _resolve_units(self) -> None:
        """Index unit spans; scan only members whose units actually need comparing.

        A paired unit at the same span of a byte-identical member is identical, so it is
        skipped without scanning anything.
        """
        wanted: dict[str, dict[str, set[tuple[int, int]]]] = {"a": {}, "b": {}}
        for unit in self.units:
            pa, pb = unit.a, unit.b
            if (
                pa is not None
                and pb is not None
                and pa.member == pb.member
                and (pa.kind, pa.start, pa.end) == (pb.kind, pb.start, pb.end)
                and self._identical(pa.member)
            ):
                self.skip.add(id(unit))
            paired = pa is not None and pb is not None and id(unit) not in self.skip
            needs_element = paired or (unit.kind == "markers" and (pa is None) != (pb is None))
            for side, prov in (("a", pa), ("b", pb)):
                if prov is None:
                    continue
                if prov.kind == "member-root":
                    self.root_units[side].add(prov.member)
                    if not needs_element:
                        continue
                    root = self.src[side].doc(prov.member).root
                    resolved = (
                        Provenance(prov.member, root.start, root.end, "element") if root else None
                    )
                    if side == "a":
                        unit.a = resolved
                    else:
                        unit.b = resolved
                    if resolved is None:
                        continue
                    prov = resolved
                self.unit_at[side][(prov.member, prov.start, prov.end)] = unit
                if needs_element:
                    wanted[side].setdefault(prov.member, set()).add((prov.start, prov.end))
        for side, members in wanted.items():
            for member, spans in members.items():
                doc = self.src[side].doc(member)
                if doc.root is None:
                    continue
                for el in doc.root.iter():
                    if (el.start, el.end) in spans:
                        self.elements[side][(member, el.start, el.end)] = el

    def _text_covers(self) -> None:
        for side, objects in self.cover_objects.items():
            for key, kind, obj in objects:
                reported = REPORTED_FIELDS.get(kind, frozenset())
                for span_key, prov in obj.spans.items():
                    name = _base_field(span_key)
                    if name not in reported:
                        continue
                    cover = _cover_for(key, name, prov, _span_value(obj, span_key))
                    if cover is not None:
                        self.covers[side].setdefault(
                            (prov.member, prov.start, prov.end), []
                        ).append(cover)

    def _clip_covers(self, unit: Unit) -> None:
        """Every ``Clip`` blob of a media clip: the reader joins its path and maybe its codec.

        Which stream's blob holds which path is checked through the residue (an index into
        the clip's path list) unless the path list itself changed, which is reported.
        """
        changed = any((unit.key, f) in self.changed for f in ("path", "blob_path", "linked_paths"))
        for side, prov, clip in (("a", unit.a, unit.obj_a), ("b", unit.b, unit.obj_b)):
            if prov is None or not isinstance(clip, MediaClip):
                continue
            el = self.elements[side].get((prov.member, prov.start, prov.end))
            if el is None:
                continue
            paths = (clip.blob_path, *clip.linked_paths)
            codec = clip.spans.get("codec")
            for node in _iter_skipping(el, _INFO_SKIP):
                if node.tag not in _INFO_TAGS:
                    continue
                blob = node.find("Clip")
                if blob is None or blob.inner_start is None or blob.inner_end is None:
                    continue
                span = (prov.member, blob.inner_start, blob.inner_end)
                takes_codec = codec is not None and (codec.start, codec.end) == span[1:]
                self.covers[side][span] = [
                    Cover(unit.key, "paths", "clip", (paths, takes_codec, changed))
                ]

    # ------------------------------------------------------------------ members

    def _whole_file(self) -> None:
        da, db = self.src["a"].whole(), self.src["b"].whole()
        if da == db:
            return
        ctx = Ctx("File", "file", None, "file")
        count, first = _diff_size(da, db)
        self.out.append(
            ctx.change(
                "opaque",
                "opaque-changed",
                f"whole file compared as bytes (not a zip on both sides): {count} bytes differ"
                f" from byte {first}",
                field_name="file",
                where_a=Provenance(self.drp["a"].path, 0, len(da), "file"),
                where_b=Provenance(self.drp["b"].path, 0, len(db), "file"),
                byte_count=count,
            )
        )

    def _duplicate_members(self) -> None:
        """A zip member name that occurs twice: only the last copy is read, the rest unseen."""
        ca = Counter(m.name for m in self.drp["a"].members)
        cb = Counter(m.name for m in self.drp["b"].members)
        for name in sorted(n for n in set(ca) | set(cb) if ca[n] > 1 or cb[n] > 1):
            self.out.append(
                _member_ctx(name).change(
                    "opaque",
                    "opaque-changed",
                    f"zip member name occurs {ca[name]} time(s) in A and {cb[name]} in B; only "
                    "the last copy is read, so the others were not compared",
                    field_name=name,
                    old=ca[name],
                    new=cb[name],
                    where_a=Provenance(name, 0, 0, "member"),
                    where_b=Provenance(name, 0, 0, "member"),
                )
            )

    def _members(self) -> None:
        a_members = {m.name: m for m in self.drp["a"].members}
        b_members = {m.name: m for m in self.drp["b"].members}
        for name in sorted(set(a_members) | set(b_members)):
            ma, mb = a_members.get(name), b_members.get(name)
            if ma is not None and mb is not None:
                if ma.role == "directory" or mb.role == "directory":
                    if ma.role != mb.role:
                        self._member_change(name, "a directory entry on one side only", 0)
                    continue
                if ma.interpreted and mb.interpreted:
                    self._xml_member(name)
                else:
                    self._raw_member(name)
                continue
            side = "a" if ma is not None else "b"
            m = ma if ma is not None else mb
            assert m is not None
            if m.role != "directory" and m.interpreted:
                if name in self.root_units[side]:
                    continue  # a model object reported (or paired) by the semantic layer
                root = self.src[side].doc(name).root
                if root is not None and self._is_unit(side, name, root):
                    continue
            size = 0 if m.role == "directory" else m.size
            what = "zip directory entry" if m.role == "directory" else "member"
            prov = Provenance(name, 0, size, "member")
            self.out.append(
                _member_ctx(name).change(
                    "opaque",
                    "opaque-removed" if side == "a" else "opaque-added",
                    f"{what} {'removed' if side == 'a' else 'added'} ({size} bytes)",
                    field_name=name,
                    where_a=prov if side == "a" else None,
                    where_b=prov if side == "b" else None,
                    byte_count=size,
                )
            )

    def _is_unit(self, side: str, member: str, el: Element) -> bool:
        return _span(member, el) in self.unit_at[side]

    def _member_change(self, name: str, why: str, count: int | None) -> None:
        self.out.append(
            _member_ctx(name).change(
                "opaque",
                "opaque-changed",
                why,
                field_name=name,
                where_a=Provenance(name, 0, 0, "member"),
                where_b=Provenance(name, 0, 0, "member"),
                byte_count=count,
            )
        )

    def _raw_member(self, name: str) -> None:
        da, db = self.src["a"].data(name), self.src["b"].data(name)
        if da == db:
            return
        count, first = _diff_size(da, db)
        self.out.append(
            _member_ctx(name).change(
                "opaque",
                "opaque-changed",
                f"member not interpreted, compared as bytes: {count} bytes differ "
                f"from byte {first}",
                field_name=name,
                where_a=Provenance(name, 0, len(da), "member"),
                where_b=Provenance(name, 0, len(db), "member"),
                byte_count=count,
            )
        )

    def _xml_member(self, name: str) -> None:
        if self._identical(name):
            return
        da, db = self.src["a"].doc(name), self.src["b"].doc(name)
        ra, rb = da.root, db.root
        self._prolog(name, da, db)
        if ra is None or rb is None:
            if da.data != db.data:
                count, _ = _diff_size(da.data, db.data)
                self._member_change(name, "has no root element on one side; bytes differ", count)
            return
        tail_a, tail_b = da.data[ra.end :].strip(), db.data[rb.end :].strip()
        if tail_a != tail_b:
            count, _ = _diff_size(tail_a, tail_b)
            self._member_change(name, "content after the root element changed", count)
        ua = name in self.root_units["a"] or self._is_unit("a", name, ra)
        ub = name in self.root_units["b"] or self._is_unit("b", name, rb)
        unit = Unit(("member", name), "member", _member_ctx(name), None, None)
        if not ua and not ub:
            self._walk(ra, name, rb, name, unit, root=False)
            return
        if not ua:
            self._lone(ra, name, "a", unit, (ra.tag,))
        if not ub:
            self._lone(rb, name, "b", unit, (rb.tag,))

    def _prolog(self, name: str, da: Document, db: Document) -> None:
        stamps = []
        cleaned = []
        for doc in (da, db):
            end = doc.root.start if doc.root is not None else len(doc.data)
            head = doc.data[:end]
            v = parse_version_header(head, name)
            if v is not None:
                head = head[: v.offset] + head[v.end :]
                stamps.append((v.app_version, v.project_version))
            else:
                stamps.append(None)
            cleaned.append(b"".join(head.split()))
        if cleaned[0] != cleaned[1]:
            count, _ = _diff_size(cleaned[0], cleaned[1])
            self._member_change(name, "XML prolog changed (outside the version stamp)", count)
        main = []
        for side in ("a", "b"):
            v = self.drp[side].versions
            main.append((v.app_version, v.project_version) if v else None)
        if stamps[0] != stamps[1] and (stamps[0], stamps[1]) != (main[0], main[1]):
            ctx = Ctx(f"Header member {name}", "member", name, "member")
            self.out.append(
                ctx.change(
                    "header",
                    "changed",
                    f"version stamp {_stamp(stamps[0])} → {_stamp(stamps[1])} "
                    "(differs from the file's main stamp change)",
                    field_name="version_stamp",
                    old=list(stamps[0]) if stamps[0] else None,
                    new=list(stamps[1]) if stamps[1] else None,
                )
            )

    # ------------------------------------------------------------------ walking

    def _walk_unit(self, unit: Unit) -> None:
        assert unit.a is not None and unit.b is not None
        ea = self.elements["a"].get((unit.a.member, unit.a.start, unit.a.end))
        eb = self.elements["b"].get((unit.b.member, unit.b.start, unit.b.end))
        if ea is None or eb is None:
            self.out.append(
                unit.ctx.change(
                    "opaque",
                    "opaque-changed",
                    "element could not be located again, so its raw content was not compared",
                    field_name=None,
                    where_a=unit.a,
                    where_b=unit.b,
                )
            )
            return
        if unit.kind in ("media", "tlclip"):
            self._clip_covers(unit)
        self._walk(ea, unit.a.member, eb, unit.b.member, unit, root=True)

    def _walk(self, ea: Element, ma: str, eb: Element, mb: str, unit: Unit, *, root: bool) -> None:
        stack: list[tuple[Element, Element, tuple[str, ...], bool]] = [(ea, eb, (), root)]
        while stack:
            ca, cb, path, is_root = stack.pop()
            pushed = self._compare(ca, ma, cb, mb, unit, path, is_root)
            stack.extend(reversed(pushed))

    def _data(self, side: str, member: str) -> bytes:
        return self.src[side].doc(member).data

    def _compare(
        self,
        ea: Element,
        ma: str,
        eb: Element,
        mb: str,
        unit: Unit,
        path: tuple[str, ...],
        root: bool,
    ) -> list[tuple[Element, Element, tuple[str, ...], bool]]:
        here = path or (ea.tag,)
        if ea.tag != eb.tag:
            self._changed(unit, here, ma, ea, mb, eb, f"element <{ea.tag}> became <{eb.tag}>")
            return []
        self._attrs(unit, here, ma, ea, mb, eb, drop_dbid=root)
        if not ea.children and not eb.children:
            self._leaf(ea, ma, eb, mb, unit, here)
            return []
        self._loose(unit, here, ma, ea, mb, eb, "text outside child elements changed")
        ka, kb = self._keyed(ea, ma, "a"), self._keyed(eb, mb, "b")
        seq_a = [o for k, c in ka.items() if k in kb and (o := _order_key(ea.tag, k, c))]
        seq_b = [o for k, c in kb.items() if k in ka and (o := _order_key(eb.tag, k, c))]
        if seq_a != seq_b:
            self._changed(unit, here, ma, ea, mb, eb, "order of child elements changed")
        pushed = []
        for key, ca in ka.items():
            cb = kb.get(key)
            seg = (*path, _segment(key, ca))
            if cb is None:
                self._lone_child(ca, ma, "a", unit, seg)
            elif ca.unit is not None:
                self._unit_child(ca, ma, cb, mb, unit, seg)
            else:
                pushed.append((ca.el, cb.el, seg, False))
        for key, cb in kb.items():
            if key not in ka:
                self._lone_child(cb, mb, "b", unit, (*path, _segment(key, cb)))
        return pushed

    def _attrs(
        self,
        unit: Unit,
        path: tuple[str, ...],
        ma: str,
        ea: Element,
        mb: str,
        eb: Element,
        *,
        drop_dbid: bool,
    ) -> None:
        attrs_a, attrs_b = dict(ea.attrs), dict(eb.attrs)
        if drop_dbid:  # a unit's identity is its match, reported by the semantic layer
            attrs_a.pop("DbId", None)
            attrs_b.pop("DbId", None)
        if attrs_a != attrs_b:
            names = sorted(
                k for k in set(attrs_a) | set(attrs_b) if attrs_a.get(k) != attrs_b.get(k)
            )
            tag_a = self._data("a", ma)[ea.start : ea.inner_start or ea.end]
            tag_b = self._data("b", mb)[eb.start : eb.inner_start or eb.end]
            self._changed(
                unit,
                path,
                ma,
                ea,
                mb,
                eb,
                f"attribute(s) {', '.join(names)} changed",
                part=(tag_a, tag_b),
            )

    def _loose(
        self,
        unit: Unit,
        path: tuple[str, ...],
        ma: str,
        ea: Element,
        mb: str,
        eb: Element,
        why: str,
    ) -> None:
        la, lb = _loose_text(ea, self._data("a", ma)), _loose_text(eb, self._data("b", mb))
        if la != lb:
            self._changed(unit, path, ma, ea, mb, eb, why, part=(la, lb))

    def _keyed(self, el: Element, member: str, side: str) -> dict[tuple[Any, ...], _Child]:
        units = self.unit_at[side]
        out: dict[tuple[Any, ...], _Child] = {}
        counts: Counter[Any] = Counter()
        for child in el.children:
            inner = (
                child.children[0] if child.tag == "Element" and len(child.children) == 1 else None
            )
            u = units.get((member, child.start, child.end))
            w = units.get((member, inner.start, inner.end)) if inner is not None else None
            key: tuple[Any, ...]
            if u is not None:
                key, entry = ("unit", u.key), _Child(child, u, False, child.tag)
            elif inner is not None and w is not None:
                key, entry = ("unit", w.key), _Child(child, w, True, inner.tag)
            elif inner is not None and inner.attrs.get("DbId"):
                key = ("Element", inner.tag, inner.attrs["DbId"])
                entry = _Child(child, None, False, inner.tag)
            elif el.tag.endswith("Set") and child.attrs.get("DbId"):
                key = ("dbid", child.tag, child.attrs["DbId"])
                entry = _Child(child, None, False, child.tag)
            else:
                counts[child.tag] += 1
                key, entry = (child.tag, counts[child.tag]), _Child(child, None, False, child.tag)
            if key in out:  # the same DbId twice among siblings
                counts[key] += 1
                key = (*key, counts[key])
            out[key] = entry
        return out

    def _unit_child(
        self, ca: _Child, ma: str, cb: _Child, mb: str, unit: Unit, path: tuple[str, ...]
    ) -> None:
        """The same unit under both parents: compare its ``<Element>`` wrapper, not the unit."""
        if ca.wrapper != cb.wrapper:
            self._changed(unit, path, ma, ca.el, mb, cb.el, "a model object is wrapped differently")
        elif ca.wrapper:
            self._attrs(unit, path, ma, ca.el, mb, cb.el, drop_dbid=False)
            self._loose(
                unit, path, ma, ca.el, mb, cb.el, "text inside the <Element> wrapper changed"
            )

    def _lone_child(
        self, child: _Child, member: str, side: str, unit: Unit, path: tuple[str, ...]
    ) -> None:
        if child.unit is None:
            before = len(self.out)
            self._lone(child.el, member, side, unit, path)
            dbid = _dbid_of(child)
            if dbid is not None and len(self.out) > before:
                self._lone_dbids[side].setdefault(dbid, []).append(before)
            return
        u = child.unit
        one_sided = u.a is None or u.b is None
        extra = child.wrapper and (
            bool(child.el.attrs) or bool(_loose_text(child.el, self._data(side, member)))
        )
        if not one_sided and u.key not in self.relocated:
            self._presence(
                unit,
                path,
                member,
                child.el,
                side,
                "a model object now sits under another parent element, which the semantic "
                "layer did not report as a move",
            )
        elif extra:
            self._presence(
                unit,
                path,
                member,
                child.el,
                side,
                "the <Element> wrapper holds more than the object",
            )

    def _covers(self, side: str, member: str, el: Element) -> list[Cover]:
        found: list[Cover] = []
        if el.inner_start is not None and el.inner_end is not None:
            found += self.covers[side].get((member, el.inner_start, el.inner_end), [])
        found += self.covers[side].get((member, el.start, el.end), [])
        return found

    def _leaf(
        self, ea: Element, ma: str, eb: Element, mb: str, unit: Unit, path: tuple[str, ...]
    ) -> None:
        ia, ib = ea.inner_raw, eb.inner_raw
        if ia == ib:
            return
        ca, cb = self._covers("a", ma, ea), self._covers("b", mb, eb)
        if ca and cb and self._explained(ca, cb, ia, ib):
            return
        if ca and cb and {c.mode for c in ca + cb} == {"text"}:
            why = "same decoded value, different text"
        elif ca or cb:
            why = "parts the reader does not decode changed"
        else:
            why = "not decoded"
        self._changed(unit, path, ma, ea, mb, eb, why)

    def _explained(self, ca: list[Cover], cb: list[Cover], ia: bytes, ib: bytes) -> bool:
        modes = {c.mode for c in ca} | {c.mode for c in cb}
        if modes == {"text"}:
            # Only this leaf: the field must be reported as changed *and* the value decoded
            # from this very leaf must differ (one entry of a list, not just the list).
            va = {(c.unit, c.field): c.value for c in ca}
            vb = {(c.unit, c.field): c.value for c in cb}
            return va.keys() == vb.keys() and all(
                k in self.changed and not same_value(va[k], vb[k]) for k in va
            )
        if len(modes) != 1:
            return False
        mode = modes.pop()
        ra, rb = _residue(mode, ia, ca), _residue(mode, ib, cb)
        return ra is not None and rb is not None and _same_residue(ra, rb)

    def _lone(self, el: Element, member: str, side: str, unit: Unit, path: tuple[str, ...]) -> None:
        """A plain element on one side only (units inside it are compared on their own)."""
        if not el.children:
            covers = self._covers(side, member, el)
            if covers and all(
                c.mode == "text" and (c.unit, c.field) in self.changed for c in covers
            ):
                return
        self._presence(unit, path, member, el, side, None)

    def _presence(
        self,
        unit: Unit,
        path: tuple[str, ...],
        member: str,
        el: Element,
        side: str,
        why: str | None,
    ) -> None:
        size = len(el.raw)
        if blobs.is_blob_tag(el.tag) and not el.children:
            data = _hex_bytes(el.inner_raw)
            size = len(data) if data is not None else size
        prov = Provenance(member, el.start, el.end, "element")
        verb = "removed" if side == "a" else "added"
        reason = f"; {why}" if why else ""
        self.out.append(
            unit.ctx.change(
                "opaque",
                f"opaque-{verb}",
                f"opaque <{'/'.join(path)}> {verb} ({size} bytes, {member} @{el.start}){reason}",
                field_name="/".join(path),
                where_a=prov if side == "a" else None,
                where_b=prov if side == "b" else None,
                byte_count=size,
            )
        )

    def _lone_carrier(self, unit: Unit) -> None:
        """A marker blob on one side only is accounted for when it is nothing but markers.

        That is: no attributes but DbId, no stray text, only ``FieldsBlob``/``BlobOwner``/
        ``DbSavedTime`` leaves, a keyed dict holding only ``BlobData``, and markers whose every
        field the reader decodes (they are reported as added or removed markers).
        """
        side, prov = ("a", unit.a) if unit.a is not None else ("b", unit.b)
        if prov is None:
            return
        el = self.elements[side].get((prov.member, prov.start, prov.end))
        if el is None:
            return
        if _plain_carrier(el, self._data(side, prov.member)):
            return
        self._presence(
            unit,
            (el.tag,),
            prov.member,
            el,
            side,
            "marker blob with content the reader does not decode",
        )

    def _changed(
        self,
        unit: Unit,
        path: tuple[str, ...],
        ma: str,
        ea: Element,
        mb: str,
        eb: Element,
        why: str,
        *,
        part: tuple[bytes, bytes] | None = None,
    ) -> None:
        leaf = not ea.children and not eb.children and part is None
        if part is not None:
            xa, xb = part
            wa = Provenance(ma, ea.start, ea.end, "element")
            wb = Provenance(mb, eb.start, eb.end, "element")
        elif leaf:
            xa, xb = ea.inner_raw, eb.inner_raw
            if blobs.is_blob_tag(ea.tag):
                ha, hb = _hex_bytes(xa), _hex_bytes(xb)
                if ha is not None and hb is not None and ha != hb:
                    xa, xb = ha, hb
                elif ha is not None and ha == hb:
                    why = "same bytes, but the hex text differs (letter case or spacing)"
            wa = _inner_prov(ma, ea)
            wb = _inner_prov(mb, eb)
        else:
            xa, xb = ea.raw, eb.raw
            wa = Provenance(ma, ea.start, ea.end, "element")
            wb = Provenance(mb, eb.start, eb.end, "element")
        count, first = _diff_size(xa, xb)
        where = f"{ma} @{wa.start}" if ma == mb else f"{ma} @{wa.start} / {mb} @{wb.start}"
        self.out.append(
            unit.ctx.change(
                "opaque",
                "opaque-changed",
                f"opaque <{'/'.join(path)}> changed: {count} of {max(len(xa), len(xb))} bytes "
                f"differ from byte {first}; {why} ({where})",
                field_name="/".join(path),
                where_a=wa,
                where_b=wb,
                byte_count=count,
            )
        )


# --------------------------------------------------------------------------- helpers


def _member_ctx(name: str) -> Ctx:
    return Ctx(f"Member {name}", "member", name, "member")


def _stamp(stamp: tuple[str | None, int | None] | None) -> str:
    if stamp is None:
        return "none"
    return f"{stamp[0]}/{stamp[1]}"


def _inner_prov(member: str, el: Element) -> Provenance:
    if el.inner_start is None or el.inner_end is None:
        return Provenance(member, el.start, el.end, "element")
    return Provenance(member, el.inner_start, el.inner_end, "text")


def _segment(key: tuple[Any, ...], child: _Child) -> str:
    if key[0] == "unit":
        return f"{child.el.tag}[{child.tag}]" if child.wrapper else child.tag
    if key[0] == "Element" and len(key) >= 3:
        return f"Element[{key[1]} {str(key[2])[:8]}]"
    if key[0] == "dbid" and len(key) >= 3:
        return f"{key[1]}[{str(key[2])[:8]}]"
    tag, n = key[0], key[1]
    return str(tag) if n == 1 else f"{tag}[{n}]"


def _order_key(parent: str, key: tuple[Any, ...], child: _Child) -> tuple[Any, ...] | None:
    """What the order of children is compared on; ``None`` = not part of the order."""
    if parent.endswith("Set") and key[0] in ("unit", "Element", "dbid"):
        return None  # a set's DbId children: matched by DbId, unordered across tags
    if key[0] == "unit" and (parent in _SEMANTIC_ORDER or parent.endswith("TrackVec")):
        return ("unit",)
    return key


def _dbid_of(child: _Child) -> str | None:
    if child.el.attrs.get("DbId"):
        return child.el.attrs["DbId"]
    if child.el.tag == "Element" and len(child.el.children) == 1:
        return child.el.children[0].attrs.get("DbId")
    return None


def _loose_text(el: Element, data: bytes) -> bytes:
    """Non-whitespace bytes of ``el`` outside its child elements (comments included).

    Slices only the gaps of ``data`` (the member), never the element's whole body.
    """
    if el.inner_start is None or el.inner_end is None:
        return b""
    parts = []
    pos = el.inner_start
    for child in el.children:
        parts.append(data[pos : child.start].strip())
        pos = child.end
    parts.append(data[pos : el.inner_end].strip())
    return b"".join(parts)


def _plain_carrier(el: Element, data: bytes) -> bool:
    if set(el.attrs) - {"DbId"} or _loose_text(el, data):
        return False
    tags = [c.tag for c in el.children]
    if len(tags) != len(set(tags)) or not set(tags) <= _CARRIER_CHILDREN:
        return False
    if any(c.children for c in el.children):
        return False
    fb = el.find("FieldsBlob")
    residue = _residue("markers", fb.inner_raw, []) if fb is not None else None
    if not isinstance(residue, tuple) or residue[1] != 1 or len(residue) != 3:
        return False
    blob_data = residue[2]
    return (
        isinstance(blob_data, tuple)
        and isinstance(blob_data[-1], _MarkerList)
        and blob_data[-1].plain()
    )


def _iter_skipping(el: Element, skip: frozenset[str]) -> Iterator[Element]:
    stack = list(reversed(el.children))
    while stack:
        node = stack.pop()
        yield node
        if node.tag not in skip:
            stack.extend(reversed(node.children))


def _diff_size(x: bytes, y: bytes) -> tuple[int, int]:
    """(bytes in the differing region, offset where it starts); symmetric in x and y."""
    n = min(len(x), len(y))
    p = 0
    while p < n and x[p] == y[p]:
        p += 1
    s = 0
    while s < n - p and x[len(x) - 1 - s] == y[len(y) - 1 - s]:
        s += 1
    return max(len(x), len(y)) - p - s, p


def _hex_bytes(inner: bytes) -> bytes | None:
    try:
        return blobs.parse_hex(inner.decode("ascii").strip())
    except (BlobError, UnicodeDecodeError):
        return None


def _cover_for(unit: tuple[Any, ...], name: str, prov: Provenance, value: Any) -> Cover | None:
    if prov.kind in ("text", "element"):
        return Cover(unit, name, "text", value=value)
    if prov.kind != "blob" or not prov.blob_path:
        return None
    bp = prov.blob_path
    if bp in ("FrameRate",):
        return Cover(unit, name, "rate", value=value)
    if bp in ("Resolution", "MediaExtents"):
        return Cover(unit, name, "fixed", value=value)
    if bp.startswith(("Time>keyed:", "Geometry>keyed:")):
        return Cover(unit, name, "keyed", bp.split(":", 1)[1], value)
    if bp == "FieldsBlob>SetupBA>message>protobuf#69":
        return Cover(unit, name, "setup", value=value)
    return None  # e.g. MediaTimemapBA: only part of it is decoded, so it stays opaque


# --------------------------------------------------------------------------- residues
#
# A residue is everything in a blob except exactly what the reader decoded into the model.
# Two blobs with equal residues differ only in decoded values, which the semantic layer
# reports (floats compared bit for bit). ``None`` means the blob could not be taken apart the
# way the reader does it, so the caller falls back to calling it opaque.


def _residue(mode: str, inner: bytes, covers: list[Cover]) -> Any:
    data = _hex_bytes(inner)
    if data is None:
        return None
    try:
        handler = _RESIDUES[mode]
        return handler(data, covers)
    except (BlobError, ValueError, IndexError, UnicodeDecodeError) as exc:
        _ = exc  # the caller reports the leaf as opaque; nothing is hidden
        return None


def _rate_residue(data: bytes, covers: list[Cover]) -> Any:
    blobs.decode_rate(data)
    return ("rate", data[8:])  # the second half is not checked by the reader


def _fixed_residue(data: bytes, covers: list[Cover]) -> Any:
    if len(data) != 16:
        raise BlobError("not a 16-byte value")
    return ("fixed",)  # every byte is part of the decoded value, compared bit for bit


def _keyed_value_rest(key: str, value: Any) -> Any:
    if isinstance(value, bool | int | float | str):
        return ()
    if isinstance(value, bytes) and key == "FrameRate":
        blobs.decode_rate(value)
        return value[8:]
    if isinstance(value, bytes) and key == "Resolution":
        blobs.decode_resolution(value)
        return b""
    raise BlobError(f"no mask for keyed value {key!r}")


def _keyed_residue(data: bytes, covers: list[Cover]) -> Any:
    consumed = {c.arg for c in covers}
    kd = blobs.decode_keyed_dict(data)
    if kd.error is not None:
        raise BlobError(kd.error)
    out: list[Any] = [kd.header, kd.count]
    used: set[str] = set()
    for e in kd.entries:
        if e.key in consumed and e.key not in used:
            used.add(e.key)
            out.append((e.key, e.type, e.subtype, _keyed_value_rest(e.key, e.value)))
        else:
            out.append(data[e.offset : e.end])
    return tuple(out)


def _pb_rest(payload: bytes, fields: list[blobs.PbField], skip: set[int]) -> tuple[bytes, ...]:
    return tuple(payload[f.offset : f.end] for f in fields if id(f) not in skip)


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _noncanonical(payload: bytes, fields: list[blobs.PbField], used: set[int]) -> tuple[Any, ...]:
    """Raw bytes of decoded fields whose encoding is not the shortest one (padded varints).

    The decoded value hides the encoding, so a non-canonical key, length or varint value is
    kept in the residue.
    """
    odd = []
    for f in fields:
        if id(f) not in used:
            continue
        raw = payload[f.offset : f.end]
        head = _varint(f.number << 3 | f.wire)
        if f.wire == 0 and isinstance(f.value, int):
            canonical = head + _varint(f.value)
        elif f.wire == 2 and isinstance(f.value, bytes):
            canonical = head + _varint(len(f.value)) + f.value
        else:
            canonical = head + (f.value if isinstance(f.value, bytes) else b"")
        if raw != canonical:
            odd.append((f.number, raw))
    return tuple(odd)


def _setup_residue(data: bytes, covers: list[Cover]) -> Any:
    kd = blobs.decode_keyed_dict(data)
    if kd.error is not None:
        raise BlobError(kd.error)
    out: list[Any] = [kd.header, kd.count]
    for e in kd.entries:
        if e.key != "SetupBA" or not isinstance(e.value, bytes):
            out.append(data[e.offset : e.end])
            continue
        env = blobs.parse_envelope(e.value)
        if env is None:
            raise BlobError("SetupBA is not an envelope")
        payload = blobs.unwrap_message(e.value)
        fields = blobs.decode_protobuf(payload, nested_depth=0)
        skip: set[int] = set()
        for f in fields:
            if f.number == 69 and f.as_text and _FRAME_RATE_HINT.fullmatch(f.as_text):
                skip.add(id(f))
                break
        out.append(
            (
                e.key,
                e.type,
                e.subtype,
                env.version,
                env.codec,
                _noncanonical(payload, fields, skip),
                _pb_rest(payload, fields, skip),
            )
        )
    return tuple(out)


def _clip_residue(data: bytes, covers: list[Cover]) -> Any:
    paths, takes_codec, paths_changed = covers[0].arg
    env = blobs.parse_envelope(data)
    if env is None:
        raise BlobError("Clip blob is not an envelope")
    payload = blobs.unwrap_message(data)
    fields = blobs.decode_protobuf(payload, nested_depth=0)
    first: dict[int, blobs.PbField] = {}
    for f in fields:
        first.setdefault(f.number, f)
    directory = first[1].as_text if 1 in first else None
    filename = first[2].as_text if 2 in first else None
    path = join_clip_path(directory, filename)
    skip: set[int] = set()
    split: Any = None
    if path and path in paths:
        # The joined path is reported; how it was split into directory and file name is not,
        # so the split is kept: which fields exist, and where it falls against the last
        # separator. Which stream holds which path is an index into the clip's path list,
        # ignored only when that list changed (the change is reported).
        tail = _PATH_SEP.split(path)[-1]
        split = (
            "*" if paths_changed else paths.index(path),
            1 in first,
            2 in first,
            len(filename or "") - len(tail),
            bool(directory) and str(directory).endswith(("\\", "/")),
        )
        for number, text in ((1, directory), (2, filename)):
            if number in first and text is not None:
                skip.add(id(first[number]))
    if takes_codec and 5 in first and first[5].as_text:
        skip.add(id(first[5]))
    return (
        env.version,
        env.codec,
        split,
        _noncanonical(payload, fields, skip),
        _pb_rest(payload, fields, skip),
    )


def _clean_text(raw: bytes) -> bool:
    return raw.decode("utf-8", "replace").encode("utf-8") == raw


def _marker_entry_residue(entry: blobs.PbField) -> tuple[Any, ...]:
    if entry.number != 1 or not isinstance(entry.value, bytes):
        raise BlobError("marker list entry is not field 1")
    value = entry.value
    parts = blobs.decode_protobuf(value, nested_depth=0)
    frame = next((p for p in parts if p.number == 1 and p.wire == 0), None)
    body = next((p for p in parts if p.number == 2 and p.wire == 2), None)
    if frame is None or body is None or not isinstance(body.value, bytes):
        raise BlobError("marker entry without frame or body")
    env = blobs.parse_envelope(body.value)
    if env is None or env.version is None:
        raise BlobError("marker body without header")
    inner = blobs.decode_protobuf(env.payload, nested_depth=0)
    detail = next((p for p in inner if p.number == 1 and p.wire == 2), None)
    detail_bytes = detail.value if detail is not None else b""
    if not isinstance(detail_bytes, bytes):
        raise BlobError("marker detail is not a message")
    fields = blobs.decode_protobuf(detail_bytes, nested_depth=0)
    color = next((f for f in fields if f.number == 1 and f.wire == 0), None)
    strings = [f for f in fields if f.number == 3 and isinstance(f.value, bytes)]
    custom = next((f for f in fields if f.number == 6 and f.wire == 2), None)
    skip: set[int] = set()
    if color is not None and isinstance(color.value, int) and color.value in MARKER_COLORS:
        skip.add(id(color))
    for i, s in enumerate(strings[:3]):
        assert isinstance(s.value, bytes)
        text = s.value
        if i == 1:  # the duration: decoded as an int, so only its canonical spelling is masked
            ok = (text.isdigit() and str(int(text)).encode() == text) or text == b""
        else:
            ok = _clean_text(text)
        if ok:
            skip.add(id(s))
    if custom is not None and isinstance(custom.value, bytes) and _clean_text(custom.value):
        skip.add(id(custom))
    head_used = {id(frame), id(body)}
    inner_used = {id(detail)} if detail is not None else set()
    noncanonical = (
        _noncanonical(value, parts, head_used)
        + _noncanonical(env.payload, inner, inner_used)
        + _noncanonical(detail_bytes, fields, skip)
    )
    return (
        _pb_rest(value, parts, head_used),
        len(strings),
        env.version,
        env.codec,
        _pb_rest(env.payload, inner, inner_used),
        _pb_rest(detail_bytes, fields, skip),
        noncanonical,
    )


@dataclass(frozen=True, slots=True)
class _MarkerList:
    """Residue of a marker list.

    ``framing`` holds the framing of markers whose content is otherwise fully decoded (string
    count, body header version, codec); ``odd`` lists, sorted, every marker entry that carries
    anything the reader does not decode (undecoded fields, non-canonical encodings). Markers
    added or removed are reported by the semantic layer, so plain markers' framing only counts
    when both sides have plain markers to compare; an ``odd`` entry always counts. The order
    of markers in the list is reported by the semantic layer.
    """

    holders: int
    framing: frozenset[Any]
    odd: tuple[Any, ...]

    def plain(self) -> bool:
        return not self.odd

    def same_as(self, other: _MarkerList) -> bool:
        if self.odd != other.odd:
            return False
        if self.framing and other.framing:
            return self.framing == other.framing and self.holders == other.holders
        return True


def _markers_payload_residue(payload: bytes) -> _MarkerList:
    top = blobs.decode_protobuf(payload, nested_depth=0)
    framing: set[Any] = set()
    odd: list[Any] = []
    for holder in top:
        if holder.number != 2 or not isinstance(holder.value, bytes):
            raise BlobError("marker list holder is not field 2")
        for entry in blobs.decode_protobuf(holder.value, nested_depth=0):
            residue = _marker_entry_residue(entry)
            rest_entry, n_strings, version, codec, rest_inner, rest_detail, noncanon = residue
            if rest_entry or rest_inner or rest_detail or noncanon:
                odd.append(residue)
            else:
                framing.add((n_strings, version, codec))
    return _MarkerList(len(top), frozenset(framing), tuple(sorted(odd, key=repr)))


def _same_residue(ra: Any, rb: Any) -> bool:
    """Equality of residues, with marker lists compared by :meth:`_MarkerList.same_as`."""
    if isinstance(ra, _MarkerList) and isinstance(rb, _MarkerList):
        return ra.same_as(rb)
    if isinstance(ra, tuple) and isinstance(rb, tuple):
        return len(ra) == len(rb) and all(_same_residue(x, y) for x, y in zip(ra, rb, strict=True))
    return bool(ra == rb)


def _marker_residue(data: bytes, covers: list[Cover]) -> Any:
    kd = blobs.decode_keyed_dict(data)
    if kd.error is not None:
        raise BlobError(kd.error)
    out: list[Any] = [kd.header, kd.count]
    done = False
    for e in kd.entries:
        if e.key == "BlobData" and isinstance(e.value, bytes) and not done:
            done = True
            env = blobs.parse_envelope(e.value)
            if env is None:
                raise BlobError("BlobData is not an envelope")
            payload = blobs.unwrap_message(e.value)
            out.append(
                (
                    e.key,
                    e.type,
                    e.subtype,
                    env.version,
                    env.codec,
                    _markers_payload_residue(payload),
                )
            )
        else:
            out.append(data[e.offset : e.end])
    if not done:
        raise BlobError("marker blob without BlobData")
    return tuple(out)


_RESIDUES: dict[str, Callable[[bytes, list[Cover]], Any]] = {
    "rate": _rate_residue,
    "fixed": _fixed_residue,
    "keyed": _keyed_residue,
    "setup": _setup_residue,
    "clip": _clip_residue,
    "markers": _marker_residue,
}
