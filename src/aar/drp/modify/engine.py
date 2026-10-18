"""Turn a :class:`~aar.drp.modify.plan.DrpEditPlan` into byte splices, and predict the diff.

:func:`prepare` resolves every edit against the reader's model and returns
:class:`Prepared`: the splices (member, byte span, new bytes), the changes the S2 diff must show
afterwards (:class:`Expected`, derived from the plan and the input model, never from the written
file), and a report of what was touched and what was not. :func:`apply` splices the bytes.

Every splice replaces a span the reader recorded (a text leaf, or the hex text of a blob it
decodes), so bytes outside those spans are never rewritten. Anything this module cannot do the
way Resolve's own files are laid out is refused with a :class:`Refusal`, never approximated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from aar.drp import blobs
from aar.drp.container import DEFAULT_MAX_MEMBER_BYTES, Container, open_container
from aar.drp.diff_model import Change
from aar.drp.errors import BlobError, ContainerError, DrpError
from aar.drp.model import DrpFile, MediaClip, Provenance, Timeline, TimelineItem
from aar.drp.modify import encode
from aar.drp.modify.encode import EncodeError
from aar.drp.modify.markers import DEFAULT_BODY_VERSION, MarkerEntry, MarkerList, props
from aar.drp.modify.paths import PathMapper, TableError
from aar.drp.modify.plan import (
    MAX_NAME,
    DrpEditPlan,
    MarkerAdd,
    MarkerDelete,
    MarkerUpdate,
    PathMap,
    Rename,
)
from aar.drp.reader import DEFAULT_MAX_ELEMENTS, DEFAULT_MAX_TOTAL_XML_BYTES, join_clip_path
from aar.drp.scan import Document, Element, scan

__all__ = [
    "EditRefused",
    "Expected",
    "Prepared",
    "Refusal",
    "Splice",
    "apply",
    "change_key",
    "prepare",
]

T = TypeVar("T")
_CONTAINER_TYPES = frozenset({"timeline", "compound", "multicam"})
_INFO_TAGS = frozenset({"BtVideoInfo", "BtAudioInfo"})
_INFO_SKIP = frozenset({"Sequence", "TimelineSharedHandle"})
_CARRIERS = {"timeline": "Sm2SequenceLockableBlob", "item": "Sm2TiItemLockableBlob"}
_TAG = re.compile(rb"^<([^\s/>]+)")


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why an edit (``edit`` = its index in the plan) or the whole plan was refused."""

    code: str
    message: str
    edit: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "edit": self.edit,
            "details": self.details,
        }


class EditRefused(DrpError):
    """The plan cannot be applied; nothing is written. ``refusals`` says why."""

    def __init__(self, refusals: list[Refusal]) -> None:
        super().__init__("; ".join(r.message for r in refusals[:3]))
        self.refusals = refusals


@dataclass(frozen=True, slots=True)
class Splice:
    """Replace ``member[start:end]`` with ``data``."""

    member: str
    start: int
    end: int
    data: bytes
    edit: int


def _norm(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class Expected:
    """A change the diff must report: the same fields as :class:`~aar.drp.diff_model.Change`."""

    section: str
    kind: str
    object_kind: str
    object_id: str | None
    field: str | None
    old: Any
    new: Any
    edit: int

    def key(self) -> tuple[Any, ...]:
        return (
            self.section,
            self.kind,
            self.object_kind,
            self.object_id,
            self.field,
            _norm(self.old),
            _norm(self.new),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "kind": self.kind,
            "object_kind": self.object_kind,
            "object_id": self.object_id,
            "field": self.field,
            "old": self.old,
            "new": self.new,
            "edit": self.edit,
        }


def change_key(c: Change) -> tuple[Any, ...]:
    """The comparison key of a diff change (same layout as :meth:`Expected.key`)."""
    d = c.to_dict()
    return (
        c.section,
        c.kind,
        c.object_kind,
        c.object_id,
        c.field_name,
        _norm(d["old"]),
        _norm(d["new"]),
    )


@dataclass(slots=True)
class Prepared:
    """The splices to make, the changes they must produce, and the input bytes they apply to."""

    splices: list[Splice]
    expected: list[Expected]
    report: list[dict[str, Any]]
    member_data: dict[str, bytes]

    def members(self) -> set[str]:
        return {s.member for s in self.splices}


def apply(member_data: dict[str, bytes], splices: Sequence[Splice]) -> dict[str, bytes]:
    """New contents of every member a splice touches. Overlapping splices are refused."""
    out: dict[str, bytes] = {}
    by_member: dict[str, list[Splice]] = {}
    for s in splices:
        by_member.setdefault(s.member, []).append(s)
    for member, items in by_member.items():
        data = member_data[member]
        items = sorted(items, key=lambda s: (s.start, s.end))
        parts: list[bytes] = []
        pos = 0
        for s in items:
            if s.start < pos or s.end < s.start or s.end > len(data):
                raise EditRefused(
                    [
                        Refusal(
                            "conflicting-edits",
                            f"two edits change the same bytes of {member} (edit {s.edit})",
                            s.edit,
                        )
                    ]
                )
            parts.append(data[pos : s.start])
            parts.append(s.data)
            pos = s.end
        parts.append(data[pos:])
        out[member] = b"".join(parts)
    return out


class _Source:
    """Raw member bytes and scans of the input, within the reader's budgets."""

    def __init__(self, path: Path) -> None:
        self.container: Container = open_container(path, max_member_bytes=DEFAULT_MAX_MEMBER_BYTES)
        self.raw: dict[str, bytes] = {}
        self.docs: dict[str, Document] = {}
        self.xml_left = DEFAULT_MAX_TOTAL_XML_BYTES
        self.elements_left = DEFAULT_MAX_ELEMENTS
        self.budget = blobs.DecodeBudget()

    def data(self, member: str) -> bytes:
        if member not in self.raw:
            data = self.container.read(member, max_bytes=max(self.xml_left, 0))
            self.xml_left -= len(data)
            self.raw[member] = data
        return self.raw[member]

    def doc(self, member: str) -> Document:
        if member not in self.docs:
            doc = scan(self.data(member), member, max_elements=max(self.elements_left, 0))
            self.elements_left -= doc.element_count
            if doc.stopped_at is not None:
                raise ContainerError(f"{member}: the element budget ran out while scanning it")
            self.docs[member] = doc
        return self.docs[member]

    def element(self, prov: Provenance) -> Element:
        doc = self.doc(prov.member)
        if doc.root is not None:
            for el in doc.root.iter():
                if (el.start, el.end) == (prov.start, prov.end):
                    return el
        raise ContainerError(f"element at {prov.member}@{prov.start} not found again")

    def close(self) -> None:
        self.container.close()


def prepare(drp: DrpFile, path: str | Path, plan: DrpEditPlan) -> Prepared:
    """Resolve ``plan`` against ``drp`` (read from ``path``); raises :class:`EditRefused`."""
    src = _Source(Path(path))
    try:
        return _Planner(drp, src).run(plan)
    finally:
        src.close()


def _one(objs: Sequence[T], match: Callable[[T], bool], what: str, sel: str, edit: int) -> T:
    found = [o for o in objs if match(o)]
    if not found:
        raise _Refuse("target-not-found", f"no {what} {sel}", edit)
    if len(found) > 1:
        raise _Refuse("target-ambiguous", f"{len(found)} {what}s match {sel}; use the DbId", edit)
    return found[0]


class _Unchanged(Exception):
    """A rename to the name the object already has: nothing to write (internal)."""


class _Refuse(Exception):
    def __init__(self, code: str, message: str, edit: int, **details: Any) -> None:
        super().__init__(message)
        self.refusal = Refusal(code, message, edit, details)


def _items(drp: DrpFile) -> Iterator[tuple[Timeline, TimelineItem]]:
    for tl in drp.timelines:
        for track in tl.tracks:
            for item in track.items:
                yield tl, item


def _timeline_key(tl: Timeline) -> str | None:
    """The key the diff pairs timelines on (``diff._Differ.run``)."""
    return tl.id or tl.sequence_id or tl.media_clip_id


class _Planner:
    def __init__(self, drp: DrpFile, src: _Source) -> None:
        self.drp = drp
        self.src = src
        self.splices: list[Splice] = []
        self.expected: list[Expected] = []
        self.report: list[dict[str, Any]] = []
        self.media = drp.media_by_id()
        self.owned = {tl.media_clip_id for tl in drp.timelines if tl.media_clip_id}
        self.marker_lists: dict[tuple[str, str], tuple[Provenance, MarkerList, list[int]]] = {}
        self.marker_before: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def run(self, plan: DrpEditPlan) -> Prepared:
        refusals: list[Refusal] = []
        path_maps = [i for i, e in enumerate(plan.edits) if isinstance(e, PathMap)]
        if len(path_maps) > 1:
            refusals.append(
                Refusal(
                    "conflicting-edits",
                    "a plan takes one path_map edit; put every row in its table",
                    path_maps[1],
                )
            )
        for i, edit in enumerate(plan.edits):
            if refusals:
                break
            try:
                if isinstance(edit, Rename):
                    self._rename(i, edit)
                elif isinstance(edit, PathMap):
                    self._path_map(i, edit)
                elif isinstance(edit, MarkerAdd | MarkerUpdate | MarkerDelete):
                    self._marker(i, edit)
                else:  # pragma: no cover - the plan model admits nothing else
                    raise _Refuse("edit-not-whitelisted", f"edit {i} is not whitelisted", i)
            except _Refuse as exc:
                refusals.append(exc.refusal)
            except (EncodeError, BlobError, ContainerError) as exc:
                refusals.append(Refusal("cannot-encode", str(exc), i))
        if not refusals:
            try:
                self._finish_markers()
            except _Refuse as exc:
                refusals.append(exc.refusal)
            except (EncodeError, BlobError, ContainerError) as exc:
                refusals.append(Refusal("cannot-encode", str(exc), None))
        if refusals:
            raise EditRefused(refusals)
        data = {s.member: self.src.data(s.member) for s in self.splices}
        return Prepared(self.splices, self.expected, self.report, data)

    def _expect(self, edit: int, *args: Any) -> None:
        self.expected.append(Expected(*args, edit=edit))

    # ------------------------------------------------------------------ rename

    def _name_splice(self, edit: int, prov: Provenance | None, new: str, what: str) -> None:
        if prov is None:
            raise _Refuse("no-span", f"{what} has no name element to rename", edit)
        if prov.kind == "text":
            data = encode.xml_text(new)
        elif prov.kind == "element":
            raw = self.src.data(prov.member)[prov.start : prov.end]
            m = _TAG.match(raw)
            if m is None or not raw.endswith(b"/>"):
                raise _Refuse("no-span", f"{what}: unexpected name element", edit)
            tag = m.group(1)
            data = b"<" + tag + b">" + encode.xml_text(new) + b"</" + tag + b">"
        else:
            raise _Refuse("no-span", f"{what}: the name is not a text leaf", edit)
        self.splices.append(Splice(prov.member, prov.start, prov.end, data, edit))

    def _rename(self, i: int, e: Rename) -> None:
        try:
            encode.check_text(e.new_name, "new_name", max_chars=MAX_NAME)
        except EncodeError as exc:
            raise _Refuse("bad-value", str(exc), i) from exc
        try:
            old = self._rename_target(i, e)
        except _Unchanged:
            self.report.append(
                {
                    "edit": i,
                    "op": "rename",
                    "target": e.target,
                    "old": e.new_name,
                    "new": e.new_name,
                    "unchanged": True,
                }
            )
            return
        self.report.append(
            {"edit": i, "op": "rename", "target": e.target, "old": old, "new": e.new_name}
        )

    def _rename_target(self, i: int, e: Rename) -> str | None:
        """Splice the new name and expect the change; returns the old name."""
        sel = f"with DbId {e.id}" if e.id is not None else f"named {e.name!r}"
        drp = self.drp
        old: str | None
        if e.target == "project":
            if drp.project is None:
                raise _Refuse("target-not-found", "the file has no project.xml project", i)
            old = drp.project.name
            self._check_old(i, e, old)
            self._name_splice(i, drp.project.spans.get("name"), e.new_name, "the project")
            self._expect(i, "header", "renamed", "project", drp.project.id, "name", old, e.new_name)
        elif e.target == "bin":
            b = _one(drp.bins, _sel(e, lambda x: x.id, lambda x: x.name), "bin", sel, i)
            if b.id == b.member:
                raise _Refuse("no-dbid", f"bin {sel} has no DbId, so the gate cannot match it", i)
            if not b.parent_id:
                raise _Refuse("root-bin", "the root (Master) bin cannot be renamed in Resolve", i)
            old = b.name
            self._check_old(i, e, old)
            self._name_splice(i, b.spans.get("name"), e.new_name, f"bin {sel}")
            self._expect(i, "media", "renamed", "bin", b.id, "name", old, e.new_name)
        elif e.target == "clip":
            clips = [m for m in drp.media if m.id not in self.owned]
            c = _one(clips, _sel(e, lambda x: x.id, lambda x: x.name), "media pool clip", sel, i)
            old = c.name
            self._check_old(i, e, old)
            self._name_splice(i, c.spans.get("name"), e.new_name, f"clip {sel}")
            self._expect(i, "media", "renamed", "media", c.id, "name", old, e.new_name)
        elif e.target == "timeline":
            tl = _one(drp.timelines, _sel(e, lambda x: x.id, lambda x: x.name), "timeline", sel, i)
            old = self._rename_timeline(i, e, tl)
        else:
            pairs = list(_items(drp))
            _, item = _one(
                pairs, _sel(e, lambda p: p[1].id, lambda p: p[1].name), "timeline item", sel, i
            )
            if not item.id:
                raise _Refuse("no-dbid", f"item {sel} has no DbId, so the gate cannot match it", i)
            old = item.name
            self._check_old(i, e, old)
            self._name_splice(i, item.spans.get("name"), e.new_name, f"item {sel}")
            self._expect(i, "timelines", "renamed", "item", item.id, "name", old, e.new_name)
        return old

    @staticmethod
    def _check_old(i: int, e: Rename, old: str | None) -> None:
        """Refuse a stale ``expect_name``; stop (as a no-op) when the name would not change."""
        if e.expect_name is not None and e.expect_name != (old or ""):
            raise _Refuse(
                "expect-mismatch", f"the current name is {old!r}, not {e.expect_name!r}", i
            )
        if old == e.new_name:
            raise _Unchanged

    def _rename_timeline(self, i: int, e: Rename, tl: Timeline) -> str | None:
        key = _timeline_key(tl)
        old = tl.name
        self._check_old(i, e, old)
        pool = self.media.get(tl.media_clip_id or "")
        own = tl.spans.get("name")
        pool_span = pool.spans.get("name") if pool is not None else None
        if own is None and pool_span is None:
            raise _Refuse("no-span", f"timeline {old!r} has no name element", i)
        if own is not None:
            self._name_splice(i, own, e.new_name, f"timeline {old!r}")
        # The pool entry carries the timeline's name too; keep the two in step when they agree.
        if pool is not None and pool_span is not None and (own is None or pool.name == old):
            self._name_splice(i, pool_span, e.new_name, f"timeline {old!r} pool entry")
        self._expect(i, "timelines", "renamed", "timeline", key, "name", old, e.new_name)
        return old

    # ------------------------------------------------------------------ paths

    def _path_map(self, i: int, e: PathMap) -> None:
        try:
            mapper = PathMapper(e.direction, [(row.windows, row.mac) for row in e.table])
        except TableError as exc:
            raise _Refuse("bad-table", str(exc), i) from exc
        touched: list[dict[str, Any]] = []
        unmapped: list[dict[str, Any]] = []
        mapped_items: dict[str, str] = {}
        for clip in self.drp.media:
            self._map_clip(i, clip, mapper, touched, unmapped)
        for tl, item in _items(self.drp):
            old = item.media_file_path
            prov = item.spans.get("media_file_path")
            if not old or prov is None:
                continue
            where = {"object": "item", "id": item.id, "timeline": tl.name, "field": "MediaFilePath"}
            if not item.id:
                unmapped.append({"path": old, "reason": "item-without-dbid", **where})
                continue
            result = mapper.map(old)
            if result.new is None:
                unmapped.append({"path": old, "reason": result.reason, **where})
                continue
            if result.new == old:
                continue
            if prov.kind != "text":
                raise _Refuse("no-span", f"item {item.id}: MediaFilePath is not a text leaf", i)
            data = encode.xml_text(result.new)
            self.splices.append(Splice(prov.member, prov.start, prov.end, data, i))
            self._expect(
                i, "timelines", "changed", "item", item.id, "media_file_path", old, result.new
            )
            touched.append({"path": old, "new": result.new, **where})
            mapped_items[old] = result.new
        self._item_path_clips(i, mapper, mapped_items)
        self.report.append(
            {
                "edit": i,
                "op": "path_map",
                "direction": e.direction,
                "touched": touched,
                "unmapped": unmapped,
            }
        )

    def _map_clip(
        self,
        i: int,
        clip: MediaClip,
        mapper: PathMapper,
        touched: list[dict[str, Any]],
        unmapped: list[dict[str, Any]],
    ) -> None:
        """Map the paths of one media clip's ``Clip`` blobs (the ones the diff covers)."""
        if clip.source is None:
            return
        where = {"object": "media", "id": clip.id, "clip": clip.name, "field": "Clip blob"}
        container = clip.type in _CONTAINER_TYPES or clip.id in self.owned
        model_paths = {p for p in (clip.blob_path, *clip.linked_paths) if p}
        new_paths: dict[str, str] = {}
        edits: list[tuple[Element, bytes, str, str]] = []
        for node in _iter_skipping(self.src.element(clip.source), _INFO_SKIP):
            blob = node.find("Clip") if node.tag in _INFO_TAGS else None
            if blob is None or blob.children or not blob.text or blob.inner_start is None:
                continue
            try:
                raw = blobs.parse_hex(blob.text)
                env = blobs.parse_envelope(raw)
                if env is None or env.version is None:
                    raise BlobError("Clip blob is not an envelope")
                payload = blobs.unwrap_message(raw, budget=self.src.budget)
                fields = blobs.decode_protobuf(payload, nested_depth=0)
            except BlobError as exc:
                unmapped.append({"path": None, "reason": f"undecodable: {exc}", **where})
                continue
            first: dict[int, blobs.PbField] = {}
            for f in fields:
                first.setdefault(f.number, f)
            directory = first[1].as_text if 1 in first else None
            filename = first[2].as_text if 2 in first else None
            path = join_clip_path(directory, filename)
            if not path:
                continue
            result = mapper.map(path)
            if result.new is None or result.new == path:
                if result.reason is not None:
                    unmapped.append({"path": path, "reason": result.reason, **where})
                continue
            if container:
                reason = "container-clip"
            elif path not in model_paths:
                reason = "not-in-model"
            elif not directory:
                reason = "filename-only-path"
            else:
                new_dir = _mapped_dir(result.new, directory, filename)
                if new_dir is None or join_clip_path(new_dir, filename) != result.new:
                    reason = "cannot-split-path"
                else:
                    payload = encode.replace_field(payload, first[1], new_dir.encode("utf-8"))
                    edits.append((blob, encode.envelope(raw, payload), path, result.new))
                    new_paths[path] = result.new
                    continue
            unmapped.append({"path": path, "reason": reason, **where})
        if not edits:
            return
        old_list = [p for p in (clip.blob_path, *clip.linked_paths) if p]
        new_list = [new_paths.get(p, p) for p in old_list]
        if len(set(new_list)) != len(new_list):
            for _blob, _raw, path, _new in edits:
                unmapped.append({"path": path, "reason": "merges-paths", **where})
            return
        member = clip.source.member
        for blob, new_raw, path, new in edits:
            assert blob.inner_start is not None and blob.inner_end is not None
            text = encode.hex_like(blob.inner_raw, new_raw)
            self.splices.append(Splice(member, blob.inner_start, blob.inner_end, text, i))
            touched.append({"path": path, "new": new, **where})
        if clip.path_source == "media_pool_blob" and clip.path in new_paths:
            new_path = new_paths[clip.path]
            self._expect(i, "media", "relinked", "media", clip.id, "path", clip.path, new_path)
        new_linked = [new_paths.get(p, p) for p in clip.linked_paths]
        if new_linked != clip.linked_paths:
            self._expect(
                i,
                "media",
                "changed",
                "media",
                clip.id,
                "linked_paths",
                clip.linked_paths,
                new_linked,
            )

    def _item_path_clips(self, i: int, mapper: PathMapper, mapped_items: dict[str, str]) -> None:
        """Clips whose path is a timeline ``<MediaFilePath>`` follow their items' mapping."""
        for clip in self.drp.media:
            if clip.path_source != "timeline_item" or not clip.path:
                continue
            if clip.type in _CONTAINER_TYPES or clip.id in self.owned:
                continue
            new = mapped_items.get(clip.path)
            if new is not None and new != clip.path:
                self._expect(i, "media", "relinked", "media", clip.id, "path", clip.path, new)

    # ------------------------------------------------------------------ markers

    def _owner(self, i: int, e: MarkerAdd | MarkerUpdate | MarkerDelete) -> str:
        """The ``BlobOwner`` DbId of the marker owner (a timeline's sequence, or an item)."""
        if e.owner == "timeline":
            tl = _one(self.drp.timelines, lambda t: t.id == e.owner_id, "timeline", e.owner_id, i)
            if not tl.sequence_id:
                raise _Refuse("no-dbid", f"timeline {e.owner_id} has no sequence DbId", i)
            return tl.sequence_id
        _, item = _one(
            list(_items(self.drp)), lambda p: p[1].id == e.owner_id, "item", e.owner_id, i
        )
        return item.id

    def _marker_list(self, i: int, kind: str, owner: str) -> tuple[Provenance, MarkerList]:
        key = (kind, owner)
        if key in self.marker_lists:
            prov, ml, edits = self.marker_lists[key]
            edits.append(i)
            return prov, ml
        doc = self.src.doc("project.xml")
        carriers = []
        if doc.root is not None:
            for el in doc.root.iter(_CARRIERS[kind]):
                if (el.child_text("BlobOwner") or "").strip() == owner:
                    carriers.append(el)
        if len(carriers) != 1:
            what = "no marker blob" if not carriers else f"{len(carriers)} marker blobs"
            raise _Refuse(
                "no-marker-list",
                f"{kind} {owner} has {what}; adding the first marker needs a new Resolve "
                "object, which the modify path does not create (add one marker in Resolve first)",
                i,
            )
        fb = carriers[0].find("FieldsBlob")
        if fb is None or fb.children or fb.inner_start is None or fb.inner_end is None:
            raise _Refuse("no-marker-list", f"{kind} {owner}: the marker blob is empty", i)
        try:
            ml = MarkerList.decode(blobs.parse_hex(fb.text or ""), self.src.budget)
        except (EncodeError, BlobError) as exc:
            raise _Refuse("no-marker-list", f"{kind} {owner}: {exc}", i) from exc
        prov = Provenance("project.xml", fb.inner_start, fb.inner_end, "text")
        self.marker_lists[key] = (prov, ml, [i])
        self.marker_before[key] = ml.values()
        return prov, ml

    def _marker(self, i: int, e: MarkerAdd | MarkerUpdate | MarkerDelete) -> None:
        owner = self._owner(i, e)
        _, ml = self._marker_list(i, e.owner, owner)
        at = [n for n, m in enumerate(ml.entries) if m.values["frame"] == e.frame]
        if isinstance(e, MarkerAdd):
            if at:
                raise _Refuse("marker-exists", f"a marker already sits at frame {e.frame}", i)
            values = props(e.frame, e.color, e.name, e.note, e.duration, e.custom_data)
            _check_marker_text(i, values)
            entry = MarkerEntry.new(
                values, ml.body_version() if ml.entries else DEFAULT_BODY_VERSION
            )
            pos = sum(1 for m in ml.entries if m.values["frame"] < e.frame)
            ml.entries.insert(pos, entry)
            return
        if not at:
            raise _Refuse("marker-not-found", f"no marker at frame {e.frame}", i)
        if len(at) > 1:
            raise _Refuse("marker-ambiguous", f"{len(at)} markers sit at frame {e.frame}", i)
        if isinstance(e, MarkerDelete):
            del ml.entries[at[0]]
            return
        changes = e.set.model_dump(exclude_unset=True)
        current = ml.entries[at[0]]
        moved_to = changes.get("frame", e.frame)
        if moved_to != e.frame and any(m.values["frame"] == moved_to for m in ml.entries):
            raise _Refuse("marker-exists", f"a marker already sits at frame {moved_to}", i)
        _check_marker_text(i, {**current.values, **changes})
        ml.entries[at[0]] = current.updated(changes)

    def _finish_markers(self) -> None:
        for key, (prov, ml, edits) in self.marker_lists.items():
            kind, owner = key
            before = self.marker_before[key]
            after = ml.values()
            data = encode.hex_like(self.src.data(prov.member)[prov.start : prov.end], ml.encode())
            self.splices.append(Splice(prov.member, prov.start, prov.end, data, edits[0]))
            diff_id = self._diff_owner_id(kind, owner)
            self._marker_changes(edits[0], diff_id, before, after)
            self.report.append(
                {
                    "edit": edits[0],
                    "op": "markers",
                    "owner": kind,
                    "owner_blob": owner,
                    "edits": edits,
                    "before": before,
                    "after": after,
                }
            )

    def _diff_owner_id(self, kind: str, owner: str) -> str:
        if kind == "item":
            return owner
        for tl in self.drp.timelines:
            if tl.sequence_id == owner:
                key = _timeline_key(tl)
                assert key is not None
                return key
        raise _Refuse("target-not-found", f"no timeline owns sequence {owner}", -1)

    def _marker_changes(
        self, edit: int, owner_id: str, before: list[dict[str, Any]], after: list[dict[str, Any]]
    ) -> None:
        """What the diff's gate-mode marker comparison reports (one marker per frame)."""
        old = {m["frame"]: m for m in before}
        new = {m["frame"]: m for m in after}
        if len(old) != len(before) or len(new) != len(after):
            raise _Refuse("marker-ambiguous", "two markers share a frame on this owner", edit)
        for frame, m in old.items():
            if frame not in new:
                self._expect(edit, "markers", "removed", "marker", owner_id, None, m, None)
        for frame, m in new.items():
            if frame not in old:
                self._expect(edit, "markers", "added", "marker", owner_id, None, None, m)
        for frame, m in old.items():
            if frame in new and new[frame] != m:
                diffs = [k for k in m if m[k] != new[frame][k]]
                self._expect(
                    edit, "markers", "changed", "marker", owner_id, ",".join(diffs), m, new[frame]
                )


def _check_marker_text(i: int, values: dict[str, Any]) -> None:
    for key in ("name", "note", "custom_data"):
        value = values.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            encode.check_text(value.replace("\n", " ").replace("\t", " "), key, max_chars=65536)
        except EncodeError as exc:
            raise _Refuse("bad-value", str(exc), i) from exc


def _mapped_dir(new: str, directory: str, filename: str | None) -> str | None:
    """The new directory field: ``new`` minus the file name and its separator."""
    if not filename:
        return new
    if not new.endswith(filename):
        return None
    head = new[: len(new) - len(filename)]
    if directory.endswith(("\\", "/")):
        return head if head.endswith(("\\", "/")) else None
    if not head.endswith(("\\", "/")) or len(head) < 2:
        return None
    return head[:-1]


def _sel(
    e: Rename, get_id: Callable[[Any], Any], get_name: Callable[[Any], Any]
) -> Callable[[Any], bool]:
    if e.id is not None:
        return lambda o: get_id(o) == e.id
    return lambda o: get_name(o) == e.name


def _iter_skipping(el: Element, skip: frozenset[str]) -> Iterator[Element]:
    stack = list(reversed(el.children))
    while stack:
        node = stack.pop()
        yield node
        if node.tag not in skip:
            stack.extend(reversed(node.children))
