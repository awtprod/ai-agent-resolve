"""The item gate: author every real item from its media clip and a *different* real item.

For every picture and sound item of the given Resolve-saved projects, :func:`run` authors the
item from the rows of the media clip it plays, its own source range, record position, track,
index and links, with another real item of the same kind as the template (leave-one-out: a
template from the same project when there is one, playing another media clip when there is
one), and compares the result with Resolve's own rows (:func:`compare_items`). Items of a shape
the writer does not support are counted per reason (``refused``), never dropped silently.

The scoreboard groups the checked items by kind and by what their media clip holds (``picture
of a file with sound``, ``picture of a file without sound``, ``sound of a picture file``,
``sound of a sound file``) and counts the shapes they carry (``linked``, ``unlinked``,
``starts at frame 0``, ``graded``, ``effects``...). Each sequence's ``MediaExtents`` length is
checked against its items too (:func:`extents_seconds`).
"""

from __future__ import annotations

import math
import struct
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.items.author import ItemPlacement, author_item
from aar.drp.items.errors import ItemError
from aar.drp.items.fields import compare_items
from aar.drp.items.media import media_view
from aar.drp.items.rows import ItemRows, ItemSet, Shape, frames_text, links_of, read_items, shape_of
from aar.drp.library import LibraryError, LibraryOptions
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool.errors import MediaPoolError, refusing
from aar.media.probe import timecode_frames

__all__ = ["GroupScore", "ItemScoreboard", "extents_seconds", "format_scoreboard", "run"]


_RANGE = ("Start", "Duration", "In")


@dataclass
class GroupScore:
    group: str
    checked: int = 0
    fields: int = 0
    equal: int = 0
    refused: int = 0
    differing: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    refusals: dict[str, int] = field(default_factory=dict)
    shapes: dict[str, int] = field(default_factory=dict)
    hidden: dict[str, int] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "differs" if self.differing or self.refused else "equal"


@dataclass
class ItemScoreboard:
    groups: list[GroupScore]
    #: Items of a shape the writer does not support, by reason.
    refused: dict[str, int]
    #: Sequences whose ``MediaExtents`` length was checked, and those that differed.
    extents_checked: int = 0
    extents_differing: list[str] = field(default_factory=list)
    extents_skipped: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def items_checked(self) -> int:
        return sum(g.checked for g in self.groups)

    def passed(self) -> bool:
        return all(g.status == "equal" for g in self.groups) and not self.extents_differing


@dataclass
class _Target:
    project: str
    items: ItemSet
    rows: ItemRows
    shape: Shape
    linked: bool


def _group(target: _Target) -> str:
    clip = target.items.clips[str(target.rows.item["MediaRef"])]
    if target.shape.kind == "picture":
        return "picture of a file " + ("with sound" if clip.audio else "without sound")
    return "sound of a " + ("picture file" if clip.video is not None else "sound file")


def _exact_start(target: _Target) -> bool:
    """The item holds the exact quotient of its media's start timecode and rate, not the
    pool's rounded one (an item a timeline import made from FCPXML)."""
    clip = target.items.clips[str(target.rows.item["MediaRef"])]
    if clip.video is None:
        return False
    time = {e[0]: e[3] for e in bt.kd_decode(clip.video["Time"], "Time")}
    timecode = time.get("Timecode")
    rate_value = time.get("FrameRate")
    if not isinstance(timecode, str) or not isinstance(rate_value, bytes):
        return False
    view = media_view(clip)
    exact = float(Fraction(timecode_frames(timecode, view.rate)) / view.rate)
    stored = target.rows.item.get("MediaStartTime")
    return stored == exact and stored != view.start_seconds


def _start_last_bit(target: _Target, rule: float) -> bool:
    """A picture of a file without sound whose ``MediaStartTime`` is one unit in the last place
    away from the rule's value: the clip stores no start of its own (no audio track), and on
    21 of 100 such real items Resolve's value is the neighbouring double."""
    clip = target.items.clips[str(target.rows.item["MediaRef"])]
    stored = target.rows.item.get("MediaStartTime")
    if clip.audio or not isinstance(stored, float) or stored == rule:
        return False
    return stored in (math.nextafter(rule, math.inf), math.nextafter(rule, -math.inf))


@refusing("bad_template", "computing a sequence length", "check the projects", error=ItemError)
def extents_seconds(rate: float, start_seconds: float, ends: Iterable[int]) -> float:
    """``MediaExtents``' length of a sequence: the end of its last item in seconds (its frame
    over the rate as a double) minus the sequence's start in seconds, as Resolve computes it
    (``(end - start) / rate`` differs in the last bits on 11 of 15 real sequences)."""
    listed = list(ends)
    if not listed:
        return 0.0
    return max(listed) / rate - start_seconds


def _check_extents(sets: list[ItemSet], board: ItemScoreboard) -> None:
    for items in sets:
        ends: dict[str, list[int] | None] = defaultdict(list)
        for rows in items.items:
            timeline, _ = items.tracks.get(str(rows.item.get("Sm2TiTrack_id")), ("", 0))
            start = frames_text(rows.item.get("Start"))
            duration = frames_text(rows.item.get("Duration"))
            known = ends[timeline]
            if start is None or duration is None or known is None:
                ends[timeline] = None
            else:
                known.append(start + duration)
        for timeline, (rate_bytes, extents) in items.sequences.items():
            listed = ends.get(timeline, [])
            if (
                listed is None
                or not isinstance(rate_bytes, bytes)
                or not isinstance(extents, bytes)
            ):
                board.extents_skipped += 1
                continue
            try:
                rate = blobs.decode_rate(rate_bytes)
                start_s, length = blobs.decode_extents(extents)
            except DrpError:
                board.extents_skipped += 1
                continue
            board.extents_checked += 1
            if not listed:
                continue  # an empty sequence keeps the length it had (seen: 0.0 and stale)
            want = extents_seconds(rate, start_s, listed)
            if struct.pack("<d", want) != struct.pack("<d", length):
                board.extents_differing.append(
                    f"{items.path.name}: a sequence of {len(listed)} items stores {length!r}, "
                    f"its items give {want!r}"
                )


@refusing(
    "bad_template",
    "running the item gate",
    "check the projects",
    error=ItemError,
    passthrough=(LibraryError,),
)
def run(
    projects: list[Path],
    *,
    options: LibraryOptions | None = None,
    other_project: bool = False,
    linked_templates: bool = False,
) -> ItemScoreboard:
    """Author and compare every supported item of ``projects`` (see the module docstring).
    ``other_project`` takes each template from another project when there is one (the
    templates of the product come from other projects than the cut's); ``linked_templates``
    takes the template of an unlinked item from the linked items of its kind (the writer uses
    a linked template for an unlinked item when no unlinked one is given)."""
    sets = [read_items(p, options=options) for p in projects]
    refused: dict[str, int] = defaultdict(int)
    targets: list[_Target] = []
    notes: list[str] = []
    for items in sets:
        notes += [f"{items.path.name}: {n}" for n in items.skipped[:20]]
        for rows in items.items:
            shape = shape_of(rows, items.clips)
            if shape.refused is not None:
                refused[f"{shape.kind or 'other'}: {shape.refused}"[:120]] += 1
                continue
            targets.append(_Target(items.path.name, items, rows, shape, bool(links_of(rows.item))))
    board = ItemScoreboard([], dict(refused), notes=notes)
    by_group: dict[str, GroupScore] = {}
    pools: dict[tuple[str | None, bool], list[_Target]] = defaultdict(list)
    for t in targets:
        pools[(t.shape.kind, t.linked)].append(t)
    for target in targets:
        group = by_group.setdefault(_group(target), GroupScore(_group(target)))
        linked = target.linked or linked_templates
        template = _template(target, pools[(target.shape.kind, linked)], other_project)
        if template is None:
            group.refused += 1
            group.refusals["no other item of its kind to use as the template"] = (
                group.refusals.get("no other item of its kind to use as the template", 0) + 1
            )
            continue
        _check(group, target, template)
    board.groups = sorted(by_group.values(), key=lambda g: g.group)
    _check_extents(sets, board)
    return board


def _template(target: _Target, pool: list[_Target], other_project: bool) -> _Target | None:
    """Another item of the same kind and linkage: the same project first (another project
    first with ``other_project``), then one playing another media clip, then one carrying no
    later additions."""
    media = target.rows.item.get("MediaRef")
    best: _Target | None = None
    best_key: tuple[int, int, int] | None = None
    for other in pool:
        if other is target:
            continue
        key = (
            (other.project == target.project) == other_project,
            other.rows.item.get("MediaRef") == media,
            len(other.shape.tags),
        )
        if best_key is None or key < best_key:
            best, best_key = other, key
            if key == (False, False, 0):
                break  # cannot do better
    return best


def _check(group: GroupScore, target: _Target, template: _Target) -> None:
    item = target.rows.item
    clip = target.items.clips[str(item["MediaRef"])]
    tags = set(target.shape.tags) | {str(target.shape.kind)}
    try:
        view = media_view(clip)
        name = item.get("Name")
        if name != view.name:
            tags.add("renamed")
        if _exact_start(target):
            tags.add("exact-start")
        elif _start_last_bit(target, view.start_seconds):
            tags.add("start-last-bit")
        start = frames_text(item.get("Start"))
        duration = frames_text(item.get("Duration"))
        src_in = frames_text(item.get("In")) if item.get("In") is not None else 0
        assert start is not None and duration is not None and src_in is not None
        placement = ItemPlacement(
            kind=str(target.shape.kind),
            track_id=str(item["Sm2TiTrack_id"]),
            index=int(target.rows.link["DbIndex"]),
            start=start,
            duration=duration,
            src_in=src_in,
            partners=links_of(item),
            name=str(name) if "renamed" in tags else None,
        )
        authored = author_item(
            clip,
            template.rows,
            template.items.clips,
            placement,
            item_id=target.rows.item_id,
        )
    except (MediaPoolError, ItemError) as exc:
        group.refused += 1
        key = f"{exc.code}: {exc.message}"[:160]
        group.refusals[key] = group.refusals.get(key, 0) + 1
        return
    group.checked += 1
    tags.add("linked" if target.linked else "unlinked")
    by_id = {r.item_id: r.item for r in target.items.items}
    for partner in placement.partners:
        other = by_id.get(partner)
        if other is not None and any(other.get(c) != item.get(c) for c in _RANGE):
            tags.add("split (its linked item has another range)")
    if src_in == 0:
        tags.add("from frame 0")
    for tag in tags - {"picture", "sound"}:
        group.shapes[tag] = group.shapes.get(tag, 0) + 1
    compared, diffs, hidden = compare_items(target.rows, authored.rows(), tags)
    group.fields += compared
    group.equal += compared - len(diffs)
    for d in diffs:
        group.differing[d.path] = group.differing.get(d.path, 0) + 1
        group.examples.setdefault(d.path, f"Resolve {d.expected} authored {d.got}"[:300])
    for rule, _ in hidden:
        group.hidden[rule.pattern] = group.hidden.get(rule.pattern, 0) + 1


@refusing("bad_template", "formatting the scoreboard", "report it", error=ItemError)
def format_scoreboard(board: ItemScoreboard) -> str:
    """The scoreboard as text (no names of files or items: only paths of fields and counts)."""
    lines: list[str] = []
    for g in board.groups:
        lines.append(
            f"{g.status:8} {g.group}: checked {g.checked}, refused {g.refused}, "
            f"fields {g.fields}, equal {g.equal}"
        )
        lines.append(
            "         shapes: " + ", ".join(f"{k} {v}" for k, v in sorted(g.shapes.items()))
        )
        for path, count in sorted(g.differing.items()):
            lines.append(f"         DIFF {path} x{count}: {g.examples[path]}")
        for reason, count in sorted(g.refusals.items()):
            lines.append(f"         REFUSED x{count}: {reason}")
        for pattern, count in sorted(g.hidden.items()):
            lines.append(f"         excluded and differing: {pattern} in {count} field(s)")
    lines.append(f"items checked: {board.items_checked}")
    lines.append("items of an unsupported shape (not authored, not compared):")
    for reason, count in sorted(board.refused.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {count:6}  {reason}")
    lines.append(
        f"sequence extents: checked {board.extents_checked}, differing "
        f"{len(board.extents_differing)}, skipped (an item not in whole frames, or unreadable) "
        f"{board.extents_skipped}"
    )
    lines += [f"  EXTENTS {line}" for line in board.extents_differing]
    return "\n".join(lines)
