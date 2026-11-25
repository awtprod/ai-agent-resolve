"""The transition gate: author every real transition from its cut and a *different* real one.

For every ``Sm2TiTransition`` of the given Resolve-saved projects, :func:`run` finds its two
neighbours in its track's ``Items`` vector (the clip before it and the clip after it, which
must abut: the frame where they meet is the cut), authors the transition from that cut, its
length, its alignment, its kind's name, its track and its index alone, with another real
transition as the template (leave-one-out, never itself: identity by project and row id), and
compares the result with Resolve's own rows field by field (the media pool's total comparison).
``Start`` is therefore checked, not copied: it is computed from the neighbours' cut and the
alignment rule.

It also checks, on every transition of a supported shape, what a transition needs of and does
to its neighbours: the media it plays (``before`` frames ahead of the incoming clip's in point,
``after`` frames past the outgoing clip's out point, which must exist in the clip's sound: the
smallest margin is reported), that its window lies inside the two clips, and that the two
neighbours themselves are reproduced by the item writer, which knows nothing of transitions
(their rows hold nothing about it).

Transitions of a shape the writer does not support are counted per reason, never dropped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.items import gate as item_gate
from aar.drp.items.errors import ItemError
from aar.drp.items.media import media_view
from aar.drp.items.rows import ItemRows, ItemSet, frames_text, links_of, read_items, shape_of
from aar.drp.items.transitions import (
    TRANSITION,
    TransitionPlacement,
    TransitionShape,
    author_transition,
    frames_before,
    transition_shape,
)
from aar.drp.library import LibraryError, LibraryOptions
from aar.drp.mediapool.errors import MediaPoolError, refusing
from aar.drp.mediapool.fields import Exclusion, compare_flat, flatten_rows
from aar.media.probe import snap_rate

__all__ = [
    "TRANSITION_EXCLUDED",
    "TransitionScoreboard",
    "format_scoreboard",
    "run",
]

#: Every field the transition gate leaves out, and why (``media`` names the shape tags it
#: applies to). Nothing else is left out.
TRANSITION_EXCLUDED: tuple[Exclusion, ...] = (
    Exclusion(
        r"I\.FieldsBlob>#1(\..*)?",
        "project-local",
        "FieldsBlob #1 (#1.#21, the transition's record start) is on 520 of the 854 sound "
        "transitions and on none of the other 334: state a later edit leaves; a transition "
        "Resolve has just made has none",
        frozenset({"history"}),
    ),
)

_CLIPS = ("Sm2TiVideoClip", "Sm2TiAudioClip")


def _excluded(path: str, tags: frozenset[str]) -> Exclusion | None:
    return next((e for e in TRANSITION_EXCLUDED if any(e.applies(path, t) for t in tags)), None)


@dataclass
class TransitionScoreboard:
    groups: list[item_gate.GroupScore]
    #: Transitions of a shape the writer does not support, by reason.
    refused: dict[str, int]
    #: The neighbours of compared transitions, reproduced by the item writer.
    neighbours: item_gate.GroupScore = field(
        default_factory=lambda: item_gate.GroupScore("neighbours")
    )
    #: Neighbours of an item shape the writer does not support, by reason.
    neighbours_refused: dict[str, int] = field(default_factory=dict)
    #: The smallest number of frames of media left beyond what a transition plays, on the
    #: incoming clip (ahead of its in point) and on the outgoing clip (past its out point).
    min_margin_in: int | None = None
    min_margin_out: int | None = None
    #: Transitions whose media margins were checked, and those that could not be (a
    #: neighbour's range not in plain whole frames, or its media not a file clip), by reason.
    margins_checked: int = 0
    margins_unchecked: dict[str, int] = field(default_factory=dict)
    #: Transitions whose window or media does not fit their neighbours (a rule that is wrong).
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def transitions_checked(self) -> int:
        return sum(g.checked for g in self.groups)

    def passed(self) -> bool:
        return (
            all(g.status == "equal" for g in self.groups)
            and self.neighbours.status == "equal"
            and not self.violations
        )


@dataclass
class _Target:
    project: str
    items: ItemSet
    rows: ItemRows
    shape: TransitionShape
    cut: int = 0
    rate: Fraction = Fraction(0)
    outgoing: ItemRows | None = None
    incoming: ItemRows | None = None


def _track_orders(items: ItemSet) -> dict[str, list[ItemRows]]:
    by_track: dict[str, list[ItemRows]] = defaultdict(list)
    for rows in items.items:
        by_track[str(rows.link.get("DbOwner"))].append(rows)
    for listed in by_track.values():
        listed.sort(
            key=lambda r: r.link.get("DbIndex") if isinstance(r.link.get("DbIndex"), int) else -1
        )
    return by_track


def _neighbours(target: _Target, order: list[ItemRows]) -> str | None:
    """Find the transition's two neighbours and its cut; the reason it has none otherwise."""
    position = next(i for i, r in enumerate(order) if r.item_id == target.rows.item_id)
    if position == 0 or position + 1 >= len(order):
        return "at the head or tail of its track (no clip on one side)"
    before, after = order[position - 1], order[position + 1]
    if before.item.get("DbType") not in _CLIPS or after.item.get("DbType") not in _CLIPS:
        return "a neighbour is not a clip"
    out_start, out_length = (
        frames_text(before.item.get("Start")),
        frames_text(before.item.get("Duration")),
    )
    in_start = frames_text(after.item.get("Start"))
    if out_start is None or out_length is None or in_start is None:
        return "a neighbour is not in whole frames"
    if out_start + out_length != in_start:
        return "its neighbours do not abut (a gap or an overlap: not a join)"
    target.cut, target.outgoing, target.incoming = in_start, before, after
    return None


def _rate(items: ItemSet, rows: ItemRows) -> Fraction | None:
    timeline, _ = items.tracks.get(str(rows.item.get("Sm2TiTrack_id")), ("", 0))
    found = items.sequences.get(timeline)
    if found is None or not isinstance(found[0], bytes):
        return None
    try:
        return snap_rate(Fraction(blobs.decode_rate(found[0])))
    except (DrpError, ValueError, ZeroDivisionError):
        return None


@refusing(
    "bad_template",
    "running the transition gate",
    "check the projects",
    error=ItemError,
    passthrough=(LibraryError,),
)
def run(
    projects: list[Path],
    *,
    options: LibraryOptions | None = None,
    other_project: bool = False,
    other_kind: bool = False,
) -> TransitionScoreboard:
    """Author and compare every supported transition of ``projects`` (module docstring).
    ``other_project`` takes each template from another project when there is one;
    ``other_kind`` prefers a template of another kind (``PrettyType`` is derived from the kind
    alone)."""
    unique: dict[str, Path] = {}
    for path in projects:
        unique.setdefault(str(Path(path).resolve()), Path(path))
    keyed = [(key, read_items(path, options=options)) for key, path in unique.items()]
    refused: dict[str, int] = defaultdict(int)
    targets: list[_Target] = []
    board = TransitionScoreboard([], {})
    if len(unique) < len(projects):
        board.notes.append(f"{len(projects) - len(unique)} project(s) given twice were read once")
    item_targets: dict[tuple[str, str], item_gate._Target] = {}
    item_pools: dict[tuple[str | None, bool], list[item_gate._Target]] = defaultdict(list)
    item_reasons: dict[tuple[str, str], str] = {}
    for project, items in keyed:
        orders = _track_orders(items)
        for rows in items.items:
            if rows.item.get("DbType") in _CLIPS:
                shape = shape_of(rows, items.clips)
                if shape.refused is None:
                    t = item_gate._Target(project, items, rows, shape, bool(links_of(rows.item)))
                    item_targets[(project, rows.item_id)] = t
                    item_pools[(shape.kind, t.linked)].append(t)
                else:
                    item_reasons[(project, rows.item_id)] = f"{shape.kind}: {shape.refused}"[:120]
                continue
            if rows.item.get("DbType") != TRANSITION:
                continue
            track = str(rows.item.get("Sm2TiTrack_id"))
            _, track_type = items.tracks.get(track, ("", None))  # type: ignore[assignment]
            shape = transition_shape(rows, track_type)
            if shape.refused is not None:
                refused[f"{shape.kind or 'other'}: {shape.refused}"[:140]] += 1
                continue
            target = _Target(project, items, rows, shape)
            why = _neighbours(target, orders[track])
            rate = _rate(items, rows)
            if why is None and rate is None:
                why = "its timeline's rate does not read"
            if why is not None:
                refused[f"sound: {why}"[:140]] += 1
                continue
            assert rate is not None
            target.rate = rate
            targets.append(target)
    board.refused = dict(refused)
    by_group: dict[str, item_gate.GroupScore] = {}
    neighbours_done: set[tuple[str, str]] = set()
    for target in targets:
        name = f"sound {target.shape.name}, {target.shape.alignment}"
        group = by_group.setdefault(name, item_gate.GroupScore(name))
        _margins(board, target)
        template = _template(target, targets, other_project, other_kind)
        if template is None:
            group.refused += 1
            reason = "no other transition to use as the template"
            group.refusals[reason] = group.refusals.get(reason, 0) + 1
        else:
            _check(group, target, template)
        for neighbour in (target.outgoing, target.incoming):
            assert neighbour is not None
            key = (target.project, neighbour.item_id)
            if key in neighbours_done:
                continue
            neighbours_done.add(key)
            _check_neighbour(board, key, item_targets, item_pools, item_reasons, other_project)
    board.groups = sorted(by_group.values(), key=lambda g: g.group)
    return board


def _margins(board: TransitionScoreboard, target: _Target) -> None:
    """What the transition plays of its neighbours' media, and that its window lies in them."""
    assert target.outgoing is not None and target.incoming is not None
    frames = target.shape.frames
    assert frames is not None and target.shape.alignment is not None
    before = frames_before(target.shape.alignment, frames)
    after = frames - before
    out_row, in_row = target.outgoing.item, target.incoming.item
    out_in = frames_text(out_row.get("In")) if out_row.get("In") is not None else 0
    in_in = frames_text(in_row.get("In")) if in_row.get("In") is not None else 0
    out_length = frames_text(out_row.get("Duration"))
    in_length = frames_text(in_row.get("Duration"))
    where = f"{target.items.path.name}: a {frames}-frame transition at {target.cut}"

    def unchecked(reason: str) -> None:
        board.margins_unchecked[reason] = board.margins_unchecked.get(reason, 0) + 1

    if out_length is None or in_length is None:
        unchecked("a neighbour's length is not in plain whole frames")
        return
    if before > out_length or after > in_length:
        board.violations.append(f"{where}: its window runs past a neighbour")
    if out_in is None or in_in is None:
        unchecked("a neighbour's in point is not in plain whole frames")
        return
    try:
        out_clip = target.items.clips[str(out_row.get("MediaRef"))]
        available = media_view(out_clip).sound_frames
    except (KeyError, MediaPoolError):
        unchecked("the outgoing clip's media is not a file clip (compound, multicam...)")
        return
    if available is None:
        board.violations.append(f"{where}: the outgoing clip has no sound")
        return
    board.margins_checked += 1
    margin_out = available - (out_in + out_length + after)
    margin_in = in_in - before
    for margin, side in ((margin_in, "in"), (margin_out, "out")):
        if margin < 0:
            board.violations.append(f"{where}: it reads {-margin} frame(s) past the {side} side")
    board.min_margin_in = (
        margin_in if board.min_margin_in is None else min(board.min_margin_in, margin_in)
    )
    board.min_margin_out = (
        margin_out if board.min_margin_out is None else min(board.min_margin_out, margin_out)
    )


def _template(
    target: _Target, pool: list[_Target], other_project: bool, other_kind: bool
) -> _Target | None:
    best: _Target | None = None
    best_key: tuple[bool, bool, int] | None = None
    for other in pool:
        if other.project == target.project and other.rows.item_id == target.rows.item_id:
            continue  # never its own template, whichever read it came from
        key = (
            (other.project == target.project) == other_project,
            (other.shape.name == target.shape.name) == other_kind,
            len(other.shape.tags),
        )
        if best_key is None or key < best_key:
            best, best_key = other, key
            if key == (False, False, 0):
                break
    return best


def _check(group: item_gate.GroupScore, target: _Target, template: _Target) -> None:
    shape = target.shape
    assert shape.name is not None and shape.alignment is not None and shape.frames is not None
    tags = set(shape.tags) | {f"{shape.frames} frames"}
    if template.shape.name != shape.name:
        tags.add("template of another kind")
    if template.project != target.project:
        tags.add("template from another project")
    try:
        placement = TransitionPlacement(
            name=shape.name,
            track_id=str(target.rows.item["Sm2TiTrack_id"]),
            index=int(target.rows.link["DbIndex"]),
            cut=target.cut,
            frames=shape.frames,
            alignment=shape.alignment,
            rate=target.rate,
        )
        authored = author_transition(template.rows, placement, transition_id=target.rows.item_id)
    except (MediaPoolError, ValueError) as exc:
        group.refused += 1
        key = f"{getattr(exc, 'code', type(exc).__name__)}: {getattr(exc, 'message', exc)}"[:160]
        group.refusals[key] = group.refusals.get(key, 0) + 1
        return
    group.checked += 1
    for tag in tags:
        group.shapes[tag] = group.shapes.get(tag, 0) + 1
    roles = {target.rows.item_id: "<self>", str(target.rows.item["Sm2TiTrack_id"]): "<T>"}
    frozen = frozenset(tags)
    compared, diffs, hidden = compare_flat(
        flatten_rows([("I", target.rows.item), ("J", target.rows.link)], roles),
        flatten_rows([("I", authored.item), ("J", authored.link)], roles),
        lambda path: _excluded(path, frozen),
    )
    group.fields += compared
    group.equal += compared - len(diffs)
    for d in diffs:
        group.differing[d.path] = group.differing.get(d.path, 0) + 1
        group.examples.setdefault(d.path, f"Resolve {d.expected} authored {d.got}"[:300])
    for rule, _ in hidden:
        group.hidden[rule.pattern] = group.hidden.get(rule.pattern, 0) + 1


def _check_neighbour(
    board: TransitionScoreboard,
    key: tuple[str, str],
    item_targets: dict[tuple[str, str], item_gate._Target],
    item_pools: dict[tuple[str | None, bool], list[item_gate._Target]],
    item_reasons: dict[tuple[str, str], str],
    other_project: bool,
) -> None:
    """Author a neighbour with the item writer (which knows nothing of transitions) and compare
    it with Resolve's: a transition that changed its neighbours' rows would show here."""
    target = item_targets.get(key)
    if target is None:
        reason = item_reasons.get(key, "not an item shape the writer supports")
        board.neighbours_refused[reason] = board.neighbours_refused.get(reason, 0) + 1
        return
    pool = item_pools[(target.shape.kind, target.linked)]
    template = item_gate._template(target, pool, other_project)
    if template is None:
        board.neighbours.refused += 1
        return
    item_gate._check(board.neighbours, target, template)


@refusing("bad_template", "formatting the scoreboard", "report it", error=ItemError)
def format_scoreboard(board: TransitionScoreboard) -> str:
    """The scoreboard as text (counts and field paths only, no names)."""
    lines: list[str] = []
    for g in [*board.groups, board.neighbours]:
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
    lines.append(f"transitions checked: {board.transitions_checked}")
    lines.append(
        f"media a transition plays, checked on {board.margins_checked}: smallest margin ahead "
        f"of the incoming clip's in point {board.min_margin_in}, past the outgoing clip's out "
        f"point {board.min_margin_out}"
    )
    for reason, count in sorted(board.margins_unchecked.items()):
        lines.append(f"  margins not checked x{count}: {reason}")
    lines += [f"  VIOLATION {v}" for v in board.violations]
    for reason, count in sorted(board.neighbours_refused.items()):
        lines.append(f"  neighbours not compared x{count}: {reason}")
    lines.append("transitions of an unsupported shape (not authored, not compared):")
    for reason, count in sorted(board.refused.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {count:6}  {reason}")
    lines += [f"  note: {n}" for n in board.notes[:20]]
    return "\n".join(lines)
