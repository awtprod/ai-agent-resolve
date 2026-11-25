"""A cut list as timeline items on the seed's two tracks.

:func:`author_cut` turns :class:`Cut` entries (a media clip, its source range and its place in
the timeline; picture, sound or both, and for a split edit the sound's own range) into authored
items (:func:`aar.drp.items.author_item`) on the seed timeline's video and audio track, with the
picture and the sound of one entry linked to each other, and returns the items as the read-back
must find them. :class:`CutTransition` entries put a sound crossfade on a join of two entries
(:func:`aar.drp.items.transitions.author_transition`): the outgoing entry's sound must end
exactly where another entry's sound starts (a join, not a gap or a split edit), and both clips
must hold the media the transition plays beyond the cut (refused ``bad_transition``, never
shortened); the two transitions on one clip must not overlap inside it (refused
``transitions_overlap``, the plan export's code for the same rule). On each track the items
are in record order, a transition before a clip that starts on the same frame (Resolve's
order). Template items come from :func:`pick_item_templates`:
plain items Resolve saved, one per kind and linkage, and one plain sound transition.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from aar.drp.items.author import AuthoredItem, ItemPlacement, author_item
from aar.drp.items.errors import ItemError
from aar.drp.items.media import media_view
from aar.drp.items.rows import ItemRows, ItemSet, links_of, shape_of
from aar.drp.items.transitions import (
    TRANSITION,
    TransitionPlacement,
    author_transition,
    frames_before,
    transition_shape,
)
from aar.drp.libwrite.errors import ProjectWriteError, refusing
from aar.drp.libwrite.seed import Seed
from aar.drp.libwrite.write import ItemSpec
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.templates import ClipRows

__all__ = ["Cut", "CutTransition", "ItemTemplates", "author_cut", "pick_item_templates"]


@dataclass(frozen=True)
class Cut:
    """One entry of the cut: ``clip`` from ``src_in`` for ``duration`` frames at ``record_in``
    (frames from the timeline's start). ``sound_*`` give the sound its own range (a split
    edit); None keeps the picture's."""

    clip: ClipRows
    record_in: int
    duration: int
    src_in: int
    picture: bool = True
    sound: bool = True
    sound_record_in: int | None = None
    sound_src_in: int | None = None
    sound_duration: int | None = None


@dataclass(frozen=True)
class CutTransition:
    """A sound crossfade on the join after cut entry ``after`` (an index into the cut list):
    Resolve's ``name`` of its kind (``Cross Fade 0 dB``...), its length and its alignment on
    the cut."""

    after: int
    frames: int
    name: str
    alignment: str = "centre"


@dataclass(frozen=True)
class ItemTemplates:
    """A template item per ``(kind, linked)``, each with the media clips of its project, and a
    template sound transition (None when none was given)."""

    by_key: dict[tuple[str, bool], tuple[ItemRows, dict[str, ClipRows]]]
    sources: tuple[str, ...] = ()
    transition: ItemRows | None = None

    def get_transition(self) -> ItemRows:
        if self.transition is None:
            raise ItemError(
                "no_template",
                "no sound transition to use as the template",
                "give --templates a project Resolve saved whose timeline has a plain sound "
                "crossfade between two clips",
            )
        return self.transition

    def get(self, kind: str, linked: bool) -> tuple[ItemRows, dict[str, ClipRows]]:
        found = self.by_key.get((kind, linked))
        if found is None and not linked:
            # The item gate's linked-templates run: an unlinked item authored from a linked
            # template equals Resolve's on every real item.
            found = self.by_key.get((kind, True))
        if found is None:
            raise ItemError(
                "no_template",
                f"no {'linked' if linked else 'unlinked'} {kind} item to use as the template",
                "give --templates a project Resolve saved whose timeline has such an item",
            )
        return found


@refusing(
    "bad_input",
    "picking the template items",
    "check the template projects",
    passthrough=(MediaPoolError,),
)
def pick_item_templates(sets: Sequence[ItemSet]) -> ItemTemplates:
    """The first plain item (supported shape, no later additions) of each kind and linkage, and
    the first plain sound transition (on an audio track, no later-edit state), in the order the
    projects and their items are given."""
    by_key: dict[tuple[str, bool], tuple[ItemRows, dict[str, ClipRows]]] = {}
    transition: ItemRows | None = None
    for items in sets:
        for rows in items.items:
            if rows.item.get("DbType") == TRANSITION:
                _, track_type = items.tracks.get(str(rows.item.get("Sm2TiTrack_id")), ("", -1))
                tshape = transition_shape(rows, track_type)
                if transition is None and tshape.refused is None and not tshape.tags:
                    transition = rows
                continue
            shape = shape_of(rows, items.clips)
            if shape.refused is not None or shape.tags or shape.kind is None:
                continue
            key = (shape.kind, bool(links_of(rows.item)))
            by_key.setdefault(key, (rows, items.clips))
    return ItemTemplates(by_key, tuple(str(s.path) for s in sets), transition)


def _bad(why: str) -> ProjectWriteError:
    return ProjectWriteError("bad_input", why, "check the cut list")


def _check_cut(cut: object) -> Cut:
    """The entry's own fields: the record frames are whole numbers from the timeline's start
    (the source range and duration are checked with the placement, against the clip), and a
    split edit's sound range belongs to an entry that has a sound."""
    if not isinstance(cut, Cut):
        raise _bad(f"a cut entry is a {type(cut).__name__}, not a Cut")
    if not isinstance(cut.picture, bool) or not isinstance(cut.sound, bool):
        raise _bad("a cut entry's picture and sound are not True or False")
    for name in ("record_in", "sound_record_in"):
        value = getattr(cut, name)
        if value is None and name != "record_in":
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _bad(f"a cut entry's {name} {value!r} is not a whole frame from 0")
    split = (cut.sound_record_in, cut.sound_src_in, cut.sound_duration)
    if not cut.sound and any(v is not None for v in split):
        raise _bad("a cut entry gives its sound its own range, but has no sound")
    return cut


@refusing("bad_input", "authoring the cut", "check the cut list", passthrough=(MediaPoolError,))
def author_cut(
    seed: Seed,
    cuts: Sequence[Cut],
    templates: ItemTemplates,
    *,
    transitions: Sequence[CutTransition] = (),
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> tuple[list[AuthoredItem], list[ItemSpec]]:
    """The items of ``cuts`` and the sound crossfades of ``transitions`` on the seed's tracks
    (see the module docstring)."""
    taken = set(seed.taken)

    def fresh() -> str:
        value = new_id()
        if not isinstance(value, str) or len(value) != 36 or value in taken:
            raise _bad(f"the id source gave {value!r}, not a fresh uuid")
        taken.add(value)
        return value

    planned: list[tuple[str, Cut, str, tuple[str, ...], int, int, int]] = []
    for cut in cuts:
        _check_cut(cut)
        if not cut.picture and not cut.sound:
            raise _bad("a cut entry has neither picture nor sound")
        view = media_view(cut.clip)
        has_sound = cut.sound and bool(view.channels)
        if cut.sound and not view.channels:
            raise _bad(f"{view.name} has no sound to place")
        picture_id = fresh() if cut.picture else ""
        sound_id = fresh() if has_sound else ""
        if cut.picture:
            partners = (sound_id,) if has_sound else ()
            planned.append(
                (picture_id, cut, "picture", partners, cut.record_in, cut.src_in, cut.duration)
            )
        if has_sound:
            partners = (picture_id,) if cut.picture else ()
            planned.append(
                (
                    sound_id,
                    cut,
                    "sound",
                    partners,
                    cut.record_in if cut.sound_record_in is None else cut.sound_record_in,
                    cut.src_in if cut.sound_src_in is None else cut.sound_src_in,
                    cut.duration if cut.sound_duration is None else cut.sound_duration,
                )
            )
    joins = _joins(seed, cuts, transitions)
    # (record start, 0 for a transition / 1 for a clip, ...): Resolve's order on a track.
    entries: dict[str, list[tuple[int, int, int, object]]] = {"picture": [], "sound": []}
    for n, plan in enumerate(planned):
        entries[plan[2]].append((plan[4], 1, n, plan))
    for n, join in enumerate(joins):
        entries["sound"].append((join.cut - join.before, 0, n, join))
    items: list[AuthoredItem] = []
    specs: list[ItemSpec] = []
    for kind, track in (("picture", seed.video_track), ("sound", seed.audio_track)):
        mine = sorted(entries[kind], key=lambda e: e[:3])
        for index, (_, is_clip, _, entry) in enumerate(mine):
            if not is_clip:
                assert isinstance(entry, _Join)
                placement_t = TransitionPlacement(
                    name=entry.name,
                    track_id=track,
                    index=index,
                    cut=seed.start_frame + entry.cut,
                    frames=entry.frames,
                    alignment=entry.alignment,
                    rate=seed.rate,
                )
                items.append(
                    author_transition(
                        templates.get_transition(),
                        placement_t,
                        transition_id=fresh(),
                        taken=seed.taken,
                    )
                )
                start = entry.cut - entry.before
                specs.append(ItemSpec("audio", start, start + entry.frames, 0, ""))
                continue
            item_id, cut, _, partners, record_in, src_in, duration = entry  # type: ignore[misc]
            template, clips = templates.get(kind, bool(partners))
            placement = ItemPlacement(
                kind=kind,
                track_id=track,
                index=index,
                start=seed.start_frame + record_in,
                duration=duration,
                src_in=src_in,
                partners=partners,
            )
            items.append(
                author_item(cut.clip, template, clips, placement, item_id=item_id, taken=seed.taken)
            )
            specs.append(
                ItemSpec(
                    "video" if kind == "picture" else "audio",
                    record_in,
                    record_in + duration,
                    src_in,
                    media_view(cut.clip).path,
                )
            )
    return items, specs


@dataclass(frozen=True)
class _Join:
    """A transition placed on a join: its cut (record frame from the timeline's start)."""

    cut: int
    frames: int
    before: int
    name: str
    alignment: str


def _refuse(why: str, **details: object) -> ProjectWriteError:
    return ProjectWriteError(
        "bad_transition",
        why,
        "put the crossfade on a join of two clips whose media run on past the cut, or add it "
        "in Resolve",
        **details,
    )


def _joins(seed: Seed, cuts: Sequence[Cut], transitions: Sequence[CutTransition]) -> list[_Join]:
    """Each transition's join: the outgoing entry's sound ends where exactly one other entry's
    sound starts, neither is a split edit, both clips hold the media it plays, and the two
    transitions on one clip do not overlap (:func:`_check_overlaps`)."""
    out: list[_Join] = []
    seen: set[int] = set()
    # Frames of each entry the transitions cover: (entry, frames) for each side of each one.
    covered: list[tuple[int, int, str]] = []
    for number, tx in enumerate(transitions):
        where = f"transition {number}"
        if not isinstance(tx, CutTransition):
            raise _bad(f"{where} is a {type(tx).__name__}, not a CutTransition")
        for name in ("after", "frames"):
            value = getattr(tx, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise _bad(f"{where}'s {name} {value!r} is not a whole number")
        if not isinstance(tx.name, str) or not isinstance(tx.alignment, str):
            raise _bad(f"{where}'s name or alignment is not text")
        if not 0 <= tx.after < len(cuts):
            raise _refuse(f"{where} follows cut entry {tx.after}, which does not exist")
        if tx.after in seen:
            raise _refuse(f"{where}: the join after cut entry {tx.after} has two transitions")
        seen.add(tx.after)
        if tx.frames < 1:
            raise _refuse(f"{where} is {tx.frames} frames long")
        out_cut = cuts[tx.after]
        if not out_cut.sound:
            raise _refuse(f"{where} follows cut entry {tx.after}, which has no sound")
        end = out_cut.record_in + out_cut.duration
        incoming = [
            i for i, c in enumerate(cuts) if i != tx.after and c.sound and c.record_in == end
        ]
        if len(incoming) != 1:
            raise _refuse(
                f"{where}: no single clip's sound starts where cut entry {tx.after}'s ends "
                f"(frame {end}): not a join (a gap, or nothing follows)"
            )
        in_cut = cuts[incoming[0]]
        for index, c in ((tx.after, out_cut), (incoming[0], in_cut)):
            if any(v is not None for v in (c.sound_record_in, c.sound_src_in, c.sound_duration)):
                raise _refuse(
                    f"{where}: cut entry {index} is a split edit; a crossfade on a split edit "
                    "was never seen"
                )
        try:
            before = frames_before(tx.alignment, tx.frames)
        except ItemError as exc:
            raise _refuse(f"{where}: {exc.message}") from None
        after = tx.frames - before
        if before > out_cut.duration or after > in_cut.duration:
            raise _refuse(
                f"{where}: a {tx.frames}-frame transition is longer than a clip it joins "
                f"({out_cut.duration} and {in_cut.duration} frames)"
            )
        out_view, in_view = media_view(out_cut.clip), media_view(in_cut.clip)
        if out_view.sound_frames is None or in_view.sound_frames is None:
            raise _refuse(f"{where}: a clip it joins has no sound")
        past = out_cut.src_in + out_cut.duration + after
        if past > out_view.sound_frames:
            raise _refuse(
                f"{where}: it plays {after} frame(s) past the outgoing clip's out point, to "
                f"source frame {past}, but its sound has {out_view.sound_frames} frames",
                needed=after,
                available=out_view.sound_frames - out_cut.src_in - out_cut.duration,
            )
        if in_cut.src_in < before:
            raise _refuse(
                f"{where}: it plays {before} frame(s) ahead of the incoming clip's in point, "
                f"but the clip starts {in_cut.src_in} frame(s) into its file",
                needed=before,
                available=in_cut.src_in,
            )
        out.append(_Join(end, tx.frames, before, tx.name, tx.alignment))
        covered.append((tx.after, before, where))
        covered.append((incoming[0], after, where))
    _check_overlaps(cuts, covered)
    return out


def _check_overlaps(cuts: Sequence[Cut], covered: list[tuple[int, int, str]]) -> None:
    """The frames the transitions on one entry cover (the part of the incoming one after its
    cut and of the outgoing one before its cut) fit in its sound: two transitions' windows
    never overlap inside one clip. The plan export's rule, with its code."""
    used: dict[int, list[tuple[int, str]]] = {}
    for index, frames, where in covered:
        used.setdefault(index, []).append((frames, where))
    for index, parts in sorted(used.items()):
        total = sum(frames for frames, _ in parts)
        if len(parts) > 1 and total > cuts[index].duration:
            raise ProjectWriteError(
                "transitions_overlap",
                f"cut entry {index}: {' and '.join(w for _, w in parts)} need {total} frames of "
                f"its {cuts[index].duration}, so they would overlap",
                "shorten the transitions or lengthen the clip",
                entry=index,
                needed=total,
                available=cuts[index].duration,
            )
