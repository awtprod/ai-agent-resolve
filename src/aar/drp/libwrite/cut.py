"""A cut list as timeline items on the seed's two tracks.

:func:`author_cut` turns :class:`Cut` entries (a media clip, its source range and its place in
the timeline; picture, sound or both, and for a split edit the sound's own range) into authored
items (:func:`aar.drp.items.author_item`) on the seed timeline's video and audio track, with the
picture and the sound of one entry linked to each other, and returns the items as the read-back
must find them. Template items come from :func:`pick_item_templates`: plain items Resolve saved,
one per kind and linkage.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from aar.drp.items.author import AuthoredItem, ItemPlacement, author_item
from aar.drp.items.errors import ItemError
from aar.drp.items.media import media_view
from aar.drp.items.rows import ItemRows, ItemSet, links_of, shape_of
from aar.drp.libwrite.errors import ProjectWriteError, refusing
from aar.drp.libwrite.seed import Seed
from aar.drp.libwrite.write import ItemSpec
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.templates import ClipRows

__all__ = ["Cut", "ItemTemplates", "author_cut", "pick_item_templates"]


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
class ItemTemplates:
    """A template item per ``(kind, linked)``, each with the media clips of its project."""

    by_key: dict[tuple[str, bool], tuple[ItemRows, dict[str, ClipRows]]]
    sources: tuple[str, ...] = ()

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
    """The first plain item (supported shape, no later additions) of each kind and linkage,
    in the order the projects and their items are given."""
    by_key: dict[tuple[str, bool], tuple[ItemRows, dict[str, ClipRows]]] = {}
    for items in sets:
        for rows in items.items:
            shape = shape_of(rows, items.clips)
            if shape.refused is not None or shape.tags or shape.kind is None:
                continue
            key = (shape.kind, bool(links_of(rows.item)))
            by_key.setdefault(key, (rows, items.clips))
    return ItemTemplates(by_key, tuple(str(s.path) for s in sets))


def _bad(why: str) -> ProjectWriteError:
    return ProjectWriteError("bad_input", why, "check the cut list")


@refusing("bad_input", "authoring the cut", "check the cut list", passthrough=(MediaPoolError,))
def author_cut(
    seed: Seed,
    cuts: Sequence[Cut],
    templates: ItemTemplates,
    *,
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> tuple[list[AuthoredItem], list[ItemSpec]]:
    """The items of ``cuts`` on the seed's tracks (see the module docstring)."""
    taken = set(seed.taken)

    def fresh() -> str:
        value = new_id()
        if not isinstance(value, str) or len(value) != 36 or value in taken:
            raise _bad(f"the id source gave {value!r}, not a fresh uuid")
        taken.add(value)
        return value

    planned: list[tuple[str, Cut, str, tuple[str, ...], int, int, int]] = []
    for cut in cuts:
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
    items: list[AuthoredItem] = []
    specs: list[ItemSpec] = []
    for kind, track in (("picture", seed.video_track), ("sound", seed.audio_track)):
        mine = sorted((p for p in planned if p[2] == kind), key=lambda p: p[4])
        for index, (item_id, cut, _, partners, record_in, src_in, duration) in enumerate(mine):
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
