"""Compare two items' rows field by field: the media-pool comparison, applied to item rows.

:func:`flatten_item` lays an item's rows out with :func:`aar.drp.mediapool.fields.flatten_rows`
(``I`` the item row, ``J`` its join row; ids replaced by the role they name: ``<self>``, the
item's media ``<M>``, its track ``<T>``, its partners ``<P0>``...), and :func:`compare_items`
compares two such layouts with :func:`aar.drp.mediapool.fields.compare_flat`: total, every
decoded level, type tags, order, envelopes and undecodable bytes. :data:`ITEM_EXCLUDED` is the
reviewed list of what is not compared, each entry with its reason and the item tags it applies
to; nothing else is left out.
"""

from __future__ import annotations

from collections.abc import Iterable

from aar.drp.items.errors import ItemError
from aar.drp.items.rows import ItemRows, links_of
from aar.drp.mediapool.errors import MediaPoolError, refusing
from aar.drp.mediapool.fields import (
    Difference,
    Exclusion,
    Leaf,
    compare_flat,
    flatten_rows,
)

__all__ = ["ITEM_EXCLUDED", "compare_items", "flatten_item", "item_excluded"]

_ALL = frozenset({"picture", "sound"})

#: Every field the item gate leaves out, and why. An exclusion's ``media`` names the item tags
#: it applies to: the item's kind (``picture``, ``sound``) or a tag of its shape.
ITEM_EXCLUDED: tuple[Exclusion, ...] = (
    Exclusion(
        r"I\.pLmVerTable([>.\[].*)?",
        "project-local",
        "the item's grade version table (ListMgt rows the writer does not author); an item "
        "Resolve has just made has none",
        _ALL,
    ),
    Exclusion(
        r"I\.CompositionTable([>.\[].*)?",
        "project-local",
        "the item's Fusion composition; an item Resolve has just made has none",
        _ALL,
    ),
    Exclusion(
        r"I\.EffectFiltersBA([>.\[].*)?",
        "project-local",
        "effects applied to the item (transform, crop, gain...); a new item has none",
        _ALL,
    ),
    Exclusion(
        r"I\.ImportExportMetadataBA([>.\[].*)?",
        "project-local",
        "metadata left by a timeline import or export; a new item has none",
        _ALL,
    ),
    Exclusion(
        r"I\.(MarkersBA|MediaMetadata)([>.\[].*)?",
        "project-local",
        "NULL or empty depending on what was done to the item (both hold nothing)",
        _ALL,
    ),
    Exclusion(
        r"I\.FieldsBlob>#1\.#1\.#(3|21)(\[\d+\])?(\..*)?|I\.FieldsBlob>#1\.#9",
        "project-local",
        "written by later edits of the item (repeated time stamps, a record frame, a flag); "
        "an item Resolve has just made (the one-clip save) has none",
        _ALL,
    ),
    Exclusion(
        r"I\.FieldsBlob>#1\.#1\.#2(\..*)?",
        "project-local",
        "Fairlight clip effects (FL::ClipFX); a new item has none",
        _ALL,
    ),
    Exclusion(
        r"I\.FieldsBlob>#1\.#1\.?",
        "project-local",
        "the group of an unlinked item that holds only later edits' fields: a new unlinked "
        "item has no group at all",
        frozenset({"unlinked-group"}),
    ),
    Exclusion(
        r"I\.FieldsBlob",
        "project-local",
        "the envelope's codec of an item whose FieldsBlob holds later edits' fields: Resolve "
        "frames a message by its length (plain up to 64 bytes, zstd above), so dropping them "
        "can change the codec; the codec is compared on every other item",
        frozenset({"blob-later"}),
    ),
    Exclusion(
        r"I\.CurrentSelectorIdx",
        "unexplained",
        "unexplained: picture items hold apparently arbitrary values (302 distinct over 4,220 "
        "in one project, most 0) and every sound item 0; the writer writes 0, so nothing of a "
        "template's reaches a new project",
        frozenset({"picture"}),
    ),
    Exclusion(
        r"I\.MediaStartTime",
        "origin",
        "items a timeline import made from FCPXML hold the start as the exact quotient of the "
        "timecode and the rate; items cut from the media pool hold the clip's own StartTime "
        "(the rounded quotient). Only for items whose value is the exact quotient and not the "
        "pool's",
        frozenset({"exact-start"}),
    ),
    Exclusion(
        r"I\.MediaStartTime",
        "last bit",
        "a picture of a file without sound: its clip stores no start of its own, and on 21 of "
        "100 real items Resolve's value is the double next to the rule's (timecode frames over "
        "the rate); only where the two are neighbours",
        frozenset({"start-last-bit"}),
    ),
)


@refusing("bad_template", "looking up an exclusion", "check the item rows", error=ItemError)
def item_excluded(path: str, tags: Iterable[str]) -> Exclusion | None:
    """The exclusion that covers ``path`` (a field) for an item with ``tags``, if any."""
    tags = frozenset(tags)
    return next(
        (e for e in ITEM_EXCLUDED if any(e.applies(path, tag) for tag in tags)),
        None,
    )


def _roles(rows: ItemRows) -> dict[str, str]:
    roles = {rows.item_id: "<self>"}
    media = rows.item.get("MediaRef")
    if isinstance(media, str):
        roles[media] = "<M>"
    track = rows.item.get("Sm2TiTrack_id")
    if isinstance(track, str):
        roles[track] = "<T>"
    try:
        partners = links_of(rows.item)
    except MediaPoolError:  # an item whose links do not decode is compared as bytes
        partners = ()
    for i, partner in enumerate(partners):
        roles.setdefault(partner, f"<P{i}>")
    return roles


@refusing("bad_template", "laying an item's rows out", "check the item rows", error=ItemError)
def flatten_item(rows: ItemRows) -> dict[str, Leaf]:
    """Every field of an item's rows as ``{path: leaf}``."""
    return flatten_rows([("I", rows.item), ("J", rows.link)], _roles(rows))


@refusing("bad_template", "comparing two items' rows", "check the item rows", error=ItemError)
def compare_items(
    expected: ItemRows, got: ItemRows, tags: Iterable[str]
) -> tuple[int, list[Difference], list[tuple[Exclusion, Difference]]]:
    """``(fields compared, differences, differences in excluded fields)`` between Resolve's
    item and an authored one, for an item with ``tags`` (its kind and shape tags)."""
    tags = frozenset(tags)
    return compare_flat(
        flatten_item(expected), flatten_item(got), lambda path: item_excluded(path, tags)
    )
