"""Author the rows of one timeline item: a picture or a sound of a media clip, at full speed.

:func:`author_item` takes the media clip the item plays (:class:`ClipRows`, Resolve's own or
authored by :mod:`aar.drp.mediapool`), a template item of the same kind that Resolve saved, and
an :class:`ItemPlacement` (track, position, source range, the items it is linked to), and
returns the ``Sm2TiItem`` row and its ``Items`` join row. The field map (docs/DESIGN.md,
"Timeline items (S25c2)"):

* **derived** from the media clip (:func:`aar.drp.items.media.media_view`): ``Name``,
  ``MediaRef``, ``MediaFilePath``, ``MediaStartTime``, ``MediaTimemapBA``, ``MediaFrameRate``,
  and for a sound ``VirtualAudioTrackBA`` (its channels) and ``MediaTrackIdx``;
* **placed**: ``Start`` (the record frame, counted from frame 0 like the timeline's own start),
  ``Duration``, ``In`` (NULL when the item reads from the file's first frame), the track, the
  ``DbIndex`` and the link list inside ``FieldsBlob``;
* **identity**: a fresh ``Sm2TiItem_id``;
* **fresh**: what a later edit adds (a grade, a Fusion composition, effects, markers, import
  metadata, the edit history inside ``FieldsBlob``) is NULL or left out, as on an item Resolve
  has just made; ``CurrentSelectorIdx`` is 0 (it is derived from nothing);
* **copied** from the template: every other column, each one of :data:`ITEM_COLUMNS`.

It writes nothing. Whatever it cannot justify is a typed :class:`ItemError`; after authoring no
id of the template (its own, its links, its media, its track) is left in any layer of any value.
"""

from __future__ import annotations

import struct
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from aar.drp import blobs
from aar.drp.items.errors import ItemError
from aar.drp.items.media import CHANNELS, MediaView, media_view, timemap
from aar.drp.items.rows import LATER_COLUMNS, ItemRows, check_item, links_of
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool.errors import refusing
from aar.drp.mediapool.templates import ClipRows, Row

__all__ = ["AuthoredItem", "ItemPlacement", "author_item"]

_UUID_TEXT = 36
#: The most frames an item may start at or run for (24 hours at 60 fps, well inside the text).
MAX_FRAMES = 24 * 3600 * 60
_HINT = "use a template item from a project Resolve saved, of the same kind"
#: The channel ids a sound item's ``VirtualAudioTrackBA`` lists, one per channel.
_CHANNEL_BASE = 0x4000


@dataclass(frozen=True)
class ItemPlacement:
    """Where the item goes and what it plays."""

    kind: str  # "picture" or "sound"
    track_id: str
    #: Position in the track's ``Items`` vector (record order).
    index: int
    #: The record frame the item starts at, counted from frame 0 (the timeline's start frame
    #: plus the item's position in the cut).
    start: int
    duration: int
    #: The first source frame, counted from the file's first frame.
    src_in: int
    #: The ids of the items it is linked to, in link-key order (its own sound or picture).
    partners: tuple[str, ...] = ()
    #: A name given to the item; None for the media clip's name (what Resolve gives it).
    name: str | None = None


@dataclass(frozen=True)
class AuthoredItem:
    """The item's rows, ready to insert (nothing is written here)."""

    kind: str
    item_id: str
    item: Row
    link: Row
    notes: tuple[str, ...] = field(default=())

    def rows(self) -> ItemRows:
        return ItemRows(self.item, self.link)


def _refuse(code: str, why: str, hint: str = _HINT) -> ItemError:
    return ItemError(code, why, hint)


def _check_placement(p: object, view: MediaView) -> ItemPlacement:
    def bad(why: str) -> ItemError:
        return _refuse("bad_placement", f"the item placement's {why}", "check the placement")

    if not isinstance(p, ItemPlacement):
        raise bad("type is not ItemPlacement")
    if p.kind not in ("picture", "sound"):
        raise bad(f"kind {p.kind!r} is not picture or sound")
    for name in ("index", "start", "duration", "src_in"):
        value = getattr(p, name)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= MAX_FRAMES:
            raise bad(f"{name} {value!r} is not a whole number from 0 to {MAX_FRAMES}")
    if p.duration < 1:
        raise bad("duration is below one frame")
    if not isinstance(p.track_id, str) or len(p.track_id) != _UUID_TEXT:
        raise bad("track id is not a uuid text")
    if not isinstance(p.partners, tuple) or not all(
        isinstance(x, str) and len(x) == _UUID_TEXT for x in p.partners
    ):
        raise bad("partners are not a tuple of uuid texts")
    if len(set(p.partners)) != len(p.partners) or len(p.partners) > 16:
        raise bad("partners repeat an id or are more than 16")
    if p.name is not None and (not isinstance(p.name, str) or not p.name or len(p.name) > 1024):
        raise bad("name is not text of 1 to 1024 characters")
    if p.kind == "picture":
        if not view.picture or view.frames is None:
            raise bad("kind is picture, but the media clip has no picture")
        if p.src_in + p.duration > view.frames:
            raise bad(
                f"source range {p.src_in}..{p.src_in + p.duration} reads past the media's "
                f"{view.frames} frames"
            )
    elif not view.channels:
        raise bad("kind is sound, but the media clip has no sound")
    return p


def _links_blob(entries: list[str], template: bt.KdEntries) -> bytes:
    """The link list: keys ``0``.. naming the partners, written highest key first (the order of
    every linked item in the corpus); each entry of the template's own entry type."""
    if not template:
        raise _refuse("bad_template", "the template item's link list is empty")
    vtype, null = template[0][1], template[0][2]
    if vtype != blobs.KD_STRING:
        raise _refuse("bad_template", f"the template's link entries are of type 0x{vtype:x}")
    rows = [[str(i), vtype, null, partner] for i, partner in enumerate(entries)]
    return bt.kd_encode(list(reversed(rows)), "item links")


def _fields_blob(template: Row, kind: str, partners: tuple[str, ...]) -> bytes:
    """The template's ``FieldsBlob`` with the link list set to ``partners`` (or none) and the
    later edits' fields (``#1.#1.#3``, ``#1.#1.#21``, ``#1.#9``, Fairlight effects
    ``#1.#1.#2``) left out, framed the way Resolve frames a message of its length."""
    original, payload = bt.message_of(template.get("FieldsBlob"), "item FieldsBlob")
    top = bt.pb_decode(payload, "item FieldsBlob")
    first = bt.pb_get(top, 1, "item FieldsBlob")
    inner = bt.pb_decode(first[2], "item FieldsBlob #1")
    link_template: bt.KdEntries = []
    groups = [f for f in inner if f[0] == 1]
    for group in groups:
        for number, _, value in bt.pb_decode(group[2], "item FieldsBlob #1.#1"):
            if number == 1:
                link_template = bt.kd_decode(value, "item links")
    keep = [f for f in inner if f[0] == 4]
    if len(keep) != 1:
        raise _refuse("bad_template", "the template item's FieldsBlob has no single #1.#4")
    rebuilt: bt.Fields = []
    if partners:
        if not link_template:
            raise _refuse(
                "bad_template",
                "the template item is not linked, so it has no link list to copy the form of",
                "use a linked template item (a picture with its own sound) for a linked item",
            )
        links = _links_blob(list(partners), link_template)
        rebuilt.append([1, 2, bt.pb_encode([[1, 2, links]], "item FieldsBlob #1.#1")])
    rebuilt.extend(keep)
    out = [[1, 2, bt.pb_encode(rebuilt, "item FieldsBlob #1")]]
    tail = 2 if kind == "picture" else 15
    tails = [f for f in top if f[0] == tail]
    if len(tails) != 1 or len(top) != 2:
        raise _refuse("bad_template", f"the template's FieldsBlob is not #1 and #{tail}")
    out.extend(tails)
    return bt.rewrap(original, bt.pb_encode(out, "item FieldsBlob"), "item FieldsBlob")


def _virtual_audio(template: Row, channels: int) -> bytes:
    """``VirtualAudioTrackBA``: ``ChannelsBA`` lists one id per channel after a 2 and the count;
    ``AudioType`` is 1 for one channel and 0 for two (every sound item in the corpus)."""
    if channels not in CHANNELS:
        raise _refuse(
            "unsupported_item",
            f"a sound of {channels} channels; only mono and stereo items were seen",
            "use a mono or stereo file",
        )
    entries = bt.kd_decode(template.get("VirtualAudioTrackBA"), "VirtualAudioTrackBA")
    keys = [e[0] for e in entries]
    if sorted(keys) != ["AudioType", "ChannelsBA"]:
        raise _refuse("bad_template", f"VirtualAudioTrackBA holds {keys}, not the two known keys")
    ids = b"".join(struct.pack(">I", _CHANNEL_BASE + i + 1) for i in range(channels))
    bt.kd_set(entries, "ChannelsBA", struct.pack(">II", 2, channels) + ids, "VirtualAudioTrackBA")
    bt.kd_set(entries, "AudioType", 1 if channels == 1 else 0, "VirtualAudioTrackBA")
    return bt.kd_encode(entries, "VirtualAudioTrackBA")


class _Ids:
    def __init__(self, taken: Collection[str], new_id: Callable[[], str]) -> None:
        self.taken = taken
        self.new_id = new_id

    def __call__(self) -> str:
        try:
            value = self.new_id()
        except Exception as exc:
            raise _refuse(
                "id_collision",
                f"the id source raised {type(exc).__name__}: {str(exc)[:200]}",
                "fix the id source, or leave it out to get uuid4 ids",
            ) from exc
        if not isinstance(value, str) or len(value) != _UUID_TEXT or value in self.taken:
            raise _refuse(
                "id_collision", f"the new item id {value!r} is not a fresh uuid", "author again"
            )
        return value


@refusing(
    "bad_template",
    "authoring a timeline item from its template and media clip",
    _HINT,
    error=ItemError,
)
def author_item(
    clip: ClipRows,
    template: ItemRows,
    template_clips: dict[str, ClipRows],
    placement: ItemPlacement,
    *,
    item_id: str | None = None,
    taken: Collection[str] = frozenset(),
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> AuthoredItem:
    """The rows of one item playing ``clip`` at ``placement``, from ``template`` (an item of
    the same kind; ``template_clips`` are the media clips of the template's project, which its
    shape is checked against). ``item_id`` is the new item's id (made by ``new_id`` if None,
    never one in ``taken``); a linked pair's ids are chosen first so each can name the other."""
    view = media_view(clip)
    check_item(template, template_clips)
    _check_placement(placement, view)
    if template.kind != placement.kind:
        raise _refuse(
            "kind_mismatch",
            f"the template is a {template.kind} item, the item to write a {placement.kind}",
            "use a template item of the same kind",
        )
    if item_id is None:
        item_id = _Ids(taken, new_id)()
    elif not isinstance(item_id, str) or len(item_id) != _UUID_TEXT or item_id in taken:
        raise _refuse(
            "id_collision", f"the item id {item_id!r} is not a fresh uuid", "author again"
        )
    if item_id in placement.partners:
        raise _refuse("bad_placement", "the item is listed as its own partner", "check the links")
    tpl = template.item
    row: Row = dict(tpl)
    for column in LATER_COLUMNS:
        row[column] = None
    row.update(
        Sm2TiItem_id=item_id,
        Name=placement.name if placement.name is not None else view.name,
        Start=str(placement.start),
        Duration=str(placement.duration),
        In=str(placement.src_in) if placement.src_in else None,
        MediaRef=view.media_id,
        MediaStartTime=view.start_seconds,
        MediaFilePath=view.path,
        MediaTimemapBA=timemap(view.last_seconds),
        MediaFrameRate=view.rate_bytes,
        Sm2TiTrack_id=placement.track_id,
        FieldsBlob=_fields_blob(tpl, placement.kind, placement.partners),
        # Not derived from anything (most real items hold 0, others what looks like
        # uninitialised memory): 0, so nothing of a template's reaches the new project.
        CurrentSelectorIdx=0,
    )
    if placement.kind == "sound":
        row["MediaTrackIdx"] = 0
        row["VirtualAudioTrackBA"] = _virtual_audio(tpl, view.channels[0])
    link = {
        "DbOwner": placement.track_id,
        "DbAssociate": item_id,
        "DbPropertyName": "Items",
        "DbIndex": placement.index,
    }
    allowed = set(placement.partners) | {item_id, placement.track_id, view.media_id}
    _no_template_ids(row, template, allowed)
    return AuthoredItem(placement.kind, item_id, row, link)


def _no_template_ids(row: Row, template: ItemRows, allowed: set[str]) -> None:
    """Refuse when an id of the template's own (its id, links, media or track) survives in any
    layer of any value: it would tie the new item to rows of another project."""
    needles = {template.item_id: "the template item's id"}
    for value, label in (
        (template.item.get("MediaRef"), "the template's media"),
        (template.item.get("Sm2TiTrack_id"), "the template's track"),
    ):
        if isinstance(value, str) and value:
            needles[value] = label
    for partner in links_of(template.item):
        needles[partner] = "the template's linked item"
    for text in allowed:
        needles.pop(text, None)
    for column, value in row.items():
        found = bt.contains_text(value, needles)
        if found is not None:
            raise _refuse(
                "bad_template",
                f"{found} is still in the authored item's {column}",
                "use another template item",
            )
