"""Transition items: a sound crossfade between two clips of one track.

A transition is one ``Sm2TiItem`` row of type ``Sm2TiTransition`` and one ``Items`` join row on
its track, like a clip (docs/DESIGN.md, "Transitions (S25e)"). It names no media, and its id
occurs nowhere else: its two neighbours (the clip that ends at the cut and the one that starts
there) hold nothing about it, and it holds nothing about them. What it plays is the media
beyond each neighbour's cut: ``before`` frames of the incoming clip ahead of its in point and
``after`` frames of the outgoing clip past its out point.

The field map, from the 920 transitions of two Resolve 21.1 saves (the transition gate,
:mod:`aar.drp.items.transition_gate`, reproduces every one of a supported shape):

* **placed**: ``Start`` (the cut minus the frames before it, counted from frame 0 like a clip's
  start), ``Duration`` (the length), the track and the ``DbIndex`` (its place in record order;
  at an equal start a transition comes before the clip);
* **derived** from the alignment: ``AlignmentType`` and ``Position`` (both 2, centred on the
  cut; 1, starting on it, and 3, ending on it, were seen only beside or in Resolve's ``N|hex``
  form and are not supported); centred, Resolve puts ``(length + 1) // 2`` frames before the
  cut;
* **derived** from the kind: ``PrettyType``, Resolve's own name of the transition
  (``Cross Fade 0DB``, ``Cross Fade +3DB``): the gain law lives there and nowhere else;
* **identity**: a fresh ``Sm2TiItem_id``;
* **later-edit state**, left out as on a transition Resolve has just made: ``FieldsBlob`` #1
  (``#1.#21``, the record start, on 520 of the 854 sound transitions; never on the other 334);
* **copied** from the template: every other column, each one a constant on every sound
  transition (``Name`` "Transition", ``FieldsBlob`` #2: one qCompress'd 44-byte value, the
  same bytes on all 854, :data:`SOUND_FIELD`; no media, no effects); a template holding
  anything else is refused.

Supported: sound transitions of the kinds in :data:`SOUND_KINDS`, centred on the cut between
two clips, of 2 to 16 frames, on a 29.97 fps timeline (the evidence's only rate). Refused with a
typed :class:`ItemError`: picture transitions (the only kind seen, Smooth Cut, holds a value no
rule derives in ``FieldsBlob`` #2 and its parameters in ``EffectFiltersBA``), other kinds, the
other alignments, other lengths and rates, and fades (no transition at a clip's head or tail
was seen: every one sits between two clips). Nothing here writes a file.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from fractions import Fraction

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.items.author import AuthoredItem
from aar.drp.items.errors import ItemError
from aar.drp.items.rows import ITEM_COLUMNS, LATER_COLUMNS, ItemRows, _range_shape, frames_text
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool.errors import MediaPoolError, refusing
from aar.drp.mediapool.templates import Row

__all__ = [
    "ALIGNMENTS",
    "MAX_FRAMES",
    "MIN_FRAMES",
    "RATES",
    "SOUND_FIELD",
    "SOUND_KINDS",
    "SOUND_PAYLOAD",
    "TRANSITION",
    "TransitionPlacement",
    "TransitionShape",
    "author_transition",
    "check_transition",
    "frames_before",
    "transition_shape",
]

TRANSITION = "Sm2TiTransition"
#: The sound transitions the gate reproduces: Resolve's menu name (the plan's
#: ``AudioTransition.type``) -> the ``PrettyType`` it stores.
SOUND_KINDS: dict[str, str] = {
    "Cross Fade 0 dB": "Cross Fade 0DB",
    "Cross Fade +3 dB": "Cross Fade +3DB",
}
_KIND_OF_PRETTY = {pretty: name for name, pretty in SOUND_KINDS.items()}
#: The picture transitions seen, none supported: why.
PICTURE_REFUSAL = (
    "a picture transition: the only kind seen (Smooth Cut) holds a value no rule derives in "
    "FieldsBlob #2 and its parameters in EffectFiltersBA, and no dissolve was seen"
)
#: The supported alignment -> its ``AlignmentType`` (and ``Position``, always equal to it).
ALIGNMENTS: dict[str, int] = {"centre": 2}
#: Every ``AlignmentType`` seen, by name.
SEEN_ALIGNMENTS: dict[int, str] = {1: "start", 2: "centre", 3: "end"}
#: Why the alignments seen but not supported are not: no example could be compared.
UNSUPPORTED_ALIGNMENTS: dict[str, str] = {
    "start": "starts on the cut: its 5 examples sit next to clips stored in Resolve's N|hex "
    "form, so none could be compared",
    "end": "ends on the cut: its 5 examples are stored in Resolve's N|hex form, so none could "
    "be compared",
}
#: The lengths the gate compared (sound: 2, 3, 4 and 16 frames).
MIN_FRAMES = 2
MAX_FRAMES = 16
#: The timeline rates the evidence covers (every real transition sits on a 29.97 timeline).
RATES = frozenset({Fraction(30000, 1001)})
#: ``FieldsBlob`` #2 of every sound transition, decompressed: 44 bytes.
SOUND_PAYLOAD = bytes.fromhex("030001") + bytes(41)
#: ``FieldsBlob`` #2 of every sound transition as stored, byte for byte (854 of 854): Qt's
#: ``qCompress`` of :data:`SOUND_PAYLOAD` (the length, then a zlib stream). A template must
#: hold exactly these bytes: a damaged stream can inflate to the same value (a flipped padding
#: bit), and the writer copies only what Resolve wrote.
SOUND_FIELD = bytes.fromhex("0000002c789c636660642016000000da0005")
#: Columns a transition never holds (it plays no media; nothing later was added).
_EMPTY_COLUMNS = (
    "In",
    "MediaRef",
    "MediaStartTime",
    "MediaFilePath",
    "MediaReelNumber",
    "MediaTimemapBA",
    "MediaFrameRate",
    "MediaTrackIdx",
    "VirtualAudioTrackBA",
    "PreConformMediaExtents",
    "Resolution",
    "Group",
    "LinkedItemSync",
    "OriginalClip",
    "Sm2TiItem_Owner_id",
    "ClipGroup",
    "RenderCacheBA",
    *LATER_COLUMNS,
)
_UUID_TEXT = 36
#: The most frames a cut may sit at (24 hours at 60 fps, as for an item).
MAX_RECORD = 24 * 3600 * 60
_HINT = "use a sound transition from a project Resolve saved as the template"


@refusing(
    "bad_placement", "placing a transition on its cut", "check the placement", error=ItemError
)
def frames_before(alignment: str, frames: int) -> int:
    """How many of a transition's ``frames`` lie before its cut: Resolve centres an odd length
    with the extra frame before the cut; ``start`` has none before it, ``end`` all."""
    if alignment == "centre":
        return (frames + 1) // 2
    if alignment == "start":
        return 0
    if alignment == "end":
        return frames
    raise ItemError("unsupported_item", f"alignment {alignment!r} is not known", _HINT)


@dataclass(frozen=True)
class TransitionShape:
    """Whether a transition row is of a supported shape (``refused`` None) and what it is."""

    kind: str | None  # "sound" or "picture" (by its track), None when not known
    name: str | None  # Resolve's menu name of a supported kind
    alignment: str | None
    frames: int | None
    refused: str | None
    tags: frozenset[str] = field(default_factory=frozenset)


def _bad(why: str) -> ItemError:
    return ItemError("bad_template", why, _HINT)


def _fields_of(item: Row) -> tuple[bytes, bt.Fields]:
    original, payload = bt.message_of(item.get("FieldsBlob"), "transition FieldsBlob")
    return original, bt.pb_decode(payload, "transition FieldsBlob")


def _payload(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise _bad("transition FieldsBlob #2 is not bytes")
    try:
        return blobs.qcompress_decompress(value, max_output=4096)
    except DrpError as exc:
        raise _bad(f"transition FieldsBlob #2 does not decompress ({exc})") from None


@refusing("bad_template", "reading a transition's shape", _HINT, error=ItemError)
def transition_shape(rows: ItemRows, track_type: int | None) -> TransitionShape:
    """The shape of a transition row on a track of ``track_type`` (0 video, 1 audio). Never
    raises on damage: a value that does not decode refuses the transition."""
    if rows.item.get("DbType") != TRANSITION:
        return TransitionShape(None, None, None, None, "not a transition")
    kind = {0: "picture", 1: "sound"}.get(track_type) if track_type is not None else None
    try:
        return _shape(rows.item, kind)
    except MediaPoolError as exc:
        return TransitionShape(kind, None, None, None, f"does not decode: {exc.message}"[:200])


def _shape(item: Row, kind: str | None) -> TransitionShape:
    def no(why: str, **known: object) -> TransitionShape:
        return TransitionShape(
            kind,
            known.get("name"),  # type: ignore[arg-type]
            known.get("alignment"),  # type: ignore[arg-type]
            known.get("frames"),  # type: ignore[arg-type]
            why,
        )

    unknown = [c for c in item if c not in ITEM_COLUMNS]
    if unknown:
        return no(f"has columns this writer does not know: {unknown[:4]}")
    if kind is None:
        return no("its track is neither a video nor an audio track")
    if kind == "picture":
        return no(PICTURE_REFUSAL)
    pretty = item.get("PrettyType")
    name = _KIND_OF_PRETTY.get(pretty) if isinstance(pretty, str) else None
    if name is None:
        return no(f"a sound transition {pretty!r}, not a kind the evidence covers")
    range_problem = _range_shape(item)
    if range_problem is not None:
        return no(range_problem, name=name)
    frames = frames_text(item.get("Duration"))
    assert frames is not None  # _range_shape checked it
    code, position = item.get("AlignmentType"), item.get("Position")
    if not isinstance(code, int) or isinstance(code, bool) or code != position:
        return no(f"AlignmentType {code!r} and Position {position!r} differ", name=name)
    alignment = SEEN_ALIGNMENTS.get(code)
    if alignment is None:
        return no(f"AlignmentType {code} was never seen", name=name)
    if alignment not in ALIGNMENTS:
        return no(UNSUPPORTED_ALIGNMENTS[alignment], name=name, alignment=alignment)
    held = [c for c in _EMPTY_COLUMNS if item.get(c) is not None]
    if held:
        return no(f"holds {held[:3]} (a sound transition holds none)", name=name)
    if not MIN_FRAMES <= frames <= MAX_FRAMES:
        return no(
            f"{frames} frames long, outside the lengths compared ({MIN_FRAMES} to {MAX_FRAMES})",
            name=name,
            alignment=alignment,
            frames=frames,
        )
    original, fields = _fields_of(item)
    envelope = blobs.parse_envelope(original)
    if envelope is None or envelope.version != 2 or envelope.codec != "protobuf":
        return no("FieldsBlob is not framed as every sound transition's (version 2, plain)")
    numbers = [f[0] for f in fields]
    tags: set[str] = set()
    if numbers == [1, 2] or numbers == [2]:
        if numbers[0] == 1:
            if fields[0][1] != 2:
                return no("FieldsBlob #1 is not a message", name=name)
            inner = bt.pb_decode(fields[0][2], "transition FieldsBlob #1")
            if [(n, w) for n, w, _ in inner] != [(21, 0)]:
                return no(f"FieldsBlob #1 holds {[n for n, _, _ in inner][:4]}", name=name)
            tags.add("history")
        tail = fields[-1]
        if tail[1] != 2 or tail[2] != SOUND_FIELD or _payload(tail[2]) != SOUND_PAYLOAD:
            return no("FieldsBlob #2 is not the value every sound transition holds", name=name)
    else:
        return no(f"FieldsBlob holds fields {numbers[:6]}, not #2 (and #1)", name=name)
    return TransitionShape(kind, name, alignment, frames, None, frozenset(tags))


@refusing("bad_template", "checking the template transition", _HINT, error=ItemError)
def check_transition(rows: object) -> ItemRows:
    """``rows`` if it is a sound transition of a supported shape whose join row places it on
    its own track; a ``bad_template`` refusal otherwise."""
    if not isinstance(rows, ItemRows):
        raise _bad("the template is not an ItemRows")
    if not isinstance(rows.item, dict) or not isinstance(rows.link, dict):
        raise _bad("the template's transition or join row is not a row")
    link = rows.link
    if set(link) != {"DbOwner", "DbAssociate", "DbPropertyName", "DbIndex"}:
        raise _bad("the template's join row does not have the join columns")
    if (
        link["DbPropertyName"] != "Items"
        or link["DbAssociate"] != rows.item.get("Sm2TiItem_id")
        or link["DbOwner"] != rows.item.get("Sm2TiTrack_id")
    ):
        raise _bad("the template's join row does not place it on its own track")
    shape = transition_shape(rows, 1)
    if shape.refused is not None:
        raise _bad(f"the template transition is not supported: {shape.refused}")
    return rows


@dataclass(frozen=True)
class TransitionPlacement:
    """Where a sound transition goes: Resolve's ``name`` of its kind, its track and place in
    the track's ``Items`` vector, the ``cut`` (record frame counted from frame 0: the
    timeline's start frame plus the join's position), its length and alignment, and the
    timeline's rate."""

    name: str
    track_id: str
    index: int
    cut: int
    frames: int
    alignment: str = "centre"
    rate: Fraction = Fraction(30000, 1001)

    @property
    def before(self) -> int:
        return frames_before(self.alignment, self.frames)

    @property
    def start(self) -> int:
        return self.cut - self.before


def _check_placement(p: object) -> TransitionPlacement:
    def bad(why: str) -> ItemError:
        return ItemError("bad_placement", f"the transition placement's {why}", "check it")

    def unsupported(why: str) -> ItemError:
        return ItemError(
            "unsupported_item",
            why,
            "use a sound crossfade of a kind, alignment and length the evidence covers, or "
            "add it in Resolve",
        )

    if not isinstance(p, TransitionPlacement):
        raise bad("type is not TransitionPlacement")
    for name in ("index", "cut", "frames"):
        value = getattr(p, name)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= MAX_RECORD:
            raise bad(f"{name} {value!r} is not a whole number from 0 to {MAX_RECORD}")
    if p.frames < 1:
        raise bad("length is below one frame")
    if not isinstance(p.track_id, str) or len(p.track_id) != _UUID_TEXT:
        raise bad("track id is not a uuid text")
    if not isinstance(p.name, str) or p.name not in SOUND_KINDS:
        raise unsupported(
            f"a {p.name!r} transition: only {sorted(SOUND_KINDS)} are supported (the kinds "
            "Resolve's saves showed)"
        )
    if not isinstance(p.alignment, str) or p.alignment not in ALIGNMENTS:
        raise unsupported(
            f"a transition aligned {p.alignment!r}: only {sorted(ALIGNMENTS)} were seen in "
            "whole frames"
        )
    if not MIN_FRAMES <= p.frames <= MAX_FRAMES:
        raise unsupported(
            f"a {p.frames}-frame transition: the lengths compared run from {MIN_FRAMES} to "
            f"{MAX_FRAMES} frames"
        )
    if not isinstance(p.rate, Fraction) or p.rate not in RATES:
        raise unsupported(
            f"a transition on a {p.rate} fps timeline: every transition seen sits on a 29.97 "
            "fps timeline"
        )
    if p.cut < p.before:
        raise bad(f"cut {p.cut} has fewer than the {p.before} frames before it")
    return p


def _fields_blob(template: Row) -> bytes:
    """The template's ``FieldsBlob`` with ``#1`` (later-edit state) left out, framed the way
    Resolve frames a message of its length."""
    original, fields = _fields_of(template)
    tails = [f for f in fields if f[0] == 2]
    if len(tails) != 1 or tails[0][1] != 2 or tails[0][2] != SOUND_FIELD:
        raise _bad("the template's FieldsBlob #2 is not the sound transition's value")
    return bt.rewrap(original, bt.pb_encode([tails[0]], "transition FieldsBlob"), "FieldsBlob")


@refusing(
    "bad_template",
    "authoring a transition from its template",
    _HINT,
    error=ItemError,
)
def author_transition(
    template: ItemRows,
    placement: TransitionPlacement,
    *,
    transition_id: str | None = None,
    taken: Collection[str] = frozenset(),
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> AuthoredItem:
    """The rows of one sound transition at ``placement``, from ``template`` (a sound
    transition Resolve saved, of any supported kind: the kind is ``PrettyType`` alone). It
    checks the placement on its own; that the cut is a join of two clips with the media the
    transition plays is the caller's (:func:`aar.drp.libwrite.cut.author_cut`) and the
    verifier's."""
    check_transition(template)
    _check_placement(placement)
    if transition_id is None:
        try:
            transition_id = new_id()
        except Exception as exc:
            raise ItemError(
                "id_collision",
                f"the id source raised {type(exc).__name__}: {str(exc)[:200]}",
                "fix the id source",
            ) from exc
    if (
        not isinstance(transition_id, str)
        or len(transition_id) != _UUID_TEXT
        or transition_id in taken
    ):
        raise ItemError(
            "id_collision", f"the transition id {transition_id!r} is not a fresh uuid", "again"
        )
    tpl = template.item
    row: Row = {c: tpl.get(c) for c in ITEM_COLUMNS if c in tpl}
    code = ALIGNMENTS[placement.alignment]
    row.update(
        Sm2TiItem_id=transition_id,
        Start=str(placement.start),
        Duration=str(placement.frames),
        Sm2TiTrack_id=placement.track_id,
        AlignmentType=code,
        Position=code,
        PrettyType=SOUND_KINDS[placement.name],
        FieldsBlob=_fields_blob(tpl),
    )
    link = {
        "DbOwner": placement.track_id,
        "DbAssociate": transition_id,
        "DbPropertyName": "Items",
        "DbIndex": placement.index,
    }
    needles = {template.item_id: "the template transition's id"}
    track = tpl.get("Sm2TiTrack_id")
    if isinstance(track, str) and track and track != placement.track_id:
        needles[track] = "the template's track"
    needles.pop(transition_id, None)
    for column, value in row.items():
        found = bt.contains_text(value, needles)
        if found is not None:
            raise _bad(f"{found} is still in the authored transition's {column}")
    return AuthoredItem("transition", transition_id, row, link)
