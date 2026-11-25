"""The refusal the timeline-item writer raises: a media-pool refusal with the item codes.

:class:`ItemError` is a :class:`aar.drp.mediapool.errors.MediaPoolError` with its own code set,
so the one typed boundary (:func:`aar.drp.mediapool.errors.refusing`) wraps both writers' entry
points, and a caller that catches media-pool refusals catches these too.
"""

from __future__ import annotations

from typing import ClassVar

from aar.drp.mediapool.errors import MediaPoolError

__all__ = ["CODES", "ItemError"]

#: Every refusal code of the item writer, with what it means.
CODES: dict[str, str] = {
    "bad_template": "the template item's rows are incomplete, inconsistent or do not decode",
    "no_template": "no template item of the needed kind was given",
    "kind_mismatch": "the template item is of another kind than the item to write",
    "unsupported_item": "the item is of a shape the writer has no evidence for",
    "bad_media": "the media clip's rows do not give what an item needs",
    "bad_placement": "the item's track, position, range or links are out of range",
    "id_collision": "a new id is already used in the project, or twice",
}


class ItemError(MediaPoolError):
    """A refusal: ``code`` (one of :data:`CODES`), ``message`` and a ``hint``."""

    CODES: ClassVar[dict[str, str]] = CODES
