"""Author the timeline items of a library project: a picture or a sound of a media clip.

:func:`author_item` builds one item's rows from the media clip it plays, a template item Resolve
saved and a placement (track, position, source range, links); :mod:`aar.drp.items.gate`
reproduces Resolve's own items leave-one-out. Nothing here writes a file. See docs/DESIGN.md,
"Timeline items (S25c2)".
"""

from aar.drp.items.author import AuthoredItem, ItemPlacement, author_item
from aar.drp.items.errors import CODES, ItemError
from aar.drp.items.media import MediaView, media_view
from aar.drp.items.rows import ItemRows, ItemSet, Shape, check_item, links_of, read_items, shape_of

__all__ = [
    "CODES",
    "AuthoredItem",
    "ItemError",
    "ItemPlacement",
    "ItemRows",
    "ItemSet",
    "MediaView",
    "Shape",
    "author_item",
    "check_item",
    "links_of",
    "media_view",
    "read_items",
    "shape_of",
]
