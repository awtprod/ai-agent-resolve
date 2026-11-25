"""Author the media-pool rows of a file Resolve has never imported, for a library project.

Templates (:func:`read_templates`) are clips of a project Resolve saved; :func:`author` builds a
new clip's rows from a file's measured facts (:class:`MediaFacts`) and a template of the same
kind (:mod:`aar.drp.mediapool.kinds`). Nothing here writes a file. See docs/DESIGN.md,
"Media-pool rows (S25c1)".
"""

from aar.drp.mediapool.author import AuthoredClip, Placement, Template, author
from aar.drp.mediapool.errors import CODES, MediaPoolError
from aar.drp.mediapool.facts import MediaFacts, facts_from_json
from aar.drp.mediapool.kinds import KINDS, KindKey, KindStatus, kind_of_facts, kind_of_rows
from aar.drp.mediapool.templates import ClipRows, TemplateSet, read_templates

__all__ = [
    "CODES",
    "KINDS",
    "AuthoredClip",
    "ClipRows",
    "KindKey",
    "KindStatus",
    "MediaFacts",
    "MediaPoolError",
    "Placement",
    "Template",
    "TemplateSet",
    "author",
    "facts_from_json",
    "kind_of_facts",
    "kind_of_rows",
    "read_templates",
]
