"""The closed list of operations the library writer performs on a seed's rows.

Every change the writer makes to a copy of the seed is one of :data:`OPERATIONS`, and every
written project's report names the ones it used with their status: ``opened in Resolve`` (a
person opened a project made that way in Resolve 21.1 and it was right) or ``not yet opened``.
A project that uses an operation not yet opened is written only when the caller allows it
explicitly (``allow_unopened``); the staging script (``scripts/stage_library_projects.py``)
builds one small project per such operation for a person to open.
"""

from __future__ import annotations

from dataclasses import dataclass

from aar.drp.libwrite.errors import ProjectWriteError, refusing

__all__ = ["OPENED", "OPERATIONS", "UNOPENED", "Operation", "unopened"]

OPENED = "opened in Resolve"
UNOPENED = "not yet opened"


@dataclass(frozen=True)
class Operation:
    name: str
    status: str
    what: str
    evidence: str

    def as_dict(self) -> dict[str, str]:
        return {"operation": self.name, "status": self.status, "what": self.what}


OPERATIONS: dict[str, Operation] = {
    op.name: op
    for op in (
        Operation(
            "new_project_identity",
            OPENED,
            "a fresh project id in every text column that held the seed's, and the new ProjectName",
            "a copy of a saved project with a fresh id and name, dropped into the library as a "
            "new folder, was listed and opened (2026-11-24)",
        ),
        Operation(
            "rename_timeline",
            UNOPENED,
            "the timeline's Sm2Timeline.Name and the Name of its media-pool clip",
            "never tried in Resolve",
        ),
        Operation(
            "remove_items",
            OPENED,
            "delete the seed timeline's items and their Items join rows (item ids are named "
            "nowhere else)",
            "the cut experiment replaced a timeline's items this way and Resolve opened and "
            "re-saved it with the cut list unchanged",
        ),
        Operation(
            "add_media",
            UNOPENED,
            "insert media-pool clips authored from measured facts (aar.drp.mediapool) into the "
            "bin that holds the timeline",
            "a project with an authored clip has not been reported on by a person yet",
        ),
        Operation(
            "add_linked_items",
            OPENED,
            "insert linked picture and sound items of one source range, abutting from the "
            "timeline's start",
            "the cut experiment's linked pairs, cut from a clip already in the pool, opened",
        ),
        Operation(
            "add_picture_only_items",
            UNOPENED,
            "insert a picture item with no sound (a file without sound, or the picture alone)",
            "never tried in Resolve",
        ),
        Operation(
            "add_sound_only_items",
            UNOPENED,
            "insert a sound item with no picture",
            "never tried in Resolve",
        ),
        Operation(
            "leave_gaps",
            UNOPENED,
            "items that do not start at the timeline's start or do not abut",
            "never tried in Resolve",
        ),
        Operation(
            "split_edit",
            UNOPENED,
            "a linked picture and sound with different source or record ranges",
            "never tried in Resolve",
        ),
        Operation(
            "set_extents",
            OPENED,
            "the sequence's MediaExtents length: the end of the last item in seconds minus the "
            "start, Resolve's own formula",
            "the trim and cut experiments set it and opened",
        ),
    )
}


@refusing("bad_input", "listing the operations not yet opened", "report it")
def unopened(names: list[str]) -> list[str]:
    """The operations of ``names`` no person has opened in Resolve yet; an unknown name is
    a ``bad_input`` refusal."""
    unknown = [n for n in names if n not in OPERATIONS]
    if unknown:
        raise ProjectWriteError(
            "bad_input", f"{unknown[0]!r} is not an operation of the writer", "report it"
        )
    return [n for n in names if OPERATIONS[n].status != OPENED]
