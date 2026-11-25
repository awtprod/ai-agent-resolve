"""Write a new project into a Resolve disk library from a seed, authored media and items.

:func:`make_project` is the one path: in a staging folder inside the library it copies the seed
(settled, through the library reader's snapshot), checks it (:func:`read_seed`), lets the caller
author the media clips and items against it, refuses an operation no person has opened in
Resolve unless allowed (:mod:`aar.drp.libwrite.ops`), changes the copy's rows
(:func:`build_project`), reads it back (:func:`check_written`) and runs the caller's own check,
and only then renames the project folder into the library (:func:`place_project`). See
docs/DESIGN.md, "Library projects (S25c2)".
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from aar.drp.items.author import AuthoredItem
from aar.drp.library import LibraryError, LibraryOptions, take_snapshot
from aar.drp.libwrite.check import check_written
from aar.drp.libwrite.cut import Cut
from aar.drp.libwrite.errors import CODES, ProjectWriteError, by_caller, refusing
from aar.drp.libwrite.ops import OPENED, OPERATIONS, UNOPENED, Operation, unopened
from aar.drp.libwrite.place import Placed, check_project_name, place_project
from aar.drp.libwrite.seed import Seed, read_seed
from aar.drp.libwrite.verify import verify_written
from aar.drp.libwrite.write import Built, ItemSpec, build_project, operations_for
from aar.drp.mediapool.author import AuthoredClip
from aar.drp.model import DrpFile

__all__ = [
    "CODES",
    "OPENED",
    "OPERATIONS",
    "UNOPENED",
    "Authored",
    "Built",
    "ItemSpec",
    "Operation",
    "Placed",
    "ProjectResult",
    "ProjectWriteError",
    "Seed",
    "build_project",
    "check_project_name",
    "check_written",
    "make_project",
    "place_project",
    "read_seed",
    "verify_written",
]

_X = TypeVar("_X")


@dataclass(frozen=True)
class Authored:
    """What the caller authored against the seed: the media clips to add, the items of the
    timeline, and the items as the read-back must find them."""

    clips: Sequence[AuthoredClip]
    items: Sequence[AuthoredItem]
    specs: Sequence[ItemSpec]
    #: Anything the caller wants back (its report data).
    extra: Any = None
    #: The cut the items were authored from (:class:`aar.drp.libwrite.cut.Cut` entries): the
    #: layout :func:`verify_written` checks the written items against, independently of the
    #: code that authored them. Required whenever there are items.
    cuts: Sequence[Cut] = ()


@dataclass
class ProjectResult(Generic[_X]):
    placed: Placed
    built: Built
    seed: Seed
    authored: Authored
    drp: DrpFile
    check: _X

    def operations(self) -> list[dict[str, str]]:
        return [OPERATIONS[name].as_dict() for name in self.built.operations]


def _stage_seed(seed_path: Path, target: Path, options: LibraryOptions) -> None:
    """A settled copy of the seed at ``target`` in the staging folder (the library reader's own
    copy: regular file, no journal beside it, settled, unchanged while copied, bounded)."""
    try:
        take_snapshot(seed_path, target, options)
    except LibraryError as exc:
        raise ProjectWriteError(
            "bad_seed",
            f"the seed {seed_path} cannot be copied: {exc.message} [{exc.code}]",
            exc.hint,
            library_code=exc.code,
        ) from None


@refusing("bad_input", "making the project", "this is a writer bug; report the seed and the plan")
def make_project(
    library: Path,
    seed_path: Path,
    *,
    project_name: str,
    timeline_name: str | None,
    author: Callable[[Seed], Authored],
    allowed_issues: Collection[str],
    check: Callable[[Path, Seed, Authored], _X] = lambda path, seed, authored: None,  # type: ignore[assignment,return-value]
    allow_unopened: bool = False,
    user: str | None = None,
    options: LibraryOptions | None = None,
) -> ProjectResult[_X]:
    """Make project ``project_name`` in ``library`` from the seed (see the module docstring).
    ``timeline_name`` None keeps the seed timeline's name. Raises :class:`ProjectWriteError`
    (or the caller's own refusal from ``author`` or ``check``, unchanged); nothing is placed
    then, and the staging folder is removed (or what is left of it named on the error)."""
    opts = options or LibraryOptions()
    check_project_name(project_name)

    @refusing("bad_input", "building the project", "this is a writer bug; report the seed")
    def build(path: Path) -> ProjectResult[_X]:
        _stage_seed(seed_path, path, opts)
        seed = read_seed(path, options=opts)
        authored = by_caller(author, seed)
        if authored.items and not authored.cuts:
            raise ProjectWriteError(
                "bad_input",
                "the items come without the cut they were authored from, so the written "
                "project cannot be verified against it",
                "pass the Cut entries in Authored.cuts",
            )
        ops = operations_for(seed, timeline_name, authored.items, media=bool(authored.clips))
        pending = unopened(ops)
        if pending and not allow_unopened:
            raise ProjectWriteError(
                "operation_not_opened",
                "the project needs operations no person has opened in Resolve yet: "
                + ", ".join(pending),
                "stage them for a person to open (scripts/stage_library_projects.py), or pass "
                "--allow-unopened-operations to write it anyway (the report lists them)",
                operations=pending,
            )
        built = build_project(
            path,
            seed,
            project_name=project_name,
            timeline_name=timeline_name,
            clips=authored.clips,
            items=authored.items,
        )
        drp = check_written(
            path,
            seed,
            authored.specs,
            timeline_name=timeline_name if timeline_name is not None else seed.timeline_name,
            project_name=project_name,
            allowed_issues=allowed_issues,
        )
        verify_written(path, seed, authored.cuts)
        extra = by_caller(check, path, seed, authored)
        placed_stub = Placed(path, path.parent, "")
        return ProjectResult(placed_stub, built, seed, authored, drp, extra)

    placed, result = place_project(library, project_name, build, user=user)
    result.placed = placed
    result.built = Built(placed.path, result.built.project_id, result.built.operations)
    return result
