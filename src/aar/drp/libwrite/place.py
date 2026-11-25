"""Place a new project in a Resolve disk library, never over an existing one.

A project is ``<library>/Resolve Projects/Users/<user>/Projects/<project name>/Project.db``.
:func:`place_project` checks the library (:func:`aar.drp.library.library_users_dir`, the
listing's own check), the name (one folder name on the Windows share:
:func:`aar.host.winpath.check_name`, and no character a reader cannot see), and that no project
folder of that name exists in any case (the share folds case). It builds the project in a
staging folder inside the library (``Resolve Projects/.aar-staging/<random>``, on the same file
system as the projects, so the last step is one rename), gives the folder and file the mode
(and, where the process may, the owner and group) of the library's own projects, and renames
the folder into place. The staging folder is removed on every path, failures included; one
that is left (a killed process) is reported by the next placement in that library, never
deleted by it. Nothing is ever written into an existing project's folder.
"""

from __future__ import annotations

import contextlib
import errno
import os
import secrets
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from aar.drp.library import LibraryError, library_users_dir
from aar.drp.libwrite.errors import ProjectWriteError
from aar.host.errors import PathRefused
from aar.host.winpath import check_name

__all__ = ["MAX_NAME", "STAGING", "Placed", "check_project_name", "place_project"]

#: The staging folder's name, inside ``Resolve Projects`` (beside ``Users``).
STAGING = ".aar-staging"
#: The longest project name accepted (a folder name; Windows allows 255 UTF-16 units).
MAX_NAME = 200
#: The most entries looked at in one ``Projects`` folder or in the staging folder.
_MAX_ENTRIES = 20_000
_T = TypeVar("_T")


@dataclass
class Placed:
    """Where the project went, and what the placement noticed."""

    path: Path
    folder: Path
    user: str
    #: Folders an interrupted placement left in the staging folder (safe to delete).
    leftovers: list[str] = field(default_factory=list)
    #: Mode and ownership as set, or why one could not be.
    permissions: list[str] = field(default_factory=list)
    #: Why the staging folder could not be removed after the project was placed.
    cleanup: list[str] = field(default_factory=list)


def _refuse(code: str, message: str, hint: str, **details: object) -> ProjectWriteError:
    return ProjectWriteError(code, message, hint, **details)


def check_project_name(name: object) -> str:
    """``name`` if it is a valid project folder name on the share; ``bad_project_name``
    otherwise."""
    hint = (
        "give a name of visible characters that Windows allows in a folder name (no \\ / : * ? "
        '" < > |, no trailing dot or space, not CON, NUL, COM1...)'
    )
    if not isinstance(name, str):
        raise _refuse("bad_project_name", "the project name is not text", hint)
    try:
        check_name(name)
    except PathRefused as exc:
        raise _refuse(
            "bad_project_name", f"the project name {name!r} is refused: {exc}", hint
        ) from None
    from aar.hub.lessons import hidden_character  # the hub's one check (as the facts use it)

    hidden = hidden_character(name)
    if hidden is not None:
        raise _refuse(
            "bad_project_name",
            f"the project name holds a character a reader cannot see (U+{ord(hidden):04X})",
            hint,
        )
    if len(name) > MAX_NAME or name.startswith("."):
        raise _refuse(
            "bad_project_name",
            f"the project name is longer than {MAX_NAME} characters or starts with a dot",
            hint,
        )
    return name


def _listdir(folder: Path) -> list[str]:
    out = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if len(out) >= _MAX_ENTRIES:
                raise _refuse(
                    "library_unwritable",
                    f"{folder} holds more than {_MAX_ENTRIES} entries",
                    "tidy the library",
                )
            out.append(entry.name)
    return out


def _projects_dir(library: Path, user: str | None) -> tuple[Path, str]:
    try:
        users = library_users_dir(library)
    except LibraryError as exc:
        raise _refuse("not_a_library", exc.message, exc.hint) from None
    if user is not None:
        check_project_name(user)
        projects = users / user / "Projects"
        if not projects.is_dir() or projects.is_symlink():
            raise _refuse(
                "not_a_library",
                f"the library has no Projects folder for user {user!r}",
                "give a user the library has (its folder under Resolve Projects/Users)",
            )
        return projects, user
    found = [
        name
        for name in sorted(_listdir(users))
        if (users / name / "Projects").is_dir() and not (users / name).is_symlink()
    ]
    if len(found) != 1:
        raise _refuse(
            "not_a_library",
            f"the library has {len(found)} users with a Projects folder ({found[:5]})",
            "name the user with --library-user",
        )
    return users / found[0] / "Projects", found[0]


def _reference(projects: Path) -> tuple[os.stat_result, os.stat_result | None]:
    """The stat of a project folder already in the library and of its ``Project.db`` (the
    pattern to follow), or of the ``Projects`` folder itself when it holds none."""
    for name in sorted(_listdir(projects)):
        folder = projects / name
        db = folder / "Project.db"
        try:
            if folder.is_symlink() or not folder.is_dir() or db.is_symlink() or not db.is_file():
                continue
            return os.stat(folder), os.stat(db)
        except OSError:
            continue
    return os.stat(projects), None


def _apply_pattern(folder: Path, db: Path, projects: Path) -> list[str]:
    """Give ``folder`` and ``db`` the mode, owner and group of the library's own projects;
    return what was set or why it was not."""
    if sys.platform == "win32":
        return ["permissions follow the share's (Windows)"]
    ref_dir, ref_db = _reference(projects)
    dir_mode = stat.S_IMODE(ref_dir.st_mode)
    file_mode = stat.S_IMODE(ref_db.st_mode) if ref_db is not None else dir_mode & 0o666
    notes = []
    os.chmod(db, file_mode)
    os.chmod(folder, dir_mode)
    notes.append(f"mode {dir_mode:o} on the folder and {file_mode:o} on Project.db")
    for path in (db, folder):
        mine = os.stat(path)
        if (mine.st_uid, mine.st_gid) == (ref_dir.st_uid, ref_dir.st_gid):
            continue
        try:
            os.chown(path, ref_dir.st_uid, ref_dir.st_gid)
        except PermissionError:
            try:
                os.chown(path, -1, ref_dir.st_gid)
                notes.append(
                    f"{path.name}: owner left as uid {mine.st_uid} (the library's is "
                    f"{ref_dir.st_uid}; only root may change it), group set"
                )
            except PermissionError:
                notes.append(
                    f"{path.name}: owner and group left as {mine.st_uid}:{mine.st_gid} (the "
                    f"library's are {ref_dir.st_uid}:{ref_dir.st_gid}; not permitted)"
                )
    return notes


def _exists(projects: Path, name: str) -> str | None:
    folded = name.casefold()
    return next((n for n in _listdir(projects) if n.casefold() == folded), None)


def place_project(
    library: Path,
    name: str,
    build: Callable[[Path], _T],
    *,
    user: str | None = None,
) -> tuple[Placed, _T]:
    """Build a project with ``build(path)`` (it writes ``path``, a ``Project.db`` in the staging
    folder, and runs its checks) and place it as project ``name``; see the module docstring.
    Raises :class:`ProjectWriteError`: ``not_a_library``, ``bad_project_name``,
    ``project_exists``, ``library_unwritable``, or whatever ``build`` raises."""
    check_project_name(name)
    projects, user_name = _projects_dir(library, user)
    clash = _exists(projects, name)
    if clash is not None:
        raise _refuse(
            "project_exists",
            f"the library already has a project folder {clash!r}",
            "choose another project name (aar never writes into an existing project)",
            folder=clash,
        )
    root = projects.parent.parent.parent / STAGING
    try:
        root.mkdir(exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise _refuse(
                "library_unwritable", f"{root} is not a folder", "remove it from the library"
            )
        leftovers = sorted(_listdir(root))
        work = root / f"build-{secrets.token_hex(8)}"
        work.mkdir()
    except ProjectWriteError:
        raise
    except OSError as exc:
        raise _refuse(
            "library_unwritable",
            f"cannot make a staging folder in {root.parent}: {exc}",
            "check that the library is writable and has room",
        ) from None
    placed = Placed(projects / name / "Project.db", projects / name, user_name, leftovers)
    try:
        result = _build_and_rename(work, projects, name, build, placed)
    except BaseException as exc:
        problem = _remove(work, root)
        if problem is not None:
            exc.add_note(problem)
        if isinstance(exc, OSError):
            raise _refuse(
                "library_unwritable",
                f"the project could not be built or placed in the library: {exc}",
                "check that the library is still there, writable and has room",
                **({"cleanup": problem} if problem else {}),
            ) from None
        raise
    problem = _remove(work, root)
    if problem is not None:
        placed.cleanup.append(problem)
    return placed, result


def _build_and_rename(
    work: Path, projects: Path, name: str, build: Callable[[Path], _T], placed: Placed
) -> _T:
    if os.stat(work).st_dev != os.stat(projects).st_dev:
        raise _refuse(
            "library_unwritable",
            "the staging folder and the projects are on different file systems",
            "keep the library's Resolve Projects folder on one file system",
        )
    folder = work / "project"
    folder.mkdir()
    result = build(folder / "Project.db")
    placed.permissions = _apply_pattern(folder, folder / "Project.db", projects)
    clash = _exists(projects, name)
    if clash is not None:
        raise _refuse(
            "project_exists",
            f"a project folder {clash!r} appeared while the project was built",
            "choose another project name",
            folder=clash,
        )
    try:
        os.rename(folder, projects / name)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise _refuse(
                "project_exists",
                f"a project folder {name!r} appeared while the project was placed",
                "choose another project name",
            ) from None
        raise
    return result


def _remove(work: Path, root: Path) -> str | None:
    """Remove the staging folder (on every path). Returns why it could not be, or None."""
    try:
        if work.exists():
            shutil.rmtree(work)
    except OSError as exc:
        return f"the staging folder {work} could not be removed ({exc}); delete it by hand"
    # Not empty: another placement is building there, or an interrupted one left a folder
    # (each placement reports those as leftovers). Neither is this placement's error.
    with contextlib.suppress(OSError):
        root.rmdir()
    return None
