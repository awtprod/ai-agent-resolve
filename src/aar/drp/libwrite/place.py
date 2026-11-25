"""Place a new project in a Resolve disk library, never over an existing one.

A project is ``<library>/Resolve Projects/Users/<user>/Projects/<project name>/Project.db``.
:func:`place_project` checks the library (:func:`aar.drp.library.library_users_dir`, the
listing's own check), the name (one folder name on the Windows share:
:func:`aar.host.winpath.check_name`, no character a reader cannot see, no leading space, not
``Project.db``, in NFC, short enough once encoded), and that no folder of that name exists in
any case or Unicode form (the share folds case). No folder below the library the caller named
may be a symbolic link or a junction (``Resolve Projects``, ``Users``, the user, ``Projects``,
the staging folders), and the resolved destination and staging folder must lie inside the
resolved library. It builds the project in a staging folder inside the library
(``Resolve Projects/.aar-staging/build-<random>``, on the same file system as the projects, so
the last step is one rename), gives the folder and file the mode (and, where the process may,
the owner and group) of the library's own projects, and renames the folder into place without
replacing anything (``renameat2(RENAME_NOREPLACE)`` where the system has it, Windows' own
rename, else a rename right after a fresh check).

The placement acts only on what it created. A staging folder that already exists is used only
when this user owns it and it is writable no more widely than the ``Projects`` folder (one this
placement makes is private, as are its build folders). The build folder, the project folder and
``Project.db`` are recorded (device and inode) when made; modes and owners are changed only
through descriptors opened without following a link (``O_NOFOLLOW``, ``O_DIRECTORY``) and
checked with ``fstat`` to be that very folder or file (a regular file with one name, this
user's), the rename's source is named relative to the build folder's checked descriptor, the
build folder is removed only when it is still the one made, and the placed file is read
(:func:`read_placed`) only when it is the one written: anything else is ``library_unwritable``
naming it. Windows has no ``O_NOFOLLOW`` or ``dir_fd``: there the same identities are checked
on the paths (never a link or junction), and modes follow the share's.

Two placements of names that fold to the same name are serialised by a lock file in the staging
folder (``lock-<hash of the folded name>``, created exclusively), held from the existence check
through the rename and removed on every path; a second placement of such a name is refused
(``project_exists``, naming the lock and any build folder a killed export left) while the first
holds it. The lock holds a random token of the placement that made it, so the placement knows
its own lock even when an interrupt lands between creating the file and recording that it did.
From the moment the staging folder, the lock or the build folder may exist, every path
(failures, ``OSError`` such as a full disk, and interrupts) either removes what this placement
made or names it in the refusal (its text and its ``residue`` detail), in the interrupt's note,
or, once placed, in :attr:`Placed.cleanup`. A folder or lock a killed process left is reported
by the next placement in that library, never deleted by it. Nothing is ever written into an
existing project's folder.
"""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import shutil
import stat
import sys
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from aar.drp.library import LibraryError, library_users_dir
from aar.drp.libwrite.errors import ProjectWriteError, by_caller, refusing, with_residue
from aar.host.errors import PathRefused
from aar.host.winpath import check_name

__all__ = [
    "MAX_NAME",
    "MAX_NAME_BYTES",
    "STAGING",
    "FileId",
    "Placed",
    "check_created",
    "check_project_name",
    "created_file",
    "fold_name",
    "place_project",
    "read_placed",
]

#: The staging folder's name, inside ``Resolve Projects`` (beside ``Users``).
STAGING = ".aar-staging"
#: The longest project name accepted, in characters.
MAX_NAME = 200
#: The longest project name accepted once encoded: a folder name on Linux is at most 255 bytes
#: of UTF-8, on Windows at most 255 UTF-16 code units.
MAX_NAME_BYTES = 255
#: The most entries looked at in one ``Projects`` folder or in the staging folder.
_MAX_ENTRIES = 20_000
_LOCK = "lock-"
#: How many times a placement makes the staging folder again when another export's cleanup
#: removed it (empty) between this placement's check and its lock.
_ROOT_TRIES = 8
_T = TypeVar("_T")
#: A file's or folder's identity: ``(st_dev, st_ino)``.
FileId = tuple[int, int]


@dataclass
class Placed:
    """Where the project went, and what the placement noticed."""

    path: Path
    folder: Path
    user: str
    #: Folders and locks an interrupted placement left in the staging folder (safe to delete
    #: when no other export is running).
    leftovers: list[str] = field(default_factory=list)
    #: Mode and ownership as set, or why one could not be.
    permissions: list[str] = field(default_factory=list)
    #: What could not be removed after the project was placed (delete it by hand), and other
    #: notes of the placement.
    cleanup: list[str] = field(default_factory=list)
    #: The identity of the ``Project.db`` this placement wrote (:func:`read_placed`).
    file_id: FileId | None = None


@dataclass
class _Made:
    """The identities of what this placement made, recorded when each was made."""

    work: FileId | None = None
    folder: FileId | None = None
    #: Whether this placement made the staging folder or accepted it (:func:`_check_root`):
    #: only then may its cleanup remove it when empty.
    root: bool = False


def _refuse(code: str, message: str, hint: str, **details: object) -> ProjectWriteError:
    return ProjectWriteError(code, message, hint, **details)


@refusing("bad_project_name", "folding a folder name", "give the name as text")
def fold_name(name: str) -> str:
    """The form two folder names share when the share may treat them as one: Unicode's caseless
    match (``casefold`` of the decomposed form) joined with Windows' upper-case comparison
    (``upper``: the dotless and dotted i, which ``casefold`` keeps apart), then composed (NFC).
    It may fold more names together than a given share does, never fewer."""
    folded = unicodedata.normalize("NFD", name).casefold().upper().casefold()
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", folded))


_NAME_HINT = (
    "give a name of visible characters that Windows allows in a folder name (no \\ / : * ? "
    '" < > |, no leading space, no trailing dot or space, not CON, NUL, COM1..., not '
    "Project.db), at most 200 characters"
)


@refusing("bad_project_name", "checking the project name", _NAME_HINT)
def check_project_name(name: object) -> str:
    """``name`` if it is a valid project folder name on the share; ``bad_project_name``
    otherwise."""
    hint = _NAME_HINT
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
    if name.startswith(" "):
        raise _refuse("bad_project_name", "the project name starts with a space", hint)
    if fold_name(name) == fold_name("Project.db"):
        raise _refuse(
            "bad_project_name", "the project name is the name of a project's own file", hint
        )
    nfc = unicodedata.normalize("NFC", name)
    if nfc != name:
        raise _refuse(
            "bad_project_name",
            f"the project name is not in Unicode's composed form (NFC): {nfc!r} is",
            "give the name in NFC (as typed on Windows), so no other form of it is in the library",
        )
    utf8 = len(name.encode("utf-8"))
    utf16 = len(name.encode("utf-16-le")) // 2
    if utf8 > MAX_NAME_BYTES or utf16 > MAX_NAME_BYTES:
        raise _refuse(
            "bad_project_name",
            f"the project name is {utf8} bytes of UTF-8 and {utf16} UTF-16 units; a folder "
            f"name may have at most {MAX_NAME_BYTES} of each",
            "give a shorter name",
        )
    return name


def _is_link(path: Path) -> bool:
    """Whether ``path`` itself is a symbolic link, or on Windows any reparse point (a
    junction, a mount point); False when it does not exist."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    attributes = getattr(st, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _real_folder(path: Path, what: str, code: str = "not_a_library") -> Path:
    """``path`` if it is a folder and not a link (see :func:`_is_link`); a refusal otherwise."""
    if _is_link(path):
        raise _refuse(
            code,
            f"{what} {path} is a symbolic link or junction",
            "aar places projects only in the library's own folders: replace the link with the "
            "folder it points to, or name that folder as the library",
        )
    if not path.is_dir():
        raise _refuse(code, f"{what} {path} is not a folder", "check the library")
    return path


def _inside(path: Path, library: Path, what: str) -> None:
    """Refuse when ``path``, resolved, does not lie inside the resolved ``library``."""
    real = os.path.realpath(path)
    top = os.path.realpath(library)
    try:
        inside = os.path.commonpath([real, top]) == top
    except ValueError:  # another drive on Windows
        inside = False
    if not inside:
        raise _refuse(
            "not_a_library",
            f"{what} resolves to {real}, outside the library {top}",
            "aar places projects only inside the library named",
        )


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


def _projects_dir(library: Path, user: str | None) -> tuple[Path, Path, str]:
    """``(Resolve Projects folder, Projects folder, user)``; every folder below ``library``
    checked to be a folder and not a link."""
    try:
        users = library_users_dir(library)
    except LibraryError as exc:
        raise _refuse("not_a_library", exc.message, exc.hint) from None
    if users.parent != library:
        _real_folder(users.parent, "the library's Resolve Projects folder")
    _real_folder(users, "the library's Users folder")
    if user is not None:
        check_project_name(user)
        folder = users / user
        projects = folder / "Projects"
        if not (projects.is_dir() or _is_link(projects) or _is_link(folder)):
            raise _refuse(
                "not_a_library",
                f"the library has no Projects folder for user {user!r}",
                "give a user the library has (its folder under Resolve Projects/Users)",
            )
        _real_folder(folder, f"the folder of user {user!r}")
        _real_folder(projects, f"the Projects folder of user {user!r}")
        return users.parent, projects, user
    found = [
        name
        for name in sorted(_listdir(users))
        if (users / name / "Projects").is_dir()
        and not _is_link(users / name)
        and not _is_link(users / name / "Projects")
    ]
    linked = [
        name
        for name in sorted(_listdir(users))
        if name not in found
        and (users / name / "Projects").is_dir()
        and (_is_link(users / name) or _is_link(users / name / "Projects"))
    ]
    if linked and not found:
        raise _refuse(
            "not_a_library",
            f"the library's user folder or Projects folder {linked[0]!r} is a symbolic link "
            "or junction",
            "aar places projects only in the library's own folders: replace the link with the "
            "folder it points to",
        )
    if len(found) != 1:
        raise _refuse(
            "not_a_library",
            f"the library has {len(found)} users with a Projects folder ({found[:5]})",
            "name the user with --library-user",
        )
    return users.parent, users / found[0] / "Projects", found[0]


def _windows() -> bool:
    return sys.platform == "win32"


#: The errors of an ``open`` without following links that say the name is not what was made:
#: a link (``ELOOP``; ``EMLINK`` on some BSDs), not a folder, or gone.
_NOT_OURS = frozenset({errno.ELOOP, errno.EMLINK, errno.ENOTDIR, errno.ENOENT})


def _replaced(what: str, path: Path, why: str) -> ProjectWriteError:
    return _refuse(
        "library_unwritable",
        f"{what} {path} is not the one this export made ({why}); something changed it while "
        "the export ran, and nothing more was done with it",
        "check who else can write to the library's staging folder (Resolve Projects/"
        f"{STAGING}) and export again",
        path=str(path),
    )


def _checked(
    st: os.stat_result, what: str, path: Path, made: FileId | None, *, folder: bool
) -> FileId:
    """``st``'s identity when it is a folder (``folder``) or a regular file, and the object
    recorded as ``made``; when nothing is recorded yet (first sight, right after this
    placement made it), a file must have one name and, on POSIX, belong to this user (later
    the placement may have given it the library's owner: the identity is the proof then).
    :func:`_replaced` otherwise."""
    if folder and not stat.S_ISDIR(st.st_mode):
        raise _replaced(what, path, "not a folder")
    if not folder and not stat.S_ISREG(st.st_mode):
        raise _replaced(what, path, "not a regular file")
    ident = (st.st_dev, st.st_ino)
    if made is not None:
        if ident != made:
            raise _replaced(what, path, "another file or folder is in its place")
        return ident
    if not folder and st.st_nlink != 1:
        raise _replaced(what, path, f"a file with {st.st_nlink} names (a hard link)")
    if not _windows() and st.st_uid != os.geteuid():
        raise _replaced(what, path, f"it belongs to uid {st.st_uid}")
    return ident


def _open_own(
    name: Path | str,
    what: str,
    path: Path,
    made: FileId | None,
    *,
    folder: bool,
    dir_fd: int | None = None,
) -> tuple[int, FileId]:
    """POSIX only: a descriptor of ``name`` (relative to ``dir_fd`` when given) opened without
    following a link, and its identity, checked by :func:`_checked`; the caller closes it."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if folder:
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(name, flags) if dir_fd is None else os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _NOT_OURS:
            why = f"{exc.strerror or exc} when opened without following a link"
            raise _replaced(what, path, why) from None
        raise
    try:
        return fd, _checked(os.fstat(fd), what, path, made, folder=folder)
    except BaseException:
        os.close(fd)
        raise


def _identify(path: Path, what: str, made: FileId | None, *, folder: bool) -> FileId:
    """The identity of ``path``, checked by :func:`_checked`: on POSIX through a descriptor
    opened without following a link; on Windows (no ``O_NOFOLLOW``) on the path itself, which
    must not be a link or junction."""
    if _windows():
        if _is_link(path):
            raise _replaced(what, path, "a symbolic link or junction")
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            raise _replaced(what, path, "it is gone") from None
        return _checked(st, what, path, made, folder=folder)
    fd, ident = _open_own(path, what, path, made, folder=folder)
    os.close(fd)
    return ident


_STAGED = "the staged Project.db"


@refusing("library_unwritable", "recording the staged project", "check the library")
def created_file(path: Path) -> FileId:
    """The identity of the ``Project.db`` a build has just created at ``path`` (a regular file
    with one name, this user's, not a link): what :func:`check_created` and the placement
    compare it with later."""
    return _identify(Path(path), _STAGED, None, folder=False)


@refusing("library_unwritable", "checking the staged project", "check the library")
def check_created(path: Path, made: FileId, step: str) -> None:
    """Refuse (``library_unwritable``) when ``path`` is no longer the file :func:`created_file`
    recorded as ``made``; ``step`` says when."""
    _identify(Path(path), f"{_STAGED} ({step})", made, folder=False)


@refusing("library_unwritable", "reading the placed project", "check the library")
def read_placed(placed: Placed) -> bytes:
    """The bytes of the placed ``Project.db``, read only when it is the file the placement
    wrote (:attr:`Placed.file_id`; on POSIX through a descriptor opened without following a
    link); ``library_unwritable`` otherwise (the project folder is placed, but its file is not
    what was written)."""
    what = f"the placed Project.db (the project folder is placed at {placed.folder})"
    if _windows():
        _identify(placed.path, what, placed.file_id, folder=False)
        data = placed.path.read_bytes()
        _identify(placed.path, what, placed.file_id, folder=False)
        return data
    fd, _ = _open_own(placed.path, what, placed.path, placed.file_id, folder=False)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def _check_root(root: Path, library: Path, projects: Path) -> None:
    """Refuse a staging folder that is a link, not a folder or not inside the library, that
    this user does not own, or that others can write to more widely than the library's
    ``Projects`` folder allows (the last two on POSIX; on Windows the share's ACLs decide). One
    ``lstat``: a folder that is gone raises ``FileNotFoundError``."""
    st = os.lstat(root)
    if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
    ):
        raise _refuse(
            "library_unwritable",
            f"the staging folder {root} is a symbolic link or junction",
            "aar places projects only in the library's own folders: delete the link (aar makes "
            "its own staging folder)",
        )
    if not stat.S_ISDIR(st.st_mode):
        raise _refuse(
            "library_unwritable", f"the staging folder {root} is not a folder", "check the library"
        )
    _inside(root, library, "the staging folder")
    if _windows():
        return
    me = os.geteuid()
    hint = (
        "aar uses only a staging folder of its own: when no export is running, delete it (the "
        "next export makes a private one)"
    )
    if st.st_uid != me:
        raise _refuse(
            "library_unwritable",
            f"the staging folder {root} belongs to uid {st.st_uid}, not to this user (uid {me})",
            hint + ", or export as its owner",
            folder=str(root),
        )
    mode = stat.S_IMODE(st.st_mode)
    allowed = stat.S_IMODE(os.stat(projects).st_mode)
    wider = mode & 0o022 & ~allowed
    if wider:
        who = " and ".join(w for bit, w in ((0o020, "its group"), (0o002, "anyone")) if wider & bit)
        raise _refuse(
            "library_unwritable",
            f"the staging folder {root} (mode {mode:o}) can be written by {who}, more widely "
            f"than the library's Projects folder allows (mode {allowed:o})",
            hint + ", or make it no more open than the Projects folder",
            folder=str(root),
        )


def _vanished(exc: OSError) -> bool:
    """Whether ``exc`` says the staging folder was gone (another export's cleanup removed it
    while it was empty; a third may have made it again since, so it is not looked at again).
    On Windows a folder being removed refuses access until it is gone, so ``PermissionError``
    counts there too (the tries are bounded: a real refusal is reported after the last)."""
    return isinstance(exc, FileNotFoundError) or (_windows() and isinstance(exc, PermissionError))


def _reference(projects: Path) -> tuple[os.stat_result, os.stat_result | None]:
    """The stat of a project folder already in the library and of its ``Project.db`` (the
    pattern to follow), or of the ``Projects`` folder itself when it holds none."""
    for name in sorted(_listdir(projects)):
        folder = projects / name
        db = folder / "Project.db"
        try:
            if _is_link(folder) or not folder.is_dir() or _is_link(db) or not db.is_file():
                continue
            return os.stat(folder), os.stat(db)
        except OSError:
            continue
    return os.stat(projects), None


def _apply_pattern(folder_fd: int, db_fd: int, projects: Path) -> list[str]:
    """POSIX: give the folder and the file (descriptors checked by :func:`_open_own`) the mode,
    owner and group of the library's own projects; return what was set or why it was not."""
    ref_dir, ref_db = _reference(projects)
    dir_mode = stat.S_IMODE(ref_dir.st_mode)
    file_mode = stat.S_IMODE(ref_db.st_mode) if ref_db is not None else dir_mode & 0o666
    notes = []
    os.fchmod(db_fd, file_mode)
    os.fchmod(folder_fd, dir_mode)
    notes.append(f"mode {dir_mode:o} on the folder and {file_mode:o} on Project.db")
    for fd, label in ((db_fd, "Project.db"), (folder_fd, "project")):
        mine = os.fstat(fd)
        if (mine.st_uid, mine.st_gid) == (ref_dir.st_uid, ref_dir.st_gid):
            continue
        try:
            os.fchown(fd, ref_dir.st_uid, ref_dir.st_gid)
        except PermissionError:
            try:
                os.fchown(fd, -1, ref_dir.st_gid)
                notes.append(
                    f"{label}: owner left as uid {mine.st_uid} (the library's is "
                    f"{ref_dir.st_uid}; only root may change it), group set"
                )
            except PermissionError:
                notes.append(
                    f"{label}: owner and group left as {mine.st_uid}:{mine.st_gid} (the "
                    f"library's are {ref_dir.st_uid}:{ref_dir.st_gid}; not permitted)"
                )
    return notes


def _settle(
    work: Path, folder: Path, made: _Made, expected: FileId | None, projects: Path
) -> tuple[list[str], FileId]:
    """Check that the build folder, the project folder and its ``Project.db`` are the ones this
    placement made (``expected``: the file's identity as the build recorded it) and give them
    the library's modes; return the notes and the file's identity."""
    db = folder / "Project.db"
    if _windows():
        _identify(work, "the build folder", made.work, folder=True)
        _identify(folder, "the built project folder", made.folder, folder=True)
        return ["permissions follow the share's (Windows)"], _identify(
            db, "the built Project.db", expected, folder=False
        )
    work_fd, _ = _open_own(work, "the build folder", work, made.work, folder=True)
    try:
        folder_fd, _ = _open_own(
            folder.name,
            "the built project folder",
            folder,
            made.folder,
            folder=True,
            dir_fd=work_fd,
        )
        try:
            db_fd, ident = _open_own(
                db.name, "the built Project.db", db, expected, folder=False, dir_fd=folder_fd
            )
            try:
                return _apply_pattern(folder_fd, db_fd, projects), ident
            finally:
                os.close(db_fd)
        finally:
            os.close(folder_fd)
    finally:
        os.close(work_fd)


def _exists(projects: Path, name: str) -> str | None:
    folded = fold_name(name)
    return next((n for n in _listdir(projects) if fold_name(n) == folded), None)


def _lock_name(name: str) -> str:
    return _LOCK + hashlib.sha256(fold_name(name).encode("utf-8")).hexdigest()[:32]


@dataclass
class _Lock:
    """This placement's claim on the lock of its folded name: ``path``, the ``token`` the lock
    file holds once written, whether this placement created the file (``created``) and whether
    another placement held it (``contended``: the file is not this placement's)."""

    path: Path
    token: bytes
    tried: bool = False
    created: bool = False
    contended: bool = False


def _take_lock(root: Path, claim: _Lock, name: str) -> None:
    """Create ``claim.path`` exclusively and write the claim's token to it; a
    ``project_exists`` refusal when another placement holds it (naming any build folder a
    killed export left beside it). Whatever fails after the file exists propagates: the
    caller's cleanup recognises the file as this placement's (:func:`_lock_left`)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    claim.tried = True
    try:
        fd = os.open(claim.path, flags, 0o600)
        claim.created = True
    except FileExistsError:
        claim.contended = True
        try:
            builds = sorted(n for n in _listdir(root) if n.startswith("build-"))
        except (OSError, ProjectWriteError):
            builds = []
        others = "".join(f"; {root / b}" for b in builds)
        raise _refuse(
            "project_exists",
            f"another export is placing a project named {name!r} (in some case) in this "
            f"library: it holds the lock {claim.path}"
            + (f" (the staging folder also holds {', '.join(builds)})" if builds else ""),
            "wait for it and choose another name; if no export is running, a killed export "
            "left them: delete the lock"
            + (" and the build folders" if builds else "")
            + f" ({claim.path}{others}); each is safe to delete when no export is running",
            lock=str(claim.path),
            leftovers=[str(root / b) for b in builds],
        ) from None
    try:
        written = os.write(fd, claim.token)
        if written != len(claim.token):
            raise OSError(errno.EIO, f"the lock took {written} of {len(claim.token)} bytes")
    finally:
        os.close(fd)


def _lock_left(claim: _Lock) -> str | None:
    """Whether this placement's lock is (or may be) still there: ``"mine"`` when this
    placement created it and it holds its token, a beginning of it or nothing (a write cut
    short); ``"maybe"`` when the file is empty and the open was neither refused nor recorded
    (an interrupt between creating the file and recording it, or another placement's file in
    the same instant: named, never deleted); None otherwise."""
    if claim.contended or not claim.tried:
        return None
    try:
        if _is_link(claim.path) or not claim.path.is_file():
            return None
        with open(claim.path, "rb") as handle:
            data = handle.read(len(claim.token) + 1)
    except FileNotFoundError:
        return None
    except OSError:
        return "maybe" if claim.created else None
    if claim.created and claim.token.startswith(data):
        return "mine"
    return "maybe" if not data else None


_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


#: The errors of ``renameat2`` that say the call is not available here (an old kernel or C
#: library, a file system without the flag, a syscall filter that forbids it), not that the
#: rename itself is refused: the placement falls back to a checked rename, which raises the
#: real error if there is one.
_UNAVAILABLE = frozenset(
    {errno.ENOSYS, errno.EINVAL, errno.EPERM, errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", 95)}
)
#: The C library's ``renameat2`` once looked up: unset, None (none here) or the call.
_NOREPLACE: list[Callable[..., tuple[int, int]] | None] = []


def _renameat2() -> Callable[..., tuple[int, int]] | None:
    """Linux's ``renameat2`` from the C library already loaded in the process (no search and
    no subprocess: ``CDLL(None)``), looked up at first use; None where there is none (another
    system, or a C library without the symbol). The call ``(old, new, old_dir=AT_FDCWD)``
    returns ``(result, errno)``; ``old`` is relative to the descriptor ``old_dir``."""
    if _NOREPLACE:
        return _NOREPLACE[0]
    found: Callable[..., tuple[int, int]] | None = None
    if sys.platform.startswith("linux"):
        import ctypes

        try:
            func = ctypes.CDLL(None, use_errno=True).renameat2
        except (OSError, AttributeError, TypeError):
            func = None
        if func is not None:
            func.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            func.restype = ctypes.c_int

            def call(old: bytes, new: bytes, old_dir: int = _AT_FDCWD) -> tuple[int, int]:
                result = int(func(old_dir, old, _AT_FDCWD, new, _RENAME_NOREPLACE))
                return result, ctypes.get_errno()

            found = call
    _NOREPLACE.append(found)
    return found


def _move(folder: Path, target: Path, *, anchor: int | None = None) -> None:
    """Rename ``folder`` to ``target``, never replacing a project there: ``FileExistsError``
    (or ``OSError`` with ``EEXIST``/``ENOTEMPTY``) when ``target`` exists. With ``anchor`` (a
    descriptor of ``folder``'s parent, POSIX) the source is ``folder``'s name in that folder,
    so a parent swapped by its name meanwhile cannot change what is moved.

    Linux: ``renameat2(RENAME_NOREPLACE)``; when that call is not available here
    (:data:`_UNAVAILABLE`) a rename right after a fresh check, which cannot replace a project
    either (renaming a folder onto a folder that is not empty fails with ``ENOTEMPTY``, onto a
    file with ``ENOTDIR``); any other error of the call is raised. Windows' rename never
    replaces an existing folder."""
    noreplace = _renameat2()
    if noreplace is not None:
        if anchor is None:
            result, code = noreplace(os.fsencode(folder), os.fsencode(target))
        else:
            result, code = noreplace(os.fsencode(folder.name), os.fsencode(target), anchor)
        if result == 0:
            return
        if code not in _UNAVAILABLE:
            raise OSError(code, os.strerror(code), str(folder), None, str(target))
    if sys.platform != "win32" and (os.path.lexists(target) or _is_link(target)):
        raise FileExistsError(errno.EEXIST, "exists", str(target))
    if anchor is None:
        os.rename(folder, target)
    else:
        os.rename(folder.name, target, src_dir_fd=anchor)


def _move_own(work: Path, folder: Path, target: Path, made: _Made) -> None:
    """Rename this placement's project folder (checked to be the one made, in the build folder
    made) to ``target``: on POSIX named relative to the build folder's checked descriptor."""
    if _windows():
        _identify(work, "the build folder", made.work, folder=True)
        _identify(folder, "the built project folder", made.folder, folder=True)
        _move(folder, target)
        return
    work_fd, _ = _open_own(work, "the build folder", work, made.work, folder=True)
    try:
        folder_fd, _ = _open_own(
            folder.name,
            "the built project folder",
            folder,
            made.folder,
            folder=True,
            dir_fd=work_fd,
        )
        os.close(folder_fd)
        _move(folder, target, anchor=work_fd)
    finally:
        os.close(work_fd)


def _identity(path: Path) -> FileId:
    st = os.lstat(path)
    return st.st_dev, st.st_ino


def _remove(path: Path) -> str | None:
    """Remove ``path`` (a folder or a file). Returns why it could not be, None when it is gone
    (an interrupt propagates: :func:`_cleanup` names what is left)."""
    try:
        if _is_link(path) or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except OSError as exc:
        return f"{path} could not be removed ({exc})"
    return None


@refusing("library_unwritable", "placing the project in the library", "check the library")
def place_project(
    library: Path,
    name: str,
    build: Callable[[Path], _T],
    *,
    user: str | None = None,
    file_id: Callable[[_T], FileId | None] | None = None,
) -> tuple[Placed, _T]:
    """Build a project with ``build(path)`` (it writes ``path``, a ``Project.db`` in the staging
    folder, and runs its checks) and place it as project ``name``; see the module docstring.
    ``file_id`` gives, from what ``build`` returned, the identity of ``Project.db`` as the
    build created it (:func:`created_file`): the placement refuses another file in its place.
    Raises :class:`ProjectWriteError`: ``not_a_library``, ``bad_project_name``,
    ``project_exists``, ``library_unwritable``, or whatever ``build`` raises (with what could
    not be removed named on it)."""
    check_project_name(name)
    library = Path(library)
    top, projects, user_name = _projects_dir(library, user)
    _inside(projects, library, "the Projects folder")
    root = top / STAGING
    claim = _Lock(root / _lock_name(name), f"{secrets.token_hex(16)} {os.getpid()}\n".encode())
    placed = Placed(projects / name / "Project.db", projects / name, user_name)
    work: Path | None = None
    made = _Made()
    moving: dict[str, FileId] = {}
    # From the first statement that may make something in the library (the staging folder)
    # to the end of the cleanup, everything runs inside this one handler.
    try:
        _stage_and_lock(library, root, projects, claim, name, placed, made)
        clash = _exists(projects, name)
        if clash is not None:
            raise _refuse(
                "project_exists",
                f"the library already has a project folder {clash!r}",
                "choose another project name (aar never writes into an existing project)",
                folder=clash,
            )
        work = root / f"build-{secrets.token_hex(8)}"
        try:
            work.mkdir(mode=0o700)
        except OSError as exc:
            work = None  # not made by this placement: never removed by it
            raise _refuse(
                "library_unwritable",
                f"cannot make a staging folder in {root}: {exc}",
                "check that the library is writable and has room",
            ) from None
        made.work = _identify(work, "the build folder", None, folder=True)
        result = _build_and_rename(
            library, work, projects, name, build, placed, moving, made, file_id
        )
        residue, interrupt = _cleanup(work, claim, root, made)
    except BaseException as exc:
        residue, interrupt = _cleanup(work, claim, root, made)
        if not isinstance(exc, Exception) and _moved(moving, placed.folder):
            # An interrupt after the rename: the project is placed; say so.
            _note_placed(exc, placed, residue)
            raise
        if residue:
            text = "; ".join(residue)
            with_residue(exc, text, f"left in the library: {text}; delete it by hand")
        if isinstance(exc, OSError):
            refusal = _refuse(
                "library_unwritable",
                f"the project could not be built or placed in the library: {exc}"
                + (
                    f" (left in the library: {'; '.join(residue)}; delete it by hand)"
                    if residue
                    else ""
                ),
                "check that the library is still there, writable and has room",
                **({"residue": "; ".join(residue)} if residue else {}),
            )
            if interrupt is not None:
                raise interrupt from refusal
            raise refusal from None
        if interrupt is not None:
            if residue:
                with_residue(
                    interrupt, "; ".join(residue), f"left in the library: {'; '.join(residue)}"
                )
            raise interrupt from exc
        raise
    placed.cleanup.extend(f"{r}; delete it by hand" for r in residue)
    if interrupt is not None:
        _note_placed(interrupt, placed, residue)
        raise interrupt
    return placed, result


def _stage_and_lock(
    library: Path,
    root: Path,
    projects: Path,
    claim: _Lock,
    name: str,
    placed: Placed,
    made: _Made,
) -> None:
    """Make (private) or accept the staging folder (:func:`_check_root`) and take the lock of
    the folded name. Another export's cleanup may remove the staging folder while it is empty,
    right after this one checked it: then it is made again, at most :data:`_ROOT_TRIES` times,
    so exports of different names never fail each other over it."""
    for attempt in range(1, _ROOT_TRIES + 1):
        last = attempt == _ROOT_TRIES
        again = f" (tried {attempt} times)" if last else ""
        try:
            try:
                root.mkdir(mode=0o700)
                made.root = True
            except FileExistsError:
                made.root = False
            _check_root(root, library, projects)
            made.root = True
            placed.leftovers = sorted(_listdir(root))
        except OSError as exc:
            if not last and _vanished(exc):
                time.sleep(0.002 * attempt)
                continue
            raise _refuse(
                "library_unwritable",
                f"cannot make a staging folder in {root.parent}: {exc}{again}",
                "check that the library is writable and has room",
            ) from None
        try:
            _take_lock(root, claim, name)
        except OSError as exc:
            if not last and _vanished(exc):
                time.sleep(0.002 * attempt)
                continue
            raise _refuse(
                "library_unwritable",
                f"cannot make a lock in the staging folder {root}: {exc}{again}",
                "check that the library is writable and has room",
            ) from None
        return


def _moved(moving: dict[str, FileId], target: Path) -> bool:
    """Whether the folder this placement was renaming is at ``target`` now."""
    mine = moving.get("folder")
    return mine is not None and _placed_anyway(target, mine)


def _note_placed(exc: BaseException, placed: Placed, residue: list[str]) -> None:
    text = f"the project was placed at {placed.folder}" + (
        f"; left in the library: {'; '.join(residue)}" if residue else ""
    )
    with_residue(exc, "; ".join(residue), text)


def _remove_root(root: Path) -> str | None:
    """Remove the staging folder when it is empty. Not empty is no one's error (another
    placement is building there, or a killed one left a folder each placement reports); an
    empty one that cannot be removed is named."""
    if _is_link(root):
        return None
    try:
        root.rmdir()
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (errno.ENOTEMPTY, errno.EEXIST) or getattr(exc, "winerror", 0) == 145:
            return None
        return f"{root} (the staging folder, empty) could not be removed ({exc})"
    return None


def _remove_work(work: Path, made: _Made) -> str | None:
    """Remove the build folder when it is the one this placement made; one put in its place
    is named, never removed."""
    if made.work is not None and os.path.lexists(work):
        try:
            _identify(work, "the build folder", made.work, folder=True)
        except ProjectWriteError:
            return (
                f"{work} (not removed: it is not the build folder this export made; something "
                "put it in its place)"
            )
    return _remove(work)


def _cleanup(
    work: Path | None, claim: _Lock, root: Path, made: _Made
) -> tuple[list[str], BaseException | None]:
    """Remove the build folder (:func:`_remove_work`), then this placement's lock, then the
    staging folder if it is empty. Returns what is left and an interrupt that landed meanwhile.
    A lock this placement may have made but cannot be sure of (:func:`_lock_left`) is named,
    never removed."""
    residue: list[str] = []
    interrupt: BaseException | None = None

    def attempt(path: Path, step: Callable[[], str | None]) -> None:
        nonlocal interrupt
        try:
            problem = step()
        except BaseException as exc:  # an interrupt (or anything else): name what is left
            problem = (
                f"{path} was not removed ({type(exc).__name__}: {exc})"
                if os.path.lexists(path)
                else None
            )
            if not isinstance(exc, Exception):
                interrupt = interrupt or exc
        if problem is not None:
            residue.append(problem)

    if work is not None:
        attempt(work, lambda: _remove_work(work, made))

    def lock_step() -> str | None:
        left = _lock_left(claim)
        if left == "mine":
            return _remove(claim.path)
        if left == "maybe":
            return (
                f"{claim.path} (an empty lock this export may have made; delete it when no "
                "export is running)"
            )
        return None

    attempt(claim.path, lock_step)
    # Else the staging folder holds what is named (each path is inside it), or it is not this
    # placement's to remove (one it refused).
    if not residue and made.root:
        attempt(root, lambda: _remove_root(root))
    return residue, interrupt


def _build_and_rename(
    library: Path,
    work: Path,
    projects: Path,
    name: str,
    build: Callable[[Path], _T],
    placed: Placed,
    moving: dict[str, FileId],
    made: _Made,
    file_id: Callable[[_T], FileId | None] | None,
) -> _T:
    if os.stat(work).st_dev != os.stat(projects).st_dev:
        raise _refuse(
            "library_unwritable",
            "the staging folder and the projects are on different file systems",
            "keep the library's Resolve Projects folder on one file system",
        )
    folder = work / "project"
    folder.mkdir(mode=0o700)
    made.folder = _identify(folder, "the built project folder", None, folder=True)
    result = by_caller(build, folder / "Project.db")
    expected = file_id(result) if file_id is not None else None
    placed.permissions, placed.file_id = _settle(work, folder, made, expected, projects)
    # Again right before the move: nothing on the way may have become a link meanwhile.
    _, again, _ = _projects_dir(library, placed.user)
    if again != projects:
        raise _refuse("not_a_library", "the library's folders changed while building", "retry")
    for path, what in ((work, "the build folder"), (folder, "the built project folder")):
        _real_folder(path, what, "library_unwritable")
        _inside(path, library, what)
    _inside(projects, library, "the Projects folder")
    clash = _exists(projects, name)
    if clash is not None:
        raise _refuse(
            "project_exists",
            f"a project folder {clash!r} appeared while the project was built",
            "choose another project name",
            folder=clash,
        )
    target = projects / name
    mine = made.folder
    assert mine is not None
    moving["folder"] = mine
    try:
        _move_own(work, folder, target, made)
    except OSError as exc:
        if _placed_anyway(target, mine):
            # The rename reported an error but the folder is in place (a share that answers
            # late): the project is placed; say so.
            placed.cleanup.append(
                f"the rename reported {type(exc).__name__}: {exc}, but the project is in place"
            )
            return result
        if isinstance(exc, FileExistsError) or exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise _refuse(
                "project_exists",
                f"a project folder {name!r} appeared while the project was placed",
                "choose another project name",
            ) from None
        raise
    return result


def _placed_anyway(target: Path, mine: FileId) -> bool:
    """Whether ``target`` is the very folder that was being moved (same device and inode)."""
    try:
        here = _identity(target)
    except OSError:
        return False
    return here == mine and mine[1] != 0
