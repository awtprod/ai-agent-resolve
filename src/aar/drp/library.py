"""Read a project saved in a Resolve disk project library (``Project.db``).

A disk library is a folder: ``<library>/Resolve Projects/Users/<user>/Projects/<project>/``
holds each project's ``Project.db``, an SQLite 3 file in rollback-journal mode. Its tables are
the element types of a ``.drp``'s XML (``SM_Project``, ``Sm2MpMedia``, ``Sm2TiItem`` and so on);
the primary key ``<Table>_id`` is the XML's ``DbId``, columns are the XML's child elements, blob
columns hold the raw bytes the XML writes as hex, and vectors are join tables ``<A>_<B>`` with
``DbOwner``, ``DbAssociate``, ``DbPropertyName`` and ``DbIndex``.

This module does not interpret a project. It presents the rows the reader needs to
:mod:`aar.drp.reader` as in-memory XML members laid out like a ``.drp``'s (``project.xml``,
``MediaPool/.../MpFolder.xml``, ``SeqContainer/<id>.xml``): blobs hex-encoded, vectors rebuilt
from the join tables in ``DbIndex`` order. Only the columns the reader interprets are carried, so
the view is never a valid ``.drp`` and is never written anywhere. Blob decoding, classification,
limits and issue codes stay in the reader. Afterwards :func:`finish` replaces every provenance
with the row and column the value came from (see :func:`finish`), and the model's ``kind`` is
``library``; :mod:`aar.drp.modify` refuses such a model.

Reading never touches the live file except to copy it: the file is opened read-only, copied to a
private snapshot outside the library once it has settled (``aar.media.settle``) and no
``-journal`` or ``-wal`` file is beside it, and the snapshot is checked with SQLite's own
``integrity_check`` and then read through an immutable, read-only connection, so no journal,
WAL or shm file is created next to either file. The snapshot is deleted on every path.

Memory is bounded while the rows are fetched, not afterwards: every row and every byte fetched is
charged against ``max_rows`` and ``max_load_bytes`` as it arrives, and SQLite itself refuses a
single value over ``max_value_bytes``. The layout is bounded the same way: every value is charged
against the layout budget (``max_layout_bytes``, twice ``max_load_bytes`` unless set) at its
laid-out size before it is laid out, once per place it is laid out, so a row referenced many
times is refused before its copies are made. The whole read after the snapshot (the SQL queries
and the layout) has one time budget, ``max_read_seconds``, and a caller's own deadline
(``stop_at``) is honoured during the layout too. A link the reader would not consume (an unknown
``DbPropertyName``, or an owner that is not read) is reported, never dropped silently.
"""

from __future__ import annotations

import binascii
import bisect
import math
import os
import re
import sqlite3
import stat
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar

from aar.drp.container import DEFAULT_MAX_MEMBER_BYTES, MemberInfo
from aar.drp.errors import ContainerError
from aar.drp.model import DrpFile, Issue, Provenance, VersionInfo
from aar.media.settle import Copied, TooLargeError, copy_settled, settle_identity

__all__ = [
    "LIBRARY_KIND",
    "SQLITE_MAGIC",
    "SUPPORTED_PROJECT_VERSIONS",
    "LibraryError",
    "LibraryListing",
    "LibraryOptions",
    "LibraryProject",
    "LibraryRows",
    "LibraryView",
    "Skipped",
    "finish",
    "is_library_path",
    "list_library",
    "open_library",
    "read_rows",
]

LIBRARY_KIND = "library"
SQLITE_MAGIC = b"SQLite format 3\x00"
#: Project versions whose layout was checked against Resolve's own .drp export of the same
#: project (15, Resolve 20) or read from Resolve 21.1 saves (17). Others are refused.
SUPPORTED_PROJECT_VERSIONS = frozenset({15, 17})
DEFAULT_SETTLE_SECONDS = 2.0
DEFAULT_DEADLINE_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 0.25
#: Budgets of one read. The largest real save measured (a show project: a 29 MB file, 9,878
#: timeline items in 39 timelines) has 21,637 rows in the tables read, 7.27 MB of load charge
#: (4.55 MB of values plus the per-value charge), lays out to 7.09 MB and has no value over
#: 29 KB. Each budget reads that project ten times over; DESIGN "Shared project library" gives
#: the measured peak memory of the worst file each budget accepts.
DEFAULT_MAX_DB_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_ROWS = 220_000
DEFAULT_MAX_LOAD_BYTES = 80 * 1024 * 1024
#: The layout budget is this many times ``max_load_bytes`` unless set: a blob's laid-out (hex)
#: form is twice its bytes.
LAYOUT_PER_LOAD_BYTE = 2
DEFAULT_MAX_VALUE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_READ_SECONDS = 120.0
#: Budgets of one listing (:func:`list_library`).
DEFAULT_LISTING_SECONDS = 120.0
DEFAULT_LISTING_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ISSUES_PER_CODE = 100
_MAX_BIN_DEPTH = 256
#: A folder's member name holds its parent folders' names, up to this many characters; a deeper
#: or longer path starts again from the folder's own name (the name only orders the members).
_MAX_BIN_MEMBER_CHARS = 1024
#: The layout checks the read's clock once per this many rows or writes.
_CLOCK_EVERY = 1024
_MAX_LISTED_PROJECTS = 500
_MAX_SCANNED_ENTRIES = 20_000
_MAX_FOLDER_DEPTH = 8
_MAX_SHOWN_SKIPPED = 10
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_:.\-]*\Z")
_LIBRARY_SUFFIXES = (".drp", ".drt", ".drb")
_BIN_TAIL = "/MpFolder.xml"

_JOIN = ("DbOwner", "DbAssociate", "DbPropertyName", "DbIndex")
#: Every table and column read. A table or column missing from a file refuses it as
#: ``not_a_project``. Owner columns (``Sm2TiTrack_id`` ...) are read only to cross-check the
#: join tables.
COLUMNS: dict[str, tuple[str, ...]] = {
    "SM_Project": (
        "SM_Project_id",
        "ProjectName",
        "ProjectVersion",
        "LastModTimeInSecs",
        "CurrentTimelineIndex",
        "CommonConfig",
    ),
    "SM_Project_Sm2Timeline": _JOIN,
    "SM_Config": ("SM_Config_id", "FieldsBlob"),
    "BtLockableBlob": ("BtLockableBlob_id", "DbType", "BlobOwner", "FieldsBlob"),
    "Sm2MpFolder": ("Sm2MpFolder_id", "Name", "MpFolder"),
    "Sm2MpFolder_Sm2MpMedia": _JOIN,
    "Sm2MpMedia": (
        "Sm2MpMedia_id",
        "DbType",
        "Name",
        "MpFolder",
        "UniqueMediaPoolItemId",
        "Video",
        "Left",
        "Right",
        "LeftEye",
        "RightEye",
        "Sequence",
        "TimelineSharedHandle",
        "Sm2MpFolder_id",
    ),
    "BtVideoInfo": ("BtVideoInfo_id", "Clip", "Time", "Geometry"),
    "BtAudioInfo": ("BtAudioInfo_id", "Clip", "TracksBA"),
    "BtAudioInfo_Sm2MpMedia": _JOIN,
    "Sm2Timeline": ("Sm2Timeline_id", "Name", "Sequence"),
    "Sm2Sequence": ("Sm2Sequence_id", "FieldsBlob", "FrameRate", "Resolution", "MediaExtents"),
    "Sm2SequenceContainer": ("Sm2SequenceContainer_id",),
    "Sm2SequenceContainer_Sm2TiTrack": _JOIN,
    "Sm2TiTrack": (
        "Sm2TiTrack_id",
        "DbType",
        "Type",
        "SubType",
        "Flags",
        "Sequence",
        "UserDefinedName",
        "Sm2SequenceContainer_id",
    ),
    "Sm2TiItem_Sm2TiTrack": _JOIN,
    "Sm2TiItem": (
        "Sm2TiItem_id",
        "DbType",
        "Name",
        "PrettyType",
        "Start",
        "Duration",
        "In",
        "MediaRef",
        "MediaFilePath",
        "MediaTimemapBA",
        "MediaTrackIdx",
        "Sm2TiTrack_id",
    ),
}
_BLOB_COLUMNS = frozenset(
    {
        "FieldsBlob",
        "Clip",
        "Time",
        "Geometry",
        "TracksBA",
        "FrameRate",
        "Resolution",
        "MediaExtents",
        "MediaTimemapBA",
    }
)
_TRACK_VECTORS = ("VideoTrackVec", "AudioTrackVec", "SubtitleTrackVec")
#: The ``DbPropertyName`` values the reader consumes, per join table: the vectors of a ``.drp``'s
#: XML that the layout below rebuilds. A link under any other property, or from an owner that is
#: not laid out, is reported as ``library-unread-link`` and not laid out.
_READ_PROPERTIES: dict[str, tuple[str, ...]] = {
    "SM_Project_Sm2Timeline": ("TimelineHandleVec",),
    "Sm2MpFolder_Sm2MpMedia": ("MediaVec",),
    "BtAudioInfo_Sm2MpMedia": ("EmbeddedAudioVec",),
    "Sm2SequenceContainer_Sm2TiTrack": _TRACK_VECTORS,
    "Sm2TiItem_Sm2TiTrack": ("Items",),
}
_ITEM_FIELDS = (
    "PrettyType",
    "Name",
    "Start",
    "Duration",
    "In",
    "MediaRef",
    "MediaFilePath",
    "MediaTimemapBA",
    "MediaTrackIdx",
)
#: Item types whose XML leaves ``<In>`` out entirely (every one of the 69 transitions in the
#: private corpus) rather than writing it empty: a NULL ``In`` stays absent for them, so the
#: reader keeps ``source_in`` None. Every other type writes ``<In/>`` for a NULL ``In``.
_NO_EMPTY_IN = frozenset({"Sm2TiTransition"})


class LibraryError(ContainerError):
    """A ``Project.db`` that is refused, with a stable ``code`` and a ``hint`` to fix it.

    Codes: ``busy`` (a journal is beside it, or it never settled before the deadline),
    ``damaged`` (SQLite cannot read it, or ``integrity_check`` fails), ``too_large`` (file, row,
    load, value, layout or time budget), ``not_a_project`` (not SQLite, or missing a table or
    column this reader needs), ``unsupported_version``, ``unreadable`` (the file cannot be
    opened or is not a regular file) and ``not_a_library`` (a folder given to
    :func:`list_library` has no ``Users`` folder).
    """

    def __init__(self, code: str, message: str, hint: str, *, timed_out: str | None = None) -> None:
        super().__init__(f"{message} [{code}]; {hint}")
        self.code = code
        self.message = message
        self.hint = hint
        #: The read ran out of time: ``"read"`` (its own ``max_read_seconds``) or ``"caller"``
        #: (the caller's ``stop_at``, such as a listing's deadline); None otherwise.
        self.timed_out = timed_out

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


@dataclass(frozen=True, slots=True)
class LibraryOptions:
    """How long to wait for a save to finish, and the budgets of one read."""

    settle_seconds: float = DEFAULT_SETTLE_SECONDS
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    max_db_bytes: int = DEFAULT_MAX_DB_BYTES
    #: Rows fetched from the tables read, in total.
    max_rows: int = DEFAULT_MAX_ROWS
    #: Bytes of text and blob values fetched, in total, plus 16 per value (charged as each row
    #: arrives).
    max_load_bytes: int = DEFAULT_MAX_LOAD_BYTES
    #: Bytes laid out (XML text, blobs as hex), charged before each value is laid out; None is
    #: ``LAYOUT_PER_LOAD_BYTE`` times ``max_load_bytes``.
    max_layout_bytes: int | None = None
    #: One value at most (SQLite's own length limit; also never more than ``max_member_bytes``).
    max_value_bytes: int = DEFAULT_MAX_VALUE_BYTES
    #: The read after the snapshot is taken: integrity check, queries and layout.
    max_read_seconds: float = DEFAULT_MAX_READ_SECONDS
    #: A time on ``clock`` by which the whole read must end, whatever is left of
    #: ``max_read_seconds`` (a listing sets its deadline here); None for none.
    stop_at: float | None = None
    #: Where the private snapshot is written (default: the system temporary folder). Never the
    #: project's own folder or a folder inside it.
    snapshot_dir: Path | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time

    @property
    def layout_limit(self) -> int:
        if self.max_layout_bytes is not None:
            return self.max_layout_bytes
        return LAYOUT_PER_LOAD_BYTE * self.max_load_bytes


class _Deadline:
    """The read's time budget: ``max_read_seconds`` from its start, or the caller's ``stop_at``
    if that comes first. :meth:`check` raises the refusal once it has passed."""

    def __init__(self, path: Path, opts: LibraryOptions) -> None:
        self.path = path
        self.opts = opts
        self.clock = opts.clock
        own = opts.clock() + opts.max_read_seconds
        caller = opts.stop_at if opts.stop_at is not None else math.inf
        self.stop_at = min(own, caller)
        self.by = "caller" if caller < own else "read"

    def passed(self) -> bool:
        return self.clock() > self.stop_at

    def check(self) -> None:
        if self.passed():
            raise self.error()

    def error(self) -> LibraryError:
        if self.by == "caller":
            return LibraryError(
                "too_large",
                f"{self.path}: the caller's deadline passed while it was read",
                "give the read more time",
                timed_out="caller",
            )
        return LibraryError(
            "too_large",
            f"{self.path}: reading it took longer than {self.opts.max_read_seconds:g} s",
            "if this really is one project, raise the limit with --max-read-seconds",
            timed_out="read",
        )


def is_library_path(path: str | Path) -> bool:
    """Whether :func:`aar.drp.read_drp` reads ``path`` as a library project.

    A ``.drp``, ``.drt`` or ``.drb`` never is (a ``.drb`` that is SQLite stays best effort, as
    before). A ``.db`` always is, so a non-SQLite ``Project.db`` is refused with a reason rather
    than listed as an unknown zip. Any other file is when its first bytes are SQLite's.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _LIBRARY_SUFFIXES:
        return False
    if suffix == ".db":
        return True
    try:
        with p.open("rb") as fh:
            return fh.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


# --------------------------------------------------------------------------- snapshot


def _sidecar(path: Path) -> str | None:
    """The name of a journal or WAL file beside ``path``: a write is in progress."""
    for suffix in ("-journal", "-wal"):
        side = path.with_name(path.name + suffix)
        if os.path.lexists(side):
            return side.name
    return None


def _settled_copy(path: Path, target: Any, opts: LibraryOptions) -> Copied:
    deadline = opts.clock() + opts.deadline_seconds
    watched: tuple[int, int, int, int] | None = None
    since = 0.0
    while True:
        side = _sidecar(path)
        if side is not None:
            reason = f"{side} is beside it (Resolve is writing the project)"
            watched = None
        else:
            try:
                identity = settle_identity(os.stat(path))
            except OSError as exc:
                raise LibraryError(
                    "unreadable", f"cannot read {path}: {exc}", "check the path and permissions"
                ) from None
            moment = opts.clock()
            if identity != watched:
                watched, since = identity, moment
            quiet = watched if moment - since >= opts.settle_seconds else None
            target.seek(0)
            target.truncate()
            try:
                copied = copy_settled(
                    path,
                    target,
                    max_bytes=opts.max_db_bytes,
                    settle_seconds=opts.settle_seconds,
                    what=path.name,
                    settled_identity=quiet,
                    now=opts.now,
                )
            except TooLargeError as exc:
                raise LibraryError(
                    "too_large",
                    f"{path}: {exc}",
                    "if this really is one project, raise the limit with --max-db-bytes",
                ) from None
            except (OSError, ValueError) as exc:
                raise LibraryError(
                    "unreadable",
                    f"cannot read {path}: {exc}",
                    "give the path of a regular Project.db file, not a link",
                ) from None
            if copied is not None:
                side = _sidecar(path)
                if side is None:
                    target.flush()
                    return copied
                reason = f"{side} appeared while it was copied"
            else:
                reason = (
                    f"it changed less than {opts.settle_seconds:g} s before the read, "
                    "or while it was copied"
                )
        if opts.clock() >= deadline:
            raise LibraryError(
                "busy",
                f"{path} is being saved: {reason}",
                "wait until Resolve has finished saving (or close the project), then read it again",
            )
        opts.sleep(max(0.0, min(opts.poll_seconds, deadline - opts.clock())))


def _snapshot_dir(path: Path, opts: LibraryOptions) -> Path:
    folder = opts.snapshot_dir if opts.snapshot_dir is not None else Path(tempfile.gettempdir())
    project = path.resolve().parent
    where = folder.resolve()
    if where == project or project in where.parents:
        raise LibraryError(
            "unreadable",
            f"the snapshot folder {folder} is inside the project's folder",
            "give a snapshot folder outside the library",
        )
    return folder


_KINDS = (
    (stat.S_ISDIR, "a folder"),
    (stat.S_ISLNK, "a symbolic link"),
    (stat.S_ISFIFO, "a named pipe (FIFO)"),
    (stat.S_ISSOCK, "a socket"),
    (stat.S_ISCHR, "a character device"),
    (stat.S_ISBLK, "a block device"),
)


def _check_kind(path: Path) -> None:
    """Refuse anything but a regular file, saying what it is (never opened, so never blocks)."""
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise LibraryError(
            "unreadable", f"cannot read {path}: {exc}", "check the path and permissions"
        ) from None
    if stat.S_ISREG(mode):
        return
    what = next((name for test, name in _KINDS if test(mode)), "not a regular file")
    hint = (
        "give the path of the Project.db file itself, not a link to it"
        if stat.S_ISLNK(mode)
        else "give the path of a regular Project.db file (a project folder holds one)"
    )
    raise LibraryError("unreadable", f"{path} is {what}, not a regular file", hint)


def _take_snapshot(path: Path, opts: LibraryOptions) -> tuple[Path, Copied]:
    """Copy ``path`` to a new private file; the caller deletes it. Nothing is left on error.

    The size limit is :func:`aar.media.settle.copy_settled`'s, checked on the opened file and
    again on every block copied.
    """
    _check_kind(path)
    fd, name = tempfile.mkstemp(prefix=".aar-library-", suffix=".db", dir=_snapshot_dir(path, opts))
    snapshot = Path(name)
    try:
        with os.fdopen(fd, "w+b") as target:
            copied = _settled_copy(path, target, opts)
    except BaseException:
        snapshot.unlink(missing_ok=True)
        raise
    return snapshot, copied


# --------------------------------------------------------------------------- SQLite


def _q(name: str) -> str:
    """An SQL identifier, quoted."""
    return '"' + name.replace('"', '""') + '"'


def _connect(
    snapshot: Path, deadline: _Deadline, max_value: int, *, text_as_bytes: bool = True
) -> sqlite3.Connection:
    """A read-only, immutable connection: SQLite creates no journal, WAL or lock files.

    Text comes back as bytes (text that is not UTF-8 is read, never a decode error), or with
    ``text_as_bytes=False`` as ``str``, where text that is not UTF-8 is an SQLite error."""
    uri = snapshot.resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        if text_as_bytes:
            conn.text_factory = bytes
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max(1, max_value))
        conn.set_progress_handler(lambda: 1 if deadline.passed() else 0, 10_000)
        conn.execute("PRAGMA trusted_schema = OFF")
        conn.execute("PRAGMA query_only = ON")
    except BaseException:
        conn.close()
        raise
    return conn


def _sql_error(path: Path, exc: sqlite3.Error, deadline: _Deadline) -> LibraryError:
    text = str(exc)
    if isinstance(exc, sqlite3.DataError) or "too big" in text:
        return LibraryError(
            "too_large",
            f"{path}: a value is larger than the per-value limit ({text})",
            "if this really is one project, raise the limit with --max-value-bytes",
        )
    if "interrupted" in text:
        return deadline.error()
    return LibraryError(
        "damaged",
        f"{path}: SQLite cannot read it ({type(exc).__name__}: {text})",
        "open the project in Resolve and save it again, or restore it from a backup",
    )


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _check_schema(conn: sqlite3.Connection, path: Path) -> None:
    found: dict[str, tuple[str, str]] = {}
    for kind, name, sql in conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table', 'view')"
    ):
        found[_text(name)] = (_text(kind), _text(sql or b""))
    missing: list[str] = []
    for table, columns in COLUMNS.items():
        kind, sql = found.get(table, ("", ""))
        if kind != "table" or sql.lstrip().upper().startswith("CREATE VIRTUAL"):
            missing.append(table)
            continue
        have = {_text(row[1]) for row in conn.execute(f"PRAGMA table_info({_q(table)})")}
        missing.extend(f"{table}.{c}" for c in columns if c not in have)
    if missing:
        shown = ", ".join(missing[:8]) + (
            f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        )
        raise LibraryError(
            "not_a_project",
            f"{path} is not a Resolve project this reader knows: it has no {shown}",
            "give the Project.db of a project in a Resolve disk library",
        )


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, bytes) and value.strip().isdigit() and len(value.strip()) <= 9:
        return int(value)
    return None


class _Row(tuple):  # type: ignore[type-arg]
    """One fetched row: a tuple (no per-row dict) whose values are also found by column name."""

    __slots__ = ()
    index: ClassVar[dict[str, int]] = {}

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return tuple.__getitem__(self, self.index[key])
        return tuple.__getitem__(self, key)


def _row_type(table: str, columns: tuple[str, ...]) -> type[_Row]:
    index = {c: i for i, c in enumerate(columns)}
    return type(f"_Row_{table.replace(':', '_')}", (_Row,), {"__slots__": (), "index": index})


_ROW_TYPES = {table: _row_type(table, columns) for table, columns in COLUMNS.items()}
#: What one fetched value costs beyond its bytes (the Python object and its slot), charged
#: against ``max_load_bytes`` so that many small values count too.
_VALUE_COST = 16

Rows = dict[str, list[_Row]]


def _load(conn: sqlite3.Connection, path: Path, opts: LibraryOptions) -> Rows:
    """Fetch the columns read, one row at a time, charging every row and byte as it arrives.

    The row count is checked first (SQLite counts without loading anything), and again while
    fetching; ``max_load_bytes`` is charged with each value's size, so a refusal comes before
    the budget's worth of memory is held, whatever the file's size says (a generated column
    can make values out of nothing).
    """
    total = 0
    for table in COLUMNS:
        (count,) = conn.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()
        total += int(count)
    if total > opts.max_rows:
        raise _too_many_rows(path, opts, total)
    rows_left = opts.max_rows
    bytes_left = opts.max_load_bytes
    tables: Rows = {}
    for table, columns in COLUMNS.items():
        make = _ROW_TYPES[table]
        out: list[_Row] = []
        sql = f"SELECT {', '.join(_q(c) for c in columns)} FROM {_q(table)}"
        for values in conn.execute(sql):  # one row per step, never a batch
            rows_left -= 1
            if rows_left < 0:
                raise _too_many_rows(path, opts, None)
            for value in values:
                bytes_left -= _VALUE_COST + (len(value) if isinstance(value, bytes) else 0)
            if bytes_left < 0:
                raise LibraryError(
                    "too_large",
                    f"{path}: the values read come to more than the {_size(opts.max_load_bytes)} "
                    f"load budget (at {table})",
                    "if this really is one project, raise the limit with --max-load-bytes",
                )
            out.append(make(values))
        tables[table] = out
    return tables


def _too_many_rows(path: Path, opts: LibraryOptions, counted: int | None) -> LibraryError:
    rows = f"{counted} rows" if counted is not None else f"more than {opts.max_rows} rows"
    return LibraryError(
        "too_large",
        f"{path} has {rows} in the tables read, over the {opts.max_rows}-row limit",
        "if this really is one project, raise the limit with --max-rows",
    )


def _size(n: int) -> str:
    return f"{n // (1024 * 1024)} MiB" if n % (1024 * 1024) == 0 else f"{n}-byte"


def _check_project(path: Path, tables: Rows) -> int:
    projects = tables["SM_Project"]
    if not projects:
        raise LibraryError(
            "not_a_project",
            f"{path} holds no project (SM_Project is empty)",
            "give the Project.db of a project in a Resolve disk library",
        )
    if len(projects) > 1:
        raise LibraryError(
            "damaged",
            f"{path} holds {len(projects)} SM_Project rows; a project file holds one",
            "open the project in Resolve and save it again, or restore it from a backup",
        )
    if _key(projects[0]["SM_Project_id"]) is None:
        raise LibraryError(
            "damaged",
            f"{path}: its SM_Project row has no SM_Project_id",
            "open the project in Resolve and save it again, or restore it from a backup",
        )
    version = _int_value(projects[0]["ProjectVersion"])
    if version not in SUPPORTED_PROJECT_VERSIONS:
        shown = "none" if version is None else str(version)
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_PROJECT_VERSIONS))
        raise LibraryError(
            "unsupported_version",
            f"{path} is project version {shown}; this reader knows versions {supported}",
            "export the project from Resolve as a .drp and read that instead",
        )
    return version


# --------------------------------------------------------------------------- the view


def _key(value: object) -> str | None:
    """A row id or reference as text (bytes kept exactly through surrogate escapes)."""
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", "surrogateescape").strip()
    else:
        text = str(value).strip()
    return text or None


def _raw(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")


def _content(column: str, value: object) -> tuple[bytes | None, bool]:
    """A column value as the XML text the reader expects: blobs as hex (which needs no
    escaping, so the flag is False), the rest as text that still has to be escaped (True)."""
    if value is None:
        return None, False
    if column in _BLOB_COLUMNS and isinstance(value, bytes):
        return binascii.hexlify(value), False
    if isinstance(value, bytes):
        return value, True
    if isinstance(value, float):
        # "86400.0" is not a frame count: read as an issue
        return repr(value).encode("ascii"), True
    return str(value).encode("utf-8", "surrogateescape"), True


def _laid_out_size(column: str, value: object) -> int:
    """What :func:`_content` makes of ``value``, in bytes before escaping, without making it."""
    if value is None:
        return 0
    if isinstance(value, bytes):
        return 2 * len(value) if column in _BLOB_COLUMNS else len(value)
    if isinstance(value, float):
        return len(repr(value))
    return len(str(value))


def _escape(data: bytes, *, attribute: bool = False) -> bytes:
    data = data.replace(b"&", b"&amp;").replace(b"<", b"&lt;").replace(b">", b"&gt;")
    return data.replace(b'"', b"&quot;") if attribute else data


def _member_part(name: str | None) -> str:
    """A folder or id as one member path part (no separators, bounded)."""
    cleaned = re.sub(r"[/\\\x00-\x1f]", "_", name or "")[:200].strip()
    return cleaned or "_"


class _Budget:
    """The layout's byte budget and the read's clock, both charged as the layout is built."""

    def __init__(self, limit: int, path: Path, deadline: _Deadline, *, hint: str) -> None:
        self.left = limit
        self.limit = limit
        self.path = path
        self.deadline = deadline
        self.hint = hint
        self.ticks = 0

    def charge(self, n: int) -> None:
        self.left -= n
        if self.left < 0:
            raise LibraryError(
                "too_large",
                f"{self.path}: the rows read lay out to more than the {_size(self.limit)} "
                "layout budget",
                self.hint,
            )
        self.tick()

    def tick(self) -> None:
        """One step of work: the read's clock is looked at once every ``_CLOCK_EVERY``."""
        self.ticks += 1
        if self.ticks >= _CLOCK_EVERY:
            self.ticks = 0
            self.deadline.check()


#: Where an element came from: its byte span in the member, the row (``<table>/<id>``, or a
#: join row's ``<join table>/<owner>/<property>/<DbIndex>``) and the column, if it is a value.
#: The row text is shared by every value of the row, so a span costs a tuple, not a string.
Span = tuple[int, int, str, str | None]


class _Doc:
    """One synthesized member, with the row and column behind every element it holds."""

    def __init__(self, name: str, budget: _Budget) -> None:
        self.name = name
        self.budget = budget
        self.buf = bytearray()
        self.stack: list[tuple[bytes, int, str | None]] = []
        self.spans: list[Span] = []
        self._write(b'<?xml version="1.0" encoding="UTF-8"?>\n')

    def _write(self, *parts: bytes) -> None:
        self.budget.charge(sum(len(part) for part in parts))
        for part in parts:
            self.buf += part

    def open(self, tag: str, ref: str | None = None, dbid: str | None = None) -> None:
        start = len(self.buf)
        name = tag.encode()
        if dbid is None:
            self._write(b"<", name, b">")
        else:
            self._write(b"<", name, b' DbId="', _escape(_raw(dbid), attribute=True), b'">')
        self.stack.append((name, start, ref))

    def close(self) -> None:
        name, start, ref = self.stack.pop()
        self._write(b"</", name, b">")
        if ref is not None:
            self.spans.append((start, len(self.buf), ref, None))

    def leaf(self, tag: str, content: bytes | None, row: str, column: str | None) -> None:
        """An element holding ``content`` as text (escaped here)."""
        start = len(self.buf)
        name = tag.encode()
        if content:
            self._write(b"<", name, b">", _escape(content), b"</", name, b">")
        else:
            self._write(b"<", name, b"/>")
        self.spans.append((start, len(self.buf), row, column))

    def value(self, column: str, value: object, row: str) -> None:
        """A column's element. Its laid-out size is charged before it is made (a blob's hex
        is twice its bytes), so a value laid out many times is refused before its copies."""
        start = len(self.buf)
        name = column.encode()
        self.budget.charge(_laid_out_size(column, value) + 2 * len(name) + 5)
        content, needs_escape = _content(column, value)
        if content:
            if needs_escape:
                escaped = _escape(content)
                self.budget.charge(len(escaped) - len(content))  # what escaping added
                content = escaped
            self.buf += b"<" + name + b">"
            self.buf += content
            self.buf += b"</" + name + b">"
        else:
            self.buf += b"<" + name + b"/>"
        self.spans.append((start, len(self.buf), row, column))

    def take(self) -> bytes:
        """The member's bytes; the buffer is released."""
        data = bytes(self.buf)
        self.buf = bytearray()
        return data


@dataclass(slots=True)
class _Link:
    owner: str
    child: str | None
    prop: str
    index: object
    ref: str


class _Builder:
    """Lay the rows out as the members of a ``.drp``, and report what does not fit."""

    def __init__(self, path: Path, tables: Rows, budget: _Budget) -> None:
        self.path = path
        self.budget = budget
        self.issues: list[Issue] = []
        self.counts: Counter[str] = Counter()
        self.docs: list[_Doc] = []
        self.member_rows: dict[str, str] = {}
        #: The next ``~n`` suffix to try for each member name asked for more than once.
        self.suffixes: dict[str, int] = {}
        #: Each folder's member path (``<parent path>/<name>``), made once per folder.
        self.bin_paths: dict[str, str] = {}
        #: Each folder's path length in the model (see :meth:`_model_path_sizes`).
        self.model_paths: dict[str, int] = {}
        self.rows: dict[str, dict[str, _Row]] = {}
        self.joins: dict[str, dict[str, dict[str, list[_Link]]]] = {}
        for table, rows in tables.items():
            if COLUMNS[table] == _JOIN:
                self.joins[table] = self._links(table, rows)
            else:
                self.rows[table] = self._index(table, rows)
        self.placed: dict[str, set[str]] = defaultdict(set)
        #: (owner, property) of every link group that was laid out, per join table.
        self.consumed: dict[str, set[tuple[str, str]]] = defaultdict(set)

    # ------------------------------------------------------------------ issues

    def issue(self, code: str, message: str, ref: str | None = None) -> None:
        self.counts[code] += 1
        if self.counts[code] <= _MAX_ISSUES_PER_CODE:
            where = Provenance(ref, 0, 0, "row") if ref is not None else None
            self.issues.append(Issue("warning", code, message, where))

    def summaries(self) -> None:
        for code, count in sorted(self.counts.items()):
            if count > _MAX_ISSUES_PER_CODE:
                more = count - _MAX_ISSUES_PER_CODE
                self.issues.append(
                    Issue("info", "issues-truncated", f"{more} more {code} issue(s) not listed")
                )

    # ------------------------------------------------------------------ rows and links

    def _index(self, table: str, rows: list[_Row]) -> dict[str, _Row]:
        out: dict[str, _Row] = {}
        id_column = f"{table}_id"
        for row in rows:
            self.budget.tick()
            rid = _key(row[id_column])
            if rid is None:
                self.issue("missing-id", f"a {table} row has no {id_column}; it is not read", table)
                continue
            if rid in out:
                self.issue(
                    "duplicate-id",
                    f"{table} row {rid} appears twice (the first is read)",
                    f"{table}/{rid}",
                )
                continue
            out[rid] = row
        return out

    def _links(self, table: str, rows: list[_Row]) -> dict[str, dict[str, list[_Link]]]:
        out: dict[str, dict[str, list[_Link]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            self.budget.tick()
            owner = _key(row["DbOwner"])
            prop = _key(row["DbPropertyName"]) or ""
            index = row["DbIndex"]
            if owner is None:
                self.issue("library-dangling-link", f"a {table} row has no DbOwner", table)
                continue
            ref = f"{table}/{owner}/{prop}/{_text(index) if index is not None else ''}"
            out[owner][prop].append(_Link(owner, _key(row["DbAssociate"]), prop, index, ref))
        for owner, props in out.items():
            for prop, links in props.items():
                seen: set[int] = set()
                for link in links:
                    self.budget.tick()
                    index = link.index
                    if not isinstance(index, int) or isinstance(index, bool) or index in seen:
                        self.issue(
                            "library-link-conflict",
                            f"{table}: the order of {prop} under {owner} is not clear "
                            f"(DbIndex {_text(index) if index is not None else 'NULL'})",
                            link.ref,
                        )
                    else:
                        seen.add(index)
                links.sort(
                    key=lambda link: (
                        not isinstance(link.index, int) or isinstance(link.index, bool),
                        link.index if isinstance(link.index, int) else 0,
                        link.child or "",
                    )
                )
        return out

    def _props(self, join: str, owner: str) -> list[tuple[str, list[_Link]]]:
        """The link groups of ``owner`` the reader consumes, in the reader's order; each one
        returned is recorded as laid out. Any other group is left for :meth:`unread_links`."""
        props = self.joins[join].get(owner, {})
        out = [(p, props[p]) for p in _READ_PROPERTIES[join] if p in props]
        self.consumed[join].update((owner, p) for p, _links in out)
        return out

    def _vector(self, join: str, owner: str, prop: str) -> list[_Link]:
        return dict(self._props(join, owner)).get(prop, [])

    def _child(self, table: str, link: _Link, join: str) -> _Row | None:
        row = self.rows[table].get(link.child or "")
        if row is None:
            self.issue(
                "library-dangling-link",
                f"{join} links {link.prop} of {link.owner} to {table} row "
                f"{link.child or '(none)'}, which does not exist",
                link.ref,
            )
        return row

    def _place(self, table: str, rid: str, link: _Link, owner_column: str | None = None) -> bool:
        """Record that ``rid`` is placed by ``link``; refuse a second placement."""
        if rid in self.placed[table]:
            self.issue(
                "library-link-conflict",
                f"{table} row {rid} is linked more than once (only the first link is read)",
                link.ref,
            )
            return False
        self.placed[table].add(rid)
        if owner_column is not None:
            owner = _key(self.rows[table][rid][owner_column])
            if owner is not None and owner != link.owner:
                self.issue(
                    "library-link-conflict",
                    f"{table} row {rid}: {owner_column} names {owner} but the link places it "
                    f"under {link.owner}",
                    f"{table}/{rid}",
                )
        return True

    def _type(self, table: str, rid: str, value: object, default: str | None = None) -> str | None:
        name = _key(value) or default
        if name is None or not _NAME.match(name):
            self.issue(
                "library-bad-type",
                f"{table} row {rid} has DbType {name!r}, which is not an element name; "
                "it is not read",
                f"{table}/{rid}",
            )
            return None
        return name

    def _doc(self, name: str, row_ref: str) -> _Doc:
        """A new member named ``name``, or with ``~<n>`` (the first free ``n``) added if that
        is taken: to the folder part of a folder's member (``MediaPool/A/B~2/MpFolder.xml``, so
        the reader still reads it as a folder), before the extension of any other. The ``n``
        last tried for each name is kept, so naming is linear in the number of members whatever
        the names."""
        unique = name
        if unique in self.member_rows:
            n = self.suffixes.get(name, 1)
            if name.endswith(_BIN_TAIL):
                stem, tail = name[: -len(_BIN_TAIL)], _BIN_TAIL
            else:
                stem, dot, ext = name.rpartition(".")
                tail = dot + ext
            while unique in self.member_rows:
                self.budget.tick()
                n += 1
                unique = f"{stem}~{n}{tail}"
            self.suffixes[name] = n
        doc = _Doc(unique, self.budget)
        self.docs.append(doc)
        self.member_rows[unique] = row_ref
        return doc

    def leaves(self, doc: _Doc, table: str, rid: str, row: _Row, cols: tuple[str, ...]) -> None:
        ref = f"{table}/{rid}"
        for col in cols:
            doc.value(col, row[col], ref)

    # ------------------------------------------------------------------ members

    def build(self) -> None:
        self.project()
        self.folders()
        self.containers()
        self.unlinked("Sm2TiItem", "Sm2TiTrack_id")
        self.unlinked("Sm2TiTrack", "Sm2SequenceContainer_id")
        self.unlinked("Sm2MpMedia", "Sm2MpFolder_id")
        self.unread_links()
        self.summaries()

    def project(self) -> None:
        [(pid, row)] = self.rows["SM_Project"].items()
        ref = f"SM_Project/{pid}"
        doc = self._doc("project.xml", ref)
        doc.open("SM_Project", ref, pid)
        self.leaves(
            doc,
            "SM_Project",
            pid,
            row,
            ("ProjectName", "ProjectVersion", "LastModTimeInSecs", "CurrentTimelineIndex"),
        )
        doc.open("TimelineHandleVec")
        for link in self._vector("SM_Project_Sm2Timeline", pid, "TimelineHandleVec"):
            doc.leaf("Element", _raw(link.child) if link.child else None, link.ref, None)
        doc.close()
        config_id = _key(row["CommonConfig"])
        if config_id is not None:
            config = self.rows["SM_Config"].get(config_id)
            if config is None:
                self.issue(
                    "library-dangling-link",
                    f"SM_Project.CommonConfig names SM_Config row {config_id}, "
                    "which does not exist",
                    f"{ref}/CommonConfig",
                )
            else:
                doc.open("CommonConfig")
                doc.open("SM_Config", f"SM_Config/{config_id}", config_id)
                self.leaves(doc, "SM_Config", config_id, config, ("FieldsBlob",))
                doc.close()
                doc.close()
        doc.open("LockableBlobMap")
        for bid, blob in self.rows["BtLockableBlob"].items():
            self.budget.tick()
            tag = self._type("BtLockableBlob", bid, blob["DbType"])
            if tag is None:
                continue
            doc.open("Element")
            doc.open(tag, f"BtLockableBlob/{bid}", bid)
            self.leaves(doc, "BtLockableBlob", bid, blob, ("BlobOwner", "FieldsBlob"))
            doc.close()
            doc.close()
        doc.close()
        doc.close()

    def _bin_path(self, fid: str) -> str:
        """The folder's names from the top down, ``/``-separated: its parent's path (made once
        per folder, so a chain of folders costs one step each) and its own name. A path longer
        than ``_MAX_BIN_MEMBER_CHARS`` or deeper than ``_MAX_BIN_DEPTH`` starts again from the
        folder's own name, and a cycle is cut where the walk meets itself: the path only names
        and orders the member, and the reader reads the folder's parent from the row."""
        folders = self.rows["Sm2MpFolder"]
        paths = self.bin_paths
        chain: list[str] = []
        walking: set[str] = set()
        node: str | None = fid
        while node is not None and node in folders and node not in paths and node not in walking:
            self.budget.tick()
            walking.add(node)
            chain.append(node)
            node = _key(folders[node]["MpFolder"])
        above = paths.get(node, "") if node is not None else ""
        for folder in reversed(chain):
            part = _member_part(_key(folders[folder]["Name"]))
            path = f"{above}/{part}" if above else part
            if len(path) > _MAX_BIN_MEMBER_CHARS or path.count("/") >= _MAX_BIN_DEPTH:
                path = part
            paths[folder] = above = path
        return paths[fid]

    def _bin_member(self, fid: str) -> str:
        return "MediaPool/" + self._bin_path(fid) + _BIN_TAIL

    def _model_path_sizes(self) -> None:
        """How long each folder's path (``Bin.path``) will be in the model, so the layout can
        charge it before the reader makes it: a chain of folders with long names makes paths
        that grow with the chain's depth, which no single value or row shows.

        The reader joins a folder's name (``?`` if it has none) with its parents' names, walking
        up until a parent is missing or a folder repeats, at most ``_MAX_BIN_DEPTH + 1`` names.
        Here a folder's size is its name's size plus one plus its parent's size, every folder of
        a cycle having the whole cycle's size; that is exact without the depth cap and never
        less with it, and it is cut to the depth cap times the longest name. One step per folder.
        """
        folders = self.rows["Sm2MpFolder"]
        sizes = self.model_paths

        def own(fid: str) -> int:
            value = folders[fid]["Name"]
            return (len(value) if isinstance(value, bytes) and value else 1) + 1

        longest = max((own(fid) for fid in folders), default=0)
        cap = (_MAX_BIN_DEPTH + 1) * longest
        for start in folders:
            chain: list[str] = []
            at: dict[str, int] = {}
            node: str | None = start
            while node is not None and node in folders and node not in sizes and node not in at:
                self.budget.tick()
                at[node] = len(chain)
                chain.append(node)
                node = _key(folders[node]["MpFolder"])
            if node is not None and node in at:  # the walk met itself: a cycle
                cycle = chain[at[node] :]
                whole = min(cap, sum(own(f) for f in cycle))
                for f in cycle:
                    sizes[f] = whole
                del chain[at[node] :]
                above = whole
            else:
                above = sizes.get(node, 0) if node is not None else 0
            for f in reversed(chain):
                above = sizes[f] = min(cap, above + own(f))

    def folders(self) -> None:
        self._model_path_sizes()
        for fid, row in self.rows["Sm2MpFolder"].items():
            self.budget.charge(self.model_paths[fid])  # its Bin.path in the model
            ref = f"Sm2MpFolder/{fid}"
            doc = self._doc(self._bin_member(fid), ref)
            doc.open("Sm2MpFolder", ref, fid)
            self.leaves(doc, "Sm2MpFolder", fid, row, ("Name", "MpFolder"))
            doc.open("MediaVec")
            for link in self._vector("Sm2MpFolder_Sm2MpMedia", fid, "MediaVec"):
                self.budget.tick()
                media = self._child("Sm2MpMedia", link, "Sm2MpFolder_Sm2MpMedia")
                if media is None or link.child is None:
                    continue
                if not self._place("Sm2MpMedia", link.child, link, "Sm2MpFolder_id"):
                    continue
                doc.open("Element")
                self.media(doc, link.child, media)
                doc.close()
            doc.close()
            doc.close()

    def _info(self, doc: _Doc, wrapper: str, info_id: str, ref: str) -> None:
        for table, cols in (
            ("BtVideoInfo", ("Clip", "Time", "Geometry")),
            ("BtAudioInfo", ("Clip", "TracksBA")),
        ):
            info = self.rows[table].get(info_id)
            if info is not None:
                doc.open(wrapper)
                doc.open(table, f"{table}/{info_id}", info_id)
                self.leaves(doc, table, info_id, info, cols)
                doc.close()
                doc.close()
                return
        self.issue(
            "library-dangling-link",
            f"{ref} names stream info {info_id}, which does not exist",
            ref,
        )

    def _sequence(self, doc: _Doc, wrapper: str, seq_id: str, ref: str) -> None:
        seq = self.rows["Sm2Sequence"].get(seq_id)
        if seq is None:
            self.issue(
                "library-dangling-link",
                f"{ref} names Sm2Sequence row {seq_id}, which does not exist",
                ref,
            )
            return
        doc.open(wrapper)
        doc.open("Sm2Sequence", f"Sm2Sequence/{seq_id}", seq_id)
        self.leaves(
            doc,
            "Sm2Sequence",
            seq_id,
            seq,
            ("FieldsBlob", "FrameRate", "Resolution", "MediaExtents"),
        )
        doc.close()
        doc.close()

    def media(self, doc: _Doc, mid: str, row: _Row) -> None:
        ref = f"Sm2MpMedia/{mid}"
        tag = self._type("Sm2MpMedia", mid, row["DbType"])
        if tag is None:
            return
        doc.open(tag, ref, mid)
        self.leaves(doc, "Sm2MpMedia", mid, row, ("Name", "MpFolder", "UniqueMediaPoolItemId"))
        info = _key(row["Video"])
        if info is not None:
            self._info(doc, "Video", info, f"{ref}/Video")
        for prop, links in self._props("BtAudioInfo_Sm2MpMedia", mid):
            doc.open(prop)
            for link in links:
                audio = self._child("BtAudioInfo", link, "BtAudioInfo_Sm2MpMedia")
                if audio is None or link.child is None:
                    continue
                doc.open("Element")
                doc.open("BtAudioInfo", f"BtAudioInfo/{link.child}", link.child)
                self.leaves(doc, "BtAudioInfo", link.child, audio, ("Clip", "TracksBA"))
                doc.close()
                doc.close()
            doc.close()
        for column in ("Left", "Right", "LeftEye", "RightEye"):
            info = _key(row[column])
            if info is not None:
                self._info(doc, column, info, f"{ref}/{column}")
        seq_id = _key(row["Sequence"])
        if seq_id is not None:
            self._sequence(doc, "Sequence", seq_id, f"{ref}/Sequence")
        tl_id = _key(row["TimelineSharedHandle"])
        if tl_id is not None:
            timeline = self.rows["Sm2Timeline"].get(tl_id)
            if timeline is None:
                self.issue(
                    "library-dangling-link",
                    f"{ref}/TimelineSharedHandle names Sm2Timeline row {tl_id}, "
                    "which does not exist",
                    f"{ref}/TimelineSharedHandle",
                )
            else:
                doc.open("TimelineSharedHandle")
                doc.open("Sm2Timeline", f"Sm2Timeline/{tl_id}", tl_id)
                self.leaves(doc, "Sm2Timeline", tl_id, timeline, ("Name",))
                seq_id = _key(timeline["Sequence"])
                if seq_id is not None:
                    self._sequence(doc, "Sequence", seq_id, f"Sm2Timeline/{tl_id}/Sequence")
                doc.close()
                doc.close()
        doc.close()

    def containers(self) -> None:
        for cid, _row in self.rows["Sm2SequenceContainer"].items():
            self.budget.tick()
            ref = f"Sm2SequenceContainer/{cid}"
            doc = self._doc(f"SeqContainer/{_member_part(cid)}.xml", ref)
            doc.open("Sm2SequenceContainer", ref, cid)
            for prop, links in self._props("Sm2SequenceContainer_Sm2TiTrack", cid):
                doc.open(prop)
                for link in links:
                    self.budget.tick()
                    track = self._child("Sm2TiTrack", link, "Sm2SequenceContainer_Sm2TiTrack")
                    if track is None or link.child is None:
                        continue
                    if not self._place("Sm2TiTrack", link.child, link, "Sm2SequenceContainer_id"):
                        continue
                    doc.open("Element")
                    self.track(doc, link.child, track)
                    doc.close()
                doc.close()
            doc.close()

    def track(self, doc: _Doc, tid: str, row: _Row) -> None:
        tag = self._type("Sm2TiTrack", tid, row["DbType"], "Sm2TiTrack")
        if tag is None:
            return
        doc.open(tag, f"Sm2TiTrack/{tid}", tid)
        self.leaves(
            doc, "Sm2TiTrack", tid, row, ("Type", "SubType", "Flags", "Sequence", "UserDefinedName")
        )
        for prop, links in self._props("Sm2TiItem_Sm2TiTrack", tid):
            doc.open(prop)
            for link in links:
                self.budget.tick()
                item = self._child("Sm2TiItem", link, "Sm2TiItem_Sm2TiTrack")
                if item is None or link.child is None:
                    continue
                if not self._place("Sm2TiItem", link.child, link, "Sm2TiTrack_id"):
                    continue
                self.item(doc, link.child, item)
            doc.close()
        doc.close()

    def item(self, doc: _Doc, iid: str, row: _Row) -> None:
        tag = self._type("Sm2TiItem", iid, row["DbType"])
        if tag is None:
            return
        fields = _ITEM_FIELDS
        if row["In"] is None and tag in _NO_EMPTY_IN:
            fields = tuple(f for f in fields if f != "In")
        doc.open("Element")
        doc.open(tag, f"Sm2TiItem/{iid}", iid)
        self.leaves(doc, "Sm2TiItem", iid, row, fields)
        doc.close()
        doc.close()

    def unlinked(self, table: str, owner_column: str) -> None:
        """Report the rows of ``table`` no link placed: they are not read (none on real saves)."""
        placed = self.placed[table]
        missed: list[str] = []
        for rid in self.rows[table]:
            self.budget.tick()
            if rid not in placed:
                missed.append(rid)
        if missed:
            owned = sum(1 for rid in missed if _key(self.rows[table][rid][owner_column]))
            self.issue(
                "library-unlinked-row",
                f"{len(missed)} {table} row(s) are placed by no link ({owned} of them name an "
                f"owner in {owner_column}), so they are not read",
                f"{table}/{missed[0]}",
            )

    def unread_links(self) -> None:
        """Report every link group that was not laid out, so no row it links disappears silently.

        A group is left out when its ``DbPropertyName`` is not one the reader consumes (an item
        under ``Other``, a track under ``Whatever``) or when its owner is never laid out (a
        missing row, a track whose type is not read, a media row no folder places).
        """
        for join, owners in self.joins.items():
            for owner, props in owners.items():
                for prop, links in props.items():
                    self.budget.tick()
                    if (owner, prop) in self.consumed[join]:
                        continue
                    read = _READ_PROPERTIES[join]
                    why = (
                        f"the reader reads only {', '.join(read)} there"
                        if prop not in read
                        else f"{owner} is not read"
                    )
                    self.issue(
                        "library-unread-link",
                        f"{join}: {len(links)} link(s) of {owner} under {prop!r} are not read "
                        f"({why}), so the rows they link are not read through them",
                        links[0].ref,
                    )


@dataclass(eq=False)
class LibraryView:
    """A ``Project.db`` presented to :mod:`aar.drp.reader` as in-memory members.

    It has the attributes and methods of :class:`~aar.drp.container.Container` the reader uses.
    ``path`` is the ``Project.db`` that was read; ``size``, ``sha256`` and ``head_hex`` are the
    snapshot's, which was that file's content when it settled.
    """

    path: Path
    size: int
    sha256: str
    head_hex: str
    members: list[MemberInfo]
    max_member_bytes: int
    fixed_versions: VersionInfo
    issues: list[Issue]
    member_rows: dict[str, str]
    spans: dict[str, list[Span]]
    _data: dict[str, bytes] = field(repr=False)
    #: The read's time budget, still running while the reader builds the model from the view.
    deadline: _Deadline | None = field(default=None, repr=False)
    kind: str = LIBRARY_KIND
    kind_source: str = "sqlite"
    sniffed_kind: str = LIBRARY_KIND
    signature: str | None = "sqlite"
    warnings: list[str] = field(default_factory=list)
    is_zip: bool = False
    has_members: bool = True

    def names(self) -> list[str]:
        return [m.name for m in self.members]

    def xml_members(self) -> list[str]:
        return self.names()

    def read(self, name: str, *, max_bytes: int | None = None) -> bytes:
        data = self._data.get(name)
        if data is None:
            raise ContainerError(f"no member {name!r}")
        limit = (
            self.max_member_bytes if max_bytes is None else min(max_bytes, self.max_member_bytes)
        )
        if len(data) > limit:
            raise ContainerError(
                f"member {name!r} is {len(data)} bytes, over the {limit}-byte limit"
            )
        return data

    def head(self, name: str) -> bytes:
        return self.read(name)[:1024]

    def check_time(self) -> None:
        """Refuse the read (:class:`LibraryError`) if its time budget has run out."""
        if self.deadline is not None:
            self.deadline.check()

    def close(self) -> None:
        self._data.clear()

    def __enter__(self) -> LibraryView:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_library(
    path: str | Path,
    *,
    options: LibraryOptions | None = None,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_total_xml_bytes: int = 600 * 1024 * 1024,
) -> LibraryView:
    """Snapshot, check and lay out one ``Project.db``; raises :class:`LibraryError` if refused.

    The layout is charged against ``options.layout_limit`` (and never more than the reader's
    ``max_total_xml_bytes``). The read's time budget starts once the snapshot is taken and
    keeps running in the returned view (:meth:`LibraryView.check_time`).
    """
    p = Path(path)
    opts = options or LibraryOptions()
    snapshot, copied = _take_snapshot(p, opts)
    deadline = _Deadline(p, opts)
    try:
        with snapshot.open("rb") as fh:
            head = fh.read(100)
        if not head.startswith(SQLITE_MAGIC):
            raise LibraryError(
                "not_a_project",
                f"{p} is not an SQLite file (leading bytes {head[:16].hex() or 'none'})",
                "give the Project.db of a project in a Resolve disk library",
            )
        try:
            conn = _connect(snapshot, deadline, min(max_member_bytes, opts.max_value_bytes))
        except sqlite3.Error as exc:
            raise _sql_error(p, exc, deadline) from None
        try:
            try:
                check = conn.execute("PRAGMA integrity_check(10)").fetchall()
                if [_text(r[0]) for r in check] != ["ok"]:
                    problems = "; ".join(_text(r[0]) for r in check[:3])
                    raise LibraryError(
                        "damaged",
                        f"{p} fails SQLite's integrity check: {problems}",
                        "open the project in Resolve and save it again, or restore it from a "
                        "backup",
                    )
                _check_schema(conn, p)
                tables = _load(conn, p, opts)
            except sqlite3.Error as exc:
                raise _sql_error(p, exc, deadline) from None
        finally:
            conn.close()
    finally:
        snapshot.unlink(missing_ok=True)
    version = _check_project(p, tables)
    deadline.check()
    if opts.layout_limit <= max_total_xml_bytes:
        budget = _Budget(
            opts.layout_limit,
            p,
            deadline,
            hint="if this really is one project, raise the limit with --max-layout-bytes",
        )
    else:
        budget = _Budget(
            max_total_xml_bytes,
            p,
            deadline,
            hint=(
                "this is the most the reader holds (max_total_xml_bytes); --max-layout-bytes "
                "cannot raise it further"
            ),
        )
    builder = _Builder(p, tables, budget)
    del tables
    builder.build()
    [pid] = builder.rows["SM_Project"]
    builder.rows.clear()  # the rows are laid out: release them before the members are made
    builder.joins.clear()
    data: dict[str, bytes] = {}
    spans: dict[str, list[Span]] = {}
    for doc in builder.docs:
        budget.tick()
        data[doc.name] = doc.take()
        doc.spans.sort()
        spans[doc.name] = doc.spans
    deadline.check()
    return LibraryView(
        path=p,
        size=copied.size,
        sha256=copied.sha256,
        head_hex=head[:16].hex(),
        members=[MemberInfo(n, len(d), len(d), 0, False) for n, d in data.items()],
        max_member_bytes=max_member_bytes,
        fixed_versions=VersionInfo(
            app_version=None,
            project_version=version,
            source=Provenance(f"SM_Project/{pid}/ProjectVersion", 0, 0, "column"),
        ),
        issues=builder.issues,
        member_rows=builder.member_rows,
        spans=spans,
        _data=data,
        deadline=deadline,
    )


@dataclass(frozen=True, slots=True)
class LibraryRows:
    """Every column of some tables of one ``Project.db``, read by :func:`read_rows`.

    ``rows[table]`` holds tuples in ``columns[table]`` order, with SQLite's own storage classes:
    ``str`` (TEXT), ``bytes`` (BLOB), ``int``, ``float`` or ``None``."""

    path: Path
    size: int
    sha256: str
    project_version: int
    columns: dict[str, tuple[str, ...]]
    rows: dict[str, list[tuple[Any, ...]]]


def read_rows(
    path: str | Path, tables: tuple[str, ...], *, options: LibraryOptions | None = None
) -> LibraryRows:
    """Every column and row of ``tables`` in one ``Project.db``, through the same private
    snapshot, integrity check, project check and budgets as :func:`open_library`.

    The live file is never opened for writing; the snapshot is deleted on every path. Rows and
    bytes are charged against ``max_rows`` and ``max_load_bytes`` as each row arrives, a value
    over ``max_value_bytes`` is refused by SQLite, and the read has ``max_read_seconds``.
    ``SM_Project`` is always read (the project check needs it). Raises :class:`LibraryError`.
    """
    p = Path(path)
    opts = options or LibraryOptions()
    wanted = tuple(dict.fromkeys(("SM_Project", *tables)))
    for table in wanted:
        if not isinstance(table, str) or not _NAME.match(table):
            raise ValueError(f"not a table name: {table!r}")
    snapshot, copied = _take_snapshot(p, opts)
    deadline = _Deadline(p, opts)
    columns: dict[str, tuple[str, ...]] = {}
    rows: dict[str, list[tuple[Any, ...]]] = {}
    try:
        with snapshot.open("rb") as fh:
            head = fh.read(100)
        if not head.startswith(SQLITE_MAGIC):
            raise LibraryError(
                "not_a_project",
                f"{p} is not an SQLite file (leading bytes {head[:16].hex() or 'none'})",
                "give the Project.db of a project in a Resolve disk library",
            )
        try:
            conn = _connect(snapshot, deadline, opts.max_value_bytes, text_as_bytes=False)
        except sqlite3.Error as exc:
            raise _sql_error(p, exc, deadline) from None
        try:
            try:
                check = conn.execute("PRAGMA integrity_check(10)").fetchall()
                if [str(r[0]) for r in check] != ["ok"]:
                    raise LibraryError(
                        "damaged",
                        f"{p} fails SQLite's integrity check: {check[0][0]}",
                        "open the project in Resolve and save it again, or restore it from a "
                        "backup",
                    )
                kinds = dict(
                    conn.execute(
                        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"
                    ).fetchall()
                )
                missing = [t for t in wanted if kinds.get(t) != "table"]
                if missing:
                    raise LibraryError(
                        "not_a_project",
                        f"{p} is not a Resolve project this reader knows: it has no "
                        f"{', '.join(missing[:8])}",
                        "give the Project.db of a project in a Resolve disk library",
                    )
                total = 0
                for table in wanted:
                    (count,) = conn.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()
                    total += int(count)
                if total > opts.max_rows:
                    raise _too_many_rows(p, opts, total)
                rows_left, bytes_left = opts.max_rows, opts.max_load_bytes
                for table in wanted:
                    names = tuple(
                        str(r[1]) for r in conn.execute(f"PRAGMA table_info({_q(table)})")
                    )
                    columns[table] = names
                    out: list[tuple[Any, ...]] = []
                    sql = f"SELECT {', '.join(_q(c) for c in names)} FROM {_q(table)}"
                    for values in conn.execute(sql):  # one row per step
                        rows_left -= 1
                        if rows_left < 0:
                            raise _too_many_rows(p, opts, None)
                        for value in values:
                            size = len(value) if isinstance(value, bytes | str) else 0
                            bytes_left -= _VALUE_COST + size
                        if bytes_left < 0:
                            raise LibraryError(
                                "too_large",
                                f"{p}: the values read come to more than the "
                                f"{_size(opts.max_load_bytes)} load budget (at {table})",
                                "if this really is one project, raise the limit with "
                                "--max-load-bytes",
                            )
                        out.append(tuple(values))
                    rows[table] = out
            except sqlite3.Error as exc:
                raise _sql_error(p, exc, deadline) from None
        finally:
            conn.close()
    finally:
        snapshot.unlink(missing_ok=True)
    index = {c: i for i, c in enumerate(columns["SM_Project"])}
    projects = [{c: r[i] for c, i in index.items()} for r in rows["SM_Project"]]
    version = _check_project(p, {"SM_Project": projects})  # type: ignore[dict-item]
    deadline.check()
    return LibraryRows(p, copied.size, copied.sha256, version, columns, rows)


# --------------------------------------------------------------------------- provenance


class _Relabel:
    def __init__(self, view: LibraryView) -> None:
        self.view = view
        self.starts = {name: [s[0] for s in spans] for name, spans in view.spans.items()}

    def ref(self, member: str, offset: int) -> str | None:
        spans = self.view.spans.get(member)
        if not spans:
            return self.view.member_rows.get(member)
        i = bisect.bisect_right(self.starts[member], offset) - 1
        while i >= 0:
            start, end, row, column = spans[i]
            if start <= offset < end:
                return row if column is None else f"{row}/{column}"
            i -= 1
        return self.view.member_rows.get(member)

    def prov(self, p: Provenance | None) -> Provenance | None:
        if p is None or p.member not in self.view.spans:
            return p
        ref = self.ref(p.member, p.start) or LIBRARY_KIND
        if p.kind == "blob":
            kind = "blob"
        elif ref.count("/") >= 2:
            kind = "column"
        else:
            kind = "row"
        return Provenance(ref, 0, 0, kind, p.blob_path)

    def spans(self, spans: dict[str, Provenance]) -> dict[str, Provenance]:
        return {k: self.prov(v) or v for k, v in spans.items()}


def finish(drp: DrpFile, view: LibraryView) -> DrpFile:
    """Make a model read from ``view`` say where its values came from in the database.

    Every :class:`~aar.drp.model.Provenance` names a row or a cell instead of a byte span:
    ``member`` is ``<table>/<row id>`` for an object (``kind="row"``) and
    ``<table>/<row id>/<column>`` for a value (``kind="column"``, or ``"blob"`` with the
    reader's ``blob_path`` inside the blob); values from a join table name
    ``<join table>/<owner id>/<property>/<DbIndex>``. ``start`` and ``end`` are 0: there is
    no text to splice. ``Bin.member`` and ``Timeline.member`` name their folder and sequence
    container rows. ``members`` and ``raw_members`` are empty (a library project has no zip
    members), and the library layer's own issues come first in ``issues``.
    """
    r = _Relabel(view)
    if drp.project is not None:
        drp.project.source = r.prov(drp.project.source)
        drp.project.spans = r.spans(drp.project.spans)
    for b in drp.bins:
        b.member = view.member_rows.get(b.member, b.member)
        b.source = r.prov(b.source)
        b.spans = r.spans(b.spans)
    for m in drp.media:
        m.source = r.prov(m.source)
        m.spans = r.spans(m.spans)
    for tl in drp.timelines:
        if tl.member is not None:
            tl.member = view.member_rows.get(tl.member, tl.member)
        tl.source = r.prov(tl.source)
        tl.spans = r.spans(tl.spans)
        for marker in tl.markers:
            marker.source = r.prov(marker.source)
        for track in tl.tracks:
            track.source = r.prov(track.source)
            track.spans = r.spans(track.spans)
            for item in track.items:
                item.source = r.prov(item.source)
                item.spans = r.spans(item.spans)
                for marker in item.markers:
                    marker.source = r.prov(marker.source)
    drp.issues = list(view.issues) + [replace(i, where=r.prov(i.where)) for i in drp.issues]
    drp.members = []
    drp.raw_members = []
    return drp


# --------------------------------------------------------------------------- listing


@dataclass(slots=True)
class LibraryProject:
    """One project folder of a library: read, refused, or not read at all (``not_read``)."""

    user: str
    folder: str
    path: str
    file_modified: str | None
    name: str | None = None
    id: str | None = None
    project_version: int | None = None
    last_modified_secs: int | None = None
    last_modified: str | None = None
    timelines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    refusal: dict[str, str] | None = None
    #: Why the project was not read: the listing's deadline or byte budget ran out first.
    not_read: str | None = None


#: What each kind of skipped entry is, as the listing says it.
SKIP_REASONS = {
    "symlink": "symbolic link(s), never followed",
    "unreadable": "folder(s) that could not be listed",
    "too_deep": f"folder(s) more than {_MAX_FOLDER_DEPTH} levels down, not looked into",
    "nested": "project(s) inside another project's folder, not read",
}
#: Why the search for projects stopped before it had looked everywhere.
CUT_REASONS = {
    "projects": f"more than {_MAX_LISTED_PROJECTS} projects",
    "entries": f"more than {_MAX_SCANNED_ENTRIES} folder entries",
    "deadline": "the listing's deadline passed while folders were searched",
}


@dataclass(slots=True)
class Skipped:
    """How many entries were skipped for one reason, and the first few (relative to ``Users``)."""

    count: int = 0
    first: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LibraryListing:
    """Every ``Project.db`` found under a library's ``Users/<user>/Projects`` folders.

    ``complete`` is true only when every folder was searched (nothing ``skipped``, the search
    not ``cut_short``) and every project found was read or refused (none ``not_read``).
    """

    root: str
    projects: list[LibraryProject]
    cut_short: list[str] = field(default_factory=list)
    skipped: dict[str, Skipped] = field(default_factory=dict)
    deadline_seconds: float = DEFAULT_LISTING_SECONDS
    max_bytes: int = DEFAULT_LISTING_BYTES
    bytes_read: int = 0

    @property
    def not_read(self) -> int:
        return sum(1 for p in self.projects if p.not_read is not None)

    @property
    def complete(self) -> bool:
        return not self.cut_short and not self.skipped and not self.not_read

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "complete": self.complete,
            "cut_short": [{"code": c, "message": CUT_REASONS[c]} for c in self.cut_short],
            "skipped": {
                reason: {"count": s.count, "first": s.first, "what": SKIP_REASONS[reason]}
                for reason, s in self.skipped.items()
            },
            "not_read": self.not_read,
            "deadline_seconds": self.deadline_seconds,
            "max_bytes": self.max_bytes,
            "bytes_read": self.bytes_read,
            "projects": [
                {f: getattr(p, f) for f in LibraryProject.__slots__}  # type: ignore[attr-defined]
                for p in self.projects
            ],
        }


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(seconds, UTC).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


class _Scan:
    """Find the project folders under ``Users`` within the entry, depth, project and time
    limits, following no link, and record everything it skips and why it stopped."""

    def __init__(self, users: Path, deadline: float, clock: Callable[[], float]) -> None:
        self.users = users
        self.deadline = deadline
        self.clock = clock
        self.entries_left = _MAX_SCANNED_ENTRIES
        self.skipped: dict[str, Skipped] = {}
        self.cut_short: list[str] = []
        self.found: list[tuple[str, Path, Path]] = []

    @property
    def stopped(self) -> bool:
        return bool(self.cut_short)

    def skip(self, reason: str, path: Path, detail: str | None = None) -> None:
        record = self.skipped.setdefault(reason, Skipped())
        record.count += 1
        if len(record.first) < _MAX_SHOWN_SKIPPED:
            try:
                name = path.relative_to(self.users).as_posix()
            except ValueError:
                name = str(path)
            record.first.append(f"{name} ({detail})" if detail else name)

    def cut(self, reason: str) -> None:
        if reason not in self.cut_short:
            self.cut_short.append(reason)

    def listdir(self, folder: Path) -> list[os.DirEntry[str]]:
        """The entries of ``folder``, each charged to the entry budget (raises ``OSError``)."""
        out: list[os.DirEntry[str]] = []
        with os.scandir(folder) as entries:
            for entry in entries:
                if self.entries_left <= 0:
                    self.cut("entries")
                    break
                self.entries_left -= 1
                out.append(entry)
        return out

    def entries(self, folder: Path) -> list[os.DirEntry[str]]:
        if self.clock() > self.deadline:
            self.cut("deadline")
            return []
        try:
            return self.listdir(folder)
        except OSError as exc:
            self.skip("unreadable", folder, exc.strerror or type(exc).__name__)
            return []

    def kind(self, entry: os.DirEntry[str]) -> str | None:
        """``"link"``, ``"folder"`` or None (a file), recording an entry that cannot be told."""
        try:
            if entry.is_symlink():
                return "link"
            return "folder" if entry.is_dir(follow_symlinks=False) else None
        except OSError as exc:
            self.skip("unreadable", Path(entry.path), exc.strerror or type(exc).__name__)
            return None

    def run(self, user_entries: list[os.DirEntry[str]]) -> None:
        for user in sorted(user_entries, key=lambda e: e.name):
            if self.stopped:
                return
            kind = self.kind(user)
            if kind == "link":
                self.skip("symlink", Path(user.path))
            elif kind == "folder":
                self.projects(user.name, Path(user.path) / "Projects")

    def projects(self, user: str, projects_dir: Path) -> None:
        try:
            mode = os.lstat(projects_dir).st_mode
        except OSError as exc:
            self.skip("unreadable", projects_dir, exc.strerror or type(exc).__name__)
            return
        if stat.S_ISLNK(mode):
            self.skip("symlink", projects_dir)
            return
        if not stat.S_ISDIR(mode):
            self.skip("unreadable", projects_dir, "not a folder")
            return
        # Each folder to look into, its depth and the project folder it is inside (None if
        # none): a project's own folders are searched too, so a project nested inside another
        # project's folder is listed as skipped rather than lost.
        stack: list[tuple[Path, int, Path | None]] = [(projects_dir, 0, None)]
        while stack and not self.stopped:
            folder, depth, inside = stack.pop()
            listed = self.entries(folder)
            if self.stopped:
                return
            if folder != projects_dir and any(e.name == "Project.db" for e in listed):
                if inside is not None:
                    self.skip("nested", folder / "Project.db", f"inside {inside.name}")
                elif len(self.found) >= _MAX_LISTED_PROJECTS:
                    self.cut("projects")
                    return
                else:
                    self.found.append((user, projects_dir, folder / "Project.db"))
                    inside = folder
            children: list[os.DirEntry[str]] = []
            for entry in listed:
                kind = self.kind(entry)
                if kind == "link":
                    self.skip("symlink", Path(entry.path))
                elif kind == "folder":
                    if depth >= _MAX_FOLDER_DEPTH:
                        self.skip("too_deep", Path(entry.path))
                    else:
                        children.append(entry)
            for entry in sorted(children, key=lambda e: e.name, reverse=True):
                stack.append((Path(entry.path), depth + 1, inside))


def list_library(
    root: str | Path,
    *,
    options: LibraryOptions | None = None,
    read: Callable[[Path, LibraryOptions], DrpFile] | None = None,
    deadline_seconds: float = DEFAULT_LISTING_SECONDS,
    max_bytes: int = DEFAULT_LISTING_BYTES,
) -> LibraryListing:
    """List every project of the library at ``root`` (or its ``Resolve Projects`` folder).

    Each project is read in full (``read``, by default :func:`aar.drp.read_drp` with
    ``options``); a project that is refused (busy, damaged, ...) is listed with its refusal.
    The whole listing has one deadline (``deadline_seconds``, on ``options.clock``) and one
    byte budget (``max_bytes``, the sizes of the files read): a project's wait for a save is cut
    to what is left and its read ends by the deadline (``LibraryOptions.stop_at``, honoured
    during the layout too), and a project not reached in time or within the budget is listed
    with ``not_read`` saying why. A project folder inside another project's folder is listed in
    ``skipped`` (``nested``), not read. At most 500 projects and 20,000 folder
    entries (the ``Users`` folder's included) are looked at, at most 8 folders deep, and no
    link is followed; every entry skipped is counted in ``skipped`` and the search stopping
    early is in ``cut_short``, so ``complete`` says whether anything was left out.
    """
    opts = options or LibraryOptions()
    deadline = opts.clock() + deadline_seconds
    base = Path(root)
    if (base / "Resolve Projects").is_dir():
        base = base / "Resolve Projects"
    users = base / "Users"
    if not users.is_dir():
        raise LibraryError(
            "not_a_library",
            f"{root} is not a Resolve disk library (no Users folder in it or in its "
            "'Resolve Projects' folder)",
            "give the library's folder, the one that holds 'Resolve Projects'",
        )
    if read is None:
        from aar.drp.reader import read_drp

        def read(p: Path, o: LibraryOptions) -> DrpFile:
            return read_drp(p, library_options=o)

    scan = _Scan(users, deadline, opts.clock)
    try:
        user_entries = scan.listdir(users)
    except OSError as exc:
        raise LibraryError(
            "unreadable", f"cannot list {users}: {exc}", "check the path and permissions"
        ) from None
    scan.run(user_entries)
    listing = LibraryListing(
        str(root),
        [],
        cut_short=scan.cut_short,
        skipped=scan.skipped,
        deadline_seconds=deadline_seconds,
        max_bytes=max_bytes,
    )
    for user_name, projects_dir, db in scan.found:
        try:
            st = os.lstat(db)
            mtime: float | None = st.st_mtime
            size = st.st_size
        except OSError:
            mtime, size = None, 0
        entry = LibraryProject(
            user=user_name,
            folder=db.parent.relative_to(projects_dir).as_posix(),
            path=str(db),
            file_modified=_iso(mtime),
        )
        listing.projects.append(entry)
        left = deadline - opts.clock()
        if left <= 0:
            entry.not_read = (
                f"the listing's {deadline_seconds:g} s deadline passed before it was reached "
                "(raise --deadline)"
            )
            continue
        if listing.bytes_read + size > max_bytes:
            entry.not_read = (
                f"it is {size} bytes, past what is left of the listing's {max_bytes}-byte "
                f"budget ({listing.bytes_read} bytes read; raise --max-listing-bytes)"
            )
            continue
        each = replace(
            opts,
            deadline_seconds=min(opts.deadline_seconds, left),
            stop_at=deadline if opts.stop_at is None else min(deadline, opts.stop_at),
        )
        try:
            drp = read(db, each)
        except LibraryError as exc:
            listing.bytes_read += size
            cut_wait = exc.code == "busy" and each.deadline_seconds < opts.deadline_seconds
            cut_query = exc.timed_out == "caller"
            if cut_wait:
                entry.not_read = (
                    f"the listing's {deadline_seconds:g} s deadline passed while it was waited "
                    f"for ({exc.message}; raise --deadline)"
                )
            elif cut_query:
                entry.not_read = (
                    f"the listing's {deadline_seconds:g} s deadline passed while it was read "
                    "(raise --deadline)"
                )
            else:
                entry.refusal = exc.to_dict()
            continue
        except ContainerError as exc:
            listing.bytes_read += size
            entry.refusal = {"code": "unreadable", "message": str(exc), "hint": "check the file"}
            continue
        listing.bytes_read += max(size, drp.size or 0)
        p = drp.project
        if p is not None:
            entry.name, entry.id = p.name, p.id
            entry.last_modified_secs = p.last_modified_secs
            entry.last_modified = _iso(p.last_modified_secs)
        entry.project_version = drp.versions.project_version if drp.versions else None
        entry.timelines = [t.name or "" for t in drp.timelines if t.kind == "timeline"]
        entry.warnings = sorted({i.code for i in drp.issues if i.severity in ("warning", "error")})
    return listing
