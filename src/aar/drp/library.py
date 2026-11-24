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
"""

from __future__ import annotations

import bisect
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

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
    "LibraryView",
    "finish",
    "is_library_path",
    "list_library",
    "open_library",
]

LIBRARY_KIND = "library"
SQLITE_MAGIC = b"SQLite format 3\x00"
#: Project versions whose layout was checked against Resolve's own .drp export of the same
#: project (15, Resolve 20) or read from Resolve 21.1 saves (17). Others are refused.
SUPPORTED_PROJECT_VERSIONS = frozenset({15, 17})
DEFAULT_SETTLE_SECONDS = 2.0
DEFAULT_DEADLINE_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 0.25
DEFAULT_MAX_DB_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_MAX_QUERY_SECONDS = 120.0
_MAX_ISSUES_PER_CODE = 100
_MAX_BIN_DEPTH = 256
_MAX_LISTED_PROJECTS = 500
_MAX_SCANNED_ENTRIES = 20_000
_MAX_FOLDER_DEPTH = 8
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_:.\-]*\Z")
_LIBRARY_SUFFIXES = (".drp", ".drt", ".drb")

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
    value or time budget), ``not_a_project`` (not SQLite, or missing a table or column this
    reader needs), ``unsupported_version``, ``unreadable`` (the file cannot be opened or is not
    a regular file) and ``not_a_library`` (a folder given to :func:`list_library` has no
    ``Users`` folder).
    """

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(f"{message} [{code}]; {hint}")
        self.code = code
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


@dataclass(frozen=True, slots=True)
class LibraryOptions:
    """How long to wait for a save to finish, and the budgets of one read."""

    settle_seconds: float = DEFAULT_SETTLE_SECONDS
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    max_db_bytes: int = DEFAULT_MAX_DB_BYTES
    max_rows: int = DEFAULT_MAX_ROWS
    max_query_seconds: float = DEFAULT_MAX_QUERY_SECONDS
    #: Where the private snapshot is written (default: the system temporary folder). Never the
    #: project's own folder or a folder inside it.
    snapshot_dir: Path | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time


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
                    "raise max_db_bytes if this really is one project",
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


def _take_snapshot(path: Path, opts: LibraryOptions) -> tuple[Path, Copied]:
    """Copy ``path`` to a new private file; the caller deletes it. Nothing is left on error."""
    try:
        size = os.stat(path).st_size
    except OSError as exc:
        raise LibraryError(
            "unreadable", f"cannot read {path}: {exc}", "check the path and permissions"
        ) from None
    if size > opts.max_db_bytes:
        raise LibraryError(
            "too_large",
            f"{path} is {size} bytes, over the {opts.max_db_bytes}-byte limit",
            "raise max_db_bytes if this really is one project",
        )
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


def _connect(snapshot: Path, opts: LibraryOptions, max_value: int) -> sqlite3.Connection:
    """A read-only, immutable connection: SQLite creates no journal, WAL or lock files."""
    uri = snapshot.resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        conn.text_factory = bytes  # text that is not UTF-8 is read, never a decode error
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max(1, max_value))
        stop_at = opts.clock() + opts.max_query_seconds
        conn.set_progress_handler(lambda: 1 if opts.clock() > stop_at else 0, 10_000)
        conn.execute("PRAGMA trusted_schema = OFF")
        conn.execute("PRAGMA query_only = ON")
    except BaseException:
        conn.close()
        raise
    return conn


def _sql_error(path: Path, exc: sqlite3.Error, opts: LibraryOptions) -> LibraryError:
    text = str(exc)
    if isinstance(exc, sqlite3.DataError) or "too big" in text:
        return LibraryError(
            "too_large",
            f"{path}: a value is larger than the reader's per-value limit ({text})",
            "raise max_member_bytes if this really is one project",
        )
    if "interrupted" in text:
        return LibraryError(
            "too_large",
            f"{path}: reading it took longer than {opts.max_query_seconds:g} s",
            "raise max_query_seconds if this really is one project",
        )
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


def _load(
    conn: sqlite3.Connection, path: Path, opts: LibraryOptions
) -> dict[str, list[dict[str, Any]]]:
    rows = 0
    for table in COLUMNS:
        (count,) = conn.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()
        rows += int(count)
    if rows > opts.max_rows:
        raise LibraryError(
            "too_large",
            f"{path} has {rows} rows in the tables read, over the {opts.max_rows} limit",
            "raise max_rows if this really is one project",
        )
    tables: dict[str, list[dict[str, Any]]] = {}
    for table, columns in COLUMNS.items():
        sql = f"SELECT {', '.join(_q(c) for c in columns)} FROM {_q(table)}"
        tables[table] = [dict(zip(columns, row, strict=True)) for row in conn.execute(sql)]
    return tables


def _check_project(path: Path, tables: dict[str, list[dict[str, Any]]]) -> int:
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


def _content(column: str, value: object) -> bytes | None:
    """A column value as the XML text the reader expects: blobs as hex, the rest as text."""
    if value is None:
        return None
    if column in _BLOB_COLUMNS and isinstance(value, bytes):
        return value.hex().encode("ascii")
    if isinstance(value, bytes):
        return value
    if isinstance(value, float):
        return repr(value).encode("ascii")  # "86400.0" is not a frame count: read as an issue
    return str(value).encode("utf-8", "surrogateescape")


def _escape(data: bytes, *, attribute: bool = False) -> bytes:
    data = data.replace(b"&", b"&amp;").replace(b"<", b"&lt;").replace(b">", b"&gt;")
    return data.replace(b'"', b"&quot;") if attribute else data


def _member_part(name: str | None) -> str:
    """A folder or id as one member path part (no separators, bounded)."""
    cleaned = re.sub(r"[/\\\x00-\x1f]", "_", name or "")[:200].strip()
    return cleaned or "_"


class _Budget:
    def __init__(self, limit: int, path: Path) -> None:
        self.left = limit
        self.limit = limit
        self.path = path

    def charge(self, n: int) -> None:
        self.left -= n
        if self.left < 0:
            raise LibraryError(
                "too_large",
                f"{self.path}: the rows read come to more than the {self.limit}-byte XML budget",
                "raise max_total_xml_bytes if this really is one project",
            )


class _Doc:
    """One synthesized member, with the row and column behind every element it holds."""

    def __init__(self, name: str, budget: _Budget) -> None:
        self.name = name
        self.budget = budget
        self.parts: list[bytes] = []
        self.pos = 0
        self.stack: list[tuple[bytes, int, str | None]] = []
        self.spans: list[tuple[int, int, str]] = []
        self._write(b'<?xml version="1.0" encoding="UTF-8"?>\n')

    def _write(self, data: bytes) -> None:
        self.budget.charge(len(data))
        self.parts.append(data)
        self.pos += len(data)

    def open(self, tag: str, ref: str | None = None, dbid: str | None = None) -> None:
        start = self.pos
        attr = b"" if dbid is None else b' DbId="' + _escape(_raw(dbid), attribute=True) + b'"'
        self._write(b"<" + tag.encode() + attr + b">")
        self.stack.append((tag.encode(), start, ref))

    def close(self) -> None:
        tag, start, ref = self.stack.pop()
        self._write(b"</" + tag + b">")
        if ref is not None:
            self.spans.append((start, self.pos, ref))

    def leaf(self, tag: str, content: bytes | None, ref: str) -> None:
        start = self.pos
        name = tag.encode()
        if content:
            self._write(b"<" + name + b">" + _escape(content) + b"</" + name + b">")
        else:
            self._write(b"<" + name + b"/>")
        self.spans.append((start, self.pos, ref))

    def data(self) -> bytes:
        return b"".join(self.parts)


@dataclass(slots=True)
class _Link:
    owner: str
    child: str | None
    prop: str
    index: object
    ref: str


class _Builder:
    """Lay the rows out as the members of a ``.drp``, and report what does not fit."""

    def __init__(
        self, path: Path, tables: dict[str, list[dict[str, Any]]], budget: _Budget
    ) -> None:
        self.path = path
        self.budget = budget
        self.issues: list[Issue] = []
        self.counts: Counter[str] = Counter()
        self.docs: list[_Doc] = []
        self.member_rows: dict[str, str] = {}
        self.rows: dict[str, dict[str, dict[str, Any]]] = {}
        self.joins: dict[str, dict[str, dict[str, list[_Link]]]] = {}
        for table, rows in tables.items():
            if COLUMNS[table] == _JOIN:
                self.joins[table] = self._links(table, rows)
            else:
                self.rows[table] = self._index(table, rows)
        self.placed: dict[str, set[str]] = defaultdict(set)

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

    def _index(self, table: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        id_column = f"{table}_id"
        for row in rows:
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

    def _links(self, table: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, list[_Link]]]:
        out: dict[str, dict[str, list[_Link]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
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

    def _props(self, join: str, owner: str, first: Iterable[str]) -> list[tuple[str, list[_Link]]]:
        props = self.joins[join].get(owner, {})
        order = [p for p in first if p in props] + sorted(p for p in props if p not in first)
        return [(p, props[p]) for p in order]

    def _child(self, table: str, link: _Link, join: str) -> dict[str, Any] | None:
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
        taken = {d.name for d in self.docs}
        unique = name
        n = 1
        while unique in taken:
            n += 1
            stem, dot, ext = name.rpartition(".")
            unique = f"{stem}~{n}{dot}{ext}"
        doc = _Doc(unique, self.budget)
        self.docs.append(doc)
        self.member_rows[unique] = row_ref
        return doc

    def leaves(
        self, doc: _Doc, table: str, rid: str, row: dict[str, Any], cols: Iterable[str]
    ) -> None:
        for col in cols:
            doc.leaf(col, _content(col, row[col]), f"{table}/{rid}/{col}")

    # ------------------------------------------------------------------ members

    def build(self) -> None:
        self.project()
        self.folders()
        self.containers()
        self.unlinked("Sm2TiItem", "Sm2TiTrack_id")
        self.unlinked("Sm2TiTrack", "Sm2SequenceContainer_id")
        self.unlinked("Sm2MpMedia", "Sm2MpFolder_id")
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
        for link in self.joins["SM_Project_Sm2Timeline"].get(pid, {}).get("TimelineHandleVec", []):
            doc.leaf("Element", _raw(link.child) if link.child else None, link.ref)
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

    def _bin_member(self, fid: str) -> str:
        folders = self.rows["Sm2MpFolder"]
        names: list[str] = []
        seen: set[str] = set()
        node: str | None = fid
        while (
            node is not None and node in folders and node not in seen and len(seen) < _MAX_BIN_DEPTH
        ):
            seen.add(node)
            names.append(_member_part(_key(folders[node]["Name"])))
            node = _key(folders[node]["MpFolder"])
        return "MediaPool/" + "/".join(reversed(names)) + "/MpFolder.xml"

    def folders(self) -> None:
        for fid, row in self.rows["Sm2MpFolder"].items():
            ref = f"Sm2MpFolder/{fid}"
            doc = self._doc(self._bin_member(fid), ref)
            doc.open("Sm2MpFolder", ref, fid)
            self.leaves(doc, "Sm2MpFolder", fid, row, ("Name", "MpFolder"))
            doc.open("MediaVec")
            for link in self.joins["Sm2MpFolder_Sm2MpMedia"].get(fid, {}).get("MediaVec", []):
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

    def media(self, doc: _Doc, mid: str, row: dict[str, Any]) -> None:
        ref = f"Sm2MpMedia/{mid}"
        tag = self._type("Sm2MpMedia", mid, row["DbType"])
        if tag is None:
            return
        doc.open(tag, ref, mid)
        self.leaves(doc, "Sm2MpMedia", mid, row, ("Name", "MpFolder", "UniqueMediaPoolItemId"))
        info = _key(row["Video"])
        if info is not None:
            self._info(doc, "Video", info, f"{ref}/Video")
        for prop, links in self._props("BtAudioInfo_Sm2MpMedia", mid, ("EmbeddedAudioVec",)):
            if not _NAME.match(prop):
                self.issue(
                    "library-bad-type",
                    f"BtAudioInfo_Sm2MpMedia property {prop!r} is not an element name",
                    links[0].ref,
                )
                continue
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
            ref = f"Sm2SequenceContainer/{cid}"
            doc = self._doc(f"SeqContainer/{_member_part(cid)}.xml", ref)
            doc.open("Sm2SequenceContainer", ref, cid)
            for prop, links in self._props("Sm2SequenceContainer_Sm2TiTrack", cid, _TRACK_VECTORS):
                if not _NAME.match(prop):
                    self.issue(
                        "library-bad-type",
                        f"track vector {prop!r} is not an element name",
                        links[0].ref,
                    )
                    continue
                doc.open(prop)
                for link in links:
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

    def track(self, doc: _Doc, tid: str, row: dict[str, Any]) -> None:
        tag = self._type("Sm2TiTrack", tid, row["DbType"], "Sm2TiTrack")
        if tag is None:
            return
        doc.open(tag, f"Sm2TiTrack/{tid}", tid)
        self.leaves(
            doc, "Sm2TiTrack", tid, row, ("Type", "SubType", "Flags", "Sequence", "UserDefinedName")
        )
        for prop, links in self._props("Sm2TiItem_Sm2TiTrack", tid, ("Items",)):
            if not _NAME.match(prop):
                self.issue(
                    "library-bad-type",
                    f"track item vector {prop!r} is not an element name",
                    links[0].ref,
                )
                continue
            doc.open(prop)
            for link in links:
                item = self._child("Sm2TiItem", link, "Sm2TiItem_Sm2TiTrack")
                if item is None or link.child is None:
                    continue
                if not self._place("Sm2TiItem", link.child, link, "Sm2TiTrack_id"):
                    continue
                self.item(doc, link.child, item)
            doc.close()
        doc.close()

    def item(self, doc: _Doc, iid: str, row: dict[str, Any]) -> None:
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
        missed = [
            rid
            for rid, row in self.rows[table].items()
            if rid not in self.placed[table] and _key(row[owner_column]) is not None
        ]
        if missed:
            self.issue(
                "library-unlinked-row",
                f"{len(missed)} {table} row(s) name an owner in {owner_column} but no link "
                "places them, so they are not read",
                f"{table}/{missed[0]}",
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
    spans: dict[str, list[tuple[int, int, str]]]
    _data: dict[str, bytes] = field(repr=False)
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
    """Snapshot, check and lay out one ``Project.db``; raises :class:`LibraryError` if refused."""
    p = Path(path)
    opts = options or LibraryOptions()
    snapshot, copied = _take_snapshot(p, opts)
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
            conn = _connect(snapshot, opts, max_member_bytes)
        except sqlite3.Error as exc:
            raise _sql_error(p, exc, opts) from None
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
                raise _sql_error(p, exc, opts) from None
        finally:
            conn.close()
    finally:
        snapshot.unlink(missing_ok=True)
    version = _check_project(p, tables)
    builder = _Builder(p, tables, _Budget(max_total_xml_bytes, p))
    builder.build()
    [pid] = builder.rows["SM_Project"]
    data = {doc.name: doc.data() for doc in builder.docs}
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
        spans={doc.name: sorted(doc.spans) for doc in builder.docs},
        _data=data,
    )


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
            start, end, ref = spans[i]
            if start <= offset < end:
                return ref
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
    """One project folder of a library, read or refused."""

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


@dataclass(slots=True)
class LibraryListing:
    """Every ``Project.db`` found under a library's ``Users/<user>/Projects`` folders."""

    root: str
    projects: list[LibraryProject]
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "truncated": self.truncated,
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


def _project_files(projects_dir: Path, budget: list[int]) -> Iterator[Path]:
    """Folders holding a ``Project.db`` under ``projects_dir`` (no links followed, bounded)."""
    stack: list[tuple[Path, int]] = [(projects_dir, 0)]
    while stack:
        folder, depth = stack.pop()
        try:
            with os.scandir(folder) as entries:
                listed = []
                for entry in entries:
                    budget[0] -= 1
                    if budget[0] < 0:
                        return
                    listed.append(entry)
        except OSError:
            continue
        names = {e.name for e in listed}
        if "Project.db" in names and folder != projects_dir:
            yield folder / "Project.db"
            continue
        if depth >= _MAX_FOLDER_DEPTH:
            continue
        for entry in sorted(listed, key=lambda e: e.name, reverse=True):
            if entry.is_dir(follow_symlinks=False):
                stack.append((Path(entry.path), depth + 1))


def list_library(
    root: str | Path,
    *,
    options: LibraryOptions | None = None,
    read: Callable[[Path], DrpFile] | None = None,
) -> LibraryListing:
    """List every project of the library at ``root`` (or its ``Resolve Projects`` folder).

    Each project is read in full (``read``, by default :func:`aar.drp.read_drp`); a project
    that is refused (busy, damaged, ...) is listed with its refusal, never skipped. At most 500
    projects and 20,000 folder entries are looked at (``truncated`` says when that cut the list
    short), at most 8 folders deep, and no link is followed.
    """
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

        def read(p: Path) -> DrpFile:
            return read_drp(p, library_options=options)

    budget = [_MAX_SCANNED_ENTRIES]
    found: list[tuple[str, Path, Path]] = []
    truncated = False
    try:
        user_dirs = sorted(
            (e for e in os.scandir(users) if e.is_dir(follow_symlinks=False)), key=lambda e: e.name
        )
    except OSError as exc:
        raise LibraryError(
            "unreadable", f"cannot list {users}: {exc}", "check the path and permissions"
        ) from None
    for user in user_dirs:
        projects_dir = Path(user.path) / "Projects"
        for db in _project_files(projects_dir, budget):
            if len(found) >= _MAX_LISTED_PROJECTS:
                truncated = True
                break
            found.append((user.name, projects_dir, db))
        if budget[0] < 0:
            truncated = True
    listing = LibraryListing(str(root), [], truncated)
    for user_name, projects_dir, db in found:
        try:
            mtime: float | None = db.stat().st_mtime
        except OSError:
            mtime = None
        entry = LibraryProject(
            user=user_name,
            folder=db.parent.relative_to(projects_dir).as_posix(),
            path=str(db),
            file_modified=_iso(mtime),
        )
        try:
            drp = read(db)
        except LibraryError as exc:
            entry.refusal = exc.to_dict()
        except ContainerError as exc:
            entry.refusal = {"code": "unreadable", "message": str(exc), "hint": "check the file"}
        else:
            p = drp.project
            if p is not None:
                entry.name, entry.id = p.name, p.id
                entry.last_modified_secs = p.last_modified_secs
                entry.last_modified = _iso(p.last_modified_secs)
            entry.project_version = drp.versions.project_version if drp.versions else None
            entry.timelines = [t.name or "" for t in drp.timelines if t.kind == "timeline"]
            entry.warnings = sorted(
                {i.code for i in drp.issues if i.severity in ("warning", "error")}
            )
        listing.projects.append(entry)
    return listing
