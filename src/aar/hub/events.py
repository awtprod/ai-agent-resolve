"""Append-only SQLite event log: the hub's single source of truth.

Everything the hub knows is derived from this log. The project registry, the work
graph and the artifact store are projections over these events; nothing mutable
lives only in memory. The table is append-only, enforced in the database by
triggers that abort any ``UPDATE``, ``DELETE`` or row-replacing ``INSERT``, so
history cannot be rewritten even by a buggy or hostile caller. ``seq`` gives a
monotonic total order; ``id`` is a time-ordered unique id (a monotonic ULID) that
sorts the same way.

Projections that need to validate against current state (the work graph's status
machine, unique-id checks) use :meth:`EventLog.transaction` so the catch-up read,
the validation and the insert happen inside one ``BEGIN IMMEDIATE`` transaction and
cannot interleave with another writer.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path, PurePath
from typing import Any

ACTOR_KINDS = frozenset({"human", "agent", "node"})

# Crockford base32, used to render a 128-bit ULID as 26 sortable characters.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RAND_BITS = 80
_RAND_MASK = (1 << _RAND_BITS) - 1

_REQUIRED_TRIGGERS = frozenset({"events_no_update", "events_no_delete", "events_no_replace"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    project_id      TEXT,
    type            TEXT NOT NULL,
    actor_kind      TEXT NOT NULL,
    actor_id        TEXT NOT NULL,
    ts              TEXT NOT NULL,
    payload         TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    correlation_id  TEXT,
    causation_id    TEXT
);
CREATE INDEX IF NOT EXISTS events_project_seq ON events(project_id, seq);
CREATE INDEX IF NOT EXISTS events_type_seq ON events(type, seq);

CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only: UPDATE is forbidden');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only: DELETE is forbidden');
END;

-- Block INSERT OR REPLACE (and any insert reusing an id, seq or idempotency key)
-- from rewriting a row. This fires on any connection, regardless of the
-- recursive_triggers setting, so a raw REPLACE cannot delete the conflicting row.
CREATE TRIGGER IF NOT EXISTS events_no_replace
BEFORE INSERT ON events
WHEN EXISTS (
    SELECT 1 FROM events
    WHERE id = NEW.id
       OR seq = NEW.seq
       OR (NEW.idempotency_key IS NOT NULL AND idempotency_key = NEW.idempotency_key)
)
BEGIN
    SELECT RAISE(ABORT, 'events are append-only: INSERT must not reuse an id, seq or key');
END;
"""

_INSERT_SQL = (
    "INSERT INTO events "
    "(id, project_id, type, actor_kind, actor_id, ts, payload, "
    " idempotency_key, correlation_id, causation_id) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_ulid_lock = threading.Lock()
_ulid_last_ms = -1
_ulid_last_rand = 0


class IdempotencyConflict(ValueError):
    """Raised when an idempotency key is reused for a different event."""


def _canonical(value: Any) -> str:
    # allow_nan=False: NaN/Infinity cannot round-trip through JSON, so an event that
    # carried one could never compare equal to its own reload. Reject it at the point
    # of serialization (append) rather than store a value that breaks idempotency.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json(value: Any) -> str:
    """The log's canonical JSON text of ``value``: sorted keys, no spaces, no NaN/Infinity.

    The same form :meth:`EventLog.append` stores, so a hash of it (a content-addressed
    plan, a write request's fingerprint) is stable across processes and retries.
    """
    return _canonical(value)


def _canonical_lenient(value: Any) -> str:
    # Comparison-only canonical form for the STORED side: tolerates a legacy stored
    # NaN/inf (allow_nan=True) so matching never raises on old data. Still distinguishes
    # True/1 and 1/1.0, because json.dumps renders them as "true"/"1"/"1.0" either way.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _sqlite_uri(path: str, cwd: str, flavour: type[PurePath] = Path) -> str:
    r"""Build a ``file:`` URI (``mode=rw``) for an existing SQLite database.

    The path is made absolute WITHOUT normalising it: ``..`` and symlinks must be
    resolved by the OS on open, never collapsed as text (collapsing can reach a decoy
    through a symlink). ``flavour`` is the pathlib flavour -- the real ``Path`` at
    runtime, ``PureWindowsPath``/``PurePosixPath`` in tests. On Windows a root-relative
    path (``\data\x.db`` or ``/data/x.db``) is not absolute, so it is anchored to the
    drive of ``cwd``; a drive-relative path (``C:foo``) that still cannot be resolved,
    and a UNC path (``\\host\share\...``) whose authority SQLite rejects, both raise
    ``ValueError``. as_uri() keeps a leading ``//``, ``%`` and non-UTF-8 names intact.
    """
    pure = flavour(path)
    if not pure.is_absolute():
        pure = flavour(cwd) / pure
    if not pure.is_absolute():
        raise ValueError(f"cannot resolve a drive-relative path to an absolute file: {path!r}")
    if pure.drive.startswith("\\\\"):
        raise ValueError(f"UNC paths are not supported; map a network drive: {path!r}")
    return pure.as_uri() + "?mode=rw"


def match_or_conflict(
    existing: Event,
    event_type: str,
    expected: dict[str, Any],
    compare_keys: Iterable[str],
    *,
    project_id: str | None = None,
    actor: Actor | None = None,
) -> None:
    """Raise :class:`IdempotencyConflict` unless ``existing`` is the same request.

    Projections short-circuit an idempotent retry on the key alone, so they must
    confirm the stored event is the *same* request before returning the stored entity.

    Fields are compared by canonical JSON, not Python ``==``: ``existing.payload`` has
    been through JSON (tuples became lists, dict keys became strings) while ``expected``
    is the live request, so a raw ``==`` both rejects genuine retries (tuple vs list)
    and accepts distinct requests (``True`` vs ``1``, ``1`` vs ``1.0``). The stored side
    is canonicalised leniently so a legacy NaN/inf payload yields a conflict rather than
    a ValueError; the expected (request) side is canonicalised strictly, so a NaN/inf in
    the request raises ValueError, consistent with :meth:`EventLog.append`. When
    ``project_id``/``actor`` are given the stored event must also match them (a key
    reused in another project's scope, or by a different actor, conflicts); ``ts`` is
    deliberately not compared. The ``event_type`` check keeps the return path (indexing
    by the stored ``node_id``/``edge_id``/``project_id``) type-correct.
    """
    # A stored payload that is not an object cannot be the same request (and must not
    # raise AttributeError on the retry path): refuse it as a conflict.
    same = existing.type == event_type and isinstance(existing.payload, dict)
    if same and project_id is not None:
        same = existing.project_id == project_id
    if same and actor is not None:
        same = existing.actor == actor
    if same:
        same = all(
            _canonical_lenient(existing.payload.get(key)) == _canonical(expected.get(key))
            for key in compare_keys
        )
    if not same:
        raise IdempotencyConflict(
            f"idempotency key {existing.idempotency_key!r} was already used for a different request"
        )


def new_event_id(ts_ms: int | None = None) -> str:
    """Return a 26-char, time-ordered, unique id (ULID: 48-bit ms time + 80-bit random).

    Ids are monotonic *within one process*: inside a millisecond the random component
    is incremented rather than redrawn, so ids minted back to back by this process sort
    in creation order. Across processes or a clock change they are not globally ordered;
    ``seq`` is the authoritative total order of the log.
    """
    global _ulid_last_ms, _ulid_last_rand
    with _ulid_lock:
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        if ts_ms == _ulid_last_ms:
            _ulid_last_rand = (_ulid_last_rand + 1) & _RAND_MASK
        else:
            _ulid_last_ms = ts_ms
            _ulid_last_rand = secrets.randbits(_RAND_BITS)
        value = (ts_ms << _RAND_BITS) | _ulid_last_rand
    chars = [""] * 26
    for i in range(25, -1, -1):
        chars[i] = _CROCKFORD[value & 0x1F]
        value >>= 5
    return "".join(chars)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Actor:
    """Who caused an event: a human, an AI agent, or a node beside a Resolve instance."""

    kind: str
    id: str

    def __post_init__(self) -> None:
        if self.kind not in ACTOR_KINDS:
            raise ValueError(f"actor kind must be one of {sorted(ACTOR_KINDS)}, got {self.kind!r}")
        if not self.id:
            raise ValueError("actor id must be non-empty")


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable fact in the log.

    ``seq`` is the monotonic insertion order, ``id`` is a time-ordered unique id,
    and ``project_id`` is ``None`` for events that are global rather than about one
    project.
    """

    seq: int
    id: str
    project_id: str | None
    type: str
    actor: Actor
    ts: str
    payload: dict[str, Any]
    idempotency_key: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        seq=row["seq"],
        id=row["id"],
        project_id=row["project_id"],
        type=row["type"],
        actor=Actor(kind=row["actor_kind"], id=row["actor_id"]),
        ts=row["ts"],
        payload=json.loads(row["payload"]),
        idempotency_key=row["idempotency_key"],
        correlation_id=row["correlation_id"],
        causation_id=row["causation_id"],
    )


def _select_rows(
    conn: sqlite3.Connection,
    since_seq: int,
    project_id: str | None,
    types: list[str] | None,
    limit: int,
) -> list[sqlite3.Row]:
    clauses = ["seq > ?"]
    params: list[Any] = [since_seq]
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if types is not None:
        if not types:
            return []
        placeholders = ",".join("?" for _ in types)
        clauses.append(f"type IN ({placeholders})")
        params.extend(types)
    params.append(limit)
    where = " AND ".join(clauses)
    sql = f"SELECT * FROM events WHERE {where} ORDER BY seq ASC LIMIT ?"
    return conn.execute(sql, params).fetchall()


class Transaction:
    """A handle to the open ``BEGIN IMMEDIATE`` transaction of an :class:`EventLog`.

    Reads and the insert done through here run inside the one write transaction, so a
    projection can catch up, validate and append atomically.
    """

    def __init__(self, log: EventLog) -> None:
        self._log = log
        self._conn = log._conn
        self._active = True

    def _deactivate(self) -> None:
        self._active = False

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("transaction is closed; it may only be used inside its `with` block")

    def read(
        self,
        since_seq: int = 0,
        *,
        project_id: str | None = None,
        types: Iterable[str] | None = None,
    ) -> list[Event]:
        """Return every event with ``seq > since_seq`` matching the filters, in order."""
        self._ensure_active()
        type_list = None if types is None else list(types)
        rows = _select_rows(self._conn, since_seq, project_id, type_list, limit=-1)
        return [_row_to_event(row) for row in rows]

    def find_by_idempotency_key(self, idempotency_key: str) -> Event | None:
        """Return the event stored under ``idempotency_key``, or ``None``."""
        self._ensure_active()
        row = self._conn.execute(
            "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return _row_to_event(row) if row is not None else None

    def append(
        self,
        event_type: str,
        actor: Actor,
        payload: dict[str, Any] | None = None,
        *,
        project_id: str | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        ts: str | None = None,
    ) -> Event:
        """Insert an event inside the open transaction and return it."""
        self._ensure_active()
        payload = payload or {}
        payload_json = _canonical(payload)
        if idempotency_key is not None:
            existing = self._log._lookup_idempotent(
                idempotency_key, event_type, project_id, payload_json, actor
            )
            if existing is not None:
                return existing
        ev_id = new_event_id()
        row = self._log._insert(
            ev_id,
            project_id,
            event_type,
            actor,
            ts or _utcnow_iso(),
            payload_json,
            idempotency_key,
            correlation_id,
            causation_id,
        )
        return _row_to_event(row)


class EventLog:
    """An append-only event log backed by a single SQLite database (WAL mode)."""

    def __init__(
        self,
        db_path: str | PathLike[str],
        *,
        create: bool = True,
        read_only: bool = False,
    ) -> None:
        self._path = str(db_path)
        self._read_only = read_only
        self._lock = threading.RLock()
        self._conn = self._open(read_only)
        try:
            self._conn.row_factory = sqlite3.Row
            # Manage transactions explicitly (BEGIN IMMEDIATE) rather than via the
            # sqlite3 module's implicit transaction handling. These settings and the
            # schema read below do not write to the database file.
            self._conn.isolation_level = None
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA recursive_triggers=ON")
            if read_only:
                self._conn.execute("PRAGMA query_only=ON")
                self._assert_schema()
            elif create:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(_SCHEMA)
            else:
                # Validate the schema BEFORE switching journal mode, so opening a
                # foreign SQLite file leaves it byte-identical (no WAL switch, no
                # -wal/-shm side files).
                self._assert_schema()
                self._conn.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            # Close a failed open deterministically, not at GC time.
            self._conn.close()
            raise

    def _open(self, read_only: bool) -> sqlite3.Connection:
        if not read_only:
            return sqlite3.connect(self._path, check_same_thread=False)
        if self._path == ":memory:":
            raise ValueError("a read-only event log needs a file-backed database, not ':memory:'")
        if "\x00" in self._path:
            # A NUL is truncated at %00 and would silently open a different file.
            raise ValueError("event log path must not contain a NUL byte")
        # Build the URI via the pure helper (non-normalising; Windows/POSIX-aware).
        # mode=rw opens an existing database but never creates one, so a missing or
        # unopenable path fails atomically (no exists()/connect() race, no stray 0-byte
        # file). query_only=ON (set by the caller) enforces read-only; rw (not ro) lets
        # a WAL database with a pending -wal still create its -shm and be read in any
        # state. Closing such a read-only handle may trigger a WAL checkpoint on the file.
        uri = _sqlite_uri(self._path, os.getcwd())
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.execute("PRAGMA schema_version")  # force the open to happen now
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            if not Path(self._path).exists():
                raise FileNotFoundError(f"no event log database at {self._path!r}") from exc
            raise OSError(f"cannot open event log database at {self._path!r}: {exc}") from exc
        return conn

    def _assert_schema(self) -> None:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
        present = {row["name"] for row in rows}
        missing = _REQUIRED_TRIGGERS - present
        if missing:
            raise RuntimeError(
                f"event log at {self._path!r} is missing append-only triggers: {sorted(missing)}"
            )

    @property
    def read_only(self) -> bool:
        """Whether this handle was opened read-only (appends and transactions refuse)."""
        return self._read_only

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()

    # -- internals shared by append() and Transaction.append() -------------

    def _lookup_idempotent(
        self,
        idempotency_key: str,
        event_type: str,
        project_id: str | None,
        payload_json: str,
        actor: Actor,
    ) -> Event | None:
        """Return the stored event for a key if it is the same request; raise on a mismatch.

        The stored payload is compared as text -- it is already canonical JSON on disk --
        and never re-``json.dumps``'d, so a legacy row whose payload contains NaN/inf does
        not raise a bare ValueError here on a retry of its key. ``actor`` is part of the
        request identity (a different actor reusing the key is a conflict); ``ts`` is not.
        """
        row = self._conn.execute(
            "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if row is None:
            return None
        if (
            row["type"] == event_type
            and row["project_id"] == project_id
            and row["actor_kind"] == actor.kind
            and row["actor_id"] == actor.id
            and row["payload"] == payload_json
        ):
            return _row_to_event(row)
        raise IdempotencyConflict(
            f"idempotency key {idempotency_key!r} was already used for a different event"
        )

    def _insert(
        self,
        ev_id: str,
        project_id: str | None,
        event_type: str,
        actor: Actor,
        ts: str,
        payload_json: str,
        idempotency_key: str | None,
        correlation_id: str | None,
        causation_id: str | None,
    ) -> sqlite3.Row:
        self._conn.execute(
            _INSERT_SQL,
            (
                ev_id,
                project_id,
                event_type,
                actor.kind,
                actor.id,
                ts,
                payload_json,
                idempotency_key,
                correlation_id,
                causation_id,
            ),
        )
        return self._conn.execute("SELECT * FROM events WHERE id = ?", (ev_id,)).fetchone()

    # -- transactions -------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """Open a ``BEGIN IMMEDIATE`` transaction, yielding a :class:`Transaction`."""
        if self._read_only:
            raise RuntimeError("event log is read-only")
        with self._lock:
            txn = Transaction(self)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield txn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
            finally:
                # A handle kept past the block must not append in autocommit.
                txn._deactivate()

    # -- public append / read ----------------------------------------------

    def append(
        self,
        event_type: str,
        actor: Actor,
        payload: dict[str, Any] | None = None,
        *,
        project_id: str | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        ts: str | None = None,
    ) -> Event:
        """Append an event and return it.

        With an ``idempotency_key``: an identical earlier event is returned unchanged;
        reusing the key for a *different* event raises :class:`IdempotencyConflict`.
        """
        if self._read_only:
            raise RuntimeError("event log is read-only")
        payload = payload or {}
        payload_json = _canonical(payload)
        with self._lock:
            if idempotency_key is not None:
                existing = self._lookup_idempotent(
                    idempotency_key, event_type, project_id, payload_json, actor
                )
                if existing is not None:
                    return existing
            ev_id = new_event_id()
            try:
                row = self._insert(
                    ev_id,
                    project_id,
                    event_type,
                    actor,
                    ts or _utcnow_iso(),
                    payload_json,
                    idempotency_key,
                    correlation_id,
                    causation_id,
                )
            except sqlite3.IntegrityError:
                # Lost an idempotency-key race against a concurrent writer: the row we
                # meant to write already exists. Run the same match check on the winner
                # (type, project, payload) so a *different* request raises a conflict
                # instead of silently receiving someone else's event.
                if idempotency_key is not None:
                    existing = self._lookup_idempotent(
                        idempotency_key, event_type, project_id, payload_json, actor
                    )
                    if existing is not None:
                        return existing
                raise
        return _row_to_event(row)

    def read(
        self,
        since_seq: int = 0,
        *,
        project_id: str | None = None,
        types: Iterable[str] | None = None,
        limit: int = 1000,
    ) -> list[Event]:
        """Return events with ``seq > since_seq`` in ascending ``seq`` order.

        ``project_id=None`` applies no project filter (global and per-project events
        are both returned). ``types`` restricts to the given event types.
        """
        type_list = None if types is None else list(types)
        with self._lock:
            rows = _select_rows(self._conn, since_seq, project_id, type_list, limit)
        return [_row_to_event(row) for row in rows]

    def iter_events(
        self,
        *,
        project_id: str | None = None,
        types: Iterable[str] | None = None,
        batch_size: int = 1000,
    ) -> Iterator[Event]:
        """Yield every matching event in ``seq`` order, paging so the log need not fit in memory."""
        type_list = None if types is None else list(types)
        since = 0
        while True:
            batch = self.read(since, project_id=project_id, types=type_list, limit=batch_size)
            if not batch:
                return
            yield from batch
            since = batch[-1].seq
            if len(batch) < batch_size:
                return

    def get_by_idempotency_key(self, idempotency_key: str) -> Event | None:
        """Return the event stored under ``idempotency_key``, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return _row_to_event(row) if row is not None else None

    def latest_seq(self) -> int:
        """Return the highest ``seq`` in the log, or ``0`` when it is empty."""
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM events").fetchone()
        return int(row["m"])
