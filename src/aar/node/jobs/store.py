"""The node-side job store: a single SQLite database that survives a process restart.

The store is the source of truth for job state. The runner holds no durable job state, so a
fresh runner opened on the same file resumes every job from its last checkpoint. A checkpoint
is one transaction: the ``jobs`` row (state, step, cursor and the remaining step deadline) is
upserted and any transition events are appended together.

Safety properties:

- **Single writer** (M6): opening a store takes an exclusive OS lock on a side file, so two
  runners can never drive the same jobs and double-execute them. ``load_all`` is further
  scoped to one ``node_id``.
- **Schema version** (L4): the version is stored and checked on open; an unknown version is
  refused rather than silently mis-read. Version 1 and 2 stores are migrated in place to 3
  (v2 added the restart counter; v3 the job's agent id and its consumed approvals).
- **Poison rows** (M5/P12): a row that cannot be parsed (bad state, bad JSON, bad target) is
  loaded as a read-only *quarantined* record -- failed, with the parse error -- and is never
  written back, so one bad row cannot brick the runner and the original data is kept.
- ``job_events`` is append-only, enforced by triggers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

from aar.node.locator import ProjectLocator

from .file_lock import FileLockUnavailable, acquire_file_lock, release_file_lock
from .model import JobState, StructuredError, WaitReason

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS jobs (
    job_id             TEXT PRIMARY KEY,
    kind               TEXT NOT NULL,
    node_id            TEXT NOT NULL,
    target             TEXT NOT NULL,
    params             TEXT NOT NULL,
    state              TEXT NOT NULL,
    current_step       INTEGER NOT NULL,
    cursor             TEXT NOT NULL,
    results            TEXT NOT NULL,
    error              TEXT,
    waiting            TEXT,
    stall              TEXT,
    approval_token     TEXT,
    approval_consumed  INTEGER NOT NULL DEFAULT 0,
    approval_pending   TEXT,
    approval_pending_key TEXT,
    approvals          TEXT NOT NULL DEFAULT '{}',
    agent_id           TEXT,
    step_deadline      REAL,
    priority           INTEGER NOT NULL DEFAULT 0,
    restarts           INTEGER NOT NULL DEFAULT 0,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_node_state ON jobs(node_id, state);

CREATE TABLE IF NOT EXISTS job_events (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL,
    at      REAL NOT NULL,
    kind    TEXT NOT NULL,
    data    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, seq);

CREATE TRIGGER IF NOT EXISTS job_events_no_update
BEFORE UPDATE ON job_events
BEGIN
    SELECT RAISE(ABORT, 'job events are append-only: UPDATE is forbidden');
END;

CREATE TRIGGER IF NOT EXISTS job_events_no_delete
BEFORE DELETE ON job_events
BEGIN
    SELECT RAISE(ABORT, 'job events are append-only: DELETE is forbidden');
END;
"""

_REQUIRED_TRIGGERS = frozenset({"job_events_no_update", "job_events_no_delete"})


class JobStoreLocked(RuntimeError):
    """Another process already holds the store's single-writer lock."""


class JobStoreVersionError(RuntimeError):
    """The store was written by an incompatible schema version."""


@dataclass
class JobRecord:
    """The persisted state of one job. Mutated by the runner, then saved."""

    job_id: str
    kind: str
    node_id: str
    target: ProjectLocator
    params: dict[str, Any]
    state: JobState = JobState.QUEUED
    current_step: int = 0
    cursor: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    error: StructuredError | None = None
    waiting: WaitReason | None = None
    stall: StructuredError | None = None
    approval_token: str | None = None
    approval_consumed: bool = False
    approval_pending: str | None = None
    """The token id a consume was started with but not yet confirmed (crash-safety, L2)."""
    approval_pending_key: str | None = None
    """What the pending consume is for: ``"project"`` (F15) or an op override key (P4)."""
    approvals: dict[str, str] = field(default_factory=dict)
    """Consumed approvals, approval key -> token id: ``"project"`` is the F15 production
    approval, ``"op:<step>/<key>/<op>"`` an override that lets one op past the branch guard.
    Verified against the hub on load; an entry the hub does not confirm is dropped."""
    agent_id: str | None = None
    """The agent the job works for: every op runs as ``Holder.agent(agent_id)`` (P2)."""
    step_deadline: float | None = None
    """Seconds the current step may still run; counts down only while the job is running."""
    priority: int = 0
    restarts: int = 0
    """How many node restarts interrupted the current step while it ran (crash-loop guard)."""
    created_at: float = 0.0
    updated_at: float = 0.0
    readonly: bool = False
    """A quarantined row loaded from a malformed record: never written back (not persisted)."""


@dataclass(frozen=True, slots=True)
class JobEvent:
    """One immutable fact in a job's history."""

    seq: int
    job_id: str
    at: float
    kind: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "at": self.at, "kind": self.kind, "data": self.data}


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record_from_row(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        kind=row["kind"],
        node_id=row["node_id"],
        target=ProjectLocator.from_dict(json.loads(row["target"])),
        params=json.loads(row["params"]),
        state=JobState(row["state"]),
        current_step=row["current_step"],
        cursor=json.loads(row["cursor"]),
        results=json.loads(row["results"]),
        error=StructuredError.from_dict(json.loads(row["error"])) if row["error"] else None,
        waiting=WaitReason.from_dict(json.loads(row["waiting"])) if row["waiting"] else None,
        stall=StructuredError.from_dict(json.loads(row["stall"])) if row["stall"] else None,
        approval_token=row["approval_token"],
        approval_consumed=bool(row["approval_consumed"]),
        approval_pending=row["approval_pending"],
        approval_pending_key=row["approval_pending_key"],
        approvals=_approvals(json.loads(row["approvals"])),
        agent_id=row["agent_id"],
        step_deadline=row["step_deadline"],
        priority=row["priority"],
        restarts=row["restarts"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _approvals(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("approvals must be an object of approval key -> token id")
    return dict(value)


def _raw(row: sqlite3.Row, column: str) -> Any:
    try:
        return row[column]
    except (IndexError, KeyError):
        return None


def _quarantined_record(row: sqlite3.Row, exc: Exception) -> JobRecord:
    """A read-only, failed stand-in for a row that cannot be parsed (the row is left as is)."""
    created = _raw(row, "created_at")
    return JobRecord(
        job_id=str(_raw(row, "job_id")),
        kind=str(_raw(row, "kind") or "?"),
        node_id=str(_raw(row, "node_id") or ""),
        target=ProjectLocator((), "<unreadable>"),
        params={},
        state=JobState.FAILED,
        error=StructuredError(
            "quarantined_bad_record",
            f"job record could not be loaded: {type(exc).__name__}: {exc}",
            "the row is malformed or from a newer version; inspect it, then remove or migrate it",
            detail={"stored_state": str(_raw(row, "state"))},
        ),
        created_at=float(created) if isinstance(created, int | float) else 0.0,
        readonly=True,
    )


def _parse_row(row: sqlite3.Row) -> JobRecord:
    try:
        return _record_from_row(row)
    except Exception as exc:  # one poison row must not brick the runner (P12)
        return _quarantined_record(row, exc)


_UPSERT = """
INSERT INTO jobs (
    job_id, kind, node_id, target, params, state, current_step, cursor, results,
    error, waiting, stall, approval_token, approval_consumed, approval_pending,
    approval_pending_key, approvals, agent_id,
    step_deadline, priority, restarts, created_at, updated_at
) VALUES (
    :job_id, :kind, :node_id, :target, :params, :state, :current_step, :cursor, :results,
    :error, :waiting, :stall, :approval_token, :approval_consumed, :approval_pending,
    :approval_pending_key, :approvals, :agent_id,
    :step_deadline, :priority, :restarts, :created_at, :updated_at
)
ON CONFLICT(job_id) DO UPDATE SET
    state=excluded.state, current_step=excluded.current_step, cursor=excluded.cursor,
    results=excluded.results, error=excluded.error, waiting=excluded.waiting,
    stall=excluded.stall, approval_token=excluded.approval_token,
    approval_consumed=excluded.approval_consumed, approval_pending=excluded.approval_pending,
    approval_pending_key=excluded.approval_pending_key, approvals=excluded.approvals,
    step_deadline=excluded.step_deadline, priority=excluded.priority,
    restarts=excluded.restarts, updated_at=excluded.updated_at
"""


class JobStore:
    """An SQLite-backed job store (WAL mode), single-writer, with checkpoint transactions."""

    def __init__(self, db_path: str | PathLike[str], *, create: bool = True) -> None:
        self._path = str(db_path)
        self._lock = threading.RLock()
        self._lock_fd = self._take_file_lock()
        try:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.isolation_level = None
            self._conn.execute("PRAGMA recursive_triggers=ON")
            if create:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(_SCHEMA)
                self._init_version()
            else:
                self._assert_schema()
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._check_version()
        except BaseException:
            self.close()
            raise

    def _take_file_lock(self) -> int:
        try:
            return acquire_file_lock(self._path + ".lock")
        except FileLockUnavailable as exc:
            raise JobStoreLocked(
                f"another process holds the job store lock for {self._path!r}"
            ) from exc

    def _init_version(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) in (1, 2):
                self._migrate()
            elif int(row["value"]) != SCHEMA_VERSION:
                raise JobStoreVersionError(
                    f"job store at {self._path!r} is schema version {row['value']}, "
                    f"not {SCHEMA_VERSION}"
                )
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def _migrate(self) -> None:
        """v1/v2 -> v3, inside the caller's transaction: v2 added the ``restarts`` counter, v3
        the agent id and the consumed approvals. A migrated job has no agent id, so it loads
        quarantined (it can never run as an agent it was not submitted for)."""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        added = {
            "restarts": "INTEGER NOT NULL DEFAULT 0",
            "approval_pending_key": "TEXT",
            "approvals": "TEXT NOT NULL DEFAULT '{}'",
            "agent_id": "TEXT",
        }
        for column, decl in added.items():
            if column not in columns:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")
        self._conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),)
        )

    def _check_version(self) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        version = int(row["value"]) if row is not None else 0
        if version != SCHEMA_VERSION:
            raise JobStoreVersionError(
                f"job store at {self._path!r} is schema version {version}, not {SCHEMA_VERSION}"
            )

    def _assert_schema(self) -> None:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
        missing = _REQUIRED_TRIGGERS - {row["name"] for row in rows}
        if missing:
            raise RuntimeError(
                f"job store at {self._path!r} is missing triggers: {sorted(missing)}"
            )

    def close(self) -> None:
        with self._lock:
            conn = getattr(self, "_conn", None)
            fd = getattr(self, "_lock_fd", None)
            self._conn = None
            self._lock_fd = None
            try:
                if conn is not None:
                    conn.close()
            finally:
                if fd is not None:
                    release_file_lock(fd)

    # -- writes ----------------------------------------------------------------------------

    def save(
        self, record: JobRecord, events: Iterable[tuple[str, dict[str, Any], float]] = ()
    ) -> None:
        """Upsert ``record`` and append ``events`` (``kind``, ``data``, ``at``) in one txn."""
        if record.readonly:
            raise ValueError(f"job {record.job_id} is a quarantined read-only record")
        params = {
            "job_id": record.job_id,
            "kind": record.kind,
            "node_id": record.node_id,
            "target": _dumps(record.target.to_dict()),
            "params": _dumps(record.params),
            "state": record.state.value,
            "current_step": record.current_step,
            "cursor": _dumps(record.cursor),
            "results": _dumps(record.results),
            "error": _dumps(record.error.to_dict()) if record.error is not None else None,
            "waiting": _dumps(record.waiting.to_dict()) if record.waiting is not None else None,
            "stall": _dumps(record.stall.to_dict()) if record.stall is not None else None,
            "approval_token": record.approval_token,
            "approval_consumed": 1 if record.approval_consumed else 0,
            "approval_pending": record.approval_pending,
            "approval_pending_key": record.approval_pending_key,
            "approvals": _dumps(record.approvals),
            "agent_id": record.agent_id,
            "step_deadline": record.step_deadline,
            "priority": record.priority,
            "restarts": record.restarts,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        rows = [(record.job_id, at, kind, _dumps(data)) for kind, data, at in events]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(_UPSERT, params)
                for job_id, at, kind, data in rows:
                    self._conn.execute(
                        "INSERT INTO job_events (job_id, at, kind, data) VALUES (?, ?, ?, ?)",
                        (job_id, at, kind, data),
                    )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def append_events(
        self, job_id: str, events: Iterable[tuple[str, dict[str, Any], float]]
    ) -> None:
        """Append ``events`` for ``job_id`` without touching its row (quarantined records)."""
        rows = [(job_id, at, kind, _dumps(data)) for kind, data, at in events]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    self._conn.execute(
                        "INSERT INTO job_events (job_id, at, kind, data) VALUES (?, ?, ?, ?)", row
                    )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # -- reads -----------------------------------------------------------------------------

    def load_all(self, node_id: str | None = None) -> list[JobRecord]:
        """Every job for ``node_id`` (or all, when ``None``), oldest first."""
        with self._lock:
            if node_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at ASC, job_id ASC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE node_id = ? ORDER BY created_at ASC, job_id ASC",
                    (node_id,),
                ).fetchall()
        return [_parse_row(row) for row in rows]

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _parse_row(row) if row is not None else None

    def events(self, job_id: str) -> list[JobEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY seq ASC", (job_id,)
            ).fetchall()
        return [
            JobEvent(row["seq"], row["job_id"], row["at"], row["kind"], _event_data(row["data"]))
            for row in rows
        ]


def _event_data(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except ValueError:
        return {"unreadable": text[:200]}
    return data if isinstance(data, dict) else {"value": data}
