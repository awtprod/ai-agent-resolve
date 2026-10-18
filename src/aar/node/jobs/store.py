"""The node-side job store: a single SQLite database that survives a process restart.

The store is the source of truth for job state. The runner holds no job state that is not
here, so a fresh runner opened on the same file resumes every job from its last checkpoint. A
checkpoint is one transaction: the ``jobs`` row (current state, step and cursor) is upserted
and any transition events are appended together, so a reader never sees a half-written job.

``jobs`` is mutable (state advances); ``job_events`` is append-only, enforced by triggers like
the hub's event log, so a job's history cannot be rewritten.
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

from .model import JobState, StructuredError, WaitReason

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,
    node_id           TEXT NOT NULL,
    target            TEXT NOT NULL,
    params            TEXT NOT NULL,
    state             TEXT NOT NULL,
    current_step      INTEGER NOT NULL,
    cursor            TEXT NOT NULL,
    results           TEXT NOT NULL,
    error             TEXT,
    waiting           TEXT,
    stall             TEXT,
    approval_token    TEXT,
    approval_consumed INTEGER NOT NULL DEFAULT 0,
    priority          INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);

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
    priority: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


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
        priority=row["priority"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


_UPSERT = """
INSERT INTO jobs (
    job_id, kind, node_id, target, params, state, current_step, cursor, results,
    error, waiting, stall, approval_token, approval_consumed, priority, created_at, updated_at
) VALUES (
    :job_id, :kind, :node_id, :target, :params, :state, :current_step, :cursor, :results,
    :error, :waiting, :stall, :approval_token, :approval_consumed, :priority,
    :created_at, :updated_at
)
ON CONFLICT(job_id) DO UPDATE SET
    state=excluded.state, current_step=excluded.current_step, cursor=excluded.cursor,
    results=excluded.results, error=excluded.error, waiting=excluded.waiting,
    stall=excluded.stall, approval_token=excluded.approval_token,
    approval_consumed=excluded.approval_consumed, priority=excluded.priority,
    updated_at=excluded.updated_at
"""


class JobStore:
    """An SQLite-backed job store (WAL mode) with checkpoint-and-event transactions."""

    def __init__(self, db_path: str | PathLike[str], *, create: bool = True) -> None:
        self._path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.isolation_level = None
            self._conn.execute("PRAGMA recursive_triggers=ON")
            if create:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(_SCHEMA)
            else:
                self._assert_schema()
                self._conn.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            self._conn.close()
            raise

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
            self._conn.close()

    # -- writes ----------------------------------------------------------------------------

    def save(
        self, record: JobRecord, events: Iterable[tuple[str, dict[str, Any], float]] = ()
    ) -> None:
        """Upsert ``record`` and append ``events`` (``kind``, ``data``, ``at``) in one txn."""
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
            "priority": record.priority,
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

    # -- reads -----------------------------------------------------------------------------

    def load_all(self) -> list[JobRecord]:
        """Every job, oldest first (by creation, then id)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at ASC, job_id ASC"
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _record_from_row(row) if row is not None else None

    def events(self, job_id: str) -> list[JobEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY seq ASC", (job_id,)
            ).fetchall()
        return [
            JobEvent(row["seq"], row["job_id"], row["at"], row["kind"], json.loads(row["data"]))
            for row in rows
        ]
