"""Project registry: stable ids, display names, aliases and source locators.

A project has a stable id (``prj_...``) that never changes, a display name that can
be renamed, zero or more aliases, and one or more sources that say where the project
lives (a cloud library, a Local Database, or a ``.drp``/``.drt`` file on a named
host). Every mutation is an event on the shared log; the in-memory state is a
projection rebuildable from the log, so renaming keeps the id and the project can be
found again by any of its source locators.

Mutations run inside one ``BEGIN IMMEDIATE`` transaction that first folds any events
appended by another writer, so two registries on the same log cannot diverge, and a
reused explicit id is rejected rather than silently overwriting.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from typing import Any

from aar.hub.common import SkippedEvent, is_member
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    IdempotencyConflict,
    Transaction,
    match_or_conflict,
)

SOURCE_KINDS = frozenset({"cloud_library", "local_db", "drp_file", "drt_file"})

# Writers refuse longer names and aliases, and more aliases than MAX_ALIASES on one
# project; the fold still accepts what is already in the log, so old data never
# disappears from projections.
MAX_NAME_CHARS = 500
MAX_ALIASES = 64

EVENT_PROJECT_CREATED = "project.created"
EVENT_PROJECT_RENAMED = "project.renamed"
EVENT_SOURCE_ADDED = "project.source_added"
EVENT_ALIAS_ADDED = "project.alias_added"

_REGISTRY_TYPES = (
    EVENT_PROJECT_CREATED,
    EVENT_PROJECT_RENAMED,
    EVENT_SOURCE_ADDED,
    EVENT_ALIAS_ADDED,
)


class DuplicateProjectId(ValueError):
    """Raised when an explicit project id is reused for a new project."""


def new_project_id() -> str:
    """Return a fresh, stable project id of the form ``prj_<hex>``."""
    return "prj_" + secrets.token_hex(8)


@dataclass(frozen=True, slots=True)
class ProjectSource:
    """Where a project lives. ``locator`` is kind-specific (e.g. a library + name, or a path)."""

    kind: str
    locator: dict[str, Any]
    host: str | None = None

    def __post_init__(self) -> None:
        if not is_member(self.kind, SOURCE_KINDS):
            raise ValueError(
                f"source kind must be one of {sorted(SOURCE_KINDS)}, got {self.kind!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "locator": self.locator, "host": self.host}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectSource:
        return cls(kind=data["kind"], locator=data["locator"], host=data.get("host"))


@dataclass
class Project:
    """A registered project. ``id`` is stable across renames."""

    id: str
    name: str
    sources: list[ProjectSource] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    created: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sources": [source.to_dict() for source in self.sources],
            "aliases": list(self.aliases),
            "created": self.created,
        }


def _reduce(
    projects: dict[str, Project], event_type: str, payload: dict[str, Any], ts: str
) -> None:
    """Fold one registry event into ``projects`` in place."""
    pid = payload["project_id"]
    if event_type == EVENT_PROJECT_CREATED:
        projects[pid] = Project(
            id=pid,
            name=payload["name"],
            sources=[ProjectSource.from_dict(s) for s in payload.get("sources", [])],
            aliases=list(payload.get("aliases", [])),
            created=ts,
        )
    elif event_type == EVENT_PROJECT_RENAMED:
        projects[pid].name = payload["name"]
    elif event_type == EVENT_SOURCE_ADDED:
        source = ProjectSource.from_dict(payload["source"])
        project = projects[pid]
        if source not in project.sources:
            project.sources.append(source)
    elif event_type == EVENT_ALIAS_ADDED:
        alias = payload["alias"]
        project = projects[pid]
        if alias not in project.aliases:
            project.aliases.append(alias)


def _source_ok(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and is_member(value.get("kind"), SOURCE_KINDS)
        and isinstance(value.get("locator"), dict)
        and (value.get("host") is None or isinstance(value.get("host"), str))
    )


def _problem(projects: dict[str, Project], event: Event) -> str | None:
    """Why ``event`` cannot be folded, or ``None``. A bad event is skipped, never raised."""
    payload = event.payload
    if not isinstance(payload, dict):
        return "payload is not an object"
    pid = payload.get("project_id")
    if not isinstance(pid, str) or not 0 < len(pid) <= 128:
        return "project_id is not an id"
    if event.type == EVENT_PROJECT_CREATED:
        if pid in projects:
            return f"project {pid} already exists"
        if not isinstance(payload.get("name"), str):
            return "name is not text"
        sources = payload.get("sources", [])
        if not isinstance(sources, list) or not all(_source_ok(s) for s in sources):
            return "sources are malformed"
        aliases = payload.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            return "aliases are not a list of text"
        return None
    if pid not in projects:
        return f"no project {pid}"
    if event.type == EVENT_PROJECT_RENAMED:
        return None if isinstance(payload.get("name"), str) else "name is not text"
    if event.type == EVENT_SOURCE_ADDED:
        return None if _source_ok(payload.get("source")) else "source is malformed"
    if event.type == EVENT_ALIAS_ADDED:
        return None if isinstance(payload.get("alias"), str) else "alias is not text"
    return f"not a registry event type: {event.type!r}"


def _fold(projects: dict[str, Project], skipped: list[SkippedEvent], event: Event) -> None:
    """Fold one event, or record it as skipped. Never raises."""
    problem = _problem(projects, event)
    if problem is None:
        _reduce(projects, event.type, event.payload, event.ts)
    else:
        skipped.append(SkippedEvent.of(event, problem))


class Registry:
    """Project registry projected from the event log."""

    def __init__(self, log: EventLog) -> None:
        self._log = log
        self._projects: dict[str, Project] = {}
        self._skipped: list[SkippedEvent] = []
        self._last_seq = 0
        self.refresh()

    # -- projection ---------------------------------------------------------

    def rebuild(self) -> dict[str, Project]:
        """Return a fresh projection built from the whole log (independent of live state)."""
        projects: dict[str, Project] = {}
        skipped: list[SkippedEvent] = []
        for event in self._log.iter_events(types=_REGISTRY_TYPES):
            _fold(projects, skipped, event)
        return projects

    def refresh(self) -> None:
        """Reload live state from the log.

        ``_last_seq`` is taken from the events actually folded, never from a separate
        ``latest_seq()`` read that could jump past events appended since the rebuild.
        """
        projects: dict[str, Project] = {}
        skipped: list[SkippedEvent] = []
        last_seq = 0
        for event in self._log.iter_events(types=_REGISTRY_TYPES):
            _fold(projects, skipped, event)
            last_seq = event.seq
        self._projects = projects
        self._skipped = skipped
        self._last_seq = last_seq

    def snapshot(self) -> dict[str, Project]:
        """Return the current live projection."""
        return self._projects

    def skipped(self) -> tuple[SkippedEvent, ...]:
        """Events the live projection could not use (malformed or forged)."""
        return tuple(self._skipped)

    # -- mutations ----------------------------------------------------------

    def create_project(
        self,
        name: str,
        actor: Actor,
        *,
        sources: list[ProjectSource] | None = None,
        aliases: list[str] | None = None,
        project_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Project:
        """Create a project and return it.

        An explicit ``project_id`` that already exists is rejected with
        :class:`DuplicateProjectId`, unless the same ``idempotency_key`` was used for
        the original create (a genuine retry), which returns the existing project.
        """
        source_dicts = [s.to_dict() for s in (sources or [])]
        alias_list = list(aliases or [])
        with self._log.transaction() as txn:
            self._catch_up(txn)
            if idempotency_key is not None:
                existing = txn.find_by_idempotency_key(idempotency_key)
                if existing is not None:
                    expected = {
                        "name": name,
                        "sources": source_dicts,
                        "aliases": alias_list,
                        "project_id": project_id,
                    }
                    compare = ["name", "sources", "aliases"]
                    if project_id is not None:
                        compare.append("project_id")
                    match_or_conflict(
                        existing,
                        EVENT_PROJECT_CREATED,
                        expected,
                        compare,
                        project_id=project_id,
                        actor=actor,
                    )
                    project = self._projects.get(existing.payload.get("project_id"))
                    if project is None:
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} names an event the "
                            "registry could not use"
                        )
                    return project
            pid = project_id or new_project_id()
            if pid in self._projects:
                raise DuplicateProjectId(f"project id already exists: {pid!r}")
            payload = {
                "project_id": pid,
                "name": name,
                "sources": source_dicts,
                "aliases": alias_list,
            }
            self._refuse_unless_foldable(EVENT_PROJECT_CREATED, actor, payload)
            event = txn.append(
                EVENT_PROJECT_CREATED,
                actor,
                payload,
                project_id=pid,
                idempotency_key=idempotency_key,
            )
        self._apply(event)
        return self._projects[pid]

    def rename_project(
        self, project_id: str, new_name: str, actor: Actor, *, idempotency_key: str | None = None
    ) -> Project:
        """Rename a project, keeping its id."""
        return self._mutate(
            project_id,
            EVENT_PROJECT_RENAMED,
            {"project_id": project_id, "name": new_name},
            actor,
            idempotency_key,
        )

    def add_source(
        self,
        project_id: str,
        source: ProjectSource,
        actor: Actor,
        *,
        idempotency_key: str | None = None,
    ) -> Project:
        """Record another place the project lives."""
        return self._mutate(
            project_id,
            EVENT_SOURCE_ADDED,
            {"project_id": project_id, "source": source.to_dict()},
            actor,
            idempotency_key,
        )

    def add_alias(
        self, project_id: str, alias: str, actor: Actor, *, idempotency_key: str | None = None
    ) -> Project:
        """Add an alternative name for the project."""
        return self._mutate(
            project_id,
            EVENT_ALIAS_ADDED,
            {"project_id": project_id, "alias": alias},
            actor,
            idempotency_key,
        )

    # -- queries ------------------------------------------------------------

    def get(self, project_id: str) -> Project | None:
        """Return the project with this id, or ``None``."""
        return self._projects.get(project_id)

    def list(self) -> list[Project]:
        """Return all projects, ordered by creation."""
        return sorted(self._projects.values(), key=lambda p: (p.created or "", p.id))

    def find_by_locator(
        self, kind: str, locator: dict[str, Any], host: str | None = None
    ) -> Project | None:
        """Return the project owning a source with this kind/locator (and host, if given)."""
        for project in self._projects.values():
            for source in project.sources:
                if (
                    source.kind == kind
                    and source.locator == locator
                    and (host is None or source.host == host)
                ):
                    return project
        return None

    def find_by_alias(self, alias: str) -> Project | None:
        """Return the project with this alias, or ``None``."""
        for project in self._projects.values():
            if alias in project.aliases:
                return project
        return None

    def find_by_name(self, name: str) -> Project | None:
        """Return the first project with this display name, or ``None``."""
        for project in self.list():
            if project.name == name:
                return project
        return None

    # -- internals ----------------------------------------------------------

    def _mutate(
        self,
        project_id: str,
        event_type: str,
        payload: dict[str, Any],
        actor: Actor,
        idempotency_key: str | None,
    ) -> Project:
        with self._log.transaction() as txn:
            self._catch_up(txn)
            if project_id not in self._projects:
                raise KeyError(f"unknown project: {project_id!r}")
            if idempotency_key is not None:
                existing = txn.find_by_idempotency_key(idempotency_key)
                if existing is not None:
                    match_or_conflict(
                        existing,
                        event_type,
                        payload,
                        payload.keys(),
                        project_id=project_id,
                        actor=actor,
                    )
                    return self._projects[project_id]
            self._refuse_unless_foldable(event_type, actor, payload)
            event = txn.append(
                event_type,
                actor,
                payload,
                project_id=project_id,
                idempotency_key=idempotency_key,
            )
        self._apply(event)
        return self._projects[project_id]

    def _refuse_unless_foldable(
        self, event_type: str, actor: Actor, payload: dict[str, Any]
    ) -> None:
        """Refuse, before writing, an event the fold would skip (a silent no-op or a
        KeyError after the commit otherwise). Checked as stored, after a JSON round trip.
        """
        for name_field in ("name", "alias"):
            value = payload.get(name_field)
            if isinstance(value, str) and len(value) > MAX_NAME_CHARS:
                raise ValueError(
                    f"refused before writing {event_type}: {name_field} is over {MAX_NAME_CHARS}"
                )
        aliases = payload.get("aliases")
        if isinstance(aliases, list):
            if len(aliases) > MAX_ALIASES:
                raise ValueError(
                    f"refused before writing {event_type}: too many aliases (over {MAX_ALIASES})"
                )
            if any(isinstance(a, str) and len(a) > MAX_NAME_CHARS for a in aliases):
                raise ValueError(
                    f"refused before writing {event_type}: alias is over {MAX_NAME_CHARS}"
                )
        if event_type == EVENT_ALIAS_ADDED:
            # One alias per event, so the cap is on what the project would hold.
            project = self._projects.get(payload.get("project_id"))
            if (
                project is not None
                and len(project.aliases) >= MAX_ALIASES
                and payload.get("alias") not in project.aliases
            ):
                raise ValueError(
                    f"refused before writing {event_type}: too many aliases (over {MAX_ALIASES})"
                )
        stored = json.loads(json.dumps(payload, allow_nan=False))
        probe = Event(0, "", stored.get("project_id"), event_type, actor, "", stored)
        problem = _problem(self._projects, probe)
        if problem is not None:
            raise ValueError(f"refused before writing {event_type}: {problem}")

    def _catch_up(self, txn: Transaction) -> None:
        for event in txn.read(self._last_seq, types=_REGISTRY_TYPES):
            if event.seq > self._last_seq:
                _fold(self._projects, self._skipped, event)
                self._last_seq = event.seq

    def _apply(self, event: Event) -> None:
        if event.seq > self._last_seq:
            _fold(self._projects, self._skipped, event)
            self._last_seq = event.seq
