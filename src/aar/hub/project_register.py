"""Register Resolve projects in the hub registry (``aar hub project`` and ``project_register``).

``aar node run`` refuses to start until the registry holds a project whose source locator
carries the Resolve project's unique id (``LocalHubLink.hub_project_id``). This module is
the one validated way to put it there; the CLI and the MCP tool are thin wrappers.

Rules, all checked before anything is written:

- Humans and agents may register; a node may not (``actor_not_allowed``).
- The source locator is an object. For ``local_db`` and ``cloud_library`` it must carry
  ``project_id`` (non-empty text), ``name`` and ``folder_path`` (an array of non-empty text),
  and nothing else; :class:`~aar.node.locator.ProjectLocator` is the one definition of that.
- One Resolve unique id belongs to one registry project (``project_exists``, naming the
  existing id). The check runs inside the registry's write transaction, after the
  idempotent-retry check, so a genuine retry of the same request still succeeds.
- A project may be registered without a source (a show edited from hub media before any
  Resolve project exists): pass neither ``source_kind`` nor ``locator`` (one without the
  other, or a ``host`` without them, is ``invalid_input``). Its name is then its identity: a
  name or alias that any project already has as its name or an alias is refused with
  ``project_exists`` naming the existing id, after the same retry check. The Resolve source
  is attached later with :func:`add_source`, under the same one-uid-one-project rule.
- :func:`rename` and :func:`add_alias` keep that identity: a name or alias a source-less
  project holds is reserved to it, and a source-less project may not take one any other
  project holds (``project_exists``). The library (:class:`Registry`) does not apply this
  rule unless asked (``unique_name``).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.registry import (
    MAX_ALIASES,
    MAX_NAME_CHARS,
    RESOLVE_SOURCE_KINDS,
    SOURCE_KINDS,
    DuplicateProjectId,
    DuplicateProjectName,
    DuplicateResolveProject,
    Project,
    ProjectSource,
    Registry,
)
from aar.hub.writes import WriteError, WriteResult, check_actor, check_id, write_key
from aar.node.locator import ProjectLocator

OP_REGISTER = "project.register"
MAX_FOLDER_SEGMENTS = 64
MAX_SEGMENT_CHARS = 255
MAX_UID_CHARS = 128
_ACTOR_KINDS_ALLOWED = frozenset({"human", "agent"})


def check_registrant(actor: Actor) -> Actor:
    """``actor`` if a human or an agent; a node (or a bad id) is refused."""
    actor = check_actor(actor)
    if actor.kind not in _ACTOR_KINDS_ALLOWED:
        raise WriteError(
            "actor_not_allowed",
            f"a {actor.kind} may not register or change projects",
            "Register as human:ID or agent:ID.",
            actor=actor.kind,
        )
    return actor


def _has_control(value: str) -> bool:
    """NUL and the other C0 control characters (and DEL), which no name should carry."""
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def _text(value: Any, field: str, *, limit: int = MAX_NAME_CHARS) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or _has_control(value):
        raise WriteError(
            "invalid_input",
            f"{field} must be text of 1-{limit} characters, not blank and without control "
            f"characters, got {value!r:.80}",
            f"Pass a printable, non-blank {field}.",
            field=field,
        )
    return value


# Control (C0 and C1), format (zero-width, BOM, bidi marks) and line/paragraph separators:
# nothing a show name needs, and they make two names look alike.
_NOT_IN_A_SHOW_NAME = frozenset({"Cc", "Cf", "Zl", "Zp"})


def _show_name(value: str, field: str) -> str:
    """``value`` (already through :func:`_text`) if it is a plain show name: no surrounding
    whitespace and nothing in :data:`_NOT_IN_A_SHOW_NAME`. Used where the name is the
    project's identity (a source-less registration, ``rename`` and ``add_alias``)."""
    bad = next((ch for ch in value if unicodedata.category(ch) in _NOT_IN_A_SHOW_NAME), None)
    if value != value.strip() or bad is not None:
        problem = "surrounding whitespace" if bad is None else f"the character U+{ord(bad):04X}"
        raise WriteError(
            "invalid_input",
            f"{field} has {problem}, got {value!r:.80}",
            f"Pass the {field} as plain text without surrounding spaces, control or "
            "zero-width characters.",
            field=field,
        )
    return value


def _check_locator_text(locator: Mapping[str, Any]) -> None:
    """The size and character rules ProjectLocator does not have (it only needs non-empty)."""
    for key, limit in (("name", MAX_NAME_CHARS), ("project_id", MAX_UID_CHARS)):
        value = locator.get(key)
        if isinstance(value, str):
            _text(value, f"locator.{key}", limit=limit)
    path = locator.get("folder_path")
    if isinstance(path, list | tuple):
        if len(path) > MAX_FOLDER_SEGMENTS:
            raise WriteError(
                "invalid_input",
                f"locator.folder_path has {len(path)} segments, at most {MAX_FOLDER_SEGMENTS}",
                "Send the Project Manager folder names from the root, not more than "
                f"{MAX_FOLDER_SEGMENTS} deep.",
                field="locator",
            )
        for segment in path:
            if isinstance(segment, str):
                _text(segment, "locator.folder_path segment", limit=MAX_SEGMENT_CHARS)


def build_source(kind: Any, locator: Any, host: Any = None) -> ProjectSource:
    """A validated :class:`ProjectSource`; ``invalid_input`` naming the field if it is not."""
    if not isinstance(kind, str) or kind not in SOURCE_KINDS:
        raise WriteError(
            "invalid_input",
            f"source kind must be one of {sorted(SOURCE_KINDS)}, got {kind!r:.80}",
            "Pass a listed source kind.",
            field="source_kind",
        )
    if not isinstance(locator, Mapping):
        raise WriteError(
            "invalid_input",
            f"locator must be a JSON object, got {type(locator).__name__}",
            "Pass the locator as an object.",
            field="locator",
        )
    if host is not None:
        host = _text(host, "host", limit=253)
    if kind not in RESOLVE_SOURCE_KINDS:
        return ProjectSource(kind=kind, locator=dict(locator), host=host)
    missing = sorted({"project_id", "name", "folder_path"} - set(locator))
    if missing:
        raise WriteError(
            "invalid_input",
            f"a {kind} locator needs {missing}",
            "Send {project_id, name, folder_path}; `aar node locate` prints them for the "
            "project open in Resolve.",
            field="locator",
        )
    try:
        _check_locator_text(locator)
    except WriteError as exc:
        exc.details["field"] = "locator"
        raise
    try:
        parsed = ProjectLocator.from_dict(locator)
    except (TypeError, ValueError) as exc:
        raise WriteError(
            "invalid_input",
            f"invalid {kind} locator: {exc}",
            "Send {project_id, name, folder_path}: project_id and name are non-empty text, "
            "folder_path is an array of non-empty folder names ([] is the root).",
            field="locator",
        ) from None
    if parsed.project_id is None:  # from_dict allows None; registration does not
        raise WriteError(
            "invalid_input",
            f"a {kind} locator needs project_id as non-empty text",
            "Use Project.GetUniqueId() of the Resolve project.",
            field="locator",
        )
    return ProjectSource(kind=kind, locator=parsed.to_dict(), host=host)


def _aliases(aliases: Sequence[Any] | None) -> list[str]:
    if aliases is None:
        return []
    if isinstance(aliases, str) or not isinstance(aliases, Sequence):
        raise WriteError(
            "invalid_input", "aliases must be an array of text", "Pass a list.", field="aliases"
        )
    if len(aliases) > MAX_ALIASES:
        raise WriteError(
            "invalid_input",
            f"at most {MAX_ALIASES} aliases",
            "Pass fewer aliases.",
            field="aliases",
        )
    return [_text(alias, "alias") for alias in aliases]


def _exists(error: DuplicateResolveProject) -> WriteError:
    return WriteError(
        "project_exists",
        f"Resolve project {error.resolve_uid!r} is already registered as {error.existing_id!r}",
        f"Use {error.existing_id} (aar hub project show), or add an alias to it.",
        existing_id=error.existing_id,
        resolve_uid=error.resolve_uid,
    )


def _name_taken(error: DuplicateProjectName) -> WriteError:
    return WriteError(
        "project_exists",
        f"{error.name!r} is already a name or alias of {error.existing_id!r}, and a project "
        "without a source is known by its name",
        f"Use {error.existing_id} (aar hub project show), or register under another name.",
        existing_id=error.existing_id,
        name=error.name,
    )


def _optional_source(source_kind: Any, locator: Any, host: Any) -> list[ProjectSource]:
    """``[]`` when neither ``source_kind`` nor ``locator`` is given, else the one source."""
    if source_kind is None and locator is None:
        if host is not None:
            raise WriteError(
                "invalid_input",
                "host names where a source lives, and no source was given",
                "Drop host, or give source_kind and locator too.",
                field="host",
            )
        return []
    if source_kind is None or locator is None:
        missing = "source_kind" if source_kind is None else "locator"
        raise WriteError(
            "invalid_input",
            f"{missing} is missing: give source_kind and locator together, or neither",
            "Pass both for a Resolve source, or neither to register the project without one "
            "(attach it later with `aar hub project source`).",
            field=missing,
        )
    return [build_source(source_kind, locator, host)]


def _conflict(error: IdempotencyConflict) -> WriteError:
    return WriteError(
        "idempotency_conflict",
        str(error),
        "Retry the original request unchanged, or use a fresh idempotency_key.",
    )


def register_project(
    log: EventLog,
    actor: Actor,
    *,
    name: Any,
    source_kind: Any = None,
    locator: Any = None,
    host: Any = None,
    aliases: Sequence[Any] | None = None,
    project_id: Any = None,
    idempotency_key: Any,
) -> WriteResult:
    """Register a project with one source, or none. A retry with the same key and request
    replays."""
    actor = check_registrant(actor)
    key = write_key(actor, idempotency_key)
    name = _text(name, "name")
    sources = _optional_source(source_kind, locator, host)
    alias_list = _aliases(aliases)
    if not sources:  # the name is the identity; a sourced registration is as before
        name = _show_name(name, "name")
        alias_list = [_show_name(alias, "alias") for alias in alias_list]
    if project_id is not None:
        project_id = check_id(project_id, "project_id")
    registry = Registry(log)
    replayed = log.get_by_idempotency_key(key) is not None
    try:
        project = registry.create_project(
            name,
            actor,
            sources=sources,
            aliases=alias_list,
            project_id=project_id,
            idempotency_key=key,
            unique_name=not sources,
        )
    except DuplicateResolveProject as exc:
        raise _exists(exc) from None
    except DuplicateProjectName as exc:
        raise _name_taken(exc) from None
    except DuplicateProjectId as exc:
        raise WriteError(
            "project_exists",
            str(exc),
            "Pick another --project-id, or omit it for a fresh one.",
            existing_id=project_id,
        ) from None
    except IdempotencyConflict as exc:
        raise _conflict(exc) from None
    except ValueError as exc:
        raise WriteError("invalid_input", str(exc), "Fix the named problem.") from None
    return WriteResult(OP_REGISTER, replayed, project.id, {"project": project.to_dict()})


def _existing(registry: Registry, project_id: Any) -> Project:
    project_id = check_id(project_id, "project_id")
    project = registry.get(project_id)
    if project is None:
        raise WriteError(
            "unknown_project",
            f"no project {project_id!r} in the hub registry",
            "Use an id from `aar hub project list` (prj_...).",
            project_id=project_id,
        )
    return project


def _change(
    log: EventLog, actor: Actor, project_id: Any, idempotency_key: Any, op: str, apply: Any
) -> WriteResult:
    actor = check_registrant(actor)
    key = write_key(actor, idempotency_key)
    registry = Registry(log)
    _existing(registry, project_id)
    replayed = log.get_by_idempotency_key(key) is not None
    try:
        project = apply(registry, key)
    except DuplicateResolveProject as exc:
        raise _exists(exc) from None
    except DuplicateProjectName as exc:
        raise _name_taken(exc) from None
    except IdempotencyConflict as exc:
        raise _conflict(exc) from None
    except KeyError:
        raise WriteError(
            "unknown_project", f"no project {project_id!r}", "Use `aar hub project list`."
        ) from None
    except ValueError as exc:
        raise WriteError("invalid_input", str(exc), "Fix the named problem.") from None
    return WriteResult(op, replayed, project.id, {"project": project.to_dict()})


def add_alias(
    log: EventLog, actor: Actor, project_id: Any, alias: Any, *, idempotency_key: Any
) -> WriteResult:
    text = _show_name(_text(alias, "alias"), "alias")
    return _change(
        log,
        actor,
        project_id,
        idempotency_key,
        "project.alias",
        lambda reg, key: reg.add_alias(
            project_id, text, actor, idempotency_key=key, unique_name=True
        ),
    )


def rename(
    log: EventLog, actor: Actor, project_id: Any, name: Any, *, idempotency_key: Any
) -> WriteResult:
    text = _show_name(_text(name, "name"), "name")
    return _change(
        log,
        actor,
        project_id,
        idempotency_key,
        "project.rename",
        lambda reg, key: reg.rename_project(
            project_id, text, actor, idempotency_key=key, unique_name=True
        ),
    )


def add_source(
    log: EventLog,
    actor: Actor,
    project_id: Any,
    *,
    source_kind: Any,
    locator: Any,
    host: Any = None,
    idempotency_key: Any,
) -> WriteResult:
    source = build_source(source_kind, locator, host)
    return _change(
        log,
        actor,
        project_id,
        idempotency_key,
        "project.source",
        lambda reg, key: reg.add_source(project_id, source, actor, idempotency_key=key),
    )
