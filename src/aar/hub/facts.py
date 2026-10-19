"""Per-project facts: what one project has learned that later work should reuse.

DESIGN R3/R6: sync and angle offsets, gains, template traps and path maps, plus what a
recipe learns on the way (render hashes, QA numbers, template ids). A fact is a typed
value under a key; the key's namespace decides the value's shape:

=========================================  ==============================================
key                                        value
=========================================  ==============================================
``sync.offset.<angle>``                    ``{frames, fps: {num, den}, relative_to}``:
                                           whole frames at that rate, and what from
``audio.gain.<track>``                     ``{db}``, -96 to +24 dB
``template.trap.<id>``                     ``{text}``, up to 2000 characters
``template.id.<name>``                     ``{unique_id}``, a Resolve unique id
``path.map.<name>``                        ``{from, to}``, paths up to 1024 characters
``render.hash.<deliverable>``              ``{sha256, bytes}``
``qa.<metric>.<deliverable>``              ``{value, unit}``: a finite number and its unit
=========================================  ==============================================

Every ``<...>`` part matches ``[A-Za-z0-9][A-Za-z0-9_-]{0,63}`` (no dots), so a key parses
one way only. The set of namespaces is closed: an unknown one, or a value that does not
fit, is refused (``unknown_namespace``, ``invalid_fact``).

Facts are events on the shared log, ``facts.recorded`` and ``facts.retracted``, scoped to
the project. :func:`facts_view` folds them: for each key the current value (the last event
in log order wins), who set it and when, its optional source (the recipe node that learned
it and/or an artifact), and the last :data:`MAX_FACT_HISTORY` changes. A malformed or
forged event is skipped and listed, never raised. Any actor kind may record facts; every
write needs an idempotency key, with the same replay and conflict rules as
:mod:`aar.hub.writes`.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from aar.hub.common import InvalidInput, SkippedEvent, actor_dict, clean_text, skipped_summary
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    IdempotencyConflict,
    Transaction,
    match_or_conflict,
)
from aar.hub.graph import WorkGraph
from aar.hub.writes import (
    SEGMENT_RE,
    NodeId,
    PathText,
    ResolveId,
    Sha256,
    StrictModel,
    TitleText,
    WriteError,
    WriteResult,
    bounded_text,
    check_actor,
    check_id,
    require_project,
    validation_details,
    write_key,
)

EVENT_RECORDED = "facts.recorded"
EVENT_RETRACTED = "facts.retracted"
FACT_EVENT_TYPES = (EVENT_RECORDED, EVENT_RETRACTED)

OP_RECORD = "facts.record"
OP_RETRACT = "facts.retract"

MAX_FACT_KEY = 200
MAX_FACT_HISTORY = 10  # changes kept per key, newest last
MAX_FACT_KEYS = 1000  # keys (set or retracted) one project may hold; writers refuse more
MAX_TRAP_TEXT = 2000
MAX_REASON = 1000
MAX_OFFSET_FRAMES = 10_000_000
MAX_RATE_PART = 1_000_000
MIN_GAIN_DB = -96.0
MAX_GAIN_DB = 24.0
MAX_QA_ABS = 1e15
MAX_RENDER_BYTES = 2**53

_SEG = SEGMENT_RE.removeprefix("^").removesuffix("$")


class FrameRate(StrictModel):
    """A frame rate as a rational, e.g. ``{"num": 24000, "den": 1001}``."""

    num: int = Field(ge=1, le=MAX_RATE_PART)
    den: int = Field(default=1, ge=1, le=MAX_RATE_PART)


class SyncOffset(StrictModel):
    """``sync.offset.<angle>``: this angle sits ``frames`` after ``relative_to``."""

    frames: int = Field(ge=-MAX_OFFSET_FRAMES, le=MAX_OFFSET_FRAMES)
    fps: FrameRate
    relative_to: TitleText


class AudioGain(StrictModel):
    """``audio.gain.<track>``: a track gain in dB."""

    db: float = Field(ge=MIN_GAIN_DB, le=MAX_GAIN_DB)


class TemplateTrap(StrictModel):
    """``template.trap.<id>``: a known trap in the project's template, as text."""

    text: bounded_text(MAX_TRAP_TEXT, multiline=True)


class TemplateId(StrictModel):
    """``template.id.<name>``: the Resolve unique id of a template item (a timeline...)."""

    unique_id: ResolveId


class PathMap(StrictModel):
    """``path.map.<name>``: a path prefix on one host and its counterpart on another."""

    from_: PathText = Field(alias="from")
    to: PathText


class RenderHash(StrictModel):
    """``render.hash.<deliverable>``: the sha256 and size of a rendered file."""

    sha256: Sha256
    bytes: int = Field(ge=0, le=MAX_RENDER_BYTES)


class QaMetric(StrictModel):
    """``qa.<metric>.<deliverable>``: a measured number and its unit (LUFS, dBTP, s, MB)."""

    value: float = Field(ge=-MAX_QA_ABS, le=MAX_QA_ABS)
    unit: Annotated[str, Field(pattern=r"^[A-Za-z%][A-Za-z0-9%/._-]{0,15}$")]


@dataclass(frozen=True, slots=True)
class Namespace:
    """One closed fact namespace: its key shape and its value model."""

    name: str
    pattern: re.Pattern[str]
    shape: str
    model: type[StrictModel]


NAMESPACES: tuple[Namespace, ...] = (
    Namespace(
        "sync.offset", re.compile(rf"sync\.offset\.{_SEG}"), "sync.offset.<angle>", SyncOffset
    ),
    Namespace("audio.gain", re.compile(rf"audio\.gain\.{_SEG}"), "audio.gain.<track>", AudioGain),
    Namespace(
        "template.trap",
        re.compile(rf"template\.trap\.{_SEG}"),
        "template.trap.<id>",
        TemplateTrap,
    ),
    Namespace(
        "template.id", re.compile(rf"template\.id\.{_SEG}"), "template.id.<name>", TemplateId
    ),
    Namespace("path.map", re.compile(rf"path\.map\.{_SEG}"), "path.map.<name>", PathMap),
    Namespace(
        "render.hash",
        re.compile(rf"render\.hash\.{_SEG}"),
        "render.hash.<deliverable>",
        RenderHash,
    ),
    Namespace("qa", re.compile(rf"qa\.{_SEG}\.{_SEG}"), "qa.<metric>.<deliverable>", QaMetric),
)


class FactSource(StrictModel):
    """Where a fact came from: the recipe node that learned it and/or an artifact."""

    recipe_node: NodeId | None = None
    artifact: Sha256 | None = None

    @model_validator(mode="after")
    def _one_at_least(self) -> FactSource:
        if self.recipe_node is None and self.artifact is None:
            raise ValueError("a source names a recipe_node, an artifact, or both")
        return self


def namespace_of(key: Any) -> Namespace:
    """The namespace ``key`` belongs to; ``unknown_namespace`` or ``invalid_fact`` if none.

    A key with a known namespace prefix but a bad ``<...>`` part is ``invalid_fact``; a
    key with no known prefix is ``unknown_namespace``.
    """
    if not isinstance(key, str) or not 0 < len(key) <= MAX_FACT_KEY:
        raise WriteError(
            "invalid_fact",
            f"a fact key is 1-{MAX_FACT_KEY} characters, got {key!r:.80}",
            "Use a key such as 'sync.offset.wide' or 'qa.lufs.master'.",
            field="key",
        )
    for namespace in NAMESPACES:
        if key.startswith(namespace.name + "."):
            if namespace.pattern.fullmatch(key) is None:
                raise WriteError(
                    "invalid_fact",
                    f"{key!r:.80} does not fit {namespace.shape}",
                    "Each <...> part is 1-64 characters of A-Z a-z 0-9 _ - (no dots).",
                    field="key",
                    shape=namespace.shape,
                )
            return namespace
    raise WriteError(
        "unknown_namespace",
        f"{key!r:.80} is not in a known fact namespace",
        "Use one of: " + ", ".join(namespace.shape for namespace in NAMESPACES) + ".",
        field="key",
        namespaces=[namespace.shape for namespace in NAMESPACES],
    )


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True)


# -- projection -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactChange:
    """One change to a key: a value recorded, or the key retracted."""

    action: Literal["recorded", "retracted"]
    value: dict[str, Any] | None
    source: dict[str, Any] | None
    reason: str | None
    by: Actor
    at: str
    event_id: str
    seq: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "value": self.value,
            "source": self.source,
            "reason": self.reason,
            "by": actor_dict(self.by),
            "at": self.at,
            "event_id": self.event_id,
            "seq": self.seq,
        }


@dataclass(frozen=True, slots=True)
class Fact:
    """A key's current state and its recent history (newest last)."""

    key: str
    namespace: str
    current: FactChange
    history: tuple[FactChange, ...]

    @property
    def is_set(self) -> bool:
        return self.current.action == "recorded"

    def to_dict(self) -> dict[str, Any]:
        latest = self.current
        return {
            "key": self.key,
            "namespace": self.namespace,
            "value": latest.value,
            "source": latest.source,
            "set_by": actor_dict(latest.by),
            "set_at": latest.at,
            "history": [change.to_dict() for change in self.history],
        }


@dataclass(frozen=True, slots=True)
class FactsView:
    """A project's facts as folded from the log, plus the events that were skipped."""

    project_id: str
    facts: dict[str, Fact]
    skipped: tuple[SkippedEvent, ...] = ()

    def current(self) -> list[Fact]:
        """Keys that are set (not retracted), sorted by key."""
        return [fact for key, fact in sorted(self.facts.items()) if fact.is_set]

    def retracted(self) -> list[Fact]:
        """Keys whose last change was a retraction, sorted by key."""
        return [fact for key, fact in sorted(self.facts.items()) if not fact.is_set]

    def get(self, key: str) -> Fact | None:
        """The key's state if it is set, else ``None``."""
        fact = self.facts.get(key)
        return fact if fact is not None and fact.is_set else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "facts": [fact.to_dict() for fact in self.current()],
            "retracted": [_retracted_dict(fact) for fact in self.retracted()],
            "skipped": skipped_summary(self.skipped),
        }


def _retracted_dict(fact: Fact) -> dict[str, Any]:
    latest = fact.current
    return {
        "key": fact.key,
        "namespace": fact.namespace,
        "reason": latest.reason,
        "retracted_by": actor_dict(latest.by),
        "retracted_at": latest.at,
        "history": [change.to_dict() for change in fact.history],
    }


@dataclass
class _State:
    current: dict[str, FactChange] = field(default_factory=dict)
    namespaces: dict[str, str] = field(default_factory=dict)
    history: dict[str, deque[FactChange]] = field(default_factory=dict)
    skipped: list[SkippedEvent] = field(default_factory=list)


def _key_problem(key: Any) -> tuple[str | None, Namespace | None]:
    try:
        return None, namespace_of(key)
    except WriteError as exc:
        return exc.message, None


def _record_problem(payload: dict[str, Any]) -> tuple[str | None, Namespace | None]:
    problem, namespace = _key_problem(payload.get("key"))
    if namespace is None:
        return problem, None
    value = payload.get("value")
    try:
        namespace.model.model_validate(value)
    except ValidationError as exc:
        return f"value does not fit {namespace.shape}: {exc.error_count()} problem(s)", None
    source = payload.get("source")
    if source is not None:
        try:
            FactSource.model_validate(source)
        except ValidationError:
            return "source is not {recipe_node, artifact}", None
    return None, namespace


def _change(event: Event, action: Literal["recorded", "retracted"]) -> FactChange:
    payload = event.payload
    return FactChange(
        action=action,
        value=payload.get("value") if action == "recorded" else None,
        source=payload.get("source") if action == "recorded" else None,
        reason=payload.get("reason") if action == "retracted" else None,
        by=event.actor,
        at=event.ts,
        event_id=event.id,
        seq=event.seq,
    )


def _fold(state: _State, event: Event) -> None:
    """Fold one facts event into ``state``, or record why it was skipped. Never raises."""
    payload = event.payload
    if not isinstance(payload, dict):
        state.skipped.append(SkippedEvent.of(event, "payload is not an object"))
        return
    if event.type == EVENT_RECORDED:
        problem, namespace = _record_problem(payload)
        action: Literal["recorded", "retracted"] = "recorded"
    elif event.type == EVENT_RETRACTED:
        problem, namespace = _key_problem(payload.get("key"))
        action = "retracted"
        reason = payload.get("reason")
        if problem is None and not (isinstance(reason, str) and 0 < len(reason) <= MAX_REASON):
            problem = f"reason is not 1-{MAX_REASON} characters"
        current = state.current.get(payload.get("key")) if problem is None else None
        if problem is None and (current is None or current.action != "recorded"):
            problem = "retracts a key that is not set"
    else:
        problem, namespace = f"not a facts event type: {event.type!r}", None
    if problem is not None or namespace is None:
        state.skipped.append(SkippedEvent.of(event, problem or "unusable"))
        return
    key = payload["key"]
    change = _change(event, action)
    state.current[key] = change
    state.namespaces[key] = namespace.name
    state.history.setdefault(key, deque(maxlen=MAX_FACT_HISTORY)).append(change)


def _view_from(project_id: str, events: list[Event] | Any) -> FactsView:
    state = _State()
    for event in events:
        _fold(state, event)
    facts = {
        key: Fact(
            key=key,
            namespace=state.namespaces[key],
            current=change,
            history=tuple(state.history[key]),
        )
        for key, change in state.current.items()
    }
    return FactsView(project_id=project_id, facts=facts, skipped=tuple(state.skipped))


def facts_view(log: EventLog, project_id: str) -> FactsView:
    """The project's facts folded from the log (malformed events skipped and listed)."""
    return _view_from(project_id, log.iter_events(project_id=project_id, types=FACT_EVENT_TYPES))


def _txn_view(txn: Transaction, project_id: str) -> FactsView:
    return _view_from(project_id, txn.read(0, project_id=project_id, types=FACT_EVENT_TYPES))


# -- writes -------------------------------------------------------------------------------


def _change_body(event: Event, action: Literal["recorded", "retracted"]) -> dict[str, Any]:
    """The result of a facts write, rebuilt from its event: the same dict on a replay."""
    namespace = namespace_of(event.payload["key"]).name
    change = _change(event, action).to_dict()
    return {"key": event.payload["key"], "namespace": namespace, **change}


def _replayed(
    txn: Transaction,
    key: str,
    event_type: str,
    payload: dict[str, Any],
    project_id: str,
    actor: Actor,
    idempotency_key: str,
) -> Event | None:
    existing = txn.find_by_idempotency_key(key)
    if existing is None:
        return None
    try:
        match_or_conflict(
            existing, event_type, payload, payload.keys(), project_id=project_id, actor=actor
        )
    except IdempotencyConflict:
        raise WriteError(
            "idempotency_conflict",
            f"idempotency key {idempotency_key!r} was already used for a different request",
            "Use a new key for a new request. Reuse a key only to retry exactly the same request.",
            idempotency_key=idempotency_key,
        ) from None
    return existing


def record_fact(
    log: EventLog,
    project_id: str,
    key: str,
    value: Mapping[str, Any],
    actor: Actor,
    *,
    idempotency_key: str,
    source: Mapping[str, Any] | None = None,
) -> WriteResult:
    """Record ``value`` under ``key`` for the project; it becomes the key's current value.

    ``source`` is optional ``{recipe_node, artifact}``: the recipe node must exist in the
    project and be a recipe, and the artifact must have been stored in the hub.
    """
    actor = check_actor(actor)
    write = write_key(actor, idempotency_key)
    project_id = check_id(project_id, "project_id")
    namespace = namespace_of(key)
    if not isinstance(value, Mapping):
        raise WriteError(
            "invalid_fact",
            f"value must be a JSON object, got {type(value).__name__}",
            f"Send the value as an object that fits {namespace.shape}.",
            field="value",
        )
    try:
        model = namespace.model.model_validate(dict(value))
    except ValidationError as exc:
        raise WriteError(
            "invalid_fact",
            f"the value does not fit {namespace.shape} ({exc.error_count()} problem(s))",
            "Fix each field listed in details.errors.",
            shape=namespace.shape,
            **validation_details(exc),
        ) from None
    source_model = None
    if source is not None:
        if not isinstance(source, Mapping):
            raise WriteError(
                "invalid_fact",
                f"source must be an object, got {type(source).__name__}",
                "Send {recipe_node, artifact} or leave source out.",
                field="source",
            )
        try:
            source_model = FactSource.model_validate(dict(source))
        except ValidationError as exc:
            raise WriteError(
                "invalid_fact",
                "the source failed validation",
                "Send {recipe_node: <recipe node id>, artifact: <sha256>} (either or both).",
                field="source",
                **validation_details(exc),
            ) from None
    payload = {
        "key": key,
        "value": _dump(model),
        "source": _dump(source_model) if source_model is not None else None,
    }
    with log.transaction() as txn:
        existing = _replayed(
            txn, write, EVENT_RECORDED, payload, project_id, actor, idempotency_key
        )
        if existing is not None:
            return WriteResult(
                OP_RECORD, True, project_id, {"fact": _change_body(existing, "recorded")}
            )
        require_project(log, project_id)
        if source_model is not None:
            _check_source(txn, log, project_id, source_model)
        view = _txn_view(txn, project_id)
        if key not in view.facts and len(view.facts) >= MAX_FACT_KEYS:
            raise WriteError(
                "too_many_facts",
                f"project {project_id!r} already holds {MAX_FACT_KEYS} fact keys",
                "Reuse an existing key, or record coarser facts.",
                limit=MAX_FACT_KEYS,
            )
        event = txn.append(
            EVENT_RECORDED, actor, payload, project_id=project_id, idempotency_key=write
        )
    return WriteResult(OP_RECORD, False, project_id, {"fact": _change_body(event, "recorded")})


def _check_source(txn: Transaction, log: EventLog, project_id: str, source: FactSource) -> None:
    if source.recipe_node is not None:
        node = WorkGraph(log, project_id).node(source.recipe_node)
        if node is None or node.kind != "recipe":
            raise WriteError(
                "unknown_source",
                f"no recipe node {source.recipe_node!r} in project {project_id!r}",
                "Name the recipe node that learned the fact (recipe_create returns its id), "
                "or leave recipe_node out.",
                recipe_node=source.recipe_node,
            )
    if (
        source.artifact is not None
        and txn.find_by_idempotency_key(f"artifact:{source.artifact}") is None
    ):
        raise WriteError(
            "unknown_source",
            f"artifact {source.artifact} was never stored in the hub",
            "Name an artifact the hub stored, or leave artifact out.",
            artifact=source.artifact,
        )


def retract_fact(
    log: EventLog,
    project_id: str,
    key: str,
    reason: str,
    actor: Actor,
    *,
    idempotency_key: str,
) -> WriteResult:
    """Retract the key's current value (it is no longer known). Its history is kept."""
    actor = check_actor(actor)
    write = write_key(actor, idempotency_key)
    project_id = check_id(project_id, "project_id")
    namespace_of(key)
    try:
        reason = clean_text(reason, "reason", max_len=MAX_REASON, multiline=True)
    except InvalidInput as exc:
        raise WriteError(
            "invalid_input", str(exc), "Say why the fact no longer holds.", field="reason"
        ) from None
    payload = {"key": key, "reason": reason}
    with log.transaction() as txn:
        existing = _replayed(
            txn, write, EVENT_RETRACTED, payload, project_id, actor, idempotency_key
        )
        if existing is not None:
            return WriteResult(
                OP_RETRACT, True, project_id, {"retracted": _change_body(existing, "retracted")}
            )
        require_project(log, project_id)
        if _txn_view(txn, project_id).get(key) is None:
            raise WriteError(
                "unknown_fact",
                f"project {project_id!r} has no fact {key!r} set",
                "Read the project's facts (facts_get) and retract a key that is set.",
                key=key,
            )
        event = txn.append(
            EVENT_RETRACTED, actor, payload, project_id=project_id, idempotency_key=write
        )
    return WriteResult(
        OP_RETRACT, False, project_id, {"retracted": _change_body(event, "retracted")}
    )
