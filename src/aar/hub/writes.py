"""The hub write surface for agents: typed brief, recipe and plan nodes, and own status.

DESIGN R6: agents create brief, plan and recipe nodes through hub write tools, while
deciding a review stays human-only. This module is that surface's library half; the MCP
tools (``aar hub serve --agent-writes``) and ``aar hub write`` are thin wrappers over it.

Node data is typed. Every model is strict (no coercion), forbids unknown keys, bounds
every string and list, refuses NaN and infinity, and carries a ``schema`` literal so a
later version can be told apart:

- :class:`Brief` (``aar.brief/1``): what to make, for which show, from which sources.
- :class:`RecipeData` (``aar.recipe/1``): one recipe run derived from a brief.
- :class:`PlanNodeData` (``aar.plan-node/1``): an EditPlan submitted for a brief. The plan
  itself is a content-addressed artifact (its canonical JSON); the node holds its sha256,
  version, rule profile and a bounded lint summary.

Writes. Every write takes an :class:`~aar.hub.events.Actor` and a required, caller-chosen
``idempotency_key``:

- The key is namespaced per actor (two agents never collide on a key) and names the whole
  request. The same key with the same request returns the first result again, marked
  ``replayed``; the same key with a different request (another project, another op, other
  data) is refused with ``idempotency_conflict``.
- A write that takes several events appends them all in ONE ``BEGIN IMMEDIATE``
  transaction, with a ``hub.write`` record (the op and a sha256 of the normalised request)
  last. A crash before the commit leaves nothing; a retry with the same key then writes it
  whole. The graph events also carry sub-keys (``#node``, ``#edge``) so the database itself
  refuses a duplicate. ``submit_plan`` stores the plan artifact first, outside the
  transaction: it is content-addressed and recorded once per hash, so a retry after a crash
  there reuses it.

Who may change what (:func:`set_own_status`): a non-human actor may move a brief, recipe,
plan or op node that it created itself (the actor on the node's creation event), through
the graph's transition table. A human may move any node except a review. Review nodes close
only through ``aar.hub.reviews`` (a human decision, or the requester's withdrawal).

Errors are :class:`WriteError` with a stable ``code``, a ``message``, a ``fix_hint`` and
details; :data:`ERROR_CODES` lists every code.

Rule profiles are built-in names only (``ccn``), never paths: ``aar.plan.load_profile``
reads any path it is given, and a path from a tool caller would make the hub read files on
its own host.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from aar.hub.artifacts import ArtifactStore
from aar.hub.common import InvalidInput, actor_dict, clean_text
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    IdempotencyConflict,
    Transaction,
    canonical_json,
)
from aar.hub.graph import (
    EVENT_EDGE_ADDED,
    EVENT_NODE_ADDED,
    STATUSES,
    Edge,
    GraphNode,
    WorkGraph,
    allowed_transitions,
)
from aar.hub.registry import Registry
from aar.node.branches import valid_agent_id
from aar.plan.lint import LintReport, all_rules, lint
from aar.plan.model import EditPlan
from aar.plan.profile import Profile, ProfileError, load_profile

SCHEMA_BRIEF = "aar.brief/1"
SCHEMA_RECIPE = "aar.recipe/1"
SCHEMA_PLAN_NODE = "aar.plan-node/1"

EVENT_WRITE = "hub.write"
"""The record a multi-event write appends last: ``{op, request_sha256, ...ids}``."""

OP_BRIEF_CREATE = "brief.create"
OP_RECIPE_CREATE = "recipe.create"
OP_PLAN_SUBMIT = "plan.submit"
OP_SET_STATUS = "node.set_status"

# -- bounds -------------------------------------------------------------------
# Strings are counted in characters after stripping. Chosen to fit real CCN briefs with a
# wide margin, while keeping every node small enough to list in a tool result.
MAX_ID = 128  # node, project and Resolve ids (matches the graph's id bound)
MAX_KEY = 128  # a caller's idempotency key
MAX_SHOW = 200
MAX_TITLE = 200
MAX_HOOK = 300
MAX_HANDLE = 100
MAX_LABEL = 200
MAX_LIBRARY = 200
MAX_TIMELINE_NAME = 500
MAX_PATH = 1024
MAX_HOST = 200
MAX_NOTES = 8000
MAX_SOURCES = 64
MAX_DELIVERABLES = 32
MAX_NAME = 64  # recipe and profile names
MAX_SEGMENT = 64  # deliverable names (also the <...> part of a facts key)

MAX_PARAMS_BYTES = 64 * 1024  # canonical JSON of a recipe's params
MAX_PARAMS_DEPTH = 8
MAX_PARAMS_ITEMS = 5_000
MAX_PLAN_BYTES = 1024 * 1024  # canonical JSON of an EditPlan
MAX_PLAN_DEPTH = 32
MAX_PLAN_ITEMS = 200_000
MAX_JSON_STRING = 100_000  # any one string inside params or a plan
MAX_JSON_INT = 2**63  # |int| bound inside params or a plan

MIN_REVIEW_ROUNDS = 1
MAX_REVIEW_ROUNDS = 10
DEFAULT_REVIEW_ROUNDS = 3

MAX_LINT_FINDINGS = 20  # findings kept on a plan node, errors first
MAX_FINDING_TEXT = 500
MAX_FINDING_REFS = 4
MAX_ERRORS_LISTED = 50  # validation errors listed in one WriteError

NAME_RE = r"^[a-z0-9][a-z0-9_-]{0,63}$"
SEGMENT_RE = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
RESOLVE_ID_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
SHA256_RE = r"^[0-9a-f]{64}$"
_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

AGENT_OWNABLE_KINDS = frozenset({"brief", "recipe", "plan", "op"})
PLAN_PARENT_KINDS = frozenset({"brief", "recipe", "plan", "review"})

ERROR_CODES = frozenset(
    {
        "invalid_input",
        "invalid_brief",
        "invalid_recipe",
        "invalid_plan",
        "invalid_params",
        "invalid_fact",
        "unknown_project",
        "unknown_parent",
        "wrong_parent_kind",
        "untyped_parent",
        "unknown_node",
        "unknown_profile",
        "profile_mismatch",
        "recipe_mismatch",
        "not_creator",
        "kind_not_allowed",
        "protected_review",
        "illegal_transition",
        "idempotency_conflict",
        "unknown_namespace",
        "unknown_fact",
        "unknown_source",
        "too_many_facts",
    }
)


class WriteError(Exception):
    """A refused hub write: ``{code, message, fix_hint, details}``.

    ``code`` is stable and machine-readable (one of :data:`ERROR_CODES`); ``fix_hint``
    says what to change; ``details`` carries the specifics (field locations, ids).
    """

    def __init__(self, code: str, message: str, fix_hint: str, **details: Any) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown write error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.fix_hint = fix_hint
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What a write returns: the op, whether it was a replay, and the entities written."""

    op: str
    replayed: bool
    project_id: str
    body: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "op": self.op,
            "replayed": self.replayed,
            "project_id": self.project_id,
            **self.body,
        }


# -- typed node data -------------------------------------------------------------


def _text(max_len: int, *, multiline: bool = False) -> AfterValidator:
    """Strip, then refuse control characters, empty text and anything over ``max_len``."""

    def check(value: str) -> str:
        try:
            return clean_text(value, "text", max_len=max_len, multiline=multiline)
        except InvalidInput as exc:
            raise ValueError(str(exc).removeprefix("text: ")) from None

    return AfterValidator(check)


_STRIP_SLACK = 256  # whitespace a value may carry around its text before it is stripped


def bounded_text(max_len: int, *, multiline: bool = False) -> Any:
    """A stripped string type of at most ``max_len`` characters, no control characters."""
    # The raw max_length keeps a huge value cheap to refuse; the exact bound is after strip.
    return Annotated[
        str, Field(max_length=max_len + _STRIP_SLACK), _text(max_len, multiline=multiline)
    ]


ShowText = bounded_text(MAX_SHOW)
TitleText = bounded_text(MAX_TITLE)
HookText = bounded_text(MAX_HOOK)
HandleText = bounded_text(MAX_HANDLE)
LabelText = bounded_text(MAX_LABEL)
LibraryText = bounded_text(MAX_LIBRARY)
TimelineNameText = bounded_text(MAX_TIMELINE_NAME)
PathText = bounded_text(MAX_PATH)
HostText = bounded_text(MAX_HOST)
NotesText = bounded_text(MAX_NOTES, multiline=True)
Name = Annotated[str, Field(pattern=NAME_RE)]
Segment = Annotated[str, Field(pattern=SEGMENT_RE)]
ResolveId = Annotated[str, Field(pattern=RESOLVE_ID_RE)]
Sha256 = Annotated[str, Field(pattern=SHA256_RE)]
NodeId = Annotated[str, Field(min_length=1, max_length=MAX_ID)]


class StrictModel(BaseModel):
    """Strict models: no coercion, no unknown keys, no NaN or infinity."""

    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, populate_by_name=False
    )

    def dump(self) -> dict[str, Any]:
        """JSON-ready data, with ``schema`` under its own name."""
        return self.model_dump(mode="json", by_alias=True)


class CloudProjectRef(StrictModel):
    """A Blackmagic Cloud project, by library and project name."""

    kind: Literal["cloud_project"]
    library: LibraryText
    project: TitleText
    label: LabelText | None = None


class TimelineSourceRef(StrictModel):
    """A timeline, by its Resolve unique id (``Timeline.GetUniqueId()``). ``name`` is a
    display cross-check only; the id is what identifies it."""

    kind: Literal["timeline"]
    unique_id: ResolveId
    name: TimelineNameText | None = None
    label: LabelText | None = None


class MediaRef(StrictModel):
    """A media file: its path, optionally the host it is on and its sha256."""

    kind: Literal["media"]
    path: PathText
    host: HostText | None = None
    sha256: Sha256 | None = None
    label: LabelText | None = None


SourceRef = Annotated[CloudProjectRef | TimelineSourceRef | MediaRef, Field(discriminator="kind")]

DeliverableKind = Literal["master", "review", "short", "thumbnail", "captions"]


class Deliverable(StrictModel):
    """One output the brief asks for. ``name`` is unique in the brief and may be used in
    facts keys (``render.hash.<name>``, ``qa.<metric>.<name>``)."""

    name: Segment
    kind: DeliverableKind


class Brief(StrictModel):
    """``aar.brief/1``: what to make, for which show, from which sources.

    Bounds: show, title 200 characters; hook 300; handle 100; notes 8000 (newlines
    allowed); at most 64 sources and 1 to 32 deliverables with unique names; recipe and
    rule profile names match ``[a-z0-9][a-z0-9_-]{0,63}``; deliverable names
    ``[A-Za-z0-9][A-Za-z0-9_-]{0,63}``.
    """

    schema_: Literal["aar.brief/1"] = Field(default=SCHEMA_BRIEF, alias="schema")
    project_id: NodeId
    show: ShowText
    kind: Literal["longform", "short", "thumbnail"]
    recipe: Name
    rule_profile: Name
    sources: list[SourceRef] = Field(default_factory=list, max_length=MAX_SOURCES)
    deliverables: list[Deliverable] = Field(min_length=1, max_length=MAX_DELIVERABLES)
    hook: HookText | None = None
    handle: HandleText | None = None
    title: TitleText | None = None
    notes: NotesText | None = None

    @model_validator(mode="after")
    def _unique_deliverables(self) -> Brief:
        seen: set[str] = set()
        for deliverable in self.deliverables:
            if deliverable.name in seen:
                raise ValueError(f"deliverable name {deliverable.name!r} is used twice")
            seen.add(deliverable.name)
        return self


class RecipeData(StrictModel):
    """``aar.recipe/1``: one recipe run derived from a brief.

    ``params`` is a JSON object of at most 64 KiB (canonical), nested at most 8 deep, with
    finite numbers. Whether the params fit the named recipe is the recipe's own validator's
    call (:data:`ParamsValidator`); the default accepts any bounded object.
    """

    schema_: Literal["aar.recipe/1"] = Field(default=SCHEMA_RECIPE, alias="schema")
    recipe: Name
    brief_id: NodeId
    params: dict[str, Any] = Field(default_factory=dict)
    max_review_rounds: int = Field(
        default=DEFAULT_REVIEW_ROUNDS, ge=MIN_REVIEW_ROUNDS, le=MAX_REVIEW_ROUNDS
    )

    @model_validator(mode="after")
    def _bounded_params(self) -> RecipeData:
        problem = json_problem(
            self.params,
            max_depth=MAX_PARAMS_DEPTH,
            max_items=MAX_PARAMS_ITEMS,
            max_bytes=MAX_PARAMS_BYTES,
        )
        if problem is not None:
            loc, message = problem
            raise ValueError(f"params{loc}: {message}")
        return self


class FindingSummary(StrictModel):
    """One lint finding as kept on a plan node (text capped at 500, at most 4 refs)."""

    rule: Annotated[str, Field(max_length=MAX_NAME * 2)]
    severity: Literal["error", "warn", "info"]
    message: Annotated[str, Field(max_length=MAX_FINDING_TEXT)]
    fix_hint: Annotated[str, Field(max_length=MAX_FINDING_TEXT)]
    refs: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_FINDING_REFS)


class LintSummary(StrictModel):
    """A plan's lint result: counts, the rules that could not run, the first findings."""

    ok: bool
    errors: int = Field(ge=0)
    warnings: int = Field(ge=0)
    infos: int = Field(ge=0)
    skipped_rules: list[Annotated[str, Field(max_length=MAX_NAME * 2)]] = Field(
        default_factory=list, max_length=200
    )
    findings: list[FindingSummary] = Field(default_factory=list, max_length=MAX_LINT_FINDINGS)
    omitted: int = Field(ge=0, description="Findings not listed (the list keeps errors first).")


class PlanNodeData(StrictModel):
    """``aar.plan-node/1``: an EditPlan submitted to the hub.

    ``plan_sha256`` is the artifact holding the plan's canonical JSON. ``brief_id`` is the
    typed brief the plan was resolved to through its parent (``None`` when the ancestry is
    untyped); when set, ``rule_profile`` equals that brief's.
    """

    schema_: Literal["aar.plan-node/1"] = Field(default=SCHEMA_PLAN_NODE, alias="schema")
    plan_sha256: Sha256
    plan_version: int = Field(ge=1)
    rule_profile: Name
    derives_from: NodeId
    brief_id: NodeId | None = None
    lint: LintSummary


ParamProblem = tuple[str, str]
"""``(location, message)`` for one bad recipe parameter, e.g. ``("params.fps", "...")``."""

ParamsValidator = Callable[[str, Mapping[str, Any], Brief], Sequence[ParamProblem]]
"""Checks a recipe's params against its definition: ``(recipe, params, brief) -> problems``.

An empty result accepts. S13g2 plugs recipe definitions in here."""


def accept_any_params(
    recipe: str, params: Mapping[str, Any], brief: Brief
) -> Sequence[ParamProblem]:
    """The default :data:`ParamsValidator`: any bounded JSON object is accepted."""
    return ()


# -- JSON bounds -------------------------------------------------------------------


def json_problem(
    value: Any, *, max_depth: int, max_items: int, max_bytes: int
) -> tuple[str, str] | None:
    """``(location, why)`` if ``value`` is not bounded JSON data, else ``None``.

    Walks iteratively (a deep value cannot hit the recursion limit): only objects with
    string keys, lists, strings, finite numbers, booleans and null; at most ``max_depth``
    levels and ``max_items`` values; strings up to :data:`MAX_JSON_STRING` characters and
    integers below :data:`MAX_JSON_INT` in size; and at most ``max_bytes`` of canonical JSON.
    """
    stack: list[tuple[Any, int, str]] = [(value, 0, "")]
    items = 0
    while stack:
        current, depth, loc = stack.pop()
        items += 1
        if items > max_items:
            return loc, f"has more than {max_items} values"
        if isinstance(current, dict):
            if depth >= max_depth:
                return loc, f"is nested deeper than {max_depth} levels"
            for key, item in current.items():
                if not isinstance(key, str):
                    return loc, "has a key that is not a string"
                if len(key) > MAX_JSON_STRING:
                    return loc, f"has a key longer than {MAX_JSON_STRING} characters"
                stack.append((item, depth + 1, f"{loc}.{key}"))
        elif isinstance(current, list | tuple):
            if depth >= max_depth:
                return loc, f"is nested deeper than {max_depth} levels"
            stack.extend((item, depth + 1, f"{loc}[{i}]") for i, item in enumerate(current))
        elif isinstance(current, str):
            if len(current) > MAX_JSON_STRING:
                return loc, f"is a string longer than {MAX_JSON_STRING} characters"
        elif isinstance(current, bool) or current is None:
            continue
        elif isinstance(current, int):
            if abs(current) >= MAX_JSON_INT:
                return loc, "is an integer too large to store"
        elif isinstance(current, float):
            if not math.isfinite(current):
                return loc, "is not a finite number"
        else:
            return loc, f"is a {type(current).__name__}, not JSON data"
    size = len(canonical_json(value).encode("utf-8"))
    if size > max_bytes:
        return "", f"is {size} bytes of JSON, over the {max_bytes}-byte limit"
    return None


# -- shared write machinery -----------------------------------------------------------


def write_key(actor: Actor, idempotency_key: Any) -> str:
    """The log key for a caller's ``idempotency_key``: namespaced per actor, unambiguous.

    ``aar.write/["agent","ed","brief-1"]``: JSON-encoding the triple means an id or key
    that contains ``:`` or ``/`` can never make two actors' keys collide.
    """
    if not isinstance(idempotency_key, str) or not _KEY_RE.fullmatch(idempotency_key):
        raise WriteError(
            "invalid_input",
            f"idempotency_key must be 1-{MAX_KEY} characters of A-Z a-z 0-9 . _ : / - "
            f"starting with a letter or digit, got {idempotency_key!r:.80}",
            "Choose a key that names this one request, e.g. 'brief-qcce-2024-longform'. "
            "Send the same key again only to retry the same request.",
            field="idempotency_key",
        )
    return "aar.write/" + json.dumps([actor.kind, actor.id, idempotency_key])


def check_actor(actor: Any) -> Actor:
    """``actor`` if it is an :class:`Actor` a write accepts, else ``invalid_input``.

    Agent ids follow the agent-branch rule (``aar.node.branches.valid_agent_id``). In v1
    an agent's id is self-declared: attribution, not authentication (DESIGN decision 6).
    """
    if not isinstance(actor, Actor):
        raise TypeError(f"actor must be an Actor, got {type(actor).__name__}")
    if actor.kind == "agent" and not valid_agent_id(actor.id):
        raise WriteError(
            "invalid_input",
            f"agent id {actor.id!r:.80} is not a valid agent id",
            f"Use 1-{MAX_ID} characters of A-Z a-z 0-9 . _ - (the agent-branch id rule).",
            field="agent_id",
        )
    if len(actor.id) > MAX_ID:
        raise WriteError(
            "invalid_input",
            f"actor id is over {MAX_ID} characters",
            "Use a shorter id.",
            field="actor",
        )
    return actor


def check_id(value: Any, field_name: str) -> str:
    """A node or project id (1-128 characters), else ``invalid_input``."""
    if not isinstance(value, str) or not 0 < len(value) <= MAX_ID:
        raise WriteError(
            "invalid_input",
            f"{field_name} must be an id of 1-{MAX_ID} characters, got {value!r:.80}",
            f"Pass the {field_name} exactly as the hub returned it.",
            field=field_name,
        )
    return value


def request_sha256(op: str, request: Mapping[str, Any]) -> str:
    """The fingerprint of a normalised write request (what idempotency compares)."""
    return hashlib.sha256(canonical_json({"op": op, **request}).encode("utf-8")).hexdigest()


def validation_details(exc: ValidationError) -> dict[str, Any]:
    """``{"errors": [{loc, msg, type}], "omitted": n}`` naming each bad field location."""
    errors = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        errors.append(
            {"loc": _dotted(error["loc"]), "msg": str(error["msg"]), "type": error["type"]}
        )
    return {
        "errors": errors[:MAX_ERRORS_LISTED],
        "omitted": max(0, len(errors) - MAX_ERRORS_LISTED),
    }


def _dotted(loc: Sequence[Any]) -> str:
    """``("sources", 0, "path")`` as ``sources[0].path``."""
    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else str(part)
    return text


def _unknown_project(project_id: str) -> WriteError:
    return WriteError(
        "unknown_project",
        f"no project {project_id!r} in the hub registry",
        "Call projects_list (or `aar hub` registry tools) and use a project's id (prj_...).",
        project_id=project_id,
    )


def require_project(log: EventLog, project_id: str) -> None:
    """Refuse with ``unknown_project`` unless the registry knows ``project_id``."""
    if Registry(log).get(project_id) is None:
        raise _unknown_project(project_id)


def _conflict(idempotency_key: str, why: str) -> WriteError:
    return WriteError(
        "idempotency_conflict",
        f"idempotency key {idempotency_key!r} was already used for a different request ({why})",
        "Use a new key for a new request. Reuse a key only to retry exactly the same request.",
        idempotency_key=idempotency_key,
    )


def _find_record(
    reader: Transaction | EventLog,
    key: str,
    idempotency_key: str,
    op: str,
    project_id: str,
    actor: Actor,
    fingerprint: str,
) -> Event | None:
    """The write record of an earlier identical request, ``None``, or a conflict."""
    if isinstance(reader, Transaction):
        existing = reader.find_by_idempotency_key(key)
    else:
        existing = reader.get_by_idempotency_key(key)
    if existing is None:
        return None
    payload = existing.payload if isinstance(existing.payload, dict) else {}
    if existing.type != EVENT_WRITE or payload.get("op") != op:
        raise _conflict(idempotency_key, f"it was used for {payload.get('op') or existing.type!r}")
    if existing.project_id != project_id:
        raise _conflict(idempotency_key, f"it was used in project {existing.project_id!r}")
    if existing.actor != actor or payload.get("request_sha256") != fingerprint:
        raise _conflict(idempotency_key, "the request differs")
    return existing


def _keyed(reader: Transaction | EventLog, key: str, event_type: str) -> Event:
    """The event a write record's sub-key names; a conflict if it is missing or wrong."""
    if isinstance(reader, Transaction):
        event = reader.find_by_idempotency_key(key)
    else:
        event = reader.get_by_idempotency_key(key)
    if event is None or event.type != event_type or not isinstance(event.payload, dict):
        raise WriteError(
            "idempotency_conflict",
            "the write record for this key names events the hub cannot find",
            "Use a new idempotency key.",
            idempotency_key=key,
        )
    return event


def _node_from(event: Event) -> dict[str, Any]:
    """A node as written by ``event`` (its creation): the same dict on a replay."""
    payload = event.payload
    return GraphNode(
        id=payload["node_id"],
        kind=payload["kind"],
        status=payload["status"],
        title=payload.get("title"),
        payload=payload.get("data", {}),
        created=event.ts,
        updated=event.ts,
    ).to_dict()


def _edge_from(event: Event) -> dict[str, Any]:
    payload = event.payload
    return Edge(
        id=payload["edge_id"],
        type=payload["type"],
        src=payload["src"],
        dst=payload["dst"],
        payload=payload.get("data", {}),
        created=event.ts,
    ).to_dict()


def _graph_body(reader: Transaction | EventLog, key: str, with_edge: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"node": _node_from(_keyed(reader, f"{key}#node", EVENT_NODE_ADDED))}
    if with_edge:
        body["edge"] = _edge_from(_keyed(reader, f"{key}#edge", EVENT_EDGE_ADDED))
    return body


def _append_record(
    txn: Transaction,
    key: str,
    op: str,
    project_id: str,
    actor: Actor,
    fingerprint: str,
    **refs: Any,
) -> None:
    txn.append(
        EVENT_WRITE,
        actor,
        {"op": op, "request_sha256": fingerprint, **refs},
        project_id=project_id,
        idempotency_key=key,
    )


@contextmanager
def _in_transaction(log: EventLog) -> Iterator[Transaction]:
    """``log.transaction()``, with a key collision on a sub-event reported as a conflict.

    The ``WriteError`` is raised inside the transaction, so it rolls back.
    """
    with log.transaction() as txn:
        try:
            yield txn
        except IdempotencyConflict as conflict:
            raise WriteError(
                "idempotency_conflict",
                f"a key this write needs is already taken: {conflict}",
                "Use a new idempotency key.",
            ) from conflict


# -- profiles and briefs ----------------------------------------------------------------


def _load_named_profile(name: Any, field_name: str) -> Profile:
    """A built-in rule profile by name, or ``unknown_profile``. Paths are never accepted."""
    if not isinstance(name, str) or not re.fullmatch(NAME_RE, name):
        raise WriteError(
            "unknown_profile",
            f"{field_name} must be a built-in rule profile name, got {name!r:.80}",
            "Use a profile name such as 'ccn'. Paths to profile files are not accepted by the hub.",
            field=field_name,
        )
    try:
        return load_profile(name)
    except ProfileError as exc:
        raise WriteError(
            "unknown_profile",
            f"rule profile {name!r} cannot be loaded: {exc}",
            "Use the name of a profile that ships with aar (for example 'ccn').",
            field=field_name,
        ) from exc


def typed_brief(node: GraphNode | None) -> Brief | None:
    """The typed brief a node holds, or ``None`` (not a brief, or untyped data)."""
    if node is None or node.kind != "brief" or not isinstance(node.payload, dict):
        return None
    try:
        return Brief.model_validate(node.payload)
    except ValidationError:
        return None


def typed_recipe(node: GraphNode | None) -> RecipeData | None:
    """The typed recipe data a node holds, or ``None``."""
    if node is None or node.kind != "recipe" or not isinstance(node.payload, dict):
        return None
    try:
        return RecipeData.model_validate(node.payload)
    except ValidationError:
        return None


def typed_plan(node: GraphNode | None) -> PlanNodeData | None:
    """The typed plan-node data a node holds, or ``None``."""
    if node is None or node.kind != "plan" or not isinstance(node.payload, dict):
        return None
    try:
        return PlanNodeData.model_validate(node.payload)
    except ValidationError:
        return None


def brief_behind(graph: WorkGraph, node: GraphNode) -> str | None:
    """The typed brief a plan parent resolves to, or ``None`` if its ancestry is untyped.

    A brief is itself; a recipe names its brief; a plan node recorded its brief; a review
    resolves through the node it reviews (one level: a review never reviews a review).
    """
    if node.kind == "review":
        subject_id = node.payload.get("subject") if isinstance(node.payload, dict) else None
        subject = graph.node(subject_id) if isinstance(subject_id, str) else None
        if subject is None or subject.kind == "review":
            return None
        node = subject
    if node.kind == "brief":
        return node.id if typed_brief(node) is not None else None
    if node.kind == "recipe":
        recipe = typed_recipe(node)
        brief_id = recipe.brief_id if recipe is not None else None
    elif node.kind == "plan":
        plan = typed_plan(node)
        brief_id = plan.brief_id if plan is not None else None
    else:
        return None
    if brief_id is None or typed_brief(graph.node(brief_id)) is None:
        return None
    return brief_id


# -- writes --------------------------------------------------------------------------------


def create_brief(
    log: EventLog,
    brief: Mapping[str, Any] | Brief,
    actor: Actor,
    *,
    idempotency_key: str,
) -> WriteResult:
    """Create a ``brief`` node (status ``pending``) holding a typed :class:`Brief`.

    Refused with ``invalid_brief`` (each bad field named), ``unknown_profile`` (the rule
    profile is not a loadable built-in name), ``unknown_project`` or
    ``idempotency_conflict``.
    """
    actor = check_actor(actor)
    key = write_key(actor, idempotency_key)
    if isinstance(brief, Brief):
        brief = brief.dump()
    if not isinstance(brief, Mapping):
        raise WriteError(
            "invalid_brief",
            f"brief must be a JSON object, got {type(brief).__name__}",
            "Send the brief as an object with project_id, show, kind, recipe, rule_profile "
            "and deliverables.",
        )
    try:
        model = Brief.model_validate(dict(brief))
    except ValidationError as exc:
        raise WriteError(
            "invalid_brief",
            f"the brief failed validation ({exc.error_count()} problem(s))",
            "Fix each field listed in details.errors (loc names the field).",
            **validation_details(exc),
        ) from None
    _load_named_profile(model.rule_profile, "rule_profile")
    data = model.dump()
    project_id = model.project_id
    fingerprint = request_sha256(OP_BRIEF_CREATE, {"project_id": project_id, "brief": data})
    title = model.title or f"{model.show} {model.kind}"
    with _in_transaction(log) as txn:
        record = _find_record(
            txn, key, idempotency_key, OP_BRIEF_CREATE, project_id, actor, fingerprint
        )
        if record is not None:
            return WriteResult(OP_BRIEF_CREATE, True, project_id, _graph_body(txn, key, False))
        require_project(log, project_id)
        graph = WorkGraph(log, project_id)
        node_event = graph.add_node_in(
            txn, "brief", actor, title=title, data=data, idempotency_key=f"{key}#node"
        )
        _append_record(
            txn,
            key,
            OP_BRIEF_CREATE,
            project_id,
            actor,
            fingerprint,
            node_id=node_event.payload["node_id"],
        )
    return WriteResult(OP_BRIEF_CREATE, False, project_id, {"node": _node_from(node_event)})


def create_recipe(
    log: EventLog,
    project_id: str,
    recipe: Mapping[str, Any],
    actor: Actor,
    *,
    idempotency_key: str,
    params_validator: ParamsValidator = accept_any_params,
) -> WriteResult:
    """Create a ``recipe`` node (``pending``) with a ``derives_from`` edge to its brief.

    ``recipe`` is ``{recipe, brief_id, params?, max_review_rounds?}``. The brief must be a
    typed brief in the project whose ``recipe`` is this recipe's name. ``params_validator``
    checks the params against the recipe's definition (default: any bounded object).
    """
    actor = check_actor(actor)
    key = write_key(actor, idempotency_key)
    project_id = check_id(project_id, "project_id")
    if not isinstance(recipe, Mapping):
        raise WriteError(
            "invalid_recipe",
            f"recipe must be a JSON object, got {type(recipe).__name__}",
            "Send {recipe, brief_id, params, max_review_rounds}.",
        )
    try:
        model = RecipeData.model_validate(dict(recipe))
    except ValidationError as exc:
        raise WriteError(
            "invalid_recipe",
            f"the recipe failed validation ({exc.error_count()} problem(s))",
            "Fix each field listed in details.errors (loc names the field).",
            **validation_details(exc),
        ) from None
    data = model.dump()
    fingerprint = request_sha256(OP_RECIPE_CREATE, {"project_id": project_id, "recipe": data})
    record = _find_record(
        log, key, idempotency_key, OP_RECIPE_CREATE, project_id, actor, fingerprint
    )
    if record is not None:
        return WriteResult(OP_RECIPE_CREATE, True, project_id, _graph_body(log, key, True))
    require_project(log, project_id)
    brief = _recipe_brief(WorkGraph(log, project_id), model)
    problems = list(params_validator(model.recipe, model.params, brief))
    if problems:
        raise WriteError(
            "invalid_params",
            f"recipe {model.recipe!r} refused {len(problems)} parameter problem(s)",
            "Fix each parameter listed in details.problems.",
            problems=[{"loc": loc, "msg": msg} for loc, msg in problems[:MAX_ERRORS_LISTED]],
            omitted=max(0, len(problems) - MAX_ERRORS_LISTED),
        )
    with _in_transaction(log) as txn:
        record = _find_record(
            txn, key, idempotency_key, OP_RECIPE_CREATE, project_id, actor, fingerprint
        )
        if record is not None:
            return WriteResult(OP_RECIPE_CREATE, True, project_id, _graph_body(txn, key, True))
        graph = WorkGraph(log, project_id)
        _recipe_brief(graph, model)
        node_event = graph.add_node_in(
            txn,
            "recipe",
            actor,
            title=f"recipe {model.recipe}",
            data=data,
            idempotency_key=f"{key}#node",
        )
        node_id = node_event.payload["node_id"]
        edge_event = graph.add_edge_in(
            txn, "derives_from", node_id, model.brief_id, actor, idempotency_key=f"{key}#edge"
        )
        _append_record(
            txn,
            key,
            OP_RECIPE_CREATE,
            project_id,
            actor,
            fingerprint,
            node_id=node_id,
            edge_id=edge_event.payload["edge_id"],
        )
    return WriteResult(
        OP_RECIPE_CREATE,
        False,
        project_id,
        {"node": _node_from(node_event), "edge": _edge_from(edge_event)},
    )


def _recipe_brief(graph: WorkGraph, model: RecipeData) -> Brief:
    """The typed brief a recipe derives from, or the refusal that says why not."""
    node = graph.node(model.brief_id)
    if node is None:
        raise WriteError(
            "unknown_parent",
            f"no node {model.brief_id!r} in project {graph.project_id!r}",
            "Pass the id of a brief node in this project (brief_create returns it).",
            brief_id=model.brief_id,
        )
    if node.kind != "brief":
        raise WriteError(
            "wrong_parent_kind",
            f"node {model.brief_id!r} is a {node.kind}, not a brief",
            "A recipe derives from a brief: pass a brief node's id.",
            brief_id=model.brief_id,
            kind=node.kind,
        )
    brief = typed_brief(node)
    if brief is None:
        raise WriteError(
            "untyped_parent",
            f"brief {model.brief_id!r} has no typed {SCHEMA_BRIEF} data",
            "Create the brief with brief_create, then derive the recipe from that brief.",
            brief_id=model.brief_id,
        )
    if brief.recipe != model.recipe:
        raise WriteError(
            "recipe_mismatch",
            f"brief {model.brief_id!r} asks for recipe {brief.recipe!r}, not {model.recipe!r}",
            f"Set recipe to {brief.recipe!r}, or create a brief for {model.recipe!r}.",
            brief_id=model.brief_id,
            brief_recipe=brief.recipe,
            recipe=model.recipe,
        )
    return brief


def _check_plan_parent(graph: WorkGraph, parent_id: str, plan_profile: str) -> str | None:
    """Refuse a parent that cannot carry a plan; return the brief it resolves to."""
    parent = graph.node(parent_id)
    if parent is None:
        raise WriteError(
            "unknown_parent",
            f"no node {parent_id!r} in project {graph.project_id!r}",
            "Pass the id of the brief, recipe, earlier plan or review this plan derives from.",
            derives_from=parent_id,
        )
    if parent.kind not in PLAN_PARENT_KINDS:
        raise WriteError(
            "wrong_parent_kind",
            f"a plan cannot derive from a {parent.kind} node ({parent_id!r})",
            "Derive the plan from a brief, a recipe, an earlier plan or a review.",
            derives_from=parent_id,
            kind=parent.kind,
            allowed=sorted(PLAN_PARENT_KINDS),
        )
    brief_id = brief_behind(graph, parent)
    if brief_id is not None:
        brief = typed_brief(graph.node(brief_id))
        assert brief is not None  # brief_behind returns only typed briefs
        if brief.rule_profile != plan_profile:
            raise WriteError(
                "profile_mismatch",
                f"the plan names rule profile {plan_profile!r}, but brief {brief_id!r} "
                f"requires {brief.rule_profile!r}",
                f"Set the plan's profile to {brief.rule_profile!r}: a plan is linted under "
                "its brief's house rules.",
                brief_id=brief_id,
                plan_profile=plan_profile,
                brief_profile=brief.rule_profile,
            )
    return brief_id


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def lint_summary(report: LintReport, profile: Profile) -> LintSummary:
    """A bounded summary of a lint report: counts, skipped rules, errors first.

    The hub lints without a transcript or laugh map, so every enabled rule that needs one
    is listed in ``skipped_rules``: a plan with no errors has not passed those rules.
    """
    rank = {"error": 0, "warn": 1, "info": 2}
    ordered = sorted(enumerate(report.findings), key=lambda pair: (rank[pair[1].severity], pair[0]))
    kept = [finding for _, finding in ordered[:MAX_LINT_FINDINGS]]
    known = all_rules()
    skipped = sorted(
        rule_id
        for rule_id, config in profile.rules.items()
        if config.enabled and rule_id in known and known[rule_id].needs
    )
    summary = LintSummary(
        ok=report.counts.get("error", 0) == 0,
        errors=report.counts.get("error", 0),
        warnings=report.counts.get("warn", 0),
        infos=report.counts.get("info", 0),
        skipped_rules=skipped,
        findings=[
            FindingSummary(
                rule=_clip(finding.rule, MAX_NAME * 2),
                severity=finding.severity,
                message=_clip(finding.message, MAX_FINDING_TEXT),
                fix_hint=_clip(finding.fix_hint, MAX_FINDING_TEXT),
                refs=[
                    ref.model_dump(mode="json", exclude_none=True)
                    for ref in finding.refs[:MAX_FINDING_REFS]
                ],
            )
            for finding in kept
        ],
        omitted=len(report.findings) - len(kept),
    )
    return summary


def submit_plan(
    log: EventLog,
    store: ArtifactStore,
    project_id: str,
    plan: Mapping[str, Any],
    actor: Actor,
    *,
    derives_from: str,
    idempotency_key: str,
    title: str | None = None,
) -> WriteResult:
    """Lint an EditPlan, store it as an artifact and add a ``plan`` node for it.

    The plan is parsed with ``aar.plan`` (any plan version it accepts) and linted under its
    own profile, which must be a built-in name and, when the parent resolves to a typed
    brief, that brief's ``rule_profile``. A plan that fails validation is refused
    (``invalid_plan``, each field location named); one with lint errors is stored as
    ``blocked``, one without as ``pending``. ``derives_from`` must be a brief, recipe, plan
    or review node of the project.
    """
    actor = check_actor(actor)
    key = write_key(actor, idempotency_key)
    project_id = check_id(project_id, "project_id")
    derives_from = check_id(derives_from, "derives_from")
    if title is not None:
        try:
            title = clean_text(title, "title", max_len=MAX_TITLE)
        except InvalidInput as exc:
            raise WriteError(
                "invalid_input", str(exc), "Shorten the title.", field="title"
            ) from None
    if not isinstance(plan, Mapping):
        raise WriteError(
            "invalid_plan",
            f"plan must be a JSON object (an EditPlan), got {type(plan).__name__}",
            "Send the EditPlan object itself (`aar plan schema` prints its JSON Schema).",
        )
    problem = json_problem(
        dict(plan), max_depth=MAX_PLAN_DEPTH, max_items=MAX_PLAN_ITEMS, max_bytes=MAX_PLAN_BYTES
    )
    if problem is not None:
        loc, why = problem
        raise WriteError(
            "invalid_plan",
            f"plan{loc} {why}",
            f"Send plain JSON with finite numbers, at most {MAX_PLAN_BYTES} bytes.",
            errors=[{"loc": loc.lstrip(".") or "plan", "msg": why, "type": "bounds"}],
            omitted=0,
        )
    try:
        parsed = EditPlan.model_validate(dict(plan))
    except ValidationError as exc:
        raise WriteError(
            "invalid_plan",
            f"the plan failed EditPlan validation ({exc.error_count()} problem(s))",
            "Fix each field listed in details.errors (loc names the field); "
            "`aar plan schema` prints the EditPlan JSON Schema.",
            **validation_details(exc),
        ) from None
    profile = _load_named_profile(parsed.profile, "plan.profile")
    try:
        report = lint(parsed, profile)
    except ProfileError as exc:  # a built-in profile the linter rejects: refuse, never store
        raise WriteError(
            "unknown_profile",
            f"rule profile {parsed.profile!r} cannot lint: {exc}",
            "Use another built-in profile name.",
            field="plan.profile",
        ) from exc
    content = canonical_json(parsed.model_dump(mode="json")).encode("utf-8")
    if len(content) > MAX_PLAN_BYTES:
        raise WriteError(
            "invalid_plan",
            f"the plan is {len(content)} bytes of canonical JSON, over {MAX_PLAN_BYTES}",
            "Split the build into smaller plans.",
            errors=[{"loc": "plan", "msg": "too large", "type": "bounds"}],
            omitted=0,
        )
    plan_sha = hashlib.sha256(content).hexdigest()
    request = {
        "project_id": project_id,
        "derives_from": derives_from,
        "plan_sha256": plan_sha,
        "title": title,
    }
    fingerprint = request_sha256(OP_PLAN_SUBMIT, request)
    if _find_record(log, key, idempotency_key, OP_PLAN_SUBMIT, project_id, actor, fingerprint):
        return WriteResult(OP_PLAN_SUBMIT, True, project_id, _plan_body(log, key, plan_sha))
    # Refuse what can be refused before storing anything; nodes and projects are never
    # removed, so these checks still hold when the transaction re-runs them.
    require_project(log, project_id)
    _check_plan_parent(WorkGraph(log, project_id), derives_from, parsed.profile)
    stored = store.put(
        content,
        actor=actor,
        mime="application/json",
        project_id=project_id,
        meta={"schema": "aar.edit-plan", "version": parsed.version},
    )
    if stored != plan_sha:  # pragma: no cover - the store hashes the same bytes
        raise RuntimeError("artifact store returned a different hash for the plan")
    status = "pending" if report.ok else "blocked"
    with _in_transaction(log) as txn:
        record = _find_record(
            txn, key, idempotency_key, OP_PLAN_SUBMIT, project_id, actor, fingerprint
        )
        if record is not None:
            return WriteResult(OP_PLAN_SUBMIT, True, project_id, _plan_body(txn, key, plan_sha))
        graph = WorkGraph(log, project_id)
        brief_id = _check_plan_parent(graph, derives_from, parsed.profile)
        data = PlanNodeData(
            plan_sha256=plan_sha,
            plan_version=parsed.version,
            rule_profile=parsed.profile,
            derives_from=derives_from,
            brief_id=brief_id,
            lint=lint_summary(report, profile),
        ).dump()
        # The plan's timeline name is free text: shown as a title, it is made one line.
        derived = _CONTROL.sub(" ", f"plan v{parsed.version}: {parsed.timeline}")
        node_title = title or _clip(derived, MAX_TITLE)
        node_event = graph.add_node_in(
            txn,
            "plan",
            actor,
            title=node_title,
            status=status,
            data=data,
            idempotency_key=f"{key}#node",
        )
        node_id = node_event.payload["node_id"]
        edge_event = graph.add_edge_in(
            txn, "derives_from", node_id, derives_from, actor, idempotency_key=f"{key}#edge"
        )
        _append_record(
            txn,
            key,
            OP_PLAN_SUBMIT,
            project_id,
            actor,
            fingerprint,
            node_id=node_id,
            edge_id=edge_event.payload["edge_id"],
            artifact_id=plan_sha,
        )
    return WriteResult(
        OP_PLAN_SUBMIT,
        False,
        project_id,
        {"node": _node_from(node_event), "edge": _edge_from(edge_event), "artifact_id": plan_sha},
    )


def _plan_body(reader: Transaction | EventLog, key: str, plan_sha: str) -> dict[str, Any]:
    return {**_graph_body(reader, key, True), "artifact_id": plan_sha}


def set_own_status(
    log: EventLog,
    project_id: str,
    node_id: str,
    status: str,
    actor: Actor,
    *,
    idempotency_key: str,
) -> WriteResult:
    """Move a node's status, if ``actor`` may.

    A non-human actor may move only brief, recipe, plan and op nodes it created; a human
    may move any node except a review. Review nodes are never moved here (they close by a
    human decision or the requester's withdrawal, ``aar.hub.reviews``). The move must be
    in the graph's transition table.
    """
    actor = check_actor(actor)
    key = write_key(actor, idempotency_key)
    project_id = check_id(project_id, "project_id")
    node_id = check_id(node_id, "node_id")
    if not isinstance(status, str) or status not in STATUSES:
        raise WriteError(
            "invalid_input",
            f"status must be one of {sorted(STATUSES)}, got {status!r:.80}",
            "Pass a status name.",
            field="status",
        )
    fingerprint = request_sha256(
        OP_SET_STATUS, {"project_id": project_id, "node_id": node_id, "to": status}
    )
    with _in_transaction(log) as txn:
        record = _find_record(
            txn, key, idempotency_key, OP_SET_STATUS, project_id, actor, fingerprint
        )
        if record is not None:
            return WriteResult(OP_SET_STATUS, True, project_id, _status_body(record.payload))
        require_project(log, project_id)
        graph = WorkGraph(log, project_id)
        node = graph.node(node_id)
        if node is None:
            raise WriteError(
                "unknown_node",
                f"no node {node_id!r} in project {project_id!r}",
                "Pass a node id from work_graph (or the id a create call returned).",
                node_id=node_id,
            )
        if node.kind == "review":
            raise WriteError(
                "protected_review",
                f"node {node_id!r} is a review: it closes only through a human decision "
                "or its requester's withdrawal",
                "Ask a human to decide it in the console, or withdraw your own review "
                "(`aar hub review withdraw`).",
                node_id=node_id,
            )
        if actor.kind != "human":
            if node.kind not in AGENT_OWNABLE_KINDS:
                raise WriteError(
                    "kind_not_allowed",
                    f"an {actor.kind} may not change the status of a {node.kind} node",
                    f"Agents move only {', '.join(sorted(AGENT_OWNABLE_KINDS))} nodes; ask a "
                    "human for anything else.",
                    node_id=node_id,
                    kind=node.kind,
                )
            creator = graph.creator(node_id)
            if creator != actor:
                raise WriteError(
                    "not_creator",
                    f"{actor.kind}:{actor.id} did not create node {node_id!r}",
                    "Only the node's creator (or a human) may change its status.",
                    node_id=node_id,
                    creator=actor_dict(creator) if creator is not None else None,
                )
        allowed = allowed_transitions(node.status)
        if status not in allowed:
            raise WriteError(
                "illegal_transition",
                f"cannot move node {node_id!r} from {node.status!r} to {status!r}",
                f"From {node.status!r} the allowed moves are "
                f"{', '.join(sorted(allowed)) or 'none (it is final)'}.",
                node_id=node_id,
                **{"from": node.status},
                to=status,
                allowed=sorted(allowed),
            )
        previous = node.status
        graph.set_status_in(txn, node_id, status, actor)
        refs = {"node_id": node_id, "kind": node.kind, "from": previous, "to": status}
        _append_record(txn, key, OP_SET_STATUS, project_id, actor, fingerprint, **refs)
    return WriteResult(OP_SET_STATUS, False, project_id, _status_body(refs))


def _status_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in ("node_id", "kind", "from", "to")}


# -- typed reads -------------------------------------------------------------------------------

MAX_DERIVED_LISTED = 100


def _derived(graph: WorkGraph, node_id: str) -> list[dict[str, Any]]:
    """The nodes that derive from ``node_id`` (at most 100), oldest first."""
    found = []
    for edge in graph.edges():
        if edge.type == "derives_from" and edge.dst == node_id:
            child = graph.node(edge.src)
            if child is not None:
                found.append({"node_id": child.id, "kind": child.kind, "status": child.status})
    return found[:MAX_DERIVED_LISTED]


def brief_view(log: EventLog, project_id: str, node_id: str) -> dict[str, Any] | None:
    """A brief node with its typed data, creator and the nodes derived from it."""
    graph = WorkGraph(log, project_id)
    node = graph.node(node_id)
    if node is None or node.kind != "brief":
        return None
    brief = typed_brief(node)
    creator = graph.creator(node_id)
    return {
        "node": node.to_dict(),
        "typed": brief is not None,
        "brief": brief.dump() if brief is not None else None,
        "creator": actor_dict(creator) if creator is not None else None,
        "derived": _derived(graph, node_id),
    }


def plan_view(
    log: EventLog,
    store: ArtifactStore,
    project_id: str,
    node_id: str,
    *,
    include_plan: bool = True,
) -> dict[str, Any] | None:
    """A plan node with its typed data, creator and (optionally) the EditPlan itself.

    The plan is read from the artifact store and checked against its hash; a missing or
    corrupt copy is reported in ``plan_error``, never served.
    """
    graph = WorkGraph(log, project_id)
    node = graph.node(node_id)
    if node is None or node.kind != "plan":
        return None
    data = typed_plan(node)
    creator = graph.creator(node_id)
    view: dict[str, Any] = {
        "node": node.to_dict(),
        "typed": data is not None,
        "creator": actor_dict(creator) if creator is not None else None,
        "derived": _derived(graph, node_id),
        "plan": None,
        "plan_error": None,
    }
    if include_plan and data is not None:
        view["plan"], view["plan_error"] = _read_plan(store, data.plan_sha256)
    return view


def _read_plan(store: ArtifactStore, sha: str) -> tuple[Any, str | None]:
    from aar.hub.artifacts import CorruptArtifact

    try:
        raw = store.read_bytes(sha)
    except FileNotFoundError:
        return None, f"the plan artifact {sha} is not in the artifact store"
    except CorruptArtifact:
        return None, f"the stored plan artifact {sha} does not match its hash"
    if len(raw) > MAX_PLAN_BYTES:
        return None, f"the plan artifact is {len(raw)} bytes, over {MAX_PLAN_BYTES}"
    try:
        return json.loads(raw), None
    except (UnicodeDecodeError, ValueError):
        return None, f"the plan artifact {sha} is not JSON"
