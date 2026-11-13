"""Lessons: one-line editing rules a human accepted after reviews, for later cut guides (S23a).

The owner reviews each cut in the console and approves or rejects it with a reason and notes at
timeline frames. That feedback drives the retry loop inside one recipe run; a **lesson** is what
carries into later runs. The ccn cut guide prints the accepted lessons that apply to its run
(:func:`lessons_for_guide`).

**Scope.** A lesson applies to every show (``global``), to one client (``client:<name>``, keyed
by the brief's ``rule_profile``, a built-in profile) or to one hub project (``project:<prj_id>``).
Where two disagree, the narrower one wins: the project over the client, the client over global.

**Events.** Lessons are global events on the shared log (``project_id`` null, like
``control.stopped`` with scope ``all`` and ``node.enrolled``); the source project and review go
in the payload.

====================  =============================  ==============  ============
event                 written by                     from            to
====================  =============================  ==============  ============
``lesson.proposed``   a human or an agent            (a new id)      ``proposed``
``lesson.added``      a human, writing one by hand   (a new id)      ``accepted``
``lesson.accepted``   a human (text, scope editable)  ``proposed``   ``accepted``
``lesson.dropped``    a human                        ``proposed``    ``dropped``
``lesson.retired``    a human                        ``accepted``    ``retired``
====================  =============================  ==============  ============

``dropped`` and ``retired`` are final. **Only a human makes a lesson count**: an agent must never
be able to write instructions into future guides. The writers refuse every other actor
(:func:`~aar.hub.common.require_human`), and the fold checks it again: an event that breaks a rule
(a human-only event by a non-human, a ``project_id``, a bad transition, a duplicate id, a payload
that fails validation) is skipped and listed, never applied. There is no CLI or MCP write path;
a person writes in the console.

**Bounds.** Text is one line (stripped, every whitespace run one space), 1-500 characters, with
no character a reader cannot see (:func:`hidden_character`: such a character could carry an
instruction past the person who accepts the lesson); a proposal's ``why`` 0-1000 and a drop or
retire reason 0-500, normalized and checked the same way. At most
:data:`MAX_PENDING` proposals wait at once, and a scope holds at most
:data:`MAX_ACCEPTED_PER_SCOPE` accepted lessons; the guide reads at most that many per scope too.

**Writes.** Each writer reads, checks and appends inside one ``BEGIN IMMEDIATE`` transaction, so
two people deciding at once cannot both win and a cap cannot be raced past. Each takes an
optional ``idempotency_key`` as the review writers do: a retry of the same request returns the
lesson as it is now; another request under the same key is refused (``idempotency_conflict``).
Errors are :class:`~aar.hub.writes.WriteError` (a stable code and a fix hint), except a human-only
write by anyone else, which is :class:`~aar.hub.common.HumanRequired`.
"""

from __future__ import annotations

import re
import secrets
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from aar.hub import reviews
from aar.hub.common import SkippedEvent, actor_dict, is_text, require_human
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    IdempotencyConflict,
    Transaction,
    match_or_conflict,
)
from aar.hub.writes import (
    MAX_ID,
    MAX_KEY,
    NAME_RE,
    WriteError,
    _conflict,
    _load_named_profile,
    check_actor,
    check_id,
    require_project,
)

EVENT_PROPOSED = "lesson.proposed"
EVENT_ADDED = "lesson.added"
EVENT_ACCEPTED = "lesson.accepted"
EVENT_DROPPED = "lesson.dropped"
EVENT_RETIRED = "lesson.retired"
LESSON_EVENT_TYPES = (EVENT_PROPOSED, EVENT_ADDED, EVENT_ACCEPTED, EVENT_DROPPED, EVENT_RETIRED)
HUMAN_ONLY_EVENTS = frozenset({EVENT_ADDED, EVENT_ACCEPTED, EVENT_DROPPED, EVENT_RETIRED})
PROPOSER_KINDS = frozenset({"human", "agent"})

STATES = ("proposed", "accepted", "dropped", "retired")

MAX_TEXT = 500
MAX_WHY = 1000
MAX_REASON = 500
MAX_PENDING = 200
MAX_ACCEPTED_PER_SCOPE = 50
#: Raw input longer than this is refused before it is normalized (whitespace runs can shrink it).
MAX_RAW = 64 * 1024

GLOBAL = "global"
LESSON_ID_RE = re.compile(r"^les_[0-9a-f]{16}$")
_PROFILE_NAME = re.compile(NAME_RE)  # the hub's rule-profile name pattern: never a path
_SPACE_RUN = re.compile(r"\s+")
#: Unicode categories a reader cannot see: control (Cc), format (Cf: zero-width, bidirectional
#: and tag characters, the soft hyphen, the byte order mark), surrogate (Cs, which no page can
#: encode), private use (Co) and unassigned (Cn).
_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})
#: Variation selectors (category Mn) show nothing themselves, and can carry bytes like tags can.
_VARIATION_SELECTORS = ((0x180B, 0x180F), (0xFE00, 0xFE0F), (0xE0100, 0xE01EF))
_SCOPE_HINT = "Use 'global', 'client:<rule profile>' (such as 'client:ccn') or 'project:<prj_id>'."


def new_lesson_id() -> str:
    """Return a fresh lesson id, ``les_<hex>`` (the pattern of the hub's node and project ids)."""
    return "les_" + secrets.token_hex(8)


def _invalid(field: str, message: str, hint: str) -> WriteError:
    return WriteError("invalid_input", f"{field} {message}", hint, field=field)


def hidden_character(text: str) -> str | None:
    """The first character of ``text`` a reader cannot see, or ``None``.

    Control, format, surrogate, private-use and unassigned code points, and variation selectors.
    A lesson is shown to a person before it reaches a guide, so it must say only what they see.
    """
    for char in text:
        point = ord(char)
        if unicodedata.category(char) in _HIDDEN_CATEGORIES or any(
            low <= point <= high for low, high in _VARIATION_SELECTORS
        ):
            return char
    return None


def _unseen(char: str) -> str:
    return f"a character a reader cannot see (U+{ord(char):04X})"


_VISIBLE_HINT = (
    "Write plain visible text: control, zero-width, bidirectional, tag, private-use and other "
    "invisible characters are refused."
)


# -- scope -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LessonScope:
    """Where a lesson applies: ``global``, ``client:<name>`` or ``project:<prj_id>``."""

    kind: str
    name: str | None = None

    def __str__(self) -> str:
        return GLOBAL if self.kind == "global" else f"{self.kind}:{self.name}"


def parse_scope(value: Any) -> LessonScope:
    """Parse scope text, checking its syntax only (a writer also checks that the target exists).

    A client is a rule-profile name (the hub's ``NAME_RE``, so never a path); a project is a hub
    id (``check_id``). Raises :class:`~aar.hub.writes.WriteError` ``invalid_input``.
    """
    if not isinstance(value, str):
        raise _invalid("scope", f"must be text, got {type(value).__name__}", _SCOPE_HINT)
    if value == GLOBAL:
        return LessonScope("global")
    kind, sep, name = value.partition(":")
    if sep and kind == "client":
        if not _PROFILE_NAME.fullmatch(name):
            raise _invalid(
                "scope",
                f"names a client that is not a rule profile name: {name!r:.80}",
                "A client is a built-in rule profile name such as 'ccn' (scope 'client:ccn'); "
                "paths to profile files are not accepted.",
            )
        return LessonScope("client", name)
    if sep and kind == "project":
        hidden = hidden_character(check_id(name, "scope"))
        if hidden is not None:
            raise _invalid("scope", f"names a project id with {_unseen(hidden)}", _VISIBLE_HINT)
        return LessonScope("project", name)
    raise _invalid(
        "scope",
        f"must be 'global', 'client:<name>' or 'project:<id>', got {value!r:.80}",
        _SCOPE_HINT,
    )


def _check_target(log: EventLog, scope: LessonScope) -> None:
    """A writer's check that the scope names something real: a loadable built-in rule profile
    (by name only, never a path), or a registered project."""
    if scope.kind == "client":
        _load_named_profile(scope.name, "scope")
    elif scope.kind == "project":
        require_project(log, scope.name or "")


def _scope_problem(value: Any) -> str | None:
    try:
        parse_scope(value)
    except WriteError as exc:
        return exc.message
    return None


# -- text --------------------------------------------------------------------------------------


def _one_line(text: str) -> str:
    return _SPACE_RUN.sub(" ", text).strip()


def _line_problem(value: Any, field: str, max_len: int, *, required: bool) -> str | None:
    """Why ``value`` is not a normalized line for ``field``, or ``None``: the fold's check, and
    the writers' own before they append."""
    if not isinstance(value, str):
        return f"{field} is not text"
    if len(value) > max_len:
        return f"{field} is over {max_len} characters"
    hidden = hidden_character(value)
    if hidden is not None:
        return f"{field} holds {_unseen(hidden)}"
    if required and not value:
        return f"{field} is empty"
    if _one_line(value) != value:
        return f"{field} is not one normalized line"
    return None


def clean_line(value: Any, field: str, *, max_len: int, required: bool = True) -> str:
    """``value`` as one line: stripped, every whitespace run (newlines included) one space.

    Refuses anything but text, more than ``max_len`` characters after normalizing, any character
    a reader cannot see (:func:`hidden_character`) and empty text when ``required``
    (``invalid_input``).
    """
    hint = f"Write {field} as one line of at most {max_len} characters."
    if not isinstance(value, str):
        raise _invalid(field, f"must be text, got {type(value).__name__}", hint)
    if len(value) > MAX_RAW:
        raise _invalid(field, f"is {len(value)} characters before normalizing", hint)
    text = _one_line(value)
    if len(text) > max_len:
        raise _invalid(
            field, f"must be at most {max_len} characters after normalizing (got {len(text)})", hint
        )
    hidden = hidden_character(text)
    if hidden is not None:
        raise _invalid(field, f"must not hold {_unseen(hidden)}", _VISIBLE_HINT)
    if required and not text:
        raise _invalid(field, "must not be empty", hint)
    return text


def check_lesson_id(value: Any) -> str:
    """``value`` if it is a lesson id (``les_`` and 16 hex digits), else ``invalid_input``."""
    if not isinstance(value, str) or not LESSON_ID_RE.fullmatch(value):
        raise _invalid(
            "lesson_id",
            f"is not a lesson id: {value!r:.80}",
            "Pass a lesson id (les_...) from the lessons page or `aar hub lessons list`.",
        )
    return value


def _check_key(idempotency_key: Any) -> str | None:
    if idempotency_key is None:
        return None
    if not isinstance(idempotency_key, str) or not 0 < len(idempotency_key) <= MAX_KEY:
        raise _invalid(
            "idempotency_key",
            f"must be 1-{MAX_KEY} characters",
            "Choose a key that names this one request; send it again only to retry it.",
        )
    return idempotency_key


def _check_source(source: Any) -> dict[str, str] | None:
    if source is None:
        return None
    if not isinstance(source, (tuple, list)) or len(source) != 2:
        raise _invalid(
            "source",
            "must be (project_id, review_id)",
            "Name the review the lesson was drawn from, or leave source out.",
        )
    project_id, review_id = source
    checked = {
        "project_id": check_id(project_id, "source.project_id"),
        "review_id": check_id(review_id, "source.review_id"),
    }
    for key, value in checked.items():
        hidden = hidden_character(value)
        if hidden is not None:
            raise _invalid(f"source.{key}", f"holds {_unseen(hidden)}", _VISIBLE_HINT)
    return checked


# -- the lesson ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LessonSource:
    """The review a proposal was drawn from."""

    project_id: str
    review_id: str

    def to_dict(self) -> dict[str, str]:
        return {"project_id": self.project_id, "review_id": self.review_id}


@dataclass(frozen=True, slots=True)
class Proposal:
    """A lesson as proposed: the suggested text and scope, why, from which review, who, when."""

    text: str
    scope: str
    why: str
    source: LessonSource | None
    by: Actor
    at: str
    event_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "scope": self.scope,
            "why": self.why,
            "source": self.source.to_dict() if self.source else None,
            "by": actor_dict(self.by),
            "at": self.at,
            "event_id": self.event_id,
        }


@dataclass(frozen=True, slots=True)
class Acceptance:
    """The lesson as a human accepted it (``via="accept"``) or wrote it (``via="add"``)."""

    text: str
    scope: str
    via: str
    by: Actor
    at: str
    event_id: str
    seq: int  # the acceptance order

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "scope": self.scope,
            "via": self.via,
            "by": actor_dict(self.by),
            "at": self.at,
            "event_id": self.event_id,
        }


@dataclass(frozen=True, slots=True)
class Closing:
    """A drop (of a proposal) or a retirement (of an accepted lesson): who, when and why."""

    reason: str
    by: Actor
    at: str
    event_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "by": actor_dict(self.by),
            "at": self.at,
            "event_id": self.event_id,
        }


@dataclass(frozen=True, slots=True)
class Lesson:
    """One lesson as folded from the log: the original proposal (when there was one), the
    accepted text and scope, who did each step and when, and the state."""

    id: str
    state: str
    proposal: Proposal | None = None
    acceptance: Acceptance | None = None
    drop: Closing | None = None
    retirement: Closing | None = None

    @property
    def text(self) -> str:
        """The text that applies: the accepted one, else the proposal's."""
        if self.acceptance is not None:
            return self.acceptance.text
        return self.proposal.text if self.proposal is not None else ""

    @property
    def scope(self) -> str:
        """The scope that applies: the accepted one, else the proposal's suggestion."""
        if self.acceptance is not None:
            return self.acceptance.scope
        return self.proposal.scope if self.proposal is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "text": self.text,
            "scope": self.scope,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "acceptance": self.acceptance.to_dict() if self.acceptance else None,
            "drop": self.drop.to_dict() if self.drop else None,
            "retirement": self.retirement.to_dict() if self.retirement else None,
        }


# -- the fold --------------------------------------------------------------------------------------


def _source_problem(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return "source is not an object or null"
    for key in ("project_id", "review_id"):
        ident = value.get(key)
        if not is_text(ident, MAX_ID):
            return f"source.{key} is not an id"
        hidden = hidden_character(ident)
        if hidden is not None:
            return f"source.{key} holds {_unseen(hidden)}"
    return None


def _payload_problem(event_type: str, payload: Any) -> str | None:
    """Why a lesson event's payload is unusable, or ``None``. Shared by the writers and the fold."""
    if not isinstance(payload, dict):
        return "payload is not an object"
    lesson_id = payload.get("lesson_id")
    if not isinstance(lesson_id, str) or not LESSON_ID_RE.fullmatch(lesson_id):
        return "lesson_id is not a lesson id (les_ and 16 hex digits)"
    if event_type in (EVENT_PROPOSED, EVENT_ADDED, EVENT_ACCEPTED):
        problem = _line_problem(payload.get("text"), "text", MAX_TEXT, required=True)
        if problem is None:
            problem = _scope_problem(payload.get("scope"))
        if problem is not None:
            return problem
    if event_type == EVENT_PROPOSED:
        why = _line_problem(payload.get("why"), "why", MAX_WHY, required=False)
        return why or _source_problem(payload.get("source"))
    if event_type in (EVENT_DROPPED, EVENT_RETIRED):
        return _line_problem(payload.get("reason"), "reason", MAX_REASON, required=False)
    return None


def _actor_problem(event_type: str, actor: Actor) -> str | None:
    who = f"{actor.kind}:{actor.id}"
    if event_type in HUMAN_ONLY_EVENTS and actor.kind != "human":
        return f"{event_type} by {who}, not a human"
    if event_type == EVENT_PROPOSED and actor.kind not in PROPOSER_KINDS:
        return f"proposal by {who}; only a human or an agent may propose a lesson"
    return None


def _transition_problem(event_type: str, lesson_id: str, current: Lesson | None) -> str | None:
    """Whether the lesson's state allows the event: the one state machine of the module."""
    if event_type in (EVENT_PROPOSED, EVENT_ADDED):
        return None if current is None else f"lesson {lesson_id} already exists; the first stands"
    if current is None:
        return f"no lesson {lesson_id}"
    if current.state == "retired":
        return f"lesson {lesson_id} was retired; it takes no more events"
    if event_type == EVENT_RETIRED:
        if current.state != "accepted":
            return f"lesson {lesson_id} is {current.state}; only an accepted lesson can be retired"
        return None
    if current.state != "proposed":  # accept or drop
        return f"lesson {lesson_id} was already decided ({current.state})"
    return None


class _Fold:
    """Lessons by id in creation order, the events skipped, and the ids of the events applied."""

    def __init__(self) -> None:
        self.lessons: dict[str, Lesson] = {}
        self.skipped: list[SkippedEvent] = []
        self.applied: set[str] = set()

    def problem(
        self, event_type: str, actor: Actor, project_id: str | None, payload: Any
    ) -> str | None:
        """Why this event cannot be applied now, or ``None``. Writers ask before they append."""
        if event_type not in LESSON_EVENT_TYPES:
            return f"{event_type} is not a lesson event"
        if project_id is not None:
            return f"a lesson event is global, but this one carries project_id {project_id!r:.80}"
        problem = _actor_problem(event_type, actor) or _payload_problem(event_type, payload)
        if problem is not None:
            return problem
        lesson_id = payload["lesson_id"]
        return _transition_problem(event_type, lesson_id, self.lessons.get(lesson_id))

    def apply(self, event: Event) -> Lesson | None:
        """Fold one event; a bad one is skipped and listed. Returns the lesson it changed."""
        problem = self.problem(event.type, event.actor, event.project_id, event.payload)
        if problem is not None:
            self.skipped.append(SkippedEvent.of(event, problem))
            return None
        lesson = _next(self.lessons.get(event.payload["lesson_id"]), event)
        self.lessons[lesson.id] = lesson
        self.applied.add(event.id)
        return lesson


def _next(current: Lesson | None, event: Event) -> Lesson:
    """The lesson after a valid event (:meth:`_Fold.problem` has passed)."""
    payload = event.payload
    lesson_id = payload["lesson_id"]
    if event.type == EVENT_PROPOSED:
        source = payload.get("source")
        proposal = Proposal(
            text=payload["text"],
            scope=payload["scope"],
            why=payload["why"],
            source=LessonSource(source["project_id"], source["review_id"]) if source else None,
            by=event.actor,
            at=event.ts,
            event_id=event.id,
        )
        return Lesson(lesson_id, "proposed", proposal=proposal)
    if event.type in (EVENT_ADDED, EVENT_ACCEPTED):
        acceptance = Acceptance(
            text=payload["text"],
            scope=payload["scope"],
            via="add" if event.type == EVENT_ADDED else "accept",
            by=event.actor,
            at=event.ts,
            event_id=event.id,
            seq=event.seq,
        )
        if current is None:
            return Lesson(lesson_id, "accepted", acceptance=acceptance)
        return replace(current, state="accepted", acceptance=acceptance)
    if current is None:  # pragma: no cover - _transition_problem refuses it
        raise ValueError(f"no lesson {lesson_id}")
    closing = Closing(reason=payload["reason"], by=event.actor, at=event.ts, event_id=event.id)
    if event.type == EVENT_DROPPED:
        return replace(current, state="dropped", drop=closing)
    return replace(current, state="retired", retirement=closing)


def _fold(events: Iterable[Event]) -> _Fold:
    fold = _Fold()
    for event in events:
        fold.apply(event)
    return fold


def scan_lessons(log: EventLog) -> tuple[tuple[Lesson, ...], tuple[SkippedEvent, ...]]:
    """Every lesson in creation order, and the events skipped. Never raises on a malformed or
    forged event: it is skipped and listed instead."""
    fold = _fold(log.iter_events(types=LESSON_EVENT_TYPES))
    return tuple(fold.lessons.values()), tuple(fold.skipped)


def get_lesson(log: EventLog, lesson_id: str) -> Lesson | None:
    """One lesson, or ``None`` if there is no lesson with that id."""
    if not isinstance(lesson_id, str) or not LESSON_ID_RE.fullmatch(lesson_id):
        return None
    lessons, _ = scan_lessons(log)
    return next((lesson for lesson in lessons if lesson.id == lesson_id), None)


# -- the guide feed --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuideLessons:
    """The accepted lesson texts a cut guide prints, per scope, each in acceptance order."""

    global_: tuple[str, ...] = ()
    client: tuple[str, ...] = ()
    project: tuple[str, ...] = ()
    client_name: str | None = None

    @property
    def empty(self) -> bool:
        return not (self.global_ or self.client or self.project)


def lessons_for_guide(log: EventLog, *, project_id: str | None, client: str | None) -> GuideLessons:
    """The accepted lessons for one run: global, the client's (the brief's ``rule_profile``;
    ``None`` without a brief) and the project's. Never proposed, dropped or retired ones. Each
    tuple holds at most :data:`MAX_ACCEPTED_PER_SCOPE`: the first ones accepted, so an event
    appended later cannot push an established lesson out."""
    lessons, _ = scan_lessons(log)
    accepted = sorted(
        (lesson for lesson in lessons if lesson.acceptance and lesson.state == "accepted"),
        key=lambda lesson: lesson.acceptance.seq if lesson.acceptance else 0,
    )

    def texts(scope: str | None) -> tuple[str, ...]:
        if scope is None:
            return ()
        found = (lesson.text for lesson in accepted if lesson.scope == scope)
        return tuple(found)[:MAX_ACCEPTED_PER_SCOPE]

    return GuideLessons(
        global_=texts(GLOBAL),
        client=texts(None if client is None else f"client:{client}"),
        project=texts(None if project_id is None else f"project:{project_id}"),
        client_name=client,
    )


# -- writers ---------------------------------------------------------------------------------------


def _txn_fold(txn: Transaction) -> _Fold:
    """The lessons as of this transaction (one read, one snapshot)."""
    return _fold(txn.read(0, types=LESSON_EVENT_TYPES))


def _replay(
    txn: Transaction,
    fold: _Fold,
    idempotency_key: str | None,
    event_type: str,
    request: dict[str, Any],
    actor: Actor,
) -> Lesson | None:
    """The lesson an earlier identical request wrote under this key (as it is now), ``None`` if
    the key is new, or ``idempotency_conflict``."""
    if idempotency_key is None:
        return None
    existing = txn.find_by_idempotency_key(idempotency_key)
    if existing is None:
        return None
    try:
        match_or_conflict(existing, event_type, request, request.keys(), actor=actor)
    except IdempotencyConflict:
        raise _conflict(idempotency_key, "the request differs") from None
    lesson = fold.lessons.get(existing.payload.get("lesson_id"))
    if lesson is None or existing.id not in fold.applied:
        raise _conflict(idempotency_key, "it names an event the lessons fold skipped")
    return lesson


def _require_state(fold: _Fold, lesson_id: str, wanted: str, done: str) -> Lesson:
    lesson = fold.lessons.get(lesson_id)
    if lesson is None:
        raise WriteError(
            "unknown_lesson",
            f"no lesson {lesson_id!r} in the hub",
            "Use a lesson id from the lessons page or `aar hub lessons list`.",
            lesson_id=lesson_id,
        )
    if lesson.state != wanted:
        only = "a proposal" if wanted == "proposed" else "an accepted lesson"
        raise WriteError(
            "illegal_transition",
            f"lesson {lesson_id} is {lesson.state}; only {only} can be {done}",
            "Reload the lessons page to see the lesson's current state.",
            lesson_id=lesson_id,
            state=lesson.state,
        )
    return lesson


def _check_room(fold: _Fold, scope: str) -> None:
    held = sum(
        1
        for lesson in fold.lessons.values()
        if lesson.state == "accepted" and lesson.scope == scope
    )
    if held >= MAX_ACCEPTED_PER_SCOPE:
        raise WriteError(
            "too_many_lessons",
            f"scope {scope} already holds {held} accepted lessons (the limit is "
            f"{MAX_ACCEPTED_PER_SCOPE})",
            "Retire one of its lessons first (or merge two into one), then try again.",
            scope=scope,
            limit=MAX_ACCEPTED_PER_SCOPE,
        )


def _check_review(log: EventLog, source: dict[str, str]) -> None:
    project_id, review_id = source["project_id"], source["review_id"]
    require_project(log, project_id)
    if reviews.get_review(log, project_id, review_id) is None:
        raise WriteError(
            "unknown_review",
            f"no review {review_id!r} in project {project_id!r}",
            "Name the review the lesson was drawn from (`aar hub review list`), or leave "
            "source out.",
            project_id=project_id,
            review_id=review_id,
        )


def _append(
    txn: Transaction,
    fold: _Fold,
    event_type: str,
    actor: Actor,
    payload: dict[str, Any],
    idempotency_key: str | None,
) -> Lesson:
    """Append one lesson event, after asking the fold whether it would apply it."""
    problem = fold.problem(event_type, actor, None, payload)
    if problem is not None:  # pragma: no cover - every writer checked each part first
        raise _invalid("payload", problem, "This is a bug in the writer; report it.")
    event = txn.append(event_type, actor, payload, idempotency_key=idempotency_key)
    lesson = fold.apply(event)
    if lesson is None:  # pragma: no cover - the same check just passed
        raise _invalid("payload", "the fold refused the event just written", "Report this bug.")
    return lesson


def _check_proposer(actor: Any) -> Actor:
    actor = check_actor(actor)
    if actor.kind not in PROPOSER_KINDS:
        raise WriteError(
            "actor_not_allowed",
            f"{actor.kind}:{actor.id:.80} may not propose a lesson",
            "Only a human or an agent proposes lessons.",
            actor=actor_dict(actor),
        )
    return actor


def _check_human(actor: Any, action: str) -> Actor:
    require_human(actor, action)
    return check_actor(actor)


def propose(
    log: EventLog,
    text: str,
    scope: str,
    actor: Actor,
    *,
    why: str = "",
    source: tuple[str, str] | None = None,
    idempotency_key: str | None = None,
) -> Lesson:
    """Record a proposed lesson. It applies to nothing until a human accepts it.

    ``actor`` is a human or an agent. ``source`` is ``(project_id, review_id)`` of the review the
    lesson was drawn from; both must exist. Refused (``too_many_proposals``) while
    :data:`MAX_PENDING` proposals are waiting.
    """
    actor = _check_proposer(actor)
    key = _check_key(idempotency_key)
    request = {
        "text": clean_line(text, "text", max_len=MAX_TEXT),
        "scope": str(parse_scope(scope)),
        "why": clean_line(why, "why", max_len=MAX_WHY, required=False),
        "source": _check_source(source),
    }
    with log.transaction() as txn:
        fold = _txn_fold(txn)
        replayed = _replay(txn, fold, key, EVENT_PROPOSED, request, actor)
        if replayed is not None:
            return replayed
        _check_target(log, parse_scope(request["scope"]))
        if request["source"] is not None:
            _check_review(log, request["source"])
        pending = sum(1 for lesson in fold.lessons.values() if lesson.state == "proposed")
        if pending >= MAX_PENDING:
            raise WriteError(
                "too_many_proposals",
                f"{pending} lesson proposals are already waiting (the limit is {MAX_PENDING})",
                "A person must accept or drop some in the console before more are proposed.",
                limit=MAX_PENDING,
            )
        payload = {"lesson_id": new_lesson_id(), **request}
        return _append(txn, fold, EVENT_PROPOSED, actor, payload, key)


def add(
    log: EventLog, text: str, scope: str, actor: Actor, *, idempotency_key: str | None = None
) -> Lesson:
    """A human writes an accepted lesson by hand (no proposal). Refused (``too_many_lessons``)
    when the scope already holds :data:`MAX_ACCEPTED_PER_SCOPE` accepted lessons."""
    actor = _check_human(actor, "add a lesson")
    key = _check_key(idempotency_key)
    request = {
        "text": clean_line(text, "text", max_len=MAX_TEXT),
        "scope": str(parse_scope(scope)),
    }
    with log.transaction() as txn:
        fold = _txn_fold(txn)
        replayed = _replay(txn, fold, key, EVENT_ADDED, request, actor)
        if replayed is not None:
            return replayed
        _check_target(log, parse_scope(request["scope"]))
        _check_room(fold, request["scope"])
        payload = {"lesson_id": new_lesson_id(), **request}
        return _append(txn, fold, EVENT_ADDED, actor, payload, key)


def accept(
    log: EventLog,
    lesson_id: str,
    text: str,
    scope: str,
    actor: Actor,
    *,
    idempotency_key: str | None = None,
) -> Lesson:
    """A human accepts a proposed lesson, with its text and scope as they should apply (they
    may differ from the proposal's, which the lesson keeps)."""
    actor = _check_human(actor, "accept a lesson")
    lesson_id = check_lesson_id(lesson_id)
    key = _check_key(idempotency_key)
    request = {
        "lesson_id": lesson_id,
        "text": clean_line(text, "text", max_len=MAX_TEXT),
        "scope": str(parse_scope(scope)),
    }
    with log.transaction() as txn:
        fold = _txn_fold(txn)
        replayed = _replay(txn, fold, key, EVENT_ACCEPTED, request, actor)
        if replayed is not None:
            return replayed
        _require_state(fold, lesson_id, "proposed", "accepted")
        _check_target(log, parse_scope(request["scope"]))
        _check_room(fold, request["scope"])
        return _append(txn, fold, EVENT_ACCEPTED, actor, request, key)


def drop(
    log: EventLog,
    lesson_id: str,
    actor: Actor,
    *,
    reason: str = "",
    idempotency_key: str | None = None,
) -> Lesson:
    """A human drops a proposed lesson; it never applies."""
    actor = _check_human(actor, "drop a lesson proposal")
    lesson_id = check_lesson_id(lesson_id)
    key = _check_key(idempotency_key)
    request = {
        "lesson_id": lesson_id,
        "reason": clean_line(reason, "reason", max_len=MAX_REASON, required=False),
    }
    with log.transaction() as txn:
        fold = _txn_fold(txn)
        replayed = _replay(txn, fold, key, EVENT_DROPPED, request, actor)
        if replayed is not None:
            return replayed
        _require_state(fold, lesson_id, "proposed", "dropped")
        return _append(txn, fold, EVENT_DROPPED, actor, request, key)


def retire(
    log: EventLog,
    lesson_id: str,
    actor: Actor,
    *,
    reason: str = "",
    idempotency_key: str | None = None,
) -> Lesson:
    """A human retires an accepted lesson (accepted or added); guides built later leave it out."""
    actor = _check_human(actor, "retire a lesson")
    lesson_id = check_lesson_id(lesson_id)
    key = _check_key(idempotency_key)
    request = {
        "lesson_id": lesson_id,
        "reason": clean_line(reason, "reason", max_len=MAX_REASON, required=False),
    }
    with log.transaction() as txn:
        fold = _txn_fold(txn)
        replayed = _replay(txn, fold, key, EVENT_RETIRED, request, actor)
        if replayed is not None:
            return replayed
        _require_state(fold, lesson_id, "accepted", "retired")
        return _append(txn, fold, EVENT_RETIRED, actor, request, key)


__all__ = [
    "EVENT_ACCEPTED",
    "EVENT_ADDED",
    "EVENT_DROPPED",
    "EVENT_PROPOSED",
    "EVENT_RETIRED",
    "GLOBAL",
    "LESSON_EVENT_TYPES",
    "MAX_ACCEPTED_PER_SCOPE",
    "MAX_PENDING",
    "MAX_REASON",
    "MAX_TEXT",
    "MAX_WHY",
    "STATES",
    "GuideLessons",
    "Lesson",
    "LessonScope",
    "accept",
    "add",
    "check_lesson_id",
    "clean_line",
    "drop",
    "get_lesson",
    "hidden_character",
    "lessons_for_guide",
    "parse_scope",
    "propose",
    "retire",
    "scan_lessons",
]
