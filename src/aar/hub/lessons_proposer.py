"""Draft lessons from decided reviews with a model, through the local model gateway (S23b).

After a person decides a review in the console (approve or reject, with a reason and notes at
timeline frames), :func:`propose_from_reviews` asks a cheap model for at most three durable
editing lessons drawn from that feedback and records each one as a **proposal**
(:func:`aar.hub.lessons.propose`, as an agent). Nothing a model writes reaches a cut guide: a
person accepts, rescopes or drops each proposal on ``/lessons`` (S23a), and guides print only
accepted lessons.

**One pass** takes the decided reviews of every project that are neither done nor given up: the
fewest strikes first, then the fewest charged attempts, then the oldest decision, so a review
that keeps failing moves behind the others instead of holding them back. They are found
through the ``review.decided`` events and each is confirmed with
:func:`aar.hub.reviews.get_review`; an open review, a withdrawn one, a decision the reviews fold
rejects and a review of an unregistered project are skipped. For each review the model gets
(:mod:`aar.hub.lesson_drafts` writes the prompt):

- the title, the decision, the reason, and the notes a person wrote, with their timecodes (notes
  an agent added are left out: the prompt says a person wrote them);
- the show (the project's registered name) and the client: the ``rule_profile`` that every typed
  brief of the project shares, if it is a built-in profile. Otherwise there is no client and only
  the global and project scopes are offered;
- the lessons already accepted or waiting for a person in global, the client and the project,
  labelled by scope (``global``, ``client``, ``project``), so it does not repeat them. Never
  another project's or another client's.

Each draft becomes ``lesson.proposed`` with the model's ``why`` and the review as its
``source``: scope ``global`` stays ``global``, ``client`` becomes ``client:<client>`` and
``project`` becomes ``project:<project_id>``. A draft the store refuses is reported, and the
others are still recorded.

**Once per review.** A review is done once its marker exists: ``lessons.drafted``, a global
event ``{project_id, review_id, proposed, model, recovered}`` with idempotency key
``lessons:drafted:<project_id>:<review_id>``. A later pass skips it without calling the model.
The marker is written after the proposals, so a crash between the two leaves proposals and no
marker. The next pass finds proposals whose source is the review, writes the marker
(``recovered`` true, ``model`` null) and does not call the model. When the store refused every
draft of a reply only because its backlog filled while the model answered
(``too_many_proposals``), no marker is written: the review is ``deferred`` and stays a
candidate.

**Failures** are classified so that neither a broken setup nor one review can use up, or hold
back, the other reviews:

- A *setup failure* stops the pass at once and charges nothing: ``not_configured``, ``origin``,
  ``redirect``, ``network`` (the gateway cannot be reached), HTTP 401, 403, 404 and 429 (once
  the client has given up retrying), and a model the gateway does not serve
  (``unknown_model``: before its first chat a pass reads ``GET /v1/models``, because the
  gateway answers a chat for an unknown model with HTTP 400).
- Every other failure of a review's chat is *ambiguous*, the review's own or the setup's: any
  other gateway error (``timeout``, ``too-large``, ``bad-response``, HTTP 5xx once retried, and
  any other 4xx such as 400, 408, 413 or 422) and a reply that is not the lessons JSON
  (``bad-output``). The pass carries on with the next review, and stops after
  :data:`MAX_IN_A_ROW` in a row. An ambiguous failure is **charged** to its review
  (``lessons.draft_failed`` ``{project_id, review_id, code, attempt}``, key
  ``lessons:draft-failed:<project_id>:<review_id>:<attempt>``) when any review of the same pass,
  before or after it, is drafted by the model (the pass has one model): a success anywhere in
  the pass shows the setup works. Otherwise it is an uncharged **strike**
  (``lessons.draft_strike`` ``{project_id, review_id, code, strike}``, key
  ``lessons:draft-strike:<project_id>:<review_id>:<strike>``).
- A review is given up once it has :data:`MAX_ATTEMPTS` charged attempts or
  :data:`MAX_STRIKES` strikes (about a day of passes 10 minutes apart: a review that fails
  alone, with no other review to show whose fault it is, cannot retry forever). Every later
  report counts it, with both counts. Nothing reopens a given-up review yet.

The marker, the attempts and the strikes are counted by their idempotency keys, whatever event
holds them, from 1 up to the first free key. So no event (malformed, forged or from a concurrent
pass) can make a review retry forever: each attempt or strike takes a key, and the last key ends
it. The three event types are bookkeeping, not lesson events: the lessons fold never reads them.

**One pass at a time.** A pass holds a non-blocking ``fcntl.flock`` on a file beside the
database (``<db>`` + :data:`LOCK_SUFFIX`, left in place after) from start to end. A pass that
finds it held does nothing and reports ``busy``, which is not a failure. Without ``fcntl``
(Windows) a pass runs without the lock. Right before recording a reply, the pass checks the
review's marker and proposals again: a review recorded meanwhile is ``recovered``, and the
reply is not recorded.

**Bounds.** At most ``limit`` chats per pass (1 to :data:`MAX_LIMIT`, default
:data:`DEFAULT_LIMIT`), failed ones included. Each reply is at most :data:`MAX_TOKENS` tokens,
and the prompt is bounded by ``lesson_drafts.PROMPT_BUDGET``. At most 3 drafts per review.
Before each chat the pass stops if ``pending + 3`` would go past the store's
:data:`~aar.hub.lessons.MAX_PENDING`; a full backlog is reported, not an error.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

from aar.hub import lessons, reviews
from aar.hub.common import Clock, actor_dict, is_text, iso, utcnow
from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.graph import WorkGraph
from aar.hub.lesson_drafts import (
    DEFAULT_LESSONS_MODEL,
    MAX_DRAFTS,
    MAX_NAME,
    DraftError,
    DraftResult,
    ReviewFeedback,
    build_messages,
    parse_drafts,
    scopes_for,
)
from aar.hub.registry import Project, Registry
from aar.hub.writes import MAX_ID, MAX_KEY, WriteError, check_actor, typed_brief
from aar.llm.gateway import ChatReply, GatewayError, is_model_id
from aar.plan.profile import builtin_profile_names

EVENT_DRAFTED = "lessons.drafted"
EVENT_DRAFT_FAILED = "lessons.draft_failed"
EVENT_DRAFT_STRIKE = "lessons.draft_strike"
DEFAULT_ACTOR = Actor("agent", "lessons-proposer")
MODEL_ENV = "AAR_LESSONS_MODEL"
DEFAULT_LIMIT = 5
MAX_LIMIT = 50
#: Charged attempts per review: failures a review drafted in the same pass showed to be its own.
MAX_ATTEMPTS = 3
#: Uncharged strikes per review: about a day of passes 10 minutes apart.
MAX_STRIKES = 144
#: Ambiguous failures in a row that stop a pass.
MAX_IN_A_ROW = 2
#: Reply tokens a chat may use: three one-line lessons need a few hundred.
MAX_TOKENS = 4000
#: Gateway failures of the setup, never of one review: each stops the pass and charges nothing.
#: Every other failure of a chat is ambiguous (see the module doc).
SETUP_FAILURES = frozenset(
    {
        "not_configured",
        "origin",
        "redirect",
        "network",
        "http-401",
        "http-403",
        "http-404",
        "http-429",
    }
)
#: The pass lock: a file beside the database, held for a whole pass.
LOCK_SUFFIX = ".lessons-propose.lock"
#: Given-up reviews listed in a report (all are counted).
MAX_LISTED = 20
#: Stops that are not failures: the pass ran.
_NOT_FAILURES = ("backlog_full", "busy")


class ChatClient(Protocol):
    """What a pass needs from the gateway client (:class:`aar.llm.gateway.GatewayClient`)."""

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> ChatReply: ...

    def models(self) -> Sequence[str]: ...


# -- the report ----------------------------------------------------------------------------------


@dataclass
class ReviewOutcome:
    """What one pass did with one review: ``drafted``, ``recovered``, ``deferred`` (the store had
    no room for any draft: no marker, so the review stays a candidate) or ``failed``."""

    project_id: str
    review_id: str
    outcome: str
    lessons: list[dict[str, str]] = field(default_factory=list)  # drafted: {id, text, scope}
    problems: list[str] = field(default_factory=list)  # drafts dropped or refused, bookkeeping
    proposed: int = 0  # recovered: the proposals found
    reply_model: str | None = None  # drafted, deferred: the model the gateway says answered
    code: str | None = None  # failed
    message: str = ""  # failed
    charged: bool = False  # failed: charged as an attempt; otherwise it is a strike
    attempts: int = 0  # failed: the review's charged attempts after this pass
    strikes: int = 0  # failed: the review's strikes after this pass
    final: bool = False  # failed: no attempt or strike left, so the review is given up

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "project_id": self.project_id,
            "review_id": self.review_id,
            "outcome": self.outcome,
        }
        if self.outcome == "drafted":
            out.update(
                proposed=len(self.lessons),
                lessons=list(self.lessons),
                reply_model=self.reply_model,
            )
        elif self.outcome == "recovered":
            out["proposed"] = self.proposed
        elif self.outcome == "deferred":
            out["reply_model"] = self.reply_model
        else:
            out.update(
                code=self.code,
                message=self.message,
                charged=self.charged,
                attempts=self.attempts,
                strikes=self.strikes,
                final=self.final,
            )
        out["problems"] = list(self.problems)
        return out


class GivenUp(NamedTuple):
    """A review this pass skipped because its attempts or its strikes are used up."""

    project_id: str
    review_id: str
    attempts: int
    strikes: int


@dataclass
class ProposeReport:
    """One pass: what it did, and why it stopped if it stopped early (``stopped``)."""

    model: str
    actor: Actor
    limit: int
    started: str
    finished: str = ""
    calls: int = 0
    usage: Counter[str] = field(default_factory=Counter)
    reviews: list[ReviewOutcome] = field(default_factory=list)
    given_up: list[GivenUp] = field(default_factory=list)
    not_decided: int = 0  # decided by event, but not a decided review once confirmed
    waiting: int = 0  # reviews left for a later pass
    stopped: dict[str, Any] | None = None

    @property
    def proposed(self) -> int:
        """The lessons proposed in this pass."""
        return sum(len(outcome.lessons) for outcome in self.reviews)

    @property
    def struck(self) -> list[ReviewOutcome]:
        """This pass's failures that are strikes: no review of the pass was drafted, so none
        showed them to be their own."""
        return [o for o in self.reviews if o.outcome == "failed" and not o.charged]

    @property
    def failed(self) -> bool:
        """Whether the pass ends in failure: a setup failure or failures in a row stopped it, or
        it recorded a strike. A full backlog and another pass holding the lock are not."""
        return self.error() is not None

    def error(self) -> dict[str, Any] | None:
        """The pass's failure (``reason``, ``code``, ``message``, ``hint``): why it stopped, else
        its last strike; ``None`` when it did not fail."""
        if self.stopped is not None and self.stopped["reason"] not in _NOT_FAILURES:
            return self.stopped
        if not self.struck:
            return None
        last = self.struck[-1]
        return {
            "reason": "strike",
            "code": last.code,
            "message": last.message,
            "hint": "No review was drafted in this pass, so it may be the setup's "
            f"failure and was not charged to the review (strike {last.strikes} of "
            f"{MAX_STRIKES}). The next pass retries it.",
            "project_id": last.project_id,
            "review_id": last.review_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "actor": actor_dict(self.actor),
            "limit": self.limit,
            "started": self.started,
            "finished": self.finished,
            "calls": self.calls,
            "usage": dict(self.usage),
            "proposed": self.proposed,
            "reviews": [outcome.to_dict() for outcome in self.reviews],
            "given_up": {
                "count": len(self.given_up),
                "reviews": [entry._asdict() for entry in self.given_up[:MAX_LISTED]],
            },
            "not_decided": self.not_decided,
            "waiting": self.waiting,
            "stopped": self.stopped,
        }


# -- settings ------------------------------------------------------------------------------------


def check_proposer(actor: Any) -> Actor:
    """``actor`` if it is an agent with a valid id; any other kind is ``actor_not_allowed``."""
    if not isinstance(actor, Actor) or actor.kind != "agent":
        who = f"{actor.kind}:{actor.id:.80}" if isinstance(actor, Actor) else type(actor).__name__
        raise WriteError(
            "actor_not_allowed",
            f"{who} may not draft lessons; the proposer runs as an agent",
            "Use an agent actor such as agent:lessons-proposer. Only a person accepts a "
            "lesson, in the console; the proposer only proposes.",
        )
    return check_actor(actor)


def lessons_model(flag: str | None, environ: Mapping[str, str]) -> str:
    """The model to draft with: ``flag`` (``--model``), else ``AAR_LESSONS_MODEL`` (blank counts
    as unset), else :data:`~aar.hub.lesson_drafts.DEFAULT_LESSONS_MODEL`. ``ValueError`` when the
    chosen value cannot be a model id; the message names where it came from, never the value."""
    if flag is not None:
        source, value = "--model", flag
    elif (environ.get(MODEL_ENV) or "").strip():
        source, value = MODEL_ENV, environ[MODEL_ENV].strip()
    else:
        return DEFAULT_LESSONS_MODEL
    if not is_model_id(value):
        raise ValueError(
            f"{source} is not a model id: 1-128 printable ASCII characters without spaces"
        )
    return value


def check_limit(limit: Any) -> int:
    """``limit`` if it is a whole number from 1 to :data:`MAX_LIMIT`, else ``ValueError``."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"the limit must be a whole number of model calls from 1 to {MAX_LIMIT}")
    return limit


# -- keys ----------------------------------------------------------------------------------------


def drafted_key(project_id: str, review_id: str) -> str:
    """The marker's idempotency key: once it is taken, the review is done."""
    return f"lessons:drafted:{project_id}:{review_id}"


def failed_key(project_id: str, review_id: str, attempt: int) -> str:
    """The idempotency key of a review's ``attempt``-th charged failure (1 to
    :data:`MAX_ATTEMPTS`)."""
    return f"lessons:draft-failed:{project_id}:{review_id}:{attempt}"


def strike_key(project_id: str, review_id: str, strike: int) -> str:
    """The idempotency key of a review's ``strike``-th strike (1 to :data:`MAX_STRIKES`)."""
    return f"lessons:draft-strike:{project_id}:{review_id}:{strike}"


def proposal_key(project_id: str, review_id: str, index: int) -> str:
    """The idempotency key of a review's ``index``-th draft. The lessons store takes keys of at
    most 128 characters, which generated ids always fit; longer ids are hashed."""
    key = f"lessons:proposal:{project_id}:{review_id}:{index}"
    if len(key) <= MAX_KEY:
        return key
    digest = hashlib.sha256(f"{project_id}\n{review_id}".encode()).hexdigest()[:32]
    return f"lessons:proposal:sha256-{digest}:{index}"


def _drafted(log: EventLog, project_id: str, review_id: str) -> bool:
    return log.get_by_idempotency_key(drafted_key(project_id, review_id)) is not None


def _used(log: EventLog, key: Callable[[int], str], cap: int) -> int:
    """How many of a review's numbered keys are used: those taken from 1 up to the first free
    one, or ``cap`` when all are taken."""
    for number in range(1, cap + 1):
        if log.get_by_idempotency_key(key(number)) is None:
            return number - 1
    return cap


@dataclass(frozen=True)
class _Standing:
    """A review's charged attempts and strikes so far."""

    attempts: int
    strikes: int

    @classmethod
    def read(cls, log: EventLog, project_id: str, review_id: str) -> _Standing:
        return cls(
            _used(log, lambda n: failed_key(project_id, review_id, n), MAX_ATTEMPTS),
            _used(log, lambda n: strike_key(project_id, review_id, n), MAX_STRIKES),
        )

    @property
    def given_up(self) -> bool:
        return self.attempts >= MAX_ATTEMPTS or self.strikes >= MAX_STRIKES


# -- the pass lock -------------------------------------------------------------------------------


class _PassLock:
    """A non-blocking exclusive ``flock`` on a file beside the database, for one pass. ``held``
    is false when another pass holds it. Closing the descriptor releases it."""

    def __init__(self, fd: int | None, held: bool) -> None:
        self.fd = fd
        self.held = held

    @classmethod
    def take(cls, db_path: str) -> _PassLock:
        """The lock for ``db_path``. ``OSError`` when its file cannot be opened or locked."""
        try:
            import fcntl
        except ImportError:  # Windows: no flock, so passes there run without the lock
            return cls(None, True)
        if db_path in ("", ":memory:"):  # no file, so no other process shares the database
            return cls(None, True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(db_path + LOCK_SUFFIX, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return cls(fd, False)
        except BaseException:
            os.close(fd)
            raise
        return cls(fd, True)

    def __enter__(self) -> _PassLock:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# -- reading the hub -----------------------------------------------------------------------------


def _decided_reviews(log: EventLog) -> list[tuple[str, str]]:
    """``(project_id, review_id)`` of every review with a ``review.decided`` event, in the order
    of its first one. The events are only a lead: each review is confirmed before use."""
    seen: dict[tuple[str, str], None] = {}
    for event in log.iter_events(types=[reviews.EVENT_DECIDED]):
        review_id = event.payload.get("review_id") if isinstance(event.payload, dict) else None
        if is_text(event.project_id, MAX_ID) and is_text(review_id, MAX_ID):
            seen.setdefault((event.project_id, review_id), None)
    return list(seen)


@dataclass(frozen=True)
class _LessonState:
    """The lessons as of one read: how many wait, how many proposals each review has, and the
    accepted (in acceptance order), then waiting, texts of each scope."""

    pending: int
    sourced: Mapping[tuple[str, str], int]
    texts: Mapping[str, tuple[str, ...]]

    @classmethod
    def read(cls, log: EventLog) -> _LessonState:
        found, _ = lessons.scan_lessons(log)
        accepted = sorted(
            (lesson for lesson in found if lesson.state == "accepted" and lesson.acceptance),
            key=lambda lesson: lesson.acceptance.seq if lesson.acceptance else 0,
        )
        waiting = [lesson for lesson in found if lesson.state == "proposed"]
        texts: dict[str, list[str]] = {}
        for lesson in (*accepted, *waiting):
            texts.setdefault(lesson.scope, []).append(lesson.text)
        sourced = Counter(
            (lesson.proposal.source.project_id, lesson.proposal.source.review_id)
            for lesson in found
            if lesson.proposal is not None and lesson.proposal.source is not None
        )
        return cls(len(waiting), sourced, {scope: tuple(t) for scope, t in texts.items()})

    def existing(self, project_id: str, client: str | None) -> dict[str, tuple[str, ...]]:
        """The lessons the model must not repeat, by the scope words it uses."""
        return {
            "global": self.texts.get(lessons.GLOBAL, ()),
            "client": self.texts.get(f"client:{client}", ()) if client is not None else (),
            "project": self.texts.get(f"project:{project_id}", ()),
        }


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _feedback(review: reviews.Review, project: Project, client: str | None) -> ReviewFeedback:
    decision = review.decision
    if decision is None:  # pragma: no cover - the pass confirmed the decision first
        raise ValueError(f"review {review.id} has no decision")
    return ReviewFeedback(
        project_id=review.project_id,
        review_id=review.id,
        title=review.title,
        decision=decision.decision,
        reason=decision.reason,
        notes=[(note.timecode, note.text) for note in review.notes if note.author.kind == "human"],
        show=_clip(project.name, MAX_NAME),
        client=client,
    )


def _store_scope(scope: str, project_id: str, client: str | None) -> str:
    """A draft's scope word as a lessons-store scope."""
    if scope == "client" and client is not None:
        return f"client:{client}"
    if scope == "project":
        return f"project:{project_id}"
    return lessons.GLOBAL if scope == "global" else scope  # anything else: the store refuses it


def _gateway_stop(
    error: GatewayError, project_id: str | None = None, review_id: str | None = None
) -> dict[str, Any]:
    return {
        "reason": "gateway",
        "code": error.code,
        "message": error.message,
        "hint": error.hint or "Nothing was recorded against the review; the next pass retries.",
        "project_id": project_id,
        "review_id": review_id,
    }


# -- the pass ------------------------------------------------------------------------------------


def propose_from_reviews(
    log: EventLog,
    client: ChatClient,
    *,
    model: str,
    actor: Actor,
    limit: int = DEFAULT_LIMIT,
    clock: Clock = utcnow,
) -> ProposeReport:
    """One bounded pass: draft lessons from the decided reviews not drafted yet, the fewest
    strikes, then charged attempts, then the oldest decision first, with at most ``limit`` model
    calls (see the module doc). Raises ``WriteError`` for an actor that is not an agent and
    ``ValueError`` for a bad model id or limit, before anything is read or sent; every other
    failure is in the report."""
    actor = check_proposer(actor)
    if not is_model_id(model):
        raise ValueError("model must be 1-128 printable ASCII characters without spaces")
    report = ProposeReport(model=model, actor=actor, limit=check_limit(limit), started="")
    report.started = iso(clock())
    try:
        lock = _PassLock.take(log.path)
    except OSError as error:
        report.stopped = {
            "reason": "setup",
            "code": "lock_unavailable",
            "message": "cannot take the pass lock beside the database: "
            f"{error.strerror or type(error).__name__}",
            "hint": "The database's folder must be writable by this user. Nothing was done.",
        }
    else:
        with lock:
            if lock.held:
                _Pass(log, client, report).run()
            else:
                report.stopped = {
                    "reason": "busy",
                    "code": "busy",
                    "message": "another pass holds this database's lessons lock",
                    "hint": "Nothing was done; that pass drafts the reviews.",
                }
    report.finished = iso(clock())
    return report


class _Pass:
    def __init__(self, log: EventLog, client: ChatClient, report: ProposeReport) -> None:
        self.log = log
        self.client = client
        self.report = report
        self.registry = Registry(log)
        self.clients: dict[str, str | None] = {}
        self.model_checked = False
        self.proven = False  # the model drafted a review in this pass: the setup works
        self.unproven: list[ReviewOutcome] = []  # ambiguous failures not charged yet
        self.in_a_row = 0  # ambiguous failures since the last review the model drafted

    def run(self) -> None:
        live: list[tuple[int, int, int, str, str]] = []
        for order, (project_id, review_id) in enumerate(_decided_reviews(self.log)):
            if _drafted(self.log, project_id, review_id):
                continue
            standing = _Standing.read(self.log, project_id, review_id)
            if standing.given_up:
                self.report.given_up.append(
                    GivenUp(project_id, review_id, standing.attempts, standing.strikes)
                )
            else:
                live.append((standing.strikes, standing.attempts, order, project_id, review_id))
        # The fewest strikes, then the fewest charged attempts, then the oldest decision first
        # (``order`` is unique, so the ids never decide): a review that keeps failing moves
        # behind the others instead of holding them back.
        candidates = [(project_id, review_id) for *_, project_id, review_id in sorted(live)]
        for index, (project_id, review_id) in enumerate(candidates):
            if not self.review(project_id, review_id):
                self.count_rest(candidates[index:])
                break
        for outcome in self.unproven:  # no review of this pass was drafted: strikes
            self.tally(outcome, charged=False)
        self.unproven.clear()

    def count_rest(self, rest: Sequence[tuple[str, str]]) -> None:
        """After a stop: the reviews left, waiting or given up (cheap key lookups only)."""
        handled = {(outcome.project_id, outcome.review_id) for outcome in self.report.reviews}
        for project_id, review_id in rest:
            if (project_id, review_id) in handled or _drafted(self.log, project_id, review_id):
                continue
            standing = _Standing.read(self.log, project_id, review_id)
            if standing.given_up:
                self.report.given_up.append(
                    GivenUp(project_id, review_id, standing.attempts, standing.strikes)
                )
            else:
                self.report.waiting += 1

    def client_of(self, project_id: str) -> str | None:
        """The rule profile every typed brief of the project shares, if it is built in."""
        if project_id not in self.clients:
            graph = WorkGraph(self.log, project_id)
            profiles = {
                brief.rule_profile
                for node in graph.nodes()
                if (brief := typed_brief(node)) is not None
            }
            name = next(iter(profiles)) if len(profiles) == 1 else None
            self.clients[project_id] = name if name in builtin_profile_names() else None
        return self.clients[project_id]

    def model_served(self) -> bool:
        """Before the pass's first chat: does the gateway serve the model? A model it does not
        serve would be answered with HTTP 400 for every review."""
        if self.model_checked:
            return True
        try:
            served = self.client.models()
        except GatewayError as error:
            self.report.stopped = _gateway_stop(error)
            return False
        if self.report.model not in served:
            self.report.stopped = {
                "reason": "setup",
                "code": "unknown_model",
                "message": f"the gateway does not serve model {self.report.model!r}",
                "hint": f"Set {MODEL_ENV} (or --model) to a model the gateway lists at "
                "GET /v1/models. Nothing was recorded against any review.",
            }
            return False
        self.model_checked = True
        return True

    def review(self, project_id: str, review_id: str) -> bool:
        """Handle one review not drafted yet; ``False`` stops the pass."""
        review = reviews.get_review(self.log, project_id, review_id)
        project = self.registry.get(project_id)
        if (
            review is None
            or review.decision is None
            or review.pending
            or review.withdrawal is not None
            or project is None
        ):
            self.report.not_decided += 1
            return True
        state = _LessonState.read(self.log)
        found = state.sourced.get((project_id, review_id), 0)
        if found:  # proposals but no marker: a pass stopped between the two
            outcome = ReviewOutcome(project_id, review_id, "recovered", proposed=found)
            self.mark(outcome, model=None)
            return True
        if self.report.calls >= self.report.limit:
            return False
        if state.pending + MAX_DRAFTS > lessons.MAX_PENDING:
            self.report.stopped = {
                "reason": "backlog_full",
                "code": "too_many_proposals",
                "message": f"{state.pending} lesson proposals are waiting and the store holds at "
                f"most {lessons.MAX_PENDING}; drafting resumes once a person accepts or drops "
                "some on /lessons",
                "pending": state.pending,
                "max_pending": lessons.MAX_PENDING,
            }
            return False
        if not self.model_served():
            return False
        client = self.client_of(project_id)
        feedback = _feedback(review, project, client)
        messages = build_messages(feedback, state.existing(project_id, client))
        self.report.calls += 1
        try:
            reply = self.client.chat(self.report.model, messages, max_tokens=MAX_TOKENS)
        except GatewayError as error:
            if error.code in SETUP_FAILURES:
                self.report.stopped = _gateway_stop(error, project_id, review_id)
                return False
            return self.ambiguous(project_id, review_id, error.code, error.message)
        self.report.usage.update(reply.usage)
        try:
            drafts = parse_drafts(reply.text, scopes=scopes_for(feedback))
        except DraftError as error:
            return self.ambiguous(project_id, review_id, error.code, error.message)
        # The model drafted for this review, so the setup works: every ambiguous failure of this
        # pass, before this review or after it, is its review's own.
        self.proven = True
        for outcome in self.unproven:
            self.tally(outcome, charged=True)
        self.unproven.clear()
        self.in_a_row = 0
        self.record(project_id, review_id, client, drafts, reply)
        return True

    def ambiguous(self, project_id: str, review_id: str, code: str, message: str) -> bool:
        """A failure that may be the review's or the setup's: charged at once when the model
        already drafted a review in this pass, else held until the pass shows which. ``False``
        (stop) after :data:`MAX_IN_A_ROW` in a row."""
        outcome = ReviewOutcome(project_id, review_id, "failed", code=code, message=message)
        self.report.reviews.append(outcome)
        if self.proven:
            self.tally(outcome, charged=True)
        else:
            self.unproven.append(outcome)
        self.in_a_row += 1
        if self.in_a_row < MAX_IN_A_ROW:
            return True
        if self.proven:
            hint = (
                f"{MAX_IN_A_ROW} reviews in a row failed after a review was drafted in this "
                "pass, so the setup works and each failure is charged to its review; a review "
                f"is given up after {MAX_ATTEMPTS} charged attempts."
            )
        else:
            hint = (
                f"{MAX_IN_A_ROW} reviews in a row failed and no review was drafted in this pass, "
                "which may be the setup's fault, so none is charged: each gets a strike. The "
                "next pass retries them."
            )
        self.report.stopped = {
            "reason": "failures_in_a_row",
            "code": code,
            "message": message,
            "hint": hint,
            "project_id": project_id,
            "review_id": review_id,
        }
        return False

    def tally(self, outcome: ReviewOutcome, *, charged: bool) -> None:
        """Record an ambiguous failure as a charged attempt or a strike, under the review's next
        free key, and note both counts in the outcome."""
        project_id, review_id = outcome.project_id, outcome.review_id
        before = _Standing.read(self.log, project_id, review_id)
        if charged:
            kind, used, cap = "attempt", before.attempts, MAX_ATTEMPTS
            event_type, key = EVENT_DRAFT_FAILED, failed_key(project_id, review_id, used + 1)
        else:
            kind, used, cap = "strike", before.strikes, MAX_STRIKES
            event_type, key = EVENT_DRAFT_STRIKE, strike_key(project_id, review_id, used + 1)
        outcome.charged = charged
        if used < cap:
            payload = {
                "project_id": project_id,
                "review_id": review_id,
                "code": outcome.code,
                kind: used + 1,
            }
            try:
                self.log.append(event_type, self.report.actor, payload, idempotency_key=key)
            except IdempotencyConflict:
                outcome.problems.append(
                    f"another event already holds {kind} {used + 1}'s key; the {kind} counts"
                )
        after = _Standing.read(self.log, project_id, review_id)
        outcome.attempts, outcome.strikes, outcome.final = (
            after.attempts,
            after.strikes,
            after.given_up,
        )

    def record(
        self,
        project_id: str,
        review_id: str,
        client: str | None,
        drafts: DraftResult,
        reply: ChatReply,
    ) -> None:
        """Propose each draft through the store, then write the marker. A review that another
        pass recorded while the model answered is left as it is."""
        marked = _drafted(self.log, project_id, review_id)
        found = _LessonState.read(self.log).sourced.get((project_id, review_id), 0)
        if marked or found:
            outcome = ReviewOutcome(project_id, review_id, "recovered", proposed=found)
            outcome.problems.append(
                "another pass recorded this review while the model answered; this reply is "
                "not recorded"
            )
            if marked:
                self.report.reviews.append(outcome)
            else:
                self.mark(outcome, model=None)
            return
        outcome = ReviewOutcome(
            project_id,
            review_id,
            "drafted",
            problems=list(drafts.problems),
            reply_model=reply.model,
        )
        refused_for: list[str] = []
        for index, draft in enumerate(drafts.drafts):
            try:
                lesson = lessons.propose(
                    self.log,
                    draft.text,
                    _store_scope(draft.scope, project_id, client),
                    self.report.actor,
                    why=draft.why,
                    source=(project_id, review_id),
                    idempotency_key=proposal_key(project_id, review_id, index),
                )
            except WriteError as refused:
                refused_for.append(refused.code)
                outcome.problems.append(
                    f"draft {index + 1}: the lessons store refused it ({refused.code}): "
                    f"{refused.message}"
                )
                continue
            outcome.lessons.append({"id": lesson.id, "text": lesson.text, "scope": lesson.scope})
        if refused_for and not outcome.lessons and set(refused_for) == {"too_many_proposals"}:
            outcome.outcome = "deferred"
            outcome.problems.append(
                "the store's backlog filled while the model answered, so no draft was recorded "
                "and no marker written: a later pass drafts this review again"
            )
            self.report.reviews.append(outcome)
            return
        self.mark(outcome, model=self.report.model)

    def mark(self, outcome: ReviewOutcome, *, model: str | None) -> None:
        """Write the review's marker (``model`` ``None``: recovered from proposals)."""
        proposed = outcome.proposed if model is None else len(outcome.lessons)
        payload = {
            "project_id": outcome.project_id,
            "review_id": outcome.review_id,
            "proposed": proposed,
            "model": model,
            "recovered": model is None,
        }
        key = drafted_key(outcome.project_id, outcome.review_id)
        try:
            self.log.append(EVENT_DRAFTED, self.report.actor, payload, idempotency_key=key)
        except IdempotencyConflict:
            outcome.problems.append(
                "another event already holds this review's marker key; it counts as drafted"
            )
        self.report.reviews.append(outcome)


__all__ = [
    "DEFAULT_ACTOR",
    "DEFAULT_LIMIT",
    "EVENT_DRAFTED",
    "EVENT_DRAFT_FAILED",
    "EVENT_DRAFT_STRIKE",
    "LOCK_SUFFIX",
    "MAX_ATTEMPTS",
    "MAX_IN_A_ROW",
    "MAX_LIMIT",
    "MAX_STRIKES",
    "MAX_TOKENS",
    "MODEL_ENV",
    "SETUP_FAILURES",
    "ChatClient",
    "GivenUp",
    "ProposeReport",
    "ReviewOutcome",
    "check_limit",
    "check_proposer",
    "drafted_key",
    "failed_key",
    "lessons_model",
    "proposal_key",
    "propose_from_reviews",
    "strike_key",
]
