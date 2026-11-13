"""Draft lessons from decided reviews with a model, through the local model gateway (S23b).

After a person decides a review in the console (approve or reject, with a reason and notes at
timeline frames), :func:`propose_from_reviews` asks a cheap model for at most three durable
editing lessons drawn from that feedback and records each one as a **proposal**
(:func:`aar.hub.lessons.propose`, as an agent). Nothing a model writes reaches a cut guide: a
person accepts, rescopes or drops each proposal on ``/lessons`` (S23a), and guides print only
accepted lessons.

**One pass** takes the decided reviews of every project, oldest decision first. They are found
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
(``recovered`` true, ``model`` null) and does not call the model.

**Failures** are classified so that a broken setup cannot use up the reviews' attempts:

- A setup or transport failure stops the pass at once and records nothing against the review,
  so the next pass retries it. This covers every :class:`~aar.llm.gateway.GatewayError` except
  ``bad-response`` and ``http-400`` (so ``not_configured``, ``origin``, ``redirect``,
  ``network``, ``timeout``, ``too-large``, and ``http-401``, ``403``, ``404``, ``429`` and
  ``5xx`` once the client has given up retrying). It also covers a model the gateway does not
  list (``unknown_model``): before its first chat a pass reads ``GET /v1/models``, because the
  gateway answers a chat for an unknown model with HTTP 400, which would otherwise count
  against every review.
- A failure of the review's own request or reply records ``lessons.draft_failed``
  (``{project_id, review_id, code, attempt}``, key
  ``lessons:draft-failed:<project_id>:<review_id>:<attempt>``): a reply that is not the lessons
  JSON (``bad-output``), one the client refuses (``bad-response``), or ``http-400``. After
  :data:`MAX_ATTEMPTS` attempts the review is given up for good, and every later report counts
  it.

The marker and the attempts are counted by their idempotency keys, whatever event holds them. So
no event (malformed, forged or from a concurrent pass) can make a review retry forever: each
failed attempt takes a key, and three keys end it. The marker and the failure event are
bookkeeping, not lesson events: the lessons fold never reads them.

**Bounds.** At most ``limit`` chats per pass (1 to :data:`MAX_LIMIT`, default
:data:`DEFAULT_LIMIT`). Each reply is at most :data:`MAX_TOKENS` tokens, and the prompt is
bounded by ``lesson_drafts.PROMPT_BUDGET``. At most 3 drafts per review. Before each chat the
pass stops if ``pending + 3`` would go past the store's :data:`~aar.hub.lessons.MAX_PENDING`;
a full backlog is reported, not an error.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

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
DEFAULT_ACTOR = Actor("agent", "lessons-proposer")
MODEL_ENV = "AAR_LESSONS_MODEL"
DEFAULT_LIMIT = 5
MAX_LIMIT = 50
MAX_ATTEMPTS = 3
#: Reply tokens a chat may use: three one-line lessons need a few hundred.
MAX_TOKENS = 4000
#: Failures of one review's own request or reply. Every other gateway error stops the pass.
REVIEW_FAILURES = frozenset({"bad-output", "bad-response", "http-400"})
#: Given-up reviews listed in a report (all are counted).
MAX_LISTED = 20


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
    """What one pass did with one review: ``drafted``, ``recovered`` or ``failed``."""

    project_id: str
    review_id: str
    outcome: str
    lessons: list[dict[str, str]] = field(default_factory=list)  # drafted: {id, text, scope}
    problems: list[str] = field(default_factory=list)  # drafts dropped or refused, bookkeeping
    proposed: int = 0  # recovered: the proposals found
    reply_model: str | None = None  # drafted: the model the gateway says answered
    code: str | None = None  # failed
    message: str = ""  # failed
    attempt: int | None = None  # failed
    final: bool = False  # failed: no attempt left

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
        else:
            out.update(code=self.code, message=self.message, attempt=self.attempt, final=self.final)
        out["problems"] = list(self.problems)
        return out


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
    given_up: list[tuple[str, str]] = field(default_factory=list)
    not_decided: int = 0  # decided by event, but not a decided review once confirmed
    waiting: int = 0  # reviews left for a later pass
    stopped: dict[str, Any] | None = None

    @property
    def proposed(self) -> int:
        """The lessons proposed in this pass."""
        return sum(len(outcome.lessons) for outcome in self.reviews)

    @property
    def failed(self) -> bool:
        """Whether a setup or transport failure stopped the pass (a full backlog is not one)."""
        return self.stopped is not None and self.stopped["reason"] != "backlog_full"

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
                "reviews": [
                    {"project_id": project_id, "review_id": review_id}
                    for project_id, review_id in self.given_up[:MAX_LISTED]
                ],
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
    """The idempotency key of a review's ``attempt``-th failure (1 to :data:`MAX_ATTEMPTS`)."""
    return f"lessons:draft-failed:{project_id}:{review_id}:{attempt}"


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


def _next_attempt(log: EventLog, project_id: str, review_id: str) -> int | None:
    """The first attempt whose key is free, or ``None`` when all are used (given up)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if log.get_by_idempotency_key(failed_key(project_id, review_id, attempt)) is None:
            return attempt
    return None


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
    """One bounded pass: draft lessons from the decided reviews not drafted yet, oldest decision
    first, with at most ``limit`` model calls (see the module doc). Raises ``WriteError`` for an
    actor that is not an agent and ``ValueError`` for a bad model id or limit, before anything is
    read or sent; every other failure is in the report."""
    actor = check_proposer(actor)
    if not is_model_id(model):
        raise ValueError("model must be 1-128 printable ASCII characters without spaces")
    report = ProposeReport(model=model, actor=actor, limit=check_limit(limit), started="")
    report.started = iso(clock())
    _Pass(log, client, report).run()
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

    def run(self) -> None:
        candidates = _decided_reviews(self.log)
        for index, (project_id, review_id) in enumerate(candidates):
            if _drafted(self.log, project_id, review_id):
                continue
            attempt = _next_attempt(self.log, project_id, review_id)
            if attempt is None:
                self.report.given_up.append((project_id, review_id))
                continue
            if not self.review(project_id, review_id, attempt):
                self.count_rest(candidates[index:])
                return

    def count_rest(self, rest: Sequence[tuple[str, str]]) -> None:
        """After a stop: the reviews left, waiting or given up (cheap key lookups only)."""
        for project_id, review_id in rest:
            if _drafted(self.log, project_id, review_id):
                continue
            if _next_attempt(self.log, project_id, review_id) is None:
                self.report.given_up.append((project_id, review_id))
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
        serve would be answered with HTTP 400, a review-specific failure, for every review."""
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

    def review(self, project_id: str, review_id: str, attempt: int) -> bool:
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
            if error.code not in REVIEW_FAILURES:
                self.report.stopped = _gateway_stop(error, project_id, review_id)
                return False
            failure = (error.code, error.message)
        else:
            self.report.usage.update(reply.usage)
            try:
                drafts = parse_drafts(reply.text, scopes=scopes_for(feedback))
            except DraftError as error:
                failure = (error.code, error.message)
            else:
                self.record(project_id, review_id, client, drafts, reply)
                return True
        self.fail(project_id, review_id, attempt, *failure)
        return True

    def record(
        self,
        project_id: str,
        review_id: str,
        client: str | None,
        drafts: DraftResult,
        reply: ChatReply,
    ) -> None:
        """Propose each draft through the store, then write the marker."""
        outcome = ReviewOutcome(
            project_id,
            review_id,
            "drafted",
            problems=list(drafts.problems),
            reply_model=reply.model,
        )
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
                outcome.problems.append(
                    f"draft {index + 1}: the lessons store refused it ({refused.code}): "
                    f"{refused.message}"
                )
                continue
            outcome.lessons.append({"id": lesson.id, "text": lesson.text, "scope": lesson.scope})
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

    def fail(self, project_id: str, review_id: str, attempt: int, code: str, message: str) -> None:
        """Record a failure of this review's own request or reply (one attempt used)."""
        outcome = ReviewOutcome(
            project_id, review_id, "failed", code=code, message=message, attempt=attempt
        )
        payload = {
            "project_id": project_id,
            "review_id": review_id,
            "code": code,
            "attempt": attempt,
        }
        key = failed_key(project_id, review_id, attempt)
        try:
            self.log.append(EVENT_DRAFT_FAILED, self.report.actor, payload, idempotency_key=key)
        except IdempotencyConflict:
            outcome.problems.append(
                f"another event already holds attempt {attempt}'s key; the attempt counts"
            )
        outcome.final = _next_attempt(self.log, project_id, review_id) is None
        self.report.reviews.append(outcome)


__all__ = [
    "DEFAULT_ACTOR",
    "DEFAULT_LIMIT",
    "EVENT_DRAFTED",
    "EVENT_DRAFT_FAILED",
    "MAX_ATTEMPTS",
    "MAX_LIMIT",
    "MAX_TOKENS",
    "MODEL_ENV",
    "REVIEW_FAILURES",
    "ChatClient",
    "ProposeReport",
    "ReviewOutcome",
    "check_limit",
    "check_proposer",
    "drafted_key",
    "failed_key",
    "lessons_model",
    "proposal_key",
    "propose_from_reviews",
]
