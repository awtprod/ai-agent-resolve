"""The lessons proposer (S23b): which reviews a pass drafts, the once-per-review marker and crash
recovery, what the model is given, how drafts are recorded, failure classes, bounds, and
``aar hub lessons propose``.

The pass tests use :class:`FakeClient` (scripted replies, every call recorded). The CLI builds a
real gateway client from the environment, so its tests point it at the loopback stub of
``tests/analyze/gemini_stub.py``, as ``tests/llm`` does.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from write_support import AGENT, HUMAN, Crash, Hub, brief_data, crash_after_appends

from aar.cli import main
from aar.hub import lessons, reviews
from aar.hub import lessons_proposer as proposer
from aar.hub.events import Actor
from aar.hub.graph import WorkGraph
from aar.hub.lesson_drafts import DEFAULT_LESSONS_MODEL
from aar.hub.registry import Registry
from aar.hub.writes import WriteError, create_brief
from aar.llm.gateway import CHAT_PATH, MODELS_PATH, ChatReply, GatewayError

_ANALYZE = str(Path(__file__).resolve().parents[1] / "analyze")
if _ANALYZE not in sys.path:
    sys.path.insert(0, _ANALYZE)
from gemini_stub import Stub  # noqa: E402

MODEL = "test-model"
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


class FakeClient:
    """Stands in for the gateway client: replies played in order (text, or an exception to
    raise), the served models, and every call recorded. A chat with no reply left fails."""

    def __init__(self, *replies: Any, served: Any = (MODEL,)) -> None:
        self.replies = list(replies)
        self.served = served
        self.calls: list[dict[str, Any]] = []
        self.model_reads = 0

    def models(self) -> tuple[str, ...]:
        self.model_reads += 1
        if isinstance(self.served, Exception):
            raise self.served
        return tuple(self.served)

    def chat(
        self, model: str, messages: Any, *, max_tokens: int, temperature: float | None = None
    ) -> ChatReply:
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        assert self.replies, "the pass called the model with no reply scripted"
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatReply(text=reply, model=model, usage=USAGE)


def item(text: str, scope: str = "project", why: str = "From the review.") -> dict[str, str]:
    return {"text": text, "scope": scope, "why": why}


def drafts(*items: dict[str, str]) -> str:
    return json.dumps({"lessons": list(items)})


def review(
    hub: Hub,
    decision: str | None = "reject",
    reason: str = "The ending runs long after the big laugh.",
    *,
    project_id: str | None = None,
    notes: tuple[tuple[int, str], ...] = (),
    agent_notes: tuple[tuple[int, str], ...] = (),
) -> str:
    """A review of a new cut (24 fps), with notes, decided by a person unless ``decision`` is
    ``None``. Returns its id."""
    project_id = project_id or hub.project_id
    op = WorkGraph(hub.log, project_id).add_node("op", AGENT, title="cut").id
    render = hub.store.put(b"render", actor=AGENT, mime="video/mp4", project_id=project_id)
    made = reviews.request_review(
        hub.log, hub.store, project_id, op, [render], AGENT, title="gate review (round 1)", fps=24
    )
    for frame, text in notes:
        reviews.add_note(hub.log, project_id, made.id, frame, text, HUMAN)
    for frame, text in agent_notes:
        reviews.add_note(hub.log, project_id, made.id, frame, text, AGENT)
    if decision is not None:
        reviews.decide(hub.log, project_id, made.id, decision, reason, HUMAN)
    return made.id


def run(hub: Hub, client: FakeClient, **options: Any) -> proposer.ProposeReport:
    return proposer.propose_from_reviews(
        hub.log, client, model=MODEL, actor=proposer.DEFAULT_ACTOR, **options
    )


def payloads(hub: Hub, event_type: str) -> list[dict[str, Any]]:
    return [event.payload for event in hub.log.iter_events(types=[event_type])]


def event_count(hub: Hub) -> int:
    return sum(1 for _ in hub.log.iter_events())


def block(call: dict[str, Any]) -> dict[str, Any]:
    """The review data the model was given: the user message's JSON block."""
    match = re.fullmatch(r"(?s)[^`]*```json\n(.*)\n```", call["messages"][1]["content"])
    assert match, call["messages"][1]["content"]
    return json.loads(match.group(1))


def proposals(hub: Hub) -> list[lessons.Lesson]:
    found, skipped = lessons.scan_lessons(hub.log)
    assert skipped == ()
    return [lesson for lesson in found if lesson.proposal is not None]


def add_ccn_brief(hub: Hub, project_id: str | None = None) -> None:
    project_id = project_id or hub.project_id
    create_brief(hub.log, brief_data(project_id), AGENT, idempotency_key=f"brief-{project_id}")


# -- which reviews, in which order, and only once ----------------------------------------------


def test_approved_and_rejected_reviews_are_drafted_and_no_others(hub: Hub) -> None:
    approved = review(hub, "approve", "Good pace; it ends on the laugh.")
    still_open = review(hub, None)
    withdrawn = review(hub, None)
    reviews.withdraw(hub.log, hub.project_id, withdrawn, "Superseded by round 2.", AGENT)
    rejected = review(hub, "reject", "The opening drags for a minute.")
    # Decisions the reviews fold rejects: an agent's, and a person's appended without the
    # status change a real decision is written with.
    forged = {"decision": "approve", "reason": "Looks fine."}
    hub.log.append(
        reviews.EVENT_DECIDED, AGENT, {"review_id": still_open, **forged}, project_id=hub.project_id
    )
    hub.log.append(
        reviews.EVENT_DECIDED, HUMAN, {"review_id": withdrawn, **forged}, project_id=hub.project_id
    )
    client = FakeClient(drafts(item("End on the laugh.")), drafts(item("Open on the first joke.")))

    report = run(hub, client)

    assert [(o.review_id, o.outcome) for o in report.reviews] == [
        (approved, "drafted"),
        (rejected, "drafted"),
    ]
    assert [(block(c)["review_id"], block(c)["decision"]) for c in client.calls] == [
        (approved, "approve"),
        (rejected, "reject"),
    ]
    assert report.not_decided == 2 and report.calls == 2 and report.stopped is None
    assert client.model_reads == 1  # the model list is read once per pass, before the first chat
    assert [m["review_id"] for m in payloads(hub, proposer.EVENT_DRAFTED)] == [approved, rejected]
    assert [lesson.text for lesson in proposals(hub)] == [
        "End on the laugh.",
        "Open on the first joke.",
    ]


def test_the_oldest_decision_goes_first_and_the_limit_is_honoured(hub: Hub) -> None:
    first, second, third = (review(hub, None) for _ in range(3))
    for review_id in (third, first, second):  # decided in another order than requested
        reviews.decide(hub.log, hub.project_id, review_id, "reject", f"Reason {review_id}.", HUMAN)
    client = FakeClient(drafts(item("One.")), drafts(item("Two.")))

    report = run(hub, client, limit=2)

    assert [block(call)["review_id"] for call in client.calls] == [third, first]
    assert [o.review_id for o in report.reviews] == [third, first]
    assert (report.calls, report.waiting, report.stopped) == (2, 1, None)
    later = FakeClient(drafts(item("Three.")))
    report = run(hub, later, limit=2)
    assert [o.review_id for o in report.reviews] == [second] and report.waiting == 0


def test_a_second_pass_calls_no_model_and_writes_nothing(hub: Hub) -> None:
    drafted = review(hub)
    run(hub, FakeClient(drafts(item("End on the laugh."))))
    # Another review whose marker key some other event already holds also counts as done.
    held = review(hub)
    hub.log.append(
        "test.other", AGENT, {}, idempotency_key=proposer.drafted_key(hub.project_id, held)
    )
    before = event_count(hub)
    silent = FakeClient(served=AssertionError("the pass must not ask for the models"))

    report = run(hub, silent)

    assert silent.calls == [] and silent.model_reads == 0
    assert report.reviews == [] and report.calls == 0 and report.waiting == 0
    assert event_count(hub) == before
    assert [m["review_id"] for m in payloads(hub, proposer.EVENT_DRAFTED)] == [drafted]


def test_proposals_left_without_a_marker_get_one_and_no_model_call(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    drafted = review(hub)
    # The pass dies while writing its second proposal: that transaction rolls back, the first
    # proposal stays, and the marker is never written.
    crash_after_appends(monkeypatch, 2)
    with pytest.raises(Crash):
        run(hub, FakeClient(drafts(item("End on the laugh."), item("Cut the applause."))))
    monkeypatch.undo()
    assert payloads(hub, proposer.EVENT_DRAFTED) == []
    [orphan] = proposals(hub)
    assert orphan.text == "End on the laugh."
    lessons.drop(hub.log, orphan.id, HUMAN)  # a person decided it before the next pass
    silent = FakeClient()

    report = run(hub, silent)

    assert silent.calls == [] and silent.model_reads == 0
    assert [(o.review_id, o.outcome, o.proposed) for o in report.reviews] == [
        (drafted, "recovered", 1)
    ]
    assert payloads(hub, proposer.EVENT_DRAFTED) == [
        {
            "project_id": hub.project_id,
            "review_id": drafted,
            "proposed": 1,
            "model": None,
            "recovered": True,
        }
    ]
    assert run(hub, FakeClient()).reviews == []  # and from then on it is done


class RacingClient(FakeClient):
    """A client during whose chat another pass writes an event under ``key`` (``payload``)."""

    def __init__(self, hub: Hub, key: str, *replies: Any) -> None:
        super().__init__(*replies)
        self.hub, self.key = hub, key

    def chat(self, model: str, messages: Any, **options: Any) -> ChatReply:
        self.hub.log.append(
            "test.other_pass", AGENT, {"by": "the other pass"}, idempotency_key=self.key
        )
        return super().chat(model, messages, **options)


def test_a_marker_or_attempt_another_pass_wrote_meanwhile_is_kept_and_this_pass_goes_on(
    hub: Hub,
) -> None:
    first, second = review(hub), review(hub)
    racing = RacingClient(
        hub, proposer.drafted_key(hub.project_id, first), drafts(item("One.")), "not json"
    )

    report = run(hub, racing, limit=1)

    [outcome] = report.reviews
    assert (outcome.outcome, len(outcome.lessons)) == ("drafted", 1)
    assert outcome.problems == [
        "another event already holds this review's marker key; it counts as drafted"
    ]
    assert payloads(hub, proposer.EVENT_DRAFTED) == []  # the other pass's event stands
    racing = RacingClient(hub, proposer.failed_key(hub.project_id, second, 1), "not json")
    report = run(hub, racing)
    [outcome] = report.reviews
    assert (outcome.review_id, outcome.outcome, outcome.attempt) == (second, "failed", 1)
    assert outcome.problems == ["another event already holds attempt 1's key; the attempt counts"]
    assert proposer.failed_key(hub.project_id, second, 1) not in {
        e.idempotency_key for e in hub.log.iter_events(types=[proposer.EVENT_DRAFT_FAILED])
    }
    assert run(hub, FakeClient("not json")).reviews[0].attempt == 2  # the next key, not 1 again


# -- what the model is given, and how drafts are recorded --------------------------------------


def test_drafts_are_proposed_by_the_agent_with_their_scope_why_and_source(hub: Hub) -> None:
    add_ccn_brief(hub)
    drafted = review(hub)
    client = FakeClient(
        drafts(
            item("Keep the host's pauses.", "global", "Note 1."),
            item("Prefer the wide camera on laughs.", "client", "Note 2."),
            item("Keep the cold open under a minute.", "project", "The reason."),
        )
    )

    report = run(hub, client)

    assert block(client.calls[0])["client"] == "ccn"
    assert '"client": this client' in client.calls[0]["messages"][0]["content"]
    assert client.calls[0]["model"] == MODEL
    assert client.calls[0]["max_tokens"] == proposer.MAX_TOKENS
    found = proposals(hub)
    assert [(lesson.text, lesson.scope, lesson.state) for lesson in found] == [
        ("Keep the host's pauses.", "global", "proposed"),
        ("Prefer the wide camera on laughs.", "client:ccn", "proposed"),
        ("Keep the cold open under a minute.", f"project:{hub.project_id}", "proposed"),
    ]
    for lesson in found:
        assert lesson.proposal is not None
        assert lesson.proposal.by == proposer.DEFAULT_ACTOR
        assert lesson.proposal.source == lessons.LessonSource(hub.project_id, drafted)
    assert [lesson.proposal.why for lesson in found if lesson.proposal] == [
        "Note 1.",
        "Note 2.",
        "The reason.",
    ]
    keys = [e.idempotency_key for e in hub.log.iter_events(types=[lessons.EVENT_PROPOSED])]
    assert keys == [f"lessons:proposal:{hub.project_id}:{drafted}:{i}" for i in range(3)]
    [outcome] = report.reviews
    assert [entry["id"] for entry in outcome.lessons] == [lesson.id for lesson in found]
    assert payloads(hub, proposer.EVENT_DRAFTED) == [
        {
            "project_id": hub.project_id,
            "review_id": drafted,
            "proposed": 3,
            "model": MODEL,
            "recovered": False,
        }
    ]
    assert report.proposed == 3 and report.usage == USAGE


def _raw_brief(hub: Hub, rule_profile: str) -> None:
    """A typed brief written straight into the graph, so its profile need not be built in."""
    data = brief_data(hub.project_id, rule_profile=rule_profile)
    WorkGraph(hub.log, hub.project_id).add_node("brief", AGENT, title="brief", data=data)


@pytest.mark.parametrize(
    "briefs",
    [
        pytest.param([], id="no-brief"),
        pytest.param(["ccn", "other-client"], id="briefs-disagree"),
        pytest.param(["other-client"], id="not-a-built-in-profile"),
    ],
)
def test_without_one_shared_built_in_client_only_global_and_project_are_offered(
    hub: Hub, briefs: list[str]
) -> None:
    for profile in briefs:
        _raw_brief(hub, profile)
    lessons.add(hub.log, "A ccn rule.", "client:ccn", HUMAN)
    review(hub)
    client = FakeClient(drafts(item("A client rule.", "client"), item("A project rule.")))

    report = run(hub, client)

    given = block(client.calls[0])
    assert given["client"] is None and "client" not in given["existing_lessons"]
    assert "do not use the client scope" in client.calls[0]["messages"][0]["content"]
    [outcome] = report.reviews
    assert outcome.problems == ["lesson 1: scope 'client' is not one of global, project"]
    assert [(lesson.text, lesson.scope) for lesson in proposals(hub)] == [
        ("A project rule.", f"project:{hub.project_id}")
    ]


def test_existing_lessons_are_the_accepted_and_waiting_ones_of_this_review_s_scopes(
    hub: Hub,
) -> None:
    acme, other = f"project:{hub.project_id}", f"project:{hub.other_project_id}"
    lessons.add(hub.log, "Global accepted.", "global", HUMAN)
    lessons.propose(hub.log, "Global waiting.", "global", AGENT)
    later = lessons.propose(hub.log, "Global, accepted after.", "global", AGENT)
    lessons.accept(hub.log, later.id, "Global, accepted after.", "global", HUMAN)
    lessons.add(hub.log, "Client accepted.", "client:ccn", HUMAN)
    lessons.add(hub.log, "Acme accepted.", acme, HUMAN)
    lessons.propose(hub.log, "Acme waiting.", acme, AGENT)
    dropped = lessons.propose(hub.log, "Acme dropped.", acme, AGENT)
    lessons.drop(hub.log, dropped.id, HUMAN)
    retired = lessons.add(hub.log, "Acme retired.", acme, HUMAN)
    lessons.retire(hub.log, retired.id, HUMAN)
    lessons.add(hub.log, "Other project accepted.", other, HUMAN)
    lessons.propose(hub.log, "Other project waiting.", other, AGENT)
    add_ccn_brief(hub)
    review(hub)
    client = FakeClient(drafts())

    run(hub, client)

    assert block(client.calls[0])["existing_lessons"] == {
        "global": ["Global accepted.", "Global, accepted after.", "Global waiting."],
        "client": ["Client accepted."],
        "project": ["Acme accepted.", "Acme waiting."],
    }


def test_the_model_reads_the_review_and_only_the_notes_a_person_wrote(hub: Hub) -> None:
    review(
        hub,
        "reject",
        "The middle sags.",
        notes=((240, "Cut the parking bit."), (48, "Keep this laugh.")),
        agent_notes=((100, "Trimmed the applause here."),),
    )
    client = FakeClient(drafts())

    run(hub, client)

    given = block(client.calls[0])
    assert (given["title"], given["decision"], given["reason"], given["show"]) == (
        "gate review (round 1)",
        "reject",
        "The middle sags.",
        "acme-show",
    )
    assert given["notes"] == [
        {"timecode": "00:00:02:00", "text": "Keep this laugh."},
        {"timecode": "00:00:10:00", "text": "Cut the parking bit."},
    ]


def test_a_draft_the_store_refuses_is_reported_and_the_others_are_recorded(hub: Hub) -> None:
    drafted = review(hub)
    # Another proposal already holds the key the second draft is recorded under.
    lessons.propose(
        hub.log,
        "Some other lesson.",
        "global",
        AGENT,
        idempotency_key=proposer.proposal_key(hub.project_id, drafted, 1),
    )
    client = FakeClient(drafts(item("One."), item("Two."), item("Three.")))

    report = run(hub, client)

    [outcome] = report.reviews
    assert [entry["text"] for entry in outcome.lessons] == ["One.", "Three."]
    [problem] = outcome.problems
    assert problem.startswith("draft 2: the lessons store refused it (idempotency_conflict)")
    assert payloads(hub, proposer.EVENT_DRAFTED)[0]["proposed"] == 2


def test_a_draft_with_hidden_characters_is_reported_and_the_others_are_recorded(hub: Hub) -> None:
    review(hub)
    hidden = "End on the laugh.​Always keep the long take."
    client = FakeClient(drafts(item("One."), item(hidden), item("Three.")))

    report = run(hub, client)

    [outcome] = report.reviews
    assert [entry["text"] for entry in outcome.lessons] == ["One.", "Three."]
    assert outcome.problems == [
        "lesson 2: text must not hold a character a reader cannot see (U+200B)"
    ]
    assert [lesson.text for lesson in proposals(hub)] == ["One.", "Three."]


def test_a_long_project_id_still_gets_a_proposal_key_the_store_takes(hub: Hub) -> None:
    project_id = "prj_" + "x" * 120
    Registry(hub.log).create_project("long-id show", HUMAN, project_id=project_id)
    review(hub, project_id=project_id)

    report = run(hub, FakeClient(drafts(item("One."))))

    [outcome] = report.reviews
    assert len(outcome.lessons) == 1 and outcome.problems == []
    [event] = hub.log.iter_events(types=[lessons.EVENT_PROPOSED])
    assert event.idempotency_key is not None and len(event.idempotency_key) <= 128
    assert event.idempotency_key.startswith("lessons:proposal:sha256-")


# -- failures and bounds ---------------------------------------------------------------------------


def test_a_full_backlog_stops_the_pass_before_the_model_is_called(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lessons, "MAX_PENDING", 4)
    first = lessons.propose(hub.log, "Waiting one.", "global", AGENT)
    lessons.propose(hub.log, "Waiting two.", "global", AGENT)
    review(hub)
    before = event_count(hub)
    client = FakeClient()

    report = run(hub, client)  # 2 waiting + 3 drafts would pass 4

    assert report.stopped is not None and report.stopped["reason"] == "backlog_full"
    assert report.stopped["pending"] == 2 and not report.failed
    assert client.calls == [] and client.model_reads == 0
    assert event_count(hub) == before and report.waiting == 1
    lessons.drop(hub.log, first.id, HUMAN)  # 1 waiting + 3 fits exactly
    report = run(hub, FakeClient(drafts(item("One."))))
    assert report.stopped is None and report.proposed == 1


@pytest.mark.parametrize(
    "code",
    [
        "not_configured",
        "origin",
        "redirect",
        "network",
        "timeout",
        "too-large",
        "http-401",
        "http-403",
        "http-404",
        "http-429",
        "http-500",
        "http-503",
    ],
)
def test_a_setup_or_transport_failure_stops_the_pass_and_records_nothing(
    hub: Hub, code: str
) -> None:
    first, second = review(hub), review(hub)
    before = event_count(hub)
    client = FakeClient(GatewayError(code, f"POST {CHAT_PATH}: {code}", "Check the gateway."))

    report = run(hub, client)

    assert report.failed and report.stopped == {
        "reason": "gateway",
        "code": code,
        "message": f"POST {CHAT_PATH}: {code}",
        "hint": "Check the gateway.",
        "project_id": hub.project_id,
        "review_id": first,
    }
    assert len(client.calls) == 1  # the second review was not tried
    assert event_count(hub) == before and report.reviews == [] and report.waiting == 2
    retry = FakeClient(drafts(item("One.")), drafts(item("Two.")))
    report = run(hub, retry)
    assert [o.review_id for o in report.reviews] == [first, second]


def test_the_model_must_be_one_the_gateway_serves_before_any_chat(hub: Hub) -> None:
    review(hub)
    before = event_count(hub)
    unknown = FakeClient(served=("another-model",))

    report = run(hub, unknown)

    assert report.failed and report.stopped is not None
    assert (report.stopped["reason"], report.stopped["code"]) == ("setup", "unknown_model")
    assert unknown.calls == [] and event_count(hub) == before
    down = FakeClient(served=GatewayError("network", "GET /v1/models: refused", "Is it up?"))
    report = run(hub, down)
    assert report.stopped is not None and report.stopped["code"] == "network"
    assert down.calls == [] and event_count(hub) == before


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        pytest.param("I think the cut is fine.", "bad-output", id="not-the-lessons-json"),
        pytest.param(GatewayError("bad-response", "no choices"), "bad-response", id="bad-response"),
        pytest.param(GatewayError("http-400", "HTTP 400: too long"), "http-400", id="http-400"),
    ],
)
def test_a_bad_reply_uses_one_attempt_and_the_third_ends_them(
    hub: Hub, failure: Any, code: str
) -> None:
    bad, good = review(hub), review(hub)
    report = run(hub, FakeClient(failure, drafts(item("End on the laugh."))))
    assert [(o.review_id, o.outcome, o.code, o.attempt, o.final) for o in report.reviews] == [
        (bad, "failed", code, 1, False),
        (good, "drafted", None, None, False),
    ]
    for attempt in (2, 3):
        report = run(hub, FakeClient(failure))
        assert [(o.review_id, o.attempt, o.final) for o in report.reviews] == [
            (bad, attempt, attempt == 3)
        ]
    silent = FakeClient()

    report = run(hub, silent)

    assert silent.calls == [] and report.reviews == []
    assert report.to_dict()["given_up"] == {
        "count": 1,
        "reviews": [{"project_id": hub.project_id, "review_id": bad}],
    }
    assert payloads(hub, proposer.EVENT_DRAFT_FAILED) == [
        {"project_id": hub.project_id, "review_id": bad, "code": code, "attempt": n}
        for n in (1, 2, 3)
    ]
    keys = [e.idempotency_key for e in hub.log.iter_events(types=[proposer.EVENT_DRAFT_FAILED])]
    assert keys == [proposer.failed_key(hub.project_id, bad, n) for n in (1, 2, 3)]
    assert [m["review_id"] for m in payloads(hub, proposer.EVENT_DRAFTED)] == [good]
    # Bookkeeping, not lessons: the lessons fold neither reads nor skips these events.
    assert lessons.scan_lessons(hub.log)[1] == ()


def test_an_attempt_key_held_by_another_event_still_ends_the_attempts(hub: Hub) -> None:
    bad = review(hub)
    hub.log.append(
        "test.other", AGENT, {}, idempotency_key=proposer.failed_key(hub.project_id, bad, 2)
    )
    attempts = [run(hub, FakeClient("not json")).reviews[0] for _ in range(2)]
    assert [(o.attempt, o.final) for o in attempts] == [(1, False), (3, True)]
    silent = FakeClient()
    report = run(hub, silent)
    assert silent.calls == [] and report.given_up == [(hub.project_id, bad)]


@pytest.mark.parametrize(
    ("actor", "code"),
    [
        (HUMAN, "actor_not_allowed"),
        (Actor("node", "node-1"), "actor_not_allowed"),
        (Actor("runner", "engine"), "actor_not_allowed"),
        (Actor("agent", "bad id!"), "invalid_input"),
    ],
)
def test_only_an_agent_runs_the_proposer(hub: Hub, actor: Actor, code: str) -> None:
    review(hub)
    before = event_count(hub)
    client = FakeClient()
    with pytest.raises(WriteError) as refused:
        proposer.propose_from_reviews(hub.log, client, model=MODEL, actor=actor)
    assert refused.value.code == code
    assert client.calls == [] and client.model_reads == 0 and event_count(hub) == before


@pytest.mark.parametrize(
    "options", [{"limit": 0}, {"limit": proposer.MAX_LIMIT + 1}, {"model": "two words"}]
)
def test_a_bad_limit_or_model_is_refused_before_anything_is_read(
    hub: Hub, options: dict[str, Any]
) -> None:
    client = FakeClient()
    with pytest.raises(ValueError):
        proposer.propose_from_reviews(
            hub.log, client, **{"model": MODEL, "actor": proposer.DEFAULT_ACTOR, **options}
        )
    assert client.calls == [] and client.model_reads == 0


# -- aar hub lessons propose ---------------------------------------------------------------------

KEY = "gw-test-key-0000"


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[Stub]:
    """The loopback stub as the model gateway, serving the default model, with the key set."""
    stub = Stub(key=KEY)
    stub.routes[("GET", MODELS_PATH)] = lambda s, h: s.reply(
        h, 200, {"data": [{"id": DEFAULT_LESSONS_MODEL}, {"id": "env-model"}]}
    )
    monkeypatch.setenv("AAR_LLM_BASE_URL", stub.base)
    monkeypatch.setenv("AAR_LLM_API_KEY", KEY)
    monkeypatch.delenv(proposer.MODEL_ENV, raising=False)
    yield stub
    stub.close()


def chat_reply(content: str, status: int = 200) -> Any:
    body = {"model": "served", "choices": [{"message": {"role": "assistant", "content": content}}]}
    return lambda s, h: s.reply(h, status, body if status == 200 else {"error": "refused"})


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, Any]:
    capsys.readouterr()
    code = main(["hub", "lessons", "propose", *argv])
    out, err = capsys.readouterr()
    return code, json.loads(out) if out.strip() else None, json.loads(err) if err.strip() else None


def test_propose_prints_the_report_and_a_second_run_calls_nothing(
    hub: Hub, gateway: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    drafted = review(hub)
    gateway.routes[("POST", CHAT_PATH)] = chat_reply(drafts(item("End on the laugh.")))
    db = str(hub.root / "hub.db")

    code, report, err = cli(capsys, "--db", db)

    assert (code, err) == (0, None)
    assert report["model"] == DEFAULT_LESSONS_MODEL and report["calls"] == 1
    assert report["actor"] == {"kind": "agent", "id": "lessons-proposer"}
    assert [(r["review_id"], r["outcome"], r["proposed"]) for r in report["reviews"]] == [
        (drafted, "drafted", 1)
    ]
    assert gateway.calls() == [f"GET {MODELS_PATH}", f"POST {CHAT_PATH}"]
    sent = json.loads(gateway.requests[1]["body"])
    assert sent["model"] == DEFAULT_LESSONS_MODEL and sent["max_tokens"] == proposer.MAX_TOKENS
    assert gateway.requests[1]["headers"]["authorization"] == f"Bearer {KEY}"

    code, report, err = cli(capsys, "--db", db)

    assert (code, err, report["calls"], report["reviews"]) == (0, None, 0, [])
    assert len(gateway.requests) == 2


def test_the_model_comes_from_the_flag_then_the_environment_then_the_default(
    hub: Hub, gateway: Stub, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = str(hub.root / "hub.db")
    assert cli(capsys, "--db", db)[1]["model"] == DEFAULT_LESSONS_MODEL
    monkeypatch.setenv(proposer.MODEL_ENV, "  ")  # blank counts as unset
    assert cli(capsys, "--db", db)[1]["model"] == DEFAULT_LESSONS_MODEL
    monkeypatch.setenv(proposer.MODEL_ENV, "env-model")
    assert cli(capsys, "--db", db, "--model", "flag-model")[1]["model"] == "flag-model"
    review(hub)
    gateway.routes[("POST", CHAT_PATH)] = chat_reply(drafts())
    code, report, _ = cli(capsys, "--db", db)
    assert (code, report["model"]) == (0, "env-model")
    assert json.loads(gateway.requests[-1]["body"])["model"] == "env-model"


def test_without_a_key_the_pass_does_not_start(
    hub: Hub, gateway: Stub, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    review(hub)
    monkeypatch.delenv("AAR_LLM_API_KEY")
    before = event_count(hub)

    code, out, err = cli(capsys, "--db", str(hub.root / "hub.db"))

    assert (code, out) == (2, None)
    assert err["error"]["code"] == "not_configured"
    assert "AAR_LLM_API_KEY" in err["error"]["message"]
    assert gateway.requests == [] and event_count(hub) == before


@pytest.mark.parametrize(
    ("status", "served", "code"),
    [(401, DEFAULT_LESSONS_MODEL, "http-401"), (200, "another-model", "unknown_model")],
)
def test_a_pass_stopped_by_the_gateway_exits_1_with_its_report(
    hub: Hub,
    gateway: Stub,
    capsys: pytest.CaptureFixture[str],
    status: int,
    served: str,
    code: str,
) -> None:
    review(hub)
    gateway.routes[("GET", MODELS_PATH)] = lambda s, h: s.reply(h, 200, {"data": [{"id": served}]})
    gateway.routes[("POST", CHAT_PATH)] = chat_reply(drafts(), status)
    before = event_count(hub)

    exit_code, report, err = cli(capsys, "--db", str(hub.root / "hub.db"))

    assert exit_code == 1
    assert report["stopped"]["code"] == code and err["error"]["code"] == code
    assert report["reviews"] == [] and event_count(hub) == before


def test_a_full_backlog_is_a_pass_that_ran(
    hub: Hub, gateway: Stub, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lessons, "MAX_PENDING", 3)
    lessons.propose(hub.log, "Waiting.", "global", AGENT)
    review(hub)

    code, report, err = cli(capsys, "--db", str(hub.root / "hub.db"))

    assert (code, err) == (0, None) and report["stopped"]["reason"] == "backlog_full"
    assert gateway.requests == []


@pytest.mark.parametrize(
    ("argv", "environ", "code"),
    [
        (["--limit", "0"], {}, "invalid_input"),
        (["--limit", "51"], {}, "invalid_input"),
        (["--model", "two words"], {}, "invalid_input"),
        ([], {proposer.MODEL_ENV: "two words"}, "invalid_input"),
        (["--actor", "human:operator-1"], {}, "actor_not_allowed"),
        (["--actor", "node:node-1"], {}, "actor_not_allowed"),
        (["--actor", "agent:"], {}, "invalid_input"),
        (["--actor", "agent:bad id"], {}, "invalid_input"),
        (["--actor", "runner:engine"], {}, "invalid_input"),
        ([], {"AAR_LLM_BASE_URL": "http://gateway.example.invalid:8317"}, "origin"),
    ],
)
def test_a_bad_argument_or_setting_exits_2_before_the_gateway_or_the_database(
    hub: Hub,
    gateway: Stub,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    environ: dict[str, str],
    code: str,
) -> None:
    review(hub)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    before = event_count(hub)

    exit_code, out, err = cli(capsys, "--db", str(hub.root / "hub.db"), *argv)

    assert (exit_code, out, err["error"]["code"]) == (2, None, code)
    assert "two words" not in err["error"]["message"]  # a setting's value is never echoed
    assert gateway.requests == [] and event_count(hub) == before


def test_a_missing_database_exits_2_and_is_not_created(
    tmp_path: Path, gateway: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "nope.db"
    code, out, err = cli(capsys, "--db", str(db))
    assert (code, out, err["error"]["code"]) == (2, None, "database_unavailable")
    assert not db.exists() and gateway.requests == []


def test_the_help_states_the_defaults_the_command_uses(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["hub", "lessons", "propose", "--help"]) == 0
    text = " ".join(capsys.readouterr().out.split())
    assert f"1-{proposer.MAX_LIMIT}, default {proposer.DEFAULT_LIMIT}" in text
    assert f"else {DEFAULT_LESSONS_MODEL}" in text
    actor = proposer.DEFAULT_ACTOR
    assert f"default {actor.kind}:{actor.id}" in text


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_a_report_to_a_full_stdout_exits_74(hub: Hub) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AAR_")}
    env["AAR_LLM_API_KEY"] = KEY  # nothing to draft, so the gateway is never asked
    program = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"
    argv = ["hub", "lessons", "propose", "--db", str(hub.root / "hub.db")]
    with open("/dev/full", "wb") as full:
        done = subprocess.run(
            [sys.executable, "-c", program, *argv],
            stdout=full,
            stderr=subprocess.PIPE,
            env=env,
            timeout=60,
            check=False,
        )
    assert done.returncode == 74, done.stderr[-300:]
    assert done.stderr.decode().endswith("No space left on device\n")
