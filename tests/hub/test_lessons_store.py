"""The lessons store (S23a): writers, the fold, bounds, idempotency and the guide feed."""

from __future__ import annotations

import json
from typing import Any

import pytest
from write_support import AGENT, HUMAN, OTHER, Hub

from aar.hub import lessons
from aar.hub.common import HumanRequired
from aar.hub.events import Actor
from aar.hub.graph import WorkGraph
from aar.hub.reviews import request_review
from aar.hub.writes import WriteError

NODE = Actor("node", "node-1")
RUNNER = Actor("runner", "engine")
OPERATOR_2 = Actor("human", "operator-2")


def _review(hub: Hub) -> tuple[str, str]:
    """A review in ``prj_acme`` to draw a lesson from: ``(project_id, review_id)``."""
    op = WorkGraph(hub.log, hub.project_id).add_node("op", AGENT, title="cut").id
    render = hub.store.put(b"render", actor=AGENT, mime="video/mp4", project_id=hub.project_id)
    review = request_review(
        hub.log, hub.store, hub.project_id, op, [render], AGENT, title="gate review (round 1)"
    )
    return hub.project_id, review.id


def _events(hub: Hub) -> list[Any]:
    return list(hub.log.iter_events(types=lessons.LESSON_EVENT_TYPES))


def _refused(code: str, call: Any, *args: Any, **kwargs: Any) -> WriteError:
    with pytest.raises(WriteError) as caught:
        call(*args, **kwargs)
    assert caught.value.code == code, caught.value.to_dict()
    assert caught.value.fix_hint
    return caught.value


# -- the writers ---------------------------------------------------------------------------


def test_each_writer_records_its_event_and_returns_the_lesson(hub: Hub) -> None:
    source = _review(hub)
    proposed = lessons.propose(
        hub.log,
        "Keep the setup with its punchline.",
        "client:ccn",
        AGENT,
        why="Round 1 was rejected for cutting the punchline.",
        source=source,
    )
    assert proposed.state == "proposed" and proposed.acceptance is None
    assert proposed.proposal is not None
    assert (proposed.text, proposed.scope) == ("Keep the setup with its punchline.", "client:ccn")
    assert proposed.proposal.why == "Round 1 was rejected for cutting the punchline."
    assert proposed.proposal.source == lessons.LessonSource(*source)
    assert proposed.proposal.by == AGENT
    assert lessons.LESSON_ID_RE.fullmatch(proposed.id)

    accepted = lessons.accept(
        hub.log, proposed.id, "Never cut between a setup and its punchline.", "global", HUMAN
    )
    assert accepted.state == "accepted" and accepted.acceptance is not None
    assert (accepted.text, accepted.scope) == (
        "Never cut between a setup and its punchline.",
        "global",
    )
    assert accepted.acceptance.via == "accept" and accepted.acceptance.by == HUMAN
    assert accepted.proposal == proposed.proposal  # the original proposal is kept

    by_human = lessons.propose(hub.log, "Open on the strongest line.", "global", HUMAN)
    dropped = lessons.drop(hub.log, by_human.id, OPERATOR_2, reason="Too vague.")
    assert dropped.state == "dropped" and dropped.drop is not None
    assert (dropped.drop.reason, dropped.drop.by) == ("Too vague.", OPERATOR_2)
    assert dropped.text == "Open on the strongest line."  # a dropped proposal keeps its text

    added = lessons.add(
        hub.log, "Hold the wide shot on a laugh.", f"project:{hub.project_id}", HUMAN
    )
    assert added.state == "accepted" and added.proposal is None
    assert added.acceptance is not None and added.acceptance.via == "add"
    assert added.scope == f"project:{hub.project_id}"

    retired = lessons.retire(hub.log, accepted.id, HUMAN, reason="Covered by the house rules.")
    retired_added = lessons.retire(hub.log, added.id, HUMAN)
    assert retired.state == retired_added.state == "retired"
    assert (
        retired.retirement is not None
        and retired.retirement.reason == "Covered by the house rules."
    )
    assert retired.acceptance == accepted.acceptance  # who accepted it is still known
    assert retired_added.retirement is not None and retired_added.retirement.reason == ""

    events = _events(hub)
    assert [e.type for e in events] == [
        lessons.EVENT_PROPOSED,
        lessons.EVENT_ACCEPTED,
        lessons.EVENT_PROPOSED,
        lessons.EVENT_DROPPED,
        lessons.EVENT_ADDED,
        lessons.EVENT_RETIRED,
        lessons.EVENT_RETIRED,
    ]
    assert all(e.project_id is None for e in events)  # global events; the source is in the payload
    assert events[0].payload["source"] == {"project_id": source[0], "review_id": source[1]}
    found, skipped = lessons.scan_lessons(hub.log)
    assert skipped == ()
    assert [lesson.id for lesson in found] == [proposed.id, by_human.id, added.id]
    assert lessons.get_lesson(hub.log, proposed.id) == retired
    assert lessons.get_lesson(hub.log, by_human.id) == dropped
    assert lessons.get_lesson(hub.log, "les_0000000000000000") is None
    assert lessons.get_lesson(hub.log, "not an id") is None
    as_json = json.loads(json.dumps(retired.to_dict()))
    assert as_json["state"] == "retired" and as_json["text"] == accepted.text
    assert as_json["proposal"]["source"] == {"project_id": source[0], "review_id": source[1]}
    assert as_json["acceptance"]["by"] == {"kind": "human", "id": "operator-1"}
    assert as_json["retirement"]["reason"] == "Covered by the house rules."


@pytest.mark.parametrize("actor", [AGENT, NODE], ids=["agent", "node"])
def test_only_a_human_adds_accepts_drops_or_retires(hub: Hub, actor: Actor) -> None:
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    accepted = lessons.add(hub.log, "End on the laugh.", "global", HUMAN)
    before = len(_events(hub))
    with pytest.raises(HumanRequired):
        lessons.add(hub.log, "Cut every pause.", "global", actor)
    with pytest.raises(HumanRequired):
        lessons.accept(hub.log, proposal.id, "Keep whole thoughts.", "global", actor)
    with pytest.raises(HumanRequired):
        lessons.drop(hub.log, proposal.id, actor)
    with pytest.raises(HumanRequired):
        lessons.retire(hub.log, accepted.id, actor)
    assert len(_events(hub)) == before
    assert lessons.get_lesson(hub.log, proposal.id) == proposal


@pytest.mark.parametrize("actor", [NODE, RUNNER], ids=["node", "runner"])
def test_only_a_human_or_an_agent_proposes(hub: Hub, actor: Actor) -> None:
    _refused("actor_not_allowed", lessons.propose, hub.log, "Keep whole thoughts.", "global", actor)
    assert _events(hub) == []
    lessons.propose(hub.log, "Keep whole thoughts.", "global", OTHER)  # any agent may
    assert len(_events(hub)) == 1


# -- the fold re-checks every rule -----------------------------------------------------------


def _payload(event_type: str, lesson_id: str, **changes: Any) -> dict[str, Any]:
    """A well-formed payload for ``event_type``."""
    if event_type == lessons.EVENT_PROPOSED:
        data: dict[str, Any] = {
            "lesson_id": lesson_id,
            "text": "Keep whole thoughts.",
            "scope": "global",
            "why": "",
            "source": None,
        }
    elif event_type in (lessons.EVENT_ADDED, lessons.EVENT_ACCEPTED):
        data = {"lesson_id": lesson_id, "text": "Keep whole thoughts.", "scope": "global"}
    else:
        data = {"lesson_id": lesson_id, "reason": ""}
    data.update(changes)
    return data


def _skips(hub: Hub, event_id: str) -> str:
    """The fold's reason for skipping ``event_id`` (it must be skipped)."""
    _, skipped = lessons.scan_lessons(hub.log)
    reasons = {item.event_id: item.reason for item in skipped}
    assert event_id in reasons, skipped
    return reasons[event_id]


@pytest.mark.parametrize(
    "event_type", [lessons.EVENT_ADDED, lessons.EVENT_ACCEPTED, lessons.EVENT_DROPPED]
)
def test_the_fold_skips_a_human_only_event_appended_by_an_agent(hub: Hub, event_type: str) -> None:
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    lesson_id = proposal.id if event_type != lessons.EVENT_ADDED else lessons.new_lesson_id()
    forged = hub.log.append(event_type, AGENT, _payload(event_type, lesson_id))
    assert "not a human" in _skips(hub, forged.id)
    assert lessons.get_lesson(hub.log, proposal.id) == proposal
    guide = lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client="ccn")
    assert guide.empty


def test_the_fold_skips_a_retire_appended_by_an_agent(hub: Hub) -> None:
    accepted = lessons.add(hub.log, "End on the laugh.", "global", HUMAN)
    forged = hub.log.append(
        lessons.EVENT_RETIRED, AGENT, _payload(lessons.EVENT_RETIRED, accepted.id)
    )
    assert "not a human" in _skips(hub, forged.id)
    assert lessons.get_lesson(hub.log, accepted.id) == accepted
    guide = lessons.lessons_for_guide(hub.log, project_id=None, client=None)
    assert guide.global_ == ("End on the laugh.",)


@pytest.mark.parametrize("actor", [NODE, RUNNER], ids=["node", "runner"])
def test_the_fold_skips_a_proposal_by_a_node_or_the_runner(hub: Hub, actor: Actor) -> None:
    lesson_id = lessons.new_lesson_id()
    forged = hub.log.append(
        lessons.EVENT_PROPOSED, actor, _payload(lessons.EVENT_PROPOSED, lesson_id)
    )
    assert "only a human or an agent" in _skips(hub, forged.id)
    assert lessons.get_lesson(hub.log, lesson_id) is None


@pytest.mark.parametrize("event_type", lessons.LESSON_EVENT_TYPES)
def test_the_fold_skips_a_lesson_event_that_carries_a_project_id(hub: Hub, event_type: str) -> None:
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    accepted = lessons.add(hub.log, "End on the laugh.", "global", HUMAN)
    targets = {
        lessons.EVENT_PROPOSED: lessons.new_lesson_id(),
        lessons.EVENT_ADDED: lessons.new_lesson_id(),
        lessons.EVENT_ACCEPTED: proposal.id,
        lessons.EVENT_DROPPED: proposal.id,
        lessons.EVENT_RETIRED: accepted.id,
    }
    forged = hub.log.append(
        event_type, HUMAN, _payload(event_type, targets[event_type]), project_id=hub.project_id
    )
    assert "global" in _skips(hub, forged.id) and hub.project_id in _skips(hub, forged.id)
    found, _ = lessons.scan_lessons(hub.log)
    assert found == (proposal, accepted)


# Every (state, event) pair the state machine refuses. A lesson is set up in ``state`` by the
# writers; the event is then appended raw by a human with a well-formed payload.
_STATES = ("none", "proposed", "accepted", "added", "dropped", "retired")
_LEGAL = {
    ("none", lessons.EVENT_PROPOSED),
    ("none", lessons.EVENT_ADDED),
    ("proposed", lessons.EVENT_ACCEPTED),
    ("proposed", lessons.EVENT_DROPPED),
    ("accepted", lessons.EVENT_RETIRED),
    ("added", lessons.EVENT_RETIRED),
}
_ILLEGAL = [
    (state, event_type)
    for state in _STATES
    for event_type in lessons.LESSON_EVENT_TYPES
    if (state, event_type) not in _LEGAL
]


def _in_state(hub: Hub, state: str) -> lessons.Lesson | None:
    if state == "none":
        return None
    if state == "added":
        return lessons.add(hub.log, "End on the laugh.", "global", HUMAN)
    lesson = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    if state in ("accepted", "retired"):
        lesson = lessons.accept(hub.log, lesson.id, lesson.text, "global", HUMAN)
    if state == "dropped":
        lesson = lessons.drop(hub.log, lesson.id, HUMAN)
    if state == "retired":
        lesson = lessons.retire(hub.log, lesson.id, HUMAN)
    return lesson


@pytest.mark.parametrize(("state", "event_type"), _ILLEGAL)
def test_the_fold_skips_every_illegal_transition(hub: Hub, state: str, event_type: str) -> None:
    lesson = _in_state(hub, state)
    lesson_id = lesson.id if lesson is not None else lessons.new_lesson_id()
    actor = AGENT if event_type == lessons.EVENT_PROPOSED else HUMAN
    forged = hub.log.append(event_type, actor, _payload(event_type, lesson_id))
    reason = _skips(hub, forged.id)
    expected = {
        "none": "no lesson",
        "retired": "was retired",
    }.get(state)
    if event_type in (lessons.EVENT_PROPOSED, lessons.EVENT_ADDED):
        expected = "already exists"
    elif expected is None and event_type == lessons.EVENT_RETIRED:
        expected = "only an accepted lesson can be retired"
    elif expected is None:
        expected = "already decided"
    assert expected in reason, reason
    assert lessons.get_lesson(hub.log, lesson_id) == lesson  # unchanged (or still absent)


def test_the_legal_transitions_are_exactly_the_writers(hub: Hub) -> None:
    """The pairs left out of the illegal table do apply when appended raw by a human."""
    for state, event_type in sorted(_LEGAL):
        lesson = _in_state(hub, state)
        lesson_id = lesson.id if lesson is not None else lessons.new_lesson_id()
        event = hub.log.append(event_type, HUMAN, _payload(event_type, lesson_id))
        _, skipped = lessons.scan_lessons(hub.log)
        assert event.id not in {item.event_id for item in skipped}, (state, event_type)


@pytest.mark.parametrize(
    ("state", "done"), [(s, d) for s in _STATES for d in ("accept", "drop", "retire")]
)
def test_the_writers_refuse_the_same_transitions(hub: Hub, state: str, done: str) -> None:
    lesson = _in_state(hub, state)
    lesson_id = lesson.id if lesson is not None else lessons.new_lesson_id()
    legal = (state == "proposed" and done in ("accept", "drop")) or (
        state in ("accepted", "added") and done == "retire"
    )
    before = len(_events(hub))
    call = {
        "accept": lambda: lessons.accept(
            hub.log, lesson_id, "Keep whole thoughts.", "global", HUMAN
        ),
        "drop": lambda: lessons.drop(hub.log, lesson_id, HUMAN),
        "retire": lambda: lessons.retire(hub.log, lesson_id, HUMAN),
    }[done]
    if legal:
        call()
        assert len(_events(hub)) == before + 1
        return
    error = _refused("unknown_lesson" if state == "none" else "illegal_transition", call)
    assert error.details["lesson_id"] == lesson_id
    assert len(_events(hub)) == before


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Keep\nwhole thoughts."},
        {"text": " Keep whole thoughts."},
        {"text": "Keep  whole thoughts."},
        {"text": ""},
        {"text": "x" * 501},
        {"text": 5},
        {"text": "Keep \x07whole thoughts."},
        {"scope": "client:../ccn"},
        {"scope": "everywhere"},
        {"scope": None},
        {"why": None},
        {"why": "y" * 1001},
        {"source": {"project_id": "prj_acme"}},
        {"source": "prj_acme/nd_1"},
        {"lesson_id": "les_XYZ"},
        {"lesson_id": None},
    ],
    ids=lambda payload: next(iter(payload)) + "=" + repr(next(iter(payload.values())))[:12],
)
def test_the_fold_skips_a_payload_that_fails_validation(hub: Hub, payload: dict[str, Any]) -> None:
    data = {**_payload(lessons.EVENT_PROPOSED, lessons.new_lesson_id()), **payload}
    forged = hub.log.append(lessons.EVENT_PROPOSED, AGENT, data)
    _skips(hub, forged.id)
    assert lessons.scan_lessons(hub.log)[0] == ()


def test_the_fold_skips_a_bad_drop_reason_and_a_non_object_payload(hub: Hub) -> None:
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    long_reason = hub.log.append(
        lessons.EVENT_DROPPED, HUMAN, _payload(lessons.EVENT_DROPPED, proposal.id, reason="r" * 501)
    )
    no_reason = hub.log.append(lessons.EVENT_DROPPED, HUMAN, {"lesson_id": proposal.id})
    assert "reason" in _skips(hub, long_reason.id) and "reason" in _skips(hub, no_reason.id)
    raw = hub.log._conn  # a payload that is JSON but not an object, as a foreign writer could leave
    raw.execute(
        "INSERT INTO events (id, project_id, type, actor_kind, actor_id, ts, payload) "
        "VALUES ('01FORGEDLISTPAYLOAD0000000', NULL, ?, 'human', 'operator-1', "
        "'2026-11-13T00:00:00+00:00', '[1, 2]')",
        (lessons.EVENT_ACCEPTED,),
    )
    assert "not an object" in _skips(hub, "01FORGEDLISTPAYLOAD0000000")
    assert lessons.get_lesson(hub.log, proposal.id) == proposal


def test_two_events_for_one_new_id_keep_the_first(hub: Hub) -> None:
    lesson_id = lessons.new_lesson_id()
    first = hub.log.append(lessons.EVENT_ADDED, HUMAN, _payload(lessons.EVENT_ADDED, lesson_id))
    second = hub.log.append(
        lessons.EVENT_ADDED, HUMAN, _payload(lessons.EVENT_ADDED, lesson_id, text="Another rule.")
    )
    assert "already exists" in _skips(hub, second.id)
    lesson = lessons.get_lesson(hub.log, lesson_id)
    assert lesson is not None and lesson.acceptance is not None
    assert (lesson.text, lesson.acceptance.event_id) == ("Keep whole thoughts.", first.id)


# -- scope -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scope", ["global", "client:ccn", "project:prj_acme", "client:a_b-1", "project:x"]
)
def test_parse_scope_round_trips(scope: str) -> None:
    assert str(lessons.parse_scope(scope)) == scope


@pytest.mark.parametrize(
    "scope",
    [
        "",
        "Global",
        "global ",
        "all",
        "client:",
        "client:CCN",
        "client: ccn",
        "client:ccn.toml",
        "client:../profiles/ccn",
        "client:/etc/aar/evil.toml",
        "client:C:\\profiles\\ccn.toml",
        "client:" + "c" * 65,
        "project:",
        "project:" + "p" * 129,
        "prj_acme",
        "node:node-1",
        None,
        5,
        ["global"],
    ],
)
def test_parse_scope_refuses_bad_syntax(scope: Any) -> None:
    error = _refused("invalid_input", lessons.parse_scope, scope)
    assert error.details["field"] == "scope"


def test_writers_check_the_scope_target(hub: Hub, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    real = lessons._load_named_profile  # the hub's built-in-name-only loader

    def spy(name: Any, field_name: str) -> Any:
        loaded.append(name)
        return real(name, field_name)

    monkeypatch.setattr(lessons, "_load_named_profile", spy)
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    before = len(_events(hub))
    for call in (
        lambda scope: lessons.add(hub.log, "Keep whole thoughts.", scope, HUMAN),
        lambda scope: lessons.propose(hub.log, "Keep whole thoughts.", scope, AGENT),
        lambda scope: lessons.accept(hub.log, proposal.id, "Keep whole thoughts.", scope, HUMAN),
    ):
        _refused("unknown_profile", call, "client:nope")
        _refused("unknown_project", call, "project:prj_missing")
        for path_like in ("client:../../etc/passwd", "client:/srv/profiles/ccn.toml"):
            _refused("invalid_input", call, path_like)
    assert loaded == ["nope", "nope", "nope"]  # a path-like client never reaches the loader
    assert len(_events(hub)) == before
    lessons.add(hub.log, "Keep whole thoughts.", "client:ccn", HUMAN)
    lessons.add(hub.log, "Keep whole thoughts.", f"project:{hub.other_project_id}", HUMAN)
    assert loaded[-1] == "ccn"


def test_propose_checks_its_source_review(hub: Hub) -> None:
    project_id, review_id = _review(hub)
    _refused(
        "unknown_review",
        lessons.propose,
        hub.log,
        "Keep whole thoughts.",
        "global",
        AGENT,
        source=(project_id, "nd_0000000000000000"),
    )
    _refused(
        "unknown_review",  # the review exists, but in another project
        lessons.propose,
        hub.log,
        "Keep whole thoughts.",
        "global",
        AGENT,
        source=(hub.other_project_id, review_id),
    )
    _refused(
        "unknown_project",
        lessons.propose,
        hub.log,
        "Keep whole thoughts.",
        "global",
        AGENT,
        source=("prj_missing", review_id),
    )
    for bad in ((project_id,), (project_id, review_id, "x"), (project_id, ""), project_id, {}):
        _refused(
            "invalid_input",
            lessons.propose,
            hub.log,
            "Keep whole thoughts.",
            "global",
            AGENT,
            source=bad,
        )
    assert _events(hub) == []
    lesson = lessons.propose(
        hub.log, "Keep whole thoughts.", "global", AGENT, source=[project_id, review_id]
    )
    assert lesson.proposal is not None and lesson.proposal.source == lessons.LessonSource(
        project_id, review_id
    )


# -- text and bounds ---------------------------------------------------------------------------


def test_text_is_normalized_to_one_line(hub: Hub) -> None:
    lesson = lessons.propose(
        hub.log,
        "  Keep\n\twhole \r\n  thoughts:\u00a0a setup\u2028with its punchline.  ",
        "global",
        AGENT,
        why="\n The owner\n\nsaid so. ",
    )
    assert lesson.text == "Keep whole thoughts: a setup with its punchline."
    assert lesson.proposal is not None and lesson.proposal.why == "The owner said so."
    accepted = lessons.accept(hub.log, lesson.id, "Keep\nwhole thoughts.\n", "global", HUMAN)
    assert accepted.text == "Keep whole thoughts."
    dropped = lessons.drop(
        hub.log, lessons.propose(hub.log, "x", "global", AGENT).id, HUMAN, reason=" Too\nvague. "
    )
    assert dropped.drop is not None and dropped.drop.reason == "Too vague."
    stored = _events(hub)[0].payload
    assert stored["text"] == lesson.text and stored["why"] == "The owner said so."


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "\n\t\r\n",
        "Keep\x00",
        "Keep\x07 it",
        "\x1b[31mred",
        "del\x7f",
        "c1\x9b",
        "a\u202eb",
        "a\u2066b",
        5,
        None,
    ],
)
def test_text_refuses_empty_values_and_control_characters(hub: Hub, bad: Any) -> None:
    error = _refused("invalid_input", lessons.add, hub.log, bad, "global", HUMAN)
    assert error.details["field"] == "text"
    _refused("invalid_input", lessons.propose, hub.log, bad, "global", AGENT)
    assert _events(hub) == []


def test_every_bound(hub: Hub) -> None:
    exact = "a" * lessons.MAX_TEXT
    padded = "  " + " \n ".join(["a" * 100] * 4) + " " + "b" * 96 + "\n\n"  # 500 after normalizing
    assert len(padded) > lessons.MAX_TEXT
    assert lessons.add(hub.log, exact, "global", HUMAN).text == exact
    assert len(lessons.add(hub.log, padded, "global", HUMAN).text) == lessons.MAX_TEXT
    _refused("invalid_input", lessons.add, hub.log, exact + "a", "global", HUMAN)
    _refused("invalid_input", lessons.add, hub.log, "x" * (lessons.MAX_RAW + 1), "global", HUMAN)

    why = "w" * lessons.MAX_WHY
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT, why=why)
    assert proposal.proposal is not None and proposal.proposal.why == why
    error = _refused("invalid_input", lessons.propose, hub.log, "x", "global", AGENT, why=why + "w")
    assert error.details["field"] == "why"
    _refused("invalid_input", lessons.propose, hub.log, "x", "global", AGENT, why="a\x00b")
    blank_why = lessons.propose(hub.log, "x", "global", AGENT, why="   ")
    assert blank_why.proposal is not None and blank_why.proposal.why == ""

    reason = "r" * lessons.MAX_REASON
    for writer, lesson_id in (
        (lessons.drop, proposal.id),
        (lessons.retire, lessons.add(hub.log, "y", "client:ccn", HUMAN).id),
    ):
        error = _refused("invalid_input", writer, hub.log, lesson_id, HUMAN, reason=reason + "r")
        assert error.details["field"] == "reason"
        _refused("invalid_input", writer, hub.log, lesson_id, HUMAN, reason="a\x1bb")
        assert writer(hub.log, lesson_id, HUMAN, reason=reason).state in ("dropped", "retired")


def test_at_most_max_pending_proposals_wait(hub: Hub) -> None:
    first = None
    for n in range(lessons.MAX_PENDING):
        lesson = lessons.propose(hub.log, f"Rule {n}.", "global", AGENT)
        first = first or lesson
    assert first is not None
    error = _refused("too_many_proposals", lessons.propose, hub.log, "One more.", "global", AGENT)
    assert error.details["limit"] == lessons.MAX_PENDING
    _refused("too_many_proposals", lessons.propose, hub.log, "One more.", "global", HUMAN)
    lessons.drop(hub.log, first.id, HUMAN)  # deciding one frees a place
    lessons.propose(hub.log, "One more.", "global", AGENT)
    _refused("too_many_proposals", lessons.propose, hub.log, "And another.", "global", AGENT)
    found, _ = lessons.scan_lessons(hub.log)
    assert sum(1 for lesson in found if lesson.state == "proposed") == lessons.MAX_PENDING


def test_a_scope_holds_at_most_max_accepted_lessons(hub: Hub) -> None:
    full = f"project:{hub.project_id}"
    held = [
        lessons.add(hub.log, f"Rule {n}.", full, HUMAN)
        for n in range(lessons.MAX_ACCEPTED_PER_SCOPE)
    ]
    error = _refused("too_many_lessons", lessons.add, hub.log, "One more.", full, HUMAN)
    assert "Retire" in error.fix_hint and error.details["scope"] == full
    proposal = lessons.propose(hub.log, "One more.", "global", AGENT)
    _refused("too_many_lessons", lessons.accept, hub.log, proposal.id, "One more.", full, HUMAN)
    assert lessons.get_lesson(hub.log, proposal.id) == proposal
    lessons.add(hub.log, "Other scopes are not full.", "client:ccn", HUMAN)
    lessons.add(hub.log, "Other scopes are not full.", f"project:{hub.other_project_id}", HUMAN)
    lessons.retire(hub.log, held[0].id, HUMAN)
    assert lessons.accept(hub.log, proposal.id, "One more.", full, HUMAN).state == "accepted"
    _refused("too_many_lessons", lessons.add, hub.log, "Yet another.", full, HUMAN)


def test_the_guide_reads_at_most_max_accepted_per_scope_the_first_accepted(hub: Hub) -> None:
    """Raw events can pass the writers' cap; the guide still prints the first 50 only."""
    over = lessons.MAX_ACCEPTED_PER_SCOPE + 10
    for n in range(over):
        payload = _payload(lessons.EVENT_ADDED, lessons.new_lesson_id(), text=f"Rule {n}.")
        hub.log.append(lessons.EVENT_ADDED, HUMAN, payload)
    guide = lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client="ccn")
    assert guide.global_ == tuple(f"Rule {n}." for n in range(lessons.MAX_ACCEPTED_PER_SCOPE))
    assert len(lessons.scan_lessons(hub.log)[0]) == over


# -- idempotency --------------------------------------------------------------------------------


def test_a_retry_with_the_same_key_writes_once(hub: Hub) -> None:
    source = _review(hub)
    first = lessons.propose(
        hub.log,
        "Keep whole thoughts.",
        "global",
        AGENT,
        why="w",
        source=source,
        idempotency_key="p-1",
    )
    again = lessons.propose(
        hub.log,
        " Keep whole\nthoughts. ",
        "global",
        AGENT,
        why="w",
        source=source,
        idempotency_key="p-1",
    )
    assert again == first
    added = lessons.add(hub.log, "End on the laugh.", "global", HUMAN, idempotency_key="a-1")
    assert (
        lessons.add(hub.log, "End on the laugh.", "global", HUMAN, idempotency_key="a-1") == added
    )
    accepted = lessons.accept(
        hub.log, first.id, "Keep it.", "client:ccn", HUMAN, idempotency_key="c-1"
    )
    retired = lessons.retire(hub.log, accepted.id, HUMAN, reason="old", idempotency_key="r-1")
    # A retry returns the lesson as it is now, even after it moved on.
    assert (
        lessons.accept(hub.log, first.id, "Keep it.", "client:ccn", HUMAN, idempotency_key="c-1")
        == retired
    )
    assert (
        lessons.retire(hub.log, accepted.id, HUMAN, reason="old", idempotency_key="r-1") == retired
    )
    other = lessons.propose(hub.log, "Open strong.", "global", AGENT)
    dropped = lessons.drop(hub.log, other.id, HUMAN, idempotency_key="d-1")
    assert lessons.drop(hub.log, other.id, HUMAN, idempotency_key="d-1") == dropped
    assert [e.type for e in _events(hub)] == [
        lessons.EVENT_PROPOSED,
        lessons.EVENT_ADDED,
        lessons.EVENT_ACCEPTED,
        lessons.EVENT_RETIRED,
        lessons.EVENT_PROPOSED,
        lessons.EVENT_DROPPED,
    ]


def test_a_key_reused_for_another_request_is_refused(hub: Hub) -> None:
    proposal = lessons.propose(
        hub.log, "Keep whole thoughts.", "global", AGENT, idempotency_key="k-1"
    )
    accepted = lessons.add(hub.log, "End on the laugh.", "global", HUMAN, idempotency_key="k-2")
    before = len(_events(hub))
    conflicts = [
        lambda: lessons.propose(hub.log, "Other text.", "global", AGENT, idempotency_key="k-1"),
        lambda: lessons.propose(
            hub.log, "Keep whole thoughts.", "client:ccn", AGENT, idempotency_key="k-1"
        ),
        lambda: lessons.propose(
            hub.log, "Keep whole thoughts.", "global", AGENT, why="new", idempotency_key="k-1"
        ),
        lambda: lessons.propose(
            hub.log, "Keep whole thoughts.", "global", OTHER, idempotency_key="k-1"
        ),
        lambda: lessons.add(
            hub.log, "Keep whole thoughts.", "global", HUMAN, idempotency_key="k-1"
        ),
        lambda: lessons.add(
            hub.log, "End on the laugh.", "global", OPERATOR_2, idempotency_key="k-2"
        ),
        lambda: lessons.accept(
            hub.log, proposal.id, "Keep whole thoughts.", "global", HUMAN, idempotency_key="k-2"
        ),
        lambda: lessons.drop(hub.log, proposal.id, HUMAN, idempotency_key="k-1"),
        lambda: lessons.retire(hub.log, accepted.id, HUMAN, idempotency_key="k-2"),
    ]
    for call in conflicts:
        _refused("idempotency_conflict", call)
    lessons.accept(hub.log, proposal.id, "Keep it.", "global", HUMAN, idempotency_key="k-3")
    _refused(
        "idempotency_conflict",
        lessons.accept,
        hub.log,
        proposal.id,
        "Keep it, edited.",
        "global",
        HUMAN,
        idempotency_key="k-3",
    )
    assert len(_events(hub)) == before + 1


def test_a_key_that_names_a_skipped_event_is_refused(hub: Hub) -> None:
    payload = _payload(lessons.EVENT_PROPOSED, lessons.new_lesson_id())
    hub.log.append(
        lessons.EVENT_PROPOSED, AGENT, payload, project_id=hub.project_id, idempotency_key="k-x"
    )
    error = _refused(
        "idempotency_conflict",
        lessons.propose,
        hub.log,
        "Keep whole thoughts.",
        "global",
        AGENT,
        idempotency_key="k-x",
    )
    assert "skipped" in error.message
    # The lesson exists but the event under the key was skipped: a retry must not report an
    # accept that never applied.
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    accept = {"lesson_id": proposal.id, "text": "Keep whole thoughts.", "scope": "global"}
    hub.log.append(
        lessons.EVENT_ACCEPTED, HUMAN, accept, project_id=hub.project_id, idempotency_key="k-y"
    )
    error = _refused(
        "idempotency_conflict",
        lessons.accept,
        hub.log,
        proposal.id,
        "Keep whole thoughts.",
        "global",
        HUMAN,
        idempotency_key="k-y",
    )
    assert "skipped" in error.message
    assert lessons.get_lesson(hub.log, proposal.id) == proposal


def test_bad_idempotency_keys_and_lesson_ids_are_refused(hub: Hub) -> None:
    for key in ("", "k" * 129, 5):
        _refused("invalid_input", lessons.add, hub.log, "x", "global", HUMAN, idempotency_key=key)
    for lesson_id in ("", "les_", "les_ABCDEF0123456789", "les_0123", "nd_0123456789abcdef", None):
        error = _refused("invalid_input", lessons.retire, hub.log, lesson_id, HUMAN)
        assert error.details["field"] == "lesson_id"
    assert _events(hub) == []


# -- the guide feed ---------------------------------------------------------------------------


def test_lessons_for_guide_keeps_the_right_scopes_in_acceptance_order(hub: Hub) -> None:
    acme, other = f"project:{hub.project_id}", f"project:{hub.other_project_id}"
    late = lessons.propose(hub.log, "Accepted second.", "global", AGENT)
    early = lessons.propose(hub.log, "Accepted first.", "client:ccn", AGENT)
    lessons.accept(hub.log, early.id, "Global, accepted first.", "global", HUMAN)
    lessons.add(hub.log, "Client rule.", "client:ccn", HUMAN)
    lessons.add(hub.log, "Acme rule.", acme, HUMAN)
    lessons.add(hub.log, "Other project's rule.", other, HUMAN)
    lessons.accept(hub.log, late.id, "Global, accepted second.", "global", HUMAN)
    hub.log.append(  # another client's lesson (a profile the hub may once have had)
        lessons.EVENT_ADDED,
        HUMAN,
        _payload(
            lessons.EVENT_ADDED,
            lessons.new_lesson_id(),
            text="Other client's rule.",
            scope="client:other",
        ),
    )
    lessons.propose(hub.log, "Still a proposal.", "global", AGENT)
    dropped = lessons.propose(hub.log, "Dropped.", acme, AGENT)
    lessons.drop(hub.log, dropped.id, HUMAN)
    retired = lessons.add(hub.log, "Retired.", "client:ccn", HUMAN)
    lessons.retire(hub.log, retired.id, HUMAN)
    lessons.add(hub.log, "Acme rule two.", acme, HUMAN)

    guide = lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client="ccn")
    assert guide.global_ == ("Global, accepted first.", "Global, accepted second.")
    assert guide.client == ("Client rule.",)
    assert guide.project == ("Acme rule.", "Acme rule two.")
    assert guide.client_name == "ccn" and not guide.empty

    no_brief = lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client=None)
    assert no_brief.client == () and no_brief.global_ == guide.global_
    assert no_brief.project == guide.project
    elsewhere = lessons.lessons_for_guide(hub.log, project_id=hub.other_project_id, client="other")
    assert elsewhere.project == ("Other project's rule.",)
    assert elsewhere.client == ("Other client's rule.",)
    assert lessons.lessons_for_guide(hub.log, project_id=None, client=None).project == ()
