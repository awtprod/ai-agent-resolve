"""The console's lessons page and its writes (S23a), driven through Starlette's TestClient."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import pytest
from test_hub_console import AGENT, HUMAN, ORIGIN, SECRET, Hub, _form

from aar.hub import console, lessons, project_register
from aar.hub.console import ConsoleConfig, CsrfSigner
from aar.hub.events import Actor
from aar.hub.registry import Registry

JSON = {**ORIGIN, "Accept": "application/json"}


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return Hub(tmp_path)


def _count(hub: Hub) -> int:
    return len(list(hub.log.iter_events(types=lessons.LESSON_EVENT_TYPES)))


def _textarea(page: str, action: str) -> str:
    start = page.index(f'action="{action}"')
    form = page[start : page.index("</form>", start)]
    match = re.search(r'<textarea[^>]*name="text"[^>]*>([^<]*)</textarea>', form)
    assert match is not None, form
    return html.unescape(match.group(1))


def _selected(page: str, action: str) -> list[str]:
    start = page.index(f'action="{action}"')
    form = page[start : page.index("</form>", start)]
    return re.findall(r'<option value="([^"]*)" selected>', form)


def test_the_page_renders_each_section(hub: Hub) -> None:
    project = f"project:{hub.project}"
    proposal = lessons.propose(
        hub.log,
        "Keep the setup with its punchline.",
        project,
        AGENT,
        why="Round 1 cut the punchline.",
        source=(hub.project, hub.plain_review),
    )
    unsourced = lessons.propose(hub.log, "Open on the strongest line.", "client:ccn", HUMAN)
    lessons.add(hub.log, "Prefer few, long ranges.", "client:ccn", HUMAN)
    lessons.add(hub.log, "Hold the wide shot on a laugh.", project, HUMAN)
    lessons.add(hub.log, "End on the laugh.", "global", HUMAN)
    response = hub.client().get("/lessons")
    assert response.status_code == 200
    page = response.text
    assert '<a href="/lessons" class="" aria-current="page">Lessons (2)</a>' in page

    # 1. Proposed: why, source review, who and when, and the decide form.
    assert "Proposed (2)" in page
    assert "Round 1 cut the punchline." in page and "No reason was given." in page
    assert f'href="/reviews/{hub.project}/{hub.plain_review}"' in page
    assert "Not drawn from a review." in page
    demo = f"Project: demo-show ({hub.project})"  # a project always shows its id
    assert "agent-1" in page and f"suggested for {demo}" in page
    decide = f"/lessons/{proposal.id}/decide"
    assert _textarea(page, decide) == "Keep the setup with its punchline."
    assert _selected(page, decide) == [project]
    assert _selected(page, f"/lessons/{unsourced.id}/decide") == ["client:ccn"]
    start = page.index(f'action="{decide}"')
    form = page[start : page.index("</form>", start)]
    assert ">Applies to</label>" in form and 'name="reason"' in form
    assert 'name="decision" value="accept"' in form and 'name="decision" value="drop"' in form
    for label in (
        "Global (every show)",
        "Client: ccn (every show whose brief uses the ccn rule profile)",
        demo,
    ):
        assert f">{label}</option>" in form, label

    # 2. Accepted, grouped: global, then each client, then each project; each with a retire form.
    accepted = page[page.index("Accepted (3)") : page.index("Add a lesson")]
    headings = re.findall(r"<h3>([^<]*)</h3>", accepted)
    assert headings == [
        "Global (every show)",
        "Client: ccn (every show whose brief uses the ccn rule profile)",
        demo,
    ]
    assert accepted.count('class="retire"') == 3
    assert accepted.index("End on the laugh.") < accepted.index("Prefer few, long ranges.")

    # 3. Add a lesson, global preselected.
    assert _selected(page, "/lessons/add") == ["global"]
    assert 'name="text"' in page[page.index('action="/lessons/add"') :]


def test_accept_with_edited_text_and_a_new_scope(hub: Hub) -> None:
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "client:ccn", AGENT, why="w")
    client = hub.client()
    action = f"/lessons/{proposal.id}/decide"
    form = _form(client.get("/lessons").text, action)
    data = {
        **form,
        "decision": "accept",
        "text": "Keep a setup\nwith its punchline.",
        "scope": f"project:{hub.project}",
        "reason": "",
    }
    first = client.post(action, data=data, headers=ORIGIN)
    second = client.post(action, data=data, headers=ORIGIN)  # a double submit acts once
    assert first.status_code == second.status_code == 303
    assert first.headers["location"] == "/lessons?done=lesson_accepted"
    lesson = lessons.get_lesson(hub.log, proposal.id)
    assert lesson is not None and lesson.state == "accepted" and lesson.acceptance is not None
    assert (lesson.text, lesson.scope) == ("Keep a setup with its punchline.", data["scope"])
    assert lesson.acceptance.by == HUMAN
    assert lesson.proposal is not None and lesson.proposal.text == "Keep whole thoughts."
    assert lesson.proposal.scope == "client:ccn"
    assert _count(hub) == 2
    page = client.get("/lessons?done=lesson_accepted").text
    assert "Lesson accepted." in page and "edited from the proposal" in page
    assert "No proposals are waiting." in page
    # The same form again, fresh key: the proposal was decided, so it is a conflict.
    again = client.post(action, data={**data, "idem": "b" * 32}, headers=JSON)
    assert again.status_code == 409 and again.json()["error"]["code"] == "illegal_transition"


def test_drop_ignores_the_text_and_keeps_the_reason(hub: Hub) -> None:
    proposal = lessons.propose(hub.log, "Cut every pause.", "global", AGENT)
    client = hub.client()
    action = f"/lessons/{proposal.id}/decide"
    form = _form(client.get("/lessons").text, action)
    data = {**form, "decision": "drop", "text": "", "scope": "", "reason": " Too\nharsh. "}
    response = client.post(action, data=data, headers=ORIGIN)
    assert response.status_code == 303
    assert response.headers["location"] == "/lessons?done=lesson_dropped"
    lesson = lessons.get_lesson(hub.log, proposal.id)
    assert lesson is not None and lesson.state == "dropped" and lesson.drop is not None
    assert (lesson.drop.reason, lesson.drop.by) == ("Too harsh.", HUMAN)
    assert "Proposal dropped." in client.get("/lessons?done=lesson_dropped").text


def test_retire_and_add(hub: Hub) -> None:
    lesson = lessons.add(hub.log, "End on the laugh.", "global", HUMAN)
    client = hub.client()
    retire = f"/lessons/{lesson.id}/retire"
    page = client.get("/lessons").text
    response = client.post(
        retire, data={**_form(page, retire), "reason": "Covered elsewhere."}, headers=ORIGIN
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/lessons?done=lesson_retired"
    retired = lessons.get_lesson(hub.log, lesson.id)
    assert retired is not None and retired.state == "retired" and retired.retirement is not None
    assert (retired.retirement.reason, retired.retirement.by) == ("Covered elsewhere.", HUMAN)
    page = client.get("/lessons?done=lesson_retired").text
    assert "Lesson retired." in page and "End on the laugh." not in page

    data = {
        **_form(page, "/lessons/add"),
        "text": "Cut to the wide shot on a big laugh.",
        "scope": f"project:{hub.project}",
    }
    response = client.post("/lessons/add", data=data, headers=ORIGIN)
    assert response.headers["location"] == "/lessons?done=lesson_added"
    assert client.post("/lessons/add", data=data, headers=ORIGIN).status_code == 303
    found, _ = lessons.scan_lessons(hub.log)
    added = [item for item in found if item.acceptance and item.acceptance.via == "add"]
    assert [(item.text, item.scope, item.state) for item in added] == [
        ("End on the laugh.", "global", "retired"),
        ("Cut to the wide shot on a big laugh.", f"project:{hub.project}", "accepted"),
    ]
    assert added[-1].acceptance is not None and added[-1].acceptance.by == HUMAN
    assert "Lesson added." in client.get("/lessons?done=lesson_added").text


def _write(hub: Hub, route: str) -> tuple[str, dict[str, str]]:
    """A valid write to ``route``: its path and its form fields (without csrf and idem)."""
    if route == "decide":
        proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
        fields = {"decision": "accept", "text": "Keep whole thoughts.", "scope": "global"}
        return f"/lessons/{proposal.id}/decide", fields
    if route == "retire":
        return f"/lessons/{lessons.add(hub.log, 'Old rule.', 'global', HUMAN).id}/retire", {}
    return "/lessons/add", {"text": "A new rule.", "scope": "global"}


@pytest.mark.parametrize("route", ["decide", "retire", "add"])
def test_writes_need_an_identity_a_csrf_token_and_the_same_origin(hub: Hub, route: str) -> None:
    path, fields = _write(hub, route)
    client = hub.client()
    form = _form(client.get("/lessons").text, path)
    before = _count(hub)

    read_only = hub.client(ConsoleConfig(csrf_secret=SECRET))
    forged = CsrfSigner(SECRET, max_age=3600).issue("")
    data = {**fields, "csrf": forged, "idem": "a" * 32}
    response = read_only.post(path, data=data, headers=ORIGIN)
    assert response.status_code == 403 and "no_identity" in response.text

    other = CsrfSigner(SECRET, max_age=3600).issue("human:operator-2")
    for csrf in (None, "", other, form["csrf"] + "0"):
        data = (
            {**fields, "idem": form["idem"]} if csrf is None else {**fields, **form, "csrf": csrf}
        )
        response = client.post(path, data=data, headers=ORIGIN)
        assert response.status_code == 403 and "csrf_failed" in response.text, csrf

    for headers in ({}, {"Origin": "http://evil.example"}, {"Referer": "http://evil.example/x"}):
        response = client.post(path, data={**fields, **form}, headers=headers)
        assert response.status_code == 403, headers
    assert _count(hub) == before
    # The real form from the page works, so each refusal above was its check's doing.
    assert client.post(path, data={**fields, **form}, headers=ORIGIN).status_code == 303
    assert _count(hub) == before + 1


def test_the_form_is_never_trusted(hub: Hub) -> None:
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    client = hub.client()
    action = f"/lessons/{proposal.id}/decide"
    form = _form(client.get("/lessons").text, action)
    good = {**form, "decision": "accept", "text": "Keep whole thoughts.", "scope": "global"}
    cases: list[tuple[str, dict[str, Any], int, str]] = [
        (action, {"decision": "maybe"}, 422, "invalid_input"),
        (action, {"scope": "client:../../etc/passwd"}, 422, "invalid_input"),
        (action, {"scope": "client:/srv/profiles/ccn.toml"}, 422, "invalid_input"),
        (action, {"scope": "client:nope"}, 422, "unknown_profile"),
        (action, {"scope": "project:prj_missing"}, 422, "unknown_project"),
        (action, {"scope": "everywhere"}, 422, "invalid_input"),
        (action, {"text": "\x07bell"}, 422, "invalid_input"),
        (action, {"text": "   "}, 422, "invalid_input"),
        (action, {"text": "x" * (lessons.MAX_TEXT + 1)}, 422, "invalid_input"),
        ("/lessons/les_0000000000000000/decide", {}, 404, "unknown_lesson"),
        ("/lessons/not-a-lesson/decide", {}, 422, "invalid_input"),
        ("/lessons/les_0000000000000000/retire", {}, 404, "unknown_lesson"),
        (f"/lessons/{proposal.id}/retire", {}, 409, "illegal_transition"),
        ("/lessons/add", {"scope": "client:nope"}, 422, "unknown_profile"),
    ]
    for path, changes, status, code in cases:
        response = client.post(path, data={**good, **changes}, headers=JSON)
        assert response.status_code == status, (path, changes, response.text)
        assert response.json()["error"]["code"] == code, (path, changes)
    missing = {key: value for key, value in good.items() if key != "scope"}
    assert client.post("/lessons/add", data=missing, headers=JSON).status_code == 422
    assert _count(hub) == 1
    assert lessons.get_lesson(hub.log, proposal.id) == proposal


def test_without_an_identity_the_page_is_read_only(hub: Hub) -> None:
    lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT, why="The owner said so.")
    lessons.add(hub.log, "End on the laugh.", "global", HUMAN)
    page = hub.client(ConsoleConfig(csrf_secret=SECRET)).get("/lessons").text
    assert "Read-only" in page and "<form" not in page and 'name="csrf"' not in page
    assert "Keep whole thoughts." in page and "The owner said so." in page
    assert "End on the laugh." in page and "Add a lesson" not in page


def test_hostile_text_and_why_are_escaped(hub: Hub) -> None:
    text, why = '<script>alert("text")</script>', '<img src=x onerror=alert("why")>'
    lessons.propose(hub.log, text, "global", AGENT, why=why)
    lessons.add(hub.log, '<b onclick="x()">bold</b>', "global", HUMAN)
    page = hub.client().get("/lessons").text
    for hostile in (text, why, '<b onclick="x()">bold</b>'):
        assert hostile not in page
        assert html.escape(hostile, quote=True) in page
    read_only = hub.client(ConsoleConfig(csrf_secret=SECRET)).get("/lessons").text
    assert text not in read_only and html.escape(text, quote=True) in read_only


def test_the_nav_counts_the_proposals_waiting(hub: Hub) -> None:
    client = hub.client()
    assert '<a href="/lessons" class="">Lessons</a>' in client.get("/").text
    first = lessons.propose(hub.log, "Rule one.", "global", AGENT)
    second = lessons.propose(hub.log, "Rule two.", "global", AGENT)
    for path in ("/", "/approvals", "/control"):  # every page says so
        assert '<a href="/lessons" class="">Lessons (2)</a>' in client.get(path).text, path
    hub.log.append(  # a forged decision does not count
        lessons.EVENT_ACCEPTED,
        AGENT,
        {"lesson_id": first.id, "text": "Rule one.", "scope": "global"},
    )
    assert ">Lessons (2)</a>" in client.get("/").text
    lessons.accept(hub.log, first.id, "Rule one.", "global", HUMAN)
    assert ">Lessons (1)</a>" in client.get("/").text
    lessons.drop(hub.log, second.id, HUMAN)
    assert '<a href="/lessons" class="">Lessons</a>' in client.get("/").text


def test_the_page_shows_the_oldest_proposals_and_says_how_many_more(hub: Hub) -> None:
    total = console.MAX_PROPOSALS_SHOWN + 2
    made = [lessons.propose(hub.log, f"Rule {n}.", "global", AGENT) for n in range(total)]
    page = hub.client().get("/lessons").text
    assert page.count('name="decision" value="accept"') == console.MAX_PROPOSALS_SHOWN
    assert f"/lessons/{made[0].id}/decide" in page and f"/lessons/{made[-1].id}/decide" not in page
    assert "2 more proposals wait behind these" in page
    assert f"Proposed ({total})" in page and f">Lessons ({total})</a>" in page


def test_a_suggestion_that_is_no_longer_a_choice_stays_selected(hub: Hub) -> None:
    lesson_id = lessons.new_lesson_id()
    hub.log.append(  # a profile that is gone since (the writer checks it; the fold cannot)
        lessons.EVENT_PROPOSED,
        AGENT,
        {
            "lesson_id": lesson_id,
            "text": "A rule.",
            "scope": "client:gone",
            "why": "",
            "source": None,
        },
    )
    client = hub.client()
    action = f"/lessons/{lesson_id}/decide"
    page = client.get("/lessons").text
    assert _selected(page, action) == ["client:gone"]
    assert "client:gone (no longer available)</option>" in page
    data = {**_form(page, action), "decision": "accept", "text": "A rule.", "scope": "client:gone"}
    response = client.post(action, data=data, headers=JSON)
    assert response.status_code == 422 and response.json()["error"]["code"] == "unknown_profile"
    lesson = lessons.get_lesson(hub.log, lesson_id)
    assert lesson is not None and lesson.state == "proposed"


def test_skipped_lesson_events_are_listed_on_the_page(hub: Hub) -> None:
    proposal = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)
    forged = hub.log.append(
        lessons.EVENT_ACCEPTED,
        AGENT,
        {"lesson_id": proposal.id, "text": "Keep whole thoughts.", "scope": "global"},
    )
    page = hub.client().get("/lessons").text
    assert "1 malformed event ignored" in page and forged.id in page
    assert "not a human" in page


def test_the_nav_says_when_the_proposals_cannot_be_counted(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(log: Any) -> Any:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(lessons, "scan_lessons", unreadable)
    response = hub.client().get("/")
    assert response.status_code == 200  # the page still renders
    assert '<a href="/lessons" class="">Lessons (?)</a>' in response.text


def test_two_projects_with_one_name_are_told_apart(hub: Hub) -> None:
    twin = Registry(hub.log).create_project("demo-show", HUMAN).id
    lessons.add(hub.log, "A rule for the twin.", f"project:{twin}", HUMAN)
    page = hub.client().get("/lessons").text
    assert f">Project: demo-show ({hub.project})</option>" in page
    assert f"<h3>Project: demo-show ({twin})</h3>" in page


def test_an_accepted_group_shows_at_most_what_the_guide_prints(hub: Hub) -> None:
    cap = lessons.MAX_ACCEPTED_PER_SCOPE
    for n in range(cap + 3):  # raw events can pass the writers' cap
        payload = {"lesson_id": lessons.new_lesson_id(), "text": f"Rule {n}.", "scope": "global"}
        hub.log.append(lessons.EVENT_ADDED, HUMAN, payload)
    page = hub.client().get("/lessons").text
    assert page.count('class="retire"') == cap
    assert "Rule 0." in page and f"Rule {cap - 1}." in page and f"Rule {cap}." not in page
    assert f"3 more past the limit of {cap}; cut guides print only the first {cap}." in page


def test_a_forged_event_with_a_lone_surrogate_cannot_break_the_page(hub: Hub) -> None:
    """Review finding: a lone surrogate (JSON-escaped in the payload) made /lessons answer 500,
    which blocked deciding every proposal. The fold skips such events, so the page renders."""
    surrogate = chr(0xDCFF)
    waiting = lessons.propose(hub.log, "Keep whole thoughts.", "global", AGENT)

    def proposal(**changes: Any) -> dict[str, Any]:
        base = {"lesson_id": lessons.new_lesson_id(), "text": "A rule.", "scope": "global"}
        return {**base, "why": "", "source": None, **changes}

    forged = [
        proposal(text="Keep the punchline." + surrogate),
        proposal(why=surrogate),
        proposal(scope=f"project:{hub.project}{surrogate}"),
        proposal(source={"project_id": hub.project + surrogate, "review_id": hub.plain_review}),
    ]
    for payload in forged:
        hub.log.append(lessons.EVENT_PROPOSED, AGENT, payload)
    response = hub.client().get("/lessons")
    assert response.status_code == 200
    assert "4 malformed events ignored" in response.text
    assert f"/lessons/{waiting.id}/decide" in response.text  # the real proposal can be decided


def test_a_project_always_shows_its_id_so_look_alike_names_differ(hub: Hub) -> None:
    """Validator probe P5: the agent-facing registration accepts a name that differs from another
    only by an invisible character (here a Hangul filler), and the two options looked the same."""
    twin = project_register.register_project(
        hub.log, AGENT, name="demo-show\u3164", idempotency_key="look-alike"
    ).project_id
    hub.log.append(  # a project the registry does not know (the writer checks it; the fold cannot)
        lessons.EVENT_ADDED,
        HUMAN,
        {"lesson_id": lessons.new_lesson_id(), "text": "A rule.", "scope": "project:prj_gone"},
    )
    page = hub.client().get("/lessons").text
    start = page.index('action="/lessons/add"')
    form = page[start : page.index("</form>", start)]
    labels = {
        value: html.unescape(label)
        for value, label in re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)</option>', form)
    }
    assert labels[f"project:{hub.project}"] == f"Project: demo-show ({hub.project})"
    assert labels[f"project:{twin}"] == f"Project: demo-show\u3164 ({twin})"
    seen = [label.replace("\u3164", "") for label in labels.values()]
    assert len(seen) == len(set(seen)), seen
    assert "<h3>Project: prj_gone (not registered)</h3>" in page


def test_a_raw_event_cannot_grow_the_page_or_break_its_times(hub: Hub) -> None:
    """Validator probes P3/P8: a raw proposal signed with a 1 MiB actor id was shown in full (25 of
    them made a 26 MB page), and one whose time overflowed when converted to UTC answered 500."""
    huge = "a" * 1024 * 1024

    def proposal() -> dict[str, Any]:
        return {
            "lesson_id": lessons.new_lesson_id(),
            "text": "A rule.",
            "scope": "global",
            "why": "",
            "source": None,
        }

    for _ in range(3):
        hub.log.append(lessons.EVENT_PROPOSED, Actor("agent", huge), proposal())
    for ts in (huge, "9999-12-31T23:59:59-23:59", "0001-01-01T00:00:00+23:59"):
        hub.log.append(lessons.EVENT_PROPOSED, AGENT, proposal(), ts=ts)
    response = hub.client().get("/lessons")
    assert response.status_code == 200
    assert "6 malformed events ignored" in response.text
    assert "Proposed (0)" in response.text and huge[:200] not in response.text
    assert len(response.content) < 100_000


@pytest.mark.parametrize("ts", ["9999-12-31T23:59:59-23:59", "0001-01-01T00:00:00+23:59"])
def test_a_time_out_of_range_in_utc_is_shown_as_written(ts: str) -> None:
    """Every page formats times with ``_when``, which raised OverflowError (a 500) for these."""
    assert console._when(ts) == ts and console._when(ts, short=True) == ts
