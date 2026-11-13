"""Lessons in the ccn cut guide (S23a): where they print, which ones, none when none apply."""

from __future__ import annotations

from typing import Any

import pytest
from ccn_recipe_support import CcnEnv, make_run, outputs, read_guide, start, work
from write_support import AGENT, HUMAN, Hub

from aar.hub import lessons
from aar.hub.lessons import GuideLessons
from aar.recipes.ccn.guide import build_guide
from ccn_support import needs_ffmpeg

SECTION_3_END = (
    "- With one camera every cut between ranges is a jump cut: prefer few, long ranges.\n"
)
SECTION_4 = "## 4. Write keep.json\n"


def _guide_args() -> dict[str, Any]:
    """Everything ``build_guide`` needs for one run, without a hub."""
    return {
        "db": "/srv/hub/hub.db",
        "artifacts": "/srv/hub/artifacts",
        "project_id": "prj_acme",
        "recipe_node": "nd_recipe",
        "brief_id": "nd_brief",
        "brief": None,
        "stub": {
            "stub": "a" * 64,
            "pause_trim_flag": "--no-pause-trim",
            "cameras": 1,
            "recording_id": "show-1",
            "fps": "30000/1001",
            "sources": [{"camera": "A", "id": "cam-a"}],
        },
        "sync": {"synced": False, "report": None},
        "transcript": {"transcript": "b" * 64, "text": "c" * 64, "wordless": {"stretches": []}},
        "laughs": {"skipped": "not requested"},
        "aar": ["/srv/venv/bin/aar"],
        "python": "/srv/venv/bin/python",
    }


def _between_3_and_4(guide: str) -> str:
    start = guide.index(SECTION_3_END) + len(SECTION_3_END)
    return guide[start : guide.index(SECTION_4)]


def test_lessons_print_at_the_end_of_section_3_in_each_scope_and_order(hub: Hub) -> None:
    acme, other = f"project:{hub.project_id}", f"project:{hub.other_project_id}"
    second = lessons.propose(hub.log, "Global B.", "global", AGENT)
    lessons.add(hub.log, "Global A.", "global", HUMAN)
    lessons.add(hub.log, "Project rule.", acme, HUMAN)
    lessons.add(hub.log, "Another project's rule.", other, HUMAN)
    lessons.add(hub.log, "Client rule.", "client:ccn", HUMAN)
    lessons.accept(hub.log, second.id, "Global B, accepted later.", "global", HUMAN)
    hub.log.append(
        lessons.EVENT_ADDED,
        HUMAN,
        {
            "lesson_id": lessons.new_lesson_id(),
            "text": "Another client's rule.",
            "scope": "client:x",
        },
    )
    lessons.propose(hub.log, "Not accepted yet.", acme, AGENT)

    found = lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client="ccn")
    guide = build_guide(**_guide_args(), lessons=found)
    assert _between_3_and_4(guide) == (
        "\n"
        "### Lessons from past reviews\n"
        "\n"
        "A person accepted these after earlier reviews. Follow them like the house rules. Where "
        "two\n"
        "disagree, the narrower one wins: this project over the client, the client over global.\n"
        "\n"
        "Global:\n"
        "- Global A.\n"
        "- Global B, accepted later.\n"
        "\n"
        "Client `ccn`:\n"
        "- Client rule.\n"
        "\n"
        "This project:\n"
        "- Project rule.\n"
        "\n"
    )
    for absent in ("Another project's rule.", "Another client's rule.", "Not accepted yet."):
        assert absent not in guide
    assert guide.count("### Lessons from past reviews") == 1
    # Sections keep their numbers: later sections refer to each other by number.
    for heading in (
        "## 1. Fetch",
        "## 3. House rules",
        "## 4. Write",
        "## 6. Submit",
        "## 8. After",
    ):
        assert heading in guide


def test_an_empty_group_is_left_out(hub: Hub) -> None:
    lessons.add(hub.log, "Only this project.", f"project:{hub.project_id}", HUMAN)
    found = lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client="ccn")
    section = _between_3_and_4(build_guide(**_guide_args(), lessons=found))
    assert "This project:\n- Only this project.\n" in section
    assert "Global:" not in section and "Client `" not in section


def test_the_client_group_is_headed_with_the_runs_own_client() -> None:
    """Validator mutant G04: a heading hard-coded to ``ccn`` survived, since every test used ccn."""
    found = GuideLessons(client=("Open on the guest's first laugh.",), client_name="acme")
    section = _between_3_and_4(build_guide(**_guide_args(), lessons=found))
    assert "Client `acme`:\n- Open on the guest's first laugh.\n" in section
    assert "ccn" not in section


def test_without_lessons_the_guide_is_byte_identical(hub: Hub) -> None:
    plain = build_guide(**_guide_args())
    assert SECTION_3_END + "\n" + SECTION_4 in plain  # section 3 runs straight into section 4
    assert build_guide(**_guide_args(), lessons=None) == plain
    assert build_guide(**_guide_args(), lessons=GuideLessons()) == plain
    # Lessons that do not apply print nothing either: proposed, dropped, retired, elsewhere.
    lessons.propose(hub.log, "Proposed.", "global", AGENT)
    dropped = lessons.propose(hub.log, "Dropped.", "global", AGENT)
    lessons.drop(hub.log, dropped.id, HUMAN)
    retired = lessons.add(hub.log, "Retired.", "global", HUMAN)
    lessons.retire(hub.log, retired.id, HUMAN)
    lessons.add(hub.log, "Another project's.", f"project:{hub.other_project_id}", HUMAN)
    lessons.add(hub.log, "A client the run is not.", "client:ccn", HUMAN)
    found = lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client=None)
    assert found.empty
    assert build_guide(**_guide_args(), lessons=found) == plain


@needs_ffmpeg
def test_the_guide_step_prints_this_runs_lessons_once_per_run(ccn_env: CcnEnv) -> None:
    env = ccn_env
    project = f"project:{env.hub.project_id}"
    lessons.add(env.hub.log, "Global rule.", "global", HUMAN)
    lessons.add(env.hub.log, "Client rule.", "client:ccn", HUMAN)  # the brief's rule_profile
    lessons.add(env.hub.log, "Project rule.", project, HUMAN)
    lessons.add(env.hub.log, "Other project.", f"project:{env.hub.other_project_id}", HUMAN)
    lessons.propose(env.hub.log, "Only proposed.", project, AGENT)
    run = make_run(env, env.params(env.show("cam-a.mov")), key="lessons")
    start(env, run, "lessons")
    assert work(env) == 5  # stub, sync, transcript, laughs, guide
    guide = read_guide(env, run)
    section = _between_3_and_4(guide)
    assert "Global:\n- Global rule.\n" in section
    assert "Client `ccn`:\n- Client rule.\n" in section
    assert "This project:\n- Project rule.\n" in section
    assert "Other project." not in guide and "Only proposed." not in guide

    # The guide is built once per run: a lesson accepted now applies from the next run.
    lessons.add(env.hub.log, "Accepted mid-run.", "global", HUMAN)
    assert work(env) == 0
    assert "Accepted mid-run." not in read_guide(env, run)
    guide_id = outputs(env, run, "guide")["guide"]
    assert "Accepted mid-run." not in env.hub.store.read_bytes(guide_id).decode()


@pytest.mark.parametrize("state", ["proposed", "dropped", "retired"])
def test_only_accepted_lessons_reach_the_feed(hub: Hub, state: str) -> None:
    lesson = lessons.propose(hub.log, "A rule.", "global", AGENT)
    if state == "dropped":
        lessons.drop(hub.log, lesson.id, HUMAN)
    if state == "retired":
        lessons.accept(hub.log, lesson.id, "A rule.", "global", HUMAN)
        assert lessons.lessons_for_guide(hub.log, project_id=None, client=None).global_ == (
            "A rule.",
        )
        lessons.retire(hub.log, lesson.id, HUMAN)
    assert lessons.lessons_for_guide(hub.log, project_id=hub.project_id, client="ccn").empty
