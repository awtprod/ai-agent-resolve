"""S12g: the registry writers cap the aliases they accept.

``create_project(aliases=[...])`` used to check only the string fields, so a 1,000,000
character alias, or 600 aliases of 500 characters, went into the log. Writers now refuse
an alias over MAX_NAME_CHARS and more than MAX_ALIASES aliases on a project, before
anything is written. The fold, and a retry of an already-stored request, are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.registry import MAX_ALIASES, MAX_NAME_CHARS, Registry

HUMAN = Actor("human", "operator-1")


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "hub.db")


def _aliases(count: int) -> list[str]:
    return [f"alias-{i}" for i in range(count)]


def test_the_limits_are_what_the_brief_says() -> None:
    assert (MAX_NAME_CHARS, MAX_ALIASES) == (500, 64)


@pytest.mark.parametrize(
    ("aliases", "match"),
    [
        (["a" * 1_000_000], f"alias is over {MAX_NAME_CHARS}"),
        (["ok", "a" * (MAX_NAME_CHARS + 1)], f"alias is over {MAX_NAME_CHARS}"),
        (["a" * (MAX_NAME_CHARS + 1)] * 10, f"alias is over {MAX_NAME_CHARS}"),
        ([f"a{i}" for i in range(MAX_ALIASES + 1)], f"too many aliases \\(over {MAX_ALIASES}\\)"),
        (["a" * MAX_NAME_CHARS] * 600, f"too many aliases \\(over {MAX_ALIASES}\\)"),
        ([5], "aliases are not a list of text"),
        (["ok", None], "aliases are not a list of text"),
        ([["nested"]], "aliases are not a list of text"),
    ],
    ids=[
        "one-million-chars",
        "one-over-the-length",
        "ten-over-the-length",
        "one-over-the-count",
        "600-at-the-length",
        "int",
        "none",
        "list",
    ],
)
def test_create_project_refuses_bad_aliases_before_writing(
    log: EventLog, aliases: list[Any], match: str
) -> None:
    registry = Registry(log)
    before = log.latest_seq()
    with pytest.raises(ValueError, match=f"refused before writing project.created: {match}"):
        registry.create_project("demo-show", HUMAN, aliases=aliases)
    assert log.latest_seq() == before
    assert registry.list() == [] and Registry(log).list() == []


def test_create_project_accepts_exactly_the_limits(log: EventLog) -> None:
    registry = Registry(log)
    aliases = [str(i).ljust(MAX_NAME_CHARS, "x") for i in range(MAX_ALIASES)]
    project = registry.create_project("demo-show", HUMAN, aliases=aliases)
    assert project.aliases == aliases
    assert Registry(log).get(project.id) == project


def test_add_alias_stops_at_the_limit_but_an_existing_alias_still_goes_through(
    log: EventLog,
) -> None:
    registry = Registry(log)
    aliases = _aliases(MAX_ALIASES)
    project = registry.create_project("demo-show", HUMAN, aliases=aliases)
    before = log.latest_seq()
    with pytest.raises(ValueError, match=r"refused before writing project\.alias_added: too many"):
        registry.add_alias(project.id, "one-too-many", HUMAN)
    assert log.latest_seq() == before
    assert Registry(log).get(project.id).aliases == aliases  # type: ignore[union-attr]
    # Naming an alias the project already has adds nothing, so the cap does not apply.
    assert registry.add_alias(project.id, aliases[3], HUMAN).aliases == aliases
    # One below the limit, the last alias fits.
    smaller = registry.create_project("smaller", HUMAN, aliases=_aliases(MAX_ALIASES - 1))
    assert len(registry.add_alias(smaller.id, "the-last-one", HUMAN).aliases) == MAX_ALIASES


def test_one_oversized_alias_added_later_is_still_refused(log: EventLog) -> None:
    registry = Registry(log)
    project = registry.create_project("demo-show", HUMAN)
    before = log.latest_seq()
    with pytest.raises(ValueError, match=f"alias is over {MAX_NAME_CHARS}"):
        registry.add_alias(project.id, "a" * 1_000_000, HUMAN)
    assert log.latest_seq() == before


# -- retries and old data: writers refuse, folds and retries accept ----------------------


def test_a_retry_of_a_create_at_the_limit_returns_the_stored_project(log: EventLog) -> None:
    registry = Registry(log)
    aliases = _aliases(MAX_ALIASES)
    first = registry.create_project("demo-show", HUMAN, aliases=aliases, idempotency_key="k-1")
    before = log.latest_seq()
    again = registry.create_project("demo-show", HUMAN, aliases=aliases, idempotency_key="k-1")
    assert again == first and log.latest_seq() == before
    # A different request under the same key is still a conflict, not a "refused".
    with pytest.raises(IdempotencyConflict):
        registry.create_project("demo-show", HUMAN, aliases=aliases[:-1], idempotency_key="k-1")


def test_a_retry_of_the_alias_that_filled_the_project_returns_it(log: EventLog) -> None:
    registry = Registry(log)
    project = registry.create_project("demo-show", HUMAN, aliases=_aliases(MAX_ALIASES - 1))
    filled = registry.add_alias(project.id, "the-last-one", HUMAN, idempotency_key="k-last")
    before = log.latest_seq()
    # The project is now full, so a fresh alias is refused ...
    with pytest.raises(ValueError, match="too many aliases"):
        registry.add_alias(project.id, "another", HUMAN, idempotency_key="k-other")
    # ... but the retry of the one that fit matches the stored event first.
    assert registry.add_alias(project.id, "the-last-one", HUMAN, idempotency_key="k-last") == filled
    with pytest.raises(IdempotencyConflict):
        registry.add_alias(project.id, "a-different-one", HUMAN, idempotency_key="k-last")
    assert log.latest_seq() == before


def test_the_fold_keeps_old_oversized_events_and_their_retries_still_work(log: EventLog) -> None:
    # Written straight to the log, the way an older writer (with no cap) would have.
    huge = ["a" * 1_000_000] + ["b" * MAX_NAME_CHARS + str(i) for i in range(600)]
    create = {"project_id": "prj_old", "name": "old", "sources": [], "aliases": huge}
    log.append("project.created", HUMAN, create, project_id="prj_old", idempotency_key="k-old")
    late = {"project_id": "prj_old", "alias": "c" * 5000}
    log.append("project.alias_added", HUMAN, late, project_id="prj_old", idempotency_key="k-late")

    registry = Registry(log)
    project = registry.get("prj_old")
    assert project is not None and project.aliases == [*huge, "c" * 5000]
    assert registry.skipped() == () and registry.rebuild()["prj_old"].aliases == project.aliases

    before = log.latest_seq()
    again = registry.create_project("old", HUMAN, aliases=huge, idempotency_key="k-old")
    assert again == project
    assert registry.add_alias("prj_old", "c" * 5000, HUMAN, idempotency_key="k-late") == project
    assert log.latest_seq() == before
    # A new write to that same project is still held to the cap.
    with pytest.raises(ValueError, match="too many aliases"):
        registry.add_alias("prj_old", "fresh", HUMAN)
    assert log.latest_seq() == before
