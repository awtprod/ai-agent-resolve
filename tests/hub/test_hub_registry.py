"""Tests for the project registry projection."""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.registry import DuplicateProjectId, Project, ProjectSource, Registry

ACTOR = Actor("human", "andrew")


def _registry(tmp_path: Path) -> tuple[EventLog, Registry]:
    log = EventLog(tmp_path / "hub.db")
    return log, Registry(log)


def test_project_source_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        ProjectSource("usb_stick", {"path": "/x"})


def test_create_assigns_stable_prefixed_id(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    project = reg.create_project("QCCE 2024", ACTOR)
    assert project.id.startswith("prj_")
    assert reg.get(project.id) is not None


def test_rename_keeps_the_id(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    project = reg.create_project("QCCE 2024", ACTOR)
    pid = project.id
    reg.rename_project(pid, "QCCE 2024 (recut)", ACTOR)
    renamed = reg.get(pid)
    assert renamed.id == pid
    assert renamed.name == "QCCE 2024 (recut)"


def test_find_by_locator_matches_kind_locator_and_optional_host(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    locator = {"library": "CCN", "project": "Roy Cooper"}
    source = ProjectSource("cloud_library", locator, host="edit-pc-1")
    project = reg.create_project("Roy Cooper", ACTOR, sources=[source])

    assert reg.find_by_locator("cloud_library", locator).id == project.id
    assert reg.find_by_locator("cloud_library", locator, host="edit-pc-1").id == project.id
    assert reg.find_by_locator("cloud_library", locator, host="other-host") is None
    assert reg.find_by_locator("drp_file", {"path": "/nope.drp"}) is None


def test_add_source_and_alias_then_lookup(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    project = reg.create_project("LNR Ep 5", ACTOR)
    reg.add_alias(project.id, "lnr5", ACTOR)
    reg.add_source(
        project.id, ProjectSource("drp_file", {"path": "/exports/lnr5.drp"}, host="mac"), ACTOR
    )

    assert reg.find_by_alias("lnr5").id == project.id
    assert reg.find_by_name("LNR Ep 5").id == project.id
    stored = reg.get(project.id)
    assert stored.sources[0].kind == "drp_file"
    assert stored.sources[0].host == "mac"


def test_mutating_unknown_project_raises(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    with pytest.raises(KeyError):
        reg.rename_project("prj_missing", "x", ACTOR)


def test_reused_explicit_id_is_rejected_and_does_not_clobber(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    reg.create_project("A", ACTOR, project_id="prj_fixed")
    reg.rename_project("prj_fixed", "A renamed", ACTOR)
    reg.add_alias("prj_fixed", "a-alias", ACTOR)

    # Reusing the id with a fresh (no) key must be rejected, not silently overwrite.
    with pytest.raises(DuplicateProjectId):
        reg.create_project("B", ACTOR, project_id="prj_fixed")

    stored = reg.get("prj_fixed")
    assert stored.name == "A renamed"
    assert stored.aliases == ["a-alias"]


def test_same_key_retry_returns_existing_project(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    first = reg.create_project("A", ACTOR, project_id="prj_fixed", idempotency_key="k")
    again = reg.create_project("A", ACTOR, project_id="prj_fixed", idempotency_key="k")
    assert again.id == first.id == "prj_fixed"
    assert len(reg.snapshot()) == 1


def test_reused_key_for_a_different_create_conflicts(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    reg.create_project("A", ACTOR, idempotency_key="K")
    with pytest.raises(IdempotencyConflict):
        reg.create_project("B", ACTOR, idempotency_key="K")
    assert len(reg.snapshot()) == 1


def test_reused_key_for_a_different_rename_conflicts(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    project = reg.create_project("A", ACTOR)
    reg.rename_project(project.id, "X", ACTOR, idempotency_key="R")
    # The same key for a different new name is not a retry.
    with pytest.raises(IdempotencyConflict):
        reg.rename_project(project.id, "Y", ACTOR, idempotency_key="R")
    assert reg.get(project.id).name == "X"


def test_projection_rebuilds_from_log_and_survives_idempotent_replay(tmp_path: Path) -> None:
    log, reg = _registry(tmp_path)
    project = reg.create_project("A", ACTOR, idempotency_key="create-A")
    reg.rename_project(project.id, "B", ACTOR)
    reg.add_alias(project.id, "b-alias", ACTOR)

    # A replayed create with the same key must return the original and not clobber the rename.
    replay = reg.create_project("A", ACTOR, idempotency_key="create-A")
    assert replay.id == project.id
    assert reg.get(project.id).name == "B"

    # Live state equals a fresh rebuild from the log...
    assert reg.rebuild() == reg.snapshot()
    # ...and equals what a brand-new Registry projects from the same log.
    reg2 = Registry(log)
    assert reg2.snapshot() == reg.snapshot()

    expected = Project(
        id=project.id, name="B", aliases=["b-alias"], created=reg.get(project.id).created
    )
    assert reg2.get(project.id) == expected


def test_cross_project_rename_key_reuse_conflicts(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    a = reg.create_project("A", ACTOR)
    b = reg.create_project("B", ACTOR)
    reg.rename_project(a.id, "Renamed", ACTOR, idempotency_key="RK")
    # Same key, same new name, different project: a conflict, not a stale-key retry.
    with pytest.raises(IdempotencyConflict):
        reg.rename_project(b.id, "Renamed", ACTOR, idempotency_key="RK")
    assert reg.get(b.id).name == "B"


@pytest.mark.parametrize(
    ("name", "override"),
    [
        ("B", {}),  # name
        ("A", {"sources": [ProjectSource("drp_file", {"path": "/b.drp"})]}),
        ("A", {"aliases": ["y"]}),
        ("A", {"project_id": "prj_other"}),
    ],
)
def test_create_conflicts_when_any_compared_field_changes(
    tmp_path: Path, name: str, override: dict
) -> None:
    _, reg = _registry(tmp_path)
    base = {
        "sources": [ProjectSource("drp_file", {"path": "/a.drp"})],
        "aliases": ["x"],
        "project_id": "prj_base",
    }
    reg.create_project("A", ACTOR, idempotency_key="C", **base)
    with pytest.raises(IdempotencyConflict):
        reg.create_project(name, ACTOR, idempotency_key="C", **{**base, **override})


def test_add_alias_conflicts_on_a_different_value(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    project = reg.create_project("A", ACTOR)
    reg.add_alias(project.id, "x", ACTOR, idempotency_key="AK")
    with pytest.raises(IdempotencyConflict):
        reg.add_alias(project.id, "y", ACTOR, idempotency_key="AK")


def test_add_source_conflicts_on_a_different_value(tmp_path: Path) -> None:
    _, reg = _registry(tmp_path)
    project = reg.create_project("A", ACTOR)
    reg.add_source(
        project.id, ProjectSource("drp_file", {"path": "/a.drp"}), ACTOR, idempotency_key="SK"
    )
    with pytest.raises(IdempotencyConflict):
        reg.add_source(
            project.id, ProjectSource("drp_file", {"path": "/b.drp"}), ACTOR, idempotency_key="SK"
        )


def test_create_key_reused_for_rename_conflicts(tmp_path: Path) -> None:
    # Pins the event_type clause in match_or_conflict: a create key reused for a rename
    # to the *same* name matches on every compared field, so only the type check catches
    # it. Without that clause the rename is silently accepted and nothing is written.
    log, reg = _registry(tmp_path)
    project = reg.create_project("C", ACTOR, idempotency_key="CK")
    with pytest.raises(IdempotencyConflict):
        reg.rename_project(project.id, "C", ACTOR, idempotency_key="CK")
    assert reg.get(project.id).name == "C"
    assert log.read(types=["project.renamed"]) == []
