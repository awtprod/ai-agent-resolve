"""Shared pytest fixtures for the hub recipe tests.

Fixtures live here (not imported from a helper module) so using one as a test parameter does not
shadow a module-level import.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from recipe_support import register_test_recipe
from write_support import Hub, make_hub

from aar.recipes import registry
from aar.recipes.models import RecipeDef


@pytest.fixture
def registered_recipe() -> Iterator[RecipeDef]:
    """Register the test recipe (every step kind) for one test; clear afterwards."""
    definition = register_test_recipe()
    yield definition
    registry.clear()


@pytest.fixture
def hub(tmp_path) -> Iterator[Hub]:  # type: ignore[no-untyped-def]
    """A fresh hub for one test (used by the adopted validator probes)."""
    made = make_hub(tmp_path)
    yield made
    made.log.close()


@pytest.fixture
def ccn_env(tmp_path, monkeypatch) -> Iterator[Any]:  # type: ignore[no-untyped-def]
    """A hub with the ccn-longform recipe registered, an injected transcriber, and the work,
    media and export folders the worker's environment names (S19a)."""
    from ccn_recipe_support import CcnEnv, fake_transcriber

    from aar.recipes.ccn import register_ccn, set_laughs_runner, set_transcriber

    registry.clear()
    register_ccn()
    root = tmp_path / "hub"
    root.mkdir()
    folders = {name: tmp_path / name for name in ("media", "work", "exports", "agent")}
    for folder in folders.values():
        folder.mkdir()
    monkeypatch.setenv("AAR_MEDIA_ROOT", str(folders["media"]))
    monkeypatch.setenv("AAR_WORK_DIR", str(folders["work"]))
    monkeypatch.setenv("AAR_EXPORT_DIR", str(folders["exports"]))
    hub = make_hub(root)
    env = CcnEnv(hub, folders["media"], folders["work"], folders["exports"], folders["agent"])
    set_transcriber(fake_transcriber(env))
    try:
        yield env
    finally:
        set_transcriber(None)
        set_laughs_runner(None)
        hub.log.close()
        registry.clear()
