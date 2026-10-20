"""Shared pytest fixtures for the hub recipe tests.

Fixtures live here (not imported from a helper module) so using one as a test parameter does not
shadow a module-level import.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from recipe_support import register_test_recipe

from aar.recipes import registry
from aar.recipes.models import RecipeDef


@pytest.fixture
def registered_recipe() -> Iterator[RecipeDef]:
    """Register the test recipe (every step kind) for one test; clear afterwards."""
    definition = register_test_recipe()
    yield definition
    registry.clear()
