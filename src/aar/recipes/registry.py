"""The recipe registry: recipes and cold callables, defined in code.

Recipes are registered with :func:`register`; it validates the definition (every bound in
:mod:`aar.recipes.models`) and installs the recipe and its params model. Cold step callables are
registered with :func:`register_callable`. :func:`registry_params_validator` is the
:data:`~aar.hub.writes.ParamsValidator` that the recipe-running surfaces plug into S13g1's hook:
it refuses an unregistered recipe name with ``unknown_recipe`` and otherwise validates the params
against the recipe's registered pydantic model (``invalid_params``).

The production registry is populated by recipe modules (named with ``--recipes`` or the
``AAR_RECIPES`` env), loaded by :func:`load_modules`. The built-in ``recipe.report`` callable is
registered by :func:`ensure_builtins`. Tests register their own recipes and callables and call
:func:`clear` to isolate.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence

from pydantic import ValidationError

from aar.hub.writes import Brief, ParamProblem, WriteError, validation_details
from aar.recipes.context import StepContext, StepOutput
from aar.recipes.models import RecipeDef, validate_recipe

ColdCallable = Callable[[StepContext], StepOutput]

_RECIPES: dict[str, RecipeDef] = {}
_CALLABLES: dict[str, ColdCallable] = {}
_BUILTINS_LOADED = False


def register(recipe: RecipeDef) -> RecipeDef:
    """Validate and install ``recipe``; return it. Re-registering the same name is refused."""
    ensure_builtins()
    validate_recipe(recipe, known_callables=tuple(_CALLABLES))
    if recipe.name in _RECIPES and _RECIPES[recipe.name] is not recipe:
        raise ValueError(f"recipe {recipe.name!r} is already registered")
    _RECIPES[recipe.name] = recipe
    return recipe


def register_callable(callable_id: str, fn: ColdCallable) -> ColdCallable:
    """Register a cold step callable under ``callable_id``; return it."""
    if not callable_id or not callable(fn):
        raise ValueError("register_callable needs a non-empty id and a callable")
    _CALLABLES[callable_id] = fn
    return fn


def get(name: str) -> RecipeDef | None:
    """The registered recipe of this name, or ``None``."""
    ensure_builtins()
    return _RECIPES.get(name)


def names() -> tuple[str, ...]:
    """Every registered recipe name, sorted."""
    ensure_builtins()
    return tuple(sorted(_RECIPES))


def callable_for(callable_id: str) -> ColdCallable:
    """The cold callable registered under ``callable_id`` (``KeyError`` if unknown)."""
    ensure_builtins()
    return _CALLABLES[callable_id]


def has_callable(callable_id: str) -> bool:
    ensure_builtins()
    return callable_id in _CALLABLES


def clear() -> None:
    """Drop every registered recipe and callable (tests), then reload the built-ins."""
    global _BUILTINS_LOADED
    _RECIPES.clear()
    _CALLABLES.clear()
    _BUILTINS_LOADED = False
    ensure_builtins()


def ensure_builtins() -> None:
    """Register the engine's built-in callables (``recipe.report``), idempotently.

    Registers directly (not via an import side effect) so :func:`clear` can drop and restore the
    built-ins even though the module stays cached. Imported here, not at module top, so the
    registry has no import cycle with ``report -> state -> registry``.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from aar.recipes.report import report_step

    _CALLABLES["recipe.report"] = report_step


def load_modules(modules: Sequence[str]) -> None:
    """Import each module name so its ``register(...)`` calls populate the registry."""
    ensure_builtins()
    for name in modules:
        name = name.strip()
        if name:
            importlib.import_module(name)


def modules_from_env(explicit: Sequence[str] | None = None) -> list[str]:
    """Recipe module names from ``--recipes`` (``explicit``) plus the ``AAR_RECIPES`` env."""
    found: list[str] = list(explicit or [])
    env = os.environ.get("AAR_RECIPES", "")
    found.extend(part for part in env.split(",") if part.strip())
    return found


def registry_params_validator(
    recipe: str, params: Mapping[str, object], brief: Brief
) -> Sequence[ParamProblem]:
    """The :data:`~aar.hub.writes.ParamsValidator` backed by this registry.

    Refuses an unregistered recipe name with ``WriteError("unknown_recipe")`` (so ``recipe_create``
    returns it to the caller), and otherwise validates ``params`` against the recipe's registered
    pydantic model, returning ``(loc, msg)`` problems that ``recipe_create`` reports as
    ``invalid_params``.
    """
    definition = get(recipe)
    if definition is None:
        raise WriteError(
            "unknown_recipe",
            f"recipe {recipe!r} is not registered",
            "Use a registered recipe name (recipe_list), or register the recipe in code.",
            recipe=recipe,
            known=list(names()),
        )
    try:
        definition.params_model.model_validate(dict(params))
    except ValidationError as exc:
        details = validation_details(exc)
        return [(error["loc"], error["msg"]) for error in details["errors"]]
    return ()
