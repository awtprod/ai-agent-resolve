"""Recipe definitions: a recipe is an ordered DAG of typed steps, defined in code.

A :class:`RecipeDef` names a recipe (matching the hub's ``NAME_RE``), carries a version and a
pydantic params model, and holds an ordered tuple of :class:`StepDef`. Each step has an id, a
kind (``cold``, ``hot``, ``agent`` or ``gate``), the ids of earlier steps it needs (a DAG,
validated here), bounded ``max_attempts`` and ``timeout_s``, and the per-kind fields that say how
to run it. Definitions live in code and never cross the wire, so a step can hold Python objects
(a pydantic model for an ``expects: json`` contract).

Every bound is checked by :func:`validate_recipe`, called when a recipe is registered. A bad
definition is a programmer error, so it raises :class:`RecipeDefError` (a ``ValueError``), not a
hub ``WriteError``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from aar.hub.writes import NAME_RE

STEP_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_NAME_RE = re.compile(NAME_RE)

MAX_STEPS = 64
MIN_ATTEMPTS = 1
MAX_ATTEMPTS = 5
DEFAULT_ATTEMPTS = 3
MAX_TIMEOUT_S = 24 * 60 * 60  # 24 hours

StepKind = Literal["cold", "hot", "agent", "gate"]
ExpectKind = Literal["plan", "artifact", "json"]
MediaKind = Literal["master", "review", "short", "thumbnail", "captions", "any"]


class RecipeDefError(ValueError):
    """A recipe or step definition that violates the engine's rules (a programmer error)."""


@dataclass(frozen=True, slots=True)
class PlanExpect:
    """An agent step must produce a plan node that passes :func:`aar.recipes.gates.plan_gate`.

    The plan node must derive from the recipe node or the current round's review. ``required_rules``
    names rules that must have actually run in the plan's lint (none, by default).
    """

    required_rules: tuple[str, ...] = ()
    kind: Literal["plan"] = "plan"


@dataclass(frozen=True, slots=True)
class ArtifactExpect:
    """An agent step must produce an artifact id of a stated media kind."""

    media_kind: MediaKind = "any"
    kind: Literal["artifact"] = "artifact"


@dataclass(frozen=True, slots=True)
class JsonExpect:
    """An agent step must produce JSON that validates against ``model``."""

    model: type[BaseModel]
    kind: Literal["json"] = "json"


Expects = PlanExpect | ArtifactExpect | JsonExpect


@dataclass(frozen=True, slots=True)
class StepDef:
    """One step of a recipe. Only the fields its ``kind`` uses are meaningful.

    - ``cold``: ``callable_id`` names a registered cold callable; ``idempotent`` says whether a
      lapsed-lease claim may be re-claimed.
    - ``hot``: ``hot_kind`` is the request kind sent to the dispatcher; ``hot_params`` are static
      params merged under the recipe params.
    - ``agent``: ``instruction`` is the text the agent reads; ``expects`` is its output contract.
    - ``gate``: ``subject`` is the earlier step whose output is reviewed; ``on_reject`` is the
      earliest step to re-run when a human rejects.
    """

    id: str
    kind: StepKind
    needs: tuple[str, ...] = ()
    max_attempts: int = DEFAULT_ATTEMPTS
    timeout_s: float = 3600.0
    # cold
    callable_id: str | None = None
    idempotent: bool = False
    # hot
    hot_kind: str | None = None
    hot_params: Mapping[str, Any] | None = None
    # agent
    instruction: str | None = None
    expects: Expects | None = None
    # gate
    subject: str | None = None
    on_reject: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeDef:
    """A recipe: a name, a version, a params model and an ordered tuple of steps."""

    name: str
    version: int
    params_model: type[BaseModel]
    steps: tuple[StepDef, ...]

    def step(self, step_id: str) -> StepDef:
        """The step with ``step_id`` (``KeyError`` if there is none)."""
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def index(self, step_id: str) -> int:
        """The position of ``step_id`` in recipe order (``KeyError`` if unknown)."""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                return i
        raise KeyError(step_id)


def _check_step(step: StepDef, seen: set[str]) -> None:
    if not isinstance(step.id, str) or STEP_ID_RE.fullmatch(step.id) is None:
        raise RecipeDefError(f"step id {step.id!r} must match {STEP_ID_RE.pattern}")
    if step.id in seen:
        raise RecipeDefError(f"step id {step.id!r} is used twice")
    if step.kind not in ("cold", "hot", "agent", "gate"):
        raise RecipeDefError(f"step {step.id!r} has unknown kind {step.kind!r}")
    if not isinstance(step.max_attempts, int) or isinstance(step.max_attempts, bool):
        raise RecipeDefError(f"step {step.id!r} max_attempts must be an int")
    if not MIN_ATTEMPTS <= step.max_attempts <= MAX_ATTEMPTS:
        raise RecipeDefError(
            f"step {step.id!r} max_attempts must be {MIN_ATTEMPTS}..{MAX_ATTEMPTS}"
        )
    if not isinstance(step.timeout_s, (int, float)) or isinstance(step.timeout_s, bool):
        raise RecipeDefError(f"step {step.id!r} timeout_s must be a number")
    if not math.isfinite(step.timeout_s) or not 0 < step.timeout_s <= MAX_TIMEOUT_S:
        raise RecipeDefError(
            f"step {step.id!r} timeout_s must be finite and 0 < t <= {MAX_TIMEOUT_S}"
        )
    for need in step.needs:
        if need not in seen:
            raise RecipeDefError(
                f"step {step.id!r} needs {need!r}, which is not an earlier step (DAG rule)"
            )
    _check_kind_fields(step, seen)


def _check_kind_fields(step: StepDef, seen: set[str]) -> None:
    if step.kind == "cold":
        if not step.callable_id:
            raise RecipeDefError(f"cold step {step.id!r} needs a callable_id")
    elif step.kind == "hot":
        if not step.hot_kind:
            raise RecipeDefError(f"hot step {step.id!r} needs a hot_kind")
    elif step.kind == "agent":
        if not step.instruction:
            raise RecipeDefError(f"agent step {step.id!r} needs an instruction")
        if step.expects is None:
            raise RecipeDefError(f"agent step {step.id!r} needs an expects contract")
        if not isinstance(step.expects, (PlanExpect, ArtifactExpect, JsonExpect)):
            raise RecipeDefError(f"agent step {step.id!r} has an unknown expects contract")
    elif step.kind == "gate":
        if not step.subject or step.subject not in seen:
            raise RecipeDefError(
                f"gate step {step.id!r} subject must name an earlier step (got {step.subject!r})"
            )
        if not step.on_reject:
            raise RecipeDefError(f"gate step {step.id!r} needs on_reject")


def validate_recipe(recipe: RecipeDef, *, known_callables: Sequence[str] = ()) -> None:
    """Raise :class:`RecipeDefError` unless ``recipe`` satisfies every bound.

    ``known_callables`` lets the registry also check that each cold step's ``callable_id`` is
    registered; when empty the callable check is skipped (the models carry no callable registry).
    """
    if not isinstance(recipe.name, str) or _NAME_RE.fullmatch(recipe.name) is None:
        raise RecipeDefError(f"recipe name {recipe.name!r} must match {NAME_RE}")
    if (
        not isinstance(recipe.version, int)
        or isinstance(recipe.version, bool)
        or recipe.version < 1
    ):
        raise RecipeDefError(f"recipe {recipe.name!r} version must be an int >= 1")
    if not (isinstance(recipe.params_model, type) and issubclass(recipe.params_model, BaseModel)):
        raise RecipeDefError(f"recipe {recipe.name!r} params_model must be a pydantic model")
    if not isinstance(recipe.steps, tuple) or not recipe.steps:
        raise RecipeDefError(f"recipe {recipe.name!r} needs at least one step")
    if len(recipe.steps) > MAX_STEPS:
        raise RecipeDefError(f"recipe {recipe.name!r} has over {MAX_STEPS} steps")
    seen: set[str] = set()
    known = set(known_callables)
    for step in recipe.steps:
        _check_step(step, seen)
        if known_callables and step.kind == "cold" and step.callable_id not in known:
            raise RecipeDefError(
                f"cold step {step.id!r} callable_id {step.callable_id!r} is not registered"
            )
        seen.add(step.id)
