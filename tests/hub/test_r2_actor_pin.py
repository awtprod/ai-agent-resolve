"""N2 AST/grep pin: the runner identity, and Actor() from external input, are centralised.

No module outside the shared parser (``aar.hub.events.parse_actor`` / ``parse_actor_text``) may
build an ``Actor`` from an input-derived kind, and ``Actor("runner", ...)`` is constructed only
inside ``aar.recipes``. This is what keeps "no string surface may produce the runner identity"
true in one place (N2); the behavioural refusals are in test_r2_surfaces.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "aar"
# events.py defines the shared parser; aar.recipes builds the engine-only RUNNER directly.
_PARSER_FILE = SRC / "hub" / "events.py"


def _actor_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Actor"
        ):
            yield node


def test_runner_literal_only_inside_recipes() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        if "recipes" in path.relative_to(SRC).parts:
            continue
        for call in _actor_calls(ast.parse(path.read_text("utf-8"))):
            if (
                call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "runner"
            ):
                offenders.append(f"{path}:{call.lineno}")
    assert offenders == [], f"Actor('runner', ...) built outside aar.recipes: {offenders}"


def test_actor_from_a_non_literal_kind_only_in_the_shared_parser() -> None:
    """R3-5: scan EVERY module under src/aar. An ``Actor(<non-literal kind>, ...)`` -- a kind that
    could come from external input -- must live only in the shared parser (hub/events.py). Every
    other surface routes through parse_actor/parse_actor_text, which refuse the runner identity."""
    allowed = {SRC / "hub" / "events.py"}  # Actor itself, parse_actor and parse_actor_text
    offenders = []
    for path in SRC.rglob("*.py"):
        if path in allowed:
            continue
        for call in _actor_calls(ast.parse(path.read_text("utf-8"))):
            first = call.args[0] if call.args else None
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                offenders.append(f"{path.relative_to(SRC)}:{call.lineno}")
    assert offenders == [], f"Actor(<non-literal kind>) outside the shared parser: {offenders}"
