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


def test_external_input_actors_route_through_the_shared_parser() -> None:
    """The external-input surfaces must not call ``Actor(<variable kind>, ...)`` directly; they go
    through parse_actor/parse_actor_text (which refuse runner). Flag any non-literal-kind Actor()
    outside events.py (the parser) and the log-fold modules that reconstruct actors from the log."""
    allowed = {
        SRC / "hub" / "events.py",  # the shared parser and Actor itself
    }
    surfaces = [
        SRC / "hub" / "cli.py",
        SRC / "hub" / "write_cli.py",
        SRC / "hub" / "nodeapi.py",
        SRC / "media" / "cli.py",
        SRC / "recipes" / "cli.py",
    ]
    offenders = []
    for path in surfaces:
        if path in allowed:
            continue
        for call in _actor_calls(ast.parse(path.read_text("utf-8"))):
            first = call.args[0] if call.args else None
            if not isinstance(first, ast.Constant):  # a variable/attribute kind from input
                offenders.append(f"{path}:{call.lineno}")
    assert offenders == [], f"Actor(<input kind>) built outside the shared parser: {offenders}"
