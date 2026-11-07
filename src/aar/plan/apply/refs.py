"""Unresolved references: a typed placeholder for an id a phase cannot know until an earlier
phase has run.

A compiled step's params are a plain JSON-ready dict. Where a real id is not yet known (a
dry-run of a later phase without its input results), the dict carries an :func:`unresolved`
marker instead of a guessed value. The dry-run prints it, clearly marked. A submission payload
is scanned with :func:`find_unresolved`; if any marker survives, the submit is refused before
the hub is ever touched -- no placeholder is ever queued.
"""

from __future__ import annotations

from typing import Any

MARKER = "__unresolved__"


def unresolved(kind: str, hint: str) -> dict[str, Any]:
    """A typed unresolved reference. ``kind`` is e.g. 'timeline', 'media', 'title_template',
    'item'; ``hint`` says what it is and which input would resolve it."""
    return {MARKER: {"kind": kind, "hint": hint}}


def is_unresolved(value: Any) -> bool:
    return isinstance(value, dict) and MARKER in value and set(value) == {MARKER}


def show(value: dict[str, Any]) -> str:
    inner = value[MARKER]
    return f"<unresolved {inner['kind']}: {inner['hint']}>"


def find_unresolved(obj: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Every unresolved marker in a nested structure, as ``(json-path, inner)`` pairs."""
    found: list[tuple[str, dict[str, Any]]] = []
    if is_unresolved(obj):
        found.append((path or "<root>", obj[MARKER]))
        return found
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(find_unresolved(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(find_unresolved(value, f"{path}[{index}]"))
    return found
