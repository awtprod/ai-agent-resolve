"""Typed errors of the plan producers.

Exit codes follow ``aar plan lint``: 2 when an input, the profile or a tool cannot be read or is
malformed (``ProduceInputError``, ``ProfileError``), 1 when the inputs were read but no plan that
lints clean could be made (``InvariantError``, ``LintGateError`` and the other
``ProduceError`` types).
"""

from __future__ import annotations

from typing import Any

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INPUT = 2


class ProduceError(ValueError):
    """The producer could not make a plan. ``code`` is a stable kebab-case id."""

    exit_code = EXIT_FAILED

    def __init__(
        self, code: str, message: str, *, hint: str = "", details: list[Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or []

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.code, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        if self.details:
            out["details"] = self.details
        return out


class ProduceInputError(ProduceError):
    """An input file is missing, unreadable, malformed or inconsistent with the others."""

    exit_code = EXIT_INPUT


class InvariantError(ProduceError):
    """A produced plan broke a house invariant (chronological order, no dialogue splice).

    The transforms are built never to do this, so it means a bug or an input they cannot handle;
    the plan is not written.
    """


class LintGateError(ProduceError):
    """The produced plan has lint errors under the profile, so it is not written."""

    def __init__(self, message: str, *, hint: str, details: list[Any], report: Any) -> None:
        super().__init__("lint-errors", message, hint=hint, details=details)
        self.report = report
