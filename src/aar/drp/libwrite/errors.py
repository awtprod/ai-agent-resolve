"""The refusal the library writer raises: a stable code, a message and a fix hint."""

from __future__ import annotations

from typing import Any

__all__ = ["CODES", "ProjectWriteError"]

#: Every refusal code of the library writer, with what it means.
CODES: dict[str, str] = {
    "bad_seed": "the seed is not a project the writer can start from",
    "seed_mismatch": "the plan's frame rate, resolution or start differs from the seed timeline's",
    "bad_input": "the media or items to write are inconsistent",
    "operation_not_opened": "the project needs an operation no person has opened in Resolve yet",
    "bad_project_name": "the project name is not a valid folder name on the share",
    "project_exists": "a project of that name is already in the library",
    "not_a_library": "the folder is not a Resolve disk library",
    "library_unwritable": "the library cannot be written (read-only, gone, or full)",
    "self_check_failed": "the written project does not read back as what was written",
}


class ProjectWriteError(ValueError):
    """A refusal: ``code`` (one of :data:`CODES`), ``message``, ``hint`` and ``details``."""

    def __init__(self, code: str, message: str, hint: str, **details: Any) -> None:
        if code not in CODES:
            raise ValueError(f"unknown library writer code {code!r}")
        super().__init__(f"{message} [{code}]; {hint}")
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details
