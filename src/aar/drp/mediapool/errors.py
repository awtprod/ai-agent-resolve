"""The one error the media-pool writer raises: a refusal with a stable code and a hint."""

from __future__ import annotations

__all__ = ["CODES", "MediaPoolError"]

#: Every refusal code, with what it means.
CODES: dict[str, str] = {
    "bad_facts": "the measured facts are missing, malformed or out of range",
    "bad_template": "the template clip's rows are incomplete, inconsistent or do not decode",
    "no_template": "no template of the file's kind was given",
    "kind_mismatch": "the template is of another kind than the file",
    "unsupported_kind": "the file's kind has no evidence that its rows can be authored",
    "cannot_derive": "a value the rows need cannot be derived from the facts",
    "vfr_without_timecode": "a variable frame rate file with no timecode track to give its rate",
    "not_exact": "a rate or duration that cannot be represented exactly",
    "id_collision": "a new id is already used in the project, or twice",
}


class MediaPoolError(ValueError):
    """A refusal: ``code`` (one of :data:`CODES`), ``message`` and a ``hint`` saying what to do."""

    def __init__(self, code: str, message: str, hint: str) -> None:
        if code not in CODES:
            raise ValueError(f"unknown media-pool error code {code!r}")
        super().__init__(f"{message} [{code}]; {hint}")
        self.code = code
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "hint": self.hint}
