"""The one error the media-pool writer raises: a refusal with a stable code and a hint.

Every public entry point of the package is wrapped in :func:`refusing`, the one place where an
error a damaged input can raise inside the shared decoders and encoders (:mod:`aar.drp.blobs`,
:mod:`aar.drp.modify.encode`) or in the arithmetic and indexing done on what they return becomes
a :class:`MediaPoolError`, with the original message kept. The specific refusals raised inside
pass through unchanged; the wrapper only catches what no specific check foresaw.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from functools import wraps
from typing import ClassVar, ParamSpec, TypeVar

from aar.drp.errors import DrpError

__all__ = ["CODES", "DAMAGE_ERRORS", "MediaPoolError", "refusing"]

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
    "bad_placement": "the placement (bin, position, project rate, dates, map) is out of range",
    "unverified": "the result would not be verified, and the caller did not allow that",
}


class MediaPoolError(ValueError):
    """A refusal: ``code`` (one of :data:`CODES`), ``message`` and a ``hint`` saying what to do.

    A sibling writer (the timeline items of :mod:`aar.drp.items`) subclasses it with its own
    ``CODES``, so the one boundary (:func:`refusing`) serves both."""

    CODES: ClassVar[dict[str, str]] = CODES

    def __init__(self, code: str, message: str, hint: str) -> None:
        if code not in self.CODES:
            raise ValueError(f"unknown {type(self).__name__} code {code!r}")
        super().__init__(f"{message} [{code}]; {hint}")
        self.code = code
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


#: What a damaged blob, row or fact can raise below the package's own checks: the shared
#: decoders' and encoders' errors (:class:`aar.drp.errors.DrpError`, which includes
#: ``EncodeError``), ``struct`` unpacking, and the arithmetic, indexing, conversion and type
#: errors of code working on what they return (an ``AttributeError``: a value of another type).
DAMAGE_ERRORS: tuple[type[BaseException], ...] = (
    DrpError,
    struct.error,
    ArithmeticError,
    LookupError,
    ValueError,
    TypeError,
    AttributeError,  # a value of another type than the code expects (``.decode`` on a number)
    RecursionError,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def refusing(
    code: str,
    what: str,
    hint: str,
    *,
    passthrough: tuple[type[BaseException], ...] = (),
    error: type[MediaPoolError] = MediaPoolError,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap a public entry point: any of :data:`DAMAGE_ERRORS` it lets escape becomes a
    ``code`` refusal of type ``error`` saying ``what`` failed, with the original error's type
    and message. A refusal of type ``error`` and the typed errors in ``passthrough`` pass
    unchanged; any other :class:`MediaPoolError` becomes an ``error``, keeping its code when
    ``error`` has it."""
    if code not in error.CODES:
        raise ValueError(f"unknown {error.__name__} code {code!r}")

    def wrap(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except MediaPoolError as exc:
                if isinstance(exc, error):
                    raise
                # A refusal of the sibling writer below this one keeps its code when this
                # writer has it, so every refusal out of an entry point is of its own type.
                kept = exc.code if exc.code in error.CODES else code
                raise error(kept, exc.message, exc.hint) from exc
            except passthrough:
                raise
            except DAMAGE_ERRORS as exc:
                detail = f"{type(exc).__name__}: {exc}"[:400]
                raise error(code, f"{what} failed on a damaged value ({detail})", hint) from exc

        return guarded

    return wrap
