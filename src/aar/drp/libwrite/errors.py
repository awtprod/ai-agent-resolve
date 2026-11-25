"""The refusal the library writer raises, and the one typed boundary around its entry points.

:class:`ProjectWriteError` is a refusal with a stable code, a message, a fix hint and details.
:func:`refusing` wraps every public entry point of :mod:`aar.drp.libwrite` (as
:func:`aar.drp.mediapool.errors.refusing` wraps the media-pool and item writers'): whatever a
damaged seed, template or input makes the code below raise (the shared decoders' errors,
``struct``, arithmetic, indexing, conversion, type errors, SQLite's errors and the
``UnicodeDecodeError`` of a damaged file's text), and the library reader's and the item and
media writers' own refusals, become a ``ProjectWriteError`` of the entry point's code, with the
original type and message kept. What a caller's own callback raised (``make_project``'s
``author`` and ``check``, ``place_project``'s ``build``) is the caller's and passes unchanged
(:func:`by_caller` marks it), and so do interrupts and ``OSError`` (the placement turns an
``OSError`` of the library into ``library_unwritable`` itself).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

__all__ = ["CODES", "ProjectWriteError", "by_caller", "from_caller", "refusing", "with_residue"]

#: Every refusal code of the library writer, with what it means.
CODES: dict[str, str] = {
    "bad_seed": "the seed is not a project the writer can start from",
    "seed_mismatch": "the plan's frame rate, resolution or start differs from the seed timeline's",
    "bad_input": "the media or items to write are inconsistent",
    "bad_transition": "a transition is not at a join of two clips, or the media it plays is "
    "not there",
    "transitions_overlap": "two transitions on one clip need more frames than it has (the plan "
    "export's code for the same rule)",
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


_MARK = "_aar_raised_by_caller"
_P = ParamSpec("_P")
_R = TypeVar("_R")


def by_caller(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    """Call a caller's callback; whatever it raises is marked as the caller's, so no boundary
    of the writer rewrites it."""
    try:
        return func(*args, **kwargs)
    except BaseException as exc:
        # An exception type without a __dict__ cannot be marked: it is left as it is.
        with contextlib.suppress(AttributeError, TypeError):
            setattr(exc, _MARK, True)
        raise


def from_caller(exc: BaseException) -> bool:
    """Whether a caller's callback raised ``exc`` (see :func:`by_caller`)."""
    return bool(getattr(exc, _MARK, False))


def _damage_errors() -> tuple[type[BaseException], ...]:
    from aar.drp.library import SQL_ERRORS
    from aar.drp.mediapool.errors import DAMAGE_ERRORS

    return (*DAMAGE_ERRORS, *SQL_ERRORS)


def converted(exc: BaseException, code: str, what: str, hint: str) -> ProjectWriteError | None:
    """``exc`` as a ``code`` refusal, or None when it is not the writer's to convert (the
    caller's, an interrupt, an ``OSError``, a refusal already)."""
    from aar.drp.library import LibraryError
    from aar.drp.mediapool.errors import MediaPoolError

    if isinstance(exc, ProjectWriteError) or from_caller(exc):
        return None
    if isinstance(exc, LibraryError):
        return ProjectWriteError(
            code, f"{what}: {exc.message} [{exc.code}]", exc.hint, library_code=exc.code
        )
    if isinstance(exc, MediaPoolError):
        return ProjectWriteError(
            code, f"{what}: {exc.message} [{exc.code}]", exc.hint, media_code=exc.code
        )
    if isinstance(exc, _damage_errors()):
        detail = f"{type(exc).__name__}: {exc}"[:400]
        return ProjectWriteError(code, f"{what} failed on a damaged value ({detail})", hint)
    return None


def refusing(
    code: str, what: str, hint: str, *, passthrough: tuple[type[BaseException], ...] = ()
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap a public entry point of the writer (see the module docstring): what it lets escape
    becomes a ``code`` refusal saying ``what`` failed, unless it is a ``ProjectWriteError``
    already, the caller's, one of ``passthrough``, or not damage (an interrupt, an
    ``OSError``, ``MemoryError``)."""
    if code not in CODES:
        raise ValueError(f"unknown library writer code {code!r}")

    def wrap(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except passthrough:
                raise
            except BaseException as exc:
                refusal = converted(exc, code, what, hint)
                if refusal is None:
                    raise
                raise refusal from exc

        return guarded

    return wrap


def with_residue(exc: BaseException, residue: str, note: str) -> None:
    """Name what a failed build left in the library on ``exc`` itself, in its text and in its
    JSON: a typed refusal (``ProjectWriteError``, an export's ``ExportError``, a media or item
    refusal) gets ``note`` added to its message and ``residue`` to its details (``residue``
    in the JSON); any other exception (an interrupt) gets ``note`` as an exception note, which
    the traceback and the CLI's interrupt message print."""
    message = getattr(exc, "message", None)
    if isinstance(message, str):
        exc.message = f"{message} ({note})"  # type: ignore[attr-defined]
        exc.args = (
            (str(exc.args[0]).replace(message, exc.message, 1),)  # type: ignore[attr-defined]
            if exc.args and isinstance(exc.args[0], str)
            else (exc.message,)  # type: ignore[attr-defined]
        )
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            details["residue"] = residue
        else:
            try:
                exc.residue = residue  # type: ignore[attr-defined]
            except AttributeError:
                exc.add_note(note)
        return
    with contextlib.suppress(AttributeError):
        exc.residue = residue  # type: ignore[attr-defined]
    exc.add_note(note)
