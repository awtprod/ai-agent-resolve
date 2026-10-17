"""Guards and helpers shared by the hub's supervision modules.

Reviews, approvals, the emergency stop and the lease board all need the same few
things: a check that an actor is a human, an injectable wall clock, and bounded,
validated text for anything an agent or a form can supply.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from aar.hub.events import Actor

Clock = Callable[[], datetime]
"""Returns the current time as a timezone-aware datetime. Injected so tests never sleep."""

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_CONTROL_SINGLE_LINE = re.compile(r"[\x00-\x1f\x7f]")


def utcnow() -> datetime:
    """Return the current UTC time (the default :data:`Clock`)."""
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    """Render an aware datetime as an ISO-8601 UTC string."""
    if moment.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return moment.astimezone(UTC).isoformat()


def parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp written by :func:`iso` (or the event log) as aware UTC."""
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError(f"timestamp has no timezone: {value!r}")
    return moment.astimezone(UTC)


class HumanRequired(PermissionError):
    """Raised when an agent or node tries something only a human may do."""

    def __init__(self, action: str, actor: Actor) -> None:
        super().__init__(f"only a human may {action}; {actor.kind}:{actor.id} is not a human")
        self.action = action
        self.actor = actor


def require_human(actor: Actor, action: str) -> None:
    """Raise :class:`HumanRequired` unless ``actor`` is ``Actor(kind="human")``."""
    if not isinstance(actor, Actor):
        raise TypeError(f"actor must be an Actor, got {type(actor).__name__}")
    if actor.kind != "human":
        raise HumanRequired(action, actor)


class InvalidInput(ValueError):
    """Raised when a supplied value fails validation. ``field`` names the bad input."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field


def clean_text(
    value: Any,
    field: str,
    *,
    max_len: int,
    required: bool = True,
    multiline: bool = False,
) -> str:
    """Return ``value`` stripped, or raise :class:`InvalidInput`.

    Rejects non-strings, control characters (newlines and tabs are allowed only when
    ``multiline``), empty values when ``required``, and anything over ``max_len``.
    """
    if not isinstance(value, str):
        raise InvalidInput(field, f"must be a string, got {type(value).__name__}")
    text = value.replace("\r\n", "\n").strip()
    pattern = _CONTROL if multiline else _CONTROL_SINGLE_LINE
    if pattern.search(text):
        raise InvalidInput(field, "must not contain control characters")
    if required and not text:
        raise InvalidInput(field, "must not be empty")
    if len(text) > max_len:
        raise InvalidInput(field, f"must be at most {max_len} characters (got {len(text)})")
    return text


def actor_dict(actor: Actor) -> dict[str, str]:
    """Return an actor as plain JSON-ready data."""
    return {"kind": actor.kind, "id": actor.id}
