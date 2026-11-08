"""Safe text of a swallowed exception: a broken ``__str__`` must never escape a handler."""

from __future__ import annotations

UNPRINTABLE = "<unprintable exception>"


def safe_str(exc: BaseException) -> str:
    """``str(exc)``, or a fixed marker when ``__str__`` itself raises."""
    try:
        return str(exc)
    except Exception:
        return UNPRINTABLE
