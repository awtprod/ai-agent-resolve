"""Typed errors for the ``aar.fusion`` package."""

from __future__ import annotations

from typing import Any

__all__ = ["FusionError"]


class FusionError(Exception):
    """Structured error raised by Fusion template parsing, rendering, and application."""

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail: dict[str, Any] = dict(detail) if detail else {}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail:
            out["detail"] = self.detail
        return out
