"""Optional lint inputs: a normalized transcript and a laugh map.

Both use **recording time** in seconds (see ``aar.plan.model``). ``recording`` names the
recording they describe; ``None`` means they apply to every recording in the plan.
Files may be either a bare JSON list or an object that wraps the list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Word(_Model):
    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    speaker: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Word:
        if self.end_s < self.start_s:
            raise ValueError(f"word {self.text!r} ends before it starts")
        return self


class Transcript(_Model):
    """Word-level transcript, sorted by start time."""

    recording: str | None = None
    words: list[Word]

    @model_validator(mode="after")
    def _sorted(self) -> Transcript:
        self.words = sorted(self.words, key=lambda word: (word.start_s, word.end_s))
        return self


class Laugh(_Model):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    strength: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Laugh:
        if self.end_s < self.start_s:
            raise ValueError("laugh ends before it starts")
        return self


class LaughMap(_Model):
    """Detected laughs, sorted by start time."""

    recording: str | None = None
    laughs: list[Laugh]

    @model_validator(mode="after")
    def _sorted(self) -> LaughMap:
        self.laughs = sorted(self.laughs, key=lambda laugh: (laugh.start_s, laugh.end_s))
        return self


def _wrap(data: Any, key: str) -> Any:
    return {key: data} if isinstance(data, list) else data


def load_transcript(path: str | Path) -> Transcript:
    """Read a transcript JSON file (a list of words, or ``{"recording":..., "words": [...]}``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Transcript.model_validate(_wrap(data, "words"))


def load_laughs(path: str | Path) -> LaughMap:
    """Read a laugh map JSON file (a list, or ``{"recording":..., "laughs": [...]}``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LaughMap.model_validate(_wrap(data, "laughs"))
