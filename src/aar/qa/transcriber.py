"""The transcriber QA calls for the join re-transcription check.

No speech model ships with this slice. A caller injects anything that has ``transcribe``;
S13c adds a faster-whisper one. Tests use a fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from aar.plan.inputs import Word


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, path: Path, start_s: float, end_s: float) -> Sequence[Word]:
        """Transcribe ``[start_s, end_s)`` of ``path`` on its own (a fresh, short window).

        Returns the words heard, with times in seconds from the start of the file (not of the
        window). Raise on failure; QA reports the join as failed, never as passed.
        """
        ...
