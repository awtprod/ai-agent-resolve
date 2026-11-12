"""A Transcript as plain text for a model to read, and the transcript's wordless stretches.

``render_text(transcript)`` is deterministic: the same transcript and settings always give the
same bytes. Each line is one spoken run, prefixed with its start and end in the transcript's
own clock (recording seconds when ``aar analyze transcript`` was given a recording offset)::

    [0612.4-0618.9] so anyway the dog got out
    [0619.8-0624.1] and nobody noticed
    [0624.3-2890.4] (no words transcribed)
    [2890.4-2893.0] welcome back

A new line starts at every gap between words longer than ``gap_s`` (0.7 s) and whenever the
text would pass ``max_chars`` characters (a word longer than that gets a line of its own). A gap
of at least ``wordless_min_s`` (60 s) between words is a line of its own, so a reader sees the
hole: **no words is not silence**. The transcriber may have missed speech there (quiet
far-field audio), or the room may be applauding. Only gaps between words are shown; the head
and tail of the file are not known from a transcript alone (``aar analyze transcript`` lists
them in its report). Times are printed to 0.1 s, zero-padded to one width over the file so the
lines align. Whitespace inside a word (a newline, a tab) becomes a single space, so one line is
always one run; a word that is only whitespace is skipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aar.plan.inputs import Transcript, Word

GAP_S = 0.7
MAX_CHARS = 120
WORDLESS_MIN_S = 60.0
MIN_MAX_CHARS = 20
NO_WORDS = "(no words transcribed)"


@dataclass(frozen=True)
class Stretch:
    """A wordless stretch, in the clock its bounds were given in."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a finite number of seconds above 0")


def wordless_stretches(
    words: Sequence[Word],
    min_s: float,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[Stretch]:
    """Every stretch of at least ``min_s`` that no word overlaps, in order.

    Words may overlap and need not be sorted. With ``start_s`` / ``end_s`` (the file's bounds in
    the words' clock) the head before the first word and the tail after the last count too; with
    no words at all the whole of ``[start_s, end_s]`` is one stretch. A stretch exactly ``min_s``
    long counts.
    """
    _positive("min_s", min_s)
    spans = sorted((w.start_s, w.end_s) for w in words)
    out: list[Stretch] = []
    cursor = start_s
    for lo, hi in spans:
        if cursor is not None and lo - cursor >= min_s:
            out.append(Stretch(cursor, lo))
        cursor = hi if cursor is None else max(cursor, hi)
    if end_s is not None and cursor is not None and end_s - cursor >= min_s:
        out.append(Stretch(cursor, end_s))
    return out


@dataclass
class _Line:
    start_s: float
    end_s: float
    text: str


def _clean(text: str) -> str:
    return " ".join(text.split())


def text_lines(
    transcript: Transcript,
    *,
    gap_s: float = GAP_S,
    max_chars: int = MAX_CHARS,
    wordless_min_s: float = WORDLESS_MIN_S,
) -> list[tuple[float, float, str]]:
    """``(start_s, end_s, text)`` per line, before formatting (see the module docstring)."""
    _positive("gap_s", gap_s)
    _positive("wordless_min_s", wordless_min_s)
    if isinstance(max_chars, bool) or not (
        isinstance(max_chars, int) and max_chars >= MIN_MAX_CHARS
    ):
        raise ValueError(f"max_chars must be a whole number of at least {MIN_MAX_CHARS}")
    lines: list[_Line] = []
    current: _Line | None = None
    words = sorted(transcript.words, key=lambda w: (w.start_s, w.end_s))
    for word in words:
        text = _clean(word.text)
        if not text:
            continue
        if current is not None:
            gap = word.start_s - current.end_s
            too_long = len(current.text) + 1 + len(text) > max_chars
            if gap >= wordless_min_s:
                lines.append(current)
                lines.append(_Line(current.end_s, word.start_s, NO_WORDS))
                current = None
            elif gap > gap_s or too_long:
                lines.append(current)
                current = None
        if current is None:
            current = _Line(word.start_s, word.end_s, text)
        else:
            current.text += " " + text
            current.end_s = max(current.end_s, word.end_s)
    if current is not None:
        lines.append(current)
    return [(line.start_s, line.end_s, line.text) for line in lines]


def render_text(
    transcript: Transcript,
    *,
    gap_s: float = GAP_S,
    max_chars: int = MAX_CHARS,
    wordless_min_s: float = WORDLESS_MIN_S,
) -> str:
    """The transcript as text, one ``[start-end] words`` line per run; ``""`` with no words."""
    lines = text_lines(transcript, gap_s=gap_s, max_chars=max_chars, wordless_min_s=wordless_min_s)
    if not lines:
        return ""
    width = max(len(f"{t:.1f}") for start, end, _ in lines for t in (start, end))
    width = max(width, len("0000.0"))
    return "".join(f"[{a:0{width}.1f}-{b:0{width}.1f}] {text}\n" for a, b, text in lines)


__all__ = [
    "GAP_S",
    "MAX_CHARS",
    "NO_WORDS",
    "WORDLESS_MIN_S",
    "Stretch",
    "render_text",
    "text_lines",
    "wordless_stretches",
]
