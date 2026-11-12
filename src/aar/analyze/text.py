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
always one run; other control characters (C0, C1 and DEL: NUL, ESC, ...) are dropped, and a
word left empty is skipped.

**Overlaps.** Gaps are measured from the furthest end any earlier word reached, across line
breaks, so a short word inside a long one never opens a false gap or wordless line.

**Exact boundaries.** Times are compared with a tolerance of ``TIME_EPS_S`` (1 us, far below the
0.1 s printed and far above float noise), so a gap of exactly ``gap_s`` (614.5 + 0.7) stays on
one line and a gap of exactly ``wordless_min_s`` is a wordless line.

**Times must be sane.** A time that is not finite or is past ``MAX_TIME_S`` (10^7 s, about 116
days, beyond any recording clock) is refused (``ValueError``; ``transcript-text`` reports it as
``invalid-transcript``), so a stray 1e300 cannot widen every line to hundreds of digits.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from aar.plan.inputs import Transcript, Word

GAP_S = 0.7
MAX_CHARS = 120
WORDLESS_MIN_S = 60.0
MIN_MAX_CHARS = 20
NO_WORDS = "(no words transcribed)"
TIME_EPS_S = 1e-6
MAX_TIME_S = 1e7


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
    long counts (within ``TIME_EPS_S``).
    """
    _positive("min_s", min_s)
    spans = sorted((w.start_s, w.end_s) for w in words)
    out: list[Stretch] = []
    cursor = start_s
    for lo, hi in spans:
        if cursor is not None and lo - cursor >= min_s - TIME_EPS_S:
            out.append(Stretch(cursor, lo))
        cursor = hi if cursor is None else max(cursor, hi)
    if end_s is not None and cursor is not None and end_s - cursor >= min_s - TIME_EPS_S:
        out.append(Stretch(cursor, end_s))
    return out


@dataclass(frozen=True)
class Line:
    """One output line: a spoken run, or a wordless stretch (``wordless``, text ``NO_WORDS``)."""

    start_s: float
    end_s: float
    text: str
    wordless: bool = False


def check_times(words: Sequence[Word]) -> None:
    """``ValueError`` for a word time that is not finite or is past ``MAX_TIME_S``."""
    for word in words:
        for t in (word.start_s, word.end_s):
            if not (math.isfinite(t) and abs(t) <= MAX_TIME_S):
                raise ValueError(
                    f"word {word.text[:40]!r} has time {t!r}, not a finite number of seconds "
                    f"up to {MAX_TIME_S:g}"
                )


def _clean(text: str) -> str:
    """Whitespace runs to one space; every other control character dropped."""
    joined = " ".join(text.split())
    return "".join(c for c in joined if unicodedata.category(c) != "Cc")


def text_lines(
    transcript: Transcript,
    *,
    gap_s: float = GAP_S,
    max_chars: int = MAX_CHARS,
    wordless_min_s: float = WORDLESS_MIN_S,
) -> list[Line]:
    """The lines before formatting (see the module docstring)."""
    _positive("gap_s", gap_s)
    _positive("wordless_min_s", wordless_min_s)
    if isinstance(max_chars, bool) or not (
        isinstance(max_chars, int) and max_chars >= MIN_MAX_CHARS
    ):
        raise ValueError(f"max_chars must be a whole number of at least {MIN_MAX_CHARS}")
    check_times(transcript.words)
    lines: list[Line] = []
    start = end = 0.0
    text = ""  # the open line's text; "" when no line is open
    covered = -math.inf  # the furthest end any word so far reached, across line breaks
    for word in sorted(transcript.words, key=lambda w: (w.start_s, w.end_s)):
        piece = _clean(word.text)
        if not piece:
            continue
        gap = word.start_s - covered
        if text:
            if gap >= wordless_min_s - TIME_EPS_S:
                lines.append(Line(start, end, text))
                lines.append(Line(covered, word.start_s, NO_WORDS, wordless=True))
                text = ""
            elif gap > gap_s + TIME_EPS_S or len(text) + 1 + len(piece) > max_chars:
                lines.append(Line(start, end, text))
                text = ""
        if text:
            text += " " + piece
            end = max(end, word.end_s)
        else:
            start, end, text = word.start_s, word.end_s, piece
        covered = max(covered, word.end_s)
    if text:
        lines.append(Line(start, end, text))
    return lines


def format_lines(lines: Sequence[Line]) -> str:
    """``[start-end] text`` per line, times zero-padded to one width; ``""`` for no lines."""
    if not lines:
        return ""
    width = max(len(f"{t:.1f}") for line in lines for t in (line.start_s, line.end_s))
    width = max(width, len("0000.0"))
    return "".join(
        f"[{line.start_s:0{width}.1f}-{line.end_s:0{width}.1f}] {line.text}\n" for line in lines
    )


def render_text(
    transcript: Transcript,
    *,
    gap_s: float = GAP_S,
    max_chars: int = MAX_CHARS,
    wordless_min_s: float = WORDLESS_MIN_S,
) -> str:
    """The transcript as text, one ``[start-end] words`` line per run; ``""`` with no words."""
    return format_lines(
        text_lines(transcript, gap_s=gap_s, max_chars=max_chars, wordless_min_s=wordless_min_s)
    )


__all__ = [
    "GAP_S",
    "MAX_CHARS",
    "MAX_TIME_S",
    "NO_WORDS",
    "TIME_EPS_S",
    "WORDLESS_MIN_S",
    "Line",
    "Stretch",
    "check_times",
    "format_lines",
    "render_text",
    "text_lines",
    "wordless_stretches",
]
