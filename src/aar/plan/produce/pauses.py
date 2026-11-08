"""Transform 1, pause trim: long pauses inside a keep range shrink to a beat.

Ported from Nicole ``trim.py`` (MINGAP 0.9 s, BEAT 0.45 s), the supoclip planner's big-silence
squeeze (-42 dB, 0.90 s, keep 0.25 + 0.20 s) and CICGTD ``edl_v7.py`` (PRE 0.25, POST 0.20,
never swallowing a transcribed word). A pause is a run of source audio below the venue's
silence threshold (``audio.pause_runs``), never a transcript gap: Whisper's word times are off
by 0.1-0.2 s at joins. But a transcribed word inside a quiet run is trusted: the run is split
around it, so a trim never removes a word. Every word counts, a zero-length one included: it
occupies at least ``min_word_ms`` around its time (``timeline.Words``).

A pause qualifies when it lies strictly inside the keep range (silence at a range's edge is the
human's choice) and lasts at least ``min_pause_s``. It keeps ``beat_lead_s`` after the
outgoing speech and ``beat_s - beat_lead_s`` before the incoming speech; the rest is removed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aar.plan.model import Fps
from aar.plan.produce.timeline import Words

#: A pause measured as exactly ``min_pause_s`` qualifies, float rounding notwithstanding.
_EPS = 1e-9


@dataclass(frozen=True)
class Trim:
    keep: int
    pause_s: tuple[float, float]  # the quiet stretch, recording seconds
    out_frame: int  # the kept audio before the pause ends here (exclusive)
    in_frame: int  # ... and resumes here

    @property
    def removed_frames(self) -> int:
        return self.in_frame - self.out_frame


def _sub_pauses(run: tuple[float, float], words: Words) -> list[tuple[float, float]]:
    """``run`` less the spans the transcribed words occupy (``min_word_ms`` at least)."""
    start, end = run
    pieces: list[tuple[float, float]] = []
    cursor = start
    for lo, hi in words.spans(start, end):
        if lo > cursor:
            pieces.append((cursor, min(lo, end)))
        cursor = max(cursor, hi)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def find_trims(
    keep: int,
    start: int,
    end: int,
    runs: Sequence[tuple[float, float]],
    words: Words,
    fps: Fps,
    *,
    min_pause_s: float,
    beat_s: float,
    beat_lead_s: float,
) -> list[Trim]:
    """The trims inside keep range ``[start, end)`` (frames) from its quiet ``runs`` (seconds)."""
    lo_s, hi_s = fps.seconds(start), fps.seconds(end)
    tail = beat_s - beat_lead_s
    trims: list[Trim] = []
    for run in runs:
        if not (run[0] > lo_s and run[1] < hi_s):
            continue
        for a, b in _sub_pauses(run, words):
            if b - a < min_pause_s - _EPS:
                continue
            out_f, in_f = fps.frames(a + beat_lead_s), fps.frames(b - tail)
            if not start < out_f < in_f < end:
                continue
            if words.inside(out_f) or words.inside(in_f):
                continue
            if words.speech_between(fps.seconds(out_f), fps.seconds(in_f)):
                continue
            trims.append(Trim(keep, (a, b), out_f, in_f))
    return sorted(trims, key=lambda t: t.out_frame)


def kept_spans(start: int, end: int, trims: Sequence[Trim]) -> list[tuple[int, int]]:
    """``[start, end)`` less the removed part of every trim."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for trim in trims:
        spans.append((cursor, trim.out_frame))
        cursor = trim.in_frame
    spans.append((cursor, end))
    return spans
