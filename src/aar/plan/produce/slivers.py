"""Transform 3, sliver fold: a piece shorter than ``min_shot_s`` merges into a neighbour.

Ported from CICGTD ``edl_v7.py`` (reaction flashes merge into a neighbour) with the R6 shortest
shot. Only a neighbour from the same keep range qualifies, so a fold never brings back material
a human cut. The gap between such neighbours is a trimmed pause (or nothing, at a camera change),
and a trimmed pause holds no word.

Across a camera change the sliver merges into the neighbour (it takes the neighbour's camera).
Across a trimmed pause, folding re-runs the trim: the sliver grows into the pause just far
enough to last ``min_shot_s`` and the rest of the pause stays trimmed. Only when the pause is too
short for that does the whole pause come back and the two pieces merge. Of the options, the one
that brings back the least removed audio wins; on a tie, the previous side (as ``edl_v7``). The
grown piece must still be covered by its source. A sliver with no such option is an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aar.plan.model import Fps
from aar.plan.produce.errors import ProduceError
from aar.plan.produce.timeline import Coverage, Piece, coalesce


@dataclass(frozen=True)
class Fold:
    source: str
    rec_in: int
    rec_out: int
    into: str  # "previous" or "next"
    readded_frames: int
    retrimmed: bool  # the sliver grew into a trimmed pause that stays trimmed


def _options(
    out: list[Piece], index: int, coverage: dict[str, Coverage], need: int
) -> list[tuple[int, int, str, bool, list[Piece], int, int]]:
    """(re-added frames, side order, side, retrimmed, replacement, first, last) per option."""
    sliver = out[index]
    short = need - sliver.length
    found = []
    if index > 0 and out[index - 1].keep == sliver.keep:
        prev = out[index - 1]
        gap = sliver.rec_in - prev.rec_out
        if 0 < short < gap:  # grow back into the trimmed pause before it
            grown = Piece(sliver.source, sliver.rec_in - short, sliver.rec_out, sliver.keep)
            if coverage[grown.source].covers(grown.rec_in, grown.rec_out):
                found.append((short, 0, "previous", True, [grown], index, index))
        merged = Piece(prev.source, prev.rec_in, sliver.rec_out, prev.keep)
        if coverage[prev.source].covers(merged.rec_in, merged.rec_out):
            found.append((gap, 0, "previous", False, [merged], index - 1, index))
    if index + 1 < len(out) and out[index + 1].keep == sliver.keep:
        nxt = out[index + 1]
        gap = nxt.rec_in - sliver.rec_out
        if 0 < short < gap:  # grow on into the trimmed pause after it
            grown = Piece(sliver.source, sliver.rec_in, sliver.rec_out + short, sliver.keep)
            if coverage[grown.source].covers(grown.rec_in, grown.rec_out):
                found.append((short, 1, "next", True, [grown], index, index))
        merged = Piece(nxt.source, sliver.rec_in, nxt.rec_out, nxt.keep)
        if coverage[nxt.source].covers(merged.rec_in, merged.rec_out):
            found.append((gap, 1, "next", False, [merged], index, index + 1))
    return found


def fold_slivers(
    pieces: Sequence[Piece], coverage: dict[str, Coverage], fps: Fps, *, need: int
) -> tuple[list[Piece], list[Fold]]:
    out = coalesce(pieces)
    folds: list[Fold] = []
    while True:
        index = next((i for i, p in enumerate(out) if p.length < need), None)
        if index is None:
            return out, folds
        sliver = out[index]
        options = _options(out, index, coverage, need)
        if not options:
            raise ProduceError(
                "unfoldable-sliver",
                f"the piece at recording {fps.seconds(sliver.rec_in):.3f}-"
                f"{fps.seconds(sliver.rec_out):.3f} s lasts {sliver.length} frames (under "
                f"{need}) and no neighbour of its keep range can take it",
                hint="lengthen that keep range, or schedule a camera that covers its neighbour",
            )
        cost, _, side, retrimmed, replacement, first, last = min(options, key=lambda o: o[:2])
        out[first : last + 1] = replacement
        out = coalesce(out)
        folds.append(Fold(sliver.source, sliver.rec_in, sliver.rec_out, side, cost, retrimmed))
