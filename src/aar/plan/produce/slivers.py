"""Transform 3, sliver fold: a piece shorter than ``min_shot_s`` merges into a neighbour.

Ported from CICGTD ``edl_v7.py`` (reaction flashes merge into a neighbour) with the R6 shortest
shot. Folding extends a neighbour over the sliver, so nothing moves in record order and the
plan stays chronological. Only a neighbour from the same keep range qualifies: the gap between
them is a trimmed pause (or nothing, at a camera change), never material a human cut, and a
trimmed pause holds no words. Of the two, the one that brings back less removed audio wins; on a
tie, the previous one (as ``edl_v7``). The merged piece must still be covered by the
neighbour's source. A sliver with no such neighbour is an error.
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
        options: list[tuple[int, int, str, int, Piece]] = []
        if index > 0 and out[index - 1].keep == sliver.keep:
            prev = out[index - 1]
            merged = Piece(prev.source, prev.rec_in, sliver.rec_out, prev.keep)
            if coverage[prev.source].covers(merged.rec_in, merged.rec_out):
                options.append((sliver.rec_in - prev.rec_out, 0, "previous", index - 1, merged))
        if index + 1 < len(out) and out[index + 1].keep == sliver.keep:
            nxt = out[index + 1]
            merged = Piece(nxt.source, sliver.rec_in, nxt.rec_out, nxt.keep)
            if coverage[nxt.source].covers(merged.rec_in, merged.rec_out):
                options.append((nxt.rec_in - sliver.rec_out, 1, "next", index + 1, merged))
        if not options:
            raise ProduceError(
                "unfoldable-sliver",
                f"the piece at recording {fps.seconds(sliver.rec_in):.3f}-"
                f"{fps.seconds(sliver.rec_out):.3f} s lasts {sliver.length} frames (under "
                f"{need}) and no neighbour of its keep range can take it",
                hint="lengthen that keep range, or schedule a camera that covers its neighbour",
            )
        cost, _, side, other, merged = min(options)
        lo, hi = sorted((index, other))
        out[lo : hi + 1] = [merged]
        out = coalesce(out)
        folds.append(Fold(sliver.source, sliver.rec_in, sliver.rec_out, side, cost))
