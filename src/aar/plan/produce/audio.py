"""The producer's only media access: probing sources, the pause scan and the join judge.

Pauses come from the source audio envelope, decoded by ffmpeg (``aar.media.read_pcm``) and
scanned by ``aar.media.quiet_runs``: 10 ms RMS windows of the loudest channel, below the venue's
silence threshold. The decode runs ``scan_chunk_s`` at a time at ``scan_rate_hz`` so memory stays
bounded; runs that meet at a chunk boundary are joined. The join judge is QA's join-quiet test
(``aar.qa.scan_join_quiet``), with QA's window, rate and thresholds, plus the out-side test of a
human cut on the same envelope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aar.media import (
    MediaError,
    MediaInputError,
    MediaToolMissing,
    probe,
    quiet_runs,
    read_pcm,
    timeout_for,
)
from aar.plan.model import EditPlan, Fps
from aar.plan.produce.errors import ProduceError, ProduceInputError
from aar.plan.produce.timeline import Piece
from aar.plan.profile import QATargets
from aar.qa.check import join_quiet_window, resolve_media, scan_join_quiet

#: Float slack when comparing a run's edges with a required stretch (seconds).
_EPS = 1e-6


@dataclass(frozen=True)
class SourceAudio:
    path: Path
    channels: int
    duration_s: float | None  # of the audio stream, else the container


def open_audio(
    plan: EditPlan, source_ids: Sequence[str], media_root: Path | None
) -> dict[str, SourceAudio]:
    """Probe every source the cut uses; each must be a local file with an audio stream."""
    sources = plan.source_map()
    out: dict[str, SourceAudio] = {}
    for source_id in source_ids:
        source = sources[source_id]
        path = resolve_media(source.media, media_root)
        if path is None:
            raise ProduceInputError(
                "source-unreadable",
                f"source {source_id!r} media {source.media!r} is not a local path",
            )
        try:
            info = probe(path)
        except MediaToolMissing as exc:
            raise ProduceInputError("tool-missing", exc.summary()) from None
        except MediaError as exc:
            raise ProduceInputError(
                "source-unreadable", f"source {source_id!r}: {exc.summary()}"
            ) from None
        if info.audio is None:
            raise ProduceInputError(
                "source-unreadable", f"source {source_id!r} has no audio stream"
            )
        duration = info.audio.duration_s or info.duration_s
        out[source_id] = SourceAudio(path, info.audio.channels or 1, duration)
    return out


def _media_failure(source_id: str, exc: MediaError) -> ProduceError:
    if isinstance(exc, MediaToolMissing):
        return ProduceInputError("tool-missing", exc.summary())
    if isinstance(exc, MediaInputError):
        return ProduceInputError("source-unreadable", f"source {source_id!r}: {exc.summary()}")
    return ProduceError("media-error", f"source {source_id!r}: {exc.summary()}")


def quiet_spans(
    audio: SourceAudio,
    source_id: str,
    start_s: float,
    end_s: float,
    *,
    threshold_db: float,
    window_s: float,
    rate: int,
    chunk_s: float,
) -> list[tuple[float, float]]:
    """Quiet runs of the source between file times ``start_s`` and ``end_s``, in file seconds,
    chunk by chunk: a run that crosses a chunk boundary comes back as two touching runs, which
    ``pause_runs`` joins."""
    runs: list[tuple[float, float]] = []
    t = max(0.0, start_s)
    while t < end_s - 2.0 / rate:
        length = min(chunk_s, end_s - t)
        try:
            pcm = read_pcm(
                audio.path,
                start_s=t,
                duration_s=length,
                channels=audio.channels,
                rate=rate,
                timeout_s=timeout_for(length, per_media_s=1.0),
            )
        except MediaError as exc:
            raise _media_failure(source_id, exc) from None
        scan = quiet_runs(pcm, threshold_dbfs=threshold_db, window_s=window_s)
        runs += [(run.start_s, run.end_s) for run in scan.runs]
        t += length
    return runs


@dataclass
class ScanStats:
    """How much of a source's audio the pause scan read, and how many pauses it found there."""

    scanned_s: float = 0.0
    pauses: int = 0


def pause_runs(
    parts: Sequence[Piece],
    audio: dict[str, SourceAudio],
    origins: dict[str, int],
    fps: Fps,
    *,
    threshold_db: float,
    window_s: float,
    rate: int,
    chunk_s: float,
    stats: dict[str, ScanStats] | None = None,
) -> list[tuple[float, float]]:
    """Quiet runs over consecutive ``parts`` (each scanned in its own source's audio), in
    recording seconds; runs that meet at a scan chunk's edge or a camera change are joined.

    ``stats`` (when given) gains, per source, the seconds scanned and the quiet runs (any length)
    that lie strictly inside a part. Speech at the right threshold has many, if only between
    words; none means silent audio (one run covering every part) or a wrong threshold.
    """
    runs: list[tuple[float, float]] = []
    join = 2.0 / rate
    for part in parts:
        origin_s = fps.seconds(origins[part.source])
        lo, hi = fps.seconds(part.rec_in) - origin_s, fps.seconds(part.rec_out) - origin_s
        found = quiet_spans(
            audio[part.source],
            part.source,
            lo,
            hi,
            threshold_db=threshold_db,
            window_s=window_s,
            rate=rate,
            chunk_s=chunk_s,
        )
        merged: list[tuple[float, float]] = []
        for a, b in found:
            if merged and a <= merged[-1][1] + join:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        if stats is not None:
            entry = stats.setdefault(part.source, ScanStats())
            entry.scanned_s += hi - lo
            entry.pauses += sum(1 for a, b in merged if a > lo + join and b < hi - join)
        for a, b in ((a + origin_s, b + origin_s) for a, b in merged):
            if runs and a <= runs[-1][1] + join:
                runs[-1] = (runs[-1][0], max(runs[-1][1], b))
            else:
                runs.append((a, b))
    return runs


class JoinJudge:
    """The audio tests of a cut on the real sources (``joins.Judge``), cached.

    ``in_quiet`` is QA's join-quiet test (``aar.qa.scan_join_quiet``, QA's window, rate and
    thresholds), plus, with ``handle`` > 0, the crossfade's incoming handle inside the same
    quiet run. ``out_quiet`` reads the outgoing source around src_out and needs one quiet run
    (QA's threshold and window) from ``out_quiet_preroll_ms`` before src_out to ``handle`` frames
    after it.
    """

    def __init__(
        self,
        audio: dict[str, SourceAudio],
        qa: QATargets,
        fps: Fps,
        out_preroll_ms: float = 20.0,
    ) -> None:
        self.audio = audio
        self.qa = qa
        self.fps = fps
        self.out_preroll_ms = out_preroll_ms
        self.cache: dict[tuple[str, str, int, int], bool] = {}

    def in_quiet(self, source: str, src_in: int, *, handle: int) -> bool:
        key = ("in", source, src_in, handle)
        if key not in self.cache:
            self.cache[key] = self._in(source, src_in, handle)
        return self.cache[key]

    def out_quiet(self, source: str, src_out: int, *, handle: int) -> bool:
        key = ("out", source, src_out, handle)
        if key not in self.cache:
            self.cache[key] = self._out(source, src_out, handle)
        return self.cache[key]

    def _in(self, source_id: str, src_in: int, handle: int) -> bool:
        audio = self.audio[source_id]
        at = self.fps.seconds(src_in)
        window = join_quiet_window(at, self.qa.join_quiet_search_s, audio.duration_s)
        if window[1] <= window[0]:
            return False
        try:
            measured = scan_join_quiet(
                audio.path,
                channels=audio.channels,
                src_in_s=at,
                window=window,
                qa=self.qa,
                timeout_s=timeout_for(0.0, per_media_s=1.0),
            )
        except MediaError as exc:
            raise _media_failure(source_id, exc) from None
        judged = measured.judged
        if judged.status != "pass":
            return False
        assert judged.at_cut is not None
        return handle == 0 or judged.at_cut.start_s <= at - self.fps.seconds(handle) + _EPS

    def _out(self, source_id: str, src_out: int, handle: int) -> bool:
        audio = self.audio[source_id]
        at = self.fps.seconds(src_out)
        lo, hi = at - self.out_preroll_ms / 1000, at + self.fps.seconds(handle)
        if lo < 0 or (audio.duration_s is not None and hi > audio.duration_s + _EPS):
            return False
        pad = self.qa.join_quiet_window_ms / 1000
        start = max(0.0, lo - pad)
        end = hi + pad if audio.duration_s is None else min(hi + pad, audio.duration_s)
        try:
            pcm = read_pcm(
                audio.path,
                start_s=start,
                duration_s=end - start,
                channels=audio.channels,
                timeout_s=timeout_for(0.0, per_media_s=1.0),
            )
        except MediaError as exc:
            raise _media_failure(source_id, exc) from None
        scan = quiet_runs(
            pcm, threshold_dbfs=self.qa.join_quiet_db, window_s=self.qa.join_quiet_window_ms / 1000
        )
        return any(r.start_s <= lo + _EPS and r.end_s >= hi - _EPS for r in scan.runs)
