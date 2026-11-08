"""The producer's only media access: probing sources, the pause scan and the join judge.

Pauses come from the source audio envelope, decoded by ffmpeg (``aar.media.read_pcm``) and
scanned by ``aar.media.quiet_runs``: 10 ms RMS windows of the loudest channel, below the venue's
silence threshold. The decode runs ``scan_chunk_s`` at a time at ``scan_rate_hz`` so memory stays
bounded; runs that meet at a chunk boundary are joined. The join judge is QA's join-quiet test
(``aar.qa.scan_join_quiet``), with QA's window, rate and thresholds.
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
) -> list[tuple[float, float]]:
    """Quiet runs over consecutive ``parts`` (each scanned in its own source's audio), in
    recording seconds; runs that meet at a scan chunk's edge or a camera change are joined."""
    runs: list[tuple[float, float]] = []
    join = 2.0 / rate
    for part in parts:
        origin_s = fps.seconds(origins[part.source])
        found = quiet_spans(
            audio[part.source],
            part.source,
            fps.seconds(part.rec_in) - origin_s,
            fps.seconds(part.rec_out) - origin_s,
            threshold_db=threshold_db,
            window_s=window_s,
            rate=rate,
            chunk_s=chunk_s,
        )
        for a, b in ((a + origin_s, b + origin_s) for a, b in found):
            if runs and a <= runs[-1][1] + join:
                runs[-1] = (runs[-1][0], max(runs[-1][1], b))
            else:
                runs.append((a, b))
    return runs


class JoinJudge:
    """``judge(source id, src_in frame)``: QA's join-quiet verdict for a cut starting there."""

    def __init__(self, audio: dict[str, SourceAudio], qa: QATargets, fps: Fps) -> None:
        self.audio = audio
        self.qa = qa
        self.fps = fps
        self.cache: dict[tuple[str, int], bool] = {}

    def __call__(self, source_id: str, src_in: int) -> bool:
        key = (source_id, src_in)
        if key not in self.cache:
            self.cache[key] = self._judge(source_id, src_in)
        return self.cache[key]

    def _judge(self, source_id: str, src_in: int) -> bool:
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
        return measured.judged.status == "pass"
