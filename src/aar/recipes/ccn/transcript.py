"""The ``transcript`` step of ``ccn-longform`` and its reuse of an earlier transcription.

**Reuse is bound to runner evidence, never to a fact's word.** The key
(:func:`transcript_cache_key`) is the sha256 of the media file's sha256, the model and every
setting that changes the words or the report, the recording and the offset. The
``transcript.<key>`` fact is only a hint: it names a transcript artifact. That artifact is reused
only when all of this holds:

- it was stored by an op node the runner created for a ``transcript`` step, in this project;
- that op's runner-recorded result (``recipe.result``, which only the engine writes) is ``done``,
  is not itself a reuse, and states **this** key, this media sha256 and these artifact ids;
- the transcript, its text and its report all read back from the store intact (the store checks
  each against its hash) and parse.

Anything else is a miss, and the file is transcribed again (which also repairs a rotted store
copy, since identical bytes are put again). So an agent that records the fact cannot point a key
at another file's transcript or at one made with other settings.

**One transcription per key at a time.** A lock file per key (:func:`~.scratch.key_lock`) guards
``<work>/aar-transcripts/<key>/``; a second worker waits for it (bounded by
:data:`LOCK_WAIT_S` and the step's time left), checking for the first worker's result, and
reuses it. The folder keeps the transcriber's checkpoint, so a killed worker's next attempt
resumes; it is removed after the artifacts are stored. Folders and lock files nobody holds are
swept at step start.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.analyze.common import offset_from_plan
from aar.analyze.errors import AnalyzeError
from aar.analyze.text import render_text
from aar.analyze.whisper import CCN_PRESET, MODEL_DIR_ENV, WhisperSettings, transcribe_file
from aar.host.transport import sha256_file
from aar.hub.artifacts import CorruptArtifact, InvalidArtifactId
from aar.hub.common import utcnow
from aar.hub.facts import facts_view
from aar.hub.graph import WorkGraph
from aar.media import timeout_for
from aar.plan.inputs import Transcript
from aar.plan.model import EditPlan
from aar.recipes.ccn import scratch
from aar.recipes.ccn.params import CcnLongformParams
from aar.recipes.ccn.scratch import SWEEP_MAX, key_lock, sweep_locks, work_root
from aar.recipes.context import StepContext, StepFailure, StepOutput
from aar.recipes.engine import op_lease_live
from aar.recipes.events import RUNNER, STEP_SCHEMA
from aar.recipes.state import OpAttempt, load_state

MAX_WORDLESS_LISTED = 20
#: How long a second worker waits for another's transcription of the same key.
LOCK_WAIT_S = 3 * 3600.0
#: Scratch of an interrupted transcription is kept this long for a resume.
STALE_S = 2 * 24 * 3600.0
AREA = "aar-transcripts"

Transcribe = Callable[..., dict[str, Any]]
_TRANSCRIBE: list[Transcribe] = [transcribe_file]


def set_transcriber(fn: Transcribe | None) -> None:
    """Use ``fn`` (``transcribe_file``'s signature) for the transcript step; None restores it."""
    _TRANSCRIBE[0] = fn or transcribe_file


class StepInterrupted(KeyboardInterrupt):
    """The worker is stopping before the step's work began. Raised past the worker (it records no
    result), so the claim's lease lapses and an idempotent step is claimed again later instead of
    using up an attempt."""


def check_cancelled(ctx: StepContext) -> None:
    if ctx.cancelled():
        raise StepInterrupted("the worker is stopping; this step is claimed again later")


def whisper_settings(params: CcnLongformParams) -> WhisperSettings:
    """The CCN preset with the params' model, threads and normalisation."""
    return replace(
        CCN_PRESET,
        model=params.whisper_model,
        cpu_threads=params.whisper_threads,
        normalize=params.normalize_audio,
    )


def transcript_cache_key(
    media_sha256: str, settings: WhisperSettings, recording: str, offset_s: float
) -> str:
    """The ``transcript.<key>`` key part: the sha256 of the media's hash, the model and every
    setting that changes the words or the report (not the thread count or the caps), the
    recording and the offset."""
    words = {
        **settings.decode_settings(),
        "no_speech_prob_max": settings.no_speech_prob_max,
        "avg_logprob_min": settings.avg_logprob_min,
        "compression_ratio_max": settings.compression_ratio_max,
        "wordless_min_s": settings.wordless_min_s,
    }
    body = {
        "media_sha256": media_sha256,
        "settings": words,
        "recording": recording,
        "offset_s": offset_s,
    }
    canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


@dataclass(frozen=True)
class Found:
    """A reusable transcription: its artifact ids, the transcript and the report."""

    transcript_id: str
    text_id: str
    report_id: str
    transcript: Transcript
    report: dict[str, Any]


def _producer(ctx: StepContext, artifact_id: str) -> OpAttempt | None:
    """The runner-created ``transcript`` op of this project that stored ``artifact_id``, folded
    from runner evidence; None for anything else."""
    meta = ctx.store.get_meta(artifact_id)
    if meta is None or meta.project_id != ctx.project_id or meta.producing_op is None:
        return None
    graph = WorkGraph(ctx.log, ctx.project_id)
    node = graph.node(meta.producing_op)
    if node is None or node.kind != "op" or graph.creator(node.id) != RUNNER:
        return None
    data = node.payload
    if not (
        isinstance(data, dict)
        and data.get("schema") == STEP_SCHEMA
        and data.get("step_id") == "transcript"
        and isinstance(data.get("recipe_node"), str)
    ):
        return None
    state = load_state(ctx.log, data["recipe_node"], project_id=ctx.project_id)
    op = next((o for o in state.ops if o.op_id == node.id), None)
    if op is not None and op.is_live and not op_lease_live(state, op, utcnow()):
        return replace(op, status="lapsed")  # a dead worker's claim: nothing will come
    return op


def producing_result(ctx: StepContext, artifact_id: str) -> dict[str, Any] | None:
    """The runner-recorded outputs of the ``transcript`` op that stored ``artifact_id`` in this
    project, when that op is done; else None."""
    op = _producer(ctx, artifact_id)
    if op is None or op.status != "done" or not isinstance(op.result_outputs, dict):
        return None
    return dict(op.result_outputs)


def result_pending(ctx: StepContext, key: str) -> bool:
    """Whether the ``transcript.<key>`` fact names artifacts of a transcript op that is still
    running under a live lease (its worker stored them and has not recorded its result yet)."""
    fact = facts_view(ctx.log, ctx.project_id).get(f"transcript.{key}")
    hint = fact.current.value.get("transcript") if fact is not None else None
    if not isinstance(hint, str):
        return False
    op = _producer(ctx, hint)
    return op is not None and op.is_live and op.status != "lapsed"


def _read_found(ctx: StepContext, transcript_id: str, text_id: str, report_id: str) -> Found:
    raw = ctx.get(transcript_id)  # each read is checked against the artifact's hash
    ctx.get(text_id).decode("utf-8")
    report = json.loads(ctx.get(report_id).decode("utf-8"))
    if not isinstance(report, dict):
        raise ValueError("the report is not a JSON object")
    return Found(transcript_id, text_id, report_id, Transcript.model_validate_json(raw), report)


def find_cached(ctx: StepContext, key: str, media_sha256: str) -> Found | None:
    """The reusable transcription of ``key`` (module docstring), or None."""
    fact = facts_view(ctx.log, ctx.project_id).get(f"transcript.{key}")
    if fact is None or not isinstance(fact.current.value, dict):
        return None
    hint = fact.current.value.get("transcript")
    if not isinstance(hint, str):
        return None
    evidence = producing_result(ctx, hint)
    if (
        evidence is None
        or evidence.get("cached") is not False
        or evidence.get("cache_key") != key
        or evidence.get("media_sha256") != media_sha256
        or evidence.get("transcript") != hint
    ):
        return None
    ids = [evidence.get(name) for name in ("transcript", "text", "report")]
    if not all(isinstance(i, str) for i in ids):
        return None
    try:
        return _read_found(ctx, *ids)
    except (
        FileNotFoundError,
        CorruptArtifact,
        InvalidArtifactId,
        UnicodeDecodeError,
        ValueError,  # JSON and pydantic validation errors too
        ValidationError,
    ):
        return None


def _wordless_summary(report: dict[str, Any]) -> dict[str, Any]:
    wordless = report.get("wordless") or {}
    stretches = list(wordless.get("stretches") or [])
    return {
        "total_s": wordless.get("total_s"),
        "file_duration_s": wordless.get("file_duration_s"),
        "longest": wordless.get("longest"),
        "count": len(stretches),
        "stretches": [
            {k: s.get(k) for k in ("start_s", "end_s", "duration_s", "level_dbfs")}
            for s in stretches[:MAX_WORDLESS_LISTED]
        ],
        "note": "no words is not silence: these stretches may hold quiet speech or laughter",
    }


def _sweep_scratch(base: Path) -> None:
    """Remove transcription folders nobody is using that are older than :data:`STALE_S`, and
    lock files nobody holds; bounded."""
    sweep_locks(base)
    if not base.is_dir():
        return
    now = time.time()
    with os.scandir(base) as entries:
        folders = [e for _, e in zip(range(SWEEP_MAX), entries, strict=False) if e.is_dir()]
    for entry in folders:
        lock = base / f"{entry.name}.lock"
        try:
            with key_lock(lock, wait_s=0):
                if now - entry.stat().st_mtime > STALE_S:
                    shutil.rmtree(entry.path, ignore_errors=True)
        except TimeoutError:
            continue  # in use


def _remaining(ctx: StepContext, cap: float) -> float:
    left = ctx.time_left()
    return max(1.0, min(cap, left)) if left == left else cap  # NaN-safe


def media_seconds(plan: EditPlan, index: int = 0) -> float | None:
    """A source's length in seconds from the stub (its frame count at the plan rate)."""
    source = plan.sources[index]
    if source.duration is None:
        return None
    return source.duration * plan.fps.den / plan.fps.num


def library_timeout(ctx: StepContext, duration_s: float | None) -> float:
    """The analysis libraries' own default timeout for ``duration_s`` of media (one decode),
    cut to the step's time left, so a call never outlives the step's deadline."""
    return _remaining(ctx, timeout_for(duration_s, per_media_s=0.5))


def _transcribe(
    ctx: StepContext,
    folder: Path,
    media: Path,
    settings: WhisperSettings,
    stub: EditPlan,
    reference: str,
) -> tuple[Path, dict[str, Any]]:
    """Run the transcriber into ``folder``, resuming a checkpoint an interrupted attempt left
    there (one made from other bytes or settings is discarded, never spliced)."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "transcript.json"
    timed = replace(settings, timeout_s=library_timeout(ctx, media_seconds(stub)))
    kwargs = {
        "settings": timed,
        "model_dir": os.environ.get(MODEL_DIR_ENV) or None,
        "offset": offset_from_plan(stub, reference),
        "recording": ctx.params.recording_id,
    }
    try:
        return out, _TRANSCRIBE[0](media, out, resume=True, **kwargs)
    except AnalyzeError as exc:
        if exc.type not in ("checkpoint-mismatch", "checkpoint-corrupt"):
            raise
    for leftover in folder.iterdir():
        if leftover.is_file():
            leftover.unlink()
    return out, _TRANSCRIBE[0](media, out, resume=False, **kwargs)


def _outputs(common: dict[str, Any], found: Found, cached: bool) -> StepOutput:
    return StepOutput(
        outputs={
            **common,
            "cached": cached,
            "transcript": found.transcript_id,
            "text": found.text_id,
            "report": found.report_id,
            "words": len(found.transcript.words),
            "wordless": _wordless_summary(found.report),
        },
        artifact_ids=(found.transcript_id, found.text_id, found.report_id),
    )


def transcript_step(ctx: StepContext, stub: EditPlan) -> StepOutput:
    """Transcribe the reference source, or reuse a transcription of the same key, store it with
    its readable text and report, and record the ``transcript.<key>`` fact."""
    params: CcnLongformParams = ctx.params
    reference = stub.sources[0]
    media = Path(reference.media)
    settings = whisper_settings(params)
    offset_s = offset_from_plan(stub, reference.id).offset_s
    base = work_root() / AREA
    _sweep_scratch(base)
    check_cancelled(ctx)
    try:
        media_sha256 = sha256_file(str(media)).lower()
    except OSError as exc:
        raise StepFailure(
            "transcript_unreadable", f"cannot read {str(media)!r}: {exc}", "check the file"
        ) from None
    key = transcript_cache_key(media_sha256, settings, params.recording_id, offset_s)
    common = {
        "media_sha256": media_sha256,
        "model": settings.model,
        "normalize": settings.normalize,
        "cache_key": key,
        "recording": params.recording_id,
    }
    found = find_cached(ctx, key, media_sha256)
    if found is not None:
        return _outputs(common, found, cached=True)
    check_cancelled(ctx)
    hit: list[Found] = []

    def other_worker_done() -> bool:
        check_cancelled(ctx)
        again = find_cached(ctx, key, media_sha256)
        if again is not None:
            hit.append(again)
        return again is not None

    try:
        with key_lock(
            base / f"{key}.lock",
            wait_s=_remaining(ctx, LOCK_WAIT_S),
            on_wait=other_worker_done,
        ) as locked:
            if not locked:
                return _outputs(common, hit[0], cached=True)
            # A holder releases the lock just before the worker records its result: wait for
            # that result (bounded), then look again before transcribing.
            deadline = time.monotonic() + _remaining(ctx, LOCK_WAIT_S)
            while result_pending(ctx, key) and time.monotonic() < deadline:
                check_cancelled(ctx)
                time.sleep(scratch.LOCK_POLL_S)
            again = find_cached(ctx, key, media_sha256)
            if again is not None:
                return _outputs(common, again, cached=True)
            made = _run(ctx, base / key, media, settings, stub, reference.id, key, common)
            return _outputs(common, made, cached=False)
    except TimeoutError:
        raise StepFailure(
            "transcript_busy",
            "another worker has been transcribing this file with these settings for longer "
            "than this step may wait",
            "retry the step when that run finishes (its transcript is then reused)",
        ) from None


def _run(
    ctx: StepContext,
    folder: Path,
    media: Path,
    settings: WhisperSettings,
    stub: EditPlan,
    reference: str,
    key: str,
    common: dict[str, Any],
) -> Found:
    try:
        try:
            out, report = _transcribe(ctx, folder, media, settings, stub, reference)
        except AnalyzeError as exc:
            raise StepFailure(f"transcript_{exc.type}", str(exc), exc.hint) from None
        raw = out.read_bytes()
        try:
            transcript = Transcript.model_validate_json(raw)
            text = render_text(transcript, wordless_min_s=settings.wordless_min_s)
        except ValueError as exc:  # pydantic's ValidationError and a time render_text refuses
            raise StepFailure(
                "transcript_invalid",
                f"the transcript cannot be used: {str(exc)[:300]}",
                "this is a transcriber bug; report it",
            ) from None
        transcript_id = ctx.put(raw, mime="application/json", meta={"role": "transcript"})
        text_id = ctx.put(text.encode("utf-8"), mime="text/plain", meta={"role": "transcript-text"})
        report_id = ctx.put(
            json.dumps(report, indent=1, sort_keys=True).encode("utf-8"),
            mime="application/json",
            meta={"role": "transcript-report"},
        )
        ctx.record_fact(
            f"transcript.{key}",
            {
                "media_sha256": common["media_sha256"],
                "model": settings.model,
                "recording": common["recording"],
                "transcript": transcript_id,
                "text": text_id,
                "report": report_id,
            },
        )
        return Found(transcript_id, text_id, report_id, transcript, report)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


__all__ = [
    "LOCK_WAIT_S",
    "StepInterrupted",
    "check_cancelled",
    "find_cached",
    "library_timeout",
    "media_seconds",
    "set_transcriber",
    "transcript_cache_key",
    "transcript_step",
    "whisper_settings",
]
