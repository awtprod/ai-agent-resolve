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

**A reference camera of several files (S24a).** Every part is transcribed at its own offset
(its ``recording_origin``), so its words are in recording time, under its own key, lock and
fact; a part's reuse evidence is its row in the op's ``parts`` output. The parts are merged
into one transcript and one readable text for the recording (:func:`merge_parts`), with a line
where two files meet that says how long the gap with no file is. A part transcribed by an
attempt is stored only when every part is done; until then its finished transcription waits in
its key's folder (:data:`DONE_REPORT`), so an attempt that stops after some parts resumes with
them, and every stored part's producer is an op whose result names it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from aar.analyze.common import offset_from_plan
from aar.analyze.errors import AnalyzeError
from aar.analyze.text import TIME_EPS_S, Line, format_lines, render_text, text_lines
from aar.analyze.whisper import CCN_PRESET, MODEL_DIR_ENV, WhisperSettings, transcribe_file
from aar.host.transport import sha256_file
from aar.hub.artifacts import CorruptArtifact, InvalidArtifactId
from aar.hub.common import utcnow
from aar.hub.facts import facts_view
from aar.hub.graph import WorkGraph
from aar.media import timeout_for
from aar.plan.inputs import Transcript, Word
from aar.plan.model import EditPlan, Source
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
#: In a key's folder: the report of a transcription that finished there but is not stored yet.
DONE_REPORT = "report.done.json"

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
    if evidence is None:
        return None
    # The op's own result, or one part of a reference camera of several files (S24a).
    rows = [evidence, *(p for p in evidence.get("parts") or [] if isinstance(p, dict))]
    row = next(
        (
            r
            for r in rows
            if r.get("cached") is False
            and r.get("cache_key") == key
            and r.get("media_sha256") == media_sha256
            and r.get("transcript") == hint
        ),
        None,
    )
    if row is None:
        return None
    ids = [row.get(name) for name in ("transcript", "text", "report")]
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
    there (one made from other bytes or settings is discarded, never spliced). A transcription
    an earlier attempt finished in ``folder`` (its :data:`DONE_REPORT` beside the transcript, for
    a part of a camera of several files whose step stopped before every part was stored) is
    taken as it is: the folder is the key's, so it is this file with these settings."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "transcript.json"
    done = folder / DONE_REPORT
    if done.is_file() and out.is_file():
        try:
            report = json.loads(done.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            report = None
        if isinstance(report, dict):
            return out, report
        done.unlink(missing_ok=True)
    timed = replace(settings, timeout_s=library_timeout(ctx, _seconds(stub, reference)))
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


def _seconds(stub: EditPlan, source_id: str) -> float | None:
    """The length of one stub source in seconds (each transcription's timeout is its own file's)."""
    index = next(n for n, s in enumerate(stub.sources) if s.id == source_id)
    return media_seconds(stub, index)


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


def reference_parts(stub: EditPlan) -> list[Source]:
    """The reference camera's files in part order: the first source alone, or, when it is a part
    (S24a), every part of its camera and recording."""
    first = stub.sources[0]
    if first.part is None:
        return [first]
    return sorted(
        (
            s
            for s in stub.sources
            if s.part is not None
            and s.camera == first.camera
            and s.recording_id == first.recording_id
        ),
        key=lambda s: s.part or 0,
    )


@dataclass(frozen=True)
class _Key:
    """One file to transcribe: its media, hash, cache key and the outputs every result shares."""

    source: Source
    media: Path
    media_sha256: str
    key: str
    offset_s: float
    common: dict[str, Any]


def _identify(ctx: StepContext, stub: EditPlan, source: Source, settings: WhisperSettings) -> _Key:
    params: CcnLongformParams = ctx.params
    media = Path(source.media)
    offset_s = offset_from_plan(stub, source.id).offset_s
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
    return _Key(source, media, media_sha256, key, offset_s, common)


T = TypeVar("T")


def _locked(
    ctx: StepContext, base: Path, file: _Key, make: Callable[[], T]
) -> tuple[Found | T, bool]:
    """A reusable transcription of ``file`` (and True), else ``make()``'s (and False) under the
    key's lock: a second worker waits for the first's result and reuses it."""
    key, media_sha256 = file.key, file.media_sha256
    found = find_cached(ctx, key, media_sha256)
    if found is not None:
        return found, True
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
                return hit[0], True
            # A holder releases the lock just before the worker records its result: wait for
            # that result (bounded), then look again before transcribing.
            deadline = time.monotonic() + _remaining(ctx, LOCK_WAIT_S)
            while result_pending(ctx, key) and time.monotonic() < deadline:
                check_cancelled(ctx)
                time.sleep(scratch.LOCK_POLL_S)
            again = find_cached(ctx, key, media_sha256)
            if again is not None:
                return again, True
            return make(), False
    except TimeoutError:
        raise StepFailure(
            "transcript_busy",
            "another worker has been transcribing this file with these settings for longer "
            "than this step may wait",
            "retry the step when that run finishes (its transcript is then reused)",
        ) from None


def transcript_step(ctx: StepContext, stub: EditPlan) -> StepOutput:
    """Transcribe the reference source, or reuse a transcription of the same key, store it with
    its readable text and report, and record the ``transcript.<key>`` fact. A reference camera
    of several files (S24a) is :func:`_parts_step`."""
    params: CcnLongformParams = ctx.params
    settings = whisper_settings(params)
    base = work_root() / AREA
    _sweep_scratch(base)
    check_cancelled(ctx)
    parts = reference_parts(stub)
    if len(parts) > 1:
        return _parts_step(ctx, stub, parts, settings, base)
    file = _identify(ctx, stub, parts[0], settings)

    def make() -> Found:
        return _run(ctx, base / file.key, file, settings, stub)

    found, cached = _locked(ctx, base, file, make)
    return _outputs(file.common, found, cached)


@dataclass(frozen=True)
class _Made:
    """A finished transcription not stored yet: its bytes, model, readable text and report."""

    raw: bytes
    transcript: Transcript
    text: str
    report: dict[str, Any]


def _make(
    ctx: StepContext, folder: Path, file: _Key, settings: WhisperSettings, stub: EditPlan
) -> _Made:
    """Transcribe ``file`` into ``folder`` (resuming what is there), check the result and mark
    the folder done (:data:`DONE_REPORT`)."""
    try:
        out, report = _transcribe(ctx, folder, file.media, settings, stub, file.source.id)
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
    marker = folder / f"{DONE_REPORT}.tmp"
    marker.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    os.replace(marker, folder / DONE_REPORT)
    return _Made(raw, transcript, text, report)


def _store(ctx: StepContext, file: _Key, made: _Made, settings: WhisperSettings) -> Found:
    """Store a transcription's three artifacts and record its ``transcript.<key>`` fact."""
    transcript_id = ctx.put(made.raw, mime="application/json", meta={"role": "transcript"})
    text_id = ctx.put(
        made.text.encode("utf-8"), mime="text/plain", meta={"role": "transcript-text"}
    )
    report_id = ctx.put(
        json.dumps(made.report, indent=1, sort_keys=True).encode("utf-8"),
        mime="application/json",
        meta={"role": "transcript-report"},
    )
    ctx.record_fact(
        f"transcript.{file.key}",
        {
            "media_sha256": file.media_sha256,
            "model": settings.model,
            "recording": file.common["recording"],
            "transcript": transcript_id,
            "text": text_id,
            "report": report_id,
        },
    )
    return Found(transcript_id, text_id, report_id, made.transcript, made.report)


def _run(
    ctx: StepContext, folder: Path, file: _Key, settings: WhisperSettings, stub: EditPlan
) -> Found:
    try:
        return _store(ctx, file, _make(ctx, folder, file, settings, stub), settings)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# -- a reference camera of several files (S24a) ------------------------------------------------


@dataclass(frozen=True)
class PartText:
    """One part's transcription for :func:`merge_parts`: the part's source, where it lies on the
    recording clock (seconds) and its transcript and report (words in recording time)."""

    source: Source
    start_s: float
    end_s: float
    transcript: Transcript
    report: dict[str, Any]


def gap_line(prev: PartText, cur: PartText) -> Line:
    """The line :func:`merge_parts` puts between two parts in the readable text."""
    gap = cur.start_s - prev.end_s
    if gap > TIME_EPS_S:
        parts = f"part {prev.source.part} and part {cur.source.part}"
        text = f"-- gap {gap:.1f} s (no file) between {parts} --"
    else:
        text = f"-- part {cur.source.part} begins (no gap) --"
    return Line(prev.end_s, max(prev.end_s, cur.start_s), text, wordless=True)


def merge_parts(
    parts: Sequence[PartText], recording: str | None, *, wordless_min_s: float
) -> tuple[Transcript, str, dict[str, Any]]:
    """One transcript, one readable text and one report for the recording from its parts' (in
    part order). Each part's words are kept to the part's own span of the recording clock (a word
    the transcriber placed past the file's last frame is cut back to it, one wholly outside is
    dropped; where two parts overlap, the span is the earlier part's). The text renders each
    part's lines and puts a :func:`gap_line` between parts, so the reader sees where the files
    meet and how long each gap with no file is. The report lists each part's own report and the
    parts' wordless stretches together."""
    words: list[Word] = []
    lines: list[Line] = []
    stretches: list[dict[str, Any]] = []
    total = duration = 0.0
    for n, part in enumerate(parts):
        # where two files overlap by a frame or two, the earlier one's words stand (produce
        # plays the earlier file there)
        start = max(part.start_s, parts[n - 1].end_s) if n else part.start_s
        mine = [
            w.model_copy(
                update={"start_s": max(w.start_s, start), "end_s": min(w.end_s, part.end_s)}
            )
            for w in part.transcript.words
            if w.end_s > start and w.start_s < part.end_s
        ]
        words += mine
        if n:
            lines.append(gap_line(parts[n - 1], part))
        lines += text_lines(Transcript(words=mine), wordless_min_s=wordless_min_s)
        wordless = part.report.get("wordless") or {}
        stretches += list(wordless.get("stretches") or [])
        total += float(wordless.get("total_s") or 0.0)
        duration += float(wordless.get("file_duration_s") or 0.0)
    longest = max(stretches, key=lambda s: s.get("duration_s") or 0.0, default=None)
    report = {
        "parts": [
            {
                "source": part.source.id,
                "part": part.source.part,
                "start_s": round(part.start_s, 3),
                "end_s": round(part.end_s, 3),
                "report": part.report,
            }
            for part in parts
        ],
        "wordless": {
            "total_s": round(total, 3),
            "file_duration_s": round(duration, 3),
            "longest": longest,
            "stretches": stretches,
        },
    }
    return Transcript(recording=recording, words=words), format_lines(lines), report


def _span(stub: EditPlan, source: Source) -> tuple[float, float]:
    start = stub.fps.seconds(source.recording_origin)
    frames = source.duration or 0
    return start, stub.fps.seconds(source.recording_origin + frames)


def _parts_step(
    ctx: StepContext,
    stub: EditPlan,
    parts: Sequence[Source],
    settings: WhisperSettings,
    base: Path,
) -> StepOutput:
    """Every part of the reference camera transcribed at its own offset (or reused by its own
    key), then merged (:func:`merge_parts`). A part transcribed here is stored only once every
    part is done: until then it waits in its key's folder (:data:`DONE_REPORT`), so an attempt
    that stops after some parts resumes with them, and the artifacts are stored by the attempt
    whose result says so (the reuse evidence)."""
    params: CcnLongformParams = ctx.params
    files: list[_Key] = []
    results: list[tuple[Found | _Made, bool]] = []
    for source in parts:
        check_cancelled(ctx)
        file = _identify(ctx, stub, source, settings)

        def make(file: _Key = file) -> _Made:
            try:
                return _make(ctx, base / file.key, file, settings, stub)
            except BaseException:
                shutil.rmtree(base / file.key, ignore_errors=True)
                raise

        files.append(file)
        results.append(_locked(ctx, base, file, make))
    check_cancelled(ctx)
    stored: list[Found] = []
    for file, (result, _) in zip(files, results, strict=True):
        stored.append(_store(ctx, file, result, settings) if isinstance(result, _Made) else result)
    texts = [
        PartText(file.source, *_span(stub, file.source), found.transcript, found.report)
        for file, found in zip(files, stored, strict=True)
    ]
    transcript, text, report = merge_parts(
        texts, params.recording_id, wordless_min_s=settings.wordless_min_s
    )
    transcript_id = ctx.put(
        transcript.model_dump_json().encode("utf-8"),
        mime="application/json",
        meta={"role": "transcript"},
    )
    text_id = ctx.put(text.encode("utf-8"), mime="text/plain", meta={"role": "transcript-text"})
    report_id = ctx.put(
        json.dumps(report, indent=1, sort_keys=True).encode("utf-8"),
        mime="application/json",
        meta={"role": "transcript-report"},
    )
    for file, (result, _) in zip(files, results, strict=True):
        if isinstance(result, _Made):
            _drop_done(base, file.key)
    rows = [
        {
            "source": file.source.id,
            "part": file.source.part,
            "offset_s": file.offset_s,
            "media_sha256": file.media_sha256,
            "cache_key": file.key,
            "cached": cached,
            "transcript": found.transcript_id,
            "text": found.text_id,
            "report": found.report_id,
            "words": len(found.transcript.words),
        }
        for file, found, (_, cached) in zip(files, stored, results, strict=True)
    ]
    return StepOutput(
        outputs={
            "model": settings.model,
            "normalize": settings.normalize,
            "recording": params.recording_id,
            "cached": all(cached for _, cached in results),
            "transcript": transcript_id,
            "text": text_id,
            "report": report_id,
            "words": len(transcript.words),
            "wordless": _wordless_summary(report),
            "parts": rows,
        },
        artifact_ids=(
            transcript_id,
            text_id,
            report_id,
            *(i for f in stored for i in (f.transcript_id, f.text_id, f.report_id)),
        ),
    )


def _drop_done(base: Path, key: str) -> None:
    """Remove a stored part's folder, unless another worker holds its key now (the sweep does it
    later)."""
    try:
        with key_lock(base / f"{key}.lock", wait_s=0):
            shutil.rmtree(base / key, ignore_errors=True)
    except TimeoutError:
        return


__all__ = [
    "DONE_REPORT",
    "LOCK_WAIT_S",
    "PartText",
    "StepInterrupted",
    "check_cancelled",
    "find_cached",
    "gap_line",
    "library_timeout",
    "media_seconds",
    "merge_parts",
    "reference_parts",
    "set_transcriber",
    "transcript_cache_key",
    "transcript_step",
    "whisper_settings",
]
