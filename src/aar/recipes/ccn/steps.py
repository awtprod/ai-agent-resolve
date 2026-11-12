"""The cold steps of ``ccn-longform``: each calls the library the CLI calls, never the CLI.

Every step reads its inputs from the params and the earlier steps' outputs (artifact ids), keeps
its scratch under ``AAR_WORK_DIR`` (never the system temp dir, never the artifact store's disk),
removes that scratch on success and failure, stores what it makes as artifacts, and fails with a
typed :class:`~aar.recipes.context.StepFailure` (a stable code and a fix hint). The transcriber and
the laughs analysis are injectable (:func:`set_transcriber`, :func:`set_laughs_runner`) so tests
need no model and no paid API.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.analyze.common import offset_from_plan
from aar.analyze.errors import AnalyzeError
from aar.analyze.laughs import analyse_laughs
from aar.analyze.sync import SyncSettings, sync_files
from aar.analyze.text import render_text
from aar.analyze.whisper import CCN_PRESET, MODEL_DIR_ENV, WhisperSettings, transcribe_file
from aar.host.transport import sha256_file
from aar.hub.facts import facts_view
from aar.hub.graph import WorkGraph
from aar.node.jobs.file_lock import FileLockUnavailable, acquire_file_lock, release_file_lock
from aar.plan.export import ExportError, export_fcpxml, media_map_from_mapping
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.model import EditPlan
from aar.plan.profile import Profile
from aar.plan.render import RenderError, RenderTerminated, render_plan
from aar.plan.stub import StubError, StubSource, build_stub
from aar.qa.report import QAReport
from aar.recipes.ccn.params import CcnLongformParams
from aar.recipes.context import StepContext, StepFailure, StepOutput
from aar.recipes.events import RUNNER, STEP_SCHEMA
from aar.recipes.gates import qa_gate
from aar.recipes.state import load_state

WORK_DIR_ENV = "AAR_WORK_DIR"
EXPORT_DIR_ENV = "AAR_EXPORT_DIR"
#: The QA checks a review render must clear. ``size-cap`` is required too, but for a render QA'd
#: at the plan's own kind it is skipped as not applicable (the gate counts that as clear): the
#: review MP4 never leaves the hub, so the 20 MB review-transfer cap does not apply to it.
REQUIRED_QA = ("decode", "loudness", "clipping", "longest-silence", "size-cap", "duration")
#: A cut's joins must be quiet; with no cut (one segment) the check passes trivially.
REQUIRED_QA_JOINS = ("join-quiet",)
ALLOW_WARN = frozenset({"longest-silence"})
SIZE_CAP_NOTE = (
    "not applied: QA runs at the plan's kind, so the 20 MB review-transfer cap (QA kind review) "
    "does not apply; this review MP4 stays on the hub and reaches the PC through the export share"
)
QA_NOT_REQUIRED = {
    "join-retranscribe": "needs a transcriber in QA; the hub render runs QA without one",
    "caption-coverage": "a long-form plan has no captions",
}
MAX_WORDLESS_LISTED = 20
MAX_COPY_SUFFIX = 50
_UNSAFE = re.compile(r"[^A-Za-z0-9 ._-]+")
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$", re.IGNORECASE)

Transcribe = Callable[..., dict[str, Any]]
LaughsRunner = Callable[..., dict[str, Any]]
_TRANSCRIBE: list[Transcribe] = [transcribe_file]
_LAUGHS: list[LaughsRunner] = [analyse_laughs]


def set_transcriber(fn: Transcribe | None) -> None:
    """Use ``fn`` (``transcribe_file``'s signature) for the transcript step; None restores it."""
    _TRANSCRIBE[0] = fn or transcribe_file


def set_laughs_runner(fn: LaughsRunner | None) -> None:
    """Use ``fn`` (``analyse_laughs``'s signature) for the laughs step; None restores it."""
    _LAUGHS[0] = fn or analyse_laughs


# -- shared helpers ---------------------------------------------------------------------------


def _params(ctx: StepContext) -> CcnLongformParams:
    if not isinstance(ctx.params, CcnLongformParams):
        raise StepFailure("invalid_params", "the params are not ccn-longform params", "")
    return ctx.params


def _profile(ctx: StepContext) -> Profile:
    if ctx.profile is None:
        raise StepFailure(
            "no_profile",
            "the recipe's brief has no loadable rule profile",
            "create the brief with a built-in rule_profile such as ccn",
        )
    return ctx.profile


def work_root() -> Path:
    """``AAR_WORK_DIR``: where every step keeps its scratch; a typed failure when unset."""
    raw = os.environ.get(WORK_DIR_ENV, "").strip()
    if not raw or not os.path.isabs(raw):
        raise StepFailure(
            "work_dir_unset",
            f"{WORK_DIR_ENV} is not set to an absolute directory",
            f"set {WORK_DIR_ENV} for the worker (a directory on the media disk, not /tmp)",
        )
    root = Path(raw)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StepFailure(
            "work_dir_unusable", f"cannot use {WORK_DIR_ENV} {raw!r}: {exc}", "fix its permissions"
        ) from None
    return root


@contextlib.contextmanager
def _scratch(parent: Path) -> Iterator[Path]:
    """A fresh private directory under ``parent``, removed afterwards whatever happens."""
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"run-{secrets.token_hex(6)}"
    path.mkdir(mode=0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _need(ctx: StepContext, step: str, key: str) -> Any:
    value = ctx.needs.get(step, {}).get(key)
    if value is None:
        raise StepFailure("missing_input", f"step {step!r} has no output {key!r}", "")
    return value


def _json_artifact(ctx: StepContext, value: Any, role: str) -> str:
    data = json.dumps(value, indent=1, sort_keys=True).encode("utf-8")
    return ctx.put(data, mime="application/json", meta={"role": role})


def _read_json(ctx: StepContext, artifact_id: str) -> Any:
    return json.loads(ctx.get(artifact_id).decode("utf-8"))


def _stub_plan(ctx: StepContext) -> EditPlan:
    return EditPlan.model_validate_json(ctx.get(_need(ctx, "stub", "stub")))


def _cut_plan(ctx: StepContext) -> tuple[str, EditPlan]:
    plan_node = _need(ctx, "cut", "plan_node")
    plan = EditPlan.model_validate_json(ctx.get(_need(ctx, "cut", "plan_sha256")))
    return plan_node, plan


def _transcript(ctx: StepContext) -> Transcript:
    return Transcript.model_validate_json(ctx.get(_need(ctx, "transcript", "transcript")))


def _laughs(ctx: StepContext) -> LaughMap | None:
    map_id = ctx.needs.get("laughs", {}).get("map")
    return None if map_id is None else LaughMap.model_validate_json(ctx.get(map_id))


def _analyze_failure(step: str, exc: AnalyzeError) -> StepFailure:
    """A typed step failure from an analysis error (``transcript_model-missing``...)."""
    return StepFailure(f"{step}_{exc.type}", str(exc), exc.hint)


def _check_cancelled(ctx: StepContext) -> None:
    if ctx.cancelled():
        raise StepFailure("cancelled", "the worker is stopping", "retry the step")


# -- 1. stub ----------------------------------------------------------------------------------


def stub_step(ctx: StepContext) -> StepOutput:
    """Build the plan stub from the params' sources (``aar.plan.stub``) and store it."""
    params = _params(ctx)
    brief = ctx.brief
    if brief is None:
        raise StepFailure(
            "no_brief", "the recipe has no typed brief", "create it with brief_create"
        )
    if brief.kind != params.kind:
        raise StepFailure(
            "kind_mismatch",
            f"the brief is a {brief.kind}, the params ask for a {params.kind}",
            "make the params' kind the brief's",
        )
    sources = [
        StubSource(s.path, camera=s.camera, shot_size=s.shot_size, id=s.id) for s in params.sources
    ]
    try:
        result = build_stub(
            sources,
            recording_id=params.recording_id,
            project=brief.show,
            timeline=params.timeline,
            kind=params.kind,
            profile=brief.rule_profile,
            allow_unknown_timecode=params.allow_unknown_timecode,
        )
    except StubError as exc:
        raise StepFailure(f"stub_{exc.code}", exc.message, exc.hint) from None
    text = result.plan.model_dump_json(indent=2, exclude_none=True) + "\n"
    stub_id = ctx.put(text.encode("utf-8"), mime="application/json", meta={"role": "stub"})
    trim = params.pause_trim_on()
    report = result.report
    return StepOutput(
        outputs={
            "stub": stub_id,
            "fps": report["fps"],
            "resolution": report["resolution"],
            "recording_id": params.recording_id,
            "reference": result.plan.sources[0].id,
            "cameras": len(params.sources),
            "sources": [
                {
                    "id": s["id"],
                    "camera": s["camera"],
                    "duration": s["duration"],
                    "start_timecode": (s["start_timecode"] or {}).get("timecode"),
                }
                for s in report["sources"]
            ],
            "unknown_timecode": report["unknown_timecode"],
            "pause_trim": "on" if trim else "off",
            "pause_trim_flag": "--pause-trim" if trim else "--no-pause-trim",
            "warnings": [w["code"] for w in report["warnings"]][:20],
        },
        artifact_ids=(stub_id,),
    )


# -- 2. sync ----------------------------------------------------------------------------------


def sync_step(ctx: StepContext) -> StepOutput:
    """``aar.analyze.sync`` of every other source against the reference, as the CLI does."""
    stub = _stub_plan(ctx)
    media = [s for s in stub.sources if s.recording_id == _params(ctx).recording_id]
    if len(media) < 2:
        return StepOutput(
            outputs={"synced": False, "report": None, "reason": "one source: nothing to sync"}
        )
    _check_cancelled(ctx)
    fps = Fraction(stub.fps.num, stub.fps.den)
    try:
        report = sync_files(
            media[0].media,
            [s.media for s in media[1:]],
            fps=fps,
            ref_origin=0,
            settings=SyncSettings(),
        )
    except AnalyzeError as exc:
        raise _analyze_failure("sync", exc) from None
    pairs = [
        {
            "file": pair.get("file"),
            "status": pair.get("status"),
            "suggested_recording_origin": pair.get("suggested_recording_origin"),
        }
        for pair in report["pairs"]
    ]
    if not report["ok"]:
        bad = [f"{p['file']} ({p['status']})" for p in pairs if p["status"] != "ok"]
        raise StepFailure(
            "sync_not_ok",
            "these files did not sync against the reference: " + ", ".join(bad),
            "check that they hold the same performance (aar analyze sync shows the candidates), "
            "or remove them from the run",
        )
    report_id = _json_artifact(ctx, report, "sync")
    return StepOutput(
        outputs={"synced": True, "report": report_id, "pairs": pairs}, artifact_ids=(report_id,)
    )


# -- 3. transcript ----------------------------------------------------------------------------


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
    """The ``transcript.<key>`` fact key part: the sha256 of the media's hash, the model and
    every setting that changes the words or the report (not the thread count or the caps)."""
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


def _made_by_a_transcript_step(ctx: StepContext, artifact_id: str) -> bool:
    """Whether the artifact was stored by a runner-created ``transcript`` step op in this project
    (so a fact anyone may record cannot point the cache at bytes a step never made)."""
    meta = ctx.store.get_meta(artifact_id)
    if meta is None or meta.project_id != ctx.project_id or meta.producing_op is None:
        return False
    graph = WorkGraph(ctx.log, ctx.project_id)
    node = graph.node(meta.producing_op)
    if node is None or node.kind != "op" or graph.creator(node.id) != RUNNER:
        return False
    data = node.payload
    return (
        isinstance(data, dict)
        and data.get("schema") == STEP_SCHEMA
        and data.get("step_id") == "transcript"
    )


def _cached(ctx: StepContext, key: str, media_sha256: str, model: str) -> dict[str, Any] | None:
    fact = facts_view(ctx.log, ctx.project_id).get(f"transcript.{key}")
    if fact is None or not isinstance(fact.current.value, dict):
        return None
    value = fact.current.value
    if value.get("media_sha256") != media_sha256 or value.get("model") != model:
        return None
    ids = [value.get(name) for name in ("transcript", "text", "report")]
    for artifact_id in ids:
        if not isinstance(artifact_id, str) or not ctx.store.has(artifact_id):
            return None
        if not _made_by_a_transcript_step(ctx, artifact_id):
            return None
    return value


def _wordless_summary(report: Mapping[str, Any]) -> dict[str, Any]:
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


def _transcribe(
    ctx: StepContext,
    media: Path,
    settings: WhisperSettings,
    stub: EditPlan,
    reference: str,
    key: str,
) -> tuple[Path, dict[str, Any], Path]:
    """Run the transcriber into ``<work>/aar-transcripts/<key>/`` under a lock (one run per file
    and settings at a time), resuming a checkpoint an interrupted attempt left there."""
    folder = work_root() / "aar-transcripts" / key
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "transcript.json"
    offset = offset_from_plan(stub, reference)
    kwargs = {
        "settings": settings,
        "model_dir": os.environ.get(MODEL_DIR_ENV) or None,
        "offset": offset,
        "recording": _params(ctx).recording_id,
    }
    try:
        report = _TRANSCRIBE[0](media, out, resume=True, **kwargs)
    except AnalyzeError as exc:
        if exc.type not in ("checkpoint-mismatch", "checkpoint-corrupt"):
            raise
        # an interrupted run of other input bytes: start over, never splice two runs
        for leftover in folder.iterdir():
            if leftover.is_file():
                leftover.unlink()
        report = _TRANSCRIBE[0](media, out, resume=False, **kwargs)
    return out, report, folder


def transcript_step(ctx: StepContext) -> StepOutput:
    """Transcribe the reference source (or reuse the stored transcript of the same file, model
    and settings), store it with its readable text and report, and record the cache fact."""
    params = _params(ctx)
    stub = _stub_plan(ctx)
    reference = stub.sources[0]
    media = Path(reference.media)
    settings = whisper_settings(params)
    offset_s = offset_from_plan(stub, reference.id).offset_s
    _check_cancelled(ctx)
    try:
        media_sha256 = sha256_file(str(media)).lower()
    except OSError as exc:
        raise StepFailure(
            "transcript_unreadable", f"cannot read {str(media)!r}: {exc}", ""
        ) from None
    key = transcript_cache_key(media_sha256, settings, params.recording_id, offset_s)
    common = {
        "media_sha256": media_sha256,
        "model": settings.model,
        "normalize": settings.normalize,
        "cache_key": key,
        "recording": params.recording_id,
    }
    hit = _cached(ctx, key, media_sha256, settings.model)
    if hit is not None:
        report = _read_json(ctx, hit["report"])
        words = len(_transcript_from(ctx.get(hit["transcript"])).words)
        return StepOutput(
            outputs={
                **common,
                "cached": True,
                "transcript": hit["transcript"],
                "text": hit["text"],
                "report": hit["report"],
                "words": words,
                "wordless": _wordless_summary(report),
            },
            artifact_ids=(hit["transcript"], hit["text"], hit["report"]),
        )
    _check_cancelled(ctx)
    lock = work_root() / "aar-transcripts" / f"{key}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = acquire_file_lock(str(lock))
    except FileLockUnavailable:
        raise StepFailure(
            "transcript_busy",
            "another worker is transcribing this file with these settings",
            "retry the step when that run finishes (its transcript is then reused)",
        ) from None
    folder: Path | None = None
    try:
        try:
            out, report, folder = _transcribe(ctx, media, settings, stub, reference.id, key)
        except AnalyzeError as exc:
            raise _analyze_failure("transcript", exc) from None
        raw = out.read_bytes()
        transcript = _transcript_from(raw)
        text = render_text(transcript, wordless_min_s=settings.wordless_min_s)
        transcript_id = ctx.put(raw, mime="application/json", meta={"role": "transcript"})
        text_id = ctx.put(text.encode("utf-8"), mime="text/plain", meta={"role": "transcript-text"})
        report_id = _json_artifact(ctx, report, "transcript-report")
        ctx.record_fact(
            f"transcript.{key}",
            {
                "media_sha256": media_sha256,
                "model": settings.model,
                "recording": params.recording_id,
                "transcript": transcript_id,
                "text": text_id,
                "report": report_id,
            },
        )
    finally:
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
        release_file_lock(fd)
    return StepOutput(
        outputs={
            **common,
            "cached": False,
            "transcript": transcript_id,
            "text": text_id,
            "report": report_id,
            "words": len(transcript.words),
            "wordless": _wordless_summary(report),
        },
        artifact_ids=(transcript_id, text_id, report_id),
    )


def _transcript_from(raw: bytes) -> Transcript:
    try:
        return Transcript.model_validate_json(raw)
    except ValidationError as exc:
        raise StepFailure(
            "transcript_invalid",
            f"the transcript does not validate ({exc.error_count()} problem(s))",
            "this is a transcriber bug; report it",
        ) from None


# -- 4. laughs --------------------------------------------------------------------------------


def laughs_step(ctx: StepContext) -> StepOutput:
    """``aar.analyze.laughs`` on the reference, only when the params ask (it calls a paid API)."""
    params = _params(ctx)
    if not params.laughs:
        return StepOutput(outputs={"skipped": "not requested", "map": None})
    stub = _stub_plan(ctx)
    reference = stub.sources[0]
    _check_cancelled(ctx)
    with _scratch(work_root() / "aar-laughs") as folder:
        out, map_out = folder / "laughs.json", folder / "laughs.map.json"
        try:
            report = _LAUGHS[0](
                reference.media,
                out,
                work_dir=folder / "work",
                allow_paid=True,
                offset=offset_from_plan(stub, reference.id),
                recording=params.recording_id,
                map_out=map_out,
            )
        except AnalyzeError as exc:
            raise _analyze_failure("laughs", exc) from None
        laughs_id = ctx.put(out.read_bytes(), mime="application/json", meta={"role": "laughs"})
        raw_map = map_out.read_bytes()
        try:
            count = len(LaughMap.model_validate_json(raw_map).laughs)
        except ValidationError as exc:
            raise StepFailure(
                "laughs_invalid", f"the laugh map does not validate: {exc}", ""
            ) from None
        map_id = ctx.put(raw_map, mime="application/json", meta={"role": "laugh-map"})
    summary = {k: report.get(k) for k in ("model", "counts") if k in report}
    return StepOutput(
        outputs={"laughs": laughs_id, "map": map_id, "count": count, **summary},
        artifact_ids=(laughs_id, map_id),
    )


# -- 6. render --------------------------------------------------------------------------------


def _stale_render_dirs(ctx: StepContext, folder: Path) -> None:
    """Remove the render dirs earlier attempts of this recipe left (a killed worker), never a
    live one's."""
    if not folder.is_dir():
        return
    state = load_state(ctx.log, ctx.recipe_node, project_id=ctx.project_id)
    live = {op.op_id for op in state.ops if op.is_live}
    for child in folder.iterdir():
        if child.is_dir() and child.name != ctx.op_node and child.name not in live:
            shutil.rmtree(child, ignore_errors=True)


def safe_name(text: str) -> str:
    """``text`` as one file or folder name both Linux and Windows accept."""
    name = _UNSAFE.sub("_", text).strip(" .")[:100].strip(" .")
    if not name or name in (".", ".."):
        name = "untitled"
    if _RESERVED.match(name):
        name = f"_{name}"
    return name


def render_step(ctx: StepContext) -> StepOutput:
    """Render the submitted plan with the review preset on the hub, store the MP4 and its report,
    and gate the render's QA (failing checks named)."""
    profile = _profile(ctx)
    plan_node, plan = _cut_plan(ctx)
    transcript = _transcript(ctx)
    laughs = _laughs(ctx)
    _check_cancelled(ctx)
    recipe_dir = work_root() / "aar-render" / ctx.recipe_node
    _stale_render_dirs(ctx, recipe_dir)
    folder = recipe_dir / ctx.op_node
    folder.mkdir(parents=True, exist_ok=True)
    try:
        out = folder / f"{safe_name(plan.timeline)}.review.mp4"
        try:
            result = render_plan(
                plan, profile, out=out, preset="review", transcript=transcript, laughs=laughs
            )
        except RenderError as exc:
            raise StepFailure(f"render_{exc.code}", exc.message, exc.fix_hint) from None
        except RenderTerminated as exc:
            # The render's own SIGTERM handler killed ffmpeg and removed its work dir; the
            # worker's handler is back now, so hand it the signal (it stops claiming work) and
            # record a retryable failure instead of dying with the claim held.
            signal.raise_signal(exc.signum)
            raise StepFailure(
                "render_interrupted",
                f"the render was stopped by signal {exc.signum}",
                "retry the render step once the worker runs again",
            ) from None
        report = result.report
        mp4_id = ctx.put(result.out, mime="video/mp4", meta={"role": "review", "file": out.name})
        report_id = _json_artifact(ctx, report, "render-report")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        with contextlib.suppress(OSError):
            recipe_dir.rmdir()  # only when empty
    qa = QAReport.model_validate(report["qa"])
    required = REQUIRED_QA + REQUIRED_QA_JOINS
    gate = qa_gate(qa, required=required, allow_warn=ALLOW_WARN)
    if not gate.passed:
        failing = [f"{r.name} ({r.detail})" for r in gate.reasons if not r.passed]
        raise StepFailure(
            "qa_failed",
            "the review render failed QA: " + "; ".join(failing),
            f"the render and its report are stored (artifacts {mp4_id}, {report_id}); fix the "
            "cut and run the recipe again (cancel this run; a new run reuses the transcript)",
        )
    loudness = report.get("loudness") or {}
    return StepOutput(
        outputs={
            "subject_node": plan_node,
            "artifact_ids": [mp4_id],
            "review_mp4": mp4_id,
            "render_report": report_id,
            "sha256": report["sha256"],
            "size_bytes": report["size_bytes"],
            "duration_s": report["timeline"]["duration_s"],
            "qa_status": qa.status,
            "qa": gate.to_dict(),
            "qa_unchecked": list(qa.unchecked),
            "qa_not_required": QA_NOT_REQUIRED,
            "size_cap": SIZE_CAP_NOTE,
            "loudness": {
                k: loudness.get(k)
                for k in ("target_lufs", "output", "gain_db", "limiter_reduction_db", "warning")
                if k in loudness
            },
        },
        artifact_ids=(mp4_id, report_id),
    )


# -- 7. export --------------------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    return sha256_file(str(path)).lower()


def _copy_targets(folder: Path, stem: str, digests: list[tuple[str, str]]) -> list[Path] | None:
    """Free (or already identical) destinations for ``(suffix, sha256)`` files named after
    ``stem``; ``-2``, ``-3``... until none of the set is taken by other bytes. None if all of
    :data:`MAX_COPY_SUFFIX` are."""
    for n in range(1, MAX_COPY_SUFFIX + 1):
        name = stem if n == 1 else f"{stem}-{n}"
        paths = [folder / f"{name}{suffix}" for suffix, _ in digests]
        if all(
            not p.exists() or (p.is_file() and _sha256_of(p) == sha)
            for p, (_, sha) in zip(paths, digests, strict=True)
        ):
            return paths
    return None


def _place_copy(ctx: StepContext, dest: Path, artifact_id: str) -> None:
    """Copy one artifact to ``dest`` (a temporary file beside it, fsynced, then placed without
    ever replacing a file); nothing to do when ``dest`` already holds those bytes."""
    from aar.drp.modify.write import _place

    if dest.exists():
        return  # _copy_targets checked it holds these bytes
    tmp = dest.with_name(f".{dest.name}.aar-copy-{secrets.token_hex(4)}")
    try:
        with ctx.store.open(artifact_id) as src, tmp.open("xb") as sink:
            shutil.copyfileobj(src, sink, 1024 * 1024)
            sink.flush()
            os.fsync(sink.fileno())
        tmp.chmod(0o644)
        _place(tmp, dest, False)
    finally:
        tmp.unlink(missing_ok=True)


def export_step(ctx: StepContext) -> StepOutput:
    """Export the submitted plan as FCPXML with the params' media map, store it and its report,
    and copy the FCPXML and the review MP4 to ``AAR_EXPORT_DIR/<project>/<timeline>/``."""
    params = _params(ctx)
    profile = _profile(ctx)
    _plan_node, plan = _cut_plan(ctx)
    try:
        result = export_fcpxml(
            plan,
            profile,
            media_map=media_map_from_mapping(params.media_map),
            transcript=_transcript(ctx),
            laughs=_laughs(ctx),
            allow_unknown_timecode=params.allow_unknown_timecode,
        )
    except ExportError as exc:
        raise StepFailure(f"export_{exc.code}", exc.message, exc.fix_hint) from None
    fcpxml = result.fcpxml.encode("utf-8")
    fcpxml_id = ctx.put(fcpxml, mime="application/xml", meta={"role": "fcpxml"})
    report_id = _json_artifact(ctx, result.report, "export-report")
    outputs: dict[str, Any] = {
        "fcpxml": fcpxml_id,
        "export_report": report_id,
        "assets": [m.get("src") for m in result.report.get("media", [])][:16],
    }
    review_mp4 = _need(ctx, "render", "review_mp4")
    outputs["copy"] = _copy_out(ctx, plan, fcpxml_id, review_mp4)
    return StepOutput(outputs=outputs, artifact_ids=(fcpxml_id, report_id))


def _copy_out(ctx: StepContext, plan: EditPlan, fcpxml_id: str, mp4_id: str) -> dict[str, Any]:
    raw = os.environ.get(EXPORT_DIR_ENV, "").strip()
    if not raw:
        return {"copied": False, "reason": f"{EXPORT_DIR_ENV} is not set: nothing was copied"}
    root = Path(raw)
    if not root.is_absolute() or not root.is_dir():
        raise StepFailure(
            "export_dir_unusable",
            f"{EXPORT_DIR_ENV} {raw!r} is not an existing absolute directory",
            "create it (the media root's share/exports) or unset it",
        )
    _check_cancelled(ctx)
    state = load_state(ctx.log, ctx.recipe_node, project_id=ctx.project_id)
    round_ = next((op.round for op in state.ops if op.op_id == ctx.op_node), state.current_round())
    folder = root / safe_name(plan.project) / safe_name(plan.timeline)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{safe_name(plan.timeline)}.r{round_}"
        digests = [(".fcpxml", fcpxml_id), (".review.mp4", mp4_id)]
        targets = _copy_targets(folder, stem, digests)
        if targets is None:
            raise StepFailure(
                "export_names_taken",
                f"{folder} already holds {MAX_COPY_SUFFIX} other copies named {stem}",
                "move old exports away",
            )
        for dest, (_, artifact_id) in zip(targets, digests, strict=True):
            _place_copy(ctx, dest, artifact_id)
    except OSError as exc:
        raise StepFailure(
            "export_copy_failed", f"cannot copy into {str(folder)!r}: {exc}", "check the share"
        ) from None
    return {
        "copied": True,
        "dir": str(folder),
        "fcpxml": str(targets[0]),
        "review_mp4": str(targets[1]),
        "round": round_,
    }


__all__ = [
    "ALLOW_WARN",
    "REQUIRED_QA",
    "export_step",
    "laughs_step",
    "render_step",
    "safe_name",
    "set_laughs_runner",
    "set_transcriber",
    "stub_step",
    "sync_step",
    "transcript_cache_key",
    "transcript_step",
    "whisper_settings",
]
