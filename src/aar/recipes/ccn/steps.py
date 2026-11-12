"""The cold steps of ``ccn-longform``: each calls the library the CLI calls, never the CLI.

Every step reads its inputs from the params and the earlier steps' runner-recorded outputs
(artifact ids), keeps its scratch under ``AAR_WORK_DIR`` (:mod:`~aar.recipes.ccn.scratch`: never
the system temp dir, never the artifact store's disk), removes it on success and failure, stores
what it makes as artifacts, and fails with a typed :class:`~aar.recipes.context.StepFailure` (a
stable code and a fix hint). A worker that is stopping before a step's work began raises
:class:`~aar.recipes.ccn.transcript.StepInterrupted` past the worker, so the claim lapses and the
(idempotent) step is claimed again instead of using an attempt. The transcriber and the laughs
analysis are injectable (:func:`set_transcriber`, :func:`set_laughs_runner`) so tests need no
model and no paid API.

**The agent's plan is checked against the run, not trusted** (:func:`check_plan`): before the
render and the export, every source of the submitted plan must be one of the stub's, with the
same media path (and that path inside ``AAR_MEDIA_ROOT``, as the params are checked) and the same
recording, frame size and start timecode, and the plan's frame rate, resolution, kind, project
and timeline must be the stub's. The export's folder and file names come from the stub step's
runner-recorded outputs, never from the plan.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aar.analyze.common import input_identity, offset_from_plan
from aar.analyze.errors import AnalyzeError
from aar.analyze.laughs import LaughSettings, analyse_laughs
from aar.analyze.sync import SyncSettings, sync_files
from aar.plan.export import ExportError, export_fcpxml, media_map_from_mapping
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.model import EditPlan
from aar.plan.profile import Profile
from aar.plan.render import RenderError, RenderTerminated, render_plan
from aar.plan.stub import StubError, StubSource, build_stub
from aar.qa.report import QAReport
from aar.recipes.ccn import transcript as transcript_mod
from aar.recipes.ccn.export_files import copy_out, safe_name
from aar.recipes.ccn.guide import artifact_get, build_guide
from aar.recipes.ccn.params import CcnLongformParams, check_hub_file
from aar.recipes.ccn.scratch import drop_scratch, op_scratch
from aar.recipes.ccn.transcript import (
    StepInterrupted,
    check_cancelled,
    library_timeout,
    media_seconds,
    set_transcriber,
    transcript_cache_key,
    whisper_settings,
)
from aar.recipes.context import StepContext, StepFailure, StepOutput
from aar.recipes.gates import GateResult, qa_gate
from aar.recipes.state import load_state

#: The QA checks a review render must clear. ``size-cap`` is required too, but for a render QA'd
#: at the plan's own kind it is skipped as not applicable (the gate counts that as clear): the
#: review MP4 never leaves the hub, so the 20 MB review-transfer cap does not apply to it.
REQUIRED_QA = (
    "decode",
    "loudness",
    "clipping",
    "longest-silence",
    "size-cap",
    "duration",
    "join-quiet",
)
#: Only the longest silence may warn; a loudness, clipping or join warning fails the gate.
ALLOW_WARN = frozenset({"longest-silence"})
SIZE_CAP_NOTE = (
    "not applied: QA runs at the plan's kind, so the 20 MB review-transfer cap (QA kind review) "
    "does not apply; this review MP4 stays on the hub and reaches the PC through the export share"
)
QA_NOT_REQUIRED = {
    "join-retranscribe": "needs a transcriber in QA; the hub render runs QA without one",
    "caption-coverage": "a long-form plan has no captions",
}
#: The plan source fields that must equal the stub's (produce copies them from the stub).
SOURCE_FIELDS = ("media", "recording_id", "frame_size", "start_timecode", "start_timecode_rate")
PLAN_FIELDS = ("fps", "resolution", "kind", "project", "timeline")

LaughsRunner = Callable[..., dict[str, Any]]
_LAUGHS: list[LaughsRunner] = [analyse_laughs]


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


def _need(ctx: StepContext, step: str, key: str) -> Any:
    value = ctx.needs.get(step, {}).get(key)
    if value is None:
        raise StepFailure("missing_input", f"step {step!r} has no output {key!r}", "")
    return value


def _json_artifact(ctx: StepContext, value: Any, role: str) -> str:
    data = json.dumps(value, indent=1, sort_keys=True).encode("utf-8")
    return ctx.put(data, mime="application/json", meta={"role": role})


def _stub_plan(ctx: StepContext) -> EditPlan:
    return EditPlan.model_validate_json(ctx.get(_need(ctx, "stub", "stub")))


def _transcript(ctx: StepContext) -> Transcript:
    return Transcript.model_validate_json(ctx.get(_need(ctx, "transcript", "transcript")))


def _laughs(ctx: StepContext) -> LaughMap | None:
    map_id = ctx.needs.get("laughs", {}).get("map")
    return None if map_id is None else LaughMap.model_validate_json(ctx.get(map_id))


def _analyze_failure(step: str, exc: AnalyzeError) -> StepFailure:
    """A typed step failure from an analysis error (``transcript_model-missing``...)."""
    return StepFailure(f"{step}_{exc.type}", str(exc), exc.hint)


def _round(ctx: StepContext) -> int:
    state = load_state(ctx.log, ctx.recipe_node, project_id=ctx.project_id)
    return next((op.round for op in state.ops if op.op_id == ctx.op_node), state.current_round())


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
    check_cancelled(ctx)
    sources = [
        StubSource(s.path, camera=s.camera, shot_size=s.shot_size, id=s.id) for s in params.sources
    ]
    try:  # before the probes: a file replaced while they run differs at the render
        identities = [input_identity(Path(s.path)) for s in params.sources]
    except OSError as exc:
        raise StepFailure(
            "stub_unreadable", f"cannot stat a source: {exc}", "check the file"
        ) from None
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
            "project": result.plan.project,
            "timeline": result.plan.timeline,
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
                    **identity,
                }
                for s, identity in zip(report["sources"], identities, strict=True)
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
    check_cancelled(ctx)
    fps = Fraction(stub.fps.num, stub.fps.den)
    longest = max((media_seconds(stub, i) or 0.0) for i in range(len(media))) or None
    try:
        report = sync_files(
            media[0].media,
            [s.media for s in media[1:]],
            fps=fps,
            ref_origin=0,
            settings=SyncSettings(timeout_s=library_timeout(ctx, longest)),
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


def transcript_step(ctx: StepContext) -> StepOutput:
    """The reference source's transcript (:mod:`aar.recipes.ccn.transcript`)."""
    _params(ctx)
    return transcript_mod.transcript_step(ctx, _stub_plan(ctx))


# -- 4. laughs --------------------------------------------------------------------------------


def laughs_step(ctx: StepContext) -> StepOutput:
    """``aar.analyze.laughs`` on the reference, only when the params ask (it calls a paid API)."""
    params = _params(ctx)
    if not params.laughs:
        return StepOutput(outputs={"skipped": "not requested", "map": None})
    stub = _stub_plan(ctx)
    reference = stub.sources[0]
    check_cancelled(ctx)
    folder = op_scratch(ctx, "aar-laughs")
    try:
        out, map_out = folder / "laughs.json", folder / "laughs.map.json"
        try:
            report = _LAUGHS[0](
                reference.media,
                out,
                work_dir=folder / "work",
                allow_paid=True,
                settings=LaughSettings(timeout_s=library_timeout(ctx, media_seconds(stub))),
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
    finally:
        drop_scratch(folder)
    summary = {k: report.get(k) for k in ("model", "counts") if k in report}
    return StepOutput(
        outputs={"laughs": laughs_id, "map": map_id, "count": count, **summary},
        artifact_ids=(laughs_id, map_id),
    )


# -- 5. guide ---------------------------------------------------------------------------------


def aar_argv() -> list[str]:
    """The argv that runs the worker's own ``aar``: the script beside this interpreter, else
    this interpreter with ``-m aar``. Never a bare ``aar``: the hub's is not on PATH."""
    script = Path(sys.executable).with_name("aar")
    return [str(script)] if script.is_file() else [sys.executable, "-m", "aar"]


def guide_step(ctx: StepContext) -> StepOutput:
    """Write the ``cut`` step's full guide (:mod:`aar.recipes.ccn.guide`) as an artifact, with
    the hub paths this worker runs on and this run's input artifact ids."""
    state = load_state(ctx.log, ctx.recipe_node, project_id=ctx.project_id)
    db = os.path.abspath(ctx.log.path)
    artifacts = os.path.abspath(ctx.store.root)
    aar = aar_argv()
    text = build_guide(
        db=db,
        artifacts=artifacts,
        project_id=ctx.project_id,
        recipe_node=ctx.recipe_node,
        brief_id=state.recipe_data.brief_id if state.recipe_data is not None else None,
        brief=ctx.brief,
        stub=dict(ctx.needs["stub"]),
        sync=dict(ctx.needs["sync"]),
        transcript=dict(ctx.needs["transcript"]),
        laughs=dict(ctx.needs["laughs"]),
        aar=aar,
        python=sys.executable,
    )
    guide_id = ctx.put(text.encode("utf-8"), mime="text/markdown", meta={"role": "cut-guide"})
    read = artifact_get(
        aar, guide_id, "guide.md", db=db, artifacts=artifacts, project_id=ctx.project_id
    )
    return StepOutput(
        outputs={"guide": guide_id, "read_guide": read, "db": db, "artifacts": artifacts},
        artifact_ids=(guide_id,),
    )


# -- the agent's plan, checked ------------------------------------------------------------------


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def check_plan(ctx: StepContext) -> tuple[str, EditPlan]:
    """The submitted plan, when it matches this run's stub (module docstring); else
    ``plan_not_from_stub`` naming each difference."""
    plan_node = _need(ctx, "cut", "plan_node")
    plan = EditPlan.model_validate_json(ctx.get(_need(ctx, "cut", "plan_sha256")))
    stub = _stub_plan(ctx)
    diffs: list[str] = []
    for name in PLAN_FIELDS:
        got, want = _dump(getattr(plan, name)), _dump(getattr(stub, name))
        if got != want:
            diffs.append(f"{name} is {got!r}, the stub's {want!r}")
    known = stub.source_map()
    for source in plan.sources:
        mine = known.get(source.id)
        if mine is None:
            diffs.append(f"source {source.id!r} is not one of the stub's ({', '.join(known)})")
            continue
        for name in SOURCE_FIELDS:
            got, want = _dump(getattr(source, name)), _dump(getattr(mine, name))
            if got != want:
                diffs.append(f"source {source.id!r} {name} is {got!r}, the stub's {want!r}")
        try:
            check_hub_file(source.media)
        except ValueError as exc:
            diffs.append(f"source {source.id!r}: {exc}")
        else:
            if os.path.realpath(source.media) != os.path.realpath(mine.media):
                diffs.append(f"source {source.id!r} media resolves to another file")
    if diffs:
        raise StepFailure(
            "plan_not_from_stub",
            "the submitted plan does not match this run's stub: " + "; ".join(diffs[:10]),
            "make the plan with aar plan produce from the stub.json the guide fetches, unedited",
        )
    _check_media_unchanged(ctx, plan)
    return plan_node, plan


def _check_media_unchanged(ctx: StepContext, plan: EditPlan) -> None:
    """``media_changed`` naming the file when a plan source's size or mtime_ns is not what the
    stub step recorded (a file replaced at the same path since: the transcript the cut was made
    on is not this file's)."""
    recorded = {s.get("id"): s for s in _need(ctx, "stub", "sources")}
    for source in plan.sources:
        want = recorded.get(source.id) or {}
        try:
            now: dict[str, Any] = input_identity(Path(source.media))
        except OSError as exc:
            now = {"error": str(exc)}
        was = {k: want.get(k) for k in ("size", "mtime_ns")}
        if now != was:
            raise StepFailure(
                "media_changed",
                f"{source.media!r} (source {source.id!r}) changed since the stub step: it was "
                f"{was}, it is {now}",
                "a file was replaced at the same path: cancel this run and start a new one, "
                "which builds its stub and transcript from the file as it is now",
            )


# -- 6. render --------------------------------------------------------------------------------


def review_qa_gate(qa: QAReport) -> GateResult:
    """The review render's QA gate: :data:`REQUIRED_QA`, warnings only in :data:`ALLOW_WARN`."""
    return qa_gate(qa, required=REQUIRED_QA, allow_warn=ALLOW_WARN)


def render_step(ctx: StepContext) -> StepOutput:
    """Render the submitted plan with the review preset on the hub, store the MP4 and its report,
    and gate the render's QA (failing checks named). Idempotent: a re-claimed op starts from an
    emptied scratch folder and stores content-addressed artifacts."""
    profile = _profile(ctx)
    plan_node, plan = check_plan(ctx)
    transcript = _transcript(ctx)
    laughs = _laughs(ctx)
    check_cancelled(ctx)
    folder = op_scratch(ctx, "aar-render")
    try:
        out = folder / f"{safe_name(_need(ctx, 'stub', 'timeline'))}.review.mp4"
        try:
            result = render_plan(
                plan, profile, out=out, preset="review", transcript=transcript, laughs=laughs
            )
        except RenderError as exc:
            raise StepFailure(f"render_{exc.code}", exc.message, exc.fix_hint) from None
        except RenderTerminated as exc:
            # The render's SIGTERM handler killed ffmpeg and removed its work dir; the worker's
            # handler is back now: hand it the signal (it stops claiming) and leave the claim to
            # lapse, so the op is claimed again rather than failed (no attempt is used).
            signal.raise_signal(exc.signum)
            raise StepInterrupted(f"the render was stopped by signal {exc.signum}") from None
        report = result.report
        mp4_id = ctx.put(result.out, mime="video/mp4", meta={"role": "review", "file": out.name})
        report_id = _json_artifact(ctx, report, "render-report")
    finally:
        drop_scratch(folder)
    qa = QAReport.model_validate(report["qa"])
    gate = review_qa_gate(qa)
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


def export_step(ctx: StepContext) -> StepOutput:
    """Export the submitted plan as FCPXML with the params' media map, store it and its report,
    and copy the FCPXML and the review MP4 to ``AAR_EXPORT_DIR/<project>/<timeline>/``, named
    from the stub step's runner-recorded outputs."""
    params = _params(ctx)
    profile = _profile(ctx)
    _plan_node, plan = check_plan(ctx)
    check_cancelled(ctx)
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
    fcpxml_id = ctx.put(
        result.fcpxml.encode("utf-8"), mime="application/xml", meta={"role": "fcpxml"}
    )
    report_id = _json_artifact(ctx, result.report, "export-report")
    review_mp4 = _need(ctx, "render", "review_mp4")
    copy = copy_out(
        ctx,
        _need(ctx, "stub", "project"),
        _need(ctx, "stub", "timeline"),
        _round(ctx),
        [(".fcpxml", fcpxml_id), (".review.mp4", review_mp4)],
    )
    if copy.get("copied"):
        copy["fcpxml"], copy["review_mp4"] = copy.pop("files")
    return StepOutput(
        outputs={
            "fcpxml": fcpxml_id,
            "export_report": report_id,
            "assets": [m.get("src") for m in result.report.get("media", [])][:16],
            "copy": copy,
        },
        artifact_ids=(fcpxml_id, report_id),
    )


__all__ = [
    "ALLOW_WARN",
    "REQUIRED_QA",
    "aar_argv",
    "check_plan",
    "export_step",
    "guide_step",
    "laughs_step",
    "render_step",
    "review_qa_gate",
    "safe_name",
    "set_laughs_runner",
    "set_transcriber",
    "stub_step",
    "sync_step",
    "transcript_cache_key",
    "transcript_step",
    "whisper_settings",
]
