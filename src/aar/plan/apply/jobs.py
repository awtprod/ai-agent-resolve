"""Build a phase envelope from a finished node ``ops.sequence`` job (rounds 3-4: R2, N1-N3, N5,
N7). This is the trust boundary between a job's raw op results and the compiler.

The recipe runner (S13l) submits one phase's compiled steps and hands back, per step, what it
submitted and the op's result::

    {"job_id": str, "steps": [{"op": str, "name": str, "params": {...}, "value": {...} | null}]}

1. The job is parsed through pydantic models; any shape error is ``malformed_job_result``.
2. It is bound to the plan: its submitted steps must equal, in canonical JSON, the steps this
   plan, phase and inputs compile to. A stale, reordered or foreign job is ``job_not_from_plan``;
   a job that stopped early, or a step without a result, is ``incomplete_job_result``.
3. Every result must parse as its op's result model (``malformed_job_result``).
4. Results are mapped by step name, never by position, and re-checked against what the step
   asked for: a find names exactly the requested clip within the requested scope, an import
   names exactly the requested files, the branch is of the plan's template, an append covers
   exactly its requested ranges.
5. The envelope then runs the same cross-checks as a hand-written one.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aar.node.ops import get_op
from aar.node.ops.build import ImportResult, TrackResult, path_key
from aar.node.ops.collab import BranchResult
from aar.node.ops.framework import MediaRef, TimelineIdentity
from aar.node.ops.read import DescribeResult, MediaFindResult, MediaInfo
from aar.node.ops.write import AppendResult
from aar.plan.apply.compile import DEFAULT_BATCH, compile_assemble, compile_prepare
from aar.plan.apply.errors import CompileError
from aar.plan.apply.inputs import (
    BRANCH_STEP,
    AssembleResult,
    FoundTitle,
    ImportedSource,
    PrepareResult,
    check_branch_id,
    classify_media,
    find_source_step,
    find_title_step,
    import_owners,
    plan_hash,
    title_templates,
    validate_assemble,
    validate_prepare,
)
from aar.plan.apply.model import Step
from aar.plan.model import EditPlan

_T = TypeVar("_T", bound=BaseModel)


class JobStep(BaseModel):
    """One step of a finished job: what was submitted plus the op's result (``None`` when the step
    did not complete)."""

    model_config = ConfigDict(extra="forbid")

    op: str = Field(min_length=1)
    name: str = Field(min_length=1)
    params: dict[str, Any]
    value: dict[str, Any] | None = None


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    steps: list[JobStep] = Field(min_length=1)


def phase_result_from_job(
    job_result: Any,
    plan: EditPlan,
    phase: str,
    *,
    template: DescribeResult | None = None,
    prepare: PrepareResult | None = None,
    batch_size: int = DEFAULT_BATCH,
) -> PrepareResult | AssembleResult:
    """Wrap a finished job's op results into a validated phase envelope.

    ``prepare`` needs ``template`` (the template describe the phase was compiled from);
    ``assemble`` needs ``prepare`` (the prepare envelope; the branch id comes from it, N2) and the
    ``batch_size`` it was compiled with.
    """
    job = _parse(job_result)
    if phase == "prepare":
        if template is None:
            raise CompileError(
                "missing_template",
                "building a prepare envelope needs the template describe the phase compiled from",
                "pass template=<timeline.describe of the plan's template>",
            )
        expected = compile_prepare(plan, template=template).steps
        results = _bind(job, expected, "prepare")
        result: PrepareResult | AssembleResult = PrepareResult(
            plan_sha256=plan_hash(plan),
            job_id=job.job_id,
            branch=_branch(results, template, plan),
            sources=_sources(results, expected, plan),
            titles=_titles(results, expected, plan),
        )
        validate_prepare(result, plan)
    elif phase == "assemble":
        if prepare is None:
            raise CompileError(
                "missing_prepare_result",
                "building an assemble envelope needs the prepare envelope (it holds the branch)",
                "pass prepare=<the prepare envelope of the same build>",
            )
        expected = compile_assemble(plan, prepare=prepare, batch_size=batch_size).steps
        results = _bind(job, expected, "assemble")
        _tracks_on_branch(results, expected, prepare)
        appends = [
            _append(_typed(results, step.name, AppendResult), step)
            for step in expected
            if step.op == "timeline.append_ranges"
        ]
        result = AssembleResult(
            plan_sha256=plan_hash(plan),
            job_id=job.job_id,
            branch=prepare.branch,  # N2: phase 2 has no timeline.branch step
            appends=appends,
        )
        validate_assemble(result, plan)
    else:
        raise ValueError(f"phase_result_from_job builds prepare/assemble, not {phase!r}")
    return result


# -- parse and bind ----------------------------------------------------------------------------


def _parse(job_result: Any) -> JobResult:
    try:
        return JobResult.model_validate(job_result)
    except (ValidationError, TypeError, ValueError) as exc:
        raise CompileError(
            "malformed_job_result",
            f"the job result is not a well-formed step list: {exc}",
            "pass {job_id, steps: [{op, name, params, value}]} for the finished job",
        ) from exc


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _bind(job: JobResult, expected: list[Step], phase: str) -> dict[str, BaseModel]:
    """Prove the job ran exactly the steps compiled for this plan, phase and inputs (N1), that
    every step has a result (N3), and parse each result as its op's result model."""
    want = [(s.op, s.name, _canon(s.params)) for s in expected]
    got = [(s.op, s.name, _canon(s.params)) for s in job.steps]
    if got != want:
        present = set(got)
        if len(present) == len(got) and got == [w for w in want if w in present]:
            missing = [name for op, name, params in want if (op, name, params) not in present]
            raise CompileError(
                "incomplete_job_result",
                f"the {phase} job has no result for {len(missing)} step(s): {', '.join(missing)}",
                "re-run the phase so every step completes",
                missing=missing,
            )
        raise CompileError(
            "job_not_from_plan",
            f"the {phase} job's steps are not the ones compiled for this plan and inputs",
            "submit exactly the steps 'aar plan apply' compiled; the plan or an input differs",
            expected=[name for _, name, _ in want],
            got=[name for _, name, _ in got],
        )
    results: dict[str, BaseModel] = {}
    for step in job.steps:
        if step.value is None:
            raise CompileError(
                "incomplete_job_result",
                f"step {step.name!r} ({step.op}) has no result; the job did not complete it",
                "re-run the phase so every step completes",
                step=step.name,
            )
        model = get_op(step.op).result
        try:
            results[step.name] = model.model_validate(step.value)
        except ValidationError as exc:
            raise CompileError(
                "malformed_job_result",
                f"step {step.name!r} ({step.op}) result is not a {model.__name__}: {exc}",
                "pass the op's own result value for every step",
                step=step.name,
            ) from exc
    return results


def _typed(results: dict[str, BaseModel], name: str, model: type[_T]) -> _T:
    value = results[name]
    assert isinstance(value, model), f"{name} is bound to {model.__name__}"
    return value


def _forged(message: str, hint: str, **details: Any) -> CompileError:
    return CompileError("forged_phase_result", message, hint, **details)


# -- prepare -----------------------------------------------------------------------------------


def _branch(
    results: dict[str, BaseModel], template: DescribeResult, plan: EditPlan
) -> TimelineIdentity:
    branch = _typed(results, BRANCH_STEP, BranchResult)
    if branch.source.unique_id != template.timeline.unique_id:
        raise CompileError(
            "template_mismatch",
            f"the job branched {branch.source.unique_id!r}, not the plan's template "
            f"{template.timeline.unique_id!r}",
            "branch the timeline the plan names as its template",
            branched=branch.source.unique_id,
            template=template.timeline.unique_id,
        )
    if branch.timeline.unique_id == template.timeline.unique_id:
        raise _forged(
            "the branch result names the template itself as the branch timeline",
            "phase 2 must never write into the template; re-run phase 'prepare'",
        )
    check_branch_id(branch.timeline.unique_id, plan)  # O1: before any envelope is built
    return branch.timeline


def _imported(
    results: dict[str, BaseModel], expected: list[Step]
) -> dict[str, tuple[MediaRef, str]]:
    """Each requested file (by ``path_key``) -> its pool ref and read-back path, from the import
    step that requested it. The result must name exactly the step's files, in the step's bin."""
    out: dict[str, tuple[MediaRef, str]] = {}
    for step in expected:
        if step.op != "media.import":
            continue
        result = _typed(results, step.name, ImportResult)
        if tuple(result.bin_path) != tuple(step.params["bin_path"]):
            raise _forged(
                f"import step {step.name!r} reports bin {'/'.join(result.bin_path)!r}, not the "
                f"requested {'/'.join(step.params['bin_path'])!r}",
                "the import result must be of the step's own request",
                step=step.name,
            )
        requested = sorted(path_key(p) for p in step.params["paths"])
        reported = sorted(path_key(m.path) for m in result.media)
        if reported != requested:
            raise _forged(
                f"import step {step.name!r} reports files {reported}, not the requested "
                f"{requested}",
                "the import result must name exactly the files the step imported",
                step=step.name,
            )
        for media in result.media:
            out[path_key(media.path)] = (media.media, media.file_path)
    return out


def _sources(
    results: dict[str, BaseModel], expected: list[Step], plan: EditPlan
) -> list[ImportedSource]:
    imported = _imported(results, expected)
    owners = import_owners(plan)
    params = {s.name: s.params for s in expected}
    out: list[ImportedSource] = []
    for source in plan.sources:
        if classify_media(source.media) == "import":
            # a source sharing a file maps to its owner's single import (N7)
            ref, file_path = imported[path_key(owners[path_key(source.media)].media)]
            out.append(ImportedSource(source_id=source.id, media=ref, file_path=file_path))
        else:
            name = find_source_step(source.id)
            info = _one_find(
                _typed(results, name, MediaFindResult),
                params[name],
                source.media,
                missing="source_not_found",
                ambiguous="source_ambiguous",
            )
            out.append(
                ImportedSource(
                    source_id=source.id,
                    media=MediaRef(unique_id=info.media_id, bin_path=info.bin_path),
                    file_path=info.file_path or source.media,
                )
            )
    return out


def _titles(
    results: dict[str, BaseModel], expected: list[Step], plan: EditPlan
) -> list[FoundTitle]:
    params = {s.name: s.params for s in expected}
    out: list[FoundTitle] = []
    for template in title_templates(plan):
        name = find_title_step(template)
        info = _one_find(
            _typed(results, name, MediaFindResult),
            params[name],
            template,
            missing="title_template_missing",
            ambiguous="title_template_ambiguous",
        )
        out.append(
            FoundTitle(
                template=template, media=MediaRef(unique_id=info.media_id, bin_path=info.bin_path)
            )
        )
    return out


def _in_scope(bin_path: tuple[str, ...] | None, scope: tuple[str, ...], recursive: bool) -> bool:
    if bin_path is None:
        return False
    return tuple(bin_path[: len(scope)]) == scope if recursive else tuple(bin_path) == scope


def _one_find(
    found: MediaFindResult,
    params: dict[str, Any],
    requested: str,
    *,
    missing: str,
    ambiguous: str,
) -> MediaInfo:
    """The single clip a ``media.find`` resolved, re-checked against the request (N1/N5): the
    result is of the requested scope, every hit is named exactly ``requested`` (case-sensitive,
    as the op matches) inside that scope, and there is exactly one hit."""
    scope = tuple(params["bin_path"])
    recursive = bool(params["recursive"])
    if tuple(found.bin_path) != scope or found.recursive != recursive:
        raise _forged(
            f"the find for {requested!r} reports scope {found.bin_path!r} (recursive="
            f"{found.recursive}), not the requested {scope!r} (recursive={recursive})",
            "the find result must be of the step's own request",
            requested=requested,
        )
    strangers = [m.name for m in found.media if m.name != requested]
    if strangers:
        raise _forged(
            f"the find for {requested!r} returned other names: {strangers!r}",
            "a find result must name exactly what the plan asked for",
            requested=requested,
            found=strangers,
        )
    outside = [m.bin_path for m in found.media if not _in_scope(m.bin_path, scope, recursive)]
    if outside:
        raise _forged(
            f"the find for {requested!r} returned hits outside its scope {scope!r}: {outside!r}",
            "a find result may only report clips inside the searched bins",
            requested=requested,
        )
    if len(found.media) != 1 or found.truncated:
        count = f"{len(found.media)}{'+' if found.truncated else ''}"
        raise CompileError(
            missing if not found.media else ambiguous,
            f"media.find for {requested!r} matched {count} media-pool items, not exactly one",
            "a pool name must name exactly one media-pool item; rename or remove the others "
            "in Resolve",
            requested=requested,
            bins=[m.bin_path for m in found.media],
        )
    return found.media[0]


# -- assemble ----------------------------------------------------------------------------------


def _tracks_on_branch(
    results: dict[str, BaseModel], expected: list[Step], prepare: PrepareResult
) -> None:
    for step in expected:
        if step.op != "track.ensure":
            continue
        tracks = _typed(results, step.name, TrackResult)
        if tracks.timeline.unique_id != prepare.branch.unique_id:
            raise CompileError(
                "phase_result_mismatch",
                f"track step {step.name!r} ran on timeline {tracks.timeline.unique_id!r}, not the "
                f"branch {prepare.branch.unique_id!r}",
                "every result must come from the same branch timeline",
            )


def _append(result: AppendResult, step: Step) -> AppendResult:
    """An append result must cover exactly its step's ranges, in order, on the requested tracks
    and record frames (the finish resolver keys items by ``(track, planned_record_in)``)."""
    ranges = step.params["ranges"]
    if not result.done or result.next_cursor is not None or len(result.ranges) != len(ranges):
        raise CompileError(
            "incomplete_job_result",
            f"append step {step.name!r} covered {len(result.ranges)} of {len(ranges)} range(s) "
            f"(done={result.done})",
            "re-run the phase so every append completes",
            step=step.name,
        )
    for index, (got, want) in enumerate(zip(result.ranges, ranges, strict=True)):
        video, audio = got.video, list(got.audio)
        if want["media_type"] == "audio":
            shape_ok = video is None and bool(audio)
        else:
            if want["media_type"] == "av" and not audio:
                raise CompileError(
                    "incomplete_job_result",
                    f"append step {step.name!r} range {index} ({want.get('label')!r}) came back "
                    "with no linked audio; finish would silently drop its crossfade, head fade "
                    "and gain",
                    "re-run the phase so the av append links its audio",
                    step=step.name,
                    range=index,
                )
            shape_ok = (
                video is not None and video.track_type == "video" and video.track == want["track"]
            )
        audio_ok = all(a.track_type == "audio" and a.track == want["track"] for a in audio)
        if (
            got.index != index
            or got.planned_record_in != want["record_in"]
            or not shape_ok
            or not audio_ok
        ):
            raise _forged(
                f"append step {step.name!r} range {index} reports index {got.index} at record "
                f"{got.planned_record_in}, not the requested {want['media_type']} range on track "
                f"{want['track']} at record {want['record_in']}",
                "the append result must be of the step's own ranges",
                step=step.name,
                range=index,
            )
    return result
