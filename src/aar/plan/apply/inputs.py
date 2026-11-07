"""The real phase-result envelopes the later phases compile from, plus the one place that builds
them from a node job's op results and cross-checks them against the plan (round 3, R2).

A phase is submittable only from the real results of the phases before it. Each result is an
envelope stamped with the plan's sha256, the job id and the branch timeline id; the compiler
refuses a stale envelope (`stale_phase_result`), a forged one (`forged_phase_result`: a pool id
that is a path, a title id equal to a template name, a `file_path` that is not the plan's media),
and duplicate or unknown source ids. ``phase_result_from_job`` is what the recipe runner (S13l)
calls to turn the raw op results into a validated envelope.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aar.node.ops.build import EditPath, ImportResult, path_key
from aar.node.ops.collab import BranchResult
from aar.node.ops.framework import MediaRef, TimelineIdentity, TimelineRef
from aar.node.ops.read import MediaFindResult
from aar.node.ops.write import AppendResult
from aar.plan.apply.errors import CompileError
from aar.plan.model import EditPlan, Source

IMPORT_ROOT = "aar"
_MEDIA_EXT = re.compile(r"\.[A-Za-z0-9]{1,5}$")
_DRIVE = re.compile(r"^[A-Za-z]:")

# Validate an edit-PC path with the op's own rule.
from pydantic import TypeAdapter  # noqa: E402

_EDIT_PATH = TypeAdapter(EditPath)


def plan_hash(plan: EditPlan) -> str:
    canonical = json.dumps(
        plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- media classification (R1) -----------------------------------------------------------------


def is_edit_path(media: str) -> bool:
    try:
        _EDIT_PATH.validate_python(media)
    except Exception:
        return False
    return True


def _looks_like_path(media: str) -> bool:
    return bool(re.search(r"[\\/]", media) or _DRIVE.match(media) or media.startswith("/"))


def classify_media(media: str) -> Literal["import", "pool", "error"]:
    """How a source's ``media`` is resolved to a pool item (R1):

    * ``import`` -- a valid absolute edit-PC path, imported by ``media.import``;
    * ``pool`` -- a media-pool name (e.g. a CCN multicam clip), found by ``media.find``;
    * ``error`` -- anything path-shaped that is not a valid edit-PC path (a hub/Linux path, a
      relative path, a malformed Windows path): not resolvable, raise ``media_not_on_edit_pc``.
    """
    if is_edit_path(media):
        return "import"
    if _looks_like_path(media) or _MEDIA_EXT.search(media.strip()):
        return "error"
    return "pool"


def import_bin(source: Source) -> tuple[str, ...]:
    return (IMPORT_ROOT, source.camera or "sources")


def title_templates(plan: EditPlan) -> list[str]:
    names: list[str] = []
    for title in plan.titles:
        if title.template is not None and title.template not in names:
            names.append(title.template)
    sting = plan.outro.sting if plan.outro is not None else None
    if sting is not None and sting.template and sting.template not in names:
        names.append(sting.template)
    return names


def check_media_classifiable(plan: EditPlan) -> None:
    """Refuse any source whose media is path-shaped but not a valid edit-PC path (R1)."""
    for source in plan.sources:
        if classify_media(source.media) == "error":
            raise CompileError(
                "media_not_on_edit_pc",
                f"source {source.id!r} media {source.media!r} is not an edit-PC path or a "
                "media-pool name",
                "stage it on the edit PC with `aar host push`, then give its edit-PC path "
                "(see docs/MEDIA-STORE.md)",
                source=source.id,
                media=source.media,
            )


# -- envelopes ---------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImportedSource(_Model):
    source_id: str = Field(min_length=1)
    media: MediaRef
    file_path: str = Field(min_length=1, description="The real clip path, for the BRAW re-check.")


class FoundTitle(_Model):
    template: str = Field(min_length=1)
    media: MediaRef


class PrepareResult(_Model):
    """Phase 1's envelope."""

    plan_sha256: str = Field(min_length=1)
    job_id: str = ""
    branch: TimelineIdentity
    sources: list[ImportedSource] = Field(default_factory=list)
    titles: list[FoundTitle] = Field(default_factory=list)

    @property
    def branch_timeline_id(self) -> str:
        return self.branch.unique_id

    def branch_ref(self) -> TimelineRef:
        return TimelineRef(name=self.branch.name, unique_id=self.branch.unique_id)

    def source_media(self) -> dict[str, ImportedSource]:
        return {s.source_id: s for s in self.sources}

    def title_media(self) -> dict[str, MediaRef]:
        return {t.template: t.media for t in self.titles}


class AssembleResult(_Model):
    """Phase 2's envelope."""

    plan_sha256: str = Field(min_length=1)
    job_id: str = ""
    branch: TimelineIdentity
    appends: list[AppendResult] = Field(default_factory=list)

    @property
    def branch_timeline_id(self) -> str:
        return self.branch.unique_id

    def branch_ref(self) -> TimelineRef:
        return TimelineRef(name=self.branch.name, unique_id=self.branch.unique_id)


# -- cross-checks (R2) -------------------------------------------------------------------------


def _check_sha(result: Any, plan: EditPlan) -> None:
    if result.plan_sha256 != plan_hash(plan):
        raise CompileError(
            "stale_phase_result",
            f"the phase result is for plan {result.plan_sha256[:12]}..., not the current plan "
            f"{plan_hash(plan)[:12]}...",
            "re-run the earlier phase on this exact plan",
        )


def _check_id_shape(unique_id: str, what: str, *, forbid: str | None = None) -> None:
    if _looks_like_path(unique_id) or (forbid is not None and unique_id == forbid):
        raise CompileError(
            "forged_phase_result",
            f"{what} id {unique_id!r} is not a Resolve unique id (it is a path or a plan name)",
            "a phase result must carry the real pool unique ids the ops returned",
            value=unique_id,
        )


def validate_prepare(prepare: PrepareResult, plan: EditPlan) -> None:
    _check_sha(prepare, plan)
    by_id = {s.id: s for s in plan.sources}
    seen: set[str] = set()
    for src in prepare.sources:
        if src.source_id not in by_id:
            raise CompileError(
                "unknown_source",
                f"prepare result has an unknown source id {src.source_id!r}",
                "a phase result may only carry the plan's own sources",
                source=src.source_id,
            )
        if src.source_id in seen:
            raise CompileError(
                "duplicate_source",
                f"source id {src.source_id!r} appears twice in the prepare result",
                "each source resolves to exactly one pool item",
                source=src.source_id,
            )
        seen.add(src.source_id)
        _check_id_shape(src.media.unique_id, f"source {src.source_id!r} pool")
        plan_media = by_id[src.source_id].media
        is_import = classify_media(plan_media) == "import"
        if is_import and path_key(src.file_path) != path_key(plan_media):
            raise CompileError(
                "forged_phase_result",
                f"source {src.source_id!r} imported {src.file_path!r}, not the plan's media "
                f"{plan_media!r}",
                "the import result must be of the plan's media for that source",
                source=src.source_id,
            )
    templates = set(title_templates(plan))
    seen_t: set[str] = set()
    for title in prepare.titles:
        if title.template not in templates:
            raise CompileError(
                "title_template_missing",
                f"prepare result has a title template {title.template!r} the plan does not use",
                "a phase result may only carry the plan's own title templates",
            )
        if title.template in seen_t:
            raise CompileError(
                "title_template_ambiguous",
                f"title template {title.template!r} appears twice in the prepare result",
                "each title template resolves to exactly one pool item",
            )
        seen_t.add(title.template)
        _check_id_shape(title.media.unique_id, f"title {title.template!r}", forbid=title.template)


def validate_assemble(assemble: AssembleResult, plan: EditPlan) -> None:
    _check_sha(assemble, plan)
    for result in assemble.appends:
        if result.timeline.unique_id != assemble.branch.unique_id:
            raise CompileError(
                "phase_result_mismatch",
                f"an append result is from timeline {result.timeline.unique_id!r}, not the "
                f"assemble branch {assemble.branch.unique_id!r}",
                "every result must come from the same branch timeline",
            )


# -- build from a node job (R2) ----------------------------------------------------------------


def phase_result_from_job(
    job_result: dict[str, Any], plan: EditPlan, phase: str
) -> PrepareResult | AssembleResult:
    """Wrap a node ops.sequence job's op results into a validated phase envelope.

    ``job_result`` is ``{"job_id": str, "steps": [{"op": str, "name": str, "value": dict}, ...]}``
    -- the recipe runner (S13l) passes the real op results. The envelope is stamped with the
    plan's sha256 and the branch timeline id, then cross-checked against the plan.
    """
    steps = job_result.get("steps", [])
    job_id = str(job_result.get("job_id", ""))
    branch = _job_branch(steps)
    sha = plan_hash(plan)
    if phase == "prepare":
        result: PrepareResult | AssembleResult = PrepareResult(
            plan_sha256=sha,
            job_id=job_id,
            branch=branch,
            sources=_job_sources(steps, plan),
            titles=_job_titles(steps, plan),
        )
        validate_prepare(result, plan)
    elif phase == "assemble":
        appends = [
            AppendResult.model_validate(s["value"])
            for s in steps
            if s["op"] == "timeline.append_ranges"
        ]
        result = AssembleResult(plan_sha256=sha, job_id=job_id, branch=branch, appends=appends)
        validate_assemble(result, plan)
    else:
        raise ValueError(f"phase_result_from_job builds prepare/assemble, not {phase!r}")
    return result


def _job_branch(steps: list[dict[str, Any]]) -> TimelineIdentity:
    for step in steps:
        if step["op"] == "timeline.branch":
            return BranchResult.model_validate(step["value"]).timeline
    raise CompileError(
        "phase_result_mismatch",
        "the job result has no timeline.branch step",
        "phase 1 must branch the template first",
    )


def _job_sources(steps: list[dict[str, Any]], plan: EditPlan) -> list[ImportedSource]:
    by_pathkey = {path_key(s.media): s for s in plan.sources if classify_media(s.media) == "import"}
    out: list[ImportedSource] = []
    for step in steps:
        if step["op"] == "media.import":
            for im in ImportResult.model_validate(step["value"]).media:
                src = by_pathkey.get(path_key(im.file_path)) or _by_requested(by_pathkey, im.path)
                if src is not None:
                    out.append(
                        ImportedSource(source_id=src.id, media=im.media, file_path=im.file_path)
                    )
    pool_sources = [s for s in plan.sources if classify_media(s.media) == "pool"]
    finds = {s["name"]: s for s in steps if step_is_find_source(s)}
    for index, src in enumerate(pool_sources):
        found = finds.get(f"find-source:{index}")
        if found is None:
            continue
        media = MediaFindResult.model_validate(found["value"]).media
        if len(media) != 1:
            code = "source_not_found" if not media else "source_ambiguous"
            raise CompileError(
                code,
                f"media.find for source {src.id!r} ({src.media!r}) returned {len(media)} items",
                "a pool-name source must name exactly one media-pool item",
                source=src.id,
            )
        info = media[0]
        out.append(
            ImportedSource(
                source_id=src.id,
                media=MediaRef(unique_id=info.media_id, bin_path=info.bin_path),
                file_path=info.file_path or src.media,
            )
        )
    return out


def _by_requested(by_pathkey: dict[str, Source], path: str) -> Source | None:
    return by_pathkey.get(path_key(path))


def step_is_find_source(step: dict[str, Any]) -> bool:
    return step["op"] == "media.find" and str(step.get("name", "")).startswith("find-source:")


def _job_titles(steps: list[dict[str, Any]], plan: EditPlan) -> list[FoundTitle]:
    templates = title_templates(plan)
    finds = {
        s["name"]: s
        for s in steps
        if s["op"] == "media.find" and str(s.get("name", "")).startswith("find-title:")
    }
    out: list[FoundTitle] = []
    for index, name in enumerate(templates):
        found = finds.get(f"find-title:{index}")
        if found is None:
            continue
        media = MediaFindResult.model_validate(found["value"]).media
        if len(media) != 1:
            code = "title_template_missing" if not media else "title_template_ambiguous"
            raise CompileError(
                code,
                f"media.find for title template {name!r} returned {len(media)} items",
                "a title template must name exactly one media-pool item",
                template=name,
            )
        info = media[0]
        out.append(
            FoundTitle(
                template=name, media=MediaRef(unique_id=info.media_id, bin_path=info.bin_path)
            )
        )
    return out
