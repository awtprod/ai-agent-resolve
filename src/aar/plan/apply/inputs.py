"""The real phase-result envelopes the later phases compile from, and the cross-checks that run
on every envelope, hand-written or built from a node job (rounds 3-4).

A phase is submittable only from the real results of the phases before it. Each result is an
envelope stamped with the plan's sha256, the job id and the branch timeline id; the compiler
refuses a stale envelope (`stale_phase_result`), a forged one (`forged_phase_result`: a pool id
that contains a path character or equals a plan-supplied string, a `file_path` that is not the
plan's media), and duplicate or unknown source ids. The builder that turns a node job's raw op
results into an envelope is :func:`aar.plan.apply.jobs.phase_result_from_job` (also importable
from here).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aar.node.ops.build import EditPath, path_key
from aar.node.ops.framework import MediaRef, TimelineIdentity, TimelineRef
from aar.node.ops.write import AppendResult
from aar.plan.apply.errors import CompileError
from aar.plan.model import EditPlan, Source

IMPORT_ROOT = "aar"
_MEDIA_EXT = re.compile(r"\.[A-Za-z0-9]{1,5}$")
_DRIVE = re.compile(r"^[A-Za-z]:")

# Validate an edit-PC path with the op's own rule.
from pydantic import TypeAdapter  # noqa: E402

_EDIT_PATH = TypeAdapter(EditPath)


#: Source fields added after ``plan_hash`` was first used (S18): left out while unset, so the
#: hash of every plan written before them, and the phase results stamped with it, stay the same.
_LATER_SOURCE_FIELDS = ("frame_size", "start_timecode", "start_timecode_rate")


def plan_hash(plan: EditPlan) -> str:
    data = plan.model_dump(mode="json")
    for source in data["sources"]:
        for name in _LATER_SOURCE_FIELDS:
            if source.get(name) is None:
                source.pop(name, None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
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


def import_owners(plan: EditPlan) -> dict[str, Source]:
    """Each distinct edit-PC file (by ``path_key``) -> the first source that uses it. That source's
    bin gets the one ``media.import`` of the file; every source sharing it maps to the same pool
    item (N7)."""
    owners: dict[str, Source] = {}
    for source in plan.sources:
        if classify_media(source.media) == "import":
            owners.setdefault(path_key(source.media), source)
    return owners


# Stable, unique step names: a job's results are mapped by these, never by position (N1).
BRANCH_STEP = "branch"


def find_source_step(source_id: str) -> str:
    return f"find:source:{source_id}"


def find_title_step(template: str) -> str:
    return f"find:title:{template}"


def import_step(bin_path: tuple[str, ...], batch: int) -> str:
    return f"import:{'/'.join(bin_path)}:{batch}"


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


_ID_FORBIDDEN = re.compile(r"[/\\:]")


def _source_strings(source: Source) -> set[str]:
    """Every plan-supplied string of one source; its pool id may equal none of them (N4)."""
    return {s for s in (source.id, source.media, source.camera, source.recording_id) if s}


def _check_id_shape(unique_id: str, what: str, *, forbid: set[str]) -> None:
    """A real Resolve unique id never contains ``/``, ``\\`` or ``:`` and is never a string the
    plan supplied (a media path or pool name, a source id, a template name) -- that would be a
    name passed through as an id (R1/N4, the W01 shape)."""
    if not unique_id.strip() or _ID_FORBIDDEN.search(unique_id) or unique_id in forbid:
        raise CompileError(
            "forged_phase_result",
            f"{what} id {unique_id!r} is not a Resolve unique id (it is empty, path-shaped or a "
            "plan string)",
            "a phase result must carry the real pool unique ids the ops returned",
            value=unique_id,
        )


def plan_strings(plan: EditPlan) -> set[str]:
    """Every string the plan supplies that could be passed through as an id."""
    out = set(title_templates(plan)) | ({plan.template} if plan.template else set())
    out |= {plan.project, plan.timeline}
    for source in plan.sources:
        out |= _source_strings(source)
    return out


def check_branch_id(unique_id: str, plan: EditPlan) -> None:
    """The branch timeline id gets the same forged-id rule as pool and title ids (O1)."""
    _check_id_shape(unique_id, "branch timeline", forbid=plan_strings(plan))


def validate_prepare(prepare: PrepareResult, plan: EditPlan) -> None:
    _check_sha(prepare, plan)
    check_branch_id(prepare.branch.unique_id, plan)
    by_id = {s.id: s for s in plan.sources}
    templates = set(title_templates(plan))
    plan_names = templates | ({plan.template} if plan.template else set())
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
        plan_media = by_id[src.source_id].media
        # A pool id never equals a plan string for that source, nor contains / \\ : (N4).
        _check_id_shape(
            src.media.unique_id,
            f"source {src.source_id!r} pool",
            forbid=_source_strings(by_id[src.source_id]) | plan_names,
        )
        is_import = classify_media(plan_media) == "import"
        if is_import and path_key(src.file_path) != path_key(plan_media):
            raise CompileError(
                "forged_phase_result",
                f"source {src.source_id!r} imported {src.file_path!r}, not the plan's media "
                f"{plan_media!r}",
                "the import result must be of the plan's media for that source",
                source=src.source_id,
            )
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
        _check_id_shape(title.media.unique_id, f"title {title.template!r}", forbid=plan_names)


def validate_assemble(assemble: AssembleResult, plan: EditPlan) -> None:
    _check_sha(assemble, plan)
    check_branch_id(assemble.branch.unique_id, plan)
    for result in assemble.appends:
        if result.timeline.unique_id != assemble.branch.unique_id:
            raise CompileError(
                "phase_result_mismatch",
                f"an append result is from timeline {result.timeline.unique_id!r}, not the "
                f"assemble branch {assemble.branch.unique_id!r}",
                "every result must come from the same branch timeline",
            )


def __getattr__(name: str) -> Any:
    # The job builder needs the compiler, which imports this module; serve it lazily (PEP 562)
    # so ``from aar.plan.apply.inputs import phase_result_from_job`` keeps working.
    if name == "phase_result_from_job":
        from aar.plan.apply.jobs import phase_result_from_job

        return phase_result_from_job
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
