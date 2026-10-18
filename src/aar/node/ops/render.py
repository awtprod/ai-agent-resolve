"""Render ops: snapshot and restore Deliver settings, configure, add, start and poll a render.

These are the hot ops the S7 render job is built from. They use the 21.1 render API
(``SaveAsNewRenderPreset`` / ``LoadRenderPreset`` / ``SetRenderSettings`` / ``AddRenderJob`` /
``StartRendering`` / ``GetRenderJobStatus`` / ``GetRenderJobList`` / ``DeleteRenderJob``),
check every return value, and report values measured back from Resolve.

Each op is idempotent so the job runner can re-run a step after a restart without doubling
work: the snapshot skips a preset that already exists, ``render.add_job`` reuses an existing
job for the same output name instead of adding a second, ``render.start`` is a no-op once the
job is rendering, and ``render.restore_deliver`` / ``render.delete_job`` treat already-done as
done. None of these ops changes the timeline's content; Deliver settings are saved before a
preset loads and restored afterwards (DESIGN §5.3, trap ``deliver-clobber``).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aar.node.ops.framework import (
    OpEnv,
    OpError,
    OpModel,
    TimelineIdentity,
    TimelineRef,
    op,
)


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _deliver(project: Any) -> tuple[str | None, str | None, int | None]:
    fmt = project.GetCurrentRenderFormatAndCodec() or {}
    mode = project.GetCurrentRenderMode()
    return (
        _str(fmt.get("format")),
        _str(fmt.get("codec")),
        (_int(mode) if mode is not None else None),
    )


# -- render.deliver_snapshot -------------------------------------------------------------------


class DeliverSnapshotParams(OpModel):
    preset_name: str = Field(min_length=1, description="Temporary preset to hold the settings.")


class DeliverSnapshotResult(OpModel):
    preset: str
    created: bool
    format: str | None
    codec: str | None
    mode: int | None


@op(
    "render.deliver_snapshot",
    params=DeliverSnapshotParams,
    result=DeliverSnapshotResult,
    mutates=True,
    traps=("deliver-clobber",),
)
def render_deliver_snapshot(env: OpEnv, params: DeliverSnapshotParams) -> DeliverSnapshotResult:
    """Save the current Deliver settings into a temporary render preset, so they can be put
    back after a delivery preset clobbers them (trap deliver-clobber)."""
    project = env.project()
    fmt, codec, mode = _deliver(project)
    presets = list(project.GetRenderPresetList() or [])
    created = False
    if params.preset_name not in presets:
        if not project.SaveAsNewRenderPreset(params.preset_name):
            raise OpError(
                "preset_save_failed",
                f"SaveAsNewRenderPreset({params.preset_name!r}) returned False",
                "choose a preset name that is not already taken",
                "deliver-clobber",
            )
        if params.preset_name not in (project.GetRenderPresetList() or []):
            raise OpError(
                "preset_save_unverified",
                f"preset {params.preset_name!r} is not listed after SaveAsNewRenderPreset",
                "save the Deliver settings as a preset by hand",
                "deliver-clobber",
            )
        created = True
    return DeliverSnapshotResult(
        preset=params.preset_name, created=created, format=fmt, codec=codec, mode=mode
    )


# -- render.configure --------------------------------------------------------------------------


class ConfigureParams(OpModel):
    target_dir: str = Field(min_length=1)
    custom_name: str = Field(min_length=1)
    preset: str | None = Field(default=None, min_length=1)
    render_format: str | None = Field(default=None, min_length=1)
    render_codec: str | None = Field(default=None, min_length=1)
    render_mode: int | None = Field(default=None, ge=0, le=1)
    mark_in: int | None = Field(default=None, ge=0)
    mark_out: int | None = Field(default=None, ge=0)


class ConfigureResult(OpModel):
    preset_loaded: str | None
    target_dir: str
    custom_name: str
    format: str | None
    codec: str | None
    mode: int | None
    select_all_frames: bool


@op(
    "render.configure",
    params=ConfigureParams,
    result=ConfigureResult,
    mutates=True,
    traps=("deliver-clobber",),
)
def render_configure(env: OpEnv, params: ConfigureParams) -> ConfigureResult:
    """Load the delivery preset (if any), then set the output path, name and range. The preset
    is loaded first because it clobbers the rest of the Deliver settings (trap deliver-clobber)."""
    project = env.project()
    if params.preset is not None and not project.LoadRenderPreset(params.preset):
        raise OpError(
            "preset_load_failed",
            f"LoadRenderPreset({params.preset!r}) returned False",
            "check the delivery preset name exactly matches one Resolve lists",
            "deliver-clobber",
        )
    select_all = params.mark_in is None or params.mark_out is None
    settings: dict[str, Any] = {
        "TargetDir": params.target_dir,
        "CustomName": params.custom_name,
        "SelectAllFrames": select_all,
    }
    if not select_all:
        settings["MarkIn"] = params.mark_in
        settings["MarkOut"] = params.mark_out
    if not project.SetRenderSettings(settings):
        raise OpError(
            "render_settings_failed",
            f"SetRenderSettings({sorted(settings)}) returned False",
            "check the target directory is writable and the mark range is inside the timeline",
        )
    if (
        params.render_format is not None
        and params.render_codec is not None
        and not project.SetCurrentRenderFormatAndCodec(params.render_format, params.render_codec)
    ):
        raise OpError(
            "render_format_failed",
            f"SetCurrentRenderFormatAndCodec({params.render_format!r}, "
            f"{params.render_codec!r}) returned False",
            "check the format and codec names against the Deliver page",
        )
    if params.render_mode is not None and not project.SetCurrentRenderMode(params.render_mode):
        raise OpError(
            "render_mode_failed",
            f"SetCurrentRenderMode({params.render_mode!r}) returned False",
            "render mode is 0 (individual clips) or 1 (single clip)",
        )
    fmt, codec, mode = _deliver(project)
    return ConfigureResult(
        preset_loaded=params.preset,
        target_dir=params.target_dir,
        custom_name=params.custom_name,
        format=fmt,
        codec=codec,
        mode=mode,
        select_all_frames=select_all,
    )


# -- render.add_job ----------------------------------------------------------------------------


class AddJobParams(OpModel):
    timeline: TimelineRef
    custom_name: str = Field(min_length=1, description="Output name; also the idempotency key.")


class AddJobResult(OpModel):
    render_job_id: str
    created: bool
    timeline: TimelineIdentity
    job: dict[str, Any]


def _job_list(project: Any) -> list[dict[str, Any]]:
    return [dict(info) for info in (project.GetRenderJobList() or [])]


@op(
    "render.add_job",
    params=AddJobParams,
    result=AddJobResult,
    mutates=True,
    needs_current_timeline=True,
)
def render_add_job(env: OpEnv, params: AddJobParams) -> AddJobResult:
    """Add a render job for the current timeline, or reuse the one an earlier attempt added."""
    timeline = env.current_timeline(params.timeline)
    project = env.project()
    existing = [j for j in _job_list(project) if j.get("OutputFilename") == params.custom_name]
    if existing:
        job = existing[0]
        return AddJobResult(
            render_job_id=str(job.get("JobId")),
            created=False,
            timeline=env.identity(timeline),
            job=job,
        )
    job_id = project.AddRenderJob()
    if not job_id:
        raise OpError(
            "add_render_job_failed",
            "AddRenderJob() returned no job id",
            "check the render settings are valid and the timeline has frames to render",
        )
    job_id = str(job_id)
    matches = [j for j in _job_list(project) if str(j.get("JobId")) == job_id]
    if len(matches) != 1:
        raise OpError(
            "add_render_job_unverified",
            f"render job {job_id!r} is not listed exactly once after AddRenderJob",
            "inspect the render queue on the Deliver page",
        )
    return AddJobResult(
        render_job_id=job_id, created=True, timeline=env.identity(timeline), job=matches[0]
    )


# -- render.start ------------------------------------------------------------------------------


class RenderJobParams(OpModel):
    render_job_id: str = Field(min_length=1)


class StartResult(OpModel):
    render_job_id: str
    status: str | None
    already: bool


_ALREADY_RUNNING = frozenset({"Rendering", "Complete"})


def _status(project: Any, job_id: str) -> dict[str, Any]:
    return dict(project.GetRenderJobStatus(job_id) or {})


@op("render.start", params=RenderJobParams, result=StartResult, mutates=True)
def render_start(env: OpEnv, params: RenderJobParams) -> StartResult:
    """Start the render job. A no-op if it is already rendering or complete (idempotent)."""
    project = env.project()
    before = _str(_status(project, params.render_job_id).get("JobStatus"))
    if before in _ALREADY_RUNNING:
        return StartResult(render_job_id=params.render_job_id, status=before, already=True)
    if not project.StartRendering([params.render_job_id], False):
        raise OpError(
            "start_rendering_failed",
            f"StartRendering([{params.render_job_id!r}]) returned False",
            "check the render job still exists in the queue and its output path is writable",
        )
    after = _str(_status(project, params.render_job_id).get("JobStatus"))
    if after not in _ALREADY_RUNNING and not project.IsRenderingInProgress():
        raise OpError(
            "render_not_started",
            f"render job {params.render_job_id!r} is {after!r} and no render is in progress",
            "start the render by hand on the Deliver page and poll again",
        )
    return StartResult(render_job_id=params.render_job_id, status=after, already=False)


# -- render.poll -------------------------------------------------------------------------------


class PollResult(OpModel):
    render_job_id: str
    status: str
    completion: int | None
    error: str | None
    is_rendering: bool
    output_path: str | None
    target_dir: str | None
    output_filename: str | None
    frame_count: int | None
    frame_rate: float | None
    duration_seconds: float | None
    render_time_ms: int | None


@op("render.poll", params=RenderJobParams, result=PollResult, mutates=False)
def render_poll(env: OpEnv, params: RenderJobParams) -> PollResult:
    """Read a render job's status and its measured output (path, frames, duration)."""
    project = env.project()
    status = _status(project, params.render_job_id)
    job_status = _str(status.get("JobStatus"))
    if job_status is None:
        raise OpError(
            "render_job_unknown",
            f"GetRenderJobStatus({params.render_job_id!r}) returned no status",
            "the render job may have been removed; list the queue and resubmit if needed",
        )
    jobs = {str(j.get("JobId")): j for j in _job_list(project)}
    info = jobs.get(params.render_job_id, {})
    target_dir = _str(info.get("TargetDir"))
    output_filename = _str(info.get("OutputFilename"))
    output_path = (
        f"{target_dir.rstrip('/')}/{output_filename}"
        if target_dir is not None and output_filename is not None
        else None
    )
    mark_in = _int(info.get("MarkIn"))
    mark_out = _int(info.get("MarkOut"))
    frame_count = (mark_out - mark_in + 1) if mark_in is not None and mark_out is not None else None
    frame_rate = _float(info.get("FrameRate"))
    duration_seconds = frame_count / frame_rate if frame_count is not None and frame_rate else None
    return PollResult(
        render_job_id=params.render_job_id,
        status=job_status,
        completion=_int(status.get("CompletionPercentage")),
        error=_str(status.get("Error")),
        is_rendering=bool(project.IsRenderingInProgress()),
        output_path=output_path,
        target_dir=target_dir,
        output_filename=output_filename,
        frame_count=frame_count,
        frame_rate=frame_rate,
        duration_seconds=duration_seconds,
        render_time_ms=_int(status.get("TimeTakenToRenderInMs")),
    )


# -- render.restore_deliver --------------------------------------------------------------------


class RestoreParams(OpModel):
    preset: str = Field(min_length=1)
    format: str | None = None
    codec: str | None = None
    mode: int | None = None


class RestoreResult(OpModel):
    restored: bool
    format: str | None
    codec: str | None
    mode: int | None


@op(
    "render.restore_deliver",
    params=RestoreParams,
    result=RestoreResult,
    mutates=True,
    traps=("deliver-clobber",),
)
def render_restore_deliver(env: OpEnv, params: RestoreParams) -> RestoreResult:
    """Load the snapshot preset back, reset the format and codec it may have dropped, and delete
    the temporary preset. Idempotent: an already-deleted preset counts as restored."""
    project = env.project()
    presets = list(project.GetRenderPresetList() or [])
    restored = False
    if params.preset in presets:
        if not project.LoadRenderPreset(params.preset):
            raise OpError(
                "preset_load_failed",
                f"LoadRenderPreset({params.preset!r}) returned False while restoring",
                "load the preset by hand on the Deliver page",
                "deliver-clobber",
            )
        restored = True
    # Reset format/codec/mode even after a preset round trip that may have dropped them.
    if (
        params.format is not None
        and params.codec is not None
        and not project.SetCurrentRenderFormatAndCodec(params.format, params.codec)
    ):
        raise OpError(
            "render_format_failed",
            f"SetCurrentRenderFormatAndCodec({params.format!r}, {params.codec!r}) returned "
            "False while restoring the Deliver settings",
            "set the format and codec back by hand",
            "deliver-clobber",
        )
    if params.mode is not None and not project.SetCurrentRenderMode(params.mode):
        raise OpError(
            "render_mode_failed",
            f"SetCurrentRenderMode({params.mode!r}) returned False while restoring",
            "set the render mode back by hand",
        )
    if restored and not project.DeleteRenderPreset(params.preset):
        raise OpError(
            "preset_delete_failed",
            f"DeleteRenderPreset({params.preset!r}) returned False",
            "delete the temporary preset by hand",
        )
    fmt, codec, mode = _deliver(project)
    return RestoreResult(restored=restored, format=fmt, codec=codec, mode=mode)


# -- render.delete_job -------------------------------------------------------------------------


class DeleteJobResult(OpModel):
    render_job_id: str
    deleted: bool


@op("render.delete_job", params=RenderJobParams, result=DeleteJobResult, mutates=True)
def render_delete_job(env: OpEnv, params: RenderJobParams) -> DeleteJobResult:
    """Remove a render job from the queue. Idempotent: a job already gone counts as deleted."""
    project = env.project()
    present = any(str(j.get("JobId")) == params.render_job_id for j in _job_list(project))
    if not present:
        return DeleteJobResult(render_job_id=params.render_job_id, deleted=False)
    if not project.DeleteRenderJob(params.render_job_id):
        raise OpError(
            "delete_render_job_failed",
            f"DeleteRenderJob({params.render_job_id!r}) returned False",
            "remove the render job from the queue by hand",
        )
    if any(str(j.get("JobId")) == params.render_job_id for j in _job_list(project)):
        raise OpError(
            "delete_render_job_unverified",
            f"render job {params.render_job_id!r} is still listed after DeleteRenderJob",
            "remove it from the queue by hand",
        )
    return DeleteJobResult(render_job_id=params.render_job_id, deleted=True)
