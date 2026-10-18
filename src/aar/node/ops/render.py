"""Render ops: build-and-start a render atomically, poll it, and delete it.

``render.build`` does the whole Deliver setup in ONE hot op that holds the Resolve instance:
it makes the job's own timeline current, snapshots the Deliver settings into a temporary
preset, loads the delivery preset, sets the output path/name/range/format, adds the render job,
starts it, and then restores the Deliver settings -- all before it returns. Because it is
atomic, two renders in one project can never cross-wire their settings (the broker runs one op
at a time), the human's Deliver settings are never left clobbered between ops, and the broker
never snapshots a clobbered state as the "human" baseline.

The snapshot preset is the durable record of the pre-job Deliver settings: if a crash leaves
it behind, a re-run loads it first to recover the true originals, then deletes it, so the
settings can never be lost and the preset never leaks.

``render.build`` creates a FRESH render job every time and reports its Resolve ``JobId``; it
never adopts an existing queue entry by name, so it can never hijack a human's render.
``render.delete_job`` only ever removes the job this aar job created (its recorded id).
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

_ALREADY_RUNNING = frozenset({"Rendering", "Complete"})


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


def _job_list(project: Any) -> list[dict[str, Any]]:
    return [dict(info) for info in (project.GetRenderJobList() or [])]


def _status(project: Any, job_id: str) -> dict[str, Any]:
    return dict(project.GetRenderJobStatus(job_id) or {})


def _restore_deliver(
    project: Any, preset: str, fmt: str | None, codec: str | None, mode: int | None
) -> None:
    """Put the pre-job Deliver settings back from the snapshot preset, then delete the preset.

    Always called (even on failure), so a render never leaves the Deliver page clobbered. A
    missing preset is an error: it is the only record of the originals and must be there.
    """
    if preset not in (project.GetRenderPresetList() or []):
        raise OpError(
            "snapshot_preset_missing",
            f"the Deliver snapshot preset {preset!r} is gone; the original settings cannot be "
            "restored",
            "do not delete aar-* render presets while a render job is running; set the Deliver "
            "settings back by hand",
            "deliver-clobber",
        )
    if not project.LoadRenderPreset(preset):
        raise OpError(
            "preset_load_failed",
            f"LoadRenderPreset({preset!r}) returned False while restoring the Deliver settings",
            "load the preset by hand on the Deliver page",
            "deliver-clobber",
        )
    if fmt is not None and codec is not None and not project.SetCurrentRenderFormatAndCodec(fmt, codec):
        raise OpError(
            "render_format_failed",
            f"SetCurrentRenderFormatAndCodec({fmt!r}, {codec!r}) returned False while restoring",
            "set the format and codec back by hand",
            "deliver-clobber",
        )
    if mode is not None and not project.SetCurrentRenderMode(mode):
        raise OpError(
            "render_mode_failed",
            f"SetCurrentRenderMode({mode!r}) returned False while restoring",
            "set the render mode back by hand",
        )
    if not project.DeleteRenderPreset(preset):
        raise OpError(
            "preset_delete_failed",
            f"DeleteRenderPreset({preset!r}) returned False",
            "delete the temporary aar- preset by hand",
        )


# -- render.build ------------------------------------------------------------------------------


class BuildParams(OpModel):
    timeline: TimelineRef
    target_dir: str = Field(min_length=1)
    custom_name: str = Field(min_length=1)
    snapshot_preset: str = Field(min_length=1, description="Temp preset holding the originals.")
    preset: str | None = Field(default=None, min_length=1, description="Delivery preset to load.")
    render_format: str | None = Field(default=None, min_length=1)
    render_codec: str | None = Field(default=None, min_length=1)
    render_mode: int | None = Field(default=None, ge=0, le=1)
    mark_in: int | None = Field(default=None, ge=0)
    mark_out: int | None = Field(default=None, ge=0)


class BuildResult(OpModel):
    render_job_id: str
    timeline: TimelineIdentity
    deliver_restored: bool
    target_dir: str | None
    output_filename: str | None
    output_path: str | None
    frame_count: int | None
    frame_rate: float | None
    duration_seconds: float | None


def _measured_from_info(info: dict[str, Any]) -> dict[str, Any]:
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
    duration = frame_count / frame_rate if frame_count is not None and frame_rate else None
    return {
        "target_dir": target_dir,
        "output_filename": output_filename,
        "output_path": output_path,
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "duration_seconds": duration,
    }


@op(
    "render.build",
    params=BuildParams,
    result=BuildResult,
    mutates=True,
    needs_current_timeline=True,
    traps=("deliver-clobber", "render-uses-original-media"),
)
def render_build(env: OpEnv, params: BuildParams) -> BuildResult:
    """Make the job's timeline current, snapshot Deliver, configure, add and start a fresh render
    job, then restore Deliver -- atomically, so nothing else can interleave."""
    timeline = env.current_timeline(params.timeline)  # the job's own timeline (never the editor's)
    project = env.project()
    snap = params.snapshot_preset
    presets = list(project.GetRenderPresetList() or [])
    if snap in presets:
        # A prior attempt crashed mid-build: the preset is the true pre-job settings. Recover
        # them before reading the originals, so a crash can never lose the human's settings.
        if not project.LoadRenderPreset(snap):
            raise OpError(
                "preset_load_failed",
                f"LoadRenderPreset({snap!r}) returned False recovering the pre-job settings",
                "load the aar- snapshot preset by hand",
                "deliver-clobber",
            )
    elif not project.SaveAsNewRenderPreset(snap):
        raise OpError(
            "preset_save_failed",
            f"SaveAsNewRenderPreset({snap!r}) returned False",
            "choose a preset name that is not already taken",
            "deliver-clobber",
        )
    fmt0, codec0, mode0 = _deliver(project)
    try:
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
        info = matches[0]
        if _str(_status(project, job_id).get("JobStatus")) not in _ALREADY_RUNNING:
            if not project.StartRendering([job_id], False):
                raise OpError(
                    "start_rendering_failed",
                    f"StartRendering([{job_id!r}]) returned False",
                    "check no other render is in progress and the output path is writable",
                )
            after = _str(_status(project, job_id).get("JobStatus"))
            if after not in _ALREADY_RUNNING and not project.IsRenderingInProgress():
                raise OpError(
                    "render_not_started",
                    f"render job {job_id!r} is {after!r} and no render is in progress",
                    "start the render by hand on the Deliver page and poll again",
                )
    finally:
        # Restore the Deliver settings no matter what, so a failure never leaves them clobbered.
        _restore_deliver(project, snap, fmt0, codec0, mode0)
    return BuildResult(
        render_job_id=job_id,
        timeline=env.identity(timeline),
        deliver_restored=True,
        **_measured_from_info(info),
    )


# -- render.poll -------------------------------------------------------------------------------


class RenderJobParams(OpModel):
    render_job_id: str = Field(min_length=1)


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
    measured = _measured_from_info(jobs.get(params.render_job_id, {}))
    return PollResult(
        render_job_id=params.render_job_id,
        status=job_status,
        completion=_int(status.get("CompletionPercentage")),
        error=_str(status.get("Error")),
        is_rendering=bool(project.IsRenderingInProgress()),
        render_time_ms=_int(status.get("TimeTakenToRenderInMs")),
        **measured,
    )


# -- render.delete_job -------------------------------------------------------------------------


class DeleteJobResult(OpModel):
    render_job_id: str
    deleted: bool


@op("render.delete_job", params=RenderJobParams, result=DeleteJobResult, mutates=True)
def render_delete_job(env: OpEnv, params: RenderJobParams) -> DeleteJobResult:
    """Remove the render job this aar job created (its recorded id). A job already gone counts
    as deleted. The runner only ever passes the id ``render.build`` returned, so this never
    touches a job the agent did not create."""
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
