"""Render ops: build-and-start a render atomically, poll it, and delete it.

``render.build`` does the whole Deliver setup in ONE hot op that holds the Resolve instance:
it makes the job's own timeline current, snapshots the Deliver settings into a temporary
preset, loads the delivery preset, sets the output path/name/range/format, adds the render job,
restores the Deliver settings, and then starts the render -- all before it returns. Because it
is atomic, two renders in one project can never cross-wire their settings (the broker runs one
op at a time), the human's Deliver settings are never left clobbered between ops, and the
broker never snapshots a clobbered state as the "human" baseline. The Deliver settings are put
back BEFORE ``StartRendering``: a queued render job keeps its own settings, so the restore never
races a running render.

Crash safety (the op runs only inside a node job, which installs a journal, see
:mod:`aar.node.ops.journal`). Every side effect is journaled to the job's durable store as it
happens, so a resumed job knows exactly what an interrupted attempt left in Resolve:

- the snapshot preset's name, BEFORE it is saved. A preset is only ever loaded or deleted when
  this job's journal says the job created it; a name that is already taken (by a human) is
  skipped, never loaded or deleted. A crashed attempt's snapshot holds the true pre-job
  settings: the next attempt loads it first, so they are never lost;
- the queue's job ids, BEFORE ``AddRenderJob``, and the new ``JobId`` right after it. A resumed
  build ADOPTS the journaled ``JobId`` (verified in the queue) instead of adding a second render
  job; a crash inside ``AddRenderJob`` itself, which cannot be attributed, fails visibly rather
  than guessing, so a human's render job is never adopted, started or deleted.

On any failure the render job this attempt added (or adopted) is deleted by its ``JobId`` unless
it is already rendering, the Deliver settings are restored, and the PRIMARY error is raised with
any cleanup or restore problem attached in its detail. A restore failure after the job was added
does not fail the render: it is reported in the result (and journaled, so the runner alerts)
while the render goes on.

``render.delete_job`` only ever removes the job this aar job created (its recorded id).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aar.node.broker import HotOpFailure
from aar.node.ops.framework import (
    OpEnv,
    OpError,
    OpModel,
    TimelineIdentity,
    TimelineRef,
    op,
)
from aar.node.ops.journal import Journal, current_journal

RENDER_GUARD_EXEMPTION = (
    "render-queue: reads the timeline, never edits its content; touches only render jobs "
    "its own aar job created (by recorded JobId) and restores the Deliver settings"
)
"""The branch-guard exemption of the render ops that change Resolve (S7 x S11 policy P3)."""

_ALREADY_RUNNING = frozenset({"Rendering", "Complete"})
_READY = frozenset({"Ready"})
_SNAPSHOT_NAMES = 10
"""How many snapshot preset names (``base``, ``base-2`` ...) to try before giving up."""


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


def _job_ids(project: Any) -> list[str]:
    return [str(j.get("JobId")) for j in _job_list(project)]


def _presets(project: Any) -> list[str]:
    return [str(p) for p in (project.GetRenderPresetList() or [])]


def _status(project: Any, job_id: str) -> dict[str, Any]:
    return dict(project.GetRenderJobStatus(job_id) or {})


def _error_dict(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, HotOpFailure):
        return {"code": exc.code, "message": exc.message}
    return {"code": "op_raised", "message": f"{type(exc).__name__}: {exc}"}


def _with_detail(exc: Exception, extra: dict[str, Any]) -> HotOpFailure:
    """The primary failure, with cleanup and restore problems attached (never masking it)."""
    if isinstance(exc, HotOpFailure):
        return HotOpFailure(exc.code, exc.message, {**exc.detail, **extra})
    return HotOpFailure(
        "op_raised",
        f"{type(exc).__name__}: {exc}",
        {
            "exception": type(exc).__name__,
            "fix_hint": "inspect the Deliver page and the render queue",
            **extra,
        },
    )


def _note(journal: Journal, event: str, data: dict[str, Any], problems: dict[str, Any]) -> bool:
    """Journal on a cleanup path: a journal failure is reported, never raised over the primary."""
    try:
        return journal(event, data)
    except Exception as exc:
        problems.setdefault("journal_errors", []).append({"event": event, **_error_dict(exc)})
        return False


def _aborted(stage: str) -> OpError:
    return OpError(
        "render_build_aborted",
        f"the job no longer wants this render ({stage}); the build undid its own side effects",
        "nothing to do: the job was cancelled or failed while its render was being built",
    )


def _restore_deliver(
    project: Any, preset: str, fmt: str | None, codec: str | None, mode: int | None
) -> None:
    """Put the pre-job Deliver settings back from the snapshot preset, then delete the preset.

    A missing preset is an error: it is the only record of the originals and must be there. A
    preset that cannot be loaded is kept (it still holds the originals).
    """
    if preset not in _presets(project):
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
            f"load the preset {preset!r} by hand on the Deliver page (it holds the original "
            "settings), then delete it",
            "deliver-clobber",
        )
    if (
        fmt is not None
        and codec is not None
        and not project.SetCurrentRenderFormatAndCodec(fmt, codec)
    ):
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


def _try_restore(
    project: Any,
    journal: Journal,
    preset: str,
    original: tuple[str | None, str | None, int | None],
    problems: dict[str, Any],
) -> dict[str, str] | None:
    """Restore the Deliver settings; return the error (as a dict) instead of raising it."""
    try:
        _restore_deliver(project, preset, *original)
    except Exception as exc:
        error = _error_dict(exc)
        _note(journal, "restore_failed", {"snapshot": preset, "restore_error": error}, problems)
        return error
    _note(journal, "snapshot_deleted", {"snapshot": preset}, problems)
    return None


def _discard_ours(project: Any, journal: Journal, job_id: str) -> dict[str, Any]:
    """Delete the render job THIS job added or adopted (by its id), unless it is rendering.

    Never touches any other job. A job that cannot be deleted is journaled as orphaned, so
    the runner raises a visible alert naming its ``JobId``.
    """
    report: dict[str, Any] = {"render_job_id": job_id}
    try:
        status = _str(_status(project, job_id).get("JobStatus"))
        if job_id not in _job_ids(project):
            report["deleted"] = True
            report["note"] = "already gone from the queue"
            _note(journal, "deleted", {"render_job_id": job_id}, report)
            return report
        if status == "Rendering":
            report.update(deleted=False, orphaned=True, status=status)
        elif project.DeleteRenderJob(job_id) and job_id not in _job_ids(project):
            report["deleted"] = True
            _note(journal, "deleted", {"render_job_id": job_id}, report)
            return report
        else:
            report.update(deleted=False, orphaned=True, status=status)
    except Exception as exc:
        report.update(deleted=False, orphaned=True, error=_error_dict(exc))
    _note(journal, "orphaned", {"render_job_id": job_id, "cleanup": dict(report)}, report)
    return report


def _free_snapshot_name(base: str, presets: list[str]) -> str:
    """The first of ``base``, ``base-2`` ... that no preset uses. A taken name is not ours."""
    taken = set(presets)
    for i in range(1, _SNAPSHOT_NAMES + 1):
        name = base if i == 1 else f"{base}-{i}"
        if name not in taken:
            return name
    raise OpError(
        "snapshot_preset_names_taken",
        f"render presets {base!r} .. {base}-{_SNAPSHOT_NAMES} all exist and none was created by "
        "this job",
        "rename or delete those presets; this job never loads or deletes a preset it did not "
        "create",
        "deliver-clobber",
    )


def _listed_once(project: Any, job_id: str, code: str) -> dict[str, Any]:
    matches = [j for j in _job_list(project) if str(j.get("JobId")) == job_id]
    if len(matches) != 1:
        raise OpError(
            code,
            f"render job {job_id!r} is listed {len(matches)} times in the render queue, not once",
            "inspect the render queue on the Deliver page; resubmit the render if its job is gone",
        )
    return matches[0]


# -- render.build ------------------------------------------------------------------------------


class BuildParams(OpModel):
    timeline: TimelineRef
    target_dir: str = Field(min_length=1)
    custom_name: str = Field(min_length=1)
    snapshot_preset: str = Field(min_length=1, description="Base name for the snapshot preset.")
    preset: str | None = Field(default=None, min_length=1, description="Delivery preset to load.")
    render_format: str | None = Field(default=None, min_length=1)
    render_codec: str | None = Field(default=None, min_length=1)
    render_mode: int | None = Field(default=None, ge=0, le=1)
    mark_in: int | None = Field(default=None, ge=0)
    mark_out: int | None = Field(default=None, ge=0)
    adopt_render_job_id: str | None = Field(
        default=None, min_length=1, description="A prior attempt added this render job: reuse it."
    )
    prior_snapshot: str | None = Field(
        default=None, min_length=1, description="A prior attempt saved this snapshot preset."
    )
    prior_pre_ids: tuple[str, ...] | None = Field(
        default=None, description="A prior attempt died inside AddRenderJob; the ids before it."
    )


class BuildResult(OpModel):
    render_job_id: str
    timeline: TimelineIdentity
    adopted: bool
    deliver_restored: bool
    restore_error: dict[str, str] | None
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


def _configure(project: Any, params: BuildParams) -> None:
    """Load the delivery preset and set the job's output, range, format and mode."""
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


def _add(project: Any, journal: Journal) -> str:
    """Add the render job, journaling the queue before and the new ``JobId`` right after."""
    pre_ids = _job_ids(project)
    if not journal("adding", {"pre_ids": pre_ids}):
        raise _aborted("before AddRenderJob")
    try:
        added = project.AddRenderJob()
    except Exception as exc:
        # It may or may not have added a job; never guess which new job is ours.
        try:
            new = sorted(set(_job_ids(project)) - set(pre_ids))
        except Exception:  # the queue cannot be read either; report what we know
            new = []
        raise _with_detail(exc, {"possible_orphans": new}) from exc
    if not added:
        raise OpError(
            "add_render_job_failed",
            "AddRenderJob() returned no job id",
            "check the render settings are valid and the timeline has frames to render",
        )
    return str(added)


@op(
    "render.build",
    params=BuildParams,
    result=BuildResult,
    mutates=True,
    needs_current_timeline=True,
    traps=("deliver-clobber", "render-uses-original-media"),
    guard_exemption=RENDER_GUARD_EXEMPTION,
)
def render_build(env: OpEnv, params: BuildParams) -> BuildResult:
    """Make the job's timeline current, snapshot Deliver, configure and add a render job (or
    adopt the one a crashed attempt added), restore Deliver, then start it -- atomically."""
    journal = current_journal()
    if journal is None:
        raise OpError(
            "render_needs_job",
            "render.build runs only inside a node job, which journals its side effects so a "
            "crash can never duplicate or orphan a render",
            "submit a 'render' job instead of running render.build directly",
        )
    if not journal("build_started", {}):
        raise _aborted("before any change")
    timeline = env.current_timeline(params.timeline)  # the job's own timeline (never the editor's)
    project = env.project()
    presets = _presets(project)
    if params.prior_snapshot is not None and params.prior_snapshot in presets:
        # A crashed attempt of THIS job saved it (journaled before the save): it is the true
        # pre-job settings. Recover them before reading the originals.
        snap = params.prior_snapshot
        if not project.LoadRenderPreset(snap):
            raise OpError(
                "preset_load_failed",
                f"LoadRenderPreset({snap!r}) returned False recovering the pre-job settings",
                f"load the aar- snapshot preset {snap!r} by hand, then delete it",
                "deliver-clobber",
            )
        journal("snapshot_recovered", {"snapshot": snap})
    else:
        snap = _free_snapshot_name(params.snapshot_preset, presets)
        journal("snapshot_saving", {"snapshot": snap})  # before the save: a crash finds it ours
        if not project.SaveAsNewRenderPreset(snap):
            raise OpError(
                "preset_save_failed",
                f"SaveAsNewRenderPreset({snap!r}) returned False",
                "check the Deliver page can save presets",
                "deliver-clobber",
            )
    original = _deliver(project)
    adopted = params.adopt_render_job_id is not None
    job_id: str | None = None
    restored = False
    restore_error: dict[str, str] | None = None
    problems: dict[str, Any] = {}
    try:
        if params.adopt_render_job_id is not None:
            job_id = params.adopt_render_job_id
            info = _listed_once(project, job_id, "render_job_missing")
        else:
            if params.prior_pre_ids is not None:
                new = sorted(set(_job_ids(project)) - set(params.prior_pre_ids))
                if new:
                    raise OpError(
                        "render_build_ambiguous",
                        "a prior attempt died inside AddRenderJob and the render queue has "
                        f"gained job(s) {new} since; which (if any) is this job's cannot be told",
                        "check the render queue: delete a Ready job this aar job added (same "
                        "timeline and output) if there is one, then resubmit the render",
                        candidates=new,
                    )
            _configure(project, params)
            job_id = _add(project, journal)
            wanted = journal("added", {"render_job_id": job_id})
            info = _listed_once(project, job_id, "add_render_job_unverified")
            if not wanted:
                raise _aborted("after AddRenderJob")
        # Put Deliver back BEFORE starting: the queued job keeps its own settings.
        restore_error = _try_restore(project, journal, snap, original, problems)
        restored = True
        status = _str(_status(project, job_id).get("JobStatus"))
        must_start = status in _READY if adopted else status not in _ALREADY_RUNNING
        if must_start:
            if not journal("starting", {"render_job_id": job_id}):
                raise _aborted("before StartRendering")
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
        journal("started", {"render_job_id": job_id, "status": status})
    except Exception as exc:
        extra: dict[str, Any] = dict(problems)
        if job_id is not None:
            extra["cleanup"] = _discard_ours(project, journal, job_id)
        if not restored:
            error = _try_restore(project, journal, snap, original, extra)
            if error is not None:
                extra["restore_error"] = error
        raise _with_detail(exc, extra) from exc
    return BuildResult(
        render_job_id=job_id,
        timeline=env.identity(timeline),
        adopted=adopted,
        deliver_restored=restore_error is None,
        restore_error=restore_error,
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


@op(
    "render.delete_job",
    params=RenderJobParams,
    result=DeleteJobResult,
    mutates=True,
    guard_exemption=RENDER_GUARD_EXEMPTION,
)
def render_delete_job(env: OpEnv, params: RenderJobParams) -> DeleteJobResult:
    """Remove the render job this aar job created (its recorded id). A job already gone counts
    as deleted. It runs only inside a node job, whose journal must confirm the id is the one
    this job recorded, so it can never delete a human's render job."""
    journal = current_journal()
    if journal is None:
        raise OpError(
            "render_needs_job",
            "render.delete_job runs only inside a node job, which proves the render job is its own",
            "submit a 'render' job; it deletes its own completed render job",
        )
    if not journal("deleting", {"render_job_id": params.render_job_id}):
        raise OpError(
            "render_job_not_ours",
            f"render job {params.render_job_id!r} is not the one this aar job created",
            "a job only ever deletes the render job its own build recorded",
        )
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
