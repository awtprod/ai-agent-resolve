r"""The render job: snapshot Deliver settings, submit a render, poll it, and restore.

A render is a hot op and queues per project (DESIGN R4). It cannot run inside a single MCP
call (F9), so it is a job: the runner polls ``render.poll`` across pumps, yielding the broker
to other projects between polls, until the render job in Resolve finishes.

The step is one resumable state machine whose phase lives in the checkpoint cursor:

    snapshot -> configure -> add -> start -> poll -> finalize_ok -> cleanup -> (done)
                                                \-> finalize_fail -> (fail)

Deliver settings are recorded before the delivery preset loads and restored afterwards
(DESIGN §5.3), on the success path and on every failure path. The restore reuses the same
save/load/delete render-preset mechanism the broker uses for a borrowed human context.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aar.node.locator import ProjectLocator
from aar.node.ops.framework import TimelineRef

from .model import Done, Fail, Pause, Step, StepAction, StepContext, StructuredError, Submit

DEFAULT_RENDER_TIMEOUT = 7200.0
"""A render may legitimately take a long time; the step's whole-step watchdog deadline."""

# GetRenderJobStatus 'JobStatus' values, grouped by what they mean for the poll loop.
_DONE_OK = frozenset({"Complete"})
_DONE_FAIL = frozenset(
    {"Cancelled", "Background Render Cancelled", "Failed", "Remote Render Cancelled"}
)


class RenderParams(BaseModel):
    """What to render and where. The timeline must carry its unique id (a render mutates)."""

    model_config = ConfigDict(extra="forbid")

    timeline: TimelineRef
    target_dir: str = Field(min_length=1)
    custom_name: str = Field(min_length=1)
    preset: str | None = Field(default=None, min_length=1, description="Delivery preset to load.")
    render_format: str | None = Field(default=None, min_length=1)
    render_codec: str | None = Field(default=None, min_length=1)
    render_mode: int | None = Field(default=None, ge=0, le=1)
    mark_in: int | None = Field(default=None, ge=0)
    mark_out: int | None = Field(default=None, ge=0)
    cleanup: bool = Field(default=True, description="Delete the completed render job when done.")
    timeout_seconds: float = Field(default=DEFAULT_RENDER_TIMEOUT, gt=0)

    @model_validator(mode="after")
    def _needs_id(self) -> RenderParams:
        if self.timeline.unique_id is None:
            raise ValueError("render needs the timeline's unique_id (a render mutates the project)")
        if self.mark_in is not None and self.mark_out is not None and self.mark_out <= self.mark_in:
            raise ValueError("mark_out must be greater than mark_in")
        return self


def _preset_name(job_id: str) -> str:
    return f"aar-job-{job_id[:40]}-deliver"


class RenderStep(Step):
    """The whole render, as one checkpointed, resumable step."""

    def __init__(self, params: RenderParams) -> None:
        self.params = params
        self.name = "render"
        self.timeout = params.timeout_seconds

    def _configure_params(self, preset: str) -> dict[str, Any]:
        p = self.params
        return {
            "preset": p.preset,
            "target_dir": p.target_dir,
            "custom_name": p.custom_name,
            "render_format": p.render_format,
            "render_codec": p.render_codec,
            "render_mode": p.render_mode,
            "mark_in": p.mark_in,
            "mark_out": p.mark_out,
        }

    def _to_finalize_fail(self, cur: dict[str, Any], error: StructuredError) -> StepAction:
        cur["error"] = error.to_dict()
        if "deliver" not in cur:
            # The snapshot never succeeded, so nothing was changed to restore.
            return Fail(error)
        cur["phase"] = "finalize_fail"
        return Submit("render.restore_deliver", cur["deliver"], key="restore")

    def advance(self, ctx: StepContext) -> StepAction:
        p = self.params
        cur = ctx.cursor
        out = ctx.outcome
        preset = _preset_name(ctx.job_id)
        phase = cur.get("phase")

        if phase is None:
            cur["phase"] = "snapshot"
            return Submit("render.deliver_snapshot", {"preset_name": preset}, key="snapshot")

        if phase == "snapshot":
            if out is None:
                return Submit("render.deliver_snapshot", {"preset_name": preset}, key="snapshot")
            if not out.ok:
                assert out.error is not None
                return Fail(out.error)  # nothing changed yet
            snap = out.value_dict()
            # Only the fields render.restore_deliver takes (it forbids extras like 'created').
            cur["deliver"] = {k: snap.get(k) for k in ("preset", "format", "codec", "mode")}
            cur["phase"] = "configure"
            return Submit("render.configure", self._configure_params(preset), key="configure")

        if phase == "configure":
            if out is None:
                return Submit("render.configure", self._configure_params(preset), key="configure")
            if not out.ok:
                assert out.error is not None
                return self._to_finalize_fail(cur, out.error)
            cur["phase"] = "add"
            return Submit(
                "render.add_job",
                {"timeline": p.timeline.model_dump(), "custom_name": p.custom_name},
                key="add",
            )

        if phase == "add":
            if out is None:
                return Submit(
                    "render.add_job",
                    {"timeline": p.timeline.model_dump(), "custom_name": p.custom_name},
                    key="add",
                )
            if not out.ok:
                assert out.error is not None
                return self._to_finalize_fail(cur, out.error)
            cur["render_job_id"] = out.value_dict()["render_job_id"]
            cur["phase"] = "start"
            return Submit("render.start", {"render_job_id": cur["render_job_id"]}, key="start")

        if phase == "start":
            if out is None:
                return Submit("render.start", {"render_job_id": cur["render_job_id"]}, key="start")
            if not out.ok:
                assert out.error is not None
                return self._to_finalize_fail(cur, out.error)
            cur["phase"] = "poll"
            return Submit("render.poll", {"render_job_id": cur["render_job_id"]}, key="poll")

        if phase == "poll":
            if out is None:
                return Submit("render.poll", {"render_job_id": cur["render_job_id"]}, key="poll")
            if not out.ok:
                assert out.error is not None
                return self._to_finalize_fail(cur, out.error)
            v = out.value_dict()
            status = v["status"]
            if status in _DONE_OK:
                cur["measured"] = _measured(v)
                cur["phase"] = "finalize_ok"
                return Submit("render.restore_deliver", cur["deliver"], key="restore")
            if status in _DONE_FAIL:
                error = StructuredError(
                    "render_failed",
                    f"render job {cur['render_job_id']} ended {status!r}"
                    + (f": {v.get('error')}" if v.get("error") else ""),
                    "inspect the render on Resolve's Deliver page; check every clip's media is "
                    "online (trap render-uses-original-media) and the output path is writable",
                    "render-uses-original-media",
                    detail={"render_status": v},
                )
                return self._to_finalize_fail(cur, error)
            # Ready / Rendering / background states: poll again next pump, do not block the broker.
            return Pause("rendering", f"render job {cur['render_job_id']} is {status!r}")

        if phase == "finalize_ok":
            if out is None:
                return Submit("render.restore_deliver", cur["deliver"], key="restore")
            if not out.ok:
                assert out.error is not None
                return Fail(
                    StructuredError(
                        "deliver_restore_failed",
                        "the render finished but the project's Deliver settings could not be "
                        "restored",
                        f"load the preset {cur['deliver'].get('preset')!r} in the Deliver page "
                        "by hand",
                        detail={"measured": cur["measured"], "restore_error": out.error.to_dict()},
                    )
                )
            if p.cleanup:
                cur["phase"] = "cleanup"
                return Submit(
                    "render.delete_job", {"render_job_id": cur["render_job_id"]}, key="delete"
                )
            return Done(cur["measured"])

        if phase == "cleanup":
            if out is None:
                return Submit(
                    "render.delete_job", {"render_job_id": cur["render_job_id"]}, key="delete"
                )
            measured = dict(cur["measured"])
            if not out.ok:
                assert out.error is not None
                measured["cleanup_note"] = (
                    f"the completed render job could not be removed from the queue: "
                    f"{out.error.message}"
                )
            return Done(measured)

        if phase == "finalize_fail":
            if out is None:
                return Submit("render.restore_deliver", cur["deliver"], key="restore")
            error = StructuredError.from_dict(cur["error"])
            if not out.ok:
                assert out.error is not None
                error = StructuredError(
                    error.code,
                    error.message,
                    error.fix_hint,
                    error.trap_id,
                    detail={
                        **error.detail,
                        "restore_error": out.error.to_dict(),
                        "deliver_preset": cur["deliver"].get("preset"),
                    },
                )
            return Fail(error)

        raise RuntimeError(f"render step reached an unknown phase {phase!r}")


def _measured(poll_value: dict[str, Any]) -> dict[str, Any]:
    """The render's measured values, taken straight from the poll op's measured result."""
    keys = (
        "render_job_id",
        "status",
        "completion",
        "output_path",
        "target_dir",
        "output_filename",
        "frame_count",
        "frame_rate",
        "duration_seconds",
        "render_time_ms",
    )
    return {k: poll_value.get(k) for k in keys}


def _build_render(params: BaseModel, target: ProjectLocator) -> list[Step]:
    assert isinstance(params, RenderParams)
    return [RenderStep(params)]


# Registered in aar.node.jobs.__init__ to keep import order explicit.
