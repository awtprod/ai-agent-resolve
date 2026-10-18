r"""The render job: build-and-start a render atomically, then poll it to completion.

A render is a hot op and queues per project (DESIGN R4). It cannot run inside a single MCP
call (F9), so it is a job: the runner polls ``render.poll`` across pumps, yielding the broker
to other projects between polls, until the render job in Resolve finishes.

The step is one resumable state machine whose phase lives in the checkpoint cursor:

    build -> poll -> (cleanup) -> done
                 \-> fail

``render.build`` is atomic (set timeline, snapshot Deliver, configure, add, start, restore
Deliver -- all in one op), so the Deliver settings are only ever clobbered inside that op while
it holds the instance (DESIGN §5.3). Nothing is left to restore on the poll/cleanup/fail paths,
so a cancel, stall or stop mid-render never leaves the Deliver page clobbered or a preset
leaked. A render job the build op added is reused on resume only by its recorded ``JobId``,
never by output name, so a crash can never duplicate it onto, or adopt, a human's render.
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
_MEASURED_KEYS = (
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
    def _check(self) -> RenderParams:
        if self.timeline.unique_id is None:
            raise ValueError("render needs the timeline's unique_id (a render mutates the project)")
        if self.mark_in is not None and self.mark_out is not None and self.mark_out <= self.mark_in:
            raise ValueError("mark_out must be greater than mark_in")
        import math

        if not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be finite")
        return self


def snapshot_preset_name(job_id: str) -> str:
    return f"aar-job-{job_id[:40]}-deliver"


class RenderStep(Step):
    """The whole render, as one checkpointed, resumable step. Renders queue per project."""

    renders_project = True  # the runner serialises these per project (one render at a time)

    def __init__(self, params: RenderParams) -> None:
        self.params = params
        self.name = "render"
        self.timeout = params.timeout_seconds

    def _build_params(self, job_id: str) -> dict[str, Any]:
        p = self.params
        return {
            "timeline": p.timeline.model_dump(),
            "target_dir": p.target_dir,
            "custom_name": p.custom_name,
            "snapshot_preset": snapshot_preset_name(job_id),
            "preset": p.preset,
            "render_format": p.render_format,
            "render_codec": p.render_codec,
            "render_mode": p.render_mode,
            "mark_in": p.mark_in,
            "mark_out": p.mark_out,
        }

    def advance(self, ctx: StepContext) -> StepAction:
        p = self.params
        cur = ctx.cursor
        out = ctx.outcome
        phase = cur.get("phase")

        if phase is None:
            cur["phase"] = "build"
            return Submit("render.build", self._build_params(ctx.job_id), key="build")

        if phase == "build":
            if out is None:
                return Submit("render.build", self._build_params(ctx.job_id), key="build")
            if not out.ok:
                assert out.error is not None
                return Fail(out.error)  # build restored Deliver in its finally; nothing to undo
            v = out.value_dict()
            cur["render_job_id"] = v["render_job_id"]
            cur["base"] = {k: v.get(k) for k in ("output_path", "target_dir", "output_filename",
                                                 "frame_count", "frame_rate", "duration_seconds")}
            cur["phase"] = "poll"
            return Submit("render.poll", {"render_job_id": cur["render_job_id"]}, key="poll")

        if phase == "poll":
            if out is None:
                return Submit("render.poll", {"render_job_id": cur["render_job_id"]}, key="poll")
            if not out.ok:
                assert out.error is not None
                return Fail(out.error)
            v = out.value_dict()
            status = v["status"]
            if status in _DONE_OK:
                cur["measured"] = {k: v.get(k) for k in _MEASURED_KEYS}
                if p.cleanup:
                    cur["phase"] = "cleanup"
                    return Submit(
                        "render.delete_job", {"render_job_id": cur["render_job_id"]}, key="delete"
                    )
                return Done(cur["measured"])
            if status in _DONE_FAIL:
                return Fail(
                    StructuredError(
                        "render_failed",
                        f"render job {cur['render_job_id']} ended {status!r}"
                        + (f": {v.get('error')}" if v.get("error") else ""),
                        "inspect the render on Resolve's Deliver page; check every clip's media "
                        "is online (trap render-uses-original-media) and the output path is "
                        "writable",
                        "render-uses-original-media",
                        detail={"render_status": v},
                    )
                )
            # Ready / Rendering / background states: poll again next pump, do not block the broker.
            return Pause("rendering", f"render job {cur['render_job_id']} is {status!r}")

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

        raise RuntimeError(f"render step reached an unknown phase {phase!r}")


def _build_render(params: BaseModel, target: ProjectLocator) -> list[Step]:
    assert isinstance(params, RenderParams)
    return [RenderStep(params)]


# Registered in aar.node.jobs.__init__ to keep import order explicit.
