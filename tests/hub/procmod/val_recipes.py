"""Recipes + a file-backed fake dispatcher shared by the validator's subprocess probes."""

from __future__ import annotations

import fcntl
import json
import os
import time

from pydantic import BaseModel, ConfigDict

from aar.recipes import registry
from aar.recipes.context import StepOutput
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.ports import HotJobView, HotResult


class P(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _mark(tag: str) -> None:
    path = os.environ.get("VAL_SIDEFILE")
    if path:
        with open(path, "a") as handle:
            handle.write(f"{tag} {os.getpid()}\n")


def _fast(step: str):
    def fn(ctx):
        _mark(step)
        return StepOutput(outputs={"step": step})

    return fn


def _slow(ctx):
    _mark("slow-start")
    time.sleep(float(os.environ.get("VAL_SLEEP", "0")))
    _mark("slow-end")
    return StepOutput(outputs={"slow": True})


registry.clear()
for i in range(4):
    registry.register_callable(f"val.f{i}", _fast(f"f{i}"))
registry.register_callable("val.slow", _slow)
registry.register(
    RecipeDef(
        name="valproc",
        version=1,
        params_model=P,
        steps=tuple(
            [
                StepDef(id=f"f{i}", kind="cold", callable_id=f"val.f{i}", idempotent=True)
                for i in range(4)
            ]
            + [StepDef(id=f"h{i}", kind="hot", hot_kind="val.hot") for i in range(3)]
        ),
    )
)
for name, idem in (("valslow_idem", True), ("valslow_non", False)):
    registry.register(
        RecipeDef(
            name=name,
            version=1,
            params_model=P,
            steps=(StepDef(id="slow", kind="cold", callable_id="val.slow", idempotent=idem),),
        )
    )
registry.register(
    RecipeDef(
        name="valdrain",
        version=1,
        params_model=P,
        steps=(
            StepDef(id="s1", kind="cold", callable_id="val.slow"),
            StepDef(id="s2", kind="cold", callable_id="val.slow"),
        ),
    )
)


class FileDispatcher:
    """Idempotent on request_id across processes; logs every submit call."""

    def __init__(self, root: str) -> None:
        self.root = root

    def _locked(self, fn):
        with open(os.path.join(self.root, "lock"), "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def submit(self, *, request_id, project_id, kind, params, agent_id):
        def go():
            jobs_path = os.path.join(self.root, "jobs.json")
            if os.path.exists(jobs_path):
                with open(jobs_path) as _jf:
                    jobs = json.load(_jf)
            else:
                jobs = {}
            with open(os.path.join(self.root, "calls.jsonl"), "a") as fh:
                fh.write(
                    json.dumps({"pid": os.getpid(), "request_id": request_id, "agent_id": agent_id})
                    + "\n"
                )
            if request_id not in jobs:
                jobs[request_id] = f"job-{len(jobs) + 1}"
                with open(jobs_path, "w") as _jf:
                    json.dump(jobs, _jf)
            return jobs[request_id]

        # widen the race window between intent and submit
        time.sleep(float(os.environ.get("VAL_SUBMIT_DELAY", "0")))
        return self._locked(go)

    def poll(self, job_ref):
        return HotJobView("done", result=HotResult(outputs={"job": job_ref}))
