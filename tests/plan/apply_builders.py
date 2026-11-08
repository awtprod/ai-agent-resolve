"""Raw EditPlan builders for the apply tests (compile_* does not lint, so these may be hostile)."""

from __future__ import annotations

import copy
import json
from typing import Any

from plan_support import FIXTURES

from aar.plan.model import EditPlan

FPS = {"num": 24000, "den": 1001}


def load_data(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def longform_data() -> dict[str, Any]:
    return copy.deepcopy(load_data("ccn_longform_v2.plan.json"))


def longform() -> EditPlan:
    return EditPlan.model_validate(longform_data())


def short() -> EditPlan:
    return EditPlan.model_validate(load_data("short_v2.plan.json"))


def source(sid: str = "a", media: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "media": media or f"E:/X/{sid}.mov",
        "camera": extra.pop("camera", "A"),
        "duration": extra.pop("duration", 100000),
        **extra,
    }


def seg(track: int = 1, src: str = "a", a: int = 0, b: int = 24, **extra: Any) -> dict[str, Any]:
    return {"track": track, "source": src, "src_in": a, "src_out": b, **extra}


def plan_data(
    segments: list[dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
    **over: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "version": 2,
        "project": "P",
        "timeline": "T",
        "template": "Tpl",
        "kind": "longform",
        "fps": FPS,
        "resolution": {"width": 1920, "height": 1080},
        "profile": "ccn",
        "sources": sources if sources is not None else [source()],
        "segments": segments if segments is not None else [seg()],
    }
    data.update(over)
    return data


def plan(*args: Any, **kwargs: Any) -> EditPlan:
    return EditPlan.model_validate(plan_data(*args, **kwargs))
