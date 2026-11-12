"""Helpers for the ``ccn-longform`` recipe tests (S19a): a hub, synthetic media under a media
root, an injected transcriber, the worker run in-process, and the agent's cut done the way the
step's instruction says (``aar hub artifact get``, ``aar plan produce``, ``aar plan lint``,
``aar hub write plan``, ``aar recipe complete``), through the real CLI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from write_support import AGENT, Hub, brief_data

from aar.analyze.whisper import wordless_report
from aar.cli import main
from aar.hub.writes import create_brief, create_recipe
from aar.plan.inputs import Transcript, Word
from aar.recipes import registry
from aar.recipes.run import recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.worker import run_worker
from ccn_support import KEEP, SECONDS, WORDS, make_show

RECIPE = "ccn-longform"
WORKER = AGENT
PC_ROOT = "E:\\Shows"


@dataclass
class CcnEnv:
    """One test's hub, folders and transcriber call log."""

    hub: Hub
    media: Path
    work: Path
    exports: Path
    agent: Path
    calls: list[dict[str, Any]] = field(default_factory=list)
    words: tuple[tuple[str, float, float], ...] = WORDS

    @property
    def db(self) -> str:
        return str(self.hub.root / "hub.db")

    @property
    def artifacts(self) -> str:
        return str(self.hub.root / "artifacts")

    def show(self, name: str = "cam-a.mov", **kw: Any) -> Path:
        return make_show(self.media / name, **kw)

    def params(self, *paths: Path, **changes: Any) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sources": [
                {"path": str(p), "camera": chr(ord("A") + n), "shot_size": "wide"}
                for n, p in enumerate(paths)
            ],
            "recording_id": "show-1",
            "timeline": "Synthetic Long v1",
            "media_map": {str(self.media): PC_ROOT},
        }
        data.update(changes)
        return data


def fake_transcriber(env: CcnEnv) -> Callable[..., dict[str, Any]]:
    """Stands in for ``transcribe_file``: writes :data:`WORDS` as the transcript (shifted to
    recording time) and returns a report with the real wordless block (at a 1 s minimum, so
    the synthetic show's two one-second pauses are listed)."""

    def transcribe(
        media: Any,
        out: Any,
        *,
        settings: Any,
        model_dir: Any = None,
        offset: Any = None,
        recording: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        env.calls.append(
            {"media": str(media), "model": settings.model, "normalize": settings.normalize}
        )
        shift = offset.offset_s if offset is not None else 0.0
        words = [Word(text=t, start_s=a + shift, end_s=b + shift) for t, a, b in env.words]
        Path(out).write_text(
            Transcript(recording=recording, words=words).model_dump_json(), encoding="utf-8"
        )
        wordless = wordless_report(
            Path(media),
            words,
            file_duration_s=SECONDS,
            offset_s=shift,
            settings=replace(settings, wordless_min_s=1.0),
        )
        return {
            "ok": True,
            "output": str(out),
            "settings": settings.to_dict(),
            "wordless": wordless,
        }

    return transcribe


def make_run(env: CcnEnv, params: dict[str, Any], *, key: str = "1", kind: str = "longform") -> str:
    """A typed brief and a ``ccn-longform`` recipe node; returns the recipe node id."""
    hub = env.hub
    brief = create_brief(
        hub.log,
        brief_data(hub.project_id, kind=kind, recipe=RECIPE, show="Synthetic Show"),
        AGENT,
        idempotency_key=f"brief-{key}",
    )
    recipe = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": RECIPE, "brief_id": brief.body["node"]["id"], "params": params},
        AGENT,
        idempotency_key=f"recipe-{key}",
        params_validator=registry.registry_params_validator,
    )
    return recipe.body["node"]["id"]


def start(env: CcnEnv, recipe_node: str, key: str = "1") -> None:
    recipe_start(env.hub.log, recipe_node, AGENT, idempotency_key=f"start-{key}")


def work(env: CcnEnv, limit: int = 30) -> int:
    """Run the cold worker in-process until it finds nothing to do; return the steps it ran."""
    ran = 0
    for _ in range(limit):
        if run_worker(env.hub.log, env.hub.store, WORKER, once=True) == 0:
            break
        ran += 1
    return ran


def outputs(env: CcnEnv, recipe_node: str, step: str) -> dict[str, Any]:
    return dict(load_state(env.hub.log, recipe_node).done_outputs(step))


def latest(env: CcnEnv, recipe_node: str, step: str) -> Any:
    state = load_state(env.hub.log, recipe_node)
    return state.latest_in_round(step, state.current_round())


def fetch(env: CcnEnv, artifact_id: str, name: str) -> Path:
    """``aar hub artifact get`` into the agent's folder, as the cut instruction says."""
    out = env.agent / name
    argv = ["hub", "artifact", "get", artifact_id, "--db", env.db, "--artifacts", env.artifacts]
    assert main([*argv, "--out", str(out), "--force"]) == 0
    return out


def agent_cut(
    env: CcnEnv,
    recipe_node: str,
    *,
    keep: dict[str, Any] | None = None,
    derives_from: str | None = None,
    key: str = "r1",
    schedule: dict[str, Any] | None = None,
) -> str:
    """Do the ``cut`` step as an agent following its instruction; return the plan node id."""
    stub_out = outputs(env, recipe_node, "stub")
    text_out = outputs(env, recipe_node, "transcript")
    stub = fetch(env, stub_out["stub"], "stub.json")
    transcript = fetch(env, text_out["transcript"], "transcript.json")
    fetch(env, text_out["text"], "transcript.txt")
    keep_file = env.agent / "keep.json"
    body = keep or KEEP
    keep_file.write_text(
        json.dumps({"recording": stub_out["recording_id"], **body}), encoding="utf-8"
    )
    plan = env.agent / f"plan.{key}.json"
    argv = ["plan", "produce", str(stub), "--transcript", str(transcript)]
    argv += ["--keep", str(keep_file), "--out", str(plan), stub_out["pause_trim_flag"]]
    sync_out = outputs(env, recipe_node, "sync")
    if sync_out.get("report"):
        argv += ["--sync", str(fetch(env, sync_out["report"], "sync.json"))]
    laughs_out = outputs(env, recipe_node, "laughs")
    if laughs_out.get("map"):
        argv += ["--laughs", str(fetch(env, laughs_out["map"], "laughs.json"))]
    if schedule is not None:
        sched = env.agent / "schedule.json"
        sched.write_text(json.dumps(schedule), encoding="utf-8")
        argv += ["--schedule", str(sched)]
    assert main(argv) == 0
    assert main(["plan", "lint", str(plan), "--transcript", str(transcript)]) == 0
    request = env.agent / f"submit.{key}.json"
    request.write_text(
        json.dumps(
            {
                "project_id": env.hub.project_id,
                "derives_from": derives_from or recipe_node,
                "plan": json.loads(plan.read_text(encoding="utf-8")),
            }
        ),
        encoding="utf-8",
    )
    write = ["hub", "write", "plan", "--db", env.db, "--artifacts", env.artifacts]
    write += ["--actor", "agent:agent-1", "--idempotency-key", f"plan-{key}", str(request)]
    plan_node = _json_main(write)["node"]["id"]
    step_complete(
        env.hub.log,
        env.hub.store,
        recipe_node,
        "cut",
        None,
        {"plan_node": plan_node},
        AGENT,
        idempotency_key=f"cut-{key}",
    )
    return plan_node


def _json_main(argv: list[str]) -> dict[str, Any]:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(argv)
    assert code == 0, buffer.getvalue()
    return json.loads(buffer.getvalue())
