"""Helpers for the ``ccn-longform`` recipe tests (S19a): a hub, synthetic media under a media
root, an injected transcriber, the worker run in-process, and the agent's cut done the way the
step's instruction says (``aar hub artifact get``, ``aar plan produce``, ``aar plan lint``,
``aar hub write plan``, ``aar recipe complete``), through the real CLI.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
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
from aar.recipes.ccn.steps import aar_argv
from aar.recipes.run import recipe_start
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
    resumes: list[bool] = field(default_factory=list)
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
        env.resumes.append(resume)
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


def make_run(
    env: CcnEnv,
    params: dict[str, Any],
    *,
    key: str = "1",
    kind: str = "longform",
    project: str | None = None,
) -> str:
    """A typed brief and a ``ccn-longform`` recipe node; returns the recipe node id."""
    hub = env.hub
    project_id = project or hub.project_id
    brief = create_brief(
        hub.log,
        brief_data(project_id, kind=kind, recipe=RECIPE, show="Synthetic Show"),
        AGENT,
        idempotency_key=f"brief-{key}",
    )
    recipe = create_recipe(
        hub.log,
        project_id,
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


GUIDE_BLOCK = re.compile(r"```sh\n(.*?)```", re.DOTALL)


def sh(env: CcnEnv, line: str, **extra: str) -> str:
    """Run one line in ``sh`` exactly as given, in the agent's folder; its stdout."""
    run = subprocess.run(
        ["sh", "-c", line],
        cwd=env.agent,
        env={**os.environ, **extra},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, f"{line}\nrc={run.returncode}\n{run.stderr}"
    return run.stdout


def read_guide(env: CcnEnv, recipe_node: str) -> str:
    """Fetch the guide with the exact command the guide step's outputs give, run in ``sh``."""
    command = outputs(env, recipe_node, "guide")["read_guide"]
    assert shlex.split(command)[: len(aar_argv())] == aar_argv()  # the worker's own aar
    sh(env, command)
    return (env.agent / "guide.md").read_text(encoding="utf-8")


def agent_cut(
    env: CcnEnv,
    recipe_node: str,
    *,
    keep: dict[str, Any] | None = None,
    derives_from: str | None = None,
    key: str = "r1",
    schedule: dict[str, Any] | None = None,
) -> str:
    """Do the ``cut`` step as an agent following the guide alone: its fetch commands, its
    produce and lint commands and its submit and complete commands, each run in ``sh`` as
    printed (only the ``<...>`` placeholders filled in; the program is never substituted).
    Returns the plan node id."""
    guide = read_guide(env, recipe_node)
    fetch_block, make_block, json_block, submit_block, *_ = GUIDE_BLOCK.findall(guide)
    recording = json.loads(re.search(r'"recording": (".*?")', guide).group(1))
    (env.agent / "keep.json").write_text(
        json.dumps({"recording": recording, **(keep or KEEP)}), encoding="utf-8"
    )
    if schedule is not None:
        (env.agent / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    for line in [*fetch_block.splitlines(), *make_block.splitlines()]:
        assert shlex.split(line)[: len(aar_argv())] == aar_argv(), line
        sh(env, line)
    make_submit = json_block.strip()
    assert shlex.split(make_submit)[0] == sys.executable
    assert shlex.split(make_submit)[-1] == recipe_node
    if derives_from is not None:  # a later round: the review that rejected the cut
        make_submit = make_submit.removesuffix(shlex.quote(recipe_node)) + shlex.quote(derives_from)
    sh(env, make_submit)
    write_line, complete_line = submit_block.strip().splitlines()
    fill = {"<you>": "agent-1", "<new key>": f"plan-{key}"}
    for old, value in fill.items():
        write_line = write_line.replace(old, value)
    plan_node = json.loads(sh(env, write_line))["node"]["id"]
    fill = {"<you>": "agent-1", "<new key>": f"cut-{key}", "<node id the write printed>": plan_node}
    for old, value in fill.items():
        complete_line = complete_line.replace(old, value)
    assert "AAR_RECIPES=aar.recipes.ccn " + shlex.join(aar_argv()) in complete_line
    sh(env, complete_line)
    return plan_node


def _json_main(argv: list[str]) -> dict[str, Any]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(argv)
    assert code == 0, buffer.getvalue()
    return json.loads(buffer.getvalue())
