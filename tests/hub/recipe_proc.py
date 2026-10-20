"""Subprocess machinery for the two-process concurrency and kill/resume tests.

Writes an importable recipe module and a tiny driver into a temp dir, then runs real
``aar.recipes`` workers / ``advance`` passes as separate OS processes against one SQLite file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from write_support import AGENT, brief_data

from aar.hub.writes import create_brief, create_recipe
from aar.recipes import registry

# A recipe module the test process and the subprocesses both import, so the registry matches.
MODULE_SRC = """
import os, time
from pydantic import BaseModel, ConfigDict
from aar.recipes import registry
from aar.recipes.context import StepOutput
from aar.recipes.models import RecipeDef, StepDef


class P(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _mark(step_id):
    path = os.environ.get("AAR_SIDEFILE")
    if path:
        with open(path, "a") as handle:
            handle.write(step_id + "\\n")


def _parallel(step_id):
    def fn(ctx):
        _mark(step_id)
        return StepOutput(outputs={"step": step_id})
    return fn


def _slow(ctx):
    _mark("slow")
    time.sleep(float(os.environ.get("AAR_SLEEP", "0")))
    return StepOutput(outputs={"slow": True})


registry.clear()
for i in range(5):
    registry.register_callable("par.%d" % i, _parallel("s%d" % i))
registry.register_callable("slow.step", _slow)
registry.register(RecipeDef(name="parallel", version=1, params_model=P, steps=tuple(
    [StepDef(id="s%d" % i, kind="cold", callable_id="par.%d" % i, idempotent=True)
     for i in range(5)]
    + [StepDef(id="report", kind="cold", callable_id="recipe.report", idempotent=True,
               needs=tuple("s%d" % i for i in range(5)))]
)))
for name, idem in (("slow_idem", True), ("slow_nonidem", False)):
    registry.register(RecipeDef(name=name, version=1, params_model=P, steps=(
        StepDef(id="slow", kind="cold", callable_id="slow.step", idempotent=idem,
                max_attempts=1),
        StepDef(id="report", kind="cold", callable_id="recipe.report", idempotent=True,
                needs=("slow",)),
    )))
"""

DRIVER_SRC = """
import os, sys
sys.path.insert(0, os.environ["AAR_MODDIR"])
import aar_test_recipes  # noqa: F401  (registers recipes)
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.recipes.engine import advance
from aar.recipes.worker import run_worker

action, db, artifacts, actor_s, recipe_node = sys.argv[1:6]
kind, _, ident = actor_s.partition(":")
actor = Actor(kind, ident)
log = EventLog(db)
try:
    if action == "advance":
        advance(log, recipe_node, actor=actor)
    elif action == "worker":
        store = ArtifactStore(artifacts, log)
        run_worker(
            log, store, actor,
            once=os.environ.get("AAR_ONCE") == "1",
            lease_s=int(os.environ.get("AAR_LEASE", "60")),
            heartbeat_s=int(os.environ.get("AAR_HB", "15")),
            poll_s=float(os.environ.get("AAR_POLL", "0.1")),
            install_signals=True,
        )
finally:
    log.close()
"""


def materialize(tmp_path: Path) -> tuple[Path, Path]:
    """Write the recipe module and driver; return (module dir, driver path)."""
    moddir = tmp_path / "recmod"
    moddir.mkdir(exist_ok=True)
    (moddir / "aar_test_recipes.py").write_text(MODULE_SRC)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER_SRC)
    load_in_process(moddir)
    return moddir, driver


def load_in_process(moddir: Path) -> None:
    """Register the same recipes in this (test) process so create_recipe accepts them.

    Drops any cached copy first so each test re-executes the freshly written module.
    """
    sys.modules.pop("aar_test_recipes", None)
    if str(moddir) not in sys.path:
        sys.path.insert(0, str(moddir))
    import aar_test_recipes  # type: ignore  # noqa: F401  (registers on import)


def make_cold_recipe(hub: Any, recipe_name: str) -> str:
    """Create a brief and recipe for ``recipe_name`` (params {}); return the recipe node id."""
    brief = create_brief(
        hub.log,
        brief_data(hub.project_id, recipe=recipe_name),
        AGENT,
        idempotency_key=f"brief-{recipe_name}",
    )
    brief_id = brief.body["node"]["id"]
    recipe = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": recipe_name, "brief_id": brief_id, "params": {}},
        AGENT,
        idempotency_key=f"recipe-{recipe_name}",
        params_validator=registry.registry_params_validator,
    )
    return recipe.body["node"]["id"]


def _env(moddir: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["AAR_MODDIR"] = str(moddir)
    env.update(extra)
    return env


def spawn(
    driver: Path,
    moddir: Path,
    action: str,
    db: Path,
    artifacts: Path,
    actor: str,
    recipe_node: str,
    **env: str,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(driver), action, str(db), str(artifacts), actor, recipe_node],
        env=_env(moddir, **env),
    )


def run(
    driver: Path,
    moddir: Path,
    action: str,
    db: Path,
    artifacts: Path,
    actor: str,
    recipe_node: str,
    timeout: float = 60,
    **env: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(driver), action, str(db), str(artifacts), actor, recipe_node],
        env=_env(moddir, **env),
        timeout=timeout,
        capture_output=True,
    )


def wait_for(predicate: Any, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` until true or ``timeout``; return whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
