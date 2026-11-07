"""Supervise `aar node run` on the edit PC (started by the 'AAR Resolve node' logon task).

Waits for Resolve, starts the worker with the locators in the projects file, and restarts it
with backoff. It must run in the interactive console session so the presence guard can see
input idle (see deploy/windows/README.md).

Configuration is a JSON file next to this script (``run_node.json``, or the path in argv[1]):

    {"hub_url": "http://100.x.y.z:8789", "node_id": "edit-pc",
     "token_file": "edit-pc.token", "store": "jobs.db",
     "projects_file": "projects.json", "log_dir": "logs"}

Relative paths resolve against the config file's directory. Optional keys: ``aar`` (the
executable; default ``venv/Scripts/aar.exe``), ``script_api`` and ``script_lib`` (Resolve's
scripting locations). ``aar`` is deliberately NOT imported here: the supervisor stays alive
across upgrades of the app and only ever launches it as a subprocess with a list argv.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SCRIPT_API = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
DEFAULT_SCRIPT_LIB = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"
REQUIRED_KEYS = ("hub_url", "node_id", "token_file", "store", "projects_file", "log_dir")

INITIAL_BACKOFF = 15.0
MAX_BACKOFF = 300.0
HEALTHY_SECONDS = 600.0  # a worker that ran longer than this resets the backoff


class ConfigError(ValueError):
    """The supervisor config file is missing, unreadable or malformed."""


@dataclass(frozen=True)
class Config:
    hub_url: str
    node_id: str
    token_file: Path
    store: Path
    projects_file: Path
    log_dir: Path
    aar: Path
    script_api: str
    script_lib: str


def load_config(path: Path) -> Config:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ConfigError(f"config {path} is missing: {', '.join(missing)}")
    for key in (*REQUIRED_KEYS, "aar", "script_api", "script_lib"):
        if key in data and (not isinstance(data[key], str) or not data[key].strip()):
            raise ConfigError(f"config key {key!r} must be a non-empty string")
    base = path.resolve().parent

    def rel(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else base / p

    return Config(
        hub_url=data["hub_url"],
        node_id=data["node_id"],
        token_file=rel(data["token_file"]),
        store=rel(data["store"]),
        projects_file=rel(data["projects_file"]),
        log_dir=rel(data["log_dir"]),
        aar=rel(data.get("aar", r"venv\Scripts\aar.exe")),
        script_api=data.get("script_api", DEFAULT_SCRIPT_API),
        script_lib=data.get("script_lib", DEFAULT_SCRIPT_LIB),
    )


def next_backoff(ran_seconds: float, backoff: float) -> float:
    """The delay before the next restart: reset after a healthy run, else double up to the cap."""
    if ran_seconds > HEALTHY_SECONDS:
        return INITIAL_BACKOFF
    return min(backoff * 2, MAX_BACKOFF)


def parse_projects(text: str) -> tuple[list[str], str | None]:
    """Validate the projects file. Returns (locator JSON strings, error message or None).

    The file must be a JSON array; entries that are not objects are dropped. A malformed file
    yields no locators, so the supervisor idles instead of starting a worker with nothing to do.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        return [], f"projects file unreadable: {exc}"
    if not isinstance(data, list):
        return [], "projects file must be a JSON array of ProjectLocator objects"
    return [json.dumps(p, ensure_ascii=False) for p in data if isinstance(p, dict)], None


def build_command(cfg: Config, locators: list[str]) -> list[str]:
    """The worker argv: always a list, never a shell string."""
    cmd = [
        str(cfg.aar),
        "node",
        "run",
        "--node-id",
        cfg.node_id,
        "--hub-url",
        cfg.hub_url,
        "--token-file",
        str(cfg.token_file),
        "--store",
        str(cfg.store),
    ]
    for loc in locators:
        cmd += ["--project", loc]
    return cmd


def build_env(cfg: Config, base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env["RESOLVE_SCRIPT_API"] = cfg.script_api
    env["RESOLVE_SCRIPT_LIB"] = cfg.script_lib
    env["PYTHONPATH"] = os.path.join(cfg.script_api, "Modules")
    env["PYTHONUTF8"] = "1"
    return env


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def resolve_running(run: Callable[..., Any] = subprocess.run) -> bool:
    """True when Resolve.exe is in the task list. ``run`` is the injectable process seam."""
    out = run(
        ["tasklist", "/FI", "IMAGENAME eq Resolve.exe", "/NH"],
        capture_output=True,
        text=True,
        creationflags=_no_window(),
    ).stdout
    return "Resolve.exe" in (out or "")


class Supervisor:
    """The loop, with every side effect (clock, sleep, process launch) injectable for tests."""

    def __init__(
        self,
        cfg: Config,
        *,
        resolve_running: Callable[[], bool] = resolve_running,
        launch: Callable[[list[str], dict[str, str], Path], int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime.datetime] = datetime.datetime.now,
    ) -> None:
        self.cfg = cfg
        self._resolve_running = resolve_running
        self._launch = launch or self._launch_subprocess
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self.backoff = INITIAL_BACKOFF

    def log(self, msg: str) -> None:
        stamp = self._now()
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.log_dir / f"supervisor-{stamp:%Y%m%d}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{stamp.isoformat(timespec='seconds')} {msg}\n")

    def _launch_subprocess(self, cmd: list[str], env: dict[str, str], log_path: Path) -> int:
        with open(log_path, "a", encoding="utf-8") as out:
            return subprocess.run(
                cmd, env=env, stdout=out, stderr=subprocess.STDOUT, creationflags=_no_window()
            ).returncode

    def projects(self) -> list[str]:
        try:
            text = self.cfg.projects_file.read_text(encoding="utf-8")
        except OSError as exc:
            self.log(f"projects file unreadable: {exc}")
            return []
        locs, error = parse_projects(text)
        if error:
            self.log(error)
        return locs

    def step(self) -> None:
        """One pass of the loop: wait, or run one worker to exit and sleep the backoff."""
        if not self._resolve_running():
            self._sleep(20)
            return
        locs = self.projects()
        if not locs:
            self._sleep(60)
            return
        self._sleep(20)  # let Resolve's scripting server come up after launch
        cmd = build_command(self.cfg, locs)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.cfg.log_dir / f"node-{self._now():%Y%m%d}.log"
        started = self._monotonic()
        self.log(f"starting worker for {len(locs)} project(s)")
        code = self._launch(cmd, build_env(self.cfg), log_path)
        ran = self._monotonic() - started
        self.log(f"worker exited code={code} after {ran:.0f}s")
        self.backoff = next_backoff(ran, self.backoff)
        self._sleep(self.backoff)

    def run(self) -> None:
        self.log("supervisor start")
        while True:
            self.step()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    config_path = Path(args[0]) if args else Path(__file__).resolve().with_name("run_node.json")
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    Supervisor(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
