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
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9  # JobObjectExtendedLimitInformation
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
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


def _last_error() -> int:
    import ctypes

    return getattr(ctypes, "get_last_error", lambda: 0)()


def _declare_job_api(k32: Any, wintypes: Any) -> None:
    """Declare argtypes/restype. Without them ctypes passes and returns C ints, which truncates
    64-bit HANDLEs and turns GetCurrentProcess()'s pseudo-handle (HANDLE)-1 into 32-bit -1."""
    import ctypes

    H, B, D, P = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD, wintypes.LPVOID
    k32.CreateJobObjectW.argtypes = [P, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = H
    k32.SetInformationJobObject.argtypes = [H, ctypes.c_int, P, D]
    k32.SetInformationJobObject.restype = B
    k32.AssignProcessToJobObject.argtypes = [H, H]
    k32.AssignProcessToJobObject.restype = B
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = H
    k32.CloseHandle.argtypes = [H]
    k32.CloseHandle.restype = B


def kill_on_close_job(os_name: str = os.name, kernel32: Any = None) -> tuple[Any, str | None]:
    """Put THIS process in a Windows job object with KILL_ON_JOB_CLOSE.

    Returns ``(handle, None)`` on success, ``(None, reason)`` on a Windows failure, and
    ``(None, None)`` off Windows (nothing to do, not an error).

    The worker (aar.exe -> venv python -> base python) is a separate process tree that nothing
    reaps if the supervisor dies, and Stop-ScheduledTask does not kill it either. Children
    inherit the supervisor's job, so when the supervisor exits and the (never closed) handle
    goes away, Windows kills the whole tree. Assigning the supervisor itself, rather than each
    worker after Popen, leaves no window in which a child escapes. ``os_name`` and ``kernel32``
    are seams for tests.
    """
    if os_name != "nt":
        return None, None
    import ctypes
    from ctypes import wintypes

    k32 = kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)
    _declare_job_api(k32, wintypes)

    def fail(call: str) -> tuple[None, str]:
        return None, f"{call} failed: winerror {_last_error()}"

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return fail("CreateJobObjectW")

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = _Extended()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(
        job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
    ):
        result = fail("SetInformationJobObject")
    elif not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        result = fail("AssignProcessToJobObject")
    else:
        return job, None
    k32.CloseHandle(job)
    return result


def self_test_job(out: Any = None) -> int:
    """``run_node.py --self-test-job``: bind a job, spawn a 30 s sleeper, exit WITHOUT waiting.

    When this process exits the job handle closes, so the sleeper must die. The operator checks
    that ``child=<pid>`` is gone. Prints ``job=ok child=<pid>`` or ``job=failed: <reason>``.
    """
    out = out or sys.stdout
    job, reason = kill_on_close_job()
    if job is None:
        print(f"job=failed: {reason or 'not Windows'}", file=out, flush=True)
        return 1
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], creationflags=_no_window()
    )
    print(f"job=ok child={child.pid}", file=out, flush=True)
    return 0


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
        self.job: Any = None

    def log(self, msg: str) -> None:
        stamp = self._now()
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.log_dir / f"supervisor-{stamp:%Y%m%d}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{stamp.isoformat(timespec='seconds')} {msg}\n")

    def log_both(self, msg: str) -> None:
        """Log to the supervisor log AND the node log, so the failure is seen in either."""
        self.log(msg)
        stamp = self._now()
        with open(self.cfg.log_dir / f"node-{stamp:%Y%m%d}.log", "a", encoding="utf-8") as f:
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
    if args[:1] == ["--self-test-job"]:
        return self_test_job()
    config_path = Path(args[0]) if args else Path(__file__).resolve().with_name("run_node.json")
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    sup = Supervisor(cfg)
    job, reason = kill_on_close_job()  # keep the handle referenced for the life of the process
    if os.name == "nt":
        if job is None:
            sup.log_both(f"supervisor: job object unavailable: {reason}")
        else:
            sup.log("worker tree bound to supervisor (job object)")
    sup.job = job
    sup.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
