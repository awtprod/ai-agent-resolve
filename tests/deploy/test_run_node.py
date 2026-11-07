"""The Windows supervisor (deploy/windows/run_node.py): config, backoff, projects, argv."""

from __future__ import annotations

import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "windows" / "run_node.py"
_spec = importlib.util.spec_from_file_location("aar_deploy_run_node", SCRIPT)
assert _spec and _spec.loader
rn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rn
_spec.loader.exec_module(rn)

GOOD = {
    "hub_url": "http://100.1.2.3:8789",
    "node_id": "edit-pc",
    "token_file": "edit-pc.token",
    "store": "jobs.db",
    "projects_file": "projects.json",
    "log_dir": "logs",
}


def _write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "run_node.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_import_does_not_pull_in_aar() -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import runpy,sys; runpy.run_path({str(SCRIPT)!r}); "
            "print(any(m == 'aar' or m.startswith('aar.') for m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False"


def test_config_resolves_relative_paths_next_to_the_file(tmp_path: Path) -> None:
    cfg = rn.load_config(_write(tmp_path, GOOD))
    base = tmp_path.resolve()
    assert cfg.token_file == base / "edit-pc.token"
    assert cfg.store == base / "jobs.db"
    assert cfg.projects_file == base / "projects.json"
    assert cfg.log_dir == base / "logs"
    assert cfg.aar == base / r"venv\Scripts\aar.exe"
    assert cfg.script_api == rn.DEFAULT_SCRIPT_API


def test_config_keeps_absolute_paths_and_overrides(tmp_path: Path) -> None:
    abs_tok = str(tmp_path / "elsewhere" / "t.token")
    cfg = rn.load_config(_write(tmp_path, {**GOOD, "token_file": abs_tok, "aar": "x/aar"}))
    assert cfg.token_file == Path(abs_tok)
    assert cfg.aar == tmp_path.resolve() / "x/aar"


@pytest.mark.parametrize("key", rn.REQUIRED_KEYS)
def test_config_missing_key(tmp_path: Path, key: str) -> None:
    data = {k: v for k, v in GOOD.items() if k != key}
    with pytest.raises(rn.ConfigError, match=key):
        rn.load_config(_write(tmp_path, data))


@pytest.mark.parametrize("bad", [[], "x", {**GOOD, "node_id": ""}, {**GOOD, "store": 3}])
def test_config_rejects_bad_shapes(tmp_path: Path, bad: object) -> None:
    with pytest.raises(rn.ConfigError):
        rn.load_config(_write(tmp_path, bad))


def test_config_unreadable_or_not_json(tmp_path: Path) -> None:
    with pytest.raises(rn.ConfigError):
        rn.load_config(tmp_path / "absent.json")
    p = tmp_path / "bad.json"
    p.write_text("{nope", encoding="utf-8")
    with pytest.raises(rn.ConfigError):
        rn.load_config(p)


def test_main_reports_config_error_with_exit_2(tmp_path: Path, capsys) -> None:
    assert rn.main([str(tmp_path / "absent.json")]) == 2
    assert "cannot read config" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("ran", "backoff", "expected"),
    [
        (601, 300.0, 15.0),
        (10_000, 15.0, 15.0),
        (600, 15.0, 30.0),  # exactly 600 is not healthy
        (0, 15.0, 30.0),
        (5, 30.0, 60.0),
        (5, 200.0, 300.0),
        (5, 300.0, 300.0),
    ],
)
def test_next_backoff(ran: float, backoff: float, expected: float) -> None:
    assert rn.next_backoff(ran, backoff) == expected


def test_parse_projects_valid_and_filters_non_objects() -> None:
    loc = {"folder_path": [], "name": "Ünï", "project_id": "p1"}
    locs, err = rn.parse_projects(json.dumps([loc, "junk", 3, None]))
    assert err is None
    assert [json.loads(x) for x in locs] == [loc]
    assert "Ünï" in locs[0]  # ensure_ascii=False


@pytest.mark.parametrize("text", ["", "{bad", '{"a": 1}', '"s"', "null"])
def test_parse_projects_invalid(text: str) -> None:
    locs, err = rn.parse_projects(text)
    assert locs == [] and err


def test_empty_array_is_valid_but_idle() -> None:
    assert rn.parse_projects("[]") == ([], None)


def test_build_command_is_a_list_with_one_project_flag_per_locator(tmp_path: Path) -> None:
    cfg = rn.load_config(_write(tmp_path, GOOD))
    locs = ['{"name": "A B", "project_id": "1"}', '{"name": "C; rm -rf /"}']
    cmd = rn.build_command(cfg, locs)
    assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd)
    assert cmd[1:3] == ["node", "run"]
    assert cmd[cmd.index("--node-id") + 1] == "edit-pc"
    assert cmd[cmd.index("--hub-url") + 1] == GOOD["hub_url"]
    assert cmd[cmd.index("--token-file") + 1] == str(cfg.token_file)
    assert cmd[cmd.index("--store") + 1] == str(cfg.store)
    assert [cmd[i + 1] for i, c in enumerate(cmd) if c == "--project"] == locs  # verbatim


def test_build_env_sets_resolve_locations_and_utf8(tmp_path: Path) -> None:
    cfg = rn.load_config(_write(tmp_path, {**GOOD, "script_api": "/api", "script_lib": "/lib"}))
    env = rn.build_env(cfg, {"KEEP": "1"})
    assert env["KEEP"] == "1"
    assert env["RESOLVE_SCRIPT_API"] == "/api"
    assert env["RESOLVE_SCRIPT_LIB"] == "/lib"
    assert env["PYTHONPATH"].startswith("/api") and env["PYTHONPATH"].endswith("Modules")
    assert env["PYTHONUTF8"] == "1"


class _Done:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_resolve_running_uses_the_injected_runner_without_a_shell() -> None:
    calls: list[list[str]] = []

    def fake(argv, **kw):
        calls.append(argv)
        assert "shell" not in kw
        return _Done("Resolve.exe   123 Console 1  500,000 K")

    assert rn.resolve_running(fake) is True
    assert calls == [["tasklist", "/FI", "IMAGENAME eq Resolve.exe", "/NH"]]
    assert rn.resolve_running(lambda *a, **k: _Done("INFO: No tasks are running")) is False
    assert rn.resolve_running(lambda *a, **k: _Done(None)) is False


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s

    def mono(self) -> float:
        return self.t


def _sup(tmp_path: Path, *, running: bool, runtime: float, projects: str | None):
    cfg = rn.load_config(_write(tmp_path, GOOD))
    if projects is not None:
        cfg.projects_file.write_text(projects, encoding="utf-8")
    clock = _Clock()
    launched: list[tuple[list[str], dict[str, str], Path]] = []

    def launch(cmd, env, log_path):
        launched.append((cmd, env, log_path))
        clock.t += runtime
        return 7

    sup = rn.Supervisor(
        cfg,
        resolve_running=lambda: running,
        launch=launch,
        sleep=clock.sleep,
        monotonic=clock.mono,
        now=lambda: datetime.datetime(2026, 11, 7, 12, 0, 0),
    )
    return sup, clock, launched


def test_step_waits_when_resolve_is_not_running(tmp_path: Path) -> None:
    sup, clock, launched = _sup(tmp_path, running=False, runtime=0, projects="[]")
    sup.step()
    assert clock.sleeps == [20] and not launched


def test_step_idles_without_projects_and_logs_why(tmp_path: Path) -> None:
    sup, clock, launched = _sup(tmp_path, running=True, runtime=0, projects="{bad")
    sup.step()
    assert clock.sleeps == [60] and not launched
    log = (sup.cfg.log_dir / "supervisor-20261107.log").read_text(encoding="utf-8")
    assert "projects file unreadable" in log


def test_step_idles_when_projects_file_is_absent(tmp_path: Path) -> None:
    sup, clock, launched = _sup(tmp_path, running=True, runtime=0, projects=None)
    sup.step()
    assert clock.sleeps == [60] and not launched


def test_step_runs_worker_then_backs_off_and_resets_after_a_healthy_run(tmp_path: Path) -> None:
    loc = {"name": "P", "project_id": "p1", "folder_path": []}
    sup, clock, launched = _sup(tmp_path, running=True, runtime=5, projects=json.dumps([loc]))
    sup.step()
    assert len(launched) == 1
    assert json.loads(launched[0][0][-1]) == loc
    assert launched[0][2].name == "node-20261107.log"
    assert clock.sleeps == [20, 30.0] and sup.backoff == 30.0
    sup.step()
    assert sup.backoff == 60.0
    (tmp_path / "b").mkdir()
    sup2, clock2, _ = _sup(tmp_path / "b", running=True, runtime=700, projects=json.dumps([loc]))
    sup2.backoff = 300.0
    sup2.step()
    assert sup2.backoff == 15.0 and clock2.sleeps[-1] == 15.0
    log = (sup.cfg.log_dir / "supervisor-20261107.log").read_text(encoding="utf-8")
    assert "worker exited code=7" in log


DEPLOY = SCRIPT.parents[1]


def test_unit_templates_use_only_known_placeholders() -> None:
    import re

    known = {"AAR_ROOT", "AAR_USER", "AAR_GROUP", "TAILNET_IP", "CONSOLE_HOSTNAME"}
    units = sorted((DEPLOY / "linux").glob("*.service.in"))
    assert [u.name for u in units] == [
        "aar-hub-console.service.in",
        "aar-hub-mcp.service.in",
        "aar-hub-nodeapi.service.in",
    ]
    for unit in units:
        text = unit.read_text(encoding="utf-8")
        assert set(re.findall(r"@([A-Z_]+)@", text)) <= known
        assert "ReadWritePaths=@AAR_ROOT@/data" in text and "NoNewPrivileges=true" in text
    console = (DEPLOY / "linux" / "aar-hub-console.service.in").read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in console and "--allowed-host @CONSOLE_HOSTNAME@" in console
    nodeapi = (DEPLOY / "linux" / "aar-hub-nodeapi.service.in").read_text(encoding="utf-8")
    assert "--host @TAILNET_IP@" in nodeapi and "0.0.0.0" not in nodeapi


def test_install_script_never_writes_the_token_and_registers_interactive() -> None:
    text = (DEPLOY / "windows" / "install.ps1").read_text(encoding="utf-8")
    assert "-LogonType Interactive" in text
    assert "Disable-ScheduledTask" in text
    assert "S-1-5-18" in text and "S-1-5-32-544" in text and "/inheritance:r" in text
    assert "WriteAllText($token" not in text
    assert "edit-pc.token'))" in text  # only tested for existence in `enable`


def test_kill_on_close_job_is_a_noop_off_windows() -> None:
    class Boom:
        def __getattr__(self, name):  # any kernel32 call would be a bug
            raise AssertionError(f"kernel32.{name} called off Windows")

    assert rn.kill_on_close_job(os_name="posix", kernel32=Boom()) is None
    if rn.os.name != "nt":
        assert rn.kill_on_close_job() is None


def test_kill_on_close_job_failure_paths_return_none() -> None:
    closed: list[int] = []

    class K32:
        def __init__(self, job, set_ok, assign_ok):
            self.job, self.set_ok, self.assign_ok = job, set_ok, assign_ok

        def CreateJobObjectW(self, *a):
            return self.job

        def SetInformationJobObject(self, *a):
            return self.set_ok

        def GetCurrentProcess(self):
            return -1

        def AssignProcessToJobObject(self, *a):
            return self.assign_ok

        def CloseHandle(self, h):
            closed.append(h)

    assert rn.kill_on_close_job("nt", K32(0, 1, 1)) is None
    assert rn.kill_on_close_job("nt", K32(5, 0, 1)) is None
    assert rn.kill_on_close_job("nt", K32(6, 1, 0)) is None
    assert closed == [5, 6]
    assert rn.kill_on_close_job("nt", K32(7, 1, 1)) == 7


def test_install_disable_kills_the_whole_tree_only_under_root() -> None:
    text = (DEPLOY / "windows" / "install.ps1").read_text(encoding="utf-8")
    assert "function Get-NodeProcesses" in text
    assert "ExecutablePath.StartsWith($prefix" in text and "*node run*" in text
    assert "disable FAILED" in text
    # Status and disable share the one selector; neither filters on a bare process name.
    assert text.count("Get-NodeProcesses") >= 4
    assert "run_node.py*'" not in text
