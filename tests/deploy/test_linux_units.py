"""The Linux unit templates render with every placeholder filled (S19a adds the worker, S23b
the lessons service and its timer)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "deploy" / "linux"
VALUES = {
    "AAR_ROOT": "/var/lib/aar",
    "AAR_USER": "aar",
    "AAR_GROUP": "aar",
    "TAILNET_IP": "tailnet-address",
    "CONSOLE_HOSTNAME": "console.example",
    "AAR_MEDIA_ROOT": "/srv/aar-media",
    "AAR_WORK_DIR": "/srv/aar-media/work",
    "AAR_EXPORT_DIR": "/srv/aar-media/share/exports",
}


def test_install_units_renders_the_worker_and_the_env_file(tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **VALUES}
    subprocess.run(["sh", str(KIT / "install-units.sh"), str(out)], env=env, check=True, timeout=30)
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "aar-hub-console.service",
        "aar-hub-lessons.service",
        "aar-hub-lessons.timer",
        "aar-hub-mcp.service",
        "aar-hub-nodeapi.service",
        "aar-hub-worker.service",
        "hub.env",
    ]
    for path in out.iterdir():
        assert "@" not in path.read_text(encoding="utf-8"), path.name
    worker = (out / "aar-hub-worker.service").read_text(encoding="utf-8")
    assert "aar recipe worker --db /var/lib/aar/data/hub.db" in worker
    assert "--recipes aar.recipes.ccn" in worker
    assert "EnvironmentFile=/var/lib/aar/hub.env" in worker
    for line in (
        "ReadWritePaths=/var/lib/aar/data",
        'ReadWritePaths="/srv/aar-media/work"',
        'ReadWritePaths="/srv/aar-media/share/exports"',
    ):
        assert line in worker.splitlines()
    assert "Restart=on-failure" in worker and "KillSignal=SIGTERM" in worker
    for unit in ("aar-hub-mcp.service", "aar-hub-console.service"):
        assert "EnvironmentFile=/var/lib/aar/hub.env" in (out / unit).read_text(encoding="utf-8")
    hub_env = dict(
        line.replace('"', "").split("=", 1)
        for line in (out / "hub.env").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    assert hub_env == {
        "AAR_RECIPES": "aar.recipes.ccn",
        "AAR_MEDIA_ROOT": "/srv/aar-media",
        "AAR_MODEL_DIR": "/srv/aar-media/models",
        "AAR_YUNET_MODEL": "/srv/aar-media/models/face_detection_yunet_2023mar.onnx",
        "AAR_WORK_DIR": "/srv/aar-media/work",
        "AAR_EXPORT_DIR": "/srv/aar-media/share/exports",
    }


def _render(tmp_path: Path, **changes: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **VALUES, **changes}
    return subprocess.run(
        ["sh", str(KIT / "install-units.sh"), str(tmp_path / "out")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_paths_with_spaces_and_sed_characters_render_quoted_and_whole(tmp_path: Path) -> None:
    work = "/srv/aar media/work | & tmp"
    run = _render(tmp_path, AAR_WORK_DIR=work, AAR_MEDIA_ROOT="/srv/a|b & c")
    assert run.returncode == 0, run.stderr
    worker = (tmp_path / "out" / "aar-hub-worker.service").read_text(encoding="utf-8")
    assert f'ReadWritePaths="{work}"' in worker.splitlines()
    env = (tmp_path / "out" / "hub.env").read_text(encoding="utf-8")
    assert f'AAR_WORK_DIR="{work}"' in env.splitlines()
    assert 'AAR_MODEL_DIR="/srv/a|b & c/models"' in env.splitlines()


def test_values_systemd_would_misread_are_refused(tmp_path: Path) -> None:
    for name, value in (
        ("AAR_WORK_DIR", 'say "hi"'),
        ("AAR_EXPORT_DIR", "/srv/100%"),
        ("AAR_MEDIA_ROOT", "/srv/$HOME"),
        ("AAR_ROOT", "/var/lib/a ar"),
    ):
        run = _render(tmp_path, **{name: value})
        assert run.returncode == 2 and name in run.stderr, (name, run.stderr)


def _settings(text: str) -> dict[str, list[str]]:
    """``key -> values`` of a unit file (a key may repeat), comments and sections left out."""
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line and not line.startswith(("#", "[")):
            key, _, value = line.partition("=")
            found.setdefault(key, []).append(value)
    return found


def test_the_lessons_timer_runs_a_hardened_oneshot_pass_with_the_key_outside_hub_env(
    tmp_path: Path,
) -> None:
    run = _render(tmp_path)
    assert run.returncode == 0, run.stderr
    out = tmp_path / "out"
    service = _settings((out / "aar-hub-lessons.service").read_text(encoding="utf-8"))
    console = _settings((out / "aar-hub-console.service").read_text(encoding="utf-8"))
    assert service["Type"] == ["oneshot"]
    assert service["ExecStart"] == [
        "/var/lib/aar/venv/bin/aar hub lessons propose --db /var/lib/aar/data/hub.db --limit 5"
    ]
    # hub.env first, then the optional lessons.env with the gateway key ("-": may be missing).
    assert service["EnvironmentFile"] == ["/var/lib/aar/hub.env", "-/var/lib/aar/lessons.env"]
    for key in (
        "User",
        "Group",
        "WorkingDirectory",
        "NoNewPrivileges",
        "ProtectSystem",
        "ProtectHome",
        "PrivateTmp",
        "PrivateDevices",
        "ReadWritePaths",
        "RestrictAddressFamilies",
    ):
        assert service[key] == console[key], key
    assert service["UMask"] == ["0077"]
    assert service["TimeoutStartSec"] == ["30min"]
    assert "Restart" not in service  # the timer starts the next pass
    timer_text = (out / "aar-hub-lessons.timer").read_text(encoding="utf-8")
    assert "[Timer]" in timer_text.splitlines() and "[Install]" in timer_text.splitlines()
    timer = _settings(timer_text)
    assert timer["OnBootSec"] == ["10min"] and timer["OnUnitInactiveSec"] == ["10min"]
    assert timer["WantedBy"] == ["timers.target"]
    # No key in any rendered file: it lives only in lessons.env, which nothing here writes.
    for path in out.iterdir():
        assert "AAR_LLM_API_KEY=" not in path.read_text(encoding="utf-8"), path.name
