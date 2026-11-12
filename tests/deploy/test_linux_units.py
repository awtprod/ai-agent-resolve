"""The Linux unit templates render with every placeholder filled (S19a adds the worker)."""

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
