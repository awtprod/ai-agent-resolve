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
    assert "ReadWritePaths=/var/lib/aar/data /srv/aar-media/work /srv/aar-media/share/exports" in (
        worker
    )
    assert "Restart=on-failure" in worker and "KillSignal=SIGTERM" in worker
    for unit in ("aar-hub-mcp.service", "aar-hub-console.service"):
        assert "EnvironmentFile=/var/lib/aar/hub.env" in (out / unit).read_text(encoding="utf-8")
    hub_env = dict(
        line.split("=", 1)
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
