"""Windows production guard: `aar` must load even though delivery's lock is POSIX-only.

`aar.cli` imports the deliver CLI to register its commands, and `aar.deliver.__init__` imports
`aar.deliver.service`. On Windows there is no `fcntl`, so a module-level `fcntl` import would break
EVERY `aar` command (including `aar node run`). The lock imports `fcntl` lazily, and delivery
refuses on Windows before taking any lock.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_aar_loads_without_fcntl(monkeypatch: pytest.MonkeyPatch) -> None:
    """With `fcntl` unimportable (as on Windows), importing `aar.cli` and every registered command
    module still works, and `aar --help` / `aar node --help` run."""
    monkeypatch.setitem(sys.modules, "fcntl", None)  # any `import fcntl` now raises
    # Drop cached deliver + cli modules so they re-import under the guard.
    for name in [n for n in list(sys.modules) if n == "aar.cli" or n.startswith("aar.deliver")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    cli = importlib.import_module("aar.cli")
    # Registering every command group (including deliver) imported nothing POSIX-only.
    cli.build_parser()
    assert cli.main(["--help"]) == 0
    assert cli.main(["node", "--help"]) == 0


def test_delivery_refuses_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, delivery raises a typed `unsupported_platform` before any lock or file work."""
    from aar.deliver import service

    monkeypatch.setattr(service.os, "name", "nt")
    with pytest.raises(service.DeliverError) as exc:
        service._require_posix()
    assert exc.value.code == "unsupported_platform"
