"""``aar deliver`` CLI: argument handling and the exit-code contract.

The edit host and the default QA are monkeypatched so the CLI runs offline and fast: 0 done,
1 QA failed, 2 bad input or a missing tool, 3 a transport error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deliver_support import FakeEditHost, RaisingEditHost, make_hub, qa_fail, qa_pass

import aar.deliver.cli as cli
import aar.deliver.service as service
from aar.cli import main
from aar.host.errors import RemoteCommandError


def _common(hub, tmp_path: Path, sources: Path) -> list[str]:
    return [
        "--remote",
        str(sources / "render_720p.mp4"),
        "--frame-count",
        "50",
        "--frame-rate",
        "25",
        "--duration",
        "2",
        "--render-job-id",
        "rj-1",
        "--deliverable",
        "short-1",
        "--project",
        hub.project_id,
        "--kind",
        "short",
        "--db",
        str(tmp_path / "hub.db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--workdir",
        str(tmp_path / "work"),
        "--actor",
        "agent:recipe-1",
        "--json",
    ]


def _use_fake_host(monkeypatch: pytest.MonkeyPatch, sources: Path) -> None:
    monkeypatch.setattr(cli, "_edit_host", lambda: FakeEditHost(sources))


def test_review_exit_0(tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_pass())
    argv = [
        "deliver",
        "review",
        *_common(hub, tmp_path, sources),
        "--subject",
        hub.subject_id,
        "--title",
        "Acme",
    ]
    assert main(argv) == 0


def test_review_exit_1_on_qa_fail(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_fail())
    argv = [
        "deliver",
        "review",
        *_common(hub, tmp_path, sources),
        "--subject",
        hub.subject_id,
        "--title",
        "Acme",
    ]
    assert main(argv) == 1


def test_review_exit_2_on_bad_input(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    # Neither --render-json nor the explicit render-output fields: a bad input (exit 2).
    argv = [
        "deliver",
        "review",
        "--deliverable",
        "short-1",
        "--project",
        hub.project_id,
        "--subject",
        hub.subject_id,
        "--title",
        "Acme",
        "--kind",
        "short",
        "--db",
        str(tmp_path / "hub.db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--workdir",
        str(tmp_path / "work"),
        "--actor",
        "agent:recipe-1",
        "--json",
    ]
    assert main(argv) == 2


def test_review_exit_3_on_transport_error(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = make_hub(tmp_path)
    monkeypatch.setattr(cli, "_edit_host", lambda: RaisingEditHost(RemoteCommandError(1, "scp")))
    argv = [
        "deliver",
        "review",
        *_common(hub, tmp_path, sources),
        "--subject",
        hub.subject_id,
        "--title",
        "Acme",
    ]
    assert main(argv) == 3


def test_master_exit_0(tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_pass("longform"))
    argv = ["deliver", "master", *_common(hub, tmp_path, sources)]
    assert main(argv) == 0


def test_master_exit_1_on_qa_fail(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_fail("longform"))
    argv = ["deliver", "master", *_common(hub, tmp_path, sources)]
    assert main(argv) == 1
