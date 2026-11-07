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


def _review_argv(hub, tmp_path: Path, sources: Path, **over: str) -> list[str]:
    argv = [
        "deliver",
        "review",
        *_common(hub, tmp_path, sources),
        "--subject",
        over.get("subject", hub.subject_id),
        "--title",
        "Acme",
    ]
    if "project" in over:
        argv[argv.index("--project") + 1] = over["project"]
    return argv


def test_v48_missing_ffmpeg_exits_2(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aar.media import MediaToolMissing

    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)

    def boom(**_: object) -> None:
        raise MediaToolMissing("ffmpeg not found")

    monkeypatch.setattr(service, "deliver_review", boom)
    assert main(_review_argv(hub, tmp_path, sources)) == 2


def test_v49_other_host_error_exits_3(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aar.host.errors import PathRefused

    hub = make_hub(tmp_path)
    monkeypatch.setattr(cli, "_edit_host", lambda: RaisingEditHost(PathRefused("x", "no")))
    assert main(_review_argv(hub, tmp_path, sources)) == 3


def test_v50_render_json_without_duration_exits_2(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_pass())
    rj = tmp_path / "render.json"
    rj.write_text(
        json.dumps(
            {
                "output_path": str(sources / "render_720p.mp4"),
                "frame_count": 50,
                "frame_rate": 25.0,
                "render_job_id": "rj-1",
            }
        )
    )
    argv = [
        "deliver",
        "review",
        "--render-json",
        str(rj),
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


def test_unknown_subject_exits_2(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_pass())
    assert main(_review_argv(hub, tmp_path, sources, subject="no-such-node")) == 2


def test_unregistered_project_exits_2(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_pass())
    assert main(_review_argv(hub, tmp_path, sources, project="proj-nope")) == 2


def test_v71_delivery_in_progress_exits_3(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_pass())
    ddir = service._delivery_dir(str(tmp_path / "work"), hub.project_id, "short-1", "rj-1")
    fd = service._acquire_delivery_lock(ddir)
    try:
        assert main(_review_argv(hub, tmp_path, sources)) == 3
    finally:
        service._release_delivery_lock(fd)


def test_r3_1_bad_db_exits_2(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --db that cannot be opened (a missing directory) exits 2 with a typed message."""
    hub = make_hub(tmp_path)
    _use_fake_host(monkeypatch, sources)
    monkeypatch.setattr(service, "_default_qa_check", qa_pass())
    common = _common(hub, tmp_path, sources)
    common[common.index("--db") + 1] = str(tmp_path / "missing_dir" / "hub.db")
    argv = ["deliver", "review", *common, "--subject", hub.subject_id, "--title", "Acme"]
    assert main(argv) == 2
