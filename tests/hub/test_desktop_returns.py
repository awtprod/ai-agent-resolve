from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest
from write_support import AGENT, make_hub

from aar.plan.model import EditPlan
from aar.recipes.desktop_returns import _return_directory, returned_edits
from aar.recipes.desktop_returns import receive_one as _receive_one

sys.path.insert(0, str(Path(__file__).parents[1] / "plan"))
from timeline_import_support import DrtClip, drt_members


@pytest.fixture(autouse=True)
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AAR_WORK_DIR", str(tmp_path / "work"))


def receive_one(log, store, project_id, request, path, actor):
    with _return_directory(path.parent, path.parent) as directory_fd:
        return _receive_one(
            log, store, project_id, request, Path(path.name), actor, directory_fd=directory_fd
        )


def packet(tmp_path: Path):
    (tmp_path / "hub").mkdir()
    hub = make_hub(tmp_path / "hub")
    source = tmp_path / "proxy.mp4"
    source.write_bytes(b"media")
    plan = EditPlan.model_validate(
        {
            "profile": "ccn",
            "version": 2,
            "project": "Show",
            "timeline": "Edit",
            "kind": "longform",
            "fps": {"num": 25, "den": 1},
            "resolution": {"width": 1920, "height": 1080},
            "sources": [{"id": "a", "media": str(source), "duration": 1000, "start_timecode": 0}],
            "segments": [{"source": "a", "src_in": 0, "src_out": 100}],
        }
    )
    aid = hub.store.put(plan.model_dump_json().encode(), actor=AGENT, project_id=hub.project_id)
    request = {
        "recipe_node": "recipe-test",
        "round": 1,
        "plan_sha256": aid,
        "media_map": {str(tmp_path): "E:\\Show"},
        "timeline": "Edit",
    }
    path = tmp_path / "edit.drt"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in drt_members([DrtClip("E:\\Show\\proxy.mp4", 0, 80, 0)]).items():
            archive.writestr(name, data)
    os.utime(path, (1, 1))
    return hub, request, path


def test_return_keeps_drt_and_reconstructed_plan_and_diff(tmp_path: Path) -> None:
    hub, request, path = packet(tmp_path)
    try:
        result = receive_one(hub.log, hub.store, hub.project_id, request, path, AGENT)
        assert result["status"] == "needs_review"
        assert result["partial_reconstruction"] is True
        assert hub.store.read_bytes(result["drt"]) == path.read_bytes()
        edited = json.loads(hub.store.read_bytes(result["plan"]))
        assert edited["segments"][0]["src_out"] == 80
        assert json.loads(hub.store.read_bytes(result["diff"]))["limits"]
        assert returned_edits(hub.log, hub.project_id, "recipe-test") == [result]
        seq = hub.log.latest_seq()
        assert receive_one(hub.log, hub.store, hub.project_id, request, path, AGENT) == result
        assert hub.log.latest_seq() == seq
    finally:
        hub.log.close()


def test_bad_return_is_visible_and_not_retried_forever(tmp_path: Path) -> None:
    hub, request, path = packet(tmp_path)
    try:
        path.write_bytes(b"not a drt")
        os.utime(path, (1, 1))
        result = receive_one(hub.log, hub.store, hub.project_id, request, path, AGENT)
        assert result["status"] == "failed"
        assert result["error"]
        seq = hub.log.latest_seq()
        receive_one(hub.log, hub.store, hub.project_id, request, path, AGENT)
        assert hub.log.latest_seq() == seq
    finally:
        hub.log.close()


def test_still_copying_and_symlink_return_are_not_ingested(tmp_path: Path) -> None:
    hub, request, path = packet(tmp_path)
    try:
        os.utime(path, None)
        assert receive_one(hub.log, hub.store, hub.project_id, request, path, AGENT) is None
        link = tmp_path / "linked.drt"
        link.symlink_to(path)
        with pytest.raises(ValueError, match="symlink"):
            receive_one(hub.log, hub.store, hub.project_id, request, link, AGENT)
    finally:
        hub.log.close()


def test_worker_picks_up_return_and_status_exposes_it(ccn_env, tmp_path: Path) -> None:
    from ccn_recipe_support import agent_cut, make_run, outputs, start, work

    from aar.recipes.run import recipe_status
    from aar.recipes.worker import run_worker

    env = ccn_env
    show = env.show()
    run = make_run(env, env.params(show))
    start(env, run)
    work(env)
    agent_cut(env, run)
    work(env)
    exported = outputs(env, run, "export")
    request = exported["desktop_return"]
    path = Path(request["directory"]) / "my-edit.drt"
    # Read the exact plan that this export represented; trim its first clip in the DRT.
    base = EditPlan.model_validate_json(env.hub.store.read_bytes(request["request"]["plan_sha256"]))
    clip = base.segments[0]
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in drt_members(
            [DrtClip("E:\\Shows\\cam-a.mov", 0, clip.src_out - clip.src_in - 10, clip.src_in)],
            rate=base.fps.num / base.fps.den,
        ).items():
            archive.writestr(name, data)
    os.utime(path, (1, 1))
    run_worker(env.hub.log, env.hub.store, AGENT, once=True)
    status = recipe_status(env.hub.log, run)
    received = status["returned_edits"][-1]
    assert received["status"] == "needs_review"
    assert received["base_plan"] == request["request"]["plan_sha256"]
    assert received["plan"] != received["base_plan"]
    assert outputs(env, run, "cut")["plan_sha256"] == received["base_plan"]
    before = len(returned_edits(env.hub.log, env.hub.project_id, run))
    run_worker(env.hub.log, env.hub.store, AGENT, once=True)
    assert len(returned_edits(env.hub.log, env.hub.project_id, run)) == before


def test_bad_entry_does_not_block_other_returned_edits(ccn_env, tmp_path: Path) -> None:
    from ccn_recipe_support import agent_cut, make_run, outputs, start, work

    from aar.recipes.desktop_returns import scan_returns

    env = ccn_env
    run = make_run(env, env.params(env.show()))
    start(env, run)
    work(env)
    agent_cut(env, run)
    work(env)
    folder = Path(outputs(env, run, "export")["desktop_return"]["directory"])
    (folder / "a-symlink.drt").symlink_to(tmp_path / "absent")
    (folder / "b-bad.drt").write_bytes(b"not a drt")
    os.utime(folder / "b-bad.drt", (1, 1))
    scan_returns(env.hub.log, env.hub.store, AGENT)
    records = returned_edits(env.hub.log, env.hub.project_id, run)
    assert {record["file"] for record in records} == {"a-symlink.drt", "b-bad.drt"}
    assert all(record["status"] == "failed" for record in records)
    count = env.hub.log.latest_seq()
    scan_returns(env.hub.log, env.hub.store, AGENT)
    assert env.hub.log.latest_seq() == count


def test_cancel_during_import_does_not_publish_return(ccn_env, monkeypatch) -> None:
    from ccn_recipe_support import agent_cut, make_run, outputs, start, work
    from write_support import HUMAN

    from aar.recipes import desktop_returns
    from aar.recipes.run import recipe_cancel

    env = ccn_env
    run = make_run(env, env.params(env.show()))
    start(env, run)
    work(env)
    agent_cut(env, run)
    work(env)
    packet = outputs(env, run, "export")["desktop_return"]
    path = Path(packet["directory"]) / "cancelled.drt"
    path.write_bytes(b"malformed file is enough to reach artifact snapshot")
    os.utime(path, (1, 1))
    original = env.hub.store.put
    cancelled = False

    def cancel_then_put(*args, **kwargs):
        nonlocal cancelled
        if not cancelled:
            recipe_cancel(env.hub.log, run, HUMAN, idempotency_key="cancel-return")
            cancelled = True
        return original(*args, **kwargs)

    monkeypatch.setattr(env.hub.store, "put", cancel_then_put)
    desktop_returns.scan_returns(env.hub.log, env.hub.store, AGENT)
    assert not returned_edits(env.hub.log, env.hub.project_id, run)


@pytest.mark.parametrize("swap_ancestor", [False, True])
def test_directory_swap_cannot_import_outside_bytes(tmp_path, monkeypatch, swap_ancestor):
    from aar.recipes import desktop_returns
    from aar.recipes.events import EVENT_RESULT, RUNNER

    hub, request, source = packet(tmp_path)
    try:
        root = tmp_path / "exports"
        folder = root / "returns" / "recipe" / "r1" / "attempt"
        folder.mkdir(parents=True)
        path = folder / source.name
        source.rename(path)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        victim = root / "returns" if swap_ancestor else folder
        outside = tmp_path / "private"
        secret_path = outside / path.relative_to(victim)
        secret_path.parent.mkdir(parents=True)
        secret = b"another project's private timeline"
        secret_path.write_bytes(secret)
        os.utime(secret_path, (1, 1))
        monkeypatch.setenv("AAR_EXPORT_DIR", str(root))
        hub.log.append(
            EVENT_RESULT,
            RUNNER,
            {
                "outcome": "done",
                "outputs": {"desktop_return": {"directory": str(folder), "request": request}},
            },
            project_id=hub.project_id,
        )
        original = desktop_returns.receive_one
        swapped = False

        def swap_then_receive(*args, **kwargs):
            nonlocal swapped
            victim.rename(tmp_path / "held")
            victim.symlink_to(outside, target_is_directory=True)
            swapped = True
            return original(*args, **kwargs)

        monkeypatch.setattr(desktop_returns, "receive_one", swap_then_receive)
        desktop_returns.scan_returns(hub.log, hub.store, AGENT)
        assert swapped
        assert not hub.store.has(hashlib.sha256(secret).hexdigest())
        assert hub.store.has(expected)
    finally:
        hub.log.close()


def test_symlink_directory_is_refused_and_open_handles_close(tmp_path, monkeypatch):
    root = tmp_path / "exports"
    parent = root / "returns"
    parent.mkdir(parents=True)
    outside = tmp_path / "private"
    outside.mkdir()
    (parent / "attempt").symlink_to(outside, target_is_directory=True)
    original = os.open
    opened = []

    def track_open(*args, **kwargs):
        fd = original(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", track_open)
    with pytest.raises(OSError), _return_directory(root, parent / "attempt"):
        pytest.fail("a symlink directory was opened")
    assert len(opened) == 2
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_descriptor_return_rejects_paths_outside_open_directory(tmp_path):
    hub, request, path = packet(tmp_path)
    try:
        folder = tmp_path / "returns"
        folder.mkdir()
        with _return_directory(folder, folder) as directory_fd:
            for outside in (path, Path("..") / path.name):
                with pytest.raises(ValueError, match="single file name"):
                    _receive_one(
                        hub.log,
                        hub.store,
                        hub.project_id,
                        request,
                        outside,
                        AGENT,
                        directory_fd=directory_fd,
                    )
        assert not hub.store.has(hashlib.sha256(path.read_bytes()).hexdigest())
        assert not list((tmp_path / "work").glob("aar-return-*"))
    finally:
        hub.log.close()
