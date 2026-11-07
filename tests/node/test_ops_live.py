"""Read ops against a running Resolve Studio. Run only with ``AAR_LIVE=1``. Read-only.

The S13h1 build tests at the end are the exception: they write, so they also need
``AAR_LIVE_BUILD=1``. They work only on a scratch timeline (and a scratch bin) they create in
the open project, and delete nothing: remove the scratch timeline and bin by hand afterwards.

The ops go through the broker as always, with a config that takes nothing from the human: no
Deliver preset is saved, the Project Manager path is not walked, nothing is saved, and the
broker is never ticked into its restore (which would call setters). The target is the project
that is already open, so the broker never switches. The Project Manager is never navigated:
the target's folder path is only known without navigating when the Project Manager is at its
root, so the tests skip otherwise.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from aar.node.branch_registry import BranchRegistry
from aar.node.broker import Broker, BrokerConfig, OpStatus
from aar.node.gateway import ScriptApiGateway
from aar.node.leases import Holder, LeaseManager
from aar.node.locator import ProjectLocator
from aar.node.ops import submit_op
from aar.node.presence import PresenceGuard

from .fake_resolve import FakeClock

pytestmark = pytest.mark.live

READ_ONLY = BrokerConfig(
    snapshot_deliver_preset=False,
    snapshot_project_manager_path=False,
    save_before_switch=False,
    restore_delay_seconds=1e9,  # never restore: the restore path calls setters
)


@pytest.fixture(scope="module")
def resolve() -> Any:
    import DaVinciResolveScript as dvr  # type: ignore[import-not-found]

    app = dvr.scriptapp("Resolve")
    assert app is not None, "scriptapp('Resolve') returned None: is Resolve Studio running?"
    return app


def _identity(resolve: Any) -> tuple[str, str, str | None, str | None]:
    project = resolve.GetProjectManager().GetCurrentProject()
    assert project is not None, "no project is open"
    timeline = project.GetCurrentTimeline()
    return (
        project.GetName(),
        project.GetUniqueId(),
        timeline.GetName() if timeline is not None else None,
        timeline.GetUniqueId() if timeline is not None else None,
    )


def _run(resolve: Any, name: str, params: dict[str, Any]) -> Any:
    pm = resolve.GetProjectManager()
    if pm.GetCurrentFolder() != "":
        pytest.skip("the Project Manager is not at its root; its path would need navigation")
    before = _identity(resolve)
    gw = ScriptApiGateway(resolve)
    clock = FakeClock()
    presence = PresenceGuard(idle_window=1.0)
    broker = Broker(
        gw, leases=LeaseManager(clock), presence=presence, clock=clock, config=READ_ONLY
    )
    presence.observe(gw.get_context(), clock())
    clock.advance(1.0)
    target = ProjectLocator((), before[0], before[1])
    op_id = submit_op(broker, name, params, target=target, holder=Holder.agent("live-test"))
    broker.run_until_idle()
    result = broker.result(op_id)
    assert _identity(resolve) == before, "the project or timeline changed"
    assert result is not None, "the op did not run (a human may be working: it was deferred)"
    assert result.status is OpStatus.OK, result.error
    return result.value


def test_live_timeline_list(resolve: Any) -> None:
    result = _run(resolve, "timeline.list", {})
    assert result.timelines and all(t.unique_id for t in result.timelines)
    assert sum(t.is_current for t in result.timelines) <= 1


def test_live_timeline_describe_current(resolve: Any) -> None:
    if _identity(resolve)[2] is None:
        pytest.skip("no current timeline")
    result = _run(resolve, "timeline.describe", {"max_items_per_track": 50})
    print(result.text)
    assert result.timeline.fps and result.timeline.duration >= 0
    for track in result.tracks:
        for item in track.items:
            assert item.end - item.start == item.duration >= 0
        for gap in track.gaps:
            assert gap.frames > 0


def test_live_media_find_root_bin(resolve: Any) -> None:
    result = _run(resolve, "media.find", {"limit": 20})
    assert result.bin_path == () and not result.recursive


def test_live_project_find_current_folder(resolve: Any) -> None:
    result = _run(resolve, "project.find", {"recursive": False})
    assert result.project_manager_path == ()
    assert all(p.locator.folder_path == () for p in result.projects)
    assert result.open_project_id == _identity(resolve)[1]


def test_live_cloud_read_ops(resolve: Any) -> None:
    """The read-only S10 cloud ops against the live instance: list, sync status, freshness.

    Strictly read-only: no LoadProject, no Project Manager navigation (``_run`` skips unless
    at the root), no media-pool folder change, no RefreshFolders, no Set*/Import*/Restore*.
    """
    open_id = _identity(resolve)[1]

    projects = _run(resolve, "cloud.projects", {})
    assert projects.open_project_id == open_id
    assert projects.database.db_name  # some current database is reported
    assert any(p.is_open and p.unique_id == open_id for p in projects.projects)

    sync = _run(resolve, "cloud.sync_status", {"max_clips": 500})
    assert sync.scanned == len(sync.clips) <= sync.total_clips
    assert sum(sync.status_counts.values()) == sync.scanned
    for clip in sync.clips:
        assert clip.media_id and clip.cloud_sync
        assert clip.proxy_only == (clip.has_proxy and clip.original_offline)
        assert clip.fully_offline == (clip.original_offline and not clip.has_proxy)
        assert not (clip.proxy_only and clip.fully_offline)
    # Report proxy-only media (the audio-loss case) and the raw status codes seen live.
    print("cloud sync codes:", sync.status_counts)
    print("proxy-only:", sync.proxy_only, "fully-offline:", sync.fully_offline)

    fresh = _run(resolve, "cloud.freshness", {})
    assert fresh.folder_count == len(fresh.folders)
    assert fresh.stale_count == sum(f.stale for f in fresh.folders)


# -- S13h1 build ops: write, on a scratch timeline this module creates --------------------------

build_only = pytest.mark.skipif(
    os.environ.get("AAR_LIVE_BUILD") != "1",
    reason="writes to the open project: set AAR_LIVE_BUILD=1 as well as AAR_LIVE=1",
)
BUILD_AGENT = "live-build-test"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run this build test")
    return value


@pytest.fixture(scope="module")
def scratch(resolve: Any) -> dict[str, Any]:
    """A scratch timeline made here, registered as this test agent's branch. Never deleted."""
    pm = resolve.GetProjectManager()
    if pm.GetCurrentFolder() != "":
        pytest.skip("the Project Manager is not at its root; its path would need navigation")
    project = pm.GetCurrentProject()
    assert project is not None, "no project is open"
    name = f"aar-scratch-s13h1-{uuid.uuid4().hex[:8]}"
    timeline = project.GetMediaPool().CreateEmptyTimeline(name)
    assert timeline is not None, "CreateEmptyTimeline returned None"
    registry = BranchRegistry()
    uid = str(timeline.GetUniqueId())
    registry.register(
        str(project.GetUniqueId()), uid, source_timeline_id=uid, agent_id=BUILD_AGENT, version=1
    )
    print("scratch timeline (delete it by hand afterwards):", name)
    return {
        "ref": {"name": name, "unique_id": uid},
        "registry": registry,
        "locator": ProjectLocator((), str(project.GetName()), str(project.GetUniqueId())),
    }


def _build(resolve: Any, scratch: dict[str, Any], name: str, params: dict[str, Any]) -> Any:
    """Run a build op with the branch guard on, on the scratch timeline only."""
    gw = ScriptApiGateway(resolve)
    clock = FakeClock()
    presence = PresenceGuard(idle_window=1.0)
    broker = Broker(
        gw, leases=LeaseManager(clock), presence=presence, clock=clock, config=READ_ONLY
    )
    presence.observe(gw.get_context(), clock())
    clock.advance(1.0)
    op_id = submit_op(
        broker,
        name,
        params,
        target=scratch["locator"],
        holder=Holder.agent(BUILD_AGENT),
        require_branch=True,
        branch_registry=scratch["registry"],
    )
    broker.run_until_idle()
    result = broker.result(op_id)
    assert result is not None, "the op did not run (a human may be working: it was deferred)"
    assert result.status is OpStatus.OK, result.error
    return result.value


@build_only
def test_live_build_tracks(resolve: Any, scratch: dict[str, Any]) -> None:
    params = {
        "timeline": scratch["ref"],
        "tracks": [
            {"type": "audio", "index": 2, "name": "aar mono", "audio_format": "mono"},
            {"type": "video", "index": 2, "name": "aar titles"},
        ],
    }
    first = _build(resolve, scratch, "track.ensure", params)
    print("track.ensure:", first.model_dump())
    assert [t.audio_format for t in first.tracks] == ["mono", None]
    again = _build(resolve, scratch, "track.ensure", params)
    assert [t.changed for t in again.tracks] == [[], []]


@build_only
def test_live_build_import_place_and_finish(resolve: Any, scratch: dict[str, Any]) -> None:
    """media.import, then two appended ranges with handles, a centred dissolve at their cut,
    a transform, a CDL and (if ``AAR_LIVE_LUT`` names one) a LUT, each read back and rerun."""
    path = _env("AAR_LIVE_MEDIA")  # an edit-PC video file, e.g. X:/acme/media/a.mov
    imported = _build(
        resolve, scratch, "media.import", {"paths": [path], "bin_path": ["aar-scratch"]}
    )
    print("media.import:", imported.model_dump())
    media = imported.media[0].media.model_dump()
    rerun = _build(resolve, scratch, "media.import", {"paths": [path], "bin_path": ["aar-scratch"]})
    assert rerun.media[0].status == "present"
    appended = _build(
        resolve,
        scratch,
        "timeline.append_ranges",
        {
            "timeline": scratch["ref"],
            "ranges": [
                {"media": media, "src_in": 100, "src_out": 200, "media_type": "video"},
                {"media": media, "src_in": 400, "src_out": 500, "media_type": "video"},
            ],
        },
    )
    a, b = (r.video for r in appended.ranges)
    a_ref = {"track_type": "video", "track": 1, "item_id": a.item_id}
    b_ref = {"track_type": "video", "track": 1, "item_id": b.item_id}
    tl = scratch["ref"]
    target = {"item": a_ref, "position": "end", "alignment": "center", "frames": 12}
    added = _build(resolve, scratch, "transition.add", {"timeline": tl, "targets": [target]})
    print("transition.add:", added.model_dump())
    again = _build(resolve, scratch, "transition.add", {"timeline": tl, "targets": [target]})
    assert again.transitions[0].status == "present"
    transform = {"item": b_ref, "zoom": 1.1, "rotation_deg": 2.0, "flip_h": True}
    moved = _build(resolve, scratch, "item.set_transform", {"timeline": tl, "targets": [transform]})
    print("item.set_transform:", moved.model_dump())
    still = _build(resolve, scratch, "item.set_transform", {"timeline": tl, "targets": [transform]})
    assert still.items[0].status == "present"
    colour: dict[str, Any] = {"item": b_ref, "cdl": {"saturation": 0.95}}
    if os.environ.get("AAR_LIVE_LUT"):
        colour["lut"] = os.environ["AAR_LIVE_LUT"]
    coloured = _build(resolve, scratch, "item.set_color", {"timeline": tl, "targets": [colour]})
    print("item.set_color:", coloured.model_dump())
    assert coloured.items[0].cdl is not None and not coloured.items[0].cdl.verified


@build_only
def test_live_build_title(resolve: Any, scratch: dict[str, Any]) -> None:
    template = _env("AAR_LIVE_TITLE_TEMPLATE")  # media-pool id of a Fusion Text+ title clip
    tl = scratch["ref"]
    _build(
        resolve,
        scratch,
        "track.ensure",
        {"timeline": tl, "tracks": [{"type": "video", "index": 2}]},
    )
    title = {
        "template": {"unique_id": template},
        "track": 2,
        "record_in": 30,
        "record_out": 120,
        "lines": ["aar line one", "aar line two"],
    }
    placed = _build(resolve, scratch, "title.place", {"timeline": tl, "titles": [title]})
    print("title.place:", placed.model_dump())
    assert placed.titles[0].text == "aar line one\naar line two"
    again = _build(resolve, scratch, "title.place", {"timeline": tl, "titles": [title]})
    assert again.titles[0].status == "present"
