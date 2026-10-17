"""Read ops against a running Resolve Studio. Run only with ``AAR_LIVE=1``. Read-only.

The ops go through the broker as always, with a config that takes nothing from the human: no
Deliver preset is saved, the Project Manager path is not walked, nothing is saved, and the
broker is never ticked into its restore (which would call setters). The target is the project
that is already open, so the broker never switches. The Project Manager is never navigated:
the target's folder path is only known without navigating when the Project Manager is at its
root, so the tests skip otherwise.
"""

from __future__ import annotations

from typing import Any

import pytest

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
