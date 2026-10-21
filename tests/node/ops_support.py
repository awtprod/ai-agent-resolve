"""A fake Resolve with a show project, a broker in front of it, and helpers to run ops."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from aar.node.branch_registry import BranchRegistry
from aar.node.broker import Broker, BrokerConfig, BrokerEvent, OpError, OpResult, OpStatus
from aar.node.gateway import ScriptApiGateway
from aar.node.leases import Holder, LeaseManager
from aar.node.locator import ProjectLocator
from aar.node.ops import submit_op
from aar.node.presence import PresenceGuard

from .fake_resolve import FakeClock, FakeMediaPoolItem, FakeProject, FakeResolve, FakeTimeline

IDLE = 60.0
AGENT = Holder.agent("thread-ops")

CONFIG = BrokerConfig(
    batch_max_ops=50,
    batch_max_seconds=1e6,
    max_wait_seconds=1e6,
    restore_delay_seconds=0.0,
)


class OpsWorld:
    """A fake Resolve with a broker in front of it, already past the presence warm-up."""

    def __init__(
        self,
        fake: FakeResolve,
        config: BrokerConfig = CONFIG,
        on_event: Callable[[BrokerEvent], None] | None = None,
    ) -> None:
        self.fake = fake
        self.gw = ScriptApiGateway(self.fake)
        self.clock = FakeClock()
        self.leases = LeaseManager(self.clock)
        self.presence = PresenceGuard(idle_window=IDLE)
        self.branch_registry = BranchRegistry()
        self.events: list[BrokerEvent] = []

        def sink(event: BrokerEvent) -> None:
            self.events.append(event)
            if on_event is not None:
                on_event(event)

        self.broker = Broker(
            self.gw,
            leases=self.leases,
            presence=self.presence,
            clock=self.clock,
            sink=sink,
            config=config,
        )
        self.presence.observe(self.gw.get_context(), self.clock())
        self.clock.advance(IDLE)
        self.fake.calls.clear()

    def locator(self, name: str = "Show") -> ProjectLocator:
        project = self.fake.project(name)
        return ProjectLocator(project.folder, name, project.GetUniqueId())

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        target: ProjectLocator | None = None,
        holder: Holder = AGENT,
        require_branch: bool = False,
    ) -> OpResult:
        # The op-mechanics tests exercise ops on any timeline, so the harness opts out of the
        # branch guard by default; the collaboration tests pass require_branch=True.
        op_id = submit_op(
            self.broker,
            name,
            params,
            target=target or self.locator(),
            holder=holder,
            require_branch=require_branch,
            branch_registry=self.branch_registry,
        )
        self.broker.run_until_idle()
        result = self.broker.result(op_id)
        assert result is not None, f"op {name} never finished"
        return result

    def ok(self, name: str, params: dict[str, Any], **kwargs: Any) -> Any:
        result = self.run(name, params, **kwargs)
        assert result.status is OpStatus.OK, result.error
        assert isinstance(result.value, BaseModel)
        return result.value

    def fails(self, name: str, params: dict[str, Any], **kwargs: Any) -> OpError:
        result = self.run(name, params, **kwargs)
        assert result.status is OpStatus.FAILED, result.value
        assert result.error is not None
        return result.error


class Show(OpsWorld):
    """Project "Show" in Project Manager folder ``Shows``: a 29.97 timeline "Main", an empty
    agent branch "Main [agent]" and a media pool with one clip of each kind."""

    def __init__(
        self,
        fake: FakeResolve | None = None,
        config: BrokerConfig = CONFIG,
        on_event: Callable[[BrokerEvent], None] | None = None,
    ) -> None:
        fake = fake or FakeResolve(
            {"Show": ["Main", "Main [agent]"], "Other": ["Other main"]},
            current="Show",
            folder=("Shows",),
        )
        self.project: FakeProject = fake.project("Show")
        pool = self.project.media_pool
        self.cam_a = pool.add_clip(("Footage", "Day 1"), "cam A", fps=29.97, frames=20000)
        self.cam_b = pool.add_clip(
            ("Footage", "Day 1"), "cam B", fps=29.97, frames=20000, braw=True
        )
        self.cam_24 = pool.add_clip(("Footage", "Day 1"), "cam 24p", fps=24.0, frames=20000)
        self.still = pool.add_clip(("Footage",), "title card", kind="still", frames=1)
        self.multicam = pool.add_clip(("Footage",), "multicam", kind="multicam", frames=20000)
        self.music = pool.add_clip(("Music",), "theme", kind="audio", frames=20000)
        self.main: FakeTimeline = self.project.timeline("Main")
        self.branch: FakeTimeline = self.project.timeline("Main [agent]")
        super().__init__(fake, config, on_event)

    @property
    def pid(self) -> str:
        return self.project.GetUniqueId()

    def ref(self, timeline: str) -> dict[str, str]:
        """A timeline ref by unique id, with its name as a cross-check (what writes need)."""
        return {"name": timeline, "unique_id": self.project.timeline(timeline).GetUniqueId()}

    def make_branch(
        self, *, source: str = "Main", holder: Holder = AGENT, if_exists: str = "reuse"
    ) -> Any:
        """Create (and register) a genuine agent branch via timeline.branch, owned by ``holder``.

        Returns the BranchResult; the branch timeline is then owned by ``holder`` in this
        world's registry, so an editing op with require_branch=True may touch it.
        """
        return self.ok(
            "timeline.branch",
            {"source": self.ref(source), "if_exists": if_exists},
            holder=holder,
        )


def media(clip: FakeMediaPoolItem) -> dict[str, Any]:
    return {"unique_id": clip.GetUniqueId()}


# Fake calls that change timeline, media or project content (not the broker's context work).
CONTENT_WRITES = frozenset(
    {
        "AppendToTimeline",
        "DeleteClips",
        "DuplicateTimeline",
        "DeleteTimelines",
        "AddMarker",
        "DeleteMarkerAtFrame",
        "DeleteMarkerByCustomData",
        "DeleteMarkersByColor",
        "UpdateMarkerCustomData",
        "AddTrack",
        "SetProperties",
        "SetProperty",
        "SetFades",
        "SetLUT",
        "CopyGrades",
        "SetClipProperty",
        "Timeline.SetSettings",
        "Project.SetSettings",
    }
)


def content_writes(fake: FakeResolve) -> list[tuple[Any, ...]]:
    return [call for call in fake.calls if call[0] in CONTENT_WRITES]


def human_settles(world: OpsWorld) -> None:
    """After a human change, let the presence guard see a quiet idle window again."""
    world.presence.observe(world.gw.get_context(), world.clock())
    world.clock.advance(IDLE)
