"""The op framework: typed, verified hot ops that run through the session broker.

An op is a function registered with ``@op(...)``. It has a dotted name (``timeline.describe``),
pydantic params and result models, and two flags: ``hot`` (needs Resolve) and ``mutates``
(changes Resolve state). Ops never run on their own: ``submit_op`` wraps one in a broker
``HotOp`` aimed at a target ``ProjectLocator``, so it queues, batches, respects leases and the
presence guard, and is context-asserted like any other hot op.

Op code runs only inside the broker. ``make_hot_op`` gives the ``HotOp`` a body that refuses
to run unless the broker is running that very op (``broker.running_op_id()``), so the lease,
project, timeline and page checks have happened; ``OpEnv`` cannot be built outside it, and a
registered op body (``OpSpec`` keeps it private) refuses to run outside it too. Inside, the
body first asserts that the open project's unique id is the target's ``project_id``; a
mismatch raises ``OpError`` and the body never acts. Op bodies raise
``OpError(code, message, fix_hint, trap_id)`` for every refusal or failed check; the broker
records its code and detail on the ``OpResult``. Results carry values measured from Resolve
after each write, never echoes of the inputs.

Timelines are leased by identity (see ``aar.node.broker``). A mutating op (or one that acts on
the current timeline) must give every ``TimelineRef`` its ``unique_id``; a name-only ref is
refused at submit with ``timeline_id_required``. A name may come with the id as a cross-check:
if it is not that timeline's name the op is refused (``timeline_ref_mismatch``). Such an op
leases each timeline id it names, so ops on different timelines of one project run side by
side; an op that names no timeline (it acts on whichever is current) leases the whole project.
Read-only ops may still name a timeline by name.

Ops are kept short (well under the 60 s cap of a live MCP call) and idempotent: re-running
one after a timeout finds its own earlier work and reports it instead of doing it twice.
Batch ops take a resume ``cursor``.

Not here yet (follow-up F15): the D3 sandbox / human-approval gate for mutating ops on a
production project. It belongs where S7 wires ``submit_op`` to the shared instance, using
S12's approval tokens; until then mutating ops are exercised against the test fake only.
"""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aar.node.branches import is_agent_branch
from aar.node.broker import Broker, HotOp, HotOpFailure, running_op_id
from aar.node.gateway import GatewayError, ResolveGateway
from aar.node.leases import Holder
from aar.node.locator import ProjectLocator
from aar.node.traps import TRAPS

# Pages ``bounce_page`` may pass through. Media is cheap to open and holds no editing state.
_BOUNCE_PAGES = ("media", "cut")


class OpError(HotOpFailure):
    """A structured op failure: what failed, how to fix it, and the known trap behind it."""

    def __init__(
        self,
        code: str,
        message: str,
        fix_hint: str,
        trap_id: str | None = None,
        **detail: object,
    ) -> None:
        if trap_id is not None and trap_id not in TRAPS:
            raise ValueError(f"unknown trap id {trap_id!r}")
        super().__init__(code, message, {"fix_hint": fix_hint, "trap_id": trap_id, **detail})
        self.fix_hint = fix_hint
        self.trap_id = trap_id

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, **self.detail}


class OpModel(BaseModel):
    """Base of op params and results: unknown keys are errors, values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Note(OpModel):
    """Something an agent should know about a result, often a trap that was handled."""

    code: str
    message: str
    trap_id: str | None = None


TrackType = Literal["video", "audio", "subtitle"]


class TimelineRef(OpModel):
    """A timeline of the target project. Ops that change Resolve, and ops that act on the
    current timeline, need its unique_id: a name-only ref is refused with
    timeline_id_required. A name given with the unique_id is a cross-check; if it is not that
    timeline's name the op is refused with timeline_ref_mismatch. Only read-only ops accept a
    name alone."""

    name: str | None = Field(default=None, min_length=1)
    unique_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _needs_one(self) -> TimelineRef:
        if self.name is None and self.unique_id is None:
            raise ValueError("a timeline ref needs a name or a unique_id")
        return self


class ItemRef(OpModel):
    """A timeline item: its track and its ``TimelineItem.GetUniqueId()``."""

    track_type: TrackType
    track: int = Field(strict=True, ge=1)
    item_id: str = Field(min_length=1)


class MediaRef(OpModel):
    """A media-pool item by ``MediaPoolItem.GetUniqueId()``, with an optional bin hint."""

    unique_id: str = Field(min_length=1)
    bin_path: tuple[str, ...] | None = Field(
        default=None, description="Bin path below the media-pool root, to look there first."
    )


class TimelineIdentity(OpModel):
    name: str
    unique_id: str


OpFn = Callable[["OpEnv", Any], BaseModel]


def _outside_broker(what: str) -> OpError:
    return OpError(
        "outside_broker",
        f"{what} runs only inside the session broker",
        "submit the op with submit_op(broker, ...) and let the broker run it",
    )


@dataclass(frozen=True)
class OpSpec:
    name: str
    params: type[OpModel]
    result: type[OpModel]
    hot: bool
    mutates: bool
    summary: str
    traps: tuple[str, ...] = ()
    needs_current_timeline: bool = False
    """The op acts on Resolve's current timeline (e.g. ``AppendToTimeline``)."""
    _body: OpFn = field(default=lambda env, params: None, repr=False, compare=False)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hot": self.hot,
            "mutates": self.mutates,
            "summary": self.summary,
            "traps": list(self.traps),
            "params_schema": self.params.model_json_schema(),
        }


_REGISTRY: dict[str, OpSpec] = {}


def op(
    name: str,
    *,
    params: type[OpModel],
    result: type[OpModel],
    mutates: bool,
    hot: bool = True,
    traps: tuple[str, ...] = (),
    needs_current_timeline: bool = False,
) -> Callable[[OpFn], OpFn]:
    """Register an op body under ``name``."""
    for trap_id in traps:
        if trap_id not in TRAPS:
            raise ValueError(f"op {name}: unknown trap id {trap_id!r}")

    def register(fn: OpFn) -> OpFn:
        if name in _REGISTRY:
            raise ValueError(f"op {name} is already registered")
        summary = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""

        @functools.wraps(fn)
        def guarded(env: OpEnv, op_params: Any) -> BaseModel:
            if running_op_id() is None or env.op_id != running_op_id():
                raise _outside_broker(f"op {name}")
            return fn(env, op_params)

        _REGISTRY[name] = OpSpec(
            name,
            params,
            result,
            hot,
            mutates,
            summary,
            traps,
            needs_current_timeline,
            _body=guarded,
        )
        return guarded

    return register


def get_op(name: str) -> OpSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no op {name!r}; known: {sorted(_REGISTRY)}") from None


def list_ops() -> list[OpSpec]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


# -- running ops ------------------------------------------------------------------------------


class OpEnv:
    """What an op body gets: the gateway, the raw ``resolve`` object and lookup helpers.

    Only the broker's run of an op can build one.
    """

    def __init__(self, gateway: ResolveGateway, target: ProjectLocator) -> None:
        op_id = running_op_id()
        if op_id is None:
            raise _outside_broker("OpEnv")
        self.op_id = op_id
        resolve = getattr(gateway, "resolve", None)
        if resolve is None:
            raise OpError(
                "no_raw_api",
                f"{type(gateway).__name__} exposes no raw 'resolve' object",
                "run ops on a ScriptApiGateway",
            )
        self.gateway = gateway
        self.resolve = resolve
        self.target = target

    # -- project -------------------------------------------------------------------------

    def project_manager(self) -> Any:
        pm = self.resolve.GetProjectManager()
        if pm is None:
            raise OpError("no_project_manager", "GetProjectManager() returned None", "restart?")
        return pm

    def project(self) -> Any:
        project = self.project_manager().GetCurrentProject()
        if project is None:
            raise OpError(
                "no_project", "no project is open", "resubmit; the broker opens the target first"
            )
        return project

    def assert_project(self) -> Any:
        """The open project, after checking its unique id is the target's ``project_id``."""
        target = self.target
        if target.project_id is None:
            raise OpError(
                "project_id_required",
                f"target {target.key!r} has no project_id, so the open project cannot be verified",
                "take the id from project.find (it is filled for the open project) or from "
                "the hub's project registry",
                "pm-folder-scoped-load",
            )
        project = self.project()
        actual_id = str(project.GetUniqueId() or "")
        if actual_id != target.project_id:
            raise OpError(
                "wrong_project",
                f"expected project {target}, but {project.GetName()!r} [{actual_id}] is open",
                "do not act: resubmit so the broker re-selects the project by locator",
                "pm-folder-scoped-load",
                expected_id=target.project_id,
                actual_id=actual_id,
                actual_name=str(project.GetName()),
            )
        return project

    # -- timelines and items -------------------------------------------------------------

    def timelines(self) -> Iterator[Any]:
        project = self.project()
        for idx in range(1, int(project.GetTimelineCount() or 0) + 1):
            timeline = project.GetTimelineByIndex(idx)
            if timeline is not None:
                yield timeline

    def timeline(self, ref: TimelineRef | None) -> Any:
        """The timeline ``ref`` names, or the current timeline when ``ref`` is ``None``."""
        if ref is None:
            current = self.project().GetCurrentTimeline()
            if current is None:
                raise OpError("no_timeline", "the project has no current timeline", "name one")
            return current
        timelines = list(self.timelines())
        if ref.unique_id is not None:
            # The id is the identity (and what the op was leased on): never fall back to the
            # name, and refuse a name that belongs to another timeline.
            matches = [t for t in timelines if t.GetUniqueId() == ref.unique_id]
            if matches and ref.name is not None and matches[0].GetName() != ref.name:
                raise OpError(
                    "timeline_ref_mismatch",
                    f"timeline id {ref.unique_id!r} is {matches[0].GetName()!r}, not {ref.name!r}",
                    "give the name and the unique_id of the same timeline, or only the unique_id",
                )
        else:
            matches = [t for t in timelines if t.GetName() == ref.name]
        if not matches:
            if ref.unique_id is not None:
                unused = f"; the name {ref.name!r} was not used to look it up" if ref.name else ""
                message = f"no timeline with id {ref.unique_id!r}{unused}"
            else:
                message = f"no timeline named {ref.name!r}"
            raise OpError(
                "timeline_not_found",
                message,
                "list timelines with timeline.list; a timeline's unique_id changes only if it "
                "is deleted and made again",
            )
        if len(matches) > 1:
            raise OpError(
                "timeline_ambiguous",
                f"{len(matches)} timelines are named {ref.name!r}",
                "refer to the timeline by unique_id",
            )
        return matches[0]

    def current_timeline(self, ref: TimelineRef | None) -> Any:
        """Make ``ref`` the current timeline (if it is not already) and return it, verified."""
        project = self.project()
        wanted = self.timeline(ref)
        current = project.GetCurrentTimeline()
        if current is None or current.GetUniqueId() != wanted.GetUniqueId():
            if not project.SetCurrentTimeline(wanted):
                raise OpError(
                    "timeline_set_failed",
                    f"SetCurrentTimeline({wanted.GetName()!r}) returned False",
                    "check the timeline is not locked by another user",
                )
            current = project.GetCurrentTimeline()
            if current is None or current.GetUniqueId() != wanted.GetUniqueId():
                raise OpError(
                    "timeline_set_failed",
                    f"timeline {wanted.GetName()!r} did not become current",
                    "retry once Resolve is idle",
                )
        return current

    @staticmethod
    def identity(timeline: Any) -> TimelineIdentity:
        return TimelineIdentity(name=str(timeline.GetName()), unique_id=str(timeline.GetUniqueId()))

    @staticmethod
    def item(timeline: Any, ref: ItemRef) -> Any:
        for item in timeline.GetItemListInTrack(ref.track_type, ref.track) or []:
            if item.GetUniqueId() == ref.item_id:
                return item
        raise OpError(
            "item_not_found",
            f"no item {ref.item_id!r} on {ref.track_type} track {ref.track} of "
            f"{timeline.GetName()!r}",
            "item ids change when items are re-created; read them again with timeline.describe",
            ref=ref.model_dump(),
        )

    # -- media pool ------------------------------------------------------------------------

    def media_pool(self) -> Any:
        media_pool = self.project().GetMediaPool()
        if media_pool is None:
            raise OpError("no_media_pool", "Project.GetMediaPool() returned None", "reopen it")
        return media_pool

    def bin(self, path: tuple[str, ...]) -> Any:
        folder = self.media_pool().GetRootFolder()
        for name in path:
            matches = [f for f in folder.GetSubFolderList() or [] if f.GetName() == name]
            if len(matches) != 1:
                raise OpError(
                    "bin_not_found" if not matches else "bin_ambiguous",
                    f"bin {'/'.join(path)!r}: {len(matches)} folders named {name!r}",
                    "list bins with media.find",
                )
            folder = matches[0]
        return folder

    def walk_bins(
        self, path: tuple[str, ...] = (), recursive: bool = True
    ) -> Iterator[tuple[tuple[str, ...], Any]]:
        stack = [(path, self.bin(path))]
        while stack:
            where, folder = stack.pop()
            yield where, folder
            if recursive:
                subs = list(folder.GetSubFolderList() or [])
                stack.extend(((*where, str(f.GetName())), f) for f in reversed(subs))

    def media(self, refs: list[MediaRef]) -> dict[str, tuple[Any, tuple[str, ...]]]:
        """Find media-pool items by unique id: ``{id: (item, bin_path)}``. All must exist."""
        wanted = {ref.unique_id for ref in refs}
        found: dict[str, tuple[Any, tuple[str, ...]]] = {}
        for hint in {ref.bin_path for ref in refs if ref.bin_path is not None}:
            self._collect(wanted, found, hint, recursive=False)
        if wanted - set(found):
            self._collect(wanted, found, (), recursive=True)
        missing = sorted(wanted - set(found))
        if missing:
            raise OpError(
                "media_not_found",
                f"no media-pool item with id(s) {missing}",
                "find media ids with media.find",
                missing=missing,
            )
        return found

    def _collect(
        self,
        wanted: set[str],
        found: dict[str, tuple[Any, tuple[str, ...]]],
        path: tuple[str, ...],
        recursive: bool,
    ) -> None:
        for where, folder in self.walk_bins(path, recursive):
            for clip in folder.GetClipList() or []:
                uid = clip.GetUniqueId()
                if uid in wanted and uid not in found:
                    found[uid] = (clip, where)
            if wanted <= set(found):
                return

    # -- pages -----------------------------------------------------------------------------

    def bounce_page(self) -> str:
        """Open another page and come back (trap ``setproperty-after-color``)."""
        page = str(self.resolve.GetCurrentPage() or "")
        other = next(p for p in _BOUNCE_PAGES if p != page)
        for step in (other, page):
            if not step or not self.resolve.OpenPage(step):
                raise OpError(
                    "page_bounce_failed",
                    f"OpenPage({step!r}) failed while bouncing from {page!r}",
                    "open another page and back by hand, then retry",
                    "setproperty-after-color",
                )
        return other


def _assert_branch(env: OpEnv, spec: OpSpec, params: OpModel) -> None:
    """Refuse an editing op aimed at a timeline that is not an agent branch.

    Agents edit a branch (``timeline.branch``), never a human's timeline (DESIGN R3). The
    guard is here, in the one choke point every op body passes, because this is where the
    open project is already asserted and the op's real, current timeline can be read by
    identity. It fires only for a mutating op that names a ``timeline`` it edits; ops that
    build a copy from a ``source`` (``timeline.duplicate``, ``timeline.branch``) carry no
    ``timeline`` field, so branching off a human timeline is never blocked, and
    ``grades.copy`` may still read a grade from a human ``reference_timeline``.
    """
    if not (spec.mutates and "timeline" in type(params).model_fields):
        return
    timeline = env.timeline(params.timeline)
    name = str(timeline.GetName())
    if not is_agent_branch(name):
        raise OpError(
            "not_an_agent_branch",
            f"op {spec.name} edits timeline {name!r}, which is not an agent branch; agents "
            f"must not edit a human's timeline",
            "branch it first with timeline.branch and edit the copy, or pass "
            "require_branch=False to edit a production timeline deliberately",
            timeline=name,
            timeline_id=str(timeline.GetUniqueId()),
        )


def _execute(
    spec: OpSpec,
    gateway: ResolveGateway,
    target: ProjectLocator,
    params: OpModel,
    *,
    require_branch: bool = False,
) -> Any:
    """Run one op body after asserting the open project. Only the broker gets here."""
    env = OpEnv(gateway, target)
    env.assert_project()
    if require_branch:
        _assert_branch(env, spec, params)
    try:
        result = spec._body(env, params)
    except GatewayError as exc:
        raise OpError(exc.code, exc.message, "see the gateway message") from exc
    if not isinstance(result, spec.result):
        raise TypeError(f"op {spec.name} returned {type(result).__name__}, not {spec.result}")
    return result


def make_hot_op(
    name: str,
    params: OpModel | dict[str, Any],
    *,
    target: ProjectLocator,
    holder: Holder,
    priority: int = 0,
    abort_batch_on_failure: bool = False,
    require_branch: bool = False,
) -> HotOp:
    """Validate ``params`` and wrap op ``name`` in a broker ``HotOp`` aimed at ``target``.

    ``require_branch`` refuses an editing op whose timeline is not an agent branch (DESIGN
    R3). It is off by default so the verified op library keeps its own behaviour; the
    agent-facing path (the S7 runner) sets it, and the override is ``require_branch=False``.
    """
    spec = get_op(name)
    if not spec.hot:
        raise ValueError(f"op {name} is not a hot op")
    if target.project_id is None:
        raise OpError(
            "project_id_required",
            f"target {target.key!r} has no project_id",
            "every op asserts the open project by id; take it from project.find or the hub",
            "pm-folder-scoped-load",
        )
    checked = params if isinstance(params, spec.params) else spec.params.model_validate(params)
    timeline, timeline_id, also_leased = _lease_timeline(spec, checked)
    op_id = uuid.uuid4().hex

    def run(gateway: ResolveGateway) -> Any:
        if running_op_id() != op_id:
            raise _outside_broker(f"op {name}")
        return _execute(spec, gateway, target, checked, require_branch=require_branch)

    return HotOp(
        project=target,
        fn=run,
        holder=holder,
        timeline=timeline,
        timeline_id=timeline_id,
        extra_timeline_ids=also_leased,
        priority=priority,
        abort_batch_on_failure=abort_batch_on_failure,
        label=name,
        id=op_id,
    )


def _timeline_refs(value: Any) -> list[TimelineRef]:
    """Every ``TimelineRef`` inside an op's params."""
    if isinstance(value, TimelineRef):
        return [value]
    if isinstance(value, BaseModel):
        return [
            ref for name in type(value).model_fields for ref in _timeline_refs(getattr(value, name))
        ]
    if isinstance(value, list | tuple):
        return [ref for item in value for ref in _timeline_refs(item)]
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _timeline_refs(item)]
    return []


def _is_timeline_ref_field(annotation: Any) -> bool:
    return annotation is TimelineRef or TimelineRef in get_args(annotation)


def _lease_timeline(
    spec: OpSpec, params: OpModel
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """What the broker selects and leases for this op: ``(name, unique id, other ids)``.

    The top-level ``timeline`` ref is made current and asserted; every timeline id the op
    names is leased. An op that leaves a top-level timeline ref empty acts on whichever
    timeline is current, so it leases the whole project (``(None, None, ())``), as does an op
    that names no timeline at all. Read-only ops lease the whole project and select nothing.
    """
    if not (spec.mutates or spec.needs_current_timeline):
        return None, None, ()
    refs = _timeline_refs(params)
    name_only = [ref.name for ref in refs if ref.unique_id is None]
    if name_only:
        raise OpError(
            "timeline_id_required",
            f"op {spec.name} changes Resolve, so every timeline it names needs its unique_id "
            f"(name-only: {name_only})",
            "take the ids from timeline.list or timeline.describe; a name may stay as a "
            "cross-check",
        )
    top_level = [
        n for n, f in type(params).model_fields.items() if _is_timeline_ref_field(f.annotation)
    ]
    if not refs or any(getattr(params, n) is None for n in top_level):
        return None, None, ()
    ids = sorted({ref.unique_id for ref in refs if ref.unique_id is not None})
    top = getattr(params, "timeline", None)
    if isinstance(top, TimelineRef):
        return top.name, top.unique_id, tuple(i for i in ids if i != top.unique_id)
    return None, None, tuple(ids)


def submit_op(
    broker: Broker,
    name: str,
    params: OpModel | dict[str, Any],
    *,
    target: ProjectLocator,
    holder: Holder,
    priority: int = 0,
    abort_batch_on_failure: bool = False,
    require_branch: bool = False,
) -> str:
    """Queue op ``name`` on ``broker``; returns the broker op id (see ``Broker.result``).

    ``require_branch`` refuses an editing op on a non-branch timeline (see
    :func:`make_hot_op`).
    """
    return broker.submit(
        make_hot_op(
            name,
            params,
            target=target,
            holder=holder,
            priority=priority,
            abort_batch_on_failure=abort_batch_on_failure,
            require_branch=require_branch,
        )
    )
