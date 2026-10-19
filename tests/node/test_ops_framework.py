"""The op framework: ops go through the broker, assert the project by id, fail structurally."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from aar.node.broker import BrokerEvent, HotOp, OpStatus
from aar.node.leases import Holder, Scope, project_key, timeline_key
from aar.node.locator import ProjectLocator
from aar.node.ops import OpEnv, OpError, get_op, list_ops, make_hot_op, submit_op
from aar.node.ops import write as write_ops

from .fake_resolve import FakeResolve
from .ops_support import AGENT, Show, content_writes, human_settles, media

OTHER = Holder.agent("thread-other")

EXPECTED = {
    "project.find": (True, False),
    "timeline.list": (True, False),
    "timeline.describe": (True, False),
    "media.find": (True, False),
    "cloud.projects": (True, False),
    "cloud.sync_status": (True, False),
    "cloud.freshness": (True, False),
    "timeline.duplicate": (True, True),
    "timeline.append_ranges": (True, True),
    "item.set_audio_gain": (True, True),
    "item.set_fades": (True, True),
    "grades.copy": (True, True),
    "clip.set_input_color_space": (True, True),
    "markers.add": (True, True),
    "timeline.branch": (True, True),
    "markers.mirror_review": (True, True),
    "markers.read_review": (True, False),
}


def test_registry_lists_the_v1_ops_with_their_flags() -> None:
    assert {s.name: (s.hot, s.mutates) for s in list_ops()} == EXPECTED
    for spec in list_ops():
        described = spec.describe()
        assert described["params_schema"]["type"] == "object"
        assert described["summary"]


def test_unknown_op_and_bad_params_are_rejected_at_submit() -> None:
    show = Show()
    with pytest.raises(KeyError):
        submit_op(show.broker, "timeline.nuke", {}, target=show.locator(), holder=AGENT)
    with pytest.raises(ValidationError):
        submit_op(
            show.broker, "timeline.describe", {"bogus": 1}, target=show.locator(), holder=AGENT
        )
    assert show.broker.pending() == []


def test_a_target_without_project_id_is_refused() -> None:
    show = Show()
    with pytest.raises(OpError) as err:
        submit_op(
            show.broker,
            "timeline.list",
            {},
            target=ProjectLocator(("Shows",), "Show"),
            holder=AGENT,
        )
    assert err.value.code == "project_id_required"
    assert err.value.trap_id == "pm-folder-scoped-load"


def test_ops_wait_in_the_broker_queue_behind_leases() -> None:
    show = Show()
    other = Holder.agent("thread-other")
    lease = show.leases.acquire(other, Scope.PROJECT, project_key(show.pid), ttl=600)
    op_id = submit_op(show.broker, "timeline.list", {}, target=show.locator(), holder=AGENT)
    show.broker.run_until_idle()
    assert show.broker.result(op_id) is None
    assert [op.id for op in show.broker.pending()] == [op_id]
    show.leases.release(lease.id, other)
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None and result.status is OpStatus.OK


def test_op_body_never_acts_in_a_project_with_another_id() -> None:
    # A human opens a same-named project between the broker's checks and the op body: the
    # body's own id assertion is the last line of defence, and a name check would pass.
    holder: dict[str, Any] = {}

    def human_switches(event: BrokerEvent) -> None:
        if event.kind == "op_started":
            fake = holder["show"].fake
            fake.current = holder["twin"]

    show = Show(on_event=human_switches)
    holder["show"] = show
    holder["twin"] = show.fake.add_project("Show", ["Main"], ("Elsewhere",))
    error = show.fails("markers.add", {"timeline": show.ref("Main"), "markers": [{"frame": 5}]})
    assert error.code == "wrong_project"
    assert error.detail["actual_id"] == holder["twin"].GetUniqueId()
    assert error.detail["actual_name"] == "Show"
    assert error.detail["trap_id"] == "pm-folder-scoped-load"
    assert content_writes(show.fake) == []


def test_op_code_cannot_run_outside_the_broker() -> None:
    show = Show()
    show.leases.acquire(OTHER, Scope.PROJECT, project_key(show.pid), ttl=600)
    hot = make_hot_op(
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 5, "name": "direct"}]},
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )
    with pytest.raises(OpError) as err:
        hot.fn(show.gw)  # the HotOp body, called by hand
    assert err.value.code == "outside_broker"
    with pytest.raises(OpError) as err:
        OpEnv(show.gw, show.locator())
    assert err.value.code == "outside_broker"
    spec = get_op("markers.add")
    assert not hasattr(spec, "fn")
    params = spec.params.model_validate({"timeline": {"name": "Main"}, "markers": [{"frame": 5}]})
    with pytest.raises(OpError) as err:
        spec._body(object(), params)  # type: ignore[arg-type]
    assert err.value.code == "outside_broker"
    with pytest.raises(OpError) as err:
        write_ops.markers_add(object(), params)  # type: ignore[arg-type]
    assert err.value.code == "outside_broker"
    assert show.main.GetMarkers() == {}
    assert content_writes(show.fake) == []


# -- timeline leases by identity (F1) -----------------------------------------------------------


def _marker(show: Show, ref: dict[str, Any], frame: int = 30) -> str:
    # These tests exercise broker leasing, not the branch guard, so they opt out of it.
    return submit_op(
        show.broker,
        "markers.add",
        {"timeline": ref, "markers": [{"frame": frame, "name": "m"}]},
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )


def _lease_main(show: Show) -> None:
    key = timeline_key(show.pid, show.main.GetUniqueId())
    show.leases.acquire(OTHER, Scope.TIMELINE, key, ttl=600)


@pytest.mark.parametrize(
    "ref",
    [
        pytest.param("mixed", id="name-of-branch-with-id-of-main"),
        pytest.param("id", id="id-only"),
        pytest.param("both", id="id-with-its-name"),
    ],
)
def test_a_timeline_lease_holds_whatever_form_the_ref_takes(ref: str) -> None:
    show = Show()
    _lease_main(show)
    refs = {
        "mixed": {"name": "Main [agent]", "unique_id": show.main.GetUniqueId()},
        "id": {"unique_id": show.main.GetUniqueId()},
        "both": show.ref("Main"),
    }
    op_id = _marker(show, refs[ref])
    show.broker.run_until_idle()
    assert show.broker.result(op_id) is None  # still queued behind OTHER's lease
    assert show.main.GetMarkers() == {} and show.branch.GetMarkers() == {}


@pytest.mark.parametrize(
    ("op_name", "params"),
    [
        ("markers.add", lambda show: {"timeline": {"name": "Main"}, "markers": [{"frame": 1}]}),
        (
            "item.set_fades",
            lambda show: {
                "timeline": {"name": "Main"},
                "targets": [
                    {"item": {"track_type": "video", "track": 1, "item_id": "x"}, "fade_in": 1}
                ],
            },
        ),
        (
            "timeline.append_ranges",
            lambda show: {
                "timeline": {"name": "Main [agent]"},
                "ranges": [{"media": media(show.cam_a), "src_in": 0, "src_out": 10}],
            },
        ),
        ("timeline.duplicate", lambda show: {"source": {"name": "Main"}, "name": "Main v2"}),
        (
            "grades.copy",
            lambda show: {
                "timeline": show.ref("Main [agent]"),
                "groups": [
                    {
                        "reference": {"track_type": "video", "track": 1, "item_id": "r"},
                        "reference_timeline": {"name": "Main"},  # nested, still name-only
                        "targets": [{"track_type": "video", "track": 1, "item_id": "t"}],
                    }
                ],
            },
        ),
    ],
)
def test_a_name_only_timeline_ref_on_a_write_is_refused_at_submit(
    op_name: str, params: Callable[[Show], dict[str, Any]]
) -> None:
    show = Show()
    with pytest.raises(OpError) as err:
        submit_op(show.broker, op_name, params(show), target=show.locator(), holder=AGENT)
    assert err.value.code == "timeline_id_required"
    assert show.broker.pending() == []
    assert show.fake.calls == []  # not a single gateway call


def test_read_ops_may_still_name_a_timeline_by_name() -> None:
    show = Show()
    result = show.ok("timeline.describe", {"timeline": {"name": "Main"}})
    assert result.timeline.unique_id == show.main.GetUniqueId()


def test_holders_of_two_timelines_of_one_project_both_get_to_work() -> None:
    """A holds Main and B holds the branch; each writes to its own timeline by id."""
    show = Show()
    a, b = Holder.agent("A"), Holder.agent("B")
    show.leases.acquire(a, Scope.TIMELINE, timeline_key(show.pid, show.main.GetUniqueId()), ttl=600)
    branch_key = timeline_key(show.pid, show.branch.GetUniqueId())
    show.leases.acquire(b, Scope.TIMELINE, branch_key, ttl=600)
    ops = [
        submit_op(
            show.broker,
            "markers.add",
            {"timeline": show.ref(name), "markers": [{"frame": 50, "name": who}]},
            target=show.locator(),
            holder=holder,
            require_branch=False,
        )
        for name, who, holder in (("Main", "a", a), ("Main [agent]", "b", b))
    ]
    show.broker.run_until_idle()
    for op_id in ops:
        result = show.broker.result(op_id)
        assert result is not None and result.status is OpStatus.OK, result
    assert show.main.GetMarkers()[50]["name"] == "a"
    assert show.branch.GetMarkers()[50]["name"] == "b"
    assert show.broker.pending() == []


def test_a_ref_whose_name_and_id_disagree_is_refused_and_writes_nothing() -> None:
    show = Show()
    op_id = _marker(show, {"name": "Main [agent]", "unique_id": show.main.GetUniqueId()})
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None and result.status is OpStatus.FAILED and result.error is not None
    assert result.error.code == "timeline_unavailable"
    assert result.error.detail["gateway_code"] == "timeline_ref_mismatch"
    assert show.main.GetMarkers() == {} and show.branch.GetMarkers() == {}


def test_the_op_itself_refuses_a_mismatched_ref() -> None:
    # A read op is not timeline-selected by the broker, so the check in OpEnv.timeline is
    # what stops it.
    show = Show()
    error = show.fails(
        "timeline.describe",
        {"timeline": {"name": "Main [agent]", "unique_id": show.main.GetUniqueId()}},
    )
    assert error.code == "timeline_ref_mismatch"


def test_an_id_ref_is_not_blocked_by_a_lease_on_another_timeline() -> None:
    show = Show()
    _lease_main(show)
    op_id = _marker(show, {"unique_id": show.branch.GetUniqueId()})
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None and result.status is OpStatus.OK
    assert list(show.branch.GetMarkers()) == [30] and show.main.GetMarkers() == {}


def _two_timeline_grades(show: Show) -> dict[str, Any]:
    reference = show.main.place("video", 1, show.cam_a, 108000, 100)
    reference.graph.nodes[0].tools = ("Primaries",)
    target = show.branch.place("video", 1, show.cam_a, 108000, 50)
    return {
        "timeline": show.ref("Main [agent]"),
        "groups": [
            {
                "reference": {
                    "track_type": "video",
                    "track": 1,
                    "item_id": reference.GetUniqueId(),
                },
                "reference_timeline": show.ref("Main"),
                "targets": [{"track_type": "video", "track": 1, "item_id": target.GetUniqueId()}],
            }
        ],
    }


def test_an_op_naming_two_timelines_leases_each_of_them() -> None:
    show = Show()
    hot = make_hot_op(
        "grades.copy",
        _two_timeline_grades(show),
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )
    assert (hot.timeline_id, hot.extra_timeline_ids) == (
        show.branch.GetUniqueId(),
        (show.main.GetUniqueId(),),
    )
    # A lease on either named timeline holds it back ...
    for leased in (show.main, show.branch):
        world = Show()
        world.leases.acquire(
            OTHER, Scope.TIMELINE, timeline_key(world.pid, leased.GetUniqueId()), ttl=600
        )
        op_id = submit_op(
            world.broker,
            "grades.copy",
            _two_timeline_grades(world),
            target=world.locator(),
            holder=AGENT,
            require_branch=False,
        )
        world.broker.run_until_idle()
        assert world.broker.result(op_id) is None, leased.GetName()
    # ... and a lease on a third timeline does not.
    world = Show()
    world.project.timelines.append(type(world.main)(world.fake, "Third", "tl-third", world.project))
    world.leases.acquire(OTHER, Scope.TIMELINE, timeline_key(world.pid, "tl-third"), ttl=600)
    op_id = submit_op(
        world.broker,
        "grades.copy",
        _two_timeline_grades(world),
        target=world.locator(),
        holder=AGENT,
        require_branch=False,
    )
    world.broker.run_until_idle()
    result = world.broker.result(op_id)
    assert result is not None and result.status is OpStatus.OK, result


def test_an_op_on_the_current_timeline_leases_the_whole_project() -> None:
    show = Show()
    hot = make_hot_op(
        "markers.add",
        {"markers": [{"frame": 1}]},
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )
    assert (hot.timeline, hot.timeline_id, hot.extra_timeline_ids) == (None, None, ())
    single = make_hot_op(
        "markers.add",
        {"timeline": {"unique_id": show.branch.GetUniqueId()}, "markers": [{"frame": 1}]},
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )
    assert single.timeline_id == show.branch.GetUniqueId()


# -- project and timeline leases survive renames and same names (F2) ---------------------------


def test_a_project_lease_holds_after_the_project_is_renamed() -> None:
    show = Show()
    show.leases.acquire(OTHER, Scope.PROJECT, project_key(show.pid), ttl=600)
    show.project._name = "Show Renamed"
    human_settles(show)
    renamed = ProjectLocator(("Shows",), "Show Renamed", show.pid)
    op_id = submit_op(
        show.broker,
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 5}]},
        target=renamed,
        holder=AGENT,
        require_branch=False,
    )
    show.broker.run_until_idle()
    assert show.broker.result(op_id) is None
    assert show.main.GetMarkers() == {}


def test_a_timeline_lease_holds_after_the_timeline_is_renamed() -> None:
    show = Show()
    _lease_main(show)
    show.main._name = "Main renamed"
    human_settles(show)
    main_id = show.main.GetUniqueId()
    for ref in ({"unique_id": main_id}, {"name": "Main renamed", "unique_id": main_id}):
        op_id = _marker(show, ref)
        show.broker.run_until_idle()
        assert show.broker.result(op_id) is None, ref
    assert show.main.GetMarkers() == {}


def test_a_lease_on_one_project_does_not_block_a_same_named_project() -> None:
    # The human works in "Other"; OTHER leases the "Show" in Shows; the agent targets the
    # "Show" in Elsewhere, which is a different project.
    fake = FakeResolve(
        {"Show": ["Main", "Main [agent]"], "Other": ["Other main"]},
        current="Other",
        folder=("Shows",),
    )
    twin = fake.add_project("Show", ["Main"], ("Elsewhere",))
    show = Show(fake)
    show.leases.acquire(OTHER, Scope.PROJECT, project_key(show.pid), ttl=600)
    op_id = submit_op(
        show.broker,
        "timeline.list",
        {},
        target=ProjectLocator(("Elsewhere",), "Show", twin.GetUniqueId()),
        holder=AGENT,
    )
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None and result.status is OpStatus.OK
    assert result.value.project_id == twin.GetUniqueId()


def test_no_lease_and_ref_combination_under_blocks() -> None:
    """Every lease on the timeline an op writes (or on its project) blocks the op, whatever
    names the lease and the op use; a lease on another timeline never matters to an id ref."""
    for lease_scope in ("project", "main", "branch"):
        for form in ("id", "name", "both"):
            for rename in (False, True):
                show = Show()
                main_id = show.main.GetUniqueId()
                if lease_scope == "project":
                    show.leases.acquire(OTHER, Scope.PROJECT, project_key(show.pid), ttl=600)
                else:
                    tl = show.main if lease_scope == "main" else show.branch
                    key = timeline_key(show.pid, tl.GetUniqueId())
                    show.leases.acquire(OTHER, Scope.TIMELINE, key, ttl=600)
                if rename:
                    show.main._name = "Main 2"
                    show.project._name = "Show 2"
                    human_settles(show)
                name = show.main.GetName()
                ref = {"id": {"unique_id": main_id}, "name": {"name": name}}.get(
                    form, {"name": name, "unique_id": main_id}
                )
                target = ProjectLocator(("Shows",), show.project.GetName(), show.pid)
                label = (lease_scope, form, rename)
                params = {"timeline": ref, "markers": [{"frame": 7}]}
                if form == "name":  # never reaches the queue, so it can never under-block
                    with pytest.raises(OpError, match="timeline_id_required"):
                        submit_op(show.broker, "markers.add", params, target=target, holder=AGENT)
                    continue
                op_id = submit_op(
                    show.broker,
                    "markers.add",
                    params,
                    target=target,
                    holder=AGENT,
                    require_branch=False,
                )
                show.broker.run_until_idle()
                wrote = bool(show.main.GetMarkers())
                if lease_scope in ("project", "main"):
                    assert not wrote and show.broker.result(op_id) is None, label
                else:
                    assert wrote, label


def test_op_errors_reach_the_broker_result_with_fix_hint_and_trap() -> None:
    show = Show()
    error = show.fails(
        "timeline.append_ranges",
        {
            "timeline": show.ref("Main [agent]"),
            "ranges": [{"media": media(show.still), "src_in": 0, "src_out": 48}],
        },
    )
    assert error.code == "still_refused"
    assert error.detail["trap_id"] == "append-still-ignores-range"
    assert "ProRes 4444" in str(error.detail["fix_hint"])


def test_get_op_names_the_known_ops_on_a_miss() -> None:
    with pytest.raises(KeyError) as err:
        get_op("nope")
    assert "timeline.describe" in str(err.value)


def test_an_op_body_cannot_run_another_ops_body_inside_the_broker() -> None:
    # Inside the broker some op is running, but not this one: its lease and context checks
    # never happened, so it must refuse.
    show = Show()
    show.leases.acquire(
        OTHER, Scope.PROJECT, project_key(show.fake.project("Other").GetUniqueId()), ttl=600
    )
    inner = make_hot_op(
        "markers.add",
        {"timeline": show.ref("Main"), "markers": [{"frame": 9, "name": "smuggled"}]},
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )
    errors: list[str] = []

    def outer(gw: Any) -> None:
        try:
            inner.fn(gw)
        except OpError as exc:
            errors.append(exc.code)

    show.broker.submit(HotOp(project=show.locator(), fn=outer, holder=AGENT))
    show.broker.run_until_idle()
    assert errors == ["outside_broker"]
    assert show.main.GetMarkers() == {}


def test_an_env_from_one_op_cannot_run_a_body_in_another() -> None:
    show = Show()
    stash: dict[str, Any] = {}
    errors: list[str] = []
    spec = get_op("markers.add")
    params = spec.params.model_validate({"timeline": show.ref("Main"), "markers": [{"frame": 3}]})

    def first(gw: Any) -> None:
        stash["env"] = OpEnv(gw, show.locator())  # allowed: an op is running

    def second(gw: Any) -> None:
        try:
            spec._body(stash["env"], params)  # the first op's env, in the second op
        except OpError as exc:
            errors.append(exc.code)

    for fn in (first, second):
        show.broker.submit(HotOp(project=show.locator(), fn=fn, holder=AGENT))
    show.broker.run_until_idle()
    assert errors == ["outside_broker"]
    assert show.main.GetMarkers() == {}


def test_an_unknown_timeline_id_says_the_name_was_not_used() -> None:
    show = Show()
    op_id = submit_op(
        show.broker,
        "markers.add",
        {"timeline": {"name": "Main", "unique_id": "tl-gone"}, "markers": [{"frame": 1}]},
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None and result.error is not None
    assert result.error.code == "timeline_unavailable"
    assert result.error.detail["gateway_code"] == "timeline_not_found"
    assert "no timeline with id 'tl-gone'" in result.error.message
    assert "the name 'Main' was not used" in result.error.message
    assert show.main.GetMarkers() == {}


def test_a_write_on_the_current_timeline_waits_for_any_timeline_lease_in_the_project() -> None:
    # grades.copy onto the current timeline, with a reference timeline by id: the current
    # timeline is unknown at submit, so the op needs the whole project, not just the reference.
    show = Show()
    params = _two_timeline_grades(show)
    params["timeline"] = None
    show.leases.acquire(
        OTHER, Scope.TIMELINE, timeline_key(show.pid, show.branch.GetUniqueId()), ttl=600
    )
    hot = make_hot_op(
        "grades.copy", params, target=show.locator(), holder=AGENT, require_branch=False
    )
    assert (hot.timeline_id, hot.extra_timeline_ids) == (None, ())
    op_id = show.broker.submit(hot)
    show.broker.run_until_idle()
    assert show.broker.result(op_id) is None


def test_a_duplicate_leases_its_source_and_nothing_else() -> None:
    def duplicate(world: Show, leased: Any) -> Any:
        world.leases.acquire(
            OTHER, Scope.TIMELINE, timeline_key(world.pid, leased.GetUniqueId()), ttl=600
        )
        op_id = submit_op(
            world.broker,
            "timeline.duplicate",
            {"source": world.ref("Main"), "name": "Main v2"},
            target=world.locator(),
            holder=AGENT,
        )
        world.broker.run_until_idle()
        return world.broker.result(op_id)

    world = Show()
    assert duplicate(world, world.main) is None  # the source is held: wait
    world = Show()
    result = duplicate(world, world.branch)  # another timeline is held: go ahead
    assert result is not None and result.status is OpStatus.OK, result
