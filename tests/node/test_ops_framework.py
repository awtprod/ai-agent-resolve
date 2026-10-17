"""The op framework: ops go through the broker, assert the project by id, fail structurally."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aar.node.broker import OpStatus
from aar.node.leases import Holder, Scope, project_key
from aar.node.locator import ProjectLocator
from aar.node.ops import OpError, get_op, list_ops, make_hot_op, submit_op

from .ops_support import AGENT, Show, media

EXPECTED = {
    "project.find": (True, False),
    "timeline.list": (True, False),
    "timeline.describe": (True, False),
    "media.find": (True, False),
    "timeline.duplicate": (True, True),
    "timeline.append_ranges": (True, True),
    "item.set_audio_gain": (True, True),
    "item.set_fades": (True, True),
    "grades.copy": (True, True),
    "clip.set_input_color_space": (True, True),
    "markers.add": (True, True),
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
    lease = show.leases.acquire(other, Scope.PROJECT, project_key("Show"), ttl=600)
    op_id = submit_op(show.broker, "timeline.list", {}, target=show.locator(), holder=AGENT)
    show.broker.run_until_idle()
    assert show.broker.result(op_id) is None
    assert [op.id for op in show.broker.pending()] == [op_id]
    show.leases.release(lease.id, other)
    show.broker.run_until_idle()
    result = show.broker.result(op_id)
    assert result is not None and result.status is OpStatus.OK


def test_op_body_never_runs_in_a_project_with_another_id() -> None:
    show = Show()
    twin = show.fake.add_project("Show", ["Main"], ("Elsewhere",))
    hot = make_hot_op(
        "timeline.duplicate",
        {"name": "copy"},
        target=ProjectLocator(("Elsewhere",), "Show", twin.GetUniqueId()),
        holder=AGENT,
    )
    with pytest.raises(OpError) as err:
        hot.fn(show.gw)  # the human's "Show" is open, not the twin
    assert err.value.code == "wrong_project"
    assert err.value.detail["actual_id"] == show.project.GetUniqueId()
    assert show.fake.calls == []  # nothing was duplicated in the wrong project


def test_op_errors_reach_the_broker_result_with_fix_hint_and_trap() -> None:
    show = Show()
    error = show.fails(
        "timeline.append_ranges",
        {
            "timeline": {"name": "Main [agent]"},
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
