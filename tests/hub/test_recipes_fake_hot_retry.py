"""FakeHotDispatcher reserves a request_id only once its job exists."""

from __future__ import annotations

from typing import Any

import pytest

from aar.recipes.ports import FakeHotDispatcher, HotJobView, HotResult


def _submit(d: FakeHotDispatcher, request_id: str = "req-1") -> str:
    return d.submit(request_id=request_id, project_id="p", kind="k", params={}, agent_id="a")


def test_behaviour_that_raises_once_leaves_no_reservation() -> None:
    state = {"n": 0}

    def flaky(_params: Any) -> HotJobView:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("node unreachable")
        return HotJobView("done", result=HotResult(outputs={"ok": True}))

    d = FakeHotDispatcher({"k": flaky})
    with pytest.raises(RuntimeError):
        _submit(d)
    assert d.submits == [] and d._by_request == {}
    ref = _submit(d)  # the retry with the same request_id creates the job
    assert d.submits == ["req-1"]
    view = d.poll(ref)
    assert view.state == "done" and view.result is not None and view.result.outputs == {"ok": True}
    assert _submit(d) == ref  # and is idempotent from then on
    assert state["n"] == 2
