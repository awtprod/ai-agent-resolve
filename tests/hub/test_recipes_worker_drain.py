"""The worker's drain: once ``stop`` is set, ``_tick`` starts no new cold step (callables cannot
be interrupted, so a fresh claim would make shutdown wait for a whole new task)."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from recipe_support import make_run
from write_support import AGENT

from aar.recipes import worker
from aar.recipes.run import recipe_start


def _claimed(hub: Any) -> int:
    return len(hub.log.read(0, project_id=hub.project_id, types=["recipe.claimed"], limit=-1))


def _tick(hub: Any, stop: threading.Event) -> bool:
    return worker._tick(
        hub.log,
        hub.store,
        AGENT,
        dispatcher=None,
        clock=worker.utcnow,
        lease_s=60,
        heartbeat_s=15,
        stop=stop,
    )


def test_stop_set_during_the_first_loop_claims_nothing(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    stop = threading.Event()
    real_advance = worker.advance

    def advance_then_stop(*args: Any, **kwargs: Any) -> Any:
        result = real_advance(*args, **kwargs)
        stop.set()  # a stop request arrives while the first loop is advancing recipes
        return result

    monkeypatch.setattr(worker, "advance", advance_then_stop)
    assert _tick(hub, stop) is False
    assert _claimed(hub) == 0


def test_without_stop_the_same_tick_claims_and_runs(hub: Any, registered_recipe: Any) -> None:
    """Control: the step the stop test left unclaimed is claimable."""
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    assert _tick(hub, threading.Event()) is True
    assert _claimed(hub) == 1


class _BadStr(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no text for you")


def test_callable_whose_str_raises_still_gets_a_failed_result(
    hub: Any, registered_recipe: Any
) -> None:
    """A broken ``__str__`` must not skip ``record_result`` (the op would hold its lease)."""
    from aar.recipes import registry

    def boom(ctx: Any) -> Any:
        raise _BadStr()

    registry.register_callable("rectest.cold", boom)
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    assert _tick(hub, threading.Event()) is True
    results = hub.log.read(0, project_id=hub.project_id, types=["recipe.result"], limit=-1)
    assert len(results) == 1
    payload = results[0].payload
    assert payload["outcome"] == "failed"
    assert payload["error"]["code"] == "error"
    assert payload["error"]["message"].startswith("_BadStr:")
