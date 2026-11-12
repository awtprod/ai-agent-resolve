"""``ccn-longform`` round 3 (S19a, s19a-val/r2): the render/laughs sweep keeps what it cannot
place, media replaced at the same path after the stub, one file given twice through a hardlink,
a re-claimed render with leftover scratch, a cancelled run's scratch, the transcript sweep."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from ccn_recipe_support import CcnEnv, agent_cut, latest, make_run, outputs, start, work
from write_support import AGENT

from aar.recipes.ccn import scratch
from aar.recipes.ccn.params import CcnLongformParams
from aar.recipes.ccn.transcript import AREA, STALE_S
from aar.recipes.run import recipe_cancel
from ccn_support import make_show, needs_ffmpeg

pytestmark = needs_ffmpeg


def _cold(env: CcnEnv, params: dict[str, object], key: str) -> str:
    run = make_run(env, params, key=key)
    start(env, run, key)
    work(env)
    return run


def _sweeper(env: CcnEnv) -> SimpleNamespace:
    """What :func:`scratch.sweep` reads of a step context: the log and the sweeping op."""
    return SimpleNamespace(log=env.hub.log, recipe_node="nd_sweeper", op_node="nd_sweeper_op")


# -- LOW-2 and n22: the sweep removes only what it knows to be dead ---------------------------


def test_the_sweep_removes_dead_scratch_only_and_keeps_what_it_cannot_place(
    ccn_env: CcnEnv,
) -> None:
    env = ccn_env
    show = env.show()
    cancelled = make_run(env, env.params(show, timeline="Cancelled"), key="c")
    recipe_cancel(env.hub.log, cancelled, AGENT, idempotency_key="cancel-c")
    live = _cold(env, env.params(show), "1")  # waits at cut
    live_cut, done_stub = latest(env, live, "cut").op_id, latest(env, live, "stub").op_id
    area = env.work / "aar-render"
    for folder in (
        area / cancelled / "nd_op",
        area / live / live_cut,
        area / live / done_stub,
        area / live / "nd_not_an_op",
        area / "nd_another_hubs_run" / "nd_op",
    ):
        folder.mkdir(parents=True)
    (area / "stray.txt").write_text("not scratch")

    assert scratch.sweep(_sweeper(env), area) == 2
    assert not (area / cancelled).exists()  # cancelled run: swept (n22)
    assert not (area / live / done_stub).exists()  # an op known not to be live: swept
    assert (area / live / live_cut).is_dir()  # live op: kept
    assert (area / live / "nd_not_an_op").is_dir()  # not one of the run's ops: kept
    assert (area / "nd_another_hubs_run" / "nd_op").is_dir()  # unknown run: kept (LOW-2)
    assert (area / "stray.txt").is_file()


def test_the_sweep_keeps_everything_when_the_state_cannot_be_read(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    cancelled = make_run(env, env.params(env.show()), key="c")
    recipe_cancel(env.hub.log, cancelled, AGENT, idempotency_key="cancel-c")
    area = env.work / "aar-render"
    (area / cancelled / "nd_op").mkdir(parents=True)

    def locked(*_a: object, **_k: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(scratch, "load_state", locked)
    assert scratch.sweep(_sweeper(env), area) == 0
    assert (area / cancelled / "nd_op").is_dir()  # unloadable: kept

    def broken(*_a: object, **_k: object) -> None:
        raise RuntimeError("a bug, not a database error")

    monkeypatch.setattr(scratch, "load_state", broken)
    with pytest.raises(RuntimeError, match="a bug"):  # not swallowed
        scratch.sweep(_sweeper(env), area)
    assert (area / cancelled / "nd_op").is_dir()
