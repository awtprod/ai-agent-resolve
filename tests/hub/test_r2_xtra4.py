"""Round-4/5 pins, adopted from the validator's ``probes/r4/xtra_r4.py`` (R4-2) and retargeted at
the round-5 worker-error rule (R4-1).

- R19/R20 (R3-4): garbage port responses count toward the bad-response limit.
- R23 (R3-1): an error message is bounded on STORED (escaped JSON) bytes.
- R21/R22/R25 (R3-3 -> R4-1): a swallowed per-recipe worker exception leaves AT MOST ONE
  ``recipe.worker_error`` hub.write record per (recipe node, exception type), whatever the message;
  a repeat writes nothing; the status view lists at most 5 byte-bounded entries plus the total
  count and the whole view stays within 64 KiB; ``logging`` gets one WARNING line per
  (recipe, exception type) per 60 s with a "suppressed N" count.
- The validator's flood probe (``probes/r4/flood.py``): 200 worker ticks with changing (emoji)
  messages.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from typing import Any

import pytest
from recipe_support import make_run
from val_support import worker_idle
from write_support import AGENT

from aar.recipes import worker
from aar.recipes.events import OP_WORKER_ERROR
from aar.recipes.ports import HotJobView, HotResult
from aar.recipes.run import recipe_start
from aar.recipes.state import load_state
from aar.recipes.status import (
    MAX_VIEW_BYTES,
    MAX_WORKER_ERROR_ENTRY_BYTES,
    MAX_WORKER_ERRORS_LISTED,
    status_view,
)

EMOJI = "\U0001f600"


@pytest.fixture(autouse=True)
def _fresh_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The WARNING rate-limit table is process state; give every test its own."""
    monkeypatch.setattr(worker, "_log_state", {})


class _Bad:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def submit(self, **kw: Any) -> Any:
        return object() if self.mode == "ref" else "job-1"

    def poll(self, job_ref: Any) -> Any:
        if self.mode == "outputs":
            return HotJobView("done", result=HotResult(outputs="not a dict"))  # type: ignore[arg-type]
        if self.mode == "error":
            return HotJobView("failed", error={"code": "x", "message": "é" * 10000})
        return HotJobView("running")


@pytest.mark.parametrize("mode", ["ref", "outputs"])
def test_r19_r20_garbage_counts_toward_the_limit(
    hub: Any, registered_recipe: Any, mode: str
) -> None:
    disp = _Bad(mode)
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s", dispatcher=disp)
    for _ in range(8):
        worker.advance(hub.log, rn, actor=AGENT, dispatcher=disp)
    h = load_state(hub.log, rn).latest_in_round("h", 1)
    assert h.status == "failed" and h.result_error["code"] == "dispatcher_bad_response"


def test_r23_non_ascii_error_bounded_on_stored_bytes(hub: Any, registered_recipe: Any) -> None:
    disp = _Bad("error")
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s", dispatcher=disp)
    worker_idle(hub, disp)
    con = sqlite3.connect(str(hub.root / "hub.db"))
    rows = [
        json.loads(r[0])
        for r in con.execute("SELECT payload FROM events WHERE type='recipe.result'").fetchall()
    ]
    con.close()
    msgs = [r["error"]["message"] for r in rows if r.get("error")]
    stored = max(len(json.dumps(m).encode()) for m in msgs)
    assert stored <= 4 * 1024


# -- the worker-error rule (R4-1) ---------------------------------------------------------


def _ticks(
    hub: Any, monkeypatch: pytest.MonkeyPatch, n: int, make_exc: Callable[[int], Exception]
) -> dict[str, int]:
    """Run ``n`` real worker ticks with ``advance`` raising ``make_exc(i)`` on its i-th call."""
    count = {"i": 0}

    def flaky(log: Any, recipe_node: str, **kw: Any) -> Any:
        count["i"] += 1
        raise make_exc(count["i"])

    monkeypatch.setattr(worker, "advance", flaky)
    for _ in range(n):
        worker._tick(
            hub.log,
            hub.store,
            AGENT,
            dispatcher=None,
            clock=worker.utcnow,
            lease_s=60,
            heartbeat_s=15,
            stop=threading.Event(),
        )
    return count


def _records(hub: Any) -> list[dict[str, Any]]:
    con = sqlite3.connect(str(hub.root / "hub.db"))
    rows = con.execute(
        "SELECT payload FROM events WHERE type='hub.write' "
        "AND json_extract(payload, '$.op') = 'recipe.worker_error'"
    ).fetchall()
    con.close()
    return [json.loads(r[0]) for r in rows]


def _record_bytes(hub: Any) -> list[int]:
    """The STORED bytes of each worker_error record's payload, as the log holds them."""
    con = sqlite3.connect(str(hub.root / "hub.db"))
    rows = con.execute(
        "SELECT length(CAST(payload AS BLOB)) FROM events WHERE type='hub.write' "
        "AND json_extract(payload, '$.op') = 'recipe.worker_error'"
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def _event_count(hub: Any) -> int:
    con = sqlite3.connect(str(hub.root / "hub.db"))
    (n,) = con.execute("SELECT count(*) FROM events").fetchone()
    con.close()
    return n


def _stored(value: Any) -> int:
    return len(json.dumps(value).encode())


def _view_bytes(view: dict[str, Any]) -> int:
    return len(json.dumps(view, sort_keys=True, separators=(",", ":")).encode())


def _types(k: int) -> list[type[Exception]]:
    return [type(f"Flap{j}Error", (RuntimeError,), {}) for j in range(k)]


def test_r21_r22_constant_worker_error_recorded_exactly_once(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    _ticks(hub, monkeypatch, 10, lambda i: RuntimeError("the node gateway is unreachable"))
    recs = _records(hub)
    assert len(recs) == 1
    assert recs[0]["recipe_node"] == rn and recs[0]["exc_type"] == "RuntimeError"
    assert recs[0]["error"] == "RuntimeError: the node gateway is unreachable"


@pytest.mark.parametrize(
    "case",
    [
        # the three cases of probes/r4/flood.py, 200 ticks each
        ("constant", lambda i: RuntimeError("the node gateway is unreachable")),
        ("varying", lambda i: TimeoutError(f"poll timed out after {i * 0.137:.6f}s (attempt {i})")),
        ("emoji", lambda i: ValueError(f"{i} " + EMOJI * 4096)),
    ],
    ids=lambda c: c[0],
)
def test_flood_probe_one_record_per_type_and_the_view_stays_bounded(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch, case: Any
) -> None:
    """R4-1 (the validator's flood probe): 200 ticks whose message changes every tick leave at
    most one record per exception type, each within 4 KiB stored, and the view under 64 KiB."""
    _, make_exc = case
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    count = _ticks(hub, monkeypatch, 200, make_exc)
    assert count["i"] >= 200
    recs = _records(hub)
    assert len(recs) == 1, f"{len(recs)} worker_error records in 200 ticks"
    assert max(_record_bytes(hub)) <= 4 * 1024  # the whole stored record, escaped JSON
    view = status_view(hub.log, rn)
    assert _view_bytes(view) <= MAX_VIEW_BYTES
    assert view["diagnostics"]["worker_error_count"] == 1
    assert len(view["diagnostics"]["worker_errors"]) == 1


def test_flood_of_mixed_types_records_one_per_type(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinds = _types(3)
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    _ticks(hub, monkeypatch, 200, lambda i: kinds[i % 3](f"{i} " + EMOJI * 512))
    recs = _records(hub)
    assert sorted(r["exc_type"] for r in recs) == ["Flap0Error", "Flap1Error", "Flap2Error"]


def test_a_repeat_writes_nothing(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    _ticks(hub, monkeypatch, 3, lambda i: KeyError(f"first {i}"))
    before = _event_count(hub)
    first = _records(hub)
    _ticks(hub, monkeypatch, 50, lambda i: KeyError(f"later, different {i}" + EMOJI * i))
    assert _event_count(hub) == before
    assert _records(hub) == first


def test_r25_diagnostics_list_capped_with_a_total_count(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinds = _types(8)
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    _ticks(hub, monkeypatch, 16, lambda i: kinds[i % 8](f"varying {i}"))
    assert len(_records(hub)) == 8
    diag = status_view(hub.log, rn)["diagnostics"]
    assert diag["worker_error_count"] == 8
    assert len(diag["worker_errors"]) == MAX_WORKER_ERRORS_LISTED == 5


def test_view_entries_bounded_on_stored_bytes_with_huge_emoji_errors(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinds = _types(30)
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    _ticks(hub, monkeypatch, 60, lambda i: kinds[i % 30](f"{i} " + EMOJI * 4096))
    assert len(_records(hub)) == 30 and max(_record_bytes(hub)) <= 4 * 1024
    view = status_view(hub.log, rn)
    diag = view["diagnostics"]
    assert diag["worker_error_count"] == 30 and len(diag["worker_errors"]) == 5
    for entry in diag["worker_errors"]:
        assert entry["error"].startswith(entry["exc_type"] + ": ")
        assert _stored(entry["error"]) <= MAX_WORKER_ERROR_ENTRY_BYTES
    assert _view_bytes(view) <= MAX_VIEW_BYTES


def test_a_forged_worker_error_record_is_not_counted(hub: Any, registered_recipe: Any) -> None:
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    hub.log.append(
        "hub.write",
        AGENT,
        {"op": OP_WORKER_ERROR, "recipe_node": rn, "exc_type": "X", "error": EMOJI * 9000},
        project_id=hub.project_id,
        idempotency_key="forged-worker-error",
    )
    diag = status_view(hub.log, rn)["diagnostics"]
    assert diag == {"worker_errors": [], "worker_error_count": 0}


def test_worker_error_log_is_rate_limited_per_recipe_and_type(
    hub: Any,
    registered_recipe: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every occurrence is accounted for in ``logging``: one WARNING line per (recipe, exception
    type) per 60 s; the next line after the window reports how many were suppressed."""
    now = {"t": 1000.0}
    monkeypatch.setattr(worker, "_monotonic", lambda: now["t"])
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    caplog.set_level(logging.WARNING, logger="aar.recipes.worker")

    def lines() -> list[str]:
        return [r.getMessage() for r in caplog.records if "worker swallowed" in r.getMessage()]

    def exc(i: int) -> Exception:
        now["t"] += 0.25  # 200 ticks span 50 s: inside one window
        return TimeoutError(f"poll timed out (attempt {i}) " + EMOJI * 4096)

    count = _ticks(hub, monkeypatch, 200, exc)
    assert len(lines()) == 1 and "suppressed" not in lines()[0]
    assert len(lines()[0]) <= 4096 + 200
    _ticks(hub, monkeypatch, 1, lambda i: KeyError("another type logs on its own"))
    assert len(lines()) == 2 and "KeyError" in lines()[1]
    seen = count["i"]
    now["t"] += 61
    _ticks(hub, monkeypatch, 1, lambda i: TimeoutError("after the window"))
    assert len(lines()) == 3
    assert f"(suppressed {seen - 1} since the last report)" in lines()[2]
    assert all(
        r.levelno == logging.WARNING for r in caplog.records if "swallowed" in r.getMessage()
    )


def test_an_unprintable_exception_does_not_stop_the_worker(
    hub: Any, registered_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Unprintable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no str for you")

    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    _ticks(hub, monkeypatch, 3, lambda i: Unprintable())
    recs = _records(hub)
    assert [r["error"] for r in recs] == ["Unprintable: <unprintable exception>"]
    assert status_view(hub.log, rn)["diagnostics"]["worker_error_count"] == 1
