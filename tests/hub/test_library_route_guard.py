"""S25d: no fault of the library route can fail the export step or lose the FCPXML hand-off.

The export step runs everything the library route does inside one guarded region
(:func:`aar.recipes.ccn.steps.library_route`). These tests run the real export step of a real
recipe run again and again, each time with a failure injected at one call the route makes.
The calls are not listed by hand: a dry run traces every call made from the route's own code
(the guard, :mod:`aar.recipes.ccn.library_handoff` and the notes' library section), and the
test then raises at each of those call sites in turn (a trace function that raises at a call
makes the called function raise on entry). Each time the FCPXML, the review video and the
notes must be published, the step must return, and the refusal ``library_failed`` must name the
injected error in the step's result and the notes. A call added later inside the route is
injected too; a library call added to the export step outside the guard changes the export
step's own calls, which another test pins to be the same with and without a library."""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import inspect
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest
from ccn_recipe_support import PC_ROOT, CcnEnv, agent_cut, make_run, outputs, start, work
from library_handoff_support import make_library_setup

from aar.recipes.ccn import handoff, library_handoff, steps
from aar.recipes.context import StepContext, StepOutput
from ccn_support import needs_ffmpeg

pytestmark = needs_ffmpeg
pytest.importorskip("zstandard")

ONE = {"ranges": [{"start_s": 0.9, "end_s": 11.1}]}


class Injected(Exception):
    """The failure injected at one call."""


@dataclasses.dataclass
class Captured:
    """The export step's real context from a worker run, to run the step again."""

    env: CcnEnv
    ctx: StepContext
    library: dict[str, Any]
    projects: Path
    fcpxml: bytes
    runs: int = 0

    def run(self, monkeypatch: pytest.MonkeyPatch, **params: Any) -> StepOutput:
        """The export step once more, into a fresh export folder, the library emptied after."""
        self.runs += 1
        exports = self.env.exports.parent / f"exports-{self.runs}"
        exports.mkdir()
        monkeypatch.setenv("AAR_EXPORT_DIR", str(exports))
        ctx = dataclasses.replace(
            self.ctx,
            op_node=f"{self.ctx.op_node}-again-{self.runs}",
            params=self.ctx.params.model_copy(update=params) if params else self.ctx.params,
            _cancelled=lambda: False,
        )
        try:
            return steps.export_step(ctx)
        finally:
            for placed in self.projects.iterdir():
                shutil.rmtree(placed)


@pytest.fixture
def captured(ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch) -> Captured:
    env = ccn_env
    show = env.show("cam-a.mp4")
    library = make_library_setup(env.media / "share", [f"{PC_ROOT}\\cam-a.mp4"])
    run = make_run(env, env.params(show, library=library), key="lib")
    start(env, run, "lib")
    assert work(env) == 5
    agent_cut(env, run, keep=ONE, key="r1")
    seen: list[StepContext] = []
    real = steps.library_route

    def capture(ctx: StepContext, *args: Any) -> Any:
        seen.append(ctx)
        return real(ctx, *args)

    with monkeypatch.context() as patch:
        patch.setattr(steps, "library_route", capture)
        assert work(env) == 2  # render, export
    export = outputs(env, run, "export")
    assert export["library_handoff"]["status"] == "placed", export["library_handoff"]
    projects = Path(library["folder"]).joinpath(*export["library_handoff"]["parts"]).parent
    for placed in projects.iterdir():
        shutil.rmtree(placed)
    [ctx] = seen
    fcpxml = Path(export["copy"]["fcpxml"]).read_bytes()
    return Captured(env, ctx, library, projects, fcpxml)


# --------------------------------------------------------------------------- tracing


def _codes(*functions: Callable[..., Any]) -> set[CodeType]:
    """The code of ``functions`` and of everything defined inside them (comprehensions,
    lambdas, nested functions)."""
    out: set[CodeType] = set()
    stack = [f.__code__ for f in functions]
    while stack:
        code = stack.pop()
        if code not in out:
            out.add(code)
            stack.extend(c for c in code.co_consts if isinstance(c, CodeType))
    return out


def route_code() -> set[CodeType]:
    """The library route's own code: the guard and its body in the export step, every function
    of :mod:`.library_handoff`, and the notes' library section."""
    module = [
        f
        for f in vars(library_handoff).values()
        if inspect.isfunction(f) and f.__module__ == library_handoff.__name__
    ]
    return _codes(
        steps.library_route,
        steps._library_route,
        *module,
        handoff.library_lines,
        handoff.refusal_lines,
    )


Site = tuple[str, int, str]


def _site(frame: FrameType) -> Site:
    caller = frame.f_back
    assert caller is not None
    return (caller.f_code.co_qualname, caller.f_lineno, frame.f_code.co_qualname)


class Tracer:
    """Records (or raises at) every call made from ``callers``' frames."""

    def __init__(self, callers: set[CodeType], inject: Site | None = None) -> None:
        self.callers = callers
        self.inject = inject
        self.sites: list[Site] = []
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "call" or frame.f_back is None or frame.f_back.f_code not in self.callers:
            return None
        site = _site(frame)
        self.sites.append(site)
        if self.inject is not None and site == self.inject and not self.fired:
            self.fired = True
            sys.settrace(None)
            raise Injected(f"injected at {site[2]} called from {site[0]} line {site[1]}")
        return None


@contextlib.contextmanager
def tracing(tracer: Tracer) -> Iterator[Tracer]:
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        yield tracer
    finally:
        sys.settrace(previous)


def _first_sites(sites: list[Site]) -> list[Site]:
    out: list[Site] = []
    for site in sites:
        if site not in out:
            out.append(site)
    return out


# --------------------------------------------------------------------------- checks


def _published(result: StepOutput, fcpxml: bytes) -> str:
    """The FCPXML hand-off is published as without a library; the notes' text."""
    copy = result.outputs["copy"]
    assert Path(copy["fcpxml"]).read_bytes() == fcpxml
    assert Path(copy["review_mp4"]).is_file()
    notes = Path(copy["handoff_notes"]).read_text(encoding="utf-8")
    assert notes.startswith("HANDOFF NOTES: ")
    for role in ("fcpxml", "export_report", "handoff_notes"):
        assert result.outputs[role] in result.artifact_ids
    json.dumps(result.outputs, allow_nan=False)
    return notes


def _failed(result: StepOutput, *needles: str) -> dict[str, Any]:
    record = result.outputs["library_handoff"]
    assert record["status"] == "refused", record
    error = record["error"]
    assert error["code"] == "library_failed", error
    for needle in needles:
        assert needle in error["message"], (needle, error["message"])
    [warning] = result.outputs["warnings"]
    assert warning == {
        "code": "library_library_failed",
        "message": error["message"],
        "fix_hint": error["hint"],
    }
    return record


# --------------------------------------------------------------------------- tests


def test_a_failure_at_every_call_of_the_library_route_keeps_the_fcpxml_hand_off(
    captured: Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    callers = route_code()
    with tracing(Tracer(callers)) as dry:
        clean = captured.run(monkeypatch)
    assert clean.outputs["library_handoff"]["status"] == "placed"
    assert "warnings" not in clean.outputs
    _published(clean, captured.fcpxml)
    sites = _first_sites(dry.sites)
    called = {site[2] for site in sites}
    # What the route does (a check that the trace sees the route's calls): the parameters'
    # files (realpath, the facts file), the snapshot folder, the export, the recording of the
    # hand-off (its report stored), the record and the notes' text.
    import aar.plan.export.library as export_module
    from aar.drp.mediapool import gate
    from aar.recipes.ccn import transcript

    for function in (
        transcript.check_cancelled,
        library_handoff.hand_over,
        os.path.realpath,
        library_handoff._facts,
        gate.load_facts,
        library_handoff.snapshot_dir,
        Path.mkdir,
        export_module.export_library,
        steps._json_artifact,
        handoff.library_lines,
        json.dumps,
    ):
        assert function.__code__.co_qualname in called, (function, sorted(called))
    assert len(sites) >= 30, sites
    placed_after = False
    for site in sites:
        with tracing(Tracer(callers, inject=site)) as tracer:
            result = captured.run(monkeypatch)
        assert tracer.fired, site
        notes = _published(result, captured.fcpxml)
        record = _failed(result, "Injected: injected at " + site[2])
        assert "LIBRARY PROJECT: NOT WRITTEN" in notes and "[library_failed]" in notes
        assert "Injected: injected at" in notes
        assert record["round"] == 1 and record["recipe_node"] == captured.ctx.recipe_node
        placed_after |= "was already placed in the library" in record["error"]["message"]
    assert placed_after  # some injections came after the project was placed, and say so


def test_real_faults_and_interrupts_of_the_library_route(
    captured: Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = captured.env
    snapshots = env.work / library_handoff.SNAPSHOT_AREA
    if snapshots.is_dir():
        shutil.rmtree(snapshots)

    # F1: a regular file where the snapshot folder belongs (mkdir: FileExistsError).
    snapshots.write_bytes(b"in the way")
    result = captured.run(monkeypatch)
    _published(result, captured.fcpxml)
    _failed(result, "FileExistsError", "writing the library project")
    snapshots.unlink()

    # F1: no space left to make it.
    real_mkdir = Path.mkdir

    def full(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == library_handoff.SNAPSHOT_AREA:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_mkdir(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", full)
        result = captured.run(monkeypatch)
    _published(result, captured.fcpxml)
    _failed(result, "No space left on device")

    # F2: a facts file with one deeply nested line is a typed refusal at its source.
    facts = Path(captured.library["facts"])
    good = facts.read_bytes()
    facts.write_bytes(b'{"path": "E:\\\\a.mp4", "x": ' + b"[" * 100_000 + b"]" * 100_000 + b"}\n")
    result = captured.run(monkeypatch)
    _published(result, captured.fcpxml)
    record = result.outputs["library_handoff"]
    assert record["error"]["code"] == "bad_facts", record["error"]
    assert "nest deeper than" in record["error"]["message"]
    # ... and so is a line longer than the hub reads.
    facts.write_bytes(b'{"path": "' + b"a" * (2 * 1024 * 1024) + b'"}\n')
    result = captured.run(monkeypatch)
    _published(result, captured.fcpxml)
    assert result.outputs["library_handoff"]["error"]["code"] == "bad_facts"
    facts.write_bytes(good)

    # Storing the library export report fails (the hand-off is placed, not recorded).
    real_put = StepContext.put

    def put(self: StepContext, source: Any, **kwargs: Any) -> str:
        if (kwargs.get("meta") or {}).get("role") == "library-export-report":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_put(self, source, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(StepContext, "put", put)
        result = captured.run(monkeypatch)
    notes = _published(result, captured.fcpxml)
    record = _failed(result, "storing the library export report", "was already placed")
    assert "report" not in record and result.artifact_ids == (
        result.outputs["fcpxml"],
        result.outputs["export_report"],
        result.outputs["handoff_notes"],
    )
    assert "was already placed" in notes

    # An exception whose own text cannot be made: still a refusal, named by its type.
    class Unprintable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no text")

    import aar.plan.export.library as export_module

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise Unprintable()

    with monkeypatch.context() as patch:
        patch.setattr(export_module, "export_library", broken)
        result = captured.run(monkeypatch)
    _published(result, captured.fcpxml)
    _failed(result, "Unprintable: <unprintable exception>")

    # The worker stopping (StepInterrupted is a KeyboardInterrupt) or code exiting: not caught.
    # The step records no result and is claimed again later (DESIGN); never a refusal.
    for stop in (KeyboardInterrupt, SystemExit):

        def stopped(*args: Any, stop: type[BaseException] = stop, **kwargs: Any) -> Any:
            raise stop()

        with monkeypatch.context() as patch:
            patch.setattr(export_module, "export_library", stopped)
            with pytest.raises(stop):
                captured.run(monkeypatch)


def test_the_export_step_makes_the_same_calls_with_and_without_a_library(
    captured: Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every library call of the export step is inside the guard: the export step's own calls
    are the same with a library as without (the guard returns None at once without one), so a
    library call added outside the guard fails this test. And without a library the step's
    result is what it was before S25d: no library keys, three artifacts, no library text."""
    callers = _codes(steps.export_step)
    with tracing(Tracer(callers)) as with_library:
        placed = captured.run(monkeypatch)
    with tracing(Tracer(callers)) as without:
        plain = captured.run(monkeypatch, library=None)
    assert placed.outputs["library_handoff"]["status"] == "placed"
    names = [site[2] for site in with_library.sites]
    assert names == [site[2] for site in without.sites]
    assert "library_route" in names
    assert "library_handoff" not in plain.outputs and "warnings" not in plain.outputs
    assert len(plain.artifact_ids) == 3
    notes = _published(plain, captured.fcpxml)
    assert "LIBRARY" not in notes and "library" not in notes.lower()
    assert sorted(plain.outputs) == [
        "assets",
        "copy",
        "desktop_return",
        "export_report",
        "fcpxml",
        "handoff_notes",
        "review_attachments",
    ]
    # The placed route adds exactly its record and its report artifact.
    assert set(placed.outputs) - set(plain.outputs) == {"library_handoff"}
    report = placed.outputs["library_handoff"]["report"]
    roles = ("fcpxml", "export_report", "handoff_notes")
    assert placed.artifact_ids == (*(placed.outputs[r] for r in roles), report)
    assert plain.artifact_ids == tuple(plain.outputs[r] for r in roles)
    assert (
        json.loads(captured.env.hub.store.read_bytes(report))["project"]["project_id"]
        == (placed.outputs["library_handoff"]["project_id"])
    )
