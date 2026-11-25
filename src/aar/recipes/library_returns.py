"""Pick up the owner's saves of handed-over library projects (S25d).

The export step records each library hand-off in its runner-written result
(``outputs.library_handoff``, :mod:`aar.recipes.ccn.library_handoff`): the library, the project
folder's path below it, the project id, the timeline, the plan it was written from and the
SHA-256 of the placed ``Project.db``. The cold worker's sweep (:func:`scan_library_returns`,
beside the ``.drt`` sweep of :mod:`aar.recipes.desktop_returns`) checks every hand-off whose
review is still open (the run neither done nor cancelled, the hand-off's round the current one,
its export attempt not superseded, and no decision on that round's gate review):

1. The library must still lead (``realpath``) inside the media root, and is then opened from
   the media root's descriptor one component of that resolved path at a time, and the project
   folder from the library's one component at a time, never following a link (``O_NOFOLLOW``):
   a link swapped in after the check is refused, not followed.
2. A ``Project.db-journal`` or ``-wal`` beside the file is a save in progress: ``busy``.
3. A settled copy (:func:`aar.media.settle.copy_settled`: unchanged for 2 s and while copied,
   at most :attr:`Limits.db_bytes`) is made into ``<AAR_WORK_DIR>/aar-library-snapshots``, the
   hub's own disk, never the system temporary folder; the journal is checked again after it.
   A file this worker has found not settled in :attr:`Limits.busy_after` sweeps in a row (one
   rewritten all the time) is reported ``busy`` (and looked at again in each sweep).
4. Its SHA-256 equal to the **most recent state the hub has recorded for the hand-off** (the
   placed file's while nothing has been returned, else the newest return's): nothing to do.
   The latest save always wins (owner, S25e): a save back to content seen earlier (an edit
   undone, or the project as placed) is a new return whenever it differs from that newest
   state. Equal to content already reported as a condition: nothing to do.
5. Read with the timeline import's reader, refusing a file of any other project id
   (``project_replaced``) before reading the timeline; a missing timeline is
   ``timeline_missing``, a damaged file ``damaged``.
6. Imported against the plan it was written from and stored exactly as a ``.drt`` return is
   (:func:`aar.recipes.desktop_returns.record_import`: the same ``recipe.timeline_return``
   event with ``needs_review``, the edited plan and the diff as artifacts, the same rule that
   pickup never replaces the active plan or decides a review), with ``source: "library_save"``
   and the saved ``Project.db`` itself as an artifact (its SHA-256 also as
   ``snapshot_sha256``). Idempotent on the run, round, plan, hand-off, the newest recorded
   state it follows (``placed``, or the return event's id) and the snapshot's hash: the same
   save seen twice (two workers, a restarted one) is one return, and the same content saved
   again later (after another return) is a new one. A return is recorded only while the state
   it follows is still the newest, checked inside the transaction that records it.

**Conditions** are ``recipe.library_pickup`` events (:data:`EVENT_PICKUP`), one per distinct
condition: a condition about a file's content (``project_replaced``, ``damaged``,
``timeline_missing``, ``too_large``) once per content hash, and one about the place
(``library_missing``, ``not_a_library``, ``project_missing``, ``path_refused``,
``path_outside_media_root``, ``busy``, ``unsupported_platform``, ``pickup_failed``) once per
cause between two returns: each place where a condition is raised names its cause (a link
below the library and a ``Project.db`` that is not a regular file are both ``path_refused``,
two events), recorded as the event's ``cause``.
They never raise: the sweep goes on with the next project, and the same project is looked at
again in the next sweep (a file whose content already gave a condition is not read again).

**Bounds** per sweep (:class:`Limits`): projects checked, bytes copied (a copy that is then
refused counts too), seconds spent; a project
larger than the bytes left waits for a later sweep, and the order the hand-offs are taken in
turns every minute, so one large, slow or hostile project cannot keep the others waiting.
Snapshots a killed worker left in the snapshot folder (``.aar-pickup-*`` and the reader's
``.aar-library-*``, an hour old) are removed at the start of each sweep; anything else there is
counted, never removed.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, Event, EventLog
from aar.media.settle import TooLargeError, copy_settled
from aar.plan.export.errors import ExportError
from aar.plan.export.media import media_map_from_mapping
from aar.recipes.events import EVENT_RESULT, RUNNER

__all__ = [
    "EVENT_PICKUP",
    "Limits",
    "library_conditions",
    "scan_library_returns",
    "sweep_snapshots",
]

EVENT_PICKUP = "recipe.library_pickup"
MAX_EVENTS = 100_000
SNAPSHOT_PREFIXES = (".aar-pickup-", ".aar-library-")
#: The pickup's own copies go one folder below the reader's (the reader refuses to take its
#: snapshot into the folder of the file it reads).
PICKUP_AREA = "pickup"
#: Codes of conditions about a file's content: reported once per content hash.
CONTENT_CONDITIONS = frozenset({"project_replaced", "damaged", "timeline_missing", "too_large"})


@dataclass(frozen=True)
class Limits:
    """What one sweep may spend."""

    projects: int = 20
    bytes: int = 1024 * 1024 * 1024
    seconds: float = 60.0
    #: One project's file at most (the library reader's own default).
    db_bytes: int = 256 * 1024 * 1024
    settle_seconds: float = 2.0
    #: A snapshot older than this in the snapshot folder was left by a killed worker.
    stale_seconds: float = 3600.0
    #: Sweeps in a row a file may be found not settled before it is reported ``busy``.
    busy_after: int = 10


class Handoff(BaseModel):
    """A runner-recorded hand-off, checked again before it is acted on."""

    model_config = ConfigDict(extra="ignore", strict=True)

    status: str = Field(pattern=r"^placed$")
    library: str = Field(min_length=1, max_length=1024)
    user: str = Field(min_length=1, max_length=255)
    folder: str = Field(min_length=1, max_length=255)
    parts: list[str] = Field(min_length=4, max_length=6)
    project_id: str = Field(min_length=1, max_length=128)
    timeline: str = Field(min_length=1, max_length=200)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    placed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    round: int = Field(ge=1, le=1000)
    recipe_node: str = Field(min_length=1, max_length=128)
    media_map: dict[str, str] = Field(min_length=1, max_length=16)


@dataclass
class _Found:
    event: Event
    handoff: Handoff
    op_node: str


class _Condition(Exception):
    """A condition to report; ``cause`` tells apart conditions of one code with different
    causes (each raise site names its own; module docstring)."""

    def __init__(
        self,
        code: str,
        message: str,
        hint: str,
        sha256: str | None = None,
        *,
        cause: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code, self.message, self.hint, self.sha256 = code, message, hint, sha256
        self.cause = cause or code


@dataclass
class Sweep:
    """What one sweep did (for the worker's log and the tests)."""

    checked: int = 0
    returned: int = 0
    unchanged: int = 0
    repeated: int = 0
    conditions: list[str] = field(default_factory=list)
    deferred: int = 0
    closed: int = 0
    invalid: int = 0
    cut_short: bool = False
    bytes: int = 0
    snapshots_removed: int = 0
    snapshot_leftovers: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- finding hand-offs


def _runner_events(
    log: EventLog, types: list[str], project_id: str | None = None
) -> Iterator[Event]:
    for count, event in enumerate(log.iter_events(types=types, project_id=project_id)):
        if count >= MAX_EVENTS:
            raise ValueError("library pickup history exceeds 100000 events; archive old runs")
        if event.actor == RUNNER:
            yield event


def _handoffs(log: EventLog, sweep: Sweep) -> list[_Found]:
    found = []
    for event in _runner_events(log, [EVENT_RESULT]):
        payload = event.payload
        out = payload.get("outputs")
        packet = out.get("library_handoff") if isinstance(out, dict) else None
        if not isinstance(packet, dict) or payload.get("outcome") != "done":
            continue
        if packet.get("status") != "placed" or packet.get("placed_sha256") is None:
            continue  # refused, or placed without a hash to compare saves with (said in notes)
        op_node = payload.get("op_node")
        try:
            handoff = Handoff.model_validate(packet)
        except ValidationError:
            sweep.invalid += 1
            continue
        if not isinstance(op_node, str) or event.project_id is None:
            sweep.invalid += 1
            continue
        found.append(_Found(event, handoff, op_node))
    return found


def _review_open(log: EventLog, found: _Found) -> bool:
    """Whether the hand-off's review is still open (module docstring)."""
    from aar.recipes.state import load_state

    state = load_state(log, found.handoff.recipe_node, project_id=found.event.project_id)
    if state.recipe_status in ("done", "cancelled"):
        return False
    if state.current_round() != found.handoff.round:
        return False
    op = state.op_by_id(found.op_node)
    if op is None or not op.is_done:
        return False  # superseded by a reopen or a rollback: a later export hands over anew
    gates = [s.id for s in state.recipe_def.steps if s.kind == "gate"] if state.recipe_def else []
    for gate in gates:
        for gop in state.all_in_round(gate, found.handoff.round):
            if not gop.is_cancelled and state.gate_decision(gop) is not None:
                return False
    return True


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _State:
    """The most recent state the hub has recorded for a hand-off: the placed project
    (``ref`` ``placed``) or its newest return (``return:<event id>``), and its content hash."""

    ref: str
    sha256: str

    @property
    def placed(self) -> bool:
        return self.ref == "placed"


def _recorded_state(log: EventLog, project_id: str, handoff: Handoff, op_node: str) -> _State:
    """The newest recorded state of the hand-off ``op_node`` (see :class:`_State`)."""
    from aar.recipes.desktop_returns import EVENT_RETURN

    newest: Event | None = None
    for event in _runner_events(log, [EVENT_RETURN], project_id):
        payload = event.payload
        if (
            payload.get("source") == "library_save"
            and payload.get("handoff") == op_node
            and payload.get("recipe_node") == handoff.recipe_node
        ):
            newest = event
    if newest is None:
        return _State("placed", handoff.placed_sha256)
    # The saved file's artifact id is its SHA-256 (returns before S25e carry only that).
    sha = newest.payload.get("snapshot_sha256") or newest.payload.get("project_db")
    if not isinstance(sha, str) or not _SHA256.match(sha):
        raise ValueError(f"the recorded return {newest.id} names no saved file's hash")
    return _State(f"return:{newest.id}", sha)


def _returns_per_handoff(log: EventLog) -> dict[str, int]:
    from aar.recipes.desktop_returns import EVENT_RETURN

    counts: dict[str, int] = {}
    for event in _runner_events(log, [EVENT_RETURN]):
        handoff = event.payload.get("handoff")
        if event.payload.get("source") == "library_save" and isinstance(handoff, str):
            counts[handoff] = counts.get(handoff, 0) + 1
    return counts


def library_conditions(log: EventLog, project_id: str, recipe_node: str) -> list[dict[str, Any]]:
    """The newest 20 pickup conditions of a run (the status view shows them)."""
    found: list[dict[str, Any]] = []
    for event in _runner_events(log, [EVENT_PICKUP], project_id):
        if event.payload.get("recipe_node") == recipe_node:
            found.append(event.payload)
    return found[-20:]


# --------------------------------------------------------------------------- snapshots


def snapshot_folder() -> Path:
    from aar.recipes.ccn.library_handoff import snapshot_dir

    return snapshot_dir()


def sweep_snapshots(folder: Path, *, stale_seconds: float, now: float) -> tuple[int, list[str]]:
    """Remove the snapshots a killed worker left (module docstring); ``(removed, left)``, the
    second naming at most 10 entries that are not ours or could not be removed."""
    removed, left = 0, []
    with os.scandir(folder) as entries:
        for n, entry in enumerate(entries):
            if n >= 10_000:
                left.append("... the snapshot folder holds more than 10000 entries")
                break
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            ours = entry.name.startswith(SNAPSHOT_PREFIXES) and (
                stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)
            )
            if not ours:
                if len(left) < 10:
                    left.append(entry.name)
                continue
            if now - st.st_mtime < stale_seconds:
                continue
            try:
                os.unlink(entry.path)
                removed += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                if len(left) < 10:
                    left.append(f"{entry.name} ({exc.strerror or exc})")
    return removed, left


# --------------------------------------------------------------------------- one project


def _open_library(
    handoff: Handoff, root: str | None, real: str, stack: contextlib.ExitStack[Any]
) -> int:
    """A descriptor of the library folder: its resolved path ``real`` (checked inside the media
    root) opened from the media root's descriptor one component at a time without following a
    link, so a link swapped in after the check is refused rather than followed. Without a
    media root there is no boundary to keep: ``real`` itself is opened."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    if root is None:
        start, below = real, []
    else:
        start = root
        below = [part for part in real[len(root.rstrip("/")) :].split("/") if part]
    try:
        fd = os.open(start, os.O_RDONLY | os.O_DIRECTORY)
    except FileNotFoundError:
        raise _Condition(
            "library_missing",
            f"the library folder {handoff.library!r} is not there",
            "reconnect or restore the library folder on the hub's share",
            cause="not_there",
        ) from None
    except OSError as exc:
        raise _Condition(
            "library_missing",
            f"the library folder {handoff.library!r} cannot be opened ({exc.strerror})",
            "check the library folder on the hub's share and its permissions",
            cause="cannot_open",
        ) from None
    stack.callback(os.close, fd)
    for part in below:
        try:
            fd = os.open(part, flags, dir_fd=fd)
        except OSError as exc:
            try:
                is_link = stat.S_ISLNK(os.stat(part, dir_fd=fd, follow_symlinks=False).st_mode)
            except OSError:
                is_link = False
            if exc.errno == errno.ELOOP or is_link:
                raise _Condition(
                    "path_refused",
                    f"the library folder {handoff.library!r} changed while it was opened "
                    f"({part!r} on its way is now a link); it is not followed",
                    "put the library folder back as it was, inside the media root",
                    cause="library_relinked",
                ) from None
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                raise _Condition(
                    "library_missing",
                    f"the library folder {handoff.library!r} is not there",
                    "reconnect or restore the library folder on the hub's share",
                    cause="not_there",
                ) from None
            raise _Condition(
                "library_missing",
                f"the library folder {handoff.library!r} cannot be opened ({exc.strerror})",
                "check the library folder on the hub's share and its permissions",
                cause="cannot_open",
            ) from None
        stack.callback(os.close, fd)
    return fd


@contextlib.contextmanager
def _project_dir(handoff: Handoff) -> Iterator[int]:
    """A descriptor of the hand-off's project folder, reached without following a link from
    the media root down (:func:`_open_library`, then the recorded parts); a
    :class:`_Condition` when it cannot be reached."""
    from aar.recipes.ccn.params import media_root

    nofollow, directory = getattr(os, "O_NOFOLLOW", None), getattr(os, "O_DIRECTORY", None)
    if os.name != "posix" or nofollow is None or directory is None:
        raise _Condition(
            "unsupported_platform",
            "library pickup needs POSIX no-follow folder descriptors (the hub is Linux)",
            "run the hub's worker on Linux",
        )
    root = media_root()
    real = os.path.realpath(handoff.library)
    if root is not None and not (real == root or real.startswith(root.rstrip("/") + "/")):
        raise _Condition(
            "path_outside_media_root",
            f"the library folder {handoff.library!r} now leads outside the media root",
            "put the library folder back inside the media root",
        )
    flags = os.O_RDONLY | nofollow | directory
    with contextlib.ExitStack() as stack:
        fd = _open_library(handoff, root, real, stack)
        for n, part in enumerate(handoff.parts):
            last = n == len(handoff.parts) - 1
            if part in ("", ".", "..") or "/" in part or "\x00" in part:
                raise _Condition(
                    "path_refused",
                    f"the recorded path holds {part!r}",
                    "hand the cut over again",
                    cause="bad_part",
                )
            try:
                fd = os.open(part, flags, dir_fd=fd)
            except OSError as exc:
                try:
                    is_link = stat.S_ISLNK(os.stat(part, dir_fd=fd, follow_symlinks=False).st_mode)
                except OSError:
                    is_link = False
                if exc.errno == errno.ELOOP or is_link:
                    raise _Condition(
                        "path_refused",
                        f"{'/'.join(handoff.parts[: n + 1])!r} below the library is a link; "
                        "links below a library are never followed",
                        "replace the link with the folder itself",
                        cause=f"link_below_library:{n}",
                    ) from None
                if last and exc.errno in (errno.ENOENT, errno.ENOTDIR):
                    raise _Condition(
                        "project_missing",
                        f"the project folder {handoff.folder!r} is not in the library any "
                        "more (deleted or renamed)",
                        "restore the project folder under its name, or hand the cut over again",
                        cause="folder_gone",
                    ) from None
                raise _Condition(
                    "not_a_library",
                    f"{'/'.join(handoff.parts[: n + 1])!r} cannot be opened in the library "
                    f"({exc.strerror})",
                    "check the library's Users and Projects folders",
                    cause=f"cannot_open:{n}",
                ) from None
            stack.callback(os.close, fd)
        yield fd


def _side_file(folder_fd: int) -> str | None:
    for name in ("Project.db-journal", "Project.db-wal"):
        try:
            os.stat(name, dir_fd=folder_fd, follow_symlinks=False)
            return name
        except FileNotFoundError:
            continue
    return None


def _busy(what: str, cause: str) -> _Condition:
    return _Condition(
        "busy",
        f"Resolve is saving the project ({what})",
        "nothing to do: the next sweep looks again once the save has finished",
        cause=cause,
    )


@dataclass
class _Snapshot:
    path: Path
    sha256: str
    size: int


#: What this worker process saw of each hand-off's file while it had not settled yet: its
#: identity and when it was first seen so (``clock`` time). A file whose identity stays the same
#: for ``settle_seconds`` across sweeps counts as settled even when its modification time runs
#: ahead of the hub's clock (a share's clock), as the library reader decides it.
_WATCHED: dict[str, tuple[tuple[int, int, int, int], float]] = {}
#: How many sweeps in a row this worker process found each hand-off's file not settled.
_UNSETTLED: dict[str, int] = {}


def _not_settled(op_node: str, limits: Limits) -> None:
    """Count one more sweep that found the file not settled; ``busy`` once that reaches
    :attr:`Limits.busy_after` (a file rewritten all the time; module docstring)."""
    if len(_UNSETTLED) > 10_000:
        _UNSETTLED.clear()
    count = _UNSETTLED.get(op_node, 0) + 1
    _UNSETTLED[op_node] = count
    if count >= limits.busy_after:
        raise _busy(
            f"its file has not stopped changing in {count} sweeps in a row", "never_settles"
        )


def _settled(op_node: str, st: os.stat_result, limits: Limits, now: float, clock: float) -> Any:
    """``(ready, settled identity or None)``: not ready while the file changed less than
    ``settle_seconds`` ago and has not been seen unchanged for that long."""
    from aar.media.settle import settle_identity

    identity = settle_identity(st)
    if now - st.st_mtime >= limits.settle_seconds:
        _WATCHED.pop(op_node, None)
        return True, None
    seen = _WATCHED.get(op_node)
    if seen is None or seen[0] != identity:
        if len(_WATCHED) > 10_000:
            _WATCHED.clear()
        _WATCHED[op_node] = (identity, clock)
        return False, None
    if clock - seen[1] >= limits.settle_seconds:
        return True, identity
    return False, None


def _snapshot(
    handoff: Handoff,
    op_node: str,
    folder_fd: int,
    target_dir: Path,
    limits: Limits,
    sweep: Sweep,
    times: tuple[float, float],
) -> _Snapshot | None:
    """A settled private copy of the project's ``Project.db``; None when it does not fit in
    the bytes the sweep has left, or has not settled yet (a save a moment ago: the next sweep
    looks again, with nothing reported, until :attr:`Limits.busy_after` sweeps in a row). The
    bytes copied count against the sweep's budget whether the copy is taken or refused. The
    caller removes it."""
    try:
        st = os.stat("Project.db", dir_fd=folder_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise _Condition(
            "project_missing",
            f"the project folder {handoff.folder!r} holds no Project.db",
            "restore the project, or hand the cut over again",
            cause="no_project_db",
        ) from None
    if not stat.S_ISREG(st.st_mode):
        raise _Condition(
            "path_refused",
            f"the Project.db of {handoff.folder!r} is not a regular file",
            "a project's Project.db must be the file itself, not a link",
            cause="db_not_regular",
        )
    side = _side_file(folder_fd)
    if side is not None:
        raise _busy(f"{side} is beside it", "side_file")
    if st.st_size > limits.db_bytes:
        raise _Condition(
            "too_large",
            f"the saved project is {st.st_size} bytes, over the pickup's {limits.db_bytes}",
            "check the project; a cut's project is far smaller",
            cause="over_bound",
        )
    if st.st_size > limits.bytes - sweep.bytes:
        return None
    ready, quiet = _settled(op_node, st, limits, *times)
    if not ready:
        _not_settled(op_node, limits)
        return None
    _UNSETTLED.pop(op_node, None)
    name = f"{SNAPSHOT_PREFIXES[0]}{os.getpid()}-{secrets.token_hex(6)}.db"
    path = target_dir / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as target, _charged(target, sweep, st.st_size):
            try:
                copied = copy_settled(
                    Path("Project.db"),
                    target,
                    max_bytes=limits.db_bytes,
                    settle_seconds=limits.settle_seconds,
                    what="the saved project",
                    dir_fd=folder_fd,
                    settled_identity=quiet,
                    now=lambda: times[0],
                )
            except TooLargeError as exc:
                raise _Condition(
                    "too_large",
                    str(exc),
                    "check the project; a cut's project is far smaller",
                    cause="grew_past_bound",
                ) from None
            except ValueError as exc:  # a link or not a regular file (ReceiptError)
                raise _Condition(
                    "path_refused", str(exc), "check the project folder", cause="copy_refused"
                ) from None
        if copied is None:
            raise _busy("it changed while it was copied", "changed_while_copied")
        side = _side_file(folder_fd)
        if side is not None:
            raise _busy(f"{side} appeared while it was copied", "side_file_after_copy")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return _Snapshot(path, copied.sha256, copied.size)


@contextlib.contextmanager
def _charged(target: Any, sweep: Sweep, size: int) -> Iterator[None]:
    """Count what was written to ``target`` against the sweep's bytes, on every path (a copy
    refused after it was made has cost its bytes too); the file's ``size`` when the position
    cannot be told."""
    try:
        yield
    finally:
        try:
            spent = target.tell()
        except (OSError, ValueError):
            spent = size
        sweep.bytes += spent


def _read(snapshot: _Snapshot, handoff: Handoff, target_dir: Path, stop_at: float) -> Any:
    from aar.drp.library import LibraryOptions
    from aar.plan.timeline_import import TimelineImportError, read_timeline

    options = LibraryOptions(
        settle_seconds=0.0,
        deadline_seconds=0.0,
        snapshot_dir=target_dir,
        stop_at=stop_at,
    )
    try:
        return read_timeline(
            snapshot.path, handoff.timeline, library_options=options, project_id=handoff.project_id
        )
    except TimelineImportError as exc:
        sha = snapshot.sha256
        if exc.code == "project_mismatch":
            raise _Condition(
                "project_replaced",
                f"the folder {handoff.folder!r} now holds another project "
                f"({exc.details.get('found')!r}, not the handed-over {handoff.project_id!r}); "
                "it is not imported",
                "open the handed-over project; a project saved under its folder's name by "
                "another project is never taken for it",
                sha,
            ) from None
        if exc.code in ("timeline_not_found", "no_timeline", "several_timelines"):
            raise _Condition(
                "timeline_missing",
                f"the saved project has no single timeline named {handoff.timeline!r} (deleted "
                f"or renamed?): {exc.message}",
                f"give the timeline its name {handoff.timeline!r} again and save",
                sha,
            ) from None
        if exc.code == "library_busy":
            raise _busy("the reader found it being saved", "reader_busy") from None
        library_code = exc.details.get("library_code")
        if library_code == "too_large":
            raise _Condition("too_large", exc.message, exc.fix_hint, sha) from None
        raise _Condition(
            "damaged", f"{exc.message} [{exc.code}]", "save the project again in Resolve", sha
        ) from None


def _key_condition(
    project_id: str, op_node: str, generation: int, code: str, sha: str | None, cause: str = ""
) -> str:
    if code in CONTENT_CONDITIONS and sha is not None:
        return f"library-pickup:{project_id}:{op_node}:{code}:{sha}"
    return f"library-pickup:{project_id}:{op_node}:{generation}:{code}:{cause}"


def _report(log: EventLog, found: _Found, generation: int, cond: _Condition, sweep: Sweep) -> None:
    """Record ``cond`` once (module docstring); never raises past the sweep."""
    from aar.recipes.state import load_state

    project_id = found.event.project_id
    assert project_id is not None
    h = found.handoff
    key = _key_condition(project_id, found.op_node, generation, cond.code, cond.sha256, cond.cause)
    sweep.conditions.append(cond.code)
    if log.get_by_idempotency_key(key) is not None:
        return
    payload: dict[str, Any] = {
        "recipe_node": h.recipe_node,
        "round": h.round,
        "handoff": found.op_node,
        "library": h.library,
        "folder": h.folder,
        "project_id": h.project_id,
        "timeline": h.timeline,
        "condition": cond.code,
        "cause": cond.cause[:200],
        "message": cond.message[:2000],
        "fix_hint": cond.hint[:1000],
    }
    if cond.sha256 is not None:
        payload["snapshot_sha256"] = cond.sha256
    with log.transaction() as txn:
        if load_state(log, h.recipe_node, project_id=project_id).recipe_status == "cancelled":
            return
        if log.get_by_idempotency_key(key) is not None:
            return
        txn.append(EVENT_PICKUP, RUNNER, payload, project_id=project_id, idempotency_key=key)


def _seen(log: EventLog, project_id: str, op_node: str, sha: str, return_key: str) -> bool:
    if log.get_by_idempotency_key(return_key) is not None:
        return True
    return any(
        log.get_by_idempotency_key(_key_condition(project_id, op_node, 0, code, sha)) is not None
        for code in CONTENT_CONDITIONS
    )


def _check_one(
    log: EventLog,
    store: ArtifactStore,
    actor: Actor,
    found: _Found,
    generation: int,
    target_dir: Path,
    limits: Limits,
    sweep: Sweep,
    stop_at: float,
    times: tuple[float, float],
) -> None:
    from aar.recipes.desktop_returns import record_import

    h = found.handoff
    project_id = found.event.project_id
    assert project_id is not None
    snapshot: _Snapshot | None = None
    try:
        with _project_dir(h) as folder_fd:
            snapshot = _snapshot(
                h,
                found.op_node,
                folder_fd,
                target_dir / PICKUP_AREA,
                limits,
                sweep,
                times,
            )
        if snapshot is None:
            sweep.deferred += 1
            return
        state = _recorded_state(log, project_id, h, found.op_node)
        if snapshot.sha256 == state.sha256:
            if state.placed:
                sweep.unchanged += 1
            else:
                sweep.repeated += 1
            return
        # The latest save always wins: the key names the state this save follows, so the same
        # content saved again after another return is a new return, and the same save seen
        # twice is one.
        key = (
            f"library-return:{project_id}:{h.recipe_node}:{h.round}:{h.plan_sha256}:"
            f"{found.op_node}:{state.ref}:{snapshot.sha256}"
        )
        if _seen(log, project_id, found.op_node, snapshot.sha256, key):
            sweep.repeated += 1
            return
        timeline = _read(snapshot, h, target_dir, stop_at)
        saved = store.put(
            snapshot.path,
            actor=actor,
            project_id=project_id,
            meta={"role": "returned-library-project"},
        )
        result: dict[str, Any] = {
            "recipe_node": h.recipe_node,
            "round": h.round,
            "base_plan": h.plan_sha256,
            "source": "library_save",
            "handoff": found.op_node,
            "library": h.library,
            "project_folder": h.folder,
            "project_id": h.project_id,
            "timeline": h.timeline,
            "project_db": saved,
            "snapshot_sha256": snapshot.sha256,
            "follows": state.ref,
            "file": "Project.db",
            "status": "needs_review",
            "partial_reconstruction": True,
        }
        stored = record_import(
            log,
            store,
            project_id,
            recipe_node=h.recipe_node,
            plan_sha256=h.plan_sha256,
            mapping=media_map_from_mapping(h.media_map),
            read=lambda: timeline,
            result=result,
            key=key,
            actor=actor,
            current=lambda: _recorded_state(log, project_id, h, found.op_node).ref == state.ref,
        )
        if stored is result:
            sweep.returned += 1
        elif stored is not None:  # another worker recorded this save first: not twice
            sweep.repeated += 1
        else:  # the run was cancelled, or another save was recorded meanwhile: look again
            sweep.deferred += 1
    except _Condition as cond:
        _report(log, found, generation, cond, sweep)
    except (OSError, ValueError, ExportError) as exc:  # the hub's own disk or a bad record
        _report(
            log,
            found,
            generation,
            _Condition(
                "pickup_failed",
                f"{type(exc).__name__}: {exc}"[:2000],
                "check the hub's work folder and artifact store",
                cause=type(exc).__name__,
            ),
            sweep,
        )
    finally:
        if snapshot is not None:
            snapshot.path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- the sweep


def scan_library_returns(
    log: EventLog,
    store: ArtifactStore,
    actor: Actor,
    *,
    limits: Limits = Limits(),
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], float] = time.time,
) -> Sweep:
    """One bounded sweep of every open library hand-off (module docstring)."""
    sweep = Sweep()
    if not os.environ.get("AAR_WORK_DIR", "").strip():
        return sweep
    stop_at = clock() + limits.seconds
    target_dir = snapshot_folder()
    (target_dir / PICKUP_AREA).mkdir(mode=0o700, exist_ok=True)
    for folder in (target_dir, target_dir / PICKUP_AREA):
        removed, left = sweep_snapshots(folder, stale_seconds=limits.stale_seconds, now=now())
        sweep.snapshots_removed += removed
        sweep.snapshot_leftovers += [n for n in left if n != PICKUP_AREA]
    found = _handoffs(log, sweep)
    if not found:
        return sweep
    returns = _returns_per_handoff(log)
    turn = int(now() // 60) % len(found)
    for item in found[turn:] + found[:turn]:
        if clock() >= stop_at or sweep.checked >= limits.projects:
            sweep.cut_short = True
            break
        if not _review_open(log, item):
            sweep.closed += 1
            continue
        sweep.checked += 1
        _check_one(
            log,
            store,
            actor,
            item,
            returns.get(item.op_node, 0),
            target_dir,
            limits,
            sweep,
            stop_at,
            (now(), clock()),
        )
    return sweep
