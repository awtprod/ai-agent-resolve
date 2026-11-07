"""What every analysis worker shares: recording time, the JSONL checkpoint and atomic writes.

**Recording time** is file time plus a ``RecordingOffset``: ``recording_origin / fps`` of a plan
source (exact rational fps), an explicit ``--offset-s``, or 0 with a note saying so.

**Checkpoint.** A worker appends one JSON record per unit of work (a transcript segment, a
sampled video frame) to ``<out>.ckpt.jsonl``, flushing and fsyncing each line. The first line is
a header: the checkpoint kind, the input's size and mtime_ns, and the sha256 of the settings
that change the records. A resumed run refuses a header that does not match
(``checkpoint-mismatch``) and a line that is not a record of its kind (``checkpoint-corrupt``);
a trailing line without its newline (a run killed mid-write) is not complete and is ignored.
The output is written to a temporary file beside it and moved into place; the worker removes
the checkpoint only after that.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from aar.analyze.errors import AnalyzeError
from aar.plan.model import EditPlan

CHECKPOINT_SUFFIX = ".ckpt.jsonl"
CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------- recording time


@dataclass(frozen=True)
class RecordingOffset:
    """Seconds added to file time to get recording time, and where they came from."""

    offset_s: float = 0.0
    source: Literal["none", "offset-s", "plan"] = "none"
    recording: str | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"offset_s": self.offset_s, "from": self.source}
        if self.detail:
            out.update(self.detail)
        if self.source == "none":
            out["note"] = "no plan source or --offset-s was given: times are file seconds"
        return out


def offset_from_plan(plan: EditPlan, source_id: str) -> RecordingOffset:
    """``recording_origin / fps`` of a plan source, with the exact rational fps."""
    source = plan.source_map().get(source_id)
    if source is None:
        known = ", ".join(repr(s.id) for s in plan.sources) or "none"
        raise AnalyzeError(
            "invalid-argument", f"the plan has no source {source_id!r} (sources: {known})"
        )
    fps = Fraction(plan.fps.num, plan.fps.den)
    offset = float(Fraction(source.recording_origin) / fps)
    return RecordingOffset(
        offset_s=offset,
        source="plan",
        recording=source.recording_id if source.recording_id is not None else source.id,
        detail={
            "source": source.id,
            "recording_origin": source.recording_origin,
            "fps": f"{plan.fps.num}/{plan.fps.den}",
        },
    )


def explicit_offset(offset_s: float) -> RecordingOffset:
    if not math.isfinite(offset_s):
        raise AnalyzeError("invalid-argument", "--offset-s must be a finite number of seconds")
    return RecordingOffset(offset_s=float(offset_s), source="offset-s")


# ---------------------------------------------------------------- files


def is_number(value: Any) -> bool:
    """An int or float, not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - not every platform opens directories
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file beside ``path``, fsync it and move it into place."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


def input_identity(path: Path) -> dict[str, int]:
    """The input's size and mtime_ns: what a checkpoint and an output header record."""
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


# ---------------------------------------------------------------- checkpoint


def checkpoint_path(out: Path) -> Path:
    return out.with_name(out.name + CHECKPOINT_SUFFIX)


def checkpoint_header(kind: str, media: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """The first line: ``kind``, the input's identity and a hash of ``settings``."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "aar_checkpoint": CHECKPOINT_VERSION,
        "kind": kind,
        "input": input_identity(media),
        "settings_sha256": digest,
        "settings": settings,
    }


#: Checks one record against the one before it (None for the first): a reason, or None if fine.
RecordCheck = Callable[[Any, Any], "str | None"]


def read_checkpoint(
    path: Path, *, kind: str, what: str, check: RecordCheck
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """(header, complete records, byte length of the complete lines).

    ``check(record, previous)`` validates each record (``previous`` is None for the first) and
    returns why it is not usable, or None.
    """
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    lines = complete.splitlines()

    def corrupt(why: str) -> AnalyzeError:
        return AnalyzeError(
            "checkpoint-corrupt",
            f"checkpoint {path} is not usable: {why}",
            hint=f"delete {path} to start over",
        )

    if not lines:
        raise corrupt("it has no complete header line")
    try:
        header = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:]]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise corrupt(f"a complete line is not JSON ({exc})") from None
    if not (isinstance(header, dict) and header.get("kind") == kind):
        raise corrupt(f"its first line is not an aar {what} checkpoint header")
    previous = None
    for i, record in enumerate(records, start=1):
        why = check(record, previous)
        if why is not None:
            raise corrupt(f"{what} line {i} {why}")
        previous = record
    return header, records, len(complete)


def check_header(path: Path, found: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse (``checkpoint-mismatch``) a checkpoint written for another input or settings."""
    differs = [
        key
        for key in ("aar_checkpoint", "input", "settings_sha256")
        if found.get(key) != expected[key]
    ]
    if differs:
        what = {
            "aar_checkpoint": "checkpoint format",
            "input": "input file (size or mtime)",
            "settings_sha256": "model or decode settings",
        }
        raise AnalyzeError(
            "checkpoint-mismatch",
            f"checkpoint {path} was written for a different "
            + " and ".join(what[k] for k in differs)
            + "; refusing to splice two runs",
            hint=f"re-run with the original input and settings, or delete {path} to start over",
        )


def append_record(handle: Any, record: dict[str, Any]) -> None:
    """One JSON line, flushed and fsynced."""
    handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


def start_checkpoint(path: Path, header: dict[str, Any]) -> None:
    with open(path, "wb") as fh:
        append_record(fh, header)
    fsync_dir(path.parent)


def trim_checkpoint(path: Path, good_bytes: int) -> None:
    """Drop a half-written last line before appending to a resumed checkpoint."""
    with open(path, "r+b") as fh:
        fh.truncate(good_bytes)


def finish_checkpoint(path: Path) -> None:
    """Remove the checkpoint once the output is safely in place."""
    path.unlink()
    fsync_dir(path.parent)
