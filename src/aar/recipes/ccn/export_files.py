"""Copying the export step's files to the edit PC's share (``AAR_EXPORT_DIR``).

The folder is ``<AAR_EXPORT_DIR>/<project>/<timeline>/`` and the files
``<timeline>.r<round>.review.mp4``, ``<timeline>.r<round>.notes.txt`` and
``<timeline>.r<round>.fcpxml``, every name made safe for Windows (:func:`safe_name`). The
project and timeline come from the stub step's runner-recorded outputs (the brief's show and the
params' timeline), never from the agent's plan.

Delivery is the step's purpose, so ``AAR_EXPORT_DIR`` must be set (``export_dir_unset``): an
export that reached nobody is not a success a review can follow.

Nothing is followed or replaced: a folder below ``AAR_EXPORT_DIR`` that is a symlink is refused
(``export_path_symlink``), and a file name that is taken (a file with other bytes, a symlink,
even a dangling one, or anything that is not a regular file) moves the set to ``-2``, ``-3``...
(at most :data:`MAX_COPY_SUFFIX`). A name that already holds the same bytes is reused, so a
repeated export copies nothing twice.

A round is published whole or not at all (:func:`_publish`): every file is first written to a
temporary name beside its place and fsynced (the slow part, where a full share or a dropped
mount fails), then each is placed without ever replacing a file, in the caller's order. The
caller puts the FCPXML last, so a timeline that can be imported is in the share only when the
review video and the notes it belongs with are already there, even after a crash between two
placements; a failure also removes what this call had placed.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import shutil
import stat
from pathlib import Path
from typing import Any

from aar.host.transport import sha256_file
from aar.recipes.context import StepContext, StepFailure

EXPORT_DIR_ENV = "AAR_EXPORT_DIR"
MAX_COPY_SUFFIX = 50
MAX_NAME = 100
_UNSAFE = re.compile(r"[^A-Za-z0-9 ._-]+")
#: Windows device names (also with an extension, and with spaces before the dot: ``con .txt``).
_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$", "clock$"}
    | {f"com{c}" for c in "0123456789"}
    | {f"lpt{c}" for c in "0123456789"}
)


def safe_name(text: str) -> str:
    """``text`` as one file or folder name both Linux and Windows accept: only letters, digits,
    space, ``.``, ``_`` and ``-``; no leading or trailing dot or space; at most :data:`MAX_NAME`
    characters; never ``.``/``..`` or a device name (``CON``, ``con .txt``, ``LPT1 ``)."""
    name = _UNSAFE.sub("_", text).strip(" .")[:MAX_NAME].strip(" .")
    if not name:
        return "untitled"
    stem = name.split(".", 1)[0].rstrip(" ").lower()
    if stem in _RESERVED:
        name = f"_{name}"[:MAX_NAME]
    return name


def export_root() -> Path:
    """``AAR_EXPORT_DIR``: an existing absolute directory. Unset or empty is
    ``export_dir_unset``, never a skipped copy."""
    raw = os.environ.get(EXPORT_DIR_ENV, "").strip()
    if not raw:
        raise StepFailure(
            "export_dir_unset",
            f"{EXPORT_DIR_ENV} is not set, so the timeline cannot be delivered to the edit PC",
            f"set {EXPORT_DIR_ENV} in the worker's environment (hub.env: the media root's "
            "share/exports), restart the worker, then retry the export (recipe_retry export)",
        )
    root = Path(raw)
    if not root.is_absolute() or not root.is_dir():
        raise StepFailure(
            "export_dir_unusable",
            f"{EXPORT_DIR_ENV} {raw!r} is not an existing absolute directory",
            "create it (the media root's share/exports), then retry the export",
        )
    return root


def _folder(root: Path, parts: list[str]) -> Path:
    """``root/parts...``, made one level at a time, refusing a symlink (or any non-directory)
    at every level instead of following it."""
    folder = root
    for part in parts:
        folder = folder / part
        with contextlib.suppress(FileExistsError):
            os.mkdir(folder)
        mode = os.lstat(folder).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise StepFailure(
                "export_path_symlink",
                f"{str(folder)!r} in the export share is a symlink or not a folder; it is "
                "never followed",
                "remove or rename it in the share",
            )
    return folder


def _holds(path: Path, sha256: str) -> bool | None:
    """True when ``path`` is a regular file (not a symlink) with these bytes, False when the
    name is taken by anything else, None when it is free."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        return False
    return sha256_file(str(path)).lower() == sha256


def _targets(folder: Path, stem: str, digests: list[tuple[str, str]]) -> list[Path] | None:
    for n in range(1, MAX_COPY_SUFFIX + 1):
        name = stem if n == 1 else f"{stem}-{n}"
        paths = [folder / f"{name}{suffix}" for suffix, _ in digests]
        if all(_holds(p, sha) is not False for p, (_, sha) in zip(paths, digests, strict=True)):
            return paths
    return None


def _stage(ctx: StepContext, tmp: Path, artifact_id: str) -> None:
    with ctx.store.open(artifact_id) as src, tmp.open("xb") as sink:
        shutil.copyfileobj(src, sink, 1024 * 1024)
        sink.flush()
        os.fsync(sink.fileno())
    tmp.chmod(0o644)


def _publish(ctx: StepContext, pairs: list[tuple[Path, str]]) -> None:
    """Place every ``(destination, artifact id)`` or none (module docstring): all staged under
    temporary names first, then placed in order, the ones this call placed removed again when a
    later one fails. A destination that already holds the bytes is left as it is."""
    from aar.drp.modify.write import _place

    todo = [(dest, artifact_id) for dest, artifact_id in pairs if not _holds(dest, artifact_id)]
    staged = [
        (dest.with_name(f".{dest.name}.aar-copy-{secrets.token_hex(4)}"), dest, artifact_id)
        for dest, artifact_id in todo
    ]
    placed: list[Path] = []
    try:
        for tmp, _dest, artifact_id in staged:
            _stage(ctx, tmp, artifact_id)
        for tmp, dest, _artifact_id in staged:
            _place(tmp, dest, False)  # FileExistsError when the name was taken meanwhile
            placed.append(dest)
    except BaseException:
        # Best effort: on a share that went away these fail too, and the first error is the
        # one to report. What is left then is never the importable file without its set
        # (it is placed last).
        for dest in reversed(placed):
            with contextlib.suppress(OSError):
                dest.unlink()
        raise
    finally:
        for tmp, _dest, _artifact_id in staged:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


def copy_out(
    ctx: StepContext, project: str, timeline: str, round_: int, files: list[tuple[str, str]]
) -> dict[str, Any]:
    """Copy ``files`` (``(suffix, artifact id)``) as ``<timeline>.r<round><suffix>``, all or
    none, placed in the order given (module docstring); the outputs' ``copy`` block."""
    root = export_root()
    stem = f"{safe_name(timeline)}.r{round_}"
    try:
        folder = _folder(root, [safe_name(project), safe_name(timeline)])
        targets = _targets(folder, stem, files)
        if targets is None:
            raise StepFailure(
                "export_names_taken",
                f"{folder} already holds {MAX_COPY_SUFFIX} other copies named {stem}",
                "move old exports away",
            )
        _publish(ctx, [(dest, sha) for dest, (_, sha) in zip(targets, files, strict=True)])
    except OSError as exc:
        raise StepFailure(
            "export_copy_failed", f"cannot copy into the export share: {exc}", "check the share"
        ) from None
    return {
        "copied": True,
        "dir": str(folder),
        "files": [str(t) for t in targets],
        "round": round_,
    }


__all__ = ["EXPORT_DIR_ENV", "MAX_COPY_SUFFIX", "copy_out", "export_root", "safe_name"]
