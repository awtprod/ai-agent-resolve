"""Copying the export step's files to the edit PC's share (``AAR_EXPORT_DIR``).

The folder is ``<AAR_EXPORT_DIR>/<project>/<timeline>/`` and the files
``<timeline>.r<round>.fcpxml`` and ``<timeline>.r<round>.review.mp4``, every name made safe for
Windows (:func:`safe_name`). The project and timeline come from the stub step's runner-recorded
outputs (the brief's show and the params' timeline), never from the agent's plan.

Nothing is followed or replaced: a folder below ``AAR_EXPORT_DIR`` that is a symlink is refused
(``export_path_symlink``), and a file name that is taken (a file with other bytes, a symlink,
even a dangling one, or anything that is not a regular file) moves the pair to ``-2``, ``-3``...
(at most :data:`MAX_COPY_SUFFIX`). A name that already holds the same bytes is reused, so a
repeated export copies nothing twice. Each file is written to a temporary name beside it,
fsynced, and placed without ever replacing a file.
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


def export_root() -> Path | None:
    """``AAR_EXPORT_DIR`` when set (an existing absolute directory), else None."""
    raw = os.environ.get(EXPORT_DIR_ENV, "").strip()
    if not raw:
        return None
    root = Path(raw)
    if not root.is_absolute() or not root.is_dir():
        raise StepFailure(
            "export_dir_unusable",
            f"{EXPORT_DIR_ENV} {raw!r} is not an existing absolute directory",
            "create it (the media root's share/exports) or unset it",
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


def _place_copy(ctx: StepContext, dest: Path, artifact_id: str) -> None:
    from aar.drp.modify.write import _place

    if _holds(dest, artifact_id):
        return
    tmp = dest.with_name(f".{dest.name}.aar-copy-{secrets.token_hex(4)}")
    try:
        with ctx.store.open(artifact_id) as src, tmp.open("xb") as sink:
            shutil.copyfileobj(src, sink, 1024 * 1024)
            sink.flush()
            os.fsync(sink.fileno())
        tmp.chmod(0o644)
        _place(tmp, dest, False)  # FileExistsError when the name was taken meanwhile
    finally:
        tmp.unlink(missing_ok=True)


def copy_out(
    ctx: StepContext, project: str, timeline: str, round_: int, files: list[tuple[str, str]]
) -> dict[str, Any]:
    """Copy ``files`` (``(suffix, artifact id)``) as ``<timeline>.r<round><suffix>`` (module
    docstring); the outputs' ``copy`` block."""
    root = export_root()
    if root is None:
        return {"copied": False, "reason": f"{EXPORT_DIR_ENV} is not set: nothing was copied"}
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
        for dest, (_, artifact_id) in zip(targets, files, strict=True):
            _place_copy(ctx, dest, artifact_id)
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
