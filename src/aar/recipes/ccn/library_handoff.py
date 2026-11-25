"""The export step's second route (S25d): the cut as a project in the shared Resolve library.

When the run's params name a library (:class:`~aar.recipes.ccn.params.LibraryParam`), the export
step also writes the cut through :func:`aar.plan.export.library.export_library` (the one
implementation of ``aar plan export --format library``): a copy of the owner's seed project,
the cut's media authored from facts the edit PC measured, self-checked and placed under a new
project folder named for the show, the timeline and the round. The FCPXML hand-off is published
exactly as before; this is an addition until the owner has confirmed library projects in
Resolve, so nothing here may fail the step: every refusal comes back as a typed record
(``status: "refused"`` with the export's code, message and hint) that the notes and the step's
result carry as a warning.

What was placed is recorded in the step's runner-written result (``library_handoff``): the
library, the user, the project folder's path below the library, the project id, the timeline,
the plan artifact it was written from and the SHA-256 of the placed ``Project.db``. The hub's
pickup (:mod:`aar.recipes.library_returns`) finds a handed-over project by that record only:
the project id and the hash, never the folder's name alone.

Paths are checked twice: by their text in the params (absolute, clean, under the media root)
and here, when they are used, by where they lead (``realpath`` inside the media root's
``realpath``): a link swapped in later is a refusal, not a write elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aar.drp.library import LibraryOptions
from aar.drp.mediapool.errors import MediaPoolError
from aar.plan.export.errors import ExportError
from aar.plan.export.media import media_map_from_mapping
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.model import EditPlan
from aar.plan.profile import Profile
from aar.recipes.ccn.export_files import safe_name
from aar.recipes.ccn.params import LibraryParam, media_root

__all__ = [
    "MAX_NAME_TRIES",
    "SNAPSHOT_AREA",
    "hand_over",
    "project_name",
    "snapshot_dir",
]

#: Names tried when the project folder's name is taken (``... r1``, ``... r1-2``, ...).
MAX_NAME_TRIES = 20
#: The folder under ``AAR_WORK_DIR`` that holds every library read's private snapshot on the
#: hub (never the system temporary folder, which may be memory-backed); the pickup sweeps it.
SNAPSHOT_AREA = "aar-library-snapshots"


def snapshot_dir() -> Path:
    """``<AAR_WORK_DIR>/aar-library-snapshots``, made private when missing."""
    from aar.recipes.ccn.scratch import work_root

    folder = work_root() / SNAPSHOT_AREA
    folder.mkdir(mode=0o700, exist_ok=True)
    return folder


def project_name(project: str, timeline: str, round_: int, attempt: int = 1) -> str:
    """The project folder's name: ``<show> - <timeline> r<round>`` (``-<n>`` from the second
    try), made of characters every share takes (:func:`~.export_files.safe_name`)."""
    stem = safe_name(f"{project} - {timeline}")[:150].rstrip(" .") or "cut"
    suffix = f" r{round_}" + (f"-{attempt}" if attempt > 1 else "")
    return stem + suffix


class _Refused(Exception):
    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def _inside_root(label: str, path: str) -> Path:
    """``path`` when it leads (``realpath``) inside the media root's ``realpath``."""
    root = media_root()
    real = os.path.realpath(path)
    if root is not None and not (real == root or real.startswith(root.rstrip("/") + "/")):
        raise _Refused(
            "path_outside_media_root",
            f"the library's {label} {path!r} leads outside the media root (to {real!r})",
            "give a path inside the media root that is not a link out of it",
        )
    return Path(path)


def _facts(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    from aar.drp.mediapool.gate import load_facts

    failures: dict[str, dict[str, str]] = {}
    try:
        return load_facts(path, failures), failures
    except FileNotFoundError:
        raise _Refused(
            "facts_unreadable",
            f"the facts file {str(path)!r} does not exist",
            "measure the cut's files on the edit PC with `aar media facts` and save the file "
            "where the run's library.facts names it",
        ) from None
    except OSError as exc:
        raise _Refused(
            "facts_unreadable",
            f"the facts file {str(path)!r} cannot be read ({exc.strerror or exc})",
            "check the file and its permissions",
        ) from None
    except MediaPoolError as exc:
        raise _Refused(
            "bad_facts",
            f"{exc.message} [{exc.code}]",
            "measure the files again with `aar media facts` on the edit PC",
        ) from None


def _refused(lib: LibraryParam, code: str, message: str, hint: str) -> dict[str, Any]:
    return {
        "status": "refused",
        "library": lib.folder,
        "label": lib.label or Path(lib.folder).name,
        "error": {"code": code, "message": message[:2000], "hint": hint[:1000]},
    }


def hand_over(
    lib: LibraryParam,
    *,
    plan: EditPlan,
    plan_sha256: str,
    profile: Profile,
    media_map: dict[str, str],
    project: str,
    round_: int,
    transcript: Transcript | None,
    laughs: LaughMap | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Write ``plan`` into the library; ``(record, export report or None)``. Never raises for
    the library's, the seed's, the facts' or the cut's faults: those are a refused record."""
    from aar.plan.export.library import export_library

    label = lib.label or Path(lib.folder).name
    try:
        folder = _inside_root("folder", lib.folder)
        seed = _inside_root("seed", lib.seed)
        facts_path = _inside_root("facts file", lib.facts)
        templates = [_inside_root("template", t) for t in lib.templates]
        facts, failures = _facts(facts_path)
    except _Refused as exc:
        return _refused(lib, exc.code, exc.message, exc.hint), None
    options = LibraryOptions(snapshot_dir=snapshot_dir())
    mapping = media_map_from_mapping(media_map)
    result = None
    last: ExportError | None = None
    name = ""
    for attempt in range(1, MAX_NAME_TRIES + 1):
        name = project_name(project, plan.timeline, round_, attempt)
        try:
            result = export_library(
                plan,
                profile,
                library=folder,
                seed=seed,
                facts=facts,
                templates=templates,
                project_name=name,
                media_map=mapping,
                transcript=transcript,
                laughs=laughs,
                allow_unverified_media=lib.allow_unverified_media,
                allow_unopened=lib.allow_unopened_operations,
                seed_resolution=lib.seed_resolution,
                user=lib.user,
                options=options,
            )
            break
        except ExportError as exc:
            last = exc
            if exc.code != "project_exists":
                break
        except Exception as exc:  # never lose the FCPXML hand-off to a library fault
            return (
                _refused(
                    lib,
                    "library_failed",
                    f"writing the library project failed ({type(exc).__name__}: {exc})"[:2000],
                    "report it with the run's id; the FCPXML hand-off is unaffected",
                ),
                None,
            )
    if result is None:
        assert last is not None
        message, hint = last.message, last.fix_hint
        if last.code == "facts_missing" and failures:
            listed = "; ".join(
                f"{path}: {error['code']}: {error['message']}"
                for path, error in sorted(failures.items())[:8]
            )
            message += f" (the facts file lists files it could not measure: {listed})"
        if last.code == "project_exists":
            message += f" (tried {MAX_NAME_TRIES} names up to {name!r})"
        residue = last.details.get("residue") if isinstance(last.details, dict) else None
        if residue:
            message += f"; left in the library: {residue}"
        return _refused(lib, last.code, message, hint), None
    report = result.report
    placed = result.result.placed
    digest = report["project"].get("sha256")
    record: dict[str, Any] = {
        "status": "placed",
        "library": lib.folder,
        "label": label,
        "user": placed.user,
        "folder": placed.folder.name,
        "parts": list(placed.folder.relative_to(folder).parts),
        "project_id": report["project"]["project_id"],
        "project_name": report["project"]["name"],
        "timeline": plan.timeline,
        "plan_sha256": plan_sha256,
        "placed_sha256": digest,
        "operations_not_opened": list(report.get("operations_not_opened") or []),
        "unverified_media": sorted(
            m.get("path", "") for m in report.get("media") or [] if m.get("status") != "verified"
        ),
        "not_carried": len(report.get("not_carried") or []),
        "cleanup": list(report.get("placement", {}).get("cleanup") or [])[:8],
    }
    if digest is None:
        # Without the placed file's hash a save cannot be told from the placed project.
        record["pickup"] = "off: the placed Project.db could not be hashed (see cleanup)"
    return record, report
