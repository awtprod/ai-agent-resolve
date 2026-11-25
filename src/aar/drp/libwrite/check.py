"""Read a written project back the way every reader of a library project reads it.

:func:`check_written` is the writer's own check, run on the staged file before anything is
placed: SQLite's integrity check and the bounded library reader (:func:`aar.drp.read_drp`, the
same reader ``aar plan import-timeline`` uses) must read it with no issue outside
``allowed_issues``, and its one timeline must hold exactly the items written (track, record
range, source in and file path), at the seed's rate and start. The export adds the plan-level
check on top (``aar.plan.export.library``): the timeline import of the written project against
the plan it was written from must report no change.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

from aar.drp import read_drp
from aar.drp.errors import DrpError
from aar.drp.library import LibraryError, LibraryOptions
from aar.drp.libwrite.errors import ProjectWriteError
from aar.drp.libwrite.seed import Seed
from aar.drp.libwrite.write import ItemSpec
from aar.drp.model import DrpFile

__all__ = ["check_written"]

_HINT = "this is a writer bug: the project was not placed; report the seed and the plan"


def _fail(message: str, **details: object) -> ProjectWriteError:
    return ProjectWriteError("self_check_failed", message, _HINT, **details)


def check_written(
    path: Path,
    seed: Seed,
    specs: Sequence[ItemSpec],
    *,
    timeline_name: str,
    project_name: str,
    allowed_issues: Collection[str],
) -> DrpFile:
    """The written project read back (see the module docstring), or ``self_check_failed``."""
    try:
        drp = read_drp(path, library_options=LibraryOptions(settle_seconds=0.0))
    except (LibraryError, DrpError, OSError) as exc:
        raise _fail(f"the written project does not read back: {exc}") from None
    issues = [i for i in drp.issues if i.code not in allowed_issues]
    if issues:
        raise _fail(
            f"the written project reads with {len(issues)} issue(s) an import refuses: "
            + ", ".join(sorted({i.code for i in issues})),
            issues=[f"{i.code}: {i.message}"[:300] for i in issues[:20]],
        )
    name = drp.project.name if drp.project is not None else None
    if name != project_name:
        raise _fail(f"the project reads back named {name!r}")
    timelines = [t for t in drp.timelines if t.kind == "timeline"]
    if len(timelines) != 1 or timelines[0].name != timeline_name:
        raise _fail(f"the project reads back with timelines {[t.name for t in timelines]!r}")
    [timeline] = timelines
    if timeline.frame_rate != float(seed.rate) or timeline.start_frame != seed.start_frame:
        raise _fail(
            f"the timeline reads back at {timeline.frame_rate} fps from frame "
            f"{timeline.start_frame}, not the seed's"
        )
    got: list[ItemSpec] = []
    for track in timeline.tracks:
        for item in track.items:
            if (
                item.record_start is None
                or item.record_end is None
                or item.fractions
                or item.timemap_form not in (None, "identity")
            ):
                raise _fail(f"an item reads back without whole-frame positions: {item.id}")
            got.append(
                ItemSpec(
                    track.type,
                    item.record_start - seed.start_frame,
                    item.record_end - seed.start_frame,
                    item.source_in or 0,
                    (item.media_path or "").casefold(),
                )
            )
    want = sorted(
        ItemSpec(s.track, s.record_in, s.record_out, s.src_in, s.path.casefold()) for s in specs
    )
    if sorted(got) != want:
        missing = [s for s in want if s not in got][:5]
        extra = [s for s in got if s not in want][:5]
        raise _fail(
            "the timeline reads back with other items than were written",
            missing=[str(s) for s in missing],
            extra=[str(s) for s in extra],
        )
    return drp
