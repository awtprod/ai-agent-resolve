"""Opt-in checks of the library reader against Resolve's own exports (S25a).

Set ``AAR_LIBRARY_PAIRS`` to a folder of pairs: each subfolder holds a ``Project.db`` copied
from a disk library and ``resolve-export.drp``, Resolve's export of the same project. A pair
whose two files carry the same ``LastModTimeInSecs`` is exact: the two models must be equal
(except for the one documented layout difference below) and ``aar drp diff`` must report no
change. A pair saved after its export only has to read and diff; its changes are printed
(``pytest -s``), never hidden. ``*.db`` files in a ``v17-unpaired`` subfolder are read on their
own and summarised. Nothing from the files is committed; test ids are numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from aar.drp import read_drp
from aar.drp.diff import diff_projects
from aar.drp.library import LibraryOptions
from aar.drp.model import DrpFile

PAIRS = os.environ.get("AAR_LIBRARY_PAIRS")
OPTIONS = LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0)
_SOURCE_ONLY = (
    "path",
    "kind",
    "kind_source",
    "sniffed_kind",
    "size",
    "sha256",
    "is_zip",
    "signature",
    "head_hex",
    "members",
    "raw_members",
)


def _pairs() -> list[Path]:
    if not PAIRS:
        return []
    return sorted(
        p
        for p in Path(PAIRS).iterdir()
        if (p / "Project.db").is_file() and (p / "resolve-export.drp").is_file()
    )


def _unpaired() -> list[Path]:
    if not PAIRS:
        return []
    return sorted((Path(PAIRS) / "v17-unpaired").glob("*.db"))


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip(v)
            for k, v in value.items()
            if k not in ("source", "spans", "where", "member")
        }
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _semantic(drp: DrpFile) -> dict[str, Any]:
    data = drp.to_dict()
    for key in _SOURCE_ONLY:
        data.pop(key)
    data["versions"] = data["versions"]["project_version"] if data["versions"] else None
    data["issues"] = [(i["severity"], i["code"], i["message"]) for i in data["issues"]]
    return _strip(data)


def _follow_project_resolution(export: DrpFile, library: DrpFile) -> int:
    """The documented difference: a timeline that follows the project's resolution stores 0x0
    in the library, where the export writes the project's. Copies the export's value onto the
    library model for those timelines only, and returns how many it changed."""
    theirs = {t.id: t for t in export.timelines}
    changed = 0
    for tl in library.timelines:
        other = theirs.get(tl.id)
        if other is not None and tl.resolution == (0, 0) and other.resolution not in (None, (0, 0)):
            tl.resolution = other.resolution
            changed += 1
    return changed


@pytest.mark.skipif(not PAIRS, reason="set AAR_LIBRARY_PAIRS to run the library pair checks")
@pytest.mark.parametrize("index", range(len(_pairs())) if PAIRS else [0])
def test_library_project_matches_resolves_export(index: int) -> None:
    folder = _pairs()[index]
    export = read_drp(folder / "resolve-export.drp")
    library = read_drp(folder / "Project.db", library_options=OPTIONS)
    assert library.kind == "library" and library.project is not None and export.project is not None
    result = diff_projects(export, library)
    exact = export.project.last_modified_secs == library.project.last_modified_secs
    print(f"pair {index}: exact={exact}; changes: {[c.text for c in result.changes]}")
    if not exact:
        return
    assert result.changes == [], [c.text for c in result.changes]
    _follow_project_resolution(export, library)
    assert _semantic(export) == _semantic(library)


@pytest.mark.skipif(not PAIRS, reason="set AAR_LIBRARY_PAIRS to run the library pair checks")
@pytest.mark.parametrize("index", range(len(_unpaired())) if PAIRS else [0])
def test_unpaired_library_project_reads(index: int) -> None:
    path = _unpaired()[index]
    drp = read_drp(path, library_options=OPTIONS)
    assert drp.kind == "library" and drp.project is not None and drp.timelines
    errors = [i for i in drp.issues if i.severity == "error"]
    assert errors == []
    for tl in drp.timelines:
        durations = sorted(
            {i.duration for t in tl.tracks for i in t.items if i.duration is not None}
        )
        print(
            f"unpaired {index}: version {drp.versions.project_version if drp.versions else None} "
            f"{tl.kind} items={sum(len(t.items) for t in tl.tracks)} "
            f"durations={durations[:4]}{'...' if len(durations) > 4 else ''}"
        )
