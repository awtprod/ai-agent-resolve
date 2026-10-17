"""Opt-in diff checks over the private corpus (``$AAR_DRP_CORPUS``); see ``test_drp_corpus``.

Test ids and printed lines use numbers, never file or project names. Run with ``-s`` to see
the timing lines (prefixed ``diff-corpus``).
"""

from __future__ import annotations

import itertools
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import pytest
from drp_edit import edit_zip, reindent

from aar.drp.container import open_container
from aar.drp.diff import ProjectDiff, diff_projects
from aar.drp.reader import read_drp

CORPUS = os.environ.get("AAR_DRP_CORPUS")
SUFFIXES = {".drp", ".drt", ".drb"}
FILES = sorted(p for p in Path(CORPUS).rglob("*") if p.suffix.lower() in SUFFIXES) if CORPUS else []
needs_corpus = pytest.mark.skipif(not CORPUS, reason="set AAR_DRP_CORPUS to run the corpus checks")
_STAMP = re.compile(r"_\d{8}_\d{6}.*$")


def family(path: Path) -> str:
    """Group the ``dat-*`` files: last ``__`` part, without timestamps and copy suffixes."""
    base = _STAMP.sub("", path.stem.split("__")[-1])
    return re.sub(r"_(original|backup)$", "", base)


def dat_families() -> list[list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in FILES:
        if p.name.startswith("dat-") and p.suffix.lower() == ".drp":
            groups[family(p)].append(p)
    return [sorted(g) for _, g in sorted(groups.items())]


def timed(a: Path, b: Path) -> tuple[ProjectDiff, float]:
    t0 = time.perf_counter()
    d = diff_projects(read_drp(a), read_drp(b))
    return d, time.perf_counter() - t0


def summary(d: ProjectDiff) -> str:
    return " ".join(f"{s}={n}" for s, n in d.counts().items())


@needs_corpus
@pytest.mark.parametrize("path", FILES, ids=[f"corpus-{i:03d}" for i in range(len(FILES))])
def test_corpus_diff_with_itself_is_empty(path: Path, tmp_path: Path) -> None:
    d, seconds = timed(path, path)
    assert d.changes == [], d.to_text()[:2000]
    line = f"diff-corpus self file={FILES.index(path):03d} {seconds:.3f}s"
    drp = read_drp(path)
    if drp.is_zip:
        # Same content with every interpreted member re-indented: nothing is byte-identical,
        # so every element of the real file is aligned and compared.
        names = [m.name for m in drp.members if m.interpreted]
        copy = edit_zip(path, tmp_path / f"indented{path.suffix}", dict.fromkeys(names, reindent))
        d, seconds = timed(path, copy)
        assert d.changes == [], d.to_text()[:2000]
        line += f" reindented={seconds:.3f}s"
    print(line)


@needs_corpus
def test_corpus_dat_families_consecutive() -> None:
    families = dat_families()
    assert sum(len(f) for f in families) >= 2
    pairs = 0
    total = 0.0
    for fi, files in enumerate(families):
        for pi, (a, b) in enumerate(itertools.pairwise(files)):
            d, seconds = timed(a, b)
            back, _ = timed(b, a)
            assert len(back.changes) == len(d.changes)
            pairs += 1
            total += seconds
            print(
                f"diff-corpus family={fi:02d} pair={pi:02d} files={FILES.index(a):03d}->"
                f"{FILES.index(b):03d} {seconds:.3f}s changes={len(d.changes)} {summary(d)} "
                f"heuristic={d.heuristic_count()}"
            )
    print(f"diff-corpus dat families={len(families)} pairs={pairs} total={total:.2f}s")
    assert pairs > 0


def _by_version(app_prefix: str, prj: int) -> Path | None:
    for p in FILES:
        if p.suffix.lower() != ".drp":
            continue
        with open_container(p) as c:
            v = c.versions()
        if v and v.project_version == prj and (v.app_version or "").startswith(app_prefix):
            return p
    return None


@needs_corpus
def test_corpus_r19_against_r21() -> None:
    r19, r21 = _by_version("19.", 14), _by_version("21.", 17)
    if r19 is None or r21 is None:
        pytest.skip("corpus has no R19 and R21 project pair")
    d, seconds = timed(r19, r21)
    fields = {c.field_name for c in d.changes if c.object_kind == "file"}
    assert {"DbAppVer", "DbPrjVer"} <= fields
    print(
        f"diff-corpus r19-vs-r21 files={FILES.index(r19):03d}->{FILES.index(r21):03d} "
        f"{seconds:.2f}s changes={len(d.changes)} {summary(d)}"
    )
