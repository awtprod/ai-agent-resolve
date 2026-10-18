"""Post-write verification: the S2 gate, then the full change list against the intended edits.

A passing gate means every difference was decoded and matched by DbId, not that the
differences were the intended ones (DESIGN R1). :func:`verify_write` therefore also requires
the diff's change list to equal the plan's expected changes exactly, as a multiset: an
unintended change, or an intended change that is missing, is a refusal. It also checks that the
version stamp of every XML member is unchanged (the writer never restamps a file) and that only
the members the plan touched differ.

This check is for the writer's own output and never uses the Resolve round-trip allow-list
(:mod:`aar.drp.modify.roundtrip`): our writer must produce zero unintended changes.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aar.drp.container import open_container, parse_version_header
from aar.drp.diff import diff_projects
from aar.drp.diff_model import ProjectDiff
from aar.drp.errors import DrpError
from aar.drp.model import DrpFile
from aar.drp.modify.engine import Expected, Refusal, change_key
from aar.drp.reader import read_drp

__all__ = ["Verification", "member_stamps", "verify_write"]


@dataclass(slots=True)
class Verification:
    """``ok`` only when the gate passed, the changes are exactly the intended ones, the version
    stamps are unchanged and no member outside the plan differs."""

    refusals: list[Refusal] = field(default_factory=list)
    diff: ProjectDiff | None = None

    @property
    def ok(self) -> bool:
        return not self.refusals and self.diff is not None


def member_stamps(path: str | Path) -> dict[str, str]:
    """The raw ``DbAppVer``/``DbPrjVer`` comment of every XML member that has one."""
    out: dict[str, str] = {}
    with open_container(path) as c:
        for name in c.xml_members():
            v = parse_version_header(c.head(name), name)
            if v is not None:
                out[name] = v.raw
    return out


def _changed_members(a: Path, b: Path) -> set[str]:
    """Members whose stored CRC or size differs, or that exist on one side only."""
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        ia = {i.filename: (i.CRC, i.file_size) for i in za.infolist()}
        ib = {i.filename: (i.CRC, i.file_size) for i in zb.infolist()}
    return {n for n in set(ia) | set(ib) if ia.get(n) != ib.get(n)}


def _version_refusals(a: DrpFile, b: DrpFile, original: Path, written: Path) -> list[Refusal]:
    va, vb = a.versions, b.versions
    main_a = (va.app_version, va.project_version) if va else None
    main_b = (vb.app_version, vb.project_version) if vb else None
    stamps_a, stamps_b = member_stamps(original), member_stamps(written)
    if main_a == main_b and stamps_a == stamps_b:
        return []
    changed = sorted(n for n in set(stamps_a) | set(stamps_b) if stamps_a.get(n) != stamps_b.get(n))
    return [
        Refusal(
            "version-gate",
            "the written file's DbAppVer/DbPrjVer stamp differs from the input's; the writer "
            "never restamps a file",
            details={"input": main_a, "output": main_b, "members": changed},
        )
    ]


def _listed(keys: Counter[tuple[Any, ...]], pool: dict[tuple[Any, ...], Any]) -> list[Any]:
    out = []
    for key, n in sorted(keys.items(), key=lambda kv: repr(kv[0])):
        out.extend([pool[key]] * n)
    return out


def verify_write(
    original: str | Path,
    written: str | Path,
    expected: Sequence[Expected],
    members: set[str],
) -> Verification:
    """Check ``written`` against ``original`` and the intended changes (see the module doc)."""
    original, written = Path(original), Path(written)
    result = Verification()
    try:
        a, b = read_drp(original), read_drp(written)
        d = diff_projects(a, b, gate=True)
    except DrpError as exc:
        result.refusals.append(Refusal("verify-unreadable", f"cannot compare the files: {exc}"))
        return result
    result.diff = d
    if not d.gate.passed:
        result.refusals.append(
            Refusal(
                "gate-failed",
                "the post-write gate failed: " + "; ".join(d.gate.reasons),
                details={"reasons": list(d.gate.reasons)},
            )
        )
    want = Counter(e.key() for e in expected)
    have = Counter(change_key(c) for c in d.changes)
    want_pool = {e.key(): e.to_dict() for e in expected}
    have_pool = {change_key(c): c.to_dict() for c in d.changes}
    missing, unintended = want - have, have - want
    if unintended:
        result.refusals.append(
            Refusal(
                "unintended-change",
                f"{sum(unintended.values())} change(s) in the written file are not in the plan",
                details={"changes": _listed(unintended, have_pool)},
            )
        )
    if missing:
        result.refusals.append(
            Refusal(
                "missing-change",
                f"{sum(missing.values())} intended change(s) do not show in the written file",
                details={"changes": _listed(missing, want_pool)},
            )
        )
    result.refusals.extend(_version_refusals(a, b, original, written))
    stray = sorted(_changed_members(original, written) - members)
    if stray:
        result.refusals.append(
            Refusal(
                "unintended-change",
                f"{len(stray)} zip member(s) changed that the plan did not touch",
                details={"members": stray},
            )
        )
    return result
