"""``edit_file``: plan -> verified new file, written atomically next to an ``unverified`` label.

Order of work, and what stops it (every stop writes nothing):

1. The input must read without warnings or errors: the post-write gate requires both files to
   read cleanly, so an input that does not (for example one with a ``blob-partial`` keyed dict)
   can never pass it. Such inputs are refused up front (``input-has-warnings``) rather than by
   weakening the gate.
2. ``input_sha256`` and ``max_project_version`` from the plan, and the output path: never the
   input itself, never an existing file unless ``force``.
3. The edits are resolved and spliced (:mod:`aar.drp.modify.engine`) and the zip is rewritten
   with every untouched byte copied (:mod:`aar.drp.modify.zipraw`).
4. The result is written to a temporary file in the output directory and verified
   (:func:`aar.drp.modify.verify.verify_write`): gate, exact change list, version stamps.
5. Only then are the sidecar ``<out>.aar.json`` (status ``unverified``) and the file moved into
   place, the sidecar first, so a written file never lacks its label. Without ``--force`` each
   move replaces nothing: it hard-links onto the name, or on a filesystem with no hard links
   (FAT/exFAT) claims the name with ``O_CREAT|O_EXCL`` and renames onto its own placeholder, so a
   file another writer put there is never overwritten (:func:`_place_no_hardlink`).

A run stopped hard between those two moves leaves a label without its file; the next run for
the same ``--out`` refuses with ``orphan-sidecar`` and lists what to remove. Temporary files a
stopped run left behind are listed in ``leftovers``. Nothing is ever deleted that this run did
not create.

Nothing here talks to Resolve. Outputs stay ``unverified`` until the sandbox round trip (D3).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aar import __version__
from aar.drp.errors import ContainerError
from aar.drp.model import DrpFile
from aar.drp.modify import engine, zipraw
from aar.drp.modify.engine import EditRefused, Refusal
from aar.drp.modify.plan import DrpEditPlan
from aar.drp.modify.verify import verify_write
from aar.drp.reader import read_drp

__all__ = [
    "DEFAULT_MAX_INPUT_BYTES",
    "LABEL",
    "SIDECAR_SUFFIX",
    "EditOutcome",
    "edit_file",
    "leftovers",
]

LABEL = "unverified"
SIDECAR_SUFFIX = ".aar.json"
UNVERIFIED_WHY = (
    "checked offline only (S2 gate plus the exact intended change list); not yet imported into "
    "a Resolve sandbox and read back, which waits on approval D3"
)
# The largest corpus file is 11.6 MB; the writer holds the input and the output in memory.
DEFAULT_MAX_INPUT_BYTES = 1024 * 1024 * 1024
_MAX_SIDECAR_BYTES = 512 * 1024 * 1024


@dataclass(slots=True)
class EditOutcome:
    """What ``aar drp edit`` prints. ``status`` is ``written``, ``dry-run`` or ``refused``."""

    status: str
    input: dict[str, Any]
    plan: dict[str, Any]
    output: dict[str, Any] | None = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    expected: list[dict[str, Any]] = field(default_factory=list)
    report: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    gate: dict[str, Any] | None = None
    leftovers: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.status == "refused"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "label": LABEL,
            "label_reason": UNVERIFIED_WHY,
            "tool": f"aar {__version__}",
            "input": self.input,
            "output": self.output,
            "plan": self.plan,
            "gate": self.gate,
            "changes": self.changes,
            "expected": self.expected,
            "report": self.report,
            "refusals": [r.to_dict() for r in self.refusals],
            "leftovers": self.leftovers,
        }


def _input_info(path: Path, drp: DrpFile | None) -> dict[str, Any]:
    v = drp.versions if drp is not None else None
    return {
        "path": str(path),
        "sha256": drp.sha256 if drp is not None else None,
        "kind": drp.kind if drp is not None else None,
        "app_version": v.app_version if v else None,
        "project_version": v.project_version if v else None,
    }


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or (b.exists() and os.path.samefile(a, b))
    except OSError:
        return False


def _precheck(
    drp: DrpFile, path: Path, plan: DrpEditPlan, out: Path | None, force: bool, dry_run: bool
) -> list[Refusal]:
    refusals: list[Refusal] = []
    noisy = sorted({i.code for i in drp.issues if i.severity in ("warning", "error")})
    if noisy:
        refusals.append(
            Refusal(
                "input-has-warnings",
                "the input reads with warnings or errors, so the post-write gate can never pass "
                "for it and no edit can be verified: " + ", ".join(noisy),
                details={"codes": noisy},
            )
        )
    if drp.kind not in ("drp", "drt") or not drp.is_zip:
        refusals.append(
            Refusal("unsupported-kind", f"only .drp and .drt archives are edited, not {drp.kind}")
        )
    if plan.input_sha256 is not None and plan.input_sha256 != drp.sha256:
        refusals.append(
            Refusal(
                "input-sha-mismatch",
                "the input is not the file the plan was made for",
                details={"plan": plan.input_sha256, "input": drp.sha256},
            )
        )
    prj = drp.versions.project_version if drp.versions else None
    if plan.max_project_version is not None and (prj is None or prj > plan.max_project_version):
        refusals.append(
            Refusal(
                "version-gate",
                f"the input is stamped DbPrjVer {prj}, newer than the target host's "
                f"{plan.max_project_version}; the writer never restamps a file",
                details={"input": prj, "max": plan.max_project_version},
            )
        )
    if out is None:
        if not dry_run:
            refusals.append(Refusal("no-output", "--out is required unless --dry-run is given"))
        return refusals
    if out.suffix.lower() != f".{drp.kind}":
        refusals.append(
            Refusal(
                "out-suffix",
                f"the output must end in .{drp.kind} like the input; Resolve and the reader "
                "decide what a file is from its suffix",
            )
        )
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    if _same_file(path, out) or _same_file(path, sidecar):
        refusals.append(Refusal("out-is-input", "the output must be a new file, not the input"))
    elif not force and _orphan_sidecar(out, sidecar):
        remove = [str(sidecar), *(str(p) for p in leftovers(out))]
        refusals.append(
            Refusal(
                "orphan-sidecar",
                f"an earlier aar drp edit was stopped after it wrote the label {sidecar} and "
                f"before it moved {out.name} into place, so the label describes a file that does "
                "not exist; remove the files listed in details.remove (or pass --force) and run "
                "again",
                details={"remove": remove},
            )
        )
    elif not force and (out.exists() or sidecar.exists()):
        existing = [str(p) for p in (out, sidecar) if p.exists()]
        refusals.append(
            Refusal(
                "out-exists",
                "the output exists; pass --force to replace it",
                details={"paths": existing},
            )
        )
    elif force and sidecar.is_file() and not _is_our_sidecar(sidecar, out):
        refusals.append(
            Refusal(
                "foreign-sidecar",
                f"a label {sidecar} that aar drp edit did not write is already here, so --force "
                "leaves it and the output untouched; move or delete it, or choose another --out",
                details={"sidecar": str(sidecar)},
            )
        )
    if not out.parent.is_dir():
        refusals.append(
            Refusal("no-output-dir", f"the output directory {out.parent} does not exist")
        )
    return refusals


def _temp_pattern(name: str, suffix: str) -> re.Pattern[str]:
    # _temp names files "." + final name + 8 random mkstemp characters + ".tmp" + suffix.
    return re.compile(rf"\.{re.escape(name)}\.[a-z0-9_]{{8}}\.tmp{re.escape(suffix)}")


def leftovers(out: Path) -> list[Path]:
    """Temporary files an interrupted run for ``out`` left in its directory (ours by name)."""
    sidecar = out.name + SIDECAR_SUFFIX
    patterns = (_temp_pattern(out.name, out.suffix), _temp_pattern(sidecar, ".json"))
    try:
        with os.scandir(out.parent) as entries:
            names = [e.name for e in entries if e.name.startswith("." + out.name)]
    except OSError:
        return []
    return sorted(out.parent / n for n in names if any(p.fullmatch(n) for p in patterns))


def _is_our_sidecar(sidecar: Path, out: Path) -> bool:
    """``sidecar`` is a label this tool wrote for this exact ``out`` (see DESIGN "Modify").

    Recognised by the stamp every ``aar drp edit`` sidecar carries: a ``tool`` field of
    ``aar <version>``, a ``label`` and ``status`` of ``unverified`` (the only states a sidecar on
    disk is ever written in), and an ``output`` object whose ``path`` and ``sidecar`` name this
    output file and this label. ``--force`` replaces only a sidecar that bears this stamp; any
    other file at the path is left untouched.
    """
    try:
        if not sidecar.is_file() or sidecar.stat().st_size > _MAX_SIDECAR_BYTES:
            return False
        record = json.loads(sidecar.read_bytes())
    except (OSError, ValueError, RecursionError):
        return False
    if not isinstance(record, dict):
        return False
    output = record.get("output")
    return (
        isinstance(output, dict)
        and str(record.get("tool", "")).startswith("aar ")
        and record.get("label") == LABEL
        and record.get("status") == LABEL
        and Path(str(output.get("path"))).name == out.name
        and Path(str(output.get("sidecar"))).name == sidecar.name
    )


def _orphan_sidecar(out: Path, sidecar: Path) -> bool:
    """``sidecar`` is a label this tool wrote for ``out``, and ``out`` itself does not exist."""
    if out.exists() or out.is_symlink():
        return False
    return _is_our_sidecar(sidecar, out)


def _read_input(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise ContainerError(f"{path.name} is {size} bytes, over the {max_bytes}-byte limit")
        with path.open("rb") as fh:
            data = fh.read(max_bytes + 1)
    except OSError as exc:
        raise ContainerError(f"cannot read {path}: {exc}") from exc
    if len(data) > max_bytes:
        raise ContainerError(f"{path.name} grew past the {max_bytes}-byte limit while read")
    return data


def _temp(directory: Path, name: str, data: bytes, mode: int | None = None) -> Path:
    """Write ``data`` to a new temporary file next to its final ``name``.

    The temporary file keeps the final suffix, so it reads exactly as the final file will.
    ``mode`` sets its permission bits (mkstemp creates files 0600).
    """
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=f".tmp{Path(name).suffix}", dir=directory)
    try:
        if mode is not None:
            os.chmod(tmp, mode)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def _backup(directory: Path, sidecar: Path) -> Path | None:
    """A hard link (else a copy) of an existing label, named like our temporary files."""
    if not sidecar.is_file():
        return None
    fd, name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp.json", dir=directory)
    os.close(fd)
    backup = Path(name)
    try:
        backup.unlink()
        try:
            os.link(sidecar, backup)
        except OSError:
            shutil.copy2(sidecar, backup)
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _place(tmp: Path, dest: Path, force: bool) -> None:
    """Move ``tmp`` to ``dest`` atomically; without ``force`` never replace an existing file."""
    if force:
        os.replace(tmp, dest)
        return
    try:
        os.link(tmp, dest)  # fails if dest exists: no window for a race
    except FileExistsError:
        raise
    except OSError:
        _place_no_hardlink(tmp, dest)  # no hard links on this filesystem (FAT/exFAT)
        return
    tmp.unlink()


def _place_no_hardlink(tmp: Path, dest: Path) -> None:
    """Place ``tmp`` at ``dest`` on a filesystem without hard links, never replacing a file
    another writer put there.

    ``os.link`` is not available, so claim the name with ``O_CREAT|O_EXCL`` (which fails with
    ``FileExistsError`` if ``dest`` already exists, leaving no check-then-rename window a
    concurrent create could slip through), then rename our verified temporary onto our own
    placeholder. A crash between the claim and the rename can leave our own partial labelled copy
    (the empty placeholder, or the finished output), never a file that was already there.
    """
    fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)  # FileExistsError if taken
    try:
        os.close(fd)
        os.rename(tmp, dest)  # atomically replaces the placeholder we just created, nothing else
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(dest)  # remove only the placeholder we created; never a pre-existing file
        raise


def edit_file(
    input_path: str | Path,
    plan: DrpEditPlan,
    out: str | Path | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> EditOutcome:
    """Apply ``plan`` to ``input_path`` and write ``out`` (see the module docstring).

    Raises :class:`~aar.drp.errors.DrpError` when the input cannot be read and ``OSError`` when
    the output cannot be written; every refusal is returned in the outcome instead.
    """
    path = Path(input_path)
    out_path = Path(out) if out is not None else None
    plan_dict = plan.model_dump(mode="json", exclude_none=True)  # parse_plan reads it back
    drp = read_drp(path)
    outcome = EditOutcome("refused", _input_info(path, drp), plan_dict)
    if out_path is not None:
        outcome.leftovers = [str(p) for p in leftovers(out_path)]
    outcome.refusals = _precheck(drp, path, plan, out_path, force, dry_run)
    if outcome.refusals:
        return outcome
    original = _read_input(path, max_input_bytes)
    # Outputs get the input's permission bits (execute and special bits dropped).
    mode = stat.S_IMODE(path.stat().st_mode) & 0o666
    if hashlib.sha256(original).hexdigest() != drp.sha256:
        outcome.refusals = [Refusal("input-changed", "the input changed on disk while it was read")]
        return outcome
    try:
        zipraw.parse_layout(original)
        prepared = engine.prepare(drp, path, plan)
        new_members = engine.apply(prepared.member_data, prepared.splices)
        new_bytes = zipraw.rewrite(original, new_members)
    except EditRefused as exc:
        outcome.refusals = exc.refusals
        return outcome
    except zipraw.ZipLayoutError as exc:
        outcome.refusals = [Refusal("zip-layout", str(exc))]
        return outcome
    outcome.expected = [e.to_dict() for e in prepared.expected]
    outcome.report = prepared.report
    with contextlib.ExitStack() as stack:
        if dry_run or out_path is None:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="aar-edit-")))
            name = (out_path.name if out_path else path.name) or "out.drp"
        else:
            directory, name = out_path.parent, out_path.name
        tmp = _temp(directory, name, new_bytes, mode)
        stack.callback(tmp.unlink, missing_ok=True)
        check = verify_write(path, tmp, prepared.expected, prepared.members())
        if check.diff is not None:
            outcome.changes = [c.to_dict() for c in check.diff.changes]
            outcome.gate = {
                "passed": check.diff.gate.passed,
                "reasons": list(check.diff.gate.reasons),
            }
        if not check.ok:
            outcome.refusals = check.refusals
            return outcome
        digest = hashlib.sha256(new_bytes).hexdigest()
        outcome.output = {
            "path": str(out_path) if out_path else None,
            "sha256": digest,
            "size": len(new_bytes),
            "sidecar": str(out_path) + SIDECAR_SUFFIX if out_path else None,
        }
        if dry_run or out_path is None:
            outcome.status = "dry-run"
            return outcome
        outcome.status = "written"
        sidecar = Path(str(out_path) + SIDECAR_SUFFIX)
        record = outcome.to_dict()
        record["status"] = LABEL
        label = json.dumps(record, indent=1, ensure_ascii=True, allow_nan=False) + "\n"
        side_tmp = _temp(directory, sidecar.name, label.encode("ascii"), mode)
        stack.callback(side_tmp.unlink, missing_ok=True)
        # With --force, the label being replaced is kept until the new file is in place.
        backup = _backup(directory, sidecar) if force else None
        if backup is not None:
            stack.callback(backup.unlink, missing_ok=True)
        try:
            _place(side_tmp, sidecar, force)
            try:
                _place(tmp, out_path, force)
            except BaseException:
                if backup is not None:
                    os.replace(backup, sidecar)
                else:
                    sidecar.unlink(missing_ok=True)
                raise
        except FileExistsError as exc:
            outcome.status = "refused"
            outcome.output = None
            outcome.refusals = [Refusal("out-exists", f"the output appeared while writing: {exc}")]
        return outcome
