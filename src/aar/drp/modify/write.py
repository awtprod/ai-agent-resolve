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
   place, the sidecar first, so a written file never lacks its label.

Nothing here talks to Resolve. Outputs stay ``unverified`` until the sandbox round trip (D3).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
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

__all__ = ["DEFAULT_MAX_INPUT_BYTES", "LABEL", "SIDECAR_SUFFIX", "EditOutcome", "edit_file"]

LABEL = "unverified"
SIDECAR_SUFFIX = ".aar.json"
UNVERIFIED_WHY = (
    "checked offline only (S2 gate plus the exact intended change list); not yet imported into "
    "a Resolve sandbox and read back, which waits on approval D3"
)
# The largest corpus file is 11.6 MB; the writer holds the input and the output in memory.
DEFAULT_MAX_INPUT_BYTES = 1024 * 1024 * 1024


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
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    if _same_file(path, out) or _same_file(path, sidecar):
        refusals.append(Refusal("out-is-input", "the output must be a new file, not the input"))
    elif not force and (out.exists() or sidecar.exists()):
        existing = [str(p) for p in (out, sidecar) if p.exists()]
        refusals.append(
            Refusal(
                "out-exists",
                "the output exists; pass --force to replace it",
                details={"paths": existing},
            )
        )
    if not out.parent.is_dir():
        refusals.append(
            Refusal("no-output-dir", f"the output directory {out.parent} does not exist")
        )
    return refusals


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


def _temp(directory: Path, name: str, data: bytes) -> Path:
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


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
        # No hard links on this filesystem (FAT/exFAT). Check, then rename; the window between
        # the two is the one place a concurrent writer could be replaced.
        if dest.exists():
            raise FileExistsError(str(dest)) from None
        os.rename(tmp, dest)
        return
    tmp.unlink()


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
    outcome.refusals = _precheck(drp, path, plan, out_path, force, dry_run)
    if outcome.refusals:
        return outcome
    original = _read_input(path, max_input_bytes)
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
        tmp = _temp(directory, name, new_bytes)
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
        side_tmp = _temp(directory, sidecar.name, label.encode("ascii"))
        stack.callback(side_tmp.unlink, missing_ok=True)
        try:
            _place(side_tmp, sidecar, force)
            try:
                _place(tmp, out_path, force)
            except BaseException:
                sidecar.unlink(missing_ok=True)
                raise
        except FileExistsError as exc:
            outcome.status = "refused"
            outcome.output = None
            outcome.refusals = [Refusal("out-exists", f"the output appeared while writing: {exc}")]
        return outcome
