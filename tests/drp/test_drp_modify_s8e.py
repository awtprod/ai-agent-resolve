"""S8e: follow-ups from the S8d validation (LOW-1 kept label snapshots, LOW-2 label rollback).

The two validator probes (``test_probe_a_label_snapshot_collides_with_leftovers_pattern`` and
``test_probe_h_label_cleanup_failure_after_output_failure_is_recoverable``) are adopted here,
inverted so they assert the fix. They run on the bundled templates only.
"""

from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from diff_testing import H264

from aar.drp.modify import parse_plan, write
from aar.drp.modify.write import SIDECAR_SUFFIX, edit_file, leftovers

# The kept-snapshot name is a documented contract (DESIGN), so it is spelled out here rather than
# imported; test_low1_the_module_uses_the_documented_snapshot_suffix ties the two together.
SNAPSHOT_SUFFIX = ".label-snapshot.json"
RENAME = {"op": "rename", "target": "project", "new_name": "x"}
HUMAN = '{"owner": "a human", "notes": "keep me"}'


def plan(*edits: dict[str, Any]) -> Any:
    return parse_plan(json.dumps({"edits": list(edits)}))


def files(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def _left(x: Any) -> list[tuple[str, str]]:
    return [(e.path, e.kind) for e in getattr(x, "left_behind", [])]


def _first_run(tmp_path: Path, name: str = "out.drp") -> tuple[Path, Path]:
    out = tmp_path / name
    assert edit_file(H264, plan(RENAME), out).status == "written"
    return out, Path(str(out) + SIDECAR_SUFFIX)


def _is_temp_name(out: Path, name: str) -> bool:
    sidecar = out.name + SIDECAR_SUFFIX
    return bool(write._temp_pattern(sidecar, ".json").fullmatch(name))


def _foreign_before_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A human label lands just before --force snapshots the label (the run then refuses)."""
    real_backup = write._backup

    def human_writes_then_backup(directory: Path, side: Path) -> Any:
        side.unlink()
        side.write_text(HUMAN)
        return real_backup(directory, side)

    monkeypatch.setattr(write, "_backup", human_writes_then_backup)


# --------------------------------------------------------------------------- LOW-1


def test_low1_a_kept_snapshot_is_not_a_leftover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Validator probe A, inverted: the kept snapshot used to keep its temporary-file name, so the
    # next run listed it in outcome.leftovers (and would have in an orphan-sidecar remove list).
    out, sidecar = _first_run(tmp_path)
    _foreign_before_snapshot(monkeypatch)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert [r.code for r in outcome.refusals] == ["foreign-sidecar"]
    [(path, kind)] = _left(outcome)
    kept = Path(path)
    assert kind == "label-snapshot" and kept.read_text() == HUMAN
    assert kept.name.startswith("." + sidecar.name + ".") and kept.name.endswith(SNAPSHOT_SUFFIX)
    assert not _is_temp_name(out, kept.name)
    assert leftovers(out) == []
    assert files(tmp_path) == sorted([out.name, sidecar.name, kept.name])
    monkeypatch.undo()
    later = edit_file(H264, plan(RENAME), out, force=True)
    assert later.leftovers == [] and kept.read_text() == HUMAN


def test_low1_a_kept_snapshot_is_not_in_an_orphan_sidecar_remove_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, sidecar = _first_run(tmp_path)
    ours = sidecar.read_bytes()
    _foreign_before_snapshot(monkeypatch)
    [(path, _)] = _left(edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True))
    monkeypatch.undo()
    # Our label comes back (a new file: the snapshot is a hard link of the human label) and the
    # output goes away, so the next run sees an orphan label.
    sidecar.unlink()
    sidecar.write_bytes(ours)
    out.unlink()
    outcome = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in outcome.refusals] == ["orphan-sidecar"]
    assert outcome.refusals[0].details["remove"] == [str(sidecar)]
    assert Path(path).read_text() == HUMAN


def test_low1_a_snapshot_rewritten_in_place_is_kept_under_the_snapshot_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The other way a snapshot is kept: a written run whose hard-linked snapshot was rewritten
    # in place after the check.
    out, sidecar = _first_run(tmp_path)
    real_place = write._place

    def human_rewrites_in_place_then_place(tmp: Path, dest: Path, force: bool) -> None:
        if dest == sidecar:
            with open(sidecar, "r+b") as fh:
                fh.truncate(0)
                fh.write(HUMAN.encode())
        real_place(tmp, dest, force)

    monkeypatch.setattr(write, "_place", human_rewrites_in_place_then_place)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert outcome.status == "written"
    [(path, kind)] = _left(outcome)
    assert kind == "label-snapshot" and path.endswith(SNAPSHOT_SUFFIX)
    assert Path(path).read_text() == HUMAN and leftovers(out) == []


def test_low1_a_snapshot_that_was_not_kept_still_has_a_temporary_name_and_is_listed(
    tmp_path: Path,
) -> None:
    # A snapshot a stopped run never settled (or one an older aar kept) keeps the temporary-file
    # name, and the leftovers listing still reports it.
    out, sidecar = _first_run(tmp_path)
    backup = write._backup(tmp_path, sidecar)
    assert backup is not None and _is_temp_name(out, backup.name)
    assert leftovers(out) == [backup]


def test_low1_an_old_style_kept_snapshot_is_still_listed(tmp_path: Path) -> None:
    # What an aar before S8e left behind: a kept snapshot under a ".tmp.json" name. DESIGN says it
    # still shows among the leftovers.
    out, _ = _first_run(tmp_path)
    old = tmp_path / ".out.drp.aar.json.ab12cd34.tmp.json"
    old.write_text(HUMAN)
    assert leftovers(out) == [old]
    assert edit_file(H264, plan(RENAME), out, force=True).leftovers == [str(old)]


def test_low1_if_the_rename_fails_the_snapshot_is_kept_under_its_temporary_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = _first_run(tmp_path)
    _foreign_before_snapshot(monkeypatch)
    real_replace = os.replace

    def replace(src: Any, dst: Any) -> None:
        if str(dst).endswith(SNAPSHOT_SUFFIX):
            raise OSError(errno.EIO, "I/O error (simulated)")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    [(path, kind)] = _left(outcome)
    assert kind == "label-snapshot" and _is_temp_name(out, Path(path).name)
    assert Path(path).read_text() == HUMAN
    assert not any(n.endswith(SNAPSHOT_SUFFIX) for n in files(tmp_path))  # the claim is gone


def _fail_snapshot_claim(monkeypatch: pytest.MonkeyPatch, err: int = errno.ENOSPC) -> None:
    """``mkstemp`` cannot claim a kept-snapshot name (no space, read-only, name too long)."""
    real_mkstemp = tempfile.mkstemp

    def mkstemp(*a: Any, **k: Any) -> Any:
        if k.get("suffix") == SNAPSHOT_SUFFIX:
            raise OSError(err, os.strerror(err))
        return real_mkstemp(*a, **k)

    monkeypatch.setattr(write.tempfile, "mkstemp", mkstemp)


def test_low1_a_failed_snapshot_claim_does_not_undo_a_written_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Round 2 (MED-1; validator repro A, inverted): the claim of the kept-snapshot name used to sit
    # outside the guard, so a failing mkstemp escaped the end-of-run settle step and turned a
    # finished --force write into an error (exit 74) with the snapshot unreported.
    out, sidecar = _first_run(tmp_path)
    real_place = write._place

    def rewrite_in_place_then_place(tmp: Path, dest: Path, force: bool) -> None:
        if dest == sidecar:
            with open(sidecar, "r+b") as fh:
                fh.truncate(0)
                fh.write(HUMAN.encode())
        real_place(tmp, dest, force)

    monkeypatch.setattr(write, "_place", rewrite_in_place_then_place)
    _fail_snapshot_claim(monkeypatch)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    monkeypatch.undo()
    assert outcome.status == "written" and not outcome.refusals
    [(path, kind)] = _left(outcome)
    assert kind == "label-snapshot" and _is_temp_name(out, Path(path).name)
    assert Path(path).read_text() == HUMAN
    assert not any(n.endswith(SNAPSHOT_SUFFIX) for n in files(tmp_path))


@pytest.mark.parametrize("err", [errno.ENOSPC, errno.EROFS, errno.ENAMETOOLONG])
def test_low1_a_failed_snapshot_claim_does_not_undo_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int
) -> None:
    out, _ = _first_run(tmp_path)
    _foreign_before_snapshot(monkeypatch)
    _fail_snapshot_claim(monkeypatch, err)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    monkeypatch.undo()
    assert [r.code for r in outcome.refusals] == ["foreign-sidecar"]
    [(path, kind)] = _left(outcome)
    assert kind == "label-snapshot" and Path(path).read_text() == HUMAN


@pytest.mark.parametrize("extra", [0, 1], ids=["name-max", "name-max-plus-1"])
def test_low1_the_snapshot_name_at_the_filesystem_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: int
) -> None:
    # Validator repro B, inverted, at the exact boundary: the kept-snapshot name is the output's
    # name plus 39 characters. At NAME_MAX it is renamed; one past it, mkstemp fails with
    # ENAMETOOLONG and the snapshot is kept under its temporary name (output + 28, which fits),
    # the refusal intact.
    limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    name = "o" * (limit - 39 + extra - len(".drp")) + ".drp"
    assert len(name) + 39 == limit + extra
    out, _ = _first_run(tmp_path, name)
    _foreign_before_snapshot(monkeypatch)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    monkeypatch.undo()
    assert [r.code for r in outcome.refusals] == ["foreign-sidecar"]
    [(path, kind)] = _left(outcome)
    kept = Path(path)
    assert kind == "label-snapshot" and kept.read_text() == HUMAN
    if extra == 0:
        assert kept.name.endswith(SNAPSHOT_SUFFIX) and len(kept.name) == limit
        assert leftovers(out) == []
    else:
        assert _is_temp_name(out, kept.name) and len(kept.name) == len(name) + 28
        assert leftovers(out) == [kept]
        assert not any(n.endswith(SNAPSHOT_SUFFIX) for n in files(tmp_path))


def test_low1_the_module_uses_the_documented_snapshot_suffix() -> None:
    assert getattr(write, "SNAPSHOT_SUFFIX", None) == SNAPSHOT_SUFFIX


def test_low1_design_states_the_snapshot_name_and_old_runs() -> None:
    text = (Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md").read_text()
    section = " ".join(
        text[text.index("- **Modify (S8).**") : text.index("- **Create (S9).**")].split()
    )
    assert ".label-snapshot.json" in section
    assert (
        "A snapshot kept by an older aar" in section
        and "still listed among the leftovers" in section
    )


# --------------------------------------------------------------------------- LOW-2


def _fail_output_placement(
    monkeypatch: pytest.MonkeyPatch, out: Path, err: OSError | None = None
) -> None:
    real_place = write._place

    def fail_output(tmp: Path, dest: Path, force: bool) -> None:
        if dest == out:
            raise err or OSError(errno.ENOSPC, "No space left on device")
        real_place(tmp, dest, force)

    monkeypatch.setattr(write, "_place", fail_output)


def _fail_label_cleanup(monkeypatch: pytest.MonkeyPatch, sidecar: Path) -> None:
    """Removing the label just placed, or restoring the old one over it, fails (a sharing
    violation). With --force the first ``os.replace`` onto the label is its placement and the
    second is the restore; without --force the label is placed by a hard link."""
    real_unlink, real_replace = os.unlink, os.replace
    onto_label: list[str] = []

    def violation() -> OSError:
        return OSError(errno.EACCES, "cannot touch the label (simulated sharing violation)")

    def unlink(p: Any, *a: Any, **k: Any) -> None:
        if Path(p) == sidecar:
            raise violation()
        real_unlink(p, *a, **k)

    def replace(src: Any, dst: Any) -> None:
        if Path(dst) == sidecar:
            onto_label.append(str(src))
            if len(onto_label) > 1:
                raise violation()
        real_replace(src, dst)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "replace", replace)


def test_low2_the_placement_error_wins_over_a_failed_label_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Validator probe H, inverted: output placement fails (ENOSPC), then removing the label just
    # placed fails (EACCES). The ENOSPC must surface, carrying the cleanup failure as a note, and
    # the label left over must be named.
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    _fail_output_placement(monkeypatch, out)
    _fail_label_cleanup(monkeypatch, sidecar)
    with pytest.raises(OSError) as raised:
        edit_file(H264, plan(RENAME), out)
    monkeypatch.undo()
    err = raised.value
    assert err.errno == errno.ENOSPC and "No space" in str(err)
    assert isinstance(err, write.PlacementError)
    assert isinstance(err.__cause__, OSError) and err.__cause__.errno == errno.ENOSPC
    assert _left(err) == [(str(sidecar), "own-label")]
    assert "orphan-sidecar" in err.left_behind[0].message
    notes = " ".join(getattr(err, "__notes__", []))
    assert "could not roll back its label" in notes and "sharing violation" in notes
    # The left-over label is aar's own, and the next run reports it.
    assert files(tmp_path) == [sidecar.name]
    again = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in again.refusals] == ["orphan-sidecar"]


def test_low2_the_cli_exit_code_is_the_placement_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    out = tmp_path / "o" / "out.drp"
    out.parent.mkdir()
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"edits": [RENAME]}))
    _fail_output_placement(monkeypatch, out)
    _fail_label_cleanup(monkeypatch, sidecar)
    code = main(["drp", "edit", str(H264), "--plan", str(plan_file), "--out", str(out)])
    result = json.loads(capsys.readouterr().out)
    assert code == 74  # the same as for the placement error alone
    assert [r["code"] for r in result["refusals"]] == ["output-error"]
    assert "No space" in result["refusals"][0]["message"]
    assert [(x["path"], x["kind"]) for x in result["left_behind"]] == [(str(sidecar), "own-label")]


def test_low2_with_force_a_failed_restore_keeps_the_old_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Round 2 (LOW-A; validator probe item4, inverted): the output placement fails, then
    # restoring the old label from its snapshot fails too. The old output is still at --out and
    # its correct label survives only in the snapshot, which used to be deleted because it bears
    # aar's stamp. It is now kept off the temporary name and named as a label-snapshot.
    out, sidecar = _first_run(tmp_path)
    old_out, old_label = out.read_bytes(), sidecar.read_bytes()
    _fail_output_placement(monkeypatch, out)
    _fail_label_cleanup(monkeypatch, sidecar)
    with pytest.raises(OSError) as raised:
        edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    monkeypatch.undo()
    err = raised.value
    assert err.errno == errno.ENOSPC
    [own, snap] = _left(err)
    assert own == (str(sidecar), "own-label") and snap[1] == "label-snapshot"
    kept = Path(snap[0])
    assert kept.name.endswith(SNAPSHOT_SUFFIX) and kept.read_bytes() == old_label
    message = err.left_behind[1].message
    assert f"output still at {out.name}" in message and f"over {sidecar.name}" in message
    assert out.read_bytes() == old_out and sidecar.read_bytes() != old_label
    assert leftovers(out) == []
    # Following the message restores the old output's label.
    os.replace(kept, sidecar)
    assert sidecar.read_bytes() == old_label
    assert [r.code for r in edit_file(H264, plan(RENAME), out).refusals] == ["out-exists"]


def test_low2_a_failed_restore_whose_snapshot_cannot_be_renamed_is_still_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same double failure, and the snapshot cannot move off its temporary name either: it is
    # kept there and named, and the end-of-run settle step must not delete it.
    out, sidecar = _first_run(tmp_path)
    old_label = sidecar.read_bytes()
    _fail_output_placement(monkeypatch, out)
    _fail_label_cleanup(monkeypatch, sidecar)
    _fail_snapshot_claim(monkeypatch)
    with pytest.raises(OSError) as raised:
        edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    monkeypatch.undo()
    [_, (path, kind)] = _left(raised.value)
    assert kind == "label-snapshot" and _is_temp_name(out, Path(path).name)
    assert Path(path).read_bytes() == old_label


def test_low2_a_placeholder_and_the_own_label_are_both_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No hard links (FAT/exFAT): the output's replace fails and leaves its placeholder, then the
    # label cleanup fails as well. Both are named, and the placement error's errno is kept.
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)

    def no_links(*a: Any, **k: Any) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    real_replace = os.replace

    def replace(src: Any, dst: Any) -> None:
        if Path(dst) == out:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_replace(src, dst)

    real_unlink = os.unlink

    def unlink(p: Any, *a: Any, **k: Any) -> None:
        if Path(p) == sidecar:
            raise OSError(errno.EACCES, "cannot remove the label (simulated sharing violation)")
        real_unlink(p, *a, **k)

    monkeypatch.setattr(os, "link", no_links)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "unlink", unlink)
    with pytest.raises(write.PlacementError) as raised:
        edit_file(H264, plan(RENAME), out)
    monkeypatch.undo()
    assert raised.value.errno == errno.ENOSPC
    assert _left(raised.value) == [(str(out), "placeholder"), (str(sidecar), "own-label")]
    assert "could not roll back its label" in " ".join(raised.value.__notes__)


def test_low2_an_output_that_appeared_is_still_a_refusal_with_the_label_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The output name was taken while writing (FileExistsError): that is the out-exists refusal
    # (exit 1), with or without a failed label cleanup; the outcome names the label left over.
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    _fail_output_placement(monkeypatch, out, FileExistsError(errno.EEXIST, "File exists"))
    _fail_label_cleanup(monkeypatch, sidecar)
    outcome = edit_file(H264, plan(RENAME), out)
    assert [r.code for r in outcome.refusals] == ["out-exists"]
    assert outcome.status == "refused"
    assert _left(outcome) == [(str(sidecar), "own-label")]
    assert outcome.to_dict()["left_behind"][0]["kind"] == "own-label"


def test_low2_a_clean_rollback_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out.drp"
    _fail_output_placement(monkeypatch, out)
    with pytest.raises(OSError) as raised:
        edit_file(H264, plan(RENAME), out)
    assert type(raised.value) is OSError and raised.value.errno == errno.ENOSPC
    assert _left(raised.value) == [] and not getattr(raised.value, "__notes__", None)
    assert files(tmp_path) == []


def test_low2_design_states_the_rollback_failure() -> None:
    text = (Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md").read_text()
    section = " ".join(
        text[text.index("- **Modify (S8).**") : text.index("- **Create (S9).**")].split()
    )
    assert "the placement error is still the one raised" in section
    assert "(kind `own-label`)" in section
