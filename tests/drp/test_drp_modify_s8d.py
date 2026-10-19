"""S8d: follow-ups from the S8c validation (round 3 LOW-1..5, round 2 L1-d, L2-a, L3-a, L4-a).

Each section names the item of the S8d brief it pins. They run on the bundled templates only.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import threading
from pathlib import Path
from typing import Any

import pytest
from diff_testing import A1_ITEM, CLIP, H264, TIMELINE
from drp_helpers import needs_zstandard

from aar.drp.modify import engine, parse_plan, write
from aar.drp.modify.paths import PathMapper
from aar.drp.modify.write import SIDECAR_SUFFIX, edit_file
from aar.drp.modify.write import leftovers as stale_temp_files
from aar.drp.reader import read_drp

RENAME = {"op": "rename", "target": "project", "new_name": "x"}
HUMAN = '{"owner": "a human", "notes": "keep me"}'
R19 = H264.parent / "media-clip-r19.drp"
R19_ARCHIVE = "7fb9ee45-9553-40cf-8f54-41d4a5132216"


def plan(*edits: dict[str, Any]) -> Any:
    return parse_plan(json.dumps({"edits": list(edits)}))


def files(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def _left(outcome_or_exc: Any) -> list[tuple[str, str]]:
    """``(path, kind)`` of what a run left behind (absent on the base, which had no such field)."""
    return [(x.path, x.kind) for x in getattr(outcome_or_exc, "left_behind", [])]


def _no_hardlinks(src: Any, dst: Any, **kw: Any) -> None:
    raise OSError(errno.EPERM, "Operation not permitted")  # FAT/exFAT has no hard links


def _first_run(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "out.drp"
    assert edit_file(H264, plan(RENAME), out).status == "written"
    return out, Path(str(out) + SIDECAR_SUFFIX)


def _design_modify_section() -> str:
    text = (Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md").read_text()
    start = text.index("- **Modify (S8).**")
    return " ".join(text[start : text.index("- **Create (S9).**")].split())


def _squash(text: str | None) -> str:
    return " ".join((text or "").split())


# --------------------------------------------------------------------------- A: --force window


def test_a_a_foreign_label_written_before_the_snapshot_is_refused_and_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # r3 LOW-1 repro: a human label lands after the last live check and before the backup. The
    # stamp is now checked on the snapshot itself, so the label is refused, not replaced.
    out, sidecar = _first_run(tmp_path)
    old_out = out.read_bytes()
    real_backup = write._backup

    def human_writes_then_backup(directory: Path, side: Path) -> Any:
        side.unlink()
        side.write_text(HUMAN)
        return real_backup(directory, side)

    monkeypatch.setattr(write, "_backup", human_writes_then_backup)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert [r.code for r in outcome.refusals] == ["foreign-sidecar"]
    assert outcome.status == "refused" and outcome.output is None
    assert sidecar.read_text() == HUMAN and out.read_bytes() == old_out  # nothing written
    # The snapshot of a label that is not ours is kept and named, never deleted.
    [(path, kind)] = _left(outcome)
    assert kind == "label-snapshot" and Path(path).read_text() == HUMAN
    assert outcome.to_dict()["left_behind"][0]["path"] == path
    assert files(tmp_path) == sorted([out.name, sidecar.name, Path(path).name])


def test_a_a_label_written_in_place_after_the_check_survives_in_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After the check, a human rewrites the label in place (same file). The hard-linked snapshot
    # shares that file, so it no longer bears our stamp when the run ends: kept, not deleted.
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
    assert kind == "label-snapshot" and Path(path).read_text() == HUMAN


def test_a_a_label_written_as_a_new_file_in_the_last_window_is_lost_as_documented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The window that remains (DESIGN says so): a new label file between the snapshot check and
    # the os.replace is replaced; rename has no compare-and-swap. Pinned so the docs stay honest.
    out, sidecar = _first_run(tmp_path)
    real_place = write._place

    def human_new_file_then_place(tmp: Path, dest: Path, force: bool) -> None:
        if dest == sidecar:
            dest.unlink()
            dest.write_text(HUMAN)
        real_place(tmp, dest, force)

    monkeypatch.setattr(write, "_place", human_new_file_then_place)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert outcome.status == "written" and HUMAN not in sidecar.read_text()
    assert _left(outcome) == [] and stale_temp_files(out) == []


def test_a_a_good_force_run_deletes_its_snapshot_of_our_label(tmp_path: Path) -> None:
    out, sidecar = _first_run(tmp_path)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert outcome.status == "written" and _left(outcome) == []
    assert files(tmp_path) == [out.name, sidecar.name]


def test_a_a_snapshot_restored_after_a_failed_placement_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, sidecar = _first_run(tmp_path)
    old_label = sidecar.read_bytes()
    real_place = write._place

    def fail_on_the_output(tmp: Path, dest: Path, force: bool) -> None:
        if dest == out:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_place(tmp, dest, force)

    monkeypatch.setattr(write, "_place", fail_on_the_output)
    with pytest.raises(OSError, match="No space"):
        edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert sidecar.read_bytes() == old_label and files(tmp_path) == [out.name, sidecar.name]


def test_a_design_and_comments_state_the_remaining_force_window() -> None:
    section = _squash(_design_modify_section())
    assert "rename has no compare-and-swap" in section
    assert "is still replaced and lost" in section
    assert "`--force` replaces only the output and a label bearing that stamp;" not in section
    source = Path(write.__file__).read_text()
    assert "--force never replaces a label aar did not write" not in source
    assert "it does not close it: rename has no compare-and-swap" in source


# --------------------------------------------------------------------------- B and H: cleanup


def _failing_replace_after(action: Any, dest: Path, err: OSError | None = None) -> Any:
    real_replace = os.replace

    def replace(src: Any, dst: Any) -> None:
        if Path(dst) == dest:
            action(dst)
            raise err or OSError(errno.EIO, "I/O error (simulated)")
        real_replace(src, dst)

    return replace


def test_b_a_failed_replace_keeps_an_empty_file_another_writer_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # r3 LOW-2 repro: another writer deletes our placeholder and O_EXCL-creates its own file,
    # not yet filled. It is empty like our placeholder; it must not be deleted.
    dest = tmp_path / "out.drp"
    held: list[int] = []

    def create_empty(dst: Any) -> None:
        os.unlink(dst)
        held.append(os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))

    monkeypatch.setattr(os, "link", _no_hardlinks)
    monkeypatch.setattr(os, "replace", _failing_replace_after(create_empty, dest))
    tmp = write._temp(tmp_path, "out.drp", b"ours")
    with pytest.raises(OSError, match="simulated") as raised:
        write._place(tmp, dest, force=False)
    os.write(held[0], b"their content")
    os.close(held[0])
    assert dest.read_bytes() == b"their content"
    assert _left(raised.value) == [(str(dest), "placeholder")]


def test_b_a_failed_replace_keeps_a_placeholder_a_plain_writer_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out.drp"
    held: list[Any] = []

    def plain_open(dst: Any) -> None:
        held.append(open(dst, "wb"))  # noqa: SIM115 - held open: truncated, nothing written yet

    monkeypatch.setattr(os, "link", _no_hardlinks)
    monkeypatch.setattr(os, "replace", _failing_replace_after(plain_open, dest))
    tmp = write._temp(tmp_path, "out.drp", b"ours")
    with pytest.raises(OSError, match="simulated") as raised:
        write._place(tmp, dest, force=False)
    held[0].write(b"their bytes")
    held[0].close()
    assert dest.read_bytes() == b"their bytes"
    assert _left(raised.value) == [(str(dest), "placeholder")]


def test_b_a_non_empty_file_at_the_name_is_kept_and_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out.drp"

    def plain_write(dst: Any) -> None:
        Path(dst).write_bytes(b"theirs")

    monkeypatch.setattr(os, "link", _no_hardlinks)
    monkeypatch.setattr(os, "replace", _failing_replace_after(plain_write, dest))
    tmp = write._temp(tmp_path, "out.drp", b"ours")
    with pytest.raises(OSError, match="simulated") as raised:
        write._place(tmp, dest, force=False)
    assert dest.read_bytes() == b"theirs" and _left(raised.value) == []
    assert type(raised.value) is OSError  # nothing left behind: the original error, unwrapped


def _sharing_violation() -> OSError:
    return OSError(errno.EACCES, "The process cannot access the file (simulated sharing violation)")


def test_h_a_sharing_violation_leaves_an_empty_unlabelled_output_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # r2 L2-a: on Windows FAT/exFAT another process holding the placeholder open makes the
    # replace fail, and an unlink would fail too. The empty out.drp stays (unlabelled), and the
    # raised error names it so an agent can act on it.
    out = tmp_path / "out.drp"
    real_unlink = os.unlink

    def unlink(path: Any, *a: Any, **k: Any) -> None:
        if Path(path) == out:
            raise _sharing_violation()
        real_unlink(path, *a, **k)

    monkeypatch.setattr(os, "link", _no_hardlinks)
    monkeypatch.setattr(
        os, "replace", _failing_replace_after(lambda d: None, out, _sharing_violation())
    )
    monkeypatch.setattr(os, "unlink", unlink)
    with pytest.raises(OSError, match="sharing violation") as raised:
        edit_file(H264, plan(RENAME), out)
    assert raised.value.errno == errno.EACCES
    assert files(tmp_path) == [out.name] and out.read_bytes() == b""  # no label, empty file
    assert _left(raised.value) == [(str(out), "placeholder")]
    assert "no label" in raised.value.left_behind[0].message


def test_h_the_cli_names_the_left_behind_file_and_exits_74(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    out = tmp_path / "o" / "out.drp"
    out.parent.mkdir()
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"edits": [RENAME]}))
    monkeypatch.setattr(os, "link", _no_hardlinks)
    monkeypatch.setattr(
        os, "replace", _failing_replace_after(lambda d: None, out, _sharing_violation())
    )
    code = main(["drp", "edit", str(H264), "--plan", str(plan_file), "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 74
    result = json.loads(captured.out)
    assert [r["code"] for r in result["refusals"]] == ["output-error"]
    assert [(x["path"], x["kind"]) for x in result["left_behind"]] == [(str(out), "placeholder")]
    assert f"left in place: {out}" in captured.err
    assert files(out.parent) == [out.name]


def test_h_a_clean_failure_reports_nothing_left_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aar.cli import main

    out = tmp_path / "o" / "out.drp"
    out.parent.mkdir()
    out.mkdir()  # --force onto a directory: the replace fails, nothing is claimed
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"edits": [RENAME]}))
    code = main(["drp", "edit", str(H264), "--plan", str(plan_file), "--out", str(out), "--force"])
    result = json.loads(capsys.readouterr().out)
    assert code == 74 and result["left_behind"] == []


def test_h_design_and_cli_help_state_the_empty_leftover() -> None:
    from aar.drp.modify.cli import EXIT_STATUS_HELP

    section = _squash(_design_modify_section())
    assert "A failed replace therefore leaves an empty file at the name, with no label" in section
    assert "`left_behind`" in section
    assert "empty placeholder" not in section.split("A crash mid-move")[0]  # B: no such claim
    assert "left_behind" in EXIT_STATUS_HELP


# --------------------------------------------------------------------------- C: bounded read


def _stamp(out: Path, **over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "status": "unverified",
        "label": "unverified",
        "tool": "aar 0.0.0",
        "output": {"path": str(out), "sidecar": str(out) + SIDECAR_SUFFIX},
    }
    rec.update(over)
    return rec


def test_c_the_label_read_is_bounded_when_the_label_grew_after_the_size_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # r3 LOW-3: the cap used to be a stat() check followed by an unbounded read. Here the label is
    # already over the cap while every stat of it still reports its earlier, small size (it grew
    # after the stat). Only a read that itself stops at cap + 1 refuses it.
    cap = 4096
    monkeypatch.setattr(write, "_MAX_SIDECAR_BYTES", cap)
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    body = json.dumps(_stamp(out)).encode()
    sidecar.write_bytes(body + b" " * (8 * cap))  # still our valid stamp: JSON allows the padding
    ino = os.stat(sidecar).st_ino
    real_stat, real_fstat = os.stat, os.fstat

    def small(st: os.stat_result) -> os.stat_result:
        if st.st_ino != ino:
            return st
        fields = list(st)
        fields[stat.ST_SIZE] = len(body)
        return os.stat_result(fields)

    monkeypatch.setattr(os, "stat", lambda *a, **k: small(real_stat(*a, **k)))
    monkeypatch.setattr(os, "fstat", lambda fd: small(real_fstat(fd)))
    real_loads = json.loads
    parsed: list[int] = []

    def loads(data: Any, *a: Any, **k: Any) -> Any:
        parsed.append(len(data))
        return real_loads(data, *a, **k)

    monkeypatch.setattr(write.json, "loads", loads)
    assert write._is_our_sidecar(sidecar, out) is False
    assert all(n <= cap for n in parsed), parsed  # never parsed past the cap


def test_c_a_label_at_the_cap_with_the_honest_size_is_still_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    body = json.dumps(_stamp(out)).encode()
    sidecar.write_bytes(body + b" " * (4096 - len(body)))
    monkeypatch.setattr(write, "_MAX_SIDECAR_BYTES", 4096)
    assert write._is_our_sidecar(sidecar, out) is True
    with open(sidecar, "ab") as fh:
        fh.write(b" ")
    assert write._is_our_sidecar(sidecar, out) is False


def test_c_a_fifo_label_is_refused_without_blocking(tmp_path: Path) -> None:
    # The label is opened once with O_NONBLOCK: a FIFO at the name (even one that appeared after
    # an is_file() check) must not hang the run waiting for a writer.
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    os.mkfifo(sidecar)
    result: list[bool] = []
    worker = threading.Thread(target=lambda: result.append(write._is_our_sidecar(sidecar, out)))
    worker.daemon = True
    worker.start()
    worker.join(5)
    if worker.is_alive():  # unblock the reader so the thread ends, then fail
        fd = os.open(sidecar, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
        worker.join(5)
        pytest.fail("reading a FIFO label blocked")
    assert result == [False]


# --------------------------------------------------------------------------- D: test gaps


def _odd_entry(path: Path, kind: str, tmp_path: Path) -> None:
    if kind == "foreign-file":
        path.write_text(HUMAN)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "dangling-symlink":
        path.symlink_to(tmp_path / "nowhere.json")
    elif kind == "symlink-to-dir":
        (tmp_path / "adir").mkdir(exist_ok=True)
        path.symlink_to(tmp_path / "adir")


def _entry(path: Path) -> tuple[int, int]:
    st = os.lstat(path)
    return (stat.S_IFMT(st.st_mode), st.st_ino)


ODD = ["foreign-file", "directory", "fifo", "dangling-symlink", "symlink-to-dir"]


@pytest.mark.parametrize("with_output", [False, True], ids=["no-output", "output-present"])
@pytest.mark.parametrize("kind", ODD)
def test_d_a_foreign_label_is_refused_before_the_edit_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, with_output: bool
) -> None:
    # r3 LOW-4: the pre-check itself refuses (its own message), so the edit never runs. Kills
    # the precheck mutants the later snapshot check used to mask (lexists -> exists / is_file,
    # "only when the output exists", the branch removed).
    out = tmp_path / "o" / "out.drp"
    out.parent.mkdir()
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    if with_output:
        out.write_bytes(b"a human's output")
    _odd_entry(sidecar, kind, tmp_path)
    before = _entry(sidecar)

    def must_not_run(*a: Any, **k: Any) -> Any:
        raise AssertionError("the edit ran although the label was foreign")

    monkeypatch.setattr(engine, "prepare", must_not_run)
    outcome = edit_file(H264, plan(RENAME), out, force=True)
    assert [r.code for r in outcome.refusals] == ["foreign-sidecar"]
    assert "is already here" in outcome.refusals[0].message  # the pre-check's own wording
    assert _entry(sidecar) == before
    assert files(out.parent) == sorted([sidecar.name] + ([out.name] if with_output else []))


@pytest.mark.parametrize("kind", ["fifo", "dangling-symlink", "directory"])
def test_d_a_non_regular_label_that_appears_mid_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # Our label at the pre-check; while the edit runs it is replaced by a non-regular entry.
    # The check just before placement must refuse it (kills F-recheck-is_file).
    out, sidecar = _first_run(tmp_path)
    old_out = out.read_bytes()
    real_verify = write.verify_write

    def verify_then_swap(*a: Any, **k: Any) -> Any:
        res = real_verify(*a, **k)
        sidecar.unlink()
        _odd_entry(sidecar, kind, tmp_path)
        return res

    monkeypatch.setattr(write, "verify_write", verify_then_swap)
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert [r.code for r in outcome.refusals] == ["foreign-sidecar"]
    assert "while the edit ran" in outcome.refusals[0].message
    assert os.path.lexists(sidecar) and not sidecar.is_file() and out.read_bytes() == old_out
    assert stale_temp_files(out) == []  # nothing to snapshot, no temporary file left


def _replace_fails_after_swapping_in(make: Any, dest: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    def swap(dst: Any) -> None:
        os.unlink(dst)
        make(dst)

    monkeypatch.setattr(os, "link", _no_hardlinks)
    monkeypatch.setattr(os, "replace", _failing_replace_after(swap, dest))


def test_d_the_cleanup_never_follows_a_symlink_to_our_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _unlink_if_ours matches our temporary's inode with lstat: a symlink pointing at the
    # temporary has its own inode, so it is neither followed nor deleted (kills lstat -> stat).
    dest = tmp_path / "out.drp"
    tmp = write._temp(tmp_path, "out.drp", b"ours")
    _replace_fails_after_swapping_in(lambda d: os.symlink(tmp, d), dest, monkeypatch)
    with pytest.raises(OSError, match="simulated") as raised:
        write._place(tmp, dest, force=False)
    assert dest.is_symlink() and os.readlink(dest) == str(tmp)
    assert _left(raised.value) == []


@pytest.mark.parametrize("kind", ["fifo", "empty-dir"])
def test_d_the_cleanup_touches_only_regular_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # An empty non-regular entry at the name is neither deleted nor reported as our placeholder
    # (kills dropping S_ISREG from _unlink_if_ours).
    dest = tmp_path / "out.drp"
    make = os.mkfifo if kind == "fifo" else os.mkdir
    _replace_fails_after_swapping_in(make, dest, monkeypatch)
    tmp = write._temp(tmp_path, "out.drp", b"ours")
    with pytest.raises(OSError, match="simulated") as raised:
        write._place(tmp, dest, force=False)
    assert os.path.lexists(dest) and not dest.is_file() and not dest.is_symlink()
    assert _left(raised.value) == []


# --------------------------------------------------------------------------- E: doc nits


def test_e_the_temporarys_identity_is_read_before_the_exclusive_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_stat, real_open = os.stat, os.open

    def spy_stat(path: Any, *a: Any, **k: Any) -> Any:
        calls.append("stat")
        return real_stat(path, *a, **k)

    def spy_open(path: Any, flags: int, *a: Any, **k: Any) -> Any:
        if flags & os.O_EXCL:
            calls.append("claim")
        return real_open(path, flags, *a, **k)

    tmp = write._temp(tmp_path, "out.drp", b"ours")
    monkeypatch.setattr(os, "stat", spy_stat)
    monkeypatch.setattr(os, "open", spy_open)
    write._place_no_hardlink(tmp, tmp_path / "out.drp")
    assert calls[:2] == ["stat", "claim"]
    assert (tmp_path / "out.drp").read_bytes() == b"ours"


def test_e_writer_docs_name_the_window_and_the_writer_correctly() -> None:
    doc = _squash(write._place_no_hardlink.__doc__)
    assert "does not create exclusively" in doc
    assert "two-syscall" not in doc and "two-adjacent-syscalls" not in doc
    assert "from the exclusive create through its close to the replace" in doc
    section = _squash(_design_modify_section())
    assert "two-syscall" not in section
    assert "from the exclusive create through its close to the replace" in section


def test_e_ntfs_upper_docstring_names_the_unicode_versions_and_the_bmp_count() -> None:
    from aar.drp.modify.paths import ntfs_upper

    doc = _squash(ntfs_upper.__doc__)
    assert "These are Unicode 1.1 characters" not in doc
    assert "Georgian" in doc and "Mtavruli" in doc and "Cherokee" in doc
    assert "246" in doc
    for version in ("3.1", "7.0", "9.0"):  # Deseret, Warang Citi, Osage and Adlam
        assert version in doc


# --------------------------------------------------------------------------- F: Windows labels


WINDOWS_FORMS = {
    "drive": "E:\\Media\\{name}",
    "drive-forward-slash": "E:/Media/{name}",
    "unc": "\\\\nas\\share\\Media\\{name}",
    "rooted": "\\Media\\{name}",
}


@pytest.mark.parametrize("form", list(WINDOWS_FORMS))
def test_f_a_label_aar_wrote_on_windows_is_ours_on_posix(tmp_path: Path, form: str) -> None:
    # r2 L1-d: the stamp records the output path as Windows wrote it; read with POSIX rules the
    # whole string was the "name", so the label was refused as foreign on a shared folder.
    out, sidecar = _first_run(tmp_path)
    recorded = WINDOWS_FORMS[form].format(name=out.name)
    label = json.loads(sidecar.read_text())
    label["output"]["path"] = recorded
    label["output"]["sidecar"] = recorded + SIDECAR_SUFFIX
    sidecar.write_text(json.dumps(label))
    assert write._is_our_sidecar(sidecar, out) is True
    outcome = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert outcome.status == "written", outcome.to_dict()["refusals"]


@pytest.mark.parametrize(
    "over",
    [
        {"output": {"path": "E:\\Media\\other.drp", "sidecar": "E:\\Media\\out.drp.aar.json"}},
        {"output": {"path": "E:\\Media\\out.drp", "sidecar": "E:\\Media\\other.drp.aar.json"}},
        {
            "tool": "aardvark 1",
            "output": {"path": "E:\\out.drp", "sidecar": "E:\\out.drp.aar.json"},
        },
        {"status": "written", "output": {"path": "E:\\out.drp", "sidecar": "E:\\out.drp.aar.json"}},
        {"output": {"path": "sub\\out.drp", "sidecar": "sub\\out.drp.aar.json"}},  # relative
    ],
    ids=["other-output", "other-label", "tool", "status", "relative-windows"],
)
def test_f_the_rest_of_the_stamp_still_has_to_match(tmp_path: Path, over: dict[str, Any]) -> None:
    out = tmp_path / "out.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    sidecar.write_text(json.dumps(_stamp(out, **over)))
    assert write._is_our_sidecar(sidecar, out) is False


def test_f_a_posix_output_name_with_a_backslash_is_not_misread(tmp_path: Path) -> None:
    # A POSIX name may contain "\". Our own label for "a\b.drp" is ours for "a\b.drp" ...
    out = tmp_path / "a\\b.drp"
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    assert edit_file(H264, plan(RENAME), out).status == "written"
    assert write._is_our_sidecar(sidecar, out) is True
    again = edit_file(H264, plan({**RENAME, "new_name": "y"}), out, force=True)
    assert again.status == "written", again.to_dict()["refusals"]
    # ... and never for "b.drp": a POSIX-rooted path is not split at the backslash.
    other = tmp_path / "b.drp"
    other_label = Path(str(other) + SIDECAR_SUFFIX)
    other_label.write_bytes(sidecar.read_bytes())
    assert write._is_our_sidecar(other_label, other) is False


@pytest.mark.parametrize(
    ("recorded", "name"),
    [
        ("/srv/acme/a\\b.drp", "a\\b.drp"),
        ("E:\\acme\\out.drp", "out.drp"),
        ("e:/acme/out.drp", "out.drp"),
        ("\\\\nas\\share\\out.drp", "out.drp"),
        ("out.drp", "out.drp"),
        ("sub\\out.drp", "sub\\out.drp"),  # relative: this host's (POSIX) rules
        ("C:out.drp", "C:out.drp"),  # no separator after the drive: not a Windows root here
    ],
)
def test_f_recorded_names_are_read_with_the_writers_rules(recorded: str, name: str) -> None:
    assert write._recorded_name(recorded) == name


# --------------------------------------------------------------------------- G: double renames


TARGETS: dict[str, dict[str, str]] = {
    "project": {"target": "project"},
    "clip": {"target": "clip", "id": CLIP},
    "timeline": {"target": "timeline", "id": TIMELINE},
    "item": {"target": "item", "id": A1_ITEM},
}


@pytest.mark.parametrize("names", ["there-and-back", "noop-then-real", "two-names"])
@pytest.mark.parametrize("target", ["project", "clip", "timeline", "item"])
def test_g_two_renames_of_one_object_conflict_whatever_the_names(
    tmp_path: Path, target: str, names: str
) -> None:
    # r2 L4-a: [-> "B", -> <original>] was written with the name B (the second rename compared
    # with the input's name looked like a no-op), while [-> B, -> C] was refused.
    sel = TARGETS[target]
    drp = read_drp(H264)
    current = {
        "project": drp.project.name,
        "clip": drp.media_by_id()[CLIP].name,
        "timeline": next(t.name for t in drp.timelines if t.id == TIMELINE),
        "item": next(
            item.name
            for t in drp.timelines
            for tr in t.tracks
            for item in tr.items
            if item.id == A1_ITEM
        ),
    }[target]
    first, second = {
        "there-and-back": ("B", current),
        "noop-then-real": (current, "B"),
        "two-names": ("B", "C"),
    }[names]
    out = tmp_path / "out.drp"
    outcome = edit_file(
        H264,
        plan(
            {"op": "rename", **sel, "new_name": first},
            {"op": "rename", **sel, "new_name": second},
        ),
        out,
    )
    assert [(r.code, r.edit) for r in outcome.refusals] == [("conflicting-edits", 1)]
    assert outcome.status == "refused" and not out.exists()


def test_g_a_name_selector_and_an_id_selector_for_one_object_conflict(tmp_path: Path) -> None:
    name = read_drp(H264).media_by_id()[CLIP].name
    outcome = edit_file(
        H264,
        plan(
            {"op": "rename", "target": "clip", "id": CLIP, "new_name": "B"},
            {"op": "rename", "target": "clip", "name": name, "new_name": name},
        ),
        tmp_path / "out.drp",
    )
    assert [(r.code, r.edit) for r in outcome.refusals] == [("conflicting-edits", 1)]
    assert outcome.refusals[0].details == {"edits": [0, 1]}


def test_g_two_renames_of_one_bin_conflict(tmp_path: Path) -> None:
    outcome = edit_file(
        R19,
        plan(
            {"op": "rename", "target": "bin", "id": R19_ARCHIVE, "new_name": "B"},
            {"op": "rename", "target": "bin", "id": R19_ARCHIVE, "new_name": "Archive"},
        ),
        tmp_path / "out.drp",
    )
    assert [(r.code, r.edit) for r in outcome.refusals] == [("conflicting-edits", 1)]


def test_g_renames_of_different_objects_still_combine(tmp_path: Path) -> None:
    out = tmp_path / "out.drp"
    outcome = edit_file(
        H264,
        plan(
            {"op": "rename", "target": "project", "new_name": "P"},
            {"op": "rename", "target": "clip", "id": CLIP, "new_name": "C"},
            {"op": "rename", "target": "timeline", "id": TIMELINE, "new_name": "T"},
        ),
        out,
    )
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    assert sorted(c["new"] for c in outcome.changes) == ["C", "P", "T"]


# --------------------------------------------------------------------------- I: uncertain folds


@pytest.mark.parametrize(
    ("row", "path", "uncertain"),
    [
        ("E:\\Media\\I", "E:\\Media\\\u0131\\clip.mov", True),  # dotless i: NTFS keeps it apart
        ("E:\\Media\\\u0131", "E:\\Media\\I\\clip.mov", True),
        ("E:\\Media\\\u00e4", "E:\\Media\\\u00c4\\clip.mov", True),  # a real fold, still flagged
        ("E:\\Media\\I", "e:\\media\\i\\clip.mov", False),  # ASCII case only
        ("E:\\Media\\\u00e4", "E:\\MEDIA\\\u00e4\\clip.mov", False),  # same non-ASCII spelling
        ("E:\\Media", "E:\\Media\\\u0131\\clip.mov", False),  # the fold is past the prefix
    ],
)
def test_i_a_prefix_match_that_needed_a_non_ascii_fold_is_uncertain(
    row: str, path: str, uncertain: bool
) -> None:
    result = PathMapper("windows_to_mac", [(row, "/Volumes/M")]).map(path)
    assert result.new is not None and result.new.endswith("clip.mov")
    assert result.uncertain is uncertain


def test_i_mac_to_windows_matches_are_exact_so_never_uncertain() -> None:
    result = PathMapper("mac_to_windows", [("E:\\\u0131", "/Volumes/M")]).map("/Volumes/M/a.mov")
    assert result.new == "E:\\\u0131\\a.mov" and result.uncertain is False


@needs_zstandard
def test_i_the_path_map_report_marks_an_uncertain_match(tmp_path: Path) -> None:
    # The validator's alias: paths under E:\<dotless i> are mapped by a row for E:\I. The edit
    # is still made (the gate cannot see the directory semantics), and the report says to check.
    there = [
        {"windows": "E:\\\u0131", "mac": "/Users/samuelgursky/Desktop"},  # the clip blobs
        {"windows": "F:\\\u0131", "mac": "/Media"},  # the timeline items' MediaFilePath
    ]
    first = edit_file(
        H264,
        plan({"op": "path_map", "direction": "mac_to_windows", "table": there}),
        tmp_path / "win.drp",
    )
    assert first.status == "written"
    assert all("uncertain" not in t for t in first.report[0]["touched"])
    back = [{"windows": "E:\\I", "mac": "/Volumes/M"}, {"windows": "F:\\I", "mac": "/Volumes/F"}]
    outcome = edit_file(
        tmp_path / "win.drp",
        plan({"op": "path_map", "direction": "windows_to_mac", "table": back}),
        tmp_path / "mac.drp",
    )
    assert outcome.status == "written", outcome.to_dict()["refusals"]
    touched = outcome.report[0]["touched"]
    assert touched and all(t["uncertain"] == engine.UNCERTAIN_FOLD for t in touched)
    assert {t["object"] for t in touched} == {"media", "item"}


def test_i_paths_module_still_has_the_shapes_s13f_couples_to() -> None:
    from aar.drp.modify import paths

    assert re.search(
        r"def _parse_win\(path: str\) -> _Win \| None:", Path(paths.__file__).read_text()
    )
    win = paths._parse_win("e:\\a\\b")
    assert win is not None and win.key() == ("E:", "A", "B")
    assert paths.ntfs_upper("\u0131") == "I"
