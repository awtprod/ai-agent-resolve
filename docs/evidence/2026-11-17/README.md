# Desktop handoff verification

Branch: `impl/desktop-handoff`, based on `dev` at `6be2f4d`.

The CCN export code path now verifies per-media receipts before publication. The existing
cold worker discovers returned DRTs only from runner-recorded export folders and surfaces
immutable source/plan/diff artifacts in recipe status. No alternative transfer system,
Resolve process, direct database connection, or approval bypass was added.

## Evidence

- Baseline: handoff + media-shrink tests, **108 passed**.
- Receipt and return tests failed before their modules existed.
- End-to-end and review-regression focused run: **23 passed in 74.56s**.
- Final receipt checks, including real ffmpeg video/audio shrink through the CLI:
  **13 passed in 9.88s**.
- `PYTHONPATH=src python3 -S -m aar.media.receipt --help` exits 0: standalone helper needs
  only Python's standard library (ffmpeg/ffprobe are external tools for shrink verification).
- `AAR_LIVE=0 make check` at `f6e92e9`: exit 0; Ruff lint and formatting passed (560 files);
  **8,872 passed, 134 skipped in 1232.92s (20m32s)**.
  Captured command output: [local-checks.txt](local-checks.txt).

The integration test creates synthetic camera media, runs the real recipe worker/export,
writes a synthetic Resolve-format DRT into that export attempt's return folder, runs the
worker again, and observes a changed reconstructed plan in `recipe_status.returned_edits`.
It proves the active plan is unchanged and a second pickup creates no duplicate record.
This is the production command/worker path with synthetic media, not a live Resolve export.

## Independent review

A fresh GPT-6 Astra reviewer found four issues and independently reproduced them. All have
regression tests observed failing before their fixes:

1. A symlink return prevented later files from being picked up: per-file/folder failures now
   produce visible, deduplicated diagnostics and the sweep continues.
2. A FIFO receipt could block an export: evidence must be a regular file, checked before
   opening and against the opened descriptor; POSIX opens are nonblocking and no-follow.
3. A source changed with restored mtime could receive a false receipt: change time and
   descriptor identity are checked, and both hashes are rechecked after verification.
4. A cancellation during import could still publish the returned plan: cancellation is
   rechecked in the final append transaction.

## Live limits and rollout

No deployment, live project writes, live Resolve import/export, or production receipt writes
were performed. The Windows/Resolve round trip remains an environment gate; the existing
reader's partial reconstruction limits remain explicit. Windows receipt tests were added to CI.

A rollout needs the AAR hub application/worker and the Windows `src` helper updated, and
receipts generated for existing proxy pairs before their next export. Existing runs without
receipts stop at export with a recovery instruction. Originals/masters remain on Windows;
proxies remain on both machines. Importing the delivered FCPXML and exporting a DRT remain
explicit Resolve actions; pickup from the shared return folder is automatic while the worker
runs. No currently running service was restarted.

## Windows CI follow-up

The first Windows/Python 3.12 CI run rejected unchanged files because `lstat().st_ctime_ns`
and `fstat().st_ctime_ns` can mean birth time and change time respectively
([CPython issue #157671](https://github.com/python/cpython/issues/157671)). The guard now
compares change timestamps through the same path API, and separately checks the opened
file's identity, size and modification time. It still refuses a replaced or changed file.

Two new regression tests failed before the correction. The expanded tests also reject
changed descriptor identity, size and modification time. After the correction, Ruff lint
and formatting passed, and the focused receipt, return-pickup and recipe end-to-end run
passed **27 tests in 63.99s**. The fresh Windows CI run remains required; no local Windows
or live Resolve validation is claimed.
