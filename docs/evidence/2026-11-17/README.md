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

The initial implementation was verified offline. The authorized JVL3RP2 roundtrip below now
covers Windows receipts and native Resolve import/render/export. No service was deployed or
restarted; the existing reader's partial reconstruction limits remain explicit.

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
passed **27 tests in 63.99s**. [Windows CI passed at `b017142`](https://github.com/awtprod/ai-agent-resolve/actions/runs/44130274584/job/134205119620).
That CI correction preceded the live Windows validation below.

## Prism directory-swap finding

Prism's HIGH finding on `b017142` was reproduced twice: replacing either the return folder
or its ancestor with a symlink after enumeration stored bytes from an outside file under
the receiving project. Both regression tests failed before the correction.

Pickup now opens each directory relative to its already-open parent with `O_DIRECTORY`
and `O_NOFOLLOW`. Enumeration and file opens use the retained directory descriptor;
`receive_one` requires that descriptor and only accepts a single file name. Snapshot
metadata checks use the same opened file, and all directory handles close on failure too.
There is no pathname fallback on platforms lacking the required POSIX handles.

After the correction, the receipt, return-pickup and recipe end-to-end tests passed
**31 tests in 58.81s**, with Ruff lint/format checks clean. Tests prove neither directory-swap
variant stores the outside bytes, while the originally enumerated DRT is still collected;
they also cover pre-existing symlink refusal, descriptor cleanup, and absolute/parent-path
rejection with temporary-file cleanup. Fresh CI and Prism review remain required.

## Authorized Resolve roundtrip on JVL3RP2

On 2026-11-17 Andrew authorized saving the open project, testing in a temporary project, and
restoring it. Resolve **21.1.0.17** ran in its interactive Windows session. The existing
300-second presence guard passed; approval grants and consumes were recorded in an isolated
event log. The project library was not switched. The existing node stayed disabled and the
MCP bridge stayed running.

The test used the PR's source and a separate hub/worker with synthetic media, through the
actual SMB share. The fixture supplied its known transcript; this was not an ASR test.
Windows generated the receipt against the PC file and its Linux copy. Their SHA-256 hashes
matched. The real CCN worker published FCPXML, review video, notes, and a return folder.

Resolve imported two video segments totaling **160 frames** at **30000/1001 fps**, linked
to the PC copy. Removing the last video segment and its matching sound left **95 video
frames** and **96 audio frames** (the fixture's outgoing audio handle). Resolve rendered
96 frames at 640×360 with stereo 48 kHz AAC, then exported a **30,900-byte DRT** to the
shared return folder and a **44,301-byte native DRP** locally. FFmpeg decoded the render
without errors; its audio mean/peak levels were **−22.8/−11.2 dB**. The still below is
exported by Resolve from the first remaining source frame.

![Synthetic media frame exported by Resolve](resolve-frame.png)

The original project, timeline, bin, playhead, and Edit page were restored and checked again
later through position-only reads. Its timeline count remained **32**. The scratch project
was deleted. Two test-driver corrections were needed: matching the overlapping sound when
deleting the last segment, and avoiding `GetCurrentRenderMode` during restoration because
that known Resolve API side effect can asynchronously open the Deliver page.

The first automatic pickup exposed a product defect: native whole-frame source positions
carried a `2.7675639557855902e-12` remainder, and a 96-frame sound duration was serialized as
`95 + 0.9999999999977263`. The importer treated these as unsupported sub-frame edits and
could retain an old segment instead of a real trim. Import now snaps only within **1e-9 of
a frame boundary**, carrying near-one remainders and recomputing ends in a copy. The forensic
reader's raw values and genuine sub-frame warnings remain unchanged.

Six regressions failed before this correction; all 15 rounding tests passed afterward.
The importer, desktop-return, recipe end-to-end, and receipt suite passed **356 tests in
62.37s**; repository Ruff lint/format checks passed. Independent review found no issues.
Native DRT/DRP files remain in the ignored `corpus/pr34-live-20261117T173926Z/`.

A fresh recipe produced byte-identical FCPXML, then the corrected worker automatically
collected the actual native DRT copied from Windows. It classified the first segment as
unchanged and the second as removed, without false sub-frame warnings. The remaining
one-frame audio handle correctly produced audio-only-edit warnings. The result stayed
`needs_review`, the active plan stayed unchanged, and another sweep added no events.
The deterministic measurements, hashes, and restoration checks are recorded in
[live-roundtrip.json](live-roundtrip.json).

This demonstrates native handoff and return on this Resolve version, not deployment of the
worker or simultaneous collaboration. Resolve authored the DRP; an offline DRP writer and
shared-library collaboration were not exercised. Audio-only edits, disabled clips, titles,
grades, Fusion, and Fairlight still require the importer's explicit partial-reconstruction
review. No returned plan is automatically approved or substituted for the active plan.
