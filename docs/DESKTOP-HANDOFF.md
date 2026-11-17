# Desktop handoff

The hub uses proxy copies; originals and final renders stay on Windows. The edit PC's
mapped share is the delivery surface. Resolve imports a timeline, not a new `.drp` project.
The optional live Resolve worker is not needed for this workflow.

## Stage media once, with evidence

On Windows, with the current repository's `src` on `PYTHONPATH` and ffmpeg/ffprobe on PATH:

```bat
set PYTHONPATH=C:\aar\src
python -m aar.media.shrink E:\Show\Proxy --out Z:\incoming\Show\Proxy --receipt
```

This retains the PC proxies and creates verified smaller hub copies with lossless audio.
`--receipt` also writes `<proxy>.aar-media.json` beside each shared copy. Rerunning verifies
and reuses completed copies. Originals are never replaced. `--receipt` refuses `--replace`.

For files already copied (or shrunk without receipts), run once per file on the PC:

```bat
python -m aar.media.receipt E:\Show\Proxy\A001.mp4 --copy Z:\incoming\Show\Proxy\A001.mp4
```

Identical copies are checked by SHA-256. Smaller copies use the existing shrink verifier,
including frame count, rate, timecode, picture samples and all decoded audio streams.
A receipt binds both file hashes, sizes, PC source path and capture time. It is a snapshot:
regenerate it if the PC source changes or moves. It cannot detect a later change on an
unconnected PC. Receipts are trusted evidence from the same protected share as the media.

Before publishing a CCN timeline, the export step checks every linked hub file against its
receipt and the mapped PC path. Missing or mismatched receipts fail export with a concrete
recovery instruction. **Existing runs need receipts before their next export.** Retry the
export after recording them; no re-transcription is required.

## Open the delivered timeline

Each round still arrives in `exports/<project>/<timeline>/` as FCPXML, review video and notes.
Read the notes and use Resolve's File > Import > Timeline to import the FCPXML into your
chosen project. The FCPXML is published last, after the accompanying files. The notes say
what was verified and which finishing work remains in Resolve.

## Send edits back through the same share

The notes identify a return folder relative to `exports`, also reported as
`desktop_return.directory` by `recipe_status`. It is unique to the recipe, round and export
attempt. After editing, use File > Export > Timeline in Resolve and save a **new `.drt`**
there. Keep every version under a different name. There is no email/upload or server command.

The existing hub cold worker checks return folders once a minute between jobs. It waits for
files to settle, snapshots them, and records the original DRT, an edited plan, and a diff as
immutable artifacts. Recipe status exposes the latest 20 versions under `returned_edits`,
even after a run finishes. Cancelled runs are not collected. The agent's guide directs it to
check these records before continuing. Restarting or running multiple workers does not
create duplicate records for the same base plan and DRT content.

Every import needs review: the existing DRT reader cannot fully reconstruct enabled/disabled
clip state, grades, Fusion, markers or Fairlight work. The diff carries its reader limits and
any `not_carried` changes. **Pickup never replaces the active plan or approves a review.**
The agent must inspect the returned plan/diff with those limitations before continuing.

A malformed return is recorded as failed; save a corrected DRT as a new file. Each DRT is
limited to 64 MiB, each return folder to 100 entries, and a sweep to 100,000 matching history
events. These limits fail visibly rather than silently truncating imports.
