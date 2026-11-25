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

## Second route: a project in the shared Resolve library (S25d)

Besides the FCPXML, a round can also be handed over as a project in a Resolve disk project
library on the hub's share (decision D10). It is an addition, not a replacement: the FCPXML,
review video and notes are published exactly as above, and they stay the dependable route
until the owner has confirmed library projects in Resolve.

### For the owner

1. Read the round's notes. Under **OR OPEN IT AS A LIBRARY PROJECT** they name the project
   library (as it is called in your Project Manager), the user, the project and its timeline.
   If they say **LIBRARY PROJECT: NOT WRITTEN**, the reason and what to do follow; import the
   FCPXML as usual.
2. In Resolve open the Project Manager, choose that project library and open the project.
3. Edit the timeline, then save the project (File > Save Project, or Ctrl+S).

Saving is all that is needed. There is nothing to export: while the round's review is open the
hub notices each save by itself (normally within a few minutes) and records it for review, one
record per save. Keep the project's folder name and the timeline's name as they are; a project
renamed or deleted, a timeline renamed, or another project saved under the same folder name is
reported to the agent and not imported.

What you will see for a few kinds of save:

- **A save that brings the project back to how it was at an earlier save** (you undid your
  edits and saved again), **or back to how the agent placed it**: nothing new is recorded. The
  hub tells saves apart by the file's content, and it has already seen that content. The
  agent's newest record is then still the later save you undid, not what is in the project
  now; tell the agent if you went back.
- **A save that changes nothing in the timeline** (you renamed the project inside Resolve,
  added a second timeline, or saved without changing anything that ends up in the file
  differently): it is recorded as a save like any other, with no change to the cut (every
  segment "unchanged").
- **A project that keeps being saved** (Resolve's auto-save every few seconds): the hub waits
  for the file to stay still for a moment; if it never does for about ten minutes, the agent is
  told the project is busy, once, and the hub keeps looking.

### For an agent

- **Measure the media on the edit PC first.** The library writer authors each clip from facts
  measured on the PC's own file. On the PC, with this repository's `src` on `PYTHONPATH`:

  ```bat
  python -m aar.media.facts E:\Show\Proxy --out Z:\incoming\Show\facts.jsonl
  ```

  One JSON line per file; a file it could not measure is a line with its error (exit 2). See
  [Media store](MEDIA-STORE.md#facts-for-library-projects-s25d).
- **Parameters.** Add `library` to the `ccn-longform` params: `folder` (the library folder on
  the hub, the one holding `Resolve Projects`), `seed` (a `Project.db` the owner saved, whose
  timeline has the cut's frame rate, size and start), `facts` (the file above, as the hub sees
  it), optionally `templates` (more saved `Project.db` files), `user`, `label` (the library's
  name in the owner's Project Manager) and `seed_resolution`. Two opt-ins default to off:
  `allow_unopened_operations` (write operations nobody has yet seen Resolve open; every real
  export needs it until the staging projects are confirmed) and `allow_unverified_media` (clips
  of a kind or size the evidence does not cover). With them off a cut that needs them is
  refused, and the refusal is in the notes. Every path must be inside `AAR_MEDIA_ROOT` when it
  is set. With `AAR_MEDIA_ROOT` unset nothing restricts where the library may be: the export
  places the project and the pickup reads saves wherever `library.folder` points (a follow-up;
  set `AAR_MEDIA_ROOT` on the hub).
- **What the notes say.** Placed: the library, user, project and timeline to open, that saving
  is all, the operations not yet seen working and the clips without full evidence. Refused:
  the export's code, message and fix hint. The export step's result carries the same as
  `library_handoff` (with `warnings` on a refusal); the step itself succeeds on the FCPXML.
- **How a save shows up.** In `recipe_status`, under `returned_edits`, like a `.drt` return,
  with `source: "library_save"`, the saved `Project.db` (`project_db`), the edited plan and
  the diff. Why a save was not taken (the project missing, being saved, replaced, damaged, the
  timeline renamed, the library gone) is under `library_pickup`, once per condition. The same
  rules as for `.drt` returns: pickup never replaces the active plan or decides a review.

### Not carried, and limits

A ccn cut of more than one continuous range has an audio crossfade at each join, which a
library project cannot carry yet: such a cut is refused and goes by FCPXML only. Titles,
transitions, fades, other tracks, gain, framing, markers and captions are not written (the
notes count the plan details left out). Saves are picked up only while the round's review is
open; after a decision or a later round they are not. The project's file is read up to 256 MiB.

Nothing the library route does can lose the FCPXML hand-off: any fault of the library, the seed,
the facts file or the hub's own disk while writing the library project is the refusal
`library_failed` (with the error) in the notes and the step's result, and the FCPXML, video and
notes are published as usual. One limit is not handled yet (S25d-f5): a library, seed or facts
file on a share that stops answering (a hung mount) holds the export step before anything is
published, and holds the worker's pickup, for as long as the share does not answer. If a round's
export does not finish, check the share.

**The facts file.** `aar media facts` lists, in its last printed line, what a folder held that
it did not measure (hidden files, other suffixes such as `.mkv` or `.jpg`, subfolders, broken
links) and why; name such a file to measure it anyway. A folder with nothing to measure is
refused (`no_media`, exit 2). A whole run stops after `--deadline` seconds (30 minutes by
default) and Ctrl+C stops it too (exit 130); either way the file is written with what was
measured, and the rest are listed. Its printed lines are ASCII, so any console shows them; the
file itself is UTF-8.
