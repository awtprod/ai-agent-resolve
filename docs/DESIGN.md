# AI-Agent-Resolve: design

Draft v1, 2026-10-17. Slices and status are tracked in [`PLAN.md`](../PLAN.md).

## 1. What this is

AI-Agent-Resolve is a companion app for DaVinci Resolve Studio, built mainly for AI agents to use. Agents plan, measure, build and check edits. Humans supervise and review, and keep working in Resolve as usual.

It does not reimplement Resolve. Resolve remains the renderer, the timeline of record and the collaboration surface. This app adds the things Resolve lacks for agent work:

- A typed edit-plan layer that is checked against house rules before anything touches Resolve.
- A scheduler in front of the single Resolve instance.
- An offline toolchain for `.drp` files.
- A shared store of projects, events and artifacts that humans and agents both see.
- Recipes that take a job, such as a CCN long-form or a set of shorts, from brief to published files.

## 2. Facts that shape the design

Status `docs` means the evidence is Blackmagic documentation or the 21.1 API stubs. `live` means it was observed on the edit PC. `UNVERIFIED` means a spike is planned.

| #   | Fact | Status |
| --- | ---- | ------ |
| F1  | Blackmagic Cloud has no public API. All cloud work goes through a Resolve instance's scripting API: `LoadCloudProject`, `CreateCloudProject`, `ImportCloudProject`, `RestoreCloudProject`, and the clip property `Cloud Sync`. Presentations (review in the browser) have no API either. The only Blackmagic REST API is the camera's own. | docs |
| F2  | The edit PC's current project library is a Blackmagic Cloud library with 33 projects, including QCCE 2024, the LNR episodes and Roy Cooper. `GetDatabaseList()` does not list it. It lists only `Local Database`, which has 0 projects. | live 2026-10-17 |
| F3  | A Resolve instance has one current project and one current timeline, and most edit calls act on that current timeline. Dynamic Project Switching keeps several projects in RAM, but the API cannot list them. | docs |
| F4  | It is unknown whether two GUI instances can run under one Windows user; community answers say no. `-nogui` headless mode and `-rr` remote render nodes both exist, but it is unknown whether either can run beside the GUI on the same PC. | UNVERIFIED (S14) |
| F5  | Collaboration is a Resolve feature on cloud and network libraries: bin, timeline and clip locks, Live Save, chat, and Compare Timelines. The API exposes only `RefreshFolders`, `Folder.GetIsFolderStale` and `liveCollaborationMode`. There is no API for locks, presence or chat. | docs |
| F6  | A `.drp` is a ZIP of XML that mirrors database tables. Binary state is stored as hex blobs (keyed dicts, protobuf, zstd). Import is version-gated: stamps must not be newer than the host. Edits can import "successfully" and still render nothing, and a renamed SeqContainer gives an empty timeline with no error. | prior art; our own checks pending |
| F7  | Resolve writes DRP, DRT and DRB files itself (`ExportProject`, `Timeline.Export` for DRT/FCPXML/OTIO/AAF/EDL, `Folder.Export`). It also imports them (`ImportProject`, `ImportTimelineFromFile`, `ImportFolderFromFile`). | docs |
| F8  | 21.1 adds scripted multicam create and flatten, transitions, fades, speed, audio normalize and a native MCP server. Advanced scripting now requires Studio. | docs |
| F9  | Today's MCP bridge stops any script after 60 s, so renders and long analyses must run as jobs and be polled. | live |
| F10 | Several agent threads and Andrew share the one Resolve instance on the edit PC, and threads have already collided (2026-10-11). | live |

## 3. Architecture

```
 agents (Command Center threads, Codex, CLI)            humans
        │ MCP (streamable HTTP) / REST / CLI                │ web console: review, approve, override
        ▼                                                   ▼
 ┌──────────────────────────── aar-hub (tailnet) ─────────────────────────────┐
 │ project registry · append-only event log · work graph                      │
 │   (brief → plan → ops → artifacts → reviews)                               │
 │ leases + presence · review threads at timecode · artifact store (by hash)  │
 │ per-project facts (offsets, gains, traps) · rule profiles · recipe runner  │
 └──────────┬──────────────────────────────────────────────┬──────────────────┘
            │ hot jobs (node pulls)                        │ cold jobs (any host)
            ▼                                              ▼
 ┌──── aar-node (beside each Resolve Studio) ───┐   ┌──── workers ──────────────────┐
 │ session broker: leases, hot-op queue,        │   │ aar.drp: read / diff /        │
 │   context save + restore, presence guard     │   │   modify / create             │
 │ op library: verified ops with traps built in │   │ transcripts, laugh maps,      │
 │ job runner: checkpointed, no 60 s cap        │   │   face tracks, ffmpeg,        │
 │ path map, offline + proxy checks             │   │   captions, thumbnails        │
 └──────────┬───────────────────────────────────┘   └───────────────────────────────┘
            │ DaVinciResolveScript (local)
            ▼
   DaVinci Resolve Studio ─── Blackmagic Cloud library · Presentations · Cloud Store
```

The app is Python 3.11+ throughout, because the Resolve API and all the existing CCN tooling are Python. It is one repo with a single package, `aar`, split into `aar.drp`, `aar.node`, `aar.hub`, `aar.plan` and `aar.recipes`.

Work is either hot or cold:

- **Hot ops** need Resolve's current project or timeline: building, grading, submitting renders, exporting. They always go through the node's broker.
- **Cold ops** need nothing from Resolve: parsing DRPs, transcription, Gemini analysis, ffmpeg, planning and linting. They run in parallel on any host.

## 4. The six requirements

### R1. Read, modify and create `.drp` files

- **Read (S1).** A lossless container reader produces a typed model: project settings, bins, media clips, timelines, tracks, items, and markers where they can be decoded. It keeps the raw text spans so later edits can splice them. It uses a tolerant scanner rather than a stock XML parser, because tags like `ListMgt::LmVersionTable` break those parsers. The version header is exposed. `.drt` is covered fully. `.drb` is best effort because its internals are still unknown.
- **Diff (S2).** A semantic diff of two files shows what an agent or a human changed between versions. Objects are matched by DbId, and every byte the reader does not decode is compared raw and reported as `opaque-changed`. Its gate mode (`aar drp diff --gate`) is S8's post-write check: it passes only when every change is decoded and DbId-matched. The gate cannot see, by construction, changes that leave both the decoded model and the compared residue identical: (a) zstd framing that decompresses to the same bytes; (b) hex letter case inside a blob the reader decodes (`FrameRate`, `Resolution`, `MediaExtents`, `Time`, `Geometry`, `Clip`, marker and settings blobs); (c) whitespace between elements; and (d) XML-equivalent attribute spellings — attribute order, quote style (`'` vs `"`) and entity-versus-literal characters — which the tolerant scanner normalizes away. Limit (d) holds only for the attributes the scanner decodes, the first 4096 of an element. The attributes of an element past that cap are not decoded; they are compared as raw bytes, from the first unread attribute to the end of the start tag, so inside that truncated tail the comparison is **stricter** than in the kept part: quote style, whitespace, entity spelling, attribute order and `/>` versus `></X>` all count as changes. That fails closed (a harmless respelling there is reported, never missed), and the reader's `scan-attributes-truncated` warning already fails the gate for such an element. A repeated attribute resolves last-wins below the cap and **first-wins past it**, because nothing past the cap is decoded: a second `DbId` there is never read, so the semantic "DbId changed" line is not produced and the pair is reported as one opaque change instead. When a `DbId` appears in the unread tail the change detail says so (a byte search for the name; a false hit only adds wording). The `DbAppVer`/`DbPrjVer` stamp is compared by its raw spelling, so a spelling-only change (`17` vs `017`, or non-ASCII digits) fails the gate. A genuine stamp *value* change also **fails** the gate on a real file: the stamp sits in the header comment of every member, including ones the reader does not interpret (`Gallery.xml`, stills), whose bytes are then compared as opaque. Bumping the stamp in only `project.xml`, or only in the members the reader interprets, fails too, but as a `version-mismatch` warning (the members no longer agree and the gate refuses any file read with warnings), not as an opaque change. A file whose stamp lived only in the decoded `project.xml` would pass a stamp change; on real files the failure comes entirely from the raw members' bytes. A project `DbId` change, by contrast, is reported but **passes** the gate — it is a single decoded header field (the project is the file's singleton, paired whatever its DbId) with no opaque collateral, so it is only listed, not failed. Because a passing gate means every change was decoded and DbId-matched, not that the changes were the ones intended, S8 must additionally compare the full change list against its intended edit set before trusting a write.
- **Modify (S8).** Only a whitelist of round-trip-verified edits is allowed:
  - rename
  - relink and path mapping (Windows drive letters to Mac `/Volumes` and back)
  - markers (add, update, delete)

  Each edit is a text-span splice plus a blob re-encode. Anything outside the whitelist is refused. Two edits that were planned are **deferred, not built**:
  - **Metadata.** It cannot be verified. The reader decodes no user-editable clip metadata (comments, keywords, scene, shot, take and camera fields live in `BtMetadataLockableBlob`, which it does not decode), so the post-write gate could not vouch for an edit. A clip's name is a rename.
  - **Fader keyframes.** No keyframe writer exists. `drp_tools` has codecs and format notes only, and its notes leave the tick encoding, the track mapping and the `AdrDatabase` checksum open. The reader does not decode `FLStudioModelBA` either.

  Every write must pass the S2 gate, and its full change list must equal the plan's intended edits. Three rules bound what is edited and what is accepted:
  - **Inputs that read with warnings are refused** (`input-has-warnings`), because the gate can never pass for them. In the 64-file private corpus that is 16 files, all `blob-partial`; the other 48 can be edited.
  - **The round-trip allow-list** (`aar.drp.modify.roundtrip`) is used only to compare a file with Resolve's own re-export of it, never for the writer's output. It accepts exactly **11 opaque lines** that Resolve rewrites on every export: 3 playhead values, 3 pan/tilt preset DbIds, 2 `DbSavedTime` values, 1 project age, 1 save date inside the settings blob, and **1 save stamp in the root bin's fields blob**. The project's `LastModTimeInSecs` is accepted as a decoded header change on top of those. Each rule checks the bytes, not just the name.
  - **Files are labelled `unverified`** (`<out>.aar.json`) until the sandbox round trip exists. The label is a JSON file aar recognises as its own by the stamp it writes: a `tool` of `aar <version>`, a `label` and `status` of `unverified`, and an `output` block naming this file and this label. A `<out>.aar.json` that aar did not write is refused (`foreign-sidecar`) and both files are left untouched; a label aar wrote on Windows is recognised on POSIX when its stamp holds an absolute Windows path (drive, UNC or `\`-rooted); one with a relative Windows path is still refused there, since `sub\out.drp` is also a valid POSIX name. `--force` checks the label when the run starts and again on a snapshot (a hard link, else a copy) taken just before the label is replaced. A label that is not aar's by then is refused, nothing is written, and the snapshot is kept and named in `left_behind`. This narrows the window but cannot close it, because rename has no compare-and-swap: a label another program writes as a new file between that last check and aar's `os.replace` (microseconds) is still replaced and lost. One written in place into the old label's file shows up in a hard-linked snapshot afterwards, which is then kept rather than deleted.
  - **Placement narrows, but does not close, the no-hard-link race.** The output and its label are written to temporary files, verified, then moved into place (the label first). Without `--force` the move hard-links onto the name — atomic, failing if the name is taken — or, on FAT/exFAT where there are no hard links, claims the name with `O_CREAT|O_EXCL` and `os.replace`s the verified temporary onto its own placeholder (`os.replace`, not `os.rename`, so it also works on Windows exFAT/FAT media, where `os.rename` fails when the destination exists). The `O_CREAT|O_EXCL` claim beats an exclusive creator or a check-then-create writer: once aar holds the name, that writer's create fails. It does **not** beat a plain non-exclusive writer (`open(dest,"wb")`, `cp`, Windows `CREATE_ALWAYS`): that writer opens and truncates aar's placeholder and writes into it, and the `os.replace` then discards those bytes. For that class of writer the window runs from the exclusive create through its close to the replace. On a failed replace aar removes the name only when it is aar's own output (same device and inode, by `lstat`). It deletes nothing else, not even an empty file: an empty file there may be another writer's (created after unlinking the placeholder, or the placeholder truncated and not yet written to), and no identity aar can record tells them apart. **A failed replace therefore leaves an empty file at the name, with no label** (the label is removed again). The same happens on Windows when a sharing violation makes the replace fail. The raised error, and the CLI's JSON (exit 74), name that file in `left_behind` (kind `placeholder`) so an agent can check and delete it. A crash mid-move may leave aar's own partial labelled copy (an empty placeholder or the output), but never a file that was already there.

- **Create (S9).** There are two routes:
  1. Preferred: build the project with the scripting API in a sandbox, then call `ExportProject`. Resolve writes the file, so it is valid by construction.
  2. Offline: fill in a version-matched template that was exported from the target Resolve version.
- **Validation gates every write.** Each written file is imported into the sandbox library, read back through the API and compared with the intended model. If it is visual, a frame is rendered as well. A file that has not been validated is labelled `unverified` and never offered to humans.
- **Version gate.** A file is never stamped newer than the target host. Writing for an older DbPrjVer is refused unless a template for that version exists.
- **Prior art.** We port the codecs from the `drp-format` JavaScript library and from Andrew's `drp_tools`. `drp_tools` supplies codecs and format notes only; it has no keyframe writer, so fader keyframes do not come from it. We do not reuse DRP-Toolkit, which is GPL.
  - **`drp-format` licence: unconfirmed.** The standalone repository `samuelgursky/drp-format` returns 404 (checked 2026-10-19). The library is vendored at `resolve-advanced/vendor/drp-format/` in `samuelgursky/davinci-resolve-mcp`. The repository's root `LICENSE` and the vendored `package.json` both say MIT, but the vendored directory has no licence file of its own and the copyright line names the repository's contributors, not the library's author. We therefore treat the licence as MIT per the host repository and unconfirmed for the directory; `tests/fixtures/drp/README.md` records the commit the four template files come from.
- **Blocked on approval.** S8 and S9 need permission to use the edit PC's empty Local Database as the sandbox. Switching databases closes the open project.

### R2. Blackmagic Cloud

- Cloud work happens only through the node (F1). The node can:
  - list cloud projects in the current library
  - load, create, import or restore a cloud project with `CloudSettings`
  - read each clip's Cloud Sync status
  - set `liveCollaborationMode`
  - check freshness with `RefreshFolders` and `GetIsFolderStale`
- **Media.** Before a build, the node checks the sync status of the project's media. It flags proxy-only sync, because in our setup proxies have no audio.
- **No direct database access.** The node never connects to the cloud library's database. That is unsupported and puts the shared library at risk.
- **Review.** Presentations have no API, so review happens in our console as a render plus timecoded notes. A human can still post a Presentation by hand. Its comments come back as timeline markers, which the node can read.
- **Limits.** The free tier allows 2 GB of storage and 20 presentations. Libraries are paid.

### R3. Collaboration between Resolve Studio users

Resolve already handles the hard parts on cloud libraries (locks, Live Save, chat, Compare Timelines). This app adds the layer above:

- **Presence and leases** across humans and agents. They are shown in the console and returned to agents in tool results, for example "bin X is held by a human since 14:02".
- **Agent branch timelines.** Agents never edit a human's timeline. An agent duplicates it to `<timeline> [agent:<id> v<n>]`, edits the copy, renders a review, and a human merges it with Compare Timelines (21.1). This rules out lock fights by design.
- **Review threads** anchored at timecode, plus approvals and handoff notes, all stored in the hub. They can be mirrored to timeline markers so humans see them inside Resolve.
- **Shared per-project facts:** sync and angle offsets, gains, template traps and path maps.
- **Remote collaborators**, Mac users included, need only Resolve Studio and access to the cloud library. The console is a web page.

### R4. Several projects at once on one machine, one user

Each Resolve instance has one current project (F3), and a second instance is unproven (F4). The plan follows from that:

- **Cold work** for any number of projects runs in parallel.
- **Hot work** is time-sliced by the broker:
  - per-project queues, batched so Resolve switches projects less often
  - leases
  - context saved and restored around every switch: project, timeline, page, playhead, Deliver settings and bin
  - a presence guard that refuses to switch while a human is working. "Working" is a heuristic: recent input, or the page or playhead has moved.
- **Renders.** Render jobs queue per project. S14 tests whether a `-rr` render node or a `-nogui` instance can run beside the GUI on the same PC. If neither can, renders are time-sliced too.
- **Fallback.** Truly parallel hot work needs a second Resolve instance. If S14 fails on Windows, that means a second machine or VM using Studio's second activation.
- **Collisions.** The broker also fixes today's thread collisions (F10).

### R5. UI and UX designed for agents

Agents are the primary users. Humans get a thin supervision console.

- **Task-level, typed tools** rather than mirrors of the Resolve API:
  - finding and describing: `project.find`; `timeline.describe`, which returns text a model can read (tracks, items with source timecode, gaps, markers)
  - analysis: `analyze.transcript`, `analyze.laughs`, `analyze.faces`, `analyze.loudness`
  - planning: `plan.lint`, `plan.apply(dry_run)`
  - looking: `frame.grab`, `frame.contact_sheet`
  - output and checks: `render.review`, `qa.check`, `review.request`
- **One EditPlan JSON for every build.** It holds source ranges with exclusive ends, tracks, transitions, audio gains, titles and a rule profile. The plan is linted against house rules before any Resolve call. A dry run returns a diff of what would change.
- **Every result carries measured values** from deterministic tools (frame counts, LUFS, durations). Errors are structured and include a fix hint and the known trap that caused them.
- **Everything resumes.** Jobs have ids and checkpoints, and the event log lets an agent replay its state after losing context.
- **Humans get four things:** a review queue (render plus timecoded notes), approve or reject, a board of who holds what, and an emergency stop.

#### The EditPlan (version 2)

`aar plan schema` prints the JSON Schema; `src/aar/plan/model.py` is the reference. Positions and lengths are integer frames at the plan's `fps`, and every range is half-open, `[in, out)`.

- **Version 1 fields** (S5): `project`, `timeline`, `kind` (`longform` or `short`), `fps`, `resolution`, `profile`; `sources` (id, media, camera, shot size, recording id and origin, duration); `segments` (track, source, `src_in`/`src_out`, `record_in`, keyframed `reframe`, `gain_db`); `transitions` between neighbouring segments (video type and length, audio transition); `titles` (lower thirds are `kind: "lower-third"` with a template name, lines and a record in/out); `captions` and `caption_style`; `markers`.
- **Version 2 adds** what the S13 producers and the `plan.apply` compiler need:
  - `template`: the timeline the build starts from. The compiler branches it with `timeline.branch`, never `timeline.duplicate`, so the copy is a registered agent branch.
  - `head`: how the first segment comes in from black (video type and frames, audio fade-in frames).
  - `outro`: the clip appended where the content ends (source id, optional source range, video track, the name of the audio track that carries it, the transition into it, and optional `sting` text over its start). The outro is not a content segment. Content rules (chronological, dialogue splice, end on the laugh, cut density) see the last content segment as the end. The outro does count towards the timeline length, so `short-max-duration` includes it.
  - `audio_tracks`: per-track name, `gain_db`, `mute` and optional `bus`, in track order (A1 first), so the CCN topology can be stated: A1–A4 muted, A5 the bus, A6 the bounce. Names are unique.
  - `Source.color`: input colour space, an optional CDL (slope, offset and power as three numbers each, plus saturation) and an optional LUT name. The safety rule still holds: never set the input colour space on BRAW.
  - `Segment.transform`: a static zoom, pan, tilt, rotation and flips, separate from the keyframed `reframe`. The effective zoom is the transform's zoom times the reframe zoom; `no-same-camera-jump` uses it.
  - `end_fade_frames`: the audio fade-out at the very end of the timeline.
- **Versioning rule.** `version` is 1 or 2, and a plan that leaves it out is version 2. Version 2 only adds fields, so every valid version 1 plan is valid, and lints the same, as version 2. A plan that says version 1 and gives a version 2 field a value is a validation error (`plan_version`) naming the field, for example `segments[1].transform`. Null, or an empty `audio_tracks`, is not a use, so a full dump of a version 1 plan reads back as version 1. `tests/plan/fixtures/v1_compat` pins the pre-v2 profile and the reports the pre-v2 code wrote for the plan fixtures; linting them today must give the same bytes.
- **Where problems are reported.** Version 2 cross-references are validation errors (exit 2), each with the field's location: an outro source that is not a plan source, an outro that plays to the end of a source with no duration, an outro range outside its source, a sting longer than the outro, an outro audio track not in `audio_tracks`, duplicate track names, a blank template or name, a CDL triple that is not three finite numbers in range, and a head or outro transition whose type and length disagree (a cut has no length; anything else needs one). The version 1 checks stay lint findings in `ranges-valid` (exit 1), so version 1 plans lint exactly as before.

**Rules.** The `ccn` profile runs these; the thresholds are in `profiles/ccn.toml`, not in code.

| Rule | Severity | What it checks (ccn values) |
| ---- | -------- | --------------------------- |
| `ranges-valid` | error | Ranges positive and inside the media, no overlap on a track, unique source ids, transitions that follow a join |
| `chronological` | error | Each recording plays forward only |
| `no-dialogue-splice` | error, warn | No cut inside a word (error); a cut sits in a pause of at least 250 ms (warn). Needs a transcript |
| `no-same-camera-jump` | error | Same camera across a time jump needs a zoom change of at least 0.2 (static transform included) |
| `timejump-close-to-wide` | warn | A time jump cuts to a wider shot |
| `no-dissolve` | error | Cuts between content segments. **New:** the head and the outro transitions are allowed when the profile says so (`allow_head`, `allow_outro`), up to `max_head_frames` / `max_outro_frames` (30, which admits the 15-frame start dissolve). Without those keys only cuts pass |
| `audio-crossfade` | error | Non-continuous joins get a "Cross Fade 0 dB" of at most 30 frames with enough handle |
| `cut-density` | warn | **New.** No more than `max_cuts` (2) cuts in any `window_s` (10 s) of record time. A cut is a join that is not unbroken playback, placed at the incoming segment's first frame; windows are half-open, so cuts exactly 10 s apart never share one. Overlapping offending windows merge into one finding with the burst's span and cut count |
| `min-shot` | warn | **New.** A content segment runs at least `min_seconds` (1.0 s) of record time; the hint says to fold it into a neighbour |
| `short-max-duration` | error | A short runs under 37 s, outro included |
| `short-end-fade` | warn | **New.** On a short, `end_fade_frames` is a de-click of at most `max_frames` (1); long-forms are not limited |
| `end-on-laugh` | warn | **Changed.** The final content out point lands `min_after_ms` to `max_after_ms` after a laugh's `anchor`, with half a frame of slack each side. ccn: onset, 0–400 ms. The code default is the old rule (end, 0–500 ms), and the old key `max_after_laugh_ms` still means the end-anchored maximum, so a profile without the new keys behaves as before. The hint names the source frame at the anchor plus `min_after_ms`. Needs a laugh map |
| `no-whip-pan` | error | Reframe keyframes move under 0.3 frame widths/s and zoom under 0.25/s |
| `caption-length` | error | Caption lines at most 16 characters |
| `caption-style` | info | Montserrat, active word yellow |

A rule's thresholds are either required (every profile listing the rule sets them) or optional with a code default. A rule can also check its threshold values, and a bad value is a profile error (exit 2) before anything is linted.

The join-based cut rules (`cut-density`, `no-same-camera-jump`, `timejump-close-to-wide`, `audio-crossfade`, and the neighbour hint of `min-shot`) consider same-track joins only, meaning neighbouring segments on one track, so an overlay entry on another track and the cut into the outro are not counted as cuts.

### R6. CCN work end to end

Recipes turn the existing, proven pipeline (the CCN shorts and long-form workflow, thumbpipe, the supoclip runner) into a work graph. The steps are:

1. Brief.
2. Locate sources (cloud project, offline media).
3. Flatten multicam and apply offsets.
4. Transcript and laugh map.
5. Cut plan, rule-checked.
6. Resolve build from the template: lower third, sting, outro, A5 bus.
7. Camera match.
8. Crossfades and loudness.
9. Review render.
10. QA: re-transcribe the joins, LUFS, offline frames, colour.
11. Human review loop.
12. Master render.
13. Shorts, then captions.
14. Thumbnails (thumbpipe).
15. Publish: YouTube through Command Center, Instagram through ccn-web-app.
16. Archive and update the project facts.

House rules ship as a rule profile, `ccn`:

- Never splice dialogue, and keep ranges chronological.
- No whip-pans, and no same-camera jump cuts.
- End hard on the laugh.
- Cut close to wide at time jumps, with no dissolves.
- Audio crossfades are at most 30 frames and sit in room tone.
- Loudness: about -22 LUFS for long-form, about -14 to -15 LUFS for shorts.
- Shorts run under 37 s.
- Captions: at most 16 characters, Montserrat, active word in yellow.
- Review MP4s stay under 20 MB.

Other shows get their own profiles.

#### R6 decisions for S13 (2026-10-19)

These are defaults taken from Andrew's recorded feedback on real CCN cuts. Each one is a profile value or a recipe parameter, so it can change without code.

- **End on the laugh means its onset.** The last frame lands between the laugh's onset and 400 ms after it: "end hard right after the punchline word or at the first hint of laugh, no dragging laugh, never fade to silence". A short's audio fade-out is a de-click (at most 1 frame), never a fade to silence. Long-forms end on the laugh, then the outro.
- **Dissolves.** None between content segments. A transition in from black on the first segment and the transition into the outro are allowed (the real long-form build uses a 15-frame start dissolve).
- **Cut density.** Warn when any 10 s window holds more than 2 cuts ("8 cuts in 13 s" was "extremely choppy"; 5 cuts in 50 s was accepted).
- **Shortest shot.** Warn under 1.0 s; fold slivers into a neighbour.
- **Pause trimming is a planner parameter, not a lint rule.** Trim pauses of at least 0.9 s down to a beat of about 0.45 s. The silence threshold is per venue: -42 dB by default, -36 dB where room laughs sit at -36 to -40 dB. QA reports the longest silence left in the output.
- **Joins sit in source-audio quiet.** Transcript word gaps are not enough, because Whisper word times are off by 0.1–0.2 s at joins. The in-side of a join needs at least 60 ms below -40 dBFS in the source audio. After the render, 2–3 overlapping short-window transcriptions around each join must agree with each other and with the plan's words.
- **Loudness.** Shorts get gain plus a limiter, not two-pass loudnorm, which flattened the laugh-to-voice range and padded the tail. Targets: -14.5 ± 0.5 LUFS for shorts, -22 ± 0.5 for long-forms. True peak at most -1.0 dBTP, no clipped samples, and the duration is unchanged.
- **Captions.** Every kept word is captioned (words snapped equals words captioned).
- **Where work runs.** Cold work (analysis, planning, QA, shorts compose) runs on the hub host. Resolve, its renders and the masters stay on the edit PC. Files move through one transport module that checks size and sha256.
- **Builds start from the template with `timeline.branch`,** so the copy is a registered agent branch. A human merges it.
- **Shorts compose with ffmpeg first** (a port of the proven runner). The Resolve template route comes later through the `plan.apply` compiler.
- **Publishing.** aar's last step emits a publish packet (final file and its hash, metadata, thumbnail) and waits on a human approval. Command Center and ccn-web-app do the upload; aar holds no YouTube or Instagram credentials.
- **Hub writes.** Agents create brief, plan and recipe nodes through hub write tools. Deciding a review stays human-only.

#### Media QA (S13b)

`aar.qa.check(file, kind=..., profile=..., plan=..., transcript=..., transcriber=...)` and `aar qa check FILE --kind short|longform|review` measure a render with ffmpeg and return one entry per check: id, status (pass, warn, fail or skipped), measured values, the target, and a one-line message saying what to do. Thresholds live in the profile's `[qa]` block (`QATargets`). Every key added after `review_mp4_max_mb` has a default, so an older profile still loads. `aar.media` runs ffmpeg and ffprobe from `AAR_FFMPEG`/`AAR_FFPROBE` or `PATH`. It passes argument lists only (the path goes in as `file:<absolute path>`), gives every call a timeout that kills the process group, keeps a bounded stderr tail, and raises `MediaError` with the exit code and that tail. The CLI exits 0 when no check failed, 1 when one did, and 2 on unreadable input, an invalid profile or missing tools.

| Check | Definition | Parameters |
| ----- | ---------- | ---------- |
| decode | ffprobe reads the file, and a full `-xerror` decode of every video and audio stream prints no error and exits 0. A file ffprobe cannot read fails here, and the render checks below are skipped as `render-unreadable`. | none |
| loudness | ebur128 on the first audio stream: integrated I within the kind's target ± tolerance (inclusive), and true peak (4x oversampled) at most the maximum. LRA is reported only. `review` uses the long-form target. A render without audio fails. | `short_lufs` -14.5, `longform_lufs` -22, `lufs_tolerance` 0.5, `true_peak_max_dbtp` -1.0 |
| clipping | A clipped sample is a decoded 32-bit float sample, in any channel, whose magnitude is at or above the clip level. When the astats sample peak is below the level, the count is 0 without a scan. Otherwise every sample is counted in pure Python, so a long master that reaches the level can run into the call's timeout (which then fails the check). NaN or infinite samples also fail. A clipped square fails and a -6 dBFS sine passes. | `clip_level_dbfs` -0.01 |
| longest-silence | ffmpeg silencedetect: a silence is every sample of every channel below the threshold for at least `silence_min_s`. The check reports the longest silence and where it starts. It passes unless `longest_silence_warn_s` is set and the longest silence is longer, which gives warn. | `silence_db` -42 (the CLI `--silence-db` or the `silence_db=` argument sets a venue value such as -36), `silence_min_s` 0.2, `longest_silence_warn_s` off |
| size-cap | `review` renders only: at most `review_mp4_max_mb` decimal MB (1 MB = 1,000,000 bytes). Other kinds are skipped as `not-applicable`. | `review_mp4_max_mb` 20 |
| duration | The render's video stream (else its audio stream, else the container) lasts the plan's record duration (`total_frames()` at the plan fps), within the tolerance in frames, inclusive. | `duration_tolerance_frames` 1 |
| join-quiet | Applies to every cut, meaning a join that does not continue the same recording within `join_continuity_tolerance_frames`. The check decodes the incoming source's first audio stream over [src_in − S, src_in + S], clipped to the start and end of the source, at 48 kHz. A window of `join_quiet_window_ms`, starting at any sample, is quiet when the RMS of its loudest channel is below `join_quiet_db`. A quiet run is the union of overlapping quiet windows, so its length is exact to the sample. The cut passes only when a run of at least `join_quiet_min_ms` contains src_in, allowing `join_quiet_slack_ms` at either end of the run: the incoming clip must start in the pause, not near it. When a cut fails, the scan becomes the fix hint. The nearest run of at least the minimum inside the window (the longer one on a tie) is reported with a suggested src_in, the frame at the plan fps nearest that run's middle ("move src_in of join N to frame F"). If there is no such run, the hint says there is no pause within ±S. Each join reports the run at the cut, the nearest pause, the longest run and the quietest window. S must cover the minimum plus the slack, so the window never shortens a run that contains src_in. With the default S, a pause whose qualifying part ends more than 40 ms before src_in is outside the window and gets no suggestion. | `join_quiet_db` -40, `join_quiet_min_ms` 60, `join_quiet_slack_ms` 10, S = `join_quiet_search_s` 0.10, `join_quiet_window_ms` 10, `join_continuity_tolerance_frames` 1 |
| join-retranscribe | Applies to every cut. An injected `Transcriber` (`aar.qa.Transcriber`) transcribes `join_windows` windows of the render, each `join_window_s` long, centred on the join and offset by `join_window_step_s`. The windows are compared on their overlap less `join_window_edge_s` at each end. A word is in that region when its midpoint is. Every pair of windows, and every window against the plan's expected words, must reach a difflib ratio of `join_agreement_min`. The expected words are the source transcript words, mapped through the plan. A source word that the cut slices by more than `word_straddle_s` on both sides fails the join, and so does a transcriber exception. | `join_windows` 3, `join_window_s` 4.0, `join_window_step_s` 0.5, `join_window_edge_s` 0.25, `join_agreement_min` 0.8, `word_straddle_s` 0.03 |
| caption-coverage | Every kept word is covered by a plan caption in record time. A kept word is a transcript word whose midpoint lies inside a kept segment's recording range, and it is covered when its record midpoint lies inside a caption's [record_in, record_out). The check lists the uncovered words. A short with no captions fails. A long-form or review plan with no captions is skipped. | none |

Skipped never means passed. A check is `skipped` only when the caller chose not to give an input (`no-plan`, `no-transcript`, `no-transcriber`, `no-captions` for a long-form or review plan), when the render could not be read or has no audio (`render-unreadable`, `no-audio`, which always come with a failed decode or loudness check), or for the size cap on a render that is not a review (`not-applicable`). Every skip carries a `skip_reason` and a message that says what was missing. A check that ran but could not check everything sets `incomplete` and puts the cause in `measured["unchecked_reason"]` and in its details. This covers a join check whose plan sources cannot be read (`source-unreadable`: the file is missing, the path is not local, or the source has no audio), a transcript that does not cover the recordings on both sides of a cut or under every segment (`transcript-does-not-cover`), and a truncated silence log (`log-truncated`). Such a check warns, or fails if something it did check failed. When nothing could be checked it warns, so it never looks better than a check that ran. The report's status is the worst status among the checks, and a skip counts as neither pass nor fail. The report's `unchecked` list names every check that was skipped or incomplete, so a runner can tell an all-clear (pass with an empty list) from "not checked" without parsing messages. The CLI prints that list under its head line. QA targets must be finite numbers: TOML allows `nan` and `inf`, and the profile refuses them. Any measurement that fails, including an ffmpeg error or a timeout, is a fail that carries the tool's message. Missing ffmpeg or ffprobe raises an error rather than producing a report in which every check is skipped. Source media paths in a plan are local hub paths. A relative path resolves against the plan file's folder in the CLI, or against `media_root` when the library is called. S13f adds the map from edit-PC paths to hub paths. Plan version 2 (S13a) works as is: `duration` measures against `total_frames()`, which includes the outro, and the step into the outro is the outro's transition, not a content cut. A v2 plan still does not say which audio track carries each segment, so every content join on a video track is treated as an audio cut.

#### Edit-PC transport (S13f)

Cold work runs on the hub; Resolve, its renders and the masters stay on the edit PC. One module (`aar.host`) moves files between them and runs PowerShell there. It is security-sensitive: it builds remote commands and moves files, so every path and string is treated as hostile.

- **Config, not code, names the host.** The edit PC is an ssh alias plus an ssh config file (`ssh -F <config>`), given by `AAR_EDIT_*` environment variables or an explicit `EditHostConfig`. There is no default host anywhere in code, and the ssh config and key material are never read. The alias must be a plain ssh host/alias (`^[A-Za-z0-9][A-Za-z0-9._@-]*$`): `/`, `[`, `:`, spaces and a leading `-` are refused, because they change how scp splits `host:path`.
- **Roots bound every transfer.** The config lists allowed remote (Windows) roots and allowed local (hub) roots. A source or destination outside them is refused before any command runs. Local paths are resolved with `realpath`, so a symlink that escapes its root is refused.
- **Runner seam.** All argv go through a `Runner` (argv list, optional stdin, timeout). The real one uses `subprocess` with an argument list and never a shell, and always passes `-F <config>`, `-o BatchMode=yes` and `-o ConnectTimeout=<n>`. Tests use a fake runner; nothing in the module or its tests contacts a real host.
- **PowerShell.** Scripts run as `powershell -NoProfile -NonInteractive -EncodedCommand <base64 of UTF-16LE>`. The command-line cap is a config value (`AAR_EDIT_CMDLINE_MAX`), not a fixed constant: Win32-OpenSSH runs the command through the sshd DefaultShell, and the default DefaultShell is `cmd.exe` (~8191 characters), far below the 32767 a PowerShell DefaultShell allows. The default cap is sized for `cmd.exe` and the refusal says so and how to raise it. Output CRLF is normalised to LF and CLIXML progress is stripped from stderr (error text is kept). Values embedded in scripts use one quoting helper (single-quoted literals), which doubles every character PowerShell treats as a single quote -- `'` and the smart quotes U+2018, U+2019, U+201A, U+201B -- so a filename like `Don’t Tell Mom.mp4` cannot break out of its literal. Parent paths inside scripts are taken with `[System.IO.Path]::GetDirectoryName`, never `Split-Path -LiteralPath ... -Parent` (an invalid parameter-set combination).
- **Transfers.** `pull`/`push` use scp in SFTP mode (the OpenSSH 9 default; `-O` is never passed). Remote size and SHA-256 come from `Get-Item`/`Get-FileHash`, local from `os`/`hashlib`. The file lands under a temporary name beside the destination and is moved into place only when both size and hash match; a mismatch or failure removes the partial and raises a typed error. The local move prefers an atomic hard link and falls back to an `O_EXCL` claim-then-replace on filesystems without hard links (FAT/exFAT, some SMB/CIFS and FUSE mounts); a local I/O failure surfaces as a typed error with the partial removed. An existing destination (file or directory) is never overwritten unless the caller asks; `--overwrite` still refuses a directory destination and a destination whose parent is a reparse point (the move fails closed).
- **Path forms and globbing.** A Windows path becomes one of `alias:E:/...` (default) or `alias:/E:/...`, selected by config, because it is unsettled which Windows OpenSSH needs. The spec is rebuilt from validated components, so no raw separators or alternate spellings leak through. scp in SFTP mode globs a remote *source* client-side, so glob metacharacters (`[ ] * ? \`) in a download source are backslash-escaped; a remote *destination* is not globbed and is sent literally. scp is always run with `-s` to force SFTP mode rather than rely on the client default; there is no toggle. The hub needs OpenSSH >= 8.7. Legacy SCP is unsupported because it is unsafe: in legacy mode scp sends the remote path unquoted through the remote shell (`cmd.exe` by default on Windows), so a data-controlled filename containing `&`, `^`, `%VAR%` or `$( )` would run a command (the round-2 HIGH-1 class of bug). A client older than 8.7 rejects `-s` and so fails closed with a typed transport error; a client of 8.7-8.9 defaults to legacy SCP, which `-s` overrides. The hub is OpenSSH 9.6p1. If the removed `AAR_EDIT_SCP_SFTP` env var is still set, config load refuses with a typed error rather than silently ignoring it.
- **Timeouts.** `connect_timeout` (ssh `ConnectTimeout`, default 10 s) bounds the handshake. `command_timeout` (default 300 s) is the wall-clock for each runner call, so it also bounds the scp of a large master and the `Get-FileHash` of one; callers pass `--timeout` (or a larger `command_timeout`) for big files.
- **What is refused.** Relative and drive-relative paths (`E:Show`), `.`/`..`/empty components, verbatim and device namespaces (`\\?\`, `\\.\`), reserved device names (`CON`, `COM1`, …), illegal or control characters (including an alternate-data-stream `:`), trailing dots or spaces, UNC paths unless a UNC root is configured, and anything that does not resolve inside an allowed root under NTFS case rules. Path parsing, NTFS upcasing and the edit-PC↔hub map reuse `aar.drp.modify.paths`.
- **Known limit.** Root validation is lexical, so a junction or symlink (reparse point) under an allowed root could still redirect a transfer outside it on the edit PC. The push move refuses when the destination's immediate parent is a reparse point, but a reparse point higher in the tree is not caught. This is acceptable for a trusted, single-tenant edit PC on a tailnet; a complete fix needs an atomic remote resolve-and-move that SFTP does not offer (any check-then-move still races).
- **Residuals for a live Windows test** (the hub has no PowerShell or Windows, so these are reasoned, not confirmed):
  - the generated move script on Windows PowerShell 5.1 (the `GetDirectoryName` parent for a drive-root and a UNC-share-root destination, HIGH-2); `Move-Item -LiteralPath ... -Destination` with `[ ]` in the name on 5.1; the no-`-Force` move refusing an existing file; and the container and reparse-point guards against a real junction;
  - the real command-line budget for a cmd.exe DefaultShell (the `cmd.exe /c` wrapper and full path);
  - which path form (`E:/...` vs `/E:/...`) Win32-OpenSSH's sftp-server accepts for stat, glob, download and upload;
  - UNC over SFTP in both forms (slash form is `///server/share/...`), including the second-hop credential, and `Get-Item -LiteralPath '\\server\share' -Force` on a share root;
  - smart-quote (`Don't ...`) and non-ASCII names end to end (UTF-8 over SFTP to UTF-16);
  - `scp -s` acceptance on the client actually used (a client older than OpenSSH 8.7 rejects it and fails closed);
  - stdin handling (ssh/scp get `/dev/null`), CLIXML noise on a first-run module load, and hidden files (`Get-Item` without `-Force` fails closed);
  - paths of 260+ characters through `GetDirectoryName`/`Get-Item` on .NET Framework (5.1);
  - whether the cleanup `Remove-Item` after an interrupted upload races the sftp-server's still-open handle and silently leaves the `.aarpart-*`;
  - legacy SCP (toggle-off) mode is unsupported because it is unsafe (see the transfers note): the remote shell parses the path, so it is never exercised and never allowed.

#### Hub write surface (S13g1)

How agents write to the hub (`aar.hub.writes`, `aar.hub.facts`). The MCP tools and `aar hub write` are thin wrappers over the same functions.

- **Opt-in.** `aar hub serve` stays read-only and lists exactly its 12 read tools. `--agent-writes` opens the existing database writable and adds six write tools (`brief_create`, `recipe_create`, `plan_submit`, `node_set_status`, `facts_record`, `facts_retract`) and three reads (`facts_get`, `brief_get`, `plan_get`). Human-only actions (deciding a review, granting or revoking an approval, resuming after a stop) are never tools. Agents without MCP use `aar hub write {brief,recipe,plan,status,fact,retract}` and `aar hub facts show`: one JSON request in (a file or stdin, with the tool's arguments), JSON out, a refusal as `{"error": {code, message, fix_hint, details}}` on stderr with exit 2. A plan stored as `blocked` exits 1, as `aar plan lint` does.
- **Identity is self-declared.** A write tool records the caller's `agent_id` as `Actor("agent", agent_id)`, checked only against the agent-id rule. That is attribution, not authentication (decision 6): any caller that reaches the server can claim any agent id. The CLI's `--actor` is the same.
- **Node kinds and schemas.** The graph gains a `recipe` kind; logs written before still fold to the same state (pinned by a fixture generated on the old code). Node data is typed with strict models (unknown keys refused, every string and list bounded, no NaN or infinity), each with a `schema` literal:
  - `aar.brief/1`: hub `project_id`, `show`, `kind` (longform, short, thumbnail), `recipe`, `rule_profile` (a built-in profile name), `sources` (at most 64 cloud-project, timeline-by-unique-id or media refs), `deliverables` (1 to 32, unique names, typed kinds) and optional `hook`, `handle`, `title`, `notes`.
  - `aar.recipe/1`: `recipe`, `brief_id`, `params` (a JSON object of at most 64 KiB, nested at most 8 deep), `max_review_rounds` (1 to 10, default 3). The recipe's own validator checks the params (a hook for S13g2; the default accepts any bounded object). The brief must be typed and ask for that recipe.
  - `aar.plan-node/1`: the EditPlan is stored as a content-addressed artifact (its canonical JSON, at most 1 MiB); the node holds its sha256, the plan version, the rule profile, the parent, the brief it resolves to, and a lint summary (counts, the first 20 findings errors first, and the rules that could not run). A plan derives from a brief, recipe, earlier plan or review, never an op or artifact. A plan that fails validation is refused with each field location; one with lint errors is stored `blocked`, one without `pending`. The hub lints without a transcript or laugh map, so the rules that need one are listed as skipped, not passed.
- **Profiles by name only.** The hub accepts built-in profile names, never paths, so a tool caller cannot make the hub read a file on its host. A plan whose parent resolves to a typed brief must use that brief's `rule_profile`, so an agent cannot dodge the house rules by naming another profile.
- **Who may change what.** An agent (or node) may move the status of a brief, recipe, plan or op node that it created (the actor on the node's creation event), through the graph's transition table. A human may move any node except a review. Reviews close only through a human decision or the requester's withdrawal.
- **Idempotency.** Every write needs a caller-chosen `idempotency_key`, namespaced per actor. The same key with the same (normalised) request returns the first result again, marked `replayed`; the same key with a different request (another project, op or content) is `idempotency_conflict`. A write that takes several events appends them in one transaction with a `hub.write` record (the op and a sha256 of the request) last, so a crash before the commit leaves nothing and a retry writes it whole. `plan_submit` stores its artifact first; it is content-addressed, so a retry reuses it.
- **Errors** are `{code, message, fix_hint, details}`. Codes include `unknown_project`, `unknown_parent`, `wrong_parent_kind`, `untyped_parent`, `unknown_node`, `not_creator`, `kind_not_allowed`, `protected_review`, `illegal_transition`, `idempotency_conflict`, `invalid_brief`, `invalid_recipe`, `invalid_plan`, `invalid_params`, `unknown_profile`, `profile_mismatch`, `recipe_mismatch`, `unknown_namespace`, `invalid_fact`, `unknown_fact`, `unknown_source` and `too_many_facts`.
- **Facts** (`facts.recorded`, `facts.retracted`, scoped to the project). Keys come from a closed set of namespaces, each with its own value model; every `<...>` part is 1 to 64 of `A-Z a-z 0-9 _ -`:
  - `sync.offset.<angle>`: whole frames, the frame rate as a rational, and what the offset is relative to.
  - `audio.gain.<track>`: dB, -96 to +24.
  - `template.trap.<id>`: text, up to 2000 characters.
  - `template.id.<name>`: a Resolve unique id.
  - `path.map.<name>`: `from` and `to` paths.
  - `render.hash.<deliverable>`: sha256 and byte size.
  - `qa.<metric>.<deliverable>`: a finite number and its unit.

  The projection gives each key's current value (the last event in log order wins), who set it and when, an optional source (a recipe node and/or a stored artifact) and its last 10 changes. A retraction keeps the history. Malformed or forged events are skipped and listed. Any actor kind may record; a project holds at most 1000 keys.

## 5. Safety rules (enforced in code, not just written here)

- Before every hot op, the node checks that the expected project is the one open. It refuses to switch project or database while a human is active, unless a human forces it from the console.
- Switching databases, creating or restoring cloud projects, and writing to a production project all need a human approval token, which is recorded in the event log.
- Deliver settings are recorded before a render preset loads and restored afterwards.
- Never change Input Color Space on BRAW clips, and never sweep a colour space across a whole bin.
- Never connect to the cloud library's database directly.
- DRP writes always go to new files. Originals are never overwritten.
- Real project files used for testing stay in a git-ignored corpus (`corpus/`, or `$AAR_DRP_CORPUS`) and are never committed.

## 6. Risks

| Risk | Mitigation |
| ---- | ---------- |
| DRP edits import but break silently (F6) | Whitelist of writes, sandbox round-trip validation, and a preference for build-and-export |
| A new Resolve version changes the DRP schema | Version-tagged templates, validation for each host version, a tolerant reader |
| The single Resolve instance is a bottleneck (R4) | Broker and batching; the S14 spike; a second machine as fallback |
| Humans and agents fight over the shared instance | Presence guard, leases, agent branch timelines |
| Cloud features need paid libraries and have no API | All cloud work goes through the node; the console replaces Presentations |
| The 60 s MCP cap, hangs and crash reporters | Node-side job runner with a watchdog, not run through the MCP bridge |
| 21.1 scripting needs Studio | The target is Studio only, as the requirement says |

## 7. Decisions (these defaults hold until Andrew says otherwise)

1. A standalone app, which Command Center threads use as an MCP server. The alternative is to build it inside Command Center.
2. The hub is tailnet-only for v1. People outside the tailnet will need accounts later.
3. The edit PC's empty Local Database becomes the sandbox for DRP validation and spikes. **This needs approval**, because switching databases closes the open project. Until then, DRP work is offline only.
4. Run the multi-instance spike (S14) on the edit PC. **This needs approval** and an idle window.
5. A GitHub remote. There is none yet, and the PR and Prism gate needs one. Until it exists, slices merge locally into `dev` after independent review. `main` gets only the reviewed `dev` → `main` PR.
6. **Open: a real credential for the console and hub.** In v1, identity is network reachability. Whoever can reach the console acts as its `--operator`, and with `--trust-tailscale-headers` any process on the hub host can claim any Tailscale login over loopback. The MCP server has no auth at all, and with `--agent-writes` any caller can write briefs, recipes, plans, status and facts under any agent id it declares. That is acceptable only while both stay tailnet-only (D2). Before either is reachable from outside the tailnet, it needs a real per-person credential, for example a Tailscale identity checked through the local API or an auth proxy, and the trusted-header mode must go.
