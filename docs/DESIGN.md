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
- **Diff (S2).** A semantic diff of two files shows what an agent or a human changed between versions. Objects are matched by DbId, and every byte the reader does not decode is compared raw and reported as `opaque-changed`. Its gate mode (`aar drp diff --gate`) is S8's post-write check: it passes only when every change is decoded and DbId-matched. Known limits: inside a blob the reader decodes (`FrameRate`, `Resolution`, `Clip`, ...), a difference only in zstd framing or only in hex letter case is not seen, because those blobs are compared after decoding.
- **Modify (S8).** Only a whitelist of round-trip-verified edits is allowed:
  - rename
  - relink and path mapping (Windows drive letters to Mac `/Volumes` and back)
  - markers
  - metadata
  - fader keyframes, ported from `drp_tools`

  Each edit is a text-span splice plus a blob re-encode. Anything outside the whitelist is refused.
- **Create (S9).** There are two routes:
  1. Preferred: build the project with the scripting API in a sandbox, then call `ExportProject`. Resolve writes the file, so it is valid by construction.
  2. Offline: fill in a version-matched template that was exported from the target Resolve version.
- **Validation gates every write.** Each written file is imported into the sandbox library, read back through the API and compared with the intended model. If it is visual, a frame is rendered as well. A file that has not been validated is labelled `unverified` and never offered to humans.
- **Version gate.** A file is never stamped newer than the target host. Writing for an older DbPrjVer is refused unless a template for that version exists.
- **Prior art.** We port the codecs from samuelgursky's `drp-format` (MIT, JavaScript) and Andrew's `drp_tools` (codecs, keyframe writer, zstd body). We do not reuse DRP-Toolkit, which is GPL.
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
6. **Open: a real credential for the console and hub.** In v1, identity is network reachability. Whoever can reach the console acts as its `--operator`, and with `--trust-tailscale-headers` any process on the hub host can claim any Tailscale login over loopback. The MCP server has no auth at all. That is acceptable only while both stay tailnet-only (D2). Before either is reachable from outside the tailnet, it needs a real per-person credential, for example a Tailscale identity checked through the local API or an auth proxy, and the trusted-header mode must go.
