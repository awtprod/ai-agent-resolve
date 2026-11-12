# Linux source media store

`aar media ingest` makes a Linux-resident, hub-owned copy of source media for cold
analysis. The bytes are written by the existing content-addressed `ArtifactStore`;
the media layer records a separate project association with the original filename,
SHA-256 and size. Identical bytes are stored once, while each project still gets its
own association.

```console
aar media ingest ./incoming/interview.mp4 \
  --db ./hub/events.sqlite --artifacts ./hub/artifacts \
  --project prj_example --actor agent:worker-1 --max-bytes 107374182400
```

Success prints JSON containing a stable `media_ref` and a verified absolute
`local_path`. The original may be removed after success. `resolve` verifies the
stored bytes every time and returns that same local path; it never falls back to the
ingest path, a Windows path, or a shared filesystem.

```console
aar media availability 'media:<sha256>' \
  --db ./hub/events.sqlite --artifacts ./hub/artifacts --project prj_example
aar media resolve 'media:<sha256>' \
  --db ./hub/events.sqlite --artifacts ./hub/artifacts --project prj_example
```

Availability exits 0 only for verified content. Missing, corrupt, unreadable or
wrong-sized content is reported as unavailable and exits 1. Bad input, an unknown
project, or an unusable hub path exits 2. A configured `--max-bytes` is enforced by
a bounded streaming read; the video is never loaded into memory as one buffer.

Existing cold commands accept the returned path directly, for example as the media
argument to `aar analyze transcript` or `aar qa check`. An EditPlan source can use
the returned `local_path` as its `media` value. Ingest does not rewrite existing
plans automatically.

Resolve still needs a path it can read on the Windows node. Before submitting an op
that imports media, stage the verified Linux copy with `aar host push` to a new path
under the configured Windows media root. The transfer compares size and SHA-256;
keep the Linux artifact as the authoritative source and use the Windows destination
in the node op's parameters. Configure `AAR_EDIT_SSH_ALIAS`, `AAR_EDIT_SSH_CONFIG`,
`AAR_EDIT_REMOTE_ROOTS` and `AAR_EDIT_LOCAL_ROOTS` explicitly; there is no default edit PC
or unrestricted transfer root. See `aar host push --help` and `aar.host.config`.

The queue does not silently translate a Linux path into a Windows path or fetch missing
media during an edit. Staging is an explicit prerequisite for media-dependent hot jobs;
cold work continues against the Linux copy while the node is absent. This reuses the
existing verified transfer commands rather than adding a second transfer protocol.

For a render already copied by the existing verified host transfer, ingest the
Linux destination with the same command:

```console
aar host pull '<configured-remote-render>' ./incoming/review.mp4
aar media ingest ./incoming/review.mp4 \
  --db ./hub/events.sqlite --artifacts ./hub/artifacts \
  --project prj_example --actor agent:worker-1
```

`aar host pull` has already compared size and SHA-256 across the transfer; ingest
hashes the local file again while placing it in the hub store. Media ingest does not
connect outbound to an edit PC and does not replace the host transfer subsystem.

## Start timecode of one file

`aar media timecode FILE` reads a QuickTime, MP4 or BRAW file's start timecode from its tmcd
track without ffprobe (S18a, `aar.media.quicktime`) and prints it as JSON, with the two fields to
copy into a plan stub's source. It needs no hub.

```console
$ aar media timecode B006_10010449_C074.braw
{"frames": 521420, "rate": "30000/1001", "drop_frame": false, "timecode": "04:49:40:20",
 "stub": {"start_timecode": 521420, "start_timecode_rate": {"num": 30000, "den": 1001}}, ...}
```

`rate` is the tmcd track's timescale/frameDuration as written; the stub's rate is that rate
snapped to a standard one (2997/100 is 30000/1001), as `aar plan produce` would. The tmcd track
cannot say whether whole-numbered video is really 23.976 or 29.97: if its 24/1 or 30/1 tmcd sits
on NTSC-rate video, give the video's rate instead.

It exits 0 when it read a timecode, 1 when the file has none it can read (the error's `code` is
`timecode_` plus `not_quicktime`, `no_timecode`, `malformed`, `unsupported`, `too_large` or
`io`), and 2 when the file is missing or not a regular file.

## Proxies-only media root (S19a, decisions D8/D9)

For the Linux-only editing run (`ccn-longform`, DESIGN "CCN long-form recipe") the hub keeps
**proxy copies only**: the camera proxies (1080p H.264 with a tmcd timecode track), never the
camera originals. They live on one media root, which is read-only inside agent sandboxes; only
the hub's own units write to it.

```
<media root>/models/          Whisper (faster-whisper-medium.en/) and YuNet models, never downloaded
<media root>/share/incoming/  proxies copied from the edit PC (<show>/Proxy/<clip>.mp4)
<media root>/share/exports/   results for the edit PC: <project>/<timeline>/<timeline>.r<N>.fcpxml
                              and .r<N>.review.mp4
<media root>/share/timelines/ timelines edited in Resolve, back from the edit PC (S20):
                              <project>/<timeline>/<timeline>.e<N>.drt
<media root>/work/            step scratch (transcripts, renders), removed after each step
```

The edit PC maps `<media root>/share` as a network drive, so `incoming/` is where the proxies
arrive and `exports/` where the timeline and the review copy come back. The hub units name these
folders in `hub.env`: `AAR_MEDIA_ROOT`, `AAR_MODEL_DIR`, `AAR_WORK_DIR`, `AAR_EXPORT_DIR`.

Recipe params name the proxies by their hub paths (under `AAR_MEDIA_ROOT`). The plan keeps
those paths, so the hub render reads them directly. The FCPXML must instead name the PC's copy
of the same file: the recipe's `media_map` (`{"<hub folder>": "<PC folder>"}`, the export's
`--media-map`) rewrites each hub path under a mapped folder into the PC path, as a `file:` URL,
and each asset starts at the proxy's own start timecode, which is how Resolve links a clip to
its file. For example `{"<media root>/share/incoming/Show_1": "E:\\Show_1"}` turns
`<media root>/share/incoming/Show_1/Proxy/A001.mp4` into `file:///E:/Show_1/Proxy/A001.mp4`.
The map must cover every source (the params refuse one it does not), and the PC's copy must be
byte-identical to the hub's, so its timecode and frames are the same.

## Timelines edited on the PC (S20)

After the owner edits an imported timeline in Resolve, File > Export > Timeline saves it as a
`.drt` into `<media root>/share/timelines/<project>/<timeline>/` on the mapped drive, one file
per round (`<timeline>.e1.drt`, `.e2.drt`, ...; a file is never overwritten). The hub reads it
with `aar plan import-timeline FILE.drt --plan PLAN.json --media-map MAP.json --out
PLAN2.json --report DIFF.json`, using the same media map as the export: an item's media path is
the PC path the FCPXML named, so the map's hub root -> PC root pairs tie it back to the plan's
hub path (DESIGN "Timeline import (S20)"). The folder is not yet named in `hub.env` or used by
the `ccn-longform` recipe; that wiring is a later slice.
