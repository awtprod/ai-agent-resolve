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
The map must cover every source (the params refuse one it does not), and the PC's copy must
describe the same frames as the hub's, with the same start timecode: byte-identical, or the hub's
copy a shrink of the PC's (below).

## Shrinking proxies before they reach the hub (S19b)

A 90-minute camera proxy is about 5.5 GB and the hub's media disk holds about one, so a multicam
show does not fit. `aar media shrink` makes a working copy of about a quarter of the size that
describes the same frames: the same frame size, frame count, constant frame rate and start
timecode, H.264 at a much lower bitrate with a keyframe at least every two seconds, and every
audio stream kept lossless (FLAC in `.mp4`, ALAC in `.mov`). The hub transcribes, syncs, finds
faces and renders the review copy from it; the exported timeline still links the PC's own
full-size proxy through the media map. Each copy is verified against its source before it gets
its name (docs/DESIGN.md "Shrunk working copies" lists the checks and the settings).

**On the edit PC, before copying to the hub.** The PC has Python and ffmpeg/ffprobe on PATH but
no `aar` installed; the command needs only the standard library, so run it from a copy of this
repository's `src` folder, folder to folder, straight into the mapped share:

```console
> set PYTHONPATH=C:\aar\src
> python -m aar.media.shrink E:\Show_1\Proxy --out Z:\incoming\Show_1\Proxy --report shrink.json
```

It takes the `.mp4`, `.m4v` and `.mov` files directly in the folder (at most 64, in name order)
to the same names in `--out` (created if missing), prints one JSON line per file and exits 0 when
every file was shrunk or already done, 2 otherwise. It never overwrites: a file already in `--out`
that verifies as a shrink of its source is skipped, so rerunning after an interruption resumes the
batch; one that does not verify is refused and left alone. A failed file does not stop the batch.
Before each file it needs half the source's size free on the destination. A single file works
too: `python -m aar.media.shrink A001.mp4 --out Z:\incoming\Show_1\Proxy\A001.mp4`.

**A full-size copy already on the hub.** `--replace` writes the copy beside the file, verifies it,
and atomically renames it over the original (the only mode that removes anything):

```console
aar media shrink <media root>/share/incoming/Show_1/Proxy/A001.mp4 --replace
aar media shrink <media root>/share/incoming/Show_1/Proxy --replace
```

A file already marked as shrunk is never shrunk again (named alone it is refused, in a folder it
is reported as `skipped`). `--replace` refuses a symbolic link (`symlink`: run it on the file the
link points to); on a file with several hard links it replaces only the name given, so the other
names keep the full-size file. Run `--replace` before a show's run starts: a run records each
source's size and modification time at its stub step and refuses (`media_changed`) a file
replaced since. Errors are JSON with a `code` (`unreadable`, `vfr`, `odd_size`,
`negative_start`, `audio_unsupported`, `already_shrunk`, `symlink`, `destination_exists`,
`not_writable`, `low_space`, `timeout`, `encode_failed`, `verify_failed`, `source_changed`,
`interrupted`, ...) and a `fix_hint`.

**Stopping a run.** Ctrl-C, `kill`, closing the terminal (Linux) or Ctrl-Break (Windows) stops
ffmpeg, removes the partial copy, prints the file's line with the code `interrupted`, writes the
report and exits 2; rerunning resumes (files already done are verified and skipped). A run killed
outright (`kill -9`, Task Manager) or a closed console window on Windows cannot clean up: it can
leave a hidden `.<name>.aar-shrink-<random>.part` file (after `kill -9` ffmpeg finishes it in
the background, full size), which is safe to delete.
