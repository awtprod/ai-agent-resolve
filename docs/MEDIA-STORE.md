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
