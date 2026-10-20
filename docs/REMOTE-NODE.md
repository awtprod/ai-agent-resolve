# Remote Resolve node

`aar node run` keeps the hub on Linux and DaVinci Resolve on a Windows edit PC. The node
dials the hub over HTTP for job control; media staging and return use the separate verified
host transfer commands. Jobs remain in the hub's existing
append-only event log while the PC is off and appear as `waiting_for_node`.

## Enrol and start

On the hub, enrol the node and save the one-time token to an access-controlled file on the
Windows PC. Do not put the token in an environment variable or a command line.

```text
aar hub node enroll edit-pc --db /srv/aar/hub.db --actor human:andrew
aar hub nodeapi --db /srv/aar/hub.db --host 127.0.0.1 --port 8789
```

The node API binds loopback unless `--allow-remote` is explicit. It is intended to sit behind
the tailnet; the enrolled bearer token is a second factor, not a replacement for that boundary.

On Windows, with Resolve Studio running and `DaVinciResolveScript` importable:

```text
aar node run ^
  --node-id edit-pc ^
  --hub-url http://hub.tailnet:8789 ^
  --token-file C:\ProgramData\aar\edit-pc.token ^
  --store C:\ProgramData\aar\jobs.db ^
  --project "{\"folder_path\":[\"Shows\"],\"name\":\"Show\",\"project_id\":\"resolve-project-uid\"}"
```

Every allowed project is an exact `ProjectLocator`; repeat `--project` to add another. A claim
must match both this authenticated node and one configured Resolve project id. The worker also
checks the complete locator before persisting the job.

Submit from the hub with an explicit idempotency key:

```text
aar hub job submit --db /srv/aar/hub.db --actor agent:editor --node edit-pc \
  --agent-id editor --kind ops.sequence --idempotency-key edit-20261020-1 \
  --project '{"folder_path":["Shows"],"name":"Show","project_id":"resolve-project-uid"}' \
  --params '{"steps":[{"op":"timeline.list","params":{}}]}'
```

Use `aar hub job list`, `show`, `cancel`, and the human-only `release`. Release means the human
has stopped the old node and checked Resolve for side effects; it increments the job generation
before reassignment. Reports and heartbeats (including old request-key replays) from the old
node/generation are then refused. The runner checks the current claim with the hub immediately
before each operation dispatch. Release cannot undo or interrupt a Resolve call already in
flight, including a call that passed that check just before release. Stop the old node and
inspect Resolve before deciding that reassignment is safe.

With `aar hub serve --agent-writes`, agents can use `job_submit` and `job_cancel`;
human release stays in the CLI. The read-only MCP surface also provides `jobs_list`, with bounded pagination and exact project
or state filters. In particular it exposes `waiting_for_node` and `node_offline` without making
either state an implicit retry.

## Failure behavior

- A claim is sticky. Its default lease is 60 seconds with a 15-second heartbeat. Lease expiry
  displays `node_offline`; it never requeues the job.
- The node uses short, immediate claim polls rather than a long poll. This keeps the pump and
  heartbeats moving even while an op is running on the broker's bounded drain thread.
- The hub job id and assignment generation are committed together in the local
  `external_assigned` event before execution. Reconnect compares that original fence with
  the current hub claim; it never adopts a newer generation for an old local record. Every
  unproven local job is quarantined, including legacy submissions without a valid mapping
  and assigned-but-unclaimed jobs. Queued operations use the runner's safe cancellation and
  precheck path. Local-only runners retain their existing behavior when no remote worker is
  attached.
- A hub claim absent locally (or present under another generation) becomes `lost_on_node`
  and is never automatically reconstructed. A terminal local job resends its result until
  the hub acknowledges it; that ACK is then recorded in the local event journal.
- Reconciliation covers complete, consecutive job-ID ranges, with at most 100 local
  summaries and 100 hub records per HTTP page. Missing jobs are checked only inside the
  range supplied, so later input pages are never mistaken for loss. Output pages also cover
  an empty local journal. The existing 10,000-job / 100,000-job-event hub limits remain
  explicit; nothing is silently truncated. Reconciliation summaries have a 512 KiB page cap,
  node API requests a 1 MiB cap, replies a 4 MiB client cap, and job results a 256 KiB cap.
- Reconnect clears the in-memory branch registry and rebuilds it only from authenticated hub
  branch records for the configured projects. A failed rebuild leaves edits fail-closed.
- Revoked or missing enrolment is a permanent refusal: the worker stops claiming. Transport
  failures fail closed and use bounded exponential backoff.
- `SIGINT` and `SIGTERM` stop new claims, let the existing runner checkpoint at its safe
  boundary, and close that runtime and `JobStore`.

All existing runner controls still apply: per-op presence and branch ownership, emergency
stops, production approval consumption, project identity checks, render journaling, and
watchdog bounds. There is no fake-gateway switch in the production CLI; tests inject a fake
gateway before startup. A live Windows/Resolve smoke remains an environment test and is not
run by the non-live suite.
