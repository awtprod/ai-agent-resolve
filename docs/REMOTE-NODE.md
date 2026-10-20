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

Cancelling a job before any node claims it commits `cancelled` on the Linux hub, even with
the Windows PC off. Cancellation and claim use the same transaction lock: if claim wins,
the job stays `cancel_requested` until that node acknowledges cancellation. Neither case
automatically reassigns the job. The creator or a human may cancel; request keys remain
idempotent. Existing unclaimed cancellation events also fold to terminal `cancelled`.

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
- `SIGINT` and `SIGTERM` stop new claims and operation dispatch. Shutdown waits up to five
  seconds for the current drain, then consumes completed and late results into durable
  checkpoints without issuing another operation. A completed step stays completed; any next
  step waits for startup. An in-flight render can still journal and restore Deliver settings.
- If the drain times out, `aar node run` reports **shutdown incomplete** and exits with code
  2. Programmatic `close(timeout)` returns `False`, retaining the store lock and node-id
  reservation while that process lives. Calling close again still waits for the same drain;
  once it finishes, a successful close checkpoints its result and releases ownership. An
  uninterruptible Resolve call is never killed by the runner. Process exit releases OS locks
  and can interrupt local cleanup; it cannot prove that Resolve abandoned the external call.
- Before entering a mutating operation, the runner commits an uncertainty marker. It removes
  that marker only as the result is checkpointed. After an abrupt exit or incomplete shutdown,
  an unfinished job with an unresolved marker stalls and blocks further work on its project:
  the side effect may have happened even when no post-call journal entry exists. A human must
  inspect Resolve
  (including any still-running call), then cancel the job or explicitly call
  `runner.resume(job_id, Actor("human", ...))` to authorize retry. This does not guarantee
  exactly-once external execution. Read-only calls may retry. `render.build` with an existing
  journal retains its established settings/queue recovery, including ambiguous-add detection;
  a build interrupted before its first journal entry requires human recovery too.

All existing runner controls still apply: per-op presence and branch ownership, emergency
stops, production approval consumption, project identity checks, render journaling, and
watchdog bounds. There is no fake-gateway switch in the production CLI; tests inject a fake
gateway before startup. A live Windows/Resolve smoke remains an environment test and is not
run by the non-live suite.
