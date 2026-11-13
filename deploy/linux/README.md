# Hub on Linux (systemd)

Four units run the hub: `aar-hub-mcp` (agent MCP server, loopback :8787, agent writes on),
`aar-hub-console` (human console, loopback :8788), `aar-hub-nodeapi` (node API for the
Windows Resolve node, tailnet IP :8789 only) and `aar-hub-worker` (the cold recipe worker: it
runs the `ccn-longform` recipe's transcript, render and export steps on the hub, with the edit
PC off). A fifth, `aar-hub-lessons`, is a one-shot pass that its timer runs every 10 minutes:
it drafts lessons from decided reviews with a model (see below). The files here are templates
(`*.service.in`, `*.timer.in`, `hub.env.in`) with `@PLACEHOLDER@` fields; nothing in this kit
writes to `/etc` or starts a service.

## Layout

Everything lives under one root (`@AAR_ROOT@`), owned by the service user:

```
<root>/app/            git checkout of this repo (pinned to a commit)
<root>/venv/           python -m venv; pip install "<root>/app[hub]"
<root>/data/hub.db     the event log (SQLite)
<root>/data/artifacts/ artifact store
```

The units set `ProtectSystem=strict` and allow writes only to `<root>/data`, and
`UMask=0077`, so create `data/` and `data/artifacts/` first, owned by the service user and
mode 0700.

`<root>/hub.env` is the units' `EnvironmentFile` (rendered from `hub.env.in`): `AAR_RECIPES`
(the recipe modules: `recipe_create` on the MCP server accepts `ccn-longform` only when it is
loaded), `AAR_MEDIA_ROOT` (recipe params must name files under it), `AAR_MODEL_DIR` and
`AAR_YUNET_MODEL` (models, never downloaded), `AAR_WORK_DIR` (step scratch) and
`AAR_EXPORT_DIR` (the edit PC's share; required: without it the `ccn-longform` export step
fails `export_dir_unset` instead of finishing with nothing delivered). The media root layout is
in `docs/MEDIA-STORE.md`.

## The worker (`aar-hub-worker`)

`aar recipe worker --recipes aar.recipes.ccn` claims the cold steps of running recipes. It
writes only to `<root>/data`, `AAR_WORK_DIR` and `AAR_EXPORT_DIR` (`ReadWritePaths`) and reads
the media root and the models. Create the work and export folders first, owned by the service
user. Its `UMask` is 0022 because the edit PC reads the exports through the share (files 0644,
folders 0755); step scratch is created 0700.

Sizing: a 90-minute show's transcript takes about 25 minutes and 4 GB of memory on the CPU, and
the review render runs at about 0.8x realtime of the cut, with at most 6 GiB for ffmpeg (less
when the cgroup or the host has less free). One worker runs one step at a time.

Stopping: on SIGTERM the worker claims nothing new. A step that has not started its real work
stops at its next check (before and after hashing the media, about 10 s for a 90-minute proxy;
while waiting for another worker's transcription; before a render or an export) and records
nothing; a render in progress is stopped too (ffmpeg's process group killed, its scratch
removed). In both cases the worker exits with the claim held: the 60 s lease lapses and the next
worker claims the step again, because these steps are idempotent, so a stop uses no attempt.
A transcription in progress cannot be interrupted: past `TimeoutStopSec` (15 minutes) systemd
kills the worker, and the next worker resumes the transcription from its checkpoint under
`AAR_WORK_DIR`. A SIGKILL looks the same to the hub: a claim whose lease lapses.

`KillMode=mixed` sends SIGTERM to the worker alone and, at the stop timeout or after a crash,
SIGKILL to every process left in the unit's cgroup. That is what reaps an ffmpeg (a render's
encode, or a decode) orphaned when the worker itself was killed: outside systemd such an ffmpeg
keeps running (re-parented to init) until it finishes.

Scratch: each render or laughs attempt works in `AAR_WORK_DIR/<area>/<recipe>/<op>/`, emptied
when the attempt starts and removed when it ends; at every start the step also removes, at most
200 entries at a time, the folders of finished or cancelled runs and of ops that are no longer
running. Transcription folders and lock files nobody holds are swept the same way (folders kept
two days for a resume).

## Choosing the root: not `/srv` for Command Center agents

Pick a root outside `/srv` if Command Center agents need to operate on it (for example to run
`aar hub ...` administrative commands). `/srv` is read-only in their sandbox, so writes to
`hub.db` or the artifacts fail there even though the service itself (a system unit) can write
to it. The 2026-11-07 deploy used `/var/lib/command-center/runtime/services/aar-hub`.

## Render and install

```
AAR_ROOT=/var/lib/aar AAR_USER=aar AAR_GROUP=aar \
TAILNET_IP=100.x.y.z CONSOLE_HOSTNAME=host.tailnet.ts.net \
AAR_MEDIA_ROOT=/srv/aar-media AAR_WORK_DIR=/srv/aar-media/work \
AAR_EXPORT_DIR=/srv/aar-media/share/exports \
  deploy/linux/install-units.sh ./rendered
sudo install -m 0640 -o root -g aar rendered/hub.env /var/lib/aar/hub.env
sudo cp rendered/*.service rendered/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aar-hub-mcp aar-hub-console aar-hub-nodeapi aar-hub-worker
```

The lessons timer is enabled separately, once its key is in place (next section).

`TAILNET_IP` is the hub's own tailnet address; the node API binds to it alone (not
`0.0.0.0`), so `Wants=tailscaled.service` orders it after tailscale.

## Lessons from reviews (`aar-hub-lessons`)

After a person decides a review in the console, `aar hub lessons propose` asks a cheap model,
through the local model gateway (OpenAI-compatible, default `http://127.0.0.1:8317`), for at
most three lessons from that review's reason and notes, and records them as proposals. A person
accepts, rescopes or drops each one on the console's `/lessons` page; nothing a model writes
reaches a cut guide without that. Each pass makes at most 5 model calls (`--limit 5`), and stops
early when the store's 200 waiting proposals have no room for 3 more. A review is drafted once,
with at most 3 charged attempts per review (and at most 144 uncharged strikes, below). Only one
pass runs at a time: a pass holds a lock file beside the database (`hub.db.lessons-propose.lock`),
and a second one that finds it held does nothing and exits 0. The timer runs a pass 10 minutes
after boot and 10 minutes after each pass ends; the unit's `TimeoutStartSec=30min` is the outer
bound on one pass.

The gateway's key is NOT in `hub.env` (the console, the MCP server and the worker do not need
it). It goes in `<root>/lessons.env`, mode 0600, owned by `root:root`. systemd reads an
`EnvironmentFile` as root before it starts the unit as the service user, so the pass gets the key
although the service user cannot read the file; the agent-facing units (the MCP server, the
worker) run as that same service user, so a file it could read would put the key within an
agent's reach. Create the file empty, then edit it, so the key never lands in a command line or a
shell history:

```
sudo install -m 0600 -o root -g root /dev/null /var/lib/aar/lessons.env
sudoedit /var/lib/aar/lessons.env
```

It holds one required line and two optional ones:

```
AAR_LLM_API_KEY=<the gateway key>
# Optional: the model (default gpt-6-luna) and the gateway (http only on a loopback address).
AAR_LESSONS_MODEL=gpt-6-luna
AAR_LLM_BASE_URL=http://127.0.0.1:8317
```

Then enable the timer, and run one pass now to check it:

```
sudo systemctl enable --now aar-hub-lessons.timer
sudo systemctl start aar-hub-lessons.service
journalctl -u aar-hub-lessons.service -n 40 --no-pager
systemctl list-timers aar-hub-lessons.timer
```

Each pass prints a JSON report (the reviews drafted, the proposals, the model calls and their
token counts, and each failure's charged attempts and strikes). Exit status 0 is a pass that
ran, even with nothing to do, another pass running, or a full backlog. Without `lessons.env` (or
with no key in it) every run exits 2 with `not_configured` and the unit shows as failed; the
timer keeps trying every 10 minutes, and nothing is recorded. A setup failure (the gateway
down, a wrong key, a model it does not serve) exits 1 and charges no review: the next pass
retries the same review. A failure that may be one review's own (a timeout, a 5xx, a 400, a
reply that is not the lessons JSON) moves the pass on to the next review, and two in a row stop
it. Such a failure is charged to its review when any review of the same pass, before or after
it, is drafted; otherwise it is a strike and the pass exits 1. A pass takes the reviews with the
fewest strikes first, then the fewest charged attempts, then the oldest decision, so reviews
that keep failing go behind the others and cannot hold them back.

So a gateway or model outage that fails every chat (the gateway's upstream down, every chat
answered HTTP 400) shows as a failed unit at every run and charges no review, but each run gives
two of the waiting reviews a strike. An outage lasting about 144 passes (a day at 10 minutes)
gives up the reviews waiting through it: with one or two waiting, after those 144 passes; with
more, the strikes spread over them, and giving them all up takes about 72 passes per waiting
review. Nothing reopens a given-up review yet, so its lessons are never drafted. A gateway that
cannot be reached at all, a wrong key or a rate limit records nothing and gives nothing up.

## Console over the tailnet

The console listens on loopback only. Publish it with tailscale serve:

```
sudo tailscale serve --bg --https=8449 http://127.0.0.1:8788
```

`CONSOLE_HOSTNAME` is the `--allowed-host` value: the tailnet DNS name the browser uses.

### `--trust-tailscale-headers` caveat (DESIGN section 7.6)

With this flag the console believes the `Tailscale-User-Login` header, which `tailscale serve`
sets. Any process on the hub host can reach loopback :8788 and send that header itself, so it
can act as any Tailscale user. Identity is network reachability in v1 (D2). The console and
MCP server must stay tailnet-only; before either is exposed beyond the tailnet, a real
per-person credential is needed and the trusted-header mode must go. The MCP server has no
auth at all, and `--agent-writes` lets any caller write under any agent id.

## Enrolling the node

The node token is minted on the hub and written on the PC over stdin; see
`deploy/windows/README.md`.
