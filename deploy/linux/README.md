# Hub on Linux (systemd)

Three units run the hub: `aar-hub-mcp` (agent MCP server, loopback :8787, agent writes on),
`aar-hub-console` (human console, loopback :8788) and `aar-hub-nodeapi` (node API for the
Windows Resolve node, tailnet IP :8789 only). The files here are templates
(`*.service.in`) with `@PLACEHOLDER@` fields; nothing in this kit writes to `/etc` or starts a
service.

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

## Choosing the root: not `/srv` for Command Center agents

Pick a root outside `/srv` if Command Center agents need to operate on it (for example to run
`aar hub ...` administrative commands). `/srv` is read-only in their sandbox, so writes to
`hub.db` or the artifacts fail there even though the service itself (a system unit) can write
to it. The 2026-11-07 deploy used `/var/lib/command-center/runtime/services/aar-hub`.

## Render and install

```
AAR_ROOT=/var/lib/aar AAR_USER=aar AAR_GROUP=aar \
TAILNET_IP=100.x.y.z CONSOLE_HOSTNAME=host.tailnet.ts.net \
  deploy/linux/install-units.sh ./rendered
sudo cp rendered/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aar-hub-mcp aar-hub-console aar-hub-nodeapi
```

`TAILNET_IP` is the hub's own tailnet address; the node API binds to it alone (not
`0.0.0.0`), so `Wants=tailscaled.service` orders it after tailscale.

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
