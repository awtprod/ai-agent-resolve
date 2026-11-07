<img src="assets/icon.svg" width="64" height="64" alt="">

# AI-Agent-Resolve

Created in [T3 Code](https://t3.codes).

## Linux hub, Windows Resolve node

The Linux host owns the event log, source media, agent MCP server and review console.
Planning, media analysis and QA can run while the Windows edit PC is shut down. Resolve
jobs stay in the hub queue until their assigned node can claim them. The console's **Jobs**
page and MCP `jobs_list` show their state.

This supports preparing several projects at once. Hot operations still share one Resolve
instance and respect human activity, project identity, branch ownership and approval gates.
Running two simultaneous Resolve instances is a separate, unverified capability.

Start with the [remote node guide](docs/REMOTE-NODE.md) and
[Linux media store guide](docs/MEDIA-STORE.md). The [slice board](PLAN.md) distinguishes
implemented work from live tests and features still awaiting approval.

For a Linux checkout with Python 3.11+ and ffmpeg installed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[hub,analyze,drp]'
.venv/bin/aar hub init --db ./hub/events.sqlite --artifacts ./hub/artifacts
```

Run these as separate processes against the same local database and artifact directory:

```bash
.venv/bin/aar hub serve --db ./hub/events.sqlite --artifacts ./hub/artifacts --port 8787
.venv/bin/aar hub console --db ./hub/events.sqlite --artifacts ./hub/artifacts --port 8788
.venv/bin/aar hub nodeapi --db ./hub/events.sqlite --port 8789
```

They bind loopback by default. Expose them only through the trusted tailnet configuration
in the design; the node API also requires an enrolled bearer token. Add `--agent-writes`
to the MCP server to enable job submission and other agent writes. Human approval and
review decisions remain separate. Model-backed transcription and face detection require
their optional extras and local models; no model download is implicit.

## Continuous integration

GitHub Actions runs lint, formatting, and the full non-live test suite on Linux with Python
3.11 and 3.12. It installs ffmpeg so media tests exercise their real tool path. Windows with
Python 3.12 runs the job-store portability tests and the full non-live node suite; those tests
do not use ffmpeg.

The exact Linux equivalent, starting from a clean checkout on Debian or Ubuntu, is:

```bash
sudo apt-get update && sudo apt-get install --yes ffmpeg
for version in 3.11 3.12; do
  "python${version}" -m venv ".venv-ci-${version}"
  ".venv-ci-${version}/bin/python" -m pip install -e '.[dev]'
  AAR_LIVE=0 ".venv-ci-${version}/bin/ruff" check .
  AAR_LIVE=0 ".venv-ci-${version}/bin/ruff" format --check .
  AAR_LIVE=0 ".venv-ci-${version}/bin/pytest" -q \
    --junitxml="junit-linux-${version}.xml"
done
```

The exact Windows PowerShell equivalent for the node job is:

```powershell
py -3.12 -m venv .venv-ci
.venv-ci\Scripts\python -m pip install -e '.[dev]'
$env:AAR_LIVE = "0"
.venv-ci\Scripts\pytest -q tests/node/test_job_store_portability.py `
  --junitxml=junit-windows-portability.xml
.venv-ci\Scripts\pytest -q tests/node -m "not live" `
  --junitxml=junit-windows-node.xml
```
