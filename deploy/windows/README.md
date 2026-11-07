# Resolve node on the edit PC (Windows)

`install.ps1` and `run_node.py` reproduce the 2026-11-07 hand deploy. Layout under
`C:\ProgramData\aar` (owner-only ACL):

```
app\            git checkout, detached at the pinned commit
venv\           py -3.X venv with the package installed
run_node.py     the supervisor (copied from app\deploy\windows)
run_node.json   supervisor config (hub_url, node_id, token_file, store, projects_file, log_dir)
projects.json   JSON array of ProjectLocator objects the worker may touch ([] = idle)
edit-pc.token   bearer token (written separately, never by the installer)
jobs.db         the node's job journal
logs\           supervisor-YYYYMMDD.log, node-YYYYMMDD.log
```

## Steps

From an elevated-or-not PowerShell as the owner (the user who sits at the PC):

```powershell
git clone https://github.com/awtprod/ai-agent-resolve $env:TEMP\aar-bootstrap   # or copy deploy\windows
.\install.ps1 install -Commit <full sha> -HubUrl http://<hub tailnet ip>:8789 -PythonVersion 3.13
```

It is idempotent: rerunning re-applies the ACL, moves the checkout to `-Commit`, reinstalls,
re-copies `run_node.py` and re-registers the task, while keeping an existing `run_node.json`,
`projects.json` and token. The task is registered **disabled**.

### Token (separate, stdin only)

Mint it on the hub (`aar hub node enroll edit-pc --db ... --actor human:...`), then on the PC
read it from stdin so it never lands in a command line, history or environment:

```powershell
$t = [Console]::In.ReadLine()
[System.IO.File]::WriteAllText('C:\ProgramData\aar\edit-pc.token', $t, (New-Object System.Text.UTF8Encoding($false)))
```

The file inherits the owner-only ACL from the directory.

### Projects, then enable

Put the exact locators in `projects.json` (the hub's `register` step lists them), then:

```powershell
.\install.ps1 enable
.\install.ps1 status
.\install.ps1 disable      # stops the task, the supervisor AND the whole worker tree; fails if any remain
```

## Why an interactive logon task

The presence guard decides whether a human is using Resolve from the console session's input
idle time. A service or a task running as SYSTEM or in session 0 cannot see that input, so the
guard would misjudge. The task therefore uses `-LogonType Interactive` for the owner's
Windows identity, so it starts at that user's logon in session 1, with no stored password,
`ExecutionTimeLimit` unlimited and `MultipleInstances IgnoreNew`. Consequence: the node is up
only while the owner is logged on at the console.

## The venv launcher process pair

On Windows `venv\Scripts\python.exe` (and `pythonw.exe`, `aar.exe`) are small redirectors that
spawn the base interpreter as a **child**. You will see two processes per Python: the
launcher and the real interpreter. Killing the launcher does not stop the child at once; the
child dies a moment later (probed on 2026-11-07 with a launcher-kill script). Consequences: (1) file locks
such as the job store's are released slightly after the parent exits, so tests and tooling poll
instead of assuming immediate release; (2) `disable` and `status` select processes by
executable path under `C:\ProgramData\aar`, plus `node run` command lines naming that root or the
store, so the pair and the base-interpreter child are all found. No other Python on the PC is touched.

## Resolve 21.1 traps (S7g)

- **Render-mode polling crashes Resolve 21.1.** Each `Project.GetCurrentRenderMode()` call logs
  "Main view page is changed to 11"; polling it at about 2.5 calls/s crashed Resolve after about
  270 calls (2026-11-07). `GetCurrentRenderFormatAndCodec` is untested and suspect. **Keep the
  worker task disabled until S7g lands and a live soak passes.**
- Studio only: 21.1 scripting needs Studio (DESIGN F8).
- Resolve's scripting server takes a while to come up after launch; the supervisor waits
  20 s after it sees `Resolve.exe`.
- `RESOLVE_SCRIPT_API`, `RESOLVE_SCRIPT_LIB` and `PYTHONPATH=<api>\Modules` must be set for the
  worker; the supervisor sets them (override paths in `run_node.json`: `script_api`,
  `script_lib`).
- The Project Manager folder is captured but not restored (PLAN S3 gap), and a project is
  loaded by name inside the current folder.

## Worker lifetime

The worker is its own process tree (`aar.exe` -> venv `python.exe` -> base `python.exe`), started
by the supervisor with `subprocess.run`. To stop it with the supervisor, `run_node.py` places
itself in a Windows job object with `KILL_ON_JOB_CLOSE` at start-up, so children inherit the job
and die when the supervisor does. This is a no-op off Windows and non-fatal if it fails (the failure reason is logged to both
the supervisor and node logs as `supervisor: job object unavailable: ...`); `disable` does not
rely on it.

**`Stop-ScheduledTask` (and ending the task in Task Scheduler) does NOT kill child processes on
this PC.** Measured 2026-11-07: the worker stays alive. That is why both the job object and
`disable`'s process enumeration exist.

Check the job object on a PC with `python run_node.py --self-test-job`: it prints
`job=ok child=<pid>` (or `job=failed: <reason>`) and exits without waiting; the child sleeper
must be gone moments later.
