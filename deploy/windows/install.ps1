<#
.SYNOPSIS
  Install the AI-Agent-Resolve Resolve node on the edit PC. Idempotent.

.DESCRIPTION
  install   Create C:\ProgramData\aar with an owner-only ACL, clone/update the app at -Commit,
            create the venv with the chosen Python, install the package, copy run_node.py and
            write run_node.json (kept if present) and an empty projects.json (kept if present),
            and register the interactive logon task. The task is registered DISABLED.
  enable    Enable the task (and start it if the owner is logged on).
  disable   Stop and disable the task.
  status    Print the task, supervisor processes and the last supervisor log lines.

  Nothing here writes the token. Write it separately over stdin (deploy/windows/README.md).

.EXAMPLE
  .\install.ps1 install -Commit <sha> -HubUrl http://100.x.y.z:8789 -PythonVersion 3.13
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('install', 'enable', 'disable', 'status')]
    [string]$Command,
    [string]$Root = 'C:\ProgramData\aar',
    [string]$Repo = 'https://github.com/awtprod/ai-agent-resolve',
    [string]$Commit,
    [string]$HubUrl,
    [string]$NodeId = 'edit-pc',
    [string]$PythonVersion = '3.13',
    [string]$TaskName = 'AAR Resolve node'
)

$ErrorActionPreference = 'Stop'

function Get-CurrentUserName {
    [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
}

function Get-AllowedSids {
    # Current user, SYSTEM, Administrators. SIDs, never names.
    $meSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    @($meSid.Value, 'S-1-5-18', 'S-1-5-32-544')
}

function Get-AclProblems([string]$Path, [bool]$RequireProtected) {
    # Returns a list of problems: 'UNPROTECTED' and/or every ACE SID outside the allowlist.
    $allowed = Get-AllowedSids
    $acl = Get-Acl -LiteralPath $Path
    $problems = @()
    if ($RequireProtected -and -not $acl.AreAccessRulesProtected) { $problems += 'UNPROTECTED' }
    foreach ($rule in $acl.Access) {
        try {
            $sid = $rule.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
        } catch {
            $sid = [string]$rule.IdentityReference
        }
        if ($allowed -notcontains $sid) { $problems += $sid }
    }
    @($problems | Select-Object -Unique)
}

function Set-OwnerOnlyAcl([string]$Path) {
    # Rebuild the root DACL from the allowlist instead of editing it: the additive icacls
    # form leaves any existing explicit ACE (Users, Everyone) in place.
    $sec = New-Object System.Security.AccessControl.DirectorySecurity
    $sec.SetAccessRuleProtection($true, $false)   # protected, drop inherited ACEs
    $inherit = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
    $none = [System.Security.AccessControl.PropagationFlags]::None
    foreach ($sidText in Get-AllowedSids) {
        $sid = New-Object System.Security.Principal.SecurityIdentifier($sidText)
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $sid, [System.Security.AccessControl.FileSystemRights]::FullControl, $inherit, $none,
            [System.Security.AccessControl.AccessControlType]::Allow)
        $sec.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $sec
    # Descendants (an existing token, jobs.db, logs) must only inherit from the new root.
    if (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1) {
        icacls "$Path\*" /reset /T /C | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "icacls /reset failed with exit code $LASTEXITCODE" }
    }
    $bad = Get-AclProblems $Path $true
    if ($bad.Count -gt 0) { throw "ACL on $Path is not owner-only: $($bad -join ', ')" }
    $token = Join-Path $Path 'edit-pc.token'
    if (Test-Path -LiteralPath $token) {
        $bad = Get-AclProblems $token $false
        if ($bad.Count -gt 0) { throw "ACL on $token is not owner-only: $($bad -join ', ')" }
    }
}

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Install-Node {
    if (-not $Commit) { throw '-Commit is required for install (a full sha; deployments are pinned).' }
    if (-not $HubUrl) { throw '-HubUrl is required for install (http://<hub tailnet ip>:8789).' }

    if (-not (Test-Path $Root)) { New-Item -ItemType Directory -Path $Root | Out-Null }
    Set-OwnerOnlyAcl $Root
    New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null

    $app = Join-Path $Root 'app'
    if (-not (Test-Path (Join-Path $app '.git'))) { Invoke-Git clone -q $Repo $app }
    Invoke-Git -C $app fetch -q origin
    Invoke-Git -C $app checkout -q --detach $Commit
    Write-Output ('app at: ' + (& git -C $app log -1 --format='%h %s'))

    $venv = Join-Path $Root 'venv'
    $py = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $py)) {
        & py "-$PythonVersion" -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "py -$PythonVersion -m venv failed ($LASTEXITCODE)" }
    }
    & $py -m pip -q install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed ($LASTEXITCODE)" }
    & $py -m pip -q install $app
    if ($LASTEXITCODE -ne 0) { throw "pip install failed ($LASTEXITCODE)" }
    & $py -c 'import aar,sys; print(aar.__file__, sys.version.split()[0])'

    # The supervisor and its config are copied, never imported from the app checkout.
    Copy-Item -Force (Join-Path $app 'deploy\windows\run_node.py') (Join-Path $Root 'run_node.py')
    $cfg = Join-Path $Root 'run_node.json'
    if (-not (Test-Path $cfg)) {
        $obj = [ordered]@{
            hub_url = $HubUrl; node_id = $NodeId; token_file = 'edit-pc.token'
            store = 'jobs.db'; projects_file = 'projects.json'; log_dir = 'logs'
        }
        Write-Utf8NoBom $cfg ($obj | ConvertTo-Json)
    }
    $projects = Join-Path $Root 'projects.json'
    if (-not (Test-Path $projects)) { Write-Utf8NoBom $projects '[]' }

    # Interactive logon task: the presence guard reads input idle, which only the console
    # session can see. -LogonType Interactive runs it in that session at logon, no password.
    $me = Get-CurrentUserName
    $action = New-ScheduledTaskAction -Execute (Join-Path $venv 'Scripts\pythonw.exe') `
        -Argument ('"' + (Join-Path $Root 'run_node.py') + '"') -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $me
    $principal = New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings -Force `
        -Description 'AI-Agent-Resolve: Resolve node worker (supervisor waits for Resolve, dials the hub)' | Out-Null
    # Registered disabled: nothing runs until the owner writes the token and runs `enable`.
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
    Write-Output "task '$TaskName' registered DISABLED. Write the token, then: install.ps1 enable"
    (icacls $Root) -join "`n"
}

function Enable-Node {
    if (-not (Test-Path (Join-Path $Root 'edit-pc.token'))) {
        throw "no token at $Root\edit-pc.token; write it first (README: Token)."
    }
    Enable-ScheduledTask -TaskName $TaskName | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Show-Status
}

function Get-NodeProcesses {
    # Every process that belongs to this install, and nothing else:
    #  (a) its executable lives under $Root (venv python.exe/pythonw.exe/aar.exe), or
    #  (b) its command line has `node run` or `run_node.py` together with $Root or the configured store path
    #      (catches the base-interpreter child the venv redirector spawns, whose executable is
    #      outside $Root).
    # Other Python processes (for example the Resolve MCP bridge's pythonw.exe) match neither.
    $prefix = ($Root.TrimEnd('\') + '\')
    $store = Join-Path $Root 'jobs.db'
    $cfg = Join-Path $Root 'run_node.json'
    if (Test-Path $cfg) {
        try {
            $c = Get-Content $cfg -Raw | ConvertFrom-Json
            if ($c.store) {
                $store = if ([System.IO.Path]::IsPathRooted($c.store)) { $c.store } else { Join-Path $Root $c.store }
            }
        } catch { }
    }
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and (
            ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) -or
            ($_.CommandLine -and ($_.CommandLine -like '*node run*' -or $_.CommandLine -like '*run_node.py*') -and
                ($_.CommandLine.IndexOf($store, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                 $_.CommandLine.IndexOf($prefix, [System.StringComparison]::OrdinalIgnoreCase) -ge 0))
        )
    })
}

function Disable-Node {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
    # The worker is its own process tree (aar.exe -> venv python -> base python), not a child
    # job of the supervisor, so killing run_node.py alone would orphan it. Kill the whole set.
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        $procs = Get-NodeProcesses
        if ($procs.Count -eq 0) { break }
        foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 2
    }
    $left = Get-NodeProcesses
    Show-Status
    if ($left.Count -gt 0) {
        throw ('disable FAILED: still running: ' + (($left | ForEach-Object { "$($_.Name)#$($_.ProcessId)" }) -join ', '))
    }
}

function Show-Status {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) { "task: $($t.TaskName) state=$($t.State)" } else { "task '$TaskName' is not registered" }
    if (Test-Path -LiteralPath $Root) {
        $bad = @(Get-AclProblems $Root $true)
        $token = Join-Path $Root 'edit-pc.token'
        if (Test-Path -LiteralPath $token) { $bad += @(Get-AclProblems $token $false) }
        if ($bad.Count -eq 0) { 'acl: owner-only' } else { "acl: EXTRA $(($bad | Select-Object -Unique) -join ' ')" }
    }
    Get-NodeProcesses | ForEach-Object { "pid=$($_.ProcessId) session=$($_.SessionId) $($_.ExecutablePath) :: $($_.CommandLine)" }
    Get-ChildItem (Join-Path $Root 'logs') -Filter 'supervisor-*.log' -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -Last 1 | ForEach-Object { Get-Content $_.FullName -Tail 5 }
}

switch ($Command) {
    'install' { Install-Node }
    'enable'  { Enable-Node }
    'disable' { Disable-Node }
    'status'  { Show-Status }
}
