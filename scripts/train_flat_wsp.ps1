<#
.SYNOPSIS
Validate the robot and warm-start a new PPO run from an exact saved actor.
.EXAMPLE
./scripts/train_flat_wsp.ps1 -MaxIterations 3000 -NumEnvs 1024
.EXAMPLE
./scripts/train_flat_wsp.ps1 -MaxIterations 2 -NumEnvs 64
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)]
    [int]$MaxIterations = 3000,

    [ValidateRange(1, 2147483647)]
    [int]$NumEnvs = 1024,

    [ValidateNotNullOrEmpty()]
    [string]$Device = 'cuda:0',

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 42,

    [ValidateNotNullOrEmpty()]
    [string]$Checkpoint = 'outputs/flat_wsp/physical_20261108/tripod_probes/model_13250_wsp_tripod_compact_v2.pt',

    [ValidateSet('Hexapod-Walk-WSP', 'Hexapod-Walk-Refine', 'Hexapod-Walk-WspStop')]
    [string]$Task = 'Hexapod-Walk-WSP',

    [Nullable[double]]$InitialStd = $null,

    [ValidatePattern('^$|^[0-9a-fA-F]{64}$')]
    [string]$CheckpointSha256 = '',

    [string]$WspProvenance = '',

    [ValidatePattern('^$|^[A-Za-z0-9_.-]+$')]
    [string]$RunName = '',

    [string]$UvPath = '',

    [switch]$NoConsoleTail,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$runSuffix = [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runPrefix = switch ($Task) {
    'Hexapod-Walk-Refine' { 'flat_refine' }
    'Hexapod-Walk-WspStop' { 'flat_wsp_stop' }
    default { 'flat_wsp' }
}
if ($null -eq $InitialStd) {
    $InitialStd = switch ($Task) {
        'Hexapod-Walk-Refine' { 0.08 }
        'Hexapod-Walk-WspStop' { 0.05 }
        default { 0.10 }
    }
}
if (-not $RunName) { $RunName = "${runPrefix}_${runStamp}_${runSuffix}" }
$smokeReport = Join-Path $repoRoot "outputs\${runName}_smoke.json"
$launchDirectory = Join-Path $repoRoot "outputs\${runPrefix}_launch\$runName"
$statusPath = Join-Path $launchDirectory 'launcher_status.json'
$uvExecutable = $null
$script:activeWorker = $null
$script:displayAvailable = -not $NoConsoleTail
$launchState = [ordered]@{
    status = 'preparing'; run_name = $runName; started_at = [DateTime]::UtcNow.ToString('o')
    max_iterations = $MaxIterations; num_envs = $NumEnvs; seed = $Seed; device = $Device
    checkpoint = $Checkpoint; task = $Task; initial_std = $InitialStd
    expected_source_sha256 = $CheckpointSha256; wsp_provenance = $WspProvenance
    console_tail_enabled = -not $NoConsoleTail; steps = @(); active_process_id = $null
    cancellation_policy = 'Ctrl+C stops only the uv process tree launched by this script. Saved checkpoints remain. Abrupt OS termination may bypass cleanup; inspect the recorded PID before relaunching.'
}

function Write-LauncherMessage {
    param([string]$Message)
    if ($script:displayAvailable) {
        try { Write-Host $Message }
        catch {
            # A disconnected console/output pipe must not terminate the training
            # worker, whose stdout and stderr go directly to persistent files.
            if ($_.Exception -is [System.Management.Automation.PipelineStoppedException]) { throw }
            $script:displayAvailable = $false
        }
    }
}

function Save-LauncherStatus {
    if ($DryRun -or -not [IO.Directory]::Exists($launchDirectory)) { return }
    $launchState['updated_at'] = [DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText($statusPath, ($launchState | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
}

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Argument)
    # Start-Process joins ArgumentList rather than escaping each argument.
    # Quote using Windows CRT rules, including backslashes before quotes/end.
    return '"' + [regex]::Replace([regex]::Replace($Argument, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function Show-NewLogTail {
    param([string]$Path, [ref]$Offset)
    if (-not $script:displayAvailable -or -not [IO.File]::Exists($Path)) { return }
    $stream = $null
    try {
        $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
            ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
        $length = $stream.Length
        if ($length -le $Offset.Value) { return }
        # Display is bounded; the complete original output remains in the file.
        $amount = [int][Math]::Min(16384, $length - $Offset.Value)
        $null = $stream.Seek($length - $amount, [IO.SeekOrigin]::Begin)
        $buffer = New-Object byte[] $amount
        $read = $stream.Read($buffer, 0, $amount)
        $Offset.Value = $stream.Position
        $text = [Text.Encoding]::UTF8.GetString($buffer, 0, $read)
        $lines = @($text -split '\r?\n')
        $start = [Math]::Max(0, $lines.Length - 30)
        for ($i = $start; $i -lt $lines.Length; $i++) {
            if ($lines[$i]) { Write-LauncherMessage $lines[$i] }
        }
    } catch {
        if ($_.Exception -is [System.Management.Automation.PipelineStoppedException]) { throw }
        # Monitoring is best effort. Native output already has durable handles.
        $launchState['last_display_error'] = $_.Exception.Message
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
    }
}

function Invoke-UvStep {
    param([string]$Id, [string]$Label, [string[]]$CommandArguments)

    Write-LauncherMessage $Label
    $displayArguments = foreach ($argument in $CommandArguments) {
        if ($argument -match '^[A-Za-z0-9_./:=\\-]+$') {
            $argument
        } else {
            "'" + $argument.Replace("'", "''") + "'"
        }
    }
    Write-LauncherMessage ('uv ' + ($displayArguments -join ' '))
    if ($DryRun) { return }
    $stdoutPath = Join-Path $launchDirectory "$Id.stdout.log"
    $stderrPath = Join-Path $launchDirectory "$Id.stderr.log"
    $step = [ordered]@{
        id = $Id; label = $Label; status = 'starting'; started_at = [DateTime]::UtcNow.ToString('o')
        executable = $uvExecutable; arguments = $CommandArguments
        stdout = $stdoutPath; stderr = $stderrPath; process_id = $null; exit_code = $null
    }
    $launchState.steps += $step
    $launchState.status = 'running'
    Save-LauncherStatus
    $nativeArguments = ($CommandArguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
    $script:activeWorker = Start-Process -FilePath $uvExecutable -ArgumentList $nativeArguments `
        -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    # Retain the process handle so ExitCode remains readable after termination.
    $null = $script:activeWorker.Handle
    $step.process_id = $script:activeWorker.Id
    $step.status = 'running'
    $launchState.active_process_id = $script:activeWorker.Id
    Save-LauncherStatus
    Write-LauncherMessage "Worker PID $($step.process_id); logs: $stdoutPath and $stderrPath"
    [long]$outOffset = 0
    [long]$errOffset = 0
    if ($NoConsoleTail) {
        # Detached jobs have persistent native handles and no progress monitor.
        $script:activeWorker.WaitForExit()
    } else {
        while (-not $script:activeWorker.HasExited) {
            Show-NewLogTail $stdoutPath ([ref]$outOffset)
            Show-NewLogTail $stderrPath ([ref]$errOffset)
            Start-Sleep -Milliseconds 500
            $script:activeWorker.Refresh()
        }
    }
    $script:activeWorker.WaitForExit()
    Show-NewLogTail $stdoutPath ([ref]$outOffset)
    Show-NewLogTail $stderrPath ([ref]$errOffset)
    $exitCode = $script:activeWorker.ExitCode
    if ($null -eq $exitCode) { throw "Could not read native exit code for $Id" }
    $step.exit_code = $exitCode
    $step.status = if ($exitCode -eq 0) { 'completed' } else { 'failed' }
    $step['finished_at'] = [DateTime]::UtcNow.ToString('o')
    $script:activeWorker.Dispose()
    $script:activeWorker = $null
    $launchState.active_process_id = $null
    Save-LauncherStatus
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode. See $stderrPath and $stdoutPath. No later step was started."
    }
}

Push-Location -LiteralPath $repoRoot
try {
    if (-not $DryRun) {
        $null = New-Item -ItemType Directory -Path $launchDirectory
        Save-LauncherStatus
        Write-LauncherMessage "Launcher status and logs: $launchDirectory"
        $uvExecutable = if ($UvPath) { (Resolve-Path -LiteralPath $UvPath).Path } else {
            @(Get-Command uv -CommandType Application -ErrorAction Stop)[0].Source
        }
    }
    if ($Task -eq 'Hexapod-Walk-WspStop' -and (-not $CheckpointSha256 -or -not $WspProvenance)) {
        throw 'WspStop requires an explicit checkpoint SHA-256 and WSP provenance JSON.'
    }
    if ($WspProvenance) {
        $WspProvenance = (Resolve-Path -LiteralPath $WspProvenance).Path
        $launchState.wsp_provenance = $WspProvenance
    }
    if ([double]::IsNaN($InitialStd) -or [double]::IsInfinity($InitialStd) -or $InitialStd -le 0 -or $InitialStd -gt 1) {
        throw 'InitialStd must be finite and in (0, 1].'
    }
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Source checkpoint not found: $Checkpoint"
    }
    $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
    # Use the framework directly: inherited PSModulePath can hide Get-FileHash
    # when Windows PowerShell 5.1 is started from a newer PowerShell host.
    $hashAlgorithm = [Security.Cryptography.SHA256]::Create()
    $checkpointStream = [IO.File]::OpenRead($checkpointPath)
    try {
        $sourceHash = [BitConverter]::ToString($hashAlgorithm.ComputeHash($checkpointStream)).Replace('-', '').ToLowerInvariant()
    } finally {
        $checkpointStream.Dispose()
        $hashAlgorithm.Dispose()
    }
    $launchState['source_sha256'] = $sourceHash
    if ($CheckpointSha256 -and $sourceHash -ne $CheckpointSha256.ToLowerInvariant()) {
        throw "Source checkpoint SHA-256 mismatch: expected $CheckpointSha256, got $sourceHash"
    }
    Invoke-UvStep -Id '01_asset' -Label '[1/3] Validating the loaded robot asset' -CommandArguments @(
        'run', '--locked', 'python', '-X', 'utf8', '-u', 'scripts/validate_robot_asset.py', '--require-manifest'
    )
    $smokeArguments = @(
        'run', '--locked', 'python', '-X', 'utf8', '-u', 'scripts/smoke_walk.py',
        '--device', $Device, '--viz', 'none', '--seed', $Seed.ToString(), '--report', $smokeReport
    )
    if ($Task -eq 'Hexapod-Walk-WspStop') { $smokeArguments += @('--task', $Task) }
    Invoke-UvStep -Id '02_smoke' -Label '[2/3] Checking neutral-action simulation' -CommandArguments $smokeArguments
    $trainingArguments = @(
        'run', '--locked', 'python', '-X', 'utf8', '-u', 'scripts/train_flat_wsp.py',
        '--task', $Task, '--initial_std', $InitialStd.ToString('R', [Globalization.CultureInfo]::InvariantCulture),
        '--checkpoint', $checkpointPath, '--num_envs', $NumEnvs.ToString(),
        '--device', $Device, '--viz', 'none', '--seed', $Seed.ToString(),
        '--max_iterations', $MaxIterations.ToString(), '--run_name', $runName
    )
    if ($CheckpointSha256) { $trainingArguments += @('--checkpoint_sha256', $CheckpointSha256.ToLowerInvariant()) }
    if ($WspProvenance) { $trainingArguments += @('--wsp_provenance', $WspProvenance) }
    Invoke-UvStep -Id '03_train' -Label "[3/3] Starting $Task with exact actor and fresh PPO state" -CommandArguments $trainingArguments
    $launchState.status = 'completed'
} catch {
    $launchState.status = 'failed'
    $launchState['error'] = $_.Exception.Message
    Save-LauncherStatus
    Write-LauncherMessage "Launch failed: $($_.Exception.Message)"
    Write-LauncherMessage "Persistent logs: $launchDirectory"
    throw
} finally {
    if ($null -ne $script:activeWorker) {
        try {
            if (-not $script:activeWorker.HasExited) {
                $launchState.status = 'interrupted'
                $launchState['cancelled_process_id'] = $script:activeWorker.Id
                # Ctrl+C cleanup targets only this invocation's worker tree.
                $killer = Start-Process -FilePath (Join-Path $env:WINDIR 'System32\taskkill.exe') `
                    -ArgumentList @('/PID', $script:activeWorker.Id.ToString(), '/T', '/F') `
                    -WindowStyle Hidden -PassThru -Wait `
                    -RedirectStandardOutput (Join-Path $launchDirectory 'cancel.stdout.log') `
                    -RedirectStandardError (Join-Path $launchDirectory 'cancel.stderr.log')
                $launchState['cancel_exit_code'] = $killer.ExitCode
                $killer.Dispose()
            }
        } finally { $script:activeWorker.Dispose() }
        $launchState.active_process_id = $null
    }
    $launchState['finished_at'] = [DateTime]::UtcNow.ToString('o')
    Save-LauncherStatus
    Pop-Location
}
