<#
.SYNOPSIS
Train harder flat velocity commands from the pre-stair actor, with persistent logs.
.EXAMPLE
./scripts/run_flat_commands.ps1 -PythonPath .venv/Scripts/python.exe -Difficulty hard
.EXAMPLE
./scripts/run_flat_commands.ps1 -PythonPath .venv/Scripts/python.exe -LinVelX @(-0.2, 0.35) -CommandInterval @(2, 4) -VerifyOnly
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$PythonPath,
    [ValidateRange(1, 2147483647)][int]$NumEnvs = 5120,
    [ValidateRange(1, 2147483647)][int]$MaxIterations = 2000,
    [ValidateNotNullOrEmpty()][string]$Device = 'cuda:0',
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateSet('standard', 'hard', 'expert', 'extreme')][string]$Difficulty = 'hard',
    [double[]]$LinVelX,
    [double[]]$LinVelY,
    [double[]]$YawRate,
    [double[]]$CommandInterval,
    [Nullable[double]]$StandingFraction = $null,
    [string]$Checkpoint = '',
    [ValidatePattern('^$|^[0-9a-fA-F]{64}$')][string]$CheckpointSha256 = '',
    [switch]$VerifyOnly,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$culture = [Globalization.CultureInfo]::InvariantCulture
$runName = 'flat_commands_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$worker = $null
$logDirectory = $null
$oldPythonPath = $env:PYTHONPATH

function Quote-Argument {
    param([string]$Argument)
    # Start-Process joins ArgumentList; apply Windows CRT quoting explicitly.
    return '"' + [regex]::Replace([regex]::Replace($Argument, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
function File-Hash {
    param([string]$Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Save-Status {
    [IO.File]::WriteAllText((Join-Path $logDirectory 'launcher_status.json'),
        ($state | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
}

Push-Location -LiteralPath $repoRoot
try {
    $pythonExecutable = (Resolve-Path -LiteralPath $PythonPath).Path
    if (-not [IO.File]::Exists($pythonExecutable)) { throw 'PythonPath must name an executable file.' }
    if ($CheckpointSha256 -and -not $Checkpoint) { throw 'CheckpointSha256 requires an explicit Checkpoint.' }
    $arguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'train_flat_wsp.py'),
        '--task', 'Hexapod-Walk-Commands', '--command-difficulty', $Difficulty.ToLowerInvariant(),
        '--num_envs', $NumEnvs.ToString($culture), '--max_iterations', $MaxIterations.ToString($culture),
        '--seed', $Seed.ToString($culture), '--device', $Device, '--viz', 'none', '--run_name', $runName)
    $ranges = [ordered]@{
        '--lin-vel-x' = $LinVelX; '--lin-vel-y' = $LinVelY
        '--yaw-rate' = $YawRate; '--command-interval' = $CommandInterval
    }
    foreach ($entry in $ranges.GetEnumerator()) {
        if ($null -eq $entry.Value) { continue }
        $values = @($entry.Value)
        if ($values.Count -ne 2) { throw "$($entry.Key) requires exactly two values." }
        foreach ($value in $values) {
            if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { throw "$($entry.Key) values must be finite." }
        }
        if ($values[0] -gt $values[1]) { throw "$($entry.Key) minimum must not exceed maximum." }
        if ($entry.Key -eq '--command-interval' -and $values[0] -le 0) { throw 'CommandInterval values must be positive.' }
        $arguments += $entry.Key
        $arguments += @($values | ForEach-Object { $_.ToString('R', $culture) })
    }
    if ($null -ne $StandingFraction) {
        if ([double]::IsNaN($StandingFraction) -or [double]::IsInfinity($StandingFraction) -or
            $StandingFraction -lt 0 -or $StandingFraction -gt 1) { throw 'StandingFraction must be finite and in [0, 1].' }
        $arguments += @('--standing-fraction', $StandingFraction.ToString('R', $culture))
    }
    $checkpointPath = $null
    $checkpointHash = $null
    if ($Checkpoint) {
        $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
        $checkpointHash = File-Hash $checkpointPath
        if ($CheckpointSha256 -and $checkpointHash -ne $CheckpointSha256.ToLowerInvariant()) {
            throw 'Checkpoint SHA-256 mismatch.'
        }
        $arguments += @('--checkpoint', $checkpointPath, '--checkpoint_sha256', $checkpointHash)
    }
    # Omitting both source flags selects the Python task's pinned pre-stair model.
    if ($VerifyOnly) { $arguments += '--verify-only' }
    $state = [ordered]@{
        status = 'prepared'; task = 'Hexapod-Walk-Commands'; run_name = $runName
        executable = $pythonExecutable; arguments = $arguments; runtime_root = $repoRoot
        checkpoint = $checkpointPath; source_sha256 = $checkpointHash
        source_selection = $(if ($Checkpoint) { 'explicit_checkpoint' } else { 'pinned_pre_stair_default' })
        command_difficulty = $Difficulty.ToLowerInvariant(); num_envs = $NumEnvs; max_iterations = $MaxIterations
        verify_only = [bool]$VerifyOnly; monitoring_enabled = $false; active_process_id = $null
        started_at = [DateTime]::UtcNow.ToString('o')
    }
    if ($DryRun) {
        $state | ConvertTo-Json -Depth 10
        return
    }
    $logDirectory = Join-Path $repoRoot "outputs\flat_commands_launch\$runName"
    $null = New-Item -ItemType Directory -Path $logDirectory
    $stdout = Join-Path $logDirectory 'train.stdout.log'
    $stderr = Join-Path $logDirectory 'train.stderr.log'
    $state['stdout'] = $stdout
    $state['stderr'] = $stderr
    Save-Status
    $env:PYTHONPATH = (Join-Path $repoRoot 'src') + [IO.Path]::PathSeparator + $oldPythonPath
    $argumentString = ($arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $worker = Start-Process -FilePath $pythonExecutable -ArgumentList $argumentString -WorkingDirectory $repoRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null = $worker.Handle
    $state.active_process_id = $worker.Id
    $state.status = 'running'
    Save-Status
    $worker.WaitForExit() # Blocking OS wait without progress polling.
    $exitCode = $worker.ExitCode
    if ($null -eq $exitCode) { throw 'Native exit code unavailable.' }
    $state['exit_code'] = $exitCode
    $worker.Dispose()
    $worker = $null
    $state.active_process_id = $null
    if ($exitCode -ne 0) { throw "Training exited with code $exitCode. See $stderr" }
    $state.status = if ($VerifyOnly) { 'verified_initialization_without_ppo' } else { 'completed' }
} catch {
    if ($logDirectory) {
        $state.status = 'failed'
        $state['error'] = $_.Exception.Message
    }
    throw
} finally {
    if ($null -ne $worker) {
        try {
            if (-not $worker.HasExited) {
                $state.status = 'interrupted'
                $state['cancelled_process_id'] = $worker.Id
                $killer = Start-Process -FilePath (Join-Path $env:WINDIR 'System32\taskkill.exe') `
                    -ArgumentList @('/PID', $worker.Id.ToString(), '/T', '/F') -WindowStyle Hidden -PassThru -Wait `
                    -RedirectStandardOutput (Join-Path $logDirectory 'cancel.stdout.log') `
                    -RedirectStandardError (Join-Path $logDirectory 'cancel.stderr.log')
                $state['cancel_exit_code'] = $killer.ExitCode
                $killer.Dispose()
            }
        } finally { $worker.Dispose() }
        $state.active_process_id = $null
    }
    if ($logDirectory) {
        $state['finished_at'] = [DateTime]::UtcNow.ToString('o')
        Save-Status
    }
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
