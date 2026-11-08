<#
.SYNOPSIS
Run one mixed-terrain training invocation with persistent native output handles.
#>
[CmdletBinding(DefaultParameterSetName = 'WarmStart')]
param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [Parameter(Mandatory = $true, ParameterSetName = 'WarmStart')][string]$Checkpoint,
    [Parameter(Mandatory = $true, ParameterSetName = 'Resume')][string]$ResumeCheckpoint,
    [Parameter(ParameterSetName = 'Resume')][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$CheckpointSha256,
    [ValidateRange(1, 2147483647)][int]$NumEnvs = 5120,
    [ValidateRange(1, 2147483647)][int]$MaxIterations = 2000,
    [ValidateRange(0.0, 1.0)][double]$FlatFraction = 0.5,
    [ValidateRange(0.005, 0.16)][double]$StairHeight = 0.04,
    [switch]$HeightCurriculum,
    [switch]$RandomStairHeight,
    [ValidateRange(0.000001, 1.0)][double]$StairRewardScale = 0.1,
    [string]$Device = 'cuda:0',
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidatePattern('^$|^[A-Za-z0-9_.-]+$')][string]$RunName = '',
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceHash = '725c9696e82ef25e4b13e20c648562d17be6c79a6c1bc6ebd81936fbf2590527'
$resume = $PSCmdlet.ParameterSetName -eq 'Resume'
if (-not $RunName) { $RunName = 'mixed_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8) }
$logDirectory = Join-Path $repoRoot "outputs\mixed_terrain_launch\$RunName"
$null = New-Item -ItemType Directory -Path $logDirectory
$statusPath = Join-Path $logDirectory 'launcher_status.json'
$worker = $null
$oldPythonPath = $env:PYTHONPATH
$state = [ordered]@{
    status = 'preparing'; run_name = $RunName; task = 'Hexapod-Mixed-Flat-Stair'
    runtime_root = $repoRoot; resume = $resume; num_envs = $NumEnvs; max_iterations = $MaxIterations
    flat_fraction = $FlatFraction; stair_height = $StairHeight; stair_reward_scale = $StairRewardScale
    automatic_height_curriculum = [bool]$HeightCurriculum; random_stair_height = [bool]$RandomStairHeight
    seed = $Seed; device = $Device; verify_only = [bool]$VerifyOnly; monitoring_enabled = $false
    started_at = [DateTime]::UtcNow.ToString('o'); steps = @(); active_process_id = $null
}
function Save-Status {
    [IO.File]::WriteAllText($statusPath, ($state | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
}
function Quote-Argument {
    param([string]$Argument)
    return '"' + [regex]::Replace([regex]::Replace($Argument, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
function File-Hash {
    param([string]$Path)
    $algorithm = [Security.Cryptography.SHA256]::Create(); $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Invoke-Stage {
    param([string]$Id, [string[]]$Arguments)
    $stdout = Join-Path $logDirectory "$Id.stdout.log"; $stderr = Join-Path $logDirectory "$Id.stderr.log"
    $stage = [ordered]@{ id = $Id; status = 'starting'; executable = $pythonExecutable; arguments = $Arguments
        stdout = $stdout; stderr = $stderr; started_at = [DateTime]::UtcNow.ToString('o') }
    $state.steps += $stage; Save-Status
    $argumentString = ($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $script:worker = Start-Process -FilePath $pythonExecutable -ArgumentList $argumentString -WorkingDirectory $repoRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null = $script:worker.Handle
    $stage['process_id'] = $script:worker.Id; $stage.status = 'running'
    $state.active_process_id = $script:worker.Id; $state.status = 'running'; Save-Status
    $script:worker.WaitForExit() # Blocking OS wait; no console/iteration/status polling.
    $code = $script:worker.ExitCode
    if ($null -eq $code) { throw "Native exit code unavailable for $Id" }
    $stage['exit_code'] = $code; $stage.status = if ($code -eq 0) { 'completed' } else { 'failed' }
    $stage['finished_at'] = [DateTime]::UtcNow.ToString('o')
    $script:worker.Dispose(); $script:worker = $null; $state.active_process_id = $null; Save-Status
    if ($code -ne 0) { throw "$Id failed with exit code $code; see $stderr" }
}
Push-Location -LiteralPath $repoRoot
try {
    Save-Status
    $pythonExecutable = (Resolve-Path -LiteralPath $PythonPath).Path
    $checkpointInput = if ($resume) { $ResumeCheckpoint } else { $Checkpoint }
    $checkpointFlag = if ($resume) { '--resume-checkpoint' } else { '--checkpoint' }
    $checkpointPath = (Resolve-Path -LiteralPath $checkpointInput).Path
    $checkpointHash = File-Hash $checkpointPath
    if ($resume) {
        if ($CheckpointSha256 -and $checkpointHash -ne $CheckpointSha256) {
            throw 'Resume checkpoint differs from the submitted SHA-256.'
        }
    } elseif ($checkpointHash -ne $sourceHash) { throw 'Source checkpoint differs from the verified final model_1999.' }
    $state['python_executable'] = $pythonExecutable; $state['checkpoint'] = $checkpointPath
    $state['source_sha256'] = $checkpointHash
    $env:PYTHONPATH = (Join-Path $repoRoot 'src') + [IO.Path]::PathSeparator + $oldPythonPath
    Invoke-Stage '01_asset' @('-X','utf8','-u',(Join-Path $PSScriptRoot 'validate_robot_asset.py'),'--require-manifest')
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('-X','utf8','-u',(Join-Path $PSScriptRoot 'train_mixed_terrain.py'),
        $checkpointFlag,$checkpointPath,'--checkpoint_sha256',$checkpointHash,
        '--num_envs',$NumEnvs.ToString(),'--max_iterations',$MaxIterations.ToString(),
        '--flat_fraction',$FlatFraction.ToString('R',$culture),'--stair_height',$StairHeight.ToString('R',$culture),
        '--stair_reward_scale',$StairRewardScale.ToString('R',$culture),
        '--seed',$Seed.ToString(),'--device',$Device,'--viz','none','--run_name',$RunName)
    if ($HeightCurriculum) { $arguments += '--height-curriculum' }
    if ($RandomStairHeight) { $arguments += '--random-stair-height' }
    if ($VerifyOnly) { $arguments += '--verify-only' }
    Invoke-Stage '02_train' $arguments
    $state.status = if ($VerifyOnly) { 'verified_initialization_without_ppo' } else { 'completed' }
} catch {
    $state.status = 'failed'; $state['error'] = $_.Exception.Message; Save-Status
    throw
} finally {
    if ($null -ne $script:worker) {
        try {
            if (-not $script:worker.HasExited) {
                $state.status = 'interrupted'; $state['cancelled_process_id'] = $script:worker.Id
                $killer = Start-Process -FilePath (Join-Path $env:WINDIR 'System32\taskkill.exe') `
                    -ArgumentList @('/PID',$script:worker.Id.ToString(),'/T','/F') -WindowStyle Hidden -PassThru -Wait `
                    -RedirectStandardOutput (Join-Path $logDirectory 'cancel.stdout.log') `
                    -RedirectStandardError (Join-Path $logDirectory 'cancel.stderr.log')
                $state['cancel_exit_code'] = $killer.ExitCode; $killer.Dispose()
            }
        } finally { $script:worker.Dispose() }
        $state.active_process_id = $null
    }
    $state['finished_at'] = [DateTime]::UtcNow.ToString('o'); Save-Status
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
