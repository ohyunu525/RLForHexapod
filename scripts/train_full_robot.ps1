<#
.SYNOPSIS
Validate the complete robot, run a neutral-action smoke test, then train a fresh walking teacher.
.EXAMPLE
./scripts/train_full_robot.ps1
.EXAMPLE
./scripts/train_full_robot.ps1 -MaxIterations 3000 -DryRun
#>
[CmdletBinding()]
param(
    # Matches HexapodStatePPORunnerCfg.max_iterations.
    [ValidateRange(1, 2147483647)]
    [int]$MaxIterations = 800,

    [ValidateRange(1, 2147483647)]
    [int]$NumEnvs = 4096,

    [ValidateNotNullOrEmpty()]
    [string]$Device = 'cuda:0',

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$runSuffix = [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runName = "full_robot_flat_${runStamp}_${runSuffix}"
$smokeReport = Join-Path $repoRoot "outputs\${runName}_smoke.json"
$uvExecutable = $null
if (-not $DryRun) {
    $uvExecutable = (Get-Command uv -CommandType Application -ErrorAction Stop).Source
}

function Invoke-UvStep {
    param(
        [string]$Label,
        [string[]]$CommandArguments
    )

    Write-Host $Label
    $displayArguments = foreach ($argument in $CommandArguments) {
        if ($argument -match '^[A-Za-z0-9_./:=\\-]+$') {
            $argument
        } else {
            "'" + $argument.Replace("'", "''") + "'"
        }
    }
    Write-Host ('uv ' + ($displayArguments -join ' '))
    if ($DryRun) {
        return
    }

    & $uvExecutable @CommandArguments
    $commandExitCode = $LASTEXITCODE
    if ($commandExitCode -ne 0) {
        throw "$Label failed with exit code $commandExitCode. No later step was started."
    }
}

Push-Location -LiteralPath $repoRoot
try {
    Invoke-UvStep -Label '[1/3] Validating the loaded robot asset' -CommandArguments @(
        'run', '--locked', 'python', 'scripts/validate_robot_asset.py', '--require-manifest'
    )

    Invoke-UvStep -Label '[2/3] Checking neutral-action simulation' -CommandArguments @(
        'run', '--locked', 'python', 'scripts/smoke_walk.py',
        '--device', $Device, '--viz', 'none', '--seed', '42', '--report', $smokeReport
    )

    Invoke-UvStep -Label '[3/3] Training a fresh walking teacher' -CommandArguments @(
        'run', '--locked', 'isaaclab', 'train', '--rl_library', 'rsl_rl',
        '--task', 'Hexapod-Walk', '--num_envs', $NumEnvs.ToString(),
        '--device', $Device, '--viz', 'none', '--seed', '42',
        '--max_iterations', $MaxIterations.ToString(), '--logger', 'tensorboard',
        '--run_name', $runName
    )
} finally {
    Pop-Location
}
