<#
.SYNOPSIS
Refine model_1999 with changed rewards, preserving its exact actor mean and normalization.
.EXAMPLE
./scripts/train_flat_refine.ps1
.EXAMPLE
./scripts/train_flat_refine.ps1 -MaxIterations 5 -NumEnvs 5120
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)]
    [int]$MaxIterations = 1000,

    [ValidateRange(1, 2147483647)]
    [int]$NumEnvs = 5120,

    [ValidateNotNullOrEmpty()]
    [string]$Device = 'cuda:0',

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 42,

    [ValidateNotNullOrEmpty()]
    [string]$Checkpoint = 'logs/rsl_rl/hexapod_walk_wsp/2026-11-08_05-09-04_flat_wsp_20261108T050840347Z_abbf5ff6_63b7c699/model_1999.pt',

    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$CheckpointSha256 = 'e8a7da57b3fbcf7a9a2bc4569b1aa93653c0691525cb2f5f1c4cf1cad98b8734',

    [ValidateRange(0.03, 0.15)]
    [double]$InitialStd = 0.08,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$parameters = @{
    Task = 'Hexapod-Walk-Refine'; MaxIterations = $MaxIterations; NumEnvs = $NumEnvs
    Device = $Device; Seed = $Seed; Checkpoint = $Checkpoint; CheckpointSha256 = $CheckpointSha256
    InitialStd = $InitialStd; DryRun = $DryRun
}
& (Join-Path $PSScriptRoot 'train_flat_wsp.ps1') @parameters
