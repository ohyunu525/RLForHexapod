<#
.SYNOPSIS
Submit one model_999 WSP training job to Windows Task Scheduler and return immediately.
.DESCRIPTION
The logged-on user owns the hidden scheduled action; no password is stored.
There are no triggers, restarts, or progress polls. Native worker output is saved
by train_flat_wsp.ps1. PrepareOnly validates inputs and writes a reviewable plan
without registering or starting a scheduled task.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$CheckpointSha256,

    [Parameter(Mandatory = $true)]
    [string]$WspProvenance,

    [ValidateRange(1, 2147483647)]
    [int]$NumEnvs = 5120,

    [ValidateRange(1, 2147483647)]
    [int]$MaxIterations = 2000,

    [ValidateNotNullOrEmpty()]
    [string]$Device = 'cuda:0',

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 42,

    [switch]$PrepareOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$parentHash = 'c8828d64d8f1257f9a140a3990cf1250c91c111d8afcb80f1aa4f2f66d648601'

function Get-ContentSha256 {
    param([string]$Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

function Quote-NativeArgument {
    param([string]$Argument)
    return '"' + [regex]::Replace([regex]::Replace($Argument, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

Push-Location -LiteralPath $repoRoot
try {
    $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
    $provenancePath = (Resolve-Path -LiteralPath $WspProvenance).Path
    $candidateHash = Get-ContentSha256 $checkpointPath
    if ($candidateHash -ne $CheckpointSha256.ToLowerInvariant()) { throw 'Candidate checkpoint SHA-256 mismatch.' }
    $provenance = [IO.File]::ReadAllText($provenancePath) | ConvertFrom-Json
    if ($provenance.source_sha256 -ne $parentHash -or $provenance.candidate_sha256 -ne $candidateHash) {
        throw 'WSP provenance must pin the reviewed model_999 and this exact candidate.'
    }
    if ($provenance.source_sha256_unchanged -ne $true -or $provenance.normalizer_preserved -ne $true -or
        $provenance.upstream_preserved -ne $true -or $provenance.runtime_inputs_added -ne $false -or
        $provenance.normalization_epsilon -ne 0.01) { throw 'WSP provenance does not preserve the actor input contract.' }
    $editScope = if ($provenance.PSObject.Properties['edit_scope']) { [string]$provenance.edit_scope } else { 'head_only' }
    if ($editScope -eq 'last_two_layers') {
        if ($provenance.method -cne 'nonlinear' -or $provenance.prefix_preserved -ne $true -or
            (@($provenance.preserved_prefix_layers) -join ',') -cne '0,2') {
            throw 'Last-two-layer WSP requires nonlinear method and preserved prefix layers 0,2.'
        }
        $allowedTensors = @('mlp.4.weight', 'mlp.4.bias', 'mlp.6.weight', 'mlp.6.bias')
        $changedTensors = @($provenance.changed_actor_tensors)
        if ($changedTensors.Count -eq 0 -or @($changedTensors | Where-Object { $_ -notin $allowedTensors }).Count -gt 0) {
            throw 'Last-two-layer WSP declares a change outside actor layers 4 and 6.'
        }
    } elseif ($editScope -ne 'head_only') { throw "Unsupported WSP edit_scope: $editScope" }
    $uvPath = @(Get-Command uv -CommandType Application -ErrorAction Stop)[0].Source
    $userName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $runName = 'flat_wsp_stop_' + $stamp + '_' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $taskName = 'RLForHexapod_' + $runName
    $submissionDirectory = Join-Path $repoRoot "outputs\flat_wsp_stop_submit\$runName"
    $launcherDirectory = Join-Path $repoRoot "outputs\flat_wsp_stop_launch\$runName"
    $null = New-Item -ItemType Directory -Path $submissionDirectory
    $receiptPath = Join-Path $submissionDirectory 'submission.json'
    $powershellPath = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $actionArguments = @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
        '-File', (Join-Path $PSScriptRoot 'train_flat_wsp.ps1'),
        '-Task', 'Hexapod-Walk-WspStop', '-Checkpoint', $checkpointPath,
        '-CheckpointSha256', $candidateHash, '-WspProvenance', $provenancePath,
        '-NumEnvs', $NumEnvs.ToString(), '-MaxIterations', $MaxIterations.ToString(),
        '-Device', $Device, '-Seed', $Seed.ToString(), '-InitialStd', '0.05',
        '-RunName', $runName, '-UvPath', $uvPath, '-NoConsoleTail'
    )
    $nativeArguments = ($actionArguments | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
    $receipt = [ordered]@{
        status = 'prepared'; created_at = [DateTime]::UtcNow.ToString('o'); task_name = $taskName
        user = $userName; logon_type = 'InteractiveToken'; requires_user_logged_on = $true
        training_task = 'Hexapod-Walk-WspStop'; experiment = 'hexapod_walk_wsp_stop'; run_name = $runName
        candidate = $checkpointPath; candidate_sha256 = $candidateHash; parent_model_999_sha256 = $parentHash
        wsp_provenance = $provenancePath; wsp_provenance_sha256 = (Get-ContentSha256 $provenancePath)
        wsp_edit_scope = $editScope
        num_envs = $NumEnvs; max_iterations = $MaxIterations; initial_std = 0.05; seed = $Seed
        executable = $powershellPath; arguments = $actionArguments; native_argument_string = $nativeArguments
        working_directory = $repoRoot; launcher_directory = $launcherDirectory
        launcher_status = (Join-Path $launcherDirectory 'launcher_status.json')
        training_stdout = (Join-Path $launcherDirectory '03_train.stdout.log')
        training_stderr = (Join-Path $launcherDirectory '03_train.stderr.log')
        receipt = $receiptPath; registered = $false; launch_accepted = $false
        monitoring_enabled = $false; training_started_verified = $false
        lifecycle = 'One on-demand scheduled action, no triggers/restarts or execution time limit. Task Scheduler owns the process independently of SSH; the user must stay logged on. No training progress is polled after submission.'
        preflight = 'The scheduled launcher validates the robot asset and neutral-action environment before starting the single requested PPO run. No PPO smoke run.'
    }
    function Save-SubmissionReceipt {
        [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
    }
    Save-SubmissionReceipt
    if (-not $PrepareOnly) {
        try {
            $scheduler = New-Object -ComObject 'Schedule.Service'
            $scheduler.Connect()
            $definition = $scheduler.NewTask(0)
            $definition.RegistrationInfo.Description = 'One model_999 WSP warm-start training run; persistent native logs; no monitoring.'
            $definition.Principal.UserId = $userName
            $definition.Principal.LogonType = 3 # TASK_LOGON_INTERACTIVE_TOKEN; no credentials
            $definition.Principal.RunLevel = 0 # Same user's ordinary desktop token
            $definition.Settings.Enabled = $true
            $definition.Settings.Hidden = $true
            $definition.Settings.AllowDemandStart = $true
            $definition.Settings.StartWhenAvailable = $false
            $definition.Settings.DisallowStartIfOnBatteries = $false
            $definition.Settings.StopIfGoingOnBatteries = $false
            $definition.Settings.ExecutionTimeLimit = 'PT0S'
            $definition.Settings.MultipleInstances = 2 # Ignore a second start of this unique task
            $action = $definition.Actions.Create(0)
            $action.Path = $powershellPath
            $action.Arguments = $nativeArguments
            $action.WorkingDirectory = $repoRoot
            [IO.File]::WriteAllText((Join-Path $submissionDirectory 'scheduled_task.xml'), $definition.XmlText,
                [Text.UTF8Encoding]::new($false))
            $folder = $scheduler.GetFolder('\')
            # TASK_CREATE=2 deliberately refuses replacing an existing task.
            $registered = $folder.RegisterTaskDefinition($taskName, $definition, 2, $userName, $null, 3, $null)
            $receipt.registered = $true
            Save-SubmissionReceipt
            $null = $registered.Run($null) # Exactly one start; no state/process/iteration polling follows.
            $receipt.launch_accepted = $true
            $receipt.status = 'submitted_without_monitoring'
            $receipt['submitted_at'] = [DateTime]::UtcNow.ToString('o')
            Save-SubmissionReceipt
        } catch {
            $receipt.status = 'submission_failed'
            $receipt['error'] = $_.Exception.Message
            Save-SubmissionReceipt
            throw
        }
    }
    $receipt | ConvertTo-Json -Depth 10
} finally { Pop-Location }
