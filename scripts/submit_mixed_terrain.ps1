<#
.SYNOPSIS
Submit one 5120-environment, 2000-iteration mixed-terrain run without monitoring.
.DESCRIPTION
PythonPath is the original installed Isaac Lab venv's absolute python.exe path.
This script and its src tree may live in an isolated runtime/worktree. PrepareOnly
records the exact action and hashes without registering or starting a task.
#>
[CmdletBinding(DefaultParameterSetName = 'WarmStart')]
param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [Parameter(Mandatory = $true, ParameterSetName = 'WarmStart')][string]$Checkpoint,
    [Parameter(Mandatory = $true, ParameterSetName = 'Resume')][string]$ResumeCheckpoint,
    [ValidateRange(1, 2147483647)][int]$NumEnvs = 5120,
    [ValidateRange(1, 2147483647)][int]$MaxIterations = 2000,
    [ValidateRange(0.0, 1.0)][double]$FlatFraction = 0.5,
    [ValidateRange(0.005, 0.16)][double]$StairHeight = 0.04,
    [switch]$HeightCurriculum,
    [switch]$RandomStairHeight,
    [ValidateRange(0.000001, 1.0)][double]$StairRewardScale = 0.1,
    [string]$Device = 'cuda:0',
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [switch]$PrepareOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceHash = '725c9696e82ef25e4b13e20c648562d17be6c79a6c1bc6ebd81936fbf2590527'
$resume = $PSCmdlet.ParameterSetName -eq 'Resume'
function File-Hash {
    param([string]$Path)
    $algorithm = [Security.Cryptography.SHA256]::Create(); $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Quote-Argument {
    param([string]$Argument)
    return '"' + [regex]::Replace([regex]::Replace($Argument, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
Push-Location -LiteralPath $repoRoot
try {
    $pythonExecutable = (Resolve-Path -LiteralPath $PythonPath).Path
    $checkpointInput = if ($resume) { $ResumeCheckpoint } else { $Checkpoint }
    $checkpointFlag = if ($resume) { '-ResumeCheckpoint' } else { '-Checkpoint' }
    $checkpointPath = (Resolve-Path -LiteralPath $checkpointInput).Path
    $checkpointHash = File-Hash $checkpointPath
    if (-not $resume -and $checkpointHash -ne $sourceHash) { throw 'Source checkpoint differs from the verified final model_1999.' }
    $userName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $runName = 'mixed_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
    $taskName = 'RLForHexapod_' + $runName
    $submission = Join-Path $repoRoot "outputs\mixed_terrain_submit\$runName"
    $launch = Join-Path $repoRoot "outputs\mixed_terrain_launch\$runName"
    $null = New-Item -ItemType Directory -Path $submission
    $receiptPath = Join-Path $submission 'submission.json'
    $powershell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-WindowStyle','Hidden',
        '-File',(Join-Path $PSScriptRoot 'run_mixed_terrain.ps1'),'-PythonPath',$pythonExecutable,
        $checkpointFlag,$checkpointPath,'-NumEnvs',$NumEnvs.ToString(),'-MaxIterations',$MaxIterations.ToString(),
        '-FlatFraction',$FlatFraction.ToString('R',$culture),'-StairHeight',$StairHeight.ToString('R',$culture),
        '-StairRewardScale',$StairRewardScale.ToString('R',$culture),'-Seed',$Seed.ToString(),'-Device',$Device,'-RunName',$runName)
    if ($resume) { $arguments += @('-CheckpointSha256', $checkpointHash) }
    if ($HeightCurriculum) { $arguments += '-HeightCurriculum' }
    if ($RandomStairHeight) { $arguments += '-RandomStairHeight' }
    $nativeArguments = ($arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $sourceFiles = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'src') -File -Recurse -Filter '*.py')
    $sourceFiles += @('train_mixed_terrain.py','mixed_actor_adapter.py','train_flat_wsp.py','run_mixed_terrain.ps1','submit_mixed_terrain.ps1','validate_robot_asset.py') | ForEach-Object { Get-Item -LiteralPath (Join-Path $PSScriptRoot $_) }
    $codeHashes = @($sourceFiles | Sort-Object FullName -Unique | ForEach-Object { [ordered]@{path=$_.FullName;sha256=(File-Hash $_.FullName)} })
    $receipt = [ordered]@{
        status='prepared';created_at=[DateTime]::UtcNow.ToString('o');task_name=$taskName;run_name=$runName
        user=$userName;logon_type='InteractiveToken';runtime_root=$repoRoot;python_executable=$pythonExecutable
        python_sha256=(File-Hash $pythonExecutable);checkpoint=$checkpointPath;source_sha256=$checkpointHash;resume=$resume
        training_task='Hexapod-Mixed-Flat-Stair';experiment='hexapod_mixed_flat_stair'
        num_envs=$NumEnvs;max_iterations=$MaxIterations;flat_fraction=$FlatFraction;stair_height=$StairHeight
        stair_reward_scale=$StairRewardScale;automatic_height_curriculum=[bool]$HeightCurriculum
        random_stair_height=[bool]$RandomStairHeight;seed=$Seed;device=$Device
        executable=$powershell;arguments=$arguments;native_argument_string=$nativeArguments;code_provenance=$codeHashes
        launcher_directory=$launch;launcher_status=(Join-Path $launch 'launcher_status.json')
        training_stdout=(Join-Path $launch '02_train.stdout.log');training_stderr=(Join-Path $launch '02_train.stderr.log')
        receipt=$receiptPath;registered=$false;launch_accepted=$false;monitoring_enabled=$false
        lifecycle='One hidden on-demand InteractiveToken action, no triggers/restarts or polling. Task Scheduler owns the process independently of SSH. The user must remain logged on.'
    }
    function Save-Receipt { [IO.File]::WriteAllText($receiptPath,($receipt | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false)) }
    Save-Receipt
    if (-not $PrepareOnly) {
        try {
            $scheduler=New-Object -ComObject 'Schedule.Service';$scheduler.Connect()
            $definition=$scheduler.NewTask(0)
            $definition.RegistrationInfo.Description='One shared mixed flat/stair PPO run with persistent logs.'
            $definition.Principal.UserId=$userName;$definition.Principal.LogonType=3;$definition.Principal.RunLevel=0
            $definition.Settings.Enabled=$true;$definition.Settings.Hidden=$true;$definition.Settings.AllowDemandStart=$true
            $definition.Settings.StartWhenAvailable=$false;$definition.Settings.DisallowStartIfOnBatteries=$false
            $definition.Settings.StopIfGoingOnBatteries=$false;$definition.Settings.ExecutionTimeLimit='PT0S'
            $definition.Settings.MultipleInstances=2
            $action=$definition.Actions.Create(0);$action.Path=$powershell;$action.Arguments=$nativeArguments;$action.WorkingDirectory=$repoRoot
            [IO.File]::WriteAllText((Join-Path $submission 'scheduled_task.xml'),$definition.XmlText,[Text.UTF8Encoding]::new($false))
            $registered=$scheduler.GetFolder('\').RegisterTaskDefinition($taskName,$definition,2,$userName,$null,3,$null)
            $receipt.registered=$true;Save-Receipt
            $null=$registered.Run($null)
            $receipt.launch_accepted=$true;$receipt.status='submitted_without_monitoring'
            $receipt['submitted_at']=[DateTime]::UtcNow.ToString('o');Save-Receipt
        } catch { $receipt.status='submission_failed';$receipt['error']=$_.Exception.Message;Save-Receipt;throw }
    }
    $receipt | ConvertTo-Json -Depth 12
} finally { Pop-Location }
