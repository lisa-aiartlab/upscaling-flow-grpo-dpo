param(
    [string]$Manifest = "flow_grpo_dataset\upscaling_dataset\manifest.json",
    [int]$Epochs = 1,
    [int]$MaxSamples = 0,
    [int]$GroupSize = 2,
    [int]$InferenceSteps = 4,
    [int]$Resolution = 120,
    [string]$OutputDirectory = "flow_grpo_output",
    [string]$LrPipeline = "",
    [ValidateSet("fp16", "bf16")]
    [string]$MixedPrecision = "bf16",
    [string]$RewardDevice = "cpu",
    [switch]$DisableClip
)

$ErrorActionPreference = "Stop"
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$cacheDirectory = Join-Path $projectDirectory ".cache"
$python = Join-Path $projectDirectory ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual-environment Python was not found: $python"
}
if ($GroupSize -lt 2) {
    throw "Flow-GRPO requires GroupSize >= 2"
}
if ($Resolution % 4 -ne 0) {
    throw "Resolution must be divisible by 4"
}

$env:HF_HOME = Join-Path $cacheDirectory "huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TORCH_HOME = Join-Path $cacheDirectory "torch"
$env:CUDA_CACHE_PATH = Join-Path $cacheDirectory "cuda"
$env:PIP_CACHE_DIR = Join-Path $cacheDirectory "pip"
$env:XDG_CACHE_HOME = $cacheDirectory
$env:TEMP = Join-Path $cacheDirectory "tmp"
$env:TMP = $env:TEMP
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"

@(
    $cacheDirectory,
    $env:HF_HOME,
    $env:HF_HUB_CACHE,
    $env:TORCH_HOME,
    $env:CUDA_CACHE_PATH,
    $env:PIP_CACHE_DIR,
    $env:TEMP
) | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

$trainingArguments = @(
    "scripts\training_scripts\flow_grpo.py",
    "--manifest", $Manifest,
    "--output-dir", $OutputDirectory,
    "--epochs", $Epochs,
    "--grpo-epochs", 1,
    "--group-size", $GroupSize,
    "--inference-steps", $InferenceSteps,
    "--resolution", $Resolution,
    "--save-every", 5,
    "--reward-device", $RewardDevice,
    "--mixed-precision", $MixedPrecision
)
if ($MaxSamples -gt 0) {
    $trainingArguments += @("--max-samples", $MaxSamples)
}
if ($LrPipeline) {
    $trainingArguments += @("--lr-pipeline", $LrPipeline)
}
if ($DisableClip) {
    $trainingArguments += @("--clip-model-id", "none")
}

Push-Location $projectDirectory
try {
    & $python @trainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
