param(
    [string]$Manifest = "flow_grpo_dataset\manifest.json",
    [int]$Epochs = 1,
    [int]$MaxSamples = 0,
    [int]$Resolution = 128,
    [double]$LearningRate = 1.0e-5,
    [double]$MaxGradNorm = 1.0,
    [double]$Beta = 500.0,
    [int]$LoraRank = 4,
    [int]$SaveEvery = 25,
    [string]$OutputDirectory = "dpo_output",
    [string]$ResumeFromCheckpoint = "",
    [string]$InitFromLora = "",
    [string]$LrPipeline = "",
    [ValidateSet("lanczos", "bicubic", "nearest")]
    [string]$SyntheticRejected = "lanczos",
    [ValidateSet("fp16", "bf16")]
    [string]$MixedPrecision = "fp16"
)

$ErrorActionPreference = "Stop"
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$cacheDirectory = Join-Path $projectDirectory ".cache"
$python = Join-Path $projectDirectory ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual-environment Python was not found: $python"
}
if ($Resolution % 16 -ne 0) {
    throw "Resolution must be divisible by 16 for FLUX.2 latent packing"
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
    "scripts\training_scripts\dpo.py",
    "--manifest", $Manifest,
    "--output-dir", $OutputDirectory,
    "--epochs", $Epochs,
    "--resolution", $Resolution,
    "--learning-rate", $LearningRate,
    "--max-grad-norm", $MaxGradNorm,
    "--beta", $Beta,
    "--lora-rank", $LoraRank,
    "--synthetic-rejected", $SyntheticRejected,
    "--save-every", $SaveEvery,
    "--mixed-precision", $MixedPrecision
)
if ($MaxSamples -gt 0) {
    $trainingArguments += @("--max-samples", $MaxSamples)
}
if ($LrPipeline) {
    $trainingArguments += @("--lr-pipeline", $LrPipeline)
}
if ($ResumeFromCheckpoint) {
    $trainingArguments += @("--resume-from-checkpoint", $ResumeFromCheckpoint)
}
if ($InitFromLora) {
    $trainingArguments += @("--init-from-lora", $InitFromLora)
}

Push-Location $projectDirectory
try {
    & $python @trainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "DPO training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
