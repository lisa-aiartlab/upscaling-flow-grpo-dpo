param(
    [string]$Manifest = "flow_grpo_dataset\upscaling_dataset\manifest.json",
    [int]$Resolution = 120,
    [string]$OutputDirectory = "flow_grpo_smoke_output",
    [string]$LrPipeline = "LR_01_resize"
)

$ErrorActionPreference = "Stop"
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$cacheDirectory = Join-Path $projectDirectory ".cache"
$python = Join-Path $projectDirectory ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual-environment Python was not found: $python"
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

Push-Location $projectDirectory
try {
    & $python "scripts\training_scripts\flow_grpo.py" `
        --manifest $Manifest `
        --lr-pipeline $LrPipeline `
        --output-dir $OutputDirectory `
        --max-samples 1 `
        --epochs 1 `
        --grpo-epochs 1 `
        --group-size 2 `
        --inference-steps 4 `
        --resolution $Resolution `
        --save-every 1 `
        --reward-device cpu `
        --mixed-precision bf16 `
        --clip-model-id none

    if ($LASTEXITCODE -ne 0) {
        throw "Smoke training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
