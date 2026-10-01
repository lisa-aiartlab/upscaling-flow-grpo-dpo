param(
    [string]$Python = "python",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu118",
    [ValidateSet("fp16", "bf16")]
    [string]$MixedPrecision = "fp16"
)

$ErrorActionPreference = "Stop"
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"

Push-Location $projectDirectory
try {
    & $Python -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Python 3.10, 3.11, or 3.12 is required"'
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10, 3.11, or 3.12 is required."
    }
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the virtual environment."
        }
    }

    & $venvPython -m pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip tooling."
    }
    & $venvPython -m pip install `
        torch==2.6.0 torchvision==0.21.0 `
        --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install CUDA-enabled PyTorch."
    }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install project dependencies."
    }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Installed dependencies are inconsistent."
    }
    & $venvPython scripts\validate_setup.py --mixed-precision $MixedPrecision

    if ($LASTEXITCODE -ne 0) {
        throw "Environment validation failed."
    }
    Write-Output "Environment is ready."
    Write-Output "Smoke tests:"
    Write-Output "  .\scripts\run_smoke_training.ps1"
    Write-Output "  .\scripts\run_smoke_dpo.ps1"
    Write-Output "Full training:"
    Write-Output "  .\scripts\run_training.ps1"
    Write-Output "  .\scripts\run_dpo.ps1"
}
finally {
    Pop-Location
}
