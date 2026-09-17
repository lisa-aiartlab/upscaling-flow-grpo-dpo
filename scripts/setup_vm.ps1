param(
    [string]$Python = "python",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124"
)

$ErrorActionPreference = "Stop"
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"

Push-Location $projectDirectory
try {
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
        torch==2.4.1 torchvision==0.19.1 `
        --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install CUDA-enabled PyTorch."
    }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install project dependencies."
    }
    & $venvPython scripts\validate_setup.py

    if ($LASTEXITCODE -ne 0) {
        throw "Environment validation failed."
    }
    Write-Output "Environment is ready. Run: .\scripts\run_smoke_training.ps1"
}
finally {
    Pop-Location
}
