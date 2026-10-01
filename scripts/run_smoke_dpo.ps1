param(
    [string]$Manifest = "flow_grpo_dataset\manifest.json",
    [int]$Resolution = 128,
    [string]$OutputDirectory = "dpo_smoke_output",
    [ValidateSet("fp16", "bf16")]
    [string]$MixedPrecision = "fp16"
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_dpo.ps1") `
    -Manifest $Manifest `
    -MaxSamples 1 `
    -Epochs 1 `
    -Resolution $Resolution `
    -SaveEvery 1 `
    -OutputDirectory $OutputDirectory `
    -MixedPrecision $MixedPrecision
