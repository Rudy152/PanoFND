# Table E: one-factor input ablations of the v2 single call. Credentials stay in this process only.
#   powershell -ExecutionPolicy Bypass -File scripts\run_ablation_e.ps1 [-Smoke]
param(
    [string]$Features = "runs\v2_tuning943_features_20261121.jsonl",
    [string]$OutputDir = "runs\v2_ablation_e_20261124",
    [int]$Workers = 4,
    [double]$MaxCost = 25.0,
    [string]$Only = "",
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")
$arguments = @("-m", "panoagent_v2.ablation_e", "run", "--dataset", (Join-Path $root "fakenewsDataSet"),
    "--features", $Features, "--output-dir", $OutputDir, "--workers", "$Workers", "--max-cost", "$MaxCost")
if ($Smoke) { $arguments += "--smoke" }
if ($Only) { $arguments += @("--only", $Only) }
Push-Location $root
try {
    $env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
    $env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath (Join-Path $root ".secrets\dashscope_host.txt") -Raw).Trim()
    $env:PYTHONIOENCODING = "utf-8"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "table E failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
}
