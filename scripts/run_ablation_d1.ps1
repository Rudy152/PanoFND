# D1 ablation: one joint call per three-modality tuning item. Credentials stay in this process only.
#   powershell -ExecutionPolicy Bypass -File scripts\run_ablation_d1.ps1 -Output runs\v2_ablation_d1_20261124.jsonl [-Limit 2]
param(
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$Features = "runs\v2_tuning943_features_20261121.jsonl",
    [int]$Workers = 4,
    [int]$Limit = 0,
    [string]$Ids = "",
    [double]$MaxCost = 5.0
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")
$arguments = @("-m", "panoagent_v2.ablation_d1", "extract", "--dataset", (Join-Path $root "fakenewsDataSet"),
    "--features", $Features, "--output", $Output, "--workers", "$Workers", "--max-cost", "$MaxCost")
if ($Limit -gt 0) { $arguments += @("--limit", "$Limit") }
if ($Ids) { $arguments += @("--ids") + @($Ids -split "[,\s]+" | Where-Object { $_ }) }
Push-Location $root
try {
    $env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
    $env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath (Join-Path $root ".secrets\dashscope_host.txt") -Raw).Trim()
    $env:PYTHONIOENCODING = "utf-8"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "D1 extract failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
}
