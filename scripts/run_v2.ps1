# Run PanoAgent v2 evidence extraction with the saved DashScope credentials.
#   powershell -ExecutionPolicy Bypass -File scripts\run_v2.ps1 -Split splits\v2_tuning180_seed42_20261121.json -Partition pilot -Output runs\v2_tuning180_features.jsonl
# The final holdout (splits\final_holdout_clean_765_20261031.json) needs -Frozen <frozen scorer JSON>;
# extraction refuses to run if any v2 code changed after the freeze.
param(
    [Parameter(Mandatory = $true)][string]$Split,
    [string]$Partition = "pilot",
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$ModelId = "qwen3.8-omni-flash",
    [int]$Workers = 4,
    [int]$Limit = 0,
    [string]$Frozen = ""
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")
$arguments = @("-m", "panoagent_v2.cli", "extract", "--dataset", (Join-Path $root "fakenewsDataSet"),
    "--split", $Split, "--partition", $Partition, "--output", $Output, "--model-id", $ModelId,
    "--workers", "$Workers")
if ($Limit -gt 0) { $arguments += @("--limit", "$Limit") }
if ($Frozen) { $arguments += @("--frozen", $Frozen) }
Push-Location $root
try {
    $env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
    $env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath (Join-Path $root ".secrets\dashscope_host.txt") -Raw).Trim()
    $env:PYTHONIOENCODING = "utf-8"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "v2 extract failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
}
