# One-time holdout run of the base-model baseline with a frozen threshold (panoagent_v2.baseline_s0). Credentials stay in this process only.
#   powershell -ExecutionPolicy Bypass -File scripts\run_s0_holdout.ps1 [-Limit N]
param(
    [int]$Workers = 4,
    [double]$MaxCost = 4.0,
    [int]$Limit = 0
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")
$arguments = @("-m", "panoagent_v2.baseline_s0", "run", "--workers", "$Workers", "--max-cost", "$MaxCost", "--limit", "$Limit")
Push-Location $root
try {
    $env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
    $env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath (Join-Path $root ".secrets\dashscope_host.txt") -Raw).Trim()
    $env:PYTHONIOENCODING = "utf-8"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "S0 holdout run failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
}
