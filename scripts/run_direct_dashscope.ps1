# Single-call (direct) baseline on DashScope with the same media protocol as the agent runs.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_direct_dashscope.ps1 -Split splits\tuning90_balanced_seed42_20261107.json -Partition pilot -Tag tuning90
param(
    [string]$Split = "",
    [string]$Partition = "pilot",
    [string]$Ids = "",
    [string]$ModelId = "qwen3-vl-plus",
    [ValidateSet("paper", "debiased")][string]$DirectPrompt = "paper",
    [int]$FrameCount = 8,
    [int]$Workers = 2,
    [string]$Tag = "smoke"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
$hostFile = Join-Path $root ".secrets\dashscope_host.txt"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")

if (-not (Test-Path -LiteralPath $python)) { throw "Missing project virtual environment: $python" }
if (-not (Test-Path -LiteralPath $hostFile)) {
    throw "Saved DashScope credentials are missing. Run scripts\save_dashscope_credentials.cmd first."
}

$stamp = Get-Date -Format "yyyyMMdd"
$modelStem = ($ModelId -replace "[^A-Za-z0-9]+", "_")
$stem = "direct_${modelStem}_${DirectPrompt}_f${FrameCount}_${Tag}_${stamp}"
$output = Join-Path $root "runs\$stem.jsonl"
$metrics = Join-Path $root "runs\${stem}_metrics.json"

$arguments = @(
    "-m", "panoagent.cli", "direct",
    "--dataset", (Join-Path $root "fakenewsDataSet"),
    "--backend", "dashscope",
    "--model-id", $ModelId,
    "--media-mode", "frames",
    "--frame-count", "$FrameCount",
    "--frame-cache", (Join-Path $root "cache\video_frames"),
    "--direct-prompt", $DirectPrompt,
    "--temperature", "0",
    "--max-new-tokens", "16",
    "--workers", "$Workers",
    "--resume",
    "--output", $output
)
if ($Split) { $arguments += @("--split", $Split, "--partition", $Partition) }
elseif ($Ids) { $arguments += @("--ids") + @($Ids -split "[,\s]+" | Where-Object { $_ }) }
else { throw "Pass -Split or -Ids." }

try {
    $env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
    $env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath $hostFile -Raw).Trim()
    $env:PYTHONIOENCODING = "utf-8"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "direct run failed with exit code $LASTEXITCODE" }
    & $python -m panoagent.cli evaluate --predictions $output --output $metrics | Out-Null
    Write-Host "Predictions: $output"
    Write-Host "Metrics:     $metrics"
}
finally {
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
    Write-Host "DashScope credentials removed from this process."
}
