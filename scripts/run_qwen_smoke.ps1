$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$dataset = Join-Path $PSScriptRoot "..\fakenewsDataSet"
$frameCache = Join-Path $PSScriptRoot "..\cache\video_frames"
$runRoot = Join-Path $PSScriptRoot "..\runs"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$credentialRoot = Join-Path $PSScriptRoot "..\.secrets"
$hostFile = Join-Path $credentialRoot "dashscope_host.txt"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing project virtual environment: $python"
}

if (-not (Test-Path -LiteralPath $hostFile)) {
    throw "Saved credentials are missing. Run scripts\save_dashscope_credentials.cmd once first."
}

$env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
$env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath $hostFile -Raw).Trim()

if (-not $env:DASHSCOPE_BASE_URL.StartsWith("https://")) {
    throw "The API Host/base_url must start with https://"
}

$common = @(
    "-m", "panoagent.cli", "direct",
    "--dataset", $dataset,
    "--backend", "dashscope",
    "--media-mode", "frames",
    "--frame-count", "16",
    "--frame-cache", $frameCache,
    "--ids", "1052",
    "--temperature", "0",
    "--max-new-tokens", "16"
)

try {
    Write-Host "Testing qwen3.5-27b with thinking disabled..."
    & $python @common `
        --model-id "qwen3.5-27b" `
        --output (Join-Path $runRoot "qwen35_27b_smoke_$runStamp.jsonl")

    Write-Host "Testing qwen2.5-omni-7b with the same controlled input..."
    & $python @common `
        --model-id "qwen2.5-omni-7b" `
        --output (Join-Path $runRoot "qwen25_omni_7b_controlled_smoke_$runStamp.jsonl")
}
finally {
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
    Write-Host "Credentials removed from this terminal process."
}
