param(
    [string]$SplitPath = "",
    [string]$RunTag = "",
    [switch]$SkipQwen35,
    [switch]$SkipOmni,
    [string]$OmniModelId = "qwen2.5-omni-7b",
    [string]$OmniOutputStem = "qwen25_omni_7b"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$dataset = Join-Path $PSScriptRoot "..\fakenewsDataSet"
$split = if ($SplitPath) { $SplitPath } else { Join-Path $PSScriptRoot "..\splits\pilot_180.json" }
$frameCache = Join-Path $PSScriptRoot "..\cache\video_frames"
$runRoot = Join-Path $PSScriptRoot "..\runs"
$credentialRoot = Join-Path $PSScriptRoot "..\.secrets"
$hostFile = Join-Path $credentialRoot "dashscope_host.txt"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing project virtual environment: $python"
}
if (-not (Test-Path -LiteralPath $split)) {
    throw "Missing pilot split: $split"
}
if (-not (Test-Path -LiteralPath $hostFile)) {
    throw "Saved credentials are missing. Run scripts\save_dashscope_credentials.cmd once first."
}

$env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
$env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath $hostFile -Raw).Trim()

$common = @(
    "-m", "panoagent.cli", "direct",
    "--dataset", $dataset,
    "--split", $split,
    "--partition", "pilot",
    "--backend", "dashscope",
    "--media-mode", "frames",
    "--frame-count", "16",
    "--frame-cache", $frameCache,
    "--temperature", "0",
    "--max-new-tokens", "16",
    "--resume"
)

$tagSuffix = if ($RunTag) { "_$RunTag" } else { "" }
$qwen35Predictions = Join-Path $runRoot "qwen35_27b_pilot_180$tagSuffix.jsonl"
$qwen35Metrics = Join-Path $runRoot "qwen35_27b_pilot_180${tagSuffix}_metrics.json"
$omniPredictions = Join-Path $runRoot "${OmniOutputStem}_pilot_180$tagSuffix.jsonl"
$omniMetrics = Join-Path $runRoot "${OmniOutputStem}_pilot_180${tagSuffix}_metrics.json"

try {
    if (-not $SkipQwen35) {
        Write-Host "Running qwen3.5-27b on the 180-sample balanced pilot..."
        & $python @common --model-id "qwen3.5-27b" --output $qwen35Predictions
        if ($LASTEXITCODE -ne 0) { throw "qwen3.5-27b pilot failed with exit code $LASTEXITCODE" }
        & $python -m panoagent.cli evaluate --predictions $qwen35Predictions --output $qwen35Metrics
        if ($LASTEXITCODE -ne 0) { throw "qwen3.5-27b evaluation failed with exit code $LASTEXITCODE" }
    }

    if (-not $SkipOmni) {
        Write-Host "Running $OmniModelId on the same 180-sample balanced pilot..."
        & $python @common --model-id $OmniModelId --output $omniPredictions
        if ($LASTEXITCODE -ne 0) { throw "$OmniModelId pilot failed with exit code $LASTEXITCODE" }
        & $python -m panoagent.cli evaluate --predictions $omniPredictions --output $omniMetrics
        if ($LASTEXITCODE -ne 0) { throw "$OmniModelId evaluation failed with exit code $LASTEXITCODE" }
    }
}
finally {
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
    Write-Host "Credentials removed from this terminal process."
}
