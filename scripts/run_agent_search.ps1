# Run PanoAgent with the web-search module (Tavily) enabled.
#
# Smoke test (default, 3 tuning-partition items):
#   powershell -ExecutionPolicy Bypass -File scripts\run_agent_search.ps1
# Full tuning partition (943 items) with the fused profile:
#   powershell -ExecutionPolicy Bypass -File scripts\run_agent_search.ps1 -Split splits\half_eval_seed42.json -Partition train -Workers 2 -Tag tuning943
#
# Never point this at splits\final_holdout_clean_765_20261031.json until the model and
# routing are frozen; that holdout may be run exactly once.
param(
    [string[]]$Ids = @("1351", "1276", "0979"),
    [string]$Split = "",
    [string]$Partition = "train",
    [ValidateSet("full", "fused", "fused-rule")][string]$AgentProfile = "fused",
    [string]$ModelId = "qwen3-vl-plus",
    [ValidateSet("model", "text", "always")][string]$RetrievalPolicy = "text",
    [int]$FrameCount = 4,
    [int]$Workers = 1,
    [string]$Tag = "search_smoke",
    [switch]$ExcludeSourceUrl,
    # "none" gives the no-retrieval arm of a paired comparison.
    [ValidateSet("tavily", "none")][string]$SearchProvider = "tavily",
    # Share one stage cache between paired arms so both reuse the identical fused report.
    [string]$CacheName = ""
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
if (-not $env:TAVILY_API_KEY) {
    $userKey = [Environment]::GetEnvironmentVariable("TAVILY_API_KEY", "User")
    if ($userKey) { $env:TAVILY_API_KEY = $userKey }
}
if ($SearchProvider -eq "tavily" -and -not $env:TAVILY_API_KEY) {
    throw "TAVILY_API_KEY is not set; web search cannot start."
}

$stamp = Get-Date -Format "yyyyMMdd"
$mode = if ($SearchProvider -eq "tavily") { "search" } else { "nosearch" }
$stem = "agent_${AgentProfile}_${mode}_${Tag}_${stamp}"
$output = Join-Path $root "runs\$stem.jsonl"
$metrics = Join-Path $root "runs\${stem}_metrics.json"
$cache = if ($CacheName) { Join-Path $root "cache\$CacheName" } else { Join-Path $root "cache\$stem" }

$arguments = @(
    "-m", "panoagent.cli", "agent",
    "--dataset", (Join-Path $root "fakenewsDataSet"),
    "--backend", "dashscope",
    "--model-id", $ModelId,
    "--media-mode", "frames",
    "--frame-count", "$FrameCount",
    "--frame-cache", (Join-Path $root "cache\video_frames"),
    "--agent-profile", $AgentProfile,
    "--search-provider", $SearchProvider,
    "--search-cache", (Join-Path $root "cache\search"),
    "--retrieval-policy", $RetrievalPolicy,
    "--strict-stage-errors",
    "--workers", "$Workers",
    "--resume",
    "--cache", $cache,
    "--output", $output
)
# -Split takes precedence over the default smoke IDs.
if ($Split) { $arguments += @("--split", $Split, "--partition", $Partition) }
elseif ($Ids.Count -gt 0) {
    # `powershell -File` passes "a,b,c" as one string; accept both forms.
    $idList = @($Ids | ForEach-Object { $_ -split "[,\s]+" } | Where-Object { $_ })
    $arguments += @("--ids") + $idList
}
else { throw "Pass -Ids or -Split." }
if ($ExcludeSourceUrl) { $arguments += "--exclude-source-url" }

try {
    $env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
    $env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath $hostFile -Raw).Trim()
    $env:PYTHONIOENCODING = "utf-8"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "agent run failed with exit code $LASTEXITCODE" }
    & $python -m panoagent.cli evaluate --predictions $output --output $metrics | Out-Null
    Write-Host "Predictions: $output"
    Write-Host "Metrics:     $metrics"
}
finally {
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
    Write-Host "DashScope credentials removed from this process."
}
