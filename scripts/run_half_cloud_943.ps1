$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$env:PYTHONUNBUFFERED = "1"
$env:DASHSCOPE_API_KEY = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "User")
$env:DASHSCOPE_BASE_URL = [Environment]::GetEnvironmentVariable("DASHSCOPE_BASE_URL", "User")

if (-not $env:DASHSCOPE_API_KEY) {
    throw "DASHSCOPE_API_KEY is not configured at User scope"
}

& ".\.venv\Scripts\python.exe" -m panoagent.cli agent `
    --dataset ".\fakenewsDataSet" `
    --split ".\splits\half_eval_seed42.json" `
    --partition train `
    --backend dashscope `
    --model-id qwen3-vl-plus `
    --media-mode frames `
    --frame-count 4 `
    --search-provider none `
    --max-new-tokens 1024 `
    --request-timeout 300 `
    --cache ".\cache\cloud_qwen3_vl_plus_half943_seed42_20261010" `
    --output ".\runs\agent_cloud_qwen3_vl_plus_half943_seed42_20261010.jsonl" `
    --resume

if ($LASTEXITCODE -ne 0) {
    throw "Half-dataset experiment exited with code $LASTEXITCODE"
}
