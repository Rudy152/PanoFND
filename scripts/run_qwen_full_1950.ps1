param(
    [string]$RunTag = "20261107",
    [switch]$SkipQwen35,
    [switch]$SkipQwen38
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$dataset = Join-Path $PSScriptRoot "..\fakenewsDataSet"
$frameCache = Join-Path $PSScriptRoot "..\cache\video_frames"
$runRoot = Join-Path $PSScriptRoot "..\runs"
$credentialRoot = Join-Path $PSScriptRoot "..\.secrets"
$hostFile = Join-Path $credentialRoot "dashscope_host.txt"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing project virtual environment: $python"
}
if (-not (Test-Path -LiteralPath (Join-Path $dataset "data.json"))) {
    throw "Missing dataset manifest under: $dataset"
}
if (-not (Test-Path -LiteralPath $hostFile)) {
    throw "Saved credentials are missing. Run scripts\save_dashscope_credentials.cmd first."
}

$env:DASHSCOPE_API_KEY = Get-PanoDashScopeCredential
$env:DASHSCOPE_BASE_URL = (Get-Content -LiteralPath $hostFile -Raw).Trim()

$common = @(
    "-m", "panoagent.cli", "direct",
    "--dataset", $dataset,
    "--backend", "dashscope",
    "--media-mode", "frames",
    "--frame-count", "16",
    "--frame-cache", $frameCache,
    "--temperature", "0",
    "--max-new-tokens", "16",
    "--workers", "1",
    "--direct-prompt", "paper",
    "--resume"
)

function Invoke-FullModelRun {
    param(
        [Parameter(Mandatory = $true)][string]$ModelId,
        [Parameter(Mandatory = $true)][string]$OutputStem
    )

    $predictions = Join-Path $runRoot "${OutputStem}_full_1950_${RunTag}.jsonl"
    $metrics = Join-Path $runRoot "${OutputStem}_full_1950_${RunTag}_metrics.json"

    for ($attempt = 1; $attempt -le 10; $attempt++) {
        Write-Host "[$ModelId] pass $attempt starting/resuming..."
        & $python @common --model-id $ModelId --output $predictions
        $modelExitCode = $LASTEXITCODE

        if ($modelExitCode -ne 0) {
            Write-Warning "[$ModelId] process exited with code $modelExitCode; restarting from saved records."
            Start-Sleep -Seconds 5
            continue
        }

        & $python -m panoagent.cli evaluate --predictions $predictions --output $metrics
        if ($LASTEXITCODE -ne 0) {
            throw "[$ModelId] evaluation failed with exit code $LASTEXITCODE"
        }
        $metricPayload = Get-Content -LiteralPath $metrics -Raw | ConvertFrom-Json
        $errors = [int]$metricPayload.overall.parse_or_runtime_errors
        $evaluated = [int]$metricPayload.overall.evaluated
        $latestById = @{}
        Get-Content -LiteralPath $predictions | ForEach-Object {
            if (-not [string]::IsNullOrWhiteSpace($_)) {
                $row = $_ | ConvertFrom-Json
                $latestById[[string]$row.id] = $row
            }
        }
        $blocked = @(
            $latestById.Values | Where-Object {
                [string]$_.error -like "*data_inspection_failed*"
            }
        ).Count
        $retryableErrors = @(
            $latestById.Values | Where-Object {
                $_.error -and [string]$_.error -notlike "*data_inspection_failed*"
            }
        ).Count
        Write-Host "[$ModelId] unique=$($latestById.Count) evaluated=$evaluated blocked=$blocked retryable_errors=$retryableErrors"

        if ($latestById.Count -eq 1950 -and $retryableErrors -eq 0) {
            if ($blocked -gt 0) {
                Write-Warning "[$ModelId] completed with $blocked provider content-safety refusals recorded as unevaluated."
            }
            return
        }
        Write-Warning "[$ModelId] incomplete or errored records remain; retrying only unfinished IDs."
    }

    throw "[$ModelId] did not reach 1950 successful records after 10 resumable passes."
}

try {
    if (-not $SkipQwen35) {
        Invoke-FullModelRun -ModelId "qwen3.5-27b" -OutputStem "qwen35_27b"
    }
    if (-not $SkipQwen38) {
        Invoke-FullModelRun -ModelId "qwen3.8-omni-flash" -OutputStem "qwen38_omni_flash"
    }
}
finally {
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:DASHSCOPE_BASE_URL -ErrorAction SilentlyContinue
    Write-Host "Credentials removed from this terminal process."
}
