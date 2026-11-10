$ErrorActionPreference = "Stop"

$credentialRoot = Join-Path $PSScriptRoot "..\.secrets"
$hostFile = Join-Path $credentialRoot "dashscope_host.txt"
$workspaceHost = "https://ws-qr0i98gfwi11a5gs.cn-beijing.maas.aliyuncs.com"
. (Join-Path $PSScriptRoot "dashscope_credentials.ps1")

Write-Host "This saves the rotated API key encrypted for the current Windows user."
Write-Host "It is not stored as plain text and cannot be decrypted by another Windows account."
$secureKey = Read-Host "Paste the ROTATED DashScope API key (input is hidden)" -AsSecureString

$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    Set-PanoDashScopeCredential -Secret $plainKey
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $plainKey = $null
}
New-Item -ItemType Directory -Path $credentialRoot -Force | Out-Null
$workspaceHost | Set-Content -LiteralPath $hostFile -Encoding UTF8

Write-Host "Saved encrypted credentials for: $workspaceHost"
Write-Host "You can now run run_qwen_smoke.cmd without entering the key again."
