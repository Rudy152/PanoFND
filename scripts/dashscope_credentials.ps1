if (-not ("PanoAgent.CredentialNative" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace PanoAgent {
    public static class CredentialNative {
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        public struct CREDENTIAL {
            public UInt32 Flags;
            public UInt32 Type;
            public string TargetName;
            public string Comment;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
            public UInt32 CredentialBlobSize;
            public IntPtr CredentialBlob;
            public UInt32 Persist;
            public UInt32 AttributeCount;
            public IntPtr Attributes;
            public string TargetAlias;
            public string UserName;
        }

        [DllImport("advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool CredWrite(ref CREDENTIAL credential, UInt32 flags);

        [DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool CredRead(string target, UInt32 type, UInt32 reservedFlag, out IntPtr credentialPtr);

        [DllImport("advapi32.dll", SetLastError = true)]
        public static extern void CredFree(IntPtr credentialPtr);
    }
}
"@
}

$script:DashScopeCredentialTarget = "PanoAgent/DashScope/API-Key"

function Set-PanoDashScopeCredential {
    param([Parameter(Mandatory = $true)][string]$Secret)

    $secretBytes = [Text.Encoding]::Unicode.GetBytes($Secret)
    $blobPointer = [Runtime.InteropServices.Marshal]::AllocCoTaskMem($secretBytes.Length)
    try {
        [Runtime.InteropServices.Marshal]::Copy($secretBytes, 0, $blobPointer, $secretBytes.Length)
        $credential = New-Object PanoAgent.CredentialNative+CREDENTIAL
        $credential.Type = 1
        $credential.TargetName = $script:DashScopeCredentialTarget
        $credential.CredentialBlobSize = $secretBytes.Length
        $credential.CredentialBlob = $blobPointer
        $credential.Persist = 2
        $credential.UserName = "DashScope API Key"
        if (-not [PanoAgent.CredentialNative]::CredWrite([ref]$credential, 0)) {
            $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            throw "Windows Credential Manager failed to save the API key (Win32 error $errorCode)."
        }
    }
    finally {
        [Array]::Clear($secretBytes, 0, $secretBytes.Length)
        [Runtime.InteropServices.Marshal]::FreeCoTaskMem($blobPointer)
    }
}

function Get-PanoDashScopeCredential {
    $credentialPointer = [IntPtr]::Zero
    if (-not [PanoAgent.CredentialNative]::CredRead(
        $script:DashScopeCredentialTarget,
        1,
        0,
        [ref]$credentialPointer
    )) {
        $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "Saved DashScope API key was not found (Win32 error $errorCode). Run scripts\save_dashscope_credentials.cmd first."
    }
    try {
        $credential = [Runtime.InteropServices.Marshal]::PtrToStructure(
            $credentialPointer,
            [type][PanoAgent.CredentialNative+CREDENTIAL]
        )
        $secretBytes = New-Object byte[] $credential.CredentialBlobSize
        [Runtime.InteropServices.Marshal]::Copy(
            $credential.CredentialBlob,
            $secretBytes,
            0,
            $credential.CredentialBlobSize
        )
        try {
            return [Text.Encoding]::Unicode.GetString($secretBytes)
        }
        finally {
            [Array]::Clear($secretBytes, 0, $secretBytes.Length)
        }
    }
    finally {
        [PanoAgent.CredentialNative]::CredFree($credentialPointer)
    }
}
