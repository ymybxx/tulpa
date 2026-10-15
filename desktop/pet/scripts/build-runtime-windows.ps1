# Build on a Windows x64 host; never cross-build or replace the macOS runtime.
#requires -Version 7.2
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows -or [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') {
    throw 'The Windows runtime currently requires a native Windows x64 build host.'
}
function Invoke-Checked([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Build command failed with exit code $LASTEXITCODE" }
}
$petRoot = Split-Path $PSScriptRoot -Parent
$projectRoot = (Resolve-Path (Join-Path $petRoot '../..')).Path
$runtimeRoot = Join-Path $petRoot 'runtime/win32-x64'
$workRoot = Join-Path $projectRoot '.cache/pyinstaller-win32-x64'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache/uv'
$env:PYTHONUTF8 = '1'
$env:PYINSTALLER_CONFIG_DIR = Join-Path $workRoot 'config'
Push-Location $projectRoot
try {
    Invoke-Checked { node (Join-Path $PSScriptRoot 'assert-packaging-idle.mjs') }
    Invoke-Checked { uv python install 3.11.15 }
    $buildPython = (& uv python find --managed-python 3.11.15 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to locate managed Python 3.11.15.' }
    Invoke-Checked { & $buildPython -c 'import platform,sys; assert sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64")' }
    Invoke-Checked { & (Join-Path $PSScriptRoot 'build-pg0-windows.ps1') -Python $buildPython }
    $runtimeArgs = @('run', '--locked', '--isolated', '--python', $buildPython, '--no-dev', '--group', 'packaging', '--extra', 'voice')
    # Share the same pinned model bytes as macOS, including an offline VAD probe.
    Invoke-Checked { uv @runtimeArgs python (Join-Path $PSScriptRoot 'prepare-bundled-models.py') (Join-Path $runtimeRoot 'models') }
    Invoke-Checked { node (Join-Path $PSScriptRoot 'assert-packaging-idle.mjs') }
    Invoke-Checked {
        uv @runtimeArgs pyinstaller --noconfirm --clean --distpath $runtimeRoot `
            --workpath (Join-Path $workRoot 'build') (Join-Path $petRoot 'packaging/fish-memory-runtime.spec')
    }
    Invoke-Checked {
        & $buildPython (Join-Path $PSScriptRoot 'verify-windows-runtime.py') `
            (Join-Path $runtimeRoot 'fish-memory-runtime') --models (Join-Path $runtimeRoot 'models')
    }
    Write-Host "Windows x64 runtime ready: $runtimeRoot"
} finally {
    Pop-Location
}
