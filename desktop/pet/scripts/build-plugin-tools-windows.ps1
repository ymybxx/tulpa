# Assemble a standard relocatable CPython + uv for plugin venvs, not frozen Python.
#requires -Version 7.2
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows -or [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') {
    throw 'Windows plugin tools currently require a native Windows x64 build host.'
}
function Invoke-Checked([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Build command failed with exit code $LASTEXITCODE" }
}
$petRoot = Split-Path $PSScriptRoot -Parent
$projectRoot = (Resolve-Path (Join-Path $petRoot '../..')).Path
$toolsRoot = Join-Path $petRoot 'runtime/win32-x64/fish-plugin-tools'
$livePlugin = if ($env:FISH_LIVE_PLUGIN_DIR) { $env:FISH_LIVE_PLUGIN_DIR } else { Join-Path $projectRoot 'plugins/live' }
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache/uv'
$env:PYTHONUTF8 = '1'
Push-Location $projectRoot
try {
    Invoke-Checked { node (Join-Path $PSScriptRoot 'assert-packaging-idle.mjs') }
    Invoke-Checked { uv python install 3.11.15 }
    $python = (& uv python find --managed-python 3.11.15 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to locate managed Python 3.11.15.' }
    $base = (& $python -c 'import sys; print(sys.base_prefix)' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to locate the standalone Python distribution.' }
    New-Item -ItemType Directory -Force (Join-Path $toolsRoot 'plugins') | Out-Null
    $pythonRoot = Join-Path $toolsRoot 'python'
    if (Test-Path $pythonRoot) { Remove-Item -Recurse -Force $pythonRoot }
    Copy-Item -Recurse $base $pythonRoot
    $toolUv = (& uv run --no-project --isolated --python $python --with 'uv==0.11.21' `
        python -c 'import uv; print(uv.find_uv_bin())' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve the pinned Windows uv tool.' }
    Copy-Item $toolUv (Join-Path $toolsRoot 'uv.exe') -Force
    Get-ChildItem $toolsRoot -Filter 'fish_plugin_sdk-*.whl' | Remove-Item -Force
    Invoke-Checked { uv build --package fish-plugin-sdk --wheel --out-dir $toolsRoot }
    Get-ChildItem (Join-Path $toolsRoot 'plugins') -Filter 'fish.live-*.fish-plugin' | Remove-Item -Force
    if (Test-Path (Join-Path $livePlugin 'fish-plugin.json')) {
        $sdkWheel = (Get-ChildItem $toolsRoot -Filter 'fish_plugin_sdk-*.whl' | Select-Object -First 1).FullName
        Invoke-Checked {
            uv run --no-project --isolated --python $python --with "$projectRoot/sdk/python[tools]" `
                --with 'httpx>=0.28,<1' --with 'Brotli>=1.1,<2' `
                python -m fish_plugin_sdk.cli build $livePlugin --sdk-wheel $sdkWheel
        }
        $version = (Get-Content (Join-Path $livePlugin 'fish-plugin.json') -Raw | ConvertFrom-Json).version
        Copy-Item (Join-Path $livePlugin "dist/fish.live-$version-win32-x86_64-py311.fish-plugin") (Join-Path $toolsRoot 'plugins')
    }
    # Prove the copied distribution can create a venv after relocation. No DB,
    # API, microphone, user profile, or real plugin instance is started.
    Invoke-Checked {
        & (Join-Path $pythonRoot 'python.exe') -c 'import ssl,venv,sys,tempfile,pathlib,subprocess; assert sys.version_info[:3] == (3,11,15); t=tempfile.TemporaryDirectory(); p=pathlib.Path(t.name)/"venv"; venv.EnvBuilder().create(p); subprocess.run([str(p/"Scripts/python.exe"),"-c","import ssl,sys; print(sys.version)"],check=True); t.cleanup()'
    }
    Write-Host "Windows plugin tools ready: $toolsRoot"
} finally {
    Pop-Location
}
