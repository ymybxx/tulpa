# Compile a pinned pg0 with UTF-8 PostgreSQL child processes, in a private cache.
#requires -Version 7.2
param([switch]$PrepareOnly, [string]$Python = "")
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows -or [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') {
    throw 'The pg0 build requires Windows x64.'
}
$petRoot = Split-Path $PSScriptRoot -Parent
$projectRoot = (Resolve-Path (Join-Path $petRoot '../..')).Path
$packaging = Join-Path $petRoot 'packaging/pg0-windows'
$cache = Join-Path $projectRoot '.cache/pg0-windows-utf8-v1'
$output = Join-Path $petRoot 'runtime/win32-x64/pg0'
& node (Join-Path $PSScriptRoot 'assert-packaging-idle.mjs')
if ($LASTEXITCODE -ne 0) { throw 'Quit tulpa before building runtime resources.' }
if (-not $Python) {
    $Python = (& uv python find --managed-python 3.11.15 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Python) { throw 'Install managed Python 3.11.15 with uv first.' }
}
$mtCommand = Get-Command mt.exe -ErrorAction SilentlyContinue
if ($mtCommand) { $mt = $mtCommand.Source }
else {
    $kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10/bin'
    $candidates = @(Get-ChildItem $kits -Filter mt.exe -Recurse | Where-Object { $_.Directory.Name -eq 'x64' } | Sort-Object FullName -Descending)
    if (-not $candidates.Count) { throw 'Windows SDK x64 mt.exe is required.' }
    $mt = $candidates[0].FullName
}
& $Python (Join-Path $packaging 'prepare.py') --cache-root $cache --mt $mt
if ($LASTEXITCODE -ne 0) { throw 'Preparing the verified UTF-8 PostgreSQL archive failed.' }
$prepared = Get-Content (Join-Path $cache 'prepared.json') -Raw | ConvertFrom-Json
if ($PrepareOnly) { return }
& (Join-Path $PSScriptRoot 'verify-windows-postgres-unicode.ps1') -PostgresRoot $prepared.postgresqlDirectory
if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL Unicode path probe failed.' }
$previousPostgresArchive = $env:TULPA_POSTGRES_ARCHIVE
$previousVectorArchive = $env:TULPA_PGVECTOR_ARCHIVE
$previousCargoTarget = $env:CARGO_TARGET_DIR
try {
    $env:TULPA_POSTGRES_ARCHIVE = $prepared.postgresqlArchive
    $env:TULPA_PGVECTOR_ARCHIVE = $prepared.pgvectorArchive
    $env:CARGO_TARGET_DIR = Join-Path $cache 'cargo-target'
    & cargo build --manifest-path (Join-Path $prepared.sourceDirectory 'Cargo.toml') --release --locked --target x86_64-pc-windows-msvc
    if ($LASTEXITCODE -ne 0) { throw 'Compiling the pinned Windows pg0 failed.' }
    & node (Join-Path $PSScriptRoot 'assert-packaging-idle.mjs')
    if ($LASTEXITCODE -ne 0) { throw 'Quit tulpa before installing runtime resources.' }
    New-Item -ItemType Directory -Force $output | Out-Null
    Copy-Item (Join-Path $env:CARGO_TARGET_DIR 'x86_64-pc-windows-msvc/release/pg0.exe') (Join-Path $output 'pg0.exe') -Force
    Copy-Item (Join-Path $packaging 'sources.json') (Join-Path $output 'sources.json') -Force
    Copy-Item (Join-Path $packaging 'NOTICE.txt') (Join-Path $output 'NOTICE.txt') -Force
    Copy-Item (Join-Path $prepared.postgresqlDirectory 'LICENSE') (Join-Path $output 'PostgreSQL-LICENSE.txt') -Force
    $licenses = Join-Path $petRoot 'runtime/win32-x64/licenses/pg0-utf8'
    New-Item -ItemType Directory -Force $licenses | Out-Null
    Copy-Item (Join-Path $packaging 'licenses/*.txt') $licenses -Force
    Copy-Item (Join-Path $packaging 'NOTICE.txt') (Join-Path $licenses 'NOTICE.txt') -Force
    Copy-Item (Join-Path $packaging 'sources.json') (Join-Path $licenses 'sources.json') -Force
    Copy-Item (Join-Path $prepared.postgresqlDirectory 'LICENSE') (Join-Path $licenses 'PostgreSQL-LICENSE.txt') -Force
    Write-Host "Pinned UTF-8 Windows pg0 ready: $output"
} finally {
    $env:TULPA_POSTGRES_ARCHIVE = $previousPostgresArchive
    $env:TULPA_PGVECTOR_ARCHIVE = $previousVectorArchive
    $env:CARGO_TARGET_DIR = $previousCargoTarget
}
