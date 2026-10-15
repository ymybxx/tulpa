param([string]$NodeIncludeDir = "", [string]$NodeLibrary = "", [string]$Configuration = "Release")
$ErrorActionPreference = "Stop"
if (-not [Environment]::Is64BitProcess -or [Environment]::OSVersion.Platform -ne "Win32NT") { throw "Windows x64 is required" }
$PetRoot = Split-Path $PSScriptRoot -Parent
& node (Join-Path $PSScriptRoot "assert-packaging-idle.mjs")
if ($LASTEXITCODE -ne 0) { throw "Quit tulpa before building native runtime resources" }
$NativeSource = Join-Path $PetRoot "native/windows"
$NativeBuild = Join-Path $PetRoot "build/windows-native"
$NativeOutput = Join-Path $PetRoot "runtime/win32-x64"
if (-not $NodeIncludeDir -or -not $NodeLibrary) {
  $ElectronPackage = Get-Content (Join-Path $PetRoot "node_modules/electron/package.json") -Raw | ConvertFrom-Json
  $ElectronVersion = $ElectronPackage.version
  $HeaderCache = Join-Path $NativeBuild "electron-$ElectronVersion"
  New-Item -ItemType Directory -Force $HeaderCache | Out-Null
  $BaseUrl = "https://electronjs.org/headers/v$ElectronVersion"
  $ArchiveName = "node-v$ElectronVersion-headers.tar.gz"
  $Archive = Join-Path $HeaderCache $ArchiveName
  $Library = Join-Path $HeaderCache "node.lib"
  # The CDN serves this as application/octet-stream. PowerShell may expose
  # Content as byte[]; explicitly decode the saved UTF-8 file instead.
  $ChecksumFile = Join-Path $HeaderCache "SHASUMS256.txt"
  Invoke-WebRequest "$BaseUrl/SHASUMS256.txt" -OutFile $ChecksumFile -UseBasicParsing
  $Checksums = Get-Content -LiteralPath $ChecksumFile -Encoding utf8
  function Get-VerifiedHeaderFile([string]$RemoteName, [string]$Destination) {
    $ChecksumLine = ($Checksums | Where-Object { ($_.Trim() -split '\s+', 2)[-1] -eq $RemoteName } | Select-Object -First 1)
    if (-not $ChecksumLine) { throw "No official checksum for $RemoteName" }
    $Expected = ($ChecksumLine.Trim() -split '\s+', 2)[0]
    if ($Expected -notmatch '^[a-fA-F0-9]{64}$') { throw "Invalid official checksum for $RemoteName" }
    if (-not (Test-Path $Destination) -or (Get-FileHash $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected.ToLowerInvariant()) {
      Invoke-WebRequest "$BaseUrl/$RemoteName" -OutFile $Destination -UseBasicParsing
    }
    if ((Get-FileHash $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected.ToLowerInvariant()) { throw "Electron header checksum mismatch: $RemoteName" }
  }
  Get-VerifiedHeaderFile $ArchiveName $Archive
  Get-VerifiedHeaderFile "win-x64/node.lib" $Library
  & tar.exe -xzf $Archive -C $HeaderCache
  if ($LASTEXITCODE -ne 0) { throw "Electron headers extraction failed" }
  $Header = Get-ChildItem $HeaderCache -Filter node_api.h -Recurse | Select-Object -First 1
  if (-not $Header) { throw "node_api.h missing from Electron headers" }
  if (-not $NodeIncludeDir) { $NodeIncludeDir = $Header.DirectoryName }
  if (-not $NodeLibrary) { $NodeLibrary = $Library }
}
& cmake -S $NativeSource -B $NativeBuild -A x64 "-DNODE_INCLUDE_DIR=$NodeIncludeDir" "-DNODE_LIBRARY=$NodeLibrary"
if ($LASTEXITCODE -ne 0) { throw "Windows native configure failed" }
& cmake --build $NativeBuild --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { throw "Windows native build failed" }
& ctest --test-dir $NativeBuild -C $Configuration --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Windows native logic tests failed" }
& node (Join-Path $PSScriptRoot "assert-packaging-idle.mjs")
if ($LASTEXITCODE -ne 0) { throw "Quit tulpa before installing native runtime resources" }
& cmake --install $NativeBuild --config $Configuration --prefix $NativeOutput
if ($LASTEXITCODE -ne 0) { throw "Windows native install failed" }
Write-Host "Windows native helpers built in $NativeOutput"
