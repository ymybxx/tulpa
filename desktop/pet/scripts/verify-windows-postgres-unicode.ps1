#requires -Version 7.2
param([Parameter(Mandatory = $true)][string]$PostgresRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows) { throw 'The PostgreSQL Unicode probe requires Windows.' }

$source = (Resolve-Path -LiteralPath $PostgresRoot).Path
foreach ($binary in @('initdb.exe', 'postgres.exe', 'pg_ctl.exe', 'psql.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $source "bin/$binary") -PathType Leaf)) {
        throw "Prepared PostgreSQL binary is missing: $binary"
    }
}

# All locations passed to native PostgreSQL deliberately include Unicode and
# spaces. The original installation and the user's databases are never used.
$probeRoot = Join-Path ([IO.Path]::GetTempPath()) "tulpa 数据库 UTF8 $([Guid]::NewGuid().ToString('N'))"
$installation = Join-Path $probeRoot '安装 目录'
$data = Join-Path $probeRoot '数据库 数据'
$temporary = Join-Path $probeRoot '临时 文件'
$passwordFile = Join-Path $temporary '密码 文件.txt'
$queryFile = Join-Path $temporary '中文 查询.sql'
$serverLog = Join-Path $probeRoot '服务器 日志.txt'
$password = [Guid]::NewGuid().ToString('N')
$utf8 = [Text.UTF8Encoding]::new($false)
$mayHaveStarted = $false
$failure = $null
$shutdownFailure = $null

function Invoke-Postgres {
    param(
        [Parameter(Mandatory = $true)][string]$Binary,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int]$TimeoutSeconds = 90
    )
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = Join-Path $installation "bin/$Binary"
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    # PostgreSQL inherits pg_ctl's handles on Windows, keeping redirected pipes
    # open after pg_ctl exits. Control commands inherit CI output instead; the
    # server writes its own -l log. initdb/psql still return captured output.
    $captureOutput = $Binary -ne 'pg_ctl.exe'
    $start.RedirectStandardOutput = $captureOutput
    $start.RedirectStandardError = $captureOutput
    if ($captureOutput) {
        $start.StandardOutputEncoding = $utf8
        $start.StandardErrorEncoding = $utf8
    }
    foreach ($argument in $Arguments) { $start.ArgumentList.Add($argument) }
    # Child-specific environment: do not change the user's system, console, or
    # caller environment. In particular this does not change the system ACP.
    $start.Environment['TEMP'] = $temporary
    $start.Environment['TMP'] = $temporary
    $start.Environment['PGPASSWORD'] = $password
    $start.Environment['PGCLIENTENCODING'] = 'UTF8'
    $start.Environment['PGCONNECT_TIMEOUT'] = '5'
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw "Unable to start $Binary" }
        if ($captureOutput) {
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()
        }
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill($true)
            [void]$process.WaitForExit(5000)
            throw "$Binary exceeded the $TimeoutSeconds second timeout"
        }
        $stdout = ''
        $stderr = ''
        if ($captureOutput) {
            $readers = [Threading.Tasks.Task[]]@($stdoutTask, $stderrTask)
            if (-not [Threading.Tasks.Task]::WaitAll($readers, 5000)) {
                throw "$Binary exited but its output pipes did not close"
            }
            $stdout = $stdoutTask.GetAwaiter().GetResult()
            $stderr = $stderrTask.GetAwaiter().GetResult()
        }
        if ($process.ExitCode -ne 0) {
            throw "$Binary failed ($($process.ExitCode))`nstdout: $stdout`nstderr: $stderr"
        }
        if ($stderr) { Write-Host $stderr }
        return $stdout
    } finally {
        $process.Dispose()
    }
}

function Show-PostgresLog {
    if (Test-Path -LiteralPath $serverLog -PathType Leaf) {
        Write-Host 'PostgreSQL Unicode probe server log:'
        Get-Content -LiteralPath $serverLog -Encoding utf8 -Tail 120 | ForEach-Object { Write-Host $_ }
    }
}

try {
    New-Item -ItemType Directory -Path $temporary -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $installation -Recurse
    [IO.File]::WriteAllText($passwordFile, "$password`n", $utf8)
    Write-Host (Invoke-Postgres -Binary 'initdb.exe' -Arguments @(
        '-D', $data, '--encoding=UTF8', '--locale=C', '--auth=password',
        '--username=postgres', "--pwfile=$passwordFile"
    ) -TimeoutSeconds 120)

    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    try {
        $listener.Start()
        $port = $listener.LocalEndpoint.Port
    } finally {
        $listener.Stop()
    }
    $mayHaveStarted = $true
    Write-Host (Invoke-Postgres -Binary 'pg_ctl.exe' -Arguments @(
        'start', '-D', $data, '-l', $serverLog, '-w', '-t', '30',
        '-o', "-h 127.0.0.1 -p $port"
    ) -TimeoutSeconds 45)

    $text = '中文数据库验证'
    $expectedHex = [Convert]::ToHexString($utf8.GetBytes($text)).ToLowerInvariant()
    # Read actual UTF-8 SQL bytes through a Unicode filename, then compare ASCII
    # hex output so the shell's own display encoding cannot mask corruption.
    $query = @"
CREATE EXTENSION vector;
SELECT current_setting('server_encoding') || ':' || encode(convert_to('$text', 'UTF8'), 'hex')
    || ':' || (('[1,2,3]'::vector(3) <-> '[1,2,4]'::vector(3)) = 1.0)::text;
"@
    [IO.File]::WriteAllText($queryFile, $query, $utf8)
    $actual = Invoke-Postgres -Binary 'psql.exe' -Arguments @(
        '-X', '-q', '-A', '-t', '-v', 'ON_ERROR_STOP=1', '-h', '127.0.0.1',
        '-p', "$port", '-U', 'postgres', '-d', 'postgres', '-f', $queryFile
    ) -TimeoutSeconds 15
    if ($actual.Trim() -cne "UTF8:${expectedHex}:true") {
        throw "PostgreSQL Unicode or pgvector roundtrip mismatch: $($actual.Trim())"
    }
} catch {
    $failure = $_
} finally {
    # pg_ctl may start PostgreSQL even when its own readiness check fails.
    # Stop our isolated instance before ever deleting its data directory.
    if ($mayHaveStarted -or (Test-Path -LiteralPath (Join-Path $data 'postmaster.pid'))) {
        try {
            Write-Host (Invoke-Postgres -Binary 'pg_ctl.exe' -Arguments @(
                'stop', '-D', $data, '-m', 'fast', '-w', '-t', '20'
            ) -TimeoutSeconds 30)
        } catch {
            if ($null -eq $failure) { $failure = $_ }
            if (Test-Path -LiteralPath (Join-Path $data 'postmaster.pid')) {
                try {
                    Write-Host (Invoke-Postgres -Binary 'pg_ctl.exe' -Arguments @(
                        'stop', '-D', $data, '-m', 'immediate', '-w', '-t', '10'
                    ) -TimeoutSeconds 20)
                } catch {
                    $shutdownFailure = $_
                }
            }
        }
    }
    if (Test-Path -LiteralPath (Join-Path $data 'postmaster.pid')) {
        if ($null -eq $shutdownFailure) {
            $shutdownFailure = 'PostgreSQL still has a postmaster.pid after shutdown'
        }
    }
    if ($null -ne $failure -or $null -ne $shutdownFailure) { Show-PostgresLog }
    if ($null -eq $shutdownFailure) {
        if (Test-Path -LiteralPath $probeRoot) {
            Remove-Item -LiteralPath $probeRoot -Recurse -Force
        }
    } else {
        Write-Host "PostgreSQL shutdown was not verified; isolated files retained at $probeRoot"
    }
}
if ($null -ne $shutdownFailure) { throw $shutdownFailure }
if ($null -ne $failure) { throw $failure }
Write-Host 'PostgreSQL UTF-8 probe passed: Unicode installation, data, TEMP/TMP, password and SQL paths; UTF-8 SQL roundtrip; pgvector distance; clean shutdown.'
