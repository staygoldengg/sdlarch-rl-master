# local_test.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Striker — The Enlightened  |  Senior Tester One-Click Runner
#
# Usage:
#   .\local_test.ps1              # Start server, run all tests, stop server
#   .\local_test.ps1 -NoServer    # Run offline tests only (server not started)
#   .\local_test.ps1 -KeepServer  # Leave server running after tests
#
# Exit code mirrors test_local_client.py: 0 = all pass, 1 = any failure.
# ─────────────────────────────────────────────────────────────────────────────
param(
    [switch]$NoServer,
    [switch]$KeepServer,
    [switch]$Fast
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot   = $PSScriptRoot
$VenvPython    = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonExe     = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$ServerPort    = 8000
$ServerTimeout = 30   # seconds to wait for server startup

# ── Colours ───────────────────────────────────────────────────────────────────
function Write-Header($msg)  { Write-Host "`n  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)      { Write-Host "  [OK] $msg"   -ForegroundColor Green  }
function Write-Warn($msg)    { Write-Host "  [!!] $msg"   -ForegroundColor Yellow }
function Write-Err($msg)     { Write-Host "  [XX] $msg"   -ForegroundColor Red    }

# ── Check server liveness ─────────────────────────────────────────────────────
function Test-ServerAlive {
    try {
        $tcp = [System.Net.Sockets.TcpClient]::new()
        $task = $tcp.ConnectAsync("127.0.0.1", $ServerPort)
        $ok   = $task.Wait(500)
        $tcp.Dispose()
        return $ok -and ($task.Status -eq "RanToCompletion")
    } catch {
        return $false
    }
}

# ── Header ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "  Striker — The Enlightened  |  Local Client Test Runner" -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Python  : $PythonExe" -ForegroundColor DarkGray
Write-Host "  Root    : $ProjectRoot" -ForegroundColor DarkGray
Write-Host ""

Set-Location $ProjectRoot

# ── Optional: start the server ────────────────────────────────────────────────
$ServerProcess = $null
$WeLaunchedIt  = $false

if (-not $NoServer) {
    if (Test-ServerAlive) {
        Write-Ok "Server already running on port $ServerPort."
    } else {
        Write-Header "Starting Striker AI backend..."
        $ServerProcess = Start-Process `
            -FilePath $PythonExe `
            -ArgumentList "server_entry.py" `
            -PassThru `
            -WindowStyle Hidden

        Write-Host "  PID $($ServerProcess.Id) — waiting up to $ServerTimeout s for port $ServerPort..." `
            -ForegroundColor DarkGray

        $deadline = (Get-Date).AddSeconds($ServerTimeout)
        while ((Get-Date) -lt $deadline) {
            if (Test-ServerAlive) { break }
            Start-Sleep -Milliseconds 500
        }

        if (Test-ServerAlive) {
            Write-Ok "Backend is up."
            $WeLaunchedIt = $true
        } else {
            Write-Err "Backend did not come up in $ServerTimeout s."
            Write-Warn "Running offline tests only."
            $NoServer = $true
        }
    }
}

# ── Build test arguments ──────────────────────────────────────────────────────
$TestArgs = @("test_local_client.py")
if ($NoServer) { $TestArgs += "--no-server" }
if ($Fast)     { $TestArgs += "--fast"      }

# ── Run test suite ────────────────────────────────────────────────────────────
Write-Header "Running test suite..."
Write-Host ""

& $PythonExe @TestArgs
$TestExitCode = $LASTEXITCODE

# ── Result banner ─────────────────────────────────────────────────────────────
Write-Host ""
if ($TestExitCode -eq 0) {
    Write-Host "  ============================================================" -ForegroundColor Green
    Write-Host "  ALL TESTS PASSED — client is fully operational." -ForegroundColor Green
    Write-Host "  ============================================================" -ForegroundColor Green
} else {
    Write-Host "  ============================================================" -ForegroundColor Red
    Write-Host "  ONE OR MORE TESTS FAILED — see output above." -ForegroundColor Red
    Write-Host "  ============================================================" -ForegroundColor Red
}
Write-Host ""

# ── Optional: stop server ─────────────────────────────────────────────────────
if ($WeLaunchedIt -and -not $KeepServer -and $ServerProcess -ne $null) {
    Write-Header "Stopping backend (PID $($ServerProcess.Id))..."
    try {
        Stop-Process -Id $ServerProcess.Id -Force
        Write-Ok "Backend stopped."
    } catch {
        Write-Warn "Could not stop backend: $_"
    }
}

exit $TestExitCode
