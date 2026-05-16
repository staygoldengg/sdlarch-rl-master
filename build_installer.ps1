# build_installer.ps1
# ===================
# Full build pipeline for Striker — The Enlightened
#
# Produces a single Windows installer:
#   src-tauri\target\release\bundle\nsis\Striker The Enlightened_1.0.0_x64-setup.exe
#
# What it does:
#   1. Checks prerequisites (Python venv, npm, cargo)
#   2. PyInstaller: bundles the FastAPI backend into dist/striker-server/
#   3. Tauri build: compiles the Rust shell + React UI into a Windows installer
#      The installer automatically includes dist/striker-server/ as a resource.
#   4. Prints the final installer path.
#
# Usage:
#   .\build_installer.ps1
#   .\build_installer.ps1 -SkipPyInstaller   # re-use existing dist/striker-server/
#   .\build_installer.ps1 -SkipTauri         # only rebuild the Python server

param(
    [switch]$SkipPyInstaller,
    [switch]$SkipTauri,
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# ── Color helpers ──────────────────────────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "`n  >> $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg) Write-Host "     OK  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "     !!  $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "     ERR $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  ==========================================" -ForegroundColor Magenta
Write-Host "   Striker The Enlightened — Build Installer" -ForegroundColor Magenta
Write-Host "  ==========================================" -ForegroundColor Magenta
Write-Host ""

# ── 1. Locate Python (prefer .venv) ───────────────────────────────────────────
Write-Step "Locating Python environment"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip    = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

if (Test-Path $VenvPython) {
    $PyExe = $VenvPython
    $PipExe = $VenvPip
    Write-OK "Using .venv: $PyExe"
} else {
    $PyExe  = "python"
    $PipExe = "pip"
    Write-Warn ".venv not found — using system python"
}

# ── 2. Install PyInstaller if missing ─────────────────────────────────────────
Write-Step "Checking PyInstaller"
$hasPI = & $PyExe -c "import PyInstaller; print('ok')" 2>$null
if ($hasPI -ne "ok") {
    Write-Warn "PyInstaller not found — installing..."
    & $PipExe install pyinstaller --quiet
    Write-OK "PyInstaller installed"
} else {
    Write-OK "PyInstaller available"
}

# ── 3. Build the Python backend ───────────────────────────────────────────────
if (-not $SkipPyInstaller) {
    Write-Step "Building Python backend (PyInstaller)"
    Write-Host "     This bundles uvicorn + fastapi + torch + weaponized_ai" -ForegroundColor DarkGray
    Write-Host "     PyTorch is large — expect 5-15 minutes on first run." -ForegroundColor DarkGray
    Write-Host ""

    $specFile = Join-Path $ProjectRoot "striker_server.spec"
    if (-not (Test-Path $specFile)) {
        Write-Fail "striker_server.spec not found at $specFile"
    }

    $piArgs = @(
        $specFile,
        "--distpath", (Join-Path $ProjectRoot "dist"),
        "--workpath", (Join-Path $ProjectRoot "build"),
        "--noconfirm"
    )
    if ($Verbose) { $piArgs += "--log-level", "INFO" }

    & $PyExe -m PyInstaller @piArgs
    if ($LASTEXITCODE -ne 0) { Write-Fail "PyInstaller failed (exit $LASTEXITCODE)" }

    $serverDir = Join-Path $ProjectRoot "dist\striker-server"
    if (-not (Test-Path "$serverDir\striker-server.exe")) {
        Write-Fail "Expected dist\striker-server\striker-server.exe — build may have failed"
    }
    Write-OK "Backend built: dist\striker-server\striker-server.exe"

    # Report size
    $sizeMB = [math]::Round((Get-ChildItem $serverDir -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
    Write-OK "Server bundle size: ${sizeMB} MB"
} else {
    Write-Warn "Skipping PyInstaller (--SkipPyInstaller)"
    $serverDir = Join-Path $ProjectRoot "dist\striker-server"
    if (-not (Test-Path "$serverDir\striker-server.exe")) {
        Write-Fail "dist\striker-server\striker-server.exe not found. Run without -SkipPyInstaller first."
    }
    Write-OK "Re-using existing dist\striker-server\"
}

# ── 4. Ensure npm dependencies ────────────────────────────────────────────────
Write-Step "Checking npm dependencies"
if (-not (Test-Path (Join-Path $ProjectRoot "node_modules"))) {
    Write-Host "     Running npm install..." -ForegroundColor DarkGray
    npm install --silent
    if ($LASTEXITCODE -ne 0) { Write-Fail "npm install failed" }
}
Write-OK "node_modules present"

# ── 5. Tauri build ────────────────────────────────────────────────────────────
if (-not $SkipTauri) {
    Write-Step "Building Tauri desktop app + Windows installer"
    Write-Host "     Compiling Rust + bundling React UI..." -ForegroundColor DarkGray
    Write-Host "     First Rust compile can take 10-20 minutes." -ForegroundColor DarkGray
    Write-Host ""

    npm run tauri build 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Fail "Tauri build failed (exit $LASTEXITCODE)" }
    Write-OK "Tauri build complete"
} else {
    Write-Warn "Skipping Tauri build (--SkipTauri)"
}

# ── 6. Locate output installer ────────────────────────────────────────────────
Write-Step "Locating output artifacts"
$bundleDir = Join-Path $ProjectRoot "src-tauri\target\release\bundle"

$installer = Get-ChildItem -Path $bundleDir -Recurse -Filter "*setup.exe" -ErrorAction SilentlyContinue |
             Select-Object -First 1
$msi       = Get-ChildItem -Path $bundleDir -Recurse -Filter "*.msi" -ErrorAction SilentlyContinue |
             Select-Object -First 1
$plainExe  = Join-Path $ProjectRoot "src-tauri\target\release\Striker The Enlightened.exe"

Write-Host ""
Write-Host "  ==========================================" -ForegroundColor Magenta
Write-Host "   BUILD COMPLETE" -ForegroundColor Green
Write-Host "  ==========================================" -ForegroundColor Magenta
Write-Host ""

if ($installer) {
    Write-Host "  Installer:  $($installer.FullName)" -ForegroundColor Green
    $sizeMB = [math]::Round($installer.Length / 1MB, 1)
    Write-Host "  Size:       ${sizeMB} MB" -ForegroundColor Green
}
if ($msi) {
    Write-Host "  MSI:        $($msi.FullName)" -ForegroundColor Green
}
if (Test-Path $plainExe) {
    Write-Host "  Exe:        $plainExe" -ForegroundColor Green
}

Write-Host ""
Write-Host "  To install: run the setup.exe above." -ForegroundColor Cyan
Write-Host "  The AI backend (striker-server.exe) starts automatically" -ForegroundColor Cyan
Write-Host "  when you launch 'Striker The Enlightened'." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Stored data location (after first run):" -ForegroundColor DarkGray
Write-Host "    <install dir>\striker-server\data\  — brain store, models" -ForegroundColor DarkGray
Write-Host ""
