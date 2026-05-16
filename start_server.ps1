# start_server.ps1
# Run: .\start_server.ps1
# Starts the weaponized_ai FastAPI server cleanly.
# The "NativeCommandError" from uvicorn writing to stderr is suppressed —
# those are just INFO logs, not real errors.

$projectRoot = $PSScriptRoot

Write-Host ""
Write-Host "  Weaponized AI Server" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:8000" -ForegroundColor Green
Write-Host ""

Set-Location $projectRoot

# Prefer .venv python if present, otherwise fall back to system python
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pyExe = $venvPython
} else {
    $pyExe = "python"
}

Write-Host "  Python: $pyExe" -ForegroundColor DarkGray
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# Route stderr to stdout so PowerShell doesn't treat INFO logs as errors
& $pyExe -m uvicorn weaponized_ai.api_server:app --reload --log-level info 2>&1
