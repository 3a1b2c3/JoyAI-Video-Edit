#!/usr/bin/env pwsh

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JoyAI-Video-Edit - Quick Test via WSL" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run in WSL
wsl bash -c @"
cd ~/JoyAI-Video-Edit
echo "[1/2] Git pull..."
git pull

echo ""
echo "[2/2] Running test_quick.py..."
python test_quick.py
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Test failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "✅ Test passed!" -ForegroundColor Green
