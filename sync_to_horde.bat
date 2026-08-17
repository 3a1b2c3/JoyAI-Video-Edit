@echo off
setlocal enableextensions enabledelayedexpansion

REM Sync checkpoints to horde via rsync in WSL

echo ========================================================================
echo Sync Checkpoints to Horde (rsync via WSL)
echo ========================================================================
echo.

REM Check WSL
echo [1/2] Checking WSL...
wsl --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: WSL not installed
    exit /b 1
)
echo   [OK] WSL found
echo.

REM Sync via rsync
echo [2/2] Syncing checkpoints to horde@10.57.233.24...
echo.

wsl bash -c "rsync -avz --progress /mnt/c/workspace/world/JoyAI-Video-Edit/deploy/deps/checkpoints/ horde@10.57.233.24:~/JoyAI-Video-Edit/deploy/deps/checkpoints/"

if errorlevel 1 (
    echo.
    echo ERROR: Sync failed
    echo Check:
    echo   1. SSH key configured for horde
    echo   2. Network connectivity to 10.57.233.24
    echo   3. rsync installed in WSL: wsl sudo apt install rsync
    exit /b 1
)

echo.
echo ========================================================================
echo Sync Complete
echo ========================================================================
echo.
