@echo off
REM Setup WSL and build joyomni_ops on Windows

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo WSL Setup for joyomni_ops
echo ==========================================
echo.

REM Check if WSL is installed
echo [1/4] Checking WSL...
wsl --list >nul 2>&1
if !errorlevel! neq 0 (
    echo WSL not installed. Installing...
    wsl --install
    echo.
    echo ⚠ Restart computer when prompted, then run this script again
    pause
    exit /b 0
)
echo OK - WSL installed

echo.
echo [2/4] Checking/Installing WSL dependencies...
REM Use the system Python 3.12 (Ubuntu 24.04 default) -- no deadsnakes needed.
REM JoyAI deps all have cp312 wheels; joyomni_ops builds against the active Python.
wsl bash -c "sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-dev python3-pip git build-essential"
if !errorlevel! neq 0 (
    echo ERROR: apt dependency install failed ^(see messages above^)
    echo   If python3.10 is unavailable, the box may not be Ubuntu -- run: wsl bash -c "python3 --version"
    exit /b 1
)
echo OK

echo.
echo [3/4] Setting up venv in WSL...
wsl bash -c "cd /mnt/c/workspace/world/JoyAI-Video-Edit && rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip setuptools wheel && pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128"
if !errorlevel! neq 0 (
    echo ERROR: venv setup failed
    exit /b 1
)
echo OK

echo.
echo [4/4] Installing requirements...
wsl bash -c "cd /mnt/c/workspace/world/JoyAI-Video-Edit && source .venv/bin/activate && pip install -r deploy/requirements.txt"
if !errorlevel! neq 0 (
    echo ERROR: requirements install failed
    exit /b 1
)
echo OK

echo.
echo ==========================================
echo ✅ WSL Setup Complete
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Open WSL:
echo    wsl
echo.
echo 2. Navigate to project:
echo    cd /mnt/c/workspace/world/JoyAI-Video-Edit
echo.
echo 3. Activate venv:
echo    source .venv/bin/activate
echo.
echo 4. Build joyomni_ops:
echo    cd deploy/joyomni_ops
echo    export JOYOMNI_OPS_NO_FP8=1
echo    python setup.py build_ext --inplace
echo.
echo 5. Run inference:
echo    cd ~/JoyAI-Video-Edit
echo    bash run.sh
echo.
pause
