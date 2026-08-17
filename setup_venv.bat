@echo off
REM Setup virtual environment for JoyAI-Video-Edit on Windows

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Setting up venv
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python not found
    exit /b 1
)

echo Python:
python --version
echo.

REM Create venv
echo [1/4] Creating venv...
python -m venv .venv
call .venv\Scripts\activate.bat
echo OK

echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if !errorlevel! neq 0 (
    echo ERROR: pip upgrade failed
    exit /b 1
)
echo OK

echo.
echo [3/4] Installing PyTorch 2.9.1+cu128...
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if !errorlevel! neq 0 (
    echo ERROR: PyTorch install failed
    exit /b 1
)
echo OK

echo.
echo [4/4] Installing requirements...
python -m pip install -r deploy\requirements.txt
if !errorlevel! neq 0 (
    echo ERROR: requirements install failed
    exit /b 1
)
echo OK

echo.
echo ==========================================
echo ✅ venv ready
echo ==========================================
echo.
echo Activate venv:
echo   .venv\Scripts\activate.bat
echo.
echo Build joyomni_ops:
echo   cd deploy\joyomni_ops
echo   set JOYOMNI_OPS_NO_FP8=1
echo   python setup.py build_ext --inplace
echo.
echo Run inference:
echo   bash run.sh
echo.
pause
