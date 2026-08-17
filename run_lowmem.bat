@echo off
setlocal enableextensions enabledelayedexpansion

REM Low memory inference (reduced resolution)

echo ========================================================================
echo DiT Inference (Low Memory - 192x192)
echo ========================================================================
echo.

set "PYTHON=C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe"

if not exist "!PYTHON!" (
    echo ERROR: Python not found
    exit /b 1
)

echo Memory optimization:
echo   - Resolution: 192x192 (56%% less than 256x256)
echo   - Frames: 1 (minimal)
echo   - Steps: 1 (fastest)
echo.

set "VIDEO=%~1"
set "OUT=%~2"

if "!VIDEO!"=="" (
    echo ERROR: Usage: run_lowmem.bat input.mp4 output.mp4
    exit /b 1
)

if "!OUT!"=="" set "OUT=output.mp4"

"!PYTHON!" run_inference_efficient.py --video "!VIDEO!" --out "!OUT!" --height 192 --width 192 --frames 1 --steps 1

if errorlevel 1 (
    echo.
    echo ERROR: Inference failed
    exit /b 1
)

echo.
