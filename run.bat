@echo off
setlocal enableextensions enabledelayedexpansion

REM DiT Inference

set "PYTHON=C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe"

if not exist "!PYTHON!" (
    echo ERROR: Python not found
    exit /b 1
)

"!PYTHON!" run_inference.py --video "%~1" --out "%~2"
