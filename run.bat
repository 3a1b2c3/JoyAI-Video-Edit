@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PYEXE=%~dp0.venv\Scripts\python.exe"

if not exist "!PYEXE!" (
  echo ERROR: venv not found. Run setup.bat first.
  exit /b 1
)

echo.
echo JoyAI-Video-Edit Server
echo.

REM Create directories
if not exist "deploy\deps\cache" mkdir "deploy\deps\cache"
if not exist "deploy\recordings" mkdir "deploy\recordings"
if not exist "deploy\deps\cache\torchinductor" mkdir "deploy\deps\cache\torchinductor"

REM Set environment
set JOYOMNI_RECORD_DIR=%~dp0deploy\recordings
set TORCHINDUCTOR_CACHE_DIR=%~dp0deploy\deps\cache\torchinductor
set TRITON_CACHE_DIR=%~dp0deploy\deps\cache\triton
set CUDA_MODULE_LOADING=LAZY
set JOYOMNI_OPS_NO_FP8=1

echo Host:     http://localhost:8081
echo Recordings: !JOYOMNI_RECORD_DIR!
echo Cache:    !TORCHINDUCTOR_CACHE_DIR!
echo.

cd /d "%~dp0deploy"
"!PYEXE!" -m uvicorn xvideo.main:app --host 0.0.0.0 --port 8081

endlocal
