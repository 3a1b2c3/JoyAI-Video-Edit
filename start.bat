@echo off
REM JoyAI-Video-Edit - Complete Setup + Run

cd /d "%~dp0"

set "PYEXE=.\.venv\Scripts\python.exe"
set "UV_EXE=C:\Users\kschmid\.local\bin\uv.exe"

echo.
echo ========================================
echo JoyAI-Video-Edit Setup
echo ========================================
echo.

REM 1. Create venv
if not exist "%PYEXE%" (
  echo [1/3] Creating venv...
  "%UV_EXE%" venv --python 3.10 .venv
  if errorlevel 1 exit /b 1
  echo ✓ Venv created
) else (
  echo [1/3] ✓ Venv exists
)

echo.
echo [2/3] Installing PyTorch + requirements...
"%UV_EXE%" pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio -p .venv -q
"%UV_EXE%" pip install -r requirements.txt -p .venv -q
echo ✓ Dependencies installed

echo.
echo [3/3] Downloading models...
"%PYEXE%" download.py

echo.
echo ========================================
echo Launching Server
echo ========================================
echo.
echo Host: http://localhost:8080
echo.

"%PYEXE%" -m uvicorn xvideo.main:app --host 0.0.0.0 --port 8080
