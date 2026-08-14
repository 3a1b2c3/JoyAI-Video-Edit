@echo off
REM JoyAI-Video-Edit Windows Setup
REM Prerequisites: Python 3.10, git-lfs, CUDA 12.8+

cd /d "%~dp0"

set "PYEXE=.\.venv\Scripts\python.exe"
set "UV_EXE=C:\Users\kschmid\.local\bin\uv.exe"

echo ========================================
echo JoyAI-Video-Edit Setup
echo ========================================
echo.

REM Check Python 3.10
echo Checking Python 3.10...
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found
  exit /b 1
)

REM Create venv if needed
if not exist "%PYEXE%" (
  echo Creating venv with Python 3.10...
  "%UV_EXE%" venv --python 3.10 .venv
  if errorlevel 1 (
    echo ERROR: Failed to create venv
    exit /b 1
  )
)

echo.
echo Installing PyTorch (cu128)...
"%UV_EXE%" pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio -p .venv
if errorlevel 1 (
  echo WARNING: PyTorch install may have failed
)

echo.
echo Installing requirements (with index fallback)...
"%UV_EXE%" pip install -r requirements.txt -p .venv --index-strategy unsafe-best-match
if errorlevel 1 (
  echo ⚠ Full requirements failed, installing critical packages...
  "%UV_EXE%" pip install triton-windows flash-attn loguru einops -p .venv
)

echo.
echo Creating deploy/deps structure...
if not exist "deploy\deps\checkpoints" mkdir deploy\deps\checkpoints
if not exist "deploy\deps\cache" mkdir deploy\deps\cache
if not exist "deploy\recordings" mkdir deploy\recordings

echo.
echo Verifying GPU...
"%PYEXE%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo [3/4] Downloading models...
"%PYEXE%" download.py

echo.
echo ========================================
echo Setup Complete
echo ========================================
echo.
echo Launch server:
echo   server.bat
echo.
echo Open browser:
echo   http://localhost:8081/
echo.
