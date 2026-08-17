@echo off
REM Build joyomni_ops with CUDA 12.8 (fixed)
REM Run from Command Prompt (NOT PowerShell)

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Building joyomni_ops with CUDA 12.8
echo ==========================================
echo.

REM Set CUDA 12.8 paths
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=!CUDA_HOME!\bin;!PATH!
set JOYOMNI_OPS_NO_FP8=1

echo [1/5] Verifying CUDA 12.8...
where nvcc >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: nvcc not found at !CUDA_HOME!
    echo Please install CUDA 12.8 first
    exit /b 1
)
nvcc --version | findstr /R "release 12.8"
if !errorlevel! neq 0 (
    echo ERROR: CUDA version is not 12.8
    nvcc --version
    exit /b 1
)
echo OK

echo.
echo [2/5] Installing PyTorch 2.9.1+cu128 for CUDA 12.8...
C:\Users\kschmid\.local\bin\uv.exe pip install --upgrade "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" --index-url https://download.pytorch.org/whl/cu128 --python C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe
if !errorlevel! neq 0 (
    echo ERROR: PyTorch install failed
    exit /b 1
)
echo OK

echo.
echo [3/5] Verifying PyTorch CUDA version...
C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
if !errorlevel! neq 0 (
    echo ERROR: PyTorch verification failed
    exit /b 1
)
echo OK

echo.
echo [4/5] Building joyomni_ops...
cd C:\workspace\world\JoyAI-Video-Edit\deploy\joyomni_ops
rmdir /s /q build 2>nul
del /q joyomni_ops\_C*.pyd 2>nul
C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe setup.py clean --all 2>nul
C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe setup.py build_ext --inplace
if !errorlevel! neq 0 (
    echo ERROR: Build failed
    exit /b 1
)
echo OK

echo.
echo [5/5] Verifying .pyd file...
if exist joyomni_ops\_C.cpython-310-x86_64.pyd (
    echo   ✅ Built: joyomni_ops\_C.cpython-310-x86_64.pyd
) else (
    echo   ❌ .pyd not found
    dir joyomni_ops
    exit /b 1
)

echo.
echo ==========================================
echo ✅ BUILD COMPLETE
echo ==========================================
echo.
echo Next: Run inference
echo   cd C:\workspace\world\JoyAI-Video-Edit
echo   bash run.sh
echo.
