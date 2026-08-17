@echo off
REM Development shell for joyomni_ops with CUDA 12.8

setlocal enabledelayedexpansion

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=!CUDA_HOME!\bin;!PATH!
set JOYOMNI_OPS_NO_FP8=1

echo.
echo ==========================================
echo Development Shell - CUDA 12.8
echo ==========================================
echo.
echo CUDA_HOME: !CUDA_HOME!
echo.

nvcc --version | findstr /R "release"

echo.
echo Python:
C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe --version

echo.
echo PyTorch CUDA:
C:\workspace\world\JoyAI-Video-Edit\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.version.cuda)"

echo.
echo Ready to build:
echo   cd C:\workspace\world\JoyAI-Video-Edit\deploy\joyomni_ops
echo   python setup.py build_ext --inplace
echo.

cmd /k
