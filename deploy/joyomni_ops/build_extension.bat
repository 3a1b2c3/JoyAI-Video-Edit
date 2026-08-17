@echo off
setlocal enableextensions enabledelayedexpansion

echo Building joyomni_ops extension...
echo.

REM Set up MSVC environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Set up CUDA environment
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0

REM Disable FP8 (no CUTLASS)
set JOYOMNI_OPS_NO_FP8=1

REM Get venv python
set VENV_PYTHON=.\..\..\.venv\Scripts\python.exe

echo MSVC setup complete. Building with:
echo   Python: !VENV_PYTHON!
echo   CUDA_HOME: !CUDA_HOME!
echo   FP8 disabled: JOYOMNI_OPS_NO_FP8=1
echo.

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Build
echo Running: !VENV_PYTHON! setup.py build_ext --inplace
!VENV_PYTHON! setup.py build_ext --inplace

if errorlevel 1 (
    echo.
    echo Build FAILED
    exit /b 1
) else (
    echo.
    echo ✓ Build successful
    exit /b 0
)
