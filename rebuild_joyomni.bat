@echo off
REM Rebuild joyomni_ops on Windows
REM NOTE: Must be run from "Developer Command Prompt for VS 2022"
REM
REM To open: Search "Developer Command Prompt" in Start menu
REM Then run this script from that command prompt

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Rebuilding joyomni_ops
echo ==========================================
echo.
echo IMPORTANT: This must run from Visual Studio Developer Command Prompt
echo If you see cl.exe not found errors, close this and:
echo   1. Search "Developer Command Prompt for VS 2022" in Start menu
echo   2. Open it
echo   3. Run this script again
echo.

cd /d "%~dp0deploy\joyomni_ops"

REM Fix setup.py CUDA path to 12.8
powershell -Command "(Get-Content setup.py) -replace 'v12\.4', 'v12.8' | Set-Content setup.py"

REM Set CUDA environment
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=!CUDA_HOME!\bin;!PATH!

REM Skip FP8 GEMM (requires CUTLASS) — build only lightweight kernels
REM (fused_norm_scale_shift, fused_qknorm_rope_3d, rmsnorm)
set JOYOMNI_OPS_NO_FP8=1
echo      JOYOMNI_OPS_NO_FP8: !JOYOMNI_OPS_NO_FP8!

REM Venv Python (has setuptools, wheel)
set PYTHON=..\..\..\.venv\Scripts\python.exe

echo [1/4] Checking CUDA...
where nvcc >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: nvcc not found
    exit /b 1
)
nvcc --version | findstr /R "release"

echo [2/4] Cleaning...
if exist build rmdir /s /q build 2>nul
for %%f in (joyomni_ops\_C.cpython-*.so joyomni_ops\_C.cpython-*.pyd) do if exist "%%f" del "%%f"

echo [3/4] Building...
!PYTHON! setup.py clean --all 2>nul
!PYTHON! setup.py build_ext --inplace
if !errorlevel! neq 0 exit /b 1

echo [4/4] Verifying...
if exist joyomni_ops\_C.cpython-310-x86_64.pyd (
    echo   ✅ Built: joyomni_ops\_C.cpython-310-x86_64.pyd
) else if exist joyomni_ops\_C.cpython-310-x86_64-linux-gnu.so (
    echo   ✅ Built: joyomni_ops\_C.cpython-310-x86_64-linux-gnu.so
) else (
    echo   ❌ Build failed - .so/.pyd not found
    dir joyomni_ops
    exit /b 1
)

echo.
echo ==========================================
echo ✅ Build complete!
echo ==========================================
echo.
echo Next: Run inference
echo   bash run_inference.sh
echo.
