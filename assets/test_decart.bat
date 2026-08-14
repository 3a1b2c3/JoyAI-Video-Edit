@echo off
setlocal enableextensions enabledelayedexpansion

echo.
echo ===================================================================
echo Decart QA Test Suite
echo ===================================================================
echo.

REM Test 1: Check dependencies
echo [TEST 1] Checking Decart dependencies...
"C:\workspace\world\scope\.venv\Scripts\python.exe" -c "import decart; import livekit; import cv2; import numpy; import imageio; print('✓ All dependencies installed')" 2>nul
if !ERRORLEVEL! neq 0 (
  echo ✗ FAILED: Missing dependencies
  exit /b 1
)

REM Test 2: Check DECART_API_KEY
echo [TEST 2] Checking DECART_API_KEY...
if "!DECART_API_KEY!"=="" (
  echo ✗ FAILED: DECART_API_KEY not set
  exit /b 1
)
echo ✓ DECART_API_KEY is set

REM Test 3: Verify scope\out directory
echo [TEST 3] Checking output directory...
if not exist "C:\workspace\world\scope\out" (
  mkdir "C:\workspace\world\scope\out"
)
echo ✓ Output directory ready

REM Test 4: Check if control video exists
echo [TEST 4] Checking control video...
if not exist "Recording 2026-08-12 205529.mp4" (
  echo ✗ FAILED: Recording 2026-08-12 205529.mp4 not found
  exit /b 1
)
echo ✓ Control video found

REM Test 5: Run Decart (dry run with 1 skip frame)
echo [TEST 5] Running Decart processing...
echo This may take 1-2 minutes...
echo.

set "DECART_SKIP=200"
"C:\workspace\world\scope\.venv\Scripts\python.exe" "C:\workspace\world\decart_long.py" "%cd%" "A high-speed racing track with vibrant neon colors."

if !ERRORLEVEL! equ 0 (
  echo.
  echo ===================================================================
  echo ✓ ALL TESTS PASSED
  echo ===================================================================
  echo.
  echo Output video saved to: C:\workspace\world\scope\out\
  echo.
) else (
  echo.
  echo ===================================================================
  echo ✗ DECART PROCESSING FAILED
  echo ===================================================================
  exit /b 1
)

endlocal
