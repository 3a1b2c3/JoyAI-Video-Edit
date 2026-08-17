@echo off
REM Build joyomni_ops on Windows via WSL (simple)

echo.
echo ==========================================
echo Building joyomni_ops via WSL
echo ==========================================
echo.

REM Check WSL
wsl --list >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: WSL not installed
    echo Run: wsl --install
    pause
    exit /b 1
)

REM Run build script in WSL
echo Launching WSL build...
echo.

wsl bash /mnt/c/workspace/world/JoyAI-Video-Edit/build_joyomni.sh

if !errorlevel! neq 0 (
    echo.
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ==========================================
echo ✅ BUILD COMPLETE
echo ==========================================
echo.
echo Next: Run inference in WSL
echo   wsl
echo   cd /mnt/c/workspace/world/JoyAI-Video-Edit
echo   bash run.sh
echo.
pause
