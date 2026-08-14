@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Decart Lucy - Neon Racing Track Restyle
REM Uses Recording 2026-08-12 205529.mp4 as control video
REM Applies vibrant neon racing aesthetic
REM Needs DECART_API_KEY set
REM Output: JoyAI-Video-Edit\assets\decart_neon_race.mp4
REM ==========================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"

REM Prompt now lives in an external text file (edit decart_prompt.txt, no need to
REM touch this .bat). Reads the first line into PROMPT.
set "PROMPTFILE=%HERE%\decart_prompt.txt"
if not exist "%PROMPTFILE%" ( echo ERROR: prompt file not found: %PROMPTFILE% & exit /b 1 )
set "PROMPT="
set /p PROMPT=<"%PROMPTFILE%"

echo.
echo ===================================================================
echo Decart Restyle: Recording to Neon Racing Track
echo ===================================================================
echo.
echo Prompt: Vibrant neon racing track aesthetic
echo Input: Recording 2026-08-12 205529.mp4
echo Output: decart_neon_race.mp4
echo.
echo Requires: DECART_API_KEY environment variable set
echo.

if "!DECART_API_KEY!"=="" (
  echo ERROR: DECART_API_KEY not set
  echo Set with: setx DECART_API_KEY your-key-here
  exit /b 1
)

REM Use screenshot as default reference image
set "DECART_REF=%HERE%\Screenshot 2026-08-14 163754.png"

"C:\workspace\world\scope\.venv\Scripts\python.exe" "C:\workspace\world\decart_long.py" "%HERE%" "!PROMPT!"

if !ERRORLEVEL! equ 0 (
  echo.
  echo ===================================================================
  echo SUCCESS! Output saved to: %HERE%\decart_neon_race.mp4
  echo ===================================================================
) else (
  echo.
  echo ===================================================================
  echo Decart failed. Check DECART_API_KEY and disk space.
  echo ===================================================================
  exit /b 1
)

endlocal
