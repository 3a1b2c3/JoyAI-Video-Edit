@echo off
REM Start server in background, then run inference

cd /d "%~dp0"

echo Starting JoyAI server...
start "JoyAI Server" cmd /k .\server.bat

echo Waiting for server to start...
timeout /t 10

echo.
echo Running inference...
.\.venv\Scripts\python.exe infer.py

echo.
echo Done. Output: outputs\edited.mp4
echo.
pause
