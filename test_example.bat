@echo off
cd /d "%~dp0"

set "PYEXE=.\.venv\Scripts\python.exe"

if not exist "%PYEXE%" (
  echo ERROR: venv not found. Run setup.bat first.
  exit /b 1
)

echo.
"%PYEXE%" example_inference.py
echo.
