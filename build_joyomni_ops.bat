@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0joyomni_ops"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set JOYOMNI_OPS_NO_FP8=1
..\\.venv\\Scripts\\python.exe setup.py develop
