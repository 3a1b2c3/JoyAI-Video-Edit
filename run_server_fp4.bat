@echo off
REM Deprecated name -- kept as a compatibility shim. See run_server_bf16.bat.
echo run_server_fp4.bat has been renamed to run_server_bf16.bat (same behavior). Update your scripts/habits.
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%run_server_bf16.bat" %*
