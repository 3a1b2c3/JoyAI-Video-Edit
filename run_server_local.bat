@echo off
REM Runs the JoyAI-Video-Edit streaming server locally via WSL, using the local
REM .venv (native WSL filesystem CUDA build -- see run_server.bat for the old
REM Git-Bash-on-Windows path, which lacks fastapi/uvicorn/joyomni_ops).
REM
REM   run_server_local.bat

wsl bash -c "cd /mnt/c/workspace/world/JoyAI-Video-Edit && source .venv/bin/activate && bash run_server_best.sh"
