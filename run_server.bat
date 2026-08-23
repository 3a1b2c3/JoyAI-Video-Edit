@echo off
REM Starts the JoyAI-Video-Edit streaming server (deploy/xvideo/serving/serve_joyomni_streaming.py)
REM via the WSL venv, since fastapi/uvicorn/joyomni_ops only live there (see run_wsl.bat).

wsl bash -c "cd /mnt/c/workspace/world/JoyAI-Video-Edit && source .venv/bin/activate && bash deploy/run_server.sh"
