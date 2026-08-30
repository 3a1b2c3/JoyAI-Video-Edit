@echo off
REM Start JoyAI-Video-Edit server with FP8 disabled (bf16 attention/MLP), low-VRAM
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "!SCRIPT_DIR!"

echo Starting JoyAI-Video-Edit server (bf16, low-VRAM, no FP8)...
echo.

wsl bash -c "cd /mnt/c/workspace/world/JoyAI-Video-Edit && source .venv/bin/activate && export JOYOMNI_LOW_VRAM=1 JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0 && bash run_server_best.sh"
