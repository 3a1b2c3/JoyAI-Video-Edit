@echo off
REM Start JoyAI-Video-Edit server with Echo FP4 + FP8 disabled + low-VRAM
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "!SCRIPT_DIR!"

echo Starting JoyAI-Video-Edit server with Echo FP4 (low-VRAM, no FP8)...
echo.

wsl bash -c "cd /mnt/c/workspace/world/JoyAI-Video-Edit && source .venv/bin/activate && export JOYOMNI_MODEL=echo_fp4 JOYOMNI_LOW_VRAM=1 JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0 && bash run_server_best.sh"
