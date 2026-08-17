@echo off
REM Run JoyAI inference in WSL (uses the venv setup_wsl.bat created in this dir).
setlocal
echo Running JoyAI inference in WSL...
wsl bash -c "cd /mnt/c/workspace/world/JoyAI-Video-Edit && source .venv/bin/activate && bash run.sh"
echo.
echo [done] output: outputs\dit_output.mp4
pause
endlocal
