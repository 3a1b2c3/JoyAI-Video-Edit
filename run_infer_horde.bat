@echo off
setlocal enabledelayedexpansion
REM Runs infer_standalone.sh on horde via WSL+SSH with FP4 model
REM (horde is where the actual Python/CUDA/joyomni_ops environment lives)
REM
REM   run_infer_horde.bat "<prompt>" <input_video_on_horde> [output_path]
REM
REM With FP4 setup:
REM   - Model: JoyAI-Echo 1.5 FP4 (22.81 GB)
REM   - GPU: RTX PRO 6000 Blackwell (96 GiB)
REM   - VRAM Mode: Full precision (no quantization on PRO 6000)
REM
REM Paths are resolved on horde, not Windows -- e.g. assets/input.mp4, not a
REM Windows path. Requires WSL SSH agent/keys set up for horde.

if "%~1"=="" (
    echo Usage: run_infer_horde.bat "<prompt>" ^<input_video_on_horde^> [output_path]
    echo Example: run_infer_horde.bat "A person dancing" assets/input.mp4
    exit /b 1
)
if "%~2"=="" (
    echo Usage: run_infer_horde.bat "<prompt>" ^<input_video_on_horde^> [output_path]
    exit /b 1
)

set "PROMPT=%~1"
set "INPUT_VIDEO=%~2"
set "OUTPUT_PATH=%~3"
if "%OUTPUT_PATH%"=="" set "OUTPUT_PATH=outputs/output_%RANDOM%.mp4"

echo Starting FP4 inference on Horde...
echo Prompt: !PROMPT!
echo Input: !INPUT_VIDEO!
echo Output: !OUTPUT_PATH!
echo.

REM SSH to horde with FP4 environment variables
wsl bash -c "ssh horde@10.57.233.24 'cd ~/JoyAI-Video-Edit && export JOYOMNI_MODEL=echo_fp4 && export JOYOMNI_LOW_VRAM=0 && bash infer_standalone.sh \"!PROMPT!\" \"!INPUT_VIDEO!\" \"!OUTPUT_PATH!\"'"
