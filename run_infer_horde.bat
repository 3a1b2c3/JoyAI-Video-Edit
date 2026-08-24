@echo off
REM Runs infer_standalone.sh on horde via WSL+SSH (horde is where the actual
REM Python/CUDA/joyomni_ops environment lives -- there is no local WSL setup for
REM this repo, only the Windows checkout used for editing).
REM
REM   run_infer_horde.bat "<prompt>" <input_video_on_horde> [output_path]
REM
REM Paths are resolved on horde, not Windows -- e.g. assets/input.mp4, not a
REM Windows path. Requires your WSL session's SSH agent/keys to already be set up
REM for horde (same as running ssh/scp/rsync to horde manually from WSL).

if "%~1"=="" (
    echo Usage: run_infer_horde.bat "<prompt>" ^<input_video_on_horde^> [output_path]
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

wsl bash -c "ssh horde@10.57.233.24 'cd ~/JoyAI-Video-Edit && bash infer_standalone.sh \"%PROMPT%\" \"%INPUT_VIDEO%\" \"%OUTPUT_PATH%\"'"
