@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PYEXE=%~dp0.venv\Scripts\python.exe"

if not exist "!PYEXE!" (
  echo ERROR: venv not found. Run setup.bat first.
  exit /b 1
)

echo.
echo JoyAI-Video-Edit Headless Inference
echo.

if not exist "deploy\deps\cache" mkdir "deploy\deps\cache"

set PYTHONPATH=%~dp0deploy;!PYTHONPATH!
set JOYOMNI_OPS_NO_FP8=1
set CUDA_MODULE_LOADING=LAZY

cd /d "%~dp0deploy"

echo Loading model (DiT required)...
echo.

"!PYEXE!" -c "
import sys
sys.path.insert(0, '.')

from pathlib import Path
import argparse

# Minimal inference test
args = argparse.Namespace(
    dit_ckpt=str(Path('.') / 'deps' / 'checkpoints' / 'JoyAI-Video-Edit' / 'dit' / 'joyai_video_edit_dit_0804.pth'),
    vae_ckpt=str(Path('.') / 'deps' / 'checkpoints' / 'JoyAI-Video-Edit' / 'vae'),
    text_encoder_ckpt=str(Path('.') / 'deps' / 'checkpoints' / 'MiMo-VL-7B-RL-2508'),
    device='cuda:0',
    dtype='bfloat16',
    num_inference_steps=8,
)

try:
    from xvideo.serving.joyomni_streaming import JoyOmniRuntime
    print('Initializing JoyOmniRuntime...')
    runtime = JoyOmniRuntime(
        dit_ckpt=str(args.dit_ckpt),
        device=args.device,
    )
    print('✓ Model loaded')
except FileNotFoundError as e:
    print(f'✗ Model file not found: {e}')
    print('Download models with: python download.py')
    sys.exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
"

endlocal
