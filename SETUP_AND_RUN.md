# JoyAI-Video-Edit Setup and Inference Guide

## Quick Start

### On Horde (A40, 48GB)
```bash
cd ~/JoyAI-Video-Edit

# Set environment
export PYTHONPATH=~/JoyAI-Video-Edit/deploy:$PYTHONPATH
export LD_LIBRARY_PATH=/home/horde/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# Run inference
bash run_with_joyomni.sh
```

### Locally (RTX 5090, 32GB)
```bash
cd C:\workspace\world\JoyAI-Video-Edit

# Build joyomni_ops first (see below)
# Then run
bash run_with_joyomni.sh
```

## Environment Setup

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- CUDA Toolkit 12.4+
- Video input file
- Style/reference image

### Dependencies
```bash
pip install -r deploy/requirements.txt
```

Key packages:
- torch==2.9.1+cu128
- transformers>=4.57.1
- diffusers==0.36.0
- opencv-python-headless
- imageio-ffmpeg

## Building joyomni_ops

### Step 1: Install CUDA Toolkit

**On Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
```

**On Windows:**
Download from https://developer.nvidia.com/cuda-toolkit

### Step 2: Build Extension

```bash
cd deploy/joyomni_ops

export CUDA_HOME=/usr/local/cuda-12.4
export JOYOMNI_OPS_NO_FP8=1  # Skip FP8 for faster build

python3 setup.py build_ext --inplace

# Verify build
ls -la joyomni_ops/_C.cpython-*.so
```

### Step 3: Install Module

The .so file is auto-copied to `joyomni_ops/` during build. To make it importable:

```bash
# Ensure __init__.py exists and has:
cat joyomni_ops/__init__.py
```

Should contain:
```python
from . import _C
fused_norm_scale_shift = _C.fused_norm_scale_shift
fused_qk_norm_rope_3d_paired = _C.fused_qk_norm_rope_3d_paired
rmsnorm = _C.rmsnorm

__all__ = ['fused_norm_scale_shift', 'fused_qk_norm_rope_3d_paired', 'rmsnorm']
```

### Step 4: Runtime Configuration

Set these environment variables before running inference:

```bash
export PYTHONPATH=~/JoyAI-Video-Edit/deploy:$PYTHONPATH
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
```

## Running Inference

### Using Wrapper Script (Recommended)

```bash
bash run_with_joyomni.sh [video] [output] [style] [frames] [height] [width] [steps]
```

**Examples:**

```bash
# Defaults (minimal, fast)
bash run_with_joyomni.sh

# Custom video
bash run_with_joyomni.sh my_video.mp4

# Full control
bash run_with_joyomni.sh input.mp4 output.mp4 style.png 1 512 512 4
```

### Direct Python

```bash
python3 run_inference_lowmem.py \
    --video assets/Recording.mp4 \
    --out outputs/result.mp4 \
    --ref-image assets/image.png \
    --frames 1 \
    --height 256 \
    --width 256 \
    --steps 1
```

## Configuration

### Default Parameters

File: `run_with_joyomni.sh`

```bash
VIDEO="assets/Recording 2026-08-12 205529.mp4"
OUTPUT="outputs/dit_output.mp4"
REF_IMAGE="assets/image.png"
FRAMES=1
HEIGHT=256
WIDTH=256
STEPS=1
```

### Model Configuration

File: `deploy/xvideo/config.py` (ExpConfig)

Key settings:
- `dit_ckpt`: Path to DiT checkpoint (default: 0811)
- `dit_precision`: Model precision (default: bf16)
- `vae_precision`: VAE precision (default: fp16)
- `context_update_interval`: KV cache update frequency (default: 2)

## Troubleshooting

### joyomni_ops Import Fails

**Error:** `ImportError: cannot import name 'fused_norm_scale_shift'`

**Fix:**
```bash
# 1. Verify .so exists
ls -la deploy/joyomni_ops/joyomni_ops/*.so

# 2. Check LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

# 3. Test import
python3 -c "from joyomni_ops import fused_norm_scale_shift; print('OK')"
```

### CUDA Not Found

**Error:** `OSError: CUDA_HOME environment variable is not set`

**Fix:**
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version  # Verify
```

### Out of Memory

Reduce resolution and frames:
```bash
bash run_with_joyomni.sh video.mp4 output.mp4 style.png 1 256 256 1
```

### Video File Not Found

```bash
ls -la assets/
# Ensure video exists in assets/ directory
```

## Performance Notes

### Horde A40 (48GB)
- Model: 16.26B parameters (float16 = ~14GB)
- Resolution: 256×256 (default)
- Frames: 1
- Steps: 1
- Estimated time: 5-10 minutes

### Local RTX 5090 (32GB)
- Same model fits with float16
- Can handle 512×512 at 1 frame
- Faster inference with proper CUDA setup

## Files Overview

```
JoyAI-Video-Edit/
├── run_with_joyomni.sh          # Main inference script
├── run_inference_lowmem.py       # Python inference (float16)
├── run_inference.py              # Python inference (float32)
├── BUILD_JOYOMNI_OPS.md          # Build instructions
├── RUN_WITH_JOYOMNI.md           # Usage guide
├── SETUP_AND_RUN.md              # This file
├── assets/                        # Input videos and images
│   ├── Recording 2026-08-12 205529.mp4
│   └── image.png
├── outputs/                       # Generated videos
├── deploy/
│   ├── joyomni_ops/             # CUDA extension
│   ├── xvideo/                  # Model code
│   ├── requirements.txt
│   └── deps/checkpoints/        # Model weights
└── .venv/                        # Python environment (optional)
```

## Next Steps

1. **Build joyomni_ops** (see BUILD_JOYOMNI_OPS.md)
2. **Run inference** (bash run_with_joyomni.sh)
3. **Check output** (outputs/dit_output.mp4)

## References

- [BUILD_JOYOMNI_OPS.md](BUILD_JOYOMNI_OPS.md) - Building the CUDA extension
- [RUN_WITH_JOYOMNI.md](RUN_WITH_JOYOMNI.md) - Usage guide
- [run_with_joyomni.sh](run_with_joyomni.sh) - Main script
