# Running Inference with joyomni_ops

## Overview

`run_with_joyomni.sh` is the main inference script. It:
1. Sets up Python path for joyomni_ops
2. Verifies joyomni_ops is installed
3. Runs DiT inference with style frame conditioning

## Requirements

- joyomni_ops built and available in `deploy/joyomni_ops/`
- Video file in `assets/`
- Style/reference image in `assets/`

## Usage

### Basic (all defaults)
```bash
bash run_with_joyomni.sh
```

Default values:
- Video: `assets/Recording 2026-08-12 205529.mp4`
- Output: `outputs/dit_output.mp4`
- Style: `assets/image.png`
- Frames: 1
- Resolution: 256×256
- Steps: 1

### Custom video
```bash
bash run_with_joyomni.sh your_video.mp4
```

### Custom everything
```bash
bash run_with_joyomni.sh video.mp4 output.mp4 style.png 2 512 512 4
```

Arguments (in order):
1. `VIDEO` - Input video file (default: Recording 2026-08-12 205529.mp4)
2. `OUTPUT` - Output video file (default: outputs/dit_output.mp4)
3. `REF_IMAGE` - Style/reference image (default: assets/image.png)
4. `FRAMES` - Number of frames to process (default: 1)
5. `HEIGHT` - Output height (default: 256)
6. `WIDTH` - Output width (default: 256)
7. `STEPS` - Denoising steps (default: 1)

## Environment

The script automatically sets:

```bash
export PYTHONPATH=$SCRIPT_DIR/deploy:$PYTHONPATH
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH
```

- **PYTHONPATH**: Allows Python to find joyomni_ops in `deploy/joyomni_ops/`
- **LD_LIBRARY_PATH**: Points to PyTorch libraries (libc10.so, etc.) that joyomni_ops depends on

### Manual Setup (if needed)

```bash
export PYTHONPATH=~/JoyAI-Video-Edit/deploy:$PYTHONPATH
TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH
```

## Building joyomni_ops

If not already built, see [BUILD_JOYOMNI_OPS.md](BUILD_JOYOMNI_OPS.md)

Quick build on Linux:
```bash
cd deploy/joyomni_ops
export CUDA_HOME=/usr/local/cuda-12.4
export JOYOMNI_OPS_NO_FP8=1
python3 setup.py build_ext --inplace
```

## Troubleshooting

### joyomni_ops not found
```bash
export PYTHONPATH=~/JoyAI-Video-Edit/deploy:$PYTHONPATH
python3 -c "from joyomni_ops import fused_norm_scale_shift; print('OK')"
```

### Video file not found
Ensure video exists:
```bash
ls -la assets/Recording*.mp4
```

### Style frame not found
Ensure image exists:
```bash
ls -la assets/image.png
```

## Output

Results saved to `outputs/` directory (default: `outputs/dit_output.mp4`)

Check GPU usage during inference:
```bash
watch -n 1 nvidia-smi
```
