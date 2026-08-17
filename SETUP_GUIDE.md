


## Quick Setup (3 Steps)

### 1. Setup Virtual Environment & Install Dependencies
```bash
bash setup_venv.sh
```

This will:
- ✅ Create Python virtual environment (`.venv`)
- ✅ Install PyTorch with CUDA support
- ✅ Install all required packages (transformers, diffusers, opencv, imageio, etc.)

**Time**: ~5-10 minutes (depending on internet speed)

### 2. Verify Checkpoints
```bash
bash download_checkpoints.sh
```

This will:
- ✅ Create checkpoint directory structure
- ✅ Check for DiT and VAE checkpoints
- ✅ Report missing files

**Note**: You must manually download the model checkpoints:
- **DiT**: `joyai_video_edit_dit_0804.pth` (28-30 GB)
- **VAE**: `xvideo_xvae-released-ckpt` directory

### 3. Check Environment
```bash
bash check_environment.sh
```

This will verify:
- ✅ Python and PyTorch installation
- ✅ GPU/CUDA availability
- ✅ GPU memory
- ✅ Checkpoint locations
- ✅ Inference script availability

---

## Detailed Setup Steps

### Step 1: Prerequisites

**System Requirements**:
- Linux, macOS, or Windows with WSL/Git Bash
- Python 3.10 or higher
- NVIDIA GPU with CUDA capability (RTX 5090, A40, A100, etc.)
- 32 GB+ VRAM (RTX 5090), 48 GB+ for A40

**Software**:
- Git
- Bash shell
- Conda/Pip package manager

### Step 2: Clone or Prepare Repository

```bash
cd JoyAI-Video-Edit
ls -la  # Verify setup_*.sh scripts are present
```

### Step 3: Run Complete Setup

**Option A**: Automated (Recommended)
```bash
bash setup_all.sh
```

This runs all setup steps automatically:
1. Creates venv
2. Installs dependencies  
3. Verifies environment
4. Checks checkpoints

**Option B**: Step-by-Step
```bash
# 1. Setup venv and install packages
bash setup_venv.sh

# 2. Verify environment
bash check_environment.sh

# 3. Setup checkpoints
bash download_checkpoints.sh
```

### Step 4: Download Model Checkpoints

The scripts will tell you where to place checkpoints:
```
deploy/deps/checkpoints/
├── JoyAI-Video-Edit/
│   ├── dit/dit/
│   │   └── joyai_video_edit_dit_0804.pth        (28-30 GB)
│   └── vae/
│       ├── config.json
│       └── diffusion_pytorch_model.bin
```

**Download locations** (you provide):
- **DiT**: [Your model source]
- **VAE**: [Your model source]

After downloading, verify with:
```bash
bash check_environment.sh
```

### Step 5: Test Installation

```bash
# Run single-frame test
bash run_inference.sh assets/Recording\ 2026-08-12\ 205529.mp4 outputs/test.mp4 1 1
```

Expected output:
```
✓ Loaded 1 frames
✓ DiT loaded (float32)
✓ VAE loaded
✓ Encoded to latents
Inference time: X.XXs (Y.Y fps)
✓ Video saved: outputs/test.mp4
✅ INFERENCE COMPLETE
```

---

## Troubleshooting Setup

### "bash: command not found"
**Solution**: You need Bash shell
- **Linux/Mac**: Bash is pre-installed
- **Windows**: Install Git Bash or use WSL

### "Python 3 not found"
**Solution**: Install Python 3.10+
```bash
# macOS with Homebrew
brew install python3

# Ubuntu/Debian
sudo apt-get install python3.10

# Windows: Download from python.org
```

### "CUDA not available"
**Solution**: Ensure NVIDIA GPU drivers are installed
```bash
nvidia-smi  # Should show GPU info

# If not installed:
# Download from https://developer.nvidia.com/cuda-downloads
```

### "PyTorch installation fails"
**Solution**: Check internet connection and disk space (5+ GB needed)
```bash
# Verify pip
pip --version

# Try manual installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Checkpoints not found"
**Solution**: Download and place in correct directories
```bash
ls -lh deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/
# Should show: joyai_video_edit_dit_0804.pth (28-30 GB)
```

---

## After Setup: Running Inference

### Quick Start
```bash
bash run_inference.sh input.mp4
```

### Custom Parameters
```bash
bash run_inference_custom.sh input.mp4 output.mp4 4 512 512 8 42
```

### Batch Processing
```bash
bash batch_inference.sh ./videos outputs 4 4
```

### Manual Python (No Scripts)
```bash
source .venv/bin/activate
python run_inference.py --video input.mp4 --out output.mp4 --steps 4
```

---

## Environment Variables (Optional)

Control GPU and memory behavior:
```bash
# Force specific GPU
export CUDA_VISIBLE_DEVICES=0

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable async operations (Windows)
export HF_DEACTIVATE_ASYNC_LOAD=1
```

---

## Verification Checklist

After setup, verify:
- [ ] `.venv` directory exists
- [ ] `python run_inference.py` works (from activated venv)
- [ ] `nvidia-smi` shows your GPU
- [ ] DiT checkpoint exists (28-30 GB)
- [ ] VAE checkpoint directory exists
- [ ] `bash check_environment.sh` reports all ✅

---

## Setup Script Reference

| Script | Purpose | Time |
|--------|---------|------|
| `setup_venv.sh` | Create venv + install packages | 5-10 min |
| `check_environment.sh` | Verify GPU/PyTorch/checkpoints | 1 min |
| `download_checkpoints.sh` | Setup checkpoint directories | 1 min |
| `setup_all.sh` | Run all above in sequence | 5-15 min |

---

## Getting Help

1. **Check documentation**:
   - `SHELL_SCRIPTS.md` — Script usage
   - `RUN_INFERENCE.md` — Inference options
   - `QUANTIZATION_ATTEMPTS.md` — Technical details

2. **Check environment**:
   ```bash
   bash check_environment.sh
   ```

3. **Test inference**:
   ```bash
   bash run_inference.sh assets/Recording\ 2026-08-12\ 205529.mp4 outputs/test.mp4 1 1
   ```

---

## Summary

| Step | Command | Status |
|------|---------|--------|
| 1. Setup | `bash setup_venv.sh` | ✅ |
| 2. Check | `bash check_environment.sh` | ✅ |
| 3. Checkpoints | `bash download_checkpoints.sh` | ✅ |
| 4. Run | `bash run_inference.sh` | ✅ |

**Estimated total time**: 10-20 minutes (first time)
