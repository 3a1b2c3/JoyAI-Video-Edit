# Setup Scripts: Python Version & Model Download Details

## Python Version Enforcement

### Requirement: Python 3.10 or 3.11 ONLY

The scripts now enforce Python version requirements:

```bash
bash setup_all.sh
```

**Checks**:
- ✅ Python must be 3.10.x or 3.11.x
- ❌ Python 3.9 or earlier → blocked (too old)
- ❌ Python 3.12+ → blocked (snapshot_download deadlock on Windows)

**Why**:
- Python 3.12 has threading issues with HuggingFace downloads on Windows
- Python 3.10/3.11 are stable and tested
- Memory notes: "Never use Python 3.12 on Windows"

**If you have 3.12**:
```bash
# Uninstall 3.12
# Install 3.11.15 (recommended)
# Then run setup again
bash setup_all.sh
```

---

## Model Download Capabilities

### Current Status: ⚠️ Semi-Automated

The scripts do NOT automatically download models. Instead, they:

1. **Check** if models exist locally
2. **Prepare** directories where models should go
3. **Attempt** to download IF `HF_TOKEN` is set
4. **Guide** you to download manually if needed

### How to Enable Auto-Download

**Set HuggingFace Token**:
```bash
export HF_TOKEN=hf_your_token_here
bash download_checkpoints.sh
```

**What gets downloaded**:
- ✅ Attempts to fetch DiT checkpoint (28-30 GB)
- ✅ Attempts to fetch VAE checkpoint

**Limitations**:
- Requires HuggingFace repo IDs to be configured
- Large downloads may fail on unstable connections
- Better to download manually on shared infrastructure

### Manual Download

If `HF_TOKEN` is not set or download fails:

1. **DiT Checkpoint**:
   ```
   Path: deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/
   File: joyai_video_edit_dit_0804.pth
   Size: 28-30 GB
   Source: [Your internal storage or HuggingFace]
   ```

2. **VAE Checkpoint**:
   ```
   Path: deploy/deps/checkpoints/JoyAI-Video-Edit/vae/
   Files: config.json, diffusion_pytorch_model.bin
   Source: xvideo_xvae-released-ckpt (HuggingFace)
   ```

3. **Verify after download**:
   ```bash
   bash check_environment.sh
   ```

---

## Script Details

### `setup_all.sh` — Complete Setup

**Steps**:
1. ✅ **Enforce Python 3.10/3.11** (exit if wrong version)
2. ✅ Run `setup_venv.sh` (create venv + dependencies)
3. ✅ Run `check_environment.sh` (verify PyTorch/GPU)
4. ✅ Run `download_checkpoints.sh` (check + attempt download)

**Output**:
```
✓ Python version OK (3.11.15)
✓ Virtual environment activated
✓ PyTorch installed with CUDA
✓ Checkpoint directories created
⚠ DiT/VAE not found (manual download needed)
```

### `setup_venv.sh` — Virtual Environment

**Creates**:
- `.venv/` directory
- Installs PyTorch cu118 (matches torch cu130 on your 5090)
- Installs: transformers, diffusers, opencv, imageio, tensorrt, etc.

**Time**: 5-10 minutes

### `download_checkpoints.sh` — Model Management

**Does**:
1. Creates checkpoint directories
2. Checks if models exist locally
3. If `HF_TOKEN` set: attempts download from HuggingFace
4. If missing: shows where to place files manually

**Output**:
```
[2/3] Checking for existing checkpoints...
  ✅ DiT checkpoint found (28.5 GB)
  ✅ VAE checkpoint found
```

OR:

```
  ⚠ DiT checkpoint NOT found
  ⚠ To download automatically:
    export HF_TOKEN=hf_xxxx
    bash download_checkpoints.sh
```

### `check_environment.sh` — Verification

**Checks**:
1. Python version
2. PyTorch installation
3. CUDA availability
4. GPU memory
5. Checkpoint locations
6. Inference script availability

---

## Environment Variables

### `HF_TOKEN` — HuggingFace Authentication

Required for automatic model downloads:

```bash
# Export in your shell
export HF_TOKEN=hf_your_token_here

# Or set in .bashrc/.zshrc for persistence
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
source ~/.bashrc
```

### Other Useful Variables

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable async (Windows)
export HF_DEACTIVATE_ASYNC_LOAD=1

# Increase compute threads (needed on Windows)
export TORCHINDUCTOR_COMPILE_THREADS=1
```

---

## Troubleshooting

### "Python 3.12+ not supported"

```bash
# Check your Python version
python3 --version

# If 3.12:
# 1. Install Python 3.11
# 2. Create new venv with 3.11
# 3. Run setup_all.sh again
```

### "Download failed"

```bash
# Verify HF_TOKEN is set
echo $HF_TOKEN

# If empty:
export HF_TOKEN=your_actual_token

# Try download again
bash download_checkpoints.sh

# If still fails, download manually
```

### "Checkpoints not found after setup"

```bash
# This is normal if HF_TOKEN not set or repos not configured
# Manually download and place at:
# deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth
# deploy/deps/checkpoints/JoyAI-Video-Edit/vae/config.json

# Then verify
bash check_environment.sh
```

---

## Quick Setup Summary

### With HuggingFace Token (Auto-Download)
```bash
export HF_TOKEN=hf_your_token
bash setup_all.sh  # Downloads models automatically
bash run_inference.sh  # Ready to use
```

### Without Token (Manual Download)
```bash
bash setup_all.sh  # Creates directories, checks for models
# (Script tells you where to put models)
# (Download models manually)
bash check_environment.sh  # Verify
bash run_inference.sh  # Ready to use
```

### Force Python Version
The scripts automatically enforce 3.10/3.11. If you have:
- 3.9 or earlier → install 3.11
- 3.12+ → uninstall and install 3.11

---

## Files

Updated scripts in `C:\workspace\world\JoyAI-Video-Edit\`:
- ✅ `setup_all.sh` — Now enforces Python 3.10/3.11
- ✅ `download_checkpoints.sh` — Now attempts auto-download with HF_TOKEN
- ✅ `setup_venv.sh` — Unchanged
- ✅ `check_environment.sh` — Unchanged
