# Setup Scripts: Use Existing Python (No Installation)

## Key Change
Scripts now **use whatever Python is already available** on your system. No installation required.

---

## How It Works

### 1. Script Finds Python Automatically
The setup scripts search for Python in this order:
```
python3  →  python  →  python.exe
```

Uses the first one found in your PATH.

### 2. Creates Virtual Environment
```bash
bash setup_all.sh
```

Uses the found Python to create `.venv/`:
- No new Python installation
- All dependencies go in `.venv/`
- Original Python unchanged

### 3. Runs Inference
The inference scripts automatically activate `.venv/` and run.

---

## Setup Steps

### Step 1: Check What Python You Have
```bash
# See which Python will be used
which python3
which python
which python.exe

# Or just run
bash check_environment.sh
```

**Expected output**:
```
[1/5] Python:
  Using: /usr/bin/python3
  Version: Python 3.11.15
  ✅ Python version OK
```

### Step 2: Run Setup
```bash
bash setup_all.sh
```

**What happens**:
1. ✅ Finds Python in PATH
2. ✅ Creates `.venv/` with that Python
3. ✅ Installs PyTorch + packages into `.venv/`
4. ✅ Checks for checkpoints
5. ✅ Shows errors/warnings (if any)

**Time**: 5-10 minutes

### Step 3: Place Checkpoints
```bash
# Script will tell you where to put:
# - joyai_video_edit_dit_0804.pth (28-30 GB)
# - vae/ directory with config.json + model file
```

### Step 4: Run Inference
```bash
bash run_inference.sh input.mp4 output.mp4
```

---

## Version Warnings (Not Blocking)

The scripts will **warn** but **not block** if Python version is:
- **Too old** (< 3.10): "⚠ Python 3.10+ recommended"
- **Too new** (> 3.11): "⚠ Python 3.12+ may have issues on Windows"
- **Just right** (3.10-3.11): "✅ Python version OK"

**Example**:
```bash
# If you have Python 3.12
bash setup_all.sh

# Output:
# ⚠ Python 3.12+ may have issues on Windows (you have 3.12)
# (Continues anyway)
```

---

## What Python Versions Work

| Version | Status |
|---------|--------|
| 3.8 | ⚠ Untested (very old) |
| 3.9 | ⚠ Old but might work |
| 3.10 | ✅ Recommended |
| 3.11 | ✅ Recommended |
| 3.12 | ⚠ May have threading issues on Windows |
| 3.13+ | ⚠ Untested |

**Best**: 3.10.x or 3.11.x
**Will still work**: Most 3.9+ versions

---

## Common Scenarios

### Scenario 1: Have Python 3.11 (Ideal)
```bash
python3 --version          # Python 3.11.x
bash setup_all.sh          # ✅ Works perfectly
bash run_inference.sh      # ✅ Ready to go
```

### Scenario 2: Have Python 3.12 (Will Warn, Still Works)
```bash
python --version           # Python 3.12.x
bash setup_all.sh          # ⚠ Warns, but continues
# Setup completes, inference works
```

### Scenario 3: Have Python 3.9 (Will Warn, May Work)
```bash
python --version           # Python 3.9.x
bash setup_all.sh          # ⚠ Warns about old version
# May work, but install 3.10/3.11 if problems
```

### Scenario 4: Multiple Python Versions
```bash
# If you have both python2 and python3
python3 --version         # Python 3.11
python --version          # Python 2.7
bash setup_all.sh         # Uses python3 (found first)
```

---

## Troubleshooting

### "Python not found"
```bash
# Check PATH
echo $PATH

# Try to find Python
which python3
which python
which python.exe

# If none found: Install Python first
# https://www.python.org/downloads/
```

### "ModuleNotFoundError: No module named..."
```bash
# Verify venv is activated
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows cmd
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Reinstall dependencies
pip install --upgrade pip
bash setup_venv.sh
```

### "PyTorch not found on GPU"
```bash
# Check GPU is detected
nvidia-smi

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, CUDA not installed or mismatched
# Install CUDA drivers from https://developer.nvidia.com/cuda-downloads
```

---

## Environment Variables

No special Python setup needed. Optional:

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Reduce memory issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Windows async issues
export HF_DEACTIVATE_ASYNC_LOAD=1
```

---

## Files Updated

All scripts now use system Python:
- ✅ `setup_all.sh` — Finds Python, warns if old/new
- ✅ `setup_venv.sh` — Uses found Python for venv
- ✅ `check_environment.sh` — Reports which Python used
- ✅ `download_checkpoints.sh` — Simplified, no downloads

---

## Quick Reference

```bash
# Find which Python script will use
bash check_environment.sh

# Setup everything (uses found Python)
bash setup_all.sh

# Place checkpoints manually
# deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth
# deploy/deps/checkpoints/JoyAI-Video-Edit/vae/

# Run inference
bash run_inference.sh input.mp4 output.mp4
```

**No Python installation needed.** Just use what's there.
