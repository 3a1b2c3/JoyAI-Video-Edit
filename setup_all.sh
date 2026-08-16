#!/bin/bash
# Complete setup: install dependencies and verify environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    DiT Inference Complete Setup                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Find Python
echo "Finding Python installation..."
PYTHON=""
for py_cmd in python3 python python.exe; do
    if command -v $py_cmd &> /dev/null; then
        PYTHON=$py_cmd
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ ERROR: Python not found in PATH"
    exit 1
fi

PYTHON_PATH=$($PYTHON -c "import sys; print(sys.executable)")
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

echo "  Using: $PYTHON_PATH"
echo "  Version: Python $PYTHON_VERSION"

# Warn if Python is old/new, but don't block
MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$MAJOR" -ne 3 ]; then
    echo "  ⚠ Python 3 required (you have $MAJOR.x)"
elif [ "$MINOR" -lt 10 ]; then
    echo "  ⚠ Python 3.10+ recommended (you have 3.$MINOR)"
elif [ "$MINOR" -gt 11 ]; then
    echo "  ⚠ Python 3.12+ may have issues on Windows (you have 3.$MINOR)"
else
    echo "  ✅ Python version OK"
fi

echo ""

# Setup CUDA environment
echo "Setting up CUDA environment..."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
echo "  CUDA_HOME: $CUDA_HOME"

echo ""

# Step 1: Install dependencies with exact versions
echo "Step 1: Installing dependencies with exact versions..."

$PYTHON -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1 || true

# Install from requirements.txt with exact versions
if [ -f "deploy/requirements.txt" ]; then
    echo "  Installing from deploy/requirements.txt (CUDA 12.8)..."
    $PYTHON -m pip install -q -r deploy/requirements.txt --index-url https://download.pytorch.org/whl/cu128
    echo "  ✅ All dependencies installed with exact versions (CUDA 12.8)"
else
    echo "  ⚠ deploy/requirements.txt not found, installing defaults..."
    $PYTHON -m pip install -q \
        torch==2.9.1 \
        torchvision==0.24.1 \
        transformers==4.57.1 \
        diffusers==0.36.0 \
        opencv-python \
        imageio \
        imageio-ffmpeg \
        numpy \
        tqdm \
        loguru \
        huggingface_hub
    echo "  ✅ Dependencies installed"
fi
echo ""

# Step 2: Check environment and CUDA
echo "Step 2: Checking environment..."

# Verify CUDA
echo "  Checking PyTorch CUDA support..."
$PYTHON << 'PYTHON_CHECK'
import torch
if torch.cuda.is_available():
    print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ❌ WARNING: CUDA not available - GPU will not be used")
PYTHON_CHECK

bash check_environment.sh
echo ""

# Step 3: Checkpoint setup
echo "Step 3: Verifying checkpoints..."
bash download_checkpoints.sh
echo ""

# Step 4: Build joyomni_ops (CUDA extension)
echo "Step 4: Building joyomni_ops CUDA extension..."
if [ -d "deploy/joyomni_ops" ]; then
    cd deploy/joyomni_ops
    echo "  Building joyomni_ops..."
    $PYTHON setup.py build_ext --inplace
    echo "  Installing joyomni_ops..."
    $PYTHON -m pip install -e . -q
    cd ../..
    echo "  ✅ joyomni_ops built successfully"
else
    echo "  ⚠ deploy/joyomni_ops not found, skipping"
fi
echo ""

# Step 6: Save installed versions
echo "Step 6: Saving installed versions..."
$PYTHON -m pip freeze > installed_versions.txt
echo "  ✅ Saved to: installed_versions.txt"
echo ""

# Final summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                         Setup Complete!                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Environment:"
echo "  Python: $PYTHON_PATH"
echo "  Versions saved: installed_versions.txt"
echo ""

echo "Next: Run inference"
echo "  bash run_inference.sh input.mp4 output.mp4"
echo ""
echo "Or: python run_inference.py --video input.mp4 --out output.mp4 --frames 1 --height 192 --width 192 --steps 1"
echo ""
