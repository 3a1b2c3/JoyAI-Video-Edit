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

# Step 1: Install dependencies
echo "Step 1: Installing dependencies via pip..."
echo "  Installing PyTorch with CUDA..."

$PYTHON -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1 || true

# Install PyTorch with CUDA support
$PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | grep -E "Successfully|Collecting|ERROR" | head -10

echo "  Installing additional packages..."
$PYTHON -m pip install -q \
    transformers \
    diffusers \
    opencv-python \
    imageio \
    imageio-ffmpeg \
    numpy \
    tqdm \
    loguru \
    tensorrt \
    huggingface_hub

echo "  ✅ All dependencies installed"
echo ""

# Step 2: Check environment
echo "Step 2: Checking environment..."
bash check_environment.sh
echo ""

# Step 3: Checkpoint setup
echo "Step 3: Verifying checkpoints..."
bash download_checkpoints.sh
echo ""

# Final summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                         Setup Complete!                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Next: Run inference"
echo "  bash run_inference.sh input.mp4 output.mp4"
echo ""
