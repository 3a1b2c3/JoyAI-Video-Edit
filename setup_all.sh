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

# Step 1: Install dependencies with exact versions
echo "Step 1: Installing dependencies with exact versions..."

$PYTHON -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1 || true

# Install from requirements.txt with exact versions
if [ -f "deploy/requirements.txt" ]; then
    echo "  Installing from deploy/requirements.txt..."
    $PYTHON -m pip install -q -r deploy/requirements.txt
    echo "  ✅ All dependencies installed with exact versions"
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
