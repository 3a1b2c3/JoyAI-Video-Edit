#!/bin/bash
# Build joyomni_ops in WSL

set -euo pipefail

PROJ="$HOME/JoyAI-Video-Edit"
if [ ! -d "$PROJ" ]; then
    PROJ="/mnt/c/workspace/world/JoyAI-Video-Edit"
fi

echo "=========================================="
echo "WSL Build: joyomni_ops"
echo "=========================================="
echo ""

# Create/activate venv (use explicit Linux Python)
cd "$PROJ"
echo "Working directory: $(pwd)"
echo ""

# Ensure we're using Linux Python in WSL, not Windows Python
LINUX_PYTHON="/usr/bin/python3"
if [ ! -f "$LINUX_PYTHON" ]; then
    LINUX_PYTHON=$(which python3)
    if [ -z "$LINUX_PYTHON" ]; then
        echo "ERROR: Python3 not found in WSL"
        exit 1
    fi
fi
echo "Using Python: $LINUX_PYTHON"
echo ""

# Remove old Windows-style venv if it exists
if [ -d ".venv/Lib" ] && [ ! -d ".venv/lib" ]; then
    echo "Removing Windows venv structure..."
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo "[1/5] Creating venv..."
    if ! $LINUX_PYTHON -m venv .venv; then
        echo "ERROR: Failed to create venv"
        exit 1
    fi
    echo "OK"
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: venv structure incorrect"
    echo "Contents of .venv:"
    ls -la .venv/ || true
    echo "Contents of .venv/bin:"
    ls -la .venv/bin/ || true
    exit 1
fi

echo "[1/5] Activating venv..."
source .venv/bin/activate
echo "OK (Python: $(python --version))"

echo "[1b/5] Installing packages..."
pip install --upgrade pip setuptools wheel >/dev/null 2>&1
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124 >/dev/null 2>&1
pip install -r deploy/requirements.txt >/dev/null 2>&1
echo "OK"

echo ""
echo "[2/5] Verifying environment..."
python --version
pip --version
echo "OK"

echo ""
echo "[3/5] Cleaning previous build..."
cd "$PROJ/deploy/joyomni_ops"
rm -rf build joyomni_ops/_C*.so 2>/dev/null || true
echo "OK"

echo ""
echo "[4/5] Building joyomni_ops..."
export JOYOMNI_OPS_NO_FP8=1
python setup.py build_ext --inplace

echo ""
echo "[5/5] Verifying build..."
if ls joyomni_ops/_C*.so >/dev/null 2>&1; then
    echo "✅ Build successful:"
    ls -lh joyomni_ops/_C*.so
else
    echo "❌ Build failed - .so not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ joyomni_ops built successfully"
echo "=========================================="
echo ""
echo "Next: Run inference"
echo "  bash $PROJ/run.sh"
echo ""
