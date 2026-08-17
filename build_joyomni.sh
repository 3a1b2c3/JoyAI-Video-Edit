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

# Create/activate venv
cd "$PROJ"
echo "Working directory: $(pwd)"
echo ""

# Remove old venv if corrupted
if [ -d ".venv/Lib" ]; then
    echo "[0/5] Cleaning corrupted venv..."
    rm -rf .venv
fi

# Create fresh venv
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating venv..."
    python3 -m venv .venv || {
        echo "ERROR: venv creation failed"
        exit 1
    }
fi

echo "[1/5] Setting up venv paths..."
export VIRTUAL_ENV="$PROJ/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages:${PYTHONPATH:-}"
echo "OK"

echo "[1b/5] Installing packages..."
$VIRTUAL_ENV/bin/python -m pip install --upgrade pip setuptools wheel 2>&1 | tail -1
echo "  Installing torch 2.4.0+cu124..."
$VIRTUAL_ENV/bin/python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
$VIRTUAL_ENV/bin/python -m pip install -r deploy/requirements.txt 2>&1 | tail -1
echo "OK"

echo ""
echo "[2/5] Verifying environment..."
$VIRTUAL_ENV/bin/python --version
$VIRTUAL_ENV/bin/pip --version
echo "OK"

echo ""
echo "[3/5] Cleaning previous build..."
cd "$PROJ/deploy/joyomni_ops"
rm -rf build joyomni_ops/_C*.so 2>/dev/null || true
echo "OK"

echo ""
echo "[4/5] Building joyomni_ops..."
export JOYOMNI_OPS_NO_FP8=1
$VIRTUAL_ENV/bin/python setup.py build_ext --inplace

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
