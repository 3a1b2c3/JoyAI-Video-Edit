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

# Create venv if needed
if [ ! -d "$PROJ/.venv" ]; then
    echo "[1/5] Creating venv..."
    cd "$PROJ"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124
    pip install -r deploy/requirements.txt
    echo "OK"
else
    echo "[1/5] Activating venv..."
    source "$PROJ/.venv/bin/activate"
    echo "OK"
fi

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
