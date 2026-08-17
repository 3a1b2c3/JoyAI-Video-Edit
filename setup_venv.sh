#!/bin/bash
# Setup virtual environment for JoyAI-Video-Edit on WSL or Linux

set -euo pipefail

echo "=========================================="
echo "Setting up venv"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
PYTHON=$(command -v python3.10 || command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Python: $($PYTHON --version)"
echo ""

# Create venv
echo "[1/4] Creating venv..."
$PYTHON -m venv .venv
source .venv/bin/activate
echo "OK"

echo ""
echo "[2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "OK"

echo ""
echo "[3/4] Installing PyTorch 2.9.1+cu124..."
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124
echo "OK"

echo ""
echo "[4/4] Installing requirements..."
pip install -r deploy/requirements.txt
echo "OK"

echo ""
echo "=========================================="
echo "✅ venv ready"
echo "=========================================="
echo ""
echo "Activate venv:"
echo "  source .venv/bin/activate"
echo ""
echo "Build joyomni_ops:"
echo "  cd deploy/joyomni_ops"
echo "  export JOYOMNI_OPS_NO_FP8=1"
echo "  python setup.py build_ext --inplace"
echo ""
echo "Run inference:"
echo "  bash run.sh"
echo ""
