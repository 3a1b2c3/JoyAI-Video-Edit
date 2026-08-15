#!/bin/bash
# Setup Python virtual environment and install dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "Setting Up Virtual Environment"
echo "========================================================================"
echo ""

# Find Python
echo "[1/4] Finding Python..."
PYTHON=""
for py_cmd in python3 python python.exe; do
    if command -v $py_cmd &> /dev/null; then
        PYTHON=$py_cmd
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python not found in PATH"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version | awk '{print $2}')
PYTHON_PATH=$($PYTHON -c "import sys; print(sys.executable)")
echo "  ✅ Found Python $PYTHON_VERSION at $PYTHON_PATH"
echo ""

# Create venv
echo "[2/4] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  ℹ .venv already exists, skipping creation"
else
    $PYTHON -m venv .venv
    echo "  ✅ Virtual environment created"
fi
echo ""

# Activate venv
echo "[3/4] Activating virtual environment..."
source .venv/bin/activate
echo "  ✅ Virtual environment activated"
echo ""

# Install dependencies
echo "[4/4] Installing dependencies..."
echo "  This may take a few minutes..."
echo ""

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install PyTorch with CUDA support (cu118)
echo "  Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1

# Install other dependencies
echo "  Installing additional packages..."
pip install -q \
    transformers \
    diffusers \
    opencv-python \
    imageio \
    imageio-ffmpeg \
    numpy \
    tqdm \
    loguru \
    tensorrt

echo "  ✅ All dependencies installed"
echo ""

echo "========================================================================"
echo "✅ Setup Complete!"
echo "========================================================================"
echo ""
echo "Virtual environment is ready at: .venv"
echo ""
echo "Next steps:"
echo "  1. Download checkpoints:"
echo "     bash download_checkpoints.sh"
echo ""
echo "  2. Check environment:"
echo "     bash check_environment.sh"
echo ""
echo "  3. Run inference:"
echo "     bash run_inference.sh"
echo ""
echo "To manually activate venv in the future:"
echo "  source .venv/bin/activate"
echo ""
