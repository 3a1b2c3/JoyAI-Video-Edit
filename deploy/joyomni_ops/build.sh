#!/bin/bash
# Rebuild joyomni_ops with FP8 support enabled

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=========================================="
echo "Building joyomni_ops with FP8 support"
echo "=========================================="
echo ""

# Check CUDA is available
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Ensure CUDA is in PATH."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
echo "[1/3] CUDA version: $CUDA_VERSION"
echo ""

# Clone cutlass if not present
if [ ! -d "cutlass" ]; then
    echo "[2/3] Cloning NVIDIA cutlass..."
    git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass
    echo "✓ Cutlass cloned"
else
    echo "[2/3] Cutlass already present at $HERE/cutlass"
fi
echo ""

# Build with FP8
echo "[3/3] Building joyomni_ops (fp8 enabled)..."
export JOYOMNI_OPS_CUTLASS_DIR="$HERE/cutlass"
unset JOYOMNI_OPS_NO_FP8 2>/dev/null || true

python -m pip install -e . --no-build-isolation --force-reinstall

echo ""
echo "=========================================="
echo "✅ Build complete!"
echo "=========================================="
echo ""
echo "Functions available:"
echo "  - sgl_per_token_quant_fp8"
echo "  - fp8_scaled_mm"
echo ""
echo "Verify:"
echo "  python -c \"import joyomni_ops; print('fp8_scaled_mm' in dir(joyomni_ops._C))\""
