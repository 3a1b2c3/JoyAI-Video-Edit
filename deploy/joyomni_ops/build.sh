#!/bin/bash
# Rebuild joyomni_ops with FP8 support enabled
#
# FP8 requires:
#   - CUDA >= 12.8 (for Blackwell support)
#   - NVIDIA cutlass library
#   - PyTorch with CUDA support
#
# If build fails or fp8 is not needed, use:
#   JOYOMNI_OPS_NO_FP8=1 python -m pip install -e .

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=========================================="
echo "Building joyomni_ops with FP8 support"
echo "=========================================="
echo ""

# Check CUDA is available
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found."
    echo ""
    echo "Ensure CUDA is installed and in PATH:"
    echo "  source /usr/local/cuda-12.x/setup_env.sh  (or your CUDA path)"
    echo "  export PATH=/usr/local/cuda-12.x/bin:\$PATH"
    echo ""
    echo "Alternatively, build without FP8:"
    echo "  JOYOMNI_OPS_NO_FP8=1 python -m pip install -e ."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+' || echo "unknown")
echo "[1/4] CUDA version: $CUDA_VERSION"
if [[ "$CUDA_VERSION" < "12.8" ]]; then
    echo "⚠️  CUDA < 12.8 — FP8 will work but without native Blackwell (sm_120) SASS."
    echo "    (Uses sm_90 PTX JIT instead; consider upgrading for full performance.)"
fi
echo ""

# Clone cutlass if not present
echo "[2/4] Checking NVIDIA cutlass..."
if [ ! -d "cutlass" ]; then
    echo "  Cloning cutlass (required for fp8_gemm.cu)..."
    git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass
    if [ ! -d "cutlass/include" ]; then
        echo "ERROR: cutlass clone failed"
        exit 1
    fi
    echo "  ✓ Cloned"
else
    if [ ! -f "cutlass/include/cutlass/cutlass.h" ]; then
        echo "ERROR: cutlass present but broken (missing cutlass.h)"
        echo "  Try: rm -rf cutlass && bash build.sh"
        exit 1
    fi
    echo "  ✓ Already present"
fi
echo ""

# Clean old build
echo "[3/4] Cleaning old build artifacts..."
rm -rf build dist *.egg-info
echo "  ✓ Cleaned"
echo ""

# Build with FP8
echo "[4/4] Building joyomni_ops (fp8 enabled)..."
export JOYOMNI_OPS_CUTLASS_DIR="$HERE/cutlass"
unset JOYOMNI_OPS_NO_FP8 2>/dev/null || true

if ! python -m pip install -e . --no-build-isolation --force-reinstall 2>&1 | tee build.log; then
    echo ""
    echo "ERROR: Build failed. See build.log for details."
    echo ""
    echo "Common fixes:"
    echo "  1. Check CUDA is in PATH:  nvcc --version"
    echo "  2. Ensure cutlass is valid:  test -f cutlass/include/cutlass/cutlass.h"
    echo "  3. Check pip output (last 50 lines):"
    echo ""
    tail -50 build.log | sed 's/^/    /'
    echo ""
    echo "If FP8 is not needed, build without it:"
    echo "  JOYOMNI_OPS_NO_FP8=1 python -m pip install -e ."
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Build complete!"
echo "=========================================="
echo ""

# Verify FP8 functions are available
echo "Verifying FP8 functions..."
if python -c "from joyomni_ops._C import fp8_scaled_mm, sgl_per_token_quant_fp8; print('  ✓ fp8_scaled_mm available'); print('  ✓ sgl_per_token_quant_fp8 available')" 2>&1; then
    echo ""
    echo "✅ FP8 support verified!"
    exit 0
else
    echo ""
    echo "⚠️  FP8 functions not available after build."
    echo ""
    echo "This may happen if:"
    echo "  - cutlass headers are missing"
    echo "  - CUDA compiler failed silently"
    echo ""
    echo "Troubleshoot:"
    echo "  1. Check build.log: tail -100 build.log"
    echo "  2. Verify cutlass:  ls -la cutlass/include/cutlass/"
    echo "  3. Check CUDA:  nvcc --version && which nvcc"
    echo ""
    echo "Workaround (build without FP8):"
    echo "  JOYOMNI_OPS_NO_FP8=1 python -m pip install -e . --force-reinstall"
    exit 1
fi
