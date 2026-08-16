#!/bin/bash
# Rebuild joyomni_ops from scratch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/deploy/joyomni_ops"

echo "=========================================="
echo "Rebuilding joyomni_ops"
echo "=========================================="

# Clean previous build
echo "[1/5] Cleaning previous build..."
rm -rf build joyomni_ops/_C.cpython-*.so joyomni_ops.egg-info
python3 setup.py clean --all 2>/dev/null || true

# Set environment
echo "[2/5] Setting CUDA environment..."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH=$CUDA_HOME/bin:$PATH
export JOYOMNI_OPS_NO_FP8=1

# Verify CUDA
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc: $(which nvcc)"
nvcc --version | head -1

# Build
echo "[3/5] Building extension..."
python3 setup.py build_ext --inplace 2>&1 | tail -20

# Verify .so
echo "[4/5] Checking build output..."
SO_FILE=$(find joyomni_ops -name '_C.cpython-*.so' -type f)
if [ -f "$SO_FILE" ]; then
    SIZE=$(du -h "$SO_FILE" | cut -f1)
    echo "  ✅ Built: $SO_FILE ($SIZE)"
else
    echo "  ❌ Build failed - .so not found"
    exit 1
fi

# Test import
echo "[5/5] Testing import..."
export PYTHONPATH=~/JoyAI-Video-Edit/deploy:$PYTHONPATH
TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$TORCH_LIB:$LD_LIBRARY_PATH

python3 << 'PYEOF'
try:
    from joyomni_ops._C import fused_norm_scale_shift
    print("✅ joyomni_ops._C imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)
PYEOF

echo ""
echo "=========================================="
echo "✅ joyomni_ops rebuilt successfully!"
echo "=========================================="
echo ""
echo "Next: Run inference"
echo "  bash run_inference.sh"
