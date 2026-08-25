#!/bin/bash
# Automated FP8 build and verification for joyomni_ops
# Handles cleanup, environment setup, and detailed diagnostics if build fails

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOYOMNI_OPS_DIR="$SCRIPT_DIR/deploy/joyomni_ops"
LOG_FILE="$JOYOMNI_OPS_DIR/fp8_build.log"

echo "=========================================="
echo "JoyOmni Ops - FP8 Build & Verification"
echo "=========================================="
echo ""

# Helper functions
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

error() {
    echo "ERROR: $@" | tee -a "$LOG_FILE"
    exit 1
}

# 1. Ensure we're in the right place
if [ ! -f "$JOYOMNI_OPS_DIR/setup.py" ]; then
    error "setup.py not found at $JOYOMNI_OPS_DIR"
fi

log "Build log: $LOG_FILE"
: > "$LOG_FILE"  # Fresh log

# 2. Find Python interpreter
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    error "Python interpreter '$PYTHON' not found. Set PYTHON=/path/to/python or activate your venv first."
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1)
log "[1/5] Python: $("$PYTHON" -c 'import sys; print(sys.executable)') ($PYTHON_VERSION)"

# 3. Check prerequisites
log "[2/5] Checking prerequisites..."

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    error "nvcc not found. Install CUDA and add /usr/local/cuda-*/bin to PATH"
fi

CUDA_VERSION=$(nvcc --version 2>&1 | grep -oP 'release \K[0-9.]+' || echo "unknown")
log "  ✓ CUDA $CUDA_VERSION"

# Check PyTorch
if ! "$PYTHON" -c "import torch; print(f'  ✓ PyTorch {torch.__version__} (CUDA: {torch.version.cuda})')" 2>&1 | tee -a "$LOG_FILE"; then
    error "PyTorch not installed or not CUDA-enabled"
fi

# Check build tools
for tool in gcc g++ cmake ninja make; do
    if command -v "$tool" &>/dev/null; then
        log "  ✓ $tool found"
    else
        log "  ⚠ $tool NOT found (may be needed)"
    fi
done
echo "" | tee -a "$LOG_FILE"

# 4. Clean and prepare
log "[3/5] Cleaning environment..."
cd "$JOYOMNI_OPS_DIR"

# Clear any stale NO_FP8 flag
unset JOYOMNI_OPS_NO_FP8 2>/dev/null || true
log "  ✓ Cleared JOYOMNI_OPS_NO_FP8"

# Remove old build artifacts
if [ -d "build" ] || [ -d "dist" ]; then
    rm -rf build dist *.egg-info 2>/dev/null || true
    log "  ✓ Removed old build artifacts"
fi

# Ensure cutlass is present
if [ ! -d "cutlass" ]; then
    log "  Cloning NVIDIA cutlass (required for FP8)..."
    if git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass 2>&1 | tee -a "$LOG_FILE"; then
        log "  ✓ Cutlass cloned"
    else
        error "Failed to clone cutlass"
    fi
else
    if [ ! -f "cutlass/include/cutlass/cutlass.h" ]; then
        log "  Cutlass directory broken, re-cloning..."
        rm -rf cutlass
        if git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass 2>&1 | tee -a "$LOG_FILE"; then
            log "  ✓ Cutlass re-cloned"
        else
            error "Failed to clone cutlass"
        fi
    else
        log "  ✓ Cutlass already present"
    fi
fi
echo "" | tee -a "$LOG_FILE"

# 5. Build
log "[4/5] Building joyomni_ops with FP8..."
export JOYOMNI_OPS_CUTLASS_DIR="$JOYOMNI_OPS_DIR/cutlass"

if VERBOSE=1 "$PYTHON" -m pip install -e . --no-build-isolation --force-reinstall -v 2>&1 | tee -a "$LOG_FILE"; then
    log "  ✓ Build completed"
else
    error "Build failed (see $LOG_FILE for details)"
fi
echo "" | tee -a "$LOG_FILE"

# 6. Verify FP8 is available
log "[5/5] Verifying FP8 functions..."

VERIFY_SCRIPT=$(cat <<'EOCHECK'
import sys
try:
    from joyomni_ops import has_fp8, fp8_scaled_mm, sgl_per_token_quant_fp8

    # Verify it's a real install
    import joyomni_ops
    if not getattr(joyomni_ops, '__file__', None):
        print("ERROR: joyomni_ops is a namespace package (not a real install)")
        sys.exit(1)

    # Check FP8 is compiled in
    if not has_fp8():
        print("ERROR: FP8 functions exist but not registered with torch.ops (not compiled in)")
        sys.exit(1)

    print(f"✓ joyomni_ops real install: {joyomni_ops.__file__}")
    print(f"✓ FP8 available: has_fp8() = {has_fp8()}")
    print(f"✓ fp8_scaled_mm available")
    print(f"✓ sgl_per_token_quant_fp8 available")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOCHECK
)

if "$PYTHON" -c "$VERIFY_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then
    log ""
    log "=========================================="
    log "✅ SUCCESS - FP8 Build Complete!"
    log "=========================================="
    log ""
    log "You can now run inference with FP8 support enabled:"
    log "  export JOYOMNI_FP8_IMG=1"
    log "  export JOYOMNI_FP8_TXT=1"
    log ""
    exit 0
else
    log ""
    log "=========================================="
    log "❌ VERIFICATION FAILED"
    log "=========================================="
    log ""
    log "FP8 functions not available. Common causes:"
    log "  1. CUDA toolchain issue (nvcc not found)"
    log "  2. Cutlass headers missing (check cutlass/include/cutlass/)"
    log "  3. Build silently skipped FP8 (JOYOMNI_OPS_NO_FP8 still set)"
    log ""
    log "Fallback (build without FP8):"
    log "  JOYOMNI_OPS_NO_FP8=1 $PYTHON -m pip install -e . --no-build-isolation --force-reinstall"
    log ""
    log "Full diagnostics in: $LOG_FILE"
    exit 1
fi
