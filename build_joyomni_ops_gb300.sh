#!/bin/bash
# Convenience wrapper around build_joyomni_ops.sh for GB300/GB200.
#
# Defaults to the NO_FP8 (light) build: bf16 only, no cutlass checkout
# needed. GB300's FP8 GEMM kernel (cutlass::arch::Sm103) does compile and
# pass isolated correctness tests (see DEPLOYMENT.md's GB300/GB200 section),
# but the CUDA-graph-capture path around it does not, and that's unresolved
# -- so FP8 is opt-in here, not the default, until that's fixed.
#
#   bash build_joyomni_ops_gb300.sh                    # bf16 only (default)
#   JOYOMNI_OPS_FP8=1 bash build_joyomni_ops_gb300.sh  # opt into FP8 (needs cutlass)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/build_joyomni_ops_gb300_$(date +%Y%m%d_%H%M%S).log"
WANT_FP8="${JOYOMNI_OPS_FP8:-0}"

echo "Log: $LOG_FILE"
echo

if [ "$WANT_FP8" = "1" ]; then
  CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass}"
  if [ ! -d "$CUTLASS_DIR" ]; then
    echo "ERROR: cutlass checkout not found at $CUTLASS_DIR"
    echo "Clone it first:"
    echo "  git clone https://github.com/NVIDIA/cutlass.git $CUTLASS_DIR"
    echo "  cd $CUTLASS_DIR && git checkout dcf215af"
    echo "(or set CUTLASS_DIR=/path/to/cutlass if you already have one elsewhere)"
    exit 1
  fi
  echo "cutlass: $CUTLASS_DIR ($(cd "$CUTLASS_DIR" && git log -1 --oneline 2>/dev/null || echo 'not a git repo'))"
  echo "Building WITH FP8 (JOYOMNI_OPS_FP8=1)..."
  echo
  JOYOMNI_OPS_CUTLASS_DIR="$CUTLASS_DIR" bash build_joyomni_ops.sh 2>&1 | tee "$LOG_FILE"
else
  echo "Building WITHOUT FP8 (bf16 only, default -- set JOYOMNI_OPS_FP8=1 to opt in)..."
  echo
  JOYOMNI_OPS_NO_FP8=1 bash build_joyomni_ops.sh 2>&1 | tee "$LOG_FILE"
fi
BUILD_STATUS="${PIPESTATUS[0]}"

echo
echo "=== Summary (full log: $LOG_FILE) ==="
grep -n "error:\|Successfully installed\|Verifying the build\|^OK:\|^FAILED:" "$LOG_FILE"

if [ "$BUILD_STATUS" -ne 0 ]; then
  echo
  echo "Build FAILED (exit $BUILD_STATUS). See errors above, or:"
  echo "  grep -n 'error:' $LOG_FILE"
  exit "$BUILD_STATUS"
fi
