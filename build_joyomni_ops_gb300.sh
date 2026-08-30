#!/bin/bash
# Convenience wrapper around build_joyomni_ops.sh for GB300/GB200: points
# JOYOMNI_OPS_CUTLASS_DIR at the cutlass checkout (default ~/cutlass, override
# with CUTLASS_DIR=... if yours lives elsewhere), logs full output to a
# timestamped file, and greps out the lines that actually matter so you don't
# have to scroll past ninja/nvcc noise to see whether it worked.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass}"
LOG_FILE="$SCRIPT_DIR/build_joyomni_ops_gb300_$(date +%Y%m%d_%H%M%S).log"

if [ ! -d "$CUTLASS_DIR" ]; then
  echo "ERROR: cutlass checkout not found at $CUTLASS_DIR"
  echo "Clone it first:"
  echo "  git clone https://github.com/NVIDIA/cutlass.git $CUTLASS_DIR"
  echo "  cd $CUTLASS_DIR && git checkout dcf215af"
  echo "(or set CUTLASS_DIR=/path/to/cutlass if you already have one elsewhere)"
  exit 1
fi

echo "cutlass: $CUTLASS_DIR ($(cd "$CUTLASS_DIR" && git log -1 --oneline 2>/dev/null || echo 'not a git repo'))"
echo "Log:     $LOG_FILE"
echo

JOYOMNI_OPS_CUTLASS_DIR="$CUTLASS_DIR" bash build_joyomni_ops.sh 2>&1 | tee "$LOG_FILE"
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
