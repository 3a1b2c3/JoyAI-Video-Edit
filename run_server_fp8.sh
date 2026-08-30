#!/bin/bash
# Start JoyAI-Video-Edit server with FP8 image/text paths enabled.
# Companion to run_server_bf16.sh, which disables FP8_IMG/FP8_TXT as a
# compatibility workaround for a joyomni_ops build without FP8 support
# (DEPLOYMENT.md "If you can't provide CUDA >= 12.8 (or cutlass)..."). This
# script is for the opposite case: joyomni_ops WAS built with FP8 (the
# default build, no JOYOMNI_OPS_NO_FP8=1) -- enabling FP8_IMG/FP8_TXT here
# uses the faster FP8 GEMM kernels for the image/text encoder Linears
# instead of falling back to bf16. Low-VRAM mode is decided by
# run_server_best.sh's GPU auto-detection, not forced here.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/server.log"

{
  echo "=== Server startup: $(date) ==="
  echo "Script dir: $SCRIPT_DIR"
  echo "Log: $LOG_FILE"
  echo

  echo "Setting environment..."
  export JOYOMNI_FP8_IMG=1
  export JOYOMNI_FP8_TXT=1
  # JOYOMNI_LOW_VRAM is intentionally left unset here -- run_server_best.sh
  # (invoked below) auto-detects the GPU and picks the right value per
  # DEPLOYMENT.md (e.g. 0 on >48 GiB cards, 1 on <=48 GiB). Still
  # overridable by exporting JOYOMNI_LOW_VRAM before calling this script.
  echo "JOYOMNI_FP8_IMG=$JOYOMNI_FP8_IMG"
  echo "JOYOMNI_FP8_TXT=$JOYOMNI_FP8_TXT"
  echo

  echo "Verifying joyomni_ops has FP8 support..."
  python - <<'PY'
import sys
try:
    import joyomni_ops
except ImportError as exc:
    print(f"joyomni_ops: MISSING -> {exc!r}")
    print("Build it first (see DEPLOYMENT.md), or use run_server_bf16.sh instead.")
    sys.exit(1)
if not joyomni_ops.has_fp8():
    print("joyomni_ops is built WITHOUT FP8 support (JOYOMNI_OPS_NO_FP8=1 build).")
    print("Rebuild with FP8 support, or use run_server_bf16.sh instead (which disables FP8_IMG/FP8_TXT).")
    sys.exit(1)
print("joyomni_ops: OK, has_fp8=True")
PY
  echo

  echo "Starting run_server_best.sh..."
  bash run_server_best.sh "$@"
} | tee -a "$LOG_FILE" 2>&1
