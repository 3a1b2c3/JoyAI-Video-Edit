#!/bin/bash
# Build (or rebuild) the joyomni_ops CUDA extension against the currently
# active venv, current source, and the GPU actually present on this
# machine. There is no other build script -- this wraps DEPLOYMENT.md's
# documented command with the checks that would otherwise catch a stale
# or wrong-target build only after a confusing runtime ImportError deep
# in a model forward pass (see TROUBLESHOOTING.md #9).
#
#   bash build_joyomni_ops.sh              # build with FP8 support
#   JOYOMNI_OPS_NO_FP8=1 bash build_joyomni_ops.sh   # light variant, no
#                                             cutlass/CUDA>=12.8 needed
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== joyomni_ops build: $(date) ==="
echo

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: no venv is active. Run:"
  echo "  source .venv/bin/activate"
  echo "first, then re-run this script."
  exit 1
fi
if [ "$VIRTUAL_ENV" != "$SCRIPT_DIR/.venv" ]; then
  echo "ERROR: active venv is $VIRTUAL_ENV,"
  echo "       expected $SCRIPT_DIR/.venv."
  echo "This usually means another project's venv (e.g. JoyAI-Echo) is still"
  echo "active from earlier in the shell session -- venv activation is shell"
  echo "state and survives 'cd'. Run:"
  echo "  deactivate; cd $SCRIPT_DIR; source .venv/bin/activate"
  echo "then re-run this script."
  exit 1
fi
echo "Venv OK: $VIRTUAL_ENV"

echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
echo "GPU:   $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i 0 2>/dev/null)"
echo

if [ "${JOYOMNI_OPS_NO_FP8:-0}" = "1" ]; then
  echo "Building WITHOUT FP8 support (JOYOMNI_OPS_NO_FP8=1)..."
  pip install --no-build-isolation --force-reinstall --no-deps ./deploy/joyomni_ops
else
  echo "Building WITH FP8 support (requires CUDA >= 12.8 + cutlass)..."
  JOYOMNI_OPS_CUTLASS_DIR="${JOYOMNI_OPS_CUTLASS_DIR:-$SCRIPT_DIR/deploy/tmp/cutlass}" \
    pip install --no-build-isolation --force-reinstall --no-deps ./deploy/joyomni_ops
fi

echo
echo "Verifying the build..."
python - <<'PY'
import sys
try:
    import joyomni_ops
except ImportError as exc:
    print(f"FAILED: joyomni_ops did not import: {exc!r}")
    sys.exit(1)

try:
    from joyomni_ops import fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm
except ImportError as exc:
    print(f"FAILED: joyomni_ops imported but is missing symbols: {exc!r}")
    print("This is the exact failure mode of a stale/wrong-target build --")
    print("see TROUBLESHOOTING.md #9.")
    sys.exit(1)

has_fp8 = joyomni_ops.has_fp8() if hasattr(joyomni_ops, "has_fp8") else None
print(f"OK: joyomni_ops imports cleanly, has_fp8={has_fp8}")
PY
