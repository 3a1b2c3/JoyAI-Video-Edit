#!/bin/bash
# Launch deploy/run_server.sh with the best-available per-GPU low-VRAM settings
# from DEPLOYMENT.md's section 4 ("Launch") -- detects the GPU via nvidia-smi and
# picks the matching documented profile (falls back to JOYOMNI_LOW_VRAM=1 on any
# other card with <=48 GB, which DEPLOYMENT.md documents as the general rule).
#
#   bash run_server_best.sh                 # auto-detect and launch
#   JOYOMNI_CONDA_ENV=my-env bash run_server_best.sh   # still overridable
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
GPU_MEM_GIB=$(( (GPU_MEM_MIB + 512) / 1024 ))

if [ -z "$GPU_NAME" ]; then
  echo "ERROR: nvidia-smi did not report a GPU -- is a driver installed?" >&2
  exit 1
fi
echo "Detected GPU: $GPU_NAME (${GPU_MEM_GIB} GiB)"

case "$GPU_NAME" in
  *B200*)
    echo "Profile: NVIDIA B200 -- 720p @ 30 FPS (DEPLOYMENT.md §4)"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_b200}"
    export JOYOMNI_WIDTH="${JOYOMNI_WIDTH:-1248}" JOYOMNI_HEIGHT="${JOYOMNI_HEIGHT:-720}" JOYOMNI_FPS="${JOYOMNI_FPS:-30}"
    ;;
  *GB300*|*GB200*)
    # No DEPLOYMENT.md entry for GB300/GB200 -- inferring the B200 profile
    # since it's the same Blackwell generation with equal-or-more memory
    # (GB300 measured 251 GiB here vs B200's ~180 GiB HBM3e), not a
    # confirmed/tested setting. Override JOYOMNI_WIDTH/HEIGHT/FPS if this
    # doesn't hold up in practice.
    echo "Profile: NVIDIA GB300/GB200 -- untested, inferring B200's 720p @ 30 FPS profile (same Blackwell generation, >= memory)"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_gb300}"
    export JOYOMNI_WIDTH="${JOYOMNI_WIDTH:-1248}" JOYOMNI_HEIGHT="${JOYOMNI_HEIGHT:-720}" JOYOMNI_FPS="${JOYOMNI_FPS:-30}"
    ;;
  *"RTX PRO 6000"*)
    echo "Profile: RTX PRO 6000 -- 480p @ 24 FPS (DEPLOYMENT.md §4)"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_pro6000}"
    ;;
  *5090*)
    echo "Profile: RTX 5090 -- 480p @ 24 FPS, FP4 Echo (DEPLOYMENT.md §4)"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_rtx5090}"
    export JOYOMNI_MODEL="${JOYOMNI_MODEL:-echo_fp4}" JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-0}" JOYOMNI_WIDTH="${JOYOMNI_WIDTH:-854}" JOYOMNI_HEIGHT="${JOYOMNI_HEIGHT:-480}" JOYOMNI_FPS="${JOYOMNI_FPS:-24}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    ;;
  *A40*)
    # No explicit DEPLOYMENT.md entry for A40 -- it's a 48 GiB Ampere card, so
    # apply the doc's general rule ("set JOYOMNI_LOW_VRAM=1 on <=48 GB cards").
    # Not an RTX 5090 (no fp32-accum SDPA penalty there), so JOYOMNI_SAGE_ATTN /
    # JOYOMNI_FP8_FAST_ACCUM are left at their defaults (0), unlike the 5090 profile.
    echo "Profile: A40 -- 480p @ 24 FPS, low-VRAM (48 GiB card, DEPLOYMENT.md general rule)"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_a40}"
    export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-1}"
    ;;
  *)
    if [ "$GPU_MEM_GIB" -le 48 ]; then
      echo "Profile: unrecognized card, ${GPU_MEM_GIB} GiB <= 48 GiB -> low-VRAM per DEPLOYMENT.md's general rule"
      export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-1}"
    else
      echo "Profile: unrecognized card, ${GPU_MEM_GIB} GiB > 48 GiB -> defaults (no low-VRAM needed)"
    fi
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_$(echo "$GPU_NAME" | tr -c '[:alnum:]' '_' | tr 'A-Z' 'a-z')}"
    ;;
esac

echo
exec bash deploy/run_server.sh "$@"
