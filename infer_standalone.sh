#!/bin/bash
# Standalone inference (no server) with best GPU settings
# Usage: bash infer_standalone.sh <prompt> <input_video> [output_path]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure joyomni_ops (editable install) and the xvideo package are both resolvable
# regardless of which interpreter/cwd context ends up running this -- same pattern
# deploy/run_server.sh uses for xvideo (PYTHONPATH="$HERE"), extended to also cover
# joyomni_ops explicitly rather than relying solely on its site-packages .pth finder.
export PYTHONPATH="$SCRIPT_DIR/deploy/joyomni_ops:$SCRIPT_DIR/deploy:${PYTHONPATH:-}"

# Argument validation
if [ $# -lt 2 ]; then
    echo "Usage: bash infer_standalone.sh <prompt> <input_video> [output_path] [--image style.jpg]"
    echo ""
    echo "  prompt          : text prompt for style/content"
    echo "  input_video     : path to input MP4/WebM"
    echo "  output_path     : where to save gen.mp4 (default: ./output_$(date +%s).mp4)"
    echo ""
    echo "Optional:"
    echo "  --image <path>  : reference/style image for conditioning"
    exit 1
fi

PROMPT="$1"
INPUT_VIDEO="$2"
OUTPUT_PATH="${3:-./output_$(date +%s).mp4}"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "ERROR: input video not found: $INPUT_VIDEO"
    exit 1
fi

# Set checkpoint paths (from deploy/deps/checkpoints)
export JOYOMNI_DIT_CHECKPOINT="$SCRIPT_DIR/deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"
export JOYOMNI_VAE_CHECKPOINT="$SCRIPT_DIR/deploy/deps/checkpoints/JoyAI-Video-Edit/vae"
export JOYOMNI_TEXT_ENCODER_CHECKPOINT="$SCRIPT_DIR/deploy/deps/checkpoints/MiMo-VL-7B-RL-2508"

if [ ! -f "$JOYOMNI_DIT_CHECKPOINT" ]; then
    echo "ERROR: DiT checkpoint not found: $JOYOMNI_DIT_CHECKPOINT"
    echo "       Run: bash download_models.sh"
    exit 1
fi

# GPU detection (same as run_server_best.sh)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
GPU_MEM_GIB=$(( (GPU_MEM_MIB + 512) / 1024 ))

if [ -z "$GPU_NAME" ]; then
  echo "ERROR: nvidia-smi did not report a GPU -- is a driver installed?" >&2
  exit 1
fi

echo "=========================================="
echo "JoyAI Standalone Inference"
echo "=========================================="
echo "GPU: $GPU_NAME (${GPU_MEM_GIB} GiB)"
echo "Prompt: $PROMPT"
echo "Input: $INPUT_VIDEO"
echo "Output: $OUTPUT_PATH"
echo "=========================================="
echo ""

# Set environment per GPU (same as run_server_best.sh)
case "$GPU_NAME" in
  *B200*)
    echo "Profile: NVIDIA B200 -- 720p @ 30 FPS"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_b200}"
    export JOYOMNI_WIDTH="${JOYOMNI_WIDTH:-1248}" JOYOMNI_HEIGHT="${JOYOMNI_HEIGHT:-720}" JOYOMNI_FPS="${JOYOMNI_FPS:-30}"
    ;;
  *"RTX PRO 6000"*)
    echo "Profile: RTX PRO 6000 -- 480p @ 24 FPS"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_pro6000}"
    ;;
  *5090*)
    echo "Profile: RTX 5090 -- 480p @ 24 FPS, low-VRAM"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_rtx5090}"
    export JOYOMNI_SAGE_ATTN="${JOYOMNI_SAGE_ATTN:-1}" JOYOMNI_FP8_FAST_ACCUM="${JOYOMNI_FP8_FAST_ACCUM:-1}" JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-1}"
    ;;
  *A40*)
    echo "Profile: A40 -- 480p @ 24 FPS, low-VRAM"
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_a40}"
    export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-1}"
    ;;
  *)
    if [ "$GPU_MEM_GIB" -le 48 ]; then
      echo "Profile: unrecognized card (${GPU_MEM_GIB} GiB) -- using low-VRAM"
      export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-1}"
    else
      echo "Profile: unrecognized card (${GPU_MEM_GIB} GiB) -- using defaults"
    fi
    export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$SCRIPT_DIR/deploy/deps/cache_default}"
    ;;
esac

mkdir -p "$JOYOMNI_CACHE_ROOT"

# Persist torch.compile/Triton's autotuned kernel choices across runs -- without this,
# every run re-benchmarks every AUTOTUNE convolution/mm from scratch (dozens of them,
# ~1-6s each), which is most of the multi-minute VAE-compile warmup time. Same vars
# deploy/run_server.sh already sets; this script was missing them entirely.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$JOYOMNI_CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$JOYOMNI_CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$JOYOMNI_CACHE_ROOT/nv_compute}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

echo ""
echo "Running inference..."

# Handle optional --image flag
IMAGE_ARG=""
if [ $# -ge 4 ] && [ "$4" = "--image" ] && [ -n "${5:-}" ]; then
    IMAGE_ARG="--image $5"
fi

python deploy/infer_standalone.py \
    --prompt "$PROMPT" \
    --input_video "$INPUT_VIDEO" \
    --output "$OUTPUT_PATH" \
    $IMAGE_ARG

if [ $? -eq 0 ]; then
    echo ""
    echo "Complete!"
    echo "   Output: $OUTPUT_PATH"
else
    echo ""
    echo "Inference failed"
    exit 1
fi
