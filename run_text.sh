#!/bin/bash
# Run DiT inference on horde with TEXT PROMPTS only (no style image)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""
echo "DEBUG: System Info"
$PYTHON -c "import sys, torch; print(f'  Python: {sys.version}'); print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')"
echo ""

# Setup PYTHONPATH for joyomni_ops (MUST be first for import to work)
export PYTHONPATH="$SCRIPT_DIR/deploy/joyomni_ops:$SCRIPT_DIR/deploy:${PYTHONPATH:-}"

# Setup LD_LIBRARY_PATH for CUDA and PyTorch
TORCH_LIB=$($PYTHON -c "import torch; print(torch.__path__[0])")/lib
CUDA_LIB="${CUDA_HOME:-/usr/local/cuda-12.4}/lib64"
export LD_LIBRARY_PATH="$CUDA_LIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# Setup checkpoint caching (faster loading on horde)
export TORCH_HOME="$SCRIPT_DIR/.cache/torch"
export HF_HOME="$SCRIPT_DIR/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

# Fix GPU memory fragmentation on 48GB systems near capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $CUDA_LIB:$TORCH_LIB"
echo "  TORCH_HOME: $TORCH_HOME (cached)"
echo "  HF_HOME: $HF_HOME (cached)"
echo ""

# Parse arguments with sensible defaults
VIDEO="${1:-assets/cases/omnidream/mattress.mp4}"
OUTPUT="${2:-$SCRIPT_DIR/outputs/text_output.mp4}"
PROMPT="${3:-A beautiful landscape with mountains and ocean}"
FRAMES="${4:-10}"
HEIGHT="${5:-auto}"
WIDTH="${6:-auto}"
STEPS="${7:-20}"
CFG="${8:-7.5}"

# Resolve full output path upfront
OUTPUT_FULL=$(cd "$(dirname "$SCRIPT_DIR/$OUTPUT")" 2>/dev/null && pwd -P)/$(basename "$OUTPUT") || echo "$SCRIPT_DIR/$OUTPUT"

echo "Configuration:"
echo "  Video:      $VIDEO"
echo "  Output:     $OUTPUT_FULL"
echo "  Prompt:     '$PROMPT'"
echo "  Frames:     $FRAMES"
echo "  Resolution: ${HEIGHT}x${WIDTH}"
echo "  Steps:      $STEPS"
echo "  CFG:        $CFG"
echo ""

# Verify input video exists
if [ ! -f "$VIDEO" ]; then
    echo "❌ ERROR: Input video not found: $VIDEO"
    exit 1
fi
echo "✓ Input video found"

# Verify joyomni_ops (fail hard if missing)
echo "Checking joyomni_ops..."
if ! JOYOMNI_CHECK=$($PYTHON -c "from joyomni_ops import fused_norm_scale_shift; print('✅ joyomni_ops available')" 2>&1); then
    echo ""
    echo "=========================================="
    echo "❌ FATAL: joyomni_ops import failed"
    echo "=========================================="
    echo ""
    echo "Error output:"
    echo "$JOYOMNI_CHECK"
    echo ""
    echo "joyomni_ops._C.cpython-310-x86_64.pyd must be built first."
    echo ""
    echo "Check if .pyd exists:"
    echo "  ls -la deploy/joyomni_ops/joyomni_ops/_C*.pyd"
    echo ""
    echo "If missing, rebuild:"
    echo "  cd deploy/joyomni_ops"
    echo "  set JOYOMNI_OPS_NO_FP8=1"
    echo "  python setup.py build_ext --inplace"
    echo ""
    exit 1
fi
echo "$JOYOMNI_CHECK"
echo ""

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: Cannot create output directory: $OUTPUT_DIR"
    exit 1
fi
echo "✓ Output directory ready: $OUTPUT_DIR"
echo ""

# Run inference via infer_text.py
echo "Running inference (TEXT PROMPTS ONLY)..."
echo "  Frames: $FRAMES"
echo "  Resolution: ${HEIGHT}x${WIDTH}"
echo "  Steps: $STEPS"
echo ""

# Build command for infer_text.py
set -- "$PYTHON" "-u" "infer_text.py" "$VIDEO" "--output" "$OUTPUT" "--prompt" "$PROMPT" "--frames" "$FRAMES" "--steps" "$STEPS" "--cfg" "$CFG"

if [ "$HEIGHT" != "auto" ]; then
    set -- "$@" "--height" "$HEIGHT"
fi
if [ "$WIDTH" != "auto" ]; then
    set -- "$@" "--width" "$WIDTH"
fi

echo "Running: $@"
echo ""

"$@"
INFER_EXIT=$?

if [ $INFER_EXIT -ne 0 ]; then
    echo "Inference failed (exit code: $INFER_EXIT)"
    exit $INFER_EXIT
fi

echo ""
echo "✅ Done!"
echo "Output: $OUTPUT_FULL"
exit 0
