#!/bin/bash
# Run DiT inference on horde with joyomni_ops (fused CUDA operations)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup Python environment
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON"

# Setup PYTHONPATH for joyomni_ops
export PYTHONPATH="$SCRIPT_DIR/deploy:$PYTHONPATH"

# Setup LD_LIBRARY_PATH for CUDA and PyTorch libraries (joyomni_ops dependencies)
TORCH_LIB=$($PYTHON -c "import torch; print(torch.__path__[0])")/lib
CUDA_LIB="/usr/local/cuda-12.4/lib64"
export LD_LIBRARY_PATH="$CUDA_LIB:$TORCH_LIB:$LD_LIBRARY_PATH"

echo "Environment:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $CUDA_LIB:$TORCH_LIB"
echo ""

# Parse arguments (handle spaces in filenames)
VIDEO="${1:-assets/Recording 2026-08-12 205529.mp4}"
OUTPUT="${2:-outputs/dit_output.mp4}"
REF_IMAGE="${3:-assets/image.png}"
FRAMES="${4:-1}"
HEIGHT="${5:-256}"
WIDTH="${6:-256}"
STEPS="${7:-1}"

echo "Configuration:"
echo "  Video:      $VIDEO"
echo "  Output:     $OUTPUT"
echo "  Style:      $REF_IMAGE"
echo "  Frames:     $FRAMES"
echo "  Resolution: ${HEIGHT}x${WIDTH}"
echo "  Steps:      $STEPS"
echo ""

# Verify joyomni_ops is available
echo "Checking joyomni_ops..."
$PYTHON -c "from joyomni_ops import fused_norm_scale_shift; print('✅ joyomni_ops available')" || {
    echo "❌ joyomni_ops not found. Rebuild with:"
    echo "   cd deploy/joyomni_ops && set JOYOMNI_OPS_NO_FP8=1 && python setup.py build_ext --inplace"
    exit 1
}
echo ""

# Run inference
echo "Running inference..."
$PYTHON run_inference_lowmem.py \
  --video "$VIDEO" \
  --out "$OUTPUT" \
  --ref-image "$REF_IMAGE" \
  --frames "$FRAMES" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --steps "$STEPS"

echo ""
echo "✅ Inference complete!"
echo "Output: $OUTPUT"
