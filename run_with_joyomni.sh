#!/bin/bash
# Run inference with joyomni_ops

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add deploy dir to Python path so joyomni_ops is found
export PYTHONPATH="$SCRIPT_DIR/deploy:$PYTHONPATH"

# Add torch libraries to LD_LIBRARY_PATH (joyomni_ops dependency)
TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

# Parse arguments (default to minimal settings)
VIDEO="${1:-assets/Recording 2026-08-12 205529.mp4}"
OUTPUT="${2:-outputs/dit_output.mp4}"
REF_IMAGE="${3:-assets/image.png}"
FRAMES="${4:-1}"
HEIGHT="${5:-256}"
WIDTH="${6:-256}"
STEPS="${7:-1}"

echo "=========================================="
echo "DiT Inference with joyomni_ops"
echo "=========================================="
echo "Video:      $VIDEO"
echo "Output:     $OUTPUT"
echo "Style:      $REF_IMAGE"
echo "Frames:     $FRAMES"
echo "Resolution: ${HEIGHT}x${WIDTH}"
echo "Steps:      $STEPS"
echo ""

# Verify joyomni_ops is available
echo "Checking joyomni_ops..."
python3 -c "from joyomni_ops import fused_norm_scale_shift; print('✅ joyomni_ops loaded')" || {
    echo "❌ joyomni_ops not found"
    exit 1
}

echo ""
echo "Running inference..."
python3 run_inference_lowmem.py \
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
