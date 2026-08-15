#!/bin/bash
# Run DiT inference with custom parameters

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Parse arguments or use defaults
VIDEO="${1:-assets/Recording 2026-08-12 205529.mp4}"
OUTPUT="${2:-outputs/dit_output.mp4}"
FRAMES="${3:-2}"
HEIGHT="${4:-256}"
WIDTH="${5:-256}"
STEPS="${6:-1}"
SEED="${7:-42}"

echo "========================================================================"
echo "DiT Inference (Float32)"
echo "========================================================================"
echo "Input video:  $VIDEO"
echo "Output:       $OUTPUT"
echo "Frames:       $FRAMES"
echo "Resolution:   ${WIDTH}x${HEIGHT}"
echo "Steps:        $STEPS"
echo "Seed:         $SEED"
echo "========================================================================"
echo ""

# Run inference
python run_inference.py \
  --video "$VIDEO" \
  --out "$OUTPUT" \
  --frames "$FRAMES" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --steps "$STEPS" \
  --seed "$SEED"

echo ""
echo "✅ Inference complete!"
if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo "Output: $OUTPUT ($SIZE)"
fi
