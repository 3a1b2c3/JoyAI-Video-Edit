#!/bin/bash
# Standalone inference script for JoyAI-Video-Edit
# Usage: bash infer_standalone.sh <prompt> <input_video> [output_video]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Arguments
PROMPT="${1:-comicbook}"
INPUT_VIDEO="${2:?Error: input_video required}"
OUTPUT_VIDEO="${3:-output_$(date +%s).mp4}"

# Optional parameters (can be overridden)
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-840}"
FPS="${FPS:-24}"
STEPS="${STEPS:-15}"
DEVICE="${DEVICE:-cuda:0}"

echo ""
echo "================================================================================"
echo "JoyAI-Video-Edit Standalone Inference"
echo "================================================================================"
echo ""
echo "Input video:  $INPUT_VIDEO"
echo "Prompt:       $PROMPT"
echo "Output:       $OUTPUT_VIDEO"
echo "Resolution:   ${WIDTH}x${HEIGHT} @ ${FPS} FPS"
echo "Steps:        $STEPS"
echo "Device:       $DEVICE"
echo ""

# Verify input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "ERROR: Input video not found: $INPUT_VIDEO"
    exit 1
fi

echo "Running inference..."
echo ""

python deploy/infer_standalone.py \
    --prompt "$PROMPT" \
    --input_video "$INPUT_VIDEO" \
    --output "$OUTPUT_VIDEO" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --fps "$FPS" \
    --num_inference_steps "$STEPS" \
    --device "$DEVICE"

echo ""
echo "================================================================================"
echo "Done!"
echo "Output saved to: $OUTPUT_VIDEO"
echo "================================================================================"
echo ""
