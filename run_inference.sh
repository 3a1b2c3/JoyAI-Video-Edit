#!/bin/bash
# Run DiT inference - no venv required

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use system Python (no venv)
PYTHON=$(command -v python3 || command -v python)

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""

# Parse arguments (handle spaces in filenames)
VIDEO="${1:-assets/Recording 2026-08-12 205529.mp4}"
OUTPUT="${2:-outputs/dit_output.mp4}"
FRAMES="${3:-2}"
HEIGHT="${4:-256}"
WIDTH="${5:-256}"
STEPS="${6:-1}"

# Run inference with properly quoted args
$PYTHON run_inference.py \
  --video "$VIDEO" \
  --out "$OUTPUT" \
  --frames "$FRAMES" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --steps "$STEPS"

echo ""
echo "✅ Inference complete!"
echo "Output: $(cd outputs && pwd)/dit_output.mp4"
