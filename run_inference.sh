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

# Run inference
$PYTHON run_inference.py \
  --video "${1:-assets/Recording 2026-08-12 205529.mp4}" \
  --out "${2:-outputs/dit_output.mp4}" \
  --frames "${3:-2}" \
  --height "${4:-256}" \
  --width "${5:-256}" \
  --steps "${6:-1}"

echo ""
echo "✅ Inference complete!"
echo "Output: $(cd outputs && pwd)/dit_output.mp4"
