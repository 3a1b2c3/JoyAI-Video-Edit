#!/bin/bash
# Run DiT inference with default settings

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run inference
python run_inference.py \
  --video "${1:-assets/Recording 2026-08-12 205529.mp4}" \
  --out "${2:-outputs/dit_output.mp4}" \
  --frames "${3:-2}" \
  --steps "${4:-1}"

echo ""
echo "✅ Inference complete!"
echo "Output: $(cd outputs && pwd)/dit_output.mp4"
